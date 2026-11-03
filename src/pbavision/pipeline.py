"""From frames to a shot-by-shot reconstruction.

The spine of the package. Four passes over the frames:

    1. per frame: table gate, cloth quad, homography, ball positions in meters
    2. rest states: runs of frames where all three balls are present and still
    3. strokes: transitions between consecutive rest states; which balls moved
       classifies the outcome (see ``classify`` — no cue-ball identity needed)
    4. innings: strokes chained by the 3-cushion rule "score and you continue,
       miss and the other player comes to the table"

Steps 3 and 4 are approximations from rest positions only: a carom that hits
both object balls but completes fewer than three cushions scores nothing, and
no rest-position data can see that. The validation report quantifies exactly
this gap instead of hiding it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .vision.camera import MainCameraTracker
from .vision.detect import WORK_WIDTH, analyse_frame, cloth_bbox, cloth_mask
from .vision.table import cloth_quad, homography, quad_plausible, to_meters

#: A rest is at least this many analysed frames with all three balls present.
REST_MIN_FRAMES = 2
#: Consecutive frames are "the same rest" while no ball moves more than this (meters).
REST_EPSILON = 0.025
#: A ball moved during a stroke if it travelled at least this far (meters).
MOVE_EPSILON = 0.08
#: A gap of this many lost frames breaks a rest run (camera cut, occlusion).
REST_GAP_FRAMES = 8
#: Total displacement across all balls in one transition above this is a cut,
#: not a stroke: three balls cannot jointly travel this far in one sample step.
TRAVEL_CAP = 7.0
#: Starting positions closer than this with the same moved pattern = replay.
REPLAY_TOLERANCE = 0.35

BALLS = ("white", "yellow", "red")


@dataclass
class RestState:
    """The table between two strokes: median positions of a stable run of frames."""

    frame: int          # first frame index of the run
    timestamp: float    # seconds into the video
    positions: dict[str, tuple[float, float]]


@dataclass
class Stroke:
    """One transition between rest states."""

    index: int
    from_rest: RestState
    to_rest: RestState
    moved: dict[str, bool]
    estimated_points: int      # 1 when the rest data implies a carom, else 0
    estimated_cue: str | None  # shooter's ball, when decidable from rest data
    cue: str | None            # shooter's ball after chaining; None until anchored
    inning_estimate: int       # 1-based, of the chained shooter; 0 while unanchored


@dataclass
class Reconstruction:
    strokes: list[Stroke] = field(default_factory=list)
    rests: list[RestState] = field(default_factory=list)
    frames_seen: int = 0
    frames_measured: int = 0
    frames_all_balls: int = 0
    camera_views: int = 0
    stable_cameras: int = 0
    main_camera_frames: int = 0
    fps: float = 4.0


def _rest_chain(mappings: list[tuple[int, dict[str, tuple[float, float]]]], fps: float) -> list[RestState]:
    """Collapse the per-frame position stream into stable rest states."""
    rests: list[RestState] = []
    run: list[tuple[int, dict[str, tuple[float, float]]]] = []

    def settled(a: dict[str, tuple[float, float]], b: dict[str, tuple[float, float]]) -> bool:
        return all(np.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) < REST_EPSILON for k in BALLS)

    def flush() -> None:
        if len(run) >= REST_MIN_FRAMES:
            median = {
                k: (
                    float(np.median([p[k][0] for _, p in run])),
                    float(np.median([p[k][1] for _, p in run])),
                )
                for k in BALLS
            }
            rests.append(RestState(frame=run[0][0], timestamp=run[0][0] / fps, positions=median))
        run.clear()

    for frame, positions in mappings:
        if run and (not settled(run[-1][1], positions) or frame - run[-1][0] > REST_GAP_FRAMES):
            flush()
        run.append((frame, positions))
    flush()
    return rests


def classify(moved: dict[str, bool]) -> tuple[int, str | None]:
    """(points, cue ball) from which balls moved — no cue-ball identity needed.

    The cue ball always moves: it is the one struck. White and yellow are the
    two players' cue balls, red is always an object ball.

        one mover                    miss; the mover is the cue ball
        two movers incl. red         miss; the other mover is the cue ball
        two movers, white + yellow   miss; which of them shot cannot matter
        three movers                 carom — unless the stroke failed the
                                     three-cushion rule, unknowable from rests
    """
    movers = [k for k, did in moved.items() if did]
    if len(movers) == 1:
        return 0, movers[0]
    if len(movers) == 2:
        if "red" in movers:
            return 0, movers[0] if movers[0] != "red" else movers[1]
        return 0, None
    if len(movers) == 3:
        return 1, None
    raise ValueError("a stroke must move at least one ball")


def _next_cue(cue: str | None) -> str | None:
    """A miss sends the other player to the table, with their own cue ball."""
    if cue == "white":
        return "yellow"
    if cue == "yellow":
        return "white"
    return None


def reconstruct(frames: list, fps: float, progress: Callable[[str], None] | None = None) -> Reconstruction:
    """Run the full reconstruction over a list of frame paths.

    Two passes. The survey pass learns the broadcast's stable cameras — a
    fixed tripod reveals itself as a dominant cloth-quad cluster. The
    measurement pass then reads positions only on stable cameras, and splits
    the stream into per-camera segments so a cut never joins two views into
    one "movement". Segments from different cameras produce strokes
    independently; because positions are in table meters, the replay filter
    then recognizes when two cameras saw the same shot and keeps one.
    """
    out = Reconstruction(fps=fps)
    say = progress or (lambda _msg: None)

    # ---- pass 1: survey the cameras -------------------------------------
    tracker = MainCameraTracker()
    gated: list[tuple[int, str]] = []  # frames the table gate passed
    for index, path in enumerate(frames):
        img = cv2.imread(str(path))
        if img is None:
            continue
        out.frames_seen += 1
        scale = WORK_WIDTH / img.shape[1]
        img = cv2.resize(img, (WORK_WIDTH, int(img.shape[0] * scale)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        found = cloth_bbox(hsv)
        if found is None:
            continue
        bbox, fraction = found
        w, h = bbox[2], max(bbox[3], 1)
        if fraction < 0.10 or not 1.35 <= w / h <= 2.4:
            continue
        quad = cloth_quad(cloth_mask(hsv))
        if quad is None or not quad_plausible(quad):
            continue
        tracker.observe(quad, img.shape[:2])
        gated.append((index, str(path)))

    survey = tracker.finalize()
    out.camera_views = survey["cameras"]
    out.stable_cameras = survey["stable_cameras"]
    out.main_camera_frames = survey["main_camera_frames"]
    say(f"survey: {survey['cameras']} views, {survey['stable_cameras']} stable cameras, "
        f"main covers {survey['main_camera_frames']}/{survey['total_frames']} table frames, "
        f"cluster sizes {survey['largest_clusters']}")
    if survey["stable_cameras"] == 0:
        say("no stable camera found — refusing to guess; no positions measured")
        return out

    # ---- pass 2: measure on stable cameras, keep the camera label --------
    labeled: list[tuple[int, int, dict[str, tuple[float, float]]]] = []
    for index, path in gated:
        img = cv2.imread(path)
        if img is None:
            continue
        scale = WORK_WIDTH / img.shape[1]
        img = cv2.resize(img, (WORK_WIDTH, int(img.shape[0] * scale)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        quad = cloth_quad(cloth_mask(hsv))
        if quad is None or not quad_plausible(quad):
            continue
        label = tracker.label(quad, img.shape[:2])
        if label is None:
            continue  # an unstable view: transition graphics, drifting zoom
        matrix = homography(quad)
        if matrix is None:
            continue
        result = analyse_frame(img)
        if not result.all_three:
            continue
        out.frames_measured += 1
        positions = {k: to_meters(matrix, result.balls[k][0], result.balls[k][1]) for k in BALLS}
        labeled.append((index, label, positions))

    say(f"{out.frames_seen} frames read, {out.frames_measured} measured on "
        f"{survey['stable_cameras']} stable cameras")

    # ---- rest states, per camera segment --------------------------------
    rest_groups = [_rest_chain([(idx, pos) for idx, _label, pos in segment], fps)
                   for segment in _segments(labeled, REST_GAP_FRAMES)]
    out.rests = [rest for group in rest_groups for rest in group]
    say(f"{len(out.rests)} rest states in {len(rest_groups)} camera segments")

    # ---- strokes: candidates inside each segment, merged in time ---------
    candidates: list[tuple[int, RestState, RestState]] = []
    for group in rest_groups:
        for before, after in zip(group, group[1:]):
            candidates.append((before.frame, before, after))
    candidates.sort(key=lambda c: c[0])

    inning_of = {"white": 1, "yellow": 1}
    anchored_cue: str | None = None
    kept: list[tuple[dict[str, bool], RestState]] = []
    for i, (_frame, before, after) in enumerate(candidates):
        moved = {
            k: bool(np.hypot(before.positions[k][0] - after.positions[k][0],
                             before.positions[k][1] - after.positions[k][1]) > MOVE_EPSILON)
            for k in BALLS
        }
        if not any(moved.values()):
            continue  # detection gap, not a stroke
        travel = sum(
            np.hypot(before.positions[k][0] - after.positions[k][0],
                     before.positions[k][1] - after.positions[k][1])
            for k in BALLS
        )
        if travel > TRAVEL_CAP:
            continue  # physically implausible as one transition
        if _is_replay(before, moved, kept):
            continue  # another camera (or a replay) already recorded this shot

        points, detected_cue = classify(moved)
        cue = detected_cue or anchored_cue
        if points == 0:
            # The miss ends this player's inning; the stroke is theirs, the
            # next one belongs to the opponent.
            stroke_cue, stroke_inning = cue, inning_of.get(cue, 0) if cue else 0
            cue = _next_cue(cue)
            if cue:
                inning_of[cue] += 1
        else:
            stroke_cue, stroke_inning = cue, inning_of.get(cue, 0) if cue else 0
        if stroke_cue:
            anchored_cue = stroke_cue
        out.strokes.append(Stroke(
            index=i, from_rest=before, to_rest=after, moved=moved,
            estimated_points=points, estimated_cue=detected_cue,
            cue=stroke_cue, inning_estimate=stroke_inning,
        ))
        kept.append((moved, before))

    say(f"{len(out.strokes)} strokes reconstructed")
    return out


def _segments(labeled: list, _gap: int) -> list[list]:
    """Split the labeled frame stream into maximal same-camera runs.

    A run breaks when the camera label changes — that is the only boundary
    that can falsify positions, because two stable cameras map the same
    resting balls a few centimetres apart. Frame gaps inside a segment are
    harmless: the broadcast cuts away and back to the *same* tripod between
    strokes all the time, and the rest chain handles gaps on its own.
    """
    runs: list[list] = []
    current: list = []
    previous_label: int | None = None
    for index, label, positions in labeled:
        if current and label != previous_label:
            runs.append(current)
            current = []
        current.append((index, label, positions))
        previous_label = label
    if current:
        runs.append(current)
    return runs


def _is_replay(before: RestState, moved: dict[str, bool], kept: list[tuple[dict[str, bool], RestState]]) -> bool:
    """A replay repeats an already-recorded shot: same movers, same starting spots.

    Tolerance is generous because replays may run from a different camera; only
    nearby starts with an identical moved pattern are rejected.
    """
    for other_moved, other_start in kept:
        if other_moved != moved:
            continue
        close = all(
            np.hypot(before.positions[k][0] - other_start.positions[k][0],
                     before.positions[k][1] - other_start.positions[k][1]) < REPLAY_TOLERANCE
            for k in BALLS
        )
        if close:
            return True
    return False
