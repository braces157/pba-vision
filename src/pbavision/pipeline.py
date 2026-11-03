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

from .vision.detect import WORK_WIDTH, analyse_frame, cloth_mask
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
    frames_on_table: int = 0
    frames_quad_locked: int = 0
    frames_all_balls: int = 0
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
    """Run the full reconstruction over a list of frame paths."""
    out = Reconstruction(fps=fps)
    mappings: list[tuple[int, dict[str, tuple[float, float]]]] = []
    last_quad: np.ndarray | None = None
    say = progress or (lambda _msg: None)

    for index, path in enumerate(frames):
        img = cv2.imread(str(path))
        if img is None:
            continue
        out.frames_seen += 1
        scale = WORK_WIDTH / img.shape[1]
        img = cv2.resize(img, (WORK_WIDTH, int(img.shape[0] * scale)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        result = analyse_frame(img)
        if result.table_bbox is None:
            continue
        out.frames_on_table += 1

        quad = cloth_quad(cloth_mask(hsv))
        if quad is not None and not quad_plausible(quad):
            quad = None
        if quad is not None:
            last_quad = quad
            out.frames_quad_locked += 1
        elif last_quad is None:
            continue
        else:
            quad = last_quad
        matrix = homography(quad)
        if matrix is None:
            continue

        if not result.all_three:
            continue
        out.frames_all_balls += 1
        mappings.append((index, {k: to_meters(matrix, result.balls[k][0], result.balls[k][1]) for k in BALLS}))

    say(f"{out.frames_seen} frames read, {out.frames_on_table} full-table, "
        f"{out.frames_quad_locked} quad-locked, {out.frames_all_balls} with all three balls")

    out.rests = _rest_chain(mappings, fps)
    say(f"{len(out.rests)} rest states")

    inning_of = {"white": 1, "yellow": 1}
    anchored_cue: str | None = None
    kept: list[tuple[dict[str, bool], RestState]] = []
    for i, (before, after) in enumerate(zip(out.rests, out.rests[1:])):
        moved = {
            k: bool(np.hypot(before.positions[k][0] - after.positions[k][0],
                             before.positions[k][1] - after.positions[k][1]) > MOVE_EPSILON)
            for k in BALLS
        }
        if not any(moved.values()):
            continue  # detection gap or camera cut, not a stroke
        travel = sum(
            np.hypot(before.positions[k][0] - after.positions[k][0],
                     before.positions[k][1] - after.positions[k][1])
            for k in BALLS
        )
        if travel > TRAVEL_CAP:
            continue  # physically implausible as one transition: a cut or lost tracking
        if _is_replay(before, moved, kept):
            continue  # replays repeat a shot that was already recorded

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
