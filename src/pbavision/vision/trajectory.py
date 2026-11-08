"""Observed frame-by-frame paths on a calibrated fixed overhead camera.

No reflected paths are inferred from endpoints. Missing detections and camera
cuts split the observations, so a plotted path cannot bridge unseen footage.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .detect import cloth_mask
from .scoreboard import TeamLeagueReader
from .table import cloth_quad

BALLS = ("white", "yellow", "red")
BED_WIDTH, BED_HEIGHT = 2.84, 1.42


def candidates(hsv: np.ndarray, bed_mask: np.ndarray) -> dict[str, list[tuple[float, float, float]]]:
    """Colour candidates inside the bed, including short motion-blur streaks."""
    h, s, v = cv2.split(hsv)
    masks = {
        "white": (s < 125) & (v > 160),
        "yellow": (h >= 8) & (h <= 44) & (s > 125) & (v > 80),
        "red": ((h < 9) | (h > 167)) & (s > 100) & (v > 60),
    }
    result = {}
    for ball, mask in masks.items():
        binary = (mask & (bed_mask > 0)).astype(np.uint8)
        # Fill the printed spots on a ball before measuring its thick core.
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if ball == "white":
            # A ball touching the cue forms one long contour. Its thick core
            # still has a local centre; a thin shaft does not. Work on cores
            # instead of taking the centroid of that merged contour.
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            count, _, stats, centres = cv2.connectedComponentsWithStats((distance > 3.2).astype(np.uint8))
            found = []
            for index in range(1, count):
                if not 2 <= stats[index, cv2.CC_STAT_AREA] <= 180:
                    continue
                cx, cy = centres[index]
                ix, iy = round(cx), round(cy)
                patch = hsv[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
                hue, saturation, brightness = np.median(patch.reshape(-1, 3), axis=0)
                # This broadcast's ivory ball is warm; white logos and sleeves
                # at the edge of the cloth are neutral or magenta.
                if not (18 <= hue <= 80 and 15 <= saturation < 125 and brightness > 185):
                    continue
                appearance = abs(stats[index, cv2.CC_STAT_AREA] - 25) / 50
                # Motion blur mixes the ivory colour with blue cloth and
                # lengthens the shape. Such candidates can continue a known
                # identity, but cannot initialize it on a hand or sleeve.
                if hue > 45 or brightness < 220 or saturation < 25:
                    appearance += 4
                if 11 <= ix < hsv.shape[1] - 11 and 11 <= iy < hsv.shape[0] - 11:
                    yy, xx = np.ogrid[-11:12, -11:12]
                    annulus = (xx * xx + yy * yy >= 64) & (xx * xx + yy * yy <= 121)
                    if np.mean(binary[iy - 11:iy + 12, ix - 11:ix + 12][annulus]) > .2:
                        appearance += 4
                found.append((float(cx), float(cy), appearance))
            result[ball] = found
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if not 12 <= area <= 450 or perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / perimeter ** 2
            if circularity < .22:
                continue
            moments = cv2.moments(contour)
            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            appearance = abs(math.log(area / 85)) + (1 - circularity)
            found.append((cx, cy, appearance))
        result[ball] = found
    return result


class BallTracker:
    """Associate detections over adjacent frames; never supply predicted output."""

    def __init__(self):
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2)
        self.t: float | None = None

    def observe(self, found: list[tuple[float, float, float]], t: float) -> tuple[float, float] | None:
        if not found:
            return None
        if self.position is None or self.t is None:
            chosen = min(found, key=lambda p: p[2])
            if chosen[2] >= 3:
                return None
            self.velocity = np.zeros(2)
        else:
            dt = t - self.t
            if dt <= 0:
                raise ValueError("tracker observations must be chronological")
            # Never abandon identity for the best-looking unrelated object
            # after an occlusion. Callers reset on a new calibrated view.
            prediction = self.position + self.velocity * min(dt, .1)
            gate = max(22, min(80, 1500 * dt))
            plausible = [p for p in found if np.linalg.norm(np.array(p[:2]) - prediction) < gate]
            if not plausible:
                return None
            chosen = min(plausible, key=lambda p: np.linalg.norm(np.array(p[:2]) - prediction) / 15 + .2 * p[2])
            observed_velocity = (np.array(chosen[:2]) - self.position) / dt
            self.velocity = .35 * self.velocity + .65 * observed_velocity
        self.position = np.array(chosen[:2])
        self.t = t
        return float(chosen[0]), float(chosen[1])


def rail_contacts(paths: list[list[list[float]]], ball: str, max_gap: float = 0.) -> list[dict]:
    """Visible direction reversals near a rail, not a three-cushion rule verdict."""
    contacts: list[dict] = []
    groups: list[list[list[float]]] = []
    gaps = []
    for segment in paths:
        if not segment:
            continue
        if groups and 0 < segment[0][0] - groups[-1][-1][0] <= max_gap:
            gaps.append((groups[-1][-1][0], segment[0][0]))
            groups[-1].extend(segment)
        else:
            groups.append(list(segment))
    for segment in groups:
        for i in range(1, len(segment) - 1):
            # Wider temporal support catches shallow, slow contacts as well
            # as fast rebounds. Spatial travel rejects stationary rail noise.
            point, before, after = segment[i], segment[max(0, i - 6)], segment[min(len(segment) - 1, i + 6)]
            t, x, y = point
            for rail, axis, bound, sign in [("left", 1, 0., 1), ("right", 1, BED_WIDTH, -1),
                                            ("top", 2, 0., 1), ("bottom", 2, BED_HEIGHT, -1)]:
                if abs(point[axis] - bound) > .075:
                    continue
                incoming = sign * (point[axis] - before[axis]) / (t - before[0])
                outgoing = sign * (after[axis] - point[axis]) / (after[0] - t)
                travel_in = sign * (before[axis] - point[axis])
                travel_out = sign * (after[axis] - point[axis])
                neighbours = segment[max(0, i - 3):min(len(segment), i + 4)]
                is_extremum = sign * point[axis] <= min(sign * p[axis] for p in neighbours) + .0005
                if is_extremum and incoming < -.04 and outgoing > .04 and min(travel_in, travel_out) >= .008:
                    recent = [c for c in contacts if c["rail"] == rail and abs(c["t"] - t) < .4]
                    if not recent:
                        bracketed = any(before[0] <= a < b <= after[0] for a, b in gaps)
                        contacts.append({"ball": ball, "t": t, "x": x, "y": y, "rail": rail,
                                         "evidence": "bracketed_gap" if bracketed else "observed_reversal"})
    return sorted(contacts, key=lambda c: c["t"])


def view_samples(
    video: Path, start: float, end: float, bed_quad: np.ndarray, outer_quad: np.ndarray,
    owner: int, latest_start: float,
):
    """Yield calibrated camera views, preserving every missing source frame.

    Eligibility comes from the live active-player counter. It prevents a
    replay with a hidden/old counter from creating a new observed strike.
    """
    destination = np.array([[0, 0], [BED_WIDTH, 0], [BED_WIDTH, BED_HEIGHT], [0, BED_HEIGHT]], np.float32)
    matrix = cv2.getPerspectiveTransform(bed_quad.astype(np.float32), destination)
    bed_mask = np.zeros((540, 960), np.uint8)
    cv2.fillConvexPoly(bed_mask, bed_quad.astype(np.int32), 1)
    bed_mask = cv2.erode(bed_mask, np.ones((3, 3), np.uint8))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    trackers = {ball: BallTracker() for ball in BALLS}
    samples = []
    last_visible = -math.inf
    try:
        first = round(start * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame_no in range(first, math.ceil(end * fps)):
            ok, frame = capture.read()
            if not ok:
                break
            t = frame_no / fps
            img = cv2.resize(frame, (960, 540))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            quad = cloth_quad(cloth_mask(hsv))
            same_view = quad is not None and float(np.mean(np.abs(quad - outer_quad))) < 12
            if not same_view:
                if samples:
                    if t - last_visible > .3:
                        yield samples, fps, "camera_cut"
                        samples = []
                        trackers = {ball: BallTracker() for ball in BALLS}
                    else:
                        samples.append({"t": t, "frame": frame_no, "points": {}, "eligible": False})
                continue
            last_visible = t
            found = candidates(hsv, bed_mask)
            points = {}
            for ball in BALLS:
                pixel = trackers[ball].observe(found[ball], t)
                if pixel is not None:
                    xy = cv2.perspectiveTransform(np.array([[pixel]], np.float32), matrix)[0, 0]
                    if 0 <= xy[0] <= BED_WIDTH and 0 <= xy[1] <= BED_HEIGHT:
                        points[ball] = [float(v) for v in xy]
            eligible = t <= latest_start and TeamLeagueReader.active_player(frame) == owner
            samples.append({"t": t, "frame": frame_no, "points": points, "eligible": eligible})
        if samples:
            yield samples, fps, "interval_end"
    finally:
        capture.release()


def trajectory_from_samples(samples: list[dict], fps: float, cue: str, completeness: dict) -> dict:
    """Serialize measurements; missing frames remain separate path segments."""
    paths: dict[str, list] = {ball: [] for ball in BALLS}
    last_frame = {ball: -2 for ball in BALLS}
    start = samples[0]["t"]
    for sample in samples:
        for ball, xy in sample["points"].items():
            if last_frame[ball] != sample["frame"] - 1:
                paths[ball].append([])
            paths[ball][-1].append([round(sample["t"] - start, 4), *[round(v, 4) for v in xy]])
            last_frame[ball] = sample["frame"]
    paths = {ball: [segment for segment in segments if len(segment) >= 2] for ball, segments in paths.items()}
    coverage = sum(len(s) for s in paths[cue]) / len(samples)
    reasons = []
    if not completeness["start_observed"]:
        reasons.append("strike not visible in this camera view")
    if not completeness["end_settled"]:
        reasons.append("path ends before a settled layout is observed")
    if len(paths[cue]) > 1:
        reasons.append("cue detections contain gaps")
    if coverage < .9:
        reasons.append("cue coverage below 90 percent")
    return {
        "sample_fps": fps, "start": round(start, 4), "duration": round(samples[-1]["t"] - start, 4),
        "bed": {"width": BED_WIDTH, "height": BED_HEIGHT}, "paths": paths,
        "cushions": rail_contacts(paths[cue], cue, max_gap=2.1 / fps), "coverage": round(coverage, 3),
        "source_frames": len(samples), "camera_frames": sum(bool(s["points"]) for s in samples),
        "motion_observed": True, "completeness": completeness,
        "validation": {"status": "needs_review" if reasons else "automatically_tracked", "reasons": reasons},
        "status": "Measured camera-view fragment" if reasons else "Observed rest-to-rest movement",
    }


def scan_turn(video: Path, turn: dict, bed_quad: np.ndarray, outer_quad: np.ndarray) -> list[dict]:
    """Discover movements throughout a turn, not just prior resting candidates."""
    from .episodes import movement_windows

    movements = []
    for samples, fps, end_reason in view_samples(
        video, turn["start"], turn["end"] + 8, bed_quad, outer_quad, turn["player"], turn["end"],
    ):
        for window in movement_windows(samples, turn["cue"]):
            selected = samples[window["first"]:window["last"] + 1]
            if selected[0]["t"] > turn["end"] or len(selected) < 6:
                continue
            completeness = {k: window[k] for k in ("start_observed", "end_settled", "end_reason")}
            if not completeness["end_settled"]:
                completeness["end_reason"] = end_reason
            trajectory = trajectory_from_samples(selected, fps, turn["cue"], completeness)
            if sum(len(s) for s in trajectory["paths"][turn["cue"]]) < 6:
                continue
            movements.append({
                "start": trajectory["start"], "end": round(selected[-1]["t"], 4),
                "from": {b: p[0][0][1:] for b, p in trajectory["paths"].items() if p},
                "to": {b: p[-1][-1][1:] for b, p in trajectory["paths"].items() if p},
                "trajectory": trajectory,
            })
    return movements
