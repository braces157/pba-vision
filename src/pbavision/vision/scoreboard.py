"""Read cumulative scores from the PBA Team League broadcast score bug.

Score changes are observations of the broadcast, not individual shot labels.
The overlay can update late or disappear during a scoring shot.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ScoreReading:
    t: float
    scores: tuple[int, int]


@dataclass
class ScoreTimeline:
    readings: list[ScoreReading] = field(default_factory=list)
    samples: int = 0
    readable_samples: int = 0
    rejected_regressions: int = 0
    target: int = 15

    def to_dict(self) -> dict:
        initial = self.readings[0].scores if self.readings else None
        final = self.readings[-1].scores if self.readings else None
        changes = []
        for before, after in zip(self.readings, self.readings[1:]):
            changes.append({
                "t": after.t,
                "scores": list(after.scores),
                "delta": [a - b for a, b in zip(after.scores, before.scores)],
            })
        return {
            "source": "broadcast_scoreboard",
            "profile": "pba-team-league",
            "player_order": "top_to_bottom",
            "target": self.target,
            "initial": list(initial) if initial is not None else None,
            "final": list(final) if final is not None else None,
            "starts_at_zero": initial == (0, 0),
            "reaches_target": final is not None and max(final) == self.target,
            "samples": self.samples,
            "readable_samples": self.readable_samples,
            "rejected_regressions": self.rejected_regressions,
            "readings": [asdict(reading) for reading in self.readings],
            "changes": changes,
            "timing": "first sample of each confirmed score; overlay updates may lag the shot",
            "resolution": "score updates, not shot outcomes; a delta may cover multiple shots",
        }


class ScoreTracker:
    """Confirm readings in nearby samples and ignore old scores in replays."""

    def __init__(self, target: int = 15, confirmations: int = 2, max_gap: float = 3.0):
        if target <= 0 or confirmations < 2 or not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("target must be positive, confirmations >= 2, and max_gap positive")
        self.timeline = ScoreTimeline(target=target)
        self.confirmations = confirmations
        self.max_gap = max_gap
        self.pending: ScoreReading | None = None
        self.count = 0
        self.last_t: float | None = None

    def observe(self, t: float, scores: tuple[int, int] | None) -> None:
        if not math.isfinite(t) or t < 0 or (self.last_t is not None and t <= self.last_t):
            raise ValueError("sample times must be finite, nonnegative and strictly increasing")
        previous_t, self.last_t = self.last_t, t
        self.timeline.samples += 1
        if scores is None or any(type(s) is not int or not 0 <= s <= self.timeline.target for s in scores):
            self.pending, self.count = None, 0
            return
        self.timeline.readable_samples += 1
        if self.timeline.readings:
            current = self.timeline.readings[-1].scores
            if any(a < b for a, b in zip(scores, current)):
                self.timeline.rejected_regressions += 1
                self.pending, self.count = None, 0
                return
            if scores == current:
                self.pending, self.count = None, 0
                return
        if (self.pending is None or self.pending.scores != scores
                or previous_t is None or t - previous_t > self.max_gap):
            self.pending, self.count = ScoreReading(round(t, 3), scores), 1
        else:
            self.count += 1
        if self.count >= self.confirmations:
            self.timeline.readings.append(self.pending)
            self.pending, self.count = None, 0


class TeamLeagueReader:
    """Fixed score cells in the 16:9 Team League layout, scaled to frame size."""

    # Coordinates measured on a 1920 x 1080 broadcast. Exclude the adjacent
    # cue-ball/run counter, inning number, names and team/set scores.
    CELLS = ((417, 885, 465, 930), (417, 933, 465, 978))
    INNING_CELL = (490, 848, 522, 880)

    @staticmethod
    def active_player(frame: np.ndarray) -> int | None:
        """The active cue-ball circle contains a dark run digit; the other is blank."""
        h, w = frame.shape[:2]
        if abs(w / h - 16 / 9) > 0.05:
            return None
        for x1, y1, x2, y2 in TeamLeagueReader.CELLS:
            tile = frame[round(y1 * h / 1080):round(y2 * h / 1080),
                         round(x1 * w / 1920):round(x2 * w / 1920)]
            if not tile.size:
                return None
            hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            background = ((hsv[:, :, 1] < 70) | ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 40)))
            if float(np.mean(background & (hsv[:, :, 2] > 160))) < .4:
                return None
        digits = []
        for y in (895, 943):
            crop = frame[round(y * h / 1080):round((y + 25) * h / 1080),
                         round(485 * w / 1920):round(503 * w / 1920)]
            if not crop.size:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark, bright = float(np.mean(gray < 100)), float(np.mean(gray > 160))
            if dark >= .12 and bright >= .20:
                digits.append(True)
            elif dark <= .05 and bright >= .80:
                digits.append(False)
            else:
                return None
        return digits.index(True) if digits.count(True) == 1 else None

    def __init__(self, target: int = 15, min_confidence: float = 0.9):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError('score OCR requires: pip install -e ".[scoreboard]"') from exc
        self.engine = RapidOCR()
        self.target = target
        self.min_confidence = min_confidence

    def read(self, frame: np.ndarray) -> tuple[int, int] | None:
        h, w = frame.shape[:2]
        if abs(w / h - 16 / 9) > 0.05:
            return None
        scores = []
        for x1, y1, x2, y2 in self.CELLS:
            crop = frame[round(y1 * h / 1080):round(y2 * h / 1080),
                         round(x1 * w / 1920):round(x2 * w / 1920)]
            if not crop.size:
                return None
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # White/yellow score tiles vanish when the score bug is hidden.
            background = ((hsv[:, :, 1] < 70) | ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 40)))
            background &= hsv[:, :, 2] > 160
            if float(np.mean(background)) < 0.4:
                return None
            result, _ = self.engine(crop, use_det=False, use_cls=False)
            if not result or len(result) != 1:
                return None
            text, confidence = result[0]
            # The broadcast's narrow single zero has lower OCR confidence
            # than other digits. Tile checks and temporal confirmation still
            # apply; never translate arbitrary letters into digits.
            threshold = 0.5 if text == "0" else self.min_confidence
            if confidence < threshold or not text.isascii() or not text.isdigit():
                return None
            value = int(text)
            if value > self.target:
                return None
            scores.append(value)
        return scores[0], scores[1]

    def read_inning(self, frame: np.ndarray) -> int | None:
        """Read the inning beside SET; active row alone cannot reject old replays."""
        h, w = frame.shape[:2]
        if abs(w / h - 16 / 9) > .05:
            return None
        x1, y1, x2, y2 = self.INNING_CELL
        crop = frame[round(y1 * h / 1080):round(y2 * h / 1080),
                     round(x1 * w / 1920):round(x2 * w / 1920)]
        if not crop.size:
            return None
        result, _ = self.engine(crop, use_det=False, use_cls=False)
        if not result or len(result) != 1:
            return None
        text, confidence = result[0]
        if confidence < self.min_confidence or not text.isascii() or not text.isdigit():
            return None
        value = int(text)
        return value if 1 <= value <= 99 else None


def read_video_scores(video: Path, sample_fps: float = 2.0, target: int = 15, progress=None) -> ScoreTimeline:
    """Sample a video directly; ball tracking and the official terminal are optional."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("score sample fps must be positive and finite")
    reader = TeamLeagueReader(target=target)
    tracker = ScoreTracker(target=target, max_gap=3 / sample_fps)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"video has no usable frame timing: {video}")
        if sample_fps > fps:
            raise ValueError("score sample fps cannot exceed the video frame rate")
        duration = frames / fps
        next_frame = 0
        for index in range(math.ceil(duration * sample_fps)):
            sample_frame = round(index * fps / sample_fps)
            if sample_frame >= frames:
                break
            t = sample_frame / fps
            # Decode forward once. Repeated seeking decodes the preceding
            # keyframe interval again for every tiny scoreboard crop.
            ok = True
            while next_frame <= sample_frame:
                ok = capture.grab()
                next_frame += 1
                if not ok:
                    break
            if ok:
                ok, frame = capture.retrieve()
            if not ok:
                tracker.observe(t, None)
                continue
            count = len(tracker.timeline.readings)
            tracker.observe(t, reader.read(frame))
            if progress and len(tracker.timeline.readings) > count:
                reading = tracker.timeline.readings[-1]
                progress(f"{reading.t:7.1f}s: {reading.scores[0]} - {reading.scores[1]}")
            elif progress and index % max(1, round(60 * sample_fps)) == 0:
                progress(f"scanned {t:.0f}/{duration:.0f}s")
    finally:
        capture.release()
    if not tracker.timeline.readings:
        raise RuntimeError("no confirmed scores; this video may not use the pba-team-league layout")
    return tracker.timeline
