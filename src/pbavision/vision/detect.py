"""Ball detection by colour segmentation.

The broadcast lights the table uniformly and the three balls are three
distinct colours on a uniform cloth, so plain HSV masks plus shape filters
find them without any learned model. A tiny YOLO would recover the remaining
20% (see the README roadmap); this module is the transparent baseline the
learned detector has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: Frames are analysed at this width; a full-table ball is ~13 px across.
WORK_WIDTH = 960

# Ball diameter ~13 px at WORK_WIDTH on a full-table view; the window is wide
# enough for wide venue shots (smaller balls) and tight overhead crops.
MIN_AREA = 30
MAX_AREA = 900
MIN_CIRCULARITY = 0.55
#: Slack around the cloth bbox so balls resting against a cushion stay inside.
BBOX_MARGIN = 24


@dataclass
class Params:
    """HSV gates and shape filters, in one place for tuning."""

    white: tuple[int, int, int, int] = (0, 180, 70, 190)   # s_lo, s_hi, v_lo, v_hi (hue unused)
    yellow: tuple[int, int, int, int] = (18, 45, 70, 80)   # h_lo, h_hi, s_lo, v_lo
    red: tuple[int, int, int, int] = (172, 6, 90, 50)      # h_lo/h_hi wrap, s_lo, v_lo
    min_area: int = MIN_AREA
    max_area: int = MAX_AREA
    min_circularity: float = MIN_CIRCULARITY
    bbox_margin: int = BBOX_MARGIN


@dataclass
class FrameResult:
    """Everything one frame contributed: table box and best ball per colour."""

    table_bbox: tuple[int, int, int, int] | None
    balls: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def all_three(self) -> bool:
        return {"white", "yellow", "red"} <= self.balls.keys()


def _ball_mask(hsv: np.ndarray, params: Params, colour: str) -> np.ndarray:
    h, s, v = cv2.split(hsv.astype(np.int16))
    if colour == "white":
        _, _, s_lo, v_lo = params.white
        mask = (s < s_lo) & (v > v_lo)
    elif colour == "yellow":
        h_lo, h_hi, s_lo, v_lo = params.yellow
        mask = (h >= h_lo) & (h <= h_hi) & (s > s_lo) & (v > v_lo)
    else:  # red wraps around the hue circle
        h_lo, h_hi, s_lo, v_lo = params.red
        mask = ((h <= h_hi) | (h >= h_lo)) & (s > s_lo) & (v > v_lo)
    return mask.astype(np.uint8)


def detect_balls(
    hsv: np.ndarray,
    bbox: tuple[int, int, int, int],
    params: Params | None = None,
) -> dict[str, tuple[float, float]]:
    """Best candidate per ball colour inside ``bbox``, as (x, y) in ``hsv`` coordinates.

    Candidates are contour-based: an area window rejects specks and blobs,
    circularity rejects the rectangles of overlay graphics. When several
    candidates pass, the one closest to the expected full-table ball area wins.
    """
    params = params or Params()
    x, y, w, h = bbox
    x0, y0 = max(0, x - params.bbox_margin), max(0, y - params.bbox_margin)
    x1 = min(hsv.shape[1], x + w + params.bbox_margin)
    y1 = min(hsv.shape[0], y + h + params.bbox_margin)
    roi = hsv[y0:y1, x0:x1]
    cloth_contours, _ = cv2.findContours(cloth_mask(hsv), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cloth_outline = cv2.convexHull(max(cloth_contours, key=cv2.contourArea)) if cloth_contours else None

    expected = 130.0  # contour area of a ~13 px ball
    out: dict[str, tuple[float, float]] = {}
    for colour in ("white", "yellow", "red"):
        mask = cv2.morphologyEx(_ball_mask(roi, params, colour), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # A white rail can surround the white ball in this mask. External-only
        # contours discard the enclosed ball along with the rail's interior.
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        best: tuple[float, float, float] | None = None
        for index, contour in enumerate(contours):
            depth, parent = 0, hierarchy[0, index, 3]
            while parent >= 0:
                depth += 1
                parent = hierarchy[0, parent, 3]
            if depth % 2:
                continue  # a hole in a bright region is not a ball component
            area = cv2.contourArea(contour)
            if area < params.min_area or area > params.max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < params.min_circularity:
                continue
            (cx, cy), _ = cv2.minEnclosingCircle(contour)
            if cloth_outline is not None and cv2.pointPolygonTest(cloth_outline, (cx + x0, cy + y0), True) < 3:
                continue
            score = circularity * (1.0 - abs(area - expected) / 900.0)
            if best is None or score > best[0]:
                best = (score, cx + x0, cy + y0)
        if best:
            out[colour] = (best[1], best[2])
    return out


def cloth_mask(hsv: np.ndarray) -> np.ndarray:
    """The cloth: green or blue, saturated, closed to swallow sight dots and logos."""
    h, s, v = cv2.split(hsv)
    green = ((h >= 35) & (h <= 90) & (s > 60) & (v > 40)).astype(np.uint8)
    blue = ((h >= 95) & (h <= 130) & (s > 60) & (v > 40)).astype(np.uint8)
    return cv2.morphologyEx(green | blue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def cloth_bbox(hsv: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
    """Bounding box of the largest cloth-coloured region, with its area fraction."""
    contours, _ = cv2.findContours(cloth_mask(hsv), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x, y, w, h), (w * h) / (hsv.shape[0] * hsv.shape[1])


def analyse_frame(bgr: np.ndarray, params: Params | None = None) -> FrameResult:
    """One frame in, one :class:`FrameResult` out.

    Frames where the cloth does not look like a carom table — title cards,
    close-ups, crowd shots — return ``table_bbox=None``. The aspect-ratio gate
    encodes the table's 2:1 proportions, which title graphics fail.
    """
    params = params or Params()
    scale = WORK_WIDTH / bgr.shape[1]
    img = cv2.resize(bgr, (WORK_WIDTH, int(bgr.shape[0] * scale)))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    found = cloth_bbox(hsv)
    if found is None:
        return FrameResult(table_bbox=None)
    bbox, fraction = found
    w, h = bbox[2], max(bbox[3], 1)
    if fraction < 0.10 or not 1.35 <= w / h <= 2.4:
        return FrameResult(table_bbox=None)
    return FrameResult(table_bbox=bbox, balls=detect_balls(hsv, bbox, params))
