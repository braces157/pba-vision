"""Synthetic broadcast frames for offline tests.

No copyrighted footage is committed; these helpers draw a plausible carom
table — cloth, cushions as a slightly darker border, three balls — at a known
geometry, so detectors and homography have exact ground truth to answer to.
"""

from __future__ import annotations

import cv2
import numpy as np

# A synthetic "overhead camera" frame: cloth quad is a gentle trapezoid, the
# way a high camera actually sees the table.
QUAD_PX = np.array([[260, 120], [980, 140], [1010, 620], [220, 600]], dtype=np.float32)


def draw_table(size=(1280, 720)) -> np.ndarray:
    """BGR image: dark surroundings, blue cloth inside QUAD_PX."""
    img = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
    img[:, :, 2] = 60  # dull red floor, like the venues
    quad = QUAD_PX.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [quad], (200, 140, 60))  # cloth: hue ~105 in HSV terms
    cv2.polylines(img, [quad], True, (60, 40, 20), 4)  # wooden cushion edge
    return img


def draw_ball(img: np.ndarray, x: float, y: float, colour: str, radius: int = 8) -> None:
    bgr = {"white": (255, 255, 255), "yellow": (60, 180, 230), "red": (60, 60, 230)}[colour]
    cv2.circle(img, (int(x), int(y)), radius, bgr, -1)


def place(img: np.ndarray, quad: np.ndarray, fx: float, fy: float) -> tuple[float, float]:
    """Fractional position on the outer rectangle -> pixel position on the quad."""
    (tl, tr, br, bl) = quad
    top = tl + (tr - tl) * fx
    bottom = bl + (br - bl) * fx
    left = tl + (bl - tl) * fy
    right = tr + (br - tr) * fy
    # bilinear patch: average of the two rails' interpolations
    px_top = top + (bottom - top) * fy
    px_side = left + (right - left) * fx
    return ((px_top[0] + px_side[0]) / 2, (px_top[1] + px_side[1]) / 2)
