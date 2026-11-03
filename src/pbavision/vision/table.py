"""Cloth quad detection and the pixel-to-table-meters homography.

The cloth mask spans the bed plus the cushion tops, so the quad it forms is
the *outer* cushion rectangle (~3.10 x 1.70 m), not the 2.84 x 1.42 m playing
bed. Ball coordinates therefore carry a small cushion bias (~one cushion
width near the rails); resting positions away from the rails are unaffected.
Refining the quad to the bed edge is on the roadmap.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Outer cushion rectangle of a tournament carom table, meters.
OUTER_W = 3.10
OUTER_H = 1.70

#: Playing bed, meters — useful as a sanity bound, not as the mapping target.
BED_W = 2.84
BED_H = 1.42

_DST = None


def _dst_corners() -> np.ndarray:
    global _DST
    if _DST is None:
        _DST = np.array([[0, 0], [OUTER_W, 0], [OUTER_W, OUTER_H], [0, OUTER_H]], dtype=np.float32)
    return _DST


def cloth_quad(mask: np.ndarray) -> np.ndarray | None:
    """Four corners of the cloth region, ordered TL, TR, BR, BL — or None.

    The convex hull is approximated with an escalating epsilon until exactly
    four points survive; a non-convex result means the mask captured something
    that is not the table and is rejected.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    for epsilon in (0.01, 0.02, 0.03, 0.05):
        quad = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(quad) == 4:
            break
    if len(quad) != 4 or not cv2.isContourConvex(quad):
        return None
    quad = quad.astype(np.float32)
    sums, diffs = quad.sum(axis=1), quad[:, 0] - quad[:, 1]
    tl, br = quad[np.argmin(sums)], quad[np.argmax(sums)]
    tr, bl = quad[np.argmax(diffs)], quad[np.argmin(diffs)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_plausible(quad: np.ndarray) -> bool:
    """Reject quads that cannot be a carom table seen from a broadcast camera."""
    (tl, tr, br, bl) = quad
    long_side = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    short_side = min(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if short_side < 90 or long_side / max(short_side, 1) > 2.6:
        return False
    return long_side >= 200


def homography(quad: np.ndarray) -> np.ndarray | None:
    """Perspective transform from pixel quad to the outer rectangle in meters."""
    if not quad_plausible(quad):
        return None
    return cv2.findHomography(quad, _dst_corners())[0]


def to_meters(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Map one pixel point through the homography."""
    pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), matrix)[0][0]
    return float(pt[0]), float(pt[1])
