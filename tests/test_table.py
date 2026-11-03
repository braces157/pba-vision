"""Homography: synthetic quad maps to meters within a few centimetres."""

from __future__ import annotations

import numpy as np
from helpers import QUAD_PX, place

from pbavision.vision.table import OUTER_H, OUTER_W, homography, quad_plausible, to_meters


def test_quad_plausible_accepts_broadcast_geometry():
    assert quad_plausible(QUAD_PX)


def test_corners_map_to_rectangle_corners():
    matrix = homography(QUAD_PX)
    assert matrix is not None
    (tl, tr, br, bl) = QUAD_PX
    checks = [
        (tl, (0.0, 0.0)),
        (tr, (OUTER_W, 0.0)),
        (br, (OUTER_W, OUTER_H)),
        (bl, (0.0, OUTER_H)),
    ]
    for pixel, meters in checks:
        x, y = to_meters(matrix, *pixel)
        assert abs(x - meters[0]) < 1e-3
        assert abs(y - meters[1]) < 1e-3


def test_fractional_positions_map_consistently():
    matrix = homography(QUAD_PX)
    assert matrix is not None
    # a point halfway along the quad should land near the middle of the rectangle
    px, py = place(None, QUAD_PX, 0.5, 0.5)
    x, y = to_meters(matrix, px, py)
    assert abs(x - OUTER_W / 2) < 0.05
    assert abs(y - OUTER_H / 2) < 0.05


def test_degenerate_quad_is_rejected():
    bad = np.array([[0, 0], [10, 0], [12, 5], [0, 5]], dtype=np.float32)
    assert not quad_plausible(bad)
    assert homography(bad) is None


def test_perspective_does_not_explode_interior_points():
    matrix = homography(QUAD_PX)
    assert matrix is not None
    for fx in np.linspace(0.05, 0.95, 7):
        for fy in np.linspace(0.05, 0.95, 7):
            px, py = place(None, QUAD_PX, float(fx), float(fy))
            x, y = to_meters(matrix, px, py)
            assert -0.1 <= x <= OUTER_W + 0.1
            assert -0.1 <= y <= OUTER_H + 0.1
