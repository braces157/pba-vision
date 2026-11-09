"""Ball detection on synthetic frames: finds the three, ignores the decoys."""

from __future__ import annotations

import cv2
import numpy as np
from helpers import QUAD_PX, draw_ball, draw_table, place

import pbavision.vision.detect
from pbavision.vision.detect import analyse_frame, cloth_mask
from pbavision.vision.table import cloth_quad


def _hsv_of(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def test_finds_all_three_balls():
    img = draw_table()
    spots = {"white": (0.5, 0.5), "yellow": (0.25, 0.3), "red": (0.8, 0.7)}
    for colour, (fx, fy) in spots.items():
        x, y = place(img, QUAD_PX, fx, fy)
        draw_ball(img, x, y, colour)

    result = analyse_frame(img)
    assert result.table_bbox is not None
    assert result.all_three


def test_white_rail_does_not_hide_the_white_ball():
    img = draw_table()
    cv2.polylines(img, [QUAD_PX.astype(np.int32)], True, (255, 255, 255), 14)
    for colour, fx in [("white", 0.3), ("yellow", 0.5), ("red", 0.7)]:
        x, y = place(img, QUAD_PX, fx, 0.5)
        draw_ball(img, x, y, colour)
    result = analyse_frame(img)
    assert result.all_three
    x, y = place(img, QUAD_PX, 0.3, 0.5)
    scale = pbavision.vision.detect.WORK_WIDTH / img.shape[1]
    assert np.linalg.norm(np.array(result.balls["white"]) - [x * scale, y * scale]) < 4


def test_white_decoy_inside_bbox_but_outside_cloth_is_ignored():
    img = draw_table()
    x, y = place(img, QUAD_PX, 0.5, 0.5)
    draw_ball(img, x, y, "white")
    draw_ball(img, 240, 170, "white", radius=10)
    result = analyse_frame(img)
    scale = pbavision.vision.detect.WORK_WIDTH / img.shape[1]
    assert np.linalg.norm(np.array(result.balls["white"]) - [x * scale, y * scale]) < 4


def test_boundary_connected_notch_does_not_hide_cushion_ball():
    img = draw_table()
    x, y = place(img, QUAD_PX, .025, .5)
    cv2.rectangle(img, (170, int(y - 35)), (int(x + 16), int(y + 35)), (20, 20, 20), -1)
    draw_ball(img, x, y, "white")
    result = analyse_frame(img)
    assert "white" in result.balls
    scale = pbavision.vision.detect.WORK_WIDTH / img.shape[1]
    assert np.linalg.norm(np.array(result.balls["white"]) - [x * scale, y * scale]) < 4


def test_ball_sized_hole_inside_white_region_is_not_a_white_ball():
    img = draw_table()
    cv2.rectangle(img, (500, 290), (600, 390), (255, 255, 255), -1)
    cv2.circle(img, (550, 340), 9, (200, 140, 60), -1)
    assert "white" not in analyse_frame(img).balls


def test_positions_are_near_truth():
    img = draw_table()
    fx, fy = 0.62, 0.41
    px, py = place(img, QUAD_PX, fx, fy)
    draw_ball(img, px, py, "yellow")

    result = analyse_frame(img)
    assert result.table_bbox is not None
    # analyse_frame reports in its 960-wide working space, place() drew in 1280-wide space
    scale = pbavision.vision.detect.WORK_WIDTH / img.shape[1]
    cx, cy = result.balls["yellow"]
    assert abs(cx - px * scale) < 4 and abs(cy - py * scale) < 4


def test_rejects_non_table_frames():
    # a title card: big saturated graphic, no table proportions
    img = np.full((720, 1280, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (200, 250), (1080, 470), (200, 160, 60), -1)
    assert analyse_frame(img).table_bbox is None


def test_decoy_outside_table_is_ignored():
    img = draw_table()
    x, y = place(img, QUAD_PX, 0.5, 0.5)
    draw_ball(img, x, y, "red")
    draw_ball(img, 60, 60, "red")  # on the floor, outside the cloth

    result = analyse_frame(img)
    assert result.table_bbox is not None
    assert set(result.balls) == {"red"}


def test_quad_is_four_ordered_corners():
    img = draw_table()
    quad = cloth_quad(cloth_mask(_hsv_of(img)))
    assert quad is not None
    (tl, tr, br, bl) = quad
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] <= bl[1] and tr[1] <= br[1]
    assert np.linalg.norm(tr - tl) > np.linalg.norm(bl - tl)  # long side first
