"""A trajectory needs observed motion; invisible gaps cannot supply bounces."""

import cv2
import numpy as np

from pbavision.vision.trajectory import BallTracker, candidates, rail_contacts


def test_rail_contact_requires_a_visible_direction_reversal():
    xs = [.20, .14, .08, .035, .08, .14, .20]
    path = [[i / 30, x, .6] for i, x in enumerate(xs)]
    hits = rail_contacts([path], "white")
    assert len(hits) == 1
    assert hits[0]["rail"] == "left"
    assert hits[0]["t"] == .1
    assert rail_contacts([path[:4], path[4:]], "white") == []


def test_change_of_direction_in_middle_of_table_is_not_a_cushion():
    path = [[i / 30, x, .6] for i, x in enumerate([1.2, 1.1, 1., .9, 1., 1.1, 1.2])]
    assert rail_contacts([path], "white") == []


def test_stationary_ball_near_rail_is_not_counted_as_repeated_contacts():
    path = [[i / 30, .03 + .001 * (i % 2), .6] for i in range(30)]
    assert rail_contacts([path], "white") == []


def test_tracker_does_not_output_predictions_when_detection_is_missing():
    tracker = BallTracker()
    assert tracker.observe([(200., 200., .1)], 0) == (200., 200.)
    assert tracker.observe([(210., 200., .1)], 1 / 30) == (210., 200.)
    assert tracker.observe([], 2 / 30) is None
    assert tracker.observe([(800., 200., 0)], 3 / 30) is None
    assert tracker.observe([(240., 200., .5), (800., 200., 0)], 4 / 30) == (240., 200.)


def test_white_ball_touching_cue_keeps_its_centre_and_rejects_neutral_logo():
    hsv = np.full((100, 180, 3), (100, 180, 160), np.uint8)
    cv2.line(hsv, (50, 50), (115, 50), (30, 55, 225), 3)
    cv2.circle(hsv, (50, 50), 6, (30, 70, 245), -1)
    cv2.circle(hsv, (145, 25), 7, (0, 10, 245), -1)
    found = candidates(hsv, np.ones(hsv.shape[:2], np.uint8))["white"]
    assert len(found) == 1
    assert np.linalg.norm(np.array(found[0][:2]) - [50, 50]) < 2


def test_printed_spots_do_not_hide_white_ball_and_skin_is_not_a_ball():
    hsv = np.full((100, 180, 3), (100, 180, 160), np.uint8)
    cv2.circle(hsv, (50, 50), 6, (30, 70, 245), -1)
    cv2.circle(hsv, (52, 47), 1, (0, 0, 60), -1)
    cv2.circle(hsv, (48, 52), 1, (0, 0, 60), -1)
    cv2.ellipse(hsv, (130, 50), (13, 7), 20, 0, 360, (12, 105, 205), -1)
    found = candidates(hsv, np.ones(hsv.shape[:2], np.uint8))["white"]
    assert len(found) == 1
    assert np.linalg.norm(np.array(found[0][:2]) - [50, 50]) < 2


def test_tracker_does_not_switch_identity_after_long_occlusion():
    tracker = BallTracker()
    tracker.observe([(200., 200., 0)], 0)
    assert tracker.observe([(700., 200., 0)], .5) is None
    assert tracker.observe([(205., 200., .5), (700., 200., 0)], .6) == (205., 200.)


def test_blurred_candidate_can_continue_identity_but_cannot_initialize_it():
    hsv = np.full((100, 180, 3), (100, 180, 160), np.uint8)
    cv2.ellipse(hsv, (80, 50), (15, 5), 0, 0, 360, (50, 25, 210), -1)
    found = candidates(hsv, np.ones(hsv.shape[:2], np.uint8))["white"]
    assert len(found) == 1
    assert BallTracker().observe(found, 0) is None
    tracker = BallTracker()
    tracker.observe([(50., 50., .1)], 0)
    point = tracker.observe(found, 1 / 30)
    assert point is not None
    assert np.linalg.norm(np.asarray(point) - [80, 50]) < 2


def test_slow_shallow_rail_contact_is_detected():
    path = [[i / 30, .03 + .07 * abs(i / 30 - .4), .6] for i in range(25)]
    hits = rail_contacts([path], "white")
    assert len(hits) == 1
    assert hits[0]["rail"] == "left"
    assert abs(hits[0]["t"] - .4) <= 1 / 30


def test_short_bracketed_contact_is_labeled_and_long_gap_not_inferred():
    path = [[i / 30, .03 + .02 * abs(i - 7), .6] for i in range(15)]
    segments = [path[:7], path[8:]]
    assert rail_contacts(segments, "white") == []
    hits = rail_contacts(segments, "white", max_gap=.07)
    assert len(hits) == 1
    assert hits[0]["evidence"] == "bracketed_gap"
    assert rail_contacts([path[:5], path[10:]], "white", max_gap=.07) == []
