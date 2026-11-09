"""A trajectory needs observed motion; invisible gaps cannot supply bounces."""

import cv2
import numpy as np

from pbavision.vision.trajectory import (
    BallTracker,
    candidates,
    rail_contacts,
    source_discontinuity,
    verify_live_identity,
)


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


def test_fast_rail_reversal_keeps_the_ball_identity():
    tracker = BallTracker(((190., 93.), (761., 377.)))
    for i, x in enumerate([626., 666., 706., 746., 724., 684., 644.]):
        point = tracker.observe([(x, 220., .1), (400., 150., .1)], i / 30)
        assert point == (x, 220.)
    assert tracker.observe([], 7 / 30) is None


def test_two_same_rail_reversals_with_intervening_travel_are_distinct():
    xs = [.2, .13, .06, .03, .06, .13, .06, .03, .06, .13, .2]
    contacts = rail_contacts([[[i / 30, x, .6] for i, x in enumerate(xs)]], "white")
    assert [c["rail"] for c in contacts] == ["left", "left"]
    assert contacts[1]["t"] - contacts[0]["t"] < .4


def test_same_camera_layout_reset_is_a_source_discontinuity():
    frame = np.zeros((300, 600, 3), np.uint8)
    mask = np.ones(frame.shape[:2], np.uint8)
    before = {"white": np.array([100, 100]), "yellow": np.array([200, 100])}
    replay = {"white": [(300, 100, .1)], "yellow": [(400, 100, .1)], "red": []}
    assert source_discontinuity(frame, frame, mask, before, replay)
    continuous = {"white": [(110, 100, .1)], "yellow": [(200, 100, .1)], "red": []}
    assert not source_discontinuity(frame, frame, mask, before, continuous)


def test_source_edit_is_detected_without_camera_geometry_change():
    frame = np.zeros((300, 600, 3), np.uint8)
    changed = frame.copy()
    changed[:, :200] = 200
    found = {"white": [], "yellow": [], "red": []}
    assert source_discontinuity(frame, changed, np.ones(frame.shape[:2], np.uint8), {}, found)


def test_same_camera_edit_splits_views_but_hidden_counter_does_not(monkeypatch):
    from pathlib import Path

    import pbavision.vision.trajectory as trajectory

    frames = [np.zeros((540, 960, 3), np.uint8) for _ in range(3)]
    frames[-1][:] = 200

    class Capture:
        def __init__(self, *args):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def get(self, prop):
            return 30

        def set(self, *args):
            pass

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            pass

    bed = np.array([[190, 93], [761, 93], [761, 377], [190, 377]], np.float32)
    monkeypatch.setattr(trajectory.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(trajectory, "cloth_quad", lambda *a: bed)
    monkeypatch.setattr(trajectory, "candidates", lambda *a: {"white": [(300, 200, .1)], "yellow": [], "red": []})
    monkeypatch.setattr(trajectory.TeamLeagueReader, "active_player", lambda *a: None)
    views = list(trajectory.view_samples(Path("unused"), 0, .1, bed, bed, 1, 1))
    assert [(len(samples), reason) for samples, _, reason in views] == [(2, "source_cut"), (1, "interval_end")]
    assert not any(s["eligible"] for samples, _, _ in views for s in samples)


def test_same_player_from_an_earlier_inning_is_ineligible():
    samples = [{"t": i / 30, "points": {}, "eligible": True} for i in range(31)]
    checked = verify_live_identity(samples, [(0, 1), (.5, 1), (1, 1)], inning=4)
    assert not any(s["eligible"] for s in checked)
    assert all(s["live_identity"] is False for s in checked)


def test_missing_counter_prevents_new_start_but_does_not_claim_a_replay():
    samples = [{"t": i / 30, "points": {}, "eligible": True} for i in range(31)]
    checked = verify_live_identity(samples, [(0, 4), (.5, None), (1, 4)], inning=4)
    assert checked[10]["eligible"] is False
    assert checked[10]["live_identity"] is None
    live = [{"t": i / 30, "points": {}, "eligible": True} for i in range(31)]
    assert all(s["eligible"] for s in verify_live_identity(live, [(0, 4), (.5, 4), (1, 4)], inning=4))
    single = [{"t": 0., "points": {}, "eligible": True}]
    assert verify_live_identity(single, [(0., 4)], inning=4)[0]["eligible"] is False
