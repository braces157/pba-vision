from pbavision.vision.episodes import movement_windows
from pbavision.vision.trajectory import trajectory_from_samples


def motion_samples(first=0, last=120, eligible=True):
    return [{"t": i / 30, "frame": i, "eligible": eligible,
             "points": {"white": [.5 + min(1, max(0, i / 30 - 1)) * .6, .6],
                        "yellow": [2., 1.], "red": [1.5, .7]}} for i in range(first, last)]


def test_observed_rest_to_motion_to_rest_is_one_movement():
    samples = motion_samples()
    windows = movement_windows(samples, "white")
    assert len(windows) == 1
    window = windows[0]
    assert window["start_observed"] is True
    assert window["end_settled"] is True
    assert window["end_reason"] == "settled"
    assert samples[window["first"]]["t"] < 1.05
    assert samples[window["last"]]["t"] > 2.6


def test_moving_camera_entry_cannot_claim_observed_strike():
    windows = movement_windows(motion_samples(first=35), "white")
    assert len(windows) == 1
    assert windows[0]["start_observed"] is False
    assert windows[0]["end_settled"] is True


def test_ineligible_replay_cannot_start_a_new_movement():
    assert movement_windows(motion_samples(eligible=False), "white") == []


def test_cut_during_motion_is_incomplete_and_serialization_preserves_gap():
    samples = motion_samples(last=55)
    windows = movement_windows(samples, "white")
    assert len(windows) == 1
    assert windows[0]["end_settled"] is False
    assert windows[0]["end_reason"] == "camera_cut"
    samples[40]["points"].pop("white")
    q = trajectory_from_samples(samples, 30, "white", {
        "start_observed": True, "end_settled": False, "end_reason": "camera_cut",
    })
    assert len(q["paths"]["white"]) == 2
    assert q["paths"]["white"][0][-1][0] == 1.3
    assert q["paths"]["white"][1][0][0] == 1.3667
    assert q["validation"]["status"] == "needs_review"


def test_rest_detection_covers_native_120_fps():
    samples = [{"t": i / 120, "frame": i, "eligible": True,
                "points": {"white": [.5 + min(1, max(0, i / 120 - 1)) * .6, .6],
                           "yellow": [2., 1.], "red": [1.5, .7]}} for i in range(480)]
    windows = movement_windows(samples, "white")
    assert len(windows) == 1
    assert windows[0]["start_observed"]
    assert windows[0]["end_settled"]


def test_hidden_departure_is_a_fragment_with_unseen_start():
    samples = motion_samples()
    for s in samples[29:39]:
        s["points"].pop("white")
    windows = movement_windows(samples, "white")
    assert len(windows) == 1
    assert windows[0]["start_observed"] is False


def test_slow_visible_fragment_is_not_omitted():
    samples = [{"t": i / 30, "frame": i, "eligible": True,
                "points": {"white": [.5 + i / 30 * .2, .6], "yellow": [2., 1.], "red": [1.5, .7]}}
               for i in range(90)]
    windows = movement_windows(samples, "white")
    assert len(windows) == 1
    assert windows[0]["start_observed"] is False


def test_singleton_endpoint_is_kept_without_connecting_a_gap():
    samples = motion_samples(last=10)
    for s in samples[5:9]:
        s["points"].pop("white")
    q = trajectory_from_samples(samples, 30, "white", {
        "start_observed": True, "end_settled": False, "end_reason": "camera_cut",
    })
    assert len(q["paths"]["white"][-1]) == 1
    assert q["paths"]["white"][-1][0][0] == .3


def test_confirmed_replay_counter_ends_a_live_window_before_replay_positions():
    samples = motion_samples(last=90)
    for sample in samples[50:]:
        sample["live_identity"] = False
    windows = movement_windows(samples, "white")
    assert len(windows) == 1
    assert windows[0]["last"] == 49
    assert windows[0]["end_reason"] == "timeline_mismatch"
    assert not windows[0]["end_settled"]


def test_temporarily_missing_counter_keeps_continuous_live_motion():
    samples = motion_samples()
    for sample in samples[50:65]:
        sample["live_identity"], sample["eligible"] = None, False
    windows = movement_windows(samples, "white")
    assert len(windows) == 1
    assert windows[0]["end_settled"] is True


def test_incoming_roll_does_not_become_the_next_players_movement(monkeypatch):
    from pathlib import Path

    import numpy as np

    import pbavision.vision.trajectory as trajectory

    incoming = motion_samples(first=35, last=55)
    own_visit = motion_samples()
    for sample in own_visit:
        sample["t"] += 4
        sample["frame"] += 120
    monkeypatch.setattr(trajectory, "view_samples", lambda *a: iter([
        (incoming, 30, "camera_cut"), (own_visit, 30, "camera_cut"),
    ]))
    turn = {"start": 1, "end": 8, "inning": 2, "player": 1, "cue": "white"}
    movements = trajectory.scan_turn(Path("unused"), turn, np.zeros((4, 2)), np.zeros((4, 2)))
    assert len(movements) == 1
    assert movements[0]["start"] > 4
    assert len(turn["unassigned_movements"]) == 1
    assert turn["unassigned_movements"][0]["trajectory"]["completeness"]["start_observed"] is False
