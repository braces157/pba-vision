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
