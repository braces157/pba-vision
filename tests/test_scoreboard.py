"""Broadcast scores must survive misses, two-point changes and replays."""

import json

import cv2
import numpy as np
import pytest

from pbavision.cli import main
from pbavision.ingest import youtube_id
from pbavision.vision.scoreboard import ScoreTracker, TeamLeagueReader, read_video_scores


def test_live_scores_ignore_glitches_and_replays_without_losing_two_points():
    tracker = ScoreTracker()
    observations = [
        (0, (0, 0)), (0.5, (0, 0)),
        (1, (8, 0)), (1.5, (0, 0)),  # one bad OCR result
        (2, (0, 2)), (2.5, (0, 2)),  # a two-point increase
        (3, (0, 0)), (3.5, (0, 0)),  # replay of the opening
        (4, (1, 2)), (4.5, (1, 2)),
    ]
    for t, scores in observations:
        tracker.observe(t, scores)
    result = tracker.timeline.to_dict()
    assert result["initial"] == [0, 0]
    assert result["final"] == [1, 2]
    assert [event["delta"] for event in result["changes"]] == [[0, 2], [1, 0]]
    assert result["rejected_regressions"] == 2
    assert not result["reaches_target"]


def test_hidden_bug_or_long_gap_breaks_confirmation():
    tracker = ScoreTracker()
    for t, scores in [(0, (0, 0)), (0.5, None), (1, (0, 0)), (10, (0, 0))]:
        tracker.observe(t, scores)
    assert not tracker.timeline.readings
    tracker.observe(10.5, (0, 0))
    assert tracker.timeline.to_dict()["initial"] == [0, 0]


def test_brief_winning_score_needs_two_distinct_samples():
    tracker = ScoreTracker()
    for t, scores in [(0, (0, 0)), (0.5, (0, 0)), (1055.5, (5, 15))]:
        tracker.observe(t, scores)
    assert not tracker.timeline.to_dict()["reaches_target"]
    tracker.observe(1056, (5, 15))
    assert tracker.timeline.to_dict()["final"] == [5, 15]
    assert tracker.timeline.to_dict()["reaches_target"]
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.observe(1056, (5, 15))


def test_ocr_requires_score_tiles_and_accepts_the_narrow_zero():
    # Exercise crop coordinates, confidence and visibility without OCR models.
    reader = TeamLeagueReader.__new__(TeamLeagueReader)
    reader.target, reader.min_confidence = 15, 0.9
    reader.engine = lambda *a, **kw: ([["0", 0.65]], None)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert reader.read(frame) is None
    for x1, y1, x2, y2 in reader.CELLS:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    assert reader.read(frame) == (0, 0)
    reader.engine = lambda *a, **kw: ([["O", 0.99]], None)
    assert reader.read(frame) is None
    reader.engine = lambda *a, **kw: ([["8", 0.65]], None)
    assert reader.read(frame) is None


def test_active_run_counter_identifies_player_without_using_points():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert TeamLeagueReader.active_player(frame) is None
    for x1, y1, x2, y2 in TeamLeagueReader.CELLS:
        frame[y1:y2, x1:x2] = 255
    frame[895:920, 485:503] = 255
    frame[943:968, 485:503] = 255
    assert TeamLeagueReader.active_player(frame) is None
    frame[898:918, 489:497] = 0
    assert TeamLeagueReader.active_player(frame) == 0
    frame[895:920, 485:503] = 255
    frame[946:966, 489:497] = 0
    assert TeamLeagueReader.active_player(frame) == 1
    frame[898:918, 489:497] = 0
    assert TeamLeagueReader.active_player(frame) is None


def test_video_sampling_confirms_a_brief_final_overlay(tmp_path, monkeypatch):
    video = tmp_path / "scores.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    # The winning score is visible for just one second before the bug hides.
    for brightness in [0] * 10 + [128] * 10 + [255] * 10:
        writer.write(np.full((48, 64, 3), brightness, dtype=np.uint8))
    writer.release()

    class Reader:
        def __init__(self, **kwargs):
            pass

        def read(self, frame):
            value = np.mean(frame)
            return (0, 0) if value < 64 else (5, 15) if value < 192 else None

    monkeypatch.setattr("pbavision.vision.scoreboard.TeamLeagueReader", Reader)
    result = read_video_scores(video).to_dict()
    assert result["initial"] == [0, 0]
    assert result["final"] == [5, 15]
    assert result["reaches_target"]
    assert result["samples"] == 6
    assert result["changes"][0]["t"] == 1.0


@pytest.mark.parametrize("value", [
    "jBbjq-VjxM4", "https://www.youtube.com/watch?v=jBbjq-VjxM4&t=10",
    "https://youtu.be/jBbjq-VjxM4", "https://youtube.com/shorts/jBbjq-VjxM4",
])
def test_video_input_accepts_the_shared_url(value):
    assert youtube_id(value) == "jBbjq-VjxM4"


def test_video_input_rejects_other_hosts():
    with pytest.raises(ValueError):
        youtube_id("https://example.com/watch?v=jBbjq-VjxM4")


def test_inning_identity_requires_a_confident_numeric_reading():
    reader = object.__new__(TeamLeagueReader)
    reader.min_confidence = .9
    frame = np.zeros((1080, 1920, 3), np.uint8)
    reader.engine = lambda *a, **k: ([["4", .99]], None)
    assert reader.read_inning(frame) == 4
    reader.engine = lambda *a, **k: ([["4", .4]], None)
    assert reader.read_inning(frame) is None
    reader.engine = lambda *a, **k: ([["SET", .99]], None)
    assert reader.read_inning(frame) is None


@pytest.mark.parametrize("initial,final,exit_code", [((0, 0), (5, 14), 2), ((0, 0), (5, 15), 0),
                                                    ((2, 3), (5, 15), 2)])
def test_scores_command_saves_observations_and_reports_partial_coverage(
    tmp_path, monkeypatch, initial, final, exit_code,
):
    tracker = ScoreTracker()
    for t, scores in [(0, initial), (0.5, initial), (1, final), (1.5, final)]:
        tracker.observe(t, scores)
    monkeypatch.setattr("pbavision.cli.read_video_scores", lambda *a, **kw: tracker.timeline)
    out = tmp_path / "scores.json"
    assert main(["scores", "--local-video", "cached.mp4", "--out", str(out)]) == exit_code
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["scoreboard"]["source"] == "broadcast_scoreboard"
    assert artifact["scoreboard"]["final"] == list(final)
