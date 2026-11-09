"""The static renderer preserves observation and scoring boundaries."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_match_history.py"
SPEC = importlib.util.spec_from_file_location("render_match_history", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_gaps_and_invalid_samples_never_join():
    paths = renderer.observed_paths(
        {
            "paths": {
                "white": [
                    [[0, 0, 0], [0.1, 1, 0]],
                    [[1, 1, 1], [float("nan"), 1, 1], [2, 0, 1]],
                ]
            }
        }
    )
    assert paths["white"] == [[[0, 0, 0], [0.1, 1, 0]], [[1, 1, 1]], [[2, 0, 1]]]
    assert paths["yellow"] == []


def test_turn_points_are_not_movement_points():
    turn = {"points": 10, "score_start": [2, 0], "score_end": [2, 10]}
    text = renderer.score_context({"meta": {"players": ["A", "B"]}}, turn, {})
    assert "Turn total: +10" in text
    assert "Movement score:" not in text
    assert "Movement score: +1" in renderer.score_context({}, turn, {"score_points": 1})


def test_unknown_turn_score_and_lagged_tile_context_can_render():
    turn = {"points": None, "observed_points": 2, "score_start": None, "score_end": [0, 0]}
    text = renderer.score_context({}, turn, {})
    assert "Score tiles at turn bounds" in text
    assert "Turn total: unknown" in text
    assert "unavailable" in text


def test_render_index_includes_every_turn_and_path_card(tmp_path, monkeypatch):
    plotted = []
    original = renderer.plt.Axes.plot

    def capture(ax, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], list):
            plotted.append((args[0], args[1]))
        return original(ax, *args, **kwargs)

    monkeypatch.setattr(renderer.plt.Axes, "plot", capture)
    base = {
        "inning": 1,
        "player_name": "A",
        "cue": "yellow",
        "points": 0,
        "score_start": [0, 0],
        "score_end": [0, 0],
        "start": 0,
        "end": 4,
    }
    movement = {
        "start": 0,
        "end": 4,
        "trajectory": {
            "duration": 4,
            "paths": {"yellow": [[[0, 0.1, 0.2], [1, 1, 0.2]], [[3, 1, 1], [4, 0.1, 1]]]},
            "status": "Camera cut",
            "completeness": {"start_observed": True, "end_settled": False},
            "validation": {"status": "needs_review", "reasons": ["Missing video"]},
        },
    }
    history = {"turns": [{**base, "turn": 1, "movements": [movement]}, {**base, "turn": 2, "movements": []}]}
    index = renderer.render_history(history, tmp_path)
    assert index["coverage_counts"] == {
        "turns": 2,
        "observed_movements": 1,
        "movements_with_paths": 1,
        "movements_without_paths": 0,
        "turns_with_paths": 1,
        "turns_without_paths": 1,
        "pngs": 2,
    }
    assert index["turns"][0]["movements"][0]["segments"]["yellow"] == 2
    assert ([.1, 1], [.2, .2]) in plotted
    assert ([1, .1], [1, 1]) in plotted
    assert ([.1, 1, 1, .1], [.2, .2, 1, 1]) not in plotted
    assert index["turns"][0]["movements"][0]["completeness"]["end_settled"] is False
    assert all((tmp_path / name).is_file() for turn in index["turns"] for name in turn["paths"])
    assert json.loads((tmp_path / "index.json").read_text())["turns"][1]["turn"] == 2
    footer = " ".join(
        renderer.status_lines(movement["trajectory"], renderer.observed_paths(movement["trajectory"]), True)
    )
    assert "Settled end not observed" in footer
    assert "Start observed" in footer


def test_before_after_snapshots_do_not_move_hidden_balls_to_another_time():
    paths = {"white": [[[0, .5, .2], [1, 1., .4]]], "yellow": [[[0, 2., 1.]]], "red": []}
    assert renderer.layout_points(paths, 1, .05) == {"white": [1., .4]}
