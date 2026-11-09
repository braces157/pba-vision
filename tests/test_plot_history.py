"""Plot exports keep player, score and onset claims within broadcast evidence."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def plots(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("plot_innings"), importlib.import_module("plot_turns")


def rest(start, end, x=0.5):
    return {"timestamp": start, "end_timestamp": end, "positions": {"white": [x, 0.5], "yellow": [1, 1], "red": [2, 1]}}


def score_data(readings=None, terminal=False):
    return {
        "meta": {"players": ["Alpha", "Beta"], "cue_colors": ["white", "yellow"]},
        "scoreboard": {"readings": readings or [{"t": 0, "scores": [0, 0]}], "reaches_target": terminal, "target": 15},
    }


def test_inning_rest_layouts_need_nearby_matching_counter_evidence(plots):
    innings, _ = plots
    nearby, hidden, wrong_inning = rest(0, 1), rest(5, 6), rest(8, 9)
    counter = {"spans": [{"inning": 1, "start": 0, "end": 10}], "readings": [[0, 1], [0.5, 1], [8, 2], [10, 1]]}
    result = innings.groups({"rests": [nearby, hidden, wrong_inning], "strokes": []}, counter, score_data())
    assert result[0]["rests"] == [nearby]


def test_delayed_score_update_belongs_to_previous_players_visit(plots):
    _, turns = plots
    counters = {
        "spans": [
            {"turn": 1, "inning": 1, "player": 0, "start": 0, "end": 9},
            {"turn": 2, "inning": 1, "player": 1, "start": 10, "end": 19},
        ],
        "readings": [],
    }
    scores = score_data(
        [{"t": 0, "scores": [0, 0]}, {"t": 11, "scores": [2, 0]}, {"t": 20, "scores": [2, 15]}], terminal=True
    )
    history = turns.build_history({"rests": [], "strokes": []}, counters, scores)
    assert history["turns"][0]["points"] == 2
    assert history["turns"][0]["score_update_times"] == [11]
    assert history["turns"][0]["score_end"] == [0, 0]  # tile context is explicitly separate
    assert history["turns"][1]["points"] == 15


def test_nonterminal_or_multi_visit_score_gaps_leave_totals_unknown(plots):
    _, turns = plots
    counters = {
        "spans": [
            {"turn": 1, "inning": 1, "player": 0, "start": 0, "end": 4},
            {"turn": 2, "inning": 1, "player": 1, "start": 5, "end": 9},
            {"turn": 3, "inning": 2, "player": 0, "start": 10, "end": 14},
        ],
        "readings": [],
    }
    partial = turns.turn_totals(counters, score_data()["scoreboard"])
    assert all(turn["points"] is None for turn in partial)
    ambiguous = score_data([{"t": 0, "scores": [0, 0]}, {"t": 15, "scores": [15, 0]}], terminal=True)
    totals = turns.turn_totals(counters, ambiguous["scoreboard"])
    assert totals[0]["points"] is None and totals[2]["points"] is None
    assert "ambiguous" in totals[0]["score_status"]
    assert totals[1]["points"] == 0


def test_other_players_score_change_does_not_resolve_a_delayed_update(plots):
    _, turns = plots
    counters = {"spans": [{"turn": 1, "inning": 1, "player": 0, "start": 0, "end": 4},
                          {"turn": 2, "inning": 1, "player": 1, "start": 5, "end": 9},
                          {"turn": 3, "inning": 2, "player": 0, "start": 10, "end": 14}]}
    scores = score_data([{"t": 0, "scores": [0, 0]}, {"t": 11, "scores": [0, 1]},
                         {"t": 15, "scores": [15, 1]}], terminal=True)
    totals = turns.turn_totals(counters, scores["scoreboard"])
    assert totals[0]["points"] is None and totals[2]["points"] is None
    assert totals[1]["points"] == 1


@pytest.mark.parametrize("mapping", [None, ["white", "white"], ["red", "yellow"], [[], []]])
def test_cue_mapping_must_be_explicit_and_valid(plots, mapping):
    _, turns = plots
    meta = {"players": ["Alpha", "Beta"]}
    if mapping is not None:
        meta["cue_colors"] = mapping
    with pytest.raises(ValueError, match="verified cue colors"):
        turns.verified_cue_colors(meta)


def test_swapped_verified_cues_follow_scoreboard_players(plots):
    _, turns = plots
    counters = {"spans": [{"turn": 1, "inning": 1, "player": 0, "start": 0, "end": 4}], "readings": []}
    history = turns.build_history({"rests": [], "strokes": []}, counters, score_data(), ["yellow", "white"])
    assert history["turns"][0]["cue"] == "yellow"
    assert history["meta"]["cue_colors"] == ["yellow", "white"]
    assert turns.verified_cue_colors(score_data()["meta"]) == ["white", "yellow"]


def test_rest_transition_has_only_an_onset_interval_and_needs_continuous_owner(plots):
    _, turns = plots
    before, after = rest(0, 2), rest(6, 7, x=1.5)
    balls = {"fps": 2, "rests": [before, after], "strokes": [{"from_rest": before, "to_rest": after}]}
    span = {"turn": 1, "inning": 1, "player": 0, "start": 0, "end": 8}
    counters = {"spans": [span], "readings": [[t, 1, 0] for t in range(9)]}
    history = turns.build_history(balls, counters, score_data())
    movement = history["turns"][0]["movements"][0]
    assert movement["start"] == 2
    assert movement["start_observed"] is False
    assert movement["onset_interval"] == [2, 6]
    counters["readings"] = [[2, 1, 0], [6, 1, 0]]  # endpoints alone cannot cover a cut
    history = turns.build_history(balls, counters, score_data())
    assert history["turns"][0]["movements"] == []
    assert history["unresolved_transitions"][0]["onset_interval"] == [2, 6]
    counters["spans"] = [{**span, "end": 4}, {**span, "turn": 2, "player": 1, "start": 5}]
    counters["readings"] = [[2, 1, 0], [6, 1, 1]]
    assert turns.build_history(balls, counters, score_data())["turns"][0]["movements"] == []


def test_score_plot_caption_uses_input_names_and_unknown_total(plots, tmp_path, monkeypatch):
    _, turns = plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = []
    original = plt.Axes.set_title

    def capture(ax, label, *args, **kwargs):
        titles.append(label)
        return original(ax, label, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "set_title", capture)
    history = {
        "meta": {"players": ["Alpha", "Beta"]},
        "turns": [
            {
                "turn": 1,
                "inning": 1,
                "player_name": "Alpha",
                "cue": "white",
                "points": None,
                "start": 0,
                "end": 2,
                "score_start": None,
                "score_end": None,
                "movements": [],
                "layout": None,
            }
        ],
    }
    turns.plot_history(history, tmp_path)
    assert "Alpha–Beta tiles unavailable" in titles[0]
    assert "Turn total unknown" in titles[0]
    assert "Cenet" not in titles[0]
    assert (tmp_path / "turn_01.png").is_file()
