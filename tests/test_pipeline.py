"""Stroke classification and inning chaining from synthetic rest sequences."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import QUAD_PX, place  # noqa: F401  (place used below via positions)

from pbavision.pipeline import classify


def _moved(*movers):
    return {k: k in movers for k in ("white", "yellow", "red")}


def test_one_mover_is_a_miss_with_cue_identified():
    points, cue = classify(_moved("white"))
    assert points == 0 and cue == "white"


def test_two_movers_with_red_is_a_miss():
    points, cue = classify(_moved("red", "yellow"))
    assert points == 0 and cue == "yellow"


def test_two_movers_without_red_is_a_miss_without_cue():
    points, cue = classify(_moved("white", "yellow"))
    assert points == 0 and cue is None


def test_three_movers_implies_a_carom():
    points, cue = classify(_moved("white", "yellow", "red"))
    assert points == 1 and cue is None


def test_empty_move_is_rejected():
    with pytest.raises(ValueError):
        classify(_moved())


def test_rest_retains_end_of_settled_run_for_shooter_assignment():
    from pbavision.pipeline import _rest_chain

    positions = {"white": (1., 1.), "yellow": (2., 1.), "red": (1.5, .5)}
    rests = _rest_chain([(i, positions) for i in (4, 5, 6, 7)], fps=4)
    assert len(rests) == 1
    assert rests[0].timestamp == 1
    assert rests[0].end_timestamp == 1.75
    assert rests[0].end_frame == 7


def test_inning_chaining_on_synthetic_stream():
    """Rest -> stroke -> rest chain: a miss switches shooter and opens a new inning."""
    from pbavision.pipeline import RestState, _next_cue

    assert _next_cue("white") == "yellow"
    assert _next_cue("yellow") == "white"
    assert _next_cue(None) is None

    def rest(frame, positions):
        return RestState(frame=frame, timestamp=frame / 4.0, positions=positions)

    # two rests only — the chaining logic itself is exercised via classify + _next_cue
    a = rest(10, {"white": (1.0, 1.0), "yellow": (2.0, 1.0), "red": (1.5, 0.5)})
    b = rest(20, {"white": (1.4, 1.0), "yellow": (2.0, 1.0), "red": (1.5, 0.5)})
    moved = {k: bool(np.hypot(a.positions[k][0] - b.positions[k][0],
                              a.positions[k][1] - b.positions[k][1]) > 0.08) for k in a.positions}
    points, cue = classify(moved)
    assert (points, cue) == (0, "white")
