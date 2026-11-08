"""Broadcast counter boundaries must not turn camera gaps into innings."""

from pbavision.align.broadcast import inning_for_interval, inning_spans, turn_spans


def test_inning_changes_need_confirmation_and_leave_unknown_time_unassigned():
    spans = inning_spans([(0, 1), (.5, 1), (1, 1), (2, 9), (3, 2), (3.5, 2), (4, 2)])
    assert [(s.inning, s.start, s.end) for s in spans] == [(1, 0, 1), (2, 3, 4)]
    assert inning_for_interval(0, 1, spans) == 1
    assert inning_for_interval(.5, 3.5, spans) is None
    assert inning_for_interval(1.5, 2.5, spans) is None


def test_replayed_counters_do_not_reopen_earlier_innings():
    readings = [(0, 1), (.5, 1), (1, 2), (1.5, 2), (2, 1), (2.5, 1), (3, 2), (3.5, 2)]
    spans = inning_spans(readings)
    assert [s.inning for s in spans] == [1, 2]
    assert spans[-1].end == 3.5


def test_distant_single_readings_do_not_confirm_an_inning():
    assert inning_spans([(0, 1), (10, 1)]) == []


def test_turns_follow_active_player_including_scoreless_visits():
    readings = [(0, 1, 1), (.25, 1, 1), (1, 1, 0), (1.25, 1, 0),
                (2, 2, 1), (2.25, 2, 1)]
    turns = turn_spans(readings)
    assert [(t.turn, t.inning, t.player) for t in turns] == [(1, 1, 1), (2, 1, 0), (3, 2, 1)]


def test_replayed_player_and_one_frame_glitches_do_not_create_turns():
    readings = [(0, 1, 1), (.25, 1, 1), (.5, 1, 0), (.75, 1, 1),
                (1, 1, 0), (1.25, 1, 0), (2, 1, 1), (2.25, 1, 1), (3, 1, 0)]
    turns = turn_spans(readings)
    assert [(t.inning, t.player) for t in turns] == [(1, 1), (1, 0)]
    assert turns[-1].end == 3
