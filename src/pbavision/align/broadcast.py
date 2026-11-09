"""Conservative inning intervals from the broadcast's inning counter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InningSpan:
    inning: int
    start: float
    end: float


@dataclass(frozen=True)
class TurnSpan:
    turn: int
    inning: int
    player: int
    start: float
    end: float


def turn_spans(readings: list[tuple[float, int, int]], max_gap: float = 1.0) -> list[TurnSpan]:
    """Confirm player changes from the active run counter, including scoreless turns."""
    result: list[TurnSpan] = []
    current: tuple[int, int] | None = None
    completed: set[tuple[int, int]] = set()
    pending: tuple[float, tuple[int, int]] | None = None
    start, end, previous_t = 0.0, 0.0, -1.0
    for t, inning, player in readings:
        if t <= previous_t or t < 0 or inning < 1 or player not in (0, 1):
            raise ValueError("turn readings require increasing times, positive innings and player 0 or 1")
        previous_t = t
        key = (inning, player)
        if key == current:
            end, pending = t, None
        elif key not in completed and (current is None or current[0] <= inning <= current[0] + 1):
            if pending is not None and pending[1] == key and t - pending[0] <= max_gap:
                if current is not None:
                    result.append(TurnSpan(len(result) + 1, *current, start, end))
                    completed.add(current)
                current, start, end = key, pending[0], t
                pending = None
            else:
                pending = (t, key)
        else:
            pending = None
    if current is not None:
        result.append(TurnSpan(len(result) + 1, *current, start, end))
    return result


def inning_spans(readings: list[tuple[float, int]], max_gap: float = 1.0) -> list[InningSpan]:
    """Confirm two nearby counter readings; leave transition gaps unassigned.

    Replayed earlier innings cannot reopen a finished interval. A jump over
    an inning is ignored rather than inventing a missing boundary.
    """
    spans: list[InningSpan] = []
    current, start, end = 0, 0.0, 0.0
    pending: tuple[float, int] | None = None
    previous_t = -1.0
    for t, inning in readings:
        if t <= previous_t or t < 0 or inning < 1:
            raise ValueError("inning readings must be positive and ordered by increasing time")
        previous_t = t
        if inning == current:
            end = t
            pending = None
        elif inning == current + 1:
            if pending is not None and pending[1] == inning and t - pending[0] <= max_gap:
                if current:
                    spans.append(InningSpan(current, start, end))
                current, start, end = inning, pending[0], t
                pending = None
            else:
                pending = (t, inning)
        else:
            pending = None
    if current:
        spans.append(InningSpan(current, start, end))
    return spans


def inning_for_interval(start: float, end: float, spans: list[InningSpan]) -> int | None:
    """Assign a rest transition only when both endpoints share a known inning."""
    if end < start:
        raise ValueError("interval end precedes its start")
    return next((s.inning for s in spans if s.start <= start <= end <= s.end), None)
