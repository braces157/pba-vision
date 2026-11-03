"""Compare a reconstruction against the official inning sheet.

The sheet is ground truth for three quantities the rest-position data can
reach: how many innings were played, how many points each player scored in
total, and how the points spread across innings. It is *not* ground truth for
the number of strokes — the terminal records points, not shots.

Two known gaps show up in every comparison and are reported, not hidden:

  overcount  a carom that contacts both object balls but completes fewer than
             three cushions scores nothing, yet three moving balls imply one
  undercount the terminal's 2-point shots (cushion-first caroms) are one
             carom each but two points on the sheet
"""

from __future__ import annotations

from dataclasses import dataclass

from .align.innings import InningSheet
from .pipeline import Reconstruction


@dataclass
class Validation:
    innings_official: int
    innings_estimated: int
    official_totals: list[int]
    estimated_totals: list[int]
    official_points_per_inning: list[list[int]]
    estimated_points_per_inning: list[int]
    strokes_detected: int
    caroms_detected: int

    @property
    def totals_gap(self) -> int:
        return sum(self.estimated_totals) - sum(self.official_totals)

    def summary(self) -> str:
        lines = [
            f"innings: official {self.innings_official}, estimated {self.innings_estimated}",
            f"final:   official {self.official_totals}, estimated caroms {self.estimated_totals}",
            f"strokes: {self.strokes_detected} detected, {self.caroms_detected} imply a carom",
            f"gap:     {self.totals_gap:+d} points "
            "(3-cushion rule failures overcount, 2-point shots undercount)",
        ]
        return "\n".join(lines)


def validate(reconstruction: Reconstruction, sheet: InningSheet) -> Validation:
    strokes = reconstruction.strokes

    # Inning estimates are per player; the sheet counts one row as one inning
    # per player, so compare against the larger of the two players' counts.
    per_player_innings = [0, 0]
    for stroke in strokes:
        if stroke.cue and stroke.inning_estimate:
            per_player_innings[0 if stroke.cue == "white" else 1] = max(
                per_player_innings[0 if stroke.cue == "white" else 1], stroke.inning_estimate
            )
    innings_estimated = max(per_player_innings) if strokes else 0

    # Estimated points per inning, matched to the sheet by inning number of the
    # scoring player. The sheet orders rows by round; row i holds each player's
    # i-th inning. CV innings are indexed per cue colour the same way.
    rows = sheet.inning_count
    n_players = len(sheet.players)
    per_inning: list[list[int]] = [[0] * rows for _ in range(n_players)]
    colour_index = {"white": 0, "yellow": 1}
    for stroke in strokes:
        if stroke.estimated_points and stroke.cue and stroke.inning_estimate:
            player = colour_index.get(stroke.cue, 0)
            inning = stroke.inning_estimate - 1
            if player < n_players and inning < rows:
                per_inning[player][inning] += stroke.estimated_points

    estimated_by_row: list[int] = []
    for i in range(rows):
        estimated_by_row.append(sum(per_inning[p][i] for p in range(n_players)))

    official_per_row = [[side.points for side in row] for row in sheet.rows]

    return Validation(
        innings_official=sheet.inning_count,
        innings_estimated=innings_estimated,
        official_totals=sheet.final_totals(),
        estimated_totals=[sum(per_inning[p]) for p in range(len(sheet.players))] if strokes else [],
        official_points_per_inning=official_per_row,
        estimated_points_per_inning=estimated_by_row,
        strokes_detected=len(strokes),
        caroms_detected=sum(1 for s in strokes if s.estimated_points),
    )
