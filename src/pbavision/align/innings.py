"""Parse the terminal's inning sheet.

The sheet is two tables: a header naming both players (the breaker is marked
선공 in Korean or (First) in English) and a body with one row per inning. Each
row starts with the inning number, then three cells per player — 1-point
shots, 2-point shots, running total.

The parser mirrors the companion scoreboard site's ``parseSetInnings`` so
both projects read the sheet identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InningSide:
    """One player's inning: shots worth one point, shots worth two, running total."""

    one_point: int
    two_point: int
    total: int

    @property
    def points(self) -> int:
        return self.one_point + 2 * self.two_point


@dataclass
class InningSheet:
    players: list[str]
    breaker: int | None
    rows: list[list[InningSide]] = field(default_factory=list)

    @property
    def inning_count(self) -> int:
        """Rows are shared between players: one row is one inning per player."""
        return len(self.rows)

    def final_totals(self) -> list[int]:
        """Running total of each player after the last row."""
        if not self.rows:
            return []
        return [side.total for side in self.rows[-1]]

    def per_inning_points(self) -> list[list[int]]:
        """Points scored in each row, per player."""
        return [[side.points for side in row] for row in self.rows]


def _cells(fragment: str) -> list[str]:
    text = re.sub(r"<!--[\s\S]*?-->", "", fragment)
    parts = re.split(r"<[^>]+>", text)
    return [p.strip() for p in parts if p.strip()]


def _number(cell: str) -> int | None:
    cleaned = cell.replace("&nbsp;", "").strip()
    return int(cleaned) if cleaned.isdigit() else None


def parse_set_innings(html: str) -> InningSheet:
    sheet = InningSheet(players=[], breaker=None)
    if not html:
        return sheet

    source = re.sub(r"<!--[\s\S]*?-->", "", html)

    head_match = re.search(r"<thead\b[^>]*>([\s\S]*?)</thead>", source, re.IGNORECASE)
    head = head_match.group(1) if head_match else ""
    # Player cells sit in <th> blocks; the name may carry nested spans (the
    # breaker mark is one), so tag-strip the whole block rather than matching
    # up to the first </span>.
    for block in re.split(r"<th\b", head, flags=re.IGNORECASE)[1:]:
        block = block.split("</th>")[0]
        block = re.sub(r"^[^>]*>", "", block)  # drop the <th ...> attributes left by the split
        if "player_name" not in block:
            continue
        is_breaker = bool(re.search(r"선공|\(First\)", block, re.IGNORECASE))
        name = re.sub(r"<[^>]+>", "", block)
        name = re.sub(r"(?:\(\s*)?(?:선공|First)(?:\s*\))?", "", name, flags=re.IGNORECASE)
        name = name.replace("&nbsp;", " ").strip()
        if not name:
            continue
        if is_breaker:
            sheet.breaker = len(sheet.players)
        sheet.players.append(name)

    body_match = re.search(r"<tbody\b[^>]*>([\s\S]*?)</tbody>", source, re.IGNORECASE)
    body = body_match.group(1) if body_match else source

    for row_html in re.findall(r"<tr\b[^>]*>([\s\S]*?)</tr>", body, re.IGNORECASE):
        values = _cells(row_html)
        if len(values) < 4:
            continue
        inning = _number(values[0])
        # The inning number is the only cell outside the per-player groups of three.
        if inning is None or (len(values) - 1) % 3 != 0:
            continue
        sides: list[InningSide] = []
        for index in range(1, len(values) - 1, 3):
            sides.append(
                InningSide(
                    one_point=_number(values[index]) or 0,
                    two_point=_number(values[index + 1]) or 0,
                    total=_number(values[index + 2]) or 0,
                )
            )
        if len(sides) >= 2:
            sheet.rows.append(sides)

    if not sheet.players:
        width = len(sheet.rows[0]) if sheet.rows else 2
        sheet.players = [""] * width
    return sheet
