"""Inning sheet parsing against a fixture mirroring the terminal's markup."""

from __future__ import annotations

from pbavision.align.innings import parse_set_innings

# Structure copied in shape (not content) from contest_result_set_record.asp:
# header names both players, the breaker's cell carries (선공), body rows are
# [inning, (1P, 2P, total) per player].
FIXTURE = """
<div class="anal_record"><table id="tbl_list1"><thead>
  <tr>
    <th rowspan="2"><span class="t14">Inning</span></th>
    <th colspan="3"><img id='img_flag' src='flag.jpg' />
      &nbsp;<span class="player_name">체네트&nbsp;</span></th>
    <th colspan="3"><img id='img_flag' src='flag.jpg' />
      &nbsp;<span class="player_name">사이그너&nbsp;<span style='color:#ff6600'>(선공)</span></span></th>
  </tr>
  <tr><td>1P</td><td>2P</td><td>Total</td><td>1P</td><td>2P</td><td>Total</td></tr>
</thead></table></div>
<div class="anal_record"><table id="tbl_list2"><tbody>
  <tr><td class="center">1</td><td>2</td><td>0</td><td><span class="keyf0_1">2</span></td>
      <td>2</td><td>0</td><td><span>2</span></td></tr>
  <tr><td>2</td><td>1</td><td>0</td><td>3</td><td>2</td><td>0</td><td>4</td></tr>
  <tr><td>3</td><td>3</td><td>0</td><td>6</td><td>2</td><td>0</td><td>6</td></tr>
  <tr><td>4</td><td>0</td><td>0</td><td>6</td><td>0</td><td>0</td><td>6</td></tr>
  <tr><td>5</td><td>0</td><td>0</td><td>6</td><td>1</td><td>0</td><td>7</td></tr>
  <tr><td>6</td><td>1</td><td>0</td><td>7</td><td>4</td><td>2</td><td>15</td></tr>
</tbody></table></div>
"""


def test_players_and_breaker():
    sheet = parse_set_innings(FIXTURE)
    assert sheet.players == ["체네트", "사이그너"]
    assert sheet.breaker == 1  # 사이그너 carried the (선공) mark in this fixture


def test_rows_and_totals():
    sheet = parse_set_innings(FIXTURE)
    assert sheet.inning_count == 6
    assert sheet.final_totals() == [7, 15]
    per_inning = sheet.per_inning_points()
    assert per_inning[0] == [2, 2]          # row 1: both players scored 2
    assert per_inning[5] == [1, 8]          # row 6: 1 one-pointer + 2 two-pointers


def test_empty_input_is_not_fatal():
    sheet = parse_set_innings("")
    assert sheet.inning_count == 0
    assert sheet.players == []
