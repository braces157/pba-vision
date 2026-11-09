"""Build a chronological match history with one player's turn per graph.

    python scripts/plot_turns.py data/balls_jBbjq-VjxM4.json \
        --turns data/turns_jBbjq-VjxM4.json --scores data/scores_jBbjq-VjxM4.json \
        --cue-colors yellow white \
        --out data/turn_graphs

Supply verified cue colors in scoreboard player order; do not infer them from
screen position. Omit the option only when score metadata already records them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_innings import COLOURS, clock, score_at, valid_rest


def verified_cue_colors(meta: dict, override: list[str] | None = None) -> list[str]:
    colors = override if override is not None else meta.get("cue_colors")
    if (
        not isinstance(colors, (list, tuple))
        or len(colors) != 2
        or any(color not in ("white", "yellow") for color in colors)
        or set(colors) != {"white", "yellow"}
    ):
        raise ValueError("verified cue colors required: meta.cue_colors or --cue-colors, in scoreboard player order")
    if len(meta.get("players", [])) != 2:
        raise ValueError("two scoreboard-ordered player names are required")
    return list(colors)


def transition_owner(start: float, end: float, turns: dict) -> int | None:
    """Require counter coverage across the entire uncertain onset interval."""
    if end <= start:
        return None
    candidates = [index for index, span in enumerate(turns["spans"]) if span["start"] <= start <= end <= span["end"]]
    if len(candidates) != 1:
        return None
    index = candidates[0]
    span = turns["spans"][index]
    readings = turns["readings"]
    before = [r for r in readings if r[0] <= start]
    after = [r for r in readings if r[0] >= end]
    if not before or not after:
        return None
    support = [before[-1], *[r for r in readings if start < r[0] < end], after[0]]
    if (
        start - support[0][0] > 1
        or support[-1][0] - end > 1
        or any(tuple(r[1:]) != (span["inning"], span["player"]) for r in support)
        or any(b[0] - a[0] > 1 for a, b in zip(support, support[1:]))
    ):
        return None
    return index


def turn_totals(turns: dict, scoreboard: dict) -> list[dict]:
    """Attribute score changes by player, including lag into the opponent's visit.

    A change bracket touching multiple visits by that player is unresolved.
    The timeline records changes, not stable observations: absent changes can
    establish a whole-turn zero only after the terminal score closes a complete
    visit sequence. Nonterminal observations remain lower-bound evidence.
    """
    spans, readings = turns["spans"], scoreboard["readings"]
    totals = [{"points": None, "observed_points": 0, "score_update_times": []} for _ in spans]
    uncertain = set()
    checkpoints = [readings[0]["t"], readings[0]["t"]] if readings else [0, 0]
    for before, after in zip(readings, readings[1:]):
        for player, (old, new) in enumerate(zip(before["scores"], after["scores"])):
            delta = new - old
            if delta <= 0:
                if delta < 0:
                    uncertain.update(i for i, s in enumerate(spans) if s["player"] == player)
                continue
            preceding = [i for i, s in enumerate(spans) if s["player"] == player and s["start"] <= checkpoints[player]]
            candidates = set(preceding[-1:])
            candidates.update(
                i for i, s in enumerate(spans)
                if s["player"] == player and checkpoints[player] < s["start"] <= after["t"]
            )
            if len(candidates) == 1:
                owner = candidates.pop()
                totals[owner]["observed_points"] += delta
                totals[owner]["score_update_times"].append(after["t"])
            else:
                uncertain.update(candidates or [i for i, s in enumerate(spans) if s["player"] == player])
            checkpoints[player] = after["t"]
    complete_order = bool(spans) and spans[0]["inning"] == 1
    for previous, current in zip(spans, spans[1:]):
        complete_order &= current["player"] == 1 - previous["player"] and current["inning"] == previous["inning"] + int(
            current["player"] == spans[0]["player"]
        )
    closed = (
        complete_order
        and bool(readings)
        and readings[0]["scores"] == [0, 0]
        and readings[0]["t"] <= spans[0]["end"]
        and scoreboard.get("reaches_target") is True
        and max(readings[-1]["scores"]) == scoreboard.get("target", 15)
        and readings[-1]["t"] >= spans[-1]["start"]
    )
    for index, total in enumerate(totals):
        if closed and index not in uncertain:
            total["points"] = total["observed_points"]
            total["score_status"] = "reconciled from terminal cumulative score and unique player-visit ownership"
        else:
            total["score_status"] = (
                "ambiguous score-update ownership"
                if index in uncertain
                else "whole-turn total unresolved without terminal score closure"
            )
    return totals


def build_history(balls: dict, turns: dict, scores: dict, cue_colors: list[str] | None = None) -> dict:
    players = scores["meta"]["players"]
    colors = verified_cue_colors(scores["meta"], cue_colors)
    score_readings = scores["scoreboard"]["readings"]
    totals = turn_totals(turns, scores["scoreboard"])
    history = []
    unresolved = []
    owned = {}
    for stroke_index, stroke in enumerate(balls["strokes"]):
        before, after = stroke["from_rest"], stroke["to_rest"]
        if before.get("end_timestamp") is None or after.get("end_timestamp") is None:
            raise ValueError("reconstruct again to retain rest end times for player assignment")
        owner = transition_owner(before["end_timestamp"], after["timestamp"], turns)
        if owner is None:
            unresolved.append(
                {
                    "stroke_index": stroke_index,
                    "onset_interval": [before["end_timestamp"], after["timestamp"]],
                    "reason": "rest endpoints do not establish an observed onset or continuous turn owner",
                }
            )
        else:
            owned[stroke_index] = owner
    for index, span in enumerate(turns["spans"]):
        player = span["player"]
        start_score = score_at(span["start"], score_readings)
        end_score = score_at(span["end"], score_readings)
        movements = []
        for stroke_index, stroke in enumerate(balls["strokes"]):
            before, after = stroke["from_rest"], stroke["to_rest"]
            if owned.get(stroke_index) != index:
                continue
            if not valid_rest(before) or not valid_rest(after):
                continue
            # Short contour flickers are not a reliable settled endpoint.
            if any(r["end_timestamp"] - r["timestamp"] < 0.5 for r in (before, after)):
                continue
            if after["timestamp"] - before["end_timestamp"] < 1:
                continue
            movements.append(
                {
                    "start": before["end_timestamp"],
                    "end": after["timestamp"],
                    "start_observed": False,
                    "onset_interval": [before["end_timestamp"], after["timestamp"]],
                    "start_meaning": "earliest possible onset bound, not an observed strike",
                    "from": before["positions"],
                    "to": after["positions"],
                }
            )
        layouts = [
            r
            for r in balls["rests"]
            if valid_rest(r)
            and span["start"] <= r["timestamp"] <= span["end"]
            and r.get("end_timestamp") is not None
            and r["end_timestamp"] - r["timestamp"] >= 0.5
            and any(
                abs(r["timestamp"] - reading[0]) <= 1 and tuple(reading[1:]) == (span["inning"], player)
                for reading in turns["readings"]
            )
        ]
        history.append(
            {
                **span,
                "player_name": players[player],
                "cue": colors[player],
                "score_start": start_score,
                "score_end": end_score,
                **totals[index],
                "movements": movements,
                "layout": layouts[0]["positions"] if layouts else None,
            }
        )
    return {
        "schema": "pba-vision/turn-history@0.1",
        "meta": {**scores["meta"], "cue_colors": colors},
        "turns": history,
        "unresolved_transitions": unresolved,
        "score_source": "broadcast cumulative score tiles",
        "turn_source": "broadcast active run counter and inning counter",
        "geometry": "fixed calibration per stable camera; meters on outer table rectangle",
        "movement_meaning": "observed rest-to-rest displacement, not the complete path or guaranteed shot list",
    }


def plot_history(history: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for turn in history["turns"]:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.add_patch(Rectangle((.13, .13), 2.84, 1.42, fill=False, ec="#333333", ls="--", lw=1))
        for ball, colour in COLOURS.items():
            points = []
            for i, move in enumerate(turn["movements"]):
                x0, y0 = move["from"][ball]
                x1, y1 = move["to"][ball]
                points.extend([(x0, y0), (x1, y1)])
                ax.annotate("", (x1, y1), (x0, y0),
                            arrowprops={"arrowstyle": "->", "color": colour, "lw": 1, "alpha": .6})
                if ball == turn["cue"]:
                    ax.annotate(str(i + 1), (x0, y0), xytext=(5, 6), textcoords="offset points", fontsize=9)
            if not points and turn["layout"]:
                points = [turn["layout"][ball]]
            if points:
                xs, ys = zip(*points)
                ax.scatter(xs, ys, s=25, c=colour, label=f"{ball} ball", zorder=3)
        start, end = turn["score_start"], turn["score_end"]
        tiles = (
            f"{'-'.join(map(str, start))} to {'-'.join(map(str, end))}"
            if start is not None and end is not None
            else "unavailable"
        )
        total = f"+{turn['points']}" if turn["points"] is not None else "unknown"
        players = "–".join(history["meta"]["players"])
        ax.set_title(
            f"Turn {turn['turn']} | Inning {turn['inning']} | {turn['player_name']} "
            f"| Turn total {total}\n"
            f"{clock(turn['start'])} - {clock(turn['end'])} | {players} tiles {tiles}",
            fontsize=13,
            pad=14,
        )
        ax.set(xlim=(-.08, 3.18), ylim=(1.78, -.08), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.12)
        if turn["movements"] or turn["layout"]:
            ax.legend(loc="upper right", fontsize=9)
        if not turn["movements"]:
            ax.text(.5, .9, "No complete movement observed on the locked camera",
                    ha="center", transform=ax.transAxes, color="#555555", fontsize=10)
        fig.text(
            0.5,
            0.025,
            "Arrows show rest displacement, not strike timing. Score tiles at counter bounds may lag play.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(out / f"turn_{turn['turn']:02d}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("balls", type=Path)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--cue-colors",
        nargs=2,
        choices=("white", "yellow"),
        metavar=("PLAYER1", "PLAYER2"),
        help="verified cue colors in scoreboard player order; overrides meta.cue_colors",
    )
    parser.add_argument("--out", type=Path, default=Path("data/turn_graphs"))
    args = parser.parse_args()

    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        history = build_history(load(args.balls), load(args.turns), load(args.scores), args.cue_colors)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    plot_history(history, args.out)
    for turn in history["turns"]:
        total = f"+{turn['points']}" if turn["points"] is not None else "total unknown"
        print(
            f"Turn {turn['turn']}: {turn['player_name']} inning {turn['inning']}, "
            f"{total}, {len(turn['movements'])} rest transitions"
        )


if __name__ == "__main__":
    main()
