"""Build a chronological match history with one player's turn per graph.

    python scripts/plot_turns.py data/balls_jBbjq-VjxM4.json \
        --turns data/turns_jBbjq-VjxM4.json --scores data/scores_jBbjq-VjxM4.json \
        --out data/turn_graphs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_innings import COLOURS, clock, score_at, valid_rest


def build_history(balls: dict, turns: dict, scores: dict) -> dict:
    players = scores["meta"]["players"]
    score_readings = scores["scoreboard"]["readings"]
    history = []
    for span in turns["spans"]:
        player = span["player"]
        start_score = score_at(span["start"], score_readings)
        end_score = score_at(span["end"], score_readings)
        if start_score is None or end_score is None:
            raise ValueError("score history does not cover a turn")
        movements = []
        for stroke in balls["strokes"]:
            before, after = stroke["from_rest"], stroke["to_rest"]
            if before.get("end_timestamp") is None or after.get("end_timestamp") is None:
                raise ValueError("reconstruct again to retain rest end times for player assignment")
            onset = before["end_timestamp"] + .5 / balls["fps"]
            if not span["start"] <= onset <= span["end"]:
                continue
            nearest = min(turns["readings"], key=lambda r: abs(r[0] - onset))
            if abs(nearest[0] - onset) > 1 or tuple(nearest[1:]) != (span["inning"], player):
                continue
            if not valid_rest(before) or not valid_rest(after):
                continue
            # Short contour flickers are not a reliable settled endpoint.
            if any(r["end_timestamp"] - r["timestamp"] < .5 for r in (before, after)):
                continue
            if after["timestamp"] - onset < 1:
                continue
            movements.append({
                "start": round(onset, 3), "end": after["timestamp"],
                "from": before["positions"], "to": after["positions"],
            })
        layouts = [r for r in balls["rests"] if valid_rest(r)
                   and span["start"] <= r["timestamp"] <= span["end"]
                   and r["end_timestamp"] - r["timestamp"] >= .5]
        history.append({
            **span, "player_name": players[player], "cue": "yellow" if player == 0 else "white",
            "score_start": start_score, "score_end": end_score,
            "points": end_score[player] - start_score[player], "movements": movements,
            "layout": layouts[0]["positions"] if layouts else None,
        })
    return {
        "schema": "pba-vision/turn-history@0.1", "meta": scores["meta"], "turns": history,
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
        ax.set_title(f"Turn {turn['turn']} | Inning {turn['inning']} | {turn['player_name']} "
                     f"| +{turn['points']} points\n"
                     f"{clock(turn['start'])} - {clock(turn['end'])} | Cenet-Nguyen {start[0]}-{start[1]} "
                     f"to {end[0]}-{end[1]}", fontsize=13, pad=14)
        ax.set(xlim=(-.08, 3.18), ylim=(1.78, -.08), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.12)
        if turn["movements"] or turn["layout"]:
            ax.legend(loc="upper right", fontsize=9)
        if not turn["movements"]:
            ax.text(.5, .9, "No complete movement observed on the locked camera",
                    ha="center", transform=ax.transAxes, color="#555555", fontsize=10)
        fig.text(.5, .025, "Arrows connect observed rest positions. Score and turn history come from the broadcast.",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .045, 1, 1))
        fig.savefig(out / f"turn_{turn['turn']:02d}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("balls", type=Path)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/turn_graphs"))
    args = parser.parse_args()
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))
    history = build_history(load(args.balls), load(args.turns), load(args.scores))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    plot_history(history, args.out)
    for turn in history["turns"]:
        print(f"Turn {turn['turn']}: {turn['player_name']} inning {turn['inning']}, "
              f"+{turn['points']}, {len(turn['movements'])} observed movements")


if __name__ == "__main__":
    main()
