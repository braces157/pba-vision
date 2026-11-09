"""Plot camera-locked rest states grouped by confirmed broadcast inning.

    python scripts/plot_innings.py data/balls_jBbjq-VjxM4.json \
        --innings data/innings_jBbjq-VjxM4.json \
        --scores data/scores_jBbjq-VjxM4.json --out data/inning_graphs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pbavision.align.broadcast import InningSpan, inning_for_interval

COLOURS = {"white": "#898989", "yellow": "#efc600", "red": "#f03939"}


def valid_rest(rest: dict) -> bool:
    return all(0 <= x <= 3.1 and 0 <= y <= 1.7 for x, y in rest["positions"].values())


def clock(t: float) -> str:
    return f"{int(t) // 60:02d}:{t % 60:04.1f}"


def score_at(t: float, readings: list[dict]) -> list[int] | None:
    earlier = [r for r in readings if r["t"] <= t]
    if earlier:
        return earlier[-1]["scores"]
    return readings[0]["scores"] if readings and readings[0]["scores"] == [0, 0] else None


def groups(ball_data: dict, counter: dict, score_data: dict) -> list[dict]:
    spans = [InningSpan(**s) for s in counter["spans"]]
    score_readings = score_data["scoreboard"]["readings"]
    result = []
    for span in spans:

        def supported_time(t: float) -> bool:
            return any(abs(t - obs_t) <= 1 and inning == span.inning for obs_t, inning in counter["readings"])

        rests = [
            r
            for r in ball_data["rests"]
            if span.start <= r["timestamp"] <= span.end and valid_rest(r) and supported_time(r["timestamp"])
        ]
        strokes = [
            s
            for s in ball_data["strokes"]
            if valid_rest(s["from_rest"])
            and valid_rest(s["to_rest"])
            and inning_for_interval(s["from_rest"]["timestamp"], s["to_rest"]["timestamp"], spans) == span.inning
        ]
        # Keep only transitions whose endpoints have nearby matching counter
        # observations. Hidden score bugs and older counters in replays do
        # not supply evidence that a position belongs to the current inning.
        supported = []
        for stroke in strokes:
            endpoints = [stroke[p]["timestamp"] for p in ("from_rest", "to_rest")]
            if all(supported_time(t) for t in endpoints):
                supported.append(stroke)
        result.append(
            {
                **asdict(span),
                "rests": rests,
                "transitions": supported,
                "score_start": score_at(span.start, score_readings),
                "score_end": score_at(span.end, score_readings),
            }
        )
    return result


def draw(ax, group: dict, compact: bool = False) -> None:
    from matplotlib.patches import Rectangle

    for name, colour in COLOURS.items():
        points = [r["positions"][name] for r in group["rests"]]
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, s=14 if compact else 23, c=colour, alpha=.8,
                       label=f"{name} ball", zorder=3)
        for stroke in group["transitions"]:
            if stroke["moved"][name]:
                x0, y0 = stroke["from_rest"]["positions"][name]
                x1, y1 = stroke["to_rest"]["positions"][name]
                ax.plot([x0, x1], [y0, y1], c=colour, lw=.85 if compact else 1, alpha=.45)
    ax.add_patch(Rectangle((.13, .13), 2.84, 1.42, fill=False, ec="#333333", ls="--", lw=1.2,
                           label="playing bed 2.84 x 1.42 m"))
    ax.set(xlim=(-.08, 3.18), ylim=(1.78, -.08), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    start, end = group["score_start"], group["score_end"]
    score = f"{start[0]}-{start[1]} to {end[0]}-{end[1]}" if start and end else "unavailable"
    ax.set_title(f"Inning {group['inning']}  |  {clock(group['start'])} - {clock(group['end'])}\n"
                 f"Broadcast score {score}  |  {len(group['rests'])} rest states, "
                 f"{len(group['transitions'])} estimated transitions", fontsize=10 if compact else 12, pad=10)
    ax.tick_params(labelsize=8 if compact else 10)
    ax.grid(alpha=.12, lw=.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("balls", type=Path, help="serialized Reconstruction from dataclasses.asdict")
    parser.add_argument("--innings", type=Path, required=True, help="confirmed broadcast spans and counter readings")
    parser.add_argument("--scores", type=Path, required=True, help="broadcast score timeline")
    parser.add_argument("--out", type=Path, default=Path("data/inning_graphs"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    balls = json.loads(args.balls.read_text(encoding="utf-8"))
    counter = json.loads(args.innings.read_text(encoding="utf-8"))
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    grouped = groups(balls, counter, scores)
    if not grouped:
        raise SystemExit("no confirmed inning spans to plot")
    args.out.mkdir(parents=True, exist_ok=True)
    players = " vs ".join(scores["meta"].get("players") or ["top player", "bottom player"])
    title = f"{players} | PBA Team League | Set {scores['meta'].get('set_number', '?')}"
    footer = ("Lines connect estimated rest positions; they are not cushion-by-cushion trajectories. "
              "Camera cuts and occlusions can hide shots.")
    for group in grouped:
        fig, ax = plt.subplots(figsize=(11, 6.6))
        draw(ax, group)
        fig.suptitle(title, fontsize=13, y=.985)
        ax.legend(loc="upper right", fontsize=8, framealpha=.9)
        fig.text(.5, .018, footer, ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, .045, 1, .96))
        fig.savefig(args.out / f"inning_{group['inning']:02d}.png", dpi=160)
        plt.close(fig)
    rows = (len(grouped) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 5.4), squeeze=False)
    for ax, group in zip(axes.flat, grouped):
        draw(ax, group, compact=True)
    for ax in list(axes.flat)[len(grouped):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, .959), fontsize=10)
    fig.suptitle(f"Three-ball analysis by inning\n{title}", fontsize=18, y=.995)
    fig.text(.5, .012, footer, ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.055, right=.985, bottom=.055, top=.895, hspace=.42, wspace=.14)
    fig.savefig(args.out / "all_innings.png", dpi=160)
    plt.close(fig)
    artifact = {
        "schema": "pba-vision/inning-positions@0.1", "meta": scores["meta"],
        "inning_source": "confirmed broadcast counter", "coordinate_units": "meters",
        "line_meaning": "estimated displacement between rest positions; not a measured trajectory",
        "fps": balls["fps"], "innings": grouped,
    }
    (args.out / "innings.json").write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"Saved {len(grouped)} inning graphs, all_innings.png and innings.json in {args.out}")
    for group in grouped:
        print(f"inning {group['inning']}: {len(group['rests'])} rests, "
              f"{len(group['transitions'])} transitions, score {group['score_start']} to {group['score_end']}")


if __name__ == "__main__":
    main()
