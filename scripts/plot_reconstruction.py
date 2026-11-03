"""Plot a reconstruction artifact: ball rest positions and stroke trails in table meters.

    python scripts/plot_reconstruction.py reconstruction.json --out scatter.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--out", type=Path, default=Path("scatter.png"))
    parser.add_argument("--title", default=None, help="override the chart title")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    strokes = data["strokes"]
    meta = data["meta"]
    validation = data["validation"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = {"white": "gray", "yellow": "gold", "red": "red"}
    for colour, c in colours.items():
        xs, ys = [], []
        for stroke in strokes:
            for phase in ("from", "to"):
                x, y = stroke[phase][colour]
                xs.append(x)
                ys.append(y)
        if xs:
            ax.scatter(xs, ys, s=8, c=c, alpha=0.5, label=f"{colour} ball")

    # stroke trails: cue-colored lines between consecutive positions
    for stroke in strokes:
        for colour, c in colours.items():
            if colour in stroke["moved"]:
                x0, y0 = stroke["from"][colour]
                x1, y1 = stroke["to"][colour]
                ax.plot([x0, x1], [y0, y1], color=c, lw=0.6, alpha=0.25)

    ax.add_patch(plt.Rectangle((0.13, 0.13), 2.84, 1.42, fill=False, color="k", ls="--", lw=1.2,
                               label="playing bed 2.84x1.42 m"))
    ax.set_xlim(-0.1, 3.2)
    ax.set_ylim(1.8, -0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    players = args.title or " vs ".join(meta.get("players_official") or ["?", "?"])
    ax.set_title(
        f"{meta.get('tournament', '')} — set {meta.get('set_number', '?')}\n"
        f"{players} · official {validation['official_totals']} · "
        f"{data['stats']['strokes']} strokes from {meta.get('fps', 4)} fps video",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
