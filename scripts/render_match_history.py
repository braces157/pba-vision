"""Render consistent static images from observed match-history trajectories.

    python scripts/render_match_history.py HISTORY.json --out reports/match_images

Only supplied observations are drawn. Separate trajectory segments stay separate;
turn totals never become individual movement scores.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

BALLS = ("white", "yellow", "red")
INK = "#334f60"
COLOURS = {"white": INK, "yellow": "#aa7e08", "red": "#be3444"}
FILLS = {"white": "#ffffff", "yellow": "#f3cb36", "red": "#dd4252"}


def finite_point(point: object, dimensions: int) -> bool:
    return (
        isinstance(point, (list, tuple))
        and len(point) >= dimensions
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point[:dimensions])
    )


def observed_paths(trajectory: dict) -> dict[str, list[list[list[float]]]]:
    """Keep declared gaps, and break at malformed or nonchronological samples."""
    paths = {ball: [] for ball in BALLS}
    for ball in BALLS:
        for segment in trajectory.get("paths", {}).get(ball, []):
            continuous = []
            for point in segment:
                valid = finite_point(point, 3) and point[0] >= 0
                if not valid or (continuous and point[0] <= continuous[-1][0]):
                    if continuous:
                        paths[ball].append(continuous)
                    continuous = []
                if valid:
                    continuous.append(point[:3])
            if continuous:
                paths[ball].append(continuous)
    return paths


def clock(seconds: float) -> str:
    minutes, seconds = divmod(max(0, seconds), 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def score_context(history: dict, turn: dict, movement: dict | None) -> str:
    players = history.get("meta", {}).get("players", [])
    names = " / ".join(players) or "Player order from source"
    before = "–".join(map(str, turn.get("score_start", []))) or "unavailable"
    after = "–".join(map(str, turn.get("score_end", []))) or "unavailable"
    total = turn.get("points")
    text = f"Turn score ({names}) {before} → {after}  |  Turn total: {f'+{total}' if total is not None else 'unknown'}"
    if movement is not None:
        points = movement.get("score_points")
        if isinstance(points, int) and not isinstance(points, bool):
            text += f"  |  Movement score: {points:+d}"
    return text


def status_lines(trajectory: dict, paths: dict, has_paths: bool) -> list[str]:
    if not has_paths:
        return ["No visible trajectory supplied. Starting layout is shown when available; no route is inferred."]
    completeness = trajectory.get("completeness") or {}
    details = []
    start = completeness.get("start_observed")
    end = completeness.get("end_settled")
    details.append(
        "Start observed" if start is True else "Start not observed" if start is False else "Start coverage unknown"
    )
    details.append(
        "Settled end observed"
        if end is True
        else "Settled end not observed"
        if end is False
        else "End coverage unknown"
    )
    if completeness.get("end_reason") == "camera_cut":
        details.append("Camera cuts away; the unseen ending is omitted")
    elif completeness.get("end_reason") == "interval_end" and end is not True:
        details.append("Video interval ends before the balls settle")
    if any(len(segments) > 1 for segments in paths.values()):
        details.append("Separate observations remain unconnected")
    return textwrap.wrap("  ·  ".join(details), width=143, max_lines=3, placeholder=" …")


def spread_positions(values: list[float], lower: float, upper: float, separation: float) -> list[float]:
    """Place outward rail labels in order without overlapping adjacent labels."""
    if not values:
        return []
    if len(values) == 1:
        return [min(upper, max(lower, values[0]))]
    separation = min(separation, (upper - lower) / (len(values) - 1))
    positions = [max(lower, min(upper, values[0]))]
    for value in values[1:]:
        positions.append(max(min(upper, value), positions[-1] + separation))
    if positions[-1] > upper:
        positions[-1] = upper
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - separation)
    return positions


def draw_contacts(ax, trajectory: dict, width: float, height: float, cue: str) -> int:
    contacts = [
        c
        for c in trajectory.get("cushions", [])
        if c.get("ball") in BALLS
        and c.get("rail") in ("top", "bottom", "left", "right")
        and finite_point([c.get("t"), c.get("x"), c.get("y")], 3)
    ]
    rails = defaultdict(list)
    for number, contact in enumerate(sorted(contacts, key=lambda c: c["t"]), 1):
        rails[contact["rail"]].append((number, contact))
    for rail, entries in rails.items():
        horizontal = rail in ("top", "bottom")
        entries.sort(key=lambda entry: entry[1]["x" if horizontal else "y"])
        maximum = width if horizontal else height
        positions = spread_positions(
            [entry[1]["x" if horizontal else "y"] for entry in entries], 0.06, maximum - 0.06, 0.19
        )
        for (number, contact), along in zip(entries, positions):
            ball = contact["ball"]
            point = (contact["x"], contact["y"])
            label = (
                (along, -0.17 if rail == "top" else height + 0.17)
                if horizontal
                else (-0.18 if rail == "left" else width + 0.18, along)
            )
            ax.plot(*point, "o", ms=4.5, color=COLOURS[ball], zorder=6)
            ax.annotate(
                f"{ball[0].upper()}{number}",
                xy=point,
                xytext=label,
                ha="center",
                va="center",
                fontsize=10,
                color=COLOURS[ball],
                weight="bold" if ball == cue else "normal",
                bbox={"boxstyle": "circle,pad=.25", "fc": "white", "ec": COLOURS[ball], "lw": 1},
                arrowprops={"arrowstyle": "-", "color": COLOURS[ball], "lw": 1},
                zorder=7,
            )
    return len(contacts)


def draw_direction(ax, segment: list, colour: str, emphasized: bool) -> None:
    # Each arrow uses one actual adjacent pair, never a chord across a gap or turn.
    distances = [math.dist(a[1:3], b[1:3]) for a, b in zip(segment, segment[1:])]
    length = sum(distances)
    if length < 0.12:
        return
    targets = [
        (index + 0.5) * length / max(1, math.ceil(length / 0.65)) for index in range(max(1, math.ceil(length / 0.65)))
    ]
    cumulative = 0.0
    target_index = 0
    for a, b, distance in zip(segment, segment[1:], distances):
        cumulative += distance
        if target_index < len(targets) and cumulative >= targets[target_index] and distance > 0.001:
            ax.annotate(
                "",
                xy=b[1:3],
                xytext=a[1:3],
                arrowprops={
                    "arrowstyle": "-|>",
                    "mutation_scale": 18 if emphasized else 11,
                    "color": colour,
                    "lw": 1.7 if emphasized else 0.9,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=4,
            )
            target_index += 1


def movement_summary(trajectory: dict, paths: dict) -> dict:
    return {
        "samples": sum(len(segment) for segments in paths.values() for segment in segments),
        "segments": {ball: len(paths[ball]) for ball in BALLS},
        "balls_observed": [ball for ball in BALLS if paths[ball]],
        "coverage": trajectory.get("coverage"),
        "status": trajectory.get("status"),
        "validation": trajectory.get("validation"),
        "completeness": trajectory.get("completeness"),
    }


def layout_points(paths: dict, t: float, tolerance: float) -> dict:
    positions = {}
    for ball in BALLS:
        points = [p for segment in paths[ball] for p in segment]
        if points:
            point = min(points, key=lambda p: abs(p[0] - t))
            if abs(point[0] - t) <= tolerance:
                positions[ball] = point[1:3]
    return positions


def draw_layout(fig, bounds, paths: dict, trajectory: dict, cue: str, width: float, height: float, last: bool):
    """Separate time snapshots make initial circles and final crosses explicit."""
    ax = fig.add_axes(bounds)
    ax.add_patch(Rectangle((0, 0), width, height, fc="#edf3f5", ec="#48616d", lw=1.5))
    ax.set(xlim=(-.05, width + .05), ylim=(height + .05, -.05), aspect="equal")
    ax.axis("off")
    reference = paths[cue] if paths.get(cue) else next(p for p in paths.values() if p)
    t = reference[-1][-1][0] if last else reference[0][0][0]
    positions = layout_points(paths, t, 1.5 / trajectory.get("sample_fps", 30))
    for ball, xy in positions.items():
        ax.add_patch(Circle(xy, radius=.055, fc=FILLS[ball], ec=COLOURS[ball], lw=1.3))
    complete = trajectory.get("completeness", {})
    if last:
        title = "AFTER · balls settled" if complete.get("end_settled") else "LAST VISIBLE · ending cut off"
    else:
        title = "BEFORE · starting balls" if complete.get("start_observed") else "FIRST VISIBLE · start unseen"
    ax.set_title(title, fontsize=10, color=INK, pad=7)
    if len(positions) < 3:
        ax.text(width / 2, height + .12, "Unseen balls omitted", ha="center", fontsize=9, color=INK)


def render_image(history: dict, turn: dict, movement: dict | None, number: int | None, destination: Path) -> dict:
    trajectory = movement.get("trajectory", {}) if movement else {}
    paths = observed_paths(trajectory)
    has_paths = any(paths.values())
    bed = trajectory.get("bed") or history.get("bed") or {"width": 2.84, "height": 1.42}
    width, height = bed["width"], bed["height"]
    if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Playing-bed dimensions must be positive finite meters")
    cue = turn.get("cue")
    footer = status_lines(trajectory, paths, has_paths)
    fig, ax = plt.subplots(figsize=(14, 11 + max(0, len(footer) - 2) * 0.18))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.36, top=0.83)
    ax.add_patch(Rectangle((0, 0), width, height, fc="#edf3f5", ec="#48616d", lw=3, zorder=0))
    ax.set(xlim=(-0.35, width + 0.35), ylim=(height + 0.32, -0.32))
    ax.set_aspect("equal")
    ax.axis("off")
    for ball in sorted(BALLS, key=lambda ball: ball == cue):
        for segment in paths[ball]:
            ax.plot(
                [point[1] for point in segment],
                [point[2] for point in segment],
                color=COLOURS[ball],
                lw=2.9 if ball == cue else 1.3,
                alpha=1 if ball == cue else 0.7,
                solid_capstyle="round",
                zorder=3 if ball == cue else 2,
            )
            draw_direction(ax, segment, COLOURS[ball], ball == cue)
        start = paths[ball][0][0][1:3] if paths[ball] else (movement or {}).get("from", {}).get(ball)
        if not finite_point(start, 2) and movement is None:
            start = (turn.get("layout") or {}).get(ball)
        if finite_point(start, 2):
            ax.add_patch(Circle(start, radius=0.027, fc=FILLS[ball], ec=COLOURS[ball], lw=1.8, zorder=9))
            if ball == cue:
                ax.add_patch(Circle(start, radius=0.039, fill=False, ec=COLOURS[ball], lw=1, zorder=8))
        if paths[ball]:
            last = paths[ball][-1][-1]
            ax.plot(last[1], last[2], marker="x", ms=9, mew=2, color=COLOURS[ball], zorder=10)
    draw_contacts(ax, trajectory, width, height, cue)
    if has_paths and cue in BALLS and paths[cue]:
        first, last = paths[cue][0][0], paths[cue][-1][-1]
        first_label = "Start" if trajectory.get("completeness", {}).get("start_observed") is True else "First visible"
        for point, label, offset in ((first, first_label, (0, 23)), (last, "Last visible", (0, -25))):
            ax.annotate(
                label,
                xy=point[1:3],
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=10,
                color=INK,
                bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 2},
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
                zorder=12,
            )
    if not has_paths:
        ax.text(width / 2, height / 2, "Trajectory unavailable", ha="center", va="center", fontsize=17, color=INK)
    start_time = trajectory.get("start", (movement or turn).get("start", 0))
    end_time = (
        start_time + trajectory["duration"] if "duration" in trajectory else (movement or turn).get("end", start_time)
    )
    subtitle = f"Observed movement {number}" if number is not None else "No visible movements"
    fig.text(
        0.5,
        0.951,
        f"{turn['player_name']}  |  Inning {turn['inning']}  |  Turn {turn['turn']}",
        ha="center",
        va="top",
        fontsize=19,
        weight="bold",
        color="#24343f",
    )
    fig.text(
        0.5,
        0.907,
        f"{subtitle}  |  {clock(start_time)}–{clock(end_time)}  |  {str(cue).capitalize()} cue ball",
        ha="center",
        va="top",
        fontsize=13,
        color="#52636f",
    )
    for index, line in enumerate(textwrap.wrap(score_context(history, turn, movement), 142)):
        fig.text(0.5, 0.868 - index * 0.023, line, ha="center", va="top", fontsize=10.5, color=INK)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="-",
            color=COLOURS[ball],
            markerfacecolor=FILLS[ball],
            label=f"{ball.capitalize()}{' (cue)' if ball == cue else ''}",
            lw=2.5 if ball == cue else 1,
        )
        for ball in BALLS
    ]
    handles += [
        Line2D([], [], marker="o", color=INK, markerfacecolor="white", ls="", label="First visible / layout"),
        Line2D([], [], marker="x", color=INK, ls="", label="Last visible"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.315), ncol=5, frameon=False, fontsize=10)
    if has_paths:
        draw_layout(fig, [.16, .14, .28, .15], paths, trajectory, cue, width, height, last=False)
        draw_layout(fig, [.56, .14, .28, .15], paths, trajectory, cue, width, height, last=True)
    fig.text(
        0.5,
        0.112,
        "Arrows follow observed travel; gaps stay blank. Numbered cushion estimates are not a scoring count.",
        ha="center",
        va="top",
        fontsize=10.5,
        color=INK,
    )
    for index, line in enumerate(footer):
        fig.text(0.5, 0.088 - index * 0.017, line, ha="center", va="top", fontsize=10, color="#52636f")
    sources = [f"Turn score source: {history.get('score_source', 'unspecified')}"]
    if movement and movement.get("score_points") is not None:
        sources.append(f"Movement score source: {movement.get('score_source', 'unspecified')}")
    fig.text(0.5, 0.028, "  |  ".join(sources), ha="center", va="bottom", fontsize=9, color="#52636f")
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)
    return movement_summary(trajectory, paths)


def write_gallery(index: dict, out: Path) -> None:
    lines = ["# Match turn images", "",
             "Each image shows the observed route, starting balls and last visible positions.",
             "Camera cuts and missing observations stay marked. Points are totals for the turn.", "",
             "Score order: " + " / ".join(index["players"]), "",
             "| Inning | Turn | Player | Turn points | Score after | Images in order |",
             "| --- | --- | --- | --- | --- | --- |"]
    for turn in index["turns"]:
        links = " · ".join(f"[{number}]({name})" for number, name in enumerate(turn["paths"], 1))
        score = "–".join(map(str, turn.get("score_end") or []))
        lines.append(f"| {turn['inning']} | {turn['turn']} | {turn['player_name']} | "
                     f"{turn['points']} | {score} | {links} |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_history(history: dict, out: Path, source: Path | None = None) -> dict:
    if not isinstance(history.get("turns"), list):
        raise ValueError("History must contain a turns array")
    out.mkdir(parents=True, exist_ok=True)
    index = {
        "schema": "pba-vision/match-images@1",
        "source": str(source) if source else None,
        "players": history.get("meta", {}).get("players", []),
        "turns": [],
    }
    totals = {
        "turns": len(history["turns"]),
        "observed_movements": 0,
        "movements_with_paths": 0,
        "movements_without_paths": 0,
        "turns_with_paths": 0,
        "turns_without_paths": 0,
        "pngs": 0,
    }
    for turn in history["turns"]:
        entry = {
            key: turn.get(key) for key in ("turn", "inning", "player_name", "cue", "points", "score_start", "score_end")
        }
        entry.update({"movements": [], "paths": []})
        visible = 0
        for number, movement in enumerate(turn.get("movements", []), 1):
            filename = f"turn_{turn['turn']:02d}_movement_{number:02d}.png"
            summary = render_image(history, turn, movement, number, out / filename)
            has_paths = summary["samples"] > 0
            visible += int(has_paths)
            entry["movements"].append(
                {
                    "observed_movement": number,
                    "start": movement.get("start"),
                    "end": movement.get("end"),
                    "path": filename,
                    **summary,
                    "score_points": movement.get("score_points"),
                    "score_source": movement.get("score_source"),
                }
            )
            entry["paths"].append(filename)
            totals["pngs"] += 1
        if not visible:
            filename = f"turn_{turn['turn']:02d}_no_visible_paths.png"
            render_image(history, turn, None, None, out / filename)
            entry["paths"].append(filename)
            totals["pngs"] += 1
        entry["coverage_counts"] = {
            "observed_movements": len(entry["movements"]),
            "with_paths": visible,
            "without_paths": len(entry["movements"]) - visible,
        }
        totals["observed_movements"] += len(entry["movements"])
        totals["movements_with_paths"] += visible
        totals["movements_without_paths"] += len(entry["movements"]) - visible
        totals["turns_with_paths" if visible else "turns_without_paths"] += 1
        index["turns"].append(entry)
    index["coverage_counts"] = totals
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_gallery(index, out)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/match_images"))
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    index = render_history(history, args.out, args.history)
    print(json.dumps(index["coverage_counts"]))
    print(f"Index: {args.out / 'index.json'}")


if __name__ == "__main__":
    main()
