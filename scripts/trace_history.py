"""Scan every turn for observed source-frame ball paths.

    python scripts/trace_history.py HISTORY.json --video VIDEO.mp4 --out OUTPUT.json

The input must calibrate its own playing-bed and fixed-camera boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pbavision.vision.trajectory import BED_HEIGHT, BED_WIDTH, scan_turn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--turn", type=int, help="Rescan only this turn")
    args = parser.parse_args(argv)
    data = json.loads(args.history.read_text(encoding="utf-8"))
    cv2.setNumThreads(1)
    calibration = data.get("calibration")
    if not calibration:
        raise ValueError("history requires calibrated playing-bed and fixed-camera quads")
    bed = np.array(calibration["bed_quad_960px"], np.float32)
    outer = np.array(calibration["fixed_outer_quad_960px"], np.float32)
    turn_ids = [turn["turn"] for turn in data["turns"]]
    if args.turn is not None:
        if args.turn not in turn_ids:
            parser.error("--turn does not match a turn in this history")
        previous = data.get("scan", {})
        completed = previous.get("completed_turns", turn_ids if previous else [])
        if data.get("schema") not in {"pba-vision/turn-history@0.2", "pba-vision/turn-history@0.3"} or (
            set(completed) != set(turn_ids)
        ) or any(
            not isinstance(m.get("trajectory"), dict) for t in data["turns"] for m in t.get("movements", [])
        ):
            raise ValueError("partial rescans require a fully traced history; scan all turns first")
    else:
        completed = []
    if data.get("schema") not in {"pba-vision/turn-history@0.2", "pba-vision/turn-history@0.3"}:
        for turn in data["turns"]:
            turn["layout"] = None  # outer-table positions cannot be plotted on the playing bed
    data["geometry"] = "meters on actual playing bed; 2.84 x 1.42; calibrated bed corners in video"
    data["schema"] = "pba-vision/turn-history@0.3"
    data["bed"] = {"width": BED_WIDTH, "height": BED_HEIGHT}
    data["movement_meaning"] = "observed source-frame trajectories; no paths inferred through missing video"
    data["trajectory_source"] = "observed source frames; separate segments for missing detections and camera cuts"
    data["scan"] = {"scope": "calibrated camera views in each turn", "source_fps": "native",
                    "white_profile": "warm ivory core", "individual_scores": "unassigned",
                    "completed_turns": [n for n in completed if n != args.turn], "complete": False,
                    "cushions": "observed reversals; short gaps explicitly marked; not a scoring verdict"}
    if args.turn is None:
        for turn in data["turns"]:
            turn["movements"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for turn in data["turns"]:
        if args.turn is not None and args.turn != turn["turn"]:
            continue
        turn["movements"] = scan_turn(args.video, turn, bed, outer)
        for index, movement in enumerate(turn["movements"], 1):
            q = movement["trajectory"]
            print(f"turn {turn['turn']}, movement {index}: {q['start']:.2f}s +{q['duration']:.1f}s, "
                  f"cue coverage {q['coverage']:.1%}, {len(q['cushions'])} rail estimates, "
                  f"start observed: {q['completeness']['start_observed']}", flush=True)
        print(f"turn {turn['turn']} scanned; {len(turn['movements'])} movements", flush=True)
        data["scan"]["completed_turns"].append(turn["turn"])
        data["scan"]["complete"] = set(data["scan"]["completed_turns"]) == set(turn_ids)
        args.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    data["calibration"] = {"bed_quad_960px": bed.tolist(), "fixed_outer_quad_960px": outer.tolist()}
    data["scan"]["complete"] = set(data["scan"]["completed_turns"]) == set(turn_ids)
    args.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
