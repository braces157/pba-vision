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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--turn", type=int, help="Rescan only this turn")
    args = parser.parse_args()
    data = json.loads(args.history.read_text(encoding="utf-8"))
    cv2.setNumThreads(1)
    calibration = data.get("calibration")
    if not calibration:
        raise ValueError("history requires calibrated playing-bed and fixed-camera quads")
    bed = np.array(calibration["bed_quad_960px"], np.float32)
    outer = np.array(calibration["fixed_outer_quad_960px"], np.float32)

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
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    data["geometry"] = "meters on actual playing bed; 2.84 x 1.42; calibrated bed corners in video"
    data["schema"] = "pba-vision/turn-history@0.3"
    data["bed"] = {"width": BED_WIDTH, "height": BED_HEIGHT}
    data["movement_meaning"] = "observed source-frame trajectories; no paths inferred through missing video"
    data["trajectory_source"] = "observed source frames; separate segments for missing detections and camera cuts"
    data["calibration"] = {"bed_quad_960px": bed.tolist(), "fixed_outer_quad_960px": outer.tolist()}
    data["scan"] = {"scope": "all calibrated camera views in each turn", "source_fps": "native",
                    "white_profile": "warm ivory core", "individual_scores": "unassigned",
                    "cushions": "observed reversals; short gaps explicitly marked; not a scoring verdict"}
    args.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
