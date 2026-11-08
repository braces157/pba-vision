"""Write source-frame contact sheets for reviewing a measured match history.

    python scripts/review_history.py HISTORY.json --video VIDEO.mp4 --out reports/review

The review images overlay observations on the exact source frame at the
start, a cushion (or midpoint), and the last visible cue sample of each view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLOURS = {"white": (255, 255, 255), "yellow": (0, 230, 255), "red": (30, 30, 255)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    bed = np.array(history["calibration"]["bed_quad_960px"], np.float32)
    corners = np.array([[0, 0], [2.84, 0], [2.84, 1.42], [0, 1.42]], np.float32)
    pixels = cv2.getPerspectiveTransform(corners, bed)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    args.out.mkdir(parents=True, exist_ok=True)
    page = []
    review = []
    try:
        for turn in history["turns"]:
            for number, movement in enumerate(turn["movements"], 1):
                q = movement["trajectory"]
                cue = [point for segment in q["paths"][turn["cue"]] for point in segment]
                if not cue:
                    continue
                contact = q["cushions"][len(q["cushions"]) // 2] if q["cushions"] else {"t": q["duration"] / 2}
                times = [cue[0][0], min(cue, key=lambda p: abs(p[0] - contact["t"]))[0], cue[-1][0]]
                tiles = []
                for relative in times:
                    frame_no = round((q["start"] + relative) * q["sample_fps"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"missing source frame {frame_no}")
                    img = cv2.resize(frame, (960, 540))
                    for ball, segments in q["paths"].items():
                        for segment in segments:
                            visible = [p for p in segment if p[0] <= relative + .0001]
                            if len(visible) > 1:
                                xy = cv2.perspectiveTransform(np.array([[p[1:3] for p in visible]], np.float32), pixels)
                                cv2.polylines(img, [xy[0].astype(np.int32)], False, COLOURS[ball], 1, cv2.LINE_AA)
                            matches = [p for p in segment if abs(p[0] - relative) < .001]
                            for p in matches:
                                xy = cv2.perspectiveTransform(np.array([[p[1:3]]], np.float32), pixels)[0, 0]
                                cv2.circle(img, tuple(np.round(xy).astype(int)), 9, COLOURS[ball], 1, cv2.LINE_AA)
                    tile = img[65:395, 160:790].copy()
                    cv2.rectangle(tile, (0, 0), (630, 25), (25, 30, 35), -1)
                    label = f"Turn {turn['turn']} / movement {number} / {q['start'] + relative:.3f}s"
                    cv2.putText(tile, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                    tiles.append(tile)
                page.append(np.hstack(tiles))
                review.append({"turn": turn["turn"], "movement": number,
                               "sample_times": [round(q["start"] + t, 4) for t in times]})
                if len(page) == 4:
                    cv2.imwrite(str(args.out / f"page_{len(review) // 4:02d}.jpg"), np.vstack(page))
                    page = []
        if page:
            cv2.imwrite(str(args.out / f"page_{(len(review) + 3) // 4:02d}.jpg"), np.vstack(page))
        (args.out / "samples.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
        print(f"Wrote source-frame review for {len(review)} movements")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
