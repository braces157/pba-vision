"""The published artifact: one JSON file per reconstructed broadcast set.

The schema is the contract with whatever consumes reconstructions (a
scoreboard site, a notebook, a plotting script). Positions are meters on the
outer cushion rectangle, x along the long rail, y across it, both from the
top-left corner of the frame's table.
"""

from __future__ import annotations

import json
from pathlib import Path

from .pipeline import Reconstruction
from .validate import Validation


def write_artifact(
    path: Path,
    meta: dict,
    reconstruction: Reconstruction,
    validation: Validation,
    broadcast_scores: dict | None = None,
) -> Path:
    artifact = {
        "schema": "pba-vision/reconstruction@0.1",
        "meta": meta,
        "stats": {
            "frames_seen": reconstruction.frames_seen,
            "camera_views": reconstruction.camera_views,
            "stable_cameras": reconstruction.stable_cameras,
            "main_camera_frames": reconstruction.main_camera_frames,
            "frames_measured": reconstruction.frames_measured,
            "frames_all_balls": reconstruction.frames_all_balls,
            "rest_states": len(reconstruction.rests),
            "strokes": len(reconstruction.strokes),
        },
        "strokes": [
            {
                "index": stroke.index,
                "t": round(stroke.from_rest.timestamp, 2),
                "from": {k: [round(v, 3) for v in stroke.from_rest.positions[k]] for k in stroke.from_rest.positions},
                "to": {k: [round(v, 3) for v in stroke.to_rest.positions[k]] for k in stroke.to_rest.positions},
                "moved": [k for k, did in stroke.moved.items() if did],
                "points_est": stroke.estimated_points,
                "cue_est": stroke.cue,
                "inning_est": stroke.inning_estimate,
            }
            for stroke in reconstruction.strokes
        ],
        "validation": {
            "innings_official": validation.innings_official,
            "innings_estimated": validation.innings_estimated,
            "official_totals": validation.official_totals,
            "estimated_carom_totals": validation.estimated_totals,
            "official_points_per_inning": validation.official_points_per_inning,
            "estimated_points_per_inning": validation.estimated_points_per_inning,
            "totals_gap": validation.totals_gap,
        },
    }
    if broadcast_scores is not None:
        artifact["scoreboard"] = broadcast_scores
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
