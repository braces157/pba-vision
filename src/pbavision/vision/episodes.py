"""Find observed cue movements without equating them with points or shots."""

from __future__ import annotations

import numpy as np

from .trajectory import BALLS


def resting(samples: list[dict], t: float, window: float = .45) -> bool:
    recent = [s for s in samples if t - window <= s["t"] <= t]
    if not recent or t - recent[0]["t"] < window - .05:
        return False
    for ball in BALLS:
        positions = [s["points"][ball] for s in recent if ball in s["points"]]
        if len(positions) < .8 * len(recent):
            return False
        xy = np.asarray(positions)
        if np.max(np.linalg.norm(xy - np.median(xy, axis=0), axis=1)) > .012:
            return False
    return True


def movement_windows(samples: list[dict], cue: str) -> list[dict]:
    """Segment a single camera view; eligibility belongs to the score timeline.

    A returning view can supply an incomplete movement. It cannot supply an
    observed strike. Callers must keep such fragments distinct from shots.
    """
    windows = []
    anchor = None
    last_rest = None
    active = None
    displaced = []
    settled_since = None
    recent_start = 0
    rest_index = None
    for i, sample in enumerate(samples):
        t, points = sample["t"], sample["points"]
        if sample.get("live_identity") is False:
            if active is not None:
                windows.append({**active, "last": max(active["first"], i - 1),
                                "end_settled": False, "end_reason": "timeline_mismatch"})
            active, anchor, last_rest, rest_index, settled_since = None, None, None, None, None
            displaced = []
            continue
        while recent_start < i and samples[recent_start]["t"] < t - .7:
            recent_start += 1
        still = resting(samples[recent_start:i + 1], t)
        if active is not None:
            if still:
                settled_since = t if settled_since is None else settled_since
                if t - settled_since >= .6:
                    windows.append({**active, "last": i, "end_settled": True, "end_reason": "settled"})
                    active, settled_since = None, None
                    anchor, last_rest, rest_index = points.get(cue), t, i
            else:
                settled_since = None
            continue
        if still:
            anchor, last_rest, rest_index = points.get(cue), t, i
            displaced = []
            continue
        if cue not in points or not sample.get("eligible", False):
            displaced = []
            continue
        if anchor is not None and last_rest is not None and t - last_rest < 2:
            if np.linalg.norm(np.asarray(points[cue]) - anchor) > .055:
                displaced.append(i)
            else:
                displaced = []
            if len(displaced) >= 3:
                first = displaced[0]
                while first > 0 and samples[first]["t"] > samples[displaced[0]]["t"] - .55:
                    first -= 1
                visible_departure = rest_index is not None and all(
                    cue in s["points"] for s in samples[rest_index:displaced[-1] + 1]
                )
                active = {"first": first, "start_observed": visible_departure, "rest_before_start": last_rest}
                anchor, last_rest, displaced = None, None, []
        else:
            # No preceding rest: record only the visible fragment, explicitly.
            recent = [(j, s) for j, s in enumerate(samples[recent_start:i + 1], recent_start)
                      if cue in s["points"]]
            if len(recent) >= 4 and t - recent[0][1]["t"] >= .2 and np.linalg.norm(
                np.asarray(points[cue]) - recent[0][1]["points"][cue]
            ) > .055:
                active = {"first": recent[0][0], "start_observed": False}
    if active is not None:
        windows.append({**active, "last": len(samples) - 1, "end_settled": False, "end_reason": "camera_cut"})
    return windows
