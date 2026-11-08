"""Lock onto the broadcast's main camera by its own geometry.

A broadcast cuts between many cameras, but the scoring view — the high
overhead table camera — is a fixed tripod: its cloth quad is geometrically
constant for the whole session, while every other view (wide venue shots,
close-ups, replay angles) has a different and usually drifting quad. So the
main camera can be *learned* from the footage itself:

    observe()   collect every plausible quad during a survey pass
    finalize()  cluster the quads; the dominant cluster is the main camera
    accept()    during measurement, approve only frames on that camera

Locking the reconstruction onto one camera is what makes positions
comparable across the whole set, and it stops camera cuts from masquerading
as strokes — the failure that wrecked inning chaining in v0.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Mean corner distance (in frame-size fractions) for two quads to be one camera.
#: A tripod quad moves by pixels; a different camera moves it by tenths.
CLUSTER_TOLERANCE = 0.05
#: A cluster needs at least this many observations to be the main camera.
MIN_CLUSTER_SIZE = 25


@dataclass
class _Cluster:
    quad: np.ndarray            # running mean of member quads (normalized)
    count: int = 0


@dataclass
class MainCameraTracker:
    """Learn and enforce the broadcast's main scoring camera."""

    tolerance: float = CLUSTER_TOLERANCE
    min_cluster_size: int = MIN_CLUSTER_SIZE
    _clusters: list[_Cluster] = field(default_factory=list)
    _main: _Cluster | None = None
    _stable: list[_Cluster] = field(default_factory=list)
    frame_shape: tuple[int, int] | None = None  # (height, width)

    # ---------------------------------------------------------------- survey

    def observe(self, quad: np.ndarray, frame_shape: tuple[int, int]) -> None:
        """Record one plausible quad, normalized by the frame size."""
        self.frame_shape = frame_shape
        normalized = self._normalize(quad, frame_shape)
        for cluster in self._clusters:
            if self._distance(normalized, cluster.quad) < self.tolerance:
                cluster.quad = (cluster.quad * cluster.count + normalized) / (cluster.count + 1)
                cluster.count += 1
                return
        self._clusters.append(_Cluster(quad=normalized, count=1))

    def finalize(self) -> dict:
        """Pick the dominant cluster and the set of stable cameras.

        Every cluster with enough frames is a usable fixed camera: since ball
        positions are mapped to table meters, two cameras' positions are
        directly comparable, and each camera's segments are reconstructed
        separately so cuts never join two views' position streams.
        Returns survey stats for the report.
        """
        if not self._clusters:
            self._main = None
            self._stable = []
            return {"cameras": 0, "stable_cameras": 0, "main_camera_frames": 0, "total_frames": 0}
        total = sum(c.count for c in self._clusters)
        best = max(self._clusters, key=lambda c: c.count)
        self._main = best if best.count >= self.min_cluster_size else None
        # A camera seen rarely is still real if it appears consistently; the
        # floor keeps single-shot flukes (zooms, transition frames) out.
        floor = max(self.min_cluster_size, int(total * 0.02))
        self._stable = [c for c in self._clusters if c.count >= floor]
        sizes = sorted((c.count for c in self._clusters), reverse=True)
        return {
            "cameras": len(self._clusters),
            "stable_cameras": len(self._stable),
            "main_camera_frames": best.count if self._main else 0,
            "total_frames": total,
            "largest_clusters": sizes[:6],
        }

    # ----------------------------------------------------------- measurement

    def accept(self, quad: np.ndarray, frame_shape: tuple[int, int]) -> bool:
        """True when the quad belongs to the learned main camera."""
        if self._main is None:
            return False
        normalized = self._normalize(quad, frame_shape)
        return self._distance(normalized, self._main.quad) < self.tolerance

    def label(self, quad: np.ndarray, frame_shape: tuple[int, int]) -> int | None:
        """Which stable camera this frame belongs to, or None for an unstable view."""
        normalized = self._normalize(quad, frame_shape)
        for index, cluster in enumerate(self._stable):
            if self._distance(normalized, cluster.quad) < self.tolerance:
                return index
        return None

    def calibration(self, label: int, frame_shape: tuple[int, int]) -> np.ndarray:
        """One fixed quad per tripod; temporary edge occlusions cannot move it."""
        h, w = frame_shape
        return (self._stable[label].quad * np.array([w, h])).astype(np.float32)

    # ---------------------------------------------------------------- private

    def _normalize(self, quad: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        h, w = frame_shape
        quad = quad.astype(np.float64).reshape(4, 2)
        quad[:, 0] /= w
        quad[:, 1] /= h
        return quad

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))
