"""The main-camera tracker learns the dominant camera and rejects the rest."""

from __future__ import annotations

import numpy as np

from pbavision.vision.camera import MainCameraTracker

SHAPE = (540, 960)

# overhead tripod: essentially constant, tiny tripod jitter
OVERHEAD = np.array([[280, 60], [940, 62], [942, 470], [278, 468]], dtype=np.float32)
# wide venue view: a very different, drifting quad
WIDE = np.array([[300, 200], [880, 215], [905, 430], [260, 415]], dtype=np.float32)


def _feed(tracker, quad, times, jitter=0.0):
    rng = np.random.default_rng(7)
    for _ in range(times):
        noisy = quad + rng.uniform(-jitter, jitter, quad.shape)
        tracker.observe(noisy, SHAPE)


def test_dominant_camera_wins():
    tracker = MainCameraTracker()
    _feed(tracker, OVERHEAD, 300, jitter=2.0)
    _feed(tracker, WIDE, 40, jitter=4.0)
    stats = tracker.finalize()
    assert stats["cameras"] == 2
    assert tracker.accept(OVERHEAD, SHAPE)
    assert not tracker.accept(WIDE, SHAPE)


def test_small_jitter_stays_on_camera():
    tracker = MainCameraTracker()
    _feed(tracker, OVERHEAD, 100, jitter=2.0)
    tracker.finalize()
    drifted = OVERHEAD + np.array([[3, -2], [-3, 2], [2, 2], [-2, -3]], dtype=np.float32)
    assert tracker.accept(drifted, SHAPE)


def test_locked_calibration_does_not_follow_temporary_edge_occlusions():
    tracker = MainCameraTracker()
    _feed(tracker, OVERHEAD, 100)
    tracker.finalize()
    occluded = OVERHEAD + np.array([[12, 5], [0, 0], [0, 0], [12, -5]], dtype=np.float32)
    label = tracker.label(occluded, SHAPE)
    assert label is not None
    assert np.allclose(tracker.calibration(label, SHAPE), OVERHEAD)
    assert np.allclose(tracker.calibration(label, (1080, 1920)), OVERHEAD * 2)


def test_rare_view_is_not_promoted():
    tracker = MainCameraTracker()
    _feed(tracker, OVERHEAD, 300, jitter=2.0)
    _feed(tracker, WIDE, 3, jitter=4.0)  # below MIN_CLUSTER_SIZE
    stats = tracker.finalize()
    assert stats["cameras"] == 2
    # the singleton cluster must not become the accepted camera
    assert tracker.accept(WIDE, SHAPE) is False


def test_empty_survey_refuses_everything():
    tracker = MainCameraTracker()
    stats = tracker.finalize()
    assert stats["cameras"] == 0
    assert tracker.accept(OVERHEAD, SHAPE) is False


def test_normalized_geometry_independent_of_resolution():
    tracker_a = MainCameraTracker()
    _feed(tracker_a, OVERHEAD, 60)
    tracker_a.finalize()

    # same camera, seen on a 1920-wide sensor: fractions match
    big = OVERHEAD * np.array([2.0, 2.0], dtype=np.float32)
    tracker_b = MainCameraTracker()
    for _ in range(60):
        tracker_b.observe(big, (1080, 1920))
    tracker_b.finalize()

    probe = OVERHEAD + 1.0
    assert tracker_a.accept(probe, SHAPE)
    assert tracker_b.accept(probe * 2.0, (1080, 1920))
