"""Tests for trajectory alignment + evaluation (synthetic, runs in CI)."""

import numpy as np
from carla_perception.trajectory import align_umeyama, evaluate_trajectory


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_alignment_recovers_a_known_similarity_transform():
    rng = np.random.default_rng(0)
    gt = np.cumsum(rng.standard_normal((100, 3)), axis=0)  # a wiggly path

    # Make a "predicted" path that is gt scaled, rotated, and shifted.
    s, R, t = 2.5, _rot_z(0.7), np.array([10.0, -3.0, 4.0])
    pred = (s * (R @ gt.T)).T + t

    aligned, params = align_umeyama(pred, gt)

    # After alignment, the predicted path should land on top of gt.
    assert np.allclose(aligned, gt, atol=1e-6)
    # Alignment maps pred -> gt. Since pred = s*gt(+...), the undo scale is 1/s.
    assert abs(params["scale"] - 1.0 / s) < 1e-6


def test_evaluate_trajectory_zero_error_for_similarity_copy():
    rng = np.random.default_rng(1)
    gt = np.cumsum(rng.standard_normal((50, 3)), axis=0)
    pred = (3.0 * (_rot_z(-0.4) @ gt.T)).T + np.array([1.0, 2.0, 3.0])

    m = evaluate_trajectory(pred, gt)
    assert m["ate"] < 1e-6
    assert m["rpe"] < 1e-6


def test_alignment_rejects_mismatched_shapes():
    try:
        align_umeyama(np.zeros((10, 3)), np.zeros((9, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched shapes")
