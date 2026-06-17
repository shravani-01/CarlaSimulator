"""Smoke tests - prove the package imports and the metrics are correct.

These run in CI (no GPU, no CARLA, no torch needed).
"""

import carla_perception
import numpy as np
from carla_perception.metrics import ate_rmse, mean_iou, rpe_rmse


def test_version():
    assert carla_perception.__version__


def test_ate_zero_for_identical_trajectories():
    traj = np.cumsum(np.random.RandomState(0).randn(50, 3), axis=0)
    assert ate_rmse(traj, traj) == 0.0


def test_ate_matches_constant_offset():
    traj = np.zeros((10, 3))
    shifted = traj + np.array([3.0, 4.0, 0.0])  # every frame off by 5m
    assert abs(ate_rmse(shifted, traj) - 5.0) < 1e-9


def test_rpe_zero_for_identical_trajectories():
    traj = np.cumsum(np.random.RandomState(1).randn(30, 3), axis=0)
    assert rpe_rmse(traj, traj, delta=1) == 0.0


def test_mean_iou_perfect():
    labels = np.array([[0, 1], [2, 1]])
    assert mean_iou(labels, labels, num_classes=3) == 1.0


def test_mean_iou_half_overlap():
    pred = np.array([0, 0, 1, 1])
    target = np.array([0, 1, 1, 1])
    # class 0: inter=1 union=2 -> 0.5 ; class 1: inter=2 union=3 -> 0.667
    assert abs(mean_iou(pred, target, num_classes=2) - (0.5 + 2 / 3) / 2) < 1e-9
