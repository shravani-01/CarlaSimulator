"""Synthetic tests for dense stereo back-projection (no images needed)."""

import numpy as np
from carla_perception.reconstruction.stereo_pointcloud import (
    disparity_to_pointcloud,
    transform_points,
)


def test_disparity_to_pointcloud_depth_and_center():
    f, cx, cy = 500.0, 320.0, 240.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    baseline = 0.5

    h, w = 480, 640
    disparity = np.full((h, w), 25.0, dtype=np.float32)  # constant disparity
    color = np.zeros((h, w, 3), dtype=np.uint8)

    pts, cols = disparity_to_pointcloud(disparity, color, K, baseline, max_depth=100)

    # Z = fx*baseline/disparity = 500*0.5/25 = 10 m everywhere.
    assert np.allclose(pts[:, 2], 10.0, atol=1e-4)
    # One point at the principal point projects to ~(0, 0, 10).
    near_center = np.linalg.norm(pts[:, :2], axis=1) < 0.05
    assert near_center.any()
    assert len(pts) == len(cols)


def test_transform_points_applies_pose():
    pts = np.array([[0.0, 0.0, 5.0]])
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]  # pure translation
    out = transform_points(pts, pose)
    assert np.allclose(out[0], [1.0, 2.0, 8.0])
