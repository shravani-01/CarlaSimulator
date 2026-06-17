"""Tests for the depth colour-map preview (no torch needed)."""

import numpy as np
from carla_perception.depth.visualize import colorize_depth, side_by_side


def test_colorize_shape_and_invalid_is_black():
    depth = np.full((10, 12), 20.0, np.float32)
    depth[0, 0] = 0.0  # invalid
    vis = colorize_depth(depth, max_depth=80.0)
    assert vis.shape == (10, 12, 3) and vis.dtype == np.uint8
    assert np.all(vis[0, 0] == 0)  # invalid pixel painted black


def test_near_is_brighter_than_far():
    near = colorize_depth(np.full((4, 4), 2.0, np.float32))
    far = colorize_depth(np.full((4, 4), 70.0, np.float32))
    assert int(near.sum()) > int(far.sum())  # near maps to a brighter colour


def test_side_by_side_widths_add_up():
    rgb = np.zeros((10, 12, 3), np.uint8)
    pred = np.full((10, 12), 15.0, np.float32)
    gt = np.full((10, 12), 16.0, np.float32)
    out = side_by_side(rgb, pred, gt)
    assert out.shape[0] == 10
    assert out.shape[1] == 12 * 3  # three equal-height panels
