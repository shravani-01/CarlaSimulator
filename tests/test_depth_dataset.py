"""Tests for the monocular-depth dataset transforms (no PyTorch needed).

We exercise the pure-NumPy path: resize, ImageNet normalisation, the valid mask,
horizontal-flip augmentation, and frame discovery on a tiny recording written to
disk via the CARLA writer.
"""

import numpy as np
from carla_perception.depth.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    list_depth_frames,
    load_depth_m,
    prepare_sample,
)

from carla_io.coords import fov_to_intrinsics
from carla_io.kitti_writer import KittiSequenceWriter


def test_prepare_sample_shapes_and_mask():
    rgb = np.full((48, 64, 3), 128, np.uint8)
    depth = np.full((48, 64), 10.0, np.float32)
    depth[0, 0] = 0.0       # invalid (no return)
    depth[0, 1] = 500.0     # beyond max_depth -> invalid
    img, d, mask = prepare_sample(rgb, depth, size=(24, 32), max_depth=80.0)
    assert img.shape == (3, 24, 32)        # CHW
    assert d.shape == (24, 32) and mask.shape == (24, 32)
    assert mask.dtype == np.bool_
    # most pixels valid, but not the zero/far ones (they map into the resized grid)
    assert mask.sum() > 0 and mask.sum() < mask.size


def test_imagenet_normalisation_values():
    # A mid-grey image should map to (v/255 - mean)/std on each channel.
    rgb = np.full((10, 10, 3), 128, np.uint8)
    depth = np.full((10, 10), 5.0, np.float32)
    img, _, _ = prepare_sample(rgb, depth, size=(10, 10))
    expected = ((128.0 / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(img[:, 0, 0], expected, atol=1e-4)


def test_depth_uses_nearest_not_blurred():
    # A sharp depth edge must stay sharp (only two distinct values after resize).
    rgb = np.zeros((40, 40, 3), np.uint8)
    depth = np.zeros((40, 40), np.float32)
    depth[:, 20:] = 30.0
    _, d, _ = prepare_sample(rgb, depth, size=(20, 20))
    assert set(np.unique(d)).issubset({0.0, 30.0})


def test_hflip_consistent_between_image_and_depth():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:, :4] = 255           # left half bright
    depth = np.ones((8, 8), np.float32)
    depth[:, :4] = 50.0        # left half far
    rng = np.random.default_rng(0)
    # find a seed that flips; default_rng(0) first .random() < 0.5 triggers flip
    img, d, _ = prepare_sample(rgb, depth, size=(8, 8), augment=True, rng=rng)
    bright_left = img[0, 0, 0] > img[0, 0, -1]
    far_left = d[0, 0] > d[0, -1]
    # whichever way it flipped, image and depth must agree on which side is which
    assert bright_left == far_left


def test_list_depth_frames(tmp_path):
    K = fov_to_intrinsics(32, 24, 90.0)
    w = KittiSequenceWriter(tmp_path, "00")
    w.write_calib(K, 0.54)
    img = np.zeros((24, 32, 3), np.uint8)
    for i in range(3):
        w.add_frame(img, img, np.eye(4), i * 0.05, depth_m=np.full((24, 32), 7.0, np.float32))
    w.finalize()
    pairs = list_depth_frames(tmp_path, "00")
    assert len(pairs) == 3
    assert np.allclose(load_depth_m(pairs[0][1]), 7.0, atol=0.01)
