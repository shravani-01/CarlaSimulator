"""Prove a CARLA capture, written in KITTI format, loads back through our own
``KITTIOdometry`` pipeline — the whole reason for matching the layout.

No CARLA needed: we synthesize a couple of tiny stereo frames + poses, write
them, then read them straight back with the loader the VO/SLAM code already uses.
"""

import numpy as np
from carla_perception.datasets.kitti import KITTIOdometry

from carla_io.coords import fov_to_intrinsics
from carla_io.kitti_writer import KittiSequenceWriter


def test_roundtrip_through_kitti_loader(tmp_path):
    K = fov_to_intrinsics(64, 48, 90.0)
    baseline = 0.54
    w = KittiSequenceWriter(tmp_path, sequence="00")
    w.write_calib(K, baseline)

    left = np.full((48, 64, 3), 100, np.uint8)
    right = np.full((48, 64, 3), 120, np.uint8)
    # Frame 0 at origin, frame 1 moved 2 m forward (+Z in OpenCV frame).
    p0 = np.eye(4)
    p1 = np.eye(4)
    p1[2, 3] = 2.0
    w.add_frame(left, right, p0, 0.0)
    w.add_frame(left, right, p1, 0.05)
    w.finalize()

    data = KITTIOdometry(tmp_path, "00")
    assert len(data) == 2
    # Baseline must round-trip via P1 (-P1[0,3]/fx).
    assert np.isclose(data.baseline, baseline, atol=1e-6)
    # Intrinsics survive.
    assert np.isclose(data.K[0, 0], K[0, 0])
    # Ground-truth positions: frame 0 at origin, frame 1 two metres ahead.
    pos = data.gt_positions()
    assert np.allclose(pos[0], [0, 0, 0], atol=1e-6)
    assert np.allclose(pos[1], [0, 0, 2.0], atol=1e-6)
