"""Synthetic tests for forward-collision warning (no images needed)."""

from dataclasses import dataclass

import numpy as np
from carla_perception.safety.collision import estimate_box_depth, forward_collision_check


@dataclass(eq=False)  # eq=False keeps identity hashing so we can use as dict keys
class FakeObj:
    label: str
    xyxy: tuple


def test_estimate_box_depth():
    disp = np.full((480, 640), 25.0, dtype=np.float32)  # constant disparity
    # fx*baseline/disp = 500*0.5/25 = 10 m
    d = estimate_box_depth(disp, (100, 100, 200, 200), fx=500.0, baseline=0.5)
    assert abs(d - 10.0) < 1e-6


def test_forward_collision_flags_close_centered_object():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    disp = np.zeros((480, 640), dtype=np.float32)

    # A car dead ahead (centered) and close: high disparity -> ~5 m.
    disp[200:300, 290:350] = 50.0   # 500*0.5/50 = 5 m
    close_center = FakeObj("car", (290, 200, 350, 300))

    # A car off to the far left edge (not in path), also close.
    disp[200:300, 0:40] = 50.0
    side = FakeObj("car", (0, 200, 40, 300))

    results = forward_collision_check([close_center, side], disp, K, baseline=0.5,
                                      warn_distance=10.0, center_fraction=0.5)
    by_obj = {r.obj: r for r in results}
    assert by_obj[close_center].hazard is True
    assert abs(by_obj[close_center].distance - 5.0) < 1e-6
    assert by_obj[side].hazard is False   # in range but not in the forward path
