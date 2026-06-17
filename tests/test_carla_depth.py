"""Pin the CARLA depth decode + the uint16-centimetre round trip.

A wrong channel order or scale here would silently corrupt every depth label, so
we check the known anchor points of CARLA's encoding.
"""

import numpy as np

from carla_io.depth import (
    carla_depth_to_meters,
    meters_to_uint16_cm,
    uint16_cm_to_meters,
)


def _bgra(b, g, r):
    return np.array([[[b, g, r, 255]]], dtype=np.uint8)


def test_zero_code_is_zero_metres():
    assert np.allclose(carla_depth_to_meters(_bgra(0, 0, 0)), 0.0)


def test_all_white_is_far_clip():
    # R=G=B=255 -> normalized = 1.0 -> 1000 m (CARLA's far clip).
    assert np.allclose(carla_depth_to_meters(_bgra(255, 255, 255)), 1000.0)


def test_red_lsb_is_smallest_step():
    # R=1, else 0 -> 1000 / (256^3 - 1) metres, the finest representable step.
    expected = 1000.0 / (256**3 - 1)
    assert np.allclose(carla_depth_to_meters(_bgra(0, 0, 1)), expected)


def test_uint16_cm_roundtrip():
    d = np.array([[0.0, 1.23, 50.0, 655.0]], dtype=np.float32)
    back = uint16_cm_to_meters(meters_to_uint16_cm(d))
    assert np.allclose(back, d, atol=0.01)  # within 1 cm


def test_uint16_cm_clips_far():
    d = np.array([[700.0]], dtype=np.float32)  # beyond 655.35 m ceiling
    assert meters_to_uint16_cm(d)[0, 0] == 65535
