"""Decode CARLA depth-camera frames into metric depth, and store them compactly.

HOW CARLA ENCODES DEPTH
-----------------------
CARLA's depth camera doesn't output a "distance image" directly. It packs the
distance into the R, G, B bytes of an ordinary image. The decode (from the CARLA
docs) is:

    normalized = (R + G * 256 + B * 256 * 256) / (256**3 - 1)   # in [0, 1]
    depth_metres = 1000 * normalized                            # CARLA's far clip is 1000 m

We isolate this here (with a unit test) because a wrong channel order silently
produces garbage depth, which would poison every downstream training run.

STORAGE
-------
Float depth maps are big. We save them as 16-bit PNGs in *centimetres*
(`uint16`, so 0-65535 cm = 0-655.35 m at 1 cm resolution), which is lossless
enough for driving distances and compresses well.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_CARLA_FAR_M = 1000.0
_MAX_CM = 65535  # uint16 ceiling -> 655.35 m


def carla_depth_to_meters(bgra: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Decode a CARLA depth frame (raw BGRA buffer) to metric depth in metres.

    Args:
        bgra: HxWx4 uint8 array as delivered by a CARLA ``sensor.camera.depth``
              (channel order B, G, R, A).

    Returns:
        HxW float32 array of depth in metres (0 .. 1000).
    """
    a = np.asarray(bgra, dtype=np.float64)
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1.0)
    return (_CARLA_FAR_M * normalized).astype(np.float32)


def meters_to_uint16_cm(depth_m: NDArray[np.floating]) -> NDArray[np.uint16]:
    """Quantise a metric depth map to uint16 centimetres (clipped at 655.35 m)."""
    cm = np.clip(np.asarray(depth_m, dtype=np.float64) * 100.0, 0, _MAX_CM)
    return cm.astype(np.uint16)


def uint16_cm_to_meters(depth_cm: NDArray[np.unsignedinteger]) -> NDArray[np.float32]:
    """Inverse of :func:`meters_to_uint16_cm` -> depth in metres."""
    return (np.asarray(depth_cm, dtype=np.float32) / 100.0).astype(np.float32)
