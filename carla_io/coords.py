"""Coordinate conversions: CARLA / Unreal frames -> KITTI / OpenCV convention.

WHY THIS MODULE EXISTS
----------------------
CARLA inherits Unreal Engine's **left-handed** coordinate system, while every
piece of geometry code we already wrote (VO, stereo, SLAM, the KITTI loader)
assumes the **right-handed OpenCV/KITTI** convention. If we get this conversion
wrong, the recorded ground-truth trajectory is mirror-flipped and nothing lines
up - the same class of bug that cost us hours on SE2 yaw and the splat export.

So we isolate the conversion into pure functions that need no CARLA install and
are pinned by unit tests (`tests/test_carla_coords.py`).

THE TWO CONVENTIONS
-------------------
CARLA / Unreal (left-handed):
    world & camera-local axes:  X forward,  Y right,  Z up

KITTI / OpenCV (right-handed):
    camera axes:                X right,    Y down,   Z forward

TWO CHANGES OF BASIS
--------------------
1. Unreal camera-local -> OpenCV camera:
       x_cv =  y_ue   (right)
       y_cv = -z_ue   (down)
       z_cv =  x_ue   (forward)
   i.e. p_cv = A @ p_ue   with   A = [[0, 1, 0], [0, 0, -1], [1, 0, 0]].

2. Unreal world (left-handed) -> a right-handed world: flip the Y axis,
       p_rh = D @ p_ue   with   D = diag(1, -1, 1).

A CARLA `Transform.get_matrix()` returns M: camera-local(UE) -> world(UE).
The camera-to-world pose in our convention is therefore

    T = blkdiag(D, 1) @ M @ blkdiag(A^-1, 1)

where A^-1 = A.T (A is orthogonal) maps an OpenCV-camera point into UE-camera
coords. The result T maps an OpenCV-camera point into the right-handed world.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Unreal-camera -> OpenCV-camera basis (p_cv = A @ p_ue).
_A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
# Unreal-world (left-handed) -> right-handed world (flip Y).
_D = np.diag([1.0, -1.0, 1.0])


def _homogeneous(rot: NDArray, trans: NDArray) -> NDArray:
    """Assemble a 4x4 from a 3x3 rotation and a length-3 translation."""
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def ue_matrix_to_opencv_c2w(m: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert a CARLA camera ``get_matrix()`` to an OpenCV camera-to-world pose.

    Args:
        m: 4x4 matrix from ``carla.Transform.get_matrix()`` - maps a point in the
           camera's local (Unreal) frame to the (Unreal) world frame.

    Returns:
        4x4 camera-to-world matrix in the right-handed OpenCV/KITTI convention:
        it maps an OpenCV-camera point (x-right, y-down, z-forward) to a
        right-handed world.
    """
    m = np.asarray(m, dtype=np.float64)
    D_h = _homogeneous(_D, np.zeros(3))
    A_inv_h = _homogeneous(_A.T, np.zeros(3))
    return D_h @ m @ A_inv_h


def poses_relative_to_first(c2w_list: list[NDArray]) -> list[NDArray[np.float64]]:
    """Express a list of camera-to-world poses relative to the first one.

    KITTI's ``poses/<seq>.txt`` gives each camera's pose in the *first frame's*
    camera coordinates, so frame 0 is the identity. This does exactly that:
    ``P_i = inv(T_0) @ T_i``.
    """
    if not c2w_list:
        return []
    T0_inv = np.linalg.inv(np.asarray(c2w_list[0], dtype=np.float64))
    return [T0_inv @ np.asarray(T, dtype=np.float64) for T in c2w_list]


def pose_to_kitti_row(pose: NDArray[np.floating]) -> str:
    """Flatten a 4x4 (or 3x4) pose to a KITTI poses-file line (12 numbers)."""
    p = np.asarray(pose, dtype=np.float64)[:3, :4]
    return " ".join(f"{v:.9e}" for v in p.reshape(-1))


def fov_to_intrinsics(width: int, height: int, fov_deg: float) -> NDArray[np.float64]:
    """Pinhole intrinsics K from a CARLA RGB camera's image size + horizontal FOV.

    CARLA cameras are ideal pinholes: the horizontal field of view fixes the focal
    length, ``fx = (width / 2) / tan(fov / 2)``. Pixels are square (fy = fx) and
    the principal point sits at the image centre.
    """
    f = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
