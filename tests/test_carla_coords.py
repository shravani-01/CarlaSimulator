"""Pin the CARLA(Unreal) -> KITTI(OpenCV) coordinate conversion.

These prove the change-of-basis is right WITHOUT needing CARLA installed: we feed
in synthetic ``get_matrix()`` values (pure translations / rotations) and check the
recovered OpenCV-frame motion has the correct sign on each axis. If any of these
flip, the recorded ground-truth trajectory would be mirrored.
"""

import numpy as np

from carla_io.coords import (
    fov_to_intrinsics,
    poses_relative_to_first,
    ue_matrix_to_opencv_c2w,
)


def _ue_translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """A CARLA get_matrix() for identity rotation at Unreal location (x, y, z)."""
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def test_output_rotation_is_proper_right_handed():
    # Converting an identity UE camera pose must yield a valid right-handed pose
    # (orthonormal rotation, determinant +1).
    T = ue_matrix_to_opencv_c2w(np.eye(4))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_forward_motion_maps_to_plus_z():
    # Driving forward in CARLA (+X) must read as +Z (forward) in the OpenCV frame.
    poses = [ue_matrix_to_opencv_c2w(_ue_translation_matrix(d, 0, 0)) for d in (0.0, 5.0)]
    rel = poses_relative_to_first(poses)
    assert np.allclose(rel[1][:3, 3], [0, 0, 5.0], atol=1e-9)


def test_rightward_motion_maps_to_plus_x():
    # Moving right in CARLA (+Y) must read as +X (right) in the OpenCV frame.
    poses = [ue_matrix_to_opencv_c2w(_ue_translation_matrix(0, d, 0)) for d in (0.0, 3.0)]
    rel = poses_relative_to_first(poses)
    assert np.allclose(rel[1][:3, 3], [3.0, 0, 0], atol=1e-9)


def test_upward_motion_maps_to_minus_y():
    # Moving up in CARLA (+Z) must read as -Y, because OpenCV's Y points DOWN.
    poses = [ue_matrix_to_opencv_c2w(_ue_translation_matrix(0, 0, d)) for d in (0.0, 2.0)]
    rel = poses_relative_to_first(poses)
    assert np.allclose(rel[1][:3, 3], [0, -2.0, 0], atol=1e-9)


def test_first_pose_is_identity():
    poses = [ue_matrix_to_opencv_c2w(_ue_translation_matrix(*t)) for t in [(1, 2, 3), (4, 5, 6)]]
    rel = poses_relative_to_first(poses)
    assert np.allclose(rel[0], np.eye(4))


def test_fov_to_intrinsics_90deg():
    # At 90 deg horizontal FOV, fx = width/2 exactly; principal point centred.
    K = fov_to_intrinsics(1280, 720, 90.0)
    assert np.isclose(K[0, 0], 640.0)
    assert np.isclose(K[0, 0], K[1, 1])  # square pixels
    assert np.isclose(K[0, 2], 640.0) and np.isclose(K[1, 2], 360.0)
