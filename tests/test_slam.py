"""Test the SE3->SE2 conversion used by stereo SLAM.

The key invariant: v2t(pose_to_se2(T)) must equal the planar (x-z) submatrix of
a pure-yaw 3D pose. If yaw sign/convention is wrong (as it was), loop-closure
optimization rotates translations the wrong way and diverges.
"""

import numpy as np
from carla_perception.slam.pose_graph import v2t
from carla_perception.slam.stereo_slam import pose_to_se2


def test_pose_to_se2_matches_planar_submatrix():
    a = 0.6  # a pure yaw about the (downward) y axis
    R = np.array(
        [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3], T[2, 3] = 2.0, 3.0  # x, z translation

    se2 = pose_to_se2(T)
    M = v2t(se2)

    # The planar pose: take rows/cols (x, z) of R plus the (x, z) translation.
    planar = np.array(
        [[R[0, 0], R[0, 2], T[0, 3]], [R[2, 0], R[2, 2], T[2, 3]], [0, 0, 1]]
    )
    assert np.allclose(M, planar, atol=1e-9)


def test_round_trip_composition_is_consistent():
    # Two pure-yaw poses; the SE2 relative must match the planar relative.
    def pose(a, x, z):
        T = np.eye(4)
        T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        T[0, 3], T[2, 3] = x, z
        return T

    Ti, Tj = pose(0.3, 1, 2), pose(0.9, 4, 6)
    rel_planar = v2t(pose_to_se2(np.linalg.inv(Ti) @ Tj))
    rel_se2 = np.linalg.inv(v2t(pose_to_se2(Ti))) @ v2t(pose_to_se2(Tj))
    assert np.allclose(rel_planar, rel_se2, atol=1e-9)
