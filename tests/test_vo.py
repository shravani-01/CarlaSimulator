"""Synthetic-geometry test for visual odometry.

We don't need any images to verify the MATH: we build a known 3D scene and two
known camera poses, project the 3D points into both cameras to get perfect 2D
correspondences, then check that estimate_relative_pose() recovers the motion we
put in. This is the gold-standard way to test geometric vision code.
"""

import cv2
import numpy as np
from carla_perception.vo.monocular_vo import estimate_relative_pose


def _project(K, R, t, points_3d):
    """Project Nx3 world points into a camera with pose (R, t) -> Nx2 pixels."""
    cam = (R @ points_3d.T + t).T  # into camera frame
    pix = (K @ cam.T).T
    return pix[:, :2] / pix[:, 2:3]


def test_recover_known_pose_from_synthetic_points():
    rng = np.random.default_rng(0)

    # Camera intrinsics.
    f, cx, cy = 800.0, 320.0, 240.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    # A cloud of 3D points spread in depth (NOT planar -> well-conditioned).
    n = 200
    points = np.column_stack(
        [
            rng.uniform(-2, 2, n),
            rng.uniform(-2, 2, n),
            rng.uniform(6, 14, n),  # in front of the camera
        ]
    )

    # Ground-truth motion: small rotation about Y + sideways/forward translation.
    angle = np.deg2rad(5.0)
    R_gt = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t_gt = np.array([[1.0], [0.0], [0.3]])  # direction matters; scale won't be recovered

    # Camera 1 is the world origin; camera 2 is moved by (R_gt, t_gt).
    pts1 = _project(K, np.eye(3), np.zeros((3, 1)), points)
    pts2 = _project(K, R_gt, t_gt, points)

    R_est, t_est, mask = estimate_relative_pose(pts1, pts2, K)

    # Rotation should match closely (no noise -> near exact).
    assert np.allclose(R_est, R_gt, atol=1e-2)

    # Translation is only up to scale, so compare DIRECTIONS via cosine similarity.
    t_gt_dir = (t_gt / np.linalg.norm(t_gt)).ravel()
    t_est_dir = (t_est / np.linalg.norm(t_est)).ravel()
    cos_sim = abs(float(t_gt_dir @ t_est_dir))
    assert cos_sim > 0.99

    # Most synthetic matches should be classed as inliers.
    assert mask.sum() > 0.8 * len(points)


def test_opencv_available():
    # Sanity: the geometry backend is importable.
    assert hasattr(cv2, "findEssentialMat")
