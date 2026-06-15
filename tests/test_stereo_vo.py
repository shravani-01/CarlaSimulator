"""Synthetic tests for stereo VO core math (no images needed).

The headline test proves the key advantage over monocular: PnP recovers the
camera translation at TRUE METRIC SCALE (magnitude matches), not just direction.
"""

import numpy as np
from carla_perception.vo.stereo_vo import estimate_pose_pnp, stereo_depth


def _project(K, R, t, points_3d):
    cam = (R @ points_3d.T + t.reshape(3, 1)).T
    pix = (K @ cam.T).T
    return pix[:, :2] / pix[:, 2:3]


def test_stereo_depth_formula():
    # Z = fx*baseline/disparity. fx=700, baseline=0.5, disparity=35 -> Z=10.
    assert abs(stereo_depth(35.0, fx=700.0, baseline=0.5) - 10.0) < 1e-9


def test_pnp_recovers_metric_pose():
    rng = np.random.default_rng(0)
    f, cx, cy = 700.0, 320.0, 240.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    # Metric 3D points in the previous camera frame.
    pts = np.column_stack(
        [rng.uniform(-3, 3, 200), rng.uniform(-3, 3, 200), rng.uniform(6, 20, 200)]
    )

    # Known METRIC motion (note the 1.5 m forward translation).
    ang = np.deg2rad(4.0)
    R_gt = np.array(
        [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
    )
    t_gt = np.array([0.8, -0.1, 1.5])

    image_points = _project(K, R_gt, t_gt, pts)
    R_est, t_est, _ = estimate_pose_pnp(pts, image_points, K)

    assert np.allclose(R_est, R_gt, atol=1e-3)
    # The whole point of stereo: translation magnitude is recovered, not just dir.
    assert np.allclose(t_est.ravel(), t_gt, atol=1e-2)
