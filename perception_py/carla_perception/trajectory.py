"""Trajectory alignment + evaluation for visual odometry.

WHY THIS MODULE EXISTS
----------------------
Monocular VO produces a trajectory in an arbitrary frame and an unknown scale
(see the scale-ambiguity note in vo/monocular_vo.py). To score it against
ground truth fairly, we first find the best similarity transform (scale +
rotation + translation) that overlays our estimated path onto the true one, then
measure the leftover error. That transform is the classic *Umeyama alignment*,
and aligning-then-scoring is the standard protocol for monocular VO/SLAM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from carla_perception.metrics import ate_rmse, rpe_rmse


def align_umeyama(
    pred: NDArray[np.floating], gt: NDArray[np.floating]
) -> tuple[NDArray[np.float64], dict]:
    """Find the similarity transform mapping `pred` onto `gt` and apply it.

    Solves for scale s, rotation R, translation t minimizing
        sum || gt_i - (s R pred_i + t) ||^2
    (Umeyama, 1991), then returns the transformed `pred`.

    Args:
        pred: Nx3 estimated positions.
        gt:   Nx3 ground-truth positions (same length).

    Returns:
        (aligned_pred, params) where params has keys "scale", "R", "t".
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.shape[1] != 3:
        raise ValueError(f"need matching Nx3 arrays, got {pred.shape} and {gt.shape}")

    n = pred.shape[0]
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    xp, xg = pred - mu_p, gt - mu_g

    cov = (xg.T @ xp) / n
    U, D, Vt = np.linalg.svd(cov)

    # Reflection guard: keep R a proper rotation (det = +1).
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    var_p = (xp**2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_p) if var_p > 0 else 1.0
    t = mu_g - scale * R @ mu_p

    aligned = (scale * (R @ pred.T)).T + t
    return aligned, {"scale": scale, "R": R, "t": t}


def evaluate_trajectory(
    pred: NDArray[np.floating], gt: NDArray[np.floating], delta: int = 1
) -> dict:
    """Align `pred` to `gt`, then report ATE, RPE, and the recovered scale."""
    aligned, params = align_umeyama(pred, gt)
    return {
        "ate": ate_rmse(aligned, gt),
        "rpe": rpe_rmse(aligned, gt, delta=delta),
        "scale": params["scale"],
    }
