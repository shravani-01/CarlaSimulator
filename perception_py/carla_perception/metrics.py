"""Evaluation metrics for the perception stack.

Implemented now (Phase 0/1):
    - ate_rmse: Absolute Trajectory Error (translation RMSE) for odometry/SLAM
    - rpe_rmse: Relative Pose Error (translation RMSE over a fixed delta)
    - mean_iou: mean Intersection-over-Union for semantic segmentation

Detection mAP is added in Phase 1 (delegated to a standard implementation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ate_rmse(
    pred_xyz: NDArray[np.floating],
    gt_xyz: NDArray[np.floating],
) -> float:
    """Absolute Trajectory Error (translation RMSE), in meters.

    Args:
        pred_xyz: estimated positions, shape (N, 3).
        gt_xyz:   ground-truth positions, shape (N, 3).

    Returns:
        RMSE of per-frame Euclidean position error.
    """
    pred = np.asarray(pred_xyz, dtype=float)
    gt = np.asarray(gt_xyz, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    err = np.linalg.norm(pred - gt, axis=1)
    return float(np.sqrt(np.mean(err**2)))


def rpe_rmse(
    pred_xyz: NDArray[np.floating],
    gt_xyz: NDArray[np.floating],
    delta: int = 1,
) -> float:
    """Relative Pose Error (translation RMSE) over a fixed frame gap `delta`.

    Measures local drift rather than global alignment.
    """
    pred = np.asarray(pred_xyz, dtype=float)
    gt = np.asarray(gt_xyz, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    if delta < 1 or delta >= len(pred):
        raise ValueError("delta must be in [1, N-1]")
    pred_step = pred[delta:] - pred[:-delta]
    gt_step = gt[delta:] - gt[:-delta]
    err = np.linalg.norm(pred_step - gt_step, axis=1)
    return float(np.sqrt(np.mean(err**2)))


def mean_iou(
    pred: NDArray[np.integer],
    target: NDArray[np.integer],
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Mean IoU over classes for semantic segmentation label maps.

    Args:
        pred:   predicted class ids, any shape.
        target: ground-truth class ids, same shape as pred.
        num_classes: number of semantic classes.
        ignore_index: class id to exclude (e.g. unlabeled), optional.

    Returns:
        Mean IoU across classes that appear in the ground truth.
    """
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")

    ious: list[float] = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = pred == c
        tgt_c = target == c
        if not tgt_c.any():
            continue  # class absent in GT -> skip
        inter = np.logical_and(pred_c, tgt_c).sum()
        union = np.logical_or(pred_c, tgt_c).sum()
        ious.append(inter / union if union > 0 else 0.0)

    return float(np.mean(ious)) if ious else 0.0
