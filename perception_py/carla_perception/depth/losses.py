"""Losses + metrics for monocular depth, computed only on valid pixels.

SILog (scale-invariant log loss, Eigen et al. 2014) is the standard depth-training
loss: it penalises the *shape* of the depth error while being lenient about a
global scale offset, which stabilises training. We add a small L1 term in metres
so the absolute scale is also pinned (we have metric ground truth, so we want it).
"""

from __future__ import annotations

import torch


def silog_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lamb: float = 0.85
) -> torch.Tensor:
    """Scale-invariant log loss over valid pixels.

    g = log(pred) - log(target);  L = sqrt(mean(g^2) - lamb * mean(g)^2) * 10.
    """
    m = mask & (pred > 0) & (target > 0)
    if m.sum() == 0:
        return pred.sum() * 0.0  # no valid pixels -> zero (keeps graph intact)
    g = torch.log(pred[m]) - torch.log(target[m])
    return torch.sqrt((g**2).mean() - lamb * (g.mean() ** 2) + 1e-7) * 10.0


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute depth error (metres) over valid pixels."""
    m = mask & (target > 0)
    if m.sum() == 0:
        return pred.sum() * 0.0
    return (pred[m] - target[m]).abs().mean()


def depth_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, l1_weight: float = 0.1
) -> torch.Tensor:
    """Combined training loss: SILog + a small absolute-scale L1 term."""
    return silog_loss(pred, target, mask) + l1_weight * masked_l1(pred, target, mask)


@torch.no_grad()
def depth_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Standard depth metrics over valid pixels: AbsRel, RMSE, delta<1.25."""
    m = mask & (target > 0) & (pred > 0)
    if m.sum() == 0:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    p, t = pred[m], target[m]
    abs_rel = ((p - t).abs() / t).mean().item()
    rmse = torch.sqrt(((p - t) ** 2).mean()).item()
    ratio = torch.maximum(p / t, t / p)
    delta1 = (ratio < 1.25).float().mean().item()
    return {"abs_rel": abs_rel, "rmse": rmse, "delta1": delta1}
