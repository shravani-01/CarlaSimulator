"""Shape + loss/metric sanity tests for the depth network.

These need PyTorch, so they're skipped automatically if torch isn't installed
(keeps the geometry CI green without the heavy dependency).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from carla_perception.depth.losses import (  # noqa: E402
    depth_loss,
    depth_metrics,
    silog_loss,
)


def test_forward_shape_and_range():
    from carla_perception.depth.model import DepthResNet

    model = DepthResNet(backbone="resnet18", pretrained=False, min_depth=0.5, max_depth=80.0)
    model.eval()
    x = torch.randn(2, 3, 64, 128)
    with torch.no_grad():
        d = model(x)
    assert d.shape == (2, 1, 64, 128)
    assert float(d.min()) >= 0.5 - 1e-4 and float(d.max()) <= 80.0 + 1e-4


def test_silog_zero_for_perfect_prediction():
    t = torch.rand(1, 1, 8, 8) * 50 + 1
    mask = torch.ones_like(t, dtype=torch.bool)
    # A perfect prediction gives ~sqrt(1e-7)*10 ≈ 0.003 from the sqrt epsilon, not 0.
    assert float(silog_loss(t.clone(), t.clone(), mask)) < 1e-2


def test_metrics_perfect_prediction():
    t = torch.rand(1, 1, 8, 8) * 50 + 1
    mask = torch.ones_like(t, dtype=torch.bool)
    m = depth_metrics(t.clone(), t.clone(), mask)
    assert m["abs_rel"] < 1e-5 and m["rmse"] < 1e-4
    assert abs(m["delta1"] - 1.0) < 1e-6


def test_loss_is_finite_and_positive_on_random():
    pred = torch.rand(1, 1, 8, 8) * 40 + 1
    target = torch.rand(1, 1, 8, 8) * 40 + 1
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = depth_loss(pred, target, mask)
    assert torch.isfinite(loss) and float(loss) > 0


def test_empty_mask_gives_zero_loss():
    pred = torch.rand(1, 1, 4, 4) + 1
    target = torch.rand(1, 1, 4, 4) + 1
    mask = torch.zeros_like(target, dtype=torch.bool)
    assert float(depth_loss(pred, target, mask)) == 0.0
    assert np.isnan(depth_metrics(pred, target, mask)["abs_rel"])
