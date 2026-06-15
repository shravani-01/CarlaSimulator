"""Synthetic loop-closure test for the pose-graph optimizer.

We simulate a square loop, corrupt the odometry so the trajectory drifts and the
loop fails to close, add a single loop-closure edge, optimize, and check the
trajectory snaps back toward ground truth.
"""

import numpy as np
from carla_perception.slam.pose_graph import PoseGraph, t2v, v2t


def _rel(a, b):
    """Measured relative pose of b in a's frame (as a vector)."""
    return tuple(t2v(np.linalg.inv(v2t(a)) @ v2t(b)))


def _ate(positions, gt):
    return float(np.sqrt(np.mean(np.sum((positions - gt) ** 2, axis=1))))


def test_loop_closure_reduces_drift():
    rng = np.random.default_rng(0)

    # Ground-truth square loop: 40 poses around a 10x10 square, heading tangent.
    n = 40
    side = np.linspace(0, 10, n // 4, endpoint=False)
    pts = np.concatenate(
        [
            np.column_stack([side, np.zeros_like(side)]),            # bottom ->
            np.column_stack([np.full_like(side, 10), side]),         # right  ^
            np.column_stack([10 - side, np.full_like(side, 10)]),    # top    <-
            np.column_stack([np.zeros_like(side), 10 - side]),       # left   v
        ]
    )
    headings = np.array([0] * (n // 4) + [np.pi / 2] * (n // 4)
                        + [np.pi] * (n // 4) + [-np.pi / 2] * (n // 4))
    gt = [(float(p[0]), float(p[1]), float(h)) for p, h in zip(pts, headings, strict=True)]

    # Build a graph whose initial nodes are a DRIFTED integration of noisy odom.
    g = PoseGraph()
    g.add_node(0, gt[0])
    cur = np.array(gt[0])
    for k in range(1, n):
        true_rel = np.array(_rel(gt[k - 1], gt[k]))
        noisy_rel = true_rel + rng.normal(0, [0.05, 0.05, 0.03])
        # integrate noisy odometry to get a drifted initial estimate
        cur = t2v(v2t(tuple(cur)) @ v2t(tuple(noisy_rel)))
        g.add_node(k, tuple(cur))
        g.add_edge(k - 1, k, tuple(noisy_rel))  # odometry edge

    drifted = g.positions()
    ate_before = _ate(drifted, pts)

    # Loop-closure edge: last node revisits the first (true relative), high weight.
    g.add_edge(n - 1, 0, _rel(gt[n - 1], gt[0]), weight=10.0)
    g.optimize()

    ate_after = _ate(g.positions(), pts)

    # The optimized trajectory should be substantially closer to ground truth.
    assert ate_after < 0.5 * ate_before
