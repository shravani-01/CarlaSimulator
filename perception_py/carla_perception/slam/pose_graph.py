"""Planar (SE2) pose-graph optimization - the SLAM back-end.

WHY THIS MODULE EXISTS
----------------------
Visual odometry estimates each step independently, so small errors accumulate
into large global drift (our KITTI loop didn't close). Pose-graph SLAM fixes this
by treating the trajectory as a graph:

    nodes  = camera poses (x, y, theta)            [we use the x-z ground plane]
    edges  = relative-pose CONSTRAINTS between nodes
             - odometry edges: consecutive poses, from VO
             - loop-closure edges: a revisited place ("node A == node B")

We then find the poses that best satisfy ALL constraints simultaneously
(nonlinear least squares). A single loop-closure edge lets the optimizer
redistribute the accumulated drift around the whole loop so it closes.

THE MATH (SE2)
--------------
A pose is (x, y, theta), represented as a 3x3 homogeneous matrix. For an edge
i->j with measured relative pose Z, the error is

    E = Z^-1 (T_i^-1 T_j)

i.e. how far the measured relative pose disagrees with the current estimate. We
minimize the sum of squared errors over all edges (anchoring one node to fix the
global gauge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

Pose = tuple[float, float, float]  # (x, y, theta)


def v2t(v: Pose) -> NDArray[np.float64]:
    """(x, y, theta) -> 3x3 SE2 homogeneous matrix."""
    x, y, th = v
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float64)


def t2v(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """3x3 SE2 matrix -> (x, y, theta) vector."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


@dataclass
class Edge:
    i: int
    j: int
    z: Pose            # measured relative pose of j in i's frame
    weight: float = 1.0


@dataclass
class PoseGraph:
    """A planar pose graph you build incrementally, then optimize.

    Usage:
        g = PoseGraph()
        g.add_node(0, (0, 0, 0)); g.add_node(1, (1, 0, 0)); ...
        g.add_edge(0, 1, measured_relative)         # odometry
        g.add_edge(800, 50, loop_relative, weight=5) # loop closure
        g.optimize()
    """

    nodes: dict[int, Pose] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, idx: int, pose: Pose) -> None:
        self.nodes[idx] = pose

    def add_edge(self, i: int, j: int, z: Pose, weight: float = 1.0) -> None:
        self.edges.append(Edge(i, j, z, weight))

    def positions(self) -> NDArray[np.float64]:
        """Nx2 (x, y) positions in node-id order."""
        return np.array([[self.nodes[k][0], self.nodes[k][1]] for k in sorted(self.nodes)])

    def optimize(
        self, max_iter: int = 30, loss: str = "soft_l1", f_scale: float = 3.0
    ) -> PoseGraph:
        """Optimize all poses to best satisfy the edges (first node anchored).

        A robust `loss` (default soft_l1) down-weights outlier edges so a few bad
        loop closures can't dominate the solution.
        """
        order = sorted(self.nodes)
        anchor = order[0]
        free = order[1:]
        index = {idx: k for k, idx in enumerate(free)}  # node id -> param block

        x0 = np.concatenate([np.array(self.nodes[idx], dtype=float) for idx in free])

        def poses_from(params: NDArray) -> dict[int, NDArray]:
            poses = {anchor: np.array(self.nodes[anchor], dtype=float)}
            for idx in free:
                k = index[idx]
                poses[idx] = params[3 * k : 3 * k + 3]
            return poses

        def residuals(params: NDArray) -> NDArray:
            poses = poses_from(params)
            res = []
            for e in self.edges:
                Ti, Tj, Z = v2t(tuple(poses[e.i])), v2t(tuple(poses[e.j])), v2t(e.z)
                err = t2v(np.linalg.inv(Z) @ np.linalg.inv(Ti) @ Tj)
                res.extend(np.sqrt(e.weight) * err)
            return np.array(res)

        # Exploit sparsity: each edge residual depends only on its two nodes, so
        # the Jacobian is mostly zeros. Telling scipy this lets it estimate the
        # Jacobian in a few evals (graph coloring) instead of one-per-parameter
        # -> orders of magnitude faster on large graphs.
        n_res, n_par = 3 * len(self.edges), len(x0)
        sparsity = None
        if n_res and n_par:
            sparsity = lil_matrix((n_res, n_par), dtype=int)
            for e_idx, e in enumerate(self.edges):
                r = 3 * e_idx
                for nid in (e.i, e.j):
                    if nid in index:
                        c = 3 * index[nid]
                        sparsity[r : r + 3, c : c + 3] = 1

        sol = least_squares(
            residuals, x0, jac_sparsity=sparsity, loss=loss, f_scale=f_scale,
            max_nfev=max_iter * 50,
        )

        updated = poses_from(sol.x)
        for idx in self.nodes:
            self.nodes[idx] = tuple(float(v) for v in updated[idx])
        return self
