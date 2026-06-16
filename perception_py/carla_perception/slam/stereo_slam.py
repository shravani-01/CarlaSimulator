"""Stereo SLAM = stereo VO front-end + loop closure + pose-graph back-end.

PIPELINE (split for fast iteration)
-----------------------------------
1. build_keyframes(): run stereo VO once and extract per-keyframe features +
   metric 3D points + pose. This is the SLOW part -> cache it to disk.
2. build_graph(): from cached keyframes, add odometry edges and detect+verify
   loop closures. FAST, so loop-closure params can be tuned cheaply.
3. optimize(): pose-graph optimization (robust loss) to redistribute drift.

We keep the graph in the planar SE2 (x-z, yaw) form (cars drive on ~flat ground).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from carla_perception.slam.pose_graph import PoseGraph, t2v, v2t
from carla_perception.vo.stereo_vo import (
    StereoVO,
    compute_stereo_points,
    estimate_pose_pnp,
    good_matches,
)


def pose_to_se2(T: NDArray[np.float64]) -> tuple[float, float, float]:
    """4x4 pose (KITTI: x right, z forward) -> planar SE2 (x, z, yaw).

    yaw uses atan2(R[2,0], R[0,0]) so that v2t(pose_to_se2(T)) reproduces the
    planar (x-z) submatrix of T exactly (KITTI's y axis points DOWN).
    """
    yaw = float(np.arctan2(T[2, 0], T[0, 0]))
    return (float(T[0, 3]), float(T[2, 3]), yaw)


class StereoSLAM:
    """Stereo VO with loop closure and pose-graph optimization."""

    def __init__(
        self,
        K: NDArray[np.floating],
        baseline: float,
        keyframe_stride: int = 10,
        loop_radius: float = 25.0,
        min_loop_gap: int = 50,
        min_loop_inliers: int = 30,
        loop_weight: float = 2.0,
        max_correction: float = 40.0,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.baseline = baseline
        self.keyframe_stride = keyframe_stride
        self.loop_radius = loop_radius
        self.min_loop_gap = min_loop_gap
        self.min_loop_inliers = min_loop_inliers
        self.loop_weight = loop_weight
        self.max_correction = max_correction

        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keyframes: list[dict] = []
        self.graph = PoseGraph()
        self.loops: list[tuple[int, int]] = []
        self.vo_positions: NDArray | None = None
        self.optimized_positions: NDArray | None = None

    # ---- Step 1: slow front-end (cacheable) -------------------------------
    def build_keyframes(self, stereo_frames, total: int | None = None, log_every: int = 200):
        """Run VO and store per-keyframe features/3D/pose (the slow part)."""
        vo = StereoVO(self.K, self.baseline)
        orb = cv2.ORB_create(nfeatures=3000)
        for f, (left, right) in enumerate(stereo_frames):
            vo.process(left, right)
            if f % self.keyframe_stride != 0:
                continue
            if total and f % log_every == 0:
                print(f"  frame {f}/{total} | keyframes={len(self.keyframes)}")
            kp, des, des3d, X3d = compute_stereo_points(
                left, right, self.K, self.baseline, orb, self._bf
            )
            kp_pts = np.array([k.pt for k in kp], dtype=np.float64) if kp else np.zeros((0, 2))
            self.keyframes.append({
                "id": len(self.keyframes), "frame": f,
                "pose": vo.pose.copy(),
                "pos": np.array([vo.pose[0, 3], vo.pose[2, 3]]),
                "kp_pts": kp_pts, "des": des, "des3d": des3d, "X3d": X3d,
            })

    def save_keyframes(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.keyframes, fh)

    def load_keyframes(self, path: str | Path) -> None:
        with open(path, "rb") as fh:
            self.keyframes = pickle.load(fh)

    # ---- Step 2: fast graph build + loop closure --------------------------
    def build_graph(self) -> None:
        """(Re)build the pose graph with odometry + loop-closure edges (fast)."""
        self.graph = PoseGraph()
        self.loops = []
        for kf in self.keyframes:
            se2 = pose_to_se2(kf["pose"])
            self.graph.add_node(kf["id"], se2)
            if kf["id"] > 0:
                prev_se2 = self.graph.nodes[kf["id"] - 1]
                rel = t2v(np.linalg.inv(v2t(prev_se2)) @ v2t(se2))
                self.graph.add_edge(kf["id"] - 1, kf["id"], tuple(rel), weight=1.0)
            self._detect_loops(kf)
        self.vo_positions = self.graph.positions().copy()

    def _detect_loops(self, kf: dict) -> None:
        if kf["des"] is None or kf["id"] <= self.min_loop_gap:
            return
        cands = [
            old for old in self.keyframes
            if old["id"] < kf["id"]
            and kf["id"] - old["id"] >= self.min_loop_gap
            and old["X3d"] is not None
            and np.linalg.norm(old["pos"] - kf["pos"]) < self.loop_radius
        ]
        cands.sort(key=lambda o: np.linalg.norm(o["pos"] - kf["pos"]))

        for old in cands[:3]:
            matches = good_matches(self._bf, old["des3d"], kf["des"])
            if len(matches) < self.min_loop_inliers:
                continue
            obj = np.array([old["X3d"][m.queryIdx] for m in matches])
            img = np.array([kf["kp_pts"][m.trainIdx] for m in matches])
            try:
                R, t, inliers = estimate_pose_pnp(obj, img, self.K)
            except RuntimeError:
                continue
            if inliers is None or len(inliers) < self.min_loop_inliers:
                continue
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t.ravel()
            z = pose_to_se2(np.linalg.inv(T_rel))

            # Gate: reject loops implying an implausible correction (false match).
            vo_rel = t2v(
                np.linalg.inv(v2t(self.graph.nodes[old["id"]])) @ v2t(self.graph.nodes[kf["id"]])
            )
            if np.linalg.norm(np.array(z[:2]) - vo_rel[:2]) > self.max_correction:
                continue

            self.graph.add_edge(old["id"], kf["id"], z, weight=self.loop_weight)
            self.loops.append((old["id"], kf["id"]))
            return

    def optimize(self, loss: str = "soft_l1", f_scale: float = 3.0) -> None:
        self.graph.optimize(loss=loss, f_scale=f_scale)
        self.optimized_positions = self.graph.positions()

    def keyframe_frames(self) -> list[int]:
        return [kf["frame"] for kf in self.keyframes]
