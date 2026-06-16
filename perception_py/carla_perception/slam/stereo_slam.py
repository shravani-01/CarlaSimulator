"""Stereo SLAM = stereo VO front-end + loop closure + pose-graph back-end.

PIPELINE
--------
1. Run stereo VO per frame for accurate frame-to-frame motion (metric scale).
2. Every `keyframe_stride` frames, add a KEYFRAME node to a pose graph, with an
   odometry edge to the previous keyframe (from VO).
3. LOOP DETECTION: when a keyframe lands spatially near a much earlier keyframe,
   VERIFY it's truly the same place by matching features and solving PnP. If it
   passes, add a loop-closure edge (a metric relative-pose constraint).
4. OPTIMIZE the pose graph: the loop edges let drift be redistributed so the
   trajectory closes.

We keep the graph in the planar SE2 (x-z, yaw) form (cars drive on ~flat ground).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from carla_perception.slam.pose_graph import PoseGraph
from carla_perception.vo.stereo_vo import (
    StereoVO,
    compute_stereo_points,
    estimate_pose_pnp,
    good_matches,
)


def pose_to_se2(T: NDArray[np.float64]) -> tuple[float, float, float]:
    """4x4 pose (KITTI: x right, z forward) -> planar SE2 (x, z, yaw)."""
    yaw = float(np.arctan2(T[0, 2], T[2, 2]))
    return (float(T[0, 3]), float(T[2, 3]), yaw)


class StereoSLAM:
    """Stereo VO with loop closure and pose-graph optimization.

    Args:
        K, baseline: left intrinsics + stereo baseline (metres).
        keyframe_stride: add a graph node every N frames.
        loop_radius: search earlier keyframes within this many metres (generous,
            to tolerate drift; PnP verification rejects false matches).
        min_loop_gap: minimum keyframe-index separation to count as a loop.
        min_loop_inliers: PnP inliers required to accept a loop.
        loop_weight: relative weight of loop edges vs odometry edges.
    """

    def __init__(
        self,
        K: NDArray[np.floating],
        baseline: float,
        keyframe_stride: int = 10,
        loop_radius: float = 25.0,
        min_loop_gap: int = 50,
        min_loop_inliers: int = 25,
        loop_weight: float = 5.0,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.baseline = baseline
        self.keyframe_stride = keyframe_stride
        self.loop_radius = loop_radius
        self.min_loop_gap = min_loop_gap
        self.min_loop_inliers = min_loop_inliers
        self.loop_weight = loop_weight

        self.vo = StereoVO(K, baseline)
        self.graph = PoseGraph()
        self._orb = cv2.ORB_create(nfeatures=3000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.keyframes: list[dict] = []   # pose, pos(x,z), kp, des, des3d, X3d, frame
        self.loops: list[tuple[int, int]] = []
        self.vo_positions: NDArray | None = None        # before optimization
        self.optimized_positions: NDArray | None = None  # after

    def build(self, stereo_frames, total: int | None = None, log_every: int = 200) -> None:
        """Consume (left, right) pairs, building the keyframe graph + loop edges."""
        for f, (left, right) in enumerate(stereo_frames):
            self.vo.process(left, right)
            if f % self.keyframe_stride != 0:
                continue
            if total and f % log_every == 0:
                print(f"  frame {f}/{total} | keyframes={len(self.keyframes)} loops={len(self.loops)}")
            self._add_keyframe(f, left, right)

        self.vo_positions = self.graph.positions().copy()

    def _add_keyframe(self, frame_idx: int, left, right) -> None:
        pose = self.vo.pose.copy()
        kp, des, des3d, X3d = compute_stereo_points(
            left, right, self.K, self.baseline, self._orb, self._bf
        )
        node_id = len(self.keyframes)
        self.graph.add_node(node_id, pose_to_se2(pose))

        if self.keyframes:  # odometry edge from VO relative motion
            prev = self.keyframes[-1]
            rel = np.linalg.inv(prev["pose"]) @ pose
            self.graph.add_edge(prev["id"], node_id, pose_to_se2(rel), weight=1.0)

        kf = {
            "id": node_id, "frame": frame_idx, "pose": pose,
            "pos": np.array([pose[0, 3], pose[2, 3]]),
            "kp": kp, "des": des, "des3d": des3d, "X3d": X3d,
        }
        self._detect_loops(kf)
        self.keyframes.append(kf)

    def _detect_loops(self, kf: dict) -> None:
        if kf["des"] is None or len(self.keyframes) <= self.min_loop_gap:
            return
        # Candidate earlier keyframes: spatially near, far enough back in time.
        cands = [
            old for old in self.keyframes
            if kf["id"] - old["id"] >= self.min_loop_gap
            and old["X3d"] is not None
            and np.linalg.norm(old["pos"] - kf["pos"]) < self.loop_radius
        ]
        cands.sort(key=lambda o: np.linalg.norm(o["pos"] - kf["pos"]))

        for old in cands[:3]:  # verify the few nearest candidates
            matches = good_matches(self._bf, old["des3d"], kf["des"])
            if len(matches) < self.min_loop_inliers:
                continue
            obj = np.array([old["X3d"][m.queryIdx] for m in matches])
            img = np.array([kf["kp"][m.trainIdx].pt for m in matches])
            try:
                R, t, inliers = estimate_pose_pnp(obj, img, self.K)
            except RuntimeError:
                continue
            if inliers is None or len(inliers) < self.min_loop_inliers:
                continue
            # T_rel maps old frame -> current camera; edge measures curr in old frame.
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t.ravel()
            z = pose_to_se2(np.linalg.inv(T_rel))
            self.graph.add_edge(old["id"], kf["id"], z, weight=self.loop_weight)
            self.loops.append((old["id"], kf["id"]))
            return  # one good loop edge per keyframe is enough

    def optimize(self) -> None:
        self.graph.optimize()
        self.optimized_positions = self.graph.positions()

    def keyframe_frames(self) -> list[int]:
        return [kf["frame"] for kf in self.keyframes]
