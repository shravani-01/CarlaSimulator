"""Run stereo SLAM (VO + loop closure) on a KITTI sequence; plot before/after.

Shows the payoff of loop closure: the raw VO trajectory drifts, and pose-graph
optimization with loop-closure edges snaps it back toward ground truth.

Usage (project root):
    PYTHONPATH=perception_py .venv/bin/python scripts/run_slam_kitti.py \
        --root ~/datasets/kitti/dataset --sequence 00
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from carla_perception.datasets.kitti import KITTIOdometry
from carla_perception.slam.stereo_slam import StereoSLAM
from carla_perception.trajectory import align_umeyama, evaluate_trajectory

OUTPUT_PATH = Path("outputs/demo/slam_kitti_trajectory.png")


def _to3d(xz: np.ndarray) -> np.ndarray:
    """(N,2) planar (x,z) -> (N,3) with y=0, for the 3D alignment/metrics."""
    return np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo SLAM (loop closure) on KITTI")
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10, help="keyframe stride")
    args = parser.parse_args()

    data = KITTIOdometry(args.root, args.sequence, max_frames=args.max_frames)
    print(f"[slam] sequence {args.sequence}: {len(data)} frames, baseline={data.baseline:.3f} m")

    slam = StereoSLAM(K=data.K, baseline=data.baseline, keyframe_stride=args.stride)
    slam.build(data.stereo_frames(), total=len(data))
    print(f"[slam] {len(slam.keyframes)} keyframes, {len(slam.loops)} loop closures detected")
    slam.optimize()

    # Ground-truth positions at keyframe frames, in the x-z plane.
    gt_all = data.gt_positions()
    kf_frames = slam.keyframe_frames()
    gt = gt_all[kf_frames][:, [0, 2]]

    before = evaluate_trajectory(_to3d(slam.vo_positions), _to3d(gt))
    after = evaluate_trajectory(_to3d(slam.optimized_positions), _to3d(gt))
    print(f"\n[slam] ATE before (VO only)        = {before['ate']:.3f} m")
    print(f"[slam] ATE after  (loop-closed SLAM) = {after['ate']:.3f} m")
    if slam.loops:
        print(f"[slam] drift reduced by {100 * (1 - after['ate'] / before['ate']):.1f}%")

    a_before, _ = align_umeyama(_to3d(slam.vo_positions), _to3d(gt))
    a_after, _ = align_umeyama(_to3d(slam.optimized_positions), _to3d(gt))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.plot(gt[:, 0], gt[:, 1], label="ground truth", linewidth=2, color="k")
    plt.plot(a_before[:, 0], a_before[:, 2], "--", label=f"VO only (ATE {before['ate']:.1f} m)", alpha=0.8)
    plt.plot(a_after[:, 0], a_after[:, 2], "--", label=f"SLAM loop-closed (ATE {after['ate']:.1f} m)", alpha=0.9)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(f"KITTI seq {args.sequence}: loop-closure SLAM ({len(slam.loops)} loops)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    print(f"[slam] plot saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
