"""Run stereo SLAM (VO + loop closure) on KITTI; plot before/after.

The slow VO/feature pass is CACHED to disk, so re-running to tune loop-closure
params is fast. Use --rebuild to force re-extraction (e.g. after changing stride).

Usage (project root):
    PYTHONPATH=perception_py .venv/bin/python scripts/run_slam_kitti.py \
        --root ~/datasets/kitti/dataset --sequence 00 --stride 20

    # then tune cheaply (reuses the cache):
    ... --loop-radius 30 --min-inliers 25 --loop-weight 3 --f-scale 2
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
    return np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]])


def main() -> None:
    p = argparse.ArgumentParser(description="Stereo SLAM (loop closure) on KITTI")
    p.add_argument("--root", required=True)
    p.add_argument("--sequence", default="00")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--rebuild", action="store_true", help="force VO re-extraction")
    # tunable loop-closure / optimizer knobs
    p.add_argument("--loop-radius", type=float, default=25.0)
    p.add_argument("--min-gap", type=int, default=50)
    p.add_argument("--min-inliers", type=int, default=30)
    p.add_argument("--loop-weight", type=float, default=2.0)
    p.add_argument("--max-correction", type=float, default=40.0)
    p.add_argument("--f-scale", type=float, default=3.0)
    args = p.parse_args()

    data = KITTIOdometry(args.root, args.sequence, max_frames=args.max_frames)
    print(f"[slam] sequence {args.sequence}: {len(data)} frames, baseline={data.baseline:.3f} m")

    slam = StereoSLAM(
        K=data.K, baseline=data.baseline, keyframe_stride=args.stride,
        loop_radius=args.loop_radius, min_loop_gap=args.min_gap,
        min_loop_inliers=args.min_inliers, loop_weight=args.loop_weight,
        max_correction=args.max_correction,
    )

    cache = Path(f"outputs/cache/slam_kf_{args.sequence}_s{args.stride}.pkl")
    if cache.exists() and not args.rebuild:
        print(f"[slam] loading cached keyframes <- {cache}")
        slam.load_keyframes(cache)
    else:
        print("[slam] building keyframes (VO + features) ... [cached for next time]")
        slam.build_keyframes(data.stereo_frames(), total=len(data))
        slam.save_keyframes(cache)
        print(f"[slam] cached keyframes -> {cache}")

    slam.build_graph()
    print(f"[slam] {len(slam.keyframes)} keyframes, {len(slam.loops)} loop closures")
    slam.optimize(f_scale=args.f_scale)

    gt = data.gt_positions()[slam.keyframe_frames()][:, [0, 2]]
    before = evaluate_trajectory(_to3d(slam.vo_positions), _to3d(gt))
    after = evaluate_trajectory(_to3d(slam.optimized_positions), _to3d(gt))
    print(f"\n[slam] ATE before (VO only)         = {before['ate']:.3f} m")
    print(f"[slam] ATE after  (loop-closed SLAM) = {after['ate']:.3f} m")
    if slam.loops:
        print(f"[slam] change: {100 * (1 - after['ate'] / before['ate']):+.1f}%")

    a_before, _ = align_umeyama(_to3d(slam.vo_positions), _to3d(gt))
    a_after, _ = align_umeyama(_to3d(slam.optimized_positions), _to3d(gt))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.plot(gt[:, 0], gt[:, 1], "k", linewidth=2, label="ground truth")
    plt.plot(a_before[:, 0], a_before[:, 2], "--", alpha=0.8, label=f"VO only ({before['ate']:.1f} m)")
    plt.plot(a_after[:, 0], a_after[:, 2], "--", alpha=0.9, label=f"SLAM ({after['ate']:.1f} m)")
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
