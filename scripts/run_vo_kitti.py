"""Run monocular VO on a KITTI sequence, score it, and plot vs ground truth.

Pipeline:
  1. load a KITTI sequence (frames + intrinsics + ground-truth poses),
  2. run MonocularVO frame-by-frame to estimate the camera trajectory,
  3. align the estimate to ground truth (Umeyama) and compute ATE / RPE,
  4. save a top-down plot of estimated vs ground-truth path.

Usage (project root, venv active):
    python scripts/run_vo_kitti.py --root /path/to/kitti_odometry --sequence 00
    python scripts/run_vo_kitti.py --root ... --sequence 00 --max-frames 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write a file
import matplotlib.pyplot as plt
import numpy as np
from carla_perception.datasets.kitti import KITTIOdometry
from carla_perception.trajectory import align_umeyama, evaluate_trajectory
from carla_perception.vo.monocular_vo import MonocularVO

OUTPUT_PATH = Path("outputs/demo/vo_kitti_trajectory.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="VO on KITTI + ATE/RPE + plot")
    parser.add_argument("--root", required=True, help="KITTI odometry dataset root")
    parser.add_argument("--sequence", default="00", help="sequence id, e.g. 00")
    parser.add_argument("--max-frames", type=int, default=None, help="cap frame count")
    args = parser.parse_args()

    data = KITTIOdometry(args.root, args.sequence, max_frames=args.max_frames)
    print(f"[vo] sequence {args.sequence}: {len(data)} frames")
    print(f"[vo] intrinsics K=\n{data.K}")

    vo = MonocularVO(K=data.K)
    for i, frame in enumerate(data.frames()):
        vo.process(frame)
        if i % 100 == 0:
            print(f"  processed frame {i}/{len(data)}")

    est = np.array(vo.trajectory).reshape(-1, 3)
    gt = data.gt_positions()
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    metrics = evaluate_trajectory(est, gt)
    print(
        f"\n[vo] ATE = {metrics['ate']:.3f} m | "
        f"RPE = {metrics['rpe']:.3f} m | recovered scale = {metrics['scale']:.3f}"
    )

    aligned, _ = align_umeyama(est, gt)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.plot(gt[:, 0], gt[:, 2], label="ground truth", linewidth=2)
    plt.plot(aligned[:, 0], aligned[:, 2], label="VO (aligned)", linestyle="--")
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(f"KITTI seq {args.sequence}: VO vs ground truth (ATE={metrics['ate']:.2f} m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    print(f"[vo] trajectory plot saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
