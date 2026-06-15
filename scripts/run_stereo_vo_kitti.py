"""Run STEREO VO on a KITTI sequence, score it, and plot vs ground truth.

Unlike monocular, stereo VO recovers metric scale, so the trajectory should
actually resemble the real loop (and the recovered alignment scale should be ~1).

Usage (project root):
    PYTHONPATH=perception_py .venv/bin/python scripts/run_stereo_vo_kitti.py \
        --root ~/datasets/kitti/dataset --sequence 00 --max-frames 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from carla_perception.datasets.kitti import KITTIOdometry
from carla_perception.trajectory import align_umeyama, evaluate_trajectory
from carla_perception.vo.stereo_vo import StereoVO

OUTPUT_PATH = Path("outputs/demo/stereo_vo_kitti_trajectory.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo VO on KITTI + ATE/RPE + plot")
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    data = KITTIOdometry(args.root, args.sequence, max_frames=args.max_frames)
    print(f"[stereo-vo] sequence {args.sequence}: {len(data)} frames, baseline={data.baseline:.3f} m")

    vo = StereoVO(K=data.K, baseline=data.baseline)
    for i, (left, right) in enumerate(data.stereo_frames()):
        vo.process(left, right)
        if i % 100 == 0:
            print(f"  processed frame {i}/{len(data)}")

    est = np.array(vo.trajectory).reshape(-1, 3)
    gt = data.gt_positions()
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    metrics = evaluate_trajectory(est, gt)
    print(
        f"\n[stereo-vo] ATE = {metrics['ate']:.3f} m | RPE = {metrics['rpe']:.3f} m | "
        f"alignment scale = {metrics['scale']:.3f} (≈1.0 means metric scale recovered)"
    )

    aligned, _ = align_umeyama(est, gt)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.plot(gt[:, 0], gt[:, 2], label="ground truth", linewidth=2)
    plt.plot(aligned[:, 0], aligned[:, 2], label="stereo VO (aligned)", linestyle="--")
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(f"KITTI seq {args.sequence}: stereo VO vs GT (ATE={metrics['ate']:.2f} m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    print(f"[stereo-vo] trajectory plot saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
