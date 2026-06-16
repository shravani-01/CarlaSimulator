"""Dense 3D reconstruction from a KITTI stereo sequence.

For each sampled frame: dense disparity (SGBM) -> colored 3D points -> transform
to world using the stereo-VO pose -> fuse into one point cloud. Saves a .ply and
a top-down preview PNG.

Usage (project root):
    PYTHONPATH=perception_py .venv/bin/python scripts/run_reconstruct_kitti.py \
        --root ~/datasets/kitti/dataset --sequence 00 --max-frames 400 --stride 5
    # add --show to open an interactive Open3D viewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from carla_perception.datasets.kitti import KITTIOdometry
from carla_perception.reconstruction.stereo_pointcloud import (
    disparity_sgbm,
    disparity_to_pointcloud,
    transform_points,
    voxel_downsample,
    write_ply,
)
from carla_perception.vo.stereo_vo import StereoVO

PLY_PATH = Path("outputs/demo/reconstruction.ply")
PNG_PATH = Path("outputs/demo/reconstruction_topdown.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Dense stereo 3D reconstruction on KITTI")
    p.add_argument("--root", required=True)
    p.add_argument("--sequence", default="00")
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-depth", type=float, default=30.0)
    p.add_argument("--per-frame-points", type=int, default=30000, help="random subsample/frame")
    p.add_argument("--voxel", type=float, default=0.2, help="voxel downsample size (m)")
    p.add_argument("--show", action="store_true", help="open Open3D viewer")
    args = p.parse_args()

    data = KITTIOdometry(args.root, args.sequence, max_frames=args.max_frames)
    print(f"[recon] sequence {args.sequence}: {len(data)} frames, baseline={data.baseline:.3f} m")

    vo = StereoVO(data.K, data.baseline)
    rng = np.random.default_rng(0)
    all_pts, all_cols = [], []

    for f, (left, right) in enumerate(data.stereo_frames()):
        vo.process(left, right)  # update pose every frame
        if f % args.stride != 0:
            continue
        if f % 100 == 0:
            print(f"  frame {f}/{len(data)} | points so far: {sum(len(p) for p in all_pts):,}")
        disp = disparity_sgbm(left, right)
        pts, cols = disparity_to_pointcloud(disp, left, data.K, data.baseline, max_depth=args.max_depth)
        if len(pts) > args.per_frame_points:  # cap memory
            sel = rng.choice(len(pts), args.per_frame_points, replace=False)
            pts, cols = pts[sel], cols[sel]
        all_pts.append(transform_points(pts, vo.pose))
        all_cols.append(cols)

    points = np.concatenate(all_pts)
    colors = np.concatenate(all_cols)
    print(f"[recon] fused {len(points):,} points")

    # Voxel downsample + save PLY (pure numpy; no Open3D required).
    dp, dc = voxel_downsample(points, colors, args.voxel)
    PLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_ply(PLY_PATH, dp, dc)
    print(f"[recon] {len(dp):,} points after voxel downsample -> {PLY_PATH}")

    # Top-down preview (x-z), coloured by image RGB.
    plt.figure(figsize=(9, 9))
    plt.scatter(dp[:, 0], dp[:, 2], c=dc, s=0.4, marker=".")
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(f"KITTI seq {args.sequence}: dense stereo reconstruction (top-down)")
    plt.savefig(PNG_PATH, dpi=130, bbox_inches="tight")
    print(f"[recon] preview saved -> {PNG_PATH}")

    if args.show:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dp)
            pcd.colors = o3d.utility.Vector3dVector(dc)
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            print(f"[recon] --show needs Open3D ({e}). The .ply opens in any 3D viewer.")


if __name__ == "__main__":
    main()
