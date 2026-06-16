"""Dense stereo reconstruction: stereo pair -> colored 3D point cloud.

WHY THIS MODULE EXISTS
----------------------
Stereo VO gave us *sparse* 3D points (just at feature locations) to estimate
motion. For a 3D *map* we want DENSE geometry: a depth value at (almost) every
pixel. We compute a dense disparity map with Semi-Global Block Matching (SGBM),
convert disparity to metric depth (Z = fx·baseline / disparity), and back-project
every valid pixel into a 3D point coloured by the image. Fuse those across frames
using the camera poses and you get a reconstructed 3D scene.

The core functions here are numpy/OpenCV only (no Open3D), so they're unit-tested
without any heavy 3D dependency.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def disparity_sgbm(left: NDArray, right: NDArray, num_disp: int = 128, block: int = 5) -> NDArray:
    """Dense disparity (pixels) for a rectified stereo pair via SGBM.

    num_disp must be divisible by 16. Returns float32 disparity (0 where unknown).
    """
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        disp12MaxDiff=1,
    )
    return sgbm.compute(gl, gr).astype(np.float32) / 16.0  # SGBM returns fixed-point ×16


def disparity_to_pointcloud(
    disparity: NDArray,
    color_bgr: NDArray,
    K: NDArray,
    baseline: float,
    min_disparity: float = 1.0,
    max_depth: float = 40.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Back-project a disparity map to a coloured 3D point cloud (camera frame).

    Returns:
        (points Nx3 in metres, colors Nx3 RGB in 0..1) for valid pixels only.
    """
    h, w = disparity.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    valid = disparity > min_disparity
    Z = np.zeros_like(disparity)
    Z[valid] = fx * baseline / disparity[valid]
    valid &= (Z > 0) & (max_depth > Z)

    Zv = Z[valid]
    X = (us[valid] - cx) * Zv / fx
    Y = (vs[valid] - cy) * Zv / fy
    points = np.stack([X, Y, Zv], axis=1).astype(np.float64)

    colors = color_bgr[valid][:, ::-1].astype(np.float64) / 255.0  # BGR -> RGB, 0..1
    return points, colors


def transform_points(points: NDArray, pose: NDArray) -> NDArray:
    """Apply a 4x4 camera->world pose to Nx3 points."""
    return (pose[:3, :3] @ points.T + pose[:3, 3:4]).T


def voxel_downsample(
    points: NDArray, colors: NDArray, voxel: float
) -> tuple[NDArray, NDArray]:
    """Keep one point per voxel cell (cheap numpy downsampling)."""
    if voxel <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx], colors[idx]


def write_ply(path, points: NDArray, colors: NDArray) -> None:
    """Write a binary PLY point cloud (no Open3D needed). colors in 0..1 RGB."""
    pts = np.asarray(points, dtype=np.float32)
    rgb = (np.clip(np.asarray(colors), 0, 1) * 255).astype(np.uint8)
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    # Interleave xyz (float32) + rgb (uint8) per vertex into a structured array.
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("r", "u1"), ("g", "u1"), ("b", "u1")])
    verts = np.empty(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    verts["r"], verts["g"], verts["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(verts.tobytes())
