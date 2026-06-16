"""Stereo visual odometry — VO with TRUE METRIC SCALE.

WHY STEREO FIXES MONOCULAR VO
-----------------------------
A single camera can't know real distances (scale ambiguity), so monocular VO
drifts in scale and collapses. Two cameras a known distance apart (the stereo
*baseline*) let us triangulate METRIC depth for each feature every frame. With
real 3D points in hand, we estimate motion by PnP (perspective-n-point):
match this frame's 3D points to where they reappear in the next frame and solve
for the camera motion that explains them — in real metres.

PIPELINE PER FRAME
------------------
1. Stereo-match left<->right features -> disparity -> metric 3D points.
2. Track left features into the next frame.
3. PnP(prev 3D points, their current 2D observations) -> metric relative pose.
4. Accumulate poses into a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray


def stereo_depth(disparity: NDArray, fx: float, baseline: float) -> NDArray:
    """Metric depth Z from disparity: Z = fx * baseline / disparity."""
    disparity = np.asarray(disparity, dtype=np.float64)
    return fx * baseline / disparity


def estimate_pose_pnp(
    object_points: NDArray[np.floating],
    image_points: NDArray[np.floating],
    K: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray]:
    """Solve for the camera pose that projects known 3D points to 2D observations.

    Args:
        object_points: Nx3 metric 3D points (in the reference/previous frame).
        image_points:  Nx2 pixel observations of those points in the new frame.
        K: 3x3 intrinsics.

    Returns:
        (R, t, inliers): rotation 3x3 and translation 3x1 mapping the reference
        frame into the new camera frame (METRIC), plus the RANSAC inlier indices.
    """
    object_points = np.ascontiguousarray(object_points, dtype=np.float64)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None,
        reprojectionError=2.0, iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnPRansac failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, inliers


def good_matches(bf, des1, des2, ratio: float = 0.75):
    """k-NN match + Lowe ratio test; returns the list of confident DMatches."""
    if des1 is None or des2 is None:
        return []
    knn = bf.knnMatch(des1, des2, k=2)
    return [
        m for pair in knn if len(pair) == 2
        for m, n in [pair] if m.distance < ratio * n.distance
    ]


def compute_stereo_points(left, right, K, baseline, orb, bf, ratio=0.75, max_row_diff=2.0):
    """Detect left features, match them in the right image, triangulate metric 3D.

    Returns (kpL, desL, des3d, X3d): all left keypoints/descriptors, plus the
    aligned descriptors and metric 3D points for the subset that got valid depth.
    Shared by StereoVO and the SLAM pipeline.
    """
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    kpL, desL = orb.detectAndCompute(gl, None)
    kpR, desR = orb.detectAndCompute(gr, None)
    if desL is None or desR is None:
        return kpL, desL, None, None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    des3d, X3d = [], []
    for m in good_matches(bf, desL, desR, ratio):
        xl, yl = kpL[m.queryIdx].pt
        xr, yr = kpR[m.trainIdx].pt
        disp = xl - xr
        if abs(yl - yr) > max_row_diff or disp <= 1e-6:
            continue
        Z = fx * baseline / disp
        X = (xl - cx) * Z / fx
        Y = (yl - cy) * Z / fy
        des3d.append(desL[m.queryIdx])
        X3d.append([X, Y, Z])

    if not X3d:
        return kpL, desL, None, None
    return kpL, desL, np.array(des3d, dtype=np.uint8), np.array(X3d, dtype=np.float64)


@dataclass
class StereoVO:
    """Feature-based stereo visual odometry with metric scale.

    Args:
        K: 3x3 intrinsics (left camera).
        baseline: stereo baseline in metres.
        orb_features: ORB keypoints per image.
        ratio: Lowe ratio for matching.
        max_row_diff: max vertical pixel difference for a valid stereo match
            (rectified images put matches on the same row).
    """

    K: NDArray[np.floating]
    baseline: float
    orb_features: int = 3000
    ratio: float = 0.75
    max_row_diff: float = 2.0

    pose: NDArray[np.float64] = field(init=False)
    trajectory: list[NDArray[np.float64]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=self.orb_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.pose = np.eye(4)
        self.trajectory = [self.pose[:3, 3].copy()]
        self._prev_des3d = None  # descriptors of prev-frame points that have 3D
        self._prev_X = None      # corresponding metric 3D points (prev frame)

    def _good(self, des1, des2):
        return good_matches(self._bf, des1, des2, self.ratio)

    def _stereo_points(self, left, right):
        """Triangulate metric 3D points for left features matched in the right image."""
        return compute_stereo_points(
            left, right, self.K, self.baseline, self._orb, self._bf,
            self.ratio, self.max_row_diff,
        )

    def process(self, left: NDArray[np.uint8], right: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Process one stereo pair; return the current camera position (3,)."""
        kpL, desL, des3d, X3d = self._stereo_points(left, right)

        if self._prev_des3d is not None and desL is not None and len(self._prev_des3d) >= 6:
            good = self._good(desL, self._prev_des3d)  # curr left -> prev 3D feats
            if len(good) >= 6:
                obj = np.array([self._prev_X[m.trainIdx] for m in good])
                img = np.array([kpL[m.queryIdx].pt for m in good])
                try:
                    R, t, _ = estimate_pose_pnp(obj, img, self.K)
                    T_rel = np.eye(4)        # maps prev frame -> current camera
                    T_rel[:3, :3] = R
                    T_rel[:3, 3] = t.ravel()
                    self.pose = self.pose @ np.linalg.inv(T_rel)  # accumulate cam->world
                except RuntimeError:
                    pass

        self.trajectory.append(self.pose[:3, 3].copy())
        self._prev_des3d, self._prev_X = des3d, X3d
        return self.pose[:3, 3]
