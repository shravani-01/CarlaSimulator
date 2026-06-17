"""Monocular visual odometry: estimate camera motion from a single camera.

WHY THIS MODULE EXISTS
----------------------
Perception (detection/segmentation/tracking) tells us about OTHER things in the
scene. Visual odometry (VO) tells us about US: how the camera itself moved
between frames. Chained over time, that's the vehicle's trajectory - the
foundation of SLAM, mapping, and localization, and exactly the "estimating
geometric entities" skill the target roles ask for.

THE GEOMETRY (intuition)
------------------------
1. Find the same feature points in two consecutive frames (ORB + matching).
2. How those matched points shifted encodes the camera's motion. The
   "essential matrix" captures that relationship for a calibrated camera.
3. Decomposing the essential matrix recovers the rotation R and translation
   direction t between the two views.

IMPORTANT - MONOCULAR SCALE AMBIGUITY
-------------------------------------
With ONE camera you cannot know absolute scale (a small nearby motion looks
identical to a large far one). So t is only a *direction* (unit length). Real
systems fix scale using an IMU, stereo, wheel odometry, or known object sizes -
which is why the target roles emphasize Visual-INERTIAL SLAM. We expose `scale`
so a caller can supply it later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray


def estimate_relative_pose(
    pts1: NDArray[np.floating],
    pts2: NDArray[np.floating],
    K: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Recover relative camera motion from matched points in two views.

    Args:
        pts1: Nx2 pixel coordinates in the first image.
        pts2: Nx2 pixel coordinates of the SAME points in the second image.
        K:    3x3 camera intrinsic matrix.

    Returns:
        (R, t, inlier_mask):
            R: 3x3 rotation taking a point from camera-1 frame to camera-2 frame.
            t: 3x1 unit translation direction (scale is unknown - see module docs).
            inlier_mask: which input matches were geometrically consistent.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    # findEssentialMat uses RANSAC to be robust to wrong matches (outliers).
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    # recoverPose picks the physically valid R,t (points in front of both cameras).
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, mask.ravel().astype(bool)


@dataclass
class MonocularVO:
    """Accumulates relative poses into a camera trajectory.

    Args:
        K: 3x3 intrinsics.
        orb_features: number of ORB keypoints to detect per frame.
        ratio: Lowe ratio for filtering ambiguous matches (lower = stricter).
    """

    K: NDArray[np.floating]
    orb_features: int = 2000
    ratio: float = 0.75

    # Internal state (not constructor args).
    R_world: NDArray[np.float64] = field(init=False)
    t_world: NDArray[np.float64] = field(init=False)
    trajectory: list[NDArray[np.float64]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=self.orb_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prev_kp = None
        self._prev_des = None
        self.R_world = np.eye(3)
        self.t_world = np.zeros((3, 1))
        self.trajectory = [self.t_world.copy()]

    def _match(self, des1, des2) -> tuple[list, list]:
        """k-NN match + Lowe ratio test to keep only confident matches."""
        knn = self._bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in knn if m.distance < self.ratio * n.distance]
        return good

    def process(self, frame: NDArray[np.uint8], scale: float = 1.0) -> NDArray[np.float64]:
        """Process one new frame; returns the current camera position (3,1).

        The first frame just initializes; from the second frame on we estimate
        motion relative to the previous frame and append to the trajectory.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        kp, des = self._orb.detectAndCompute(gray, None)

        if self._prev_des is not None and des is not None:
            good = self._match(self._prev_des, des)
            if len(good) >= 8:  # need enough matches for a stable estimate
                pts1 = np.float64([self._prev_kp[m.queryIdx].pt for m in good])
                pts2 = np.float64([kp[m.trainIdx].pt for m in good])
                R, t, _ = estimate_relative_pose(pts1, pts2, self.K)
                # Accumulate into a world pose. Translation is scaled by `scale`
                # (unknown for pure monocular; default 1.0).
                self.t_world = self.t_world + scale * (self.R_world @ t)
                self.R_world = R @ self.R_world
                self.trajectory.append(self.t_world.copy())

        self._prev_kp, self._prev_des = kp, des
        return self.t_world
