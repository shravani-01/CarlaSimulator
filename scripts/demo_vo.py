"""Demo: visualize ORB feature matches between two frames (the basis of VO).

VO starts by finding the same points in consecutive frames. This script shows
those matches so you can SEE what the geometry is computed from. By default it
makes two frames by panning a crop across the sample image (good for visualizing
matches); for a *meaningful trajectory* use a real driving sequence (KITTI/CARLA).

Usage (project root, venv active):
    python scripts/demo_vo.py
    python scripts/demo_vo.py --image1 a.jpg --image2 b.jpg
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from carla_perception.vo.monocular_vo import estimate_relative_pose

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = Path("data/sample/bus.jpg")
OUTPUT_PATH = Path("outputs/demo/vo_matches.jpg")


def ensure_sample_image() -> Path:
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def default_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two frames simulating a small sideways camera shift, from the sample image."""
    img = cv2.imread(str(ensure_sample_image()))
    h, w = img.shape[:2]
    cw, ch = int(w * 0.8), h
    a = img[0:ch, 0:cw].copy()
    b = img[0:ch, (w - cw) : w].copy()  # shifted crop
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description="VO feature-matching demo")
    parser.add_argument("--image1", type=str, default=None)
    parser.add_argument("--image2", type=str, default=None)
    args = parser.parse_args()

    if args.image1 and args.image2:
        a, b = cv2.imread(args.image1), cv2.imread(args.image2)
    else:
        a, b = default_pair()

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    print(f"[demo] {len(kp1)} & {len(kp2)} keypoints, {len(good)} good matches")

    # Draw the matches.
    vis = cv2.drawMatches(a, kp1, b, kp2, good[:80], None, flags=2)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), vis)
    print(f"[demo] match visualization saved -> {OUTPUT_PATH}")

    # Attempt a relative-pose estimate (illustrative only on this planar pair).
    if len(good) >= 8:
        h, w = a.shape[:2]
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
        R, t, mask = estimate_relative_pose(pts1, pts2, K)
        print(f"[demo] estimated translation direction (unit): {t.ravel().round(3)}")
        print(f"[demo] inlier matches: {int(mask.sum())}/{len(good)}")
        print("[demo] NOTE: this pair is ~planar, so the pose is illustrative. "
              "Use a real driving sequence for a meaningful trajectory.")


if __name__ == "__main__":
    main()
