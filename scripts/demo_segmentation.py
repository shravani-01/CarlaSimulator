"""Demo: run semantic segmentation on one image and save a colored overlay.

It:
  1. loads an image (reusing the sample street scene by default),
  2. runs our Segmenter to label every pixel,
  3. prints which classes were found,
  4. saves an overlay image (original blended with the class colors).

Usage (project root, venv active):
    python scripts/demo_segmentation.py
    python scripts/demo_segmentation.py --image path/to/your_image.jpg
    python scripts/demo_segmentation.py --device cpu

First run downloads the DeepLabV3 weights (~160 MB).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
from carla_perception.segmentation.segmenter import Segmenter

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = Path("data/sample/bus.jpg")
OUTPUT_PATH = Path("outputs/demo/segmentation_overlay.jpg")


def ensure_sample_image() -> Path:
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[demo] downloading sample image -> {SAMPLE_PATH}")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic segmentation demo")
    parser.add_argument("--image", type=str, default=None, help="path to an image")
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else ensure_sample_image()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    segmenter = Segmenter(device=args.device)
    print(f"[demo] running segmentation on {segmenter.device} ...")
    label_map = segmenter.segment(image)

    found = segmenter.classes_present(label_map)
    print(f"\n[demo] classes found: {', '.join(found)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    overlay = segmenter.overlay(image, label_map, alpha=args.alpha)
    cv2.imwrite(str(OUTPUT_PATH), overlay)
    print(f"\n[demo] segmentation overlay saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
