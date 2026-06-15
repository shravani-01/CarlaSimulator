"""Demo: run the object detector on one image and save an annotated copy.

This is the "see it work" script. It:
  1. loads an image (a sample street scene if you don't pass one),
  2. runs our Detector on it,
  3. prints every object found,
  4. saves an annotated image with boxes drawn.

Usage (from the project root, venv active):
    python scripts/demo_detection.py
    python scripts/demo_detection.py --image path/to/your_image.jpg
    python scripts/demo_detection.py --device mps      # use Apple Silicon GPU

First run downloads the YOLO weights (~6 MB) and, if needed, the sample image.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
from carla_perception.detection.detector import Detector

# A classic sample image (a street scene with a bus and people).
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = Path("data/sample/bus.jpg")
OUTPUT_PATH = Path("outputs/demo/detection_annotated.jpg")


def ensure_sample_image() -> Path:
    """Download the sample image once if the user didn't provide their own."""
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[demo] downloading sample image -> {SAMPLE_PATH}")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Object detection demo")
    parser.add_argument("--image", type=str, default=None, help="path to an image")
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else ensure_sample_image()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    # Build the detector and run it.
    detector = Detector(conf=args.conf, device=args.device)
    detections = detector.detect(image)

    # Report what we found.
    print(f"\n[demo] found {len(detections)} objects in {image_path}:")
    for d in detections:
        print(f"  - {d.label:<12} conf={d.confidence:.2f}  box={tuple(round(v) for v in d.xyxy)}")

    # Save an annotated copy.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    annotated = detector.annotate(image, detections)
    cv2.imwrite(str(OUTPUT_PATH), annotated)
    print(f"\n[demo] annotated image saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
