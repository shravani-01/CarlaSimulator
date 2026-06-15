"""Demo: run the full perception pipeline (detection + segmentation) on one image.

It loads an image, runs the combined PerceptionPipeline, prints a summary, and
saves a single image showing the segmentation overlay WITH detection boxes on top.

Usage (project root, venv active):
    python scripts/demo_perception.py
    python scripts/demo_perception.py --image path/to/your_image.jpg
    python scripts/demo_perception.py --device cpu
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
from carla_perception.pipeline import PerceptionPipeline

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = Path("data/sample/bus.jpg")
OUTPUT_PATH = Path("outputs/demo/perception_combined.jpg")


def ensure_sample_image() -> Path:
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[demo] downloading sample image -> {SAMPLE_PATH}")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined perception demo")
    parser.add_argument("--image", type=str, default=None, help="path to an image")
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else ensure_sample_image()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    pipeline = PerceptionPipeline(device=args.device)
    result = pipeline.process(image)

    print(f"\n[demo] {pipeline.summarize(result)}")
    for d in result.detections:
        print(f"  - {d.label:<12} conf={d.confidence:.2f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    vis = pipeline.render(image, result)
    cv2.imwrite(str(OUTPUT_PATH), vis)
    print(f"\n[demo] combined visualization saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
