"""Demo: detection + tracking over a sequence of frames -> annotated video.

Two ways to feed it frames:
  * --video path.mp4   : track objects through a real video clip, OR
  * (default)          : SYNTHESIZE a short clip by panning a crop across the
                         sample image. The objects shift frame-to-frame, so you
                         can watch the tracker keep stable IDs as they move.

It writes outputs/demo/tracking.mp4 with each object boxed and labelled
"id:label" - the id should stay the SAME for an object across the whole clip.

Usage (project root, venv active):
    python scripts/demo_tracking.py
    python scripts/demo_tracking.py --video path/to/clip.mp4
    python scripts/demo_tracking.py --frames 40 --device cpu
"""

from __future__ import annotations

import argparse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from carla_perception.detection.detector import Detector
from carla_perception.tracking.tracker import IoUTracker, Track

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = Path("data/sample/bus.jpg")
OUTPUT_PATH = Path("outputs/demo/tracking.mp4")


def ensure_sample_image() -> Path:
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def synth_frames(image: np.ndarray, n: int) -> Iterator[np.ndarray]:
    """Fake a moving camera by sliding a crop window left-to-right across an image."""
    h, w = image.shape[:2]
    cw, ch = int(w * 0.7), int(h * 0.9)
    xs = np.linspace(0, w - cw, n).astype(int)
    for x in xs:
        yield image[0:ch, x : x + cw].copy()


def video_frames(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"id:{t.track_id} {t.label}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection + tracking demo")
    parser.add_argument("--video", type=str, default=None, help="path to a video clip")
    parser.add_argument("--frames", type=int, default=30, help="synth frame count")
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    args = parser.parse_args()

    detector = Detector(device=args.device)
    tracker = IoUTracker()

    # Build the frame source.
    if args.video:
        frames = list(video_frames(Path(args.video)))
    else:
        image = cv2.imread(str(ensure_sample_image()))
        frames = list(synth_frames(image, args.frames))
    if not frames:
        raise SystemExit("No frames to process.")

    h, w = frames[0].shape[:2]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h)
    )

    print(f"[demo] processing {len(frames)} frames ...")
    for i, frame in enumerate(frames):
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        writer.write(draw_tracks(frame, tracks))
        ids = sorted(t.track_id for t in tracks)
        print(f"  frame {i:>3}: {len(tracks)} tracks, ids={ids}")

    writer.release()
    print(f"\n[demo] tracking video saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
