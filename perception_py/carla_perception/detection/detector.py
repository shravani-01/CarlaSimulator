"""Object detection wrapped around Ultralytics YOLO.

WHY THIS MODULE EXISTS
----------------------
Our perception stack needs to find dynamic objects (vehicles, pedestrians,
cyclists, traffic lights, ...) in every camera frame. Training a detector from
scratch is slow and data-hungry, so we start from a *pretrained* YOLO model and
wrap it behind a small, stable API.

The wrapper matters: the rest of the codebase talks to OUR `Detector` and gets
back clean `Detection` objects. It never imports `ultralytics` directly. That
means later we can fine-tune on CARLA data, change the YOLO version, or even
swap in a totally different detector — and no calling code has to change.

This is the same design idea as the whole project: a stable interface on the
outside, swappable implementation on the inside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Detection:
    """One detected object in an image.

    Attributes:
        cls_id:     integer class id from the model (e.g. 2 == "car" in COCO).
        label:      human-readable class name (e.g. "car").
        confidence: model's confidence in this detection, 0..1.
        xyxy:       bounding box as (x1, y1, x2, y2) pixel coordinates,
                    i.e. top-left corner and bottom-right corner.
    """

    cls_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class Detector:
    """Thin, stable wrapper over an Ultralytics YOLO model.

    Args:
        weights: path or name of the YOLO weights. "yolo11n.pt" is the smallest
                 ("n" = nano) pretrained model — fast, downloads automatically
                 the first time, good enough to see results on a laptop.
        conf:    minimum confidence to keep a detection (filters weak guesses).
        iou:     IoU threshold for non-max suppression (removes duplicate boxes
                 covering the same object).
        device:  None lets Ultralytics pick; "cpu" forces CPU; "mps" uses Apple
                 Silicon GPU; "cuda" uses an NVIDIA GPU.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | None = None,
    ) -> None:
        # Imported lazily (inside __init__, not at top of file) so that simply
        # importing this module doesn't require the heavy torch/ultralytics
        # stack. You only pay that cost when you actually build a Detector.
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        """Run detection on a single image.

        Args:
            image: an HxWx3 image array (as returned by cv2.imread, i.e. BGR).

        Returns:
            A list of Detection objects (possibly empty).
        """
        # verbose=False keeps Ultralytics from printing a log line per call.
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou, device=self.device, verbose=False
        )
        result = results[0]  # we passed one image, so we take the first result
        names = result.names  # dict: class_id -> class_name

        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    cls_id=cls_id,
                    label=names[cls_id],
                    confidence=float(box.conf[0]),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    def annotate(
        self, image: NDArray[np.uint8], detections: list[Detection]
    ) -> NDArray[np.uint8]:
        """Draw boxes + labels on a copy of the image (for visualization)."""
        import cv2

        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{det.label} {det.confidence:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return out
