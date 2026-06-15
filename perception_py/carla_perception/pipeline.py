"""Combined perception pipeline: detection + segmentation in one pass.

WHY THIS MODULE EXISTS
----------------------
So far Detector and Segmenter are separate. But the real system processes each
camera frame through *several* perception tasks together. Instead of scattering
`Detector(...)` and `Segmenter(...)` calls all over the codebase, we compose them
into ONE component with a single, stable API:

    pipeline = PerceptionPipeline()
    result   = pipeline.process(frame)     # -> detections + per-pixel labels
    vis      = pipeline.render(frame, result)

Everything downstream — the CARLA capture loop, the web demo, the evaluation
scripts — will call THIS pipeline. That keeps the "what perception does to a
frame" logic in exactly one place, which is the whole point of an integration
layer.

NOTE ON IMPORTS: this module only imports our own light wrapper classes at the
top. The heavy torch/ultralytics work still happens lazily inside those wrappers'
constructors, so importing `pipeline` (e.g. in tests) stays cheap; the cost is
only paid when you actually build a PerceptionPipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from carla_perception.detection.detector import Detection, Detector
from carla_perception.segmentation.segmenter import Segmenter


@dataclass
class PerceptionResult:
    """Everything we perceive about a single frame.

    Attributes:
        detections: list of detected objects (boxes + labels).
        label_map:  HxW array of per-pixel class ids (semantic segmentation).
    """

    detections: list[Detection]
    label_map: NDArray[np.uint8]


class PerceptionPipeline:
    """Runs detection + segmentation on a frame and renders the combined result.

    Args:
        device: passed through to both sub-models (None auto-picks).
        detector / segmenter: inject your own instances (useful for tests or to
            reuse already-loaded models); otherwise they're created for you.
    """

    def __init__(
        self,
        device: str | None = None,
        detector: Detector | None = None,
        segmenter: Segmenter | None = None,
    ) -> None:
        self.detector = detector or Detector(device=device)
        self.segmenter = segmenter or Segmenter(device=device)

    def process(self, frame: NDArray[np.uint8]) -> PerceptionResult:
        """Run all perception tasks on one frame."""
        detections = self.detector.detect(frame)
        label_map = self.segmenter.segment(frame)
        return PerceptionResult(detections=detections, label_map=label_map)

    def render(
        self,
        frame: NDArray[np.uint8],
        result: PerceptionResult,
        alpha: float = 0.5,
    ) -> NDArray[np.uint8]:
        """Draw both layers on one image: segmentation overlay first, boxes on top.

        Order matters: we lay the translucent class colors down first, then draw
        the detection boxes over them so the boxes stay crisp and readable.
        """
        out = self.segmenter.overlay(frame, result.label_map, alpha=alpha)
        out = self.detector.annotate(out, result.detections)
        return out

    def summarize(self, result: PerceptionResult) -> str:
        """Short text summary of a result (handy for logging/printing)."""
        n = len(result.detections)
        classes = self.segmenter.classes_present(result.label_map)
        return f"{n} objects detected; segmented classes: {', '.join(classes)}"
