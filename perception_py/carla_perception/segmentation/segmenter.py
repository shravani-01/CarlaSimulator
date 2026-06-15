"""Semantic segmentation wrapped around torchvision DeepLabV3.

WHY THIS MODULE EXISTS
----------------------
Detection gives us boxes ("there is a car here"). Segmentation goes finer: it
labels EVERY pixel with a class (road, car, person, sidewalk, ...). That pixel
map is what lets a self-driving stack reason about the *drivable surface* and
free space, not just object locations.

We start from a torchvision DeepLabV3 model pretrained on Pascal VOC (21 classes,
including car / bus / bicycle / person) so we can see real segmentation output
on a laptop with zero training. Later we fine-tune / swap to road-centric classes
(Cityscapes / CARLA: road, sidewalk, lane, ...). As with the Detector, callers
talk only to OUR `Segmenter`, never to torchvision directly.

KEY IDEA: the model outputs one score per class per pixel. The predicted label
for a pixel is simply the class with the highest score (argmax).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _voc_palette(num_classes: int = 256) -> NDArray[np.uint8]:
    """Build the classic Pascal-VOC color map (one distinct RGB per class id).

    This is the standard bit-shuffling palette used by VOC so each class gets a
    visually distinct color. We only need the first ~21 entries here.
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for shift in range(8):
            r |= ((c >> 0) & 1) << (7 - shift)
            g |= ((c >> 1) & 1) << (7 - shift)
            b |= ((c >> 2) & 1) << (7 - shift)
            c >>= 3
        palette[i] = (r, g, b)
    return palette


class Segmenter:
    """Thin, stable wrapper over a pretrained DeepLabV3 segmentation model.

    Args:
        device: None lets us auto-pick (cuda > mps > cpu); or force "cpu"/"mps"/"cuda".
    """

    def __init__(self, device: str | None = None) -> None:
        # Lazy heavy imports (see Detector for the same reasoning).
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            deeplabv3_resnet50,
        )

        self._torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.preprocess = self.weights.transforms()  # resize + normalize
        self.classes: list[str] = list(self.weights.meta["categories"])
        self.palette = _voc_palette()

        self.model = deeplabv3_resnet50(weights=self.weights).eval().to(device)

    def segment(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Predict a per-pixel class-id map for one image.

        Args:
            image: HxWx3 BGR image (as from cv2.imread).

        Returns:
            HxW array of class ids (uint8), same height/width as the input.
        """
        import cv2
        torch = self._torch

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # HxWxC uint8 -> CxHxW tensor, then the model's own preprocessing.
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        batch = self.preprocess(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(batch)["out"][0]  # (num_classes, h', w')
        labels = out.argmax(0).byte().cpu().numpy()  # (h', w') at model resolution

        # Resize the label map back to the original image size.
        # INTER_NEAREST so we don't invent in-between class ids.
        return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

    def colorize(self, label_map: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Turn a class-id map into a BGR color image for visualization."""
        rgb = self.palette[label_map]              # HxWx3 in RGB
        return rgb[:, :, ::-1].copy()              # -> BGR for OpenCV

    def overlay(
        self, image: NDArray[np.uint8], label_map: NDArray[np.uint8], alpha: float = 0.5
    ) -> NDArray[np.uint8]:
        """Blend the color mask over the original image."""
        import cv2

        color = self.colorize(label_map)
        return cv2.addWeighted(image, 1 - alpha, color, alpha, 0.0)

    def classes_present(self, label_map: NDArray[np.uint8]) -> list[str]:
        """Human-readable names of the classes that appear in a label map."""
        return [self.classes[i] for i in np.unique(label_map) if i < len(self.classes)]
