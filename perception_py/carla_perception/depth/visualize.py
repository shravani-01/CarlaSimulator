"""Turn a metric depth map into a readable colour image (for eval previews).

Pure NumPy/OpenCV (no torch), so it's unit-tested and usable anywhere.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def colorize_depth(
    depth_m: NDArray[np.floating], max_depth: float = 80.0
) -> NDArray[np.uint8]:
    """Map metric depth (metres) to a BGR colour image.

    Near = bright/warm, far = dark; invalid pixels (depth <= 0) are black.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    norm = np.clip(d, 0.0, max_depth) / max_depth
    inv = (255.0 * (1.0 - norm)).astype(np.uint8)  # closer -> larger value
    color = cv2.applyColorMap(inv, cv2.COLORMAP_MAGMA)
    color[d <= 0.0] = 0
    return color


def side_by_side(
    rgb: NDArray[np.uint8],
    pred_m: NDArray[np.floating],
    gt_m: NDArray[np.floating] | None = None,
    max_depth: float = 80.0,
) -> NDArray[np.uint8]:
    """Stack [RGB | predicted depth | (optional) GT depth] horizontally (BGR)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    panels = [bgr, colorize_depth(pred_m, max_depth)]
    if gt_m is not None:
        panels.append(colorize_depth(gt_m, max_depth))
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    return np.hstack(panels)
