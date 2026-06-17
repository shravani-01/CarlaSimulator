"""Dataset + transforms for monocular depth training on CARLA recordings.

Reads (RGB left image, metric depth) pairs from a recording in KITTI layout that
also has a ``depth/`` folder (written by carla_io with ground-truth depth).

DESIGN NOTE
-----------
The image/depth *transforms* are plain NumPy functions (resize, normalize, flip,
valid-mask) so they're unit-tested without needing PyTorch installed. The
``CarlaDepthDataset`` class is a thin wrapper that imports torch lazily, so the
heavy dependency is only required when you actually train.

Key correctness points:
  * RGB is resized with bilinear interpolation; **depth with nearest-neighbour**
    (averaging depth across an edge would invent distances that don't exist).
  * RGB is normalised with ImageNet statistics, because the encoder backbone is
    pretrained on ImageNet and expects that input distribution.
  * A **valid mask** marks pixels with usable depth (0 < d <= max_depth); the sky
    and far clip are excluded so they don't dominate the loss.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from carla_io.depth import uint16_cm_to_meters

# ImageNet normalisation (the pretrained encoder expects this).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_depth_frames(root: str | Path, sequence: str = "00") -> list[tuple[Path, Path]]:
    """Return (rgb_path, depth_path) pairs for frames that have BOTH files."""
    seq = Path(root) / "sequences" / sequence
    img_dir, depth_dir = seq / "image_0", seq / "depth"
    pairs = []
    for img in sorted(img_dir.glob("*.png")):
        d = depth_dir / img.name
        if d.exists():
            pairs.append((img, d))
    return pairs


def load_rgb(path: str | Path) -> NDArray[np.uint8]:
    """Load an image as HxWx3 RGB uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth_m(path: str | Path) -> NDArray[np.float32]:
    """Load a 16-bit-centimetre depth PNG as HxW float32 metres."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"failed to read {path}")
    return uint16_cm_to_meters(raw)


def prepare_sample(
    rgb_u8: NDArray[np.uint8],
    depth_m: NDArray[np.floating],
    size: tuple[int, int] = (192, 640),
    augment: bool = False,
    max_depth: float = 80.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Resize + normalise one (RGB, depth) pair into model-ready arrays.

    Returns:
        image: 3xHxW float32, ImageNet-normalised.
        depth: HxW float32 metres (resized nearest).
        mask:  HxW bool, True where depth is usable (0 < d <= max_depth).
    """
    rng = rng or np.random.default_rng()
    h, w = size
    img = cv2.resize(rgb_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(
        np.asarray(depth_m, dtype=np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    )

    if augment:
        if rng.random() < 0.5:  # horizontal flip (both image and depth)
            img = img[:, ::-1, :].copy()
            depth = depth[:, ::-1].copy()
        # mild brightness jitter on the image only
        factor = float(rng.uniform(0.8, 1.2))
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    mask = (depth > 0.0) & (depth <= max_depth)

    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    img_chw = np.transpose(img_f, (2, 0, 1)).copy()
    return img_chw, depth.astype(np.float32), mask


def make_dataset(
    root: str | Path,
    sequence: str = "00",
    split: str = "train",
    size: tuple[int, int] = (192, 640),
    val_frac: float = 0.2,
    augment: bool | None = None,
    max_depth: float = 80.0,
):
    """Build a ``CarlaDepthDataset`` (imports torch lazily)."""
    return CarlaDepthDataset(root, sequence, split, size, val_frac, augment, max_depth)


class CarlaDepthDataset:
    """PyTorch ``Dataset`` of (image, depth, mask) tensors from a CARLA recording.

    The train/val split is a **contiguous** block (val = the last ``val_frac`` of
    the drive) rather than random, so near-identical neighbouring frames don't leak
    between train and val.
    """

    def __init__(
        self,
        root: str | Path,
        sequence: str = "00",
        split: str = "train",
        size: tuple[int, int] = (192, 640),
        val_frac: float = 0.2,
        augment: bool | None = None,
        max_depth: float = 80.0,
    ):
        import torch  # noqa: F401  (validate torch is present early)

        self.size = size
        self.max_depth = max_depth
        self.augment = (split == "train") if augment is None else augment
        pairs = list_depth_frames(root, sequence)
        if not pairs:
            raise FileNotFoundError(
                f"no (image, depth) pairs under {root}/sequences/{sequence}; "
                "record with sensors.depth=true (see docs/SETUP_CARLA.md)."
            )
        n_train = int(round(len(pairs) * (1.0 - val_frac)))
        self.pairs = pairs[:n_train] if split == "train" else pairs[n_train:]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import torch

        img_path, depth_path = self.pairs[idx]
        img, depth, mask = prepare_sample(
            load_rgb(img_path),
            load_depth_m(depth_path),
            size=self.size,
            augment=self.augment,
            max_depth=self.max_depth,
        )
        return {
            "image": torch.from_numpy(img),
            "depth": torch.from_numpy(depth)[None],  # 1xHxW
            "mask": torch.from_numpy(mask)[None],
        }
