"""Loader for the KITTI odometry dataset.

Expected layout (the standard KITTI odometry format):

    <root>/
      sequences/<seq>/image_0/000000.png ...   # grayscale LEFT camera frames
      sequences/<seq>/image_1/000000.png ...   # grayscale RIGHT camera frames
      sequences/<seq>/calib.txt                # projection matrices P0..P3
      poses/<seq>.txt                          # ground-truth poses (seq 00-10)

Each pose line is 12 numbers = a 3x4 [R|t] matrix giving the left camera's pose
in the first frame's coordinates. The camera position is its translation column.

Stereo note: P0 and P1 are the left/right projection matrices. The stereo
baseline (metres) = -P1[0,3] / P1[0,0] = fx*baseline / fx.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class KITTIOdometry:
    """Read frames, intrinsics, baseline, and ground-truth poses for one sequence.

    Args:
        root: dataset root containing `sequences/` and `poses/`.
        sequence: sequence id, e.g. "00".
        max_frames: optionally cap the number of frames.
    """

    def __init__(self, root: str | Path, sequence: str = "00", max_frames: int | None = None):
        self.root = Path(root)
        self.sequence = sequence
        self.seq_dir = self.root / "sequences" / sequence
        self.image_dir = self.seq_dir / "image_0"
        self.image_dir_right = self.seq_dir / "image_1"
        self.calib_file = self.seq_dir / "calib.txt"
        self.poses_file = self.root / "poses" / f"{sequence}.txt"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"KITTI images not found at {self.image_dir}. See docs/SETUP_KITTI.md."
            )

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.image_paths_right = sorted(self.image_dir_right.glob("*.png"))
        if max_frames is not None:
            self.image_paths = self.image_paths[:max_frames]
            self.image_paths_right = self.image_paths_right[:max_frames]

        calib = self._load_calib()
        self.P0 = calib["P0"]
        self.P1 = calib.get("P1")
        self.K = self.P0[:, :3]
        # Stereo baseline in metres (positive). Only meaningful if P1 exists.
        self.baseline = (
            float(-self.P1[0, 3] / self.P1[0, 0]) if self.P1 is not None else None
        )

    def _load_calib(self) -> dict[str, NDArray[np.float64]]:
        """Parse every 'Pk: ...' projection matrix from calib.txt into 3x4 arrays."""
        mats: dict[str, NDArray[np.float64]] = {}
        with open(self.calib_file) as f:
            for line in f:
                if ":" not in line:
                    continue
                name, rest = line.split(":", 1)
                if name.startswith("P"):
                    mats[name.strip()] = np.array(rest.split(), dtype=np.float64).reshape(3, 4)
        if "P0" not in mats:
            raise ValueError(f"P0 not found in {self.calib_file}")
        return mats

    def __len__(self) -> int:
        return len(self.image_paths)

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        """Yield each LEFT frame as a BGR image (mono VO uses this)."""
        for path in self.image_paths:
            img = cv2.imread(str(path))
            if img is None:
                raise OSError(f"failed to read {path}")
            yield img

    def stereo_frames(self) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
        """Yield (left, right) BGR image pairs for stereo VO."""
        for lp, rp in zip(self.image_paths, self.image_paths_right, strict=False):
            left, right = cv2.imread(str(lp)), cv2.imread(str(rp))
            if left is None or right is None:
                raise OSError(f"failed to read stereo pair {lp} / {rp}")
            yield left, right

    def gt_positions(self) -> NDArray[np.float64]:
        """Ground-truth camera positions, Nx3 (truncated to the frames we use)."""
        if not self.poses_file.exists():
            raise FileNotFoundError(
                f"Ground-truth poses not found at {self.poses_file} "
                "(only sequences 00-10 have them)."
            )
        poses = np.loadtxt(self.poses_file).reshape(-1, 3, 4)
        positions = poses[:, :, 3]
        return positions[: len(self.image_paths)]
