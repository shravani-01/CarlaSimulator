"""Loader for the KITTI odometry dataset.

Expected layout (the standard KITTI odometry format):

    <root>/
      sequences/<seq>/image_0/000000.png ...   # grayscale left camera frames
      sequences/<seq>/calib.txt                # projection matrices P0..P3
      poses/<seq>.txt                          # ground-truth poses (seq 00-10)

Each pose line is 12 numbers = a 3x4 [R|t] matrix giving the left camera's pose
in the first frame's coordinates. The camera position is its translation column.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class KITTIOdometry:
    """Read frames, intrinsics, and ground-truth poses for one KITTI sequence.

    Args:
        root: dataset root containing `sequences/` and `poses/`.
        sequence: sequence id, e.g. "00".
        max_frames: optionally cap the number of frames (handy for quick runs).
    """

    def __init__(self, root: str | Path, sequence: str = "00", max_frames: int | None = None):
        self.root = Path(root)
        self.sequence = sequence
        self.seq_dir = self.root / "sequences" / sequence
        self.image_dir = self.seq_dir / "image_0"
        self.calib_file = self.seq_dir / "calib.txt"
        self.poses_file = self.root / "poses" / f"{sequence}.txt"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"KITTI images not found at {self.image_dir}. See docs/SETUP_KITTI.md."
            )

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if max_frames is not None:
            self.image_paths = self.image_paths[:max_frames]
        self.K = self._load_intrinsics()

    def _load_intrinsics(self) -> NDArray[np.float64]:
        """Parse the 3x4 P0 projection matrix from calib.txt; K is its left 3x3."""
        with open(self.calib_file) as f:
            for line in f:
                if line.startswith("P0:"):
                    vals = np.array(line.split()[1:], dtype=np.float64).reshape(3, 4)
                    return vals[:, :3]
        raise ValueError(f"P0 not found in {self.calib_file}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        """Yield each frame as a BGR image (VO converts to grayscale internally)."""
        for path in self.image_paths:
            img = cv2.imread(str(path))
            if img is None:
                raise OSError(f"failed to read {path}")
            yield img

    def gt_positions(self) -> NDArray[np.float64]:
        """Ground-truth camera positions, Nx3 (truncated to the frames we use)."""
        if not self.poses_file.exists():
            raise FileNotFoundError(
                f"Ground-truth poses not found at {self.poses_file} "
                "(only sequences 00-10 have them)."
            )
        poses = np.loadtxt(self.poses_file).reshape(-1, 3, 4)
        positions = poses[:, :, 3]  # translation column per frame
        return positions[: len(self.image_paths)]
