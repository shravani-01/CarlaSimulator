"""Write a CARLA capture as a KITTI-odometry dataset.

The payoff of matching KITTI's layout exactly: our existing ``KITTIOdometry``
loader, stereo VO, SLAM and the splat exporter all read CARLA data with ZERO new
code. CARLA just becomes another (better) data source.

Produced layout::

    <out>/
      sequences/<seq>/image_0/000000.png ...   # left camera
      sequences/<seq>/image_1/000000.png ...   # right camera
      sequences/<seq>/calib.txt                # P0, P1 (encodes the baseline)
      sequences/<seq>/times.txt                # timestamp per frame
      poses/<seq>.txt                          # 3x4 GT pose per frame (cam0 frame)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from carla_io.coords import pose_to_kitti_row


class KittiSequenceWriter:
    """Create the directory tree and stream frames + poses into KITTI format."""

    def __init__(self, root: str | Path, sequence: str = "00"):
        self.root = Path(root)
        self.sequence = sequence
        self.seq_dir = self.root / "sequences" / sequence
        self.left_dir = self.seq_dir / "image_0"
        self.right_dir = self.seq_dir / "image_1"
        for d in (self.left_dir, self.right_dir, self.root / "poses"):
            d.mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._poses: list[NDArray] = []
        self._times: list[float] = []

    def write_calib(self, K: NDArray, baseline: float) -> None:
        """Write KITTI calib.txt. P1's last column encodes the stereo baseline:
        ``P1[0,3] = -fx * baseline`` (so loaders recover ``baseline = -P1[0,3]/fx``).
        """
        fx = float(K[0, 0])
        P0 = np.hstack([K, np.zeros((3, 1))])
        P1 = P0.copy()
        P1[0, 3] = -fx * baseline
        lines = []
        for name, P in (("P0", P0), ("P1", P1), ("P2", P0), ("P3", P1)):
            vals = " ".join(f"{v:.12e}" for v in P.reshape(-1))
            lines.append(f"{name}: {vals}")
        (self.seq_dir / "calib.txt").write_text("\n".join(lines) + "\n")

    def add_frame(
        self,
        left_bgr: NDArray[np.uint8],
        right_bgr: NDArray[np.uint8],
        pose_c2w: NDArray,
        timestamp: float,
    ) -> None:
        """Save one stereo pair + its ground-truth camera-to-world pose."""
        import cv2

        name = f"{self._n:06d}.png"
        cv2.imwrite(str(self.left_dir / name), left_bgr)
        cv2.imwrite(str(self.right_dir / name), right_bgr)
        self._poses.append(np.asarray(pose_c2w, dtype=np.float64))
        self._times.append(float(timestamp))
        self._n += 1

    def finalize(self) -> None:
        """Write poses (relative to frame 0) and timestamps once capture is done."""
        from carla_io.coords import poses_relative_to_first

        rel = poses_relative_to_first(self._poses)
        (self.root / "poses" / f"{self.sequence}.txt").write_text(
            "\n".join(pose_to_kitti_row(p) for p in rel) + ("\n" if rel else "")
        )
        (self.seq_dir / "times.txt").write_text(
            "\n".join(f"{t:.6e}" for t in self._times) + ("\n" if self._times else "")
        )

    def __len__(self) -> int:
        return self._n
