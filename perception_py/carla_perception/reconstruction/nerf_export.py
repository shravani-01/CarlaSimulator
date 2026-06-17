"""Export posed images to a nerfstudio dataset (transforms.json) for splatting.

WHY THIS MODULE EXISTS
----------------------
Gaussian-Splatting trainers (nerfstudio's splatfacto, etc.) need posed images:
each image plus the camera-to-world pose it was taken from. The usual pipeline
runs COLMAP to *estimate* those poses. We already have poses - from our own
stereo VO/SLAM - so we skip COLMAP and feed our poses directly. That's the
integration: our geometry pipeline supplies the camera trajectory.

POSE CONVENTION (the classic gotcha)
------------------------------------
Our poses are OpenCV convention (camera +x right, +y down, +z forward).
nerfstudio's transforms.json expects OpenGL/Blender convention (+x right,
+y up, +z back). Converting is a flip of the camera y and z axes:
    c2w_opengl = c2w_opencv @ diag(1, -1, -1, 1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def opencv_to_opengl(c2w: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert an OpenCV camera-to-world 4x4 pose to OpenGL/Blender convention."""
    return np.asarray(c2w, dtype=np.float64) @ _OPENCV_TO_OPENGL


def build_transforms(
    frames: list[tuple[str, NDArray]],
    K: NDArray,
    width: int,
    height: int,
    convention: str = "opengl",
) -> dict:
    """Build a nerfstudio transforms.json dict.

    Args:
        frames: list of (relative_image_path, 4x4 camera-to-world pose [OpenCV]).
        K: 3x3 intrinsics. width/height: image size.
        convention: "opengl" applies the OpenCV->OpenGL flip (nerfstudio default);
            "opencv" leaves poses as-is (use if the result looks inside-out).
    """
    if convention not in ("opengl", "opencv"):
        raise ValueError("convention must be 'opengl' or 'opencv'")

    def pose(c2w):
        m = opencv_to_opengl(c2w) if convention == "opengl" else np.asarray(c2w, float)
        return m.tolist()

    return {
        "camera_model": "OPENCV",
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": int(width),
        "h": int(height),
        "frames": [
            {"file_path": path, "transform_matrix": pose(c2w)} for path, c2w in frames
        ],
    }
