"""Tests for the nerfstudio export (pose convention + transforms structure)."""

import numpy as np
from carla_perception.reconstruction.nerf_export import build_transforms, opencv_to_opengl


def test_opencv_to_opengl_is_an_involution_on_pose():
    rng = np.random.default_rng(0)
    c2w = np.eye(4)
    c2w[:3, 3] = rng.standard_normal(3)
    # Applying the flip twice returns the original pose.
    assert np.allclose(opencv_to_opengl(opencv_to_opengl(c2w)), c2w)


def test_opencv_to_opengl_flips_forward_axis():
    # Identity OpenCV pose looks along +z; OpenGL convention looks along -z,
    # so the third rotation column should flip sign.
    gl = opencv_to_opengl(np.eye(4))
    assert np.allclose(gl[:3, 2], [0, 0, -1])
    assert np.allclose(gl[:3, 1], [0, -1, 0])  # up axis flips too


def test_build_transforms_structure():
    K = np.array([[700, 0, 620], [0, 700, 187], [0, 0, 1]], dtype=float)
    frames = [("images/000000.png", np.eye(4)), ("images/000002.png", np.eye(4))]
    t = build_transforms(frames, K, width=1241, height=376)
    assert t["fl_x"] == 700 and t["cx"] == 620 and t["w"] == 1241
    assert len(t["frames"]) == 2
    assert t["frames"][0]["file_path"] == "images/000000.png"
    assert np.array(t["frames"][0]["transform_matrix"]).shape == (4, 4)
