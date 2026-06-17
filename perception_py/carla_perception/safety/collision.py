"""Forward-collision warning - the perception behind automatic emergency braking.

WHY THIS MODULE EXISTS
----------------------
A real car can't *decide to brake* without first perceiving that something is
dangerously close ahead. That perception is exactly what we can do from stereo:
measure the metric distance to each detected object, and flag the ones that are
(a) in the car's forward path and (b) closer than a safe distance. This is a
warning system - it does NOT steer or brake (that's planning/control, and needs
a simulator we control, e.g. CARLA).

We reuse the stereo disparity already available from the reconstruction module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

HAZARD_CLASSES = ("car", "truck", "bus", "person", "bicycle", "motorcycle")


def estimate_box_depth(
    disparity: NDArray, box, fx: float, baseline: float, min_disparity: float = 1.0
) -> float | None:
    """Median metric depth (m) of valid pixels inside a bounding box, or None."""
    x1, y1, x2, y2 = (int(v) for v in box)
    patch = disparity[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    valid = patch[patch > min_disparity]
    if valid.size < 10:
        return None
    return float(fx * baseline / np.median(valid))


@dataclass
class CollisionResult:
    obj: object            # the Detection or Track (has .xyxy, .label)
    distance: float | None  # metres, or None if depth unavailable
    hazard: bool            # in-path AND too close AND a relevant class


def forward_collision_check(
    objects: list,
    disparity: NDArray,
    K: NDArray,
    baseline: float,
    warn_distance: float = 10.0,
    center_fraction: float = 0.5,
    classes: tuple[str, ...] = HAZARD_CLASSES,
) -> list[CollisionResult]:
    """Flag objects that are in the ego forward path and closer than warn_distance.

    The "forward path" is approximated as a central horizontal band of the image
    (center_fraction of the width around the principal point).
    """
    fx, cx = K[0, 0], K[0, 2]
    width = disparity.shape[1]
    half = center_fraction * width / 2.0
    lo, hi = cx - half, cx + half

    results: list[CollisionResult] = []
    for o in objects:
        x1, _, x2, _ = o.xyxy
        box_cx = 0.5 * (x1 + x2)
        depth = estimate_box_depth(disparity, o.xyxy, fx, baseline)
        in_path = lo <= box_cx <= hi
        hazard = bool(
            depth is not None
            and in_path
            and depth < warn_distance
            and getattr(o, "label", None) in classes
        )
        results.append(CollisionResult(obj=o, distance=depth, hazard=hazard))
    return results
