"""Export a KITTI sequence + our stereo-VO poses to a nerfstudio dataset.

Produces a folder ready for Gaussian-Splatting training on a GPU (RunPod):
    <out>/images/000000.png ...
    <out>/transforms.json    (intrinsics + per-image camera poses from our VO)

Runs locally on CPU (no GPU needed) - it's just VO + writing files.

Usage (project root):
    PYTHONPATH=perception_py .venv/bin/python scripts/export_nerfstudio.py \
        --root ~/datasets/kitti/dataset --sequence 00 --max-frames 200 --stride 2 \
        --out outputs/nerf/seq00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from carla_perception.datasets.kitti import KITTIOdometry
from carla_perception.reconstruction.nerf_export import build_transforms
from carla_perception.vo.stereo_vo import StereoVO


def main() -> None:
    p = argparse.ArgumentParser(description="Export KITTI+VO poses to a nerfstudio dataset")
    p.add_argument("--root", required=True)
    p.add_argument("--sequence", default="00")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", default="outputs/nerf/seq00")
    p.add_argument("--convention", choices=["opengl", "opencv"], default="opengl",
                   help="pose convention; flip to 'opencv' if the splat looks inside-out")
    args = p.parse_args()

    data = KITTIOdometry(args.root, args.sequence, max_frames=args.max_frames)
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] sequence {args.sequence}: {len(data)} frames -> {out}")

    vo = StereoVO(data.K, data.baseline)
    frames: list[tuple[str, object]] = []
    width = height = None
    for f, (left, right) in enumerate(data.stereo_frames()):
        vo.process(left, right)
        if f % args.stride != 0:
            continue
        if width is None:
            height, width = left.shape[:2]
        name = f"{f:06d}.png"
        cv2.imwrite(str(img_dir / name), left)
        frames.append((f"images/{name}", vo.pose.copy()))
        if f % 100 == 0:
            print(f"  exported frame {f} ({len(frames)} kept)")

    transforms = build_transforms(frames, data.K, width, height, convention=args.convention)
    (out / "transforms.json").write_text(json.dumps(transforms, indent=2))
    print(f"[export] wrote {len(frames)} posed images + transforms.json")
    print(f"[export] upload '{out}' to RunPod - see docs/SETUP_GAUSSIAN_SPLATTING.md")


if __name__ == "__main__":
    main()
