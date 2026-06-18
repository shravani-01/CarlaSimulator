"""Predict depth on images that have NO ground truth (e.g. real KITTI photos).

Used for the sim-to-real check: run the CARLA-trained model on real images and
save [RGB | predicted depth] panels so you can see how a sim-trained network
transfers to the real world.

Usage:
    PYTHONPATH=perception_py python scripts/predict_depth.py \
        --root ~/datasets/kitti/dataset --sequence 00 \
        --ckpt outputs/depth/carla00/best.pt --max-vis 8 \
        --out outputs/depth/carla00/predict_kitti
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from carla_perception.depth.dataset import IMAGENET_MEAN, IMAGENET_STD, load_rgb
from carla_perception.depth.model import DepthResNet
from carla_perception.depth.visualize import side_by_side


def main() -> None:
    p = argparse.ArgumentParser(description="Predict depth on label-free images")
    p.add_argument("--root", help="KITTI-layout root (uses sequences/<seq>/image_0)")
    p.add_argument("--sequence", default="00")
    p.add_argument("--images", help="alternatively, a glob like 'path/*.png'")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-vis", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", default="outputs/depth/predict")
    args = p.parse_args()

    device = (
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.images:
        paths = sorted(Path().glob(args.images))
    else:
        paths = sorted((Path(args.root) / "sequences" / args.sequence / "image_0").glob("*.png"))
    paths = paths[:: args.stride][: args.max_vis]
    if not paths:
        raise FileNotFoundError("no images found (check --root/--sequence or --images)")

    ckpt = torch.load(args.ckpt, map_location=device)
    backbone = ckpt.get("args", {}).get("backbone", "resnet18")
    model = DepthResNet(backbone, pretrained=False, max_depth=args.max_depth).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[predict] {len(paths)} images; device={device}")

    mean = IMAGENET_MEAN.reshape(3, 1, 1)
    std = IMAGENET_STD.reshape(3, 1, 1)
    with torch.no_grad():
        for i, path in enumerate(paths):
            rgb = load_rgb(path)
            img = cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            x = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            x = (x - mean) / std
            pred = model(torch.from_numpy(x)[None].to(device))
            panel = side_by_side(img, pred[0, 0].cpu().numpy(), None, max_depth=args.max_depth)
            cv2.imwrite(str(out / f"pred_{i:03d}.png"), panel)

    print(f"[predict] {len(paths)} previews -> {out}")


if __name__ == "__main__":
    main()
