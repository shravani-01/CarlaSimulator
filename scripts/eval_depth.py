"""Evaluate a trained depth model + save side-by-side preview images.

Computes the standard depth metrics (AbsRel, RMSE, delta<1.25) on a recording's
validation split, and writes a few [RGB | predicted depth | GT depth] panels.

Usage:
    PYTHONPATH=perception_py python scripts/eval_depth.py \
        --root data/recordings/carla_town10 --sequence 00 \
        --ckpt outputs/depth/carla00/best.pt --out outputs/depth/carla00/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from carla_perception.depth.dataset import IMAGENET_MEAN, IMAGENET_STD, make_dataset
from carla_perception.depth.losses import depth_metrics
from carla_perception.depth.model import DepthResNet
from carla_perception.depth.visualize import side_by_side


def _denormalize(img_chw: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation -> HxWx3 uint8 RGB."""
    x = img_chw.cpu().numpy().transpose(1, 2, 0)
    x = (x * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a monocular-depth checkpoint")
    p.add_argument("--root", required=True)
    p.add_argument("--sequence", default="00")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-vis", type=int, default=6)
    p.add_argument("--out", default="outputs/depth/eval")
    args = p.parse_args()

    device = (
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = (args.height, args.width)

    ckpt = torch.load(args.ckpt, map_location=device)
    backbone = ckpt.get("args", {}).get("backbone", "resnet18")
    model = DepthResNet(backbone, pretrained=False, max_depth=args.max_depth).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = make_dataset(args.root, args.sequence, args.split, size, max_depth=args.max_depth)
    print(f"[eval] {len(ds)} {args.split} frames; device={device}")

    tot = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}
    saved = 0
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            img = s["image"][None].to(device)
            depth = s["depth"][None].to(device)
            mask = s["mask"][None].to(device)
            pred = model(img)
            m = depth_metrics(pred, depth, mask)
            for k in tot:
                tot[k] += m[k]
            if saved < args.max_vis:
                rgb = _denormalize(s["image"])
                panel = side_by_side(
                    rgb, pred[0, 0].cpu().numpy(), depth[0, 0].cpu().numpy(),
                    max_depth=args.max_depth,
                )
                cv2.imwrite(str(out / f"depth_{i:03d}.png"), panel)
                saved += 1

    n = max(len(ds), 1)
    summary = {k: v / n for k, v in tot.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] AbsRel={summary['abs_rel']:.4f}  RMSE={summary['rmse']:.3f}  "
          f"delta1={summary['delta1']:.3f}")
    print(f"[eval] {saved} previews + metrics.json -> {out}")


if __name__ == "__main__":
    main()
