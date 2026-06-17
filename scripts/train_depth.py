"""Train the monocular-depth network on a CARLA recording (with GT depth).

Runs on GPU (RunPod) for real training; falls back to CPU/MPS for a quick smoke
test. Saves the best checkpoint (by val AbsRel) and prints a per-epoch scorecard.

Usage:
    PYTHONPATH=perception_py python scripts/train_depth.py \
        --root data/recordings/carla_town10 --sequence 00 \
        --epochs 20 --batch-size 8 --out outputs/depth/carla00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from carla_perception.depth.dataset import make_dataset
from carla_perception.depth.losses import depth_loss, depth_metrics
from carla_perception.depth.model import DepthResNet
from torch.utils.data import DataLoader


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    tot = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            pred = model(img)
            m = depth_metrics(pred, depth, mask)
            for k in tot:
                tot[k] += m[k]
            n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Train monocular depth on CARLA depth labels")
    p.add_argument("--root", required=True)
    p.add_argument("--sequence", default="00")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="outputs/depth/carla00")
    args = p.parse_args()

    device = pick_device(args.device)
    size = (args.height, args.width)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} size={size} backbone={args.backbone}")

    train_ds = make_dataset(args.root, args.sequence, "train", size, max_depth=args.max_depth)
    val_ds = make_dataset(args.root, args.sequence, "val", size, max_depth=args.max_depth)
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val frames")
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = DepthResNet(args.backbone, pretrained=True, max_depth=args.max_depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_dl:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad()
            loss = depth_loss(model(img), depth, mask)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        train_loss = running / max(len(train_dl), 1)
        metrics = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(f"[train] epoch {epoch:02d}  loss={train_loss:.4f}  "
              f"AbsRel={metrics['abs_rel']:.4f}  RMSE={metrics['rmse']:.3f}  "
              f"d1={metrics['delta1']:.3f}")

        torch.save({"model": model.state_dict(), "args": vars(args)}, out / "last.pt")
        if metrics["abs_rel"] < best:
            best = metrics["abs_rel"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "best.pt")
            print(f"[train]   new best AbsRel={best:.4f} -> {out/'best.pt'}")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] done. best val AbsRel={best:.4f}. history -> {out/'history.json'}")


if __name__ == "__main__":
    main()
