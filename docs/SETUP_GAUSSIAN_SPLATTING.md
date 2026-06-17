# Gaussian Splatting on RunPod

Turn your KITTI sequence + **your own VO poses** into a photorealistic 3D
Gaussian-Splatting model and a flythrough video. Training needs a CUDA GPU, so
this runs on RunPod; everything before it runs locally.

> The novelty: instead of COLMAP estimating camera poses (the usual step), we
> feed nerfstudio the poses from *our* stereo visual odometry.

> **Result of an actual run** (KITTI seq 00, 100 frames, RTX-class GPU):
> splatfacto trained 30k steps in ~10-15 min and exported a **641K-Gaussian**
> `.ply` plus a 90-frame flythrough (`docs/images/gaussian_splat.gif`). Total pod
> cost ≈ $0.30-0.50. The reconstruction is sharp along the driving axis and
> streaks at the edges (forward-only parallax) - see the README writeup.

---

## Step 0 - Export the dataset (local, no GPU)

```bash
export PYTHONPATH="$PWD/perception_py"
.venv/bin/python scripts/export_nerfstudio.py \
    --root ~/datasets/kitti/dataset --sequence 00 \
    --max-frames 200 --stride 2 --out outputs/nerf/seq00
```

This writes `outputs/nerf/seq00/` with `images/` + `transforms.json`. It's small
(~a couple hundred images), so it uploads quickly.

> Tip: a segment with some turning gives splatting better multi-view coverage
> than pure straight-line driving. Try a few start ranges.

---

## Step 1 - Launch a RunPod GPU pod

- GPU: **RTX 4090** or **A40** is plenty (~$0.34-0.45/hr).
- Template: the **nerfstudio** community template, or a PyTorch+CUDA image.
  (The official nerfstudio Docker image `ghcr.io/nerfstudio-project/nerfstudio`
  has everything preinstalled - easiest.)
- Expose a terminal / Jupyter / SSH.

Budget: training is ~10-15 min; total pod time ~30-45 min ⇒ roughly **$0.20-0.35**.

## Step 2 - Get the dataset onto the pod

Use RunPod's file upload, or `runpodctl`, or `scp`. Put it at e.g.
`/workspace/seq00/` (so `/workspace/seq00/transforms.json` exists).

> **Gotcha (transfers):** `runpodctl send/receive` uses the `croc` relay on
> **TCP port 9009**, which many campus/corporate firewalls block (`i/o timeout`
> connecting to `relay*.runpod.net`). If that happens, transfer over HTTPS
> instead: on the pod run `cd /workspace && python3 -m http.server 8888`, then
> open the pod's **Connect → HTTP Service :8888** proxy URL in your browser and
> download/upload through it - that traffic rides port 443 and clears most
> firewalls. No `unzip` on the pod? Use `python3 -m zipfile -e file.zip dest/`.

## Step 3 - Install nerfstudio (skip if using the nerfstudio image)

```bash
pip install nerfstudio
# (the Docker image already has this + CUDA + tinycudann)
```

## Step 4 - Train Gaussian Splatting (splatfacto)

```bash
ns-train splatfacto --data /workspace/seq00 nerfstudio-data
```

nerfstudio reads `transforms.json`, trains, and prints a config path like
`outputs/seq00/splatfacto/<timestamp>/config.yml`. Watch the viewer URL it
prints if you want a live 3D view.

## Step 5 - Render a flythrough video

Interpolate a camera path through the training views (your driving trajectory):

```bash
ns-render interpolate \
    --load-config outputs/.../config.yml \
    --output-path /workspace/flythrough.mp4
```

## Step 6 - Export the splat + download

```bash
ns-export gaussian-splat --load-config outputs/.../config.yml --output-dir /workspace/splat
```

Download `flythrough.mp4` and `splat/*.ply`. The `.ply` opens in browser splat
viewers (e.g. PlayCanvas SuperSplat, antimatter15/splat). Drop the flythrough in
the README/LinkedIn post.

---

## Troubleshooting

- **Scene looks "inside-out" / mirrored** → pose convention. Re-export with
  `--convention opencv` and retrain.
- **Blurry / streaky** → forward-only driving gives weak parallax; use a segment
  with turns, more frames, or a shorter span.
- **OOM** → fewer images (`--max-frames`), or a bigger GPU.

## Optional: the research angle

The roadmap's original question - *does a Gaussian-Splatting map relocalize the
camera better than a classical feature map?* - can be tested by rendering novel
views from the splat and matching a held-out frame against them. That's a strong
"finding" for the blog if you want to take it further.
