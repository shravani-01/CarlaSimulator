# Training the monocular-depth network (RunPod GPU)

Train a CNN to predict metric depth from a single camera, supervised by CARLA's
perfect ground-truth depth, then study how well it transfers to *real* KITTI
images (the sim-to-real gap). Everything is written + unit-tested already; this
doc is just the GPU run.

## Prerequisites - two data recordings

1. **CARLA with depth** (the training data). Re-run the CARLA recorder
   (`docs/SETUP_CARLA.md`) - depth capture is on by default now
   (`sensors.depth: true`), so the recording will include a `depth/` folder.
   A few thousand frames across a couple of weathers is ideal:
   ```bash
   python -m carla_io.record_dataset data.capture.num_frames=3000 data.weather=ClearNoon
   ```
2. **KITTI** (the sim-to-real test set) - you already have it locally. KITTI depth
   labels are optional; if absent we evaluate qualitatively (depth previews).

## Step 1 - Launch a PyTorch GPU pod

- Image: a RunPod **PyTorch 2.x / CUDA** template (torch + torchvision preinstalled).
- GPU: RTX 4090 / A40 is plenty; depth training on ~3k frames is ~15-30 min.
- Expose HTTP port **8888** (for the download trick) and SSH.

## Step 2 - Get the code + data onto the pod

```bash
cd /workspace
git clone https://github.com/shravani-01/CarlaSimulator.git && cd CarlaSimulator
pip install opencv-python-headless numpy
export PYTHONPATH="$PWD:$PWD/perception_py"
```
Upload your CARLA-with-depth recording (HTTP-proxy upload, or `git`/`scp`) to
`data/recordings/carla_town10/`.

## Step 3 - Train

```bash
python scripts/train_depth.py \
    --root data/recordings/carla_town10 --sequence 00 \
    --epochs 25 --batch-size 12 --lr 1e-4 \
    --height 192 --width 640 --device cuda \
    --out outputs/depth/carla00
```
Watch `AbsRel` fall and `delta1` (fraction of pixels within 25%) rise each epoch.
The best checkpoint (lowest val AbsRel) is saved to `outputs/depth/carla00/best.pt`.

## Step 4 - Evaluate + previews (on CARLA val)

```bash
python scripts/eval_depth.py \
    --root data/recordings/carla_town10 --sequence 00 \
    --ckpt outputs/depth/carla00/best.pt \
    --height 192 --width 640 --device cuda \
    --out outputs/depth/carla00/eval_carla
```
Gives `metrics.json` (AbsRel/RMSE/delta1) + `[RGB | predicted | GT]` panels.

## Step 5 - Sim-to-real: test on KITTI

Run the same model on KITTI images (no retraining). KITTI has no `depth/` folder,
so this is a **qualitative** transfer check - the previews show how a sim-trained
net does on real photos:
```bash
python scripts/eval_depth.py \
    --root ~/datasets/kitti/dataset --sequence 00 \
    --ckpt outputs/depth/carla00/best.pt \
    --height 192 --width 640 --device cuda \
    --out outputs/depth/carla00/eval_kitti_qualitative
```
> If you have KITTI depth maps in our `depth/` format you'll also get numbers here,
> and the CARLA-vs-KITTI AbsRel difference *is* the sim-to-real gap - the headline
> finding. Otherwise, report it qualitatively from the previews.

## Step 6 - (optional) Close the gap by fine-tuning

A few epochs of fine-tuning on a little real KITTI data usually shrinks the gap.
Resume from the CARLA checkpoint and train at a low LR on KITTI, then re-evaluate.

## Step 7 - Download artifacts + terminate

Pull `best.pt`, `history.json`, and the `eval_*` preview folders via the HTTP
proxy (`python3 -m http.server 8888`), then **terminate the pod**.

The previews + metrics feed straight into the README/LinkedIn post.
