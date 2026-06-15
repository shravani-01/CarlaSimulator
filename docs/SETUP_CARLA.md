# CARLA Setup Guide

CARLA is the free driving simulator we use for data + validation. This guide gets you from zero to a recorded, ground-truth-labeled dataset.

## 1. System requirements

- **GPU:** NVIDIA with ≥6 GB VRAM (8 GB+ recommended). CARLA is GPU-heavy.
- **OS:** Linux or Windows. (On macOS, run CARLA on a RunPod/cloud GPU box and connect remotely.)
- **Disk:** ~20 GB for CARLA + room for recordings.

> On your Mac: run CARLA headless on a RunPod GPU instance and stream sensor data, or just record datasets there and pull them down via DVC.

## 2. Install CARLA

Pick a packaged release (recommended — no build needed):

```bash
# Example: download a packaged CARLA release (check carla.org for the current version)
mkdir -p ~/carla && cd ~/carla
# Download CARLA_X.Y.Z.tar.gz from https://github.com/carla-simulator/carla/releases
tar -xzf CARLA_*.tar.gz
```

Install the matching Python API into this project's environment:

```bash
# The version MUST match the simulator release you downloaded
pip install carla
```

## 3. Launch the simulator

```bash
cd ~/carla
./CarlaUE4.sh                 # add -RenderOffScreen for headless servers
# server now listening on localhost:2000
```

## 4. Record a dataset

With the simulator running:

```bash
make record
# or:
python -m carla_io.record_dataset data=carla_town10
```

This spawns an ego vehicle with an RGB camera, depth, semantic segmentation, IMU, and GPS,
drives it on autopilot, and writes synchronized frames + ground-truth poses to
`data/recordings/`. Tune frame count, weather, and traffic in `configs/data/carla_town10.yaml`.

## 5. Version the data with DVC (don't commit raw frames to git)

```bash
dvc add data/recordings
git add data/recordings.dvc .gitignore
git commit -m "Track CARLA recording with DVC"
# optional remote (S3/GCS/local):
# dvc remote add -d storage <url> && dvc push
```

## Troubleshooting

- **`carla` import version mismatch** → the pip `carla` version must equal the simulator release.
- **Black frames / no render on a server** → launch with `-RenderOffScreen`.
- **Low FPS** → reduce camera resolution and `num_vehicles` in the data config.
