# KITTI Odometry Setup Guide

We use the **KITTI odometry** dataset to get a real driving trajectory with
ground-truth poses, so we can score our visual odometry with ATE/RPE.

## What to download

From the official KITTI odometry page (free account required):
https://www.cvlibs.net/datasets/kitti/eval_odometry.php

Grab these three:

| File | Size | What it is |
|---|---|---|
| `data_odometry_calib.zip` | ~1 MB | camera calibration for all sequences |
| `data_odometry_poses.zip` | ~4 MB | ground-truth poses (sequences 00–10) |
| `data_odometry_gray.zip` | ~22 GB | grayscale images (all sequences) |

> The grayscale image zip is large because it bundles every sequence. You only
> need one sequence to get started — once unzipped, you can keep just
> `sequences/00/` and delete the rest. Sequence **00** is the classic loop.

## Where to put it

Unzip so the layout looks like this (anywhere on disk — you'll pass `--root`):

```
kitti_odometry/
├── poses/
│   └── 00.txt
└── sequences/
    └── 00/
        ├── calib.txt
        └── image_0/
            ├── 000000.png
            └── ...
```

Keep it **outside** the repo (it's big and shouldn't be committed — `data/` is
gitignored anyway).

## Run VO on it

```bash
# Start small with --max-frames to sanity-check quickly, then run the full seq.
python scripts/run_vo_kitti.py --root /path/to/kitti_odometry --sequence 00 --max-frames 300
python scripts/run_vo_kitti.py --root /path/to/kitti_odometry --sequence 00
```

This prints ATE / RPE and saves `outputs/demo/vo_kitti_trajectory.png` — your
estimated path overlaid on ground truth.

## Note on monocular results

Pure monocular VO has no absolute scale and drifts over long sequences, so the
plotted path won't perfectly match — that's expected and exactly what motivates
the next steps (loop closure, bundle adjustment, and IMU fusion for scale).
