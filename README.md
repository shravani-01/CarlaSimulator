# CarlaPerception — Self-Driving Visual Perception Stack

A camera-based self-driving perception system built and validated in the **CARLA** simulator:
**visual odometry / SLAM + 3D object detection & tracking + semantic segmentation + a Gaussian-Splatting 3D reconstruction**, with a C++/CUDA core, a simulation-based validation testbed, and a live web demo.

> **Research question:** Can a modern Gaussian-Splatting 3D map make camera-only ego-localization more
> accurate and more robust to appearance change (rain, night, glare) than a classical feature-based map —
> and what does each cost in latency for onboard use?

---

## Architecture

```
                          ┌──────────────────────────────┐
                          │            CARLA              │
                          │  RGB · depth · semseg · IMU   │
                          │  GPS · 3D boxes · ego pose GT │
                          └───────────────┬──────────────┘
                                          │  carla_io (capture + GT export)
                 ┌────────────────────────┼────────────────────────┐
                 ▼                         ▼                         ▼
      ┌───────────────────┐    ┌────────────────────┐    ┌────────────────────┐
      │  GEOMETRY (C++)   │    │  PERCEPTION (Py)   │    │   NEURAL 3D (Py)   │
      │  VO front-end     │    │  detection (YOLO)  │    │  Gaussian Splatting │
      │  SLAM back-end    │    │  tracking          │    │  reconstruction     │
      │  cam–IMU fusion   │    │  segmentation      │    │  relocalization     │
      │  Ceres/GTSAM BA   │    │  mono depth        │    │                     │
      └─────────┬─────────┘    └─────────┬──────────┘    └──────────┬─────────┘
                └────────────────────────┼──────────────────────────┘
                                         ▼
                  ┌───────────────────────────────────────────────┐
                  │  eval (ATE/RPE · mIoU · mAP · robustness)      │
                  │  testbed (CARLA scenario suite in CI)          │
                  │  serving (ONNX/TensorRT) · frontend (web demo) │
                  └───────────────────────────────────────────────┘
```

## Repository layout

| Path | Purpose |
|---|---|
| `core_cpp/` | C++ geometry + real-time path (VO, SLAM, fusion, calib) + pybind11 bindings |
| `perception_py/` | PyTorch perception: detection, tracking, segmentation, depth |
| `neural3d/` | Gaussian-Splatting reconstruction + relocalization |
| `carla_io/` | CARLA client: sensor capture, scenario scripting, ground-truth export |
| `eval/` | Metrics: ATE/RPE, mIoU, mAP, calibration, robustness slices |
| `serving/` | ONNX/TensorRT export + real-time runner |
| `testbed/` | CARLA scenario suite (weather × time × traffic) + CI hooks |
| `frontend/` | Live web demo (dashboard + 3D viewer) |
| `configs/` | Hydra config tree |
| `pipelines/`, `infra/` | DVC pipelines, Dockerfiles, RunPod scripts |

## Quickstart

```bash
# 1. Install base + dev tooling
make setup
pre-commit install

# 2. (when ready) deep-learning + 3D extras
make setup-ml
make setup-3d

# 3. Install CARLA — see docs/SETUP_CARLA.md, then record a dataset
make record

# 4. Sanity checks
make lint && make test
```

## Status — Phase 0 (foundation)

- [x] Monorepo scaffold, git + DVC initialized
- [x] Build/config/CI skeleton (pyproject, CMake, Hydra, GitHub Actions, pre-commit)
- [x] CARLA setup guide + recording-script stub
- [ ] Phase 1 — perception MVP (detection + tracking + segmentation)

See `../Computer Vision AI/PROJECT_ROADMAP.md` for the full plan and phased milestones.

## Tech stack

C++17 · Python 3.10+ · CUDA · CMake/pybind11 · OpenCV · Eigen · Ceres/GTSAM ·
PyTorch · Ultralytics YOLO · gsplat · Open3D · Rerun · CARLA · ONNX/TensorRT ·
DVC · Weights & Biases · Hydra · Docker · GitHub Actions.

## License

MIT
