# Project Status — CarlaPerception

_A self-driving visual-perception stack: perception (detection/segmentation/tracking) + geometry (VO → stereo VO → loop-closure SLAM), built from scratch and validated on KITTI._

**Last updated:** 2026-06 · **Tests:** 28 passing · **Status:** perception + geometry + neural-3D complete & working

---

## Headline results

| Result | Metric | Notes |
|---|---|---|
| Object detection | bus + 4 pedestrians @ 0.94/0.62 conf | YOLO11, our `Detector` wrapper |
| Semantic segmentation | per-pixel class map | DeepLabV3 wrapper (VOC classes) |
| Multi-object tracking | persistent IDs across frames | our IoU tracker (unit-tested) |
| Monocular VO (KITTI 00) | RPE ≈ 0.26 m; scale **collapses** | shows the monocular scale problem |
| **Stereo VO (KITTI 00)** | **metric scale; ATE ≈ 26 m / ~0.7% over 3.7 km** | PnP + stereo depth |
| **Loop-closure SLAM (KITTI 00)** | **ATE 25 m → 9.5 m (−62%)** | 7 loops, pose-graph optimization |
| Dense 3D reconstruction | colored point cloud (~440K pts) | SGBM stereo + pose fusion |
| **Gaussian Splatting (KITTI 00)** | **641K-Gaussian splat + flythrough, COLMAP-free** | splatfacto on our stereo-VO poses |

**Portfolio one-liners:**
- _"Stereo SLAM on KITTI seq 00 — loop closure + pose-graph optimization cut trajectory drift 62% (ATE 25 m → 9.5 m)."_
- _"Built monocular → stereo VO with metric scale and ATE/RPE evaluation; debugged SE2 coordinate-frame and Jacobian-sparsity issues to make loop closure work and run fast."_
- _"Reconstructed a photorealistic Gaussian-Splatting map of a KITTI street by feeding my own stereo-VO camera poses into nerfstudio — skipping COLMAP entirely."_

---

## What's done ✅

**Foundation / engineering**
- Monorepo scaffold; `pyproject.toml` with grouped extras; Hydra configs; Makefile
- Git + DVC initialized; GitHub repo + CI workflow; pre-commit; ruff/mypy
- `conftest` import bootstrap (robust to conda/pyenv); 28 unit tests; lint clean

**Phase 1 — Perception (Python)**
- `detection/` — YOLO `Detector` wrapper + demo
- `segmentation/` — DeepLabV3 `Segmenter` wrapper + demo
- `tracking/` — `IoUTracker` with persistent IDs + **unit tests** + video demo
- `pipeline.py` — combined `PerceptionPipeline(process → render)` + demo
- `metrics.py` — ATE, RPE, mean-IoU + **unit tests**

**Phase 2 — Geometry / VO**
- `vo/monocular_vo.py` — ORB + essential-matrix relative pose + trajectory; **synthetic-geometry test**
- `vo/stereo_vo.py` — stereo depth + PnP, **metric scale**; **synthetic metric-pose test**
- `datasets/kitti.py` — KITTI odometry loader (stereo + intrinsics + baseline + GT poses)
- `trajectory.py` — Umeyama alignment + `evaluate_trajectory`; **unit tests**
- KITTI run scripts for mono + stereo VO with ATE/RPE + plots

**Phase 3 — SLAM**
- `slam/pose_graph.py` — SE2 pose-graph optimizer, **robust loss + sparse Jacobian**; **loop-closure unit test**
- `slam/stereo_slam.py` — keyframes + loop detection + PnP verification + gating; **SE2-conversion tests**
- `run_slam_kitti.py` — cached keyframes (fast tuning) + tunable CLI knobs + before/after plot

**Bugs debugged the hard way** (good interview material): SE2 yaw handedness, odometry/edge consistency, sparse-Jacobian performance.

---

## What's left ⬜ (roadmap remaining)

**A. Polish & proof (low effort, high credibility)**
- Update `README.md` with the result plots + numbers + architecture diagram
- Run a **second KITTI sequence** (e.g. 05 or 07) to show the pipeline generalizes
- Write the LinkedIn/blog post around the SLAM before/after plot

**B. Live demo (the LinkedIn centerpiece)**
- `frontend/` web app: load a clip → live detection + segmentation + trajectory + 3D view (Rerun/Streamlit). Fast to iterate, big visual payoff.

**C. Dense 3D / neural reconstruction (the "original contribution")** — ✅ done
- ✅ Stereo point-cloud / dense reconstruction from the keyframes
- ✅ Gaussian-Splatting reconstruction (splatfacto on our VO poses, COLMAP-free) on RunPod GPU
- ⬜ remaining stretch: splat-based **relocalization study** (the roadmap's headline research question)

**D. The CARLA spine (the project's namesake)**
- Set up CARLA on RunPod, implement the real `carla_io/record_dataset.py` capture loop
- Re-run perception + VO/SLAM on self-recorded multi-sensor data

**E. Production / MLOps & edge**
- ONNX/TensorRT export + latency benchmarks (the Tesla "onboard" angle)
- CARLA scenario validation testbed in CI; Evidently-style monitoring
- C++ core for the geometry hot path (g2o/GTSAM/Ceres) — the production-optimization story

---

## Recommended next step

**Update the README with results, then build the live web demo (B).** Rationale: you now have genuinely strong, demonstrable results but they only live in terminal output and PNGs. A clickable demo + a polished README turns this from "scripts that print numbers" into a project a recruiter can *see* in 10 seconds — the highest return on effort right now. The CARLA spine (D) and 3D reconstruction (C) are bigger, and best tackled once the demo makes the current work shine.

> Alternative if you'd rather show technical depth first: run **sequence 05/07** (A) to prove generalization — a quick win with the caching already in place.
