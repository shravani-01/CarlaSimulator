# Building Visual SLAM from Scratch: From Drifting Odometry to a Closed Loop

*Draft — a build log of a from-scratch self-driving perception + SLAM project on KITTI.*

---

## The question I started with

Self-driving perception roles (Tesla Autopilot, Apple AR) keep asking for the same
things: structure-from-motion, visual-inertial SLAM, mapping and localization,
metric 3D. I wanted to actually *build* that stack — not glue together a tutorial,
but implement the geometry myself and prove it on a real benchmark. So I picked
KITTI odometry and set a concrete goal: estimate a car's trajectory from its
cameras, measure how wrong I am, and then make it less wrong.

This is the story of getting from a trajectory that collapses into a scribble to
one that traces a 3.7 km loop and closes it — and the three bugs I had to debug
along the way.

## Step 1 — Monocular visual odometry, and the scale problem

Visual odometry estimates how the camera moved between two frames: find the same
feature points in both images (I used ORB), then recover the rotation and
translation from how those points shifted (the essential matrix).

It worked — locally. Frame-to-frame error (RPE) was ~0.26 m. But over the full
KITTI loop the trajectory **collapsed into a small tangle**, with an absolute
error (ATE) of 185 m. The culprit is fundamental: **a single camera cannot know
scale.** A small nearby motion and a large far motion look identical, so the
trajectory drifts in scale and curls up. The metric drives it home — after
aligning to ground truth, the recovered scale was 0.089 instead of 1.0.

![Monocular VO collapse](images/monocular_vo.png)

That failure is exactly why job descriptions say *visual-**inertial** SLAM*: you
need another sensor (IMU, or a second camera) to pin down scale.

## Step 2 — Stereo fixes scale

KITTI is a stereo dataset: two cameras a known distance apart (the baseline,
~0.54 m). That known distance lets you triangulate **metric depth** for every
feature. With real 3D points in hand, you can estimate motion by PnP — match this
frame's 3D points to where they appear in the next frame and solve for the camera
pose that explains them, *in real metres*.

The change was dramatic:

| | recovered scale | ATE (full loop) |
|---|---|---|
| Monocular | 0.089 (collapsed) | 185 m |
| **Stereo** | **1.0 (metric)** | **26 m (~0.7%)** |

![Stereo VO](images/stereo_vo.png)

The trajectory now traces the actual loop. But it still **drifts** — small errors
accumulate, so by the end of the loop the streets don't quite line up. The loop
doesn't close.

## Step 3 — Loop closure, and three bugs

The fix for drift is loop closure: notice when the car revisits a place it's seen
before, and use that as a constraint to correct the whole trajectory. I built a
**pose graph** — nodes are camera poses, edges are constraints (sequential
odometry, plus loop-closure edges between revisited places) — and optimized all
poses to satisfy every constraint at once.

The first run made things **worse** (ATE 25 m → 29 m). The second run, worse
still. Loop closure that *increases* error means the constraints are wrong, and
chasing that down was the real work:

1. **Coordinate-frame handedness.** I represented poses in the ground plane (SE2).
   KITTI's vertical axis points *down*, which flips the sign of planar rotation
   relative to the standard SE2 convention. My yaw extraction was mirror-flipped,
   so loop constraints rotated translations the wrong way and the optimizer
   diverged. I added a unit test pinning the invariant: converting a 3D pose to
   SE2 and back must reproduce the planar geometry exactly.
2. **Graph consistency.** I'd derived odometry edges from full-3D relative poses
   while the nodes were 2D projections — so the backbone disagreed with itself
   before any loop closure. Fix: derive odometry edges from the node poses
   themselves, so *only* loop closures drive the correction.
3. **Outlier loops + speed.** Wide-baseline loop matches are noisy; a few bad ones
   dominate a plain least-squares solve. A robust loss (soft-L1) plus gating out
   implausible loops handled the outliers. And the optimization was painfully slow
   until I told the solver the Jacobian is **sparse** (each edge touches only two
   nodes) — that took the solve from minutes to seconds.

With those fixed, and a little parameter tuning (trust good loops, reject outliers
hard), loop closure finally did its job:

**ATE 25 m → 9.5 m — a 62% reduction in drift.**

![Loop-closure SLAM](images/slam_loop_closure.png)

The orange (SLAM) path sits almost on top of the black ground truth; the blue
(VO-only) path bows away from it. The loops are closed.

## What I'd do next

This is a planar (SE2) pose graph in Python — great for learning and for this
result, but production SLAM uses full 6-DOF graphs in C++ (g2o / GTSAM / Ceres)
with sparse analytic Jacobians. Natural next steps: a second sensor for tighter
scale (visual-inertial), dense 3D reconstruction (Gaussian Splatting) from the
keyframes, and porting the geometry hot path to C++.

## Takeaways

- **Build the metrics first.** I couldn't have debugged any of this without ATE/RPE
  and trajectory-alignment in place from day one.
- **Test the geometry synthetically.** Projecting known 3D points through known
  poses and checking the math recovers them caught real bugs with zero data.
- **Coordinate frames are where these systems quietly break.** A single flipped
  sign cost me two debugging sessions — and is exactly the kind of thing these
  roles probe for.

*Code: [github.com/shravani-01/CarlaSimulator](https://github.com/shravani-01/CarlaSimulator)*
