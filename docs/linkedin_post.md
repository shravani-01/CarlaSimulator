# LinkedIn post

## The post (copy-paste this)

Detecting objects in an image is the easy part now - point YOLO at it, or just ask a vision-language model. I'd done plenty of that.

But it left me with a nagging question: I could tell *what* was in a frame, yet I had no idea how a self-driving car knows *where it is* in the world. So I stopped reaching for the easy tools and spent a few weeks building the geometry myself, from scratch.

The result: a self-driving perception + SLAM stack, tested on the KITTI benchmark.

The journey, in one line each:
→ Monocular visual odometry - works, but drifts and can't recover real-world scale
→ Stereo - fixes scale using the two-camera baseline (metric trajectory)
→ Loop-closure SLAM - recognizes revisited places and cut trajectory drift 62% (25 m → 9.5 m over a 3.7 km loop)
→ Gaussian Splatting - reconstructed the street in photorealistic 3D, feeding in my *own* camera poses and skipping the usual COLMAP step
→ CARLA - generated my own driving data in simulation and validated the whole pipeline against perfect ground truth (0.59 m error)

The honest part: there's plenty I'd still improve - it's a planar pose graph in Python, not production 6-DOF C++. But I understand every line, and debugging a flipped coordinate frame at 1am taught me more than any course could.

Biggest lesson: build your metrics first. I couldn't have debugged any of this without ATE/RPE in place from day one.

Code + writeup 👉 [your repo link]

#ComputerVision #SLAM #SelfDriving #Robotics #MachineLearning

---

## Visuals to attach (in this order)

1. `docs/images/slam_loop_closure.png` - the SLAM before/after (your headline result)
2. `docs/images/gaussian_splat.gif` - the 3D flythrough (the eye-catcher)
3. `docs/images/carla_stereo_vo.png` - VO vs perfect CARLA ground truth

LinkedIn shows up to ~3 images well; the GIF will autoplay and is your scroll-stopper.

## Tips
- Put the repo link in the FIRST comment, not the post body - LinkedIn suppresses
  reach on posts with external links. Mention "link in comments" in the post.
- Post Tue-Thu morning for best reach.
- Reply to every early comment in the first hour - it boosts distribution.
