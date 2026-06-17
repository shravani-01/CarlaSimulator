# Recording a CARLA dataset on RunPod

Generate your **own** self-driving dataset - stereo frames + **perfect**
ground-truth poses - by driving a virtual car in the CARLA simulator, then run the
existing VO/SLAM/splat pipeline on it. CARLA needs an NVIDIA GPU on Linux, so the
*simulator* runs on RunPod; the analysis runs locally afterward.

> **Why bother when KITTI already works?** Two things CARLA adds that KITTI can't:
> (1) **perfect** ground-truth poses (no GPS/INS noise), and (2) **scriptable
> scenarios** - you choose the route, traffic, and weather. A route *with turns*
> directly fixes the Gaussian-Splatting parallax problem we hit on KITTI.

The recorder writes data in **KITTI odometry format**, so every existing script
(`run_stereo_vo_kitti.py`, `run_slam_kitti.py`, `export_nerfstudio.py`) works on
it unchanged - just point `--root` at the recording.

---

## What gets produced

```
data/recordings/carla_town10/
  sequences/00/image_0/000000.png ...   # left camera
  sequences/00/image_1/000000.png ...   # right camera
  sequences/00/calib.txt                # P0,P1 (encodes the 0.54 m baseline)
  sequences/00/times.txt
  poses/00.txt                          # 3x4 ground-truth pose per frame
```

---

## Step 1 - Launch a CARLA pod

- GPU: **RTX 4090 / A40** (CARLA wants ≥8 GB VRAM). ~$0.40-0.50/hr.
- Container image: **`carlasim/carla:0.9.15`** (ships the simulator + Python API).
  - Pin the version - the `carla` *client* package **must match** the server
    version exactly, or the API handshake times out.
- Disk: ~30 GB (the image is large).
- Start command: `sleep infinity` (keeps the container up; we start the server by
  hand over SSH).

Connect via **SSH** (the web terminal is flaky - see `SETUP_GAUSSIAN_SPLATTING.md`).

## Step 2 - Start the CARLA server (headless)

CARLA is a client/server app: the **server** is the simulator (GPU); your script
is the **client**. Start the server off-screen, in the background:

```bash
cd /home/carla
DISPLAY= ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 &
sleep 20   # give it time to boot
```

`-RenderOffScreen` runs without a display; the GPU still renders the cameras.

## Step 3 - Get the recorder onto the pod + install client deps

Bring the repo over (git clone, or the HTTP-proxy upload from the splat doc), then:

```bash
pip install carla==0.9.15 numpy opencv-python-headless hydra-core omegaconf
cd /path/to/CarlaPerception
export PYTHONPATH="$PWD:$PWD/perception_py"
```

Use `opencv-python-headless` on the server (no GUI libs needed there).

## Step 4 - Record

```bash
python -m carla_io.record_dataset \
    data.town=Town10HD_Opt \
    data.capture.num_frames=1000 \
    data.weather=ClearNoon
```

It loads the town, spawns a Tesla Model 3 with a stereo rig + background traffic,
drives on autopilot in **synchronous mode**, and saves a stereo pair + the exact
left-camera pose every tick. ~1000 frames at 20 fps ≈ a 50 s drive.

> **Want turns** (better splat coverage)? Autopilot turns at intersections, so a
> longer capture naturally includes them. For a guaranteed loop, re-record and keep
> the run with the most turning, or raise `num_frames`.

## Step 5 - Pull the data to your Mac

Zip and download through the RunPod HTTP proxy (campus firewalls block
`runpodctl`'s relay - same workaround as the splat doc):

```bash
# on the pod
cd /path/to/CarlaPerception && python3 -m zipfile -c carla00.zip data/recordings/carla_town10
python3 -m http.server 8888        # open Connect -> HTTP Service :8888 and download
```

Unzip into your project locally at `data/recordings/carla_town10/`, then
**terminate the pod**.

## Step 6 - Run the existing pipeline on your CARLA data (local, no GPU)

```bash
ROOT=data/recordings/carla_town10
export PYTHONPATH="$PWD/perception_py"

# Stereo VO vs CARLA's PERFECT ground truth (ATE should be very low).
.venv/bin/python scripts/run_stereo_vo_kitti.py --root $ROOT --sequence 00

# Loop-closure SLAM
.venv/bin/python scripts/run_slam_kitti.py --root $ROOT --sequence 00 --stride 20

# Gaussian-Splatting export (then train on RunPod per SETUP_GAUSSIAN_SPLATTING.md)
.venv/bin/python scripts/export_nerfstudio.py --root $ROOT --sequence 00 \
    --max-frames 200 --stride 2 --out outputs/nerf/carla00
```

Because CARLA's ground truth is exact, this is also a clean way to **validate the
VO/SLAM accuracy itself** - any error is the algorithm's, not the sensor's.

## Step 7 - Version the data with DVC (don't commit raw frames to git)

```bash
dvc add data/recordings/carla_town10
git add data/recordings/carla_town10.dvc .gitignore
git commit -m "Track CARLA recording with DVC"
# optional remote: dvc remote add -d storage <url> && dvc push
```

---

## Troubleshooting

- **`RuntimeError: time-out ... version mismatch`** → the `carla` pip version
  doesn't match the server. Install the exact same version (`carla==0.9.15`).
- **Server exits / black frames** → missing `-RenderOffScreen`, or too little
  VRAM. Use a bigger GPU or a lighter town (`Town01`, `Town02`).
- **Frames/poses look mirrored** → coordinate convention; `carla_io/coords.py` +
  `tests/test_carla_coords.py` exist to prevent exactly this. Run the tests if you
  touch that code.
- **Capture hangs** → a sensor frame was dropped; we match frames by id, so a stall
  usually means the server fell out of sync - restart the server.
- **Low FPS** → reduce camera resolution and `num_vehicles` in the data config.
