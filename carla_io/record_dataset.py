"""Record a CARLA driving dataset with a synchronized stereo rig + ground truth.

Output is written in **KITTI odometry format** (see ``kitti_writer``), so the
recording is consumable by our existing loader, stereo VO, SLAM, and the splat
exporter with no changes.

How it works (the CARLA client/server split):
  * CARLA runs as a *server* (the simulator, needs a GPU). This script is the
    *client*: it connects over the Python API, spawns a car ("ego"), bolts a
    left+right camera onto it, and drives.
  * We put the world in **synchronous mode** so each ``world.tick()`` produces
    exactly one frame per sensor with matching timestamps — essential for stereo.
  * Every tick we save the stereo pair and the car's *exact* pose (the reason to
    use a simulator: perfect ground truth, plus we can script routes with turns).

Run on the CARLA host (server already listening on :2000):
    python -m carla_io.record_dataset                  # uses configs/data/carla_town10.yaml
    python -m carla_io.record_dataset data.capture.num_frames=600

See docs/SETUP_CARLA.md for the full RunPod runbook.
"""

from __future__ import annotations

import contextlib
import queue
import sys
from pathlib import Path

import numpy as np

from carla_io.coords import fov_to_intrinsics, ue_matrix_to_opencv_c2w
from carla_io.kitti_writer import KittiSequenceWriter

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    _HAS_HYDRA = True
except Exception:  # pragma: no cover - hydra optional at import time
    _HAS_HYDRA = False


def _carla_image_to_bgr(image) -> np.ndarray:
    """Convert a CARLA RGB image to an OpenCV BGR uint8 array (drop alpha)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))  # BGRA
    return arr[:, :, :3].copy()


def _spawn_stereo_cameras(world, blueprint_lib, ego, cfg, baseline: float):
    """Attach a left + right RGB camera to the ego vehicle, ``baseline`` apart.

    In the camera's local (Unreal) frame +Y is *right*, so the right camera is the
    left camera shifted by ``+baseline`` along Y. Both look forward (+X).
    """
    import carla

    rgb = cfg.data.sensors.rgb
    cam_bp = blueprint_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(rgb.width))
    cam_bp.set_attribute("image_size_y", str(rgb.height))
    cam_bp.set_attribute("fov", str(rgb.fov))

    # Mount roughly at windshield height, looking ahead.
    left_tf = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.6))
    right_tf = carla.Transform(carla.Location(x=1.5, y=baseline, z=1.6))
    left = world.spawn_actor(cam_bp, left_tf, attach_to=ego)
    right = world.spawn_actor(cam_bp, right_tf, attach_to=ego)
    return left, right


def _spawn_traffic(world, blueprint_lib, tm, cfg) -> list:
    """Populate the town with some moving vehicles so scenes aren't empty."""
    actors = []
    spawn_points = world.get_map().get_spawn_points()
    n = min(int(cfg.data.capture.get("num_vehicles", 0)), max(0, len(spawn_points) - 1))
    vehicle_bps = blueprint_lib.filter("vehicle.*")
    for sp in spawn_points[1 : 1 + n]:
        bp = np.random.choice(vehicle_bps)
        v = world.try_spawn_actor(bp, sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)
    return actors


def record(cfg: DictConfig) -> None:
    """Connect to CARLA, spawn an ego rig, drive, and dump frames + GT poses."""
    out_dir = Path(cfg.data.recordings_dir) / cfg.data.name
    seq = str(cfg.data.get("sequence", "00"))
    fps = int(cfg.data.capture.fps)
    num_frames = int(cfg.data.capture.num_frames)
    warmup = int(cfg.data.capture.get("warmup_frames", 30))
    baseline = float(cfg.data.sensors.rgb.get("baseline", 0.54))
    print(f"[record] -> {out_dir} (KITTI seq {seq}); town={cfg.data.town} "
          f"weather={cfg.data.weather} frames={num_frames} fps={fps} baseline={baseline}m")

    try:
        import carla
    except Exception:
        print("[record] `carla` not installed — dry run only. "
              "See docs/SETUP_CARLA.md.", file=sys.stderr)
        return

    client = carla.Client(cfg.data.get("host", "localhost"), int(cfg.data.get("port", 2000)))
    client.set_timeout(30.0)
    world = client.load_world(cfg.data.town)

    # Weather (used by the robustness study).
    try:
        world.set_weather(getattr(carla.WeatherParameters, str(cfg.data.weather)))
    except AttributeError:
        print(f"[record] unknown weather '{cfg.data.weather}', keeping default", file=sys.stderr)

    original_settings = world.get_settings()
    tm = client.get_trafficmanager()
    actors: list = []
    try:
        # --- synchronous mode: deterministic, timestamp-aligned sensor capture ---
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        bp = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        ego_bp = bp.find("vehicle.tesla.model3")
        ego = world.spawn_actor(ego_bp, spawn_points[0])
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)
        actors += _spawn_traffic(world, bp, tm, cfg)

        left, right = _spawn_stereo_cameras(world, bp, ego, cfg, baseline)
        actors += [left, right]

        # Sensor callbacks push into queues; we pair them by frame id each tick.
        lq: queue.Queue = queue.Queue()
        rq: queue.Queue = queue.Queue()
        left.listen(lq.put)
        right.listen(rq.put)

        K = fov_to_intrinsics(
            int(cfg.data.sensors.rgb.width),
            int(cfg.data.sensors.rgb.height),
            float(cfg.data.sensors.rgb.fov),
        )
        writer = KittiSequenceWriter(out_dir, sequence=seq)
        writer.write_calib(K, baseline)

        def _grab(q, frame_id):
            """Pull from a sensor queue until we get the frame matching this tick."""
            while True:
                img = q.get(timeout=10.0)
                if img.frame == frame_id:
                    return img

        # Warm up so traffic settles and the car starts moving before we record.
        for _ in range(warmup):
            world.tick()
            lq.get(timeout=10.0)
            rq.get(timeout=10.0)

        for i in range(num_frames):
            snap = world.tick()
            limg = _grab(lq, snap)
            rimg = _grab(rq, snap)
            # Ground-truth pose = the LEFT camera's world transform, converted to
            # our OpenCV/KITTI convention.
            c2w = ue_matrix_to_opencv_c2w(np.array(left.get_transform().get_matrix()))
            writer.add_frame(
                _carla_image_to_bgr(limg),
                _carla_image_to_bgr(rimg),
                c2w,
                timestamp=snap * (1.0 / fps),
            )
            if i % 100 == 0:
                print(f"[record]   frame {i}/{num_frames}")

        writer.finalize()
        print(f"[record] done: {len(writer)} frames -> {out_dir}")
        print(f"[record] run VO/SLAM with --root {out_dir} --sequence {seq}")
    finally:
        # Always restore async mode and clean up actors, even on error/Ctrl-C.
        with contextlib.suppress(Exception):
            tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)
        for a in actors:
            with contextlib.suppress(Exception):
                a.destroy()
        print("[record] cleaned up actors and restored async mode")


if _HAS_HYDRA:

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        print(OmegaConf.to_yaml(cfg.data))
        record(cfg)

else:  # pragma: no cover

    def main() -> None:
        print("Install dev deps first: `make setup` (provides hydra-core).")


if __name__ == "__main__":
    main()
