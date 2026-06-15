"""Record a CARLA dataset with synchronized sensors + ground truth.

Phase 0: this is a runnable *stub*. It validates config loading and lays out the
capture loop; the CARLA-specific calls are guarded so the file imports and the
`--help`/dry-run path works even without `carla` installed. We flesh out the
actual sensor callbacks in Phase 1.

Run (with a CARLA server listening on :2000):
    python -m carla_io.record_dataset
    python -m carla_io.record_dataset data=carla_town10
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    _HAS_HYDRA = True
except Exception:  # pragma: no cover - hydra optional at import time
    _HAS_HYDRA = False


def record(cfg: DictConfig) -> None:
    """Connect to CARLA, spawn an ego rig, drive, and dump frames + GT poses."""
    out_dir = Path(cfg.data.recordings_dir) / cfg.data.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[record] output dir: {out_dir}")
    print(f"[record] town={cfg.data.town} weather={cfg.data.weather} "
          f"frames={cfg.data.capture.num_frames} fps={cfg.data.capture.fps}")

    try:
        import carla  # noqa: F401
    except Exception:
        print("[record] `carla` not installed — dry run only. "
              "See docs/SETUP_CARLA.md.", file=sys.stderr)
        return

    # --- Fleshed out in Phase 1 ---
    # client = carla.Client("localhost", 2000); client.set_timeout(10.0)
    # world = client.load_world(cfg.data.town)
    # ... set synchronous mode, spawn ego vehicle, attach RGB/depth/semseg/IMU/GPS
    # ... tick the world, save frames + ego transform (ground-truth pose) per step
    print("[record] CARLA found — capture loop is implemented in Phase 1.")


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
