"""CarlaPerception - interactive web demo (Streamlit).

Two tabs:
  1. Live perception  - run detection + segmentation on a KITTI driving frame,
     a built-in sample, or an uploaded image, and see the annotated result.
  2. SLAM results     - view the monocular vs. stereo vs. loop-closure trajectory
     plots and the headline metrics.

Run from the project root:
    pip install -e ".[ml,frontend]"
    streamlit run frontend/app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from carla_perception.datasets.kitti import KITTIOdometry
from carla_perception.pipeline import PerceptionPipeline
from carla_perception.reconstruction.stereo_pointcloud import disparity_sgbm
from carla_perception.safety.collision import forward_collision_check
from carla_perception.tracking.tracker import IoUTracker, Track

st.set_page_config(page_title="CarlaPerception", layout="wide")

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = Path("data/sample/bus.jpg")
IMAGES_DIR = Path("docs/images")
OUTPUTS_DIR = Path("outputs/demo")


@st.cache_resource(show_spinner="Loading perception models...")
def get_pipeline(device: str | None) -> PerceptionPipeline:
    return PerceptionPipeline(device=device)


@st.cache_resource(show_spinner="Indexing KITTI sequence...")
def get_kitti(root: str, seq: str) -> KITTIOdometry:
    return KITTIOdometry(root, seq)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"id:{t.track_id} {t.label}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def draw_collision(frame: np.ndarray, checks) -> np.ndarray:
    """Draw tracked boxes with distance; red box + BRAKE banner for hazards."""
    out = frame.copy()
    hazards = []
    for r in checks:
        t = r.obj
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        color = (0, 0, 255) if r.hazard else (0, 255, 0)
        label = f"id:{t.track_id} {t.label}"
        if r.distance is not None:
            label += f" {r.distance:.0f}m"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if r.hazard else 2)
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        if r.hazard and r.distance is not None:
            hazards.append(r.distance)
    if hazards:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 46), (0, 0, 255), -1)
        cv2.putText(out, f"!! BRAKE  -  obstacle {min(hazards):.0f} m ahead",
                    (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    h, w = frames[0].shape[:2]
    writer = None
    for codec in ("avc1", "mp4v"):  # prefer browser-friendly H.264, fall back
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            break
    for f in frames:
        writer.write(f)
    writer.release()


def sample_image() -> np.ndarray | None:
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
        except Exception:
            return None
    return cv2.imread(str(SAMPLE_PATH))


# ---------------------------------------------------------------- sidebar
st.sidebar.title("CarlaPerception")
st.sidebar.caption("Self-driving visual perception & SLAM")
device = st.sidebar.selectbox("Compute device", ["auto", "cpu", "mps", "cuda"])
device = None if device == "auto" else device

tab_perc, tab_track, tab_slam = st.tabs(
    ["🔍 Live perception", "🎥 Tracking video", "🛰️ SLAM results"]
)

# ---------------------------------------------------------------- perception
with tab_perc:
    st.subheader("Detection + semantic segmentation on one frame")
    source = st.radio("Image source", ["KITTI frame", "Sample image", "Upload"], horizontal=True)

    image: np.ndarray | None = None
    if source == "KITTI frame":
        root = st.text_input("KITTI dataset root", "~/datasets/kitti/dataset")
        seq = st.text_input("Sequence", "00")
        try:
            data = get_kitti(str(Path(root).expanduser()), seq)
            idx = st.slider("Frame", 0, len(data) - 1, 0)
            image = cv2.imread(str(data.image_paths[idx]))
        except Exception as e:
            st.warning(f"Couldn't load KITTI ({e}). See docs/SETUP_KITTI.md, or use another source.")
    elif source == "Sample image":
        image = sample_image()
    else:
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if up is not None:
            arr = np.frombuffer(up.read(), np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        st.info("Pick an image source above.")
    elif st.button("Run perception", type="primary"):
        pipe = get_pipeline(device)
        with st.spinner("Running detection + segmentation..."):
            result = pipe.process(image)
            vis = pipe.render(image, result)
        c1, c2 = st.columns(2)
        c1.image(to_rgb(image), caption="input", use_column_width=True)
        c2.image(to_rgb(vis), caption="detection + segmentation", use_column_width=True)
        st.success(pipe.summarize(result))
        if result.detections:
            st.dataframe(
                [{"label": d.label, "confidence": round(d.confidence, 2)} for d in result.detections],
                use_container_width=True,
            )

# ---------------------------------------------------------------- tracking video
with tab_track:
    st.subheader("Detection + tracking over a clip -> annotated video")
    st.caption("Each object keeps a stable `id:` as it moves - that's tracking.")
    tsrc = st.radio("Source", ["KITTI clip", "Upload video"], horizontal=True, key="tsrc")
    seg_overlay = st.checkbox("Also overlay segmentation (slower)", value=False)
    fcw = st.checkbox("⚠️ Forward-collision warning (KITTI stereo)", value=False,
                      help="Measures distance to objects ahead and flags ones that are too close.")
    warn_dist = st.slider("Warn distance (m)", 5, 25, 12) if fcw else 12.0

    frames_in: list[np.ndarray] = []
    right_in: list[np.ndarray] = []
    Kmat = baseline = None
    if tsrc == "KITTI clip":
        troot = st.text_input("KITTI dataset root", "~/datasets/kitti/dataset", key="troot")
        tseq = st.text_input("Sequence", "00", key="tseq")
        try:
            tdata = get_kitti(str(Path(troot).expanduser()), tseq)
            start = st.slider("Start frame", 0, max(0, len(tdata) - 2), 0)
            count = st.slider("Number of frames", 20, 150, 80)
            if st.button("Generate tracking video", type="primary"):
                left_paths = tdata.image_paths[start : start + count]
                frames_in = [cv2.imread(str(p)) for p in left_paths]
                if fcw:  # need the right camera for stereo distance
                    right_paths = tdata.image_paths_right[start : start + count]
                    right_in = [cv2.imread(str(p)) for p in right_paths]
                    Kmat, baseline = tdata.K, tdata.baseline
        except Exception as e:
            st.warning(f"Couldn't load KITTI ({e}). See docs/SETUP_KITTI.md, or upload a video.")
    else:
        if fcw:
            st.info("Collision warning needs stereo - available for KITTI clips only.")
        up = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi"], key="tup")
        if up is not None and st.button("Generate tracking video", type="primary"):
            tmp = Path("outputs/demo/_upload.mp4")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(up.read())
            cap = cv2.VideoCapture(str(tmp))
            while len(frames_in) < 200:
                ok, fr = cap.read()
                if not ok:
                    break
                frames_in.append(fr)
            cap.release()

    if frames_in:
        pipe = get_pipeline(device)
        tracker = IoUTracker()
        use_fcw = fcw and len(right_in) == len(frames_in) and Kmat is not None
        annotated = []
        bar = st.progress(0.0, text="Running detection + tracking...")
        for i, fr in enumerate(frames_in):
            tracks = tracker.update(pipe.detector.detect(fr))
            base = fr
            if seg_overlay:
                base = pipe.segmenter.overlay(fr, pipe.segmenter.segment(fr), alpha=0.4)
            if use_fcw:
                disp = disparity_sgbm(fr, right_in[i])
                checks = forward_collision_check(tracks, disp, Kmat, baseline, warn_distance=warn_dist)
                annotated.append(draw_collision(base, checks))
            else:
                annotated.append(draw_tracks(base, tracks))
            bar.progress((i + 1) / len(frames_in))
        bar.empty()

        out_path = Path("outputs/demo/tracking_app.mp4")
        write_video(out_path, annotated)
        st.success(f"Done - {len(annotated)} frames" + (" (with collision warning)" if use_fcw else ""))
        st.video(str(out_path))
        st.download_button("Download MP4", out_path.read_bytes(), "tracking.mp4", "video/mp4")

# ---------------------------------------------------------------- slam
with tab_slam:
    st.subheader("Visual odometry -> loop-closure SLAM (KITTI)")
    st.markdown(
        "Monocular VO drifts and loses scale; **stereo** recovers metric scale; "
        "**loop closure** removes the remaining drift."
    )
    cols = st.columns(3)
    panels = [
        ("Monocular VO (scale collapse)", "monocular_vo.png", "vo_kitti_trajectory.png"),
        ("Stereo VO (metric)", "stereo_vo.png", "stereo_vo_kitti_trajectory.png"),
        ("Loop-closure SLAM", "slam_loop_closure.png", "slam_kitti_trajectory.png"),
    ]
    for col, (title, committed, fresh) in zip(cols, panels, strict=True):
        path = IMAGES_DIR / committed
        if not path.exists():
            path = OUTPUTS_DIR / fresh
        with col:
            st.caption(title)
            if path.exists():
                st.image(str(path), use_column_width=True)
            else:
                st.info(f"Run the KITTI script to generate {fresh}.")

    st.markdown(
        """
        | Stage | Metric (KITTI 00) |
        |---|---|
        | Monocular VO | scale collapses; ATE ~185 m |
        | Stereo VO | metric; **ATE ~26 m (~0.7%)** |
        | Loop-closure SLAM | **ATE 25 -> 9.5 m (-62%)** |
        | SLAM on seq 05 | **8.6 -> 5.8 m (-33%)**, same params |
        """
    )
