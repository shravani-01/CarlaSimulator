"""CarlaPerception — interactive web demo (Streamlit).

Two tabs:
  1. Live perception  — run detection + segmentation on a KITTI driving frame,
     a built-in sample, or an uploaded image, and see the annotated result.
  2. SLAM results     — view the monocular vs. stereo vs. loop-closure trajectory
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

st.set_page_config(page_title="CarlaPerception", layout="wide")

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = Path("data/sample/bus.jpg")
IMAGES_DIR = Path("docs/images")
OUTPUTS_DIR = Path("outputs/demo")


@st.cache_resource(show_spinner="Loading perception models…")
def get_pipeline(device: str | None) -> PerceptionPipeline:
    return PerceptionPipeline(device=device)


@st.cache_resource(show_spinner="Indexing KITTI sequence…")
def get_kitti(root: str, seq: str) -> KITTIOdometry:
    return KITTIOdometry(root, seq)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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

tab_perc, tab_slam = st.tabs(["🔍 Live perception", "🛰️ SLAM results"])

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
        with st.spinner("Running detection + segmentation…"):
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

# ---------------------------------------------------------------- slam
with tab_slam:
    st.subheader("Visual odometry → loop-closure SLAM (KITTI)")
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
        | Loop-closure SLAM | **ATE 25 → 9.5 m (−62%)** |
        | SLAM on seq 05 | **8.6 → 5.8 m (−33%)**, same params |
        """
    )
