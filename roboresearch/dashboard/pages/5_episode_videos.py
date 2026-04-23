from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Episode Videos — RoboResearch", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "registry"
VIDEOS_DIR = REGISTRY / "videos"

st.title("Episode Videos")

if not VIDEOS_DIR.exists() or not list(VIDEOS_DIR.glob("**/*.mp4")):
    st.info(
        "No episode videos found. Videos will appear in `registry/videos/` once "
        "the system records evaluation episodes."
    )
    st.stop()

videos = sorted(VIDEOS_DIR.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)

video_info = []
for v in videos:
    name = v.stem
    run_id = v.parent.name if v.parent != VIDEOS_DIR else name
    is_best = "best" in name.lower()
    is_worst = "worst" in name.lower()
    label = "Best" if is_best else "Worst" if is_worst else "Episode"
    video_info.append({"path": v, "run_id": run_id, "outcome": label, "name": name})

bests = [v for v in video_info if v["outcome"] == "Best"]
worsts = [v for v in video_info if v["outcome"] == "Worst"]

if bests or worsts:
    st.subheader("Best vs Worst Episodes")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Best Episode**")
        if bests:
            best = bests[0]
            st.video(str(best["path"]))
            st.caption(f"Run: {best['run_id']}")
        else:
            st.info("No best episodes recorded yet.")

    with col2:
        st.markdown("**Worst Episode**")
        if worsts:
            worst = worsts[0]
            st.video(str(worst["path"]))
            st.caption(f"Run: {worst['run_id']}")
        else:
            st.info("No worst episodes recorded yet.")

st.divider()
st.subheader("All Recorded Episodes")

cols_per_row = 3
for i in range(0, len(video_info), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        idx = i + j
        if idx < len(video_info):
            v = video_info[idx]
            with col:
                st.video(str(v["path"]))
                st.caption(f"{v['run_id']} | {v['outcome']}")
