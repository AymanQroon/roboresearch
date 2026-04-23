import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Experiment Log — RoboResearch", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "registry"
EXPERIMENTS_TSV = REGISTRY / "experiments.tsv"
METADATA_DIR = REGISTRY / "metadata"


def load_experiments() -> pd.DataFrame | None:
    if not EXPERIMENTS_TSV.exists():
        return None
    df = pd.read_csv(EXPERIMENTS_TSV, sep="\t")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_metadata(run_id: str) -> dict | None:
    path = METADATA_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


st.title("Experiment Log")

df = load_experiments()

if df is None or df.empty:
    st.warning("No experiment data found.")
    st.stop()

display = df.copy()
display["delta"] = display["success_rate"].diff()
display["delta"] = display["delta"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "—")

display_cols = ["run_id", "timestamp", "algorithm", "env_name", "success_rate", "mean_reward", "delta", "notes"]
display = display[display_cols]


def highlight_delta(row):
    delta_str = row["delta"]
    if delta_str == "—":
        return [""] * len(row)
    val = float(delta_str)
    if val > 0:
        return ["background-color: rgba(0, 200, 0, 0.12)"] * len(row)
    elif val < 0:
        return ["background-color: rgba(200, 0, 0, 0.12)"] * len(row)
    return [""] * len(row)


styled = display.style.apply(highlight_delta, axis=1)

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=600,
)

st.divider()
st.subheader("Experiment Details")

run_ids = df["run_id"].tolist()
selected_run = st.selectbox("Select an experiment to view full config", run_ids)

if selected_run:
    meta = load_metadata(selected_run)
    if meta:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Configuration**")
            st.json(meta.get("config", {}))
        with col2:
            st.markdown("**Metrics**")
            st.json(meta.get("metrics", {}))
            if meta.get("notes"):
                st.markdown(f"**Notes:** {meta['notes']}")
            st.markdown(f"**Model path:** `{meta.get('model_path', 'N/A')}`")
    else:
        st.info(f"No metadata file found for {selected_run}.")
