from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RoboResearch", layout="wide", page_icon="\U0001f916")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "registry"
EXPERIMENTS_TSV = REGISTRY / "experiments.tsv"


def load_experiments() -> pd.DataFrame | None:
    if not EXPERIMENTS_TSV.exists():
        return None
    df = pd.read_csv(EXPERIMENTS_TSV, sep="\t")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def is_system_running(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    latest = df["timestamp"].max()
    now = datetime.now(timezone.utc) if latest.tzinfo else datetime.now()
    return (now - latest) < timedelta(minutes=5)


st.title("RoboResearch — Autonomous Robotics Research")

df = load_experiments()

if df is None or df.empty:
    st.warning("No experiment data found. Start the autonomous loop to generate data.")
    st.code("python -m roboresearch.main --max-experiments 50 --time-budget 300", language="bash")
    st.stop()

running = is_system_running(df)
best_sr = df["success_rate"].max()
best_run = df.loc[df["success_rate"].idxmax()]
current_env = df.iloc[-1]["env_name"]
current_algo = df.iloc[-1]["algorithm"]
total = len(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Experiments", total)
col2.metric("Best Success Rate", f"{best_sr:.1%}")
col3.metric("Current Task", current_env)
col4.metric("Current Algorithm", current_algo)
if running:
    col5.metric("System Status", "RUNNING")
    col5.caption(":green[Active]")
else:
    col5.metric("System Status", "IDLE")
    col5.caption("No recent activity")

st.divider()

st.subheader("Learning Curve (Last 20 Experiments)")

recent = df.tail(20).copy()
recent["experiment_num"] = range(len(recent))

fig = px.line(
    recent,
    x="experiment_num",
    y="success_rate",
    color="algorithm",
    color_discrete_map={"SAC": "#636EFA", "PPO": "#EF553B", "TD3": "#00CC96"},
    markers=True,
    labels={"experiment_num": "Experiment #", "success_rate": "Success Rate"},
)
fig.update_layout(
    yaxis_range=[0, 1],
    height=350,
    margin=dict(t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Latest Experiments")
    latest = df.tail(5)[["run_id", "algorithm", "env_name", "success_rate", "mean_reward"]].copy()
    latest = latest.iloc[::-1]
    st.dataframe(latest, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Experiments by Algorithm")
    algo_counts = df["algorithm"].value_counts().reset_index()
    algo_counts.columns = ["Algorithm", "Count"]
    fig2 = px.pie(algo_counts, names="Algorithm", values="Count", hole=0.4)
    fig2.update_layout(height=250, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

if st.button("Refresh Data"):
    st.rerun()
