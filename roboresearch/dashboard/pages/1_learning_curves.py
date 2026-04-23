from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Learning Curves — RoboResearch", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "registry"
EXPERIMENTS_TSV = REGISTRY / "experiments.tsv"


def load_experiments() -> pd.DataFrame | None:
    if not EXPERIMENTS_TSV.exists():
        return None
    df = pd.read_csv(EXPERIMENTS_TSV, sep="\t")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


st.title("Learning Curves")

df = load_experiments()

if df is None or df.empty:
    st.warning("No experiment data found.")
    st.stop()

st.sidebar.header("Filters")

envs = df["env_name"].unique().tolist()
selected_envs = st.sidebar.multiselect("Environment", envs, default=envs)

algos = df["algorithm"].unique().tolist()
selected_algos = st.sidebar.multiselect("Algorithm", algos, default=algos)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

filtered = df[
    (df["env_name"].isin(selected_envs))
    & (df["algorithm"].isin(selected_algos))
].copy()
if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["timestamp"].dt.date >= start) & (filtered["timestamp"].dt.date <= end)
    ]

if filtered.empty:
    st.info("No experiments match the selected filters.")
    st.stop()

filtered = filtered.sort_values("timestamp")
filtered["experiment_num"] = range(1, len(filtered) + 1)

algo_color_map = {"SAC": "#636EFA", "PPO": "#EF553B", "TD3": "#00CC96"}


def _find_switch_experiment_nums(filtered_df: pd.DataFrame, column: str) -> list[int]:
    """Find experiment_num values where a column value changes."""
    nums = []
    values = filtered_df[column].tolist()
    exp_nums = filtered_df["experiment_num"].tolist()
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            nums.append(exp_nums[i])
    return nums


algo_switch_nums = _find_switch_experiment_nums(filtered, "algorithm")
env_switch_nums = _find_switch_experiment_nums(filtered, "env_name")


def _add_switch_lines(fig, algo_nums, env_nums):
    for x in algo_nums:
        fig.add_vline(
            x=x, line_dash="dash", line_color="gray", opacity=0.5,
            annotation_text="algo switch", annotation_position="top left",
            annotation_font_size=9,
        )
    for x in env_nums:
        fig.add_vline(
            x=x, line_dash="dash", line_color="red", opacity=0.5,
            annotation_text="task graduation", annotation_position="top right",
            annotation_font_size=9,
        )


st.subheader("Success Rate over Experiments")

fig_sr = px.line(
    filtered,
    x="experiment_num",
    y="success_rate",
    color="algorithm",
    color_discrete_map=algo_color_map,
    markers=True,
    labels={"experiment_num": "Experiment #", "success_rate": "Success Rate"},
)
fig_sr.update_layout(yaxis_range=[0, 1], height=450, margin=dict(t=20, b=40))
_add_switch_lines(fig_sr, algo_switch_nums, env_switch_nums)

st.plotly_chart(fig_sr, use_container_width=True)

st.subheader("Mean Reward over Experiments")

fig_rw = px.line(
    filtered,
    x="experiment_num",
    y="mean_reward",
    color="algorithm",
    color_discrete_map=algo_color_map,
    markers=True,
    labels={"experiment_num": "Experiment #", "mean_reward": "Mean Reward"},
)
fig_rw.update_layout(height=450, margin=dict(t=20, b=40))
_add_switch_lines(fig_rw, algo_switch_nums, env_switch_nums)

st.plotly_chart(fig_rw, use_container_width=True)
