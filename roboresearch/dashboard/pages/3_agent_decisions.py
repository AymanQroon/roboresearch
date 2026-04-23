import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agent Decisions — RoboResearch", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "registry"
AGENT_LOG = REGISTRY / "agent_log.jsonl"


def load_agent_log() -> pd.DataFrame | None:
    if not AGENT_LOG.exists():
        return None
    entries = []
    with open(AGENT_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        return None
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


st.title("Agent Decisions")

df = load_agent_log()

if df is None or df.empty:
    st.info(
        "No agent decision log found. The file `registry/agent_log.jsonl` will be "
        "populated once the autonomous agent loop starts running."
    )
    st.stop()

st.sidebar.header("Filters")
agents = sorted(df["agent_name"].unique().tolist())
selected_agents = st.sidebar.multiselect("Agent", agents, default=agents)

filtered = df[df["agent_name"].isin(selected_agents)].sort_values("timestamp", ascending=False)

if filtered.empty:
    st.info("No decisions match the selected filters.")
    st.stop()

st.metric("Total Decisions", len(filtered))

st.divider()

for _, row in filtered.iterrows():
    ts = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    agent = row["agent_name"]
    action = row["action"]
    run_id = row.get("run_id", "")
    reasoning = row.get("reasoning", "")

    agent_colors = {
        "Orchestrator": ":blue",
        "Experiment Coder": ":green",
        "Failure Analyst": ":orange",
        "Quick Evaluator": ":violet",
    }
    color = agent_colors.get(agent, "")
    badge = f"{color}[{agent}]" if color else agent

    with st.expander(f"{ts}  |  {badge}  |  **{action}**  |  `{run_id}`", expanded=False):
        st.markdown(f"**Agent:** {agent}")
        st.markdown(f"**Action:** {action}")
        st.markdown(f"**Run ID:** {run_id}")
        if "model" in row:
            st.markdown(f"**Model:** {row['model']}")
        if "tokens_used" in row and isinstance(row["tokens_used"], dict):
            tokens = row["tokens_used"]
            st.markdown(f"**Tokens:** {tokens.get('input', 0):,} input / {tokens.get('output', 0):,} output")
        st.markdown("---")
        st.markdown(f"**Reasoning:**\n\n{reasoning}")
