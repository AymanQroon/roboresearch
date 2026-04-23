import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cost Tracker — RoboResearch", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "registry"
AGENT_LOG = REGISTRY / "agent_log.jsonl"

COST_RATES = {
    "opus": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    "sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "haiku": {"input": 0.80 / 1_000_000, "output": 4.0 / 1_000_000},
}

MODEL_TO_TIER = {
    "claude-opus-4-6": "opus",
    "claude-opus-4-7": "opus",
    "claude-sonnet-4-5@20250929": "sonnet",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
}


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
    return pd.DataFrame(entries)


def compute_cost(row) -> float:
    tokens = row.get("tokens_used")
    if not isinstance(tokens, dict):
        return 0.0
    model = row.get("model", "")
    tier = MODEL_TO_TIER.get(model, "sonnet")
    rates = COST_RATES[tier]
    input_cost = tokens.get("input", 0) * rates["input"]
    output_cost = tokens.get("output", 0) * rates["output"]
    return input_cost + output_cost


st.title("Cost Tracker")

df = load_agent_log()

if df is None or df.empty:
    st.info("No agent log data available for cost tracking.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["input_tokens"] = df["tokens_used"].apply(lambda t: t.get("input", 0) if isinstance(t, dict) else 0)
df["output_tokens"] = df["tokens_used"].apply(lambda t: t.get("output", 0) if isinstance(t, dict) else 0)
df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
df["cost"] = df.apply(compute_cost, axis=1)

total_cost = df["cost"].sum()
total_tokens = df["total_tokens"].sum()
total_input = df["input_tokens"].sum()
total_output = df["output_tokens"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cost", f"${total_cost:.2f}")
col2.metric("Total Tokens", f"{total_tokens:,}")
col3.metric("Input Tokens", f"{total_input:,}")
col4.metric("Output Tokens", f"{total_output:,}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Tokens by Agent")
    agent_tokens = df.groupby("agent_name")[["input_tokens", "output_tokens"]].sum().reset_index()
    agent_tokens_melted = agent_tokens.melt(
        id_vars="agent_name", value_vars=["input_tokens", "output_tokens"],
        var_name="token_type", value_name="tokens"
    )
    fig_tokens = px.bar(
        agent_tokens_melted,
        x="agent_name",
        y="tokens",
        color="token_type",
        barmode="group",
        labels={"agent_name": "Agent", "tokens": "Tokens", "token_type": "Type"},
        color_discrete_map={"input_tokens": "#636EFA", "output_tokens": "#EF553B"},
    )
    fig_tokens.update_layout(height=400, margin=dict(t=20, b=40))
    st.plotly_chart(fig_tokens, use_container_width=True)

with col_right:
    st.subheader("Cost by Agent")
    agent_cost = df.groupby("agent_name")["cost"].sum().reset_index()
    fig_cost_bar = px.bar(
        agent_cost,
        x="agent_name",
        y="cost",
        labels={"agent_name": "Agent", "cost": "Cost ($)"},
        color="agent_name",
    )
    fig_cost_bar.update_layout(height=400, margin=dict(t=20, b=40), showlegend=False)
    st.plotly_chart(fig_cost_bar, use_container_width=True)

st.divider()
st.subheader("Cumulative Cost over Time")

df_sorted = df.sort_values("timestamp").copy()
df_sorted["cumulative_cost"] = df_sorted["cost"].cumsum()

fig_cum = px.line(
    df_sorted,
    x="timestamp",
    y="cumulative_cost",
    labels={"timestamp": "Time", "cumulative_cost": "Cumulative Cost ($)"},
    markers=True,
)
fig_cum.update_layout(height=400, margin=dict(t=20, b=40))
st.plotly_chart(fig_cum, use_container_width=True)

st.divider()
st.subheader("Cost by Model Tier")

df["tier"] = df["model"].map(MODEL_TO_TIER).fillna("unknown")
tier_cost = df.groupby("tier")["cost"].sum().reset_index()
fig_tier = px.pie(tier_cost, names="tier", values="cost", hole=0.4)
fig_tier.update_layout(height=300, margin=dict(t=10, b=10))
st.plotly_chart(fig_tier, use_container_width=True)
