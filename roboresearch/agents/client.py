from __future__ import annotations

import os

import anthropic

_VERTEX_PROJECT_ID = "your-gcp-project-id"
_VERTEX_REGION = "us-central1"

MODELS = {
    "orchestrator": "claude-opus-4-6",
    "experiment_coder": "claude-sonnet-4-5@20250929",
    "failure_analyst": "claude-opus-4-6",
    "quick_evaluator": "claude-haiku-4-5-20251001",
}


def create_client() -> anthropic.Anthropic | anthropic.AnthropicVertex:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()
    return anthropic.AnthropicVertex(
        project_id=os.environ.get("VERTEX_PROJECT_ID", _VERTEX_PROJECT_ID),
        region=os.environ.get("VERTEX_REGION", _VERTEX_REGION),
    )
