from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_ROOT = PROJECT_ROOT / "registry"
MODELS_DIR = REGISTRY_ROOT / "models"
METADATA_DIR = REGISTRY_ROOT / "metadata"
EXPERIMENTS_TSV = REGISTRY_ROOT / "experiments.tsv"

TSV_HEADER = "run_id\ttimestamp\talgorithm\tenv_name\tsuccess_rate\tmean_reward\tnotes"

mcp = FastMCP("registry")


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENTS_TSV.exists():
        EXPERIMENTS_TSV.write_text(TSV_HEADER + "\n")


def _read_metadata(run_id: str) -> dict[str, Any]:
    path = METADATA_DIR / f"{run_id}.json"
    if not path.exists():
        raise ValueError(f"No experiment found with run_id: {run_id}")
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Corrupted metadata for run_id {run_id}: {exc}") from exc


def _all_metadata() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in METADATA_DIR.glob("*.json"):
        try:
            results.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return results


@mcp.tool()
def save_checkpoint(
    run_id: str,
    model_path: str,
    config: dict,
    metrics: dict,
    notes: str = "",
) -> dict:
    """Save a model checkpoint and its metadata to the registry."""
    _ensure_dirs()

    if (METADATA_DIR / f"{run_id}.json").exists():
        raise ValueError(f"Duplicate run_id: {run_id} already exists in the registry")

    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    dest_dir = MODELS_DIR / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / source.name
    shutil.copy2(source, dest_path)

    timestamp = datetime.now(timezone.utc).isoformat()

    metadata = {
        "run_id": run_id,
        "model_path": str(dest_path),
        "config": config,
        "metrics": metrics,
        "notes": notes,
        "timestamp": timestamp,
    }
    (METADATA_DIR / f"{run_id}.json").write_text(json.dumps(metadata, indent=2))

    algorithm = config.get("algorithm", "")
    env_name = config.get("env_name", "")
    success_rate = metrics.get("success_rate", "")
    mean_reward = metrics.get("mean_reward", "")
    tsv_row = f"{run_id}\t{timestamp}\t{algorithm}\t{env_name}\t{success_rate}\t{mean_reward}\t{notes}"

    with open(EXPERIMENTS_TSV, "a") as f:
        f.write(tsv_row + "\n")

    return {"status": "saved", "run_id": run_id, "registry_path": str(dest_path)}


@mcp.tool()
def load_checkpoint(run_id: str) -> dict:
    """Load metadata and model path for a given experiment run."""
    _ensure_dirs()
    meta = _read_metadata(run_id)
    return {
        "model_path": meta["model_path"],
        "config": meta["config"],
        "metrics": meta["metrics"],
        "notes": meta["notes"],
        "timestamp": meta["timestamp"],
    }


@mcp.tool()
def list_experiments(
    filter_by: dict | None = None,
    sort_by: str = "timestamp",
    limit: int = 20,
) -> dict:
    """Query experiment history with optional filtering and sorting."""
    _ensure_dirs()
    experiments = _all_metadata()

    if filter_by:
        if "env_name" in filter_by:
            target_env = filter_by["env_name"]
            experiments = [e for e in experiments if e.get("config", {}).get("env_name") == target_env]
        if "algorithm" in filter_by:
            target_algo = filter_by["algorithm"]
            experiments = [
                e for e in experiments if e.get("config", {}).get("algorithm") == target_algo
            ]
        if "min_success_rate" in filter_by:
            threshold = float(filter_by["min_success_rate"])
            experiments = [
                e for e in experiments if float(e.get("metrics", {}).get("success_rate", 0)) >= threshold
            ]

    valid_sort_keys = {"timestamp", "success_rate", "reward"}
    if sort_by not in valid_sort_keys:
        sort_by = "timestamp"

    def sort_key(exp: dict) -> Any:
        if sort_by == "timestamp":
            return exp.get("timestamp", "")
        if sort_by == "success_rate":
            return float(exp.get("metrics", {}).get("success_rate", 0))
        if sort_by == "reward":
            return float(exp.get("metrics", {}).get("mean_reward", 0))
        return ""

    experiments.sort(key=sort_key, reverse=True)

    experiments = experiments[:limit]

    summaries = []
    for exp in experiments:
        summaries.append({
            "run_id": exp.get("run_id"),
            "timestamp": exp.get("timestamp"),
            "algorithm": exp.get("config", {}).get("algorithm"),
            "env_name": exp.get("config", {}).get("env_name"),
            "success_rate": exp.get("metrics", {}).get("success_rate"),
            "mean_reward": exp.get("metrics", {}).get("mean_reward"),
            "notes": exp.get("notes", ""),
        })

    return {"experiments": summaries, "total": len(summaries)}


@mcp.tool()
def get_best_model(env_name: str, metric: str = "success_rate") -> dict:
    """Find the experiment with the highest value for a given metric, filtered by environment."""
    _ensure_dirs()
    experiments = _all_metadata()
    experiments = [
        e for e in experiments if e.get("config", {}).get("env_name") == env_name
    ]

    if not experiments:
        raise ValueError(f"No experiments found for env_name: {env_name}")

    best = max(experiments, key=lambda e: float(e.get("metrics", {}).get(metric, 0)))

    return {
        "run_id": best["run_id"],
        "model_path": best["model_path"],
        "config": best["config"],
        "metrics": best["metrics"],
    }


@mcp.tool()
def diff_configs(run_id_a: str, run_id_b: str) -> dict:
    """Compare configs between two experiment runs and return the differences."""
    _ensure_dirs()
    config_a = _read_metadata(run_id_a)["config"]
    config_b = _read_metadata(run_id_b)["config"]

    keys_a = set(config_a.keys())
    keys_b = set(config_b.keys())

    changed = {}
    for key in keys_a & keys_b:
        if config_a[key] != config_b[key]:
            changed[key] = {"old": config_a[key], "new": config_b[key]}

    return {
        "changed": changed,
        "added": {k: config_b[k] for k in keys_b - keys_a},
        "removed": {k: config_a[k] for k in keys_a - keys_b},
    }


if __name__ == "__main__":
    mcp.run()
