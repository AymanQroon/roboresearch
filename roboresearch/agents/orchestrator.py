from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from rich.console import Console
from rich.table import Table

from roboresearch.agents.client import MODELS, create_client
from roboresearch.agents.experiment_coder import ExperimentCoder
from roboresearch.agents.failure_analyst import FailureAnalyst, TASK_INFO
from roboresearch.agents.quick_evaluator import QuickEvaluator
from roboresearch.training import (
    create_model,
    load_model,
    train_model,
    save_model,
    evaluate_model,
    record_best_and_worst,
    frames_to_base64,
    get_default_config,
)

_PLANNING_MODEL = MODELS["orchestrator"]

_CURRICULUM = ["FetchReach-v4", "FetchPush-v4", "FetchPickAndPlace-v4"]

_GRADUATION_THRESHOLD = 0.80
_GRADUATION_CONSECUTIVE = 3
_STALE_THRESHOLD = 5

_PLANNING_SYSTEM = """\
You are the chief scientist of an autonomous robotics research lab. You plan experiments \
to improve robot manipulation policies trained in MuJoCo simulation.

Your job is to propose the NEXT experiment based on what has been tried, what worked, \
what failed, and the overall research direction.

Be specific and actionable. Your plan will be given to an experiment coder who generates \
training configurations. Focus on ONE clear hypothesis per experiment.

Consider:
- Hyperparameter changes (learning rate, batch size, network architecture)
- Algorithmic switches (SAC, PPO, TD3) when progress stalls
- Reward shaping strategies
- Exploration-exploitation tradeoffs
- Training duration adjustments

Keep plans to 3-5 sentences. State the hypothesis clearly.\
"""


console = Console()


def _normalize_token_usage(usage: dict) -> dict:
    if "input" in usage:
        return usage
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
    }


class RegistryClient:
    def __init__(self, registry_dir: str):
        self.root = Path(registry_dir)
        self.metadata_dir = self.root / "metadata"
        self.models_dir = self.root / "models"
        self.videos_dir = self.root / "videos"
        self.experiments_tsv = self.root / "experiments.tsv"
        self.agent_log = self.root / "agent_log.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        if not self.experiments_tsv.exists():
            self.experiments_tsv.write_text(
                "run_id\ttimestamp\talgorithm\tenv_name\tsuccess_rate\tmean_reward\tnotes\n"
            )

    def next_run_id(self) -> str:
        existing = sorted(self.metadata_dir.glob("run_*.json"))
        if not existing:
            return "run_001"
        last = existing[-1].stem
        num = int(last.split("_")[1]) + 1
        return f"run_{num:03d}"

    def load_all_metadata(self) -> list[dict]:
        results = []
        for path in sorted(self.metadata_dir.glob("run_*.json")):
            try:
                results.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def load_recent(self, n: int = 10) -> list[dict]:
        all_meta = self.load_all_metadata()
        return all_meta[-n:]

    def get_best_metrics(self, env_name: str) -> dict | None:
        all_meta = self.load_all_metadata()
        env_runs = [
            m for m in all_meta if m.get("config", {}).get("env_name") == env_name
        ]
        if not env_runs:
            return None
        best = max(
            env_runs,
            key=lambda m: float(m.get("metrics", {}).get("success_rate", 0)),
        )
        return best.get("metrics")

    def save_checkpoint(
        self,
        run_id: str,
        model_path: str,
        config: dict,
        metrics: dict,
        notes: str = "",
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()

        dest_dir = self.models_dir / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        source = Path(model_path)
        for f in source.parent.glob(f"{source.stem}*"):
            shutil.copy2(f, dest_dir / f.name)

        metadata = {
            "run_id": run_id,
            "model_path": str(dest_dir / source.name),
            "config": config,
            "metrics": metrics,
            "notes": notes,
            "timestamp": timestamp,
        }
        (self.metadata_dir / f"{run_id}.json").write_text(json.dumps(metadata, indent=2))

        algorithm = config.get("algorithm", "")
        env_name = config.get("env_name", "")
        success_rate = metrics.get("success_rate", "")
        mean_reward = metrics.get("mean_reward", "")
        tsv_row = (
            f"{run_id}\t{timestamp}\t{algorithm}\t{env_name}"
            f"\t{success_rate}\t{mean_reward}\t{notes}"
        )
        with open(self.experiments_tsv, "a") as f:
            f.write(tsv_row + "\n")

    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        reasoning: str,
        tokens_used: dict,
        model: str,
        run_id: str | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": agent_name,
            "model": model,
            "action": action,
            "reasoning": reasoning,
            "run_id": run_id,
            "tokens_used": tokens_used,
        }
        with open(self.agent_log, "a") as f:
            f.write(json.dumps(entry) + "\n")


class Orchestrator:
    def __init__(self, registry_dir: str = "registry", **kwargs):
        self._client = create_client()
        self.registry = RegistryClient(registry_dir)
        self.experiment_coder = ExperimentCoder()
        self.failure_analyst = FailureAnalyst()
        self.quick_evaluator = QuickEvaluator()

    def run(
        self,
        program_path: str = "program.md",
        max_experiments: int = 50,
        time_budget_seconds: int = 300,
    ) -> None:
        program = self._read_program(program_path)
        past_experiments = self.registry.load_all_metadata()

        current_env = "FetchReach-v4"
        current_algorithm = "SAC"
        if past_experiments:
            last = past_experiments[-1]
            current_env = last.get("config", {}).get("env_name", current_env)
            current_algorithm = last.get("config", {}).get("algorithm", current_algorithm)

        failure_analysis_text: str | None = None
        experiments_without_improvement = 0
        results_log: list[dict] = []

        console.rule("[bold blue]RoboResearch Autonomous Loop[/bold blue]")
        console.print(f"Environment: [cyan]{current_env}[/cyan]")
        console.print(f"Algorithm: [cyan]{current_algorithm}[/cyan]")
        console.print(f"Max experiments: [cyan]{max_experiments}[/cyan]")
        console.print(f"Training budget: [cyan]{time_budget_seconds}s[/cyan] per run")
        console.print(f"Past experiments in registry: [cyan]{len(past_experiments)}[/cyan]")
        console.rule()

        for i in range(1, max_experiments + 1):
            console.print(f"\n[bold]=== Experiment {i}/{max_experiments} ===[/bold]")

            recent = self.registry.load_recent(10)

            new_env = self._should_graduate_task(recent, current_env)
            if new_env:
                console.print(
                    f"[bold green]GRADUATING[/bold green] from {current_env} to {new_env}"
                )
                current_env = new_env
                experiments_without_improvement = 0
                failure_analysis_text = None

            new_algo = self._should_switch_algorithm(recent, current_env)
            if new_algo and new_algo != current_algorithm:
                console.print(
                    f"[bold yellow]SWITCHING ALGORITHM[/bold yellow] "
                    f"from {current_algorithm} to {new_algo}"
                )
                current_algorithm = new_algo
                experiments_without_improvement = 0

            try:
                result = self._run_single_experiment(
                    experiment_number=i,
                    program=program,
                    recent_experiments=recent,
                    failure_analysis_text=failure_analysis_text,
                    current_env=current_env,
                    current_algorithm=current_algorithm,
                    experiments_without_improvement=experiments_without_improvement,
                    time_budget_seconds=time_budget_seconds,
                )
            except Exception:
                console.print("[bold red]EXPERIMENT CRASHED[/bold red]")
                console.print(traceback.format_exc())
                self.registry.log_agent_action(
                    agent_name="Orchestrator",
                    action="experiment_crash",
                    reasoning=traceback.format_exc()[-500:],
                    tokens_used={"input": 0, "output": 0},
                    model=_PLANNING_MODEL,
                )
                continue

            results_log.append(result)

            if result["decision"] == "keep":
                experiments_without_improvement = 0
            else:
                experiments_without_improvement += 1

            failure_analysis_text = result.get("failure_analysis_text")

            self._print_progress_table(results_log[-10:])

        self._print_final_summary(results_log)

    def _run_single_experiment(
        self,
        experiment_number: int,
        program: str,
        recent_experiments: list[dict],
        failure_analysis_text: str | None,
        current_env: str,
        current_algorithm: str,
        experiments_without_improvement: int,
        time_budget_seconds: int,
    ) -> dict:
        run_id = self.registry.next_run_id()
        console.print(f"Run ID: [cyan]{run_id}[/cyan] | Env: {current_env} | Algo: {current_algorithm}")

        # --- Step 1: Plan experiment ---
        console.print("  Planning experiment...")
        plan, plan_usage = self._plan_experiment(
            program=program,
            past_experiments=recent_experiments,
            failure_analysis=failure_analysis_text,
            current_env=current_env,
            current_algorithm=current_algorithm,
        )
        console.print(f"  Plan: [dim]{plan[:120]}...[/dim]")
        self.registry.log_agent_action(
            agent_name="Orchestrator",
            action="plan_experiment",
            reasoning=plan,
            tokens_used=plan_usage,
            model=_PLANNING_MODEL,
            run_id=run_id,
        )

        # --- Step 2: Generate config ---
        console.print("  Generating training config...")
        current_config = get_default_config(current_algorithm, current_env)

        past_for_coder = []
        for exp in recent_experiments:
            past_for_coder.append({
                "config": exp.get("config", {}),
                "metrics": exp.get("metrics", {}),
                "notes": exp.get("notes", ""),
            })

        coder_result = self.experiment_coder.generate_config(
            experiment_plan=plan,
            current_config=current_config,
            past_experiments=past_for_coder,
            failure_analysis=failure_analysis_text,
            env_name=current_env,
        )
        config = coder_result["config"]
        changes = coder_result.get("changes", [])
        console.print(f"  Changes: {changes}")
        self.registry.log_agent_action(
            agent_name="ExperimentCoder",
            action="generate_config",
            reasoning=coder_result.get("reasoning", ""),
            tokens_used=_normalize_token_usage(coder_result.get("usage", {})),
            model=MODELS["experiment_coder"],
            run_id=run_id,
        )

        # --- Step 3: Train model (warm-start from best checkpoint if available) ---
        console.print(f"  Training for up to {time_budget_seconds}s...")
        algorithm = config.get("algorithm", current_algorithm)
        env_name = config.get("env_name", current_env)

        best_checkpoint = self._find_best_checkpoint(env_name, algorithm)
        if best_checkpoint:
            console.print(f"  Warm-starting from {best_checkpoint['run_id']}")
            model = load_model(algorithm, best_checkpoint["model_path"], env_name)
            lr = config.get("learning_rate")
            if lr is not None:
                model.learning_rate = lr
        else:
            hyperparams = {k: v for k, v in config.items() if k not in ("algorithm", "env_name")}
            model = create_model(algorithm, env_name, hyperparams)

        train_result = train_model(model, time_budget_seconds=time_budget_seconds)
        console.print(
            f"  Trained {train_result['total_timesteps_trained']} steps "
            f"in {train_result['elapsed_time']}s"
        )

        tmp_model_path = f"/tmp/roboresearch_{run_id}"
        save_model(model, tmp_model_path)

        # --- Step 4: Evaluate ---
        console.print("  Evaluating (20 episodes)...")
        eval_result = evaluate_model(model, env_name, num_episodes=20, capture_frames=True)
        metrics = eval_result["summary"]
        console.print(
            f"  success_rate={metrics['success_rate']:.3f}  "
            f"mean_reward={metrics['mean_reward']:.2f}"
        )

        # --- Step 5: Quick evaluation decision ---
        best_metrics = self.registry.get_best_metrics(env_name)
        notes = "; ".join(changes) if changes else coder_result.get("reasoning", "")

        eval_decision = self.quick_evaluator.evaluate_decision(
            current_metrics=metrics,
            best_metrics=best_metrics,
            experiment_notes=notes,
            num_experiments_without_improvement=experiments_without_improvement,
        )
        decision = eval_decision["decision"]
        console.print(
            f"  Decision: [{'green' if decision == 'keep' else 'red'}]"
            f"{decision.upper()}[/{'green' if decision == 'keep' else 'red'}] "
            f"({eval_decision['reasoning'][:100]})"
        )
        self.registry.log_agent_action(
            agent_name="QuickEvaluator",
            action="evaluate_decision",
            reasoning=eval_decision["reasoning"],
            tokens_used=_normalize_token_usage(eval_decision.get("usage", {})),
            model=MODELS["quick_evaluator"],
            run_id=run_id,
        )

        # --- Step 6: Post-decision actions ---
        failure_analysis_text = None
        failed_episodes = [
            ep for ep in eval_result["episodes"] if not ep["success"]
        ]

        # Always save checkpoint so warm-starting accumulates training steps.
        # The keep/discard decision determines whether this counts as "best."
        self.registry.save_checkpoint(
            run_id=run_id,
            model_path=tmp_model_path,
            config=config,
            metrics=metrics,
            notes=notes + (" [DISCARDED]" if decision == "discard" else ""),
        )

        if decision == "keep":
            try:
                video_dir = str(self.registry.videos_dir / run_id)
                record_best_and_worst(model, env_name, num_episodes=5, output_dir=video_dir)
                console.print(f"  Videos saved to {video_dir}")
            except Exception:
                console.print("  [dim]Video recording skipped (non-critical error)[/dim]")

        if failed_episodes:
            failure_analysis_text = self._run_failure_analysis(
                failed_episodes, env_name, run_id
            )

        delta = None
        if best_metrics:
            delta = metrics["success_rate"] - best_metrics.get("success_rate", 0)

        return {
            "run_id": run_id,
            "env": env_name,
            "algorithm": algorithm,
            "success_rate": metrics["success_rate"],
            "mean_reward": metrics["mean_reward"],
            "decision": decision,
            "delta": delta,
            "failure_analysis_text": failure_analysis_text,
        }

    def _run_failure_analysis(
        self,
        failed_episodes: list[dict],
        env_name: str,
        run_id: str,
    ) -> str | None:
        task_info = TASK_INFO.get(env_name, {})
        if not task_info:
            return None

        episodes_with_frames = []
        for ep in failed_episodes:
            raw_frames = ep.get("frames", [])
            if not raw_frames:
                continue
            b64_frames = frames_to_base64(raw_frames)
            episodes_with_frames.append({
                "frames": b64_frames,
                "reward": ep.get("total_reward"),
                "episode_length": ep.get("episode_length"),
                "final_distance": ep.get("final_distance"),
            })

        if not episodes_with_frames:
            return None

        try:
            console.print("  Running failure analysis...")
            batch_result = self.failure_analyst.analyze_batch(
                failed_episodes=episodes_with_frames,
                task_description=task_info["description"],
                success_criteria=task_info["success_criteria"],
                env_name=env_name,
                max_episodes=3,
            )
            self.registry.log_agent_action(
                agent_name="FailureAnalyst",
                action="analyze_batch",
                reasoning=batch_result.get("pattern_summary", ""),
                tokens_used=_normalize_token_usage(batch_result.get("total_token_usage", {})),
                model=MODELS["failure_analyst"],
                run_id=run_id,
            )
            summary = batch_result.get("pattern_summary", "")
            fixes = batch_result.get("overall_suggested_fixes", [])
            if fixes:
                summary += "\nSuggested fixes: " + "; ".join(fixes)
            return summary
        except Exception:
            console.print("  [dim]Failure analysis skipped (non-critical error)[/dim]")
            return None

    def _plan_experiment(
        self,
        program: str,
        past_experiments: list[dict],
        failure_analysis: str | None,
        current_env: str,
        current_algorithm: str,
    ) -> tuple[str, dict]:
        parts = [
            f"## Research Program\n{program}",
            f"\n## Current Setup\nEnvironment: {current_env}\nAlgorithm: {current_algorithm}",
        ]

        if past_experiments:
            parts.append("\n## Recent Experiments (most recent last)")
            for exp in past_experiments[-5:]:
                run_id = exp.get("run_id", "?")
                cfg = exp.get("config", {})
                met = exp.get("metrics", {})
                parts.append(
                    f"- {run_id}: algo={cfg.get('algorithm')} env={cfg.get('env_name')} "
                    f"sr={met.get('success_rate', '?')} reward={met.get('mean_reward', '?')} "
                    f"notes={exp.get('notes', '')}"
                )

        if failure_analysis:
            parts.append(f"\n## Latest Failure Analysis\n{failure_analysis}")

        parts.append(
            "\nPropose the next experiment. Be specific about what to change and why. "
            "State your hypothesis clearly."
        )

        user_message = "\n".join(parts)

        try:
            response = self._client.messages.create(
                model=_PLANNING_MODEL,
                max_tokens=1024,
                system=_PLANNING_SYSTEM,
                messages=[{"role": "user", "content": user_message}],
            )
            plan = response.content[0].text
            usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
            return plan, usage
        except anthropic.APIError as exc:
            fallback = (
                f"Continue with {current_algorithm} on {current_env}. "
                f"Try adjusting learning rate slightly. (Planning failed: {exc})"
            )
            return fallback, {"input": 0, "output": 0}

    def _should_graduate_task(
        self,
        recent_experiments: list[dict],
        current_env: str,
    ) -> str | None:
        try:
            current_idx = _CURRICULUM.index(current_env)
        except ValueError:
            return None

        if current_idx >= len(_CURRICULUM) - 1:
            return None

        env_runs = [
            m for m in recent_experiments
            if m.get("config", {}).get("env_name") == current_env
        ]

        if len(env_runs) < _GRADUATION_CONSECUTIVE:
            return None

        last_n = env_runs[-_GRADUATION_CONSECUTIVE:]
        all_above = all(
            float(r.get("metrics", {}).get("success_rate", 0)) >= _GRADUATION_THRESHOLD
            for r in last_n
        )

        if all_above:
            return _CURRICULUM[current_idx + 1]
        return None

    def _should_switch_algorithm(
        self,
        recent_experiments: list[dict],
        current_env: str,
    ) -> str | None:
        env_runs = [
            m for m in recent_experiments
            if m.get("config", {}).get("env_name") == current_env
        ]

        if len(env_runs) < _STALE_THRESHOLD:
            return None

        last_n = env_runs[-_STALE_THRESHOLD:]
        success_rates = [
            float(r.get("metrics", {}).get("success_rate", 0)) for r in last_n
        ]

        best_sr = max(success_rates)
        first_sr = success_rates[0]
        if best_sr <= first_sr:
            current_algo = last_n[-1].get("config", {}).get("algorithm", "SAC")
            alternatives = [a for a in ["SAC", "TD3", "PPO"] if a != current_algo]
            return alternatives[0] if alternatives else None

        return None

    def _find_best_checkpoint(self, env_name: str, algorithm: str) -> dict | None:
        all_meta = self.registry.load_all_metadata()
        matching = [
            m for m in all_meta
            if m.get("config", {}).get("env_name") == env_name
            and m.get("config", {}).get("algorithm") == algorithm
            and Path(m.get("model_path", "")).exists()
        ]
        if not matching:
            return None
        # Use the latest checkpoint — in RL, cumulative training steps matter
        # more than picking the "best" snapshot, since the latest model has
        # the most experience even if its eval was noisy.
        return matching[-1]

    def _read_program(self, program_path: str) -> str:
        path = Path(program_path)
        if not path.exists():
            console.print(f"[yellow]Warning: {program_path} not found, using defaults[/yellow]")
            return "Improve robot manipulation success rate through iterative experimentation."
        return path.read_text()

    def _print_progress_table(self, results: list[dict]) -> None:
        table = Table(title="Recent Experiments")
        table.add_column("Run ID", style="cyan")
        table.add_column("Env")
        table.add_column("Algo")
        table.add_column("Success Rate", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Decision")

        for r in results:
            delta_str = ""
            if r.get("delta") is not None:
                delta_str = f"{r['delta']:+.3f}"

            decision_style = "green" if r["decision"] == "keep" else "red"
            table.add_row(
                r["run_id"],
                r["env"],
                r["algorithm"],
                f"{r['success_rate']:.3f}",
                delta_str,
                f"[{decision_style}]{r['decision'].upper()}[/{decision_style}]",
            )

        console.print(table)

    def _print_final_summary(self, results: list[dict]) -> None:
        console.rule("[bold blue]Final Summary[/bold blue]")

        total = len(results)
        kept = sum(1 for r in results if r["decision"] == "keep")
        discarded = total - kept

        console.print(f"Total experiments: {total}")
        console.print(f"Kept: [green]{kept}[/green]  Discarded: [red]{discarded}[/red]")

        if results:
            best = max(results, key=lambda r: r["success_rate"])
            console.print(
                f"Best run: [cyan]{best['run_id']}[/cyan] "
                f"({best['env']}, {best['algorithm']}) "
                f"success_rate={best['success_rate']:.3f}"
            )
        console.rule()
