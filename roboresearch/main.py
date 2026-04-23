from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from roboresearch.agents.orchestrator import Orchestrator

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="roboresearch",
        description="RoboResearch — Autonomous robot manipulation research loop",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=50,
        help="Maximum number of experiments to run (default: 50)",
    )
    parser.add_argument(
        "--program",
        type=str,
        default="program.md",
        help="Path to the research program file (default: program.md)",
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        default="registry",
        help="Path to the registry directory (default: registry)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=300,
        help="Training time budget in seconds per experiment (default: 300)",
    )
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold blue]RoboResearch[/bold blue]\n"
            "[dim]Autonomous robot manipulation research through iterative experimentation[/dim]\n"
            "\n"
            f"Program:         {args.program}\n"
            f"Registry:        {args.registry_dir}\n"
            f"Max experiments: {args.max_experiments}\n"
            f"Time budget:     {args.time_budget}s per run",
            title="[bold]v0.1.0[/bold]",
            border_style="blue",
        )
    )

    orchestrator = Orchestrator(
        registry_dir=args.registry_dir,
    )

    try:
        orchestrator.run(
            program_path=args.program,
            max_experiments=args.max_experiments,
            time_budget_seconds=args.time_budget,
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        console.print("Printing summary of experiments completed so far...\n")
        recent = orchestrator.registry.load_all_metadata()
        if recent:
            best = max(
                recent,
                key=lambda m: float(m.get("metrics", {}).get("success_rate", 0)),
            )
            console.print(f"Total runs in registry: {len(recent)}")
            console.print(
                f"Best run: {best['run_id']} "
                f"(success_rate={best['metrics'].get('success_rate', 0):.3f})"
            )
        else:
            console.print("No experiments completed yet.")
        sys.exit(0)


if __name__ == "__main__":
    main()
