from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from coreot.benchmarks.missing_state import run_missing_state_benchmark_build
from coreot.candidates.runner import run_candidate_cost
from coreot.config.load import load_yaml
from coreot.data.raw_import import run_raw_import
from coreot.embeddings.runner import run_embedding
from coreot.evaluation.runner import run_evaluation
from coreot.preprocessing.runner import run_model_visible_derivation
from coreot.reports.runner import run_report
from coreot.scoring.runner import run_scoring
from coreot.transport.runner import run_transport
from coreot.workflows.orchestrator import (
    WorkflowError,
    get_step_definition,
    ordered_step_names,
    runnable_step_names,
    step_plan,
)
from coreot.workflows.retention import prune_intermediate_artifacts


class CliError(RuntimeError):
    """Raised for user-facing CLI errors."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coreot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("steps", help="List the default pipeline steps.")

    run_parser = subparsers.add_parser("run", help="Run or dry-run one pipeline step.")
    run_parser.add_argument("--step", required=True, choices=runnable_step_names())
    run_parser.add_argument("--config", type=Path, default=None)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config path and report the planned action without executing it.",
    )

    run_all_parser = subparsers.add_parser("run-all", help="Run or dry-run an experiment.")
    run_all_parser.add_argument("--experiment", required=True, choices=["missing_celltype"])
    run_all_parser.add_argument(
        "--through",
        choices=ordered_step_names(),
        default=None,
        help="Stop the plan after this step.",
    )
    run_all_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all config paths and report the planned actions without executing them.",
    )
    run_all_parser.add_argument(
        "--keep-intermediate-artifacts",
        action="store_true",
        help="Keep large upstream run artifacts instead of pruning them after their last consumer.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "steps":
            return _steps()
        if args.command == "run":
            return _run(args.step, args.config, args.dry_run)
        if args.command == "run-all":
            return _run_all(
                args.experiment,
                args.through,
                args.dry_run,
                args.keep_intermediate_artifacts,
            )
    except (CliError, WorkflowError, FileNotFoundError, NotImplementedError) as exc:
        print(f"coreot: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _steps() -> int:
    for step_name in ordered_step_names():
        print(step_name)
    return 0


def _run(step_name: str, config: Path | None, dry_run: bool) -> int:
    definition = get_step_definition(step_name)
    config_path = config if config is not None else definition.config
    _load_existing_config(config_path)

    if dry_run:
        print(f"DRY RUN {definition.name}: config={config_path}")
        return 0

    if definition.name == "raw-import":
        result = run_raw_import(config_path)
        print(f"raw-import: input={result.raw_path}")
        print(f"raw-import: manifest={result.manifest_path}")
        return 0
    if definition.name == "benchmark-build":
        result = run_missing_state_benchmark_build(config_path)
        print(f"benchmark-build: root={result.benchmark_root}")
        print(f"benchmark-build: conditions={','.join(result.conditions)}")
        return 0
    if definition.name == "model-visible-derivation":
        result = run_model_visible_derivation(config_path)
        print(f"model-visible-derivation: root={result.derived_root}")
        print(f"model-visible-derivation: conditions={','.join(result.conditions)}")
        return 0
    if definition.name == "embedding":
        result = run_embedding(config_path)
        print(f"embedding: root={result.embeddings_root}")
        print(f"embedding: conditions={','.join(result.conditions)}")
        print(f"embedding: providers={','.join(result.providers)}")
        return 0
    if definition.name == "candidate-cost":
        result = run_candidate_cost(config_path)
        print(f"candidate-cost: root={result.candidates_root}")
        print(f"candidate-cost: conditions={','.join(result.conditions)}")
        print(f"candidate-cost: providers={','.join(result.providers)}")
        return 0
    if definition.name == "transport":
        result = run_transport(config_path)
        print(f"transport: root={result.transport_root}")
        print(f"transport: conditions={','.join(result.conditions)}")
        print(f"transport: candidate_sets={','.join(result.candidate_sets)}")
        print(f"transport: methods={','.join(result.methods)}")
        return 0
    if definition.name == "score-abstain":
        result = run_scoring(config_path)
        print(f"score-abstain: root={result.scoring_root}")
        print(f"score-abstain: conditions={','.join(result.conditions)}")
        print(f"score-abstain: candidate_sets={','.join(result.candidate_sets)}")
        print(f"score-abstain: methods={','.join(str(method) for method in result.methods)}")
        return 0
    if definition.name == "evaluation":
        result = run_evaluation(config_path)
        print(f"evaluation: root={result.evaluation_root}")
        print(f"evaluation: conditions={','.join(result.conditions)}")
        print(f"evaluation: candidate_sets={','.join(result.candidate_sets)}")
        print(f"evaluation: methods={','.join(str(method) for method in result.methods)}")
        return 0
    if definition.name == "report":
        result = run_report(config_path)
        print(f"report: root={result.reports_root}")
        print(f"report: markdown={result.markdown_path}")
        return 0

    raise NotImplementedError(
        f"Runner for step {definition.name!r} is not implemented yet. "
        "Use --dry-run until the stage has a tested implementation."
    )


def _run_all(
    experiment: str,
    through: str | None,
    dry_run: bool,
    keep_intermediate_artifacts: bool,
) -> int:
    if experiment != "missing_celltype":
        raise CliError(f"Unsupported experiment: {experiment}")

    plan = step_plan(through)
    for definition in plan:
        _load_existing_config(definition.config)

    if dry_run:
        for definition in plan:
            print(f"DRY RUN {definition.name}: config={definition.config}")
        return 0

    implemented_prefix = [
        "raw-import",
        "benchmark-build",
        "model-visible-derivation",
        "embedding",
        "candidate-cost",
        "transport",
        "score-abstain",
        "evaluation",
        "report",
    ]
    requested_steps = [definition.name for definition in plan]
    if requested_steps != implemented_prefix[: len(requested_steps)]:
        raise NotImplementedError(
            "Only raw-import, benchmark-build, model-visible-derivation, embedding, "
            "candidate-cost, transport baselines, baseline score-abstain, baseline evaluation, "
            "and Markdown report generation are implemented for execution."
        )

    for definition in plan:
        exit_code = _run(definition.name, definition.config, dry_run=False)
        if exit_code != 0:
            return exit_code
        if not keep_intermediate_artifacts:
            config = _load_existing_config(definition.config)
            run_id = str(config.get("run_id", ""))
            outputs = config.get("outputs", {})
            if run_id and isinstance(outputs, dict) and isinstance(outputs.get("root"), str):
                run_root = Path(outputs["root"]) / run_id
                prune_intermediate_artifacts(run_root, definition.name)
    return 0


def _load_existing_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return load_yaml(path)
