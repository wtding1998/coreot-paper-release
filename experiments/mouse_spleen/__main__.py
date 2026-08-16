from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from experiments.mouse_spleen.pipeline import MouseSpleenConfigError, run_stage


STAGES = (
    "validate_data",
    "prepare_embedding",
    "prepare_task_families",
    "run_controlled",
    "prepare_baselines",
    "run_internal_baselines",
    "run_external_baselines",
    "run_baselines",
    "aggregate_baselines",
    "run_natural_mismatch",
    "aggregate_natural_mismatch",
    "make_natural_mismatch_figure",
    "prepare_natural_baselines",
    "run_natural_external_baselines",
    "aggregate_natural_baselines",
    "run_natural_constant_tau_sensitivity",
    "aggregate_natural_constant_tau_sensitivity",
    "aggregate_natural_constant_tau_target8_sensitivity",
    "run_natural_match_only_sensitivity",
    "aggregate_natural_match_only_sensitivity",
    "aggregate_natural_match_only_target8_sensitivity",
    "run_natural_full_tau_range_sensitivity",
    "aggregate_natural_full_tau_range_sensitivity",
    "run_natural_component_ablation",
    "aggregate_natural_component_ablation",
    "aggregate_metrics",
    "make_figures",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.mouse_spleen")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_stage(args.config, args.stage)
    except (MouseSpleenConfigError, FileNotFoundError, ValueError) as exc:
        print(f"mouse-spleen: {exc}", file=sys.stderr)
        return 2
    print(f"mouse-spleen: stage={result.stage}")
    print(f"mouse-spleen: root={result.root}")
    print(f"mouse-spleen: manifest={result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
