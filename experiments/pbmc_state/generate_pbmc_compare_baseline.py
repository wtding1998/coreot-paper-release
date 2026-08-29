from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coreot.external_baselines.schema import (  # noqa: E402
    DEFAULT_EXTERNAL_BASELINE_METHODS,
)
from coreot.results.compare_baselines import write_compare_baselines_results  # noqa: E402
from experiments.pbmc_state.generate_pbmc_coreot_full_tau_min_tau_max_labelwise_configs import (  # noqa: E402
    DEFAULT_HELD_OUT_ALPHA,
    DEFAULT_SEEDS,
    DEFAULT_TAU_TARGET,
    _slug,
    _value_slug,
)

DEFAULT_SELECTED_TAU_RANGES: dict[str, tuple[float, float]] = {
    "B cells": (0.5, 1.0),
    "NK cells": (0.5, 1.0),
    "Dendritic cells": (0.75, 1.0),
    "CD8 T cells": (0.5, 1.5),
}
PBMC_EXTERNAL_METHODS = DEFAULT_EXTERNAL_BASELINE_METHODS
PBMC_INTERNAL_BASELINES: tuple[dict[str, object], ...] = (
    {"name": "prior_only"},
    {"name": "nn"},
    {
        "name": "uniform_uot",
        "epsilon": 0.05,
        "tau_source": 1.0,
        "tau_target": 1.0,
        "max_iter": 2000,
        "tol": 1.0e-6,
        "numerical_floor": 1.0e-300,
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the selected PBMC coreot_full and external-baseline "
            "results, then write merged comparison tables."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--source-coreot-grid-dir",
        type=Path,
        default=Path(
            "experiments/pbmc_state/generated_configs/"
            "pbmc_coreot_full_tau_min_tau_max_labelwise_grid"
        ),
        help="PBMC coreot_full tau-range grid containing the selected run configs.",
    )
    parser.add_argument(
        "--selected-coreot-grid-dir",
        type=Path,
        default=Path(
            "experiments/pbmc_state/generated_configs/"
            "pbmc_coreot_full_compare_baseline"
        ),
        help="Generated selected-grid view used for compare-baseline summaries.",
    )
    parser.add_argument(
        "--external-grid-dir",
        type=Path,
        default=Path("experiments/pbmc_state/generated_configs/pbmc_ifnb_stim_state_main_grid"),
        help="PBMC source grid containing benchmark.yaml per external-baseline run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/PBMC/compare_baselines"),
    )
    parser.add_argument("--condition", default="incomplete_reference")
    parser.add_argument("--candidate-set", default="pca30_k100")
    parser.add_argument("--embedding-name", default="pca30")
    parser.add_argument("--tau-target", type=float, default=DEFAULT_TAU_TARGET)
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help=(
            "Allow the internal selected-grid bundle to summarize completed "
            "runs only. External-baseline generation still requires complete "
            "selected PBMC external-baseline artifacts."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_grid_dir = sync_selected_coreot_grid(
        source_grid_dir=args.source_coreot_grid_dir,
        selected_grid_dir=args.selected_coreot_grid_dir,
        tau_ranges=DEFAULT_SELECTED_TAU_RANGES,
    )
    paths = write_compare_baselines_results(
        runs_root=args.runs_root,
        grid_dir=selected_grid_dir,
        external_grid_dir=args.external_grid_dir,
        output_root=args.output_root,
        condition=args.condition,
        internal_candidate_set=args.candidate_set,
        embedding_name=args.embedding_name,
        internal_methods=(
            "prior_only",
            "nn",
            "uniform_uot",
            "coreot_full",
            "coreot_match_only",
        ),
        internal_score_overrides={},
        expected_external_held_out_labels=tuple(DEFAULT_HELD_OUT_ALPHA),
        expected_external_seeds=DEFAULT_SEEDS,
        external_methods=PBMC_EXTERNAL_METHODS,
        external_report_title="External Baselines Report - PBMC stimulated-state",
        external_report_description=(
            "External reference-mapping baselines evaluated on selected "
            "IFN-beta PBMC condition-specific held-out-state run roots."
        ),
        summary_title="PBMC Baseline Comparison",
        summary_description=(
            "This bundle regenerates the label-specific selected PBMC coreot_full "
            "operating points and the external reference-mapping baseline report "
            "from run artifacts, then writes merged comparison tables."
        ),
        skip_incomplete=args.skip_incomplete,
    )
    print(paths.output_root)
    return 0


def sync_selected_coreot_grid(
    *,
    source_grid_dir: Path,
    selected_grid_dir: Path,
    tau_ranges: dict[str, tuple[float, float]],
) -> Path:
    if not source_grid_dir.is_dir():
        raise FileNotFoundError(f"Source coreot_full grid does not exist: {source_grid_dir}")

    expected_run_ids = set(_selected_coreot_run_ids(tau_ranges=tau_ranges))
    selected_grid_dir.mkdir(parents=True, exist_ok=True)
    for child in selected_grid_dir.iterdir():
        if child.is_dir() and child.name not in expected_run_ids:
            shutil.rmtree(child)

    for run_id in sorted(expected_run_ids):
        source_run_dir = source_grid_dir / run_id
        if not source_run_dir.is_dir():
            raise FileNotFoundError(
                f"Selected coreot_full config directory is missing: {source_run_dir}"
            )
        shutil.copytree(source_run_dir, selected_grid_dir / run_id, dirs_exist_ok=True)
        _add_match_only_method(selected_grid_dir / run_id)
        _add_internal_baselines(selected_grid_dir / run_id)
    return selected_grid_dir


def _add_match_only_method(run_dir: Path) -> None:
    for config_name in ("transport.yaml", "scoring.yaml", "evaluation.yaml"):
        path = run_dir / config_name
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        methods = payload.get("methods", [])
        methods = [
            method
            for method in methods
            if not (
                method == "coreot_match_only"
                or isinstance(method, dict)
                and method.get("name") == "coreot_match_only"
            )
        ]
        full_index = next(
            index
            for index, method in enumerate(methods)
            if method == "coreot_full"
            or isinstance(method, dict) and method.get("name") == "coreot_full"
        )
        if config_name == "transport.yaml":
            full_method = methods[full_index]
            if not isinstance(full_method, dict):
                raise ValueError(f"Expected mapped coreot_full config in {path}")
            match_only = copy.deepcopy(full_method)
            match_only["name"] = "coreot_match_only"
            match_only["alpha"] = 0.0
        else:
            match_only = "coreot_match_only"
        methods.insert(full_index + 1, match_only)
        payload["methods"] = methods
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _add_internal_baselines(run_dir: Path) -> None:
    baseline_names = tuple(method["name"] for method in PBMC_INTERNAL_BASELINES)
    for config_name in ("transport.yaml", "scoring.yaml", "evaluation.yaml"):
        path = run_dir / config_name
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        methods = [
            method
            for method in payload.get("methods", [])
            if not (
                method in baseline_names
                if isinstance(method, str)
                else method.get("name") in baseline_names
            )
        ]
        baselines: list[object]
        if config_name == "transport.yaml":
            baselines = [copy.deepcopy(method) for method in PBMC_INTERNAL_BASELINES]
        else:
            baselines = list(baseline_names)
        payload["methods"] = methods + baselines
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _selected_coreot_run_ids(
    *, tau_ranges: dict[str, tuple[float, float]]
) -> tuple[str, ...]:
    run_ids: list[str] = []
    for held_out_label, alpha in DEFAULT_HELD_OUT_ALPHA.items():
        try:
            tau_min, tau_max = tau_ranges[held_out_label]
        except KeyError as exc:
            raise ValueError(f"No selected tau range for {held_out_label!r}") from exc
        tau_min_slug = _value_slug(tau_min)
        tau_max_slug = _value_slug(tau_max)
        label_slug = _slug(held_out_label)
        alpha_slug = _value_slug(alpha)
        for seed in DEFAULT_SEEDS:
            run_ids.append(
                f"pbmc_ifnb_{label_slug}_stim_seed{seed}"
                f"_taumin{tau_min_slug}_taumax{tau_max_slug}"
                f"_alpha{alpha_slug}_coreot_full"
            )
    return tuple(run_ids)


if __name__ == "__main__":
    raise SystemExit(main())
