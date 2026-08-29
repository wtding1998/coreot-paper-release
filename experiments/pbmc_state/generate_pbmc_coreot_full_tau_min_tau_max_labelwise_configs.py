from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pbmc_state.generate_pbmc_coreot_constant_tau_alpha_broad_grid_configs import (  # noqa: E402
    CONFIG_NAMES,
    DEFAULT_SEEDS,
    REMOVED_STATE_CONDITION,
    _apply_method_overrides,
    _read_base_configs,
    _slug,
)

DEFAULT_HELD_OUT_ALPHA: dict[str, float] = {
    "B cells": 4.0,
    "NK cells": 2.0,
    "Dendritic cells": 3.0,
    "CD8 T cells": 4.0,
}
DEFAULT_TAU_MIN_VALUES: tuple[float, ...] = (0.5, 0.75, 1.0)
DEFAULT_TAU_MAX_VALUES: tuple[float, ...] = (1.0, 1.25, 1.5)
DEFAULT_TAU_TARGET = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PBMC coreot_full tau_min/tau_max sweep configs using "
            "the label-specific alpha values selected in docs/state_pbmc.md."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments/pbmc_state/configs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/pbmc_state/generated_configs/"
            "pbmc_coreot_full_tau_min_tau_max_labelwise_grid"
        ),
    )
    parser.add_argument("--held-out-cell-type", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--tau-min-values", type=float, nargs="*", default=None)
    parser.add_argument("--tau-max-values", type=float, nargs="*", default=None)
    parser.add_argument("--tau-target", type=float, default=DEFAULT_TAU_TARGET)
    return parser


def _value_slug(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(value).replace(".", "p")


def _write_run_configs(
    *,
    base_configs: dict[str, dict[str, Any]],
    output_dir: Path,
    cell_type: str,
    seed: int,
    run_id: str,
    tau_min: float,
    tau_max: float,
    alpha: float,
    tau_target: float,
) -> None:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_NAMES:
        payload = copy.deepcopy(base_configs[name])
        payload["run_id"] = run_id
        if name == "benchmark":
            payload["removed_state"] = cell_type
            payload["removed_state_condition"] = REMOVED_STATE_CONDITION
            payload.setdefault("split", {})["seed"] = seed
        if name in ("transport", "scoring", "evaluation"):
            _apply_method_overrides(
                payload,
                method_filter={"coreot_full", "prior_only"},
                float_overrides={
                    "tau_min": tau_min,
                    "tau_max": tau_max,
                    "tau_target": tau_target,
                    "alpha": alpha,
                },
            )
        (run_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_configs = _read_base_configs(args.base_dir)
    held_out_cell_types = tuple(args.held_out_cell_type or DEFAULT_HELD_OUT_ALPHA)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    tau_min_values = tuple(args.tau_min_values or DEFAULT_TAU_MIN_VALUES)
    tau_max_values = tuple(args.tau_max_values or DEFAULT_TAU_MAX_VALUES)
    if not tau_min_values:
        raise SystemExit("--tau-min-values must not be empty.")
    if not tau_max_values:
        raise SystemExit("--tau-max-values must not be empty.")
    if args.tau_target <= 0:
        raise SystemExit("--tau-target must be positive.")

    written: list[Path] = []
    for cell_type in held_out_cell_types:
        if cell_type not in DEFAULT_HELD_OUT_ALPHA:
            known = ", ".join(DEFAULT_HELD_OUT_ALPHA)
            raise SystemExit(f"No selected alpha for {cell_type!r}. Known labels: {known}.")
        alpha = DEFAULT_HELD_OUT_ALPHA[cell_type]
        slug = _slug(cell_type)
        for seed in seeds:
            for tau_min in tau_min_values:
                for tau_max in tau_max_values:
                    if tau_min > tau_max:
                        continue
                    run_id = (
                        f"pbmc_ifnb_{slug}_stim_seed{seed}"
                        f"_taumin{_value_slug(tau_min)}"
                        f"_taumax{_value_slug(tau_max)}"
                        f"_alpha{_value_slug(alpha)}_coreot_full"
                    )
                    _write_run_configs(
                        base_configs=base_configs,
                        output_dir=args.output_dir,
                        cell_type=cell_type,
                        seed=seed,
                        run_id=run_id,
                        tau_min=float(tau_min),
                        tau_max=float(tau_max),
                        alpha=float(alpha),
                        tau_target=float(args.tau_target),
                    )
                    written.append(args.output_dir / run_id)

    for run_dir in written:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
