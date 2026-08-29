from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Sequence

import yaml

from experiments.missing_celltype.generate_grid_configs import (
    CONFIG_NAMES,
    DEFAULT_HELD_OUT_LABELS,
    DEFAULT_SEEDS,
    _apply_method_overrides,
    _held_out_slug,
    _read_base_configs,
)


DEFAULT_TAU_MIN_VALUES: tuple[float, ...] = (5.0, 6.0, 7.0)
DEFAULT_TAU_MAX_VALUES: tuple[float, ...] = (7.0, 8.0, 9.0)
DEFAULT_ALPHA = 5.0
DEFAULT_TAU_TARGET = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate HIHA DC coreot_full tau_min/tau_max configs at alpha=5."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments/missing_celltype/configs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/"
            "hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid"
        ),
    )
    parser.add_argument("--held-out-label", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument(
        "--tau-min-values",
        type=float,
        nargs="*",
        default=None,
        help="Values for coreot_full tau_min. Defaults to 5 6 7.",
    )
    parser.add_argument(
        "--tau-max-values",
        type=float,
        nargs="*",
        default=None,
        help="Values for coreot_full tau_max. Defaults to 7 8 9.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--tau-target", type=float, default=DEFAULT_TAU_TARGET)
    return parser


def _value_slug(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(value).replace(".", "p")


def _write_run_configs(
    *,
    base_configs: dict[str, dict[str, object]],
    output_dir: Path,
    held_out_label: str,
    seed: int,
    run_id: str,
    tau_min: float,
    tau_max: float,
    alpha: float,
    tau_target: float = 1.0,
) -> None:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_NAMES:
        payload = copy.deepcopy(base_configs[name])
        payload["run_id"] = run_id
        if name == "benchmark":
            payload["removed_state"] = held_out_label
            payload.setdefault("split", {})["seed"] = seed
        if name in ("transport", "scoring", "evaluation"):
            _apply_method_overrides(
                payload,
                method_filter={"coreot_full", "prior_only"},
                float_overrides={
                    "tau_min": tau_min,
                    "tau_max": tau_max,
                    "alpha": alpha,
                    "tau_target": tau_target,
                },
            )
        (run_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tau_min_values = tuple(args.tau_min_values or DEFAULT_TAU_MIN_VALUES)
    tau_max_values = tuple(args.tau_max_values or DEFAULT_TAU_MAX_VALUES)
    if not tau_min_values:
        raise SystemExit("--tau-min-values must not be empty.")
    if not tau_max_values:
        raise SystemExit("--tau-max-values must not be empty.")
    if args.alpha <= 0:
        raise SystemExit("--alpha must be positive.")

    base_configs = _read_base_configs(args.base_dir)
    held_out_labels = tuple(args.held_out_label or DEFAULT_HELD_OUT_LABELS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)

    written: list[Path] = []
    for held_out_label in held_out_labels:
        slug = _held_out_slug(held_out_label)
        for seed in seeds:
            for tau_min in tau_min_values:
                for tau_max in tau_max_values:
                    if tau_min > tau_max:
                        continue
                    run_id = (
                        f"hiha_dc_{slug}_seed{seed}_taumin{_value_slug(tau_min)}"
                        f"_taumax{_value_slug(tau_max)}_alpha{_value_slug(args.alpha)}"
                        "_coreot_full"
                    )
                    _write_run_configs(
                        base_configs=base_configs,
                        output_dir=args.output_dir,
                        held_out_label=held_out_label,
                        seed=seed,
                        run_id=run_id,
                        tau_min=float(tau_min),
                        tau_max=float(tau_max),
                        alpha=float(args.alpha),
                        tau_target=float(args.tau_target),
                    )
                    written.append(args.output_dir / run_id)

    for run_dir in written:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
