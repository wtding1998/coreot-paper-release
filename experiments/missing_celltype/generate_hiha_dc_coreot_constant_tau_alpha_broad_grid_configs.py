from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.missing_celltype.generate_grid_configs import (  # noqa: E402
    CONFIG_NAMES,
    DEFAULT_SEEDS,
    _apply_method_overrides,
    _held_out_slug,
    _read_base_configs,
)

DEFAULT_BROAD_HELD_OUT_LABELS = ("HLA-DRhi cDC2", "ISG+ cDC2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate broad full-sweep configs for coreot_constant_tau and uniform_uot."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments/missing_celltype/configs"),
    )
    parser.add_argument(
        "--coreot-output-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/hiha_dc_coreot_constant_tau_alpha_broad_grid/coreot_constant_tau"
        ),
    )
    parser.add_argument(
        "--uniform-output-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/hiha_dc_coreot_constant_tau_alpha_broad_grid/uniform_uot_tau"
        ),
    )
    parser.add_argument("--held-out-label", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--tau-min", type=int, default=1)
    parser.add_argument("--tau-max", type=int, default=9)
    parser.add_argument("--tau-step", type=int, default=1)
    parser.add_argument("--tau-values", type=float, nargs="*", default=None)
    parser.add_argument("--alpha-min", type=int, default=1)
    parser.add_argument("--alpha-max", type=int, default=10)
    parser.add_argument("--alpha-step", type=int, default=1)
    parser.add_argument("--alpha-values", type=float, nargs="*", default=None)
    return parser


def _write_run_configs(
    *,
    base_configs: dict[str, dict[str, object]],
    output_dir: Path,
    held_out_label: str,
    seed: int,
    run_id: str,
    methods: tuple[str, ...],
    overrides: dict[str, float],
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
                method_filter=set(methods),
                float_overrides=overrides,
            )
        path = run_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tau_min > args.tau_max:
        raise SystemExit("--tau-min must be <= --tau-max.")
    if args.alpha_min > args.alpha_max:
        raise SystemExit("--alpha-min must be <= --alpha-max.")

    base_configs = _read_base_configs(args.base_dir)
    held_out_labels = tuple(args.held_out_label or DEFAULT_BROAD_HELD_OUT_LABELS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    tau_values: tuple[float, ...] = (
        tuple(args.tau_values)
        if args.tau_values
        else tuple(range(args.tau_min, args.tau_max + 1, args.tau_step))
    )
    alpha_values: tuple[float, ...] = (
        tuple(args.alpha_values)
        if args.alpha_values
        else tuple(range(args.alpha_min, args.alpha_max + 1, args.alpha_step))
    )

    for held_out_label in held_out_labels:
        slug = _held_out_slug(held_out_label)
        for seed in seeds:
            for tau in tau_values:
                uniform_run_id = f"hiha_dc_{slug}_seed{seed}_tau{tau}_alpha0_uniform"
                _write_run_configs(
                    base_configs=base_configs,
                    output_dir=args.uniform_output_dir,
                    held_out_label=held_out_label,
                    seed=seed,
                    run_id=uniform_run_id,
                    methods=("uniform_uot", "prior_only"),
                    overrides={"tau_source": float(tau), "tau_target": float(tau)},
                )
            for tau in tau_values:
                for alpha in alpha_values:
                    coreot_run_id = (
                        f"hiha_dc_{slug}_seed{seed}_tau{tau}_alpha{alpha}_coreot_constant_tau"
                    )
                    _write_run_configs(
                        base_configs=base_configs,
                        output_dir=args.coreot_output_dir,
                        held_out_label=held_out_label,
                        seed=seed,
                        run_id=coreot_run_id,
                        methods=("coreot_constant_tau", "prior_only"),
                        overrides={
                            "tau_source": float(tau),
                            "tau_target": float(tau),
                            "alpha": float(alpha),
                        },
                    )

    print(args.coreot_output_dir)
    print(args.uniform_output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
