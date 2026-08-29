from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_NAMES = (
    "raw_import",
    "benchmark",
    "derivation",
    "embedding",
    "candidates",
    "transport",
    "scoring",
    "evaluation",
    "report",
)

DEFAULT_HELD_OUT_CELL_TYPES = (
    "CD14+ Monocytes",
    "CD4 T cells",
    "B cells",
    "NK cells",
    "Dendritic cells",
    "CD8 T cells",
    "FCGR3A+ Monocytes",
)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
REMOVED_STATE_CONDITION = "stim"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate broad full-sweep configs for PBMC coreot_constant_tau and uniform_uot."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments/pbmc_state/configs"),
    )
    parser.add_argument(
        "--coreot-output-dir",
        type=Path,
        default=Path(
            "experiments/pbmc_state/generated_configs/pbmc_coreot_constant_tau_alpha_broad_grid/coreot_constant_tau"
        ),
    )
    parser.add_argument(
        "--uniform-output-dir",
        type=Path,
        default=Path(
            "experiments/pbmc_state/generated_configs/pbmc_coreot_constant_tau_alpha_broad_grid/uniform_uot_tau"
        ),
    )
    parser.add_argument("--held-out-cell-type", action="append", default=None)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tau_min > args.tau_max:
        raise SystemExit("--tau-min must be <= --tau-max.")
    if args.alpha_min > args.alpha_max:
        raise SystemExit("--alpha-min must be <= --alpha-max.")

    base_configs = _read_base_configs(args.base_dir)
    held_out_cell_types = tuple(args.held_out_cell_type or DEFAULT_HELD_OUT_CELL_TYPES)
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

    for cell_type in held_out_cell_types:
        slug = _slug(cell_type)
        for seed in seeds:
            for tau in tau_values:
                uniform_run_id = (
                    f"pbmc_ifnb_{slug}_stim_seed{seed}_tau{tau}_alpha0_uniform"
                )
                _write_run_configs(
                    base_configs=base_configs,
                    output_dir=args.uniform_output_dir,
                    cell_type=cell_type,
                    seed=seed,
                    run_id=uniform_run_id,
                    methods=("uniform_uot", "prior_only"),
                    overrides={"tau_source": float(tau), "tau_target": float(tau)},
                )
            for tau in tau_values:
                for alpha in alpha_values:
                    coreot_run_id = (
                        f"pbmc_ifnb_{slug}_stim_seed{seed}"
                        f"_tau{tau}_alpha{alpha}_coreot_constant_tau"
                    )
                    _write_run_configs(
                        base_configs=base_configs,
                        output_dir=args.coreot_output_dir,
                        cell_type=cell_type,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(label: str) -> str:
    cleaned = label.lower().replace("+", "plus")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned


def _read_base_configs(base_dir: Path) -> dict[str, dict[str, Any]]:
    configs = {}
    for name in CONFIG_NAMES:
        path = base_dir / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Base config does not exist: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Base config must be a mapping: {path}")
        configs[name] = payload
    return configs


def _method_name(entry: object) -> str | None:
    if isinstance(entry, dict):
        name = entry.get("name")
        return str(name) if isinstance(name, str) else None
    if isinstance(entry, str):
        return entry
    return None


def _apply_method_overrides(
    payload: dict[str, Any],
    *,
    method_filter: set[str] | None,
    float_overrides: dict[str, float],
) -> None:
    methods: list[dict[str, Any]] = payload.get("methods", [])

    if method_filter is not None:
        methods = [m for m in methods if _method_name(m) in method_filter]
        payload["methods"] = methods

    for method in methods:
        if not isinstance(method, dict):
            continue
        for key, value in float_overrides.items():
            if key in method:
                method[key] = value


def _write_run_configs(
    *,
    base_configs: dict[str, dict[str, object]],
    output_dir: Path,
    cell_type: str,
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
            payload["removed_state"] = cell_type
            payload["removed_state_condition"] = REMOVED_STATE_CONDITION
            payload.setdefault("split", {})["seed"] = seed
        if name in ("transport", "scoring", "evaluation"):
            _apply_method_overrides(
                payload,
                method_filter=set(methods),
                float_overrides=overrides,
            )
        path = run_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
