from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Sequence

import yaml


DEFAULT_HELD_OUT_LABELS = ("CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
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

# Method parameters that accept a float value through CLI overrides.
FLOAT_OVERRIDES = (
    "alpha",
    "epsilon",
    "tau_source",
    "tau_target",
    "tau_min",
    "tau_max",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-run configs for the HIHA DC main held-out-state grid."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments/missing_celltype/configs"),
        help="Directory containing singleton pilot configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/missing_celltype/generated_configs/hiha_dc_main_grid"),
        help="Directory to write generated per-run config directories.",
    )
    parser.add_argument(
        "--held-out-label",
        action="append",
        default=None,
        help="Held-out AIFI_L3 label. May be supplied multiple times.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Donor split seed. May be supplied multiple times.",
    )

    # --- tuning / parameter-sweep arguments ---
    parser.add_argument(
        "--run-id-suffix",
        default="",
        help="Appended to each generated run_id (e.g. '_tau010').",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Transport methods to keep. Omit to keep all methods from the base config.",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None, help="Convenience: sets both tau_source and tau_target.")
    parser.add_argument("--tau-source", type=float, default=None)
    parser.add_argument("--tau-target", type=float, default=None)
    parser.add_argument("--tau-min", type=float, default=None)
    parser.add_argument("--tau-max", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --- resolve --tau convenience flag ---
    if args.tau is not None:
        if args.tau_source is not None and args.tau_source != args.tau:
            raise ValueError(
                f"--tau={args.tau} conflicts with --tau-source={args.tau_source}; "
                "use one or the other."
            )
        if args.tau_target is not None and args.tau_target != args.tau:
            raise ValueError(
                f"--tau={args.tau} conflicts with --tau-target={args.tau_target}; "
                "use one or the other."
            )
        if args.tau_source is None:
            args.tau_source = args.tau
        if args.tau_target is None:
            args.tau_target = args.tau

    # --- guard: require explicit --output-dir when sweep flags are in use ---
    _sweep_flags = (
        "methods",
        "alpha",
        "epsilon",
        "tau_source",
        "tau_target",
        "tau",
        "tau_min",
        "tau_max",
    )
    _any_sweep = any(getattr(args, flag, None) is not None for flag in _sweep_flags)
    _custom_suffix = bool(args.run_id_suffix)
    if (_any_sweep or _custom_suffix) and args.output_dir == Path(
        "experiments/missing_celltype/generated_configs/hiha_dc_main_grid"
    ):
        raise ValueError(
            "Sensitivity/sweep flags detected (e.g. --tau, --alpha, --methods, --run-id-suffix). "
            "Please specify --output-dir explicitly to avoid overwriting the main grid configs."
        )

    held_out_labels = tuple(args.held_out_label or DEFAULT_HELD_OUT_LABELS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    base_configs = _read_base_configs(args.base_dir)

    # Build override dict for transport method parameters.
    float_overrides: dict[str, float] = {}
    for key in FLOAT_OVERRIDES:
        value = getattr(args, key, None)
        if value is not None:
            float_overrides[key] = float(value)

    method_filter: set[str] | None = None
    if args.methods is not None:
        method_filter = set(args.methods)

    written = []
    for held_out_label in held_out_labels:
        for seed in seeds:
            run_id = f"hiha_dc_{_held_out_slug(held_out_label)}_seed{seed}{args.run_id_suffix}"
            run_dir = args.output_dir / run_id
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
                        method_filter=method_filter,
                        float_overrides=float_overrides,
                    )
                path = run_dir / f"{name}.yaml"
                path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            written.append(run_dir)

    for run_dir in written:
        print(run_dir)
    return 0


def _apply_method_overrides(
    payload: dict[str, Any],
    *,
    method_filter: set[str] | None,
    float_overrides: dict[str, float],
) -> None:
    methods: list[dict[str, Any]] = payload.get("methods", [])

    # Filter to requested methods.  Handles both list-of-dicts (transport)
    # and list-of-strings (scoring, evaluation).
    if method_filter is not None:
        methods = [
            m
            for m in methods
            if _method_name(m) in method_filter
        ]
        payload["methods"] = methods

    # Apply float overrides to every method dict (transport only).
    for method in methods:
        if not isinstance(method, dict):
            continue
        for key, value in float_overrides.items():
            if key in method:
                method[key] = value


def _method_name(entry: object) -> str | None:
    """Return the method name from a dict or string entry."""
    if isinstance(entry, dict):
        name = entry.get("name")
        return str(name) if isinstance(name, str) else None
    if isinstance(entry, str):
        return entry
    return None


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


def _held_out_slug(label: str) -> str:
    cleaned = label.lower().replace("+", "").replace("-", "")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if cleaned == "cd14_cdc2":
        return "cd14_cdc2"
    if cleaned == "hla_drhi_cdc2":
        return "hladrhi_cdc2"
    if cleaned == "isg_cdc2":
        return "isg_cdc2"
    return cleaned


if __name__ == "__main__":
    raise SystemExit(main())
