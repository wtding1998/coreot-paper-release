from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.missing_celltype.generate_grid_configs import (  # noqa: E402
    CONFIG_NAMES,
    DEFAULT_SEEDS,
    _held_out_slug,
    _method_name,
    _read_base_configs,
)


DEFAULT_HELD_OUT_LABELS = ("HLA-DRhi cDC2", "ISG+ cDC2")
SELECTED_METHODS = (
    "nn",
    "uniform_uot",
    "coreot_full",
    "coreot_match_only",
    "prior_only",
)


@dataclass(frozen=True)
class SelectedOperatingPoint:
    uniform_tau: float
    coreot_tau_min: float
    coreot_tau_max: float
    coreot_tau_target: float
    coreot_alpha: float


SELECTED_OPERATING_POINTS: dict[str, SelectedOperatingPoint] = {
    "HLA-DRhi cDC2": SelectedOperatingPoint(
        uniform_tau=0.5,
        coreot_tau_min=2.5,
        coreot_tau_max=3.0,
        coreot_tau_target=2.0,
        coreot_alpha=2.0,
    ),
    "ISG+ cDC2": SelectedOperatingPoint(
        uniform_tau=0.5,
        coreot_tau_min=0.5,
        coreot_tau_max=0.625,
        coreot_tau_target=1.0,
        coreot_alpha=0.25,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate selected HIHA DC report_leave_one_HIHA_DC configs containing "
            "baseline, uniform UOT, and full CoRe-OT methods in each run."
        )
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
            "experiments/missing_celltype/generated_configs/report_leave_one_HIHA_DC"
        ),
    )
    parser.add_argument("--held-out-label", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument(
        "--run-id-suffix",
        default="_report_leave_one_HIHA_DC",
        help="Appended to each generated run_id.",
    )
    parser.add_argument(
        "--uniform-uot-only",
        action="store_true",
        help=(
            "Generate the uniform-UOT execution slice, retaining prior_only as "
            "its scoring dependency."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_configs = _read_base_configs(args.base_dir)
    held_out_labels = tuple(args.held_out_label or DEFAULT_HELD_OUT_LABELS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    selected_methods = (
        ("uniform_uot", "prior_only") if args.uniform_uot_only else SELECTED_METHODS
    )

    unknown_labels = sorted(set(held_out_labels) - set(SELECTED_OPERATING_POINTS))
    if unknown_labels:
        raise SystemExit(
            "No selected operating point is defined for held-out label(s): "
            + ", ".join(unknown_labels)
        )

    written: list[Path] = []
    for held_out_label in held_out_labels:
        operating_point = SELECTED_OPERATING_POINTS[held_out_label]
        slug = _held_out_slug(held_out_label)
        for seed in seeds:
            run_id = f"hiha_dc_{slug}_seed{seed}{args.run_id_suffix}"
            _write_run_configs(
                base_configs=base_configs,
                output_dir=args.output_dir,
                held_out_label=held_out_label,
                seed=seed,
                run_id=run_id,
                operating_point=operating_point,
                selected_methods=selected_methods,
            )
            written.append(args.output_dir / run_id)

    for run_dir in written:
        print(run_dir)
    return 0


def _write_run_configs(
    *,
    base_configs: dict[str, dict[str, object]],
    output_dir: Path,
    held_out_label: str,
    seed: int,
    run_id: str,
    operating_point: SelectedOperatingPoint,
    selected_methods: tuple[str, ...] = SELECTED_METHODS,
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
            _apply_selected_methods(payload, operating_point, selected_methods)
        (run_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


def _apply_selected_methods(
    payload: dict[str, object],
    operating_point: SelectedOperatingPoint,
    selected_methods: tuple[str, ...] = SELECTED_METHODS,
) -> None:
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        raise ValueError("Config payload field 'methods' must be a list.")

    filtered_methods = [
        method for method in methods if _method_name(method) in selected_methods
    ]
    full_method = next(
        (method for method in filtered_methods if _method_name(method) == "coreot_full"),
        None,
    )
    if isinstance(full_method, dict):
        match_only = copy.deepcopy(full_method)
        match_only["name"] = "coreot_match_only"
        match_only["alpha"] = 0.0
        filtered_methods.insert(filtered_methods.index(full_method) + 1, match_only)
    elif full_method is not None:
        filtered_methods.insert(
            filtered_methods.index(full_method) + 1, "coreot_match_only"
        )
    payload["methods"] = filtered_methods

    for method in filtered_methods:
        if not isinstance(method, dict):
            continue
        name = method.get("name")
        if name == "uniform_uot":
            method["tau_source"] = operating_point.uniform_tau
            method["tau_target"] = operating_point.uniform_tau
        elif name in {"coreot_full", "coreot_match_only"}:
            method["tau_min"] = operating_point.coreot_tau_min
            method["tau_max"] = operating_point.coreot_tau_max
            method["alpha"] = (
                operating_point.coreot_alpha if name == "coreot_full" else 0.0
            )
            method["tau_target"] = operating_point.coreot_tau_target


if __name__ == "__main__":
    raise SystemExit(main())
