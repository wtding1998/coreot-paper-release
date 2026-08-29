from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from coreot.results.grid import write_broad_grid_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate broad (tau,alpha) grid results for coreot_constant_tau."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--coreot-grid-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/hiha_dc_coreot_constant_tau_alpha_broad_grid/coreot_constant_tau"
        ),
    )
    parser.add_argument(
        "--uniform-grid-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/hiha_dc_coreot_constant_tau_alpha_broad_grid/uniform_uot_tau"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/HIHA_DC/sensitivity/constant_tau_alpha"),
    )
    parser.add_argument(
        "--main-grid-root",
        type=Path,
        default=Path("results/HIHA_DC/main"),
    )
    parser.add_argument("--condition", default="incomplete_reference")
    parser.add_argument("--candidate-set", default="hiha_harmony30_k100")
    parser.add_argument("--figures", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_broad_grid_results(
        runs_root=args.runs_root,
        coreot_grid_dir=args.coreot_grid_dir,
        uniform_grid_dir=args.uniform_grid_dir,
        output_root=args.output_root,
        main_grid_root=args.main_grid_root,
        condition=args.condition,
        candidate_set=args.candidate_set,
        write_figures=args.figures,
    )
    print(paths.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
