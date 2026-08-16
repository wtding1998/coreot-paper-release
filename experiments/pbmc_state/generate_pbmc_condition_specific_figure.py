from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from coreot.results.pbmc_figures import write_pbmc_condition_specific_figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the PBMC condition-specific weak-correspondence figure."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--raw-data", type=Path, default=Path("data/raw/kang_2018.h5ad"))
    parser.add_argument(
        "--comparison-path", type=Path,
        default=Path("results/PBMC/compare_baselines/tables/compare_detection_by_run.csv"),
    )
    parser.add_argument(
        "--shared-transfer-path", type=Path,
        default=Path(
            "results/PBMC/compare_baselines/tables/compare_shared_label_transfer_by_run.csv"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/PBMC/figures"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_pbmc_condition_specific_figure(
        runs_root=args.runs_root,
        raw_data_path=args.raw_data,
        comparison_path=args.comparison_path,
        shared_transfer_path=args.shared_transfer_path,
        output_root=args.output_root,
    )
    print("Figure 3 validation: all figure-data and artist-bound assertions passed")
    print("Figure 3 generation used preserved artifacts; no transport fitting was run")
    print(paths.png)
    print(paths.pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
