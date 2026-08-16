from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coreot.results.pbmc_s7_calibration import (  # noqa: E402
    write_pbmc_s7_calibration_figure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PBMC Supplementary Figure S7."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(
            "results/PBMC/figures/data/figure_3_abstention_tradeoff_by_seed.csv"
        ),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path(
            "experiments/pbmc_state/generated_configs/"
            "pbmc_coreot_full_compare_baseline"
        ),
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/kang_2018.h5ad"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/PBMC/figures"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_pbmc_s7_calibration_figure(
        manifest_path=args.manifest_path,
        runs_root=args.runs_root,
        config_root=args.config_root,
        raw_path=args.raw_path,
        output_root=args.output_root,
    )
    print(paths.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
