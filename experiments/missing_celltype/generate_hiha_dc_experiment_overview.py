from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from coreot.results.experiment_overview import write_experiment_overview_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate experiment overview Table 1 for the HIHA DC "
        "leave-one-cell-type-out benchmark."
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path(
            "data/derived/hiha_dc/"
            "human_immune_health_atlas_dc.with_recomputed_AIFI_L2_score.h5ad"
        ),
        help="Path to the HIHA DC .h5ad file with recomputed AIFI_L2_score.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/HIHA_DC/main/experiment_overview"),
        help="Output directory for table_1_experiment_overview.csv and .md.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_experiment_overview_table(
        raw_data_path=args.raw_data,
        output_root=args.output_root,
    )
    print(f"CSV  : {paths.csv_path}")
    print(f"MD   : {paths.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
