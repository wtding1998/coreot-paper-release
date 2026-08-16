from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from coreot.results.hiha_figure2 import (
    generate_panel_a,
    generate_panel_b,
    generate_panel_c,
    generate_panel_d,
    generate_panel_e,
    generate_panel_f,
)
from coreot.results.hiha_figure2_composite import generate_main_figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate panels for the HIHA controlled missing-state Figure 2."
    )
    parser.add_argument(
        "--panel",
        choices=("a", "b", "c", "d", "e", "f", "main"),
        default="a",
    )
    parser.add_argument("--docs-root", type=Path, default=Path("docs"))
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=Path("results/HIHA_DC/manuscript/figure2_panels"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/HIHA_DC/manuscript/figure2"),
    )
    parser.add_argument(
        "--overview",
        type=Path,
        default=Path(
            "results/HIHA_DC/main/experiment_overview/table_1_experiment_overview.csv"
        ),
    )
    parser.add_argument(
        "--detection-table",
        type=Path,
        default=Path(
            "results/HIHA_DC/compare_baselines/tables/compare_detection_by_run.csv"
        ),
    )
    parser.add_argument(
        "--panel-b-detection-table",
        type=Path,
        default=Path("results/HIHA_DC/main/tables/main_detection_by_run.csv"),
    )
    parser.add_argument(
        "--label-transfer-table",
        type=Path,
        default=Path(
            "results/HIHA_DC/compare_baselines/tables/"
            "compare_shared_label_transfer_by_run.csv"
        ),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.panel == "a":
        paths = generate_panel_a(
            panel_root=args.panel_root,
            result_root=args.output_root,
            overview_path=args.overview,
        )
    elif args.panel == "b":
        paths = generate_panel_b(
            panel_root=args.panel_root,
            result_root=args.output_root,
            detection_path=args.panel_b_detection_table,
        )
    elif args.panel == "c":
        paths = generate_panel_c(
            panel_root=args.panel_root,
            result_root=args.output_root,
            detection_path=args.detection_table,
            runs_root=args.runs_root,
        )
    elif args.panel == "d":
        paths = generate_panel_d(
            panel_root=args.panel_root,
            result_root=args.output_root,
            label_transfer_path=args.label_transfer_table,
        )
    elif args.panel == "e":
        paths = generate_panel_e(
            panel_root=args.panel_root,
            result_root=args.output_root,
            detection_path=args.detection_table,
            runs_root=args.runs_root,
        )
    elif args.panel == "f":
        paths = generate_panel_f(
            panel_root=args.panel_root,
            result_root=args.output_root,
            label_transfer_path=args.label_transfer_table,
            runs_root=args.runs_root,
        )
    else:
        paths = generate_main_figure(
            docs_root=args.docs_root,
            result_root=args.output_root,
            source_root=args.output_root / "source_data",
        )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
