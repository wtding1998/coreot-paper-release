from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from coreot.results.pbmc_figure3 import (
    COMPARISON_DETECTION_PATH,
    COMPARISON_LABEL_TRANSFER_PATH,
    DOCS_ROOT,
    RAW_DATA_PATH,
    REVIEW_ROOT,
    RESULT_ROOT,
    RUNS_ROOT,
    generate_panel_a,
    generate_panel_b,
    generate_panel_c,
    generate_panel_d,
    generate_panel_e,
    generate_panel_f,
    generate_destination_support_supplement,
)
from coreot.results.pbmc_figure3_composite import (
    generate_main_figure,
    generate_panel_images,
    generate_review_panels,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate standalone panels and the composite PBMC manuscript Figure 3."
        )
    )
    parser.add_argument(
        "--panel",
        choices=(
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "all",
            "main",
            "review",
            "supp-destination",
        ),
        default="a",
    )
    parser.add_argument("--docs-root", type=Path, default=DOCS_ROOT)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=COMPARISON_DETECTION_PATH,
    )
    parser.add_argument(
        "--label-transfer-path",
        type=Path,
        default=COMPARISON_LABEL_TRANSFER_PATH,
    )
    return parser


def _generate_panel_source(
    panel: str,
    args: argparse.Namespace,
) -> dict[str, Path]:
    common = {
        "docs_root": args.review_root,
        "result_root": args.output_root,
    }
    if panel == "a":
        return generate_panel_a(**common)
    if panel == "b":
        return generate_panel_b(
            **common,
            comparison_path=args.comparison_path,
            raw_data_path=args.raw_data,
            runs_root=args.runs_root,
        )
    if panel == "c":
        return generate_panel_c(
            **common,
            comparison_path=args.comparison_path,
            raw_data_path=args.raw_data,
            runs_root=args.runs_root,
        )
    if panel == "d":
        return generate_panel_d(
            **common,
            label_transfer_path=args.label_transfer_path,
        )
    if panel == "e":
        return generate_panel_e(
            **common,
            comparison_path=args.comparison_path,
            raw_data_path=args.raw_data,
            runs_root=args.runs_root,
        )
    if panel == "f":
        return generate_panel_f(
            **common,
            label_transfer_path=args.label_transfer_path,
            runs_root=args.runs_root,
        )
    if panel == "supp-destination":
        return generate_destination_support_supplement(
            docs_root=args.docs_root,
            result_root=args.output_root,
            comparison_path=args.comparison_path,
            raw_data_path=args.raw_data,
            runs_root=args.runs_root,
        )
    raise ValueError(f"Unsupported panel: {panel}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = args.output_root / "figure_3_pbmc_source_data"
    if args.panel == "main":
        artifacts = generate_main_figure(
            docs_root=args.docs_root,
            result_root=args.output_root,
            source_root=source_root,
        )
        for path in artifacts.values():
            print(path)
        return 0
    if args.panel == "review":
        artifacts = generate_review_panels(
            review_root=args.review_root,
            source_root=source_root,
        )
        for path in artifacts.values():
            print(path)
        return 0
    panels = (
        ("a", "b", "c", "d", "e", "f")
        if args.panel == "all"
        else (args.panel,)
    )
    for panel in panels:
        artifacts = _generate_panel_source(panel, args)
        for path in artifacts.values():
            print(path)
    if args.panel == "all":
        rendered = generate_panel_images(
            panel_root=args.review_root,
            source_root=source_root,
        )
    elif args.panel in {"a", "b", "c", "d", "e", "f"}:
        rendered = generate_panel_images(
            letters=(args.panel,),
            panel_root=args.review_root,
            source_root=source_root,
        )
    else:
        return 0
    for path in rendered.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
