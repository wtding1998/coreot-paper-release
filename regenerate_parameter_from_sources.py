from __future__ import annotations

import argparse
from pathlib import Path

from coreot.results.hiha_parameter_sensitivity import (
    collect_hiha_parameter_sensitivity,
    render_hiha_parameter_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    by_split, summary = collect_hiha_parameter_sensitivity(
        hla_within_cdc2_detection_path=(
            args.source_root / "hla_detection_within_cdc2_by_run.csv"
        ),
        hla_transfer_path=args.source_root / "hla_shared_label_transfer_by_run.csv",
        isg_path=args.source_root / "isg_discovery_by_split.csv",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    by_split.to_csv(
        args.output_root / "hiha_parameter_sensitivity_by_split.csv", index=False
    )
    summary.to_csv(
        args.output_root / "hiha_parameter_sensitivity_summary.csv", index=False
    )
    render_hiha_parameter_sensitivity(
        summary,
        {
            suffix: args.output_root
            / f"manuscript_fig_hiha_supp_parameter_sensitivity.{suffix}"
            for suffix in ("png", "pdf", "svg")
        },
    )


if __name__ == "__main__":
    main()
