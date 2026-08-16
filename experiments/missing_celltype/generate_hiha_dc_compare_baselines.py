from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from coreot.results.compare_baselines import write_compare_baselines_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the selected HIHA DC internal-method and external-baseline "
            "results, then write merged comparison tables."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/"
            "report_leave_one_HIHA_DC"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/HIHA_DC/compare_baselines"),
    )
    parser.add_argument(
        "--uniform-uot-grid-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/"
            "hiha_dc_uniform_uot_tau05"
        ),
        help=(
            "Completed grid providing the canonical uniform-UOT rows with "
            "tau_source=tau_target=0.5."
        ),
    )
    parser.add_argument("--condition", default="incomplete_reference")
    parser.add_argument("--internal-candidate-set", default="hiha_harmony30_k100")
    parser.add_argument("--embedding-name", default="hiha_harmony30")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help=(
            "Allow the internal selected-report bundle to summarize completed "
            "runs only. External-baseline generation still requires the complete "
            "selected-report grid."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_compare_baselines_results(
        runs_root=args.runs_root,
        grid_dir=args.grid_dir,
        output_root=args.output_root,
        condition=args.condition,
        internal_candidate_set=args.internal_candidate_set,
        embedding_name=args.embedding_name,
        internal_method_grid_dirs={"uniform_uot": args.uniform_uot_grid_dir},
        internal_score_overrides={},
        skip_incomplete=args.skip_incomplete,
    )
    print(paths.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
