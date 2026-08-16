from __future__ import annotations

import argparse
from pathlib import Path

from coreot.results.hiha import write_hiha_reformulation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate scope-aware HIHA manuscript results from preserved artifacts."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("experiments/missing_celltype/generated_configs/report_leave_one_HIHA_DC"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/HIHA_DC"))
    parser.add_argument(
        "--uniform-grid-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/"
            "hiha_dc_uniform_uot_tau05"
        ),
    )
    args = parser.parse_args()
    for path in write_hiha_reformulation(
        runs_root=args.runs_root,
        grid_dir=args.grid_dir,
        output_root=args.output_root,
        method_grid_dirs={"uniform_uot": args.uniform_grid_dir},
    ).values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
