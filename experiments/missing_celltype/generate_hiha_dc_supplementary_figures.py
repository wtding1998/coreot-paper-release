"""Generate the canonical HIHA supplementary figures one at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

from coreot.results.hiha_supplement import (
    write_hiha_s1,
    write_hiha_s2,
    write_hiha_s4,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("s1", "s2", "s4"),
        default="s1",
        help="Supplementary figure to generate.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(
            "data/derived/hiha_dc/"
            "human_immune_health_atlas_dc.with_recomputed_AIFI_L2_score.h5ad"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/HIHA_DC/figures")
    )
    parser.add_argument("--candidate-set", default="hiha_harmony30_k100")
    parser.add_argument("--eta", type=float, default=1.0e-12)
    args = parser.parse_args()

    if args.figure == "s1":
        outputs = write_hiha_s1(
            runs_root=args.runs_root,
            input_path=args.input_path,
            output_root=args.output_root,
            candidate_set=args.candidate_set,
            eta=args.eta,
        )
    elif args.figure == "s2":
        outputs = write_hiha_s2(
            runs_root=args.runs_root,
            input_path=args.input_path,
            output_root=args.output_root,
            candidate_set=args.candidate_set,
        )
    elif args.figure == "s4":
        outputs = write_hiha_s4(output_root=args.output_root)
    else:
        raise SystemExit(
            f"Supplementary Figure {args.figure.upper()} is not implemented yet; "
            "run with --figure s1."
        )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
