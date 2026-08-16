from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.component_ablation_surfaces import METRICS, _render_variant, selected_specs


TAU_SOURCE = {1.0, 2.0, 3.0, 4.0, 5.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for endpoint in ("hladrhi_cdc2", "isg_cdc2"):
        for seed in range(1, 6):
            path = args.input_root / endpoint / f"seed{seed}" / "compatibility_only.csv"
            frame = pd.read_csv(path)
            frames.append(frame.loc[frame["tau_source"].isin(TAU_SOURCE)].copy())
    by_seed = pd.concat(frames, ignore_index=True)
    if len(by_seed) != 250 or not by_seed["converged"].astype(bool).all():
        raise ValueError("Expected 250 converged scoped compatibility fits")
    parameter_columns = ["tau_min", "tau_max", "tau_source", "alpha"]
    for column in parameter_columns:
        if column not in by_seed:
            by_seed[column] = np.nan
    groups = ["experiment", "endpoint", "variant", "method", *parameter_columns]
    summary = (
        by_seed.groupby(groups, dropna=False, sort=False)
        .agg(
            n_splits=("seed", "nunique"),
            all_converged=("converged", "all"),
            max_n_iterations=("n_iterations", "max"),
            **{f"{metric}_mean": (metric, "mean") for metric, _ in METRICS},
            **{f"{metric}_sd": (metric, "std") for metric, _ in METRICS},
        )
        .reset_index()
    )
    if len(summary) != 50 or not summary["n_splits"].eq(5).all():
        raise ValueError("Expected 50 complete scoped compatibility cells")
    args.output_root.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(args.output_root / "component_ablation_by_seed.csv", index=False)
    summary.to_csv(args.output_root / "component_ablation_summary.csv", index=False)
    _render_variant(
        experiment="hiha",
        specs=selected_specs("hiha", None),
        summary=summary,
        variant="compatibility_only",
        output_path=args.output_root / "manuscript_fig_hiha_compatibility_sensitivity.png",
    )


if __name__ == "__main__":
    main()
