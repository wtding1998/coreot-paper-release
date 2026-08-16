from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from coreot.artifacts.hashes import sha256_file


METHOD = "coreot_full"
CONDITION = "natural_mismatch"
N_TARGET_INTERVALS = 20
SELECTED_PARAMETERS = {
    "tau_min": 3.0,
    "tau_max": 5.0,
    "tau_target": 8.0,
    "alpha": 40.0,
    "epsilon": 0.05,
}
SCORES_RELATIVE = Path(
    "results/mouse_spleen_core_ot/runs/"
    "mouse_spleen_natural_proliferating/scoring/natural_mismatch/"
    "mouse_spleen_provider_k100/cell_scores.parquet"
)
PRIORS_RELATIVE = Path(
    "results/mouse_spleen_core_ot/runs/"
    "mouse_spleen_natural_proliferating/derived/natural_mismatch/"
    "prior_profiles/default/source_priors.csv"
)
PARAMETERS_RELATIVE = Path(
    "results/mouse_spleen_core_ot/runs/"
    "mouse_spleen_natural_proliferating/transport/natural_mismatch/"
    "mouse_spleen_provider_k100/coreot_full/method_params.yaml"
)
TRANSPORT_MANIFEST_RELATIVE = Path(
    "results/mouse_spleen_core_ot/runs/"
    "mouse_spleen_natural_proliferating/transport/natural_mismatch/"
    "mouse_spleen_provider_k100/coreot_full/transport_manifest.yaml"
)
TRUTH_RELATIVE = Path(
    "results/mouse_spleen_core_ot/runs/"
    "mouse_spleen_natural_proliferating/benchmark/natural_mismatch/"
    "evaluation_truth/query_truth.csv"
)


def _require_columns(
    frame: pd.DataFrame, required: set[str], path: Path
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} omits required columns: {missing}")


def load_selected_scores(project_root: Path) -> pd.DataFrame:
    scores_path = project_root / SCORES_RELATIVE
    priors_path = project_root / PRIORS_RELATIVE
    parameters_path = project_root / PARAMETERS_RELATIVE
    transport_manifest_path = project_root / TRANSPORT_MANIFEST_RELATIVE
    truth_path = project_root / TRUTH_RELATIVE
    for path in (
        scores_path,
        priors_path,
        parameters_path,
        transport_manifest_path,
        truth_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected-fit input: {path}")

    parameters = yaml.safe_load(parameters_path.read_text(encoding="utf-8"))
    for name, expected in SELECTED_PARAMETERS.items():
        observed = float(parameters[name])
        if not np.isclose(observed, expected, atol=0.0, rtol=0.0):
            raise ValueError(
                f"Selected-fit parameter mismatch for {name}: "
                f"expected {expected}, observed {observed}"
            )
    transport_manifest = yaml.safe_load(
        transport_manifest_path.read_text(encoding="utf-8")
    )
    if transport_manifest.get("metadata", {}).get("converged") is not True:
        raise ValueError("Selected transport is not recorded as converged")

    scores = pd.read_parquet(scores_path)
    _require_columns(
        scores,
        {"cell_id", "condition_id", "method", "u", "prior_risk"},
        scores_path,
    )
    scores = scores.loc[
        scores["condition_id"].astype(str).eq(CONDITION)
        & scores["method"].astype(str).eq(METHOD),
        ["cell_id", "u", "prior_risk"],
    ].copy()
    priors = pd.read_csv(priors_path)
    _require_columns(priors, {"cell_id", "prior_risk"}, priors_path)
    priors = priors.loc[:, ["cell_id", "prior_risk"]].copy()
    truth = pd.read_csv(truth_path, usecols=["cell_id"])
    if (
        scores["cell_id"].duplicated().any()
        or priors["cell_id"].duplicated().any()
        or truth["cell_id"].duplicated().any()
    ):
        raise ValueError("Selected-fit diagnostic inputs contain duplicate cell IDs")
    if set(scores["cell_id"]) != set(truth["cell_id"]):
        raise ValueError("Selected-fit score rows do not cover the all-query cohort")

    merged = scores.merge(
        priors,
        on="cell_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_score", "_prior"),
    )
    if len(merged) != len(scores) or len(merged) != len(priors):
        raise ValueError("Selected-fit scores and prior profiles have different cell sets")
    if not np.allclose(
        merged["prior_risk_score"],
        merged["prior_risk_prior"],
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("Score-table prior risk disagrees with the saved prior profile")

    frame = merged.rename(
        columns={"prior_risk_score": "prior_risk"}
    ).loc[:, ["cell_id", "u", "prior_risk"]]
    finite = np.isfinite(frame["u"]) & np.isfinite(frame["prior_risk"])
    if not finite.all():
        raise ValueError("Selected-fit prior diagnostic contains nonfinite values")
    return frame.sort_values("cell_id", kind="mergesort").reset_index(drop=True)


def equal_frequency_bins(
    frame: pd.DataFrame, n_target_intervals: int = N_TARGET_INTERVALS
) -> pd.DataFrame:
    _require_columns(frame, {"cell_id", "u", "prior_risk"}, Path("selected_scores"))
    if len(frame) < 2:
        raise ValueError("Prior diagnostic requires at least two cells")
    average_rank = frame["prior_risk"].rank(method="average")
    prior_bin = np.minimum(
        (
            np.floor(
                (average_rank.to_numpy(dtype=float) - 1.0)
                * n_target_intervals
                / len(frame)
            ).astype(int)
            + 1
        ),
        n_target_intervals,
    )
    working = frame.assign(prior_bin=prior_bin)
    bins = (
        working.groupby("prior_bin", as_index=False)
        .agg(
            n_cells=("cell_id", "size"),
            mean_prior_risk=("prior_risk", "mean"),
            mean_u=("u", "mean"),
        )
        .sort_values("prior_bin", kind="mergesort")
        .reset_index(drop=True)
    )
    bins.insert(1, "n_target_intervals", n_target_intervals)
    bins.insert(2, "n_occupied_intervals", len(bins))
    return bins


def correlation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    prior_rank = frame["prior_risk"].rank(method="average")
    deficit_rank = frame["u"].rank(method="average")
    return pd.DataFrame(
        [
            {
                "metric": "u_vs_prior_risk",
                "spearman": float(prior_rank.corr(deficit_rank)),
                "pearson": float(frame["prior_risk"].corr(frame["u"])),
                "n_cells": len(frame),
                "n_excluded": 0,
            }
        ]
    )


def render_figure(
    frame: pd.DataFrame,
    bins: pd.DataFrame,
    correlation: pd.DataFrame,
    output_path: Path,
) -> None:
    text_color = "#2D2926"
    split_color = "#B1AAA4"
    mean_color = "#D55E00"
    zero_color = "#6F655E"
    grid_color = "#ECE6E1"
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "text.color": text_color,
            "axes.labelcolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
            "axes.edgecolor": text_color,
            "axes.linewidth": 0.6,
        }
    )
    figure = plt.figure(figsize=(183 / 25.4, 62 / 25.4), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(3.2, 1.0),
        left=0.08,
        right=0.98,
        bottom=0.23,
        top=0.88,
        wspace=0.32,
    )
    curve_axis = figure.add_subplot(grid[0, 0])
    curve_axis.scatter(
        frame["prior_risk"],
        frame["u"],
        s=3,
        color=split_color,
        alpha=0.28,
        edgecolor="none",
        rasterized=True,
        label="Query cell",
    )
    curve_axis.plot(
        bins["mean_prior_risk"],
        bins["mean_u"],
        color=mean_color,
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Occupied-interval mean",
    )
    curve_axis.set_xlim(0.44, 0.56)
    curve_axis.set_ylim(0.0, 1.0)
    curve_axis.set_xlabel(r"Prior risk $r_{q,i}=1-\rho_{q,i}$")
    curve_axis.set_ylabel(r"Query-marginal deficit $u_{q,i}$")
    curve_axis.set_title("Selected CoRe-OT fit", weight="bold", fontsize=7)
    curve_axis.grid(color=grid_color, linewidth=0.35)
    curve_axis.spines[["top", "right"]].set_visible(False)
    curve_axis.legend(frameon=False, fontsize=6, loc="upper left")
    curve_axis.text(
        -0.08,
        1.03,
        "A",
        transform=curve_axis.transAxes,
        fontsize=8,
        weight="bold",
    )

    correlation_axis = figure.add_subplot(grid[0, 1])
    spearman = float(correlation["spearman"].iloc[0])
    correlation_axis.axvline(0.0, color=zero_color, linewidth=0.8)
    correlation_axis.scatter(
        [spearman],
        [0.0],
        marker="D",
        s=28,
        color=mean_color,
        edgecolor=text_color,
        linewidth=0.45,
        zorder=3,
    )
    correlation_axis.set_xlim(-1.0, 1.0)
    correlation_axis.set_ylim(-0.5, 0.5)
    correlation_axis.set_yticks([0.0], [r"$u$ vs prior risk"])
    correlation_axis.set_xlabel("Spearman correlation")
    correlation_axis.set_title("Rank association", weight="bold", fontsize=7)
    correlation_axis.grid(axis="x", color=grid_color, linewidth=0.35)
    correlation_axis.spines[["top", "right"]].set_visible(False)
    correlation_axis.text(
        -0.18,
        1.03,
        "B",
        transform=correlation_axis.transAxes,
        fontsize=8,
        weight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=450)
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".svg"))
    plt.close(figure)


def caption_text(n_cells: int, n_occupied_intervals: int) -> str:
    return (
        "**Association of query-marginal deficit with prior risk in the "
        "mouse-spleen application.** CoRe-OT was evaluated at the selected "
        f"operating point over all {n_cells:,} RNA query cells. **(A)** Gray points "
        "show cell-level deficit against prior risk. Orange points show means "
        "in 20 target equal-frequency intervals constructed from average "
        "prior-risk ranks; tied values remain together, leaving "
        f"{n_occupied_intervals} occupied "
        "intervals. **(B)** The diamond shows the average-rank Spearman "
        "correlation using all finite cell-level pairs; the vertical line "
        "marks zero. Pearson correlation is provided in Supplementary Data 2. "
        "The display is a descriptive within-fit diagnostic from one dataset."
    )


def generate(project_root: Path) -> dict[str, Path]:
    frame = load_selected_scores(project_root)
    bins = equal_frequency_bins(frame)
    correlation = correlation_summary(frame)
    output_root = (
        project_root
        / "results/mouse_spleen_core_ot/manuscript/prior_dependence"
    )
    figure_path = (
        project_root
        / "docs/figs/manuscript_fig_mouse_spleen_prior_dependence.png"
    )
    cells_path = output_root / "prior_cell_values.csv"
    bins_path = output_root / "prior_bins.csv"
    correlation_path = output_root / "prior_correlation.csv"
    caption_path = output_root / "caption.md"
    alt_text_path = output_root / "alt_text.txt"
    manifest_path = output_root / "manifest.yaml"

    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cells_path, index=False)
    bins.to_csv(bins_path, index=False)
    correlation.to_csv(correlation_path, index=False)
    render_figure(frame, bins, correlation, figure_path)
    caption_path.write_text(
        caption_text(len(frame), len(bins)) + "\n",
        encoding="utf-8",
    )
    alt_text_path.write_text(
        "Two-panel diagnostic of query-marginal deficit versus prior risk "
        "for the selected mouse-spleen CoRe-OT fit. Cell-level values show "
        "substantial deficit variation at the dominant tied prior-risk value; "
        "the second panel shows a positive but nonunit Spearman correlation.\n",
        encoding="utf-8",
    )
    outputs = {
        "cells": cells_path,
        "bins": bins_path,
        "correlation": correlation_path,
        "figure_png": figure_path,
        "figure_pdf": figure_path.with_suffix(".pdf"),
        "figure_svg": figure_path.with_suffix(".svg"),
        "caption": caption_path,
        "alt_text": alt_text_path,
    }
    manifest = {
        "stage": "generate_mouse_spleen_prior_dependence_figure",
        "condition": CONDITION,
        "method": METHOD,
        "cohort": "all_query_cells",
        "truth_stratification": False,
        "selected_parameters": SELECTED_PARAMETERS,
        "binning": {
            "n_target_intervals": N_TARGET_INTERVALS,
            "tie_method": "average_rank_keep_ties_together",
            "empty_interval": "omit",
            "n_occupied_intervals": len(bins),
        },
        "correlations": {
            "primary": "spearman_average_rank",
            "sensitivity": "pearson",
            "p_values": False,
        },
        "lineage": {
            "cell_scores": str(SCORES_RELATIVE),
            "cell_scores_sha256": sha256_file(project_root / SCORES_RELATIVE),
            "prior_profile": str(PRIORS_RELATIVE),
            "prior_profile_sha256": sha256_file(project_root / PRIORS_RELATIVE),
            "method_parameters": str(PARAMETERS_RELATIVE),
            "method_parameters_sha256": sha256_file(
                project_root / PARAMETERS_RELATIVE
            ),
            "transport_manifest": str(TRANSPORT_MANIFEST_RELATIVE),
            "transport_manifest_sha256": sha256_file(
                project_root / TRANSPORT_MANIFEST_RELATIVE
            ),
            "evaluation_truth": str(TRUTH_RELATIVE),
            "evaluation_truth_sha256": sha256_file(project_root / TRUTH_RELATIVE),
            "generator": (
                "experiments/mouse_spleen/"
                "generate_prior_dependence_figure.py"
            ),
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return {**outputs, "manifest": manifest_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the selected-fit mouse-spleen prior-dependence "
            "diagnostic from existing per-cell artifacts."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in generate(args.project_root.resolve()).values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
