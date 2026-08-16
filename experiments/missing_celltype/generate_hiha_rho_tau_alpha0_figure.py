from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
from pathlib import Path
import sys
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from experiments.rho_attribution_discovery_confirmation import (  # noqa: E402
    summarize_surface,
)


ENDPOINTS = ("HLA-DRhi cDC2", "ISG+ cDC2")
TAU_TARGETS = {"HLA-DRhi cDC2": 2.0, "ISG+ cDC2": 3.0}
ALPHA = 0.0
TAU_VALUES = (0.25, 0.5, 0.75, 1.0, 1.25)
METRICS = (
    ("auprc", "AP"),
    ("auroc", "AUROC"),
    ("forced_accuracy", "Forced accuracy"),
    ("forced_macro_f1", "Forced macro-F1"),
)
def effect_column(metric: str) -> str:
    return f"delta_{metric}_heterogeneous_minus_uniform_mean"


def relative_effect_column(metric: str) -> str:
    return f"relative_{metric}_heterogeneous_minus_uniform_percent_mean"


def relative_effect_sd_column(metric: str) -> str:
    return f"relative_{metric}_heterogeneous_minus_uniform_percent_sd"


EFFECT_COLUMNS = tuple(effect_column(metric) for metric, _ in METRICS)
RELATIVE_EFFECT_COLUMNS = tuple(
    relative_effect_column(metric) for metric, _ in METRICS
)
METRIC_SOURCE_COLUMNS = tuple(
    column
    for metric, _ in METRICS
    for column in (
        f"heterogeneous_{metric}_mean",
        f"heterogeneous_{metric}_sd",
        f"uniform_{metric}_mean",
        f"uniform_{metric}_sd",
        effect_column(metric),
        f"delta_{metric}_heterogeneous_minus_uniform_sd",
        f"delta_{metric}_heterogeneous_minus_uniform_n_positive",
        relative_effect_column(metric),
        relative_effect_sd_column(metric),
    )
)
SOURCE_COLUMNS = (
    "experiment",
    "analysis_stage",
    "cohort",
    "endpoint",
    "alpha",
    "alpha_ratio",
    "tau_min",
    "tau_max",
    "n_splits",
    "all_heterogeneous_converged",
    "all_uniform_converged",
    "mean_matched_tau_mean",
    "mean_matched_tau_sd",
) + METRIC_SOURCE_COLUMNS + ("discovery_support",)


def focused_surface(summary: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(summary.columns))
    if missing:
        raise ValueError(f"Coarse summary lacks required columns: {missing}")

    selected = summary.loc[
        summary["experiment"].astype(str).eq("hiha")
        & summary["analysis_stage"].astype(str).eq("alpha0_focused025_125")
        & summary["cohort"].astype(str).eq("discovery")
        & summary["endpoint"].astype(str).isin(ENDPOINTS)
        & np.isclose(summary["alpha"].astype(float), ALPHA)
        & summary["tau_min"].astype(float).isin(TAU_VALUES)
        & summary["tau_max"].astype(float).isin(TAU_VALUES),
        SOURCE_COLUMNS,
    ].copy()

    expected_pairs = set(combinations_with_replacement(TAU_VALUES, 2))
    for endpoint in ENDPOINTS:
        local = selected.loc[selected["endpoint"].eq(endpoint)]
        observed_pairs = {
            (float(row.tau_min), float(row.tau_max))
            for row in local.itertuples(index=False)
        }
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"{endpoint} has incorrect triangular coverage: "
                f"missing={sorted(expected_pairs - observed_pairs)}, "
                f"unexpected={sorted(observed_pairs - expected_pairs)}"
            )
        if local.duplicated(["tau_min", "tau_max"]).any():
            raise ValueError(f"{endpoint} contains duplicate tau cells")

    if not selected["n_splits"].astype(int).eq(5).all():
        raise ValueError("Focused HIHA surface must contain five splits per cell")
    if not (
        selected["all_heterogeneous_converged"].eq(True).all()
        and selected["all_uniform_converged"].eq(True).all()
    ):
        raise ValueError("Focused HIHA surface contains a nonconverged paired fit")

    numerical_columns = (
        "mean_matched_tau_mean",
        "mean_matched_tau_sd",
    ) + tuple(
        column
        for metric, _ in METRICS
        for column in (
            f"heterogeneous_{metric}_mean",
            f"heterogeneous_{metric}_sd",
            f"uniform_{metric}_mean",
            f"uniform_{metric}_sd",
            effect_column(metric),
            f"delta_{metric}_heterogeneous_minus_uniform_sd",
            relative_effect_column(metric),
            relative_effect_sd_column(metric),
        )
    )
    if not np.isfinite(
        selected.loc[:, numerical_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Focused HIHA surface contains nonfinite values")
    for metric, _ in METRICS:
        if not (selected[f"uniform_{metric}_mean"].astype(float) > 0.0).all():
            raise ValueError(f"Relative {metric} requires a positive comparator")

    diagonal = np.isclose(
        selected["tau_min"].astype(float),
        selected["tau_max"].astype(float),
    )
    diagonal_values = selected.loc[
        diagonal, (*EFFECT_COLUMNS, *RELATIVE_EFFECT_COLUMNS)
    ].to_numpy(dtype=float)
    if np.max(np.abs(diagonal_values)) > 1.0e-12:
        raise ValueError("Diagonal heterogeneous/uniform identity failed")
    positive_count_columns = [
        f"delta_{metric}_heterogeneous_minus_uniform_n_positive"
        for metric, _ in METRICS
    ]
    if not selected.loc[diagonal, positive_count_columns].astype(int).eq(0).all(
        axis=None
    ):
        raise ValueError("Diagonal cells must have zero positive splits")

    return selected.sort_values(
        ["endpoint", "tau_min", "tau_max"], kind="mergesort"
    ).reset_index(drop=True)


def render_heatmap(surface: pd.DataFrame, path: Path) -> None:
    matrices = {
        (metric, endpoint): (
            surface.loc[surface["endpoint"].eq(endpoint)]
            .pivot(
                index="tau_min",
                columns="tau_max",
                values=relative_effect_column(metric),
            )
            .reindex(index=TAU_VALUES, columns=TAU_VALUES)
            .astype(float)
        )
        for metric, _ in METRICS
        for endpoint in ENDPOINTS
    }
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    figure, axes = plt.subplots(
        len(METRICS),
        len(ENDPOINTS),
        figsize=(8.4, 12.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    labels = [f"{value:g}" for value in TAU_VALUES]
    for metric_index, (metric, metric_label) in enumerate(METRICS):
        limit = max(
            float(np.nanmax(np.abs(matrices[(metric, endpoint)].to_numpy())))
            for endpoint in ENDPOINTS
        )
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(
                f"{metric_label} heatmap requires a finite nonzero effect"
            )
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = None
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            axis = axes[metric_index, endpoint_index]
            matrix = matrices[(metric, endpoint)]
            image = axis.imshow(
                matrix.to_numpy(),
                origin="lower",
                aspect="equal",
                cmap=cmap,
                norm=norm,
            )
            for row_index, tau_min in enumerate(TAU_VALUES):
                for column_index, tau_max in enumerate(TAU_VALUES):
                    value = matrix.loc[tau_min, tau_max]
                    if pd.isna(value):
                        continue
                    red, green, blue, _ = image.cmap(image.norm(float(value)))
                    axis.text(
                        column_index,
                        row_index,
                        f"{float(value):.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=(
                            "white"
                            if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                            else "black"
                        ),
                    )
            axis.set_xticks(range(len(TAU_VALUES)), labels)
            axis.set_yticks(range(len(TAU_VALUES)), labels)
            axis.set_xlabel(r"$\tau_{\max}$")
            axis.set_ylabel(
                f"Relative {metric_label} difference (%)\n$\\tau_{{\\min}}$"
                if endpoint_index == 0
                else r"$\tau_{\min}$"
            )
            if metric_index == 0:
                axis.set_title(endpoint, fontsize=10, fontweight="bold")
        if image is not None:
            figure.colorbar(
                image,
                ax=axes[metric_index, :],
                location="right",
                fraction=0.046,
                pad=0.03,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def caption_text() -> str:
    return (
        "**Matchability-penalty attribution at fixed $\\alpha=0$.** "
        "Rows show paired relative heterogeneous-minus-mean-matched-uniform "
        "differences (%) in AP, AUROC, represented-state forced accuracy, "
        "and represented-state forced macro-F1; columns show the HLA-DRhi "
        "cDC2 and ISG+ cDC2 endpoints. For metric $M$, each split-level "
        "value is $100(M_{\\mathrm{heterogeneous}}-M_{\\mathrm{uniform}})/"
        "M_{\\mathrm{uniform}}$, and each displayed cell is the arithmetic "
        "mean across five donor-aware discovery splits. Positive values favor the "
        "heterogeneous query penalty, and negative values favor the "
        "mean-matched uniform comparator. The vertical and horizontal axes "
        "show $\\tau_{\\min}$ and $\\tau_{\\max}$, respectively, with "
        "$\\tau_{\\min}\\leq\\tau_{\\max}$. Diagonal cells are zero "
        "because the paired penalties coincide. Color scales are centered "
        "at zero and shared across endpoints within each metric but differ "
        "across metrics. Corresponding absolute differences are retained "
        "as scale context in Supplementary Data 3.\n"
    )


def generate(project_root: Path) -> tuple[Path, Path, Path, Path]:
    analysis_root = (
        project_root
        / "results/HIHA_DC/sensitivity/"
        "rho_attribution_discovery_confirmation"
    )
    by_split_path = (
        analysis_root / "tables/discovery_alpha0_focused025_125_by_split.csv"
    )
    summary_path = (
        analysis_root / "tables/discovery_alpha0_focused025_125_summary.csv"
    )
    for path in (by_split_path,):
        if not path.is_file():
            raise FileNotFoundError(f"Missing regenerated input: {path}")

    by_split = pd.read_csv(by_split_path, float_precision="round_trip")
    summary = summarize_surface(by_split, expected_splits=5)
    summary.to_csv(summary_path, index=False)
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    surface = focused_surface(summary)
    source_table_path = (
        analysis_root / "tables/rho_tau_surface_alpha0_range025_125.csv"
    )
    figure_path = (
        project_root
        / "docs/figs/manuscript_supp_rho_attribution_hiha_alpha0.png"
    )
    caption_path = figure_path.with_name(
        "manuscript_supp_rho_attribution_hiha_alpha0_caption.md"
    )
    manifest_path = (
        analysis_root / "design/rho_tau_surface_alpha0_range025_125.yaml"
    )

    source_table_path.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(source_table_path, index=False)
    render_heatmap(surface, figure_path)
    caption_path.write_text(caption_text(), encoding="utf-8")

    manifest = {
        "experiment": "HIHA_DC",
        "endpoints": list(ENDPOINTS),
        "analysis": "mean_matched_rho_attribution_surface",
        "cohort": "donor_separated_discovery",
        "n_splits": 5,
        "alpha": ALPHA,
        "tau_values": list(TAU_VALUES),
        "tau_target_by_endpoint": TAU_TARGETS,
        "metrics": [metric for metric, _ in METRICS],
        "effect": (
            "arithmetic mean over splits of split-level relative "
            "heterogeneous-minus-mean-matched-uniform differences in percent"
        ),
        "display": {
            "rows": [label for _, label in METRICS],
            "columns": list(ENDPOINTS),
            "panel_letters": False,
            "units": "percent",
            "aggregation": "mean_of_split_level_relative_differences",
            "color_scale": "symmetric about zero and shared by endpoint within metric",
        },
        "scope": "post_hoc_exploratory_narrower_display",
        "selection_provenance": {
            "requested_after_inspecting": (
                "the broader fixed-alpha-zero tau surface"
            ),
            "prespecified": False,
            "independent_confirmation": False,
        },
        "lineage": {
            "paired_split_table": str(by_split_path.relative_to(project_root)),
            "paired_split_table_sha256": sha256_file(by_split_path),
            "aggregation_code": (
                "experiments/rho_attribution_discovery_confirmation.py"
                "::summarize_surface"
            ),
            "focused_summary": str(summary_path.relative_to(project_root)),
            "focused_summary_sha256": sha256_file(summary_path),
            "focused_generator": (
                "experiments/missing_celltype/"
                "generate_hiha_rho_tau_alpha0_figure.py"
            ),
        },
        "artifacts": {
            "source_table": str(source_table_path.relative_to(project_root)),
            "source_table_sha256": sha256_file(source_table_path),
            "figure": str(figure_path.relative_to(project_root)),
            "figure_sha256": sha256_file(figure_path),
            "caption": str(caption_path.relative_to(project_root)),
            "caption_sha256": sha256_file(caption_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return source_table_path, figure_path, caption_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the focused HIHA alpha=0 rho-attribution heatmap from "
            "the regenerated donor-separated 0.25--1.25 summary."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in generate(args.project_root.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
