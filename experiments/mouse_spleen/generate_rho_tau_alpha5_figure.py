from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from coreot.artifacts.hashes import sha256_file


ENDPOINT = "Proliferating"
ALPHA = 5.0
TAU_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
METRICS = (
    ("auprc", "Relative AP difference (%)"),
    ("auroc", "Relative AUROC difference (%)"),
    ("forced_accuracy", "Relative forced accuracy difference (%)"),
    ("forced_macro_f1", "Relative forced macro-F1 difference (%)"),
)


def absolute_effect_column(metric: str) -> str:
    return f"delta_{metric}_heterogeneous_minus_uniform_mean"


def relative_effect_column(metric: str) -> str:
    return f"relative_{metric}_heterogeneous_minus_uniform_percent_mean"


ABSOLUTE_EFFECT_COLUMNS = tuple(
    absolute_effect_column(metric) for metric, _ in METRICS
)
RELATIVE_EFFECT_COLUMNS = tuple(
    relative_effect_column(metric) for metric, _ in METRICS
)
STORED_RELATIVE_AP_COLUMN = relative_effect_column("auprc")
SOURCE_COLUMNS = (
    "analysis_stage",
    "endpoint",
    "n_splits",
    "alpha",
    "tau_min",
    "tau_max",
    "mean_matched_tau_mean",
    *(
        column
        for metric, _ in METRICS
        for column in (
            f"heterogeneous_{metric}_mean",
            f"uniform_{metric}_mean",
            absolute_effect_column(metric),
        )
    ),
    STORED_RELATIVE_AP_COLUMN,
    "all_heterogeneous_converged",
    "all_uniform_converged",
)


def focused_surface(summary: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(summary.columns))
    if missing:
        raise ValueError(f"Coarse summary lacks required columns: {missing}")

    selected = summary.loc[
        summary["endpoint"].astype(str).eq(ENDPOINT)
        & np.isclose(summary["alpha"].astype(float), ALPHA)
        & summary["tau_min"].astype(float).isin(TAU_VALUES)
        & summary["tau_max"].astype(float).isin(TAU_VALUES),
        SOURCE_COLUMNS,
    ].copy()
    expected = set(combinations_with_replacement(TAU_VALUES, 2))
    observed = {
        (float(row.tau_min), float(row.tau_max))
        for row in selected.itertuples(index=False)
    }
    if observed != expected:
        missing_cells = sorted(expected - observed)
        unexpected_cells = sorted(observed - expected)
        raise ValueError(
            "Focused surface has incorrect triangular coverage: "
            f"missing={missing_cells}, unexpected={unexpected_cells}"
        )
    if selected.duplicated(["tau_min", "tau_max"]).any():
        raise ValueError("Focused surface contains duplicate tau cells")
    if not selected["n_splits"].astype(int).eq(1).all():
        raise ValueError("Mouse-spleen focused surface must contain one dataset")
    if not (
        selected["all_heterogeneous_converged"].eq(True).all()
        and selected["all_uniform_converged"].eq(True).all()
    ):
        raise ValueError("Focused surface contains a nonconverged paired fit")

    for metric, _ in METRICS:
        comparator = selected[f"uniform_{metric}_mean"].astype(float)
        if not comparator.gt(0.0).all():
            raise ValueError(f"Relative {metric} requires a positive comparator")
        relative = (
            100.0
            * selected[absolute_effect_column(metric)].astype(float)
            / comparator
        )
        if metric == "auprc" and not np.allclose(
            relative,
            selected[STORED_RELATIVE_AP_COLUMN].astype(float),
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("Stored relative AP differences do not match AP values")
        selected[relative_effect_column(metric)] = relative

    numerical_columns = (
        "mean_matched_tau_mean",
        *(
            column
            for metric, _ in METRICS
            for column in (
                f"heterogeneous_{metric}_mean",
                f"uniform_{metric}_mean",
                absolute_effect_column(metric),
                relative_effect_column(metric),
            )
        ),
    )
    if not np.isfinite(
        selected.loc[:, numerical_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Focused surface contains nonfinite numerical values")
    diagonal = np.isclose(
        selected["tau_min"].astype(float),
        selected["tau_max"].astype(float),
    )
    diagonal_effects = selected.loc[
        diagonal, [*ABSOLUTE_EFFECT_COLUMNS, *RELATIVE_EFFECT_COLUMNS]
    ].to_numpy(dtype=float)
    if np.max(np.abs(diagonal_effects)) > 1.0e-12:
        raise ValueError("Diagonal heterogeneous/uniform identity failed")

    return selected.sort_values(
        ["tau_min", "tau_max"], kind="mergesort"
    ).reset_index(drop=True)


def render_heatmap(surface: pd.DataFrame, path: Path) -> None:
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), squeeze=False)
    tick_labels = [f"{value:g}" for value in TAU_VALUES]
    for axis, (metric, label) in zip(axes.flat, METRICS, strict=True):
        matrix = (
            surface.pivot(
                index="tau_min",
                columns="tau_max",
                values=relative_effect_column(metric),
            )
            .reindex(index=TAU_VALUES, columns=TAU_VALUES)
            .astype(float)
        )
        limit = float(np.nanmax(np.abs(matrix.to_numpy())))
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(f"{label} heatmap requires a finite nonzero effect")
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
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
                axis.text(
                    column_index,
                    row_index,
                    f"{float(value):.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if abs(float(value)) > 0.55 * limit
                        else "black"
                    ),
                )
        axis.set_xticks(range(len(TAU_VALUES)), tick_labels)
        axis.set_yticks(range(len(TAU_VALUES)), tick_labels)
        axis.set_xlabel(r"$\tau_{\max}$")
        axis.set_ylabel(r"$\tau_{\min}$")
        axis.set_title(label)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def caption_text() -> str:
    return (
        "**Matchability-penalty attribution at fixed $\\alpha=5$.** Panels "
        "show paired relative heterogeneous-minus-mean-matched-uniform "
        "differences (%) in AP, AUROC, represented-state forced accuracy, "
        "and represented-state forced macro-F1, ordered row-wise. For metric "
        "$M$, each value is $100(M_{\\mathrm{heterogeneous}}-"
        "M_{\\mathrm{uniform}})/M_{\\mathrm{uniform}}$. Positive values "
        "favor the heterogeneous query "
        "penalty, and negative values favor the uniform comparator. The "
        "heterogeneous penalty is "
        "$\\tau_{q,i}=\\tau_{\\min}+(\\tau_{\\max}-"
        "\\tau_{\\min})\\rho_{q,i}$; its paired uniform comparator uses "
        "the empirical-query-mass-weighted mean penalty. The vertical and "
        "horizontal axes give $\\tau_{\\min}$ and $\\tau_{\\max}$, "
        "respectively, with "
        "$\\tau_{\\min}\\leq\\tau_{\\max}$. Diagonal cells are zero "
        "because the two penalties coincide. The target penalty, candidate "
        "graph, costs, priors, entropy regularization, and solver settings "
        "are held fixed within each pair. The displayed compatibility "
        "weight and penalty range were examined in a follow-up exploratory "
        "search; values are point estimates from one dataset. Corresponding "
        "absolute differences are retained as scale context in Supplementary "
        "Data 2.\n"
    )


def generate(project_root: Path) -> tuple[Path, Path, Path, Path]:
    analysis_root = (
        project_root
        / "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "rho_attribution_tau_alpha_search"
    )
    by_dataset_path = analysis_root / "tables/coarse_by_dataset.csv"
    summary_path = analysis_root / "tables/coarse_summary.csv"
    for path in (by_dataset_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing regenerated input: {path}")

    summary = pd.read_csv(summary_path, float_precision="round_trip")
    surface = focused_surface(summary)
    source_table_path = (
        analysis_root / "tables/rho_tau_surface_alpha5_range025_4.csv"
    )
    figure_path = (
        project_root
        / "docs/figs/"
        "manuscript_supp_rho_attribution_mouse_spleen_alpha5.png"
    )
    caption_path = figure_path.with_name(
        "manuscript_supp_rho_attribution_mouse_spleen_alpha5_caption.md"
    )
    manifest_path = (
        analysis_root / "design/rho_tau_surface_alpha5_range025_4.yaml"
    )

    source_table_path.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(source_table_path, index=False)
    render_heatmap(surface, figure_path)
    caption_path.write_text(caption_text(), encoding="utf-8")

    manifest = {
        "experiment": "mouse_spleen",
        "endpoint": ENDPOINT,
        "analysis": "exploratory_mean_matched_rho_attribution_surface",
        "alpha": ALPHA,
        "tau_values": list(TAU_VALUES),
        "tau_target": 8.0,
        "effect": "relative heterogeneous minus mean-matched uniform differences",
        "effect_units": "percent",
        "metric_panels": [metric for metric, _ in METRICS],
        "layout": "2_by_2_row_major",
        "relative_effect_definition": (
            "100 * (heterogeneous metric - mean-matched uniform metric) "
            "/ mean-matched uniform metric"
        ),
        "scope": "single_dataset_exploratory_point_estimates",
        "lineage": {
            "paired_dataset_table": str(by_dataset_path.relative_to(project_root)),
            "paired_dataset_table_sha256": sha256_file(by_dataset_path),
            "aggregation_code": (
                "experiments/rho_attribution_discovery_confirmation.py"
                "::summarize_surface"
            ),
            "coarse_summary": str(summary_path.relative_to(project_root)),
            "coarse_summary_sha256": sha256_file(summary_path),
            "focused_generator": (
                "experiments/mouse_spleen/"
                "generate_rho_tau_alpha5_figure.py"
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
            "Generate the focused mouse-spleen alpha=5 rho-attribution "
            "heatmap from the regenerated coarse summary."
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
