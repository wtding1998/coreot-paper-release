from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text


EXPECTED_LABELS = ("B cells", "NK cells", "Dendritic cells", "CD8 T cells")
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
REPRESENTATIVE_LABEL = "B cells"
REPRESENTATIVE_SEED = 1
DISPLAY_LABELS = {
    "B cells": "B cells",
    "NK cells": "NK cells",
    "Dendritic cells": "DCs",
    "CD8 T cells": "CD8 T cells",
}
REFERENCE_ORDER = ("ablated", "full")
DESTINATION_CATEGORIES = ("B control", "B stimulated", "Other targets")
RAW_MASS_COLUMN = "median_a_hat_over_a"
TRANSPORT_ETA = 1.0e-12

FIGURE_STYLE = {
    "control": "#377EB8",
    "stimulated": "#E69F00",
    "atlas": "#D9D9D9",
    "diagnostic": "#006D77",
    "control_group": "#8A8A8A",
    "grid": "#E5E7EB",
    "baseline": "#667085",
    "text": "#202124",
    "muted": "#667085",
    "warning": "#A65E00",
}

EXPECTED_AUPRC_MEANS = {
    "B cells": {"coreot_full": 0.970, "scmap_cluster": 0.913, "seurat_anchor": 0.883, "chetah": 0.783},
    "NK cells": {"coreot_full": 0.972, "scmap_cluster": 0.895, "seurat_anchor": 0.601, "chetah": 0.818},
    "Dendritic cells": {"coreot_full": 0.997, "scmap_cluster": 0.894, "seurat_anchor": 0.579, "chetah": 0.682},
    "CD8 T cells": {"coreot_full": 0.898, "scmap_cluster": 0.808, "seurat_anchor": 0.674, "chetah": 0.500},
}
EXPECTED_ABSTENTION_MEANS = {
    "B cells": {
        "coreot_full": (0.990, 0.954), "scmap_cluster": (0.474, 0.966),
        "seurat_anchor": (0.221, 0.945), "chetah": (0.132, 0.949),
    },
    "NK cells": {
        "coreot_full": (0.962, 0.945), "scmap_cluster": (0.249, 0.946),
        "seurat_anchor": (0.200, 0.949), "chetah": (0.119, 0.974),
    },
    "Dendritic cells": {
        "coreot_full": (1.000, 0.955), "scmap_cluster": (0.105, 0.949),
        "seurat_anchor": (0.138, 0.951), "chetah": (0.049, 0.950),
    },
    "CD8 T cells": {
        "coreot_full": (0.574, 0.952), "scmap_cluster": (0.125, 0.956),
        "seurat_anchor": (0.457, 0.958), "chetah": (0.045, 0.956),
    },
}
EXPECTED_FORCED_MACRO_F1_MEANS = {
    "B cells": 0.954,
    "NK cells": 0.956,
    "Dendritic cells": 0.955,
    "CD8 T cells": 0.981,
}


@dataclass(frozen=True)
class MethodScore:
    display_name: str
    method: str
    score: str
    family: str


METHOD_SCORES = (
    MethodScore(r"CoRe-OT $u$", "coreot_full", "u", "internal"),
    MethodScore("Seurat", "seurat_anchor", "u", "external"),
    MethodScore("SingleR", "singleR", "u", "external"),
    MethodScore("CellTypist", "celltypist_l3", "u", "external"),
    MethodScore("scmap-cell", "scmap_cell", "u", "external"),
    MethodScore("scmap-cluster", "scmap_cluster", "u", "external"),
    MethodScore("CHETAH", "chetah", "u", "external"),
)

_METHOD_SCORE_LOOKUP = {(item.method, item.score): item for item in METHOD_SCORES}
MAIN_METHOD_SCORES = tuple(
    _METHOD_SCORE_LOOKUP[key]
    for key in (
        ("coreot_full", "u"),
        ("scmap_cluster", "u"),
        ("seurat_anchor", "u"),
        ("chetah", "u"),
    )
)

METHOD_COLORS = {
    "coreot_full": "#006D77",
    "seurat_anchor": "#009E73",
    "singleR": "#E69F00",
    "celltypist_l3": "#CC79A7",
    "scmap_cell": "#F0E442",
    "scmap_cluster": "#56B4E9",
    "chetah": "#AA4499",
}
METHOD_MARKERS = {
    ("coreot_full", "u"): "o",
    ("seurat_anchor", "u"): "s",
    ("singleR", "u"): "D",
    ("celltypist_l3", "u"): "P",
    ("scmap_cell", "u"): "X",
    ("scmap_cluster", "u"): "v",
    ("chetah", "u"): "D",
}


@dataclass(frozen=True)
class PBMCFigurePaths:
    output_root: Path
    data_root: Path
    umap_scores: Path
    detection_by_seed: Path
    detection_summary: Path
    destination_by_seed: Path
    destination_summary: Path
    restoration_by_seed: Path
    restoration_summary: Path
    mass_retention_by_seed: Path
    mass_retention_summary: Path
    abstention_by_seed: Path
    abstention_summary: Path
    forced_macro_f1_by_seed: Path
    forced_macro_f1_summary: Path
    png: Path
    pdf: Path
    svg: Path
    description: Path


class PBMCFigureError(ValueError):
    pass


def _paths(output_root: Path) -> PBMCFigurePaths:
    data_root = output_root / "data"
    return PBMCFigurePaths(
        output_root=output_root,
        data_root=data_root,
        umap_scores=data_root / "figure_3_seed1_umap_scores.csv",
        detection_by_seed=data_root / "figure_3_within_celltype_detection_by_seed.csv",
        detection_summary=data_root / "figure_3_within_celltype_detection_summary.csv",
        destination_by_seed=data_root / "figure_3_destination_rescue_by_seed.csv",
        destination_summary=data_root / "figure_3_destination_rescue_summary.csv",
        restoration_by_seed=data_root / "figure_3_restoration_by_seed.csv",
        restoration_summary=data_root / "figure_3_restoration_summary.csv",
        mass_retention_by_seed=data_root / "figure_3_mass_retention_by_seed.csv",
        mass_retention_summary=data_root / "figure_3_mass_retention_summary.csv",
        abstention_by_seed=data_root / "figure_3_abstention_tradeoff_by_seed.csv",
        abstention_summary=data_root / "figure_3_abstention_tradeoff_summary.csv",
        forced_macro_f1_by_seed=data_root / "figure_3_forced_macro_f1_by_seed.csv",
        forced_macro_f1_summary=data_root / "figure_3_forced_macro_f1_summary.csv",
        png=output_root / "figure_3_condition_specific_weak_correspondence.png",
        pdf=output_root / "figure_3_condition_specific_weak_correspondence.pdf",
        svg=output_root / "figure_3_condition_specific_weak_correspondence.svg",
        description=output_root / "figure_3_condition_specific_weak_correspondence.md",
    )


def _validate_comparison_coverage(comparison: pd.DataFrame) -> None:
    required = {
        "result_family", "run_id", "held_out_label", "seed", "method", "score",
        "auroc", "auprc", "auprc_baseline", "absent_abstention_rate",
        "shared_false_abstention_rate",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise PBMCFigureError(f"Comparison table is missing columns {sorted(missing)}.")
    expected = {
        (label, seed, item.method, item.score)
        for label in EXPECTED_LABELS
        for seed in EXPECTED_SEEDS
        for item in METHOD_SCORES
    }
    key_columns = ["held_out_label", "seed", "method", "score"]
    displayed = comparison.loc[
        comparison.apply(
            lambda row: (row["method"], row["score"])
            in {(item.method, item.score) for item in METHOD_SCORES},
            axis=1,
        ),
        key_columns,
    ]
    actual = set(displayed.itertuples(index=False, name=None))
    duplicate_keys = displayed.loc[displayed.duplicated(keep=False)]
    if actual != expected or not duplicate_keys.empty:
        raise PBMCFigureError(
            "Comparison table must contain each displayed method-score row exactly once "
            f"for four labels and five seeds; missing={sorted(expected - actual)}, "
            f"unexpected_displayed={sorted(actual - expected)}, "
            f"duplicate_displayed={duplicate_keys.to_dict(orient='records')}."
        )


def _read_truth(run_root: Path, condition: str) -> pd.DataFrame:
    path = run_root / "benchmark" / condition / "evaluation_truth" / "query_truth.csv"
    truth = pd.read_csv(path)
    required = {"cell_id", "true_label", "is_absent_state"}
    missing = required - set(truth.columns)
    if missing:
        raise PBMCFigureError(f"{path} is missing columns {sorted(missing)}.")
    if truth["cell_id"].duplicated().any():
        raise PBMCFigureError(f"{path} contains duplicate cell identifiers.")
    return truth


def _read_scores(
    run_root: Path, item: MethodScore, condition: str, internal_candidate_set: str,
    external_candidate_set: str,
) -> pd.DataFrame:
    candidate = internal_candidate_set if item.family == "internal" else external_candidate_set
    path = run_root / "scoring" / condition / candidate / "cell_scores.parquet"
    frame = pd.read_parquet(path)
    required = {"cell_id", "method", item.score}
    missing = required - set(frame.columns)
    if missing:
        raise PBMCFigureError(f"{path} is missing columns {sorted(missing)}.")
    selected = frame.loc[frame["method"] == item.method, ["cell_id", item.score]].copy()
    if selected["cell_id"].duplicated().any():
        raise PBMCFigureError(f"Duplicate cells for {item.method}/{item.score} in {path}.")
    values = selected[item.score].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PBMCFigureError(f"Non-finite scores for {item.method}/{item.score} in {path}.")
    return selected


def _read_transport_scores(
    run_root: Path,
    condition: str,
    internal_candidate_set: str,
    expected_cell_ids: set[str] | None = None,
) -> pd.DataFrame:
    path = (
        run_root / "transport" / condition / internal_candidate_set / "coreot_full"
        / "cell_transport_scores.parquet"
    )
    frame = pd.read_parquet(path)
    required = {"cell_id", "method", "a", "a_hat", "u", "e", "rho", "tau_source"}
    missing = required - set(frame.columns)
    if missing:
        raise PBMCFigureError(f"{path} is missing columns {sorted(missing)}.")
    selected = frame.loc[frame["method"] == "coreot_full"].copy()
    selected["cell_id"] = selected["cell_id"].astype(str)
    if selected["cell_id"].duplicated().any():
        raise PBMCFigureError(f"Duplicate CoRe-OT transport rows in {path}.")
    if expected_cell_ids is not None and set(selected["cell_id"]) != expected_cell_ids:
        raise PBMCFigureError(f"CoRe-OT transport cells do not match evaluation truth in {path}.")
    numeric = ["a", "a_hat", "u", "e", "rho", "tau_source"]
    values = selected[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PBMCFigureError(f"Non-finite CoRe-OT transport scores in {path}.")
    if (selected["a"] <= 0.0).any() or (selected["a_hat"] < 0.0).any():
        raise PBMCFigureError(f"Invalid source masses in {path}.")
    expected_u = np.maximum(selected["a"].to_numpy() - selected["a_hat"].to_numpy(), 0.0)
    expected_u /= selected["a"].to_numpy() + TRANSPORT_ETA
    if not np.allclose(selected["u"].to_numpy(), expected_u, rtol=0.0, atol=1.0e-10):
        raise PBMCFigureError(f"Raw source-deficit formula mismatch in {path}.")
    selected[RAW_MASS_COLUMN] = selected["a_hat"] / selected["a"]
    if not np.isfinite(selected[RAW_MASS_COLUMN].to_numpy()).all():
        raise PBMCFigureError(f"Non-finite relative transported mass in {path}.")
    return selected


def compute_within_celltype_metrics(
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    cell_metadata: pd.DataFrame,
    held_out_label: str,
    score_column: str,
) -> dict[str, float | int]:
    if not truth["cell_id"].is_unique or not scores["cell_id"].is_unique:
        raise PBMCFigureError("Truth and score cell identifiers must be unique.")
    if set(truth["cell_id"]) != set(scores["cell_id"]):
        raise PBMCFigureError("Score cell set does not match evaluation truth.")
    metadata = cell_metadata[["cell_type", "condition"]].copy()
    metadata.index = metadata.index.astype(str)
    merged = truth[["cell_id", "true_label", "is_absent_state"]].merge(
        scores[["cell_id", score_column]], on="cell_id", validate="one_to_one"
    )
    joined = metadata.reindex(merged["cell_id"].astype(str))
    if joined.isna().any().any():
        raise PBMCFigureError("Some evaluated cells do not join to raw PBMC metadata.")
    merged["raw_cell_type"] = joined["cell_type"].to_numpy()
    merged["condition"] = joined["condition"].to_numpy()
    if not (merged["true_label"].astype(str) == merged["raw_cell_type"].astype(str)).all():
        raise PBMCFigureError("Evaluation and raw-data cell-type annotations disagree.")
    expected_positive = (
        (merged["raw_cell_type"] == held_out_label) & (merged["condition"] == "stim")
    )
    if not np.array_equal(expected_positive.to_numpy(), merged["is_absent_state"].astype(bool)):
        raise PBMCFigureError("Evaluation positives disagree with cell type x condition metadata.")
    subset = merged.loc[
        (merged["raw_cell_type"] == held_out_label) & merged["condition"].isin(["ctrl", "stim"])
    ].copy()
    y_true = (subset["condition"] == "stim").to_numpy()
    y_score = subset[score_column].to_numpy(dtype=float)
    if y_true.sum() == 0 or (~y_true).sum() == 0:
        raise PBMCFigureError(f"Within-cell-type detection is undefined for {held_out_label}.")
    return {
        "n_positives": int(y_true.sum()),
        "n_negatives": int((~y_true).sum()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "auprc_baseline": float(y_true.mean()),
    }


def compute_destination_composition(
    coupling: pd.DataFrame,
    positive_cell_ids: set[str],
    target_metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {"source_cell_id", "target_cell_id", "coupling"}
    missing = required - set(coupling.columns)
    if missing:
        raise PBMCFigureError(f"Coupling is missing columns {sorted(missing)}.")
    source_ids = {str(cell_id) for cell_id in positive_cell_ids}
    selected = coupling.loc[coupling["source_cell_id"].astype(str).isin(source_ids)].copy()
    selected["source_cell_id"] = selected["source_cell_id"].astype(str)
    selected["target_cell_id"] = selected["target_cell_id"].astype(str)
    if not np.isfinite(selected["coupling"].to_numpy(dtype=float)).all():
        raise PBMCFigureError("Coupling contains non-finite entries.")
    if (selected["coupling"] < 0.0).any():
        raise PBMCFigureError("Coupling contains negative entries.")
    totals = selected.groupby("source_cell_id")["coupling"].sum()
    present_sources = set(totals.index)
    zero_sources = (source_ids - present_sources) | set(totals.index[totals <= 0.0])
    eligible_sources = source_ids - zero_sources
    if not eligible_sources:
        raise PBMCFigureError("No held-out source cells have positive transported mass.")
    selected = selected.loc[selected["source_cell_id"].isin(eligible_sources)].copy()
    row_totals = selected["source_cell_id"].map(totals)
    if (row_totals <= 0.0).any() or not np.isfinite(row_totals).all():
        raise PBMCFigureError("Eligible source cells must have positive finite transported mass.")
    metadata = target_metadata[["cell_type", "condition"]].copy()
    metadata.index = metadata.index.astype(str)
    joined = metadata.reindex(selected["target_cell_id"])
    if joined.isna().any().any():
        raise PBMCFigureError("Some coupling targets do not join to raw PBMC metadata.")
    selected["target_cell_type"] = joined["cell_type"].to_numpy()
    selected["target_condition"] = joined["condition"].to_numpy()
    selected["conditional_mass"] = selected["coupling"].to_numpy(dtype=float) / row_totals
    per_cell = (
        selected.groupby(
            ["source_cell_id", "target_cell_type", "target_condition"], sort=True
        )["conditional_mass"]
        .sum()
        .reset_index()
    )
    composition = (
        per_cell.groupby(["target_cell_type", "target_condition"], sort=True)[
            "conditional_mass"
        ]
        .sum()
        .div(len(eligible_sources))
        .rename("proportion")
        .reset_index()
    )
    if not np.isclose(composition["proportion"].sum(), 1.0, atol=1e-9):
        raise PBMCFigureError("Destination composition does not sum to one.")
    composition["n_positive_transport_sources"] = len(eligible_sources)
    composition["n_zero_transport_sources"] = len(zero_sources)
    return composition


def compute_restoration_summaries(
    truth: pd.DataFrame,
    ablated_scores: pd.DataFrame,
    full_scores: pd.DataFrame,
    cell_metadata: pd.DataFrame,
    held_out_label: str,
) -> pd.DataFrame:
    """Return cell-group medians without treating cells as replicates."""
    ablated = ablated_scores.loc[
        ablated_scores["method"] == "coreot_full", ["cell_id", "u"]
    ].rename(columns={"u": "u_ablated"})
    full = full_scores.loc[
        full_scores["method"] == "coreot_full", ["cell_id", "u"]
    ].rename(columns={"u": "u_full"})
    merged = truth[["cell_id", "is_absent_state"]].merge(
        ablated, on="cell_id", validate="one_to_one"
    ).merge(full, on="cell_id", validate="one_to_one")
    metadata = cell_metadata[["cell_type", "condition"]].copy()
    metadata.index = metadata.index.astype(str)
    joined = metadata.reindex(merged["cell_id"].astype(str))
    if joined.isna().any().any():
        raise PBMCFigureError("Restoration-score cells do not join to PBMC metadata.")
    merged["cell_type"] = joined["cell_type"].to_numpy()
    merged["condition"] = joined["condition"].to_numpy()
    selected = merged.loc[merged["cell_type"] == held_out_label].copy()
    selected["cell_group"] = np.where(
        selected["condition"] == "stim", "held_out_stimulated", "same_type_control"
    )
    expected_absent = selected["cell_group"] == "held_out_stimulated"
    if not np.array_equal(expected_absent.to_numpy(), selected["is_absent_state"].astype(bool)):
        raise PBMCFigureError("Restoration groups disagree with held-out-state truth.")
    rows: list[dict[str, object]] = []
    for group, group_frame in selected.groupby("cell_group", sort=False):
        for reference, column in (("ablated", "u_ablated"), ("full", "u_full")):
            rows.append({
                "cell_group": group,
                "reference": reference,
                "n_cells": len(group_frame),
                "median_u": float(group_frame[column].median()),
            })
    return pd.DataFrame(rows)


def _summarize(frame: pd.DataFrame, keys: list[str], value_columns: list[str]) -> pd.DataFrame:
    summary = frame.groupby(keys, sort=False)[value_columns].agg(["mean", "std", "count"])
    summary.columns = [f"{value}_{stat}" for value, stat in summary.columns]
    summary = summary.reset_index()
    for value in value_columns:
        summary[f"{value}_sem"] = summary[f"{value}_std"] / np.sqrt(
            summary[f"{value}_count"]
        )
    return summary


def _validate_summary_means(
    by_seed: pd.DataFrame,
    summary: pd.DataFrame,
    keys: list[str],
    value_columns: list[str],
    name: str,
) -> None:
    for _, row in summary.iterrows():
        subset = by_seed
        for key in keys:
            subset = subset.loc[subset[key] == row[key]]
        if len(subset) != 5:
            raise PBMCFigureError(f"{name} summary key {tuple(row[key] for key in keys)} has {len(subset)} seeds.")
        for value in value_columns:
            if not np.isclose(float(row[f"{value}_mean"]), float(subset[value].mean()),
                              rtol=0.0, atol=1.0e-12):
                raise PBMCFigureError(f"{name} summary mean mismatch for {value}.")
            expected_sd = float(subset[value].std(ddof=1))
            if not np.isclose(float(row[f"{value}_std"]), expected_sd, rtol=0.0, atol=1.0e-12):
                raise PBMCFigureError(f"{name} summary sample SD mismatch for {value}.")


def validate_figure_data(
    *,
    umap: pd.DataFrame,
    detection: pd.DataFrame,
    detection_summary: pd.DataFrame,
    restoration: pd.DataFrame,
    restoration_summary: pd.DataFrame,
    destination: pd.DataFrame,
    destination_summary: pd.DataFrame,
    mass_retention: pd.DataFrame,
    mass_retention_summary: pd.DataFrame,
    abstention: pd.DataFrame,
    abstention_summary: pd.DataFrame,
    forced_macro_f1: pd.DataFrame,
    forced_macro_f1_summary: pd.DataFrame,
) -> dict[str, object]:
    """Validate the exact data contract used by the six displayed panels."""
    expected_pairs = {(label, seed, item.method, item.score)
                      for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS
                      for item in MAIN_METHOD_SCORES}
    all_expected_pairs = {(label, seed, item.method, item.score)
                          for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS
                          for item in METHOD_SCORES}
    detection_keys = detection[["held_out_label", "seed", "method", "score"]]
    if set(detection_keys.itertuples(index=False, name=None)) != all_expected_pairs:
        raise PBMCFigureError("Detection source table does not cover all preserved states, seeds, and methods exactly.")
    if detection_keys.duplicated().any():
        raise PBMCFigureError("Detection source table contains duplicate displayed rows.")
    if not np.isfinite(detection[["auprc", "auroc", "auprc_baseline"]].to_numpy(dtype=float)).all():
        raise PBMCFigureError("Detection source table contains non-finite values.")
    if not detection["seed"].isin(EXPECTED_SEEDS).all():
        raise PBMCFigureError("Detection source table contains an unexpected seed.")
    _validate_summary_means(
        detection, detection_summary,
        ["held_out_label", "display_name", "method", "score"],
        ["auroc", "auprc", "auprc_baseline"], "detection",
    )
    for label, method_values in EXPECTED_AUPRC_MEANS.items():
        for method, expected in method_values.items():
            actual = float(detection.loc[
                (detection["held_out_label"] == label) & (detection["method"] == method)
                & (detection["score"] == "u"), "auprc"
            ].mean())
            if not np.isclose(actual, expected, atol=0.002, rtol=0.0):
                raise PBMCFigureError(f"AUPRC mean changed for {label}/{method}: {actual:.6f}.")

    restoration_keys = restoration[["held_out_label", "seed", "run_id", "cell_group", "reference"]]
    if restoration_keys.duplicated().any() or set(restoration["seed"]) != set(EXPECTED_SEEDS):
        raise PBMCFigureError("Restoration source table is not uniquely paired over five seeds.")
    if set(restoration["reference"]) != set(REFERENCE_ORDER):
        raise PBMCFigureError("Restoration source table has an unexpected reference condition.")
    if not np.isfinite(restoration[["median_u", "delta_median_u"]].to_numpy(dtype=float)).all():
        raise PBMCFigureError("Restoration source table contains non-finite values.")
    _validate_summary_means(
        restoration, restoration_summary,
        ["held_out_label", "cell_group", "reference"],
        ["median_u", "delta_median_u"], "restoration",
    )

    if set(destination["seed"]) != set(EXPECTED_SEEDS):
        raise PBMCFigureError("Destination source table does not cover all five seeds.")
    if destination[["seed", "condition_id", "target_cell_type", "target_condition"]].duplicated().any():
        raise PBMCFigureError("Destination source table contains duplicate category rows.")
    if not np.isfinite(destination["proportion"].to_numpy(dtype=float)).all():
        raise PBMCFigureError("Destination source table contains non-finite proportions.")
    for (seed, condition_id), group in destination.groupby(["seed", "condition_id"]):
        if not np.isclose(group["proportion"].sum(), 1.0, atol=1.0e-9):
            raise PBMCFigureError(f"Destination composition does not sum to one for seed {seed}/{condition_id}.")
        if group["n_zero_transport_sources"].nunique() != 1:
            raise PBMCFigureError("Zero-transport source count is not recorded consistently.")
    _validate_summary_means(
        destination, destination_summary,
        ["condition_id", "target_cell_type", "target_condition"], ["proportion"], "destination",
    )

    mass_keys = mass_retention[["held_out_label", "seed", "run_id", "reference"]]
    expected_mass_keys = {(REPRESENTATIVE_LABEL, seed, reference)
                          for seed in EXPECTED_SEEDS for reference in REFERENCE_ORDER}
    if set(mass_keys[["held_out_label", "seed", "reference"]].itertuples(index=False, name=None)) != expected_mass_keys:
        raise PBMCFigureError("Mass-retention source table does not cover the paired B-cell splits exactly.")
    if mass_keys.duplicated().any() or (mass_retention[RAW_MASS_COLUMN] < 0.0).any():
        raise PBMCFigureError("Mass-retention source table is duplicated or negative.")
    if not np.isfinite(mass_retention[RAW_MASS_COLUMN].to_numpy(dtype=float)).all():
        raise PBMCFigureError("Mass-retention source table contains non-finite values.")
    _validate_summary_means(
        mass_retention, mass_retention_summary,
        ["held_out_label", "reference"], [RAW_MASS_COLUMN], "mass retention",
    )

    abstention_keys = abstention[["held_out_label", "seed", "method", "score"]]
    if set(abstention_keys.itertuples(index=False, name=None)) != expected_pairs or abstention_keys.duplicated().any():
        raise PBMCFigureError("Abstention source table does not cover displayed states, seeds, and methods exactly.")
    if not np.isfinite(abstention[["absent_abstention_rate", "shared_false_abstention_rate"]].to_numpy(dtype=float)).all():
        raise PBMCFigureError("Abstention source table contains non-finite values.")
    coverage = 1.0 - abstention["shared_false_abstention_rate"].to_numpy(dtype=float)
    if ((coverage < 0.0) | (coverage > 1.0)).any():
        raise PBMCFigureError("Shared-cell coverage is outside [0, 1].")
    _validate_summary_means(
        abstention, abstention_summary,
        ["held_out_label", "display_name", "method", "score"],
        ["absent_abstention_rate", "shared_false_abstention_rate"], "abstention",
    )
    for label, method_values in EXPECTED_ABSTENTION_MEANS.items():
        for method, (expected_absent, expected_coverage) in method_values.items():
            subset = abstention.loc[(abstention["held_out_label"] == label)
                                    & (abstention["method"] == method) & (abstention["score"] == "u")]
            actual_absent = float(subset["absent_abstention_rate"].mean())
            actual_coverage = float((1.0 - subset["shared_false_abstention_rate"]).mean())
            if not np.isclose(actual_absent, expected_absent, atol=0.002, rtol=0.0):
                raise PBMCFigureError(f"Abstention mean changed for {label}/{method}.")
            if not np.isclose(actual_coverage, expected_coverage, atol=0.002, rtol=0.0):
                raise PBMCFigureError(f"Coverage mean changed for {label}/{method}.")

    forced_keys = forced_macro_f1[["held_out_label", "seed"]]
    if set(forced_keys.itertuples(index=False, name=None)) != {
        (label, seed) for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS
    } or forced_keys.duplicated().any():
        raise PBMCFigureError("Forced macro-F1 source table is incomplete or duplicated.")
    if not np.isfinite(forced_macro_f1["forced_macro_f1"].to_numpy(dtype=float)).all():
        raise PBMCFigureError("Forced macro-F1 source table contains non-finite values.")
    _validate_summary_means(
        forced_macro_f1, forced_macro_f1_summary, ["held_out_label"], ["forced_macro_f1"],
        "forced macro-F1",
    )
    for label, expected in EXPECTED_FORCED_MACRO_F1_MEANS.items():
        actual = float(forced_macro_f1.loc[forced_macro_f1["held_out_label"] == label,
                                           "forced_macro_f1"].mean())
        if not np.isclose(actual, expected, atol=0.002, rtol=0.0):
            raise PBMCFigureError(f"Forced macro-F1 mean changed for {label}: {actual:.6f}.")

    required_umap = {
        "cell_id", "umap_1", "umap_2", "cell_type", "condition", "u_ablated", "u_full", "delta_u"
    }
    missing_umap = required_umap - set(umap.columns)
    if missing_umap or umap["cell_id"].duplicated().any():
        raise PBMCFigureError(f"UMAP source table is missing columns or duplicate cells: {sorted(missing_umap)}.")
    numeric_umap = umap[["umap_1", "umap_2", "u_ablated", "u_full", "delta_u"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_umap).all():
        raise PBMCFigureError("UMAP source table contains non-finite values.")
    if not np.allclose(umap["delta_u"], umap["u_ablated"] - umap["u_full"], atol=1.0e-12, rtol=0.0):
        raise PBMCFigureError("UMAP restoration response is not u_ablated - u_full.")
    if ((umap["u_ablated"] < 0.0) | (umap["u_ablated"] > 1.0)).any():
        raise PBMCFigureError("Ablated raw deficit is outside [0, 1].")
    b_delta = umap.loc[umap["cell_type"].eq(REPRESENTATIVE_LABEL), "delta_u"].to_numpy(dtype=float)
    delta_limit = float(np.max(np.abs(b_delta)))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "delta_limit": delta_limit,
        "delta_boundary_count": int(np.isclose(np.abs(b_delta), delta_limit, atol=1.0e-12).sum()),
        "mass_max": float(mass_retention[RAW_MASS_COLUMN].max()),
        "n_zero_transport_sources": int(destination["n_zero_transport_sources"].max()),
    }


def _compute_umap_scores(
    *, raw_data_path: Path, query_ids: list[str], truth: pd.DataFrame,
    ablated_scores: pd.DataFrame, full_scores: pd.DataFrame,
) -> pd.DataFrame:
    adata = ad.read_h5ad(raw_data_path)
    raw_ids = set(adata.obs_names.astype(str))
    if set(query_ids) - raw_ids:
        raise PBMCFigureError("Some seed-1 query cells are absent from the raw PBMC dataset.")
    query = adata[query_ids].copy()
    sc.pp.normalize_total(query, target_sum=10_000)
    sc.pp.log1p(query)
    sc.pp.highly_variable_genes(query, n_top_genes=2_000, flavor="seurat")
    query = query[:, query.var["highly_variable"]].copy()
    sc.pp.scale(query, max_value=10)
    sc.tl.pca(query, n_comps=30, svd_solver="arpack", random_state=1)
    sc.pp.neighbors(query, n_neighbors=15, n_pcs=30, metric="euclidean", random_state=1)
    sc.tl.umap(query, min_dist=0.3, random_state=1)
    coordinates = pd.DataFrame(
        {"cell_id": query.obs_names.astype(str), "umap_1": query.obsm["X_umap"][:, 0],
         "umap_2": query.obsm["X_umap"][:, 1]}
    )
    metadata = adata.obs.reindex(query_ids)
    coordinates["cell_type"] = metadata["cell_type"].astype(str).to_numpy()
    coordinates["condition"] = metadata["label"].astype(str).to_numpy()
    merged = coordinates.merge(
        truth[["cell_id", "is_absent_state"]], on="cell_id", validate="one_to_one"
    )
    for frame, suffix in ((ablated_scores, "ablated"), (full_scores, "full")):
        selected = frame.loc[frame["method"] == "coreot_full", ["cell_id", "u", "u_tilde"]]
        if not selected["cell_id"].is_unique or set(selected["cell_id"]) != set(query_ids):
            raise PBMCFigureError(f"CoRe-OT {suffix} scores do not match seed-1 query cells.")
        selected = selected.rename(columns={"u": f"u_{suffix}", "u_tilde": f"u_tilde_{suffix}"})
        merged = merged.merge(selected, on="cell_id", validate="one_to_one")
    merged["delta_u"] = merged["u_ablated"] - merged["u_full"]
    merged["delta_u_tilde"] = merged["u_tilde_ablated"] - merged["u_tilde_full"]
    return merged.sort_values("cell_id", ignore_index=True)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10, 1.10, label, transform=ax.transAxes, fontsize=15,
        fontweight="bold", va="top", ha="left", color=FIGURE_STYLE["text"],
        clip_on=False,
    )


def _rounded_box(ax: plt.Axes, x: float, y: float, width: float, height: float,
                 title: str, subtitle: str, *, facecolor: str = "#F7F7F7") -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=facecolor, edgecolor="#5B616B", linewidth=0.9,
    ))
    ax.text(x + width / 2, y + height * 0.70, title, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=FIGURE_STYLE["text"])
    ax.text(x + width / 2, y + height * 0.30, subtitle, ha="center", va="center",
            fontsize=7.3, color=FIGURE_STYLE["text"])


def _state_icon(ax: plt.Axes, x: float, y: float, color: str, label: str,
                *, crossed: bool = False, size: float = 50.0) -> None:
    ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    if crossed:
        ax.plot([x - 0.025, x + 0.025], [y - 0.025, y + 0.025], color=FIGURE_STYLE["warning"],
                linewidth=1.3, zorder=5)
        ax.plot([x - 0.025, x + 0.025], [y + 0.025, y - 0.025], color=FIGURE_STYLE["warning"],
                linewidth=1.3, zorder=5)
    if label:
        ax.text(x, y - 0.065, label, ha="center", va="top", fontsize=6.7,
                color=FIGURE_STYLE["text"])


def _draw_schematic(ax: plt.Axes) -> None:
    """Draw two separate, matched reference runs without crossing paths."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _rounded_box(ax, 0.22, 0.80, 0.56, 0.16, "Same query donors", "")
    for x, color, label in (
        (0.43, FIGURE_STYLE["control"], "control"),
        (0.57, FIGURE_STYLE["stimulated"], "stimulated"),
    ):
        _state_icon(ax, x, 0.845, color, label, size=42)
    for x, title, subtitle, stim_crossed in (
        (0.03, "Ablated reference", "stimulated B state removed", True),
        (0.54, "Matched full reference", "stimulated B state present", False),
    ):
        _rounded_box(ax, x, 0.46, 0.43, 0.24, title, "")
        _state_icon(ax, x + 0.13, 0.565, FIGURE_STYLE["control"], "", size=42)
        _state_icon(ax, x + 0.28, 0.565, FIGURE_STYLE["stimulated"], "",
                    crossed=stim_crossed, size=42)
        ax.text(x + 0.215, 0.475, subtitle, ha="center", va="center", fontsize=6.8,
                color=FIGURE_STYLE["text"])
    _rounded_box(
        ax, 0.03, 0.22, 0.43, 0.16, "Paired CoRe-OT run 1",
        r"$u$ · destination · abstain / label transfer",
        facecolor="#F2F8F8",
    )
    _rounded_box(
        ax, 0.54, 0.22, 0.43, 0.16, "Paired CoRe-OT run 2",
        r"$u$ · destination · abstain / label transfer",
        facecolor="#F2F8F8",
    )
    ax.plot([0.50, 0.50], [0.17, 0.20], color="#5B616B", linewidth=0.8)
    ax.plot([0.31, 0.69], [0.17, 0.17], color="#5B616B", linewidth=0.8)
    ax.text(0.50, 0.12, "compare paired outputs", ha="center", va="center",
            fontsize=7.4, color=FIGURE_STYLE["diagnostic"])

    ax.text(0.50, 0.985,
            "same query cells · same donor split · same source prior · same operating point",
            ha="center", va="center", fontsize=6.7, color=FIGURE_STYLE["muted"])
    for x in (0.50,):
        ax.add_patch(FancyArrowPatch(
            (x, 0.80), (0.25, 0.705), arrowstyle="-|>", mutation_scale=9,
            linewidth=0.8, color="#5B616B", connectionstyle="arc3,rad=0.0",
        ))
        ax.add_patch(FancyArrowPatch(
            (x, 0.80), (0.75, 0.705), arrowstyle="-|>", mutation_scale=9,
            linewidth=0.8, color="#5B616B", connectionstyle="arc3,rad=0.0",
        ))
    ax.add_patch(FancyArrowPatch((0.245, 0.48), (0.245, 0.385), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.8, color="#5B616B"))
    ax.add_patch(FancyArrowPatch((0.755, 0.48), (0.755, 0.385), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.8, color="#5B616B"))
    ax.text(0.50, 0.055, "Model inputs: PCA, broad-lineage anchor, condition-blind reliability prior",
            ha="center", va="center", fontsize=7.2, color=FIGURE_STYLE["text"])
    ax.text(0.50, 0.015,
            "Benchmark-only, not model-visible: condition and held-out-state identity",
            ha="center", va="center", fontsize=7.2, color=FIGURE_STYLE["warning"])


def _clean_embedding_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=9.0, pad=4, color=FIGURE_STYLE["text"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7.7, labelpad=2)
    ax.set_ylabel("UMAP2", fontsize=7.7, labelpad=2)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _b_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    b = frame.loc[frame["cell_type"] == REPRESENTATIVE_LABEL]
    x0, x1 = np.quantile(b["umap_1"], [0.02, 0.98])
    y0, y1 = np.quantile(b["umap_2"], [0.02, 0.98])
    px = max((x1 - x0) * 0.08, 0.3)
    py = max((y1 - y0) * 0.08, 0.3)
    return (x0 - px, x1 + px), (y0 - py, y1 + py)


def _draw_b_zoom(
    ax: plt.Axes, frame: pd.DataFrame, title: str, limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    frame = frame.sort_values("cell_id")
    b = frame.loc[frame["cell_type"] == REPRESENTATIVE_LABEL]
    ax.scatter(
        frame["umap_1"], frame["umap_2"], s=3.0, color=FIGURE_STYLE["atlas"], linewidth=0,
        alpha=0.45, rasterized=True,
    )
    for condition, color, label in (
        ("ctrl", FIGURE_STYLE["control"], "B control"),
        ("stim", FIGURE_STYLE["stimulated"], "B stimulated"),
    ):
        group = b.loc[b["condition"] == condition]
        ax.scatter(group["umap_1"], group["umap_2"], s=10, color=color, linewidth=0,
                   label=label, alpha=0.88, rasterized=True)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    _clean_embedding_axis(ax, title)


def _draw_score_map(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    norm: matplotlib.colors.Normalize | TwoSlopeNorm,
    cmap: str,
    colorbar_ticks: list[float],
) -> None:
    frame = frame.sort_values("cell_id")
    b = frame.loc[frame["cell_type"] == REPRESENTATIVE_LABEL].copy()
    ax.scatter(
        frame["umap_1"], frame["umap_2"], s=3.0, color=FIGURE_STYLE["atlas"], linewidth=0,
        alpha=0.45, rasterized=True,
    )
    values = b[column].to_numpy(dtype=float)
    order = np.argsort(values)
    scatter = ax.scatter(b["umap_1"].to_numpy()[order], b["umap_2"].to_numpy()[order],
                         c=values[order], s=9, cmap=cmap, norm=norm, linewidth=0,
                         rasterized=True)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    _clean_embedding_axis(ax, title)
    colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.035, pad=0.01)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.tick_params(labelsize=6.0, length=2, pad=1)


def _draw_restoration_facets(axes: list[plt.Axes], by_seed: pd.DataFrame) -> None:
    colors = {
        "held_out_stimulated": FIGURE_STYLE["stimulated"],
        "same_type_control": FIGURE_STYLE["control_group"],
    }
    for ax, label in zip(axes, EXPECTED_LABELS):
        subset = by_seed.loc[by_seed["held_out_label"] == label]
        for group, color in colors.items():
            rows = subset.loc[subset["cell_group"] == group]
            for seed, seed_rows in rows.groupby("seed"):
                values = seed_rows.set_index("reference")["median_u"]
                ax.plot(
                    [0, 1], [values["ablated"], values["full"]], color=color,
                    alpha=0.38, linewidth=0.9, marker="o", markersize=3.0,
                    markeredgecolor="white", markeredgewidth=0.35, clip_on=False,
                )
            means = rows.groupby("reference")["median_u"].mean()
            ax.plot(
                [0, 1], [means["ablated"], means["full"]], color=color,
                linewidth=2.2, marker="o", markersize=5.0, markeredgecolor="#2F3337",
                markeredgewidth=0.55, clip_on=False,
                label=("held-out stimulated" if group == "held_out_stimulated"
                       else "same-type control"),
            )
        ax.set_title(DISPLAY_LABELS[label], fontsize=9.5, pad=3)
        ax.set_xticks([0, 1], ["Ablated", "Full"])
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.grid(axis="y", color=FIGURE_STYLE["grid"], linewidth=0.55)
        ax.tick_params(labelsize=7.3)
    axes[0].set_ylabel(r"Median source deficit $u$", fontsize=8.5)
    split_handle = Line2D([0], [0], color="#555B64", linewidth=0.9, marker="o", markersize=3,
                          label="thin = donor split; thick = arithmetic mean")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles + [split_handle], labels + [split_handle.get_label()], frameon=False,
        fontsize=7.0, loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=1,
        handlelength=2.0, borderaxespad=0.0,
    )


def _destination_category_values(summary: pd.DataFrame) -> pd.DataFrame:
    conditions = ["incomplete_reference", "full_reference_control"]
    values = pd.DataFrame(0.0, index=conditions, columns=DESTINATION_CATEGORIES)
    for condition in conditions:
        subset = summary.loc[summary["condition_id"] == condition]
        if not np.isclose(subset["proportion_mean"].sum(), 1.0, atol=1.0e-9):
            raise PBMCFigureError(f"Destination summary for {condition} does not sum to one.")
        is_b = subset["target_cell_type"].eq(REPRESENTATIVE_LABEL)
        values.loc[condition, "B control"] = float(
            subset.loc[is_b & subset["target_condition"].eq("ctrl"), "proportion_mean"].sum()
        )
        values.loc[condition, "B stimulated"] = float(
            subset.loc[is_b & subset["target_condition"].eq("stim"), "proportion_mean"].sum()
        )
        values.loc[condition, "Other targets"] = float(
            subset.loc[~is_b, "proportion_mean"].sum()
        )
    if not np.allclose(values.sum(axis=1).to_numpy(), 1.0, atol=1.0e-9):
        raise PBMCFigureError("Collapsed destination categories do not sum to one.")
    return values


def _draw_destination(ax: plt.Axes, summary: pd.DataFrame) -> None:
    values = _destination_category_values(summary)
    y = np.array([1.0, 0.0])
    conditions = ("incomplete_reference", "full_reference_control")
    colors = {
        "B control": FIGURE_STYLE["control"],
        "B stimulated": FIGURE_STYLE["stimulated"],
        "Other targets": "#A7ADB5",
    }
    left = np.zeros(2)
    for category in DESTINATION_CATEGORIES:
        segment = values.loc[list(conditions), category].to_numpy(dtype=float)
        ax.barh(y, segment, left=left, height=0.52, color=colors[category],
                edgecolor="white", linewidth=0.65, label=category)
        for yi, start, width in zip(y, left, segment):
            if width >= 0.08:
                text_color = "white" if category != "Other targets" else FIGURE_STYLE["text"]
                ax.text(start + width / 2, yi, f"{100 * width:.0f}%", ha="center", va="center",
                        fontsize=8.0, fontweight="bold", color=text_color, clip_on=False)
        left += segment
    ax.set_title("Conditional destination composition", loc="left", fontsize=9.0, pad=2)
    ax.set_yticks(y, ["Ablated", "Full"])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Destination composition conditional on\npositive transported mass", fontsize=7.2)
    ax.tick_params(labelsize=7.4)
    ax.grid(axis="x", color=FIGURE_STYLE["grid"], linewidth=0.55)
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.35, -0.86),
              ncol=3, handlelength=1.2, columnspacing=0.6, borderaxespad=0.0)


def _draw_mass_retention(ax: plt.Axes, by_seed: pd.DataFrame) -> None:
    for seed, rows in by_seed.groupby("seed", sort=True):
        paired = rows.set_index("reference")[RAW_MASS_COLUMN]
        if not set(REFERENCE_ORDER).issubset(paired.index):
            raise PBMCFigureError(f"Mass-retention seed {seed} is not paired.")
        ax.plot(
            [0, 1], [paired["ablated"], paired["full"]], color=FIGURE_STYLE["diagnostic"],
            alpha=0.35, linewidth=0.9, marker="o", markersize=3.0, clip_on=False,
        )
    for x, reference in enumerate(REFERENCE_ORDER):
        rows = by_seed.loc[by_seed["reference"] == reference]
        ax.scatter(np.full(len(rows), x), rows[RAW_MASS_COLUMN], s=20,
                   color=FIGURE_STYLE["diagnostic"], alpha=0.46, linewidth=0, zorder=2,
                   clip_on=False)
        ax.scatter(x, rows[RAW_MASS_COLUMN].mean(), s=52,
                   color=FIGURE_STYLE["diagnostic"], edgecolor="#202124", linewidth=0.7,
                   zorder=3, clip_on=False)
    upper = max(1.04, float(by_seed[RAW_MASS_COLUMN].max()) * 1.08)
    ax.axhline(1.0, color=FIGURE_STYLE["muted"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_title("Retained source mass", loc="left", fontsize=9.0, pad=2)
    ax.set_xticks([0, 1], ["Ablated", "Full"])
    ax.set_ylabel("Median relative source mass\n" r"$\widehat a_i/a_i$", fontsize=7.5)
    ax.set_ylim(0, upper)
    ax.set_yticks(np.linspace(0, upper, 5))
    ax.grid(axis="y", color=FIGURE_STYLE["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7.4)


def _draw_detection_facets(axes: list[plt.Axes], by_seed: pd.DataFrame) -> None:
    for ax, label in zip(axes, EXPECTED_LABELS):
        subset = by_seed.loc[by_seed["held_out_label"] == label]
        for x, item in enumerate(MAIN_METHOD_SCORES):
            rows = subset.loc[(subset["method"] == item.method) & (subset["score"] == item.score)]
            values = rows["auprc"].to_numpy(dtype=float)
            ax.scatter(np.full(len(values), x), values, s=18, color=METHOD_COLORS[item.method],
                       marker=METHOD_MARKERS[(item.method, item.score)], alpha=0.42, linewidth=0,
                       clip_on=False)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            ax.errorbar(x, mean, yerr=sd, fmt=METHOD_MARKERS[(item.method, item.score)],
                        color=METHOD_COLORS[item.method], markeredgecolor="#222222",
                        markeredgewidth=0.6, markersize=5.8, linewidth=1.0, capsize=2.2,
                        clip_on=False)
        for baseline in subset.groupby("seed")["auprc_baseline"].first():
            ax.axhline(baseline, color=FIGURE_STYLE["baseline"], linewidth=0.8,
                       linestyle=(0, (3, 2)), alpha=0.55, zorder=0)
        ax.set_title(DISPLAY_LABELS[label], fontsize=9.5, pad=3)
        ax.set_xticks(
            range(len(MAIN_METHOD_SCORES)),
            ["CoRe-OT\nu", "scmap-\ncluster", "Seurat", "CHETAH"],
        )
        ax.set_ylim(-0.02, 1.04)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color=FIGURE_STYLE["grid"], linewidth=0.45)
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_ha("right")
        ax.tick_params(labelsize=7.0, pad=5)
    axes[0].set_ylabel("Within-cell-type AUPRC", fontsize=8.5)


def _draw_abstention_facets(axes: list[plt.Axes], by_seed: pd.DataFrame) -> None:
    for ax, label in zip(axes, EXPECTED_LABELS):
        subset = by_seed.loc[by_seed["held_out_label"] == label]
        for item in MAIN_METHOD_SCORES:
            rows = subset.loc[(subset["method"] == item.method) & (subset["score"] == item.score)]
            color = METHOD_COLORS[item.method]
            marker = METHOD_MARKERS[(item.method, item.score)]
            coverage = 1.0 - rows["shared_false_abstention_rate"]
            ax.scatter(coverage, rows["absent_abstention_rate"],
                       s=20, color=color, marker=marker, alpha=0.40, linewidth=0,
                       clip_on=False)
            ax.scatter(coverage.mean(),
                       rows["absent_abstention_rate"].mean(), s=45, color=color, marker=marker,
                       edgecolor="#222222", linewidth=0.7, zorder=3, clip_on=False)
        ax.set_title(DISPLAY_LABELS[label], fontsize=9.5, pad=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.grid(color=FIGURE_STYLE["grid"], linewidth=0.5)
        ax.tick_params(labelsize=7.2)
    axes[2].set_xlabel("Shared-cell coverage\n(= 1 − shared false\nabstention)", fontsize=7.6, labelpad=4)
    axes[3].set_xlabel("Shared-cell coverage\n(= 1 − shared false\nabstention)", fontsize=7.6, labelpad=4)
    axes[0].set_ylabel("Held-out-state abstention", fontsize=8.2)
    axes[0].text(0.57, 0.92, "better ↗", ha="left", va="top", fontsize=7.5,
                 color=FIGURE_STYLE["muted"])


def _draw_forced_macro_f1(ax: plt.Axes, by_seed: pd.DataFrame) -> None:
    for x, label in enumerate(EXPECTED_LABELS):
        rows = by_seed.loc[by_seed["held_out_label"] == label]
        ax.scatter(np.full(len(rows), x), rows["forced_macro_f1"], s=18,
                   color=METHOD_COLORS["coreot_full"], alpha=0.45, linewidth=0,
                   clip_on=False)
        ax.scatter(x, rows["forced_macro_f1"].mean(), s=38,
                   color=METHOD_COLORS["coreot_full"], edgecolor="#222222", linewidth=0.7,
                   clip_on=False)
    ax.set_xticks(range(4), ["B", "NK", "DC", "CD8 T"])
    ax.set_ylim(0.90, 1.005)
    ax.set_yticks([0.90, 0.95, 1.00])
    ax.set_ylabel("CoRe-OT forced macro-F1", fontsize=7.2, labelpad=2)
    ax.text(0.98, 0.92, "zoomed scale: 0.90–1.00", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color=FIGURE_STYLE["muted"])
    ax.grid(axis="y", color=FIGURE_STYLE["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7.2)


def validate_artist_bounds(figure: plt.Figure, *, tolerance_px: float = 4.0) -> None:
    """Fail before export when visible text is clipped by the canvas."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    violations: list[str] = []
    for artist in figure.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer)
        if (
            bbox.x0 < -tolerance_px or bbox.y0 < -tolerance_px
            or bbox.x1 > width + tolerance_px or bbox.y1 > height + tolerance_px
        ):
            violations.append(
                f"{artist.get_text()!r}@({bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f})"
                f" axes={artist.axes.get_position().bounds if artist.axes is not None else None}"
            )
    if violations:
        raise PBMCFigureError(
            f"Visible text falls outside the figure canvas ({width}x{height} px): "
            + ", ".join(violations[:8])
        )


def _write_figure(
    *, umap: pd.DataFrame, restoration: pd.DataFrame, mass_retention: pd.DataFrame,
    detection: pd.DataFrame, destination: pd.DataFrame, abstention: pd.DataFrame,
    forced_macro_f1: pd.DataFrame, png: Path, pdf: Path, svg: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 8.0,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    figure = plt.figure(figsize=(9.4, 10.2), constrained_layout=False, facecolor="white")
    outer = figure.add_gridspec(
        nrows=3, ncols=2, height_ratios=(1.08, 1.08, 1.52),
        width_ratios=(1.65, 1.00), hspace=0.42, wspace=0.26,
    )

    ax_a = figure.add_subplot(outer[0, 0])
    _draw_schematic(ax_a)
    _add_panel_label(ax_a, "A")

    b_container = outer[0, 1].subgridspec(3, 1, height_ratios=(0.24, 0.12, 0.64), hspace=0.04)
    ax_b_header = figure.add_subplot(b_container[0, 0])
    ax_b_header.set_axis_off()
    ax_b_header.text(0.0, 0.75, "B cells, seed 1, prespecified illustration",
                     fontsize=9.5, fontweight="bold", va="center", ha="left")
    ax_b_header.text(0.0, 0.18, "Source-only UMAP; same coordinates\nand B-cell coordinate window",
                     fontsize=7.2, color=FIGURE_STYLE["muted"], va="center", ha="left")
    _add_panel_label(ax_b_header, "B")
    ax_b_legend = figure.add_subplot(b_container[1, 0])
    ax_b_legend.set_axis_off()
    ax_b_legend.set_xlim(0, 1)
    ax_b_legend.set_ylim(0, 1)
    legend_items = (
        (FIGURE_STYLE["atlas"], "other query"),
        (FIGURE_STYLE["control"], "B control"),
        (FIGURE_STYLE["stimulated"], "B stimulated"),
    )
    for x, (color, label) in zip((0.03, 0.30, 0.57), legend_items):
        ax_b_legend.scatter([x], [0.5], s=22, color=color, edgecolor="white", linewidth=0.5)
        ax_b_legend.text(x + 0.025, 0.5, label, ha="left", va="center", fontsize=6.7)
    grid_b = b_container[2, 0].subgridspec(1, 3, wspace=0.15)
    ax_b1 = figure.add_subplot(grid_b[0, 0])
    ax_b2 = figure.add_subplot(grid_b[0, 1])
    ax_b3 = figure.add_subplot(grid_b[0, 2])
    limits = _b_limits(umap)
    _draw_b_zoom(ax_b1, umap, "Biological state", limits)
    _draw_score_map(
        ax_b2, umap, "u_ablated", r"$u^{\mathrm{ablated}}$", limits,
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0), cmap="magma",
        colorbar_ticks=[0.0, 0.5, 1.0],
    )
    delta_limit = max(float(np.max(np.abs(umap.loc[umap["cell_type"].eq(REPRESENTATIVE_LABEL), "delta_u"]))), 1.0e-8)
    _draw_score_map(
        ax_b3, umap, "delta_u", "Restoration\nresponse, $\\Delta u$", limits,
        norm=TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit), cmap="coolwarm",
        colorbar_ticks=[-delta_limit, 0.0, delta_limit],
    )
    ax_b3.text(0.0, -0.23, "positive = deficit\nreduction", transform=ax_b3.transAxes,
               ha="left", va="top", fontsize=6.2, color=FIGURE_STYLE["muted"])

    grid_c = outer[1, 0].subgridspec(1, 4, wspace=0.27)
    c_axes = [figure.add_subplot(grid_c[0, j]) for j in range(4)]
    _draw_restoration_facets(c_axes, restoration)
    _add_panel_label(c_axes[0], "C")

    d_container = outer[1, 1].subgridspec(3, 1, height_ratios=(0.16, 0.78, 1.20), hspace=0.58)
    ax_d_header = figure.add_subplot(d_container[0, 0])
    ax_d_header.set_axis_off()
    ax_d_header.text(0.0, 0.60, "Held-out stimulated B cells", fontsize=10.0,
                     fontweight="bold", va="center", ha="left")
    _add_panel_label(ax_d_header, "D")
    ax_d1 = figure.add_subplot(d_container[1, 0])
    ax_d2 = figure.add_subplot(d_container[2, 0])
    _draw_mass_retention(ax_d1, mass_retention)
    _draw_destination(ax_d2, destination)
    grid_e = outer[2, 0].subgridspec(1, 4, wspace=0.27)
    e_axes = [figure.add_subplot(grid_e[0, j]) for j in range(4)]
    _draw_detection_facets(e_axes, detection)
    _add_panel_label(e_axes[0], "E")
    grid_f = outer[2, 1].subgridspec(3, 2, height_ratios=(1.0, 1.0, 0.55),
                                      hspace=1.40, wspace=0.42)
    f_axes = [figure.add_subplot(grid_f[j // 2, j % 2]) for j in range(4)]
    _draw_abstention_facets(f_axes, abstention)
    ax_f_strip = figure.add_subplot(grid_f[2, :])
    _draw_forced_macro_f1(ax_f_strip, forced_macro_f1)
    _add_panel_label(f_axes[0], "F")
    method_handles = [
        Line2D([0], [0], marker=METHOD_MARKERS[(item.method, item.score)], linestyle="none",
               color=METHOD_COLORS[item.method], label=item.display_name, markersize=7)
        for item in MAIN_METHOD_SCORES
    ]
    baseline_handle = Line2D([0], [0], color=FIGURE_STYLE["baseline"], linewidth=1.0,
                             linestyle=(0, (3, 2)), label="dashed = split prevalence")
    figure.legend(
        handles=method_handles + [baseline_handle], loc="lower center",
        bbox_to_anchor=(0.5, 0.015), ncol=5, frameon=False, fontsize=7.8,
        handletextpad=0.45, columnspacing=1.05,
    )
    figure.subplots_adjust(left=0.075, right=0.975, top=0.965, bottom=0.105)
    validate_artist_bounds(figure)
    figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(svg, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def _render_description(paths: PBMCFigurePaths, validation_report: dict[str, object]) -> str:
    return f"""# Figure 3: Condition-specific weak correspondence and matched-reference rescue

## Generation record

- Generated at (UTC): `{validation_report['generated_at']}`
- Generator: `experiments/pbmc_state/generate_pbmc_condition_specific_figure.py`
- No new transport fit was run; all summaries use preserved run artifacts and
  the existing Figure 3 source tables.
- Validation status: **all figure-data assertions passed before export**.

## Files

- PNG: `{paths.png}`
- PDF: `{paths.pdf}`
- SVG: `{paths.svg}`

## Source data

- Seed-1 source-query UMAP and scores: `{paths.umap_scores}`
- Within-cell-type metrics by seed: `{paths.detection_by_seed}`
- Within-cell-type metric summary: `{paths.detection_summary}`
- Transport destinations by seed: `{paths.destination_by_seed}`
- Transport-destination summary: `{paths.destination_summary}`
- Restoration summaries by seed: `{paths.restoration_by_seed}`
- Restoration summary: `{paths.restoration_summary}`
- Relative transported source mass by seed: `{paths.mass_retention_by_seed}`
- Relative transported source mass summary: `{paths.mass_retention_summary}`
- Abstention trade-offs by seed: `{paths.abstention_by_seed}`
- Abstention trade-off summary: `{paths.abstention_summary}`
- CoRe-OT forced macro-F1 by seed: `{paths.forced_macro_f1_by_seed}`
- CoRe-OT forced macro-F1 summary: `{paths.forced_macro_f1_summary}`

## Panel definitions

- **A:** The same query cells are mapped in two separate matched CoRe-OT runs,
  ordered `Ablated -> Full`; only stimulated cells of the selected state are
  removed in the ablated reference, while same-type control cells remain.
- **B:** Prespecified stimulated B cells, donor-split seed 1. Coordinates are
  generated from a source-only UMAP before evaluation truth or score columns are
  joined. All three views use the same score-independent 2nd--98th percentile
  B-cell coordinate window. The biological map plots other query cells first in
  light gray. The raw score map uses a fixed `[0, 1]` color scale. The
  restoration map uses a symmetric diverging scale with
  `limit = max(abs(Delta u)) = {float(validation_report['delta_limit']):.6f}`;
  `{int(validation_report['delta_boundary_count'])}` displayed point(s) define
  the absolute boundary and no value is silently clipped. Positive `Delta u`
  means deficit reduction after restoration.
- **C:** Within each state and donor split, cell-level values are first reduced
  to a median for held-out stimulated cells and same-type controls. Thin lines
  are the five donor splits; thick lines and outlined points are arithmetic
  means of split-level medians. The order is `Ablated -> Full`.
- **D:** The left component is the raw relative transported source mass
  `a_hat / a`, not `1 - u`; `u` uses the positive-part deficit definition
  `[a - a_hat]_+ / (a + eta)`, so these quantities are not interchangeable.
  The maximum displayed raw ratio is
  `{float(validation_report['mass_max']):.6f}` and no value is capped. The
  right component is destination composition conditional on positive
  transported mass. Each eligible source row is normalized to sum to one,
  source cells are averaged equally within each seed, and the five seed-level
  compositions are then averaged equally. Zero-transport sources are excluded
  from row normalization and recorded in the destination source table; the
  maximum recorded zero-source count is
  `{int(validation_report['n_zero_transport_sources'])}`.
- **E:** Within-cell-type AUPRC uses the fixed method order `CoRe-OT u`,
  `scmap-cluster`, `Seurat`, `CHETAH`. Points are the five donor splits,
  larger outlined markers are arithmetic means, and intervals are mean ± sample
  SD with `ddof=1`. Dashed lines are split-specific prevalence baselines.
  External method scores are method-specific diagnostics, not estimates of the
  same physical quantity as CoRe-OT `u`.
- **F:** The upper facets show held-out-state abstention versus shared-cell
  coverage `1 - shared false abstention`, with both axes on the 0--1 domain.
  The lower strip contains only CoRe-OT forced macro-F1 on an explicitly
  zoomed 0.90--1.00 scale. All markers are donor-split values or arithmetic
  means; no population uncertainty is implied.

## State and method order

States are fixed as `B cells`, `NK cells`, `Dendritic cells`, `CD8 T cells`.
The compact comparison is fixed as `CoRe-OT u`, `scmap-cluster`, `Seurat`,
`CHETAH`. The B-cell maps and destination view are prespecified
paired-reference illustrations, not independent validation and not component
attribution. The figure supports a controlled restoration claim only; it does
not prove biological absence or isolate the effects of individual CoRe-OT
components.

## Change log

- Rebuilt Panel A as two parallel matched runs with `Ablated -> Full` order,
  explicit crossed-out B-stimulated state, and corrected benchmark-only footer.
- Added the prespecified B-cell title/subtitle, non-overlapping biological
  labels, fixed score scales, and signed restoration annotation.
- Reworked Panel D as a readable retained-mass plot plus horizontal conditional
  destination bars. The mass table was regenerated from preserved transport
  `a_hat` and `a` columns; no fitted artifact changed.
- Improved Panels E--F labels, method palette, prevalence key, boundary-marker
  visibility, shared coverage labels, and the zoomed forced macro-F1 strip.
- Added source-table, paired-reference, arithmetic mean/SD, destination-sum,
  coordinate, score-sign, expected-summary, and artist-bound assertions.
"""


def write_pbmc_condition_specific_figure(
    *, runs_root: Path = Path("runs"), raw_data_path: Path = Path("data/raw/kang_2018.h5ad"),
    comparison_path: Path = Path("results/PBMC/compare_baselines/tables/compare_detection_by_run.csv"),
    shared_transfer_path: Path = Path(
        "results/PBMC/compare_baselines/tables/compare_shared_label_transfer_by_run.csv"
    ),
    output_root: Path = Path("results/PBMC/figures"), condition: str = "incomplete_reference",
    internal_candidate_set: str = "pca30_k100",
    external_candidate_set: str = "external_reference_mapping", metric_tolerance: float = 1e-10,
) -> PBMCFigurePaths:
    paths = _paths(output_root)
    paths.data_root.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(comparison_path)
    _validate_comparison_coverage(comparison)
    raw = ad.read_h5ad(raw_data_path, backed="r")
    metadata = raw.obs[["cell_type", "label"]].rename(columns={"label": "condition"}).copy()
    metadata.index = metadata.index.astype(str)

    detection_rows: list[dict[str, object]] = []
    destination_rows: list[pd.DataFrame] = []
    restoration_rows: list[pd.DataFrame] = []
    mass_retention_rows: list[dict[str, object]] = []
    representative: tuple[Path, pd.DataFrame] | None = None
    for label in EXPECTED_LABELS:
        for seed in EXPECTED_SEEDS:
            cell_sets: list[set[str]] = []
            for item in METHOD_SCORES:
                expected = comparison.loc[
                    (comparison["held_out_label"] == label) & (comparison["seed"] == seed)
                    & (comparison["method"] == item.method) & (comparison["score"] == item.score)
                ]
                if len(expected) != 1:
                    raise PBMCFigureError(f"Expected one comparison row for {label}, seed {seed}, {item}.")
                expected_row = expected.iloc[0]
                run_root = runs_root / str(expected_row["run_id"])
                truth = _read_truth(run_root, condition)
                scores = _read_scores(run_root, item, condition, internal_candidate_set,
                                      external_candidate_set)
                cell_sets.append(set(truth["cell_id"].astype(str)))
                metrics = compute_within_celltype_metrics(truth, scores, metadata, label, item.score)
                joined = truth[["cell_id", "is_absent_state"]].merge(
                    scores, on="cell_id", validate="one_to_one"
                )
                y_true = joined["is_absent_state"].astype(bool).to_numpy()
                y_score = joined[item.score].to_numpy(dtype=float)
                global_auroc = roc_auc_score(y_true, y_score)
                global_auprc = average_precision_score(y_true, y_score)
                if not np.isclose(global_auroc, float(expected_row["auroc"]), atol=metric_tolerance, rtol=0):
                    raise PBMCFigureError(f"Global AUROC mismatch for {label}, seed {seed}, {item}.")
                if not np.isclose(global_auprc, float(expected_row["auprc"]), atol=metric_tolerance, rtol=0):
                    raise PBMCFigureError(f"Global AUPRC mismatch for {label}, seed {seed}, {item}.")
                detection_rows.append({
                    "held_out_label": label, "seed": seed, "run_id": expected_row["run_id"],
                    "display_name": item.display_name, "method": item.method, "score": item.score,
                    **metrics,
                })
                if item.family == "internal" and item.score == "u":
                    truth_ids = set(truth["cell_id"].astype(str))
                    ablated_all = pd.read_parquet(
                        run_root / "scoring" / condition / internal_candidate_set
                        / "cell_scores.parquet"
                    )
                    full_all = pd.read_parquet(
                        run_root / "scoring" / "full_reference_control"
                        / internal_candidate_set / "cell_scores.parquet"
                    )
                    ablated_transport = _read_transport_scores(
                        run_root, condition, internal_candidate_set, truth_ids
                    )
                    full_transport = _read_transport_scores(
                        run_root, "full_reference_control", internal_candidate_set, truth_ids
                    )
                    transport_pairs = ablated_transport.merge(
                        full_transport,
                        on="cell_id",
                        suffixes=("_ablated", "_full"),
                        validate="one_to_one",
                    )
                    for column in ("a", "rho", "tau_source"):
                        if not np.allclose(
                            transport_pairs[f"{column}_ablated"],
                            transport_pairs[f"{column}_full"],
                            rtol=0.0,
                            atol=1.0e-12,
                        ):
                            raise PBMCFigureError(
                                f"Paired source/prior values differ for {label}, seed {seed}, {column}."
                            )
                    for score_frame, score_condition in (
                        (ablated_all, condition), (full_all, "full_reference_control")
                    ):
                        coreot = score_frame.loc[
                            score_frame["method"] == "coreot_full", ["cell_id", "u", "u_tilde"]
                        ]
                        if (
                            not coreot["cell_id"].is_unique
                            or set(coreot["cell_id"].astype(str))
                            != set(truth["cell_id"].astype(str))
                            or not np.isfinite(coreot[["u", "u_tilde"]].to_numpy(dtype=float)).all()
                        ):
                            raise PBMCFigureError(
                                "Ablated/full CoRe-OT score validation failed for "
                                f"{label}, seed {seed}, {score_condition}."
                            )
                    restoration = compute_restoration_summaries(
                        truth, ablated_all, full_all, metadata, label
                    )
                    restoration.insert(0, "run_id", str(expected_row["run_id"]))
                    restoration.insert(0, "seed", seed)
                    restoration.insert(0, "held_out_label", label)
                    restoration_rows.append(restoration)
                if item.family == "internal" and item.score == "u" and label == REPRESENTATIVE_LABEL:
                    if seed == REPRESENTATIVE_SEED:
                        representative = (run_root, truth)
                    positive_ids = set(truth.loc[truth["is_absent_state"].astype(bool), "cell_id"].astype(str))
                    for reference, transport_frame in (
                        ("ablated", ablated_transport), ("full", full_transport)
                    ):
                        selected = transport_frame.loc[
                            transport_frame["cell_id"].isin(positive_ids), RAW_MASS_COLUMN
                        ].to_numpy(dtype=float)
                        if len(selected) != len(positive_ids):
                            raise PBMCFigureError("Mass-retention scores do not cover held-out B cells.")
                        mass_retention_rows.append({
                            "held_out_label": label, "seed": seed,
                            "run_id": str(expected_row["run_id"]), "reference": reference,
                            "n_cells": len(selected),
                            RAW_MASS_COLUMN: float(np.median(selected)),
                        })
                    for condition_id in ("incomplete_reference", "full_reference_control"):
                        coupling_path = (
                            run_root / "transport" / condition_id / internal_candidate_set
                            / "coreot_full" / "sparse_coupling.parquet"
                        )
                        composition = compute_destination_composition(
                            pd.read_parquet(coupling_path), positive_ids, metadata
                        )
                        composition.insert(0, "condition_id", condition_id)
                        composition.insert(0, "seed", seed)
                        destination_rows.append(composition)
            if any(cell_set != cell_sets[0] for cell_set in cell_sets[1:]):
                raise PBMCFigureError(f"Internal and external query cell sets differ for {label}, seed {seed}.")

    detection = pd.DataFrame(detection_rows)
    detection_summary = _summarize(
        detection,
        ["held_out_label", "display_name", "method", "score"],
        ["auroc", "auprc", "auprc_baseline"],
    )
    destination = pd.concat(destination_rows, ignore_index=True)
    destination_summary = _summarize(
        destination, ["condition_id", "target_cell_type", "target_condition"], ["proportion"]
    )
    restoration = pd.concat(restoration_rows, ignore_index=True)
    paired_restoration = restoration.pivot(
        index=["held_out_label", "seed", "run_id", "cell_group"],
        columns="reference", values="median_u",
    ).reset_index()
    paired_restoration["delta_median_u"] = (
        paired_restoration["ablated"] - paired_restoration["full"]
    )
    restoration = restoration.merge(
        paired_restoration[
            ["held_out_label", "seed", "run_id", "cell_group", "delta_median_u"]
        ],
        on=["held_out_label", "seed", "run_id", "cell_group"],
        validate="many_to_one",
    )
    restoration_summary = _summarize(
        restoration, ["held_out_label", "cell_group", "reference"],
        ["median_u", "delta_median_u"],
    )
    mass_retention = pd.DataFrame(mass_retention_rows)
    mass_retention_summary = _summarize(
        mass_retention, ["held_out_label", "reference"], [RAW_MASS_COLUMN]
    )
    displayed_pairs = {(item.method, item.score) for item in MAIN_METHOD_SCORES}
    abstention = comparison.loc[
        comparison.apply(
            lambda row: (row["method"], row["score"]) in displayed_pairs,
            axis=1,
        ),
        [
        "held_out_label", "seed", "run_id", "method", "score",
        "absent_abstention_rate", "shared_false_abstention_rate",
        ],
    ].copy()
    display_lookup = {(item.method, item.score): item.display_name for item in METHOD_SCORES}
    abstention["display_name"] = [display_lookup[(m, s)] for m, s in zip(abstention["method"], abstention["score"])]
    abstention_summary = _summarize(
        abstention, ["held_out_label", "display_name", "method", "score"],
        ["absent_abstention_rate", "shared_false_abstention_rate"],
    )
    shared_transfer = pd.read_csv(shared_transfer_path)
    required_shared = {"held_out_label", "seed", "condition_id", "method", "score", "forced_macro_f1"}
    missing_shared = required_shared - set(shared_transfer.columns)
    if missing_shared:
        raise PBMCFigureError(f"Shared-transfer table is missing columns {sorted(missing_shared)}.")
    forced_macro_f1 = shared_transfer.loc[
        (shared_transfer["condition_id"] == condition)
        & (shared_transfer["method"] == "coreot_full")
        & (shared_transfer["score"] == "u")
        & shared_transfer["held_out_label"].isin(EXPECTED_LABELS),
        ["held_out_label", "seed", "run_id", "forced_macro_f1"],
    ].copy()
    expected_forced = {(label, seed) for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS}
    actual_forced = set(forced_macro_f1[["held_out_label", "seed"]].itertuples(index=False, name=None))
    if actual_forced != expected_forced or forced_macro_f1.duplicated(["held_out_label", "seed"]).any():
        raise PBMCFigureError("Forced macro-F1 rows do not cover four labels by five seeds exactly.")
    forced_macro_f1_summary = _summarize(
        forced_macro_f1, ["held_out_label"], ["forced_macro_f1"]
    )

    if representative is None:
        raise PBMCFigureError("Representative B-cell seed-1 run was not discovered.")
    representative_root, representative_truth = representative
    split = pd.read_csv(representative_root / "benchmark" / "split_manifest.csv")
    query_ids = split.loc[split["split_domain"] == "query", "cell_id"].astype(str).tolist()
    if set(query_ids) != set(representative_truth["cell_id"].astype(str)):
        raise PBMCFigureError("Seed-1 split membership does not match evaluation truth cell set.")
    ablated = pd.read_parquet(
        representative_root / "scoring" / "incomplete_reference" / internal_candidate_set / "cell_scores.parquet"
    )
    full = pd.read_parquet(
        representative_root / "scoring" / "full_reference_control" / internal_candidate_set / "cell_scores.parquet"
    )
    umap = _compute_umap_scores(
        raw_data_path=raw_data_path, query_ids=query_ids, truth=representative_truth,
        ablated_scores=ablated, full_scores=full,
    )
    validation_report = validate_figure_data(
        umap=umap,
        detection=detection,
        detection_summary=detection_summary,
        restoration=restoration,
        restoration_summary=restoration_summary,
        destination=destination,
        destination_summary=destination_summary,
        mass_retention=mass_retention,
        mass_retention_summary=mass_retention_summary,
        abstention=abstention,
        abstention_summary=abstention_summary,
        forced_macro_f1=forced_macro_f1,
        forced_macro_f1_summary=forced_macro_f1_summary,
    )

    detection.to_csv(paths.detection_by_seed, index=False)
    detection_summary.to_csv(paths.detection_summary, index=False)
    destination.to_csv(paths.destination_by_seed, index=False)
    destination_summary.to_csv(paths.destination_summary, index=False)
    restoration.to_csv(paths.restoration_by_seed, index=False)
    restoration_summary.to_csv(paths.restoration_summary, index=False)
    mass_retention.to_csv(paths.mass_retention_by_seed, index=False)
    mass_retention_summary.to_csv(paths.mass_retention_summary, index=False)
    abstention.to_csv(paths.abstention_by_seed, index=False)
    abstention_summary.to_csv(paths.abstention_summary, index=False)
    forced_macro_f1.to_csv(paths.forced_macro_f1_by_seed, index=False)
    forced_macro_f1_summary.to_csv(paths.forced_macro_f1_summary, index=False)
    umap.to_csv(paths.umap_scores, index=False)
    _write_figure(
        umap=umap, restoration=restoration, mass_retention=mass_retention,
        detection=detection, destination=destination_summary, abstention=abstention,
        forced_macro_f1=forced_macro_f1, png=paths.png, pdf=paths.pdf, svg=paths.svg,
    )
    paths.description.write_text(_render_description(paths, validation_report), encoding="utf-8")
    return paths
