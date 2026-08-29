from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import subprocess
from typing import Callable, Iterable

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from coreot.results.compare_baselines import select_primary_comparison_rows
from coreot.results.matched_reference import (
    CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE,
    TYPICAL_CELL_CONDITIONAL_DESTINATION,
    control_adjusted_median_deficit_decrease,
)


PROJECT_ROOT = Path(".")
DOCS_ROOT = PROJECT_ROOT / "docs"
RESULT_ROOT = PROJECT_ROOT / "results/PBMC/manuscript/figure3"
SOURCE_DATA_ROOT = RESULT_ROOT / "figure_3_pbmc_source_data"
REVIEW_ROOT = PROJECT_ROOT / "results/PBMC/manuscript/figure3_panels"
RUNS_ROOT = PROJECT_ROOT / "runs"
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/kang_2018.h5ad"
COMPARISON_DETECTION_PATH = (
    PROJECT_ROOT / "results/PBMC/compare_baselines/tables/compare_detection_by_run.csv"
)
COMPARISON_LABEL_TRANSFER_PATH = (
    PROJECT_ROOT / "results/PBMC/compare_baselines/tables/compare_shared_label_transfer_by_run.csv"
)
FIXED_QUERY_UMAP_PATH = PROJECT_ROOT / "results/PBMC/figures/data/figure_3_seed1_umap_scores.csv"
BENCHMARK_CONFIG_PATH = PROJECT_ROOT / "experiments/pbmc_state/configs/benchmark.yaml"
INTERNAL_CANDIDATE_SET = "pca30_k100"
EXTERNAL_CANDIDATE_SET = "external_reference_mapping"
INCOMPLETE_REFERENCE = "incomplete_reference"
FULL_REFERENCE = "full_reference_control"
TRANSPORT_ETA = 1.0e-12

ENDPOINTS = ("B cells", "NK cells", "Dendritic cells")
SEEDS = (1, 2, 3, 4, 5)
ENDPOINT_DISPLAY = {
    "B cells": "Reference-omitted stimulated B cells",
    "NK cells": "Reference-omitted stimulated NK cells",
    "Dendritic cells": "Reference-omitted stimulated DCs",
}
ENDPOINT_SHORT = {
    "B cells": "B",
    "NK cells": "NK",
    "Dendritic cells": "DC",
}

PANEL_B_METHODS = (
    ("coreot_full", "u", "CoRe-OT"),
    ("scdot", "z_absent_score", "scDOT"),
    ("tacco_ot", "z_absent_score", "TACCO-OT"),
    ("seurat_anchor", "u", "Seurat"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
PANEL_B_AP_LIMITS = (0.0, 1.0)
PANEL_B_AP_DISPLAY_LIMITS = (-0.025, 1.025)
PANEL_B_AP_TICKS = (0.0, 0.5, 1.0)
PANEL_C_PROBABILITY_DISPLAY_LIMITS = (-0.03, 1.08)
PANEL_C_METHODS = (
    ("coreot_full", "u"),
    ("coreot_match_only", "u"),
    ("uniform_uot", "u"),
    ("prior_only", "prior_risk"),
)
PANEL_C_CONTRASTS = (
    ("delta_full_match_only", "CoRe-OT − CoRe-OT (−C)"),
    ("delta_match_only_uniform", "CoRe-OT (−C) − Uniform UOT"),
    ("delta_full_prior", "CoRe-OT − Prior-only"),
)
PANEL_F_METHODS = (
    ("coreot_full", "u", "CoRe-OT"),
    ("scdot", "z_absent_score", "scDOT"),
    ("tacco_ot", "z_absent_score", "TACCO-OT"),
    ("seurat_anchor", "u", "Seurat"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
PANEL_D_METHODS = PANEL_F_METHODS
PANEL_D_LIM = (0.5, 1.0)
PANEL_D_METRICS = (
    ("forced_macro_f1", "Forced macro-F1", "#4C78A8"),
    ("forced_accuracy", "Forced accuracy", "#F2A65A"),
)
PANEL_E_METHODS = PANEL_B_METHODS
PANEL_D_SEED = 1
PANEL_E_MAPS = (
    ("truth", "Reference-omitted cells"),
    *((method, display) for method, _, display in PANEL_E_METHODS),
)
PANEL_F_MAPS = (
    ("truth", "Evaluation labels"),
    *((method, display) for method, _, display in PANEL_F_METHODS),
)

CELL_TYPE_COLORS = {
    "B cells": "#4C78A8",
    "CD14+ Monocytes": "#F58518",
    "CD4 T cells": "#54A24B",
    "CD8 T cells": "#E45756",
    "Dendritic cells": "#B279A2",
    "FCGR3A+ Monocytes": "#72B7B2",
    "Megakaryocytes": "#FF9DA6",
    "NK cells": "#9D755D",
    "Held-out state (not evaluated)": "#D9D9D9",
}

METHOD_COLORS = {
    "coreot_full": "#0072B2",
    "uniform_uot": "#D55E00",
    "coreot_match_only": "#009E73",
    "prior_only": "#009E73",
    "scdot": "#D55E00",
    "tacco_ot": "#009E73",
    "seurat_anchor": "#CC79A7",
    "scmap_cluster": "#E69F00",
    "chetah": "#6A3D9A",
}
METHOD_MARKERS = {
    "coreot_full": "o",
    "uniform_uot": "s",
    "coreot_match_only": "^",
    "prior_only": "D",
    "scdot": "s",
    "tacco_ot": "^",
    "seurat_anchor": "s",
    "scmap_cluster": "v",
    "chetah": "D",
}
CONTRAST_COLORS = {
    "delta_full_match_only": "#009E73",
    "delta_match_only_uniform": "#4D4D4D",
    "delta_full_prior": "#6F58A8",
}
CONTRAST_MARKERS = {
    "delta_full_match_only": "o",
    "delta_match_only_uniform": "s",
    "delta_full_prior": "D",
}

INK = "#242424"
MUTED_INK = "#5E5E5E"
LIGHT_EDGE = "#A8A8A8"
GRID = "#E3E3E3"
QUERY_FILL = "#DCECF5"
REFERENCE_FILL = "#F2F2F2"
CONTROL = "#377EB8"
STIMULATED = "#D55E00"
RESTORED = "#009E73"
PANEL_E_ELIGIBLE_GRAY = "#C8C8C8"
PANEL_E_CONTEXT_GRAY = "#E8E8E8"
PANEL_E_TRUTH_COLOR = "#D73027"

PANEL_RASTER_DPI = 350
PANEL_SIZES = {
    "A": (178 / 25.4, 48 / 25.4),
    "B": (178 / 25.4, 72 / 25.4),
    "C": (178 / 25.4, 66 / 25.4),
    "D": (178 / 25.4, 76 / 25.4),
    "E": (178 / 25.4, 82 / 25.4),
    "F": (178 / 25.4, 82 / 25.4),
}
MAIN_FIGURE_SIZE_INCHES = (178 / 25.4, 230 / 25.4)
MAIN_FIGURE_MIN_FONT_SIZE = 5.0


class PBMCFigure3Error(ValueError):
    """Raised when retained artifacts cannot support PBMC Figure 3."""


@dataclass(frozen=True)
class RunDescriptor:
    endpoint: str
    seed: int
    run_id: str
    root: Path


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path | str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PBMCFigure3Error(f"{path} is missing columns {missing}.")


def _require_exact_keys(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    expected: set[tuple[object, ...]],
    label: str,
) -> None:
    if frame.duplicated(columns).any():
        duplicates = frame.loc[frame.duplicated(columns, keep=False), columns]
        raise PBMCFigure3Error(
            f"{label} contains duplicate keys: {duplicates.to_dict(orient='records')}"
        )
    observed = set(frame[columns].itertuples(index=False, name=None))
    if observed != expected:
        raise PBMCFigure3Error(
            f"{label} keys differ from the figure contract; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _read_raw_metadata(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    raw = ad.read_h5ad(path, backed="r")
    required = {"cell_type", "label", "replicate"}
    missing = required - set(raw.obs.columns)
    if missing:
        raise PBMCFigure3Error(f"{path} is missing raw metadata columns {sorted(missing)}.")
    metadata = raw.obs.loc[:, ["cell_type", "label", "replicate"]].copy()
    metadata = metadata.rename(columns={"label": "condition", "replicate": "donor"})
    metadata.index = metadata.index.astype(str)
    if metadata.index.duplicated().any():
        raise PBMCFigure3Error(f"{path} contains duplicate cell identifiers.")
    if not set(metadata["condition"].astype(str)).issubset({"ctrl", "stim"}):
        raise PBMCFigure3Error(f"{path} contains unexpected PBMC condition labels.")
    return metadata


def _read_comparison(path: Path = COMPARISON_DETECTION_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {
            "run_id",
            "held_out_label",
            "seed",
            "method",
            "score",
            "auprc",
            "auroc",
        },
        path,
    )
    return select_primary_comparison_rows(frame, source=path)


def _comparison_row(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    seed: int,
    method: str,
    score: str,
) -> pd.Series:
    selected = frame.loc[
        frame["held_out_label"].eq(endpoint)
        & frame["seed"].eq(seed)
        & frame["method"].eq(method)
        & frame["score"].eq(score)
    ]
    if len(selected) != 1:
        raise PBMCFigure3Error(
            "Expected one comparison row for "
            f"{endpoint}, seed {seed}, method={method}, score={score}; found {len(selected)}."
        )
    return selected.iloc[0]


def _selected_internal_runs(
    comparison: pd.DataFrame,
    *,
    runs_root: Path = RUNS_ROOT,
) -> tuple[RunDescriptor, ...]:
    descriptors = []
    for endpoint in ENDPOINTS:
        for seed in SEEDS:
            row = _comparison_row(
                comparison,
                endpoint=endpoint,
                seed=seed,
                method="coreot_full",
                score="u",
            )
            root = runs_root / str(row["run_id"])
            if not root.is_dir():
                raise PBMCFigure3Error(f"Selected PBMC run root is missing: {root}")
            descriptors.append(
                RunDescriptor(
                    endpoint=endpoint,
                    seed=seed,
                    run_id=str(row["run_id"]),
                    root=root,
                )
            )
    return tuple(descriptors)


def _read_truth(root: Path, condition: str = INCOMPLETE_REFERENCE) -> pd.DataFrame:
    path = root / "benchmark" / condition / "evaluation_truth" / "query_truth.csv"
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {"cell_id", "true_label", "is_absent_state", "is_shared_state"},
        path,
    )
    frame["cell_id"] = frame["cell_id"].astype(str)
    if frame["cell_id"].duplicated().any():
        raise PBMCFigure3Error(f"{path} contains duplicate query cells.")
    return frame


def _read_scores(root: Path, condition: str, candidate_set: str) -> pd.DataFrame:
    path = root / "scoring" / condition / candidate_set / "cell_scores.parquet"
    frame = pd.read_parquet(path)
    _require_columns(frame, {"cell_id", "method"}, path)
    frame["cell_id"] = frame["cell_id"].astype(str)
    return frame


def _method_scores(
    frame: pd.DataFrame,
    *,
    method: str,
    score: str,
    expected_cell_ids: set[str],
    source: str,
) -> pd.DataFrame:
    _require_columns(frame, {"cell_id", "method", score}, source)
    selected = frame.loc[frame["method"].eq(method), ["cell_id", score]].copy()
    if selected["cell_id"].duplicated().any():
        raise PBMCFigure3Error(f"{source} contains duplicate {method} score rows.")
    if set(selected["cell_id"]) != expected_cell_ids:
        raise PBMCFigure3Error(f"{source} has a query-cell mismatch for {method}/{score}.")
    values = selected[score].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PBMCFigure3Error(f"{source} contains non-finite {method}/{score} values.")
    return selected


def compute_within_celltype_metrics(
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    endpoint: str,
    score_column: str,
) -> dict[str, float | int]:
    if not truth["cell_id"].is_unique or not scores["cell_id"].is_unique:
        raise PBMCFigure3Error("Truth and score cell identifiers must be unique.")
    if set(truth["cell_id"]) != set(scores["cell_id"]):
        raise PBMCFigure3Error("Truth and score cell sets must match exactly.")
    joined = truth.loc[:, ["cell_id", "true_label", "is_absent_state"]].merge(
        scores, on="cell_id", validate="one_to_one"
    )
    raw = metadata.reindex(joined["cell_id"])
    if raw[["cell_type", "condition"]].isna().any().any():
        raise PBMCFigure3Error("Some query cells do not join to raw PBMC metadata.")
    joined["raw_cell_type"] = raw["cell_type"].astype(str).to_numpy()
    joined["condition"] = raw["condition"].astype(str).to_numpy()
    if not joined["true_label"].astype(str).eq(joined["raw_cell_type"]).all():
        raise PBMCFigure3Error("Evaluation truth and raw cell-type metadata disagree.")
    expected_positive = joined["raw_cell_type"].eq(endpoint) & joined["condition"].eq("stim")
    if not np.array_equal(
        expected_positive.to_numpy(),
        joined["is_absent_state"].astype(bool).to_numpy(),
    ):
        raise PBMCFigure3Error(
            f"Condition-specific positive labels disagree with raw metadata for {endpoint}."
        )
    local = joined.loc[
        joined["raw_cell_type"].eq(endpoint) & joined["condition"].isin(("ctrl", "stim"))
    ].copy()
    y_true = local["condition"].eq("stim").to_numpy(dtype=bool)
    y_score = local[score_column].to_numpy(dtype=float)
    if y_true.sum() == 0 or (~y_true).sum() == 0:
        raise PBMCFigure3Error(f"Within-cell-type AP is undefined for {endpoint}.")
    return {
        "n_positive": int(y_true.sum()),
        "n_negative": int((~y_true).sum()),
        "prevalence": float(y_true.mean()),
        "auprc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
    }


def _validate_global_metrics(
    *,
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    score_column: str,
    expected_row: pd.Series,
    label: str,
    tolerance: float = 1.0e-10,
) -> None:
    joined = truth.loc[:, ["cell_id", "is_absent_state"]].merge(
        scores, on="cell_id", validate="one_to_one"
    )
    y_true = joined["is_absent_state"].astype(bool).to_numpy()
    y_score = joined[score_column].to_numpy(dtype=float)
    actual_ap = average_precision_score(y_true, y_score)
    actual_auroc = roc_auc_score(y_true, y_score)
    if not np.isclose(actual_ap, float(expected_row["auprc"]), atol=tolerance, rtol=0):
        raise PBMCFigure3Error(f"Global AP lineage check failed for {label}.")
    if not np.isclose(actual_auroc, float(expected_row["auroc"]), atol=tolerance, rtol=0):
        raise PBMCFigure3Error(f"Global AUROC lineage check failed for {label}.")


def collect_panel_b_data(
    *,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    comparison = _read_comparison(comparison_path)
    metadata = _read_raw_metadata(raw_data_path)
    rows: list[dict[str, object]] = []
    source_paths = {str(comparison_path), str(raw_data_path)}
    for endpoint in ENDPOINTS:
        for seed in SEEDS:
            query_sets: list[set[str]] = []
            for method, score, display in PANEL_B_METHODS:
                expected = _comparison_row(
                    comparison,
                    endpoint=endpoint,
                    seed=seed,
                    method=method,
                    score=score,
                )
                root = runs_root / str(expected["run_id"])
                candidate_set = (
                    EXTERNAL_CANDIDATE_SET
                    if method in {"seurat_anchor", "scmap_cluster", "chetah"}
                    else INTERNAL_CANDIDATE_SET
                )
                truth = _read_truth(root)
                all_scores = _read_scores(root, INCOMPLETE_REFERENCE, candidate_set)
                selected = _method_scores(
                    all_scores,
                    method=method,
                    score=score,
                    expected_cell_ids=set(truth["cell_id"]),
                    source=str(root),
                )
                _validate_global_metrics(
                    truth=truth,
                    scores=selected,
                    score_column=score,
                    expected_row=expected,
                    label=f"{endpoint}/{seed}/{method}",
                )
                query_sets.append(set(truth["cell_id"]))
                metrics = compute_within_celltype_metrics(
                    truth,
                    selected,
                    metadata,
                    endpoint=endpoint,
                    score_column=score,
                )
                rows.append(
                    {
                        "endpoint": endpoint,
                        "seed": seed,
                        "run_id": str(expected["run_id"]),
                        "method": method,
                        "score": score,
                        "display_name": display,
                        "score_orientation": "larger_is_weaker_reference_support",
                        **metrics,
                    }
                )
                source_paths.add(str(root))
            if any(query_set != query_sets[0] for query_set in query_sets[1:]):
                raise PBMCFigure3Error(
                    f"Panel B internal and external query cells differ for {endpoint}, seed {seed}."
                )
    by_seed = pd.DataFrame(rows)
    expected_keys = {
        (endpoint, seed, method, score)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for method, score, _ in PANEL_B_METHODS
    }
    _require_exact_keys(
        by_seed,
        columns=["endpoint", "seed", "method", "score"],
        expected=expected_keys,
        label="Panel B source data",
    )
    summary = (
        by_seed.groupby(["endpoint", "method", "score", "display_name"], sort=False)
        .agg(
            mean_auprc=("auprc", "mean"),
            sample_sd_auprc=("auprc", lambda values: values.std(ddof=1)),
            mean_auroc=("auroc", "mean"),
            sample_sd_auroc=("auroc", lambda values: values.std(ddof=1)),
            mean_prevalence=("prevalence", "mean"),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(5).all():
        raise PBMCFigure3Error("Panel B summaries must contain five fixed splits.")
    return by_seed, summary, sorted(source_paths)


def collect_panel_c_data(
    *,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    comparison = _read_comparison(comparison_path)
    metadata = _read_raw_metadata(raw_data_path)
    rows = []
    source_paths = {str(comparison_path), str(raw_data_path)}
    for descriptor in _selected_internal_runs(comparison, runs_root=runs_root):
        truth = _read_truth(descriptor.root)
        truth_ids = set(truth["cell_id"])
        all_scores = _read_scores(
            descriptor.root,
            INCOMPLETE_REFERENCE,
            INTERNAL_CANDIDATE_SET,
        )
        metrics: dict[str, float] = {}
        cohort_signature: tuple[int, int, float] | None = None
        for method, score in PANEL_C_METHODS:
            selected = _method_scores(
                all_scores,
                method=method,
                score=score,
                expected_cell_ids=truth_ids,
                source=str(descriptor.root),
            )
            local = compute_within_celltype_metrics(
                truth,
                selected,
                metadata,
                endpoint=descriptor.endpoint,
                score_column=score,
            )
            current_signature = (
                int(local["n_positive"]),
                int(local["n_negative"]),
                float(local["prevalence"]),
            )
            if cohort_signature is None:
                cohort_signature = current_signature
            elif current_signature != cohort_signature:
                raise PBMCFigure3Error(
                    f"Panel C methods use different local cohorts: {descriptor.run_id}"
                )
            metrics[method] = float(local["auprc"])
        if cohort_signature is None:
            raise PBMCFigure3Error(f"Panel C has no local cohort: {descriptor.run_id}")
        rows.append(
            {
                "endpoint": descriptor.endpoint,
                "seed": descriptor.seed,
                "run_id": descriptor.run_id,
                "n_positive": cohort_signature[0],
                "n_negative": cohort_signature[1],
                "prevalence": cohort_signature[2],
                "auprc_full": metrics["coreot_full"],
                "auprc_match_only": metrics["coreot_match_only"],
                "auprc_uniform_uot": metrics["uniform_uot"],
                "auprc_prior_only": metrics["prior_only"],
                "delta_full_match_only": (metrics["coreot_full"] - metrics["coreot_match_only"]),
                "delta_match_only_uniform": (metrics["coreot_match_only"] - metrics["uniform_uot"]),
                "delta_full_prior": metrics["coreot_full"] - metrics["prior_only"],
            }
        )
        source_paths.add(str(descriptor.root))
    by_seed = pd.DataFrame(rows)
    _require_exact_keys(
        by_seed,
        columns=["endpoint", "seed"],
        expected={(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS},
        label="Panel C source data",
    )
    long = by_seed.melt(
        id_vars=["endpoint", "seed", "run_id"],
        value_vars=[column for column, _ in PANEL_C_CONTRASTS],
        var_name="contrast",
        value_name="delta_auprc",
    )
    label_lookup = dict(PANEL_C_CONTRASTS)
    long["display_name"] = long["contrast"].map(label_lookup)
    summary = (
        long.groupby(["endpoint", "contrast", "display_name"], sort=False)
        .agg(
            mean_delta_auprc=("delta_auprc", "mean"),
            sample_sd_delta_auprc=(
                "delta_auprc",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    return by_seed, summary, sorted(source_paths)


def _read_coreot_transport(
    root: Path,
    *,
    condition: str,
    expected_cell_ids: set[str],
) -> pd.DataFrame:
    path = (
        root
        / "transport"
        / condition
        / INTERNAL_CANDIDATE_SET
        / "coreot_full"
        / "cell_transport_scores.parquet"
    )
    frame = pd.read_parquet(path)
    _require_columns(frame, {"cell_id", "method", "a", "a_hat", "u"}, path)
    selected = frame.loc[
        frame["method"].eq("coreot_full"),
        ["cell_id", "a", "a_hat", "u"],
    ].copy()
    selected["cell_id"] = selected["cell_id"].astype(str)
    if selected["cell_id"].duplicated().any():
        raise PBMCFigure3Error(f"{path} contains duplicate CoRe-OT transport rows.")
    if set(selected["cell_id"]) != expected_cell_ids:
        raise PBMCFigure3Error(f"{path} does not match the paired query cell set.")
    numeric = selected[["a", "a_hat", "u"]].to_numpy(dtype=float)
    if (
        not np.isfinite(numeric).all()
        or (selected["a"] <= 0).any()
        or (selected["a_hat"] < 0).any()
    ):
        raise PBMCFigure3Error(f"{path} contains invalid transported marginals.")
    expected_u = np.maximum(
        selected["a"].to_numpy(dtype=float) - selected["a_hat"].to_numpy(dtype=float),
        0.0,
    ) / (selected["a"].to_numpy(dtype=float) + TRANSPORT_ETA)
    if not np.allclose(
        selected["u"].to_numpy(dtype=float),
        expected_u,
        rtol=0,
        atol=1.0e-10,
    ):
        raise PBMCFigure3Error(f"{path} violates the deficit formula.")
    return selected


def _read_coupling(root: Path, condition: str) -> tuple[pd.DataFrame, Path]:
    path = (
        root
        / "transport"
        / condition
        / INTERNAL_CANDIDATE_SET
        / "coreot_full"
        / "sparse_coupling.parquet"
    )
    frame = pd.read_parquet(
        path,
        columns=["source_cell_id", "target_cell_id", "coupling"],
    )
    frame["source_cell_id"] = frame["source_cell_id"].astype(str)
    frame["target_cell_id"] = frame["target_cell_id"].astype(str)
    values = frame["coupling"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise PBMCFigure3Error(f"{path} contains invalid coupling entries.")
    return frame, path


def _conditional_destination_table(
    *,
    coupling: pd.DataFrame,
    source_ids: set[str],
    metadata: pd.DataFrame,
    transported: pd.DataFrame,
    numerator_mask: Callable[[pd.DataFrame], pd.Series],
    probability_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = coupling.loc[coupling["source_cell_id"].isin(source_ids)].copy()
    if selected.empty:
        raise PBMCFigure3Error("No selected query cells occur in the sparse coupling.")
    row_totals = selected.groupby("source_cell_id", sort=False)["coupling"].sum()
    transport_indexed = transported.set_index("cell_id")
    expected = transport_indexed.reindex(sorted(source_ids))
    if expected[["a_hat"]].isna().any().any():
        raise PBMCFigure3Error("Conditional destinations lack transported marginals.")
    joined_totals = row_totals.reindex(sorted(source_ids)).fillna(0.0)
    if not np.allclose(
        joined_totals.to_numpy(dtype=float),
        expected["a_hat"].to_numpy(dtype=float),
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise PBMCFigure3Error(
            "Sparse-coupling row sums do not reproduce recorded transported marginals."
        )
    target = metadata.reindex(selected["target_cell_id"])
    if target[["cell_type", "condition"]].isna().any().any():
        raise PBMCFigure3Error("Some coupling targets do not join to PBMC metadata.")
    selected["target_cell_type"] = target["cell_type"].astype(str).to_numpy()
    selected["target_condition"] = target["condition"].astype(str).to_numpy()
    mask = numerator_mask(selected)
    if mask.dtype != bool or len(mask) != len(selected):
        raise PBMCFigure3Error("Conditional-destination numerator mask is invalid.")
    numerator = (
        selected.loc[mask]
        .groupby("source_cell_id", sort=False)["coupling"]
        .sum()
        .reindex(sorted(source_ids))
        .fillna(0.0)
    )
    per_cell = expected.loc[:, ["a", "a_hat", "u"]].copy()
    per_cell.index.name = "cell_id"
    per_cell = per_cell.reset_index()
    per_cell["eligible_destination"] = per_cell["a_hat"].gt(TRANSPORT_ETA)
    per_cell[probability_name] = np.where(
        per_cell["eligible_destination"],
        numerator.to_numpy(dtype=float) / joined_totals.to_numpy(dtype=float),
        np.nan,
    )
    eligible_values = per_cell.loc[per_cell["eligible_destination"], probability_name].to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(eligible_values).all()
        or (eligible_values < -1.0e-12).any()
        or (eligible_values > 1.0 + 1.0e-12).any()
    ):
        raise PBMCFigure3Error("Conditional destination probabilities leave [0, 1].")
    per_cell[probability_name] = per_cell[probability_name].clip(lower=0.0, upper=1.0)
    return per_cell, selected


def _forced_destinations(selected_coupling: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        selected_coupling.groupby(
            ["source_cell_id", "target_cell_type", "target_condition"],
            sort=True,
        )["coupling"]
        .sum()
        .reset_index()
    )
    grouped["condition_specific_destination"] = (
        grouped["target_cell_type"].astype(str) + "::" + grouped["target_condition"].astype(str)
    )
    ranked = grouped.sort_values(
        ["source_cell_id", "coupling", "condition_specific_destination"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    forced_state = ranked.drop_duplicates("source_cell_id", keep="first").loc[
        :, ["source_cell_id", "condition_specific_destination"]
    ]
    by_type = (
        grouped.groupby(["source_cell_id", "target_cell_type"], sort=True)["coupling"]
        .sum()
        .reset_index()
        .sort_values(
            ["source_cell_id", "coupling", "target_cell_type"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("source_cell_id", keep="first")
        .loc[:, ["source_cell_id", "target_cell_type"]]
        .rename(columns={"target_cell_type": "coupling_forced_cell_type"})
    )
    return forced_state.merge(by_type, on="source_cell_id", validate="one_to_one")


def collect_panel_d_data(
    *,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    comparison = _read_comparison(comparison_path)
    metadata = _read_raw_metadata(raw_data_path)
    cell_rows = []
    seed_rows = []
    source_paths = {str(comparison_path), str(raw_data_path)}
    for descriptor in _selected_internal_runs(comparison, runs_root=runs_root):
        truth = _read_truth(descriptor.root)
        truth_ids = set(truth["cell_id"])
        incomplete = _read_coreot_transport(
            descriptor.root,
            condition=INCOMPLETE_REFERENCE,
            expected_cell_ids=truth_ids,
        )
        full = _read_coreot_transport(
            descriptor.root,
            condition=FULL_REFERENCE,
            expected_cell_ids=truth_ids,
        )
        paired = (
            truth.loc[:, ["cell_id", "true_label", "is_absent_state"]]
            .merge(
                incomplete.loc[:, ["cell_id", "u"]].rename(columns={"u": "u_ablated"}),
                on="cell_id",
                validate="one_to_one",
            )
            .merge(
                full.loc[:, ["cell_id", "u"]].rename(columns={"u": "u_full"}),
                on="cell_id",
                validate="one_to_one",
            )
        )
        raw = metadata.reindex(paired["cell_id"])
        if raw[["cell_type", "condition"]].isna().any().any():
            raise PBMCFigure3Error("Panel D query cells do not join to raw metadata.")
        paired["cell_type"] = raw["cell_type"].astype(str).to_numpy()
        paired["condition"] = raw["condition"].astype(str).to_numpy()
        local = paired.loc[paired["cell_type"].eq(descriptor.endpoint)].copy()
        local["truth_group"] = np.where(
            local["condition"].eq("stim"),
            "held_out_stimulated",
            "same_type_control",
        )
        if not np.array_equal(
            local["truth_group"].eq("held_out_stimulated").to_numpy(),
            local["is_absent_state"].astype(bool).to_numpy(),
        ):
            raise PBMCFigure3Error("Panel D truth groups disagree with absent-state truth.")
        local["u_ablated_minus_full"] = local["u_ablated"] - local["u_full"]
        local.insert(0, "run_id", descriptor.run_id)
        local.insert(0, "seed", descriptor.seed)
        local.insert(0, "endpoint", descriptor.endpoint)

        held_ids = set(
            local.loc[local["truth_group"].eq("held_out_stimulated"), "cell_id"].astype(str)
        )
        full_coupling, coupling_path = _read_coupling(descriptor.root, FULL_REFERENCE)
        restored, _ = _conditional_destination_table(
            coupling=full_coupling,
            source_ids=held_ids,
            metadata=metadata,
            transported=full,
            numerator_mask=lambda frame, endpoint=descriptor.endpoint: (
                frame["target_cell_type"].eq(endpoint) & frame["target_condition"].eq("stim")
            ),
            probability_name="restored_state_conditional_probability",
        )
        local = local.merge(
            restored.loc[
                :,
                [
                    "cell_id",
                    "a_hat",
                    "eligible_destination",
                    "restored_state_conditional_probability",
                ],
            ].rename(columns={"a_hat": "transported_mass_full"}),
            on="cell_id",
            how="left",
            validate="one_to_one",
        )
        cell_rows.append(local)

        held = local.loc[local["truth_group"].eq("held_out_stimulated")]
        control = local.loc[local["truth_group"].eq("same_type_control")]
        eligible_held = held.loc[held["eligible_destination"].eq(True)]
        if eligible_held.empty:
            raise PBMCFigure3Error(f"No eligible held-out cells for Panel D: {descriptor.run_id}")
        deficit_response = control_adjusted_median_deficit_decrease(
            incomplete=local["u_ablated"],
            full=local["u_full"],
            heldout=local["truth_group"].eq("held_out_stimulated"),
            control=local["truth_group"].eq("same_type_control"),
        )
        seed_rows.append(
            {
                "endpoint": descriptor.endpoint,
                "seed": descriptor.seed,
                "run_id": descriptor.run_id,
                "heldout_u_ablated": float(held["u_ablated"].median()),
                "heldout_u_full": float(held["u_full"].median()),
                "control_u_ablated": float(control["u_ablated"].median()),
                "control_u_full": float(control["u_full"].median()),
                "restoration_specificity": (deficit_response.control_adjusted_decrease),
                "restored_state_conditional_probability": float(
                    eligible_held["restored_state_conditional_probability"].median()
                ),
                "n_heldout": int(len(held)),
                "n_control": int(len(control)),
                "n_destination_eligible": int(len(eligible_held)),
                "n_destination_excluded": int(len(held) - len(eligible_held)),
            }
        )
        source_paths.update(
            {
                str(descriptor.root),
                str(coupling_path),
            }
        )
    cells = pd.concat(cell_rows, ignore_index=True)
    by_seed = pd.DataFrame(seed_rows)
    _require_exact_keys(
        by_seed,
        columns=["endpoint", "seed"],
        expected={(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS},
        label="Panel D source data",
    )
    return cells, by_seed, sorted(source_paths)


def collect_panel_e_data(
    *,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    comparison = _read_comparison(comparison_path)
    metadata = _read_raw_metadata(raw_data_path)
    rows = []
    source_paths = {str(comparison_path), str(raw_data_path)}
    for descriptor in _selected_internal_runs(comparison, runs_root=runs_root):
        truth = _read_truth(descriptor.root)
        truth_ids = set(truth["cell_id"])
        transported = _read_coreot_transport(
            descriptor.root,
            condition=INCOMPLETE_REFERENCE,
            expected_cell_ids=truth_ids,
        )
        score_frame = _read_scores(
            descriptor.root,
            INCOMPLETE_REFERENCE,
            INTERNAL_CANDIDATE_SET,
        )
        _require_columns(score_frame, {"forced_label", "u"}, descriptor.root)
        coreot_scores = score_frame.loc[
            score_frame["method"].eq("coreot_full"),
            ["cell_id", "u", "forced_label"],
        ].copy()
        if (
            set(coreot_scores["cell_id"]) != truth_ids
            or coreot_scores["cell_id"].duplicated().any()
        ):
            raise PBMCFigure3Error(f"Panel E score rows mismatch {descriptor.run_id}.")
        local_truth = truth.loc[truth["true_label"].eq(descriptor.endpoint)].copy()
        local_ids = set(local_truth["cell_id"])
        coupling, coupling_path = _read_coupling(
            descriptor.root,
            INCOMPLETE_REFERENCE,
        )
        conditional, selected_coupling = _conditional_destination_table(
            coupling=coupling,
            source_ids=local_ids,
            metadata=metadata,
            transported=transported,
            numerator_mask=lambda frame, endpoint=descriptor.endpoint: frame["target_cell_type"].eq(
                endpoint
            ),
            probability_name="same_celltype_conditional_mass",
        )
        forced = _forced_destinations(selected_coupling)
        local = (
            local_truth.loc[:, ["cell_id", "is_absent_state"]]
            .merge(
                conditional,
                on="cell_id",
                validate="one_to_one",
            )
            .merge(
                coreot_scores.loc[:, ["cell_id", "forced_label"]],
                on="cell_id",
                validate="one_to_one",
            )
            .merge(
                forced,
                left_on="cell_id",
                right_on="source_cell_id",
                validate="one_to_one",
            )
        )
        raw = metadata.reindex(local["cell_id"])
        if raw[["cell_type", "condition"]].isna().any().any():
            raise PBMCFigure3Error("Panel E query cells do not join to raw metadata.")
        local["truth_group"] = np.where(
            raw["condition"].astype(str).to_numpy() == "stim",
            "held_out_stimulated",
            "same_type_control",
        )
        if not np.array_equal(
            local["truth_group"].eq("held_out_stimulated").to_numpy(),
            local["is_absent_state"].astype(bool).to_numpy(),
        ):
            raise PBMCFigure3Error("Panel E truth groups disagree with absent-state truth.")
        eligible = local["eligible_destination"].astype(bool)
        if (
            not local.loc[eligible, "forced_label"]
            .astype(str)
            .eq(local.loc[eligible, "coupling_forced_cell_type"].astype(str))
            .all()
        ):
            raise PBMCFigure3Error(
                f"Forced cell-type assignments disagree with coupling destinations: {descriptor.run_id}"
            )
        local = local.rename(
            columns={
                "a_hat": "transported_mass",
                "forced_label": "forced_cell_type_assignment",
                "condition_specific_destination": ("forced_condition_specific_destination"),
            }
        )
        local.insert(0, "run_id", descriptor.run_id)
        local.insert(0, "seed", descriptor.seed)
        local.insert(0, "endpoint", descriptor.endpoint)
        rows.append(
            local.loc[
                :,
                [
                    "endpoint",
                    "seed",
                    "run_id",
                    "cell_id",
                    "truth_group",
                    "u",
                    "a",
                    "transported_mass",
                    "eligible_destination",
                    "same_celltype_conditional_mass",
                    "forced_cell_type_assignment",
                    "forced_condition_specific_destination",
                ],
            ]
        )
        source_paths.update({str(descriptor.root), str(coupling_path)})
    cells = pd.concat(rows, ignore_index=True)
    if cells.duplicated(["endpoint", "seed", "cell_id"]).any():
        raise PBMCFigure3Error("Panel E source data contain duplicate run-cell rows.")
    eligible = cells.loc[cells["eligible_destination"]].copy()
    summary = (
        eligible.groupby(["endpoint", "seed", "run_id", "truth_group"], sort=False)
        .agg(
            n_cells=("cell_id", "size"),
            median_u=("u", "median"),
            q1_u=("u", lambda values: values.quantile(0.25)),
            q3_u=("u", lambda values: values.quantile(0.75)),
            median_same_celltype_conditional_mass=(
                "same_celltype_conditional_mass",
                "median",
            ),
            q1_same_celltype_conditional_mass=(
                "same_celltype_conditional_mass",
                lambda values: values.quantile(0.25),
            ),
            q3_same_celltype_conditional_mass=(
                "same_celltype_conditional_mass",
                lambda values: values.quantile(0.75),
            ),
        )
        .reset_index()
    )
    expected = {
        (endpoint, seed, group)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for group in ("held_out_stimulated", "same_type_control")
    }
    _require_exact_keys(
        summary,
        columns=["endpoint", "seed", "truth_group"],
        expected=expected,
        label="Panel E seed summaries",
    )
    return cells, summary, sorted(source_paths)


def collect_panel_f_data(
    *,
    label_transfer_path: Path = COMPARISON_LABEL_TRANSFER_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = pd.read_csv(label_transfer_path)
    _require_columns(
        frame,
        {
            "held_out_label",
            "seed",
            "condition_id",
            "method",
            "score",
            "forced_accuracy",
            "forced_macro_f1",
            "coverage",
            "post_abstention_macro_f1",
        },
        label_transfer_path,
    )
    frame = select_primary_comparison_rows(frame, source=label_transfer_path)
    rows = []
    for endpoint in ENDPOINTS:
        for seed in SEEDS:
            for method, score, display in PANEL_F_METHODS:
                selected = frame.loc[
                    frame["held_out_label"].eq(endpoint)
                    & frame["seed"].eq(seed)
                    & frame["condition_id"].eq(INCOMPLETE_REFERENCE)
                    & frame["method"].eq(method)
                    & frame["score"].eq(score)
                ]
                if len(selected) != 1:
                    raise PBMCFigure3Error(
                        "Expected one represented-cell-type transfer row for "
                        f"{endpoint}, seed {seed}, {method}/{score}; found {len(selected)}."
                    )
                row = selected.iloc[0]
                rows.append(
                    {
                        "endpoint": endpoint,
                        "seed": seed,
                        "run_id": str(row["run_id"]),
                        "method": method,
                        "score": score,
                        "display_name": display,
                        "forced_accuracy": float(row["forced_accuracy"]),
                        "forced_macro_f1": float(row["forced_macro_f1"]),
                        "coverage": float(row["coverage"]),
                        "post_abstention_macro_f1": float(row["post_abstention_macro_f1"]),
                    }
                )
    by_seed = pd.DataFrame(rows)
    _require_exact_keys(
        by_seed,
        columns=["endpoint", "seed", "method", "score"],
        expected={
            (endpoint, seed, method, score)
            for endpoint in ENDPOINTS
            for seed in SEEDS
            for method, score, _ in PANEL_F_METHODS
        },
        label="Panel F source data",
    )
    summary = (
        by_seed.groupby(["endpoint", "method", "score", "display_name"], sort=False)
        .agg(
            mean_forced_macro_f1=("forced_macro_f1", "mean"),
            sample_sd_forced_macro_f1=(
                "forced_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            mean_forced_accuracy=("forced_accuracy", "mean"),
            sample_sd_forced_accuracy=(
                "forced_accuracy",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    return by_seed, summary, [str(label_transfer_path)]


def _candidate_set_for_method(method: str) -> str:
    if method in {"seurat_anchor", "scmap_cluster", "chetah"}:
        return EXTERNAL_CANDIDATE_SET
    return INTERNAL_CANDIDATE_SET


def _read_fixed_query_umap(
    *,
    path: Path = FIXED_QUERY_UMAP_PATH,
    expected_cell_ids: set[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, {"cell_id", "umap_1", "umap_2"}, path)
    selected = frame.loc[:, ["cell_id", "umap_1", "umap_2"]].copy()
    selected["cell_id"] = selected["cell_id"].astype(str)
    if selected["cell_id"].duplicated().any():
        raise PBMCFigure3Error(f"{path} contains duplicate fixed-UMAP cells.")
    if set(selected["cell_id"]) != expected_cell_ids:
        raise PBMCFigure3Error(f"{path} does not match the seed-1 query cell set.")
    if not np.isfinite(selected[["umap_1", "umap_2"]].to_numpy(dtype=float)).all():
        raise PBMCFigure3Error(f"{path} contains non-finite UMAP coordinates.")
    return selected


def collect_panel_d_umap_data(
    *,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
    umap_path: Path = FIXED_QUERY_UMAP_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    comparison = _read_comparison(comparison_path)
    metadata = _read_raw_metadata(raw_data_path)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    source_paths = {str(comparison_path), str(raw_data_path), str(umap_path)}
    for endpoint in ENDPOINTS:
        coreot_row = _comparison_row(
            comparison,
            endpoint=endpoint,
            seed=PANEL_D_SEED,
            method="coreot_full",
            score="u",
        )
        truth = _read_truth(runs_root / str(coreot_row["run_id"]))
        truth_ids = set(truth["cell_id"].astype(str))
        coordinates = _read_fixed_query_umap(
            path=umap_path,
            expected_cell_ids=truth_ids,
        )
        base = truth.merge(coordinates, on="cell_id", validate="one_to_one")
        base.insert(0, "endpoint", endpoint)
        raw = metadata.reindex(base["cell_id"])
        if raw[["cell_type", "condition"]].isna().any().any():
            raise PBMCFigure3Error(f"Panel E metadata join failed for {endpoint}.")
        base["cell_type"] = raw["cell_type"].astype(str).to_numpy()
        base["condition"] = raw["condition"].astype(str).to_numpy()
        base["is_evaluation_cohort"] = base["cell_type"].eq(endpoint)
        base["is_held_out"] = base["is_absent_state"].astype(bool)
        n_selected = int(base["is_held_out"].sum())
        if n_selected <= 0:
            raise PBMCFigure3Error(f"Panel E has no reference-omitted cells for {endpoint}.")

        truth_map = base.copy()
        truth_map["map_id"] = "truth"
        truth_map["map_display"] = "Held-out truth"
        truth_map["method"] = "truth"
        truth_map["score"] = np.nan
        truth_map["is_selected"] = truth_map["is_held_out"]
        frames.append(truth_map)

        for method, score, display in PANEL_E_METHODS:
            expected = _comparison_row(
                comparison,
                endpoint=endpoint,
                seed=PANEL_D_SEED,
                method=method,
                score=score,
            )
            root = runs_root / str(expected["run_id"])
            method_truth = _read_truth(root)
            if set(method_truth["cell_id"].astype(str)) != truth_ids:
                raise PBMCFigure3Error(f"Panel E query cells differ for {endpoint}/{method}.")
            score_frame = _read_scores(
                root,
                INCOMPLETE_REFERENCE,
                _candidate_set_for_method(method),
            )
            selected_scores = _method_scores(
                score_frame,
                method=method,
                score=score,
                expected_cell_ids=truth_ids,
                source=str(root),
            ).rename(columns={score: "score"})
            method_map = base.merge(
                selected_scores,
                on="cell_id",
                validate="one_to_one",
            )
            ranked = method_map.loc[method_map["is_evaluation_cohort"]].sort_values(
                ["score", "cell_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            selected_ids = set(ranked.head(n_selected)["cell_id"])
            if len(selected_ids) != n_selected:
                raise PBMCFigure3Error(f"Panel E top-N selection failed for {endpoint}/{method}.")
            method_map["map_id"] = method
            method_map["map_display"] = display
            method_map["method"] = method
            method_map["is_selected"] = method_map["cell_id"].isin(selected_ids)
            frames.append(method_map)
            summaries.append(
                {
                    "endpoint": endpoint,
                    "seed": PANEL_D_SEED,
                    "method": method,
                    "display_name": display,
                    "run_id": str(expected["run_id"]),
                    "score_name": score,
                    "n_query": len(method_map),
                    "n_evaluation_cohort": int(method_map["is_evaluation_cohort"].sum()),
                    "n_held_out": n_selected,
                    "n_selected": int(method_map["is_selected"].sum()),
                    "n_selected_held_out": int(
                        (method_map["is_selected"] & method_map["is_held_out"]).sum()
                    ),
                }
            )
            source_paths.add(str(root))
    cells = pd.concat(frames, ignore_index=True)
    cells.insert(1, "seed", PANEL_D_SEED)
    expected_maps = {(endpoint, map_id) for endpoint in ENDPOINTS for map_id, _ in PANEL_E_MAPS}
    observed_maps = set(
        cells[["endpoint", "map_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_maps != expected_maps:
        raise PBMCFigure3Error("Panel E maps differ from the accepted contract.")
    if cells.duplicated(["endpoint", "map_id", "cell_id"]).any():
        raise PBMCFigure3Error("Panel E contains duplicate map-cell rows.")
    summary = pd.DataFrame(summaries)
    if not summary["n_selected"].eq(summary["n_held_out"]).all():
        raise PBMCFigure3Error("Panel E does not select exactly N_l cells.")
    return cells, summary, sorted(source_paths)


def collect_panel_f_assignment_data(
    *,
    label_transfer_path: Path = COMPARISON_LABEL_TRANSFER_PATH,
    runs_root: Path = RUNS_ROOT,
    umap_path: Path = FIXED_QUERY_UMAP_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    by_seed, _, source_paths = collect_panel_f_data(label_transfer_path=label_transfer_path)
    seed_rows = by_seed.loc[by_seed["seed"].eq(PANEL_D_SEED)]
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    source_set = set(source_paths) | {str(umap_path)}
    for endpoint in ENDPOINTS:
        endpoint_rows = seed_rows.loc[seed_rows["endpoint"].eq(endpoint)]
        coreot_row = endpoint_rows.loc[endpoint_rows["method"].eq("coreot_full")]
        if len(coreot_row) != 1:
            raise PBMCFigure3Error(f"Panel F lacks one CoRe-OT row for {endpoint}.")
        truth_root = runs_root / str(coreot_row.iloc[0]["run_id"])
        truth = _read_truth(truth_root)
        truth_ids = set(truth["cell_id"].astype(str))
        coordinates = _read_fixed_query_umap(
            path=umap_path,
            expected_cell_ids=truth_ids,
        )
        base = truth.merge(coordinates, on="cell_id", validate="one_to_one")
        base.insert(0, "endpoint", endpoint)
        base["is_represented"] = base["is_shared_state"].astype(bool)
        base["is_held_out"] = base["is_absent_state"].astype(bool)
        if not (base["is_represented"] ^ base["is_held_out"]).all():
            raise PBMCFigure3Error(f"Panel F truth does not partition query cells for {endpoint}.")
        represented_labels = sorted(
            base.loc[base["is_represented"], "true_label"].astype(str).unique()
        )

        truth_map = base.copy()
        truth_map["map_id"] = "truth"
        truth_map["map_display"] = "Ground truth"
        truth_map["method"] = "truth"
        truth_map["forced_label"] = ""
        truth_map["displayed_assignment"] = np.where(
            truth_map["is_represented"],
            truth_map["true_label"],
            "Held-out state (not evaluated)",
        )
        frames.append(truth_map)

        for method, score, display in PANEL_F_METHODS:
            metric_row = endpoint_rows.loc[
                endpoint_rows["method"].eq(method) & endpoint_rows["score"].eq(score)
            ]
            if len(metric_row) != 1:
                raise PBMCFigure3Error(f"Panel F lacks one metric row for {endpoint}/{method}.")
            root = runs_root / str(metric_row.iloc[0]["run_id"])
            method_truth = _read_truth(root)
            if set(method_truth["cell_id"].astype(str)) != truth_ids:
                raise PBMCFigure3Error(f"Panel F query cells differ for {endpoint}/{method}.")
            score_frame = _read_scores(
                root,
                INCOMPLETE_REFERENCE,
                _candidate_set_for_method(method),
            )
            _require_columns(score_frame, {"forced_label"}, root)
            assignments = score_frame.loc[
                score_frame["method"].eq(method),
                ["cell_id", "forced_label"],
            ].copy()
            assignments["cell_id"] = assignments["cell_id"].astype(str)
            if (
                assignments["cell_id"].duplicated().any()
                or set(assignments["cell_id"]) != truth_ids
            ):
                raise PBMCFigure3Error(f"Panel F assignment cells differ for {endpoint}/{method}.")
            method_map = base.merge(
                assignments,
                on="cell_id",
                validate="one_to_one",
            )
            represented = method_map["is_represented"]
            method_map["map_id"] = method
            method_map["map_display"] = display
            method_map["method"] = method
            method_map["displayed_assignment"] = np.where(
                represented,
                method_map["forced_label"].astype(str),
                "Held-out state (not evaluated)",
            )
            unexpected = sorted(
                set(method_map.loc[represented, "displayed_assignment"]) - set(CELL_TYPE_COLORS)
            )
            if unexpected:
                raise PBMCFigure3Error(
                    f"Panel F has unmapped labels for {endpoint}/{method}: {unexpected}"
                )
            truth_values = method_map.loc[represented, "true_label"].astype(str)
            predicted = method_map.loc[represented, "forced_label"].astype(str)
            forced_accuracy = float(accuracy_score(truth_values, predicted))
            forced_macro_f1 = float(
                f1_score(
                    truth_values,
                    predicted,
                    labels=represented_labels,
                    average="macro",
                    zero_division=0,
                )
            )
            expected_accuracy = float(metric_row.iloc[0]["forced_accuracy"])
            expected_macro_f1 = float(metric_row.iloc[0]["forced_macro_f1"])
            if not np.isclose(forced_accuracy, expected_accuracy, atol=1.0e-12):
                raise PBMCFigure3Error(
                    f"Panel F accuracy does not reproduce Panel D for "
                    f"{endpoint}/{method}: {forced_accuracy} != {expected_accuracy}."
                )
            if not np.isclose(forced_macro_f1, expected_macro_f1, atol=1.0e-12):
                raise PBMCFigure3Error(
                    f"Panel F macro-F1 does not reproduce Panel D for "
                    f"{endpoint}/{method}: {forced_macro_f1} != {expected_macro_f1}."
                )
            frames.append(method_map)
            summaries.append(
                {
                    "endpoint": endpoint,
                    "seed": PANEL_D_SEED,
                    "method": method,
                    "display_name": display,
                    "run_id": str(metric_row.iloc[0]["run_id"]),
                    "n_represented": int(represented.sum()),
                    "forced_accuracy": forced_accuracy,
                    "forced_macro_f1": forced_macro_f1,
                    "panel_d_forced_accuracy": expected_accuracy,
                    "panel_d_forced_macro_f1": expected_macro_f1,
                }
            )
            source_set.add(str(root))
    cells = pd.concat(frames, ignore_index=True)
    cells.insert(1, "seed", PANEL_D_SEED)
    expected_maps = {(endpoint, map_id) for endpoint in ENDPOINTS for map_id, _ in PANEL_F_MAPS}
    observed_maps = set(
        cells[["endpoint", "map_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_maps != expected_maps:
        raise PBMCFigure3Error("Panel F maps differ from the accepted contract.")
    if cells.duplicated(["endpoint", "map_id", "cell_id"]).any():
        raise PBMCFigure3Error("Panel F contains duplicate map-cell rows.")
    return cells, pd.DataFrame(summaries), sorted(source_set)


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LIGHT_EDGE)
    ax.tick_params(labelsize=6.2, colors=INK, width=0.6, length=2.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def _panel_heading(
    target: plt.Axes,
    letter: str,
    title: str,
    *,
    x: float = 0.0,
    y: float = 1.13,
    title_x: float = 0.075,
    fontsize: float = 8.0,
) -> None:
    target.text(
        x,
        y,
        letter,
        transform=target.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    if title:
        target.text(
            title_x,
            y,
            title,
            transform=target.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color=INK,
            clip_on=False,
        )


def _rounded_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = LIGHT_EDGE,
    fontsize: float = 6.2,
    fontweight: str = "normal",
    linewidth: float = 0.8,
    gid: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    patch.set_gid(gid)
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=INK,
        linespacing=1.08,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED_INK,
    linewidth: float = 0.8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=linewidth,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def _draw_tile_matrix(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    crossed_endpoint: str | None = None,
    fontsize: float = 5.4,
) -> None:
    col_width = width / 3
    row_height = height / 2
    for column, endpoint in enumerate(ENDPOINTS):
        short = ENDPOINT_SHORT[endpoint]
        ax.text(
            x + (column + 0.5) * col_width,
            y + height + 0.025,
            short,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=INK,
            fontweight="bold",
        )
        for row, condition in enumerate(("Stim", "Ctrl")):
            y0 = y + (1 - row) * row_height
            fill = "#FDE8DE" if condition == "Stim" else "#E6F0F8"
            ax.add_patch(
                Rectangle(
                    (x + column * col_width, y0),
                    col_width * 0.88,
                    row_height * 0.82,
                    facecolor=fill,
                    edgecolor=LIGHT_EDGE,
                    linewidth=0.55,
                )
            )
            crossed = condition == "Stim" and endpoint == crossed_endpoint
            ax.text(
                x + column * col_width + col_width * 0.44,
                y0 + row_height * 0.41,
                "×" if crossed else "✓",
                ha="center",
                va="center",
                fontsize=fontsize + 1.4,
                color=STIMULATED if crossed else INK,
                fontweight="bold",
            )
    ax.text(
        x - 0.012,
        y + row_height * 1.41,
        "Stim",
        ha="right",
        va="center",
        fontsize=fontsize - 0.3,
        color=STIMULATED,
    )
    ax.text(
        x - 0.012,
        y + row_height * 0.41,
        "Ctrl",
        ha="right",
        va="center",
        fontsize=fontsize - 0.3,
        color=CONTROL,
    )


def draw_panel_a(
    ax: plt.Axes,
    *,
    compact: bool = False,
    include_heading: bool = True,
) -> None:
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    if include_heading:
        _panel_heading(
            ax,
            "A",
            "",
            y=0.99 if compact else 0.98,
            title_x=0.11 if compact else 0.045,
            fontsize=7.2 if compact else 8.2,
        )
    font = 5.2 if compact else 6.0
    ax.text(
        0.5,
        0.84,
        "Donor-level split",
        ha="center",
        va="center",
        fontsize=font + 0.5,
        color=MUTED_INK,
    )
    _rounded_box(
        ax,
        x=0.04,
        y=0.42,
        width=0.22,
        height=0.22,
        text="Query cells\nall states retained",
        facecolor=QUERY_FILL,
        edgecolor=CONTROL,
        fontsize=font,
    )
    _rounded_box(
        ax,
        x=0.34,
        y=0.42,
        width=0.20,
        height=0.22,
        text="Reference branch\nB · NK · DC\nCtrl + Stim",
        facecolor=REFERENCE_FILL,
        fontsize=font,
    )
    _rounded_box(
        ax,
        x=0.60,
        y=0.59,
        width=0.36,
        height=0.20,
        text=("Reference-omitted condition\nomit one target Stim state\nsame-type Ctrl remains"),
        facecolor="#FFF4EE",
        edgecolor=STIMULATED,
        fontsize=font - 0.2,
        gid="panel-a-reference-omitted-box",
    )
    _rounded_box(
        ax,
        x=0.60,
        y=0.27,
        width=0.36,
        height=0.20,
        text="Restored-reference condition\ntarget Stim state retained",
        facecolor="#E8F4ED",
        edgecolor=RESTORED,
        fontsize=font,
        gid="panel-a-restored-reference-box",
    )
    _arrow(ax, (0.54, 0.56), (0.60, 0.69), color=STIMULATED)
    _arrow(ax, (0.54, 0.49), (0.60, 0.37), color=RESTORED)
    # The unchanged query enters both paired mappings through separate routes.
    ax.plot(
        [0.26, 0.30, 0.30, 0.59, 0.59],
        [0.57, 0.72, 0.72, 0.72, 0.69],
        color=CONTROL,
        linewidth=0.8,
    )
    _arrow(ax, (0.59, 0.69), (0.64, 0.69), color=CONTROL)
    ax.plot(
        [0.26, 0.30, 0.30, 0.59, 0.59],
        [0.48, 0.31, 0.31, 0.31, 0.37],
        color=CONTROL,
        linewidth=0.8,
    )
    _arrow(ax, (0.59, 0.37), (0.64, 0.37), color=CONTROL)
    ax.text(
        0.42,
        0.75,
        "Same query cells",
        ha="center",
        va="bottom",
        fontsize=font - 0.4,
        color=CONTROL,
    )
    _rounded_box(
        ax,
        x=0.06,
        y=0.02,
        width=0.88,
        height=0.15,
        text="within-cell-type omitted-state ranking\npaired reference-restoration response",
        facecolor="#F7F7F7",
        edgecolor=INK,
        fontsize=font,
    )


def _method_axis_labels(methods: Iterable[tuple[str, str, str]]) -> list[str]:
    labels = []
    for _, _, display in methods:
        labels.append(
            display.replace("scmap-cluster", "scmap-\ncluster").replace(
                "Uniform UOT", "Uniform\nUOT"
            )
        )
    return labels


def draw_panel_b(
    axes: Iterable[plt.Axes],
    by_seed: pd.DataFrame,
    *,
    include_heading: bool = True,
    heading_y: float = 1.13,
    compact: bool = False,
) -> None:
    axes = list(axes)
    method_order = [method for method, _, _ in PANEL_B_METHODS]
    y = np.arange(len(method_order))
    seed_offsets = dict(zip(SEEDS, np.linspace(-0.13, 0.13, len(SEEDS))))
    display_by_method = {method: display for method, _, display in PANEL_B_METHODS}
    for endpoint_index, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS)):
        local = by_seed.loc[by_seed["endpoint"].eq(endpoint)]
        for method_index, method in enumerate(method_order):
            values = local.loc[local["method"].eq(method)].sort_values("seed")
            for row in values.itertuples(index=False):
                ax.scatter(
                    row.auprc,
                    method_index + seed_offsets[int(row.seed)],
                    s=10,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    alpha=0.55,
                    linewidth=0,
                    zorder=3,
                )
            mean = float(values["auprc"].mean())
            sample_sd = float(values["auprc"].std(ddof=1))
            ax.errorbar(
                mean,
                method_index,
                xerr=sample_sd,
                fmt=METHOD_MARKERS[method],
                markersize=5.0,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor=INK,
                markeredgewidth=0.7,
                ecolor=METHOD_COLORS[method],
                elinewidth=0.9,
                capsize=2,
                zorder=5,
            )
        for prevalence in local.groupby("seed")["prevalence"].first():
            ax.axvline(
                prevalence,
                color="#888888",
                linestyle=(0, (3, 2)),
                linewidth=0.55,
                alpha=0.25,
                zorder=1,
            )
        ax.set(
            xlim=PANEL_B_AP_DISPLAY_LIMITS,
            ylim=(len(method_order) - 0.5, -0.5),
            yticks=y,
            title={
                "B cells": "Stimulated B",
                "NK cells": "Stimulated NK",
                "Dendritic cells": "Stimulated DC",
            }[endpoint],
        )
        ax.set_xticks(PANEL_B_AP_TICKS)
        if endpoint_index == 0:
            ax.set_yticklabels(
                [display_by_method[method] for method in method_order],
                fontsize=5.6,
            )
        else:
            ax.set_yticklabels([])
        ax.title.set_fontsize(5.8 if compact else 6.8)
        ax.title.set_fontweight("bold")
        if endpoint_index == 0:
            ax.set_xlabel("AP for omitted-state ranking", fontsize=6.4)
        _style_axis(ax, grid_axis="x")
    if include_heading:
        _panel_heading(
            axes[0],
            "B",
            "",
            title_x=0.12,
            y=heading_y,
        )
    axes[1].set_xlabel("AP for omitted-state ranking", fontsize=6.4)
    axes[-1].set_xlabel("AP for omitted-state ranking", fontsize=6.4)
    axes[-1].text(
        0.02,
        0.98,
        "dashed: split prevalence",
        transform=axes[-1].transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=MUTED_INK,
    )


def draw_panel_c(
    ax: plt.Axes,
    by_seed: pd.DataFrame,
    *,
    include_heading: bool = True,
) -> None:
    endpoint_x = np.arange(len(ENDPOINTS))
    offsets = (-0.18, 0.0, 0.18)
    for contrast_index, (column, display) in enumerate(PANEL_C_CONTRASTS):
        color = CONTRAST_COLORS[column]
        marker = CONTRAST_MARKERS[column]
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            values = by_seed.loc[by_seed["endpoint"].eq(endpoint), ["seed", column]].sort_values(
                "seed"
            )
            center = endpoint_index + offsets[contrast_index]
            jitter = np.linspace(-0.045, 0.045, len(values))
            ax.scatter(
                center + jitter,
                values[column],
                s=10,
                color=color,
                marker=marker,
                alpha=0.55,
                linewidth=0,
                zorder=3,
            )
            mean = float(values[column].mean())
            sample_sd = float(values[column].std(ddof=1))
            ax.errorbar(
                center,
                mean,
                yerr=sample_sd,
                fmt=marker,
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                ecolor=color,
                elinewidth=0.85,
                capsize=2,
                zorder=5,
            )
    ax.axhline(0, color=INK, linewidth=0.75, zorder=2)
    values = by_seed[[column for column, _ in PANEL_C_CONTRASTS]].to_numpy(dtype=float)
    lower = min(-0.06, float(np.nanmin(values)) - 0.05)
    upper = max(0.08, float(np.nanmax(values)) + 0.05)
    ax.set(
        xlim=(-0.55, len(ENDPOINTS) - 0.45),
        ylim=(lower, upper),
        xticks=endpoint_x,
        xticklabels=["B", "NK", "DC"],
        ylabel="Paired ΔAP",
    )
    ax.set_xlabel("Reference-omitted cells", fontsize=6.4)
    ax.set_ylabel("Control-adjusted decrease, $\\Delta^{\\mathrm{CA}}$", fontsize=6.8)
    _style_axis(ax)
    if include_heading:
        _panel_heading(ax, "C", "Transport and prior controls", title_x=0.12)
    handles = [
        Line2D(
            [0],
            [0],
            marker=CONTRAST_MARKERS[column],
            color="none",
            markerfacecolor="white",
            markeredgecolor=CONTRAST_COLORS[column],
            markersize=4.8,
            label=display,
        )
        for column, display in PANEL_C_CONTRASTS
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=5.1,
        loc="upper left",
        borderaxespad=0.15,
        handletextpad=0.35,
        labelspacing=0.3,
    )


def draw_panel_d(
    deficit_ax: plt.Axes,
    specificity_ax: plt.Axes,
    destination_ax: plt.Axes,
    by_seed: pd.DataFrame,
    *,
    include_heading: bool = True,
    compact: bool = False,
    letter: str = "D",
) -> None:
    row_specs = []
    for endpoint in ENDPOINTS:
        row_specs.append((endpoint, "heldout", f"{ENDPOINT_SHORT[endpoint]} reference-omitted"))
        row_specs.append((endpoint, "control", f"{ENDPOINT_SHORT[endpoint]} control"))
    y_positions = np.arange(len(row_specs))[::-1]
    seed_offsets = dict(zip(SEEDS, np.linspace(-0.18, 0.18, len(SEEDS))))
    for y, (endpoint, group, _) in zip(y_positions, row_specs):
        local = by_seed.loc[by_seed["endpoint"].eq(endpoint)].sort_values("seed")
        ablated_column = f"{group}_u_ablated"
        full_column = f"{group}_u_full"
        color = METHOD_COLORS["coreot_full"] if group == "heldout" else "#777777"
        for row in local.itertuples(index=False):
            y_seed = y + seed_offsets[int(row.seed)]
            ablated = float(getattr(row, ablated_column))
            full = float(getattr(row, full_column))
            deficit_ax.plot(
                [full, ablated],
                [y_seed, y_seed],
                color=color,
                alpha=0.35,
                linewidth=0.65,
                zorder=2,
            )
            deficit_ax.scatter(
                ablated,
                y_seed,
                s=10,
                facecolor="white",
                edgecolor=color,
                linewidth=0.6,
                zorder=3,
            )
            deficit_ax.scatter(
                full,
                y_seed,
                s=10,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                zorder=3,
            )
        mean_ablated = float(local[ablated_column].mean())
        mean_full = float(local[full_column].mean())
        deficit_ax.plot(
            [mean_full, mean_ablated],
            [y, y],
            color=color,
            linewidth=1.8,
            zorder=5,
        )
        deficit_ax.scatter(
            mean_ablated,
            y,
            s=27,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=6,
        )
        deficit_ax.scatter(
            mean_full,
            y,
            s=27,
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=6,
        )
    deficit_ax.set(
        xlim=(0, 0.85),
        ylim=(-0.55, 5.85),
        yticks=y_positions,
        yticklabels=[label for _, _, label in row_specs],
        xlabel=r"Median query-marginal deficit, $u$",
    )
    deficit_ax.tick_params(axis="y", labelsize=5.8)
    for y, endpoint in zip((5.55, 3.55, 1.55), ENDPOINTS, strict=True):
        deficit_ax.text(
            0.42,
            y,
            {
                "B cells": "Stimulated B",
                "NK cells": "Stimulated NK",
                "Dendritic cells": "Stimulated DC",
            }[endpoint],
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold",
            clip_on=False,
        )
    deficit_ax.set_xlabel("Decrease in median\nquery-marginal deficit", fontsize=6.4, labelpad=2)
    _style_axis(deficit_ax, grid_axis="x")
    deficit_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=INK,
                markersize=4.4,
                label="Reference-omitted condition",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=INK,
                markeredgecolor=INK,
                markersize=4.4,
                label="Restored-reference condition",
            ),
        ],
        frameon=False,
        fontsize=5.2,
        loc="lower right",
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.25,
    )
    if include_heading:
        _panel_heading(
            deficit_ax,
            letter,
            "",
            title_x=0.10,
        )

    x = np.arange(len(ENDPOINTS))
    for index, endpoint in enumerate(ENDPOINTS):
        local = by_seed.loc[by_seed["endpoint"].eq(endpoint)].sort_values("seed")
        jitter = np.linspace(-0.08, 0.08, len(local))
        specificity_ax.scatter(
            index + jitter,
            local["restoration_specificity"],
            s=11,
            color=STIMULATED,
            alpha=0.55,
            linewidth=0,
        )
        specificity_ax.scatter(
            index,
            local["restoration_specificity"].mean(),
            s=28,
            facecolor="white",
            edgecolor=STIMULATED,
            linewidth=1.0,
            zorder=4,
        )
    specificity_ax.axhline(0, color=INK, linewidth=0.7)
    specificity_values = by_seed["restoration_specificity"].to_numpy(dtype=float)
    lower = min(-0.05, float(specificity_values.min()) - 0.06)
    upper = max(0.1, float(specificity_values.max()) + 0.06)
    specificity_ax.set(
        xlim=(-0.45, 2.45),
        ylim=(lower, upper),
        xticks=x,
        xticklabels=["B", "NK", "DC"],
        title=(
            r"Control-adjusted ($\Delta^{\mathrm{CA}}$)"
            if compact
            else "Control-adjusted decrease, $\\Delta^{\\mathrm{CA}}$"
        ),
        ylabel=r"$\Delta^{\mathrm{CA}}$",
    )
    specificity_ax.title.set_fontsize(6.1)
    specificity_ax.title.set_fontweight("bold")
    specificity_ax.set_ylabel(
        (r"$\Delta^{\mathrm{CA}}$" if compact else r"$\Delta^{\mathrm{CA}}$"),
        fontsize=6.0,
    )
    _style_axis(specificity_ax)

    for index, endpoint in enumerate(ENDPOINTS):
        local = by_seed.loc[by_seed["endpoint"].eq(endpoint)].sort_values("seed")
        jitter = np.linspace(-0.08, 0.08, len(local))
        destination_ax.scatter(
            index + jitter,
            local["restored_state_conditional_probability"],
            s=11,
            color=RESTORED,
            alpha=0.55,
            linewidth=0,
        )
        destination_ax.scatter(
            index,
            local["restored_state_conditional_probability"].mean(),
            s=28,
            facecolor="white",
            edgecolor=RESTORED,
            linewidth=1.0,
            zorder=4,
        )
    destination_ax.set(
        xlim=(-0.45, 2.45),
        ylim=(0, 1.02),
        xticks=x,
        xticklabels=["B", "NK", "DC"],
        title="Median restored-state destination fraction",
        ylabel="Destination\nfraction",
    )
    destination_ax.title.set_fontsize(6.1)
    destination_ax.title.set_fontweight("bold")
    destination_ax.set_ylabel("Destination\nfraction", fontsize=6.0)
    _style_axis(destination_ax)


def draw_panel_e(
    axes: Iterable[plt.Axes],
    cells: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    include_heading: bool = True,
    heading_y: float = 1.13,
    compact: bool = False,
) -> None:
    axes = list(axes)
    group_styles = {
        "held_out_stimulated": (STIMULATED, "Reference-omitted stimulated"),
        "same_type_control": (CONTROL, "Same-type control"),
    }
    for endpoint_index, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS)):
        local = cells.loc[cells["endpoint"].eq(endpoint) & cells["eligible_destination"]]
        for group in ("same_type_control", "held_out_stimulated"):
            color, _ = group_styles[group]
            group_cells = local.loc[local["truth_group"].eq(group)]
            ax.scatter(
                group_cells["same_celltype_conditional_mass"],
                group_cells["u"],
                s=2.0,
                color=color,
                alpha=0.08 if group == "same_type_control" else 0.11,
                linewidth=0,
                rasterized=True,
                zorder=2,
            )
            group_summary = summary.loc[
                summary["endpoint"].eq(endpoint) & summary["truth_group"].eq(group)
            ].sort_values("seed")
            ax.scatter(
                group_summary["median_same_celltype_conditional_mass"],
                group_summary["median_u"],
                s=20,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                marker="o" if group == "held_out_stimulated" else "s",
                zorder=5,
            )
        ax.set(
            xlim=(0, 1.02),
            ylim=(0, 1.02),
            title=(
                f"{ENDPOINT_SHORT[endpoint]} reference-omitted"
                if compact
                else ENDPOINT_DISPLAY[endpoint]
            ),
        )
        ax.set_xticks(np.linspace(0, 1, 3))
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.title.set_fontsize(5.8 if compact else 6.5)
        ax.title.set_fontweight("bold")
        if endpoint_index == 0:
            ax.set_ylabel(r"Query-marginal deficit, $u$", fontsize=6.4)
        else:
            ax.set_yticklabels([])
        _style_axis(ax)
    if include_heading:
        _panel_heading(
            axes[0],
            "E",
            "Reference destination versus fitted mass retention",
            title_x=0.12,
            y=heading_y,
        )
    axes[1].set_xlabel(
        "Same-cell-type conditional mass",
        fontsize=6.4,
        labelpad=3,
    )
    axes[-1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=STIMULATED,
                markersize=4.3,
                label="Reference-omitted stimulated",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="white",
                markeredgecolor=CONTROL,
                markersize=4.3,
                label="Same-type control",
            ),
        ],
        frameon=False,
        fontsize=5.0,
        loc="lower left",
        borderaxespad=0.25,
        handletextpad=0.25,
    )


def draw_panel_f(
    axes: Iterable[plt.Axes],
    by_seed: pd.DataFrame,
    *,
    include_heading: bool = True,
    heading_y: float = 1.13,
) -> None:
    axes = list(axes)
    method_order = [method for method, _, _ in PANEL_F_METHODS]
    x = np.arange(len(method_order))
    seed_offsets = dict(zip(SEEDS, np.linspace(-0.13, 0.13, len(SEEDS))))
    for endpoint_index, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS)):
        local = by_seed.loc[by_seed["endpoint"].eq(endpoint)]
        for method_index, method in enumerate(method_order):
            values = local.loc[local["method"].eq(method)].sort_values("seed")
            for row in values.itertuples(index=False):
                ax.scatter(
                    method_index + seed_offsets[int(row.seed)],
                    row.forced_macro_f1,
                    s=9,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    alpha=0.55,
                    linewidth=0,
                    zorder=3,
                )
            mean = float(values["forced_macro_f1"].mean())
            sample_sd = float(values["forced_macro_f1"].std(ddof=1))
            ax.errorbar(
                method_index,
                mean,
                yerr=sample_sd,
                fmt=METHOD_MARKERS[method],
                markersize=4.7,
                markerfacecolor="white",
                markeredgecolor=METHOD_COLORS[method],
                markeredgewidth=1.0,
                ecolor=METHOD_COLORS[method],
                elinewidth=0.8,
                capsize=2,
                zorder=5,
            )
        ax.set(
            ylim=(0, 1.02),
            xticks=x,
            xticklabels=_method_axis_labels(PANEL_F_METHODS),
            title={
                "B cells": "Stimulated B",
                "NK cells": "Stimulated NK",
                "Dendritic cells": "Stimulated DC",
            }[endpoint],
        )
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="x", labelsize=5.0, pad=1)
        ax.title.set_fontsize(6.5)
        ax.title.set_fontweight("bold")
        if endpoint_index == 0:
            ax.set_ylabel("Forced macro-F1", fontsize=6.5)
        else:
            ax.set_yticklabels([])
        _style_axis(ax)
    if include_heading:
        _panel_heading(
            axes[0],
            "F",
            "",
            title_x=0.12,
            y=heading_y,
        )


def draw_panel_d_umap(axes: np.ndarray, cells: pd.DataFrame) -> None:
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_cells = cells.loc[cells["endpoint"].eq(endpoint)]
        coordinate_rows = endpoint_cells.loc[endpoint_cells["map_id"].eq("truth")]
        x_span = max(float(coordinate_rows["umap_1"].max() - coordinate_rows["umap_1"].min()), 1e-6)
        y_span = max(float(coordinate_rows["umap_2"].max() - coordinate_rows["umap_2"].min()), 1e-6)
        xlim = (
            float(coordinate_rows["umap_1"].min() - 0.025 * x_span),
            float(coordinate_rows["umap_1"].max() + 0.025 * x_span),
        )
        ylim = (
            float(coordinate_rows["umap_2"].min() - 0.025 * y_span),
            float(coordinate_rows["umap_2"].max() + 0.025 * y_span),
        )
        for column, (map_id, display) in enumerate(PANEL_E_MAPS):
            ax = axes[row, column]
            frame = endpoint_cells.loc[endpoint_cells["map_id"].eq(map_id)]
            outside = frame.loc[~frame["is_evaluation_cohort"]]
            cohort = frame.loc[frame["is_evaluation_cohort"] & ~frame["is_selected"]]
            selected = frame.loc[frame["is_selected"]]
            ax.scatter(
                outside["umap_1"],
                outside["umap_2"],
                s=1.2,
                color=PANEL_E_CONTEXT_GRAY,
                alpha=0.55,
                linewidth=0,
                rasterized=True,
                zorder=1,
            )
            ax.scatter(
                cohort["umap_1"],
                cohort["umap_2"],
                s=2.0,
                color=PANEL_E_ELIGIBLE_GRAY,
                alpha=0.65,
                linewidth=0,
                rasterized=True,
                zorder=2,
            )
            selected_color = PANEL_E_TRUTH_COLOR if map_id == "truth" else METHOD_COLORS[map_id]
            ax.scatter(
                selected["umap_1"],
                selected["umap_2"],
                s=4.2,
                color=selected_color,
                alpha=0.95,
                linewidth=0,
                rasterized=True,
                zorder=3,
            )
            ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(
                    display,
                    fontsize=6.6,
                    pad=4,
                )
        axes[row, 0].set_ylabel(
            f"{ENDPOINT_SHORT[endpoint]} reference-omitted",
            fontsize=6.8,
            fontweight="bold",
            labelpad=5,
        )


def draw_panel_e_transfer(axes: Iterable[plt.Axes], by_seed: pd.DataFrame) -> None:
    axes = list(axes)
    positions = np.arange(len(PANEL_D_METHODS), dtype=float)
    offsets = (-0.15, 0.15)
    metrics = PANEL_D_METRICS
    display_by_method = {method: display for method, _, display in PANEL_D_METHODS}
    for facet, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS)):
        local = by_seed.loc[by_seed["endpoint"].eq(endpoint)]
        for method_index, (method, score, _) in enumerate(PANEL_D_METHODS):
            method_rows = local.loc[
                local["method"].eq(method) & local["score"].eq(score)
            ].sort_values("seed")
            for metric_index, (metric, _, color) in enumerate(metrics):
                values = method_rows[metric].to_numpy(dtype=float)
                mean = float(values.mean())
                sample_sd = float(values.std(ddof=1))
                center = positions[method_index] + offsets[metric_index]
                jitter = np.linspace(-0.035, 0.035, len(values))
                ax.scatter(
                    values,
                    center + jitter,
                    s=7,
                    color=color,
                    alpha=0.45,
                    linewidth=0,
                    zorder=3,
                )
                ax.errorbar(
                    mean,
                    center,
                    xerr=sample_sd,
                    fmt="o",
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=0.8,
                    ecolor=color,
                    elinewidth=0.7,
                    capsize=1.8,
                    zorder=4,
                )
        ax.set(
            xlim=(0.5, 1.0),
            ylim=(len(PANEL_D_METHODS) - 0.5, -0.5),
            yticks=positions,
            title={
                "B cells": "Stimulated B",
                "NK cells": "Stimulated NK",
                "Dendritic cells": "Stimulated DC",
            }[endpoint],
        )
        ax.set_xticks(np.arange(0.5, 1.01, 0.1))
        ax.title.set_fontsize(6.8)
        ax.title.set_fontweight("bold")
        if facet == 0:
            ax.set_yticklabels(
                [display_by_method[method] for method, _, _ in PANEL_D_METHODS],
                fontsize=5.8,
            )
        else:
            ax.set_yticklabels([])
        _style_axis(ax, grid_axis="x")
    axes[1].set_xlabel(
        "Metric value",
        fontsize=6.5,
    )
    _panel_heading(
        axes[0],
        "D",
        "",
        title_x=0.12,
        y=1.22,
    )
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markersize=4,
                linewidth=0.8,
                label=label,
            )
            for _, label, color in metrics
        ],
        frameon=False,
        fontsize=5.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        columnspacing=1.0,
    )


def draw_panel_f_assignments(axes: np.ndarray, cells: pd.DataFrame) -> None:
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_cells = cells.loc[cells["endpoint"].eq(endpoint)]
        coordinate_rows = endpoint_cells.loc[endpoint_cells["map_id"].eq("truth")]
        x_span = max(float(coordinate_rows["umap_1"].max() - coordinate_rows["umap_1"].min()), 1e-6)
        y_span = max(float(coordinate_rows["umap_2"].max() - coordinate_rows["umap_2"].min()), 1e-6)
        xlim = (
            float(coordinate_rows["umap_1"].min() - 0.025 * x_span),
            float(coordinate_rows["umap_1"].max() + 0.025 * x_span),
        )
        ylim = (
            float(coordinate_rows["umap_2"].min() - 0.025 * y_span),
            float(coordinate_rows["umap_2"].max() + 0.025 * y_span),
        )
        for column, (map_id, display) in enumerate(PANEL_F_MAPS):
            ax = axes[row, column]
            frame = endpoint_cells.loc[endpoint_cells["map_id"].eq(map_id)]
            represented = frame.loc[frame["is_represented"]]
            held_out = frame.loc[frame["is_held_out"]]
            colors = represented["displayed_assignment"].map(CELL_TYPE_COLORS)
            if colors.isna().any():
                raise PBMCFigure3Error("Panel F contains an unmapped cell type.")
            ax.scatter(
                represented["umap_1"],
                represented["umap_2"],
                c=colors,
                s=2.0,
                alpha=0.80,
                linewidth=0,
                rasterized=True,
                zorder=1,
            )
            ax.scatter(
                held_out["umap_1"],
                held_out["umap_2"],
                color=CELL_TYPE_COLORS["Held-out state (not evaluated)"],
                s=3.5,
                alpha=1.0,
                linewidth=0,
                rasterized=True,
                zorder=2,
            )
            ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(
                    display,
                    fontsize=6.6,
                    pad=4,
                )
        axes[row, 0].set_ylabel(
            f"{ENDPOINT_SHORT[endpoint]} reference-omitted",
            fontsize=6.8,
            fontweight="bold",
            labelpad=5,
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=4,
            label=label,
        )
        for label, color in CELL_TYPE_COLORS.items()
    ]
    axes[-1, 2].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=5.0,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=0.8,
        handletextpad=0.25,
    )


def _draw_composite_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    *,
    title_x: float = 0.13,
) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.70,
        letter,
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        title_x,
        0.70,
        title,
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
    )


def draw_panel_f_compact(ax: plt.Axes, by_seed: pd.DataFrame) -> None:
    endpoint_x = np.arange(len(ENDPOINTS))
    method_offsets = np.linspace(-0.28, 0.28, len(PANEL_F_METHODS))
    seed_jitter = np.linspace(-0.022, 0.022, len(SEEDS))
    for method_index, (method, score, display) in enumerate(PANEL_F_METHODS):
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            values = by_seed.loc[
                by_seed["endpoint"].eq(endpoint)
                & by_seed["method"].eq(method)
                & by_seed["score"].eq(score)
            ].sort_values("seed")
            center = endpoint_index + method_offsets[method_index]
            ax.scatter(
                center + seed_jitter,
                values["forced_macro_f1"],
                s=7,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                alpha=0.5,
                linewidth=0,
                zorder=3,
            )
            mean = float(values["forced_macro_f1"].mean())
            sample_sd = float(values["forced_macro_f1"].std(ddof=1))
            ax.errorbar(
                center,
                mean,
                yerr=sample_sd,
                fmt=METHOD_MARKERS[method],
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor=METHOD_COLORS[method],
                markeredgewidth=0.9,
                ecolor=METHOD_COLORS[method],
                elinewidth=0.7,
                capsize=1.5,
                zorder=5,
            )
    ax.set(
        xlim=(-0.52, len(ENDPOINTS) - 0.48),
        ylim=(0, 1.02),
        xticks=endpoint_x,
        xticklabels=["B", "NK", "DC"],
        ylabel="Forced macro-F1",
    )
    ax.set_xlabel("Reference-omitted stimulated state", fontsize=5.8)
    ax.set_ylabel("Forced macro-F1", fontsize=5.8)
    _style_axis(ax)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKERS[method],
                color="none",
                markerfacecolor="white",
                markeredgecolor=METHOD_COLORS[method],
                markersize=3.7,
                label=display,
            )
            for method, _, display in PANEL_F_METHODS
        ],
        frameon=False,
        fontsize=5.0,
        loc="lower left",
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.25,
        labelspacing=0.25,
        borderaxespad=0.2,
    )


def _panel_outputs(letter: str, docs_root: Path) -> dict[str, Path]:
    stem = f"figure_3_pbmc_panel_{letter.lower()}"
    legacy_svg = docs_root / f"{stem}.svg"
    if legacy_svg.is_file():
        legacy_svg.unlink()
    return {suffix: docs_root / f"{stem}.{suffix}" for suffix in ("png", "pdf", "tiff")}


def _save_figure(fig: plt.Figure, outputs: dict[str, Path]) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "dpi": PANEL_RASTER_DPI if suffix in {"png", "tiff"} else None,
            "facecolor": "white",
        }
        if suffix == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **save_kwargs)


def _validate_raster(path: Path, size_inches: tuple[float, float]) -> None:
    with Image.open(path) as image:
        expected = (
            round(size_inches[0] * PANEL_RASTER_DPI),
            round(size_inches[1] * PANEL_RASTER_DPI),
        )
        if image.width < expected[0] - 4 or image.height < expected[1] - 4:
            raise PBMCFigure3Error(
                f"Raster is below the requested size: {path} has {image.size}, expected {expected}."
            )


def _write_panel_manifest(
    *,
    letter: str,
    result_root: Path,
    artifacts: dict[str, str],
    sources: list[str],
    parameters: dict[str, object],
) -> Path:
    path = result_root / f"panel_{letter.lower()}_manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "PBMC condition-specific weak correspondence Figure 3",
                "panel": letter,
                "generator": ("experiments/pbmc_state/generate_pbmc_figure3_panels.py"),
                "artifacts": artifacts,
                "sources": sources,
                "parameters": parameters,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _validate_required(paths: Iterable[Path], label: str) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise PBMCFigure3Error(f"Missing or empty {label} artifact: {path}")


def generate_panel_a(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
) -> dict[str, Path]:
    source_root = result_root / "figure_3_pbmc_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    source_path = source_root / "panel_a_design.csv"
    pd.DataFrame(
        [
            {
                "field": "displayed_endpoints",
                "value": "B cells | NK cells | Dendritic cells",
                "source_path": "docs/state_pbmc.md",
            },
            {
                "field": "split",
                "value": "donor-aware; query fraction 0.25; same query cells paired",
                "source_path": str(BENCHMARK_CONFIG_PATH),
            },
            {
                "field": "ablated_reference",
                "value": (
                    "remove stimulated cells of one target cell type; retain "
                    "same-type controls and other stimulated cell types"
                ),
                "source_path": str(BENCHMARK_CONFIG_PATH),
            },
            {
                "field": "full_reference",
                "value": "restore the omitted stimulated target state",
                "source_path": str(BENCHMARK_CONFIG_PATH),
            },
            {
                "field": "primary_evaluation",
                "value": (
                    "held-out stimulated query cells versus control query cells "
                    "of the same cell type"
                ),
                "source_path": ("docs/manuscript_figure3_pbmc_condition_specific_states.md"),
            },
        ]
    ).to_csv(source_path, index=False)

    outputs = _panel_outputs("A", docs_root)
    fig, ax = plt.subplots(figsize=PANEL_SIZES["A"])
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.98)
    draw_panel_a(ax, compact=False)
    _save_figure(fig, outputs)
    plt.close(fig)
    manifest = _write_panel_manifest(
        letter="A",
        result_root=result_root,
        artifacts={
            **{suffix: str(path) for suffix, path in outputs.items()},
            "source_data": str(source_path),
        },
        sources=[
            str(BENCHMARK_CONFIG_PATH),
            "docs/state_pbmc.md",
            "docs/manuscript_figure3_pbmc_condition_specific_states.md",
        ],
        parameters={
            "endpoints": list(ENDPOINTS),
            "layout": "condition_by_cell_type_schematic",
            "canvas_inches": list(PANEL_SIZES["A"]),
            "raster_dpi": PANEL_RASTER_DPI,
        },
    )
    _validate_required([*outputs.values(), source_path, manifest], "Panel A")
    _validate_raster(outputs["png"], PANEL_SIZES["A"])
    return {**outputs, "source_data": source_path, "manifest": manifest}


def generate_panel_b(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Path]:
    by_seed, summary, source_paths = collect_panel_b_data(
        comparison_path=comparison_path,
        raw_data_path=raw_data_path,
        runs_root=runs_root,
    )
    source_root = result_root / "figure_3_pbmc_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    by_seed_path = source_root / "figure_3_within_celltype_detection_by_seed.csv"
    summary_path = source_root / "figure_3_within_celltype_detection_summary.csv"
    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    outputs = _panel_outputs("B", docs_root)
    fig, axes = plt.subplots(1, 3, figsize=PANEL_SIZES["B"])
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.78, wspace=0.12)
    draw_panel_b(axes, by_seed, heading_y=1.28)
    _save_figure(fig, outputs)
    plt.close(fig)
    manifest = _write_panel_manifest(
        letter="B",
        result_root=result_root,
        artifacts={
            **{suffix: str(path) for suffix, path in outputs.items()},
            "by_seed": str(by_seed_path),
            "summary": str(summary_path),
        },
        sources=source_paths,
        parameters={
            "endpoints": list(ENDPOINTS),
            "seeds": list(SEEDS),
            "methods": [
                {
                    "method": method,
                    "score": score,
                    "display": "Seurat" if method == "seurat_anchor" else display,
                }
                for method, score, display in PANEL_B_METHODS
            ],
            "method_colors": {method: METHOD_COLORS[method] for method, _, _ in PANEL_B_METHODS},
            "cohort": ("reference-omitted stimulated state versus same-type control cells"),
            "metric_implementation": "sklearn.metrics.average_precision_score",
            "artifact_field": "auprc",
            "display_label": "AP",
            "display_axis": list(PANEL_B_AP_LIMITS),
            "rendered_axis": list(PANEL_B_AP_DISPLAY_LIMITS),
            "display_ticks": list(PANEL_B_AP_TICKS),
            "axis_truncation": (
                "display_only; source AP values are unchanged; "
                "unlabeled right padding prevents clipping at 1.0"
            ),
            "interval": "mean_plus_or_minus_sample_sd_ddof1",
            "canvas_inches": list(PANEL_SIZES["B"]),
            "raster_dpi": PANEL_RASTER_DPI,
        },
    )
    _validate_required(
        [*outputs.values(), by_seed_path, summary_path, manifest],
        "Panel B",
    )
    _validate_raster(outputs["png"], PANEL_SIZES["B"])
    return {
        **outputs,
        "by_seed": by_seed_path,
        "summary": summary_path,
        "manifest": manifest,
    }


def generate_panel_c(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Path]:
    cells, by_seed, source_paths = collect_panel_d_data(
        comparison_path=comparison_path,
        raw_data_path=raw_data_path,
        runs_root=runs_root,
    )
    source_root = result_root / "figure_3_pbmc_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    cells_path = source_root / "figure_3_matched_reference_by_cell.parquet"
    by_seed_path = source_root / "figure_3_matched_reference_by_seed.csv"
    cells.to_parquet(cells_path, index=False)
    by_seed.to_csv(by_seed_path, index=False)

    outputs = _panel_outputs("C", docs_root)
    fig = plt.figure(figsize=PANEL_SIZES["C"])
    grid = fig.add_gridspec(
        2,
        3,
        left=0.16,
        right=0.99,
        bottom=0.24,
        top=0.80,
        width_ratios=(1.2, 1.2, 1.0),
        hspace=0.55,
        wspace=0.55,
    )
    deficit_ax = fig.add_subplot(grid[:, :2])
    specificity_ax = fig.add_subplot(grid[0, 2])
    destination_ax = fig.add_subplot(grid[1, 2])
    draw_panel_d(
        deficit_ax,
        specificity_ax,
        destination_ax,
        by_seed,
        letter="C",
    )
    _save_figure(fig, outputs)
    plt.close(fig)
    manifest = _write_panel_manifest(
        letter="C",
        result_root=result_root,
        artifacts={
            **{suffix: str(path) for suffix, path in outputs.items()},
            "by_cell": str(cells_path),
            "by_seed": str(by_seed_path),
        },
        sources=source_paths,
        parameters={
            "endpoints": list(ENDPOINTS),
            "seeds": list(SEEDS),
            "score": "raw_query_marginal_deficit_u",
            "restoration_specificity": CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE,
            "restored_destination": TYPICAL_CELL_CONDITIONAL_DESTINATION,
            "probability_ticks": [0.0, 0.5, 1.0],
            "probability_rendered_axis": list(PANEL_C_PROBABILITY_DISPLAY_LIMITS),
            "probability_right_padding": (
                "unlabeled padding prevents clipping at the 1.0 boundary"
            ),
            "interpretation": (
                "matched_full_reference_response_with_condition_specific_provider_refit"
            ),
            "canvas_inches": list(PANEL_SIZES["C"]),
            "raster_dpi": PANEL_RASTER_DPI,
        },
    )
    _validate_required(
        [*outputs.values(), cells_path, by_seed_path, manifest],
        "Panel C",
    )
    _validate_raster(outputs["png"], PANEL_SIZES["C"])
    return {
        **outputs,
        "by_cell": cells_path,
        "by_seed": by_seed_path,
        "manifest": manifest,
    }


def generate_panel_e(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Path]:
    cells, summary, source_paths = collect_panel_d_umap_data(
        comparison_path=comparison_path,
        raw_data_path=raw_data_path,
        runs_root=runs_root,
    )
    source_root = result_root / "figure_3_pbmc_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    cells_path = source_root / "figure_3_weak_support_umap_cells.parquet"
    summary_path = source_root / "figure_3_weak_support_umap_summary.csv"
    cells.to_parquet(cells_path, index=False)
    summary.to_csv(summary_path, index=False)

    outputs = _panel_outputs("E", docs_root)
    fig, axes = plt.subplots(
        3,
        7,
        figsize=PANEL_SIZES["E"],
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.14,
        top=0.80,
        wspace=0.04,
        hspace=0.06,
    )
    fig.text(0.012, 0.975, "E", fontsize=10.5, fontweight="bold", va="top")
    draw_panel_d_umap(axes, cells)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=PANEL_E_TRUTH_COLOR,
                markeredgecolor="none",
                markersize=4,
                label="Reference-omitted cells",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=METHOD_COLORS["coreot_full"],
                markeredgecolor="none",
                markersize=4,
                label=r"Method-specific top-ranked $N_+$ cells",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=PANEL_E_ELIGIBLE_GRAY,
                markeredgecolor="none",
                markersize=4,
                label="Other ranking-cohort cells",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=PANEL_E_CONTEXT_GRAY,
                markeredgecolor="none",
                markersize=4,
                label="Outside ranking cohort",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=5.5,
        columnspacing=1.2,
        handletextpad=0.35,
    )
    _save_figure(fig, outputs)
    plt.close(fig)
    manifest = _write_panel_manifest(
        letter="E",
        result_root=result_root,
        artifacts={
            **{suffix: str(path) for suffix, path in outputs.items()},
            "by_cell": str(cells_path),
            "summary": str(summary_path),
        },
        sources=source_paths,
        parameters={
            "endpoints": list(ENDPOINTS),
            "seed": PANEL_D_SEED,
            "maps": [
                {
                    "map_id": map_id,
                    "display": "Seurat" if map_id == "seurat_anchor" else display,
                }
                for map_id, display in PANEL_E_MAPS
            ],
            "method_colors": {method: METHOD_COLORS[method] for method, _, _ in PANEL_E_METHODS},
            "truth_color": PANEL_E_TRUTH_COLOR,
            "eligible_cohort_color": PANEL_E_ELIGIBLE_GRAY,
            "outside_cohort_color": PANEL_E_CONTEXT_GRAY,
            "selection": "top_N_l_within_same_celltype_cohort",
            "tie_break": "score_descending_then_cell_id_ascending",
            "coordinates": str(FIXED_QUERY_UMAP_PATH),
            "canvas_inches": list(PANEL_SIZES["E"]),
            "raster_dpi": PANEL_RASTER_DPI,
        },
    )
    _validate_required(
        [*outputs.values(), cells_path, summary_path, manifest],
        "Panel E",
    )
    _validate_raster(outputs["png"], PANEL_SIZES["E"])
    return {
        **outputs,
        "by_cell": cells_path,
        "summary": summary_path,
        "manifest": manifest,
    }


def generate_panel_d(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    label_transfer_path: Path = COMPARISON_LABEL_TRANSFER_PATH,
) -> dict[str, Path]:
    by_seed, summary, source_paths = collect_panel_f_data(
        label_transfer_path=label_transfer_path,
    )
    source_root = result_root / "figure_3_pbmc_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    by_seed_path = source_root / "figure_3_represented_celltype_transfer_by_seed.csv"
    summary_path = source_root / "figure_3_represented_celltype_transfer_summary.csv"
    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    outputs = _panel_outputs("D", docs_root)
    fig, axes = plt.subplots(1, 3, figsize=PANEL_SIZES["D"])
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.77, wspace=0.12)
    draw_panel_e_transfer(axes, by_seed)
    _save_figure(fig, outputs)
    plt.close(fig)
    manifest = _write_panel_manifest(
        letter="D",
        result_root=result_root,
        artifacts={
            **{suffix: str(path) for suffix, path in outputs.items()},
            "by_seed": str(by_seed_path),
            "summary": str(summary_path),
        },
        sources=source_paths,
        parameters={
            "endpoints": list(ENDPOINTS),
            "seeds": list(SEEDS),
            "condition": INCOMPLETE_REFERENCE,
            "methods": [
                {
                    "method": method,
                    "score": score,
                    "display": "Seurat" if method == "seurat_anchor" else display,
                }
                for method, score, display in PANEL_D_METHODS
            ],
            "metrics": ["forced_macro_f1", "forced_accuracy"],
            "metric_colors": {metric: color for metric, _, color in PANEL_D_METRICS},
            "axis": list(PANEL_D_LIM),
            "encoding": ("five_donor_split_points_plus_arithmetic_mean_and_sample_sd_interval"),
            "split_points": "five_donor_split_values",
            "mean": "arithmetic_mean",
            "interval": "mean_plus_or_minus_sample_sd_ddof1",
            "bar_length_encoding": False,
            "evaluation_population": "represented_cell_types_only",
            "canvas_inches": list(PANEL_SIZES["D"]),
            "raster_dpi": PANEL_RASTER_DPI,
        },
    )
    _validate_required(
        [*outputs.values(), by_seed_path, summary_path, manifest],
        "Panel D",
    )
    _validate_raster(outputs["png"], PANEL_SIZES["D"])
    return {
        **outputs,
        "by_seed": by_seed_path,
        "summary": summary_path,
        "manifest": manifest,
    }


def generate_panel_f(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    label_transfer_path: Path = COMPARISON_LABEL_TRANSFER_PATH,
    runs_root: Path = RUNS_ROOT,
    umap_path: Path = FIXED_QUERY_UMAP_PATH,
) -> dict[str, Path]:
    cells, summary, source_paths = collect_panel_f_assignment_data(
        label_transfer_path=label_transfer_path,
        runs_root=runs_root,
        umap_path=umap_path,
    )
    source_root = result_root / "figure_3_pbmc_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    cells_path = source_root / "figure_3_label_assignment_umap_cells.parquet"
    summary_path = source_root / "figure_3_label_assignment_umap_summary.csv"
    cells.to_parquet(cells_path, index=False)
    summary.to_csv(summary_path, index=False)

    outputs = _panel_outputs("F", docs_root)
    fig, axes = plt.subplots(3, 6, figsize=PANEL_SIZES["F"])
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.20,
        top=0.80,
        wspace=0.04,
        hspace=0.06,
    )
    fig.text(0.012, 0.975, "F", fontsize=10.5, fontweight="bold", va="top")
    draw_panel_f_assignments(axes, cells)
    _save_figure(fig, outputs)
    plt.close(fig)
    manifest = _write_panel_manifest(
        letter="F",
        result_root=result_root,
        artifacts={
            **{suffix: str(path) for suffix, path in outputs.items()},
            "by_cell": str(cells_path),
            "summary": str(summary_path),
        },
        sources=source_paths,
        parameters={
            "endpoints": list(ENDPOINTS),
            "seed": PANEL_D_SEED,
            "maps": [
                {
                    "map_id": map_id,
                    "display": "Seurat" if map_id == "seurat_anchor" else display,
                }
                for map_id, display in PANEL_F_MAPS
            ],
            "coordinates": str(umap_path),
            "evaluation_population": "represented_query_cells_only",
            "reference_omitted_display": "neutral_gray_not_evaluated",
            "cell_type_colors": CELL_TYPE_COLORS,
            "label_contract": "cell_type_not_cell_type_by_condition",
            "panel_d_crosscheck": ["forced_macro_f1", "forced_accuracy"],
            "canvas_inches": list(PANEL_SIZES["F"]),
            "raster_dpi": PANEL_RASTER_DPI,
        },
    )
    _validate_required(
        [*outputs.values(), cells_path, summary_path, manifest],
        "Panel F",
    )
    _validate_raster(outputs["png"], PANEL_SIZES["F"])
    return {
        **outputs,
        "by_cell": cells_path,
        "summary": summary_path,
        "manifest": manifest,
    }


def generate_destination_support_supplement(
    *,
    result_root: Path = RESULT_ROOT,
    comparison_path: Path = COMPARISON_DETECTION_PATH,
    raw_data_path: Path = RAW_DATA_PATH,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Path]:
    cells, summary, source_paths = collect_panel_e_data(
        comparison_path=comparison_path,
        raw_data_path=raw_data_path,
        runs_root=runs_root,
    )
    supplement_root = result_root.parent / "supp_destination_support"
    supplement_root.mkdir(parents=True, exist_ok=True)
    cells_path = supplement_root / "destination_support_by_cell.parquet"
    summary_path = supplement_root / "destination_support_by_seed.csv"
    cells.to_parquet(cells_path, index=False)
    summary.to_csv(summary_path, index=False)

    outputs = {
        suffix: supplement_root / f"destination_support.{suffix}"
        for suffix in ("png", "pdf", "tiff")
    }
    legacy_svg = supplement_root / "destination_support.svg"
    if legacy_svg.is_file():
        legacy_svg.unlink()
    fig, axes = plt.subplots(1, 3, figsize=PANEL_SIZES["E"])
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.78, wspace=0.16)
    draw_panel_e(axes, cells, summary, include_heading=False)
    fig.text(
        0.02,
        0.95,
        "Reference destination versus fitted mass retention",
        fontsize=8.0,
        fontweight="bold",
        va="top",
    )
    _save_figure(fig, outputs)
    plt.close(fig)
    manifest = _write_panel_manifest(
        letter="supp_destination_support",
        result_root=supplement_root,
        artifacts={
            **{suffix: str(path) for suffix, path in outputs.items()},
            "by_cell": str(cells_path),
            "by_seed": str(summary_path),
        },
        sources=source_paths,
        parameters={
            "status": "supplementary_candidate",
            "endpoints": list(ENDPOINTS),
            "seeds": list(SEEDS),
            "condition": INCOMPLETE_REFERENCE,
            "score": "raw_query_marginal_deficit_u",
            "same_celltype_destination": (
                "sum_coupling_to_target_cell_type_divided_by_source_a_hat"
            ),
            "destination_eligibility": f"a_hat > {TRANSPORT_ETA}",
            "canvas_inches": list(PANEL_SIZES["E"]),
            "raster_dpi": PANEL_RASTER_DPI,
        },
    )
    _validate_required(
        [*outputs.values(), cells_path, summary_path, manifest],
        "destination-support supplement",
    )
    _validate_raster(outputs["png"], PANEL_SIZES["E"])
    return {
        **outputs,
        "by_cell": cells_path,
        "by_seed": summary_path,
        "manifest": manifest,
    }


def _read_composite_sources(
    source_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    panel_b_path = source_root / "figure_3_within_celltype_detection_by_seed.csv"
    panel_c_path = source_root / "figure_3_within_celltype_internal_contrasts_by_seed.csv"
    panel_d_path = source_root / "figure_3_matched_reference_by_seed.csv"
    panel_e_cells_path = source_root / "figure_3_destination_support_by_cell.parquet"
    panel_e_summary_path = source_root / "figure_3_destination_support_by_seed.csv"
    panel_f_path = source_root / "figure_3_represented_celltype_transfer_by_seed.csv"

    panel_b = pd.read_csv(panel_b_path)
    panel_c = pd.read_csv(panel_c_path)
    panel_d = pd.read_csv(panel_d_path)
    panel_e_cells = pd.read_parquet(panel_e_cells_path)
    panel_e_summary = pd.read_csv(panel_e_summary_path)
    panel_f = pd.read_csv(panel_f_path)

    _require_exact_keys(
        panel_b,
        columns=["endpoint", "seed", "method", "score"],
        expected={
            (endpoint, seed, method, score)
            for endpoint in ENDPOINTS
            for seed in SEEDS
            for method, score, _ in PANEL_B_METHODS
        },
        label="Composite Panel B",
    )
    _require_exact_keys(
        panel_c,
        columns=["endpoint", "seed"],
        expected={(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS},
        label="Composite Panel C",
    )
    _require_exact_keys(
        panel_d,
        columns=["endpoint", "seed"],
        expected={(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS},
        label="Composite Panel D",
    )
    _require_exact_keys(
        panel_e_summary,
        columns=["endpoint", "seed", "truth_group"],
        expected={
            (endpoint, seed, group)
            for endpoint in ENDPOINTS
            for seed in SEEDS
            for group in ("held_out_stimulated", "same_type_control")
        },
        label="Composite Panel E",
    )
    _require_exact_keys(
        panel_f,
        columns=["endpoint", "seed", "method", "score"],
        expected={
            (endpoint, seed, method, score)
            for endpoint in ENDPOINTS
            for seed in SEEDS
            for method, score, _ in PANEL_F_METHODS
        },
        label="Composite Panel F",
    )
    if panel_e_cells.empty:
        raise PBMCFigure3Error("Composite Panel E cell source is empty.")
    return panel_b, panel_c, panel_d, panel_e_cells, panel_e_summary, panel_f


def _draw_main_figure(
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e_cells: pd.DataFrame,
    panel_e_summary: pd.DataFrame,
    panel_f: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
    outer = fig.add_gridspec(
        3,
        12,
        left=0.07,
        right=0.99,
        bottom=0.055,
        top=0.99,
        height_ratios=(0.27, 0.40, 0.33),
        hspace=0.28,
        wspace=0.65,
    )

    panel_a_grid = outer[0, :4].subgridspec(
        2,
        1,
        height_ratios=(0.13, 0.87),
        hspace=0.02,
    )
    panel_a_header = fig.add_subplot(panel_a_grid[0, 0])
    _draw_composite_header(panel_a_header, "A", "PBMC reference removal", title_x=0.16)
    panel_a_ax = fig.add_subplot(panel_a_grid[1, 0])
    draw_panel_a(panel_a_ax, compact=True, include_heading=False)

    panel_b_outer = outer[0, 4:].subgridspec(
        2,
        1,
        height_ratios=(0.13, 0.87),
        hspace=0.02,
    )
    panel_b_header = fig.add_subplot(panel_b_outer[0, 0])
    _draw_composite_header(panel_b_header, "B", "External detection", title_x=0.075)
    panel_b_grid = panel_b_outer[1, 0].subgridspec(1, 3, wspace=0.18)
    panel_b_axes = [fig.add_subplot(panel_b_grid[0, index]) for index in range(3)]
    draw_panel_b(panel_b_axes, panel_b, include_heading=False, compact=True)

    panel_c_grid = outer[1, :4].subgridspec(
        2,
        1,
        height_ratios=(0.12, 0.88),
        hspace=0.02,
    )
    panel_c_header = fig.add_subplot(panel_c_grid[0, 0])
    _draw_composite_header(
        panel_c_header,
        "C",
        "Transport and prior controls",
        title_x=0.16,
    )
    panel_c_ax = fig.add_subplot(panel_c_grid[1, 0])
    draw_panel_c(panel_c_ax, panel_c, include_heading=False)

    panel_d_outer = outer[1, 4:].subgridspec(
        2,
        1,
        height_ratios=(0.12, 0.88),
        hspace=0.02,
    )
    panel_d_header = fig.add_subplot(panel_d_outer[0, 0])
    _draw_composite_header(
        panel_d_header,
        "D",
        "Paired reference-restoration response",
        title_x=0.075,
    )
    panel_d_grid = panel_d_outer[1, 0].subgridspec(
        2,
        3,
        width_ratios=(1.2, 1.2, 1.0),
        hspace=0.58,
        wspace=0.72,
    )
    panel_d_deficit_ax = fig.add_subplot(panel_d_grid[:, :2])
    panel_d_specificity_ax = fig.add_subplot(panel_d_grid[0, 2])
    panel_d_destination_ax = fig.add_subplot(panel_d_grid[1, 2])
    draw_panel_d(
        panel_d_deficit_ax,
        panel_d_specificity_ax,
        panel_d_destination_ax,
        panel_d,
        include_heading=False,
        compact=True,
    )

    panel_e_outer = outer[2, :6].subgridspec(
        2,
        1,
        height_ratios=(0.13, 0.87),
        hspace=0.02,
    )
    panel_e_header = fig.add_subplot(panel_e_outer[0, 0])
    _draw_composite_header(
        panel_e_header,
        "E",
        "Reference destination versus fitted mass retention",
        title_x=0.10,
    )
    panel_e_grid = panel_e_outer[1, 0].subgridspec(1, 3, wspace=0.18)
    panel_e_axes = [fig.add_subplot(panel_e_grid[0, index]) for index in range(3)]
    draw_panel_e(
        panel_e_axes,
        panel_e_cells,
        panel_e_summary,
        include_heading=False,
        compact=True,
    )

    panel_f_grid = outer[2, 6:].subgridspec(
        2,
        1,
        height_ratios=(0.13, 0.87),
        hspace=0.02,
    )
    panel_f_header = fig.add_subplot(panel_f_grid[0, 0])
    _draw_composite_header(
        panel_f_header,
        "F",
        "",
        title_x=0.10,
    )
    panel_f_ax = fig.add_subplot(panel_f_grid[1, 0])
    draw_panel_f_compact(panel_f_ax, panel_f)
    return fig, {
        "A": [panel_a_header, panel_a_ax],
        "B": [panel_b_header, *panel_b_axes],
        "C": [panel_c_header, panel_c_ax],
        "D": [
            panel_d_header,
            panel_d_deficit_ax,
            panel_d_specificity_ax,
            panel_d_destination_ax,
        ],
        "E": [panel_e_header, *panel_e_axes],
        "F": [panel_f_header, panel_f_ax],
    }


def _panel_crop_bbox_inches(
    fig: plt.Figure,
    axes: Iterable[plt.Axes],
    *,
    padding_inches: float = 0.04,
) -> Bbox:
    renderer = fig.canvas.get_renderer()
    tight_boxes = [
        box
        for ax in axes
        if ax.get_visible()
        for box in [ax.get_tightbbox(renderer)]
        if box is not None
    ]
    if not tight_boxes:
        raise PBMCFigure3Error("Cannot crop a panel without visible axes.")
    combined = Bbox.union(tight_boxes).transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(
        max(0.0, combined.x0 - padding_inches),
        max(0.0, combined.y0 - padding_inches),
        min(fig.get_figwidth(), combined.x1 + padding_inches),
        min(fig.get_figheight(), combined.y1 + padding_inches),
    )


def _validate_review_crop(path: Path) -> None:
    with Image.open(path) as image:
        if image.width < 400 or image.height < 300:
            raise PBMCFigure3Error(
                f"Review crop is too small for panel inspection: {path} "
                f"({image.width}x{image.height})."
            )
        pixels = np.asarray(image.convert("RGB"))
    if not np.any(pixels < 245):
        raise PBMCFigure3Error(f"Review crop contains no visible content: {path}.")


def _validate_main_fonts(fig: plt.Figure) -> None:
    sizes = [float(text.get_fontsize()) for text in fig.findobj(match=Text) if text.get_text()]
    if not sizes or min(sizes) < MAIN_FIGURE_MIN_FONT_SIZE:
        raise PBMCFigure3Error(
            "Main Figure 3 contains text below the final-size font floor; "
            f"minimum={min(sizes) if sizes else None}, required={MAIN_FIGURE_MIN_FONT_SIZE}."
        )


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_versions() -> dict[str, str]:
    packages = (
        "anndata",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "PyYAML",
    )
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _caption_text() -> str:
    return (
        "**Figure 3. PBMC condition-specific reference omission and restoration.** "
        "**(A) Controlled benchmark design.** Across five donor splits, stimulated B, "
        "NK, and dendritic cells are analyzed in separate reference-omission scenarios. "
        "The same query cells retain all states and the same reference donors are used; "
        "the selected stimulated state is omitted in the reference-omitted condition "
        "and restored in the paired restored-reference condition, while same-type "
        "controls remain represented. **(B) Paired reference-restoration response.** "
        "Bars show the decreases in median query-marginal deficit for reference-omitted "
        "cells and same-type controls, the control-adjusted decrease "
        "\\(\\Delta^{\\mathrm{CA}}\\), and the restored-state destination fraction. "
        "The latter is the within-split median, among reference-omitted cells satisfying "
        "\\(\\widehat a_{q,i}>\\eta\\) with \\(\\eta=10^{-12}\\), of the fraction "
        "of transported mass assigned "
        "to the restored state. **(C) Within-cell-type omitted-state ranking.** AP and "
        "AUROC compare each omitted stimulated state with same-type controls. "
        "**(D) Forced cell-type label transfer among represented states.** Forced "
        "accuracy and forced macro-F1 are evaluated only for cells whose states remain "
        "represented. Panels B--D use a common summary convention. Bar lengths are "
        "arithmetic means across five donor splits; whiskers show one sample standard "
        "deviation. Donor-split points are omitted, and all metric axes start at zero; "
        "all four quantities in Panel B share a 0--1 metric-value axis. In Panel C, "
        "solid bars show AP and hollow bars show AUROC; in Panel D, solid and hollow "
        "bars show forced accuracy and forced macro-F1, respectively. Dashed lines in "
        "Panel C show mean prevalence for AP; no AUROC 0.5 "
        "reference is shown. Method colors in Panels C and D match Panel E. "
        "**(E) Spatial localization of top-ranked cells.** For donor split 1, the "
        "top-ranked \\(N_+\\) cells in each cell-type ranking cohort are shown on fixed "
        "query coordinates for every Panel C method, where \\(N_+\\) is the number of "
        "reference-omitted cells. Gray layers indicate selection status rather than "
        "represented-state truth. **(F) Spatial view of forced cell-type assignments "
        "among represented states.** Evaluation labels and method-specific assignments "
        "among represented cell-type labels are shown on the corresponding coordinates "
        "for donor split 1; row labels identify the reference-omission scenario. "
        "Reference-omitted cells are gray and excluded from label-transfer evaluation.\n"
    )


def generate_review_panels(
    *,
    review_root: Path = REVIEW_ROOT,
    source_root: Path = SOURCE_DATA_ROOT,
    docs_root: Path = DOCS_ROOT,
) -> dict[str, Path]:
    del source_root
    readme_path = review_root / "README.md"
    if not readme_path.is_file():
        raise PBMCFigure3Error(
            "The durable panel-review log is missing. Create "
            f"{readme_path} before regenerating review PNGs."
        )
    outputs: dict[str, Path] = {}
    for letter in "ABCDEF":
        source_path = _panel_outputs(letter, docs_root)["png"]
        if not source_path.is_file():
            raise PBMCFigure3Error(f"Generate standalone Panel {letter} before its review image.")
        output_path = review_root / f"figure_3_pbmc_panel_{letter.lower()}.png"
        with Image.open(source_path) as image:
            image.convert("RGB").save(output_path)
        _validate_review_crop(output_path)
        outputs[f"panel_{letter.lower()}"] = output_path
    return {**outputs, "readme": readme_path}


def generate_main_figure(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    source_root: Path = SOURCE_DATA_ROOT,
) -> dict[str, Path]:
    (
        panel_b,
        panel_c,
        panel_d,
        panel_e_cells,
        panel_e_summary,
        panel_f,
    ) = _read_composite_sources(source_root)
    result_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    result_outputs = {
        suffix: result_root / f"figure_3_pbmc_main.{suffix}" for suffix in ("png", "pdf", "tiff")
    }
    docs_figure_root = docs_root / "figs"
    docs_figure_root.mkdir(parents=True, exist_ok=True)
    docs_outputs = {
        suffix: docs_figure_root / f"manuscript_fig_pbmc_main.{suffix}"
        for suffix in ("png", "pdf", "tiff")
    }
    for legacy_svg in (
        result_root / "figure_3_pbmc_main.svg",
        docs_figure_root / "manuscript_fig_pbmc_main.svg",
    ):
        if legacy_svg.is_file():
            legacy_svg.unlink()

    fig, _ = _draw_main_figure(
        panel_b,
        panel_c,
        panel_d,
        panel_e_cells,
        panel_e_summary,
        panel_f,
    )
    _validate_main_fonts(fig)
    _save_figure(fig, result_outputs)
    _save_figure(fig, docs_outputs)
    plt.close(fig)

    caption_path = result_root / "figure_3_pbmc_caption.md"
    docs_caption_path = docs_root / "manuscript_caption_pbmc.md"
    caption = _caption_text()
    caption_path.write_text(caption, encoding="utf-8")
    docs_caption_path.write_text(caption, encoding="utf-8")

    source_paths = sorted(path for path in source_root.iterdir() if path.is_file())
    run_ids = sorted(
        set(panel_b["run_id"].astype(str))
        | set(panel_c["run_id"].astype(str))
        | set(panel_d["run_id"].astype(str))
        | set(panel_f["run_id"].astype(str))
    )
    plotting_script = Path(__file__)
    manifest_path = result_root / "figure_3_pbmc_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "PBMC condition-specific weak correspondence Figure 3",
                "generator": ("experiments/pbmc_state/generate_pbmc_figure3_panels.py"),
                "artifacts": {
                    "result": {suffix: str(path) for suffix, path in result_outputs.items()},
                    "manuscript": {suffix: str(path) for suffix, path in docs_outputs.items()},
                    "caption": str(caption_path),
                    "manuscript_caption": str(docs_caption_path),
                    "source_data_directory": str(source_root),
                },
                "input_tables": [
                    str(COMPARISON_DETECTION_PATH),
                    str(COMPARISON_LABEL_TRANSFER_PATH),
                ],
                "run_identifiers": run_ids,
                "endpoint_filters": {
                    "included": list(ENDPOINTS),
                    "excluded_main_figure": ["CD8 T cells"],
                    "detection_positive": ("condition == stim and cell_type == held_out endpoint"),
                    "detection_negative": ("condition == ctrl and cell_type == held_out endpoint"),
                    "unrelated_cell_types": "excluded from Panels B and C",
                },
                "metric_implementation": {
                    "average_precision": (
                        "sklearn.metrics.average_precision_score; stored as auprc"
                    ),
                    "auroc": "sklearn.metrics.roc_auc_score",
                    "restoration_specificity": (
                        "(median heldout u_ablated - median heldout u_full) - "
                        "(median control u_ablated - median control u_full)"
                    ),
                    "conditional_destination": (
                        "coupling mass to selected target labels divided by a_hat "
                        f"for cells with a_hat > {TRANSPORT_ETA}"
                    ),
                    "label_transfer": ("forced macro-F1 on represented cell-type labels"),
                },
                "method_score_orientation": {
                    "all_detection_scores": "larger_is_weaker_reference_support",
                    "coreot_full": "query-marginal deficit u",
                    "coreot_match_only": "query-marginal deficit u",
                    "uniform_uot": "query-marginal deficit u",
                    "prior_only": "prior_risk = 1 - rho",
                    "external_methods": (
                        "method-specific oriented diagnostics from preserved artifacts"
                    ),
                },
                "displayed_donor_splits": list(SEEDS),
                "aggregation": {
                    "split_unit": "fixed donor-aware benchmark split",
                    "interval": "sample standard deviation across five splits, ddof=1",
                    "panel_d_destination": (
                        "median cell-level conditional probability within split"
                    ),
                    "panel_e_cloud": (
                        "descriptive eligible run-cell observations with split medians"
                    ),
                },
                "software_versions": _software_versions(),
                "plotting_script": {
                    "path": str(plotting_script),
                    "git_commit": _git_head(),
                    "repository_worktree_dirty": _git_dirty(),
                    "sha256": _file_sha256(plotting_script),
                },
                "source_data_sha256": {str(path): _file_sha256(path) for path in source_paths},
                "regeneration_commands": [
                    ("uv run python experiments/pbmc_state/generate_pbmc_compare_baseline.py"),
                    (
                        "uv run python "
                        "experiments/pbmc_state/generate_pbmc_figure3_panels.py "
                        "--panel all"
                    ),
                    (
                        "uv run python "
                        "experiments/pbmc_state/generate_pbmc_figure3_panels.py "
                        "--panel main"
                    ),
                ],
                "interpretation_limits": [
                    (
                        "The paired reference-restoration response refits the input PCA "
                        "for each reference condition."
                    ),
                    (
                        "The displayed settings were selected post hoc using the "
                        "same endpoints and donor splits; the spatial panels are not "
                        "independent quantitative evidence."
                    ),
                    (
                        "Split-level variability is descriptive and not a "
                        "population-level confidence interval."
                    ),
                ],
                "canvas_inches": list(MAIN_FIGURE_SIZE_INCHES),
                "raster_dpi": PANEL_RASTER_DPI,
                "minimum_font_size_points": MAIN_FIGURE_MIN_FONT_SIZE,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    required = [
        *result_outputs.values(),
        *docs_outputs.values(),
        caption_path,
        docs_caption_path,
        manifest_path,
    ]
    _validate_required(required, "main Figure 3")
    _validate_raster(result_outputs["png"], MAIN_FIGURE_SIZE_INCHES)
    return {
        **{f"result_{key}": value for key, value in result_outputs.items()},
        **{f"docs_{key}": value for key, value in docs_outputs.items()},
        "caption": caption_path,
        "docs_caption": docs_caption_path,
        "manifest": manifest_path,
    }
