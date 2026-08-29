"""Generate PBMC supplementary source tables, figures, and provenance."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from coreot.results.compare_baselines import select_primary_comparison_rows
from coreot.results.matched_reference import (
    CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE,
    MEAN_CELL_CONDITIONAL_DESTINATION,
    POOLED_TRANSPORTED_MASS_DESTINATION_COMPOSITION,
    TYPICAL_CELL_CONDITIONAL_DESTINATION,
)
from coreot.results.pbmc_figure3 import compute_within_celltype_metrics


ENDPOINTS = ("B cells", "NK cells", "Dendritic cells")
SEEDS = (1, 2, 3, 4, 5)
ENDPOINT_SHORT = {
    "B cells": "B",
    "NK cells": "NK",
    "Dendritic cells": "DC",
}
SELECTED_PARAMETERS = {
    "B cells": (0.5, 1.0, 4.0),
    "NK cells": (0.5, 1.0, 2.0),
    "Dendritic cells": (0.75, 1.0, 3.0),
}
CONDITIONS = ("incomplete_reference", "full_reference_control")
CONDITION_LABELS = {
    "incomplete_reference": "Reference-omitted condition",
    "full_reference_control": "Restored-reference condition",
}
INCOMPLETE_REFERENCE = "incomplete_reference"
INTERNAL_CANDIDATE_SET = "pca30_k100"
RASTER_DPI = 350
N_PRIOR_INTERVALS = 20
CALIBRATION_PERCENTILES = (0.9, 0.95, 0.975)
CALIBRATION_ENTROPY_THRESHOLD = 0.8
TRANSPORT_ETA = 1.0e-12


@dataclass(frozen=True)
class MethodSpec:
    method: str
    score: str
    display: str
    forced_labels: bool
    calibrated_score_threshold: bool


METHODS = (
    MethodSpec("coreot_full", "u", "CoRe-OT", True, True),
    MethodSpec("uniform_uot", "u", "Uniform UOT", True, True),
    MethodSpec("prior_only", "prior_risk", "Prior only", False, False),
    MethodSpec("nn", "nn_distance", "Nearest neighbor", True, False),
    MethodSpec("seurat_anchor", "u", "Seurat", True, True),
    MethodSpec("singleR", "u", "SingleR", True, True),
    MethodSpec("celltypist_l3", "u", "CellTypist", True, True),
    MethodSpec("scmap_cell", "u", "scmap-cell", True, True),
    MethodSpec("scmap_cluster", "u", "scmap-cluster", True, True),
    MethodSpec("chetah", "u", "CHETAH", True, True),
)
METHOD_BY_KEY = {(item.method, item.score): item for item in METHODS}
METHOD_ORDER = {item.method: index for index, item in enumerate(METHODS)}
SELECTIVE_METHODS = tuple(item for item in METHODS if item.calibrated_score_threshold)
TRANSFER_METHODS = tuple(item for item in METHODS if item.forced_labels)
CALIBRATION_SENSITIVITY_METHODS = METHODS[:2]

CONTRASTS = (
    (
        "full_minus_prior",
        "CoRe-OT − prior only",
        "coreot_full",
        "prior_only",
    ),
)

ENDPOINT_COLORS = {
    "B cells": "#0072B2",
    "NK cells": "#D55E00",
    "Dendritic cells": "#009E73",
}
CONDITION_COLORS = {
    "incomplete_reference": "#E69F00",
    "full_reference_control": "#0072B2",
}
DESTINATION_CATEGORIES = (
    "same_type_control",
    "same_type_stimulated",
    "other_reference_states",
)
DESTINATION_LABELS = {
    "same_type_control": "Same-type\ncontrol",
    "same_type_stimulated": "Same-type\nstimulated",
    "other_reference_states": "Other reference\nstates",
}


class PBMCSupplementError(RuntimeError):
    """Raised when retained PBMC artifacts do not satisfy the supplement contract."""


@dataclass(frozen=True)
class SupplementPaths:
    project_root: Path
    data_root: Path
    docs_root: Path
    figures_root: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> SupplementPaths:
        return cls(
            project_root=project_root,
            data_root=project_root / "results/PBMC/manuscript/supplement",
            docs_root=project_root / "docs",
            figures_root=project_root / "docs/figs",
        )


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    source: Path | str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PBMCSupplementError(f"{source} is missing columns {missing}.")


def _require_exact_keys(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    expected: set[tuple[object, ...]],
    label: str,
) -> None:
    if frame.duplicated(list(columns)).any():
        duplicated = frame.loc[frame.duplicated(list(columns), keep=False), list(columns)]
        raise PBMCSupplementError(
            f"{label} contains duplicate keys: {duplicated.to_dict(orient='records')[:5]}"
        )
    observed = set(frame.loc[:, list(columns)].itertuples(index=False, name=None))
    if observed != expected:
        raise PBMCSupplementError(
            f"{label} keys differ from the contract; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _read_raw_metadata(path: Path) -> pd.DataFrame:
    raw = ad.read_h5ad(path, backed="r")
    try:
        _require_columns(raw.obs, ("cell_type", "label", "replicate"), path)
        metadata = raw.obs.loc[:, ["cell_type", "label", "replicate"]].copy()
    finally:
        raw.file.close()
    metadata = metadata.rename(columns={"label": "condition", "replicate": "donor"})
    metadata.index = metadata.index.astype(str)
    if metadata.index.duplicated().any():
        raise PBMCSupplementError(f"{path} contains duplicate cell identifiers.")
    if not set(metadata["condition"].astype(str)).issubset({"ctrl", "stim"}):
        raise PBMCSupplementError(f"{path} contains unexpected condition labels.")
    return metadata


def _read_truth(run_root: Path) -> pd.DataFrame:
    path = run_root / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
    truth = pd.read_csv(path)
    _require_columns(
        truth,
        ("cell_id", "true_label", "is_absent_state", "is_shared_state"),
        path,
    )
    truth["cell_id"] = truth["cell_id"].astype(str)
    if truth["cell_id"].duplicated().any():
        raise PBMCSupplementError(f"{path} contains duplicate query cells.")
    return truth


def _read_method_scores(
    run_root: Path,
    *,
    candidate_set: str,
    method: str,
    score: str,
    expected_ids: set[str],
) -> pd.DataFrame:
    path = run_root / "scoring" / INCOMPLETE_REFERENCE / candidate_set / "cell_scores.parquet"
    scores = pd.read_parquet(path)
    _require_columns(scores, ("cell_id", "method", score), path)
    scores["cell_id"] = scores["cell_id"].astype(str)
    selected = scores.loc[scores["method"].eq(method), ["cell_id", score]].copy()
    if selected["cell_id"].duplicated().any():
        raise PBMCSupplementError(f"{path} contains duplicate rows for {method}/{score}.")
    if set(selected["cell_id"]) != expected_ids:
        raise PBMCSupplementError(f"{path} has a query-cell mismatch for {method}/{score}.")
    if not np.isfinite(selected[score].to_numpy(dtype=float)).all():
        raise PBMCSupplementError(f"{path} contains non-finite values for {method}/{score}.")
    return selected


def _comparison_rows(path: Path) -> pd.DataFrame:
    comparison = pd.read_csv(path)
    _require_columns(
        comparison,
        (
            "run_id",
            "held_out_label",
            "seed",
            "candidate_set",
            "method",
            "score",
            "auroc",
            "auprc",
        ),
        path,
    )
    comparison = select_primary_comparison_rows(comparison, source=path)
    keys = set(METHOD_BY_KEY)
    selected = comparison.loc[
        comparison["held_out_label"].isin(ENDPOINTS)
        & comparison[["method", "score"]].apply(tuple, axis=1).isin(keys)
    ].copy()
    expected = {
        (endpoint, seed, item.method, item.score)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for item in METHODS
    }
    _require_exact_keys(
        selected,
        columns=("held_out_label", "seed", "method", "score"),
        expected=expected,
        label="selected comparison rows",
    )
    selected["method_order"] = selected["method"].map(METHOD_ORDER)
    return selected.sort_values(["method_order", "held_out_label", "seed"]).drop(
        columns="method_order"
    )


def collect_within_celltype_detection(
    *,
    comparison_path: Path,
    raw_data_path: Path,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = _comparison_rows(comparison_path)
    metadata = _read_raw_metadata(raw_data_path)
    rows: list[dict[str, object]] = []
    query_sets: dict[tuple[str, int], set[str]] = {}
    for record in comparison.itertuples(index=False):
        endpoint = str(record.held_out_label)
        seed = int(record.seed)
        method = str(record.method)
        score = str(record.score)
        run_root = runs_root / str(record.run_id)
        truth = _read_truth(run_root)
        truth_ids = set(truth["cell_id"])
        cohort_key = (endpoint, seed)
        previous_ids = query_sets.setdefault(cohort_key, truth_ids)
        if previous_ids != truth_ids:
            raise PBMCSupplementError(
                f"Comparison methods use different query cells for {endpoint}, seed {seed}."
            )
        selected = _read_method_scores(
            run_root,
            candidate_set=str(record.candidate_set),
            method=method,
            score=score,
            expected_ids=truth_ids,
        )
        joined = truth.loc[:, ["cell_id", "is_absent_state"]].merge(
            selected,
            on="cell_id",
            validate="one_to_one",
        )
        y_true = joined["is_absent_state"].astype(bool).to_numpy()
        y_score = joined[score].to_numpy(dtype=float)
        observed_ap = average_precision_score(y_true, y_score)
        observed_auroc = roc_auc_score(y_true, y_score)
        if not np.isclose(observed_ap, float(record.auprc), atol=1.0e-10, rtol=0):
            raise PBMCSupplementError(
                f"Global AP lineage check failed for {endpoint}/{seed}/{method}."
            )
        if not np.isclose(
            observed_auroc,
            float(record.auroc),
            atol=1.0e-10,
            rtol=0,
        ):
            raise PBMCSupplementError(
                f"Global AUROC lineage check failed for {endpoint}/{seed}/{method}."
            )
        local = compute_within_celltype_metrics(
            truth,
            selected,
            metadata,
            endpoint=endpoint,
            score_column=score,
        )
        rows.append(
            {
                "held_out_label": endpoint,
                "seed": seed,
                "run_id": str(record.run_id),
                "candidate_set": str(record.candidate_set),
                "method": method,
                "score": score,
                "display_name": METHOD_BY_KEY[(method, score)].display,
                "n_positive": int(local["n_positive"]),
                "n_negative": int(local["n_negative"]),
                "prevalence": float(local["prevalence"]),
                "auprc": float(local["auprc"]),
                "auroc": float(local["auroc"]),
            }
        )
    by_seed = pd.DataFrame.from_records(rows)
    expected_keys = {
        (endpoint, seed, item.method, item.score)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for item in METHODS
    }
    _require_exact_keys(
        by_seed,
        columns=("held_out_label", "seed", "method", "score"),
        expected=expected_keys,
        label="within-cell-type detection",
    )
    summary = (
        by_seed.groupby(
            ["held_out_label", "method", "score", "display_name"],
            sort=False,
        )
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", lambda values: values.std(ddof=1)),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", lambda values: values.std(ddof=1)),
            prevalence_mean=("prevalence", "mean"),
            prevalence_std=("prevalence", lambda values: values.std(ddof=1)),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(5).all():
        raise PBMCSupplementError("Within-cell-type summaries must contain five fixed splits.")
    contrast_by_seed, contrast_summary = _within_celltype_contrasts(by_seed)
    return by_seed, summary, contrast_by_seed, contrast_summary


def _within_celltype_contrasts(
    by_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for seed in SEEDS:
            split = by_seed.loc[
                by_seed["held_out_label"].eq(endpoint) & by_seed["seed"].eq(seed)
            ].set_index("method")
            for contrast_id, display, minuend, subtrahend in CONTRASTS:
                for metric in ("auprc", "auroc"):
                    records.append(
                        {
                            "held_out_label": endpoint,
                            "seed": seed,
                            "contrast": contrast_id,
                            "display_name": display,
                            "metric": metric,
                            "difference": float(
                                split.loc[minuend, metric] - split.loc[subtrahend, metric]
                            ),
                        }
                    )
    by_contrast = pd.DataFrame.from_records(records)
    expected = {
        (endpoint, seed, contrast[0], metric)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for contrast in CONTRASTS
        for metric in ("auprc", "auroc")
    }
    _require_exact_keys(
        by_contrast,
        columns=("held_out_label", "seed", "contrast", "metric"),
        expected=expected,
        label="within-cell-type paired contrasts",
    )
    summary = (
        by_contrast.groupby(
            ["held_out_label", "contrast", "display_name", "metric"],
            sort=False,
        )
        .agg(
            mean_difference=("difference", "mean"),
            std_difference=("difference", lambda values: values.std(ddof=1)),
            n_pairs=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_pairs"].eq(5).all():
        raise PBMCSupplementError("Each paired contrast must contain five splits.")
    return by_contrast, summary


def collect_prior_dependence(
    *,
    selected_runs: pd.DataFrame,
    raw_data_path: Path,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = _read_raw_metadata(raw_data_path)
    runs = selected_runs.loc[
        selected_runs["held_out_label"].isin(ENDPOINTS)
        & selected_runs["method"].eq("coreot_full")
        & selected_runs["score"].eq("u")
    ].copy()
    expected_runs = {(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS}
    _require_exact_keys(
        runs,
        columns=("held_out_label", "seed"),
        expected=expected_runs,
        label="selected CoRe-OT prior-dependence runs",
    )

    bin_records: list[dict[str, object]] = []
    correlation_records: list[dict[str, object]] = []
    for run in runs.itertuples(index=False):
        endpoint = str(run.held_out_label)
        seed = int(run.seed)
        run_root = runs_root / str(run.run_id)
        truth = _read_truth(run_root)
        scores = _read_method_scores(
            run_root,
            candidate_set=str(run.candidate_set),
            method="coreot_full",
            score="u",
            expected_ids=set(truth["cell_id"]),
        )
        score_path = (
            run_root
            / "scoring"
            / INCOMPLETE_REFERENCE
            / str(run.candidate_set)
            / "cell_scores.parquet"
        )
        complete_scores = pd.read_parquet(score_path)
        _require_columns(
            complete_scores,
            ("cell_id", "method", "u", "prior_risk"),
            score_path,
        )
        complete_scores = complete_scores.loc[
            complete_scores["method"].eq("coreot_full"),
            ["cell_id", "u", "prior_risk"],
        ].copy()
        complete_scores["cell_id"] = complete_scores["cell_id"].astype(str)
        if complete_scores["cell_id"].duplicated().any():
            raise PBMCSupplementError(f"{score_path} contains duplicate CoRe-OT query cells.")
        if set(complete_scores["cell_id"]) != set(scores["cell_id"]):
            raise PBMCSupplementError(f"{score_path} has inconsistent CoRe-OT score columns.")
        complete_scores["cell_type"] = complete_scores["cell_id"].map(
            metadata["cell_type"].astype(str)
        )
        if complete_scores["cell_type"].isna().any():
            raise PBMCSupplementError(f"{score_path} cannot be joined to PBMC cell-type metadata.")
        cohort = complete_scores.loc[complete_scores["cell_type"].eq(endpoint)].copy()
        for column in ("u", "prior_risk"):
            cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
        n_cohort = len(cohort)
        finite = np.isfinite(cohort["u"]) & np.isfinite(cohort["prior_risk"])
        cohort = cohort.loc[finite].copy()
        if len(cohort) < 2:
            raise PBMCSupplementError(
                f"Fewer than two finite prior-dependence cells for {endpoint}, seed={seed}."
            )

        average_rank = cohort["prior_risk"].rank(method="average")
        cohort["prior_bin"] = np.minimum(
            (
                np.floor(
                    (average_rank.to_numpy(dtype=float) - 1.0) * N_PRIOR_INTERVALS / len(cohort)
                ).astype(int)
                + 1
            ),
            N_PRIOR_INTERVALS,
        )
        occupied_bins = int(cohort["prior_bin"].nunique())
        for prior_bin, subset in cohort.groupby("prior_bin", sort=True):
            bin_records.append(
                {
                    "held_out_label": endpoint,
                    "seed": seed,
                    "run_id": str(run.run_id),
                    "prior_bin": int(prior_bin),
                    "n_cells": int(len(subset)),
                    "n_cohort": int(n_cohort),
                    "n_finite": int(len(cohort)),
                    "n_nonfinite_excluded": int(n_cohort - len(cohort)),
                    "n_occupied_bins": occupied_bins,
                    "mean_prior_risk": float(subset["prior_risk"].mean()),
                    "mean_u": float(subset["u"].mean()),
                }
            )
        correlation_records.append(
            {
                "held_out_label": endpoint,
                "seed": seed,
                "run_id": str(run.run_id),
                "spearman": float(cohort["u"].corr(cohort["prior_risk"], method="spearman")),
                "pearson": float(cohort["u"].corr(cohort["prior_risk"])),
                "n_cells": int(len(cohort)),
                "n_cohort": int(n_cohort),
                "n_excluded": int(n_cohort - len(cohort)),
            }
        )

    bins_by_seed = pd.DataFrame.from_records(bin_records)
    bins_summary = (
        bins_by_seed.groupby(["held_out_label", "prior_bin"], as_index=False)
        .agg(
            n_splits=("seed", "nunique"),
            mean_prior_risk=("mean_prior_risk", "mean"),
            mean_u=("mean_u", "mean"),
        )
        .sort_values(["held_out_label", "prior_bin"])
        .reset_index(drop=True)
    )
    correlations_by_seed = pd.DataFrame.from_records(correlation_records)
    correlations_summary = (
        correlations_by_seed.groupby("held_out_label", as_index=False)
        .agg(
            n_splits=("seed", "size"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", lambda values: values.std(ddof=1)),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", lambda values: values.std(ddof=1)),
            n_cells_min=("n_cells", "min"),
            n_cells_max=("n_cells", "max"),
            n_excluded_total=("n_excluded", "sum"),
        )
        .reset_index(drop=True)
    )
    if not correlations_summary["n_splits"].eq(5).all():
        raise PBMCSupplementError("Each PBMC prior-dependence summary requires five donor splits.")
    return (
        bins_by_seed,
        bins_summary,
        correlations_by_seed,
        correlations_summary,
    )


def collect_benchmark_summary(
    path: Path,
    *,
    selected_runs: pd.DataFrame,
    runs_root: Path,
    raw_data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = (
        "run_id",
        "held_out_label",
        "seed",
        "query_donors",
        "reference_donors",
        "n_source_cells",
        "n_full_reference_target_cells",
        "n_incomplete_reference_target_cells",
        "n_positives",
        "n_same_celltype_controls",
    )
    _require_columns(frame, required, path)
    by_seed = frame.loc[frame["held_out_label"].isin(ENDPOINTS), required].copy()
    expected = {(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS}
    _require_exact_keys(
        by_seed,
        columns=("held_out_label", "seed"),
        expected=expected,
        label="PBMC benchmark construction",
    )
    selected = selected_runs.loc[
        selected_runs["method"].eq("coreot_full") & selected_runs["score"].eq("u"),
        ["held_out_label", "seed", "run_id", "candidate_set"],
    ].drop_duplicates()
    _require_exact_keys(
        selected,
        columns=("held_out_label", "seed"),
        expected=expected,
        label="selected PBMC benchmark runs",
    )
    metadata = _read_raw_metadata(raw_data_path)
    executed_rows: list[dict[str, object]] = []
    for record in by_seed.itertuples(index=False):
        run = selected.loc[
            selected["held_out_label"].eq(record.held_out_label) & selected["seed"].eq(record.seed)
        ]
        if len(run) != 1:
            raise PBMCSupplementError(
                f"Expected one selected run for {record.held_out_label}, seed {record.seed}."
            )
        selected_record = run.iloc[0]
        run_id = str(selected_record["run_id"])
        candidate_set = str(selected_record["candidate_set"])
        run_root = runs_root / run_id
        condition_ids: dict[str, tuple[set[str], set[str]]] = {}
        for condition in CONDITIONS:
            coupling_path = (
                run_root
                / "transport"
                / condition
                / candidate_set
                / "coreot_full"
                / "sparse_coupling.parquet"
            )
            coupling = pd.read_parquet(
                coupling_path,
                columns=["source_cell_id", "target_cell_id"],
            )
            source_ids = set(coupling["source_cell_id"].astype(str))
            target_ids = set(coupling["target_cell_id"].astype(str))
            if not source_ids or not target_ids:
                raise PBMCSupplementError(f"{coupling_path} does not contain both fitted domains.")
            condition_ids[condition] = source_ids, target_ids

        source_ids, incomplete_ids = condition_ids["incomplete_reference"]
        full_source_ids, full_ids = condition_ids["full_reference_control"]
        if source_ids != full_source_ids:
            raise PBMCSupplementError(
                f"Paired PBMC conditions use different query cells for {run_id}."
            )
        if len(source_ids) != int(record.n_source_cells):
            raise PBMCSupplementError(f"Executed query count differs from {path} for {run_id}.")
        if not incomplete_ids < full_ids:
            raise PBMCSupplementError(
                f"The incomplete-reference target is not a strict subset for {run_id}."
            )
        all_ids = source_ids | full_ids
        missing_metadata = all_ids - set(metadata.index)
        if missing_metadata:
            raise PBMCSupplementError(
                f"Raw PBMC metadata are missing fitted cells for {run_id}: "
                f"{sorted(missing_metadata)[:5]}"
            )
        restored = metadata.loc[sorted(full_ids - incomplete_ids)]
        if not (
            restored["cell_type"].astype(str).eq(str(record.held_out_label))
            & restored["condition"].astype(str).eq("stim")
        ).all():
            raise PBMCSupplementError(
                f"Restored PBMC cells do not match the held-out state for {run_id}."
            )
        query_donors = sorted(metadata.loc[sorted(source_ids), "donor"].astype(str).unique())
        reference_donors = sorted(metadata.loc[sorted(full_ids), "donor"].astype(str).unique())
        if set(query_donors) & set(reference_donors):
            raise PBMCSupplementError(
                f"Executed PBMC query and reference donors overlap for {run_id}."
            )
        row = record._asdict()
        row.update(
            {
                "run_id": run_id,
                "query_donors": ", ".join(query_donors),
                "reference_donors": ", ".join(reference_donors),
                "n_full_reference_target_cells": len(full_ids),
                "n_incomplete_reference_target_cells": len(incomplete_ids),
            }
        )
        executed_rows.append(row)
    by_seed = pd.DataFrame.from_records(executed_rows)
    denominator = by_seed["n_positives"] + by_seed["n_same_celltype_controls"]
    if (denominator <= 0).any():
        raise PBMCSupplementError("Within-cell-type prevalence is undefined.")
    by_seed["within_celltype_prevalence"] = by_seed["n_positives"] / denominator
    summary = (
        by_seed.groupby("held_out_label", sort=False)
        .agg(
            query_min=("n_source_cells", "min"),
            query_max=("n_source_cells", "max"),
            positive_min=("n_positives", "min"),
            positive_max=("n_positives", "max"),
            control_min=("n_same_celltype_controls", "min"),
            control_max=("n_same_celltype_controls", "max"),
            full_reference_min=("n_full_reference_target_cells", "min"),
            full_reference_max=("n_full_reference_target_cells", "max"),
            incomplete_reference_min=(
                "n_incomplete_reference_target_cells",
                "min",
            ),
            incomplete_reference_max=(
                "n_incomplete_reference_target_cells",
                "max",
            ),
            prevalence_min=("within_celltype_prevalence", "min"),
            prevalence_max=("within_celltype_prevalence", "max"),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(5).all():
        raise PBMCSupplementError("Benchmark summaries require five splits.")
    return by_seed, summary


def collect_realized_penalties(
    *,
    selected_runs: pd.DataFrame,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows = selected_runs.loc[
        selected_runs["method"].eq("coreot_full") & selected_runs["score"].eq("u")
    ]
    expected = {(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS}
    _require_exact_keys(
        full_rows,
        columns=("held_out_label", "seed"),
        expected=expected,
        label="selected CoRe-OT runs",
    )
    records = []
    for row in full_rows.itertuples(index=False):
        endpoint = str(row.held_out_label)
        tau_min, tau_max, alpha = SELECTED_PARAMETERS[endpoint]
        for condition in CONDITIONS:
            path = (
                runs_root
                / str(row.run_id)
                / "transport"
                / condition
                / INTERNAL_CANDIDATE_SET
                / "coreot_full"
                / "transport_manifest.yaml"
            )
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            observed = (
                float(metadata["tau_source_min"]),
                float(metadata["tau_source_max"]),
                float(metadata["tau_target"]),
                float(metadata["alpha"]),
            )
            expected_values = (tau_min, tau_max, 1.0, alpha)
            if observed[0] < tau_min - 1.0e-12 or observed[1] > tau_max + 1.0e-12:
                raise PBMCSupplementError(
                    f"{path} records realized penalties outside configured bounds."
                )
            if not np.isclose(observed[2:], expected_values[2:]).all():
                raise PBMCSupplementError(f"{path} disagrees with the selected operating point.")
            records.append(
                {
                    "held_out_label": endpoint,
                    "seed": int(row.seed),
                    "run_id": str(row.run_id),
                    "condition": condition,
                    "configured_tau_min": tau_min,
                    "configured_tau_max": tau_max,
                    "realized_tau_min": observed[0],
                    "realized_tau_max": observed[1],
                    "tau_target": observed[2],
                    "alpha": observed[3],
                }
            )
    by_seed = pd.DataFrame.from_records(records)
    expected_keys = {
        (endpoint, seed, condition)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for condition in CONDITIONS
    }
    _require_exact_keys(
        by_seed,
        columns=("held_out_label", "seed", "condition"),
        expected=expected_keys,
        label="realized PBMC penalties",
    )
    summary = (
        by_seed.groupby(
            [
                "held_out_label",
                "condition",
                "configured_tau_min",
                "configured_tau_max",
                "tau_target",
                "alpha",
            ],
            sort=False,
        )
        .agg(
            realized_tau_min=("realized_tau_min", "min"),
            realized_tau_max=("realized_tau_max", "max"),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    return by_seed, summary


def collect_transfer_results(
    path: Path,
    *,
    detection_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = (
        "held_out_label",
        "seed",
        "run_id",
        "candidate_set",
        "method",
        "score",
        "forced_accuracy",
        "forced_macro_f1",
        "post_abstention_macro_f1",
        "coverage",
        "shared_false_abstention_rate",
    )
    _require_columns(frame, required, path)
    frame = select_primary_comparison_rows(frame, source=path)
    keys = {(item.method, item.score) for item in TRANSFER_METHODS}
    by_seed = frame.loc[
        frame["held_out_label"].isin(ENDPOINTS)
        & frame[["method", "score"]].apply(tuple, axis=1).isin(keys),
        required,
    ].copy()
    expected = {
        (endpoint, seed, item.method, item.score)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for item in TRANSFER_METHODS
    }
    _require_exact_keys(
        by_seed,
        columns=("held_out_label", "seed", "method", "score"),
        expected=expected,
        label="represented-state cell-type label transfer",
    )
    by_seed["display_name"] = [
        METHOD_BY_KEY[(method, score)].display
        for method, score in by_seed[["method", "score"]].itertuples(
            index=False,
            name=None,
        )
    ]
    by_seed["selective_metrics_applicable"] = [
        METHOD_BY_KEY[(method, score)].calibrated_score_threshold
        for method, score in by_seed[["method", "score"]].itertuples(
            index=False,
            name=None,
        )
    ]
    detection = pd.read_csv(detection_path)
    _require_columns(
        detection,
        (
            "held_out_label",
            "seed",
            "method",
            "score",
            "absent_abstention_rate",
        ),
        detection_path,
    )
    detection = select_primary_comparison_rows(detection, source=detection_path)
    detection = detection.loc[
        detection["held_out_label"].isin(ENDPOINTS)
        & detection[["method", "score"]].apply(tuple, axis=1).isin(keys),
        [
            "held_out_label",
            "seed",
            "method",
            "score",
            "absent_abstention_rate",
        ],
    ].copy()
    _require_exact_keys(
        detection,
        columns=("held_out_label", "seed", "method", "score"),
        expected={
            (endpoint, seed, item.method, item.score)
            for endpoint in ENDPOINTS
            for seed in SEEDS
            for item in TRANSFER_METHODS
        },
        label="held-out-state abstention",
    )
    by_seed = by_seed.merge(
        detection,
        on=["held_out_label", "seed", "method", "score"],
        how="left",
        validate="one_to_one",
    )
    not_applicable = ~by_seed["selective_metrics_applicable"]
    by_seed.loc[
        not_applicable,
        (
            "absent_abstention_rate",
            "post_abstention_macro_f1",
            "coverage",
            "shared_false_abstention_rate",
        ),
    ] = np.nan
    summary = (
        by_seed.groupby(
            [
                "held_out_label",
                "method",
                "score",
                "display_name",
                "selective_metrics_applicable",
            ],
            sort=False,
        )
        .agg(
            forced_accuracy_mean=("forced_accuracy", "mean"),
            forced_accuracy_std=(
                "forced_accuracy",
                lambda values: values.std(ddof=1),
            ),
            forced_macro_f1_mean=("forced_macro_f1", "mean"),
            forced_macro_f1_std=(
                "forced_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            post_abstention_macro_f1_mean=(
                "post_abstention_macro_f1",
                "mean",
            ),
            post_abstention_macro_f1_std=(
                "post_abstention_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            held_out_abstention_mean=("absent_abstention_rate", "mean"),
            held_out_abstention_std=(
                "absent_abstention_rate",
                lambda values: values.std(ddof=1),
            ),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", lambda values: values.std(ddof=1)),
            shared_false_abstention_mean=(
                "shared_false_abstention_rate",
                "mean",
            ),
            shared_false_abstention_std=(
                "shared_false_abstention_rate",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(5).all():
        raise PBMCSupplementError("Transfer summaries require five splits.")
    return by_seed, summary


def collect_full_reference_calibration(
    path: Path,
    *,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = (
        "held_out_label",
        "seed",
        "run_id",
        "candidate_set",
        "method",
        "score",
        "full_reference_false_abstention_rate",
        "post_abstention_macro_f1",
        "coverage",
    )
    _require_columns(frame, required, path)
    frame = select_primary_comparison_rows(frame, source=path)
    keys = {(item.method, item.score) for item in SELECTIVE_METHODS}
    by_run = frame.loc[
        frame["held_out_label"].isin(ENDPOINTS)
        & frame[["method", "score"]].apply(tuple, axis=1).isin(keys),
        required,
    ].copy()
    expected = {
        (endpoint, seed, item.method, item.score)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for item in SELECTIVE_METHODS
    }
    _require_exact_keys(
        by_run,
        columns=("held_out_label", "seed", "method", "score"),
        expected=expected,
        label="restored-reference calibration",
    )
    for row in by_run.itertuples(index=False):
        manifest_path = (
            runs_root
            / str(row.run_id)
            / "scoring"
            / "full_reference_control"
            / str(row.candidate_set)
            / "scoring_manifest.yaml"
        )
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        threshold_mode = payload.get("metadata", {}).get("threshold_mode")
        if threshold_mode != "full_reference_quantile":
            raise PBMCSupplementError(
                f"{manifest_path} does not record an executed calibrated threshold."
            )
    by_run["display_name"] = [
        METHOD_BY_KEY[(method, score)].display
        for method, score in by_run[["method", "score"]].itertuples(
            index=False,
            name=None,
        )
    ]
    summary = (
        by_run.groupby(["method", "score", "display_name"], sort=False)
        .agg(
            false_abstention_mean=(
                "full_reference_false_abstention_rate",
                "mean",
            ),
            false_abstention_std=(
                "full_reference_false_abstention_rate",
                lambda values: values.std(ddof=1),
            ),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", lambda values: values.std(ddof=1)),
            post_abstention_macro_f1_mean=(
                "post_abstention_macro_f1",
                "mean",
            ),
            post_abstention_macro_f1_std=(
                "post_abstention_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            n_endpoint_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_endpoint_splits"].eq(15).all():
        raise PBMCSupplementError("Calibration summaries require 15 endpoint-split evaluations.")
    return by_run, summary


def collect_global_detection(
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = comparison.copy()
    by_seed["display_name"] = [
        METHOD_BY_KEY[(method, score)].display
        for method, score in by_seed[["method", "score"]].itertuples(
            index=False,
            name=None,
        )
    ]
    summary = (
        by_seed.groupby(
            ["held_out_label", "method", "score", "display_name"],
            sort=False,
        )
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", lambda values: values.std(ddof=1)),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", lambda values: values.std(ddof=1)),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(5).all():
        raise PBMCSupplementError("Global detection summaries require five splits.")
    return by_seed, summary


def _read_sensitivity_fit_convergence(
    *,
    runs_root: Path,
    run_id: str,
    candidate_set: str,
    tau_min: float,
    tau_max: float,
    alpha: float,
) -> dict[str, object]:
    fit_root = (
        runs_root
        / run_id
        / "transport"
        / INCOMPLETE_REFERENCE
        / candidate_set
        / "coreot_full"
    )
    manifest_path = fit_root / "transport_manifest.yaml"
    method_params_path = fit_root / "method_params.yaml"
    for path in (manifest_path, method_params_path):
        if not path.is_file():
            raise PBMCSupplementError(
                f"PBMC sensitivity fit evidence is missing for {run_id}: {path}"
            )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    params = yaml.safe_load(method_params_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(params, dict):
        raise PBMCSupplementError(
            f"PBMC sensitivity fit evidence must be YAML mappings for {run_id}."
        )
    metadata = manifest.get("metadata")
    artifacts = manifest.get("artifacts")
    if not isinstance(metadata, dict) or not isinstance(artifacts, dict):
        raise PBMCSupplementError(
            f"PBMC sensitivity transport manifest is malformed for {run_id}."
        )
    if metadata.get("method") != "coreot_full" or params.get("name") != "coreot_full":
        raise PBMCSupplementError(
            f"PBMC sensitivity fit method identity is inconsistent for {run_id}."
        )

    declared_params = artifacts.get("method_params")
    if not isinstance(declared_params, str) or not declared_params:
        raise PBMCSupplementError(
            f"PBMC sensitivity transport manifest omits method_params for {run_id}."
        )
    declared_path = Path(declared_params)
    declared_candidates = (
        (declared_path,)
        if declared_path.is_absolute()
        else (runs_root.parent / declared_path, fit_root / declared_path)
    )
    if method_params_path.resolve() not in {
        candidate.resolve() for candidate in declared_candidates
    }:
        raise PBMCSupplementError(
            f"PBMC sensitivity transport manifest points outside its fit directory for {run_id}."
        )

    expected_params = {
        "tau_min": float(tau_min),
        "tau_max": float(tau_max),
        "alpha": float(alpha),
        "epsilon": 0.05,
        "tol": 1.0e-6,
        "numerical_floor": 1.0e-300,
    }
    for key, expected in expected_params.items():
        value = params.get(key)
        if not isinstance(value, int | float) or not np.isclose(
            float(value),
            expected,
            rtol=0.0,
            atol=0.0,
        ):
            raise PBMCSupplementError(
                f"PBMC sensitivity method parameter {key} disagrees with the retained "
                f"contract for {run_id}."
            )
    for key in ("alpha", "epsilon"):
        value = metadata.get(key)
        if not isinstance(value, int | float) or not np.isclose(
            float(value),
            expected_params[key],
            rtol=0.0,
            atol=0.0,
        ):
            raise PBMCSupplementError(
                f"PBMC sensitivity manifest metadata {key} disagrees with "
                f"method_params.yaml for {run_id}."
            )

    max_iter = params.get("max_iter")
    n_iter = metadata.get("n_iter")
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, int)
        or max_iter <= 0
        or isinstance(n_iter, bool)
        or not isinstance(n_iter, int)
        or n_iter <= 0
        or n_iter > max_iter
    ):
        raise PBMCSupplementError(
            f"PBMC sensitivity iteration evidence is invalid for {run_id}."
        )
    if metadata.get("converged") is not True:
        raise PBMCSupplementError(
            f"PBMC sensitivity fit did not satisfy its scaling-change tolerance for {run_id}."
        )

    relative_fit_root = Path(runs_root.name) / fit_root.relative_to(runs_root)
    return {
        "condition": INCOMPLETE_REFERENCE,
        "candidate_set": candidate_set,
        "convergence_evidence": "verified_converged",
        "converged": True,
        "n_iter": n_iter,
        "max_iter": max_iter,
        "tol": float(params["tol"]),
        "transport_manifest": (relative_fit_root / manifest_path.name).as_posix(),
        "method_params": (relative_fit_root / method_params_path.name).as_posix(),
    }


def collect_sensitivity_results(
    *,
    detection_path: Path,
    transfer_path: Path,
    raw_data_path: Path,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detection = pd.read_csv(detection_path)
    required_detection = (
        "run_id",
        "held_out_label",
        "seed",
        "candidate_set",
        "method",
        "score",
        "auprc",
        "auroc",
        "tau_min",
        "tau_max",
        "alpha",
    )
    _require_columns(detection, required_detection, detection_path)
    selected = detection.loc[
        detection["held_out_label"].isin(ENDPOINTS)
        & detection["method"].eq("coreot_full")
        & detection["score"].eq("u"),
        required_detection,
    ].copy()
    expected = {
        (endpoint, seed, tau_min, tau_max)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for tau_min in (0.5, 0.75, 1.0)
        for tau_max in (1.0, 1.25, 1.5)
    }
    _require_exact_keys(
        selected,
        columns=("held_out_label", "seed", "tau_min", "tau_max"),
        expected=expected,
        label="PBMC tau sensitivity runs",
    )
    metadata = _read_raw_metadata(raw_data_path)
    rows = []
    for record in selected.itertuples(index=False):
        run_root = runs_root / str(record.run_id)
        convergence = _read_sensitivity_fit_convergence(
            runs_root=runs_root,
            run_id=str(record.run_id),
            candidate_set=str(record.candidate_set),
            tau_min=float(record.tau_min),
            tau_max=float(record.tau_max),
            alpha=float(record.alpha),
        )
        truth = _read_truth(run_root)
        scores = _read_method_scores(
            run_root,
            candidate_set=str(record.candidate_set),
            method="coreot_full",
            score="u",
            expected_ids=set(truth["cell_id"]),
        )
        joined = truth.loc[:, ["cell_id", "is_absent_state"]].merge(
            scores,
            on="cell_id",
            validate="one_to_one",
        )
        y_true = joined["is_absent_state"].astype(bool).to_numpy()
        y_score = joined["u"].to_numpy(dtype=float)
        if not np.isclose(
            average_precision_score(y_true, y_score),
            float(record.auprc),
            atol=1.0e-10,
            rtol=0,
        ):
            raise PBMCSupplementError(f"Sensitivity AP lineage failed for {record.run_id}.")
        if not np.isclose(
            roc_auc_score(y_true, y_score),
            float(record.auroc),
            atol=1.0e-10,
            rtol=0,
        ):
            raise PBMCSupplementError(f"Sensitivity AUROC lineage failed for {record.run_id}.")
        local = compute_within_celltype_metrics(
            truth,
            scores,
            metadata,
            endpoint=str(record.held_out_label),
            score_column="u",
        )
        rows.append(
            {
                "held_out_label": str(record.held_out_label),
                "seed": int(record.seed),
                "run_id": str(record.run_id),
                "tau_min": float(record.tau_min),
                "tau_max": float(record.tau_max),
                "alpha": float(record.alpha),
                "within_celltype_auprc": float(local["auprc"]),
                "within_celltype_auroc": float(local["auroc"]),
                "within_celltype_prevalence": float(local["prevalence"]),
                **convergence,
            }
        )
    local_by_seed = pd.DataFrame.from_records(rows)
    transfer = pd.read_csv(transfer_path)
    required_transfer = (
        "run_id",
        "held_out_label",
        "seed",
        "method",
        "score",
        "tau_min",
        "tau_max",
        "alpha",
        "coverage",
        "shared_false_abstention_rate",
        "forced_macro_f1",
    )
    _require_columns(transfer, required_transfer, transfer_path)
    transfer = transfer.loc[
        transfer["held_out_label"].isin(ENDPOINTS)
        & transfer["method"].eq("coreot_full")
        & transfer["score"].eq("u"),
        required_transfer,
    ].copy()
    joined = local_by_seed.merge(
        transfer,
        on=[
            "run_id",
            "held_out_label",
            "seed",
            "tau_min",
            "tau_max",
            "alpha",
        ],
        how="outer",
        validate="one_to_one",
    )
    if joined.isna().any().any():
        raise PBMCSupplementError("Sensitivity detection and transfer rows do not align exactly.")
    summary = (
        joined.groupby(
            ["held_out_label", "tau_min", "tau_max", "alpha"],
            sort=False,
        )
        .agg(
            within_celltype_auprc_mean=("within_celltype_auprc", "mean"),
            within_celltype_auprc_std=(
                "within_celltype_auprc",
                lambda values: values.std(ddof=1),
            ),
            within_celltype_auroc_mean=("within_celltype_auroc", "mean"),
            within_celltype_auroc_std=(
                "within_celltype_auroc",
                lambda values: values.std(ddof=1),
            ),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", lambda values: values.std(ddof=1)),
            forced_macro_f1_mean=("forced_macro_f1", "mean"),
            forced_macro_f1_std=(
                "forced_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            shared_false_abstention_mean=(
                "shared_false_abstention_rate",
                "mean",
            ),
            shared_false_abstention_std=(
                "shared_false_abstention_rate",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(5).all():
        raise PBMCSupplementError("Sensitivity summaries require five splits.")
    compact_records = []
    for endpoint in ENDPOINTS:
        endpoint_rows = summary.loc[summary["held_out_label"].eq(endpoint)]
        tau_min, tau_max, alpha = SELECTED_PARAMETERS[endpoint]
        selected_row = endpoint_rows.loc[
            np.isclose(endpoint_rows["tau_min"], tau_min)
            & np.isclose(endpoint_rows["tau_max"], tau_max)
        ]
        if len(selected_row) != 1:
            raise PBMCSupplementError(f"Sensitivity summary lacks selected point for {endpoint}.")
        row = selected_row.iloc[0]
        compact_records.append(
            {
                "held_out_label": endpoint,
                "selected_tau_min": tau_min,
                "selected_tau_max": tau_max,
                "alpha": alpha,
                "selected_auroc_mean": row["within_celltype_auroc_mean"],
                "selected_auroc_std": row["within_celltype_auroc_std"],
                "grid_auroc_min": endpoint_rows["within_celltype_auroc_mean"].min(),
                "grid_auroc_max": endpoint_rows["within_celltype_auroc_mean"].max(),
                "selected_coverage_mean": row["coverage_mean"],
                "selected_coverage_std": row["coverage_std"],
                "grid_coverage_min": endpoint_rows["coverage_mean"].min(),
                "grid_coverage_max": endpoint_rows["coverage_mean"].max(),
                "selected_false_abstention_mean": row["shared_false_abstention_mean"],
                "selected_false_abstention_std": row["shared_false_abstention_std"],
                "grid_false_abstention_min": endpoint_rows["shared_false_abstention_mean"].min(),
                "grid_false_abstention_max": endpoint_rows["shared_false_abstention_mean"].max(),
            }
        )
    return joined, summary, pd.DataFrame.from_records(compact_records)


def collect_calibration_sensitivity(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = (
        "held_out_label",
        "seed",
        "run_id",
        "quantile",
        "heldout_rejection",
        "shared_coverage",
        "shared_false_abstention",
    )
    _require_columns(frame, required, path)
    by_seed = frame.loc[frame["held_out_label"].isin(ENDPOINTS), required].copy()
    quantiles = tuple(sorted(by_seed["quantile"].unique()))
    expected = {
        (endpoint, seed, quantile)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for quantile in quantiles
    }
    _require_exact_keys(
        by_seed,
        columns=("held_out_label", "seed", "quantile"),
        expected=expected,
        label="PBMC calibration sensitivity",
    )
    summary = (
        by_seed.groupby(["held_out_label", "quantile"], sort=False)
        .agg(
            heldout_rejection_mean=("heldout_rejection", "mean"),
            heldout_rejection_std=(
                "heldout_rejection",
                lambda values: values.std(ddof=1),
            ),
            shared_coverage_mean=("shared_coverage", "mean"),
            shared_coverage_std=(
                "shared_coverage",
                lambda values: values.std(ddof=1),
            ),
            shared_false_abstention_mean=(
                "shared_false_abstention",
                "mean",
            ),
            shared_false_abstention_std=(
                "shared_false_abstention",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    return by_seed, summary


def collect_calibration_percentile_table(
    *,
    selected_runs: pd.DataFrame,
    transfer_by_seed: pd.DataFrame,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = selected_runs.loc[
        selected_runs["held_out_label"].isin(ENDPOINTS)
        & selected_runs["method"].eq("coreot_full")
        & selected_runs["score"].eq("u"),
        ["held_out_label", "seed", "run_id", "candidate_set"],
    ].copy()
    expected_runs = {(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS}
    _require_exact_keys(
        runs,
        columns=("held_out_label", "seed"),
        expected=expected_runs,
        label="selected PBMC calibration-sensitivity runs",
    )
    runs = runs.set_index(["held_out_label", "seed"])

    records: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for seed in SEEDS:
            run = runs.loc[(endpoint, seed)]
            run_root = runs_root / str(run["run_id"])
            candidate_set = str(run["candidate_set"])
            truth = _read_truth(run_root)
            truth_ids = set(truth["cell_id"])
            held_out = truth["is_absent_state"].astype(bool)
            represented = truth["is_shared_state"].astype(bool)
            if (held_out & represented).any() or not (held_out | represented).all():
                raise PBMCSupplementError(
                    f"{run_root} does not partition query cells into held-out and "
                    "represented states."
                )

            manifest_path = (
                run_root
                / "scoring"
                / INCOMPLETE_REFERENCE
                / candidate_set
                / "scoring_manifest.yaml"
            )
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            metadata = manifest.get("metadata", {})
            thresholds = metadata.get("thresholds", {})
            try:
                theta_h = float(thresholds["theta_H"]["value"])
                primary_percentile = float(thresholds["theta_u"]["quantile"])
                source_condition = str(thresholds["theta_u"]["source_condition"])
                source_method = str(thresholds["theta_u"]["source_method"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PBMCSupplementError(
                    f"{manifest_path} lacks the calibrated abstention contract."
                ) from exc
            if (
                metadata.get("threshold_mode") != "full_reference_quantile"
                or not np.isclose(theta_h, CALIBRATION_ENTROPY_THRESHOLD)
                or not np.isclose(primary_percentile, 0.95)
                or source_condition != "full_reference_control"
                or source_method != "method"
            ):
                raise PBMCSupplementError(
                    f"{manifest_path} disagrees with the PBMC calibration-sensitivity contract."
                )

            score_frames: dict[str, pd.DataFrame] = {}
            for condition in CONDITIONS:
                score_path = (
                    run_root / "scoring" / condition / candidate_set / "cell_scores.parquet"
                )
                frame = pd.read_parquet(
                    score_path,
                    columns=["cell_id", "method", "u", "label_entropy", "forced_label"],
                )
                frame["cell_id"] = frame["cell_id"].astype(str)
                score_frames[condition] = frame

            for method_spec in CALIBRATION_SENSITIVITY_METHODS:
                method = method_spec.method
                incomplete = (
                    score_frames[INCOMPLETE_REFERENCE]
                    .loc[
                        score_frames[INCOMPLETE_REFERENCE]["method"].eq(method),
                        ["cell_id", "u", "label_entropy", "forced_label"],
                    ]
                    .copy()
                )
                full = (
                    score_frames["full_reference_control"]
                    .loc[
                        score_frames["full_reference_control"]["method"].eq(method),
                        ["cell_id", "u"],
                    ]
                    .copy()
                )
                for condition, frame in (
                    (INCOMPLETE_REFERENCE, incomplete),
                    ("full_reference_control", full),
                ):
                    if frame["cell_id"].duplicated().any() or set(frame["cell_id"]) != truth_ids:
                        raise PBMCSupplementError(
                            f"{run_root} has a query-cell mismatch for {method} under {condition}."
                        )
                joined = truth.merge(
                    incomplete,
                    on="cell_id",
                    validate="one_to_one",
                ).merge(
                    full.rename(columns={"u": "full_reference_u"}),
                    on="cell_id",
                    validate="one_to_one",
                )
                incomplete_numeric = joined[["u", "label_entropy"]].to_numpy(dtype=float)
                if not np.isfinite(incomplete_numeric).all():
                    raise PBMCSupplementError(
                        f"{run_root} contains non-finite incomplete-reference scores for {method}."
                    )
                full_scores = joined["full_reference_u"].to_numpy(dtype=float)
                finite_full_scores = full_scores[np.isfinite(full_scores)]
                if not len(finite_full_scores):
                    raise PBMCSupplementError(
                        f"{run_root} has no finite restored-reference scores for {method}."
                    )
                joined["forced_label"] = joined["forced_label"].fillna("").astype(str)
                held_out_mask = joined["is_absent_state"].astype(bool)
                represented_mask = joined["is_shared_state"].astype(bool)
                labels = sorted(
                    joined.loc[represented_mask, "true_label"].dropna().astype(str).unique()
                )
                for percentile in CALIBRATION_PERCENTILES:
                    threshold = float(np.quantile(finite_full_scores, percentile))
                    abstain = joined["u"].gt(threshold) | joined["label_entropy"].gt(theta_h)
                    retained = represented_mask & ~abstain
                    retained_labeled = retained & joined["forced_label"].str.len().gt(0)
                    if not labels or not retained_labeled.any():
                        raise PBMCSupplementError(
                            f"{run_root} has no evaluable retained labels for {method} at "
                            f"percentile {percentile}."
                        )
                    records.append(
                        {
                            "held_out_label": endpoint,
                            "seed": seed,
                            "run_id": str(run["run_id"]),
                            "candidate_set": candidate_set,
                            "method": method,
                            "score": method_spec.score,
                            "display_name": method_spec.display,
                            "calibration_percentile": percentile,
                            "threshold": threshold,
                            "entropy_threshold": theta_h,
                            "n_full_reference_finite": int(len(finite_full_scores)),
                            "n_held_out_state": int(held_out_mask.sum()),
                            "n_represented_state": int(represented_mask.sum()),
                            "n_retained_represented_state": int(retained.sum()),
                            "held_out_state_abstention_rate": float(
                                abstain.loc[held_out_mask].mean()
                            ),
                            "represented_state_coverage": float(
                                (~abstain.loc[represented_mask]).mean()
                            ),
                            "post_abstention_macro_f1": float(
                                f1_score(
                                    joined.loc[retained_labeled, "true_label"].astype(str),
                                    joined.loc[retained_labeled, "forced_label"],
                                    labels=labels,
                                    average="macro",
                                    zero_division=0.0,
                                )
                            ),
                        }
                    )

    by_seed = pd.DataFrame.from_records(records)
    expected = {
        (endpoint, seed, method.method, percentile)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for method in CALIBRATION_SENSITIVITY_METHODS
        for percentile in CALIBRATION_PERCENTILES
    }
    _require_exact_keys(
        by_seed,
        columns=("held_out_label", "seed", "method", "calibration_percentile"),
        expected=expected,
        label="PBMC calibration-percentile table",
    )

    primary = by_seed.loc[np.isclose(by_seed["calibration_percentile"], 0.95)].merge(
        transfer_by_seed.loc[
            transfer_by_seed["method"].isin(
                [item.method for item in CALIBRATION_SENSITIVITY_METHODS]
            ),
            [
                "held_out_label",
                "seed",
                "method",
                "absent_abstention_rate",
                "coverage",
                "post_abstention_macro_f1",
            ],
        ],
        on=["held_out_label", "seed", "method"],
        validate="one_to_one",
        suffixes=("", "_primary"),
    )
    for observed, expected_column in (
        ("held_out_state_abstention_rate", "absent_abstention_rate"),
        ("represented_state_coverage", "coverage"),
        ("post_abstention_macro_f1", "post_abstention_macro_f1_primary"),
    ):
        if not np.allclose(primary[observed], primary[expected_column], atol=1.0e-12, rtol=0):
            raise PBMCSupplementError(f"Primary calibration lineage check failed for {observed}.")

    summary = (
        by_seed.groupby(
            [
                "held_out_label",
                "method",
                "score",
                "display_name",
                "calibration_percentile",
            ],
            sort=False,
        )
        .agg(
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", lambda values: values.std(ddof=1)),
            held_out_state_abstention_rate_mean=(
                "held_out_state_abstention_rate",
                "mean",
            ),
            held_out_state_abstention_rate_std=(
                "held_out_state_abstention_rate",
                lambda values: values.std(ddof=1),
            ),
            represented_state_coverage_mean=("represented_state_coverage", "mean"),
            represented_state_coverage_std=(
                "represented_state_coverage",
                lambda values: values.std(ddof=1),
            ),
            post_abstention_macro_f1_mean=("post_abstention_macro_f1", "mean"),
            post_abstention_macro_f1_std=(
                "post_abstention_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(len(SEEDS)).all():
        raise PBMCSupplementError(
            "Calibration-percentile summaries require every retained donor split."
        )
    return by_seed, summary


def _filter_mechanistic_sources(
    *,
    score_path: Path,
    specificity_path: Path,
    mass_path: Path,
    destination_cells_path: Path,
) -> dict[str, pd.DataFrame]:
    output = {
        "score": pd.read_csv(score_path),
        "specificity": pd.read_csv(specificity_path),
        "mass": pd.read_csv(mass_path),
        "destination_cells": pd.read_parquet(destination_cells_path),
    }
    for key, frame in output.items():
        endpoint_column = "endpoint" if key == "destination_cells" else "held_out_label"
        _require_columns(frame, (endpoint_column,), key)
        output[key] = frame.loc[frame[endpoint_column].isin(ENDPOINTS)].copy()
        if output[key].empty:
            raise PBMCSupplementError(f"Mechanistic source {key} has no retained endpoints.")
    return output


def collect_matched_destination_response(
    *,
    selected_runs: pd.DataFrame,
    runs_root: Path,
    raw_data_path: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    expected_runs = {(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS}
    runs = selected_runs.loc[
        selected_runs["held_out_label"].isin(ENDPOINTS)
        & selected_runs["method"].eq("coreot_full")
        & selected_runs["score"].eq("u"),
        ["held_out_label", "seed", "run_id", "candidate_set"],
    ].drop_duplicates()
    _require_exact_keys(
        runs,
        columns=("held_out_label", "seed"),
        expected=expected_runs,
        label="selected PBMC matched-destination runs",
    )
    metadata = _read_raw_metadata(raw_data_path)
    records: list[dict[str, object]] = []
    source_paths: list[Path] = []

    for run in runs.itertuples(index=False):
        endpoint = str(run.held_out_label)
        run_root = runs_root / str(run.run_id)
        candidate_set = str(run.candidate_set)
        truth = _read_truth(run_root)
        truth_ids = set(truth["cell_id"])
        local = truth.loc[truth["true_label"].astype(str).eq(endpoint)].copy()
        raw_local = metadata.reindex(local["cell_id"])
        if raw_local[["cell_type", "condition"]].isna().any().any():
            raise PBMCSupplementError(
                f"Matched-destination query cells do not join to metadata for {run.run_id}."
            )
        if not raw_local["cell_type"].astype(str).eq(endpoint).all():
            raise PBMCSupplementError(
                f"Matched-destination truth disagrees with metadata for {run.run_id}."
            )
        local["cell_group"] = np.where(
            raw_local["condition"].astype(str).to_numpy() == "stim",
            "held_out_stimulated",
            "same_type_control",
        )
        if not np.array_equal(
            local["cell_group"].eq("held_out_stimulated").to_numpy(),
            local["is_absent_state"].astype(bool).to_numpy(),
        ):
            raise PBMCSupplementError(
                f"Matched-destination groups disagree with absent-state truth for {run.run_id}."
            )

        for condition in CONDITIONS:
            transport_path = (
                run_root
                / "transport"
                / condition
                / candidate_set
                / "coreot_full"
                / "cell_transport_scores.parquet"
            )
            transport = pd.read_parquet(transport_path)
            _require_columns(
                transport,
                ("cell_id", "method", "a", "a_hat"),
                transport_path,
            )
            transport["cell_id"] = transport["cell_id"].astype(str)
            transport = transport.loc[
                transport["method"].eq("coreot_full"),
                ["cell_id", "a", "a_hat"],
            ].copy()
            if transport["cell_id"].duplicated().any() or set(transport["cell_id"]) != truth_ids:
                raise PBMCSupplementError(
                    f"{transport_path} does not match the paired query-cell set."
                )
            numeric = transport[["a", "a_hat"]].to_numpy(dtype=float)
            if (
                not np.isfinite(numeric).all()
                or (transport["a"] <= 0).any()
                or (transport["a_hat"] < 0).any()
            ):
                raise PBMCSupplementError(
                    f"{transport_path} contains invalid transported marginals."
                )

            coupling_path = (
                run_root
                / "transport"
                / condition
                / candidate_set
                / "coreot_full"
                / "sparse_coupling.parquet"
            )
            coupling = pd.read_parquet(
                coupling_path,
                columns=["source_cell_id", "target_cell_id", "coupling"],
            )
            coupling["source_cell_id"] = coupling["source_cell_id"].astype(str)
            coupling["target_cell_id"] = coupling["target_cell_id"].astype(str)
            coupling_values = coupling["coupling"].to_numpy(dtype=float)
            if not np.isfinite(coupling_values).all() or (coupling_values < 0).any():
                raise PBMCSupplementError(f"{coupling_path} contains invalid entries.")
            target_metadata = metadata.reindex(coupling["target_cell_id"])
            if target_metadata[["cell_type", "condition"]].isna().any().any():
                raise PBMCSupplementError(
                    f"{coupling_path} contains targets missing from PBMC metadata."
                )
            target_type = target_metadata["cell_type"].astype(str).to_numpy()
            target_condition = target_metadata["condition"].astype(str).to_numpy()
            coupling["destination_category"] = np.select(
                [
                    (target_type == endpoint) & (target_condition == "ctrl"),
                    (target_type == endpoint) & (target_condition == "stim"),
                ],
                ["same_type_control", "same_type_stimulated"],
                default="other_reference_states",
            )

            transport_indexed = transport.set_index("cell_id")
            for cell_group in ("held_out_stimulated", "same_type_control"):
                group_ids = sorted(
                    local.loc[local["cell_group"].eq(cell_group), "cell_id"].astype(str)
                )
                group_transport = transport_indexed.reindex(group_ids)
                if group_transport[["a", "a_hat"]].isna().any().any():
                    raise PBMCSupplementError(
                        f"Matched-destination marginals are incomplete for {run.run_id}."
                    )
                row_totals = (
                    coupling.loc[coupling["source_cell_id"].isin(group_ids)]
                    .groupby("source_cell_id", sort=False)["coupling"]
                    .sum()
                    .reindex(group_ids)
                    .fillna(0.0)
                )
                if not np.allclose(
                    row_totals.to_numpy(dtype=float),
                    group_transport["a_hat"].to_numpy(dtype=float),
                    rtol=1.0e-10,
                    atol=1.0e-14,
                ):
                    raise PBMCSupplementError(
                        f"Sparse-coupling rows do not reproduce marginals for {run.run_id}."
                    )
                eligible_ids = [
                    cell_id
                    for cell_id in group_ids
                    if float(group_transport.loc[cell_id, "a_hat"]) > TRANSPORT_ETA
                ]
                if not eligible_ids:
                    raise PBMCSupplementError(
                        f"No positive transported mass for {run.run_id}/{cell_group}/{condition}."
                    )
                grouped_mass = (
                    coupling.loc[coupling["source_cell_id"].isin(eligible_ids)]
                    .groupby(["source_cell_id", "destination_category"], sort=False)["coupling"]
                    .sum()
                    .unstack(fill_value=0.0)
                    .reindex(index=eligible_ids, columns=DESTINATION_CATEGORIES, fill_value=0.0)
                )
                conditional = grouped_mass.div(row_totals.reindex(eligible_ids), axis=0)
                if not np.allclose(
                    conditional.sum(axis=1).to_numpy(dtype=float),
                    1.0,
                    rtol=0,
                    atol=1.0e-10,
                ):
                    raise PBMCSupplementError(
                        f"Conditional destinations do not sum to one for {run.run_id}."
                    )
                relative_mass = group_transport["a_hat"] / group_transport["a"]
                for destination_category in DESTINATION_CATEGORIES:
                    records.append(
                        {
                            "held_out_label": endpoint,
                            "seed": int(run.seed),
                            "run_id": str(run.run_id),
                            "cell_group": cell_group,
                            "reference_condition": condition,
                            "destination_category": destination_category,
                            "destination_estimand": (
                                MEAN_CELL_CONDITIONAL_DESTINATION
                            ),
                            "mean_destination_fraction": float(
                                conditional[destination_category].mean()
                            ),
                            "mean_relative_transported_query_mass": float(relative_mass.mean()),
                            "n_group_cells": int(len(group_ids)),
                            "n_positive_transport_cells": int(len(eligible_ids)),
                            "structurally_unavailable": bool(
                                condition == INCOMPLETE_REFERENCE
                                and destination_category == "same_type_stimulated"
                            ),
                        }
                    )
            source_paths.append(transport_path)

    by_seed = pd.DataFrame.from_records(records)
    expected = {
        (endpoint, seed, group, condition, destination)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for group in ("held_out_stimulated", "same_type_control")
        for condition in CONDITIONS
        for destination in DESTINATION_CATEGORIES
    }
    _require_exact_keys(
        by_seed,
        columns=(
            "held_out_label",
            "seed",
            "cell_group",
            "reference_condition",
            "destination_category",
        ),
        expected=expected,
        label="PBMC matched-destination response",
    )
    return by_seed, source_paths


def _save_figure(fig: plt.Figure, outputs: dict[str, Path]) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = RASTER_DPI
        if suffix == "svg":
            # Keep visible labels in the same path-and-comment representation
            # regardless of global Matplotlib state left by another renderer.
            with matplotlib.rc_context({"svg.fonttype": "path"}):
                fig.savefig(path, **kwargs)
        else:
            fig.savefig(path, **kwargs)
        if suffix == "svg":
            content = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                encoding="utf-8",
            )


def _panel_label(axis: plt.Axes, label: str, title: str) -> None:
    axis.set_title(f"{label}. {title}", loc="left", fontsize=8.5, fontweight="bold")


def render_mechanistic_figure(
    sources: dict[str, pd.DataFrame],
    outputs: dict[str, Path],
) -> None:
    response = sources["destination_response"]
    fig = plt.figure(figsize=(11.8, 9.2))
    grid = fig.add_gridspec(
        len(ENDPOINTS),
        2,
        width_ratios=(5.6, 1.15),
        wspace=0.12,
        hspace=0.62,
    )
    row_specs = (
        ("held_out_stimulated", INCOMPLETE_REFERENCE),
        ("held_out_stimulated", "full_reference_control"),
        ("same_type_control", INCOMPLETE_REFERENCE),
        ("same_type_control", "full_reference_control"),
    )
    group_labels = {
        "held_out_stimulated": "Reference-omitted stimulated cells",
        "same_type_control": "Same-type controls",
    }
    split_mass = response.drop_duplicates(
        ["held_out_label", "seed", "cell_group", "reference_condition"]
    )
    mass_summary = split_mass.groupby(
        ["held_out_label", "cell_group", "reference_condition"],
        sort=False,
    )["mean_relative_transported_query_mass"].agg(["mean", "std"])
    mass_upper = max(1.0, float((mass_summary["mean"] + mass_summary["std"]).max()))
    mass_upper = float(np.ceil(mass_upper / 0.25) * 0.25)
    mass_ticks = np.arange(0.0, mass_upper + 0.125, 0.25)

    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        heat_axis = fig.add_subplot(grid[endpoint_index, 0])
        mass_axis = fig.add_subplot(grid[endpoint_index, 1])
        endpoint_rows = response.loc[response["held_out_label"].eq(endpoint)]
        heat_rows = []
        mass_means = []
        mass_sds = []
        mass_colors = []
        row_labels = []
        for cell_group, condition in row_specs:
            subset = endpoint_rows.loc[
                endpoint_rows["cell_group"].eq(cell_group)
                & endpoint_rows["reference_condition"].eq(condition)
            ]
            category_means = (
                subset.groupby("destination_category", sort=False)["mean_destination_fraction"]
                .mean()
                .reindex(DESTINATION_CATEGORIES)
            )
            if category_means.isna().any():
                raise PBMCSupplementError(
                    f"Incomplete matched destinations for {endpoint}/{cell_group}/{condition}."
                )
            heat_rows.append(category_means.to_numpy(dtype=float))
            split_masses = (
                subset.groupby("seed", sort=True)["mean_relative_transported_query_mass"]
                .first()
                .to_numpy(dtype=float)
            )
            if len(split_masses) != len(SEEDS):
                raise PBMCSupplementError(
                    f"Incomplete matched mass summaries for {endpoint}/{cell_group}/{condition}."
                )
            mass_means.append(float(split_masses.mean()))
            mass_sds.append(float(split_masses.std(ddof=1)))
            mass_colors.append(CONDITION_COLORS[condition])
            row_labels.append(f"{group_labels[cell_group]}\n{CONDITION_LABELS[condition]}")

        heat = np.asarray(heat_rows, dtype=float)
        heat_axis.imshow(
            heat,
            vmin=0,
            vmax=1,
            cmap=plt.cm.Blues,
            aspect="auto",
            interpolation="nearest",
        )
        for row_index, (_, condition) in enumerate(row_specs):
            for column_index, destination in enumerate(DESTINATION_CATEGORIES):
                value = heat[row_index, column_index]
                display = (
                    "—"
                    if condition == INCOMPLETE_REFERENCE and destination == "same_type_stimulated"
                    else f"{value:.2f}"
                )
                heat_axis.text(
                    column_index,
                    row_index,
                    display,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#111111" if value < 0.63 else "white",
                )
        heat_axis.set_yticks(range(len(row_labels)), row_labels)
        heat_axis.set_xticks(
            range(len(DESTINATION_CATEGORIES)),
            [DESTINATION_LABELS[item] for item in DESTINATION_CATEGORIES],
        )
        heat_axis.tick_params(axis="both", length=0, labelsize=6.5)
        heat_axis.set_title(
            f"{chr(65 + endpoint_index)}  {endpoint}",
            loc="left",
            weight="bold",
            fontsize=10,
        )
        heat_axis.set_xlabel("Reference state")
        for spine in heat_axis.spines.values():
            spine.set_visible(False)

        mass_axis.barh(
            np.arange(len(row_labels)),
            mass_means,
            xerr=mass_sds,
            color=mass_colors,
            edgecolor="#333333",
            linewidth=0.45,
            alpha=0.82,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
            },
        )
        mass_axis.set_yticks([])
        mass_axis.set_xlim(0.0, mass_upper)
        mass_axis.set_xticks(mass_ticks)
        mass_axis.set_xlabel(
            "Mean relative transported\nquery mass, $\\widehat a_i/a_i$",
            fontsize=7,
        )
        mass_axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        mass_axis.spines[["top", "right", "left"]].set_visible(False)
        mass_axis.tick_params(labelsize=6.5)
        mass_axis.invert_yaxis()

    _save_figure(fig, outputs)
    plt.close(fig)


def render_robustness_figure(
    sensitivity_summary: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    fig = plt.figure(figsize=(7.2, 5.9))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.95,
    )
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=(1, 0.18, 1, 0.18, 1, 0.065),
        wspace=0.08,
        hspace=0.55,
    )
    axes = [
        [fig.add_subplot(grid[row, 2 * index]) for index in range(3)]
        for row in range(2)
    ]
    colorbar_axes = [fig.add_subplot(grid[row, 5]) for row in range(2)]
    tau_min_values = (0.5, 0.75, 1.0)
    tau_max_values = (1.0, 1.25, 1.5)
    metric_rows = (
        ("within_celltype_auprc_mean", "AP", "Mean within-cell-type AP"),
        (
            "forced_macro_f1_mean",
            "forced macro-F1",
            "Mean represented-state forced macro-F1",
        ),
    )
    for metric_index, (column, metric_label, colorbar_label) in enumerate(metric_rows):
        metric_values = sensitivity_summary[column].to_numpy(dtype=float)
        vmin = float(metric_values.min())
        vmax = float(metric_values.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0e-6
        heatmap_image = None
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            axis = axes[metric_index][endpoint_index]
            subset = sensitivity_summary.loc[sensitivity_summary["held_out_label"].eq(endpoint)]
            matrix = np.full((3, 3), np.nan)
            for row in subset.itertuples(index=False):
                row_index = tau_min_values.index(float(row.tau_min))
                column_index = tau_max_values.index(float(row.tau_max))
                matrix[row_index, column_index] = float(getattr(row, column))
            if np.isnan(matrix).any():
                raise PBMCSupplementError(
                    f"The {metric_label} sensitivity heatmap is incomplete for {endpoint}."
                )
            heatmap_image = axis.imshow(
                matrix,
                origin="lower",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            for row_index in range(3):
                for column_index in range(3):
                    value = matrix[row_index, column_index]
                    scaled = (value - vmin) / (vmax - vmin)
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=("white" if scaled < 0.22 or scaled > 0.82 else "#1F1F1F"),
                    )
            axis.set_xticks(range(3), [f"{value:g}" for value in tau_max_values])
            axis.set_yticks(range(3), [f"{value:g}" for value in tau_min_values])
            axis.set_xlabel(r"$\tau_{\max}$")
            if endpoint_index == 0:
                axis.set_ylabel(r"$\tau_{\min}$")
            panel_index = metric_index * len(ENDPOINTS) + endpoint_index
            _panel_label(
                axis,
                chr(ord("A") + panel_index),
                f"{ENDPOINT_SHORT[endpoint]} {metric_label}",
            )
            axis.tick_params(labelsize=6.5)
        if heatmap_image is not None:
            colorbar = fig.colorbar(
                heatmap_image,
                cax=colorbar_axes[metric_index],
            )
            colorbar.set_label(colorbar_label, fontsize=6.5)
            colorbar.ax.tick_params(labelsize=6)
    _save_figure(fig, outputs)
    plt.close(fig)


def render_prior_dependence_figure(
    prior_bins_by_seed: pd.DataFrame,
    prior_bins_summary: pd.DataFrame,
    prior_correlations_by_seed: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    """Render the S2-aligned PBMC prior-dependence diagnostic."""
    split_color = "#B1AAA4"
    mean_color = "#D55E00"
    text_color = "#2D2926"
    zero_color = "#6F655E"
    grid_color = "#ECE6E1"
    x_upper = _rounded_axis_upper(float(prior_bins_by_seed["mean_prior_risk"].max()), 0.05)
    y_upper = _rounded_axis_upper(float(prior_bins_by_seed["mean_u"].max()), 0.08)
    fig = plt.figure(figsize=(183 / 25.4, 65 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.15, 1.15, 1.15, 0.9),
        left=0.075,
        right=0.985,
        bottom=0.30,
        top=0.91,
        wspace=0.44,
    )
    profile_axes: list[plt.Axes] = []
    for index, endpoint in enumerate(ENDPOINTS):
        if index == 0:
            axis = fig.add_subplot(grid[0, index])
        else:
            axis = fig.add_subplot(
                grid[0, index],
                sharex=profile_axes[0],
                sharey=profile_axes[0],
            )
        profile_axes.append(axis)
        endpoint_bins = prior_bins_by_seed.loc[prior_bins_by_seed["held_out_label"].eq(endpoint)]
        if endpoint_bins["seed"].nunique() != len(SEEDS):
            raise PBMCSupplementError(f"Prior-dependence profiles lack five splits for {endpoint}.")
        for _, split_bins in endpoint_bins.groupby("seed", sort=True):
            ordered = split_bins.set_index("prior_bin").reindex(range(1, N_PRIOR_INTERVALS + 1))
            axis.plot(
                ordered["mean_prior_risk"],
                ordered["mean_u"],
                color=split_color,
                linewidth=0.6,
                solid_capstyle="round",
            )
        mean_bins = (
            prior_bins_summary.loc[
                prior_bins_summary["held_out_label"].eq(endpoint)
                & prior_bins_summary["n_splits"].eq(len(SEEDS))
            ]
            .set_index("prior_bin")
            .reindex(range(1, N_PRIOR_INTERVALS + 1))
        )
        axis.plot(
            mean_bins["mean_prior_risk"],
            mean_bins["mean_u"],
            color=mean_color,
            linewidth=1.6,
            marker="o",
            markersize=2.2,
            markerfacecolor=mean_color,
            markeredgewidth=0,
            solid_capstyle="round",
        )
        axis.set_xlim(0, x_upper)
        axis.set_ylim(0, y_upper)
        axis.set_ylabel(r"Query-marginal deficit $u_{q,i}$", fontsize=7)
        axis.grid(color=grid_color, linewidth=0.35)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(width=0.6, length=2.5, labelsize=6.5)
        _panel_label(axis, chr(ord("A") + index), endpoint)

    jitter = dict(zip(SEEDS, (-0.06, -0.03, 0.0, 0.03, 0.06), strict=True))
    correlation_axis = fig.add_subplot(grid[0, 3])
    y_positions = {endpoint: 2 - index for index, endpoint in enumerate(ENDPOINTS)}
    for endpoint in ENDPOINTS:
        values = prior_correlations_by_seed.loc[
            prior_correlations_by_seed["held_out_label"].eq(endpoint)
        ].sort_values("seed")
        if len(values) != len(SEEDS):
            raise PBMCSupplementError(
                f"Prior-dependence correlations lack five splits for {endpoint}."
            )
        correlation_axis.scatter(
            values["spearman"],
            [y_positions[endpoint] + jitter[int(seed)] for seed in values["seed"]],
            color=split_color,
            s=9,
            edgecolor="none",
            zorder=2,
        )
        correlation_axis.scatter(
            values["spearman"].mean(),
            y_positions[endpoint],
            marker="D",
            color=mean_color,
            s=25,
            edgecolor=text_color,
            linewidth=0.45,
            zorder=3,
        )
    correlation_axis.axvline(0, color=zero_color, linewidth=0.8)
    correlation_axis.set_xlim(-1, 1)
    correlation_axis.set_ylim(-0.4, 2.4)
    correlation_axis.set_yticks(
        [y_positions[endpoint] for endpoint in ENDPOINTS],
        ["B cells", "NK cells", "Dendritic\ncells"],
    )
    correlation_axis.set_xlabel("Spearman correlation", fontsize=7)
    correlation_axis.grid(axis="x", color=grid_color, linewidth=0.35)
    correlation_axis.spines[["top", "right"]].set_visible(False)
    correlation_axis.tick_params(width=0.6, length=2.5, labelsize=6.5)
    _panel_label(correlation_axis, "D", "Within-split association")

    profile_legend = [
        Line2D(
            [0],
            [0],
            color=split_color,
            linewidth=0.6,
            label="Donor split",
        ),
        Line2D(
            [0],
            [0],
            color=mean_color,
            linewidth=1.6,
            marker="o",
            markersize=2.2,
            label="Donor-equal mean",
        ),
    ]
    association_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=split_color,
            markeredgewidth=0,
            markersize=3,
            label="Donor split",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=mean_color,
            markeredgecolor=text_color,
            markeredgewidth=0.45,
            markersize=5,
            label="Donor-equal mean",
        ),
    ]
    fig.legend(
        handles=profile_legend,
        frameon=False,
        fontsize=6,
        loc="lower center",
        bbox_to_anchor=(0.37, 0.025),
        ncol=2,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0,
    )
    fig.legend(
        handles=association_legend,
        frameon=False,
        fontsize=6,
        loc="lower center",
        bbox_to_anchor=(0.88, 0.025),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.9,
        borderaxespad=0,
    )
    fig.text(
        0.37,
        0.15,
        r"Prior risk $r_{q,i}=1-\rho_{q,i}$",
        va="center",
        ha="center",
        fontsize=7,
    )
    _save_figure(fig, outputs)
    plt.close(fig)


def _rounded_axis_upper(value: float, padding: float) -> float:
    if not np.isfinite(value) or value < 0:
        raise PBMCSupplementError(f"Invalid diagnostic axis maximum: {value}.")
    padded = value * (1.0 + padding)
    step = 0.05 if padded <= 0.5 else 0.1
    return min(1.0, max(step, float(np.ceil(padded / step) * step)))


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path, project_root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(project_root)),
        "sha256": _sha256(path),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def generate_pbmc_supplement(project_root: Path) -> dict[str, Path]:
    project_root = project_root.resolve()
    paths = SupplementPaths.from_project_root(project_root)
    paths.data_root.mkdir(parents=True, exist_ok=True)
    paths.figures_root.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        project_root / "results/PBMC/compare_baselines/tables/compare_detection_by_run.csv"
    )
    transfer_path = (
        project_root / "results/PBMC/compare_baselines/tables/"
        "compare_shared_label_transfer_by_run.csv"
    )
    full_reference_path = (
        project_root / "results/PBMC/compare_baselines/tables/compare_full_reference_by_run.csv"
    )
    split_path = project_root / "results/PBMC/main/tables/split_summary_by_run.csv"
    raw_data_path = project_root / "data/raw/kang_2018.h5ad"
    runs_root = project_root / "runs"
    sensitivity_detection_path = (
        project_root / "results/PBMC/sensitivity/full_tau_labelwise/tables/detection_by_run.csv"
    )
    sensitivity_transfer_path = (
        project_root / "results/PBMC/sensitivity/full_tau_labelwise/tables/"
        "shared_label_transfer_by_run.csv"
    )
    calibration_sensitivity_path = (
        project_root / "results/PBMC/figures/data/figure_s7_calibration_by_seed.csv"
    )
    figure_data_root = project_root / "results/PBMC/figures/data"
    destination_cells_path = (
        project_root / "results/PBMC/manuscript/supp_destination_support/"
        "destination_support_by_cell.parquet"
    )
    component_root = project_root / "results/PBMC/sensitivity/component_ablation"
    component_by_seed_path = component_root / "tables/component_ablation_by_seed.csv"
    component_summary_path = component_root / "tables/component_ablation_summary.csv"
    component_minus_m_path = paths.docs_root / "figs/manuscript_fig_pbmc_component_minus_m.png"
    rho_tau_root = (
        project_root / "results/PBMC/sensitivity/rho_attribution_tau_surface_alpha0_range075_175"
    )
    rho_tau_by_seed_path = rho_tau_root / "tables/rho_tau_surface_range075_175_by_seed.csv"
    rho_tau_summary_path = rho_tau_root / "tables/rho_tau_surface_range075_175_summary.csv"
    rho_tau_figure_path = (
        paths.docs_root / "figs/manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png"
    )
    rho_tau_caption_path = (
        paths.docs_root / "figs/manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175_caption.md"
    )
    rho_alpha_summary_path = (
        project_root / "results/PBMC/sensitivity/rho_attribution_alpha_search/tables/"
        "rho_attribution_summary.csv"
    )
    for component_path in (
        component_by_seed_path,
        component_summary_path,
        component_minus_m_path,
        rho_tau_by_seed_path,
        rho_tau_summary_path,
        rho_tau_figure_path,
        rho_tau_caption_path,
        rho_alpha_summary_path,
    ):
        if not component_path.is_file():
            raise FileNotFoundError(component_path)
    comparison = _comparison_rows(comparison_path)
    selected_benchmark_runs = comparison.loc[
        comparison["method"].eq("coreot_full") & comparison["score"].eq("u"),
        ["run_id", "candidate_set"],
    ].drop_duplicates()
    benchmark_coupling_inputs = [
        runs_root
        / str(run.run_id)
        / "transport"
        / condition
        / str(run.candidate_set)
        / "coreot_full"
        / "sparse_coupling.parquet"
        for run in selected_benchmark_runs.itertuples(index=False)
        for condition in CONDITIONS
    ]
    benchmark_by_seed, benchmark_summary = collect_benchmark_summary(
        split_path,
        selected_runs=comparison,
        runs_root=runs_root,
        raw_data_path=raw_data_path,
    )
    penalty_by_seed, penalty_summary = collect_realized_penalties(
        selected_runs=comparison,
        runs_root=runs_root,
    )
    (
        within_by_seed,
        within_summary,
        contrast_by_seed,
        contrast_summary,
    ) = collect_within_celltype_detection(
        comparison_path=comparison_path,
        raw_data_path=raw_data_path,
        runs_root=runs_root,
    )
    transfer_by_seed, transfer_summary = collect_transfer_results(
        transfer_path,
        detection_path=comparison_path,
    )
    calibration_by_run, calibration_summary = collect_full_reference_calibration(
        full_reference_path,
        runs_root=runs_root,
    )
    global_by_seed, global_summary = collect_global_detection(comparison)
    (
        sensitivity_by_seed,
        sensitivity_summary,
        sensitivity_compact,
    ) = collect_sensitivity_results(
        detection_path=sensitivity_detection_path,
        transfer_path=sensitivity_transfer_path,
        raw_data_path=raw_data_path,
        runs_root=runs_root,
    )
    (
        calibration_curve_by_seed,
        calibration_curve_summary,
    ) = collect_calibration_sensitivity(calibration_sensitivity_path)
    (
        calibration_percentile_by_seed,
        calibration_percentile_summary,
    ) = collect_calibration_percentile_table(
        selected_runs=comparison,
        transfer_by_seed=transfer_by_seed,
        runs_root=runs_root,
    )
    (
        prior_bins_by_seed,
        prior_bins_summary,
        prior_correlations_by_seed,
        prior_correlations_summary,
    ) = collect_prior_dependence(
        selected_runs=comparison,
        raw_data_path=raw_data_path,
        runs_root=runs_root,
    )
    mechanistic = _filter_mechanistic_sources(
        score_path=figure_data_root / "figure_s6_score_rescue_by_seed.csv",
        specificity_path=(figure_data_root / "figure_s6_restoration_specificity_by_seed.csv"),
        mass_path=figure_data_root / "figure_s6_mass_rescue_by_seed.csv",
        destination_cells_path=destination_cells_path,
    )
    destination_response, destination_response_sources = collect_matched_destination_response(
        selected_runs=comparison,
        runs_root=runs_root,
        raw_data_path=raw_data_path,
    )
    mechanistic["destination_response"] = destination_response

    output_frames = {
        "pbmc_benchmark_by_seed.csv": benchmark_by_seed,
        "pbmc_benchmark_summary.csv": benchmark_summary,
        "pbmc_selected_penalties_by_seed.csv": penalty_by_seed,
        "pbmc_selected_penalties_summary.csv": penalty_summary,
        "pbmc_within_celltype_detection_by_seed.csv": within_by_seed,
        "pbmc_within_celltype_detection_summary.csv": within_summary,
        "pbmc_within_celltype_contrasts_by_seed.csv": contrast_by_seed,
        "pbmc_within_celltype_contrasts_summary.csv": contrast_summary,
        "pbmc_represented_transfer_by_seed.csv": transfer_by_seed,
        "pbmc_represented_transfer_summary.csv": transfer_summary,
        "pbmc_full_reference_calibration_by_run.csv": calibration_by_run,
        "pbmc_full_reference_calibration_summary.csv": calibration_summary,
        "pbmc_global_detection_by_seed.csv": global_by_seed,
        "pbmc_global_detection_summary.csv": global_summary,
        "pbmc_sensitivity_by_seed.csv": sensitivity_by_seed,
        "pbmc_sensitivity_summary.csv": sensitivity_summary,
        "pbmc_sensitivity_compact.csv": sensitivity_compact,
        "pbmc_calibration_sensitivity_by_seed.csv": calibration_curve_by_seed,
        "pbmc_calibration_sensitivity_summary.csv": calibration_curve_summary,
        "pbmc_calibration_percentile_by_seed.csv": calibration_percentile_by_seed,
        "pbmc_calibration_percentile_summary.csv": calibration_percentile_summary,
        "pbmc_prior_dependence_bins_by_seed.csv": prior_bins_by_seed,
        "pbmc_prior_dependence_bins_summary.csv": prior_bins_summary,
        "pbmc_prior_dependence_correlations_by_seed.csv": prior_correlations_by_seed,
        "pbmc_prior_dependence_correlations_summary.csv": prior_correlations_summary,
        "pbmc_mechanistic_score_by_seed.csv": mechanistic["score"],
        "pbmc_mechanistic_specificity_by_seed.csv": mechanistic["specificity"],
        "pbmc_mechanistic_mass_by_seed.csv": mechanistic["mass"],
        "pbmc_matched_destination_response_by_seed.csv": mechanistic["destination_response"],
    }
    generated_tables = [
        _write_csv(frame, paths.data_root / filename) for filename, frame in output_frames.items()
    ]
    mechanistic_cells_output = _write_parquet(
        mechanistic["destination_cells"],
        paths.data_root / "pbmc_mechanistic_destination_by_cell.parquet",
    )

    mechanistic_outputs = {
        suffix: paths.figures_root / f"manuscript_fig_pbmc_supp_mechanistic.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    robustness_outputs = {
        suffix: paths.figures_root / f"manuscript_fig_pbmc_supp_robustness.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    prior_dependence_outputs = {
        suffix: paths.figures_root / f"manuscript_fig_pbmc_supp_prior_dependence.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    render_mechanistic_figure(mechanistic, mechanistic_outputs)
    render_robustness_figure(
        sensitivity_summary,
        robustness_outputs,
    )
    render_prior_dependence_figure(
        prior_bins_by_seed,
        prior_bins_summary,
        prior_correlations_by_seed,
        prior_dependence_outputs,
    )
    all_artifacts = [
        *generated_tables,
        mechanistic_cells_output,
        *mechanistic_outputs.values(),
        *robustness_outputs.values(),
        *prior_dependence_outputs.values(),
        component_minus_m_path,
        rho_tau_figure_path,
        rho_tau_caption_path,
    ]
    source_inputs = [
        comparison_path,
        transfer_path,
        full_reference_path,
        split_path,
        raw_data_path,
        sensitivity_detection_path,
        sensitivity_transfer_path,
        calibration_sensitivity_path,
        figure_data_root / "figure_s6_score_rescue_by_seed.csv",
        figure_data_root / "figure_s6_restoration_specificity_by_seed.csv",
        figure_data_root / "figure_s6_mass_rescue_by_seed.csv",
        destination_cells_path,
        component_by_seed_path,
        component_summary_path,
        rho_tau_by_seed_path,
        rho_tau_summary_path,
        rho_alpha_summary_path,
        *[
            runs_root
            / str(run.run_id)
            / "scoring"
            / condition
            / str(run.candidate_set)
            / "cell_scores.parquet"
            for run in selected_benchmark_runs.itertuples(index=False)
            for condition in CONDITIONS
        ],
        *[
            runs_root
            / str(run.run_id)
            / "scoring"
            / INCOMPLETE_REFERENCE
            / str(run.candidate_set)
            / "scoring_manifest.yaml"
            for run in selected_benchmark_runs.itertuples(index=False)
        ],
        *benchmark_coupling_inputs,
        *destination_response_sources,
        *[
            project_root / artifact
            for artifact in sensitivity_by_seed[
                ["transport_manifest", "method_params"]
            ].to_numpy().ravel()
        ],
    ]
    manifest_path = project_root / "results/PBMC/manuscript/pbmc_supplement_manifest.yaml"
    manifest = {
        "stage": "generate_pbmc_supplement",
        "generator": "experiments/pbmc_state/generate_pbmc_supplement.py",
        "artifacts": [_artifact_record(path, project_root) for path in all_artifacts],
        "sources": [_artifact_record(path, project_root) for path in source_inputs],
        "metadata": {
            "held_out_states": list(ENDPOINTS),
            "seeds": list(SEEDS),
            "primary_score": "u",
            "primary_endpoint": "within_celltype_condition_detection",
            "variability": "mean_and_sample_sd_over_five_fixed_splits",
            "inference": "descriptive_no_confidence_intervals_or_hypothesis_tests",
            "matched_reference_estimands": {
                "restoration_specificity_compatibility_column": (
                    CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE
                ),
                "figure3_destination": TYPICAL_CELL_CONDITIONAL_DESTINATION,
                "manuscript_figure_s6_destination": (
                    MEAN_CELL_CONDITIONAL_DESTINATION
                ),
                "legacy_four_endpoint_figure_s6_destination": (
                    POOLED_TRANSPORTED_MASS_DESTINATION_COMPOSITION
                ),
                "legacy_four_endpoint_figure_s6_role": (
                    "reproducibility_artifact_not_manuscript_authority"
                ),
            },
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "mechanistic_png": mechanistic_outputs["png"],
        "robustness_png": robustness_outputs["png"],
        "prior_dependence_png": prior_dependence_outputs["png"],
        "data_root": paths.data_root,
    }
