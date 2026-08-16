from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.artifacts.run_artifacts import (
    ArtifactBatchError,
    RunArtifacts,
)
from coreot.results.grid_common import (
    DISPLAY_NAME_BY_METHOD,
    LABEL_TRANSFER_METHODS,
    MAIN_TABLE_1_COLUMN_HEADERS,
    MAIN_TABLE_1_QUANTITIES,
    MAIN_TABLE_2_COLUMN_HEADERS,
    MAIN_TABLE_2_QUANTITIES,
    METHOD_GROUP_BY_METHOD,
    PRIMARY_SCORE_BY_METHOD,
    STAGE,
    TABLE_ROW_ORDER,
    ResultsGridError,
    _accuracy,
    _format_table_value,
    _macro_f1,
    _markdown_table,
    _nonempty_string,
    _partition_main_grid_runs,
    build_detection_by_run,
    discover_run_descriptors,
    summarize_run_metrics,
)
from coreot.config.load import load_yaml
from coreot.data.raw_import import _normalize_pbmc_condition, _pbmc_broad_lineage

# ---------------------------------------------------------------------------
# PBMC-specific constants
# ---------------------------------------------------------------------------

RESCUE_CELL_GROUPS: tuple[str, ...] = (
    "held_out_positives",
    "same_celltype_controls",
    "other_stimulated",
    "other_shared",
)

RESCUE_GROUP_DISPLAY: dict[str, str] = {
    "held_out_positives": "Held-out stimulated",
    "same_celltype_controls": "Same cell-type controls",
    "other_stimulated": "Other stimulated",
    "other_shared": "Other shared",
}


# ---------------------------------------------------------------------------
# Paths dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PbmcArtifactPaths:
    output_root: Path
    split_summary_by_run: Path
    split_summary_md: Path
    global_detection_by_run: Path
    global_detection_summary: Path
    within_celltype_detection_by_run: Path
    within_celltype_detection_summary: Path
    rescue_by_run: Path
    rescue_summary: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    detection_table_1: Path
    detection_table_2: Path
    rescue_table: Path
    shared_label_transfer_table: Path
    summary_md: Path
    manifest: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_raw_obs(
    run_root: Path,
    *,
    grid_dir: Path | None = None,
    descriptor: Any | None = None,
) -> pd.DataFrame:
    """Read the raw AnnData .obs from a completed run."""
    h5ad_path = run_root / "raw" / "input.h5ad"
    if h5ad_path.is_file():
        return _standardize_raw_obs_index(anndata.read_h5ad(h5ad_path).obs)
    if grid_dir is not None and descriptor is not None:
        return _read_configured_raw_obs(grid_dir / descriptor.run_id / "raw_import.yaml")
    raise ResultsGridError(f"Raw input not found: {h5ad_path}")


def _read_configured_raw_obs(raw_import_config: Path) -> pd.DataFrame:
    """Read standardized .obs metadata from a retained raw-import config."""
    if not raw_import_config.is_file():
        raise ResultsGridError(f"Raw-import config not found: {raw_import_config}")
    config = load_yaml(raw_import_config)
    input_config = config.get("input", {})
    if not isinstance(input_config, dict) or not isinstance(input_config.get("path"), str):
        raise ResultsGridError(f"Missing input.path in raw-import config: {raw_import_config}")
    source = Path(input_config["path"])
    if not source.is_file():
        raise ResultsGridError(f"Configured raw input not found: {source}")

    adapter = config.get("adapter")
    if adapter != "pbmc_ifnb":
        return _standardize_raw_obs_index(anndata.read_h5ad(source).obs)

    adapter_config = config.get("pbmc_ifnb", {})
    if adapter_config is not None and not isinstance(adapter_config, dict):
        raise ResultsGridError(f"pbmc_ifnb config must be a mapping: {raw_import_config}")
    pbmc_config = adapter_config if isinstance(adapter_config, dict) else {}

    adata = anndata.read_h5ad(source)
    obs = adata.obs.copy()
    celltype_column = str(pbmc_config.get("cell_type_column", "cell_type"))
    condition_column = str(pbmc_config.get("condition_column", "label"))
    donor_column = str(pbmc_config.get("donor_column", "replicate"))
    sample_column = str(pbmc_config.get("sample_column", donor_column))
    batch_column = str(pbmc_config.get("batch_column", donor_column))
    missing = [
        column
        for column in (
            celltype_column,
            condition_column,
            donor_column,
            sample_column,
            batch_column,
        )
        if column not in obs.columns
    ]
    if missing:
        raise ResultsGridError(
            "Configured PBMC input .obs is missing required column(s): "
            + ", ".join(missing)
        )

    cell_ids = _configured_cell_ids(
        adata,
        obs,
        str(pbmc_config.get("cell_id_source", "obs_names")),
        raw_import_config,
    )
    condition = _normalize_pbmc_condition(
        obs[condition_column], pbmc_config.get("condition_values", {})
    )
    standardized = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": obs[celltype_column].astype(str).to_numpy(),
            "broad_label": obs[celltype_column]
            .astype(str)
            .map(_pbmc_broad_lineage)
            .astype(str)
            .to_numpy(),
            "sample_id": obs[sample_column].astype(str).to_numpy(),
            "donor_id": obs[donor_column].astype(str).to_numpy(),
            "batch_id": obs[batch_column].astype(str).to_numpy(),
            "state_condition": condition.to_numpy(),
        },
        index=pd.Index(cell_ids, name=adata.obs_names.name),
    )
    return standardized


def _standardize_raw_obs_index(obs: pd.DataFrame) -> pd.DataFrame:
    standardized = obs.copy()
    if "cell_id" not in standardized.columns:
        standardized["cell_id"] = standardized.index.astype(str)
    return standardized.set_index("cell_id", drop=False)


def _configured_cell_ids(
    adata: anndata.AnnData,
    obs: pd.DataFrame,
    source: str,
    config_path: Path,
) -> pd.Index:
    if source == "obs_names":
        return pd.Index(adata.obs_names.astype(str))
    if source not in obs.columns:
        raise ResultsGridError(
            f"Configured cell_id_source is absent from .obs in {config_path}: {source}"
        )
    return pd.Index(obs[source].astype(str))


def _read_split_manifest(run_root: Path) -> pd.DataFrame:
    """Read the split manifest."""
    path = run_root / "benchmark" / "split_manifest.csv"
    if not path.is_file():
        raise ResultsGridError(f"Split manifest not found: {path}")
    return pd.read_csv(path)


def _read_query_truth(run_root: Path, condition: str) -> pd.DataFrame:
    """Read query_truth.csv via RunArtifacts."""
    return RunArtifacts(run_root, STAGE).evaluation_truth(condition).query_truth().read()


def _read_cell_scores(
    run_root: Path, condition: str, candidate_set: str
) -> pd.DataFrame:
    """Read cell_scores.parquet via RunArtifacts."""
    return (
        RunArtifacts(run_root, STAGE)
        .scoring(condition, candidate_set)
        .cell_scores()
        .read()
    )


# ---------------------------------------------------------------------------
# Issue 06: Split summary
# ---------------------------------------------------------------------------


def build_pbmc_split_summary_by_run(
    *,
    runs_root: Path,
    descriptors: tuple,
    grid_dir: Path | None = None,
) -> pd.DataFrame:
    """Build per-run split summary for PBMC stimulated-state experiment.

    Returns a DataFrame with donor counts, source/target cell counts,
    positive counts, same-cell-type controls, other stimulated counts,
    broad lineage, and positive fraction.
    """
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        raw_obs = _read_raw_obs(run_root, grid_dir=grid_dir, descriptor=descriptor)
        split_manifest = _read_split_manifest(run_root)
        truth = _read_query_truth(run_root, "incomplete_reference")
        raw_obs_indexed = raw_obs.set_index("cell_id", drop=False)
        query_cell_ids = split_manifest.loc[
            split_manifest["split_domain"] == "query", "cell_id"
        ].astype(str)
        reference_cell_ids = split_manifest.loc[
            split_manifest["split_domain"] == "reference", "cell_id"
        ].astype(str)

        def _donor_id(cell_id: str) -> str:
            if cell_id not in raw_obs_indexed.index:
                return "unknown"
            row = raw_obs_indexed.loc[cell_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return str(row.get("donor_id", "unknown"))

        # Donors
        query_donors = sorted({_donor_id(str(cid)) for cid in query_cell_ids})
        reference_donors = sorted({_donor_id(str(cid)) for cid in reference_cell_ids})

        # Cell counts
        n_source = int(len(query_cell_ids))
        n_full_target = int(len(reference_cell_ids))
        removed_reference = 0
        for cid in reference_cell_ids:
            if cid not in raw_obs_indexed.index:
                continue
            obs_row = raw_obs_indexed.loc[cid]
            if isinstance(obs_row, pd.DataFrame):
                obs_row = obs_row.iloc[0]
            if (
                str(obs_row.get("cell_type", "")) == descriptor.held_out_label
                and str(obs_row.get("state_condition", "")) == "stim"
            ):
                removed_reference += 1
        n_incomplete_target = n_full_target - removed_reference

        # Positives and same-cell-type controls
        n_positives = int(truth["is_absent_state"].astype(bool).sum())
        same_ct_mask = (
            (truth["true_label"] == descriptor.held_out_label)
            & (~truth["is_absent_state"].astype(bool))
        )
        n_same_ct_controls = int(same_ct_mask.sum())

        # Other stimulated cells: query cells with state_condition == stim
        # that are not the held-out type
        other_stim_count = 0
        broad_lineage = "unknown"
        for cid in query_cell_ids:
            if cid in raw_obs_indexed.index:
                obs_row = raw_obs_indexed.loc[cid]
                if isinstance(obs_row, pd.DataFrame):
                    obs_row = obs_row.iloc[0]
                cell_type = str(obs_row.get("cell_type", ""))
                state_cond = str(obs_row.get("state_condition", ""))
                if cell_type == descriptor.held_out_label:
                    broad_lineage = str(obs_row.get("broad_label", "unknown"))
                elif state_cond == "stim":
                    other_stim_count += 1

        positive_fraction = n_positives / n_source if n_source > 0 else float("nan")

        rows.append(
            {
                "run_id": descriptor.run_id,
                "held_out_label": descriptor.held_out_label,
                "seed": descriptor.seed,
                "query_donors": ", ".join(query_donors),
                "reference_donors": ", ".join(reference_donors),
                "n_source_cells": n_source,
                "n_full_reference_target_cells": n_full_target,
                "n_incomplete_reference_target_cells": n_incomplete_target,
                "n_positives": n_positives,
                "n_same_celltype_controls": n_same_ct_controls,
                "n_other_stimulated": other_stim_count,
                "broad_lineage": broad_lineage,
                "positive_fraction": positive_fraction,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Issue 07: Within-cell-type detection
# ---------------------------------------------------------------------------


def build_pbmc_within_celltype_detection_by_run(
    *,
    runs_root: Path,
    descriptors: tuple,
    candidate_set: str,
    condition: str,
    methods: tuple,
) -> pd.DataFrame:
    """Build within-cell-type detection metrics for PBMC.

    Restricts source cells to the held-out cell type only.
    Positives = held-out stimulated cells (is_absent_state=True).
    Negatives = same-cell-type control cells (is_absent_state=False).
    """
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _read_query_truth(run_root, condition)
        score_frame = _read_cell_scores(run_root, condition, candidate_set)

        # Filter to same cell type
        same_ct_mask = truth["true_label"] == descriptor.held_out_label
        same_ct_truth = truth.loc[same_ct_mask].copy()

        n_positives = int(same_ct_truth["is_absent_state"].astype(bool).sum())
        n_negatives = int((~same_ct_truth["is_absent_state"].astype(bool)).sum())
        if n_positives < 1 or n_negatives < 1:
            continue

        auprc_baseline = n_positives / (n_positives + n_negatives)

        for method in methods:
            method_scores = score_frame.loc[
                score_frame["method"] == method
            ].copy()
            if method_scores.empty:
                continue

            primary_score = PRIMARY_SCORE_BY_METHOD[method]
            joined = method_scores.merge(
                same_ct_truth[["cell_id", "is_absent_state"]],
                on="cell_id",
                how="inner",
            )
            if joined.empty:
                continue

            y_true = joined["is_absent_state"].astype(bool).astype(int)
            score_vals = pd.to_numeric(joined[primary_score], errors="coerce")
            valid = ~score_vals.isna()
            if valid.sum() < 2:
                continue

            y_valid = y_true.loc[valid]
            s_valid = score_vals.loc[valid]

            try:
                auroc = float(roc_auc_score(y_valid, s_valid))
            except ValueError:
                auroc = float("nan")
            try:
                auprc = float(average_precision_score(y_valid, s_valid))
            except ValueError:
                auprc = float("nan")

            # Abstention rate for positives
            pos_mask = joined["is_absent_state"].astype(bool)
            pos_cells = joined.loc[pos_mask]
            if "abstain_u_or_entropy" in pos_cells.columns:
                absent_abstention = float(
                    pos_cells["abstain_u_or_entropy"].astype(bool).mean()
                ) if len(pos_cells) > 0 else float("nan")
            else:
                absent_abstention = float("nan")

            # Median scores
            median_pos = (
                float(s_valid.loc[y_valid == 1].median())
                if (y_valid == 1).sum() > 0
                else float("nan")
            )
            median_neg = (
                float(s_valid.loc[y_valid == 0].median())
                if (y_valid == 0).sum() > 0
                else float("nan")
            )

            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(method, "unknown"),
                    "primary_score": primary_score,
                    "n_positives": n_positives,
                    "n_negatives": n_negatives,
                    "auroc": auroc,
                    "auprc": auprc,
                    "auprc_baseline": auprc_baseline,
                    "absent_abstention_rate": absent_abstention,
                    "median_positives": median_pos,
                    "median_negatives": median_neg,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Issue 08: Full-reference rescue
# ---------------------------------------------------------------------------


def build_pbmc_rescue_by_run(
    *,
    runs_root: Path,
    descriptors: tuple,
    candidate_set: str,
    methods: tuple,
    grid_dir: Path | None = None,
) -> pd.DataFrame:
    """Build full-reference rescue metrics for PBMC.

    Aligns source cells between incomplete_reference and full_reference_control,
    computes per-cell score drops, and summarizes by cell group.
    """
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        raw_obs = _read_raw_obs(run_root, grid_dir=grid_dir, descriptor=descriptor)
        truth_inc = _read_query_truth(run_root, "incomplete_reference")
        scores_inc = _read_cell_scores(
            run_root, "incomplete_reference", candidate_set
        )
        scores_full = _read_cell_scores(
            run_root, "full_reference_control", candidate_set
        )

        for method in methods:
            primary_score = PRIMARY_SCORE_BY_METHOD[method]

            inc_method = scores_inc.loc[scores_inc["method"] == method].copy()
            full_method = scores_full.loc[scores_full["method"] == method].copy()
            if inc_method.empty or full_method.empty:
                continue

            inc_vals = pd.to_numeric(inc_method[primary_score], errors="coerce")
            full_vals = pd.to_numeric(full_method[primary_score], errors="coerce")

            inc_scored = inc_method.loc[~inc_vals.isna()].copy()
            full_scored = full_method.loc[~full_vals.isna()].copy()

            # Align by cell_id
            merged = inc_scored[["cell_id", primary_score]].merge(
                full_scored[["cell_id", primary_score]],
                on="cell_id",
                how="inner",
                suffixes=("_incomplete", "_full"),
            )
            if merged.empty:
                continue

            merged["score_drop"] = (
                merged[f"{primary_score}_incomplete"]
                - merged[f"{primary_score}_full"]
            )

            # Merge with truth for cell grouping
            merged = merged.merge(
                truth_inc[["cell_id", "true_label", "is_absent_state"]],
                on="cell_id",
                how="left",
            )

            # Classify into cell groups
            def _classify_group(row: pd.Series) -> str:
                if row["is_absent_state"]:
                    return "held_out_positives"
                if row["true_label"] == descriptor.held_out_label:
                    return "same_celltype_controls"
                # Check state_condition from raw for "other_stimulated"
                return _classify_stimulated(row["cell_id"], raw_obs)

            merged["cell_group"] = merged.apply(_classify_group, axis=1)

            for group in RESCUE_CELL_GROUPS:
                group_data = merged.loc[merged["cell_group"] == group]
                n_cells = len(group_data)
                if n_cells == 0:
                    continue
                rows.append(
                    {
                        "run_id": descriptor.run_id,
                        "held_out_label": descriptor.held_out_label,
                        "seed": descriptor.seed,
                        "candidate_set": candidate_set,
                        "method": method,
                        "method_group": METHOD_GROUP_BY_METHOD.get(method, "unknown"),
                        "primary_score": primary_score,
                        "cell_group": group,
                        "n_cells": n_cells,
                        "mean_score_incomplete": float(
                            group_data[f"{primary_score}_incomplete"].mean()
                        ),
                        "mean_score_full": float(
                            group_data[f"{primary_score}_full"].mean()
                        ),
                        "mean_score_drop": float(group_data["score_drop"].mean()),
                        "median_score_drop": float(group_data["score_drop"].median()),
                    }
                )
    return pd.DataFrame(rows)


def _classify_stimulated(cell_id: str, raw_obs: pd.DataFrame) -> str:
    """Check raw data to classify a cell as other_stimulated or other_shared."""
    if cell_id not in raw_obs.index:
        return "other_shared"
    row = raw_obs.loc[cell_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    state_cond = str(row.get("state_condition", ""))
    return "other_stimulated" if state_cond == "stim" else "other_shared"


# ---------------------------------------------------------------------------
# Issue 09: Shared label transfer
# ---------------------------------------------------------------------------


def build_pbmc_shared_label_transfer_by_run(
    *,
    runs_root: Path,
    descriptors: tuple,
    candidate_set: str,
    condition: str,
    methods: tuple,
) -> pd.DataFrame:
    """Build PBMC shared-cell label-transfer metrics.

    PBMC uses a condition-specific held-out state: only held-out stimulated
    source cells are absent. Same-cell-type control cells remain shared and
    must be included in the cell-type label-transfer label support.
    """
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _read_query_truth(run_root, condition)
        score_frame = _read_cell_scores(run_root, condition, candidate_set)
        for method in methods:
            method_scores = score_frame.loc[score_frame["method"] == method].copy()
            if method_scores.empty:
                raise ResultsGridError(f"No cell scores found for method={method}")
            joined = method_scores.merge(truth, on="cell_id", how="left", validate="many_to_one")
            if joined["true_label"].isna().any():
                raise ResultsGridError(f"Missing query truth for method={method}")
            shared = joined.loc[joined["is_shared_state"].astype(bool)].copy()
            forced_labeled = shared.loc[_nonempty_string(shared["forced_label"])]
            non_abstained = shared.loc[~shared["abstain_u_or_entropy"].astype(bool)]
            post_labeled = non_abstained.loc[
                _nonempty_string(non_abstained["final_label_abstention_aware"])
            ]
            labels = sorted(str(label) for label in shared["true_label"].dropna().unique())
            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(method, "unknown"),
                    "n_shared": int(len(shared)),
                    "n_forced_labeled": int(len(forced_labeled)),
                    "n_non_abstained": int(len(non_abstained)),
                    "forced_accuracy": _accuracy(
                        forced_labeled["true_label"], forced_labeled["forced_label"]
                    ),
                    "forced_macro_f1": _macro_f1(
                        forced_labeled["true_label"], forced_labeled["forced_label"], labels
                    ),
                    "post_abstention_accuracy": _accuracy(
                        post_labeled["true_label"],
                        post_labeled["final_label_abstention_aware"],
                    ),
                    "post_abstention_macro_f1": _macro_f1(
                        post_labeled["true_label"],
                        post_labeled["final_label_abstention_aware"],
                        labels,
                    ),
                    "coverage": float(len(non_abstained) / len(shared))
                    if len(shared)
                    else math.nan,
                    "shared_false_abstention_rate": float(
                        shared["abstain_u_or_entropy"].astype(bool).mean()
                    )
                    if len(shared)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Issue 10: Orchestrator
# ---------------------------------------------------------------------------


def write_pbmc_results(
    *,
    runs_root: str | Path,
    grid_dir: str | Path,
    output_root: str | Path,
    candidate_set: str = "pca30_k100",
    condition: str = "incomplete_reference",
    methods: tuple | None = None,
    skip_incomplete: bool = False,
) -> PbmcArtifactPaths:
    """Generate the PBMC stimulated-state manuscript-results bundle.

    Writes split-summary, global detection, within-cell-type detection,
    rescue, and shared-label-transfer tables, plus markdown tables,
    a summary report, and a manifest.
    """
    if methods is None:
        methods = tuple(PRIMARY_SCORE_BY_METHOD)

    descriptors = discover_run_descriptors(grid_dir)
    if not descriptors:
        raise ResultsGridError(f"No run descriptors found in {grid_dir}")

    completed_descriptors, failures = _partition_main_grid_runs(
        runs_root=Path(runs_root),
        descriptors=descriptors,
        candidate_set=candidate_set,
        conditions=(condition, "full_reference_control"),
    )
    if failures and not skip_incomplete:
        raise ArtifactBatchError(failures)
    descriptors_to_use = completed_descriptors
    if not descriptors_to_use:
        raise ResultsGridError(
            "No completed runs available for manuscript-results generation"
        )

    output = Path(output_root)
    tables_root = output / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    # --- Split summary (Issue 06) ---
    split_by_run = build_pbmc_split_summary_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        grid_dir=Path(grid_dir),
    )

    # --- Global detection (Issue 07, reused from grid_common) ---
    global_det_by_run = build_detection_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        candidate_set=candidate_set,
        condition=condition,
        methods=methods,
    )
    global_det_summary = summarize_run_metrics(
        global_det_by_run,
        group_columns=("held_out_label", "method", "method_group", "primary_score"),
        value_columns=(
            "auroc",
            "auprc",
            "auprc_baseline",
            "absent_abstention_rate",
            "shared_false_abstention_rate",
            "median_absent",
            "median_shared",
            "absent_minus_shared_median",
        ),
        include_overall=True,
    )

    # --- Within-cell-type detection (Issue 07, new) ---
    within_det_by_run = build_pbmc_within_celltype_detection_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        candidate_set=candidate_set,
        condition=condition,
        methods=methods,
    )
    within_det_summary: pd.DataFrame
    if not within_det_by_run.empty:
        within_det_summary = summarize_run_metrics(
            within_det_by_run,
            group_columns=(
                "held_out_label",
                "method",
                "method_group",
                "primary_score",
            ),
            value_columns=(
                "auroc",
                "auprc",
                "auprc_baseline",
                "absent_abstention_rate",
                "median_positives",
                "median_negatives",
            ),
            include_overall=True,
        )
    else:
        within_det_summary = pd.DataFrame()

    # --- Rescue (Issue 08) ---
    rescue_by_run = build_pbmc_rescue_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        candidate_set=candidate_set,
        methods=methods,
        grid_dir=Path(grid_dir),
    )
    rescue_summary: pd.DataFrame
    if not rescue_by_run.empty:
        rescue_summary = summarize_run_metrics(
            rescue_by_run,
            group_columns=(
                "held_out_label",
                "method",
                "method_group",
                "primary_score",
                "cell_group",
            ),
            value_columns=(
                "mean_score_incomplete",
                "mean_score_full",
                "mean_score_drop",
                "median_score_drop",
                "n_cells",
            ),
            include_overall=True,
        )
    else:
        rescue_summary = pd.DataFrame()

    # --- Shared label transfer (Issue 09, PBMC condition-specific shared states) ---
    label_transfer_methods = tuple(
        m for m in methods if m in LABEL_TRANSFER_METHODS
    )
    shared_by_run = build_pbmc_shared_label_transfer_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        candidate_set=candidate_set,
        condition=condition,
        methods=label_transfer_methods,
    )
    shared_summary = summarize_run_metrics(
        shared_by_run,
        group_columns=("held_out_label", "method", "method_group"),
        value_columns=(
            "forced_accuracy",
            "forced_macro_f1",
            "post_abstention_accuracy",
            "post_abstention_macro_f1",
            "coverage",
            "shared_false_abstention_rate",
        ),
        include_overall=True,
    )

    # --- Build paths ---
    paths = PbmcArtifactPaths(
        output_root=output,
        split_summary_by_run=tables_root / "split_summary_by_run.csv",
        split_summary_md=tables_root / "split_summary.md",
        global_detection_by_run=tables_root / "global_detection_by_run.csv",
        global_detection_summary=tables_root / "global_detection_summary.csv",
        within_celltype_detection_by_run=tables_root
        / "within_celltype_detection_by_run.csv",
        within_celltype_detection_summary=tables_root
        / "within_celltype_detection_summary.csv",
        rescue_by_run=tables_root / "rescue_by_run.csv",
        rescue_summary=tables_root / "rescue_summary.csv",
        shared_label_transfer_by_run=tables_root
        / "shared_label_transfer_by_run.csv",
        shared_label_transfer_summary=tables_root
        / "shared_label_transfer_summary.csv",
        detection_table_1=tables_root / "detection_table_1.md",
        detection_table_2=tables_root / "detection_table_2.md",
        rescue_table=tables_root / "rescue_table.md",
        shared_label_transfer_table=tables_root / "shared_label_transfer_table.md",
        summary_md=output / "summary.md",
        manifest=output / "manifest.yaml",
    )

    # --- Write CSVs ---
    split_by_run.to_csv(paths.split_summary_by_run, index=False)
    global_det_by_run.to_csv(paths.global_detection_by_run, index=False)
    global_det_summary.to_csv(paths.global_detection_summary, index=False)
    within_det_by_run.to_csv(paths.within_celltype_detection_by_run, index=False)
    within_det_summary.to_csv(paths.within_celltype_detection_summary, index=False)
    rescue_by_run.to_csv(paths.rescue_by_run, index=False)
    rescue_summary.to_csv(paths.rescue_summary, index=False)
    shared_by_run.to_csv(paths.shared_label_transfer_by_run, index=False)
    shared_summary.to_csv(paths.shared_label_transfer_summary, index=False)

    # --- Write markdown tables ---
    paths.detection_table_1.write_text(
        _render_global_detection_table(global_det_summary), encoding="utf-8"
    )
    paths.detection_table_2.write_text(
        _render_within_celltype_table(within_det_summary), encoding="utf-8"
    )
    paths.rescue_table.write_text(
        _render_rescue_table(rescue_summary), encoding="utf-8"
    )
    paths.shared_label_transfer_table.write_text(
        _render_shared_label_table(shared_summary), encoding="utf-8"
    )
    paths.split_summary_md.write_text(
        _render_split_summary_md(split_by_run), encoding="utf-8"
    )
    paths.summary_md.write_text(
        _render_pbmc_summary(
            split_by_run,
            global_det_summary,
            within_det_summary,
            rescue_summary,
            shared_summary,
            skipped_failures=tuple(failures),
        ),
        encoding="utf-8",
    )

    # --- Write manifest ---
    artifacts = {
        "split_summary_by_run": str(paths.split_summary_by_run),
        "split_summary_md": str(paths.split_summary_md),
        "global_detection_by_run": str(paths.global_detection_by_run),
        "global_detection_summary": str(paths.global_detection_summary),
        "within_celltype_detection_by_run": str(
            paths.within_celltype_detection_by_run
        ),
        "within_celltype_detection_summary": str(
            paths.within_celltype_detection_summary
        ),
        "rescue_by_run": str(paths.rescue_by_run),
        "rescue_summary": str(paths.rescue_summary),
        "shared_label_transfer_by_run": str(paths.shared_label_transfer_by_run),
        "shared_label_transfer_summary": str(paths.shared_label_transfer_summary),
        "detection_table_1": str(paths.detection_table_1),
        "detection_table_2": str(paths.detection_table_2),
        "rescue_table": str(paths.rescue_table),
        "shared_label_transfer_table": str(paths.shared_label_transfer_table),
        "summary_md": str(paths.summary_md),
    }
    write_manifest(
        paths.manifest,
        Manifest(
            stage=STAGE,
            artifacts=artifacts,
            metadata={
                "experiment": "pbmc_stimulated_state",
                "benchmark_type": "condition_specific_held_out_state",
                "runs_root": str(runs_root),
                "grid_dir": str(grid_dir),
                "condition": condition,
                "candidate_set": candidate_set,
                "skip_incomplete": skip_incomplete,
                "n_runs_expected": len(descriptors),
                "n_runs_completed": len(descriptors_to_use),
                "skipped_runs": [
                    {
                        "run_id": f.run_id,
                        "artifact_kind": f.artifact_kind,
                        "state": f.state,
                        "condition": f.condition,
                        "path": str(f.path),
                    }
                    for f in failures
                ],
                "primary_score_by_method": PRIMARY_SCORE_BY_METHOD,
                "method_group_by_method": METHOD_GROUP_BY_METHOD,
                "variability": "descriptive_seed_level_mean_std_sem",
                "rescue_interpretation": (
                    "Positive score drop means the weak-correspondence score "
                    "decreased when the missing stimulated state was restored "
                    "to the target domain (full-reference control). "
                    "This is the expected direction for informative scores."
                ),
                "reviewer_caveats": [
                    "Cell-level rescue comparisons are descriptive, "
                    "not formal population-inference tests.",
                    "Weak target-domain support is a diagnostic for "
                    "correspondence quality, not a claim of biological novelty.",
                ],
            },
        ),
    )
    return paths


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def _render_split_summary_md(split_by_run: pd.DataFrame) -> str:
    """Render the split summary as a Markdown table."""
    lines = [
        "# PBMC Split Summary",
        "",
        "Per-run dataset and split characteristics.",
        "",
    ]
    if split_by_run.empty:
        lines.append("No completed runs.")
        return "\n".join(lines)

    display_cols = [
        "run_id",
        "held_out_label",
        "seed",
        "n_source_cells",
        "n_full_reference_target_cells",
        "n_incomplete_reference_target_cells",
        "n_positives",
        "n_same_celltype_controls",
        "n_other_stimulated",
        "broad_lineage",
        "positive_fraction",
    ]
    available = [c for c in display_cols if c in split_by_run.columns]
    table_frame = split_by_run[available]
    lines.extend(_markdown_table(table_frame))
    return "\n".join(lines)


def _render_global_detection_table(detection_summary: pd.DataFrame) -> str:
    """Render global detection table (Main Table 1 style)."""
    lines = [
        "# Global Detection: Missing Stimulated-State Detection",
        "",
        "Positives: held-out stimulated source cells. Negatives: all other source cells.",
        "",
        "Values are mean ± std across donor-split seeds.",
        "",
    ]
    if detection_summary.empty:
        lines.append("No detection data.")
        return "\n".join(lines)

    held_out_labels = sorted(
        set(detection_summary["held_out_label"]) - {"overall"}
    )
    metric_header = [MAIN_TABLE_1_COLUMN_HEADERS.get(q, q) for q in MAIN_TABLE_1_QUANTITIES]
    header = ["Method"] + metric_header
    sep = ["---"] * len(header)

    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in TABLE_ROW_ORDER:
            if method not in set(detection_summary["method"]):
                continue
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            row = [display]
            for qty in MAIN_TABLE_1_QUANTITIES:
                cell = _lookup_summary(detection_summary, method, lbl, qty)
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Overall
    lines.append("## Mean across all held-out labels")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for method in TABLE_ROW_ORDER:
        if method not in set(detection_summary["method"]):
            continue
        display = DISPLAY_NAME_BY_METHOD.get(method, method)
        row = [display]
        for qty in MAIN_TABLE_1_QUANTITIES:
            cell = _lookup_summary(detection_summary, method, "overall", qty)
            row.append(_format_table_value(cell, qty))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "_Scores: UOT/CoRe-OT methods use u; balanced OT uses label uncertainty; NN uses distance; prior-only uses prior_risk._", ""])
    return "\n".join(lines)


def _render_within_celltype_table(within_summary: pd.DataFrame) -> str:
    """Render within-cell-type detection table."""
    lines = [
        "# Within-Cell-Type Detection: Held-Out Stimulated vs Same-Cell-Type Controls",
        "",
        "Restricted to source cells of the held-out cell type.",
        "Positives: held-out stimulated cells. Negatives: same-cell-type control cells.",
        "",
        "Values are mean ± std across donor-split seeds.",
        "",
    ]
    if within_summary.empty:
        lines.append("No within-cell-type detection data (insufficient cells per run).")
        return "\n".join(lines)

    held_out_labels = sorted(
        set(within_summary["held_out_label"]) - {"overall"}
    )
    quantities = ("auroc", "auprc", "auprc_baseline", "absent_abstention_rate")
    q_headers = {q: q.upper().replace("_", " ") for q in quantities}
    header = ["Method"] + [q_headers[q] for q in quantities]
    sep = ["---"] * len(header)

    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in TABLE_ROW_ORDER:
            if method not in set(within_summary["method"]):
                continue
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            row = [display]
            for qty in quantities:
                cell = _lookup_summary(within_summary, method, lbl, qty)
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Overall
    if "overall" in set(within_summary["held_out_label"]):
        lines.append("## Mean across all held-out labels")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in TABLE_ROW_ORDER:
            if method not in set(within_summary["method"]):
                continue
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            row = [display]
            for qty in quantities:
                cell = _lookup_summary(within_summary, method, "overall", qty)
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "_Within-cell-type detection is a primary endpoint for the PBMC stimulated-state experiment._", ""])
    return "\n".join(lines)


def _render_rescue_table(rescue_summary: pd.DataFrame) -> str:
    """Render full-reference rescue table."""
    lines = [
        "# Full-Reference Rescue: Score Changes When Stimulated State Is Restored",
        "",
        "Compares weak-correspondence scores between incomplete-reference "
        "and full-reference control conditions.",
        "Positive score drop = score decreased when state was restored "
        "(expected direction for informative detection).",
        "",
        "Values are mean ± std across donor-split seeds.",
        "",
        "**Important**: Cell-level comparisons are descriptive, "
        "not formal population-inference tests.",
        "",
    ]
    if rescue_summary.empty:
        lines.append("No rescue data.")
        return "\n".join(lines)

    for group in RESCUE_CELL_GROUPS:
        group_display = RESCUE_GROUP_DISPLAY.get(group, group)
        group_data = rescue_summary.loc[
            rescue_summary["cell_group"] == group
        ]
        if group_data.empty:
            continue
        lines.append(f"## {group_display}")
        lines.append("")
        header = ["Method", "Mean Drop", "Median Drop", "N Cells"]
        sep = ["---"] * 4
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in TABLE_ROW_ORDER:
            if method not in set(group_data["method"]):
                continue
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            drop_cell = _lookup_summary(group_data, method, "overall", "mean_score_drop")
            med_drop_cell = _lookup_summary(group_data, method, "overall", "median_score_drop")
            n_cell = _lookup_summary(group_data, method, "overall", "n_cells")
            row = [
                display,
                _format_table_value(drop_cell, "auroc"),
                _format_table_value(med_drop_cell, "auroc"),
                str(int(float(n_cell["mean"]))) if n_cell is not None and not isinstance(n_cell, float) else (
                    f"{n_cell['mean']:.0f}" if n_cell is not None and hasattr(n_cell, '__getitem__') and not (isinstance(n_cell.get('mean', float('nan')), float) and math.isnan(float(n_cell.get('mean', 0)))) else "—"
                ),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.extend(["", "_Score drop = incomplete-reference score − full-reference score. Positive = expected rescue._", ""])
    return "\n".join(lines)


def _render_shared_label_table(shared_summary: pd.DataFrame) -> str:
    """Render shared-cell label transfer table."""
    lines = [
        "# Shared-Cell Label Transfer",
        "",
        "Restricted to shared source cells (held-out positives excluded).",
        "Prior-only excluded from label-transfer tables.",
        "",
        "Values are mean ± std across donor-split seeds, "
        "aggregated over all held-out labels.",
        "",
    ]
    if shared_summary.empty:
        lines.append("No label transfer data.")
        return "\n".join(lines)

    overall = shared_summary.loc[
        shared_summary["held_out_label"] == "overall"
    ]
    methods = [m for m in TABLE_ROW_ORDER if m != "prior_only" and m in set(overall["method"])]
    header = ["Method"] + [MAIN_TABLE_2_COLUMN_HEADERS.get(q, q) for q in MAIN_TABLE_2_QUANTITIES]
    sep = ["---"] * len(header)

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for method in methods:
        display = DISPLAY_NAME_BY_METHOD.get(method, method)
        row = [display]
        for qty in MAIN_TABLE_2_QUANTITIES:
            cell = _lookup_summary(shared_summary, method, "overall", qty)
            row.append(_format_table_value(cell, qty))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "_Coverage reported alongside post-abstention accuracy. Held-out positives excluded from shared-cell metrics._", ""])
    return "\n".join(lines)


def _render_pbmc_summary(
    split_by_run: pd.DataFrame,
    detection_summary: pd.DataFrame,
    within_summary: pd.DataFrame,
    rescue_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    *,
    skipped_failures: tuple = (),
) -> str:
    """Render the top-level PBMC manuscript-results summary."""
    lines = [
        "# PBMC Stimulated-State Experiment — Manuscript Results",
        "",
        "## Benchmark Definition",
        "",
        "Condition-specific held-out state: for each cell type, "
        "IFN-β stimulated cells are removed from the reference domain "
        "(incomplete reference), while the same cell type's control (unstimulated) "
        "cells remain. The full-reference control retains all states.",
        "",
        "**Primary endpoint**: within-cell-type detection — "
        "held-out stimulated cells vs same-cell-type controls.",
        "",
        "## Table Index",
        "",
        "- [Split Summary](tables/split_summary.md)",
        "- [Global Detection](tables/detection_table_1.md)",
        "- [Within-Cell-Type Detection](tables/detection_table_2.md)",
        "- [Full-Reference Rescue](tables/rescue_table.md)",
        "- [Shared-Cell Label Transfer](tables/shared_label_transfer_table.md)",
        "",
        "## Dataset Summary",
        "",
    ]
    if not split_by_run.empty:
        n_runs = len(split_by_run)
        n_types = split_by_run["held_out_label"].nunique()
        lines.append(f"- {n_runs} completed runs across {n_types} held-out cell types")
        total_pos = int(split_by_run["n_positives"].sum())
        total_src = int(split_by_run["n_source_cells"].sum())
        lines.append(f"- {total_pos} total held-out stimulated positives across {total_src} source cells")
    lines.append("")

    if skipped_failures:
        lines.extend(
            [
                "## Skipped Runs",
                "",
                "Skip-incomplete mode excluded the following runs because "
                "required artifacts were missing:",
                "",
            ]
        )
        for f in skipped_failures:
            lines.append(f"- `{f.run_id}`: {f.artifact_kind} ({f.state})")
        lines.append("")

    lines.extend(
        [
            "## Reviewer-Facing Caveats",
            "",
            "- Weak target-domain support is a diagnostic for correspondence quality, "
            "not a claim of biological novelty or perturbation discovery.",
            "- Cell-level rescue comparisons are descriptive summaries, "
            "not formal population-inference tests.",
            "- The condition-specific held-out state tests whether CoRe-OT "
            "can detect condition-driven weak correspondence, "
            "not whether IFN-β response is predictable from reference data.",
            "- All scores are derived from model-visible artifacts; "
            "condition labels and IFN scores were never model inputs.",
            "- Variability summaries are descriptive (mean ± std) across "
            "donor-split seeds, not inferential population estimates.",
        ]
    )
    return "\n".join(lines)


def _lookup_summary(
    summary: pd.DataFrame, method: str, held_out_label: str, quantity: str
) -> object:
    """Look up a single (method, held_out_label, quantity) row from a summary DataFrame."""
    mask = (
        (summary["method"] == method)
        & (summary["held_out_label"] == held_out_label)
        & (summary["quantity"] == quantity)
    )
    matches = summary.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]
