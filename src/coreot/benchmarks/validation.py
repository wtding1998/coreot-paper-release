from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from coreot.benchmarks.conditions import (
    DEFAULT_CONDITIONS,
    FULL_REFERENCE_CONTROL,
    INCOMPLETE_REFERENCE,
)
from coreot.data.schemas import (
    BROAD_ANCHOR_PRIORS_COLUMNS,
    CONDITION_MANIFEST_COLUMNS,
    INITIAL_PRIORS_COLUMNS,
    MODEL_VISIBLE_CELLS_COLUMNS,
    MODEL_VISIBLE_FORBIDDEN_COLUMNS,
    QUERY_TRUTH_COLUMNS,
    SPLIT_MANIFEST_COLUMNS,
    STATE_PRESENCE_COLUMNS,
    TARGET_LABELS_COLUMNS,
)
from coreot.data.validation import validate_required_columns
from coreot.data.model_visible import read_condition_adata


class BenchmarkArtifactValidationError(ValueError):
    """Raised when benchmark-build artifacts violate the repository contract."""


def validate_missing_state_benchmark_artifacts(
    benchmark_root: str | Path,
    removed_state: str,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    *,
    removed_state_condition: str | None = None,
) -> None:
    root = Path(benchmark_root)
    split_manifest = _read_required_csv(root / "split_manifest.csv", SPLIT_MANIFEST_COLUMNS)
    condition_manifest = _read_required_csv(
        root / "condition_manifest.csv", CONDITION_MANIFEST_COLUMNS
    )
    _validate_condition_manifest(condition_manifest, removed_state, conditions)

    query_sets: dict[str, set[str]] = {}
    for condition in conditions:
        condition_root = root / condition
        model_visible = condition_root / "model_visible"
        evaluation_truth = condition_root / "evaluation_truth"
        _require_files(
            model_visible,
            (
                "counts.h5ad",
                "cells.csv",
                "genes.csv",
                "target_labels.csv",
                "broad_anchor_priors.csv",
                "initial_priors.csv",
                "dataset_config.yaml",
            ),
        )
        _require_files(
            evaluation_truth,
            ("query_truth.csv", "state_presence.csv", "benchmark_truth_manifest.yaml"),
        )

        cells = _read_required_csv(model_visible / "cells.csv", MODEL_VISIBLE_CELLS_COLUMNS)
        target_labels = _read_required_csv(
            model_visible / "target_labels.csv", TARGET_LABELS_COLUMNS
        )
        broad_anchor_priors = _read_required_csv(
            model_visible / "broad_anchor_priors.csv", BROAD_ANCHOR_PRIORS_COLUMNS
        )
        initial_priors = _read_required_csv(
            model_visible / "initial_priors.csv", INITIAL_PRIORS_COLUMNS
        )
        query_truth = _read_required_csv(evaluation_truth / "query_truth.csv", QUERY_TRUTH_COLUMNS)
        _read_required_csv(evaluation_truth / "state_presence.csv", STATE_PRESENCE_COLUMNS)

        _validate_model_visible_cells(cells, condition)
        _validate_model_visible_no_forbidden_columns(cells, "model_visible/cells.csv")
        _validate_counts_h5ad(model_visible / "counts.h5ad", cells)
        _validate_target_labels(
            target_labels,
            cells,
            condition,
            removed_state,
            removed_state_condition,
        )
        _validate_model_visible_priors(broad_anchor_priors, initial_priors, cells)
        _validate_query_truth(query_truth, cells, condition, removed_state)

        query_sets[condition] = set(cells.loc[cells["domain"] == "query", "cell_id"].astype(str))

    if query_sets[INCOMPLETE_REFERENCE] != query_sets[FULL_REFERENCE_CONTROL]:
        raise BenchmarkArtifactValidationError(
            "incomplete_reference and full_reference_control must share the same query cells"
        )
    split_query_ids = set(
        split_manifest.loc[split_manifest["split_domain"] == "query", "cell_id"].astype(str)
    )
    if query_sets[INCOMPLETE_REFERENCE] != split_query_ids:
        raise BenchmarkArtifactValidationError(
            "condition query cells must match split_manifest query cells"
        )


def _validate_condition_manifest(
    condition_manifest: pd.DataFrame, removed_state: str, conditions: tuple[str, ...]
) -> None:
    seen = tuple(condition_manifest["condition_id"].astype(str))
    if seen != conditions:
        raise BenchmarkArtifactValidationError(
            f"condition_manifest condition order must be {conditions}; got {seen}"
        )
    if set(condition_manifest["removed_state"].astype(str)) != {removed_state}:
        raise BenchmarkArtifactValidationError("condition_manifest removed_state is inconsistent")


def _validate_model_visible_cells(cells: pd.DataFrame, condition: str) -> None:
    if cells["cell_id"].astype(str).duplicated().any():
        raise BenchmarkArtifactValidationError(f"{condition} cells.csv has duplicate cell_id values")
    domains = set(cells["domain"].astype(str))
    if not domains <= {"query", "reference"}:
        raise BenchmarkArtifactValidationError(f"{condition} cells.csv has invalid domain values")
    if {"query", "reference"} - domains:
        raise BenchmarkArtifactValidationError(f"{condition} must contain query and reference cells")
    if set(cells["condition_id"].astype(str)) != {condition}:
        raise BenchmarkArtifactValidationError(f"{condition} cells.csv has inconsistent condition_id")


def _validate_counts_h5ad(path: Path, cells: pd.DataFrame) -> None:
    stored_counts = ad.read_h5ad(path, backed="r")
    try:
        _validate_model_visible_no_forbidden_columns(
            stored_counts.obs, "model_visible/counts.h5ad .obs"
        )
    finally:
        stored_counts.file.close()
    counts = read_condition_adata(path, cells)
    validate_required_columns(counts.obs, MODEL_VISIBLE_CELLS_COLUMNS, "model_visible/counts.h5ad .obs")
    _validate_model_visible_no_forbidden_columns(counts.obs, "model_visible/counts.h5ad .obs")
    counts_obs = counts.obs.loc[:, MODEL_VISIBLE_CELLS_COLUMNS].astype(str).reset_index(drop=True)
    cells_as_text = cells.loc[:, MODEL_VISIBLE_CELLS_COLUMNS].astype(str).reset_index(drop=True)
    if not counts_obs.equals(cells_as_text):
        raise BenchmarkArtifactValidationError(
            "model_visible/counts.h5ad .obs metadata must match cells.csv"
        )


def _validate_target_labels(
    target_labels: pd.DataFrame,
    cells: pd.DataFrame,
    condition: str,
    removed_state: str,
    removed_state_condition: str | None,
) -> None:
    reference_ids = set(cells.loc[cells["domain"] == "reference", "cell_id"].astype(str))
    target_ids = set(target_labels["cell_id"].astype(str))
    if not target_ids <= reference_ids:
        raise BenchmarkArtifactValidationError(
            f"{condition} target_labels.csv must contain only reference cells"
        )
    if target_ids != reference_ids:
        raise BenchmarkArtifactValidationError(
            f"{condition} target_labels.csv must label every reference cell"
        )
    if (
        condition == INCOMPLETE_REFERENCE
        and removed_state_condition is None
        and removed_state in set(target_labels["target_label"].astype(str))
    ):
        raise BenchmarkArtifactValidationError(
            "incomplete_reference target_labels.csv contains the removed state"
        )
    if condition == FULL_REFERENCE_CONTROL and removed_state not in set(
        target_labels["target_label"].astype(str)
    ):
        raise BenchmarkArtifactValidationError(
            "full_reference_control target_labels.csv must contain the removed state"
        )


def _validate_model_visible_priors(
    broad_anchor_priors: pd.DataFrame, initial_priors: pd.DataFrame, cells: pd.DataFrame
) -> None:
    cell_ids = set(cells["cell_id"].astype(str))
    for name, priors in (
        ("broad_anchor_priors.csv", broad_anchor_priors),
        ("initial_priors.csv", initial_priors),
    ):
        prior_ids = set(priors["cell_id"].astype(str))
        if prior_ids != cell_ids:
            raise BenchmarkArtifactValidationError(f"{name} must contain exactly cells.csv cell IDs")


def _validate_query_truth(
    query_truth: pd.DataFrame, cells: pd.DataFrame, condition: str, removed_state: str
) -> None:
    query_ids = set(cells.loc[cells["domain"] == "query", "cell_id"].astype(str))
    truth_ids = set(query_truth["cell_id"].astype(str))
    if truth_ids != query_ids:
        raise BenchmarkArtifactValidationError(
            f"{condition} query_truth.csv must contain exactly query cell IDs"
        )
    if set(query_truth["removed_state"].astype(str)) != {removed_state}:
        raise BenchmarkArtifactValidationError(f"{condition} query_truth.csv removed_state mismatch")
    if condition == INCOMPLETE_REFERENCE and not query_truth["is_absent_state"].any():
        raise BenchmarkArtifactValidationError(
            f"{condition} query_truth.csv must contain at least one absent-state query cell"
        )
    if condition == FULL_REFERENCE_CONTROL and query_truth["is_absent_state"].any():
        raise BenchmarkArtifactValidationError(
            f"{condition} query_truth.csv must not contain absent-state query cells"
        )
    if not query_truth["is_shared_state"].any():
        raise BenchmarkArtifactValidationError(
            f"{condition} query_truth.csv must contain at least one shared-state query cell"
        )


def _validate_model_visible_no_forbidden_columns(frame: pd.DataFrame, name: str) -> None:
    forbidden = [column for column in MODEL_VISIBLE_FORBIDDEN_COLUMNS if column in frame.columns]
    if forbidden:
        raise BenchmarkArtifactValidationError(
            f"{name} contains hidden-truth column(s): {', '.join(forbidden)}"
        )


def _read_required_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise BenchmarkArtifactValidationError(f"Required artifact does not exist: {path}")
    frame = pd.read_csv(path)
    validate_required_columns(frame, required_columns, str(path))
    return frame


def _require_files(root: Path, filenames: tuple[str, ...]) -> None:
    for filename in filenames:
        path = root / filename
        if not path.is_file():
            raise BenchmarkArtifactValidationError(f"Required artifact does not exist: {path}")
