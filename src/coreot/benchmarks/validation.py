from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from coreot.benchmarks.conditions import (
    DEFAULT_CONDITIONS,
    FULL_REFERENCE_CONTROL,
    INCOMPLETE_REFERENCE,
    SUPPORTED_CONDITION_CONTRACTS,
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
    if conditions not in SUPPORTED_CONDITION_CONTRACTS:
        raise BenchmarkArtifactValidationError(
            "benchmark artifact validation requires the paired primary conditions "
            f"{DEFAULT_CONDITIONS} or the incomplete-reference-only sensitivity "
            f"condition; got {conditions}"
        )
    root = Path(benchmark_root)
    split_manifest = _read_required_csv(root / "split_manifest.csv", SPLIT_MANIFEST_COLUMNS)
    _validate_split_manifest(split_manifest)
    condition_manifest = _read_required_csv(
        root / "condition_manifest.csv", CONDITION_MANIFEST_COLUMNS
    )
    _validate_condition_manifest(condition_manifest, removed_state, conditions)

    query_sets: dict[str, set[str]] = {}
    condition_cells: dict[str, pd.DataFrame] = {}
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

        condition_cells[condition] = cells
        query_sets[condition] = set(cells.loc[cells["domain"] == "query", "cell_id"].astype(str))

    if (
        FULL_REFERENCE_CONTROL in query_sets
        and query_sets[INCOMPLETE_REFERENCE] != query_sets[FULL_REFERENCE_CONTROL]
    ):
        raise BenchmarkArtifactValidationError(
            "incomplete_reference and full_reference_control must share the same query cells"
        )
    split_query_ids = _split_ids(split_manifest, "query")
    if query_sets[INCOMPLETE_REFERENCE] != split_query_ids:
        raise BenchmarkArtifactValidationError(
            "condition query cells must match split_manifest query cells"
        )
    _validate_condition_membership(split_manifest, condition_cells)


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


def _validate_split_manifest(split_manifest: pd.DataFrame) -> None:
    raw_cell_ids = split_manifest["cell_id"]
    cell_ids = raw_cell_ids.astype(str)
    if raw_cell_ids.isna().any() or (cell_ids.str.strip() == "").any():
        raise BenchmarkArtifactValidationError("split_manifest.csv cell_id values must be nonempty")
    if cell_ids.duplicated().any():
        raise BenchmarkArtifactValidationError("split_manifest.csv has duplicate cell_id values")
    domains = set(split_manifest["split_domain"].astype(str))
    allowed = {"query", "reference", "excluded"}
    if not domains <= allowed:
        raise BenchmarkArtifactValidationError(
            "split_manifest.csv has invalid split_domain values; expected query, reference, or excluded"
        )


def _split_ids(split_manifest: pd.DataFrame, domain: str) -> set[str]:
    return set(
        split_manifest.loc[split_manifest["split_domain"].astype(str) == domain, "cell_id"]
        .astype(str)
    )


def _validate_condition_membership(
    split_manifest: pd.DataFrame, condition_cells: dict[str, pd.DataFrame]
) -> None:
    """Require model-visible rows to agree with split domains.

    The full-reference control is expected to expose every post-subsampling
    query/reference row.  The incomplete-reference condition may omit a
    subset of split-reference rows by design, but it may not expose excluded
    rows or change a row's query/reference domain.
    """

    split_domains = split_manifest.set_index(split_manifest["cell_id"].astype(str))[
        "split_domain"
    ].astype(str)
    split_query_ids = _split_ids(split_manifest, "query")
    split_reference_ids = _split_ids(split_manifest, "reference")
    for condition, cells in condition_cells.items():
        cell_ids = cells["cell_id"].astype(str)
        observed_ids = set(cell_ids)
        unknown_ids = observed_ids - set(split_domains.index)
        if unknown_ids:
            raise BenchmarkArtifactValidationError(
                f"{condition} cells.csv contains IDs absent from split_manifest.csv"
            )
        expected_query_ids = set(cell_ids.loc[cells["domain"].astype(str) == "query"])
        expected_reference_ids = set(cell_ids.loc[cells["domain"].astype(str) == "reference"])
        if expected_query_ids != split_query_ids:
            raise BenchmarkArtifactValidationError(
                f"{condition} query cells must match split_manifest query cells"
            )
        if not expected_reference_ids <= split_reference_ids:
            raise BenchmarkArtifactValidationError(
                f"{condition} reference cells must match split_manifest reference cells"
            )
        expected_domains = split_domains.reindex(cell_ids)
        observed_domains = cells["domain"].astype(str)
        if expected_domains.isna().any() or not (
            expected_domains.to_numpy() == observed_domains.to_numpy()
        ).all():
            raise BenchmarkArtifactValidationError(
                f"{condition} model-visible domains disagree with split_manifest.csv"
            )

    if FULL_REFERENCE_CONTROL in condition_cells:
        full_reference_ids = set(
            condition_cells[FULL_REFERENCE_CONTROL].loc[
                condition_cells[FULL_REFERENCE_CONTROL]["domain"].astype(str) == "reference",
                "cell_id",
            ].astype(str)
        )
        if full_reference_ids != split_reference_ids:
            raise BenchmarkArtifactValidationError(
                "full_reference_control reference cells must match split_manifest reference cells"
            )

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
