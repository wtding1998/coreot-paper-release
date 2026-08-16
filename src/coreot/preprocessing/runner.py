from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.config.load import load_yaml
from coreot.data.schemas import (
    BROAD_ANCHOR_PRIORS_COLUMNS,
    INITIAL_PRIORS_COLUMNS,
    MODEL_VISIBLE_CELLS_COLUMNS,
    TARGET_LABELS_COLUMNS,
)
from coreot.data.validation import validate_required_columns, validate_stage_can_read
from coreot.data.model_visible import read_condition_adata
from coreot.preprocessing.expression import preprocess_expression


STAGE = "model-visible-derivation"


class ModelVisibleDerivationError(ValueError):
    """Raised when model-visible derivation cannot construct valid artifacts."""


@dataclass(frozen=True)
class ModelVisibleDerivationResult:
    derived_root: Path
    conditions: tuple[str, ...]


def run_model_visible_derivation(config_path: str | Path) -> ModelVisibleDerivationResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    output_root = _required_path(config, ("outputs", "root"))
    conditions = tuple(config.get("conditions", ()))
    if not conditions:
        raise ModelVisibleDerivationError("model-visible-derivation requires conditions")

    run_root = output_root / run_id
    derived_root = run_root / "derived"
    preprocessing_config = _required_mapping(config, ("preprocessing",))
    priors_config = _required_mapping(config, ("priors",))
    prior_profile = _required_str(priors_config, ("profile_name",))
    mass_mode = str(priors_config.get("empirical_mass", "uniform"))

    for condition in conditions:
        _derive_condition(
            run_root, derived_root, condition, preprocessing_config, prior_profile, mass_mode
        )

    return ModelVisibleDerivationResult(derived_root=derived_root, conditions=conditions)


def _derive_condition(
    run_root: Path,
    derived_root: Path,
    condition: str,
    preprocessing_config: dict[str, Any],
    prior_profile: str,
    mass_mode: str,
) -> None:
    model_visible = run_root / "benchmark" / condition / "model_visible"
    counts_path = model_visible / "counts.h5ad"
    cells_path = model_visible / "cells.csv"
    target_labels_path = model_visible / "target_labels.csv"
    broad_anchor_priors_path = model_visible / "broad_anchor_priors.csv"
    initial_priors_path = model_visible / "initial_priors.csv"
    for path in (counts_path, cells_path, target_labels_path, broad_anchor_priors_path, initial_priors_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required model-visible artifact does not exist: {path}")

    cells = pd.read_csv(cells_path)
    target_labels = pd.read_csv(target_labels_path)
    broad_anchor_priors = pd.read_csv(broad_anchor_priors_path)
    initial_priors = pd.read_csv(initial_priors_path)
    validate_required_columns(cells, MODEL_VISIBLE_CELLS_COLUMNS, str(cells_path))
    validate_required_columns(target_labels, TARGET_LABELS_COLUMNS, str(target_labels_path))
    validate_required_columns(
        broad_anchor_priors, BROAD_ANCHOR_PRIORS_COLUMNS, str(broad_anchor_priors_path)
    )
    validate_required_columns(initial_priors, INITIAL_PRIORS_COLUMNS, str(initial_priors_path))

    adata = read_condition_adata(counts_path, cells)

    preprocessed = preprocess_expression(adata.X, preprocessing_config)
    genes = pd.DataFrame(
        {
            "gene_id": adata.var_names.astype(str)[preprocessed.selected_gene_indices],
            "gene_name": adata.var_names.astype(str)[preprocessed.selected_gene_indices],
            "selected": True,
        }
    )

    condition_root = derived_root / condition
    condition_root.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(condition_root / "processed_expression.npz", preprocessed.matrix)
    genes.to_csv(condition_root / "processed_genes.csv", index=False)

    profile_root = condition_root / "prior_profiles" / prior_profile
    profile_root.mkdir(parents=True, exist_ok=True)
    source_priors = _build_source_priors(cells, initial_priors, broad_anchor_priors, mass_mode)
    target_priors = _build_target_priors(cells, target_labels, mass_mode)
    source_priors.to_csv(profile_root / "source_priors.csv", index=False)
    target_priors.to_csv(profile_root / "target_priors.csv", index=False)
    write_manifest(
        profile_root / "prior_manifest.yaml",
        Manifest(
            stage=STAGE,
            artifacts={
                "source_priors": str(profile_root / "source_priors.csv"),
                "target_priors": str(profile_root / "target_priors.csv"),
            },
            metadata={
                "condition": condition,
                "profile_name": prior_profile,
                "prior_mode": "neutral_model_visible_copy",
            },
        ),
    )

    write_manifest(
        condition_root / "derivation_manifest.yaml",
        Manifest(
            stage=STAGE,
            artifacts={
                "processed_expression": str(condition_root / "processed_expression.npz"),
                "processed_genes": str(condition_root / "processed_genes.csv"),
                "prior_profile": str(profile_root),
            },
            metadata={
                "condition": condition,
                "n_cells": int(preprocessed.matrix.shape[0]),
                "n_genes": int(preprocessed.matrix.shape[1]),
                "preprocessing": preprocessing_config,
            },
        ),
    )


def _build_source_priors(
    cells: pd.DataFrame,
    initial_priors: pd.DataFrame,
    broad_anchor_priors: pd.DataFrame,
    mass_mode: str,
) -> pd.DataFrame:
    source_cells = cells.loc[cells["domain"] == "query", ["cell_id"]].copy()
    merged = source_cells.merge(initial_priors, on="cell_id", how="left", validate="one_to_one")
    merged = merged.merge(
        broad_anchor_priors, on="cell_id", how="left", validate="one_to_one"
    )
    if merged["rho"].isna().any() or merged["prior_risk"].isna().any():
        raise ModelVisibleDerivationError("initial_priors.csv must contain every source/query cell")
    if merged["broad_anchor_class"].isna().any() or merged["anchor_confidence"].isna().any():
        raise ModelVisibleDerivationError(
            "broad_anchor_priors.csv must contain every source/query cell"
        )
    output = pd.DataFrame(
        {
            "cell_id": merged["cell_id"].astype(str),
            "rho": merged["rho"].astype(float),
            "rho_source": merged["prior_source"].astype(str),
            "rho_recipe": merged["prior_source"].astype(str),
            "prior_risk": merged["prior_risk"].astype(float),
            "pmax_reference_classifier": 1.0,
            "anchor_class_pred": merged["broad_anchor_class"].astype(str),
            "anchor_confidence": merged["anchor_confidence"].astype(float),
        }
    )
    _add_empirical_mass(output, cells.loc[cells["domain"] == "query"], mass_mode)
    for column in broad_anchor_priors.columns:
        if column.startswith("anchor_probability::"):
            output[column] = merged[column].astype(float)
    return output


def _build_target_priors(
    cells: pd.DataFrame, target_labels: pd.DataFrame, mass_mode: str
) -> pd.DataFrame:
    target_cells = cells.loc[cells["domain"] == "reference", ["cell_id", "donor_id"]].copy()
    merged = target_cells.merge(target_labels, on="cell_id", how="left", validate="one_to_one")
    if merged[["target_label", "broad_label"]].isna().any().any():
        raise ModelVisibleDerivationError("target_labels.csv must contain every target cell")
    output = pd.DataFrame(
        {
            "cell_id": merged["cell_id"].astype(str),
            "target_label_visible": merged["target_label"].astype(str),
            "broad_anchor_class": merged["broad_label"].astype(str),
            "rho_target": 1.0,
        }
    )
    _add_empirical_mass(output, merged, mass_mode)
    return output


def _add_empirical_mass(output: pd.DataFrame, cells: pd.DataFrame, mass_mode: str) -> None:
    if mass_mode == "uniform":
        return
    if mass_mode != "donor_balanced":
        raise ModelVisibleDerivationError(
            "priors.empirical_mass must be 'uniform' or 'donor_balanced'"
        )
    output["empirical_mass"] = _donor_balanced_mass(cells)


def _donor_balanced_mass(cells: pd.DataFrame) -> pd.Series:
    if cells.empty:
        raise ModelVisibleDerivationError("cannot construct masses for an empty domain")
    donor = cells["donor_id"].astype(str)
    donor_counts = donor.value_counts()
    mass = donor.map(lambda value: 1.0 / (len(donor_counts) * donor_counts[value]))
    return mass.reset_index(drop=True).astype(float)


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise ModelVisibleDerivationError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_mapping(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value = _required_value(config, path)
    if not isinstance(value, dict):
        dotted = ".".join(path)
        raise ModelVisibleDerivationError(f"Expected mapping config value: {dotted}")
    return value


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise ModelVisibleDerivationError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
