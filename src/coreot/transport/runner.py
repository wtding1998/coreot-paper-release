from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.config.load import load_yaml
from coreot.data.schemas import (
    CELL_TRANSPORT_SCORES_COLUMNS,
    EMBEDDING_CELLS_COLUMNS,
    PRIOR_ONLY_SCORES_COLUMNS,
    SPARSE_COUPLING_COLUMNS,
    SOURCE_PRIORS_COLUMNS,
    TARGET_PRIORS_COLUMNS,
)
from coreot.data.validation import validate_required_columns, validate_stage_can_read
from coreot.preprocessing.provider_reliability import (
    ProviderReliabilityError,
    construct_provider_reliability,
    write_provider_reliability_artifacts,
)
from coreot.transport.label_transfer import coupling_label_probabilities, one_hot_label_probabilities
from coreot.transport.sinkhorn import (
    solve_dense_balanced_sinkhorn,
    solve_sparse_balanced_sinkhorn,
    solve_sparse_unbalanced_sinkhorn,
)


STAGE = "transport"
IMPLEMENTED_METHODS = frozenset(
    {
        "nn",
        "prior_only",
        "balanced_ot",
        "uniform_uot",
        "coreot_constant_tau",
        "coreot_full",
        "coreot_match_only",
    }
)


class TransportRunnerError(ValueError):
    """Raised when baseline transport artifacts cannot be constructed."""


@dataclass(frozen=True)
class TransportResult:
    transport_root: Path
    conditions: tuple[str, ...]
    candidate_sets: tuple[str, ...]
    methods: tuple[str, ...]


def run_transport(config_path: str | Path) -> TransportResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    output_root = _required_path(config, ("outputs", "root"))
    conditions = tuple(config.get("conditions", ()))
    candidate_sets = tuple(config.get("candidate_sets", ()))
    methods = tuple(_method_name(method) for method in config.get("methods", ()))
    if not conditions:
        raise TransportRunnerError("transport requires conditions")
    if not candidate_sets:
        raise TransportRunnerError("transport requires candidate_sets")
    if not methods:
        raise TransportRunnerError("transport requires methods")

    runnable_methods = tuple(method for method in methods if method in IMPLEMENTED_METHODS)
    if not runnable_methods:
        raise TransportRunnerError("transport config contains no implemented methods")

    run_root = output_root / run_id
    transport_root = run_root / "transport"
    rho_config = config.get("rho")
    provider_priors: dict[str, pd.DataFrame] = {}
    candidate_names: list[str] = []
    for candidate_set in candidate_sets:
        if not isinstance(candidate_set, dict):
            raise TransportRunnerError("transport candidate_sets must be mappings")
        candidate_name = _required_str(candidate_set, ("name",))
        provider = _required_str(candidate_set, ("provider",))
        prior_profile = _required_str(candidate_set, ("prior_profile",))
        if rho_config is not None and provider not in provider_priors:
            if not isinstance(rho_config, dict):
                raise TransportRunnerError("rho must be a mapping when configured")
            provider_priors[provider] = _construct_provider_rho_for_transport(
                run_root=run_root,
                provider=provider,
                rho_config=rho_config,
            )
        candidate_names.append(candidate_name)
        for condition in conditions:
            _run_candidate_set(
                run_root=run_root,
                transport_root=transport_root,
                condition=condition,
                candidate_name=candidate_name,
                provider=provider,
                prior_profile=prior_profile,
                methods=runnable_methods,
                method_configs=config.get("methods", ()),
                provider_rho=provider_priors.get(provider),
            )

    return TransportResult(
        transport_root=transport_root,
        conditions=conditions,
        candidate_sets=tuple(candidate_names),
        methods=runnable_methods,
    )


def _construct_provider_rho_for_transport(
    *,
    run_root: Path,
    provider: str,
    rho_config: dict[str, Any],
) -> pd.DataFrame:
    if rho_config.get("mode") != "calibrate_provider_score":
        raise TransportRunnerError(
            "canonical provider construction requires rho.mode: calibrate_provider_score"
        )
    condition = "incomplete_reference"
    cells_path = run_root / "benchmark" / condition / "model_visible" / "cells.csv"
    labels_path = run_root / "benchmark" / condition / "model_visible" / "target_labels.csv"
    embedding_path = run_root / "embeddings" / condition / provider / "embedding.npy"
    embedding_cells_path = (
        run_root / "embeddings" / condition / provider / "embedding_cells.csv"
    )
    for path in (cells_path, labels_path, embedding_path, embedding_cells_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required provider-reliability input does not exist: {path}")
    cells = pd.read_csv(cells_path)
    labels = pd.read_csv(labels_path)
    embedding_cells = pd.read_csv(embedding_cells_path)
    embedding = np.load(embedding_path)
    cell_ids = cells["cell_id"].astype(str).reset_index(drop=True)
    if cell_ids.tolist() != embedding_cells["cell_id"].astype(str).tolist():
        raise TransportRunnerError("provider embedding and model-visible cell order differ")
    reference = cells["domain"].astype(str).eq("reference").to_numpy()
    if labels["cell_id"].astype(str).duplicated().any():
        raise TransportRunnerError("target labels contain duplicate cell IDs")
    label_by_id = pd.Series(
        labels["broad_label"].astype(str).to_numpy(),
        index=labels["cell_id"].astype(str),
    )
    reference_labels = cell_ids[reference].map(label_by_id)
    if reference_labels.isna().any():
        raise TransportRunnerError("reference broad labels are incomplete for rho calibration")
    calibration = rho_config.get("calibration", {})
    if not isinstance(calibration, dict):
        raise TransportRunnerError("rho.calibration must be a mapping")
    try:
        result = construct_provider_reliability(
            cell_ids=cell_ids,
            embedding=embedding,
            is_reference=reference,
            reference_labels=reference_labels,
            reference_donor_ids=cells.loc[reference, "donor_id"],
            clip_eps=_float_config(rho_config, "numerical_clip_eps", 1.0e-6),
            classifier_c=_float_config(rho_config, "classifier_c", 1.0),
            min_correct_isotonic=int(calibration.get("min_correct_isotonic", 25)),
            min_error_isotonic=int(calibration.get("min_error_isotonic", 25)),
            min_distinct_isotonic=int(calibration.get("min_distinct_score_isotonic", 10)),
        )
    except ProviderReliabilityError as exc:
        raise TransportRunnerError(str(exc)) from exc
    output_root = run_root / "transport" / "provider_reliability" / provider
    write_provider_reliability_artifacts(output_root, result)
    source = result.source.copy()
    source["coreot_rho_cell_id_hash"] = result.metadata["cell_id_rho_sha256"]
    return source


def _apply_provider_rho(
    source_priors: pd.DataFrame,
    provider_rho: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    source_ids = source_priors["cell_id"].astype(str)
    provider_ids = provider_rho["cell_id"].astype(str)
    if set(source_ids) != set(provider_ids) or len(source_ids) != len(provider_ids):
        raise TransportRunnerError(
            f"{condition} source cells do not match the fixed provider-reliability artifact"
        )
    columns = [
        "cell_id",
        "coreot_provider_score_raw",
        "coreot_rho",
        "coreot_rho_cell_id_hash",
    ]
    merged = source_priors.drop(columns=["rho", "prior_risk"], errors="ignore").merge(
        provider_rho.loc[:, columns],
        on="cell_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["coreot_rho"].isna().any():
        raise TransportRunnerError(f"{condition} provider rho alignment is incomplete")
    merged["rho"] = merged.pop("coreot_rho").astype(float)
    merged["prior_risk"] = 1.0 - merged["rho"]
    merged["rho_source"] = "calibrated_provider_reliability"
    merged["rho_recipe"] = "donor_grouped_oof_pca_broad_lineage_reliability"
    merged["pmax_reference_classifier"] = merged["coreot_provider_score_raw"].astype(float)
    return merged


def _run_candidate_set(
    *,
    run_root: Path,
    transport_root: Path,
    condition: str,
    candidate_name: str,
    provider: str,
    prior_profile: str,
    methods: tuple[str, ...],
    method_configs: tuple[object, ...],
    provider_rho: pd.DataFrame | None,
) -> None:
    candidate_edges_path = run_root / "candidates" / condition / candidate_name / "candidate_edges.parquet"
    source_priors_path = (
        run_root / "derived" / condition / "prior_profiles" / prior_profile / "source_priors.csv"
    )
    target_priors_path = (
        run_root / "derived" / condition / "prior_profiles" / prior_profile / "target_priors.csv"
    )
    for path in (candidate_edges_path, source_priors_path, target_priors_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required transport input does not exist: {path}")

    candidate_edges = pd.read_parquet(candidate_edges_path)
    source_priors = pd.read_csv(source_priors_path)
    target_priors = pd.read_csv(target_priors_path)
    validate_required_columns(source_priors, SOURCE_PRIORS_COLUMNS, str(source_priors_path))
    validate_required_columns(target_priors, TARGET_PRIORS_COLUMNS, str(target_priors_path))
    if provider_rho is not None:
        source_priors = _apply_provider_rho(source_priors, provider_rho, condition)

    for method in methods:
        method_root = transport_root / condition / candidate_name / method
        method_root.mkdir(parents=True, exist_ok=True)
        _write_method_params(method_root / "method_params.yaml", method, method_configs)
        if method == "nn":
            _run_nn(method_root, condition, candidate_edges, source_priors, target_priors)
        elif method == "prior_only":
            _run_prior_only(method_root, condition, source_priors)
        elif method == "uniform_uot":
            _run_uniform_uot(
                method_root,
                condition,
                candidate_edges,
                source_priors,
                target_priors,
                _method_config(method, method_configs),
            )
        elif method == "coreot_constant_tau":
            _run_coreot_constant_tau(
                method_root,
                condition,
                candidate_edges,
                source_priors,
                target_priors,
                _method_config(method, method_configs),
            )
        elif method == "balanced_ot":
            _run_balanced_ot(
                method_root,
                condition,
                candidate_edges,
                source_priors,
                target_priors,
                _method_config(method, method_configs),
                run_root,
                provider,
            )
        elif method == "coreot_full":
            _run_coreot_full(
                method_root,
                condition,
                candidate_edges,
                source_priors,
                target_priors,
                _method_config(method, method_configs),
            )
        elif method == "coreot_match_only":
            _run_coreot_match_only(
                method_root,
                condition,
                candidate_edges,
                source_priors,
                target_priors,
                _method_config(method, method_configs),
            )
        else:
            raise TransportRunnerError(f"Unsupported implemented method: {method}")


def _run_nn(
    method_root: Path,
    condition: str,
    candidate_edges: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
) -> None:
    nearest = (
        candidate_edges.sort_values(["source_cell_id", "scaled_distance", "distance", "target_cell_id"])
        .groupby("source_cell_id", as_index=False)
        .first()
    )
    scores = source_priors[["cell_id", "rho"]].merge(
        nearest,
        left_on="cell_id",
        right_on="source_cell_id",
        how="left",
        validate="one_to_one",
    )
    if scores["target_cell_id"].isna().any():
        missing = scores.loc[scores["target_cell_id"].isna(), "cell_id"].tolist()
        raise TransportRunnerError(f"nn transport missing candidate edge(s) for source cells: {missing}")

    target_labels = target_priors[["cell_id", "target_label_visible"]].rename(
        columns={"cell_id": "target_cell_id"}
    )
    scores = scores.merge(target_labels, on="target_cell_id", how="left", validate="many_to_one")
    if scores["target_label_visible"].isna().any():
        raise TransportRunnerError("nn transport target labels are incomplete")

    n_source = len(scores)
    mass = 1.0 / n_source if n_source else 0.0
    output = pd.DataFrame(
        {
            "cell_id": scores["cell_id"].astype(str),
            "condition_id": condition,
            "method": "nn",
            "domain": "query",
            "a": mass,
            "a_hat": mass,
            "u": 0.0,
            "e": 0.0,
            "rho": scores["rho"].astype(float),
            "tau_source": np.nan,
            "max_label_probability": 1.0,
            "label_entropy": 0.0,
            "forced_label": scores["target_label_visible"].astype(str),
            "hub_exposure": 0.0,
            "nn_distance": scores["distance"].astype(float),
        }
    )
    output = output.loc[:, CELL_TRANSPORT_SCORES_COLUMNS]
    output.to_parquet(method_root / "cell_transport_scores.parquet", index=False)
    label_order = sorted(target_priors["target_label_visible"].astype(str).unique())
    np.savez_compressed(
        method_root / "label_probabilities.npz",
        **one_hot_label_probabilities(output["cell_id"], output["forced_label"], label_order),
    )
    _write_manifest(method_root, "nn")


def _run_prior_only(method_root: Path, condition: str, source_priors: pd.DataFrame) -> None:
    output = pd.DataFrame(
        {
            "cell_id": source_priors["cell_id"].astype(str),
            "condition_id": condition,
            "method": "prior_only",
            "domain": "query",
            "rho": source_priors["rho"].astype(float),
            "prior_risk": source_priors["prior_risk"].astype(float),
        }
    )
    output = output.loc[:, PRIOR_ONLY_SCORES_COLUMNS]
    output.to_parquet(method_root / "cell_transport_scores.parquet", index=False)
    _write_manifest(method_root, "prior_only")


def _run_uniform_uot(
    method_root: Path,
    condition: str,
    candidate_edges: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    method_config: dict[str, Any],
) -> None:
    _run_sparse_uot_method(
        method_root=method_root,
        condition=condition,
        candidate_edges=candidate_edges,
        source_priors=source_priors,
        target_priors=target_priors,
        method_config=method_config,
        method_name="uniform_uot",
        use_anchor_cost=False,
    )


def _run_coreot_constant_tau(
    method_root: Path,
    condition: str,
    candidate_edges: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    method_config: dict[str, Any],
) -> None:
    _run_sparse_uot_method(
        method_root=method_root,
        condition=condition,
        candidate_edges=candidate_edges,
        source_priors=source_priors,
        target_priors=target_priors,
        method_config=method_config,
        method_name="coreot_constant_tau",
        use_anchor_cost=True,
    )


def _run_coreot_full(
    method_root: Path,
    condition: str,
    candidate_edges: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    method_config: dict[str, Any],
) -> None:
    _run_sparse_uot_method(
        method_root=method_root,
        condition=condition,
        candidate_edges=candidate_edges,
        source_priors=source_priors,
        target_priors=target_priors,
        method_config=method_config,
        method_name="coreot_full",
        use_anchor_cost=True,
    )


def _run_coreot_match_only(
    method_root: Path,
    condition: str,
    candidate_edges: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    method_config: dict[str, Any],
) -> None:
    if _float_config(method_config, "alpha", 0.0) != 0.0:
        raise TransportRunnerError("coreot_match_only requires alpha = 0")
    _run_sparse_uot_method(
        method_root=method_root,
        condition=condition,
        candidate_edges=candidate_edges,
        source_priors=source_priors,
        target_priors=target_priors,
        method_config=method_config,
        method_name="coreot_match_only",
        use_anchor_cost=True,
    )


def _run_balanced_ot(
    method_root: Path,
    condition: str,
    candidate_edges: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    method_config: dict[str, Any],
    run_root: Path,
    provider: str,
) -> None:
    source = source_priors.reset_index(drop=True).copy()
    target = target_priors.reset_index(drop=True).copy()
    source_mass = _empirical_mass(source)
    target_mass = _empirical_mass(target)
    epsilon = _float_config(method_config, "epsilon", 0.05)
    eta = _float_config(method_config, "eta", 1.0e-12)
    alpha = _float_config(method_config, "alpha", 0.0)

    if str(method_config.get("support", "dense")) == "sparse":
        result, label_arrays = _run_sparse_balanced_ot(
            candidate_edges=candidate_edges,
            source=source,
            target=target,
            source_mass=source_mass,
            target_mass=target_mass,
            epsilon=epsilon,
            alpha=alpha,
            eta=eta,
            method_config=method_config,
        )
        support = "sparse"
    else:
        result, label_arrays = _run_dense_balanced_ot(
            run_root=run_root,
            condition=condition,
            provider=provider,
            source=source,
            target=target,
            source_mass=source_mass,
            target_mass=target_mass,
            epsilon=epsilon,
            alpha=alpha,
            method_config=method_config,
        )
        support = "dense"

    probabilities = label_arrays["probabilities"]
    labels = label_arrays["labels"].astype(str)
    max_probability = np.where(probabilities.sum(axis=1) > 0, probabilities.max(axis=1), np.nan)
    forced_label = np.where(
        probabilities.sum(axis=1) > 0,
        labels[np.argmax(probabilities, axis=1)],
        "",
    )
    label_entropy = _normalized_entropy(probabilities)
    output = pd.DataFrame(
        {
            "cell_id": source["cell_id"].astype(str),
            "condition_id": condition,
            "method": "balanced_ot",
            "domain": "query",
            "a": source_mass,
            "a_hat": result.source_marginal,
            "u": np.nan,
            "e": np.nan,
            "rho": source["rho"].astype(float),
            "tau_source": np.nan,
            "max_label_probability": max_probability,
            "label_entropy": label_entropy,
            "forced_label": forced_label,
            "hub_exposure": np.nan,
            "nn_distance": np.nan,
        }
    )
    output = output.loc[:, CELL_TRANSPORT_SCORES_COLUMNS]
    output.to_parquet(method_root / "cell_transport_scores.parquet", index=False)
    np.savez_compressed(method_root / "label_probabilities.npz", **label_arrays)
    _write_manifest(
        method_root,
        "balanced_ot",
        metadata={
            "epsilon": epsilon,
            "alpha": alpha,
            "support": support,
            "n_iter": result.n_iter,
            "converged": result.converged,
            "total_transported_mass": float(result.source_marginal.sum()),
            "max_source_marginal_error": float(np.max(np.abs(result.source_marginal - source_mass))),
            "max_target_marginal_error": float(np.max(np.abs(result.target_marginal - target_mass))),
        },
    )


def _run_sparse_balanced_ot(
    *,
    candidate_edges: pd.DataFrame,
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    epsilon: float,
    alpha: float,
    eta: float,
    method_config: dict[str, Any],
) -> tuple[object, dict[str, np.ndarray]]:
    edges = _indexed_candidate_edges(candidate_edges, source, target, "balanced_ot")
    _validate_sparse_support(edges, len(source), len(target), "balanced_ot")
    cost = edges["scaled_distance"].to_numpy(dtype=float) + alpha * _anchor_compatibility_cost(
        edges, source, target
    )
    result = solve_sparse_balanced_sinkhorn(
        source_index=edges["source_index"].to_numpy(dtype=int),
        target_index=edges["target_index"].to_numpy(dtype=int),
        cost=cost,
        source_mass=source_mass,
        target_mass=target_mass,
        epsilon=epsilon,
        max_iter=int(method_config.get("max_iter", 2000)),
        tol=_float_config(method_config, "tol", 1.0e-6),
        numerical_floor=_float_config(method_config, "numerical_floor", 1.0e-300),
    )
    label_arrays = coupling_label_probabilities(
        cell_ids=source["cell_id"],
        source_index=result.source_index,
        target_index=result.target_index,
        coupling=result.coupling,
        source_marginal=result.source_marginal,
        target_labels=target["target_label_visible"],
        eta=eta,
    )
    return result, label_arrays


def _run_dense_balanced_ot(
    *,
    run_root: Path,
    condition: str,
    provider: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    epsilon: float,
    alpha: float,
    method_config: dict[str, Any],
) -> tuple[object, dict[str, np.ndarray]]:
    embedding, embedding_cells = _read_embedding(run_root, condition, provider)
    cell_to_row = {
        cell_id: int(row_index)
        for cell_id, row_index in zip(
            embedding_cells["cell_id"].astype(str), embedding_cells["row_index"], strict=True
        )
    }
    source_rows = source["cell_id"].astype(str).map(cell_to_row)
    target_rows = target["cell_id"].astype(str).map(cell_to_row)
    if source_rows.isna().any() or target_rows.isna().any():
        raise TransportRunnerError("balanced_ot dense support requires embeddings for every source and target cell")
    source_embedding = embedding[source_rows.to_numpy(dtype=int)]
    target_embedding = embedding[target_rows.to_numpy(dtype=int)]
    distances = _pairwise_euclidean(source_embedding, target_embedding)
    scaled = distances / (float(np.median(distances)) + _float_config(method_config, "distance_delta", 1.0e-8))
    anchor_cost = _dense_anchor_compatibility_cost(source, target)
    cost = scaled + alpha * anchor_cost
    result = solve_dense_balanced_sinkhorn(
        cost=cost,
        source_mass=source_mass,
        target_mass=target_mass,
        epsilon=epsilon,
        max_iter=int(method_config.get("max_iter", 2000)),
        tol=_float_config(method_config, "tol", 1.0e-6),
        numerical_floor=_float_config(method_config, "numerical_floor", 1.0e-300),
    )
    label_arrays = _dense_coupling_label_probabilities(
        cell_ids=source["cell_id"],
        coupling=result.coupling,
        target_labels=target["target_label_visible"],
    )
    return result, label_arrays


def _read_embedding(run_root: Path, condition: str, provider: str) -> tuple[np.ndarray, pd.DataFrame]:
    embedding_root = run_root / "embeddings" / condition / provider
    embedding_path = embedding_root / "embedding.npy"
    cells_path = embedding_root / "embedding_cells.csv"
    for path in (embedding_path, cells_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required balanced OT embedding input does not exist: {path}")
    embedding = np.load(embedding_path)
    embedding_cells = pd.read_csv(cells_path)
    validate_required_columns(embedding_cells, EMBEDDING_CELLS_COLUMNS, str(cells_path))
    if embedding.shape[0] != len(embedding_cells):
        raise TransportRunnerError("embedding rows must match embedding_cells.csv rows")
    return embedding, embedding_cells


def _pairwise_euclidean(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.sum(source * source, axis=1)[:, None]
    target_norm = np.sum(target * target, axis=1)[None, :]
    squared = np.maximum(source_norm + target_norm - 2.0 * source @ target.T, 0.0)
    return np.sqrt(squared)


def _dense_anchor_compatibility_cost(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    probability_columns = _anchor_probability_columns(source)
    if probability_columns:
        probabilities = source.loc[:, probability_columns].to_numpy(dtype=float)
        class_to_column = {
            column.removeprefix("anchor_probability::"): index
            for index, column in enumerate(probability_columns)
        }
        target_anchor = target["broad_anchor_class"].astype(str).to_numpy()
        compatibility = np.zeros((len(source), len(target)), dtype=float)
        for target_index, anchor_class in enumerate(target_anchor):
            column_index = class_to_column.get(anchor_class)
            if column_index is not None:
                compatibility[:, target_index] = probabilities[:, column_index]
        confidence = source["anchor_confidence"].astype(float).to_numpy()[:, None]
        return confidence * (1.0 - compatibility)
    source_anchor = source["anchor_class_pred"].astype(str).to_numpy()
    source_confidence = source["anchor_confidence"].astype(float).to_numpy()
    target_anchor = target["broad_anchor_class"].astype(str).to_numpy()
    mismatch = source_anchor[:, None] != target_anchor[None, :]
    return source_confidence[:, None] * mismatch.astype(float)


def _dense_coupling_label_probabilities(
    *, cell_ids: pd.Series, coupling: np.ndarray, target_labels: pd.Series
) -> dict[str, np.ndarray]:
    label_order = sorted(target_labels.astype(str).unique())
    probabilities = np.zeros((coupling.shape[0], len(label_order)), dtype=float)
    source_marginal = np.maximum(coupling.sum(axis=1), 1.0e-300)
    for label_index, label in enumerate(label_order):
        target_mask = target_labels.astype(str).to_numpy() == label
        probabilities[:, label_index] = coupling[:, target_mask].sum(axis=1) / source_marginal
    return {
        "cell_ids": cell_ids.astype(str).to_numpy(),
        "labels": np.asarray(label_order, dtype=object),
        "probabilities": probabilities,
    }


def _run_sparse_uot_method(
    *,
    method_root: Path,
    condition: str,
    candidate_edges: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    method_config: dict[str, Any],
    method_name: str,
    use_anchor_cost: bool,
) -> None:
    source = source_priors.reset_index(drop=True).copy()
    target = target_priors.reset_index(drop=True).copy()
    edges = _indexed_candidate_edges(candidate_edges, source, target, method_name)
    _validate_sparse_support(edges, len(source), len(target), method_name)

    n_source = len(source)
    n_target = len(target)
    source_mass = _empirical_mass(source)
    target_mass = _empirical_mass(target)
    epsilon = _float_config(method_config, "epsilon", 0.05)
    if epsilon <= 0.0:
        raise TransportRunnerError(f"{method_name} requires epsilon > 0")
    if method_name in {"coreot_full", "coreot_match_only"}:
        tau_min = _float_config(method_config, "tau_min", 0.05)
        tau_max = _float_config(method_config, "tau_max", 1.0)
        _validate_tau_range(tau_min, tau_max, method_name)
        rho = source["rho"].astype(float).to_numpy()
        if not np.all((0.0 <= rho) & (rho <= 1.0)):
            raise TransportRunnerError(
                f"{method_name} requires source rho values in [0, 1]"
            )
        tau_source = tau_min + (tau_max - tau_min) * rho
    else:
        tau_value = method_config.get("tau_source", 1.0)
        if tau_value == "matched_coreot_mean":
            tau_min = _float_config(method_config, "matched_tau_min", 0.05)
            tau_max = _float_config(method_config, "matched_tau_max", 1.0)
            _validate_tau_range(tau_min, tau_max, method_name)
            rho = source["rho"].astype(float).to_numpy()
            if not np.all((0.0 <= rho) & (rho <= 1.0)):
                raise TransportRunnerError(
                    f"{method_name} requires source rho values in [0, 1]"
                )
            tau_value = float(np.sum(source_mass * (tau_min + (tau_max - tau_min) * rho)))
        if not isinstance(tau_value, int | float):
            raise TransportRunnerError(
                f"{method_name} requires numeric tau_source or 'matched_coreot_mean'"
            )
        tau_source = np.full(n_source, float(tau_value))
    tau_target = np.full(n_target, _float_config(method_config, "tau_target", 1.0))
    if np.any(tau_target <= 0.0):
        raise TransportRunnerError(f"{method_name} requires tau_target > 0")
    eta = _float_config(method_config, "eta", 1.0e-12)
    cost = edges["scaled_distance"].to_numpy(dtype=float)
    alpha = _float_config(method_config, "alpha", 0.0)
    if alpha < 0.0:
        raise TransportRunnerError(f"{method_name} requires alpha >= 0")
    if use_anchor_cost:
        cost = cost + alpha * _anchor_compatibility_cost(edges, source, target)
    result = solve_sparse_unbalanced_sinkhorn(
        source_index=edges["source_index"].to_numpy(dtype=int),
        target_index=edges["target_index"].to_numpy(dtype=int),
        cost=cost,
        source_mass=source_mass,
        target_mass=target_mass,
        epsilon=epsilon,
        tau_source=tau_source,
        tau_target=tau_target,
        max_iter=int(method_config.get("max_iter", 2000)),
        tol=_float_config(method_config, "tol", 1.0e-6),
        numerical_floor=_float_config(method_config, "numerical_floor", 1.0e-300),
    )

    target_expansion = np.maximum(result.target_marginal - target_mass, 0.0) / (target_mass + eta)
    hub_numerator = np.bincount(
        result.source_index,
        weights=result.coupling * target_expansion[result.target_index],
        minlength=n_source,
    )
    hub_exposure = hub_numerator / np.maximum(result.source_marginal, eta)
    label_arrays = coupling_label_probabilities(
        cell_ids=source["cell_id"],
        source_index=result.source_index,
        target_index=result.target_index,
        coupling=result.coupling,
        source_marginal=result.source_marginal,
        target_labels=target["target_label_visible"],
        eta=eta,
    )
    probabilities = label_arrays["probabilities"]
    labels = label_arrays["labels"].astype(str)
    max_probability = np.where(probabilities.sum(axis=1) > 0, probabilities.max(axis=1), np.nan)
    forced_label = np.where(
        probabilities.sum(axis=1) > 0,
        labels[np.argmax(probabilities, axis=1)],
        "",
    )
    label_entropy = _normalized_entropy(probabilities)

    output = pd.DataFrame(
        {
            "cell_id": source["cell_id"].astype(str),
            "condition_id": condition,
            "method": method_name,
            "domain": "query",
            "a": source_mass,
            "a_hat": result.source_marginal,
            "u": np.maximum(source_mass - result.source_marginal, 0.0) / (source_mass + eta),
            "e": np.maximum(result.source_marginal - source_mass, 0.0) / (source_mass + eta),
            "rho": source["rho"].astype(float),
            "tau_source": tau_source,
            "max_label_probability": max_probability,
            "label_entropy": label_entropy,
            "forced_label": forced_label,
            "hub_exposure": hub_exposure,
            "nn_distance": np.nan,
        }
    )
    output = output.loc[:, CELL_TRANSPORT_SCORES_COLUMNS]
    output.to_parquet(method_root / "cell_transport_scores.parquet", index=False)
    _write_sparse_coupling(method_root, result, source, target)
    np.savez_compressed(method_root / "label_probabilities.npz", **label_arrays)
    metadata = {
        "epsilon": epsilon,
        "tau_source_min": float(np.min(tau_source)),
        "tau_source_max": float(np.max(tau_source)),
        "tau_target": float(tau_target[0]),
        "alpha": alpha,
        "n_iter": result.n_iter,
        "converged": result.converged,
        "total_transported_mass": float(result.coupling.sum()),
    }
    if "coreot_rho_cell_id_hash" in source.columns:
        hashes = source["coreot_rho_cell_id_hash"].astype(str).unique()
        if len(hashes) != 1:
            raise TransportRunnerError("source rows contain inconsistent rho cell-ID hashes")
        metadata["rho_cell_id_hash"] = hashes[0]
    if np.allclose(tau_source, tau_source[0]):
        metadata["tau_source"] = float(tau_source[0])
    _write_manifest(
        method_root,
        method_name,
        metadata=metadata,
    )


def _write_method_params(path: Path, method: str, method_configs: tuple[object, ...]) -> None:
    path.write_text(yaml.safe_dump(_method_config(method, method_configs), sort_keys=True), encoding="utf-8")


def _write_manifest(
    method_root: Path, method: str, metadata: dict[str, object] | None = None
) -> None:
    artifacts = {
        "method_params": str(method_root / "method_params.yaml"),
        "cell_transport_scores": str(method_root / "cell_transport_scores.parquet"),
    }
    if method in {
        "nn",
        "balanced_ot",
        "uniform_uot",
        "coreot_constant_tau",
        "coreot_full",
        "coreot_match_only",
    }:
        artifacts["label_probabilities"] = str(method_root / "label_probabilities.npz")
    if method in {
        "uniform_uot",
        "coreot_constant_tau",
        "coreot_full",
        "coreot_match_only",
    }:
        artifacts["sparse_coupling"] = str(method_root / "sparse_coupling.parquet")
    write_manifest(
        method_root / "transport_manifest.yaml",
        Manifest(stage=STAGE, artifacts=artifacts, metadata={"method": method} | (metadata or {})),
    )


def _write_sparse_coupling(
    method_root: Path,
    result: object,
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> None:
    coupling = pd.DataFrame(
        {
            "source_cell_id": source["cell_id"].astype(str).to_numpy()[result.source_index],
            "target_cell_id": target["cell_id"].astype(str).to_numpy()[result.target_index],
            "source_index": result.source_index.astype(int),
            "target_index": result.target_index.astype(int),
            "coupling": result.coupling.astype(float),
        }
    )
    coupling.loc[:, SPARSE_COUPLING_COLUMNS].to_parquet(
        method_root / "sparse_coupling.parquet", index=False
    )


def _method_config(method: str, method_configs: tuple[object, ...]) -> dict[str, Any]:
    config = next(
        (
            candidate
            for candidate in method_configs
            if isinstance(candidate, dict) and candidate.get("name") == method
        ),
        {"name": method},
    )
    return dict(config)


def _indexed_candidate_edges(
    candidate_edges: pd.DataFrame, source: pd.DataFrame, target: pd.DataFrame, method_name: str
) -> pd.DataFrame:
    source_lookup = {cell_id: index for index, cell_id in enumerate(source["cell_id"].astype(str))}
    target_lookup = {cell_id: index for index, cell_id in enumerate(target["cell_id"].astype(str))}
    edges = candidate_edges.copy()
    edges["source_index"] = edges["source_cell_id"].astype(str).map(source_lookup)
    edges["target_index"] = edges["target_cell_id"].astype(str).map(target_lookup)
    edges = edges.dropna(subset=["source_index", "target_index"]).copy()
    if edges.empty:
        raise TransportRunnerError(f"{method_name} has no candidate edges after source/target filtering")
    edges["source_index"] = edges["source_index"].astype(int)
    edges["target_index"] = edges["target_index"].astype(int)
    return edges


def _validate_sparse_support(
    edges: pd.DataFrame, n_source: int, n_target: int, method_name: str
) -> None:
    source_ids = set(edges["source_index"].astype(int))
    target_ids = set(edges["target_index"].astype(int))
    if source_ids != set(range(n_source)):
        missing = sorted(set(range(n_source)) - source_ids)
        raise TransportRunnerError(f"{method_name} missing candidate support for source rows: {missing}")
    if target_ids != set(range(n_target)):
        missing = sorted(set(range(n_target)) - target_ids)
        raise TransportRunnerError(f"{method_name} missing candidate support for target rows: {missing}")


def _anchor_compatibility_cost(
    edges: pd.DataFrame, source: pd.DataFrame, target: pd.DataFrame
) -> np.ndarray:
    probability_columns = _anchor_probability_columns(source)
    if probability_columns:
        probabilities = source.loc[:, probability_columns].to_numpy(dtype=float)
        class_to_column = {
            column.removeprefix("anchor_probability::"): index
            for index, column in enumerate(probability_columns)
        }
        source_index = edges["source_index"].to_numpy(dtype=int)
        target_index = edges["target_index"].to_numpy(dtype=int)
        target_anchor = target["broad_anchor_class"].astype(str).to_numpy()[target_index]
        compatibility = np.zeros(len(edges), dtype=float)
        for edge_index, anchor_class in enumerate(target_anchor):
            column_index = class_to_column.get(anchor_class)
            if column_index is not None:
                compatibility[edge_index] = probabilities[source_index[edge_index], column_index]
        confidence = source["anchor_confidence"].astype(float).to_numpy()[source_index]
        return confidence * (1.0 - compatibility)
    source_anchor = source["anchor_class_pred"].astype(str).to_numpy()
    source_confidence = source["anchor_confidence"].astype(float).to_numpy()
    target_anchor = target["broad_anchor_class"].astype(str).to_numpy()
    source_index = edges["source_index"].to_numpy(dtype=int)
    target_index = edges["target_index"].to_numpy(dtype=int)
    mismatch = source_anchor[source_index] != target_anchor[target_index]
    return source_confidence[source_index] * mismatch.astype(float)


def _anchor_probability_columns(source: pd.DataFrame) -> list[str]:
    columns = sorted(
        column for column in source.columns if column.startswith("anchor_probability::")
    )
    if columns:
        probabilities = source.loc[:, columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
            raise TransportRunnerError("soft anchor probabilities must be finite and nonnegative")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6):
            raise TransportRunnerError("soft anchor probability rows must sum to one")
    return columns


def _empirical_mass(priors: pd.DataFrame) -> np.ndarray:
    if "empirical_mass" not in priors.columns:
        return np.full(len(priors), 1.0 / len(priors))
    mass = pd.to_numeric(priors["empirical_mass"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise TransportRunnerError("empirical_mass must contain finite positive values")
    total = float(mass.sum())
    if total <= 0.0:
        raise TransportRunnerError("empirical_mass must have positive total mass")
    return mass / total


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] <= 1:
        return np.zeros(probabilities.shape[0], dtype=float)
    clipped = np.where(probabilities > 0, probabilities, 1.0)
    entropy = -(probabilities * np.log(clipped)).sum(axis=1) / np.log(probabilities.shape[1])
    return np.where(probabilities.sum(axis=1) > 0, entropy, np.nan)


def _float_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float):
        raise TransportRunnerError(f"Expected numeric {key} for method {config.get('name')!r}")
    return float(value)


def _validate_tau_range(tau_min: float, tau_max: float, method_name: str) -> None:
    if tau_min <= 0.0:
        raise TransportRunnerError(f"{method_name} requires tau_min > 0")
    if tau_max < tau_min:
        raise TransportRunnerError(f"{method_name} requires tau_max >= tau_min")


def _method_name(method: object) -> str:
    if not isinstance(method, dict) or not isinstance(method.get("name"), str):
        raise TransportRunnerError("transport methods must be mappings with a name")
    return method["name"]


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise TransportRunnerError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise TransportRunnerError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
