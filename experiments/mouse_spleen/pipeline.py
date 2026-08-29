from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
import yaml

from coreot.artifacts.hashes import sha256_file
from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.candidates.runner import run_candidate_cost
from coreot.config.load import load_yaml
from coreot.evaluation.metrics import safe_auprc, safe_auroc, safe_median
from coreot.evaluation.runner import run_evaluation
from coreot.external_baselines.runner import run_external_baselines
from coreot.scoring.prior_adjusted import compute_prior_adjusted_deficit
from coreot.scoring.runner import run_scoring
from coreot.transport.runner import run_transport
from coreot.transport.runner import (
    _run_coreot_constant_tau,
    _run_coreot_full,
    _run_coreot_match_only,
)


class MouseSpleenConfigError(ValueError):
    """Raised when the mouse-spleen experiment configuration is invalid."""


RETAINED_SENSITIVITY_FIT_FILES = (
    "cell_transport_scores.parquet",
    "label_probabilities.npz",
    "method_params.yaml",
    "sparse_coupling.parquet",
    "transport_manifest.yaml",
)


@dataclass(frozen=True)
class StageResult:
    stage: str
    root: Path
    artifacts: dict[str, Path]
    manifest_path: Path


def run_stage(config_path: str | Path, stage: str) -> StageResult:
    config = load_yaml(config_path)
    if stage == "validate_data":
        return _validate_data(config)
    if stage == "prepare_embedding":
        return _prepare_embedding(config)
    if stage == "prepare_task_families":
        return _prepare_task_families(config)
    if stage == "run_controlled":
        return _run_controlled(config)
    if stage == "prepare_baselines":
        return _prepare_baselines(config)
    if stage == "run_internal_baselines":
        return _run_internal_baselines(config)
    if stage == "run_external_baselines":
        return _run_external_baselines(config)
    if stage == "run_baselines":
        _run_internal_baselines(config)
        return _run_external_baselines(config)
    if stage == "aggregate_baselines":
        return _aggregate_baselines(config)
    if stage == "run_natural_mismatch":
        return _run_natural_mismatch(config)
    if stage == "aggregate_natural_mismatch":
        return _aggregate_natural_mismatch(config)
    if stage == "make_natural_mismatch_figure":
        return _make_natural_mismatch_figure(config)
    if stage == "prepare_natural_baselines":
        return _prepare_natural_baselines(config)
    if stage == "run_natural_external_baselines":
        return _run_natural_external_baselines(config)
    if stage == "aggregate_natural_baselines":
        return _aggregate_natural_baselines(config)
    if stage == "run_natural_constant_tau_sensitivity":
        return _run_natural_constant_tau_sensitivity(config)
    if stage == "aggregate_natural_constant_tau_sensitivity":
        return _aggregate_natural_constant_tau_sensitivity(config)
    if stage == "aggregate_natural_constant_tau_target8_sensitivity":
        tau_target, max_iterations = _natural_target_penalty_settings(config)
        return _aggregate_natural_constant_tau_sensitivity(
            config,
            root_name="constant_tau_alpha_tau_target_8",
            tau_target_override=tau_target,
            max_iterations_override=max_iterations,
        )
    if stage == "run_natural_match_only_sensitivity":
        return _run_natural_match_only_sensitivity(config)
    if stage == "aggregate_natural_match_only_sensitivity":
        return _aggregate_natural_match_only_sensitivity(config)
    if stage == "aggregate_natural_match_only_target8_sensitivity":
        tau_target, max_iterations = _natural_target_penalty_settings(config)
        return _aggregate_natural_match_only_sensitivity(
            config,
            root_name="match_only_tau_range_tau_target_8",
            tau_target_override=tau_target,
            max_iterations_override=max_iterations,
        )
    if stage == "run_natural_full_tau_range_sensitivity":
        return _run_natural_full_tau_range_sensitivity(config)
    if stage == "aggregate_natural_full_tau_range_sensitivity":
        return _aggregate_natural_full_tau_range_sensitivity(config)
    if stage == "run_natural_component_ablation":
        from experiments.mouse_spleen.component_ablation import (
            run_component_ablation,
        )

        return run_component_ablation(config)
    if stage == "aggregate_natural_component_ablation":
        from experiments.mouse_spleen.component_ablation import (
            aggregate_component_ablation,
        )

        return aggregate_component_ablation(config)
    if stage == "aggregate_metrics":
        return _aggregate_metrics(config)
    if stage == "make_figures":
        return _make_figures(config)
    if stage == "all":
        _run_controlled(config)
        _run_external_baselines(config)
        _run_natural_mismatch(config)
        _aggregate_metrics(config)
        return _make_figures(config)
    raise MouseSpleenConfigError(f"Unsupported mouse-spleen stage: {stage}")


def _validate_data(config: dict[str, Any]) -> StageResult:
    data_config = _mapping(config, "data")
    labels_config = _mapping(config, "labels")
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    manifest_root = output_root / "manifest"
    manifest_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "rna": Path(_string(data_config, "rna_h5ad")),
        "atac_gene_activity": Path(_string(data_config, "atac_gene_activity_h5ad")),
        "atac_peaks": Path(_string(data_config, "atac_peaks_h5ad")),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Mouse-spleen {name} input does not exist: {path}")

    datasets = {name: ad.read_h5ad(path, backed="r") for name, path in paths.items()}
    try:
        label_key = str(data_config.get("label_key", "cell_type"))
        source_key = str(data_config.get("source_key", "source"))
        for name, adata in datasets.items():
            missing = [key for key in (label_key, source_key) if key not in adata.obs]
            if missing:
                raise MouseSpleenConfigError(
                    f"{name} .obs is missing required column(s): {', '.join(missing)}"
                )
            if not adata.obs_names.is_unique:
                raise MouseSpleenConfigError(f"{name} observation names must be unique")
        if list(datasets["atac_gene_activity"].obs_names.astype(str)) != list(
            datasets["atac_peaks"].obs_names.astype(str)
        ):
            raise MouseSpleenConfigError(
                "ATAC gene-activity and peak objects must have identical observation IDs and order"
            )
        _validate_expected_shapes(datasets, data_config)

        raw_labels = {
            "rna": datasets["rna"].obs[label_key].astype(str).str.strip(),
            "atac": datasets["atac_gene_activity"].obs[label_key].astype(str).str.strip(),
        }
        raw_to_canonical = _resolve_label_maps(raw_labels, labels_config)
        broad_by_fine = _broad_label_map(labels_config)
        composition = _composition(
            raw_labels,
            raw_to_canonical,
            broad_by_fine,
            labels_config,
        )

        composition_path = manifest_root / "cell_type_composition.csv"
        composition.to_csv(composition_path, index=False)
        resolved_path = manifest_root / "resolved_label_map.yaml"
        resolved_path.write_text(
            yaml.safe_dump(
                {
                    "raw_to_canonical": raw_to_canonical,
                    "canonical_to_broad": broad_by_fine,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        manifest_path = manifest_root / "dataset_manifest.yaml"
        metadata = {
            "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
            "sha256": {name: sha256_file(path) for name, path in paths.items()},
            "shapes": {name: list(map(int, adata.shape)) for name, adata in datasets.items()},
            "obs_keys": {name: list(map(str, adata.obs.columns)) for name, adata in datasets.items()},
            "obsm_keys": {name: list(map(str, adata.obsm.keys())) for name, adata in datasets.items()},
            "raw_labels": {
                modality: sorted(values.unique().tolist()) for modality, values in raw_labels.items()
            },
        }
        write_manifest(
            manifest_path,
            Manifest(
                stage="validate_data",
                artifacts={
                    "cell_type_composition": str(composition_path),
                    "resolved_label_map": str(resolved_path),
                },
                metadata=metadata,
            ),
        )
    finally:
        for adata in datasets.values():
            if adata.file is not None:
                adata.file.close()

    return StageResult(
        stage="validate_data",
        root=manifest_root,
        artifacts={
            "cell_type_composition": composition_path,
            "resolved_label_map": resolved_path,
        },
        manifest_path=manifest_path,
    )


def _prepare_embedding(config: dict[str, Any]) -> StageResult:
    validation = _validate_data(config)
    data_config = _mapping(config, "data")
    embedding_config = _mapping(config, "embedding")
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    embedding_root = output_root / "embedding"
    embedding_root.mkdir(parents=True, exist_ok=True)

    validation_payload = _read_manifest_payload(validation.manifest_path)
    input_sha256 = validation_payload["metadata"]["sha256"]
    cached_manifest = embedding_root / "embedding_manifest.yaml"
    cached_embedding = embedding_root / "embedding.npy"
    cached_cells = embedding_root / "embedding_cells.csv"
    if bool(_mapping(config, "experiment").get("resume", True)) and all(
        path.is_file() for path in (cached_manifest, cached_embedding, cached_cells)
    ):
        cached_payload = _read_manifest_payload(cached_manifest)
        cached_metadata = cached_payload.get("metadata", {})
        requested_mode = str(embedding_config.get("mode", "fixed_multimap"))
        dimension_matches = requested_mode != "fixed_multimap" or int(
            cached_metadata.get("dimension", -1)
        ) == int(_mapping(embedding_config, "multimap").get("n_components", 50))
        if (
            cached_metadata.get("input_sha256") == input_sha256
            and cached_metadata.get("mode") == requested_mode
            and dimension_matches
        ):
            return StageResult(
                stage="prepare_embedding",
                root=embedding_root,
                artifacts={"embedding": cached_embedding, "embedding_cells": cached_cells},
                manifest_path=cached_manifest,
            )

    rna_path = Path(_string(data_config, "rna_h5ad"))
    atac_genes_path = Path(_string(data_config, "atac_gene_activity_h5ad"))
    mode = str(embedding_config.get("mode", "fixed_multimap"))
    if mode == "precomputed":
        provider_path = Path(_string(embedding_config, "precomputed_path"))
        if not provider_path.is_file():
            raise FileNotFoundError(f"Precomputed provider does not exist: {provider_path}")
        arrays = np.load(provider_path, allow_pickle=False)
        required = {"rna_cell_ids", "atac_cell_ids", "rna_embedding", "atac_embedding"}
        missing = required - set(arrays.files)
        if missing:
            raise MouseSpleenConfigError(
                f"Precomputed provider is missing array(s): {', '.join(sorted(missing))}"
            )
        rna_ids = arrays["rna_cell_ids"].astype(str)
        atac_ids = arrays["atac_cell_ids"].astype(str)
        rna_embedding = np.asarray(arrays["rna_embedding"], dtype=float)
        atac_embedding = np.asarray(arrays["atac_embedding"], dtype=float)
        provider_metadata: dict[str, object] = {
            "mode": mode,
            "source_path": str(provider_path.resolve()),
            "source_sha256": sha256_file(provider_path),
        }
    elif mode == "precomputed_combined":
        provider_path = Path(_string(embedding_config, "precomputed_embedding_path"))
        provider_cells_path = Path(_string(embedding_config, "precomputed_cells_path"))
        for path in (provider_path, provider_cells_path):
            if not path.is_file():
                raise FileNotFoundError(f"Precomputed provider artifact does not exist: {path}")
        combined = np.load(provider_path, allow_pickle=False)
        provider_cells = pd.read_csv(provider_cells_path).sort_values("row_index")
        required_columns = {"row_index", "raw_cell_id", "modality"}
        missing_columns = required_columns - set(provider_cells.columns)
        if missing_columns:
            raise MouseSpleenConfigError(
                "Precomputed provider cells are missing column(s): "
                + ", ".join(sorted(missing_columns))
            )
        if provider_cells["row_index"].astype(int).tolist() != list(range(len(provider_cells))):
            raise MouseSpleenConfigError("Precomputed provider row indices must be contiguous")
        if combined.ndim != 2 or combined.shape[0] != len(provider_cells):
            raise MouseSpleenConfigError(
                "Precomputed combined embedding rows must match precomputed provider cells"
            )
        rna_mask = provider_cells["modality"].astype(str).eq("rna").to_numpy()
        atac_mask = provider_cells["modality"].astype(str).eq("atac").to_numpy()
        if not np.all(rna_mask | atac_mask):
            raise MouseSpleenConfigError("Precomputed provider modalities must be rna or atac")
        rna_ids = provider_cells.loc[rna_mask, "raw_cell_id"].astype(str).to_numpy()
        atac_ids = provider_cells.loc[atac_mask, "raw_cell_id"].astype(str).to_numpy()
        rna_embedding = np.asarray(combined[rna_mask], dtype=float)
        atac_embedding = np.asarray(combined[atac_mask], dtype=float)
        provider_metadata = {
            "mode": mode,
            "source_embedding_path": str(provider_path.resolve()),
            "source_embedding_sha256": sha256_file(provider_path),
            "source_cells_path": str(provider_cells_path.resolve()),
            "source_cells_sha256": sha256_file(provider_cells_path),
        }
    elif mode == "fixed_multimap":
        (
            rna_ids,
            atac_ids,
            rna_embedding,
            atac_embedding,
            provider_metadata,
        ) = _generate_multimap_embedding(config)
    else:
        raise MouseSpleenConfigError(
            "Unsupported embedding.mode "
            f"{mode!r}; expected 'fixed_multimap', 'precomputed', or 'precomputed_combined'"
        )

    expected_rna_ids = _obs_names(rna_path)
    expected_atac_ids = _obs_names(atac_genes_path)
    if rna_ids.tolist() != expected_rna_ids.tolist():
        raise MouseSpleenConfigError("Provider RNA cell IDs/order do not match rna_h5ad")
    if atac_ids.tolist() != expected_atac_ids.tolist():
        raise MouseSpleenConfigError(
            "Provider ATAC cell IDs/order do not match atac_gene_activity_h5ad"
        )
    if rna_embedding.ndim != 2 or atac_embedding.ndim != 2:
        raise MouseSpleenConfigError("Provider embeddings must be two-dimensional")
    if rna_embedding.shape[0] != len(rna_ids) or atac_embedding.shape[0] != len(atac_ids):
        raise MouseSpleenConfigError("Provider embedding rows must match provider cell IDs")
    if rna_embedding.shape[1] != atac_embedding.shape[1]:
        raise MouseSpleenConfigError("RNA and ATAC provider dimensions must match")
    embedding = np.vstack([rna_embedding, atac_embedding])
    if not np.isfinite(embedding).all():
        raise MouseSpleenConfigError("Provider embedding contains nonfinite values")

    embedding_path = embedding_root / "embedding.npy"
    np.save(embedding_path, embedding)
    cells_path = embedding_root / "embedding_cells.csv"
    cells = pd.DataFrame(
        {
            "row_index": np.arange(len(embedding)),
            "cell_id": [
                *(f"rna::{cell_id}" for cell_id in rna_ids),
                *(f"atac::{cell_id}" for cell_id in atac_ids),
            ],
            "raw_cell_id": [*rna_ids, *atac_ids],
            "modality": ["rna"] * len(rna_ids) + ["atac"] * len(atac_ids),
        }
    )
    cells.to_csv(cells_path, index=False)
    manifest_path = embedding_root / "embedding_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="prepare_embedding",
            artifacts={"embedding": str(embedding_path), "embedding_cells": str(cells_path)},
            metadata={
                **provider_metadata,
                "dimension": int(embedding.shape[1]),
                "n_rna": int(len(rna_ids)),
                "n_atac": int(len(atac_ids)),
                "frozen": bool(embedding_config.get("freeze_for_primary_paired_runs", True)),
                "embedding_sha256": sha256_file(embedding_path),
                "validation_manifest": str(validation.manifest_path),
                "input_sha256": input_sha256,
            },
        ),
    )
    return StageResult(
        stage="prepare_embedding",
        root=embedding_root,
        artifacts={"embedding": embedding_path, "embedding_cells": cells_path},
        manifest_path=manifest_path,
    )


def _prepare_task_families(config: dict[str, Any]) -> StageResult:
    embedding_result = _prepare_embedding(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    task_root = output_root / "task_families"
    task_root.mkdir(parents=True, exist_ok=True)
    labels_config = _mapping(config, "labels")
    experiments_config = _mapping(config, "experiments")
    controlled_config = _mapping(experiments_config, "controlled")
    shared_labels = [str(value) for value in labels_config.get("shared_labels", ())]
    requested = [str(value) for value in controlled_config.get("primary_holdouts", ())]
    if bool(controlled_config.get("run_all_shared_holdouts", False)):
        requested = shared_labels
    if not requested:
        raise MouseSpleenConfigError("controlled experiment requires at least one holdout")

    rna, atac = _load_canonical_modalities(config)
    embedding = np.load(embedding_result.artifacts["embedding"])
    embedding_cells = pd.read_csv(embedding_result.artifacts["embedding_cells"])
    embedding_by_id = {
        cell_id: embedding[int(row_index)]
        for cell_id, row_index in zip(
            embedding_cells["cell_id"].astype(str),
            embedding_cells["row_index"].astype(int),
            strict=True,
        )
    }
    query = atac.loc[atac["fine_label"].isin(shared_labels)].copy()
    reference_full = rna.loc[rna["fine_label"].isin(shared_labels)].copy()
    if query["broad_label"].isna().any() or reference_full["broad_label"].isna().any():
        raise MouseSpleenConfigError("Every shared canonical fine label must have a broad mapping")
    minimum = int(controlled_config.get("supplementary_min_query_positive_cells", 20))
    smoke_value = experiments_config.get("smoke", {})
    smoke_config = smoke_value if isinstance(smoke_value, dict) else {}
    smoke_enabled = bool(smoke_config.get("enabled", False))
    if smoke_enabled:
        smoke_holdout = _string(smoke_config, "holdout_label")
        if requested != [smoke_holdout]:
            raise MouseSpleenConfigError(
                "Enabled smoke configuration requires exactly its holdout_label as the controlled holdout"
            )
    artifacts: dict[str, Path] = {}
    prepared = []
    for holdout in requested:
        n_positive = int(query["fine_label"].eq(holdout).sum())
        if n_positive < minimum:
            continue
        task_query = query
        task_reference = reference_full
        preparation_metadata: dict[str, object] = {}
        if smoke_enabled:
            task_query, task_reference, preparation_metadata = _bounded_smoke_sample(
                query=query,
                reference_full=reference_full,
                holdout=holdout,
                config=smoke_config,
                default_seed=int(_mapping(config, "experiment").get("master_seed", 0)),
            )
        slug = _slug(holdout)
        result = _write_controlled_task_family(
            config=config,
            task_root=task_root / slug,
            output_root=output_root,
            holdout=holdout,
            query=task_query,
            reference_full=task_reference,
            embedding_by_id=embedding_by_id,
            preparation_metadata=preparation_metadata,
        )
        artifacts[f"{slug}_manifest"] = result.manifest_path
        prepared.append({"holdout_label": holdout, "n_query_positive": n_positive, "slug": slug})
    if not prepared:
        raise MouseSpleenConfigError("No controlled holdout meets the configured cell threshold")

    manifest_path = task_root / "task_families_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="prepare_task_families",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={"controlled_task_families": prepared},
        ),
    )
    return StageResult(
        stage="prepare_task_families",
        root=task_root,
        artifacts=artifacts,
        manifest_path=manifest_path,
    )


def _run_controlled(config: dict[str, Any]) -> StageResult:
    task_result = _prepare_task_families(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    runs_root = output_root / "runs"
    graph_config = _mapping(config, "candidate_graph")
    scaling_config = _mapping(config, "cost_scaling")
    configured_methods = [str(value) for value in _mapping(config, "methods").get("transport", ())]
    if not configured_methods:
        raise MouseSpleenConfigError("methods.transport must contain at least one method")
    methods = list(dict.fromkeys([*configured_methods, "prior_only"]))
    candidate_name = f"mouse_spleen_provider_k{int(graph_config.get('k_source_to_target', 100))}"
    scoring_config = _mapping(config, "scoring")
    controlled_artifacts: dict[str, Path] = {}

    for artifact_name, task_manifest_path in task_result.artifacts.items():
        task_manifest = _read_manifest_payload(task_manifest_path)
        run_root = Path(str(task_manifest["artifacts"]["run_root"]))
        run_id = run_root.name
        generated = run_root / "generated_configs"
        generated.mkdir(parents=True, exist_ok=True)
        conditions = [
            str(item["condition_id"]) for item in task_manifest["metadata"]["conditions"]
        ]
        common = {"run_id": run_id, "conditions": conditions, "outputs": {"root": str(runs_root)}}

        candidate_path = generated / "candidates.yaml"
        _write_yaml(
            candidate_path,
            {
                **common,
                "providers": ["mouse_spleen_provider"],
                "candidate_graph": graph_config,
                "cost_scaling": scaling_config,
            },
        )
        run_candidate_cost(candidate_path)

        transport_path = generated / "transport.yaml"
        _write_yaml(
            transport_path,
            {
                **common,
                "candidate_sets": [
                    {
                        "name": candidate_name,
                        "provider": "mouse_spleen_provider",
                        "prior_profile": "default",
                    }
                ],
                "methods": _transport_method_configs(config, methods),
            },
        )
        run_transport(transport_path)

        score_path = generated / "scoring.yaml"
        _write_yaml(
            score_path,
            {
                **common,
                "candidate_sets": [candidate_name],
                "methods": methods,
                "prior_adjustment": scoring_config.get("prior_adjustment", {}),
                "thresholds": scoring_config.get("thresholds", {}),
            },
        )
        run_scoring(score_path)

        evaluation_path = generated / "evaluation.yaml"
        _write_yaml(
            evaluation_path,
            {
                **common,
                "candidate_sets": [candidate_name],
                "methods": methods,
                "primary_negative_set": "same_broad",
            },
        )
        run_evaluation(evaluation_path)
        controlled_artifacts[artifact_name] = task_manifest_path

    manifest_path = output_root / "controlled_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_controlled",
            artifacts={key: str(value) for key, value in controlled_artifacts.items()},
            metadata={
                "candidate_set": candidate_name,
                "conditions": "task_manifest",
                "methods": methods,
                "evaluation_scope": "same_broad",
            },
        ),
    )
    return StageResult(
        stage="run_controlled",
        root=output_root,
        artifacts=controlled_artifacts,
        manifest_path=manifest_path,
    )


def _baseline_task_families(config: dict[str, Any]) -> StageResult:
    baseline_config = copy.deepcopy(config)
    experiments = _mapping(baseline_config, "experiments")
    experiments["support_dose"] = {"fractions": [0.0, 1.0], "n_seeds_for_partial_fractions": 0}
    experiments["random_deletion"] = {"n_seeds": 0, "controls": []}
    return _prepare_task_families(baseline_config)


def _prepare_baselines(config: dict[str, Any]) -> StageResult:
    tasks = _baseline_task_families(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    baseline_root = output_root / "compare_baselines"
    baseline_root.mkdir(parents=True, exist_ok=True)
    counts_path, counts_manifest = _prepare_external_pseudocounts(config, baseline_root)
    baseline_config = _mapping(config, "baselines")
    external_methods = [str(value) for value in baseline_config.get("external", ())]
    if not external_methods:
        raise MouseSpleenConfigError("baselines.external must contain at least one method")
    config_root = baseline_root / "external_configs"
    artifacts: dict[str, Path] = {"pseudocount_manifest": counts_manifest}
    for task_manifest_path in tasks.artifacts.values():
        task = _read_manifest_payload(task_manifest_path)
        run_root = Path(str(task["artifacts"]["run_root"]))
        run_id = run_root.name
        run_config_root = config_root / run_id
        run_config_root.mkdir(parents=True, exist_ok=True)
        config_path = run_config_root / "external_baselines.yaml"
        repeat = int(_mapping(config, "experiment").get("master_seed", 0))
        _write_yaml(
            config_path,
            {
                "run_id": run_id,
                "heldout_label": str(task["metadata"]["holdout_label"]),
                "repeat": repeat,
                "conditions": ["incomplete_reference", "full_reference_control"],
                "candidate_set": "external_reference_mapping",
                "methods": external_methods,
                "outputs": {"root": str(output_root / "runs")},
                "threshold_quantile": float(baseline_config.get("threshold_quantile", 0.95)),
                "r_scripts_dir": str(
                    baseline_config.get(
                        "r_scripts_dir", "experiments/missing_celltype/external_baselines"
                    )
                ),
                "model_visible_counts_source": str(counts_path.resolve()),
                "keep_intermediate_artifacts": bool(
                    baseline_config.get("keep_intermediate_artifacts", False)
                ),
            },
        )
        _write_yaml(
            run_config_root / "benchmark.yaml",
            {
                "run_id": run_id,
                "removed_state": str(task["metadata"]["holdout_label"]),
                "repeat": repeat,
                "split": {"seed": repeat},
            },
        )
        artifacts[f"{run_id}_external_config"] = config_path
    manifest_path = baseline_root / "baseline_inputs_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="prepare_baselines",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "conditions": ["incomplete_reference", "full_reference_control"],
                "external_methods": external_methods,
                "expression_input": "rounded_expm1_library_10000_pseudocounts",
            },
        ),
    )
    return StageResult("prepare_baselines", baseline_root, artifacts, manifest_path)


def _prepare_external_pseudocounts(
    config: dict[str, Any], baseline_root: Path
) -> tuple[Path, Path]:
    data_config = _mapping(config, "data")
    counts_root = baseline_root / "input"
    counts_root.mkdir(parents=True, exist_ok=True)
    counts_path = counts_root / "model_visible_pseudocounts.h5ad"
    manifest_path = counts_root / "pseudocount_manifest.yaml"
    source_paths = {
        "rna": Path(_string(data_config, "rna_h5ad")),
        "atac_gene_activity": Path(_string(data_config, "atac_gene_activity_h5ad")),
    }
    source_hashes = {key: sha256_file(path) for key, path in source_paths.items()}
    if counts_path.is_file() and manifest_path.is_file():
        manifest = _read_manifest_payload(manifest_path)
        if manifest.get("metadata", {}).get("source_sha256") == source_hashes:
            return counts_path, manifest_path

    rna = ad.read_h5ad(source_paths["rna"])
    atac = ad.read_h5ad(source_paths["atac_gene_activity"])
    try:
        common_genes = rna.var_names.intersection(atac.var_names, sort=False)
        if len(common_genes) == 0:
            raise MouseSpleenConfigError(
                "External baselines require shared RNA and ATAC gene-activity features"
            )
        rna_full_counts, rna_deviation = _inverse_log1p_pseudocounts(
            sparse.csr_matrix(rna.X)
        )
        atac_full_counts, atac_deviation = _inverse_log1p_pseudocounts(
            sparse.csr_matrix(atac.X)
        )
        rna_indices = rna.var_names.get_indexer(common_genes)
        atac_indices = atac.var_names.get_indexer(common_genes)
        rna_counts = rna_full_counts[:, rna_indices]
        atac_counts = atac_full_counts[:, atac_indices]
        matrix = sparse.vstack([atac_counts, rna_counts], format="csr")
        obs = pd.DataFrame(
            {
                "cell_id": [
                    *(f"atac::{value}" for value in atac.obs_names.astype(str)),
                    *(f"rna::{value}" for value in rna.obs_names.astype(str)),
                ],
                "domain": ["query"] * atac.n_obs + ["reference"] * rna.n_obs,
            }
        )
        obs.index = pd.Index(obs["cell_id"], name="cell_id_index")
        combined = ad.AnnData(
            X=matrix,
            obs=obs,
            var=pd.DataFrame(index=pd.Index(common_genes.astype(str), name="gene")),
        )
        combined.layers["counts"] = matrix.copy()
        combined.write_h5ad(counts_path, compression="gzip")
    finally:
        del rna, atac
    write_manifest(
        manifest_path,
        Manifest(
            stage="prepare_baseline_pseudocounts",
            artifacts={"model_visible_pseudocounts": str(counts_path)},
            metadata={
                "source_sha256": source_hashes,
                "transformation": "round(expm1(log1p_normalized_expression))",
                "assumed_library_size": 10000,
                "raw_library_sizes_recoverable": False,
                "n_cells": int(matrix.shape[0]),
                "n_common_genes": int(matrix.shape[1]),
                "max_pre_round_library_sum_deviation": {
                    "rna": rna_deviation,
                    "atac_gene_activity": atac_deviation,
                },
                "scientific_status": "normalized_pseudocount_sensitivity_input",
            },
        ),
    )
    return counts_path, manifest_path


def _inverse_log1p_pseudocounts(matrix: sparse.csr_matrix) -> tuple[sparse.csr_matrix, float]:
    counts = matrix.astype(np.float64, copy=True)
    counts.data = np.expm1(counts.data)
    sums = np.asarray(counts.sum(axis=1)).ravel()
    deviation = float(np.max(np.abs(sums - 10000.0)))
    if deviation > 0.1:
        raise MouseSpleenConfigError(
            "Expression matrix is not log1p-normalized to library size 10000; "
            f"maximum inverse-log library deviation is {deviation:.6g}"
        )
    counts.data = np.rint(counts.data)
    counts.eliminate_zeros()
    return counts.astype(np.int32), deviation


def _run_internal_baselines(config: dict[str, Any]) -> StageResult:
    prepared = _prepare_baselines(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    runs_root = output_root / "runs"
    baseline_config = _mapping(config, "baselines")
    methods = [str(value) for value in baseline_config.get("internal", ())]
    if not methods:
        raise MouseSpleenConfigError("baselines.internal must contain at least one method")
    graph_config = _mapping(config, "candidate_graph")
    candidate_name = f"mouse_spleen_provider_k{int(graph_config.get('k_source_to_target', 100))}"
    task_result = _baseline_task_families(config)
    artifacts: dict[str, Path] = {}
    for task_manifest_path in task_result.artifacts.values():
        task = _read_manifest_payload(task_manifest_path)
        run_root = Path(str(task["artifacts"]["run_root"]))
        run_id = run_root.name
        generated = run_root / "generated_configs/baselines"
        generated.mkdir(parents=True, exist_ok=True)
        common = {
            "run_id": run_id,
            "conditions": ["incomplete_reference", "full_reference_control"],
            "outputs": {"root": str(runs_root)},
        }
        candidate_path = generated / "candidates.yaml"
        _write_yaml(
            candidate_path,
            {
                **common,
                "providers": ["mouse_spleen_provider"],
                "candidate_graph": graph_config,
                "cost_scaling": _mapping(config, "cost_scaling"),
            },
        )
        run_candidate_cost(candidate_path)
        transport_path = generated / "transport.yaml"
        _write_yaml(
            transport_path,
            {
                **common,
                "candidate_sets": [
                    {
                        "name": candidate_name,
                        "provider": "mouse_spleen_provider",
                        "prior_profile": "default",
                    }
                ],
                "methods": _transport_method_configs(config, methods),
            },
        )
        run_transport(transport_path)
        scoring_path = generated / "scoring.yaml"
        _write_yaml(
            scoring_path,
            {
                **common,
                "candidate_sets": [candidate_name],
                "methods": methods,
                "prior_adjustment": _mapping(config, "scoring").get("prior_adjustment", {}),
                "thresholds": _mapping(config, "scoring").get("thresholds", {}),
            },
        )
        run_scoring(scoring_path)
        evaluation_path = generated / "evaluation.yaml"
        _write_yaml(
            evaluation_path,
            {
                **common,
                "candidate_sets": [candidate_name],
                "methods": methods,
                "primary_negative_set": "same_broad",
            },
        )
        run_evaluation(evaluation_path)
        artifacts[run_id] = task_manifest_path
    manifest_path = prepared.root / "internal_baselines_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_internal_baselines",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={"methods": methods, "candidate_set": candidate_name},
        ),
    )
    return StageResult("run_internal_baselines", prepared.root, artifacts, manifest_path)


def _run_external_baselines(config: dict[str, Any]) -> StageResult:
    prepared = _prepare_baselines(config)
    artifacts: dict[str, Path] = {}
    for name, config_path in prepared.artifacts.items():
        if not name.endswith("_external_config"):
            continue
        result = run_external_baselines(config_path)
        artifacts[result.run_root.name] = config_path
    manifest_path = prepared.root / "external_baselines_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_external_baselines",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "methods": list(_mapping(config, "baselines").get("external", ())),
                "candidate_set": "external_reference_mapping",
            },
        ),
    )
    return StageResult("run_external_baselines", prepared.root, artifacts, manifest_path)


def _aggregate_baselines(config: dict[str, Any]) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    baseline_root = output_root / "compare_baselines"
    config_root = baseline_root / "external_configs"
    if not config_root.is_dir():
        raise FileNotFoundError("Baseline configs do not exist; run prepare_baselines first")
    internal_candidate = f"mouse_spleen_provider_k{int(_mapping(config, 'candidate_graph').get('k_source_to_target', 100))}"
    baseline_config = _mapping(config, "baselines")
    method_sources = {
        **{str(method): internal_candidate for method in baseline_config.get("internal", ())},
        **{
            str(method): "external_reference_mapping"
            for method in baseline_config.get("external", ())
        },
    }
    detection_rows: list[dict[str, object]] = []
    shared_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    for run_config_root in sorted(path for path in config_root.iterdir() if path.is_dir()):
        descriptor = _read_manifest_payload(run_config_root / "benchmark.yaml")
        run_id = str(descriptor["run_id"])
        holdout = str(descriptor["removed_state"])
        run_root = output_root / "runs" / run_id
        truths = {
            condition: pd.read_csv(
                run_root / f"benchmark/{condition}/evaluation_truth/query_truth.csv"
            )
            for condition in ("incomplete_reference", "full_reference_control")
        }
        for method, candidate in method_sources.items():
            score_paths = {
                condition: run_root / f"scoring/{condition}/{candidate}/cell_scores.parquet"
                for condition in truths
            }
            if not all(path.is_file() for path in score_paths.values()):
                raise FileNotFoundError(
                    f"Baseline outputs for run={run_id}, method={method} are incomplete"
                )
            joined = {
                condition: pd.read_parquet(path)
                .loc[lambda frame: frame["method"].astype(str).eq(method)]
                .merge(truths[condition], on="cell_id", validate="one_to_one")
                for condition, path in score_paths.items()
            }
            incomplete = joined["incomplete_reference"]
            positive_broad = incomplete.loc[
                incomplete["true_label"].astype(str).eq(holdout), "true_broad_label"
            ].astype(str).unique()
            if len(positive_broad) != 1:
                raise MouseSpleenConfigError(
                    f"Baseline holdout {holdout!r} must have one evaluation broad class"
                )
            scoped = incomplete.loc[
                incomplete["true_broad_label"].astype(str).eq(positive_broad[0])
            ]
            absent = scoped["true_label"].astype(str).eq(holdout)
            score_column = {
                "prior_only": "prior_risk",
            }.get(method, "u")
            detection_rows.append(
                {
                    "run_id": run_id,
                    "holdout_label": holdout,
                    "method": method,
                    "candidate_set": candidate,
                    "score": score_column,
                    "evaluation_scope": "within_broad",
                    "n_query": int(len(scoped)),
                    "n_positive": int(absent.sum()),
                    "auroc": safe_auroc(absent, scoped[score_column]),
                    "auprc": safe_auprc(absent, scoped[score_column]),
                    "auprc_baseline": float(absent.mean()),
                    "absent_abstention_rate": _boolean_mean(
                        scoped.loc[absent, "abstain_u_or_entropy"]
                    ),
                    "shared_false_abstention_rate": _boolean_mean(
                        scoped.loc[~absent, "abstain_u_or_entropy"]
                    ),
                }
            )
            shared = incomplete.loc[incomplete["is_shared_state"].astype(bool)]
            shared_rows.append(
                {
                    "run_id": run_id,
                    "holdout_label": holdout,
                    "method": method,
                    "candidate_set": candidate,
                    **_label_transfer_metrics(shared),
                }
            )
            full = joined["full_reference_control"]
            full_rows.append(
                {
                    "run_id": run_id,
                    "holdout_label": holdout,
                    "method": method,
                    "candidate_set": candidate,
                    "score": score_column,
                    "full_reference_false_abstention_rate": _boolean_mean(
                        full["abstain_u_or_entropy"]
                    ),
                    "score_median": safe_median(full[score_column]),
                    "score_p95": float(np.nanquantile(full[score_column], 0.95)),
                    **_label_transfer_metrics(full),
                }
            )
    tables_root = baseline_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "detection": tables_root / "baseline_detection_by_run.csv",
        "detection_summary": tables_root / "baseline_detection_summary.csv",
        "shared_label_transfer": tables_root / "baseline_shared_label_transfer_by_run.csv",
        "shared_label_transfer_summary": tables_root
        / "baseline_shared_label_transfer_summary.csv",
        "full_reference": tables_root / "baseline_full_reference_by_run.csv",
        "full_reference_summary": tables_root / "baseline_full_reference_summary.csv",
    }
    detection = pd.DataFrame(detection_rows)
    shared = pd.DataFrame(shared_rows)
    full = pd.DataFrame(full_rows)
    detection.to_csv(artifacts["detection"], index=False)
    shared.to_csv(artifacts["shared_label_transfer"], index=False)
    full.to_csv(artifacts["full_reference"], index=False)
    _summarize_baseline_table(
        detection,
        metrics=(
            "auroc",
            "auprc",
            "auprc_baseline",
            "absent_abstention_rate",
            "shared_false_abstention_rate",
        ),
    ).to_csv(artifacts["detection_summary"], index=False)
    _summarize_baseline_table(
        shared,
        metrics=(
            "forced_accuracy",
            "forced_macro_f1",
            "post_abstention_accuracy",
            "post_abstention_macro_f1",
            "coverage",
            "shared_false_abstention_rate",
        ),
    ).to_csv(artifacts["shared_label_transfer_summary"], index=False)
    _summarize_baseline_table(
        full,
        metrics=(
            "full_reference_false_abstention_rate",
            "score_median",
            "score_p95",
            "forced_accuracy",
            "forced_macro_f1",
            "post_abstention_accuracy",
            "post_abstention_macro_f1",
            "coverage",
        ),
    ).to_csv(artifacts["full_reference_summary"], index=False)
    manifest_path = baseline_root / "baseline_metrics_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_baselines",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "internal_candidate_set": internal_candidate,
                "external_candidate_set": "external_reference_mapping",
                "primary_detection_scope": "within_broad",
                "prior_only_detection_score": "prior_risk",
                "summary_variability": "descriptive_across_holdouts_not_biological_replicates",
            },
        ),
    )
    return StageResult("aggregate_baselines", baseline_root, artifacts, manifest_path)


def _boolean_mean(values: pd.Series) -> float:
    return float(values.astype(bool).mean()) if len(values) else float("nan")


def _summarize_baseline_table(
    frame: pd.DataFrame, *, metrics: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, candidate), group in frame.groupby(["method", "candidate_set"], sort=True):
        row: dict[str, object] = {
            "method": method,
            "candidate_set": candidate,
            "n_holdouts": int(group["holdout_label"].nunique()),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            if not values.notna().any():
                for statistic in ("mean", "std", "median", "min", "max"):
                    row[f"{metric}_{statistic}"] = float("nan")
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_true_label_macro_f1(
    true_labels: pd.Series | np.ndarray,
    predicted_labels: pd.Series | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
) -> float:
    true = pd.Series(true_labels, copy=False).astype(str)
    predicted = pd.Series(predicted_labels, copy=False).astype(str)
    fixed_labels = sorted(true.unique()) if labels is None else list(labels)
    if true.empty or not fixed_labels:
        return float("nan")
    return float(
        f1_score(
            true,
            predicted,
            labels=fixed_labels,
            average="macro",
            zero_division=0.0,
        )
    )


def _label_transfer_metrics(
    frame: pd.DataFrame, *, applicable: bool = True
) -> dict[str, float]:
    if frame.empty or not applicable:
        return {
            "forced_accuracy": float("nan"),
            "forced_macro_f1": float("nan"),
            "post_abstention_accuracy": float("nan"),
            "post_abstention_macro_f1": float("nan"),
            "coverage": float("nan"),
            "shared_false_abstention_rate": float("nan"),
        }
    true = frame["true_label"].astype(str)
    forced = frame["forced_label"].fillna("").astype(str)
    accepted = ~frame["abstain_u_or_entropy"].astype(bool)
    labels = sorted(true.unique())
    return {
        "forced_accuracy": float(accuracy_score(true, forced)),
        "forced_macro_f1": fixed_true_label_macro_f1(true, forced, labels=labels),
        "post_abstention_accuracy": (
            float(accuracy_score(true[accepted], forced[accepted]))
            if accepted.any()
            else float("nan")
        ),
        "post_abstention_macro_f1": (
            fixed_true_label_macro_f1(
                true[accepted], forced[accepted], labels=labels
            )
            if accepted.any()
            else float("nan")
        ),
        "coverage": float(accepted.mean()),
        "shared_false_abstention_rate": float((~accepted).mean()),
    }


def _aggregate_metrics(config: dict[str, Any]) -> StageResult:
    task_result = _prepare_task_families(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    metrics_root = output_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    graph_config = _mapping(config, "candidate_graph")
    candidate_name = f"mouse_spleen_provider_k{int(graph_config.get('k_source_to_target', 100))}"
    paired_frames: list[pd.DataFrame] = []
    detection_rows: list[dict[str, object]] = []
    prior_detection_rows: list[dict[str, object]] = []
    restoration_rows: list[dict[str, object]] = []
    dose_run_rows: list[dict[str, object]] = []
    deletion_run_rows: list[dict[str, object]] = []

    for task_manifest_path in task_result.artifacts.values():
        task_manifest = _read_manifest_payload(task_manifest_path)
        metadata = task_manifest["metadata"]
        holdout = str(metadata["holdout_label"])
        holdout_broad = str(metadata["holdout_broad_label"])
        run_root = Path(str(task_manifest["artifacts"]["run_root"]))
        full_path = (
            run_root
            / "scoring/full_reference_control"
            / candidate_name
            / "cell_scores.parquet"
        )
        incomplete_path = (
            run_root
            / "scoring/incomplete_reference"
            / candidate_name
            / "cell_scores.parquet"
        )
        truth_path = run_root / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
        for path in (full_path, incomplete_path, truth_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Controlled outputs are incomplete; run run_controlled first: {path}"
                )
        full = pd.read_parquet(full_path)
        incomplete = pd.read_parquet(incomplete_path)
        truth = pd.read_csv(truth_path)
        prior = incomplete.loc[
            incomplete["method"].astype(str).eq("prior_only"),
            ["cell_id", "prior_risk"],
        ].merge(truth, on="cell_id", how="inner", validate="one_to_one")
        for scope, mask in (
            ("within_broad", prior["true_broad_label"].astype(str).eq(holdout_broad)),
            ("global_all_query", pd.Series(True, index=prior.index)),
        ):
            scoped_prior = prior.loc[mask]
            y_prior = scoped_prior["true_label"].astype(str).eq(holdout)
            prior_detection_rows.append(
                {
                    "holdout_label": holdout,
                    "method": "prior_only",
                    "score": "one_minus_rho",
                    "evaluation_scope": scope,
                    "n_query": int(len(scoped_prior)),
                    "n_positive": int(y_prior.sum()),
                    "auroc": safe_auroc(y_prior, scoped_prior["prior_risk"]),
                    "auprc": safe_auprc(y_prior, scoped_prior["prior_risk"]),
                    "auprc_baseline": (
                        float(y_prior.mean()) if len(y_prior) else float("nan")
                    ),
                    "median_absent": safe_median(
                        scoped_prior.loc[y_prior, "prior_risk"]
                    ),
                    "median_shared": safe_median(
                        scoped_prior.loc[~y_prior, "prior_risk"]
                    ),
                }
            )
        methods = sorted(
            set(full.loc[full["u"].notna(), "method"].astype(str))
            & set(incomplete.loc[incomplete["u"].notna(), "method"].astype(str))
        )
        for method in methods:
            left = incomplete.loc[
                incomplete["method"].astype(str).eq(method),
                ["cell_id", "u", "prior_risk"],
            ].rename(
                columns={
                    "u": "u_incomplete",
                    "prior_risk": "one_minus_rho_incomplete",
                }
            )
            right = full.loc[
                full["method"].astype(str).eq(method),
                ["cell_id", "u", "prior_risk"],
            ].rename(
                columns={"u": "u_full", "prior_risk": "one_minus_rho_full"}
            )
            paired = left.merge(right, on="cell_id", how="inner", validate="one_to_one")
            if not np.allclose(
                paired["one_minus_rho_incomplete"],
                paired["one_minus_rho_full"],
                equal_nan=False,
            ):
                raise MouseSpleenConfigError(
                    "Paired conditions do not have identical matchability priors"
                )
            paired["one_minus_rho"] = paired["one_minus_rho_incomplete"]
            paired = paired.merge(truth, on="cell_id", how="left", validate="one_to_one")
            if paired[["true_label", "true_broad_label"]].isna().any().any():
                raise MouseSpleenConfigError("Paired scores are missing evaluation truth")
            paired.insert(0, "method", method)
            paired.insert(0, "holdout_label", holdout)
            paired = paired.rename(columns={"cell_id": "query_cell_id"})
            paired["delta_u"] = paired["u_incomplete"] - paired["u_full"]
            paired_frames.append(paired)

            positive = paired["true_label"].astype(str).eq(holdout)
            for scope, mask in (
                ("within_broad", paired["true_broad_label"].astype(str).eq(holdout_broad)),
                ("global_all_query", pd.Series(True, index=paired.index)),
            ):
                scoped = paired.loc[mask]
                y = scoped["true_label"].astype(str).eq(holdout)
                detection_rows.append(
                    {
                        "holdout_label": holdout,
                        "method": method,
                        "score": "u",
                        "evaluation_scope": scope,
                        "n_query": int(len(scoped)),
                        "n_positive": int(y.sum()),
                        "auroc": safe_auroc(y, scoped["u_incomplete"]),
                        "auprc": safe_auprc(y, scoped["u_incomplete"]),
                        "auprc_baseline": float(y.mean()) if len(y) else float("nan"),
                        "median_absent": safe_median(scoped.loc[y, "u_incomplete"]),
                        "median_shared": safe_median(scoped.loc[~y, "u_incomplete"]),
                    }
                )
            restoration_rows.append(
                {
                    "holdout_label": holdout,
                    "method": method,
                    "score": "delta_u",
                    "median_delta_u_positive": safe_median(paired.loc[positive, "delta_u"]),
                    "median_delta_u_shared": safe_median(paired.loc[~positive, "delta_u"]),
                    "auroc_delta_u": safe_auroc(positive, paired["delta_u"]),
                    "auprc_delta_u": safe_auprc(positive, paired["delta_u"]),
                }
            )

            condition_items = list(metadata.get("conditions", ()))
            for condition_item in condition_items:
                condition = str(condition_item["condition_id"])
                family = str(condition_item.get("family", ""))
                if family not in {"controlled", "support_dose", "random_deletion"}:
                    continue
                condition_path = (
                    run_root / "scoring" / condition / candidate_name / "cell_scores.parquet"
                )
                if not condition_path.is_file():
                    raise FileNotFoundError(f"Missing condition score artifact: {condition_path}")
                condition_scores = pd.read_parquet(condition_path)
                condition_scores = condition_scores.loc[
                    condition_scores["method"].astype(str).eq(method), ["cell_id", "u"]
                ]
                condition_paired = condition_scores.merge(
                    truth, on="cell_id", how="inner", validate="one_to_one"
                )
                condition_paired = condition_paired.merge(
                    right, on="cell_id", how="inner", validate="one_to_one"
                )
                condition_positive = condition_paired["true_label"].astype(str).eq(holdout)
                delta = condition_paired["u"] - condition_paired["u_full"]
                if family in {"controlled", "support_dose"}:
                    dose_run_rows.append(
                        {
                            "holdout_label": holdout,
                            "method": method,
                            "condition_id": condition,
                            "support_fraction": float(condition_item["support_fraction"]),
                            "random_seed": condition_item.get("random_seed"),
                            "median_positive_u": safe_median(
                                condition_paired.loc[condition_positive, "u"]
                            ),
                        }
                    )
                if family == "random_deletion":
                    deletion_run_rows.append(
                        {
                            "holdout_label": holdout,
                            "method": method,
                            "condition_id": condition,
                            "deletion_control_type": condition_item["deletion_control_type"],
                            "random_seed": condition_item["random_seed"],
                            "median_delta_u_positive": safe_median(delta[condition_positive]),
                            "median_delta_u_shared": safe_median(delta[~condition_positive]),
                        }
                    )

    paired_output = pd.concat(paired_frames, ignore_index=True)
    paired_path = metrics_root / "paired_cells.parquet"
    detection_path = metrics_root / "table_controlled_detection.csv"
    prior_detection_path = metrics_root / "table_prior_only_detection.csv"
    restoration_path = metrics_root / "table_restoration.csv"
    support_dose_path = metrics_root / "table_support_dose.csv"
    random_deletion_path = metrics_root / "table_random_deletion.csv"
    natural_path = metrics_root / "table_natural_mismatch.csv"
    paired_output.to_parquet(paired_path, index=False)
    pd.DataFrame(detection_rows).to_csv(detection_path, index=False)
    pd.DataFrame(prior_detection_rows).to_csv(prior_detection_path, index=False)
    pd.DataFrame(restoration_rows).to_csv(restoration_path, index=False)
    dose_runs = pd.DataFrame(dose_run_rows)
    dose_summary = (
        dose_runs.groupby(["holdout_label", "method", "support_fraction"], as_index=False)
        .agg(
            mean=("median_positive_u", "mean"),
            sd=("median_positive_u", lambda values: float(np.std(values, ddof=0))),
            median=("median_positive_u", "median"),
            min=("median_positive_u", "min"),
            max=("median_positive_u", "max"),
            n_seeds=("median_positive_u", "size"),
        )
        .sort_values(["holdout_label", "method", "support_fraction"])
    )
    dose_summary.to_csv(support_dose_path, index=False)
    pd.DataFrame(deletion_run_rows).to_csv(random_deletion_path, index=False)
    _natural_metric_rows(output_root).to_csv(natural_path, index=False)
    manifest_path = metrics_root / "metrics_manifest.yaml"
    artifacts = {
        "paired_cells": paired_path,
        "controlled_detection": detection_path,
        "prior_only_detection": prior_detection_path,
        "restoration": restoration_path,
        "support_dose": support_dose_path,
        "random_deletion": random_deletion_path,
        "natural_mismatch": natural_path,
    }
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_metrics",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "primary_score": "u",
                "primary_evaluation_scope": "within_broad",
                "pairing_key": "query_cell_id",
            },
        ),
    )
    return StageResult(
        stage="aggregate_metrics",
        root=metrics_root,
        artifacts=artifacts,
        manifest_path=manifest_path,
    )


def _natural_metric_rows(output_root: Path) -> pd.DataFrame:
    manifest_path = output_root / "natural_mismatch/natural_mismatch_manifest.yaml"
    if not manifest_path.is_file():
        return pd.DataFrame(columns=_natural_metric_columns())
    metrics, _, _ = _build_natural_mismatch_tables(output_root)
    return metrics


def _aggregate_natural_mismatch(config: dict[str, Any]) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    natural_root = output_root / "natural_mismatch"
    metrics, cell_scores, broad_predictions = _build_natural_mismatch_tables(output_root)
    tables_root = natural_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "metrics": tables_root / "table_natural_mismatch.csv",
        "cell_scores": tables_root / "natural_cell_scores.parquet",
        "broad_predictions": tables_root / "table_natural_broad_predictions.csv",
    }
    metrics.to_csv(artifacts["metrics"], index=False)
    cell_scores.to_parquet(artifacts["cell_scores"], index=False)
    broad_predictions.to_csv(artifacts["broad_predictions"], index=False)
    manifest_path = natural_root / "natural_mismatch_metrics_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_natural_mismatch",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "primary_score": "u",
                "ifit_b_primary_scope": "within_broad",
                "proliferating_primary_scope": "global_all_query",
                "restoration_metric_applicability": "undefined",
            },
        ),
    )
    return StageResult("aggregate_natural_mismatch", natural_root, artifacts, manifest_path)


def _natural_endpoint_manifests(config: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    manifest_path = output_root / "natural_mismatch/natural_mismatch_manifest.yaml"
    if not manifest_path.is_file():
        _run_natural_mismatch(config)
    manifest = _read_manifest_payload(manifest_path)
    return [
        _read_manifest_payload(Path(str(path)))
        for path in manifest.get("artifacts", {}).values()
    ]


def _prepare_natural_baselines(config: dict[str, Any]) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    baseline_root = output_root / "natural_mismatch/compare_baselines"
    baseline_root.mkdir(parents=True, exist_ok=True)
    counts_path, counts_manifest = _prepare_external_pseudocounts(config, baseline_root)
    external_methods = [
        str(value) for value in _mapping(config, "baselines").get("external", ())
    ]
    if not external_methods:
        raise MouseSpleenConfigError("baselines.external must contain at least one method")
    config_root = baseline_root / "external_configs"
    artifacts: dict[str, Path] = {"pseudocount_manifest": counts_manifest}
    repeat = int(_mapping(config, "experiment").get("master_seed", 0))
    baseline_config = _mapping(config, "baselines")
    for endpoint_manifest in _natural_endpoint_manifests(config):
        endpoint = str(endpoint_manifest["metadata"]["natural_endpoint"])
        run_root = Path(str(endpoint_manifest["artifacts"]["run_root"]))
        run_config_root = config_root / run_root.name
        run_config_root.mkdir(parents=True, exist_ok=True)
        config_path = run_config_root / "external_baselines.yaml"
        _write_yaml(
            config_path,
            {
                "run_id": run_root.name,
                "heldout_label": endpoint,
                "repeat": repeat,
                "conditions": ["natural_mismatch"],
                "candidate_set": "external_reference_mapping",
                "methods": external_methods,
                "outputs": {"root": str(output_root / "runs")},
                "threshold_mode": "none",
                "r_scripts_dir": str(
                    baseline_config.get(
                        "r_scripts_dir",
                        "experiments/missing_celltype/external_baselines",
                    )
                ),
                "model_visible_counts_source": str(counts_path.resolve()),
                "keep_intermediate_artifacts": bool(
                    baseline_config.get("keep_intermediate_artifacts", False)
                ),
            },
        )
        _write_yaml(
            run_config_root / "benchmark.yaml",
            {
                "run_id": run_root.name,
                "removed_state": endpoint,
                "repeat": repeat,
                "split": {"seed": repeat},
                "condition": "natural_mismatch",
            },
        )
        artifacts[f"{run_root.name}_external_config"] = config_path
    manifest_path = baseline_root / "baseline_inputs_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="prepare_natural_baselines",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "condition": "natural_mismatch",
                "external_methods": external_methods,
                "threshold_mode": "none_no_matched_full_reference",
                "expression_input": "rounded_expm1_library_10000_pseudocounts",
            },
        ),
    )
    return StageResult("prepare_natural_baselines", baseline_root, artifacts, manifest_path)


def _run_natural_external_baselines(config: dict[str, Any]) -> StageResult:
    prepared = _prepare_natural_baselines(config)
    artifacts: dict[str, Path] = {}
    for name, config_path in prepared.artifacts.items():
        if name.endswith("_external_config"):
            result = run_external_baselines(config_path)
            artifacts[result.run_root.name] = config_path
    manifest_path = prepared.root / "external_baselines_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_natural_external_baselines",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "methods": list(_mapping(config, "baselines").get("external", ())),
                "candidate_set": "external_reference_mapping",
                "threshold_applicability": "undefined_without_matched_full_reference",
            },
        ),
    )
    return StageResult(
        "run_natural_external_baselines", prepared.root, artifacts, manifest_path
    )


def _natural_primary_scope(endpoint: str, truth: pd.DataFrame) -> pd.Series:
    if endpoint == "Ifit B":
        return truth["true_broad_label"].astype(str).eq("B")
    return pd.Series(True, index=truth.index)


def _aggregate_natural_baselines(config: dict[str, Any]) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    baseline_root = output_root / "natural_mismatch/compare_baselines"
    natural_config = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    coreot_full_operating_point = _mapping(natural_config, "coreot_full")
    internal_candidate = f"mouse_spleen_provider_k{int(_mapping(config, 'candidate_graph').get('k_source_to_target', 100))}"
    baseline_config = _mapping(config, "baselines")
    natural_internal_methods = list(
        dict.fromkeys(
            [
                *map(str, baseline_config.get("internal", ())),
                "uniform_uot",
                "coreot_constant_tau",
                "coreot_match_only",
                "coreot_full",
            ]
        )
    )
    required_converged_internal_methods = {
        "coreot_full",
        "coreot_constant_tau",
        "coreot_match_only",
        "uniform_uot",
    }
    natural_config = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    endpoint_configs = _mapping(natural_config, "endpoints")
    uniform_uot_tau_source_by_endpoint = {
        endpoint: (
            float(endpoint_config["uniform_uot_tau_source"])
            if "uniform_uot_tau_source" in endpoint_config
            else "matched_coreot_mean"
        )
        for endpoint, endpoint_config in endpoint_configs.items()
    }
    method_sources = {
        **{
            str(method): internal_candidate
            for method in natural_internal_methods
        },
        **{
            str(method): "external_reference_mapping"
            for method in baseline_config.get("external", ())
        },
    }
    detection_rows: list[dict[str, object]] = []
    shared_rows: list[dict[str, object]] = []
    for endpoint_manifest in _natural_endpoint_manifests(config):
        endpoint = str(endpoint_manifest["metadata"]["natural_endpoint"])
        run_root = Path(str(endpoint_manifest["artifacts"]["run_root"]))
        truth = pd.read_csv(
            run_root / "benchmark/natural_mismatch/evaluation_truth/query_truth.csv"
        )
        for method, candidate in method_sources.items():
            if method in required_converged_internal_methods:
                transport_manifest_path = (
                    run_root
                    / "transport/natural_mismatch"
                    / candidate
                    / method
                    / "transport_manifest.yaml"
                )
                if not transport_manifest_path.is_file():
                    raise FileNotFoundError(
                        "Natural factorial transport manifest is missing for "
                        f"run={run_root.name}, method={method}: "
                        f"{transport_manifest_path}"
                    )
                transport_manifest = _read_manifest_payload(transport_manifest_path)
                if not bool(transport_manifest.get("metadata", {}).get("converged")):
                    raise MouseSpleenConfigError(
                        "Natural factorial method did not converge and cannot enter "
                        f"the baseline comparison: run={run_root.name}, method={method}"
                    )
            score_path = (
                run_root / f"scoring/natural_mismatch/{candidate}/cell_scores.parquet"
            )
            if not score_path.is_file():
                raise FileNotFoundError(
                    f"Natural baseline output is missing for run={run_root.name}, "
                    f"method={method}: {score_path}"
                )
            method_scores = pd.read_parquet(score_path).loc[
                lambda frame: frame["method"].astype(str).eq(method)
            ]
            if method_scores.empty:
                raise FileNotFoundError(
                    f"Natural baseline scores omit run={run_root.name}, method={method}"
                )
            joined = truth.merge(method_scores, on="cell_id", validate="one_to_one")
            scoped = joined.loc[_natural_primary_scope(endpoint, joined)]
            positive = scoped["true_label"].astype(str).eq(endpoint)
            score_column = {
                "prior_only": "prior_risk",
            }.get(method, "u")
            detection_rows.append(
                {
                    "run_id": run_root.name,
                    "holdout_label": endpoint,
                    "method": method,
                    "candidate_set": candidate,
                    "score": score_column,
                    "evaluation_scope": (
                        "within_broad" if endpoint == "Ifit B" else "global_all_query"
                    ),
                    "n_query": int(len(scoped)),
                    "n_positive": int(positive.sum()),
                    "auroc": safe_auroc(positive, scoped[score_column]),
                    "auprc": safe_auprc(positive, scoped[score_column]),
                    "auprc_baseline": float(positive.mean()),
                    "median_absent": safe_median(scoped.loc[positive, score_column]),
                    "median_shared": safe_median(scoped.loc[~positive, score_column]),
                    "threshold_applicability": "undefined",
                }
            )
            shared = joined.loc[joined["is_shared_state"].astype(bool)]
            shared_rows.append(
                {
                    "run_id": run_root.name,
                    "holdout_label": endpoint,
                    "method": method,
                    "candidate_set": candidate,
                    "label_transfer_applicability": (
                        "not_applicable_detection_only_method"
                        if method == "prior_only"
                        else "applicable"
                    ),
                    **_label_transfer_metrics(
                        shared, applicable=method != "prior_only"
                    ),
                }
            )
    tables_root = baseline_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "detection": tables_root / "baseline_detection_by_run.csv",
        "detection_summary": tables_root / "baseline_detection_summary.csv",
        "shared_label_transfer": tables_root / "baseline_shared_label_transfer_by_run.csv",
        "shared_label_transfer_summary": tables_root
        / "baseline_shared_label_transfer_summary.csv",
        "report": baseline_root / "baseline_comparison_report.md",
    }
    detection = pd.DataFrame(detection_rows)
    shared = pd.DataFrame(shared_rows)
    detection.to_csv(artifacts["detection"], index=False)
    shared.to_csv(artifacts["shared_label_transfer"], index=False)
    _summarize_baseline_table(
        detection,
        metrics=("auroc", "auprc", "auprc_baseline", "median_absent", "median_shared"),
    ).to_csv(artifacts["detection_summary"], index=False)
    _summarize_baseline_table(
        shared,
        metrics=(
            "forced_accuracy",
            "forced_macro_f1",
            "post_abstention_accuracy",
            "post_abstention_macro_f1",
            "coverage",
            "shared_false_abstention_rate",
        ),
    ).to_csv(artifacts["shared_label_transfer_summary"], index=False)
    manifest_path = baseline_root / "baseline_metrics_manifest.yaml"
    _write_natural_baseline_comparison_report(
        detection=detection,
        transfer=shared,
        report_path=artifacts["report"],
        manifest_path=manifest_path,
        artifacts=artifacts,
        coreot_full_operating_point=coreot_full_operating_point,
        uniform_uot_tau_source_by_endpoint=uniform_uot_tau_source_by_endpoint,
        method_order=[
            "coreot_full",
            "coreot_constant_tau",
            "coreot_match_only",
            "prior_only",
            "uniform_uot",
            "nn",
            *map(str, baseline_config.get("external", ())),
        ],
    )
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_natural_baselines",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "internal_candidate_set": internal_candidate,
                "external_candidate_set": "external_reference_mapping",
                "ifit_b_primary_scope": "within_broad",
                "proliferating_primary_scope": "global_all_query",
                "threshold_applicability": "undefined_without_matched_full_reference",
                "summary_variability": "descriptive_across_endpoints_not_biological_replicates",
                "coreot_full_operating_point": coreot_full_operating_point,
                "coreot_full_convergence_required": True,
                "required_converged_internal_methods": sorted(
                    required_converged_internal_methods
                ),
                "internal_transport_convergence_required": True,
                "uniform_uot_tau_source_by_endpoint": (
                    uniform_uot_tau_source_by_endpoint
                ),
            },
        ),
    )
    return StageResult("aggregate_natural_baselines", baseline_root, artifacts, manifest_path)


def _write_natural_baseline_comparison_report(
    *,
    detection: pd.DataFrame,
    transfer: pd.DataFrame,
    report_path: Path,
    manifest_path: Path,
    artifacts: dict[str, Path],
    coreot_full_operating_point: dict[str, Any],
    uniform_uot_tau_source_by_endpoint: dict[str, float | str],
    method_order: list[str],
) -> None:
    merged = detection.merge(
        transfer.drop(columns=["candidate_set"]),
        on=["run_id", "holdout_label", "method"],
        how="left",
        validate="one_to_one",
    )
    order = {method: index for index, method in enumerate(method_order)}
    merged["method_order"] = merged["method"].map(order).fillna(len(order))
    columns = [
        ("method", "Method"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("median_absent", "Median endpoint"),
        ("median_shared", "Median shared"),
        ("forced_accuracy", "Forced accuracy"),
        ("forced_macro_f1", "Forced macro-F1"),
        ("coverage", "Coverage"),
        ("shared_false_abstention_rate", "Shared false-abstention rate"),
    ]
    point = coreot_full_operating_point
    content = [
        "# Mouse-spleen natural-mismatch baseline comparison",
        (
            "This pipeline audit report is generated from the endpoint-level "
            "detection and shared-cell label-transfer tables. It is not a "
            "replicate-based statistical comparison."
        ),
        (
            "Sources: "
            f"[detection by endpoint](tables/{artifacts['detection'].name}); "
            f"[detection summary](tables/{artifacts['detection_summary'].name}); "
            "[shared-cell label transfer by endpoint]"
            f"(tables/{artifacts['shared_label_transfer'].name}); "
            "[shared-cell label-transfer summary]"
            f"(tables/{artifacts['shared_label_transfer_summary'].name}); "
            f"[aggregation manifest]({manifest_path.name})."
        ),
        (
            "Canonical `coreot_full` operating point: "
            f"`tau_min={float(point['tau_min']):g}`, "
            f"`tau_max={float(point['tau_max']):g}`, "
            f"`tau_target={float(point['tau_target']):g}`, "
            f"`alpha={float(point['alpha']):g}`, and "
            f"`max_iterations={int(point['max_iterations'])}`. The comparison "
            "includes `coreot_full` only after convergence is verified."
        ),
        (
            "Uniform UOT query penalties: "
            + ", ".join(
                f"`{endpoint}`="
                + (
                    "empirical mean matched to CoRe-OT"
                    if tau_source == "matched_coreot_mean"
                    else f"`tau_source={float(tau_source):g}`"
                )
                for endpoint, tau_source in uniform_uot_tau_source_by_endpoint.items()
            )
            + "."
        ),
        (
            "No calibrated abstention threshold is available for this natural-"
            "mismatch comparison. Consequently, coverage is one for every "
            "applicable method, post-abstention metrics duplicate forced-label "
            "metrics and are omitted, and `prior_only` label-transfer entries "
            "are reported as `NA`."
        ),
    ]
    for endpoint in ("Ifit B", "Proliferating"):
        endpoint_rows = merged.loc[merged["holdout_label"].eq(endpoint)].sort_values(
            ["method_order", "method"]
        )
        content.extend(
            [
                f"## {endpoint}",
                _markdown_metric_table(endpoint_rows, columns),
            ]
        )
        coreot = endpoint_rows.loc[endpoint_rows["method"].eq("coreot_full")]
        if len(coreot) != 1:
            raise MouseSpleenConfigError(
                f"Expected one coreot_full baseline row for endpoint={endpoint!r}"
            )
        row = coreot.iloc[0]
        if endpoint == "Ifit B":
            interpretation = (
                "For `coreot_full`, the endpoint median query-marginal deficit is lower "
                "than the shared-cell median and the AUROC is below 0.5. Under "
                "this fitted model, the deficit does not provide useful ranking "
                "for the Ifit B endpoint."
            )
        else:
            interpretation = (
                "For `coreot_full`, the endpoint median query-marginal deficit is higher "
                "than the shared-cell median, with AUROC "
                f"{float(row['auroc']):.2f} and AUPRC {float(row['auprc']):.2f}. "
                "This is descriptive evidence of weak correspondence for the "
                "Proliferating endpoint under the fitted model, not proof of "
                "biological absence or novelty."
            )
        content.append(interpretation)
    content.extend(
        [
            "## Cross-endpoint summaries",
            (
                "The summary CSVs average two biologically different endpoints. "
                "Their means and standard deviations are descriptive only and "
                "must not be interpreted as replicate-based uncertainty, "
                "confidence intervals, or significance estimates."
            ),
        ]
    )
    report_path.write_text("\n\n".join(content) + "\n", encoding="utf-8")


def _natural_sensitivity_grid(config: dict[str, Any]) -> tuple[list[float], list[float]]:
    natural_config = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    sensitivity = _mapping(natural_config, "sensitivity")
    alpha_values = [float(value) for value in sensitivity.get("alpha_values", range(11))]
    tau_values = [float(value) for value in sensitivity.get("tau_values", range(1, 10))]
    if not alpha_values or not tau_values:
        raise MouseSpleenConfigError("Natural sensitivity alpha_values and tau_values cannot be empty")
    if any(value < 0.0 for value in alpha_values):
        raise MouseSpleenConfigError("Natural sensitivity alpha_values must be nonnegative")
    if any(value <= 0.0 for value in tau_values):
        raise MouseSpleenConfigError("Natural sensitivity tau_values must be positive")
    return alpha_values, tau_values


def _natural_target_penalty_settings(config: dict[str, Any]) -> tuple[float, int]:
    natural = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    settings = natural.get("target_penalty_sensitivity", {})
    if not isinstance(settings, dict):
        raise MouseSpleenConfigError(
            "natural_mismatch.target_penalty_sensitivity must be a mapping"
        )
    tau_target = float(settings.get("tau_target", 8.0))
    max_iterations = int(settings.get("max_iterations", 5000))
    if tau_target <= 0.0 or max_iterations <= 0:
        raise MouseSpleenConfigError(
            "Natural target-penalty sensitivity settings must be positive"
        )
    return tau_target, max_iterations


def _natural_sensitivity_max_iterations(
    config: dict[str, Any], endpoint: str
) -> int:
    natural_config = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    sensitivity = _mapping(natural_config, "sensitivity")
    transport_default = int(_mapping(config, "transport").get("max_iterations", 2000))
    endpoint_values = sensitivity.get("max_iterations_by_endpoint", {})
    if not isinstance(endpoint_values, dict):
        raise MouseSpleenConfigError(
            "natural_mismatch.sensitivity.max_iterations_by_endpoint must be a mapping"
        )
    value = int(
        endpoint_values.get(
            endpoint,
            sensitivity.get("max_iterations", transport_default),
        )
    )
    if value <= 0:
        raise MouseSpleenConfigError("Natural sensitivity max iterations must be positive")
    return value


def _run_natural_constant_tau_sensitivity(
    config: dict[str, Any],
    *,
    root_name: str = "constant_tau_alpha",
    tau_target_override: float | None = None,
    max_iterations_override: int | None = None,
    endpoints: set[str] | None = None,
) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    sensitivity_root = (
        output_root / "natural_mismatch/sensitivity" / root_name
    )
    checkpoint_root = sensitivity_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    alpha_values, tau_values = _natural_sensitivity_grid(config)
    transport = _mapping(config, "transport")
    epsilon = float(transport.get("epsilon", 0.05))
    original_max_iter = int(transport.get("max_iterations", 2000))
    tol = float(transport.get("tolerance", 1.0e-6))
    eta = float(transport.get("eta", 1.0e-12))
    prior_adjustment = _mapping(_mapping(config, "scoring"), "prior_adjustment")
    if not bool(prior_adjustment.get("enabled", False)):
        raise MouseSpleenConfigError(
            "Natural tilde-u sensitivity requires scoring.prior_adjustment.enabled=true"
        )
    prior_adjustment_n_folds = int(prior_adjustment.get("n_folds", 5))
    prior_adjustment_stratify = bool(
        prior_adjustment.get("stratify_by_anchor", False)
    )
    prior_adjustment_min_anchor_size = int(
        prior_adjustment.get("min_anchor_size", 20)
    )
    endpoint_manifests = [
        manifest
        for manifest in _natural_endpoint_manifests(config)
        if endpoints is None
        or str(manifest["metadata"]["natural_endpoint"]) in endpoints
    ]
    if not endpoint_manifests:
        raise MouseSpleenConfigError("Natural sensitivity endpoint filter is empty")
    artifacts: dict[str, Path] = {}
    for endpoint_manifest in endpoint_manifests:
        endpoint = str(endpoint_manifest["metadata"]["natural_endpoint"])
        max_iter = (
            int(max_iterations_override)
            if max_iterations_override is not None
            else _natural_sensitivity_max_iterations(config, endpoint)
        )
        candidate = str(endpoint_manifest["metadata"]["candidate_set"])
        run_root = Path(str(endpoint_manifest["artifacts"]["run_root"]))
        condition = "natural_mismatch"
        checkpoint_path = checkpoint_root / f"{_slug(endpoint)}.csv"
        columns = [
            "run_id", "natural_endpoint", "condition", "candidate_set", "method",
            "alpha", "tau", "tau_target", "max_iterations", "converged",
            "n_iterations",
            "runtime_seconds",
            "evaluation_scope", "n_query", "n_positive", "auroc", "auprc",
            "auprc_baseline", "median_absent", "median_shared",
            "u_tilde_auroc", "u_tilde_auprc", "u_tilde_auprc_baseline",
            "u_tilde_median_absent", "u_tilde_median_shared",
            "u_tilde_model", "u_tilde_cross_fitting", "u_tilde_n_folds",
            "u_tilde_fold_source", "u_tilde_anchor_stratification",
            "u_tilde_fallback",
            "shared_forced_accuracy", "shared_forced_macro_f1",
        ]
        if checkpoint_path.is_file():
            checkpoint = pd.read_csv(checkpoint_path)
            if "u_tilde_auroc" not in checkpoint:
                checkpoint = pd.DataFrame(columns=columns)
            if "n_iterations" not in checkpoint:
                checkpoint["n_iterations"] = np.nan
            if "max_iterations" not in checkpoint:
                checkpoint["max_iterations"] = original_max_iter
            if "tau_target" not in checkpoint:
                checkpoint["tau_target"] = checkpoint["tau"].astype(float)
            checkpoint = checkpoint.loc[
                checkpoint["converged"].astype(bool)
                | checkpoint["max_iterations"].astype(int).ge(max_iter)
            ].copy()
        else:
            checkpoint = pd.DataFrame(columns=columns)
        completed = {
            (float(row.alpha), float(row.tau))
            for row in checkpoint.itertuples(index=False)
        }
        candidates = pd.read_parquet(
            run_root / f"candidates/{condition}/{candidate}/candidate_edges.parquet"
        )
        profile = run_root / f"derived/{condition}/prior_profiles/default"
        source_priors = pd.read_csv(profile / "source_priors.csv")
        target_priors = pd.read_csv(profile / "target_priors.csv")
        truth = pd.read_csv(
            run_root / f"benchmark/{condition}/evaluation_truth/query_truth.csv"
        )
        for alpha in alpha_values:
            for tau in tau_values:
                if (alpha, tau) in completed:
                    continue
                method_root = (
                    sensitivity_root
                    / "tmp"
                    / run_root.name
                    / f"alpha_{alpha:g}_tau_{tau:g}"
                )
                method_root.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                _run_coreot_constant_tau(
                    method_root,
                    condition,
                    candidates,
                    source_priors,
                    target_priors,
                    {
                        "name": "coreot_constant_tau",
                        "epsilon": epsilon,
                        "tau_source": tau,
                        "tau_target": (
                            float(tau_target_override)
                            if tau_target_override is not None
                            else tau
                        ),
                        "alpha": alpha,
                        "max_iter": max_iter,
                        "tol": tol,
                        "eta": eta,
                    },
                )
                runtime = time.perf_counter() - started
                scores = pd.read_parquet(method_root / "cell_transport_scores.parquet")
                method_manifest = _read_manifest_payload(
                    method_root / "transport_manifest.yaml"
                )
                calibration_input = scores.loc[:, ["cell_id", "u"]].merge(
                    source_priors.loc[
                        :, ["cell_id", "prior_risk", "anchor_class_pred"]
                    ],
                    on="cell_id",
                    validate="one_to_one",
                )
                adjusted = compute_prior_adjusted_deficit(
                    calibration_input,
                    n_folds=prior_adjustment_n_folds,
                    stratify_by_anchor=prior_adjustment_stratify,
                    min_anchor_size=prior_adjustment_min_anchor_size,
                )
                scores = scores.merge(
                    pd.DataFrame(
                        {
                            "cell_id": calibration_input["cell_id"].astype(str),
                            "u_tilde": adjusted.u_tilde.astype(float),
                        }
                    ),
                    on="cell_id",
                    validate="one_to_one",
                )
                joined = truth.merge(scores, on="cell_id", validate="one_to_one")
                scoped = joined.loc[_natural_primary_scope(endpoint, joined)]
                positive = scoped["true_label"].astype(str).eq(endpoint)
                shared = joined.loc[joined["is_shared_state"].astype(bool)]
                shared_true = shared["true_label"].astype(str)
                shared_forced = shared["forced_label"].fillna("").astype(str)
                row = {
                    "run_id": run_root.name,
                    "natural_endpoint": endpoint,
                    "condition": condition,
                    "candidate_set": candidate,
                    "method": "coreot_constant_tau",
                    "alpha": alpha,
                    "tau": tau,
                    "tau_target": (
                        float(tau_target_override)
                        if tau_target_override is not None
                        else tau
                    ),
                    "max_iterations": max_iter,
                    "converged": bool(
                        method_manifest.get("metadata", {}).get("converged", False)
                    ),
                    "n_iterations": int(
                        method_manifest.get("metadata", {}).get("n_iter", 0)
                    ),
                    "runtime_seconds": runtime,
                    "evaluation_scope": (
                        "within_broad" if endpoint == "Ifit B" else "global_all_query"
                    ),
                    "n_query": int(len(scoped)),
                    "n_positive": int(positive.sum()),
                    "auroc": safe_auroc(positive, scoped["u"]),
                    "auprc": safe_auprc(positive, scoped["u"]),
                    "auprc_baseline": float(positive.mean()),
                    "median_absent": safe_median(scoped.loc[positive, "u"]),
                    "median_shared": safe_median(scoped.loc[~positive, "u"]),
                    "u_tilde_auroc": safe_auroc(positive, scoped["u_tilde"]),
                    "u_tilde_auprc": safe_auprc(positive, scoped["u_tilde"]),
                    "u_tilde_auprc_baseline": float(positive.mean()),
                    "u_tilde_median_absent": safe_median(
                        scoped.loc[positive, "u_tilde"]
                    ),
                    "u_tilde_median_shared": safe_median(
                        scoped.loc[~positive, "u_tilde"]
                    ),
                    "u_tilde_model": str(adjusted.metadata.get("model", "")),
                    "u_tilde_cross_fitting": str(
                        adjusted.metadata.get("cross_fitting", "")
                    ),
                    "u_tilde_n_folds": int(adjusted.metadata.get("n_folds", 0)),
                    "u_tilde_fold_source": str(
                        adjusted.metadata.get("fold_source", "")
                    ),
                    "u_tilde_anchor_stratification": str(
                        adjusted.metadata.get("anchor_stratification", "")
                    ),
                    "u_tilde_fallback": str(
                        adjusted.metadata.get("fallback", "")
                    ),
                    "shared_forced_accuracy": float(
                        accuracy_score(shared_true, shared_forced)
                    ),
                    "shared_forced_macro_f1": fixed_true_label_macro_f1(
                        shared_true, shared_forced
                    ),
                }
                new_row = pd.DataFrame([row], columns=columns)
                checkpoint = (
                    new_row
                    if checkpoint.empty
                    else pd.concat([checkpoint, new_row], ignore_index=True)
                )
                checkpoint.to_csv(checkpoint_path, index=False)
                shutil.rmtree(method_root)
        checkpoint.to_csv(checkpoint_path, index=False)
        artifacts[_slug(endpoint)] = checkpoint_path
    tmp_root = sensitivity_root / "tmp"
    if tmp_root.is_dir() and not any(tmp_root.rglob("*")):
        shutil.rmtree(tmp_root)
    manifest_path = sensitivity_root / "run_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_natural_constant_tau_sensitivity",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "method": "coreot_constant_tau",
                "secondary_score": "u_tilde_cross_fitted_isotonic_residual",
                "alpha_values": alpha_values,
                "tau_values": tau_values,
                "tau_target_policy": (
                    f"fixed_{float(tau_target_override):g}"
                    if tau_target_override is not None
                    else "matched_to_source_tau"
                ),
                "max_iterations_by_endpoint": {
                    str(endpoint_manifest["metadata"]["natural_endpoint"]): (
                        int(max_iterations_override)
                        if max_iterations_override is not None
                        else _natural_sensitivity_max_iterations(
                            config,
                            str(endpoint_manifest["metadata"]["natural_endpoint"]),
                        )
                    )
                    for endpoint_manifest in endpoint_manifests
                },
                "n_expected": len(artifacts) * len(alpha_values) * len(tau_values),
                "checkpointed": True,
            },
        ),
    )
    return StageResult(
        "run_natural_constant_tau_sensitivity",
        sensitivity_root,
        artifacts,
        manifest_path,
    )


def _aggregate_natural_constant_tau_sensitivity(
    config: dict[str, Any],
    *,
    root_name: str = "constant_tau_alpha",
    tau_target_override: float | None = None,
    max_iterations_override: int | None = None,
) -> StageResult:
    run_result = _run_natural_constant_tau_sensitivity(
        config,
        root_name=root_name,
        tau_target_override=tau_target_override,
        max_iterations_override=max_iterations_override,
    )
    sensitivity_root = run_result.root
    tables_root = sensitivity_root / "tables"
    figures_root = sensitivity_root / "figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    detection = pd.concat(
        [pd.read_csv(path) for path in run_result.artifacts.values()],
        ignore_index=True,
    ).sort_values(["natural_endpoint", "alpha", "tau"])
    detection_path = tables_root / "detection_by_run.csv"
    detection["valid_for_interpretation"] = detection["converged"].astype(bool)
    detection.to_csv(detection_path, index=False)
    summary = detection.copy()
    summary.insert(summary.columns.get_loc("auroc"), "n_runs", 1)
    invalid = ~summary["valid_for_interpretation"]
    summary.loc[
        invalid,
        [
            "auroc",
            "auprc",
            "median_absent",
            "median_shared",
            "u_tilde_auroc",
            "u_tilde_auprc",
            "u_tilde_median_absent",
            "u_tilde_median_shared",
            "shared_forced_accuracy",
            "shared_forced_macro_f1",
        ],
    ] = np.nan
    summary_path = tables_root / "detection_summary.csv"
    summary.to_csv(summary_path, index=False)
    endpoints = list(detection["natural_endpoint"].drop_duplicates())
    figure, axes = plt.subplots(
        len(endpoints), 2, figsize=(10, max(4.0, 3.6 * len(endpoints))), squeeze=False
    )
    for row_index, endpoint in enumerate(endpoints):
        endpoint_frame = summary.loc[summary["natural_endpoint"].eq(endpoint)]
        for column_index, metric in enumerate(("auroc", "auprc")):
            pivot = endpoint_frame.pivot(index="alpha", columns="tau", values=metric)
            axis = axes[row_index, column_index]
            image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
            axis.set_xticks(range(len(pivot.columns)), labels=[f"{x:g}" for x in pivot.columns])
            axis.set_yticks(range(len(pivot.index)), labels=[f"{x:g}" for x in pivot.index])
            axis.set_xlabel("tau")
            axis.set_ylabel("alpha")
            axis.set_title(f"{endpoint}: {metric.upper()}")
            figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure_path = figures_root / "heatmaps.png"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    report_path = sensitivity_root / "report.md"
    report_path.write_text(
        "# Natural-mismatch constant-tau sensitivity\n\n"
        "Each grid point uses query-marginal deficit `u`. Ifit B is evaluated "
        "within the B broad class; Proliferating is evaluated over all query cells. "
        "There is one deterministic dataset per endpoint, so `n_runs = 1` and the "
        "summary does not estimate biological variability. Grid points that did not "
        "converge within the configured iteration cap are retained in `detection_by_run.csv` "
        "as diagnostics, set to missing in `detection_summary.csv`, and masked in the "
        "heatmaps.\n",
        encoding="utf-8",
    )
    artifacts = {
        "detection_by_run": detection_path,
        "detection_summary": summary_path,
        "heatmaps": figure_path,
        "report": report_path,
    }
    manifest_path = sensitivity_root / "manifest.yaml"
    alpha_values, tau_values = _natural_sensitivity_grid(config)
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_natural_constant_tau_sensitivity",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "method": "coreot_constant_tau",
                "primary_score": "u",
                "alpha_values": alpha_values,
                "tau_values": tau_values,
                "tau_target_policy": (
                    f"fixed_{float(tau_target_override):g}"
                    if tau_target_override is not None
                    else "matched_to_source_tau"
                ),
                "n_completed": int(len(detection)),
                "n_expected": len(endpoints) * len(alpha_values) * len(tau_values),
                "n_converged": int(detection["converged"].astype(bool).sum()),
                "nonconverged_policy": "diagnostic_only_excluded_from_summary_and_heatmaps",
                "summary_variability": "none_single_deterministic_dataset_per_endpoint",
            },
        ),
    )
    return StageResult(
        "aggregate_natural_constant_tau_sensitivity",
        sensitivity_root,
        artifacts,
        manifest_path,
    )


def _natural_match_only_settings(
    config: dict[str, Any],
) -> tuple[list[tuple[float, float]], float, int]:
    natural = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    sensitivity = _mapping(natural, "match_only_sensitivity")
    values = sorted(
        {float(value) for value in sensitivity.get(
            "tau_values", (0.05, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
        )}
    )
    if not values or any(value <= 0.0 for value in values):
        raise MouseSpleenConfigError(
            "Natural match-only sensitivity tau_values must be nonempty and positive"
        )
    pairs = [
        (tau_min, tau_max)
        for tau_min in values
        for tau_max in values
        if tau_min <= tau_max
    ]
    tau_target = float(sensitivity.get("tau_target", 1.0))
    max_iter = int(sensitivity.get("max_iterations", 3500))
    if tau_target <= 0.0 or max_iter <= 0:
        raise MouseSpleenConfigError(
            "Natural match-only tau_target and max_iterations must be positive"
        )
    return pairs, tau_target, max_iter


def _run_natural_match_only_sensitivity(config: dict[str, Any]) -> StageResult:
    return _run_natural_match_only_sensitivity_variant(config)


def _natural_full_tau_range_settings(
    config: dict[str, Any],
) -> tuple[list[tuple[float, float]], float, float, int]:
    natural = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    sensitivity = _mapping(natural, "full_tau_range_sensitivity")
    values = sorted(
        {
            float(value)
            for value in sensitivity.get(
                "tau_values", (0.05, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
            )
        }
    )
    if not values or any(value <= 0.0 for value in values):
        raise MouseSpleenConfigError(
            "Natural full tau-range tau_values must be nonempty and positive"
        )
    pairs = [
        (tau_min, tau_max)
        for tau_min in values
        for tau_max in values
        if tau_min <= tau_max
    ]
    alpha = float(sensitivity.get("alpha", 20.0))
    tau_target = float(sensitivity.get("tau_target", 8.0))
    max_iter = int(sensitivity.get("max_iterations", 5000))
    if alpha <= 0.0 or tau_target <= 0.0 or max_iter <= 0:
        raise MouseSpleenConfigError(
            "Natural full tau-range alpha, tau_target, and max_iterations "
            "must be positive"
        )
    return pairs, alpha, tau_target, max_iter


def _write_fixed_natural_source_priors(
    config: dict[str, Any],
    *,
    run_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    condition = "natural_mismatch"
    provider_root = run_root / f"embeddings/{condition}/mouse_spleen_provider"
    cells = pd.read_csv(provider_root / "embedding_cells.csv")
    embedding = np.load(provider_root / "embedding.npy")
    if len(cells) != len(embedding):
        raise MouseSpleenConfigError(
            "Natural-mismatch embedding rows and cell metadata are misaligned"
        )
    embedding_by_id = dict(
        zip(cells["cell_id"].astype(str), embedding, strict=True)
    )
    query = cells.loc[cells["domain"].eq("query"), ["cell_id"]].copy()
    reference = pd.read_csv(
        run_root / f"benchmark/{condition}/model_visible/target_labels.csv"
    )
    natural = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    if "priors" not in natural:
        raise MouseSpleenConfigError(
            "Full tau-range sensitivity requires experiments.natural_mismatch.priors"
        )
    configured = _fit_source_priors(
        config,
        query=query,
        reference=reference,
        embedding_by_id=embedding_by_id,
        priors_config=_mapping(natural, "priors"),
    )
    default_path = (
        run_root / f"derived/{condition}/prior_profiles/default/source_priors.csv"
    )
    default = pd.read_csv(default_path)
    anchor_columns = sorted(
        column
        for column in configured.columns
        if column.startswith("anchor_probability::")
    )
    if anchor_columns != sorted(
        column
        for column in default.columns
        if column.startswith("anchor_probability::")
    ):
        raise MouseSpleenConfigError(
            "Configured and default natural priors have different anchor classes"
        )
    default_aligned = configured[["cell_id"]].merge(
        default[["cell_id", "rho", *anchor_columns]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    compared_columns = ["rho", *anchor_columns]
    if default_aligned[compared_columns].isna().any().any() or not np.allclose(
        configured[compared_columns].to_numpy(dtype=float),
        default_aligned[compared_columns].to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise MouseSpleenConfigError(
            "Configured natural prior differs from the stored default; rerun the "
            "canonical natural-mismatch endpoints before the full tau-range sweep"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configured.to_csv(output_path, index=False)
    recipe = str(configured["rho_recipe"].iloc[0])
    manifest_path = output_path.with_name("prior_manifest.yaml")
    write_manifest(
        manifest_path,
        Manifest(
            stage="derive_natural_fixed_default_rho",
            artifacts={
                "source_priors": str(output_path),
                "default_source_priors": str(default_path),
            },
            metadata={
                "source_priors_sha256": sha256_file(output_path),
                "default_source_priors_sha256": sha256_file(default_path),
                "rho_recipe": recipe,
                "anchor_classifier": {
                    "C": float(configured["anchor_classifier_C"].iloc[0]),
                    "max_iter": int(configured["anchor_classifier_max_iter"].iloc[0]),
                    "random_state": int(
                        configured["anchor_classifier_random_state"].iloc[0]
                    ),
                },
                "matchability_classifier": {
                    "C": float(configured["matchability_classifier_C"].iloc[0]),
                    "max_iter": int(
                        configured["matchability_classifier_max_iter"].iloc[0]
                    ),
                    "random_state": int(
                        configured["matchability_classifier_random_state"].iloc[0]
                    ),
                },
                "fixed_default_rho_verified": True,
                "n_query": int(len(configured)),
            },
        ),
    )
    return configured


def _run_natural_full_tau_range_sensitivity(
    config: dict[str, Any],
) -> StageResult:
    pairs, alpha, tau_target, max_iter = _natural_full_tau_range_settings(config)
    return _run_natural_match_only_sensitivity_variant(
        config,
        root_name="full_tau_range_alpha_20_tau_target_8",
        tau_target_override=tau_target,
        max_iterations_override=max_iter,
        method_name="coreot_full",
        alpha=alpha,
        fixed_default_rho=True,
        pairs_override=pairs,
    )


def _run_natural_match_only_sensitivity_variant(
    config: dict[str, Any],
    *,
    root_name: str = "match_only_tau_range",
    tau_target_override: float | None = None,
    max_iterations_override: int | None = None,
    method_name: str = "coreot_match_only",
    alpha: float = 0.0,
    fixed_default_rho: bool = False,
    pairs_override: list[tuple[float, float]] | None = None,
    endpoints: set[str] | None = None,
    retain_fit_artifacts: bool = False,
    retention_exempt_fit_roots: dict[tuple[float, float], Path] | None = None,
) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    sensitivity_root = (
        output_root / "natural_mismatch/sensitivity" / root_name
    )
    checkpoint_root = sensitivity_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pairs, tau_target, max_iter = _natural_match_only_settings(config)
    if pairs_override is not None:
        pairs = pairs_override
    if tau_target_override is not None:
        tau_target = float(tau_target_override)
    if max_iterations_override is not None:
        max_iter = int(max_iterations_override)
    configured_pairs = set(pairs)
    retention_exemptions = {
        pair: Path(path)
        for pair, path in (retention_exempt_fit_roots or {}).items()
    }
    if retention_exemptions and not retain_fit_artifacts:
        raise MouseSpleenConfigError(
            "Sensitivity retention exemptions require retain_fit_artifacts=True"
        )
    if not set(retention_exemptions) <= configured_pairs:
        raise MouseSpleenConfigError(
            "Sensitivity retention exemptions must belong to the configured grid"
        )
    transport = _mapping(config, "transport")
    epsilon = float(transport.get("epsilon", 0.05))
    tol = float(transport.get("tolerance", 1.0e-6))
    eta = float(transport.get("eta", 1.0e-12))
    prior_adjustment = _mapping(_mapping(config, "scoring"), "prior_adjustment")
    if not bool(prior_adjustment.get("enabled", False)):
        raise MouseSpleenConfigError(
            "Natural match-only sensitivity requires prior adjustment"
        )
    n_folds = int(prior_adjustment.get("n_folds", 5))
    stratify = bool(prior_adjustment.get("stratify_by_anchor", False))
    min_anchor_size = int(prior_adjustment.get("min_anchor_size", 20))
    columns = [
        "run_id", "natural_endpoint", "condition", "candidate_set", "method",
        "tau_min", "tau_max", "tau_target", "alpha", "max_iterations",
        "converged", "n_iterations", "runtime_seconds", "evaluation_scope", "n_query",
        "n_positive", "auroc", "auprc", "auprc_baseline", "median_absent",
        "median_shared", "u_tilde_auroc", "u_tilde_auprc",
        "u_tilde_auprc_baseline", "u_tilde_median_absent",
        "u_tilde_median_shared", "u_tilde_model", "u_tilde_cross_fitting",
        "u_tilde_n_folds", "u_tilde_fold_source",
        "u_tilde_anchor_stratification", "u_tilde_fallback",
        "shared_forced_accuracy", "shared_forced_macro_f1",
        "rho_recipe", "source_priors_sha256", "candidate_edges_sha256",
        "anchor_prior_reuse_verified",
        "anchor_classifier_C", "anchor_classifier_max_iter",
        "anchor_classifier_random_state", "matchability_classifier_C",
        "matchability_classifier_max_iter", "matchability_classifier_random_state",
    ]
    endpoint_manifests = [
        manifest
        for manifest in _natural_endpoint_manifests(config)
        if endpoints is None
        or str(manifest["metadata"]["natural_endpoint"]) in endpoints
    ]
    if not endpoint_manifests:
        raise MouseSpleenConfigError("Natural sensitivity endpoint filter is empty")
    if retention_exemptions and len(endpoint_manifests) != 1:
        raise MouseSpleenConfigError(
            "Sensitivity retention exemptions require exactly one endpoint"
        )
    artifacts: dict[str, Path] = {}
    candidate_identities: dict[str, dict[str, str]] = {}
    for endpoint_manifest in endpoint_manifests:
        endpoint = str(endpoint_manifest["metadata"]["natural_endpoint"])
        endpoint_slug = _slug(endpoint)
        candidate = str(endpoint_manifest["metadata"]["candidate_set"])
        run_root = Path(str(endpoint_manifest["artifacts"]["run_root"]))
        condition = "natural_mismatch"
        checkpoint_path = checkpoint_root / f"{_slug(endpoint)}.csv"
        checkpoint = (
            pd.read_csv(checkpoint_path)
            if checkpoint_path.is_file()
            else pd.DataFrame(columns=columns)
        )
        if "n_iterations" not in checkpoint:
            checkpoint["n_iterations"] = np.nan
        candidate_path = (
            run_root / f"candidates/{condition}/{candidate}/candidate_edges.parquet"
        )
        candidates = pd.read_parquet(candidate_path)
        candidate_identities[endpoint_slug] = {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        }
        candidate_edges_hash = candidate_identities[endpoint_slug]["sha256"]
        profile = run_root / f"derived/{condition}/prior_profiles/default"
        fixed_prior_path = (
            sensitivity_root
            / "priors"
            / _slug(endpoint)
            / "source_priors.csv"
        )
        source_priors_path = (
            fixed_prior_path
            if fixed_default_rho
            else profile / "source_priors.csv"
        )
        source_priors = (
            _write_fixed_natural_source_priors(
                config,
                run_root=run_root,
                output_path=fixed_prior_path,
            )
            if fixed_default_rho
            else pd.read_csv(source_priors_path)
        )
        source_priors_hash = sha256_file(source_priors_path)
        if fixed_default_rho and not checkpoint.empty:
            expected_checkpoint = {
                "method": method_name,
                "alpha": alpha,
                "tau_target": tau_target,
                "max_iterations": max_iter,
                "source_priors_sha256": source_priors_hash,
            }
            if retain_fit_artifacts:
                expected_checkpoint["candidate_edges_sha256"] = candidate_edges_hash
            for column, expected in expected_checkpoint.items():
                if column not in checkpoint:
                    raise MouseSpleenConfigError(
                        f"Fixed-rho checkpoint lacks identity column {column!r}: "
                        f"{checkpoint_path}"
                    )
                observed = checkpoint[column]
                matches = (
                    np.isclose(observed.astype(float), float(expected)).all()
                    if isinstance(expected, int | float)
                    else observed.astype(str).eq(str(expected)).all()
                )
                if not matches:
                    raise MouseSpleenConfigError(
                        f"Fixed-rho checkpoint identity mismatch for {column!r}: "
                        f"{checkpoint_path}"
                    )
            checkpoint_provenance = {
                "anchor_classifier_C": float(
                    source_priors["anchor_classifier_C"].iloc[0]
                ),
                "anchor_classifier_max_iter": int(
                    source_priors["anchor_classifier_max_iter"].iloc[0]
                ),
                "anchor_classifier_random_state": int(
                    source_priors["anchor_classifier_random_state"].iloc[0]
                ),
                "matchability_classifier_C": float(
                    source_priors["matchability_classifier_C"].iloc[0]
                ),
                "matchability_classifier_max_iter": int(
                    source_priors["matchability_classifier_max_iter"].iloc[0]
                ),
                "matchability_classifier_random_state": int(
                    source_priors["matchability_classifier_random_state"].iloc[0]
                ),
            }
            checkpoint_provenance_changed = False
            for column, value in checkpoint_provenance.items():
                if column not in checkpoint or not np.isclose(
                    checkpoint[column].astype(float),
                    float(value),
                ).all():
                    checkpoint[column] = value
                    checkpoint_provenance_changed = True
            if checkpoint_provenance_changed:
                checkpoint.to_csv(checkpoint_path, index=False)

        def method_root(tau_min: float, tau_max: float) -> Path:
            storage = "fits" if retain_fit_artifacts else "tmp"
            return (
                sensitivity_root
                / storage
                / run_root.name
                / f"taumin_{tau_min:g}_taumax_{tau_max:g}"
            )

        def complete_fit_artifacts(path: Path) -> bool:
            return all(
                (path / filename).is_file()
                for filename in RETAINED_SENSITIVITY_FIT_FILES
            )

        def expected_method_config(
            tau_min: float,
            tau_max: float,
        ) -> dict[str, object]:
            return {
                "name": method_name,
                "epsilon": epsilon,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "tau_target": tau_target,
                "alpha": alpha,
                "max_iter": max_iter,
                "tol": tol,
                "eta": eta,
            }

        for pair, exempt_root in retention_exemptions.items():
            if not complete_fit_artifacts(exempt_root):
                raise MouseSpleenConfigError(
                    "Sensitivity retention exemption lacks a complete fit bundle: "
                    f"{pair} -> {exempt_root}"
                )
            observed_method_config = yaml.safe_load(
                (exempt_root / "method_params.yaml").read_text(encoding="utf-8")
            )
            if observed_method_config != expected_method_config(*pair):
                raise MouseSpleenConfigError(
                    "Sensitivity retention exemption has unexpected method parameters: "
                    f"{pair} -> {exempt_root}"
                )

        if retain_fit_artifacts:
            keep_rows = []
            for row in checkpoint.itertuples(index=False):
                pair = (float(row.tau_min), float(row.tau_max))
                keep_rows.append(
                    pair in configured_pairs
                    and (
                        pair in retention_exemptions
                        or complete_fit_artifacts(method_root(*pair))
                    )
                )
            if not all(keep_rows):
                checkpoint = checkpoint.loc[keep_rows].copy()
                checkpoint.to_csv(checkpoint_path, index=False)
        completed = {
            (float(row.tau_min), float(row.tau_max))
            for row in checkpoint.itertuples(index=False)
        }
        missing_exemptions = set(retention_exemptions) - completed
        if missing_exemptions:
            raise MouseSpleenConfigError(
                "Sensitivity retention exemptions lack prevalidated checkpoint rows: "
                f"{sorted(missing_exemptions)}"
            )
        target_priors = pd.read_csv(profile / "target_priors.csv")
        truth = pd.read_csv(
            run_root / f"benchmark/{condition}/evaluation_truth/query_truth.csv"
        )
        for tau_min, tau_max in pairs:
            if (tau_min, tau_max) in completed:
                continue
            fit_root = method_root(tau_min, tau_max)
            if fit_root.exists() and not complete_fit_artifacts(fit_root):
                shutil.rmtree(fit_root)
            fit_root.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            run_method = (
                _run_coreot_full
                if method_name == "coreot_full"
                else _run_coreot_match_only
            )
            method_config = expected_method_config(tau_min, tau_max)
            _write_yaml(fit_root / "method_params.yaml", method_config)
            run_method(
                fit_root,
                condition,
                candidates,
                source_priors,
                target_priors,
                method_config,
            )
            runtime = time.perf_counter() - started
            if retain_fit_artifacts and not complete_fit_artifacts(fit_root):
                missing = [
                    filename
                    for filename in RETAINED_SENSITIVITY_FIT_FILES
                    if not (fit_root / filename).is_file()
                ]
                raise MouseSpleenConfigError(
                    f"Retained sensitivity fit lacks artifacts {missing}: {fit_root}"
                )
            scores = pd.read_parquet(fit_root / "cell_transport_scores.parquet")
            method_manifest = _read_manifest_payload(
                fit_root / "transport_manifest.yaml"
            )
            calibration = scores.loc[:, ["cell_id", "u"]].merge(
                source_priors.loc[
                    :, ["cell_id", "prior_risk", "anchor_class_pred"]
                ],
                on="cell_id",
                validate="one_to_one",
            )
            adjusted = compute_prior_adjusted_deficit(
                calibration,
                n_folds=n_folds,
                stratify_by_anchor=stratify,
                min_anchor_size=min_anchor_size,
            )
            scores = scores.merge(
                pd.DataFrame(
                    {
                        "cell_id": calibration["cell_id"].astype(str),
                        "u_tilde": adjusted.u_tilde.astype(float),
                    }
                ),
                on="cell_id",
                validate="one_to_one",
            )
            joined = truth.merge(scores, on="cell_id", validate="one_to_one")
            scoped = joined.loc[_natural_primary_scope(endpoint, joined)]
            positive = scoped["true_label"].astype(str).eq(endpoint)
            shared = joined.loc[joined["is_shared_state"].astype(bool)]
            shared_true = shared["true_label"].astype(str)
            shared_forced = shared["forced_label"].fillna("").astype(str)
            row = {
                "run_id": run_root.name,
                "natural_endpoint": endpoint,
                "condition": condition,
                "candidate_set": candidate,
                "method": method_name,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "tau_target": tau_target,
                "alpha": alpha,
                "max_iterations": max_iter,
                "converged": bool(
                    method_manifest.get("metadata", {}).get("converged", False)
                ),
                "n_iterations": int(
                    method_manifest.get("metadata", {}).get("n_iter", 0)
                ),
                "runtime_seconds": runtime,
                "evaluation_scope": (
                    "within_broad" if endpoint == "Ifit B" else "global_all_query"
                ),
                "n_query": int(len(scoped)),
                "n_positive": int(positive.sum()),
                "auroc": safe_auroc(positive, scoped["u"]),
                "auprc": safe_auprc(positive, scoped["u"]),
                "auprc_baseline": float(positive.mean()),
                "median_absent": safe_median(scoped.loc[positive, "u"]),
                "median_shared": safe_median(scoped.loc[~positive, "u"]),
                "u_tilde_auroc": safe_auroc(positive, scoped["u_tilde"]),
                "u_tilde_auprc": safe_auprc(positive, scoped["u_tilde"]),
                "u_tilde_auprc_baseline": float(positive.mean()),
                "u_tilde_median_absent": safe_median(
                    scoped.loc[positive, "u_tilde"]
                ),
                "u_tilde_median_shared": safe_median(
                    scoped.loc[~positive, "u_tilde"]
                ),
                "u_tilde_model": str(adjusted.metadata.get("model", "")),
                "u_tilde_cross_fitting": str(
                    adjusted.metadata.get("cross_fitting", "")
                ),
                "u_tilde_n_folds": int(adjusted.metadata.get("n_folds", 0)),
                "u_tilde_fold_source": str(
                    adjusted.metadata.get("fold_source", "")
                ),
                "u_tilde_anchor_stratification": str(
                    adjusted.metadata.get("anchor_stratification", "")
                ),
                "u_tilde_fallback": str(adjusted.metadata.get("fallback", "")),
                "shared_forced_accuracy": float(
                    accuracy_score(shared_true, shared_forced)
                ),
                "shared_forced_macro_f1": fixed_true_label_macro_f1(
                    shared_true, shared_forced
                ),
                "rho_recipe": str(source_priors["rho_recipe"].iloc[0]),
                "source_priors_sha256": source_priors_hash,
                "candidate_edges_sha256": candidate_edges_hash,
                "anchor_prior_reuse_verified": bool(fixed_default_rho),
                "anchor_classifier_C": float(
                    source_priors["anchor_classifier_C"].iloc[0]
                ),
                "anchor_classifier_max_iter": int(
                    source_priors["anchor_classifier_max_iter"].iloc[0]
                ),
                "anchor_classifier_random_state": int(
                    source_priors["anchor_classifier_random_state"].iloc[0]
                ),
                "matchability_classifier_C": float(
                    source_priors["matchability_classifier_C"].iloc[0]
                ),
                "matchability_classifier_max_iter": int(
                    source_priors["matchability_classifier_max_iter"].iloc[0]
                ),
                "matchability_classifier_random_state": int(
                    source_priors["matchability_classifier_random_state"].iloc[0]
                ),
            }
            new_row = pd.DataFrame([row], columns=columns)
            checkpoint = (
                new_row
                if checkpoint.empty
                else pd.concat([checkpoint, new_row], ignore_index=True)
            ).sort_values(["tau_min", "tau_max"])
            checkpoint.to_csv(checkpoint_path, index=False)
            if not retain_fit_artifacts:
                shutil.rmtree(fit_root)
        artifacts[endpoint_slug] = checkpoint_path
        if retain_fit_artifacts:
            artifacts[f"{endpoint_slug}_fit_root"] = (
                sensitivity_root / "fits" / run_root.name
            )
            for pair, exempt_root in sorted(retention_exemptions.items()):
                artifacts[
                    f"{endpoint_slug}_retained_external_taumin_{pair[0]:g}_taumax_{pair[1]:g}"
                ] = exempt_root
    tmp_root = sensitivity_root / "tmp"
    if tmp_root.is_dir() and not any(tmp_root.rglob("*")):
        shutil.rmtree(tmp_root)
    manifest_path = sensitivity_root / "run_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_natural_tau_range_sensitivity",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "method": method_name,
                "alpha": alpha,
                "fixed_default_rho": fixed_default_rho,
                "tau_pairs": [list(pair) for pair in pairs],
                "tau_target": tau_target,
                "max_iterations": max_iter,
                "n_expected": len(endpoint_manifests) * len(pairs),
                "checkpointed": True,
                "retained_fit_artifacts": retain_fit_artifacts,
                "retained_fit_files": list(RETAINED_SENSITIVITY_FIT_FILES),
                "retention_exempt_fit_roots": {
                    f"taumin_{pair[0]:g}_taumax_{pair[1]:g}": str(path)
                    for pair, path in sorted(retention_exemptions.items())
                },
                "candidate_edges": candidate_identities,
            },
        ),
    )
    return StageResult(
        "run_natural_tau_range_sensitivity",
        sensitivity_root,
        artifacts,
        manifest_path,
    )


def _markdown_metric_table(
    frame: pd.DataFrame, columns: list[tuple[str, str]]
) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = row._asdict()
        formatted: list[str] = []
        for column, _ in columns:
            value = values[column]
            if pd.isna(value):
                formatted.append("NA")
            elif isinstance(value, (bool, np.bool_)):
                formatted.append("yes" if bool(value) else "no")
            elif isinstance(value, (int, np.integer)):
                formatted.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.2f}")
            else:
                formatted.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _write_match_only_endpoint_report(
    frame: pd.DataFrame,
    endpoint: str,
    path: Path,
    *,
    target8: pd.DataFrame | None = None,
) -> None:
    frame = frame.loc[frame["natural_endpoint"].eq(endpoint)].sort_values(
        ["tau_min", "tau_max"]
    )
    shared_keys = [
        ("tau_min", "Tau min"),
        ("tau_max", "Tau max"),
        ("converged", "Converged"),
    ]
    detection_columns = [
        *shared_keys,
        ("evaluation_scope", "Scope"),
        ("n_query", "N query"),
        ("n_positive", "N positive"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("auprc_baseline", "AUPRC baseline"),
        ("median_absent", "Median endpoint"),
        ("median_shared", "Median shared"),
    ]
    tilde_columns = [
        *shared_keys,
        ("evaluation_scope", "Scope"),
        ("n_query", "N query"),
        ("n_positive", "N positive"),
        ("u_tilde_auroc", "AUROC"),
        ("u_tilde_auprc", "AUPRC"),
        ("u_tilde_auprc_baseline", "AUPRC baseline"),
        ("u_tilde_median_absent", "Median endpoint"),
        ("u_tilde_median_shared", "Median shared"),
    ]
    transfer_columns = [
        *shared_keys,
        ("shared_forced_accuracy", "Forced accuracy"),
        ("shared_forced_macro_f1", "Forced macro-F1"),
    ]
    target_endpoint = None
    if target8 is not None:
        target_endpoint = target8.loc[
            target8["natural_endpoint"].eq(endpoint)
        ].sort_values(["tau_min", "tau_max"])
    content = (
        f"# {endpoint}: match-only tau-range metrics\n\n"
        "Terminal-iterate metrics are retained when `converged = no`.\n\n"
        "## $u$-based detection metrics — tau target = 1\n\n"
        + _markdown_metric_table(frame, detection_columns)
    )
    if target_endpoint is not None:
        content += (
            "\n\n## $u$-based detection metrics — tau target = 8\n\n"
            + _markdown_metric_table(target_endpoint, detection_columns)
        )
    content += (
        "\n\n## Tilde-u detection metrics — tau target = 1\n\n"
        + _markdown_metric_table(frame, tilde_columns)
    )
    if target_endpoint is not None:
        content += (
            "\n\n## Tilde-u detection metrics — tau target = 8\n\n"
            + _markdown_metric_table(target_endpoint, tilde_columns)
        )
    content += (
        "\n\n## Shared-cell label-transfer metrics — tau target = 1\n\n"
        + _markdown_metric_table(frame, transfer_columns)
    )
    if target_endpoint is not None:
        content += (
            "\n\n## Shared-cell label-transfer metrics — tau target = 8\n\n"
            + _markdown_metric_table(target_endpoint, transfer_columns)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _aggregate_natural_match_only_sensitivity(
    config: dict[str, Any],
    *,
    root_name: str = "match_only_tau_range",
    tau_target_override: float | None = None,
    max_iterations_override: int | None = None,
) -> StageResult:
    run_result = _run_natural_match_only_sensitivity_variant(
        config,
        root_name=root_name,
        tau_target_override=tau_target_override,
        max_iterations_override=max_iterations_override,
    )
    root = run_result.root
    tables_root = root / "tables"
    figures_root = root / "figures"
    reports_root = root / "reports"
    for path in (tables_root, figures_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [pd.read_csv(path) for path in run_result.artifacts.values()],
        ignore_index=True,
    ).sort_values(["natural_endpoint", "tau_min", "tau_max"])
    table_path = tables_root / "metrics_by_grid.csv"
    frame.to_csv(table_path, index=False)
    endpoints = list(frame["natural_endpoint"].drop_duplicates())
    metrics = (
        ("auroc", "$u$-based AUROC"),
        ("auprc", "$u$-based AUPRC"),
        ("u_tilde_auroc", "Tilde u AUROC"),
        ("u_tilde_auprc", "Tilde u AUPRC"),
    )
    figure, axes = plt.subplots(
        len(endpoints), len(metrics), figsize=(16, 3.8 * len(endpoints)), squeeze=False
    )
    tau_values = sorted(set(frame["tau_min"]) | set(frame["tau_max"]))
    for row_index, endpoint in enumerate(endpoints):
        endpoint_frame = frame.loc[frame["natural_endpoint"].eq(endpoint)]
        for column_index, (metric, title) in enumerate(metrics):
            pivot = endpoint_frame.pivot(
                index="tau_min", columns="tau_max", values=metric
            ).reindex(index=tau_values, columns=tau_values)
            axis = axes[row_index, column_index]
            image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
            axis.set_xticks(
                range(len(tau_values)), labels=[f"{value:g}" for value in tau_values]
            )
            axis.set_yticks(
                range(len(tau_values)), labels=[f"{value:g}" for value in tau_values]
            )
            axis.set_xlabel("tau max")
            axis.set_ylabel("tau min")
            axis.set_title(f"{endpoint}: {title}")
            figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure_path = figures_root / "heatmaps.png"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    report_artifacts: dict[str, Path] = {}
    report_frame = frame
    target8_frame: pd.DataFrame | None = None
    report_output_root = reports_root
    if tau_target_override is not None:
        original_root = root.parent / "match_only_tau_range"
        original_table = original_root / "tables/metrics_by_grid.csv"
        if not original_table.is_file():
            raise FileNotFoundError(
                f"Original match-only table is required for paired reports: {original_table}"
            )
        report_frame = pd.read_csv(original_table)
        target8_frame = frame
        report_output_root = original_root / "reports"
    else:
        target8_table = (
            root.parent
            / "match_only_tau_range_tau_target_8/tables/metrics_by_grid.csv"
        )
        if target8_table.is_file():
            target8_frame = pd.read_csv(target8_table)
    for endpoint in endpoints:
        report_path = report_output_root / f"{_slug(endpoint)}.md"
        _write_match_only_endpoint_report(
            report_frame,
            endpoint,
            report_path,
            target8=target8_frame,
        )
        report_artifacts[f"{_slug(endpoint)}_report"] = report_path
    artifacts = {
        "metrics_by_grid": table_path,
        "heatmaps": figure_path,
        **report_artifacts,
    }
    pairs, tau_target, max_iter = _natural_match_only_settings(config)
    if tau_target_override is not None:
        tau_target = float(tau_target_override)
    if max_iterations_override is not None:
        max_iter = int(max_iterations_override)
    manifest_path = root / "manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_natural_match_only_sensitivity",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "method": "coreot_match_only",
                "alpha": 0.0,
                "tau_pairs": [list(pair) for pair in pairs],
                "tau_target": tau_target,
                "max_iterations": max_iter,
                "n_completed": int(len(frame)),
                "n_expected": len(endpoints) * len(pairs),
                "n_converged": int(frame["converged"].astype(bool).sum()),
                "nonconverged_policy": "terminal_metrics_retained_with_flag",
            },
        ),
    )
    return StageResult(
        "aggregate_natural_match_only_sensitivity",
        root,
        artifacts,
        manifest_path,
    )


def _write_full_tau_range_endpoint_report(
    frame: pd.DataFrame,
    endpoint: str,
    path: Path,
    *,
    table_path: Path,
) -> None:
    endpoint_frame = frame.loc[frame["natural_endpoint"].eq(endpoint)].sort_values(
        ["tau_min", "tau_max"]
    )
    shared_keys = [
        ("tau_min", "Tau min"),
        ("tau_max", "Tau max"),
        ("converged", "Converged"),
    ]
    detection_columns = [
        *shared_keys,
        ("evaluation_scope", "Scope"),
        ("n_query", "N query"),
        ("n_positive", "N positive"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("auprc_baseline", "AUPRC baseline"),
        ("median_absent", "Median endpoint"),
        ("median_shared", "Median shared"),
    ]
    tilde_columns = [
        *shared_keys,
        ("evaluation_scope", "Scope"),
        ("n_query", "N query"),
        ("n_positive", "N positive"),
        ("u_tilde_auroc", "AUROC"),
        ("u_tilde_auprc", "AUPRC"),
        ("u_tilde_auprc_baseline", "AUPRC baseline"),
        ("u_tilde_median_absent", "Median endpoint"),
        ("u_tilde_median_shared", "Median shared"),
    ]
    transfer_columns = [
        *shared_keys,
        ("shared_forced_accuracy", "Forced accuracy"),
        ("shared_forced_macro_f1", "Forced macro-F1"),
    ]
    rho_recipe = str(endpoint_frame["rho_recipe"].iloc[0])
    tau_upper = float(max(endpoint_frame["tau_min"].max(), endpoint_frame["tau_max"].max()))
    content = (
        f"# {endpoint}: full CoRe-OT tau-range metrics\n\n"
        "This grid fixes the natural-mismatch default rho, `alpha = 20`, and "
        "`tau_target = 8`. Terminal-iterate metrics are retained when "
        "`converged = no`.\n\n"
        f"The upper bound `{tau_upper:g}` is an exploratory stress-test limit "
        "for stronger source-marginal penalties, not a performance-selected "
        "operating point. This report displays the complete surface without "
        "ranking settings or selecting an optimum.\n\n"
        f"Default rho recipe: `{rho_recipe}`.\n\n"
        f"Full-precision source: `{table_path}`.\n\n"
        "## $u$-based detection metrics\n\n"
        + _markdown_metric_table(endpoint_frame, detection_columns)
        + "\n\n## Tilde-u detection metrics\n\n"
        + _markdown_metric_table(endpoint_frame, tilde_columns)
        + "\n\n## Shared-cell label-transfer metrics\n\n"
        + _markdown_metric_table(endpoint_frame, transfer_columns)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _aggregate_natural_full_tau_range_sensitivity(
    config: dict[str, Any],
) -> StageResult:
    run_result = _run_natural_full_tau_range_sensitivity(config)
    root = run_result.root
    tables_root = root / "tables"
    figures_root = root / "figures"
    reports_root = root / "reports"
    for path in (tables_root, figures_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [pd.read_csv(path) for path in run_result.artifacts.values()],
        ignore_index=True,
    ).sort_values(["natural_endpoint", "tau_min", "tau_max"])
    table_path = tables_root / "metrics_by_grid.csv"
    frame.to_csv(table_path, index=False)
    endpoints = list(frame["natural_endpoint"].drop_duplicates())
    metrics = (
        ("auroc", "$u$-based AUROC"),
        ("auprc", "$u$-based AUPRC"),
        ("u_tilde_auroc", "Tilde u AUROC"),
        ("u_tilde_auprc", "Tilde u AUPRC"),
    )
    tau_values = sorted(set(frame["tau_min"]) | set(frame["tau_max"]))
    figure, axes = plt.subplots(
        len(endpoints),
        len(metrics),
        figsize=(16, 3.8 * len(endpoints)),
        squeeze=False,
    )
    for row_index, endpoint in enumerate(endpoints):
        endpoint_frame = frame.loc[frame["natural_endpoint"].eq(endpoint)]
        for column_index, (metric, title) in enumerate(metrics):
            pivot = endpoint_frame.pivot(
                index="tau_min", columns="tau_max", values=metric
            ).reindex(index=tau_values, columns=tau_values)
            axis = axes[row_index, column_index]
            image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
            axis.set_xticks(
                range(len(tau_values)),
                labels=[f"{value:g}" for value in tau_values],
                rotation=45,
                ha="right",
            )
            axis.set_yticks(
                range(len(tau_values)), labels=[f"{value:g}" for value in tau_values]
            )
            axis.set_xlabel("tau max")
            axis.set_ylabel("tau min")
            axis.set_title(f"{endpoint}: {title}")
            figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure_path = figures_root / "heatmaps.png"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    report_artifacts: dict[str, Path] = {}
    for endpoint in endpoints:
        report_path = reports_root / f"{_slug(endpoint)}.md"
        _write_full_tau_range_endpoint_report(
            frame,
            endpoint,
            report_path,
            table_path=table_path,
        )
        report_artifacts[f"{_slug(endpoint)}_report"] = report_path
    pairs, alpha, tau_target, max_iter = _natural_full_tau_range_settings(config)
    artifacts = {
        "metrics_by_grid": table_path,
        "heatmaps": figure_path,
        **report_artifacts,
    }
    manifest_path = root / "manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_natural_full_tau_range_sensitivity",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "method": "coreot_full",
                "experiment_status": "exploratory_fixed_default_rho",
                "alpha": alpha,
                "alpha_selection": "specified_independently_of_existing_high_alpha_results",
                "tau_pairs": [list(pair) for pair in pairs],
                "tau_target": tau_target,
                "max_iterations": max_iter,
                "rho_recipes": sorted(frame["rho_recipe"].astype(str).unique()),
                "anchor_classifier": {
                    "C": float(frame["anchor_classifier_C"].iloc[0]),
                    "max_iter": int(frame["anchor_classifier_max_iter"].iloc[0]),
                    "random_state": int(
                        frame["anchor_classifier_random_state"].iloc[0]
                    ),
                },
                "matchability_classifier": {
                    "C": float(frame["matchability_classifier_C"].iloc[0]),
                    "max_iter": int(
                        frame["matchability_classifier_max_iter"].iloc[0]
                    ),
                    "random_state": int(
                        frame["matchability_classifier_random_state"].iloc[0]
                    ),
                },
                "source_priors_sha256": sorted(
                    frame["source_priors_sha256"].astype(str).unique()
                ),
                "anchor_prior_reuse_verified": bool(
                    frame["anchor_prior_reuse_verified"].astype(bool).all()
                ),
                "n_completed": int(len(frame)),
                "n_expected": len(endpoints) * len(pairs),
                "n_converged": int(frame["converged"].astype(bool).sum()),
                "nonconverged_policy": "terminal_metrics_retained_with_flag",
            },
        ),
    )
    return StageResult(
        "aggregate_natural_full_tau_range_sensitivity",
        root,
        artifacts,
        manifest_path,
    )


def _natural_metric_columns() -> list[str]:
    return [
        "natural_endpoint",
        "method",
        "score",
        "evaluation_scope",
        "n_query",
        "n_positive",
        "auroc",
        "auprc",
        "auprc_baseline",
        "median_absent",
        "median_shared",
        "restoration_metric_applicability",
    ]


def _build_natural_mismatch_tables(
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = output_root / "natural_mismatch/natural_mismatch_manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Natural-mismatch outputs do not exist; run run_natural_mismatch first: {manifest_path}"
        )
    root_manifest = _read_manifest_payload(manifest_path)
    metric_rows: list[dict[str, object]] = []
    cell_rows: list[pd.DataFrame] = []
    broad_rows: list[dict[str, object]] = []
    for endpoint_manifest_path in root_manifest.get("artifacts", {}).values():
        endpoint_manifest = _read_manifest_payload(Path(str(endpoint_manifest_path)))
        endpoint = str(endpoint_manifest["metadata"]["natural_endpoint"])
        candidate = str(endpoint_manifest["metadata"]["candidate_set"])
        run_root = Path(str(endpoint_manifest["artifacts"]["run_root"]))
        scores = pd.read_parquet(
            run_root / "scoring/natural_mismatch" / candidate / "cell_scores.parquet"
        )
        truth = pd.read_csv(
            run_root / "benchmark/natural_mismatch/evaluation_truth/query_truth.csv"
        )
        source_priors = pd.read_csv(
            run_root / "derived/natural_mismatch/prior_profiles/default/source_priors.csv"
        )
        candidate_edges = pd.read_parquet(
            run_root / "candidates/natural_mismatch" / candidate / "candidate_edges.parquet"
        )
        geometry = (
            candidate_edges.sort_values(["source_cell_id", "distance", "target_cell_id"])
            .drop_duplicates(["source_cell_id", "target_cell_id"])
            .groupby("source_cell_id", sort=False)["distance"]
            .agg(
                nearest_reference_distance="min",
                mean_k_reference_distance=lambda values: float(
                    np.mean(np.sort(values.to_numpy(dtype=float))[:10])
                ),
            )
            .reset_index()
            .rename(columns={"source_cell_id": "cell_id"})
        )
        base = truth.merge(source_priors, on="cell_id", validate="one_to_one").merge(
            geometry, on="cell_id", validate="one_to_one"
        )
        base["is_positive"] = base["true_label"].astype(str).eq(endpoint)
        base["one_minus_max_broad_probability"] = 1.0 - base[
            "pmax_reference_classifier"
        ].astype(float)
        base["broad_probability_entropy"] = 1.0 - base["anchor_confidence"].astype(float)
        support_scores = {
            "nearest_reference_distance": ("geometry", "nearest_reference_distance"),
            "mean_k_reference_distance": ("geometry", "mean_k_reference_distance"),
            "one_minus_max_broad_probability": (
                "broad_prior",
                "one_minus_max_broad_probability",
            ),
            "broad_probability_entropy": ("broad_prior", "broad_probability_entropy"),
            "one_minus_rho": ("prior_only", "prior_risk"),
        }
        endpoint_cell_frames: list[pd.DataFrame] = []
        for score_name, (method, column) in support_scores.items():
            endpoint_cell_frames.append(
                _natural_cell_score_frame(base, endpoint, method, score_name, column)
            )
        for method, method_scores in scores.groupby("method", sort=True):
            method_frame = truth.merge(method_scores, on="cell_id", validate="one_to_one")
            if method_frame["u"].notna().any() and str(method) != "balanced_ot":
                endpoint_cell_frames.append(
                    _natural_cell_score_frame(
                        method_frame, endpoint, str(method), "u", "u"
                    )
                )
            for score_name, column in (
                (
                    "one_minus_max_conditional_transport_label_probability",
                    "label_uncertainty",
                ),
                ("conditional_transport_label_entropy", "label_entropy"),
            ):
                if method_frame[column].notna().any():
                    endpoint_cell_frames.append(
                        _natural_cell_score_frame(
                            method_frame, endpoint, str(method), score_name, column
                        )
                    )
        endpoint_cells = pd.concat(endpoint_cell_frames, ignore_index=True)
        cell_rows.append(endpoint_cells)
        for (method, score_name), frame in endpoint_cells.groupby(
            ["method", "score"], sort=True
        ):
            scopes = [("global_all_query", pd.Series(True, index=frame.index))]
            if endpoint == "Ifit B":
                scopes.insert(
                    0,
                    ("within_broad", frame["true_broad_label"].astype(str).eq("B")),
                )
            for scope, mask in scopes:
                scoped = frame.loc[mask]
                positive = scoped["is_positive"].astype(bool)
                metric_rows.append(
                    {
                        "natural_endpoint": endpoint,
                        "method": method,
                        "score": score_name,
                        "evaluation_scope": scope,
                        "n_query": int(len(scoped)),
                        "n_positive": int(positive.sum()),
                        "auroc": safe_auroc(positive, scoped["value"]),
                        "auprc": safe_auprc(positive, scoped["value"]),
                        "auprc_baseline": (
                            float(positive.mean()) if len(positive) else float("nan")
                        ),
                        "median_absent": safe_median(scoped.loc[positive, "value"]),
                        "median_shared": safe_median(scoped.loc[~positive, "value"]),
                        "restoration_metric_applicability": "undefined",
                    }
                )
        positive_priors = base.loc[base["is_positive"]]
        predicted_counts = positive_priors["anchor_class_pred"].astype(str).value_counts()
        for broad_class, count in predicted_counts.items():
            broad_rows.append(
                {
                    "natural_endpoint": endpoint,
                    "predicted_broad_class": broad_class,
                    "n_positive": int(len(positive_priors)),
                    "predicted_count": int(count),
                    "predicted_fraction": float(count / len(positive_priors)),
                    "median_normalized_broad_entropy": safe_median(
                        1.0 - positive_priors["anchor_confidence"].astype(float)
                    ),
                }
            )
    return (
        pd.DataFrame(metric_rows, columns=_natural_metric_columns()),
        pd.concat(cell_rows, ignore_index=True),
        pd.DataFrame(broad_rows),
    )


def _natural_cell_score_frame(
    frame: pd.DataFrame,
    endpoint: str,
    method: str,
    score: str,
    value_column: str,
) -> pd.DataFrame:
    result = frame.loc[
        frame[value_column].notna(),
        ["cell_id", "true_label", "true_broad_label", value_column],
    ].copy()
    result.insert(0, "natural_endpoint", endpoint)
    result["method"] = method
    result["score"] = score
    result["value"] = result.pop(value_column).astype(float)
    result["is_positive"] = result["true_label"].astype(str).eq(endpoint)
    return result


def _make_natural_mismatch_figure(config: dict[str, Any]) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    natural_root = output_root / "natural_mismatch"
    tables_root = natural_root / "tables"
    cell_path = tables_root / "natural_cell_scores.parquet"
    broad_path = tables_root / "table_natural_broad_predictions.csv"
    for path in (cell_path, broad_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Natural-mismatch figure input does not exist; aggregate first: {path}"
            )
    cells = pd.read_parquet(cell_path)
    broad = pd.read_csv(broad_path)
    endpoints = [str(value) for value in broad["natural_endpoint"].drop_duplicates()]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    deficit = cells.loc[
        cells["method"].eq("coreot_full") & cells["score"].eq("u")
    ]
    box_values = []
    box_labels = []
    for endpoint in endpoints:
        frame = deficit.loc[deficit["natural_endpoint"].eq(endpoint)]
        for positive, label in ((False, "shared"), (True, "endpoint")):
            box_values.append(frame.loc[frame["is_positive"].eq(positive), "value"])
            box_labels.append(f"{endpoint}\n{label}")
    axes[0].boxplot(box_values, tick_labels=box_labels, showfliers=False)
    axes[0].set_ylabel("Query-marginal deficit u")
    axes[0].tick_params(axis="x", rotation=25)
    pivot = broad.pivot(
        index="natural_endpoint",
        columns="predicted_broad_class",
        values="predicted_fraction",
    ).fillna(0.0)
    bottom = np.zeros(len(pivot))
    for broad_class in pivot.columns:
        axes[1].bar(pivot.index, pivot[broad_class], bottom=bottom, label=broad_class)
        bottom += pivot[broad_class].to_numpy(dtype=float)
    axes[1].set_ylabel("Predicted broad-class fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure_root = natural_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_path = figure_root / "natural_mismatch.png"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    manifest_path = figure_root / "natural_mismatch_figure_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="make_natural_mismatch_figure",
            artifacts={"natural_mismatch": str(figure_path)},
            metadata={"generated_from_saved_tables": True},
        ),
    )
    return StageResult(
        "make_natural_mismatch_figure",
        figure_root,
        {"natural_mismatch": figure_path},
        manifest_path,
    )


def _make_figures(config: dict[str, Any]) -> StageResult:
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    metrics_root = output_root / "metrics"
    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    required = {
        "controlled_detection": metrics_root / "table_controlled_detection.csv",
        "support_dose": metrics_root / "table_support_dose.csv",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Figure input does not exist; aggregate metrics first: {path}")
    artifacts: dict[str, Path] = {}
    detection = pd.read_csv(required["controlled_detection"])
    primary = detection.loc[detection["evaluation_scope"].eq("within_broad")]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for method, frame in primary.groupby("method"):
        axis.plot(frame["holdout_label"], frame["auprc"], marker="o", label=method)
    axis.set_ylabel("AUPRC")
    axis.set_xlabel("Held-out fine state")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(fontsize=7)
    figure.tight_layout()
    detection_figure = figures_root / "controlled_detection.png"
    figure.savefig(detection_figure, dpi=300)
    plt.close(figure)
    artifacts["controlled_detection"] = detection_figure

    dose = pd.read_csv(required["support_dose"])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for (holdout, method), frame in dose.groupby(["holdout_label", "method"]):
        axis.plot(frame["support_fraction"], frame["mean"], marker="o", label=f"{holdout}: {method}")
    axis.set_xlabel("Retained reference fraction")
    axis.set_ylabel("Median positive deficit")
    axis.legend(fontsize=6)
    figure.tight_layout()
    dose_figure = figures_root / "support_dose.png"
    figure.savefig(dose_figure, dpi=300)
    plt.close(figure)
    artifacts["support_dose"] = dose_figure
    manifest_path = figures_root / "figures_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="make_figures",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={"generated_from_saved_metrics": True},
        ),
    )
    return StageResult("make_figures", figures_root, artifacts, manifest_path)


def _run_natural_mismatch(config: dict[str, Any]) -> StageResult:
    embedding_result = _prepare_embedding(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    natural_root = output_root / "natural_mismatch"
    natural_root.mkdir(parents=True, exist_ok=True)
    natural_config = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    if not bool(natural_config.get("enabled", True)):
        raise MouseSpleenConfigError("natural-mismatch stage is disabled")
    endpoints_config = _mapping(natural_config, "endpoints")
    endpoints = list(endpoints_config) or [
        str(value) for value in _mapping(config, "labels").get("rna_only_labels", ())
    ]
    rna, atac = _load_canonical_modalities(config)
    shared = [str(value) for value in _mapping(config, "labels").get("shared_labels", ())]
    target = atac.loc[atac["fine_label"].isin(shared)].copy()
    embedding = np.load(embedding_result.artifacts["embedding"])
    cells = pd.read_csv(embedding_result.artifacts["embedding_cells"])
    embedding_by_id = {
        str(cell_id): embedding[int(row)]
        for cell_id, row in zip(cells["cell_id"], cells["row_index"], strict=True)
    }
    configured_methods = natural_config.get(
        "methods", _mapping(config, "methods").get("transport", ())
    )
    methods = list(
        dict.fromkeys([*map(str, configured_methods), "prior_only"])
    )
    graph_config = _mapping(config, "candidate_graph")
    candidate_name = f"mouse_spleen_provider_k{int(graph_config.get('k_source_to_target', 100))}"
    artifacts: dict[str, Path] = {}
    natural_priors = (
        _mapping(natural_config, "priors")
        if "priors" in natural_config
        else _mapping(config, "priors")
    )
    for endpoint in endpoints:
        endpoint_config = _mapping(endpoints_config, endpoint)
        other_endpoints = set(endpoints) - {endpoint}
        query = rna.loc[
            rna["fine_label"].isin([*shared, endpoint])
            & ~rna["fine_label"].isin(other_endpoints)
        ].copy()
        if not query["fine_label"].eq(endpoint).any():
            raise MouseSpleenConfigError(f"Natural endpoint {endpoint!r} has no RNA cells")
        source_priors = _fit_source_priors(
            config,
            query,
            target,
            embedding_by_id,
            priors_config=natural_priors,
        )
        run_id = f"mouse_spleen_natural_{_slug(endpoint)}"
        run_root = output_root / "runs" / run_id
        condition = "natural_mismatch"
        _write_condition_artifacts(
            run_root=run_root,
            condition=condition,
            holdout=endpoint,
            query=query,
            reference=target,
            source_priors=source_priors,
            embedding_by_id=embedding_by_id,
            split_seed=int(_mapping(config, "experiment").get("master_seed", 0)),
            absent_condition=True,
        )
        generated = run_root / "generated_configs"
        generated.mkdir(parents=True, exist_ok=True)
        common = {
            "run_id": run_id,
            "conditions": [condition],
            "outputs": {"root": str(output_root / "runs")},
        }
        candidate_path = generated / "candidates.yaml"
        _write_yaml(
            candidate_path,
            {
                **common,
                "providers": ["mouse_spleen_provider"],
                "candidate_graph": graph_config,
                "cost_scaling": _mapping(config, "cost_scaling"),
            },
        )
        run_candidate_cost(candidate_path)
        transport_path = generated / "transport.yaml"
        _write_yaml(
            transport_path,
            {
                **common,
                "candidate_sets": [
                    {
                        "name": candidate_name,
                        "provider": "mouse_spleen_provider",
                        "prior_profile": "default",
                    }
                ],
                "methods": _transport_method_configs(
                    config,
                    methods,
                    balanced_support=str(
                        natural_config.get("balanced_ot_support", "dense")
                    ),
                    coreot_full_override=(
                        _mapping(natural_config, "coreot_full")
                        if "coreot_full" in natural_config
                        else None
                    ),
                    mean_matched_factorial=bool(
                        natural_config.get("mean_matched_primary_factorial", False)
                    ),
                    uniform_uot_tau_source=(
                        float(endpoint_config["uniform_uot_tau_source"])
                        if "uniform_uot_tau_source" in endpoint_config
                        else None
                    ),
                ),
            },
        )
        run_transport(transport_path)
        score_path = generated / "scoring.yaml"
        _write_yaml(
            score_path,
            {
                **common,
                "candidate_sets": [candidate_name],
                "methods": methods,
                "prior_adjustment": _mapping(config, "scoring").get("prior_adjustment", {}),
                "thresholds": {
                    "theta_H": _mapping(config, "scoring").get("thresholds", {}).get(
                        "theta_H", {"value": 0.8}
                    )
                },
            },
        )
        run_scoring(score_path)
        endpoint_manifest = natural_root / f"{_slug(endpoint)}_manifest.yaml"
        write_manifest(
            endpoint_manifest,
            Manifest(
                stage="run_natural_mismatch",
                artifacts={"run_root": str(run_root)},
                metadata={
                    "natural_endpoint": endpoint,
                    "condition_id": condition,
                    "candidate_set": candidate_name,
                    "n_query_positive": int(query["fine_label"].eq(endpoint).sum()),
                    "restoration_metric_applicability": "undefined",
                },
            ),
        )
        artifacts[_slug(endpoint)] = endpoint_manifest
    manifest_path = natural_root / "natural_mismatch_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_natural_mismatch",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={"mapping_direction": "rna_to_atac", "endpoints": endpoints},
        ),
    )
    return StageResult("run_natural_mismatch", natural_root, artifacts, manifest_path)


def _transport_method_configs(
    config: dict[str, Any],
    methods: list[str],
    *,
    balanced_support: str = "sparse",
    coreot_full_override: dict[str, Any] | None = None,
    mean_matched_factorial: bool = False,
    uniform_uot_tau_source: float | None = None,
) -> list[dict[str, Any]]:
    transport = _mapping(config, "transport")
    epsilon = float(transport.get("epsilon", 0.05))
    tau_constant = float(transport.get("tau_source_constant", 1.0))
    tau_target = float(transport.get("tau_target", 1.0))
    alpha = float(transport.get("alpha", 0.25))
    max_iter = int(transport.get("max_iterations", 2000))
    tol = float(transport.get("tolerance", 1.0e-6))
    eta = float(transport.get("eta", 1.0e-12))
    full = {**transport, **(coreot_full_override or {})}
    if mean_matched_factorial and coreot_full_override is None:
        raise MouseSpleenConfigError(
            "mean_matched_factorial requires a coreot_full override"
        )
    result: list[dict[str, Any]] = []
    for method in methods:
        item: dict[str, Any] = {"name": method}
        if method == "uniform_uot":
            item |= {
                "epsilon": epsilon,
                "tau_source": tau_constant,
                "tau_target": tau_target,
                "max_iter": max_iter,
                "tol": tol,
                "eta": eta,
            }
            if mean_matched_factorial:
                item |= {
                    "tau_source": "matched_coreot_mean",
                    "matched_tau_min": float(full["tau_min"]),
                    "matched_tau_max": float(full["tau_max"]),
                    "tau_target": float(full["tau_target"]),
                    "max_iter": int(full["max_iterations"]),
                }
            if uniform_uot_tau_source is not None:
                if uniform_uot_tau_source <= 0.0:
                    raise MouseSpleenConfigError(
                        "uniform_uot_tau_source must be positive"
                    )
                item["tau_source"] = float(uniform_uot_tau_source)
                item.pop("matched_tau_min", None)
                item.pop("matched_tau_max", None)
        elif method == "balanced_ot":
            if balanced_support not in {"dense", "sparse"}:
                raise MouseSpleenConfigError(
                    "balanced_ot_support must be 'dense' or 'sparse'"
                )
            item |= {
                "epsilon": epsilon,
                "alpha": 0.0,
                "support": balanced_support,
                "max_iter": max_iter,
                "tol": tol,
                "eta": eta,
            }
        elif method == "coreot_constant_tau":
            item |= {
                "epsilon": epsilon,
                "tau_source": tau_constant,
                "tau_target": tau_target,
                "alpha": alpha,
                "max_iter": max_iter,
                "tol": tol,
                "eta": eta,
            }
            if mean_matched_factorial:
                item |= {
                    "tau_source": "matched_coreot_mean",
                    "matched_tau_min": float(full["tau_min"]),
                    "matched_tau_max": float(full["tau_max"]),
                    "tau_target": float(full["tau_target"]),
                    "alpha": float(full["alpha"]),
                    "max_iter": int(full["max_iterations"]),
                }
        elif method == "coreot_full":
            item |= {
                "epsilon": epsilon,
                "tau_min": float(full.get("tau_min", 0.05)),
                "tau_max": float(full.get("tau_max", 1.0)),
                "tau_target": float(full.get("tau_target", 1.0)),
                "alpha": float(full.get("alpha", 0.25)),
                "max_iter": int(full.get("max_iterations", max_iter)),
                "tol": tol,
                "eta": eta,
            }
        elif method == "coreot_match_only":
            item |= {
                "epsilon": epsilon,
                "tau_min": float(full.get("tau_min", 0.05)),
                "tau_max": float(full.get("tau_max", 1.0)),
                "tau_target": float(full.get("tau_target", 1.0)),
                "alpha": 0.0,
                "max_iter": int(full.get("max_iterations", max_iter)),
                "tol": tol,
                "eta": eta,
            }
        result.append(item)
    return result


def _read_manifest_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MouseSpleenConfigError(f"Invalid task manifest: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_controlled_task_family(
    *,
    config: dict[str, Any],
    task_root: Path,
    output_root: Path,
    holdout: str,
    query: pd.DataFrame,
    reference_full: pd.DataFrame,
    embedding_by_id: dict[str, np.ndarray],
    preparation_metadata: dict[str, object] | None = None,
) -> StageResult:
    task_root.mkdir(parents=True, exist_ok=True)
    reference_zero = reference_full.loc[~reference_full["fine_label"].eq(holdout)].copy()
    if reference_zero.empty:
        raise MouseSpleenConfigError(f"Holdout {holdout!r} removes the entire reference")
    holdout_broad = reference_full.loc[
        reference_full["fine_label"].eq(holdout), "broad_label"
    ].unique()
    if len(holdout_broad) != 1:
        raise MouseSpleenConfigError(f"Holdout {holdout!r} must have exactly one broad group")
    if not reference_zero["broad_label"].eq(holdout_broad[0]).any():
        raise MouseSpleenConfigError(
            f"Holdout {holdout!r} leaves no represented fine state in broad group {holdout_broad[0]!r}"
        )

    query_embedding = np.vstack([embedding_by_id[value] for value in query["cell_id"]])
    reference_zero_embedding = np.vstack(
        [embedding_by_id[value] for value in reference_zero["cell_id"]]
    )
    priors_config = _mapping(config, "priors")
    classifier_config = _mapping(priors_config, "broad_classifier")
    classifier = LogisticRegression(
        C=float(classifier_config.get("C", 1.0)),
        class_weight="balanced",
        max_iter=int(classifier_config.get("max_iter", 5000)),
        random_state=int(classifier_config.get("random_state", 0)),
    ).fit(reference_zero_embedding, reference_zero["broad_label"].astype(str))
    h_source = classifier.predict_proba(query_embedding)
    classes = classifier.classes_.astype(str)
    confidence = _entropy_confidence(h_source)
    matchability_config = _mapping(priors_config, "matchability")
    agreement = _query_neighbor_agreement(
        query_embedding,
        h_source,
        k=int(matchability_config.get("query_neighbor_k", 15)),
    )
    confidence_weight = float(matchability_config.get("confidence_weight", 0.5))
    neighbor_weight = float(matchability_config.get("neighbor_weight", 0.5))
    if not np.isclose(confidence_weight + neighbor_weight, 1.0):
        raise MouseSpleenConfigError("Matchability confidence and neighbor weights must sum to one")
    clip = matchability_config.get("clip", [0.0, 1.0])
    if not isinstance(clip, list) or len(clip) != 2:
        raise MouseSpleenConfigError("priors.matchability.clip must have two values")
    rho = np.clip(confidence_weight * confidence + neighbor_weight * agreement, *map(float, clip))
    predicted = classes[np.argmax(h_source, axis=1)]

    query_prior_path = task_root / "query_prior_h.npy"
    query_rho_path = task_root / "query_rho.csv"
    classifier_path = task_root / "broad_classifier.joblib"
    np.save(query_prior_path, h_source)
    pd.DataFrame(
        {"cell_id": query["cell_id"], "rho": rho, "prior_risk": 1.0 - rho}
    ).to_csv(query_rho_path, index=False)
    joblib.dump(classifier, classifier_path)

    source_priors = pd.DataFrame(
        {
            "cell_id": query["cell_id"].astype(str),
            "rho": rho,
            "rho_source": "entropy_confidence_query_neighbor_agreement",
            "rho_recipe": "rho=clip(0.5*entropy_confidence+0.5*query_neighbor_agreement)",
            "prior_risk": 1.0 - rho,
            "pmax_reference_classifier": h_source.max(axis=1),
            "anchor_class_pred": predicted,
            "anchor_confidence": confidence,
        }
    )
    for index, broad in enumerate(classes):
        source_priors[f"anchor_probability::{broad}"] = h_source[:, index]

    run_id = f"mouse_spleen_{_slug(holdout)}"
    run_root = output_root / "runs" / run_id
    conditions: dict[str, pd.DataFrame] = {
        "full_reference_control": reference_full,
        "incomplete_reference": reference_zero,
    }
    master_seed = int(_mapping(config, "experiment").get("master_seed", 0))
    condition_metadata: dict[str, dict[str, object]] = {
        "full_reference_control": {"family": "controlled", "support_fraction": 1.0},
        "incomplete_reference": {"family": "controlled", "support_fraction": 0.0},
    }
    experiments = _mapping(config, "experiments")
    support_value = experiments.get("support_dose", {})
    support_config = support_value if isinstance(support_value, dict) else {}
    fractions = [float(value) for value in support_config.get("fractions", ())]
    partial_seeds = int(support_config.get("n_seeds_for_partial_fractions", 0))
    heldout_reference = reference_full.loc[reference_full["fine_label"].eq(holdout)]
    for fraction in fractions:
        if not 0.0 <= fraction <= 1.0:
            raise MouseSpleenConfigError("support-dose fractions must lie in [0, 1]")
        if fraction in (0.0, 1.0):
            continue
        retained_count = int(round(fraction * len(heldout_reference)))
        for seed in range(partial_seeds):
            rng = np.random.default_rng(master_seed + seed)
            retained_ids = rng.choice(
                heldout_reference["cell_id"].to_numpy(), retained_count, replace=False
            )
            condition = f"support_{_fraction_slug(fraction)}_seed_{seed}"
            conditions[condition] = pd.concat(
                [reference_zero, heldout_reference.loc[heldout_reference["cell_id"].isin(retained_ids)]],
                ignore_index=True,
            )
            condition_metadata[condition] = {
                "family": "support_dose",
                "support_fraction": fraction,
                "random_seed": seed,
            }

    deletion_value = experiments.get("random_deletion", {})
    deletion_config = deletion_value if isinstance(deletion_value, dict) else {}
    deletion_seeds = int(deletion_config.get("n_seeds", 0))
    controls = [str(value) for value in deletion_config.get("controls", ())]
    removed_count = len(heldout_reference)
    for control in controls:
        if control not in {"global_stratified", "same_broad_stratified"}:
            raise MouseSpleenConfigError(f"Unsupported random-deletion control: {control}")
        for seed in range(deletion_seeds):
            rng = np.random.default_rng(master_seed + 10000 + seed)
            eligible = reference_full
            if control == "same_broad_stratified":
                same_broad = reference_full.loc[
                    reference_full["broad_label"].eq(str(holdout_broad[0]))
                ]
                eligible = same_broad if len(same_broad) >= removed_count else reference_full
            removed_ids = rng.choice(
                eligible["cell_id"].to_numpy(), removed_count, replace=False
            )
            condition = f"{control}_seed_{seed}"
            conditions[condition] = reference_full.loc[
                ~reference_full["cell_id"].isin(removed_ids)
            ].copy()
            condition_metadata[condition] = {
                "family": "random_deletion",
                "deletion_control_type": control,
                "random_seed": seed,
            }

    retained_root = task_root / "retained_reference_sets"
    retained_root.mkdir(parents=True, exist_ok=True)
    condition_rows: list[dict[str, object]] = []
    for condition, reference in conditions.items():
        retained_path = retained_root / f"{condition}.csv"
        pd.DataFrame({"cell_id": reference["cell_id"].astype(str)}).to_csv(
            retained_path, index=False
        )
        _write_condition_artifacts(
            run_root=run_root,
            condition=condition,
            holdout=holdout,
            query=query,
            reference=reference,
            source_priors=source_priors,
            embedding_by_id=embedding_by_id,
            split_seed=master_seed,
        )
        condition_rows.append(
            {
                "condition_id": condition,
                **condition_metadata[condition],
                "n_reference": int(len(reference)),
                "retained_reference_ids": str(retained_path),
                "retained_reference_sha256": sha256_file(retained_path),
            }
        )

    manifest_path = task_root / "task_manifest.yaml"
    artifacts = {
        "run_root": str(run_root),
        "query_prior_h": str(query_prior_path),
        "query_rho": str(query_rho_path),
        "broad_classifier": str(classifier_path),
    }
    write_manifest(
        manifest_path,
        Manifest(
            stage="prepare_task_families",
            artifacts=artifacts,
            metadata={
                "task_id": run_id,
                "holdout_label": holdout,
                "holdout_broad_label": str(holdout_broad[0]),
                "n_query": int(len(query)),
                "n_query_positive": int(query["fine_label"].eq(holdout).sum()),
                "n_reference_full": int(len(reference_full)),
                "n_reference_incomplete": int(len(reference_zero)),
                "broad_class_order": classes.tolist(),
                "classifier_training_fine_labels": {
                    str(key): int(value)
                    for key, value in reference_zero["fine_label"].value_counts().items()
                },
                "query_prior_sha256": sha256_file(query_prior_path),
                "query_rho_sha256": sha256_file(query_rho_path),
                "conditions": condition_rows,
                **(preparation_metadata or {}),
            },
        ),
    )
    return StageResult(
        stage="prepare_task_families",
        root=task_root,
        artifacts={key: Path(value) for key, value in artifacts.items()},
        manifest_path=manifest_path,
    )


def _bounded_smoke_sample(
    *,
    query: pd.DataFrame,
    reference_full: pd.DataFrame,
    holdout: str,
    config: dict[str, Any],
    default_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    query_total = int(config.get("query_total", 100))
    query_positive = int(config.get("query_positive", 30))
    reference_total = int(config.get("reference_total", 150))
    reference_holdout = int(config.get("reference_holdout", 30))
    seed = int(config.get("seed", default_seed))
    if not (0 < query_positive < query_total <= 100):
        raise MouseSpleenConfigError(
            "Smoke query sizes must satisfy 0 < query_positive < query_total <= 100"
        )
    if not (0 < reference_holdout < reference_total <= 150):
        raise MouseSpleenConfigError(
            "Smoke reference sizes must satisfy "
            "0 < reference_holdout < reference_total <= 150"
        )

    query_heldout = query.loc[query["fine_label"].eq(holdout)]
    query_other = query.loc[~query["fine_label"].eq(holdout)]
    reference_heldout = reference_full.loc[reference_full["fine_label"].eq(holdout)]
    reference_other = reference_full.loc[~reference_full["fine_label"].eq(holdout)]
    if len(query_heldout) < query_positive or len(reference_heldout) < reference_holdout:
        raise MouseSpleenConfigError(
            f"Smoke holdout {holdout!r} lacks the requested positive cells"
        )

    sampled_query = pd.concat(
        [
            _deterministic_sample(query_heldout, query_positive, seed=seed),
            _stratified_sample(
                query_other,
                query_total - query_positive,
                group_column="fine_label",
                seed=seed + 1,
            ),
        ]
    ).sort_index()
    sampled_reference = pd.concat(
        [
            _deterministic_sample(reference_heldout, reference_holdout, seed=seed + 2),
            _stratified_sample(
                reference_other,
                reference_total - reference_holdout,
                group_column="fine_label",
                seed=seed + 3,
            ),
        ]
    ).sort_index()

    holdout_broad = sampled_reference.loc[
        sampled_reference["fine_label"].eq(holdout), "broad_label"
    ].iloc[0]
    same_broad_negative = sampled_query.loc[
        ~sampled_query["fine_label"].eq(holdout)
        & sampled_query["broad_label"].eq(holdout_broad)
    ]
    if same_broad_negative.empty:
        raise MouseSpleenConfigError("Smoke query must retain a same-broad negative cell")
    if len(sampled_query) != query_total or len(sampled_reference) != reference_total:
        raise MouseSpleenConfigError("Smoke sampling did not produce the requested sizes")

    return (
        sampled_query,
        sampled_reference,
        {
            "smoke": True,
            "smoke_interpretation": "pipeline_integrity_only",
            "smoke_seed": seed,
            "smoke_query_total": query_total,
            "smoke_query_positive": query_positive,
            "smoke_reference_full_total": reference_total,
            "smoke_reference_holdout": reference_holdout,
            "smoke_query_cell_id_sha256": _cell_id_sha256(sampled_query["cell_id"]),
            "smoke_reference_cell_id_sha256": _cell_id_sha256(
                sampled_reference["cell_id"]
            ),
        },
    )


def _deterministic_sample(frame: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if n > len(frame):
        raise MouseSpleenConfigError(f"Cannot sample {n} cells from {len(frame)} available")
    if n == len(frame):
        return frame.copy()
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), size=n, replace=False))
    return frame.iloc[positions].copy()


def _stratified_sample(
    frame: pd.DataFrame,
    n: int,
    *,
    group_column: str,
    seed: int,
) -> pd.DataFrame:
    if n > len(frame):
        raise MouseSpleenConfigError(f"Cannot sample {n} cells from {len(frame)} available")
    counts = frame[group_column].astype(str).value_counts().sort_index()
    if n < len(counts):
        raise MouseSpleenConfigError(
            f"Stratified smoke sample of {n} cannot retain all {len(counts)} groups"
        )
    allocation = pd.Series(1, index=counts.index, dtype=int)
    remaining = n - int(allocation.sum())
    capacity = counts - allocation
    if remaining:
        raw = remaining * capacity / int(capacity.sum())
        additions = np.floor(raw).astype(int).clip(upper=capacity)
        allocation += additions
        remaining = n - int(allocation.sum())
        remainders = (raw - additions).sort_values(ascending=False, kind="stable")
        while remaining:
            progressed = False
            for label in remainders.index:
                if allocation[label] < counts[label]:
                    allocation[label] += 1
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                raise MouseSpleenConfigError("Unable to complete stratified smoke allocation")

    sampled = []
    for offset, label in enumerate(allocation.index):
        group = frame.loc[frame[group_column].astype(str).eq(label)]
        sampled.append(
            _deterministic_sample(group, int(allocation[label]), seed=seed + offset)
        )
    return pd.concat(sampled).sort_index()


def _cell_id_sha256(cell_ids: pd.Series) -> str:
    payload = "\n".join(cell_ids.astype(str)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_condition_artifacts(
    *,
    run_root: Path,
    condition: str,
    holdout: str,
    query: pd.DataFrame,
    reference: pd.DataFrame,
    source_priors: pd.DataFrame,
    embedding_by_id: dict[str, np.ndarray],
    split_seed: int,
    absent_condition: bool | None = None,
) -> None:
    model_visible = run_root / "benchmark" / condition / "model_visible"
    evaluation_truth = run_root / "benchmark" / condition / "evaluation_truth"
    model_visible.mkdir(parents=True, exist_ok=True)
    evaluation_truth.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame(
        {
            "cell_id": [*query["cell_id"], *reference["cell_id"]],
            "domain": ["query"] * len(query) + ["reference"] * len(reference),
            "sample_id": "not_available",
            "donor_id": "not_available",
            "batch_id": "not_available",
            "condition_id": condition,
            "split_seed": split_seed,
        }
    )
    cells.to_csv(model_visible / "cells.csv", index=False)
    target_labels = pd.DataFrame(
        {
            "cell_id": reference["cell_id"].astype(str),
            "target_label": reference["fine_label"].astype(str),
            "broad_label": reference["broad_label"].astype(str),
        }
    )
    target_labels.to_csv(model_visible / "target_labels.csv", index=False)
    is_positive = query["fine_label"].eq(holdout).to_numpy()
    is_absent_condition = condition == "incomplete_reference" if absent_condition is None else absent_condition
    query_truth = pd.DataFrame(
        {
            "cell_id": query["cell_id"].astype(str),
            "true_label": query["fine_label"].astype(str),
            "true_broad_label": query["broad_label"].astype(str),
            "removed_state": holdout,
            "is_absent_state": is_positive if is_absent_condition else False,
            "is_shared_state": ~is_positive if is_absent_condition else True,
        }
    )
    query_truth.to_csv(evaluation_truth / "query_truth.csv", index=False)

    profile_root = run_root / "derived" / condition / "prior_profiles" / "default"
    profile_root.mkdir(parents=True, exist_ok=True)
    source_priors.to_csv(profile_root / "source_priors.csv", index=False)
    pd.DataFrame(
        {
            "cell_id": reference["cell_id"].astype(str),
            "target_label_visible": reference["fine_label"].astype(str),
            "broad_anchor_class": reference["broad_label"].astype(str),
            "rho_target": 1.0,
        }
    ).to_csv(profile_root / "target_priors.csv", index=False)

    provider_root = run_root / "embeddings" / condition / "mouse_spleen_provider"
    provider_root.mkdir(parents=True, exist_ok=True)
    condition_embedding = np.vstack(
        [embedding_by_id[value] for value in [*query["cell_id"], *reference["cell_id"]]]
    )
    np.save(provider_root / "embedding.npy", condition_embedding)
    pd.DataFrame(
        {
            "row_index": np.arange(len(cells)),
            "cell_id": cells["cell_id"].astype(str),
            "domain": cells["domain"].astype(str),
            "condition_id": condition,
        }
    ).to_csv(provider_root / "embedding_cells.csv", index=False)


def _fit_source_priors(
    config: dict[str, Any],
    query: pd.DataFrame,
    reference: pd.DataFrame,
    embedding_by_id: dict[str, np.ndarray],
    *,
    priors_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    query_embedding = np.vstack([embedding_by_id[value] for value in query["cell_id"]])
    reference_embedding = np.vstack(
        [embedding_by_id[value] for value in reference["cell_id"]]
    )
    if priors_config is None:
        priors_config = _mapping(config, "priors")
    anchor_classifier_config = (
        _mapping(priors_config, "anchor_classifier")
        if "anchor_classifier" in priors_config
        else _mapping(priors_config, "broad_classifier")
    )
    matchability_classifier_config = (
        _mapping(priors_config, "matchability_classifier")
        if "matchability_classifier" in priors_config
        else anchor_classifier_config
    )
    anchor_classifier = LogisticRegression(
        C=float(anchor_classifier_config.get("C", 1.0)),
        class_weight="balanced",
        max_iter=int(anchor_classifier_config.get("max_iter", 5000)),
        random_state=int(anchor_classifier_config.get("random_state", 0)),
    ).fit(reference_embedding, reference["broad_label"].astype(str))
    matchability_classifier = LogisticRegression(
        C=float(matchability_classifier_config.get("C", 1.0)),
        class_weight="balanced",
        max_iter=int(matchability_classifier_config.get("max_iter", 5000)),
        random_state=int(matchability_classifier_config.get("random_state", 0)),
    ).fit(reference_embedding, reference["broad_label"].astype(str))
    anchor_probabilities = anchor_classifier.predict_proba(query_embedding)
    matchability_probabilities = matchability_classifier.predict_proba(query_embedding)
    classes = anchor_classifier.classes_.astype(str)
    if not np.array_equal(classes, matchability_classifier.classes_.astype(str)):
        raise MouseSpleenConfigError(
            "Anchor and matchability classifiers must use the same broad classes"
        )
    anchor_confidence = _entropy_confidence(anchor_probabilities)
    matchability_confidence = _entropy_confidence(matchability_probabilities)
    matchability = _mapping(priors_config, "matchability")
    agreement = _query_neighbor_agreement(
        query_embedding,
        matchability_probabilities,
        k=int(matchability.get("query_neighbor_k", 15)),
    )
    confidence_weight = float(matchability.get("confidence_weight", 0.5))
    neighbor_weight = float(matchability.get("neighbor_weight", 0.5))
    if not np.isclose(confidence_weight + neighbor_weight, 1.0):
        raise MouseSpleenConfigError("Matchability confidence and neighbor weights must sum to one")
    clip = matchability.get("clip", [0.0, 1.0])
    if not isinstance(clip, list) or len(clip) != 2:
        raise MouseSpleenConfigError("priors.matchability.clip must have two values")
    clip_lower, clip_upper = map(float, clip)
    rho = np.clip(
        confidence_weight * matchability_confidence + neighbor_weight * agreement,
        clip_lower,
        clip_upper,
    )
    rho_recipe = (
        f"rho=clip({confidence_weight:g}*matchability_classifier_entropy_confidence+"
        f"{neighbor_weight:g}*query_neighbor_agreement,{clip_lower:g},{clip_upper:g})"
    )
    source_priors = pd.DataFrame(
        {
            "cell_id": query["cell_id"].astype(str),
            "rho": rho,
            "rho_source": "entropy_confidence_query_neighbor_agreement",
            "rho_recipe": rho_recipe,
            "prior_risk": 1.0 - rho,
            "pmax_reference_classifier": anchor_probabilities.max(axis=1),
            "anchor_class_pred": classes[np.argmax(anchor_probabilities, axis=1)],
            "anchor_confidence": anchor_confidence,
            "matchability_classifier_confidence": matchability_confidence,
            "query_neighbor_agreement": agreement,
            "anchor_classifier_C": float(anchor_classifier_config.get("C", 1.0)),
            "anchor_classifier_max_iter": int(
                anchor_classifier_config.get("max_iter", 5000)
            ),
            "anchor_classifier_random_state": int(
                anchor_classifier_config.get("random_state", 0)
            ),
            "matchability_classifier_C": float(
                matchability_classifier_config.get("C", 1.0)
            ),
            "matchability_classifier_max_iter": int(
                matchability_classifier_config.get("max_iter", 5000)
            ),
            "matchability_classifier_random_state": int(
                matchability_classifier_config.get("random_state", 0)
            ),
        }
    )
    for index, broad in enumerate(classes):
        source_priors[f"anchor_probability::{broad}"] = anchor_probabilities[:, index]
    return source_priors


def _load_canonical_modalities(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_config = _mapping(config, "data")
    labels_config = _mapping(config, "labels")
    label_key = str(data_config.get("label_key", "cell_type"))
    rna_adata = ad.read_h5ad(_string(data_config, "rna_h5ad"), backed="r")
    atac_adata = ad.read_h5ad(_string(data_config, "atac_gene_activity_h5ad"), backed="r")
    try:
        raw = {
            "rna": rna_adata.obs[label_key].astype(str).str.strip(),
            "atac": atac_adata.obs[label_key].astype(str).str.strip(),
        }
        resolved = _resolve_label_maps(raw, labels_config)
        broad = _broad_label_map(labels_config)
        rna = pd.DataFrame(
            {
                "cell_id": [f"rna::{value}" for value in rna_adata.obs_names.astype(str)],
                "raw_label": raw["rna"].to_numpy(),
                "fine_label": raw["rna"].map(resolved["rna"]).to_numpy(),
            }
        )
        atac = pd.DataFrame(
            {
                "cell_id": [f"atac::{value}" for value in atac_adata.obs_names.astype(str)],
                "raw_label": raw["atac"].to_numpy(),
                "fine_label": raw["atac"].map(resolved["atac"]).to_numpy(),
            }
        )
        rna["broad_label"] = rna["fine_label"].map(broad)
        atac["broad_label"] = atac["fine_label"].map(broad)
        return rna, atac
    finally:
        if rna_adata.file is not None:
            rna_adata.file.close()
        if atac_adata.file is not None:
            atac_adata.file.close()


def _entropy_confidence(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] <= 1:
        return np.ones(probabilities.shape[0], dtype=float)
    clipped = np.clip(probabilities, 1.0e-300, 1.0)
    entropy = -(probabilities * np.log(clipped)).sum(axis=1)
    return 1.0 - entropy / np.log(probabilities.shape[1])


def _query_neighbor_agreement(
    embedding: np.ndarray, probabilities: np.ndarray, *, k: int
) -> np.ndarray:
    if len(embedding) <= 1:
        return np.ones(len(embedding), dtype=float)
    n_neighbors = min(max(k, 1) + 1, len(embedding))
    indices = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", algorithm="brute"
    ).fit(embedding).kneighbors(embedding, return_distance=False)
    agreements = np.empty(len(embedding), dtype=float)
    for row, neighbors in enumerate(indices):
        neighbors = neighbors[neighbors != row][:k]
        agreements[row] = (
            float(np.mean(probabilities[neighbors] @ probabilities[row]))
            if len(neighbors)
            else 1.0
        )
    return agreements


def _slug(value: str) -> str:
    return "_".join(value.strip().lower().replace("+", " plus ").split())


def _fraction_slug(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _generate_multimap_embedding(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    data_config = _mapping(config, "data")
    embedding_config = _mapping(config, "embedding")
    multimap_config = _mapping(embedding_config, "multimap")
    # MultiMAP commit 681e608 imports np.infty, removed by NumPy 2. Keep the
    # pinned provider usable without constraining the repository-wide NumPy.
    if not hasattr(np, "infty"):
        np.infty = np.inf  # type: ignore[attr-defined]
    try:
        multimap = importlib.import_module("MultiMAP")
    except ImportError as exc:
        raise MouseSpleenConfigError(
            "fixed_multimap embedding requires the optional pinned MultiMAP dependency"
        ) from exc

    rna = ad.read_h5ad(_string(data_config, "rna_h5ad"))
    atac_genes = ad.read_h5ad(_string(data_config, "atac_gene_activity_h5ad"))
    atac_peaks = ad.read_h5ad(_string(data_config, "atac_peaks_h5ad"))
    rna_ids = rna.obs_names.astype(str).to_numpy()
    atac_ids = atac_genes.obs_names.astype(str).to_numpy()
    rna.obs_names = pd.Index([f"rna::{value}" for value in rna_ids])
    atac_genes.obs_names = pd.Index([f"atac::{value}" for value in atac_ids])
    atac_peaks.obs_names = atac_genes.obs_names.copy()

    n_components = int(multimap_config.get("n_components", 50))
    rna_pca = rna.copy()
    sc.pp.scale(rna_pca)
    sc.tl.pca(rna_pca, n_comps=min(n_components, rna_pca.n_obs - 1, rna_pca.n_vars - 1))
    rna.obsm["X_pca"] = np.asarray(rna_pca.obsm["X_pca"])
    multimap.TFIDF_LSI(atac_peaks, n_comps=n_components)
    atac_genes.obsm["X_lsi"] = np.asarray(atac_peaks.obsm["X_lsi"]).copy()
    integrated = multimap.Integration(
        [rna, atac_genes],
        ["X_pca", "X_lsi"],
        n_components=n_components,
        seed=int(_mapping(config, "experiment").get("master_seed", 0)),
    )
    key = str(multimap_config.get("output_key", "X_multimap"))
    if key not in integrated.obsm:
        raise MouseSpleenConfigError(f"MultiMAP output is missing obsm key {key!r}")
    coordinates = np.asarray(integrated.obsm[key], dtype=float)
    integrated_ids = integrated.obs_names.astype(str).tolist()
    expected_ids = [*(f"rna::{value}" for value in rna_ids), *(f"atac::{value}" for value in atac_ids)]
    if integrated_ids != expected_ids:
        lookup = {cell_id: index for index, cell_id in enumerate(integrated_ids)}
        if set(lookup) != set(expected_ids):
            raise MouseSpleenConfigError("MultiMAP output cell IDs do not match the input cells")
        coordinates = coordinates[[lookup[cell_id] for cell_id in expected_ids]]
    return (
        rna_ids,
        atac_ids,
        coordinates[: len(rna_ids)],
        coordinates[len(rna_ids) :],
        {
            "mode": "fixed_multimap",
            "package_version": str(getattr(multimap, "__version__", "0.0.1")),
            "git_repository": str(multimap_config.get("git_repository", "")),
            "git_commit": str(multimap_config.get("git_commit", "")),
            "data_scope": "complete_unlabeled_rna_atac",
        },
    )


def _obs_names(path: Path) -> np.ndarray:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.obs_names.astype(str).to_numpy()
    finally:
        if adata.file is not None:
            adata.file.close()


def _resolve_label_maps(
    raw_labels: dict[str, pd.Series], labels_config: dict[str, Any]
) -> dict[str, dict[str, str]]:
    aliases = _mapping(labels_config, "aliases")
    case_insensitive = bool(labels_config.get("allow_case_insensitive_exact_match", True))
    if bool(labels_config.get("allow_fuzzy_match", False)):
        raise MouseSpleenConfigError("Mouse-spleen label resolution does not support fuzzy matching")

    normalized_aliases: dict[str, str] = {}
    for canonical, configured_aliases in aliases.items():
        if not isinstance(configured_aliases, list) or not configured_aliases:
            raise MouseSpleenConfigError(f"labels.aliases.{canonical} must be a nonempty list")
        for alias in configured_aliases:
            normalized = _normalize_label(str(alias), case_insensitive)
            previous = normalized_aliases.get(normalized)
            if previous is not None and previous != str(canonical):
                raise MouseSpleenConfigError(
                    f"Raw label {alias!r} maps to multiple canonical labels: "
                    f"{previous!r}, {canonical!r}"
                )
            normalized_aliases[normalized] = str(canonical)

    resolved: dict[str, dict[str, str]] = {}
    for modality, values in raw_labels.items():
        mapping: dict[str, str] = {}
        unresolved = []
        for raw in sorted(values.unique()):
            canonical = normalized_aliases.get(_normalize_label(raw, case_insensitive))
            if canonical is None:
                unresolved.append(raw)
            else:
                mapping[raw] = canonical
        if unresolved:
            raise MouseSpleenConfigError(
                f"Unresolved {modality} raw label(s): {', '.join(unresolved)}"
            )
        resolved[modality] = mapping

    shared = [str(value) for value in labels_config.get("shared_labels", ())]
    rna_only = [str(value) for value in labels_config.get("rna_only_labels", ())]
    for canonical in shared:
        for modality in ("rna", "atac"):
            if canonical not in set(resolved[modality].values()):
                raise MouseSpleenConfigError(
                    f"Shared canonical label {canonical!r} is absent from {modality}"
                )
    for canonical in rna_only:
        if canonical not in set(resolved["rna"].values()):
            raise MouseSpleenConfigError(f"RNA-only canonical label {canonical!r} is absent from RNA")
        if canonical in set(resolved["atac"].values()):
            raise MouseSpleenConfigError(
                f"RNA-only canonical label {canonical!r} must be absent from ATAC"
            )
    return resolved


def _broad_label_map(labels_config: dict[str, Any]) -> dict[str, str]:
    broad_map = _mapping(labels_config, "broad_map")
    result: dict[str, str] = {}
    for broad, fine_labels in broad_map.items():
        if not isinstance(fine_labels, list):
            raise MouseSpleenConfigError(f"labels.broad_map.{broad} must be a list")
        for fine in fine_labels:
            fine = str(fine)
            previous = result.get(fine)
            if previous is not None and previous != str(broad):
                raise MouseSpleenConfigError(
                    f"Canonical label {fine!r} belongs to multiple broad groups"
                )
            result[fine] = str(broad)
    return result


def _composition(
    raw_labels: dict[str, pd.Series],
    raw_to_canonical: dict[str, dict[str, str]],
    broad_by_fine: dict[str, str],
    labels_config: dict[str, Any],
) -> pd.DataFrame:
    counts: dict[str, pd.Series] = {}
    for modality, values in raw_labels.items():
        canonical = values.map(raw_to_canonical[modality])
        counts[modality] = canonical.value_counts()
    labels = sorted(set(counts["rna"].index) | set(counts["atac"].index))
    shared = set(map(str, labels_config.get("shared_labels", ())))
    rows = []
    for fine in labels:
        rna_count = int(counts["rna"].get(fine, 0))
        atac_count = int(counts["atac"].get(fine, 0))
        rows.append(
            {
                "fine_label": fine,
                "broad_label": broad_by_fine.get(fine, ""),
                "rna_count": rna_count,
                "atac_count": atac_count,
                "rna_fraction": rna_count / len(raw_labels["rna"]),
                "atac_fraction": atac_count / len(raw_labels["atac"]),
                "shared_status": "shared" if fine in shared else "rna_only",
            }
        )
    return pd.DataFrame(rows)


def _validate_expected_shapes(datasets: dict[str, ad.AnnData], config: dict[str, Any]) -> None:
    expected = config.get("expected_shapes", {})
    if not isinstance(expected, dict) or bool(config.get("allow_shape_mismatch", False)):
        return
    key_map = {"rna": "rna", "atac_gene_activity": "atac_gene_activity"}
    for config_key, dataset_key in key_map.items():
        value = expected.get(config_key)
        if value is None:
            continue
        if list(datasets[dataset_key].shape) != list(value):
            raise MouseSpleenConfigError(
                f"{dataset_key} shape {datasets[dataset_key].shape} does not match expected {value}"
            )


def _normalize_label(value: str, case_insensitive: bool) -> str:
    stripped = value.strip()
    return stripped.casefold() if case_insensitive else stripped


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise MouseSpleenConfigError(f"Expected mapping config value: {key}")
    return value


def _string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise MouseSpleenConfigError(f"Expected nonempty string config value: {key}")
    return value
