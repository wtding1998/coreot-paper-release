from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.config.load import load_yaml
from coreot.data.schemas import EMBEDDING_CELLS_COLUMNS, MODEL_VISIBLE_CELLS_COLUMNS
from coreot.data.validation import validate_required_columns, validate_stage_can_read
from coreot.data.model_visible import read_condition_adata
from coreot.embeddings.pca import compute_pca_embedding
from coreot.embeddings.row_identity import assert_embedding_row_identity
from coreot.embeddings.incremental import fit_incremental_pca, transform_incremental_pca


STAGE = "embedding"


class EmbeddingRunnerError(ValueError):
    """Raised when embedding artifacts cannot be constructed."""


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings_root: Path
    conditions: tuple[str, ...]
    providers: tuple[str, ...]


def run_embedding(config_path: str | Path) -> EmbeddingResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    output_root = _required_path(config, ("outputs", "root"))
    conditions = tuple(config.get("conditions", ()))
    providers = tuple(config.get("providers", ()))
    if not conditions:
        raise EmbeddingRunnerError("embedding requires conditions")
    if not providers:
        raise EmbeddingRunnerError("embedding requires at least one provider")

    run_root = output_root / run_id
    embeddings_root = run_root / "embeddings"
    provider_names: list[str] = []
    for provider in providers:
        if not isinstance(provider, dict):
            raise EmbeddingRunnerError("embedding providers must be mappings")
        name = _required_str(provider, ("name",))
        provider_names.append(name)
        method = str(provider.get("method", "pca"))
        if method == "paired_incremental_pca":
            _run_paired_incremental_pca(
                run_root, embeddings_root, conditions, name, provider
            )
            continue
        for condition in conditions:
            if method == "pca":
                if not name.startswith("pca"):
                    raise EmbeddingRunnerError(f"PCA provider names must start with 'pca'; got {name!r}")
                n_components = _required_int(provider, ("n_components",))
                _run_pca_provider(run_root, embeddings_root, condition, name, n_components)
            elif method == "obsm":
                obsm_key = _required_str(provider, ("obsm_key",))
                n_components = _required_int(provider, ("n_components",))
                _run_obsm_provider(
                    run_root, embeddings_root, condition, name, obsm_key, n_components
                )
            else:
                raise EmbeddingRunnerError(f"Unsupported embedding provider method: {method!r}")

    return EmbeddingResult(
        embeddings_root=embeddings_root,
        conditions=conditions,
        providers=tuple(provider_names),
    )


def _run_paired_incremental_pca(
    run_root: Path,
    embeddings_root: Path,
    conditions: tuple[str, ...],
    provider_name: str,
    provider: dict[str, Any],
) -> None:
    shared_root = run_root / "benchmark" / "shared" / "model_visible"
    counts_path = shared_root / "counts.h5ad"
    fit_cells_path = shared_root / "pca_fit_cells.csv"
    if not counts_path.is_file() or not fit_cells_path.is_file():
        raise FileNotFoundError("paired incremental PCA requires shared counts and pca_fit_cells.csv")
    shared = ad.read_h5ad(counts_path)
    shared_ids = shared.obs["cell_id"].astype(str)
    lookup = pd.Series(np.arange(len(shared_ids)), index=shared_ids)
    fit_ids = pd.read_csv(fit_cells_path)["cell_id"].astype(str)
    fit_rows = fit_ids.map(lookup)
    if fit_rows.isna().any():
        raise EmbeddingRunnerError("pca_fit_cells.csv contains cells absent from shared counts")
    n_components = _required_int(provider, ("n_components",))
    chunk_size = int(provider.get("chunk_size", 2048))
    model = fit_incremental_pca(
        shared.X[fit_rows.to_numpy(dtype=int)],
        n_components=n_components,
        n_top_genes=int(provider.get("n_top_genes", 2000)),
        chunk_size=chunk_size,
        target_sum=float(provider.get("target_sum", 10_000.0)),
    )
    shared_embedding = transform_incremental_pca(
        shared.X,
        model,
        n_components=n_components,
        chunk_size=chunk_size,
        target_sum=float(provider.get("target_sum", 10_000.0)),
    )
    for condition in conditions:
        cells = pd.read_csv(
            run_root / "benchmark" / condition / "model_visible" / "cells.csv"
        )
        rows = cells["cell_id"].astype(str).map(lookup)
        if rows.isna().any():
            raise EmbeddingRunnerError(f"{condition} contains cells absent from shared counts")
        embedding = shared_embedding[rows.to_numpy(dtype=int)]
        _write_embedding_artifacts(
            embeddings_root=embeddings_root,
            condition=condition,
            provider_name=provider_name,
            embedding=embedding,
            embedding_2d=embedding[:, :2],
            cells=cells,
            metadata={
                "condition": condition,
                "provider": provider_name,
                "method": "paired_incremental_pca",
                "fit_condition": "ablated_reference_training_cells",
                "shared_fit": True,
                "requested_components": n_components,
                "fitted_components": int(model.components.shape[0]),
            },
        )


def _run_obsm_provider(
    run_root: Path,
    embeddings_root: Path,
    condition: str,
    provider_name: str,
    obsm_key: str,
    n_components: int,
) -> None:
    counts_path = run_root / "benchmark" / condition / "model_visible" / "counts.h5ad"
    cells_path = run_root / "benchmark" / condition / "model_visible" / "cells.csv"
    for path in (counts_path, cells_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required embedding input does not exist: {path}")

    cells = pd.read_csv(cells_path)
    validate_required_columns(cells, MODEL_VISIBLE_CELLS_COLUMNS, str(cells_path))
    counts = read_condition_adata(counts_path, cells)
    if obsm_key not in counts.obsm:
        raise EmbeddingRunnerError(f"counts.h5ad is missing obsm embedding {obsm_key!r}")

    matrix = np.asarray(counts.obsm[obsm_key])
    if matrix.ndim != 2:
        raise EmbeddingRunnerError(f"obsm embedding {obsm_key!r} must be two-dimensional")
    if matrix.shape[0] != len(cells):
        raise EmbeddingRunnerError(
            f"obsm embedding rows ({matrix.shape[0]}) must match cells.csv rows ({len(cells)})"
        )
    if matrix.shape[1] < n_components:
        raise EmbeddingRunnerError(
            f"obsm embedding {obsm_key!r} has {matrix.shape[1]} columns; need {n_components}"
        )

    embedding = matrix[:, :n_components].astype(float, copy=True)
    embedding_2d = embedding[:, :2] if embedding.shape[1] >= 2 else np.pad(embedding, ((0, 0), (0, 1)))
    _write_embedding_artifacts(
        embeddings_root=embeddings_root,
        condition=condition,
        provider_name=provider_name,
        embedding=embedding,
        embedding_2d=embedding_2d,
        cells=cells,
        metadata={
            "condition": condition,
            "provider": provider_name,
            "method": "obsm",
            "obsm_key": obsm_key,
            "requested_components": n_components,
            "fitted_components": int(embedding.shape[1]),
        },
    )


def _run_pca_provider(
    run_root: Path,
    embeddings_root: Path,
    condition: str,
    provider_name: str,
    n_components: int,
) -> None:
    expression_path = run_root / "derived" / condition / "processed_expression.npz"
    cells_path = run_root / "benchmark" / condition / "model_visible" / "cells.csv"
    for path in (expression_path, cells_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required embedding input does not exist: {path}")

    matrix = sparse.load_npz(expression_path)
    cells = pd.read_csv(cells_path)
    validate_required_columns(cells, MODEL_VISIBLE_CELLS_COLUMNS, str(cells_path))
    if matrix.shape[0] != len(cells):
        raise EmbeddingRunnerError(
            f"processed_expression rows ({matrix.shape[0]}) must match cells.csv rows ({len(cells)})"
        )

    pca = compute_pca_embedding(matrix, n_components=n_components)
    _write_embedding_artifacts(
        embeddings_root=embeddings_root,
        condition=condition,
        provider_name=provider_name,
        embedding=pca.embedding,
        embedding_2d=pca.embedding_2d,
        cells=cells,
        metadata={
            "condition": condition,
            "provider": provider_name,
            "method": "pca",
            "requested_components": pca.requested_components,
            "fitted_components": pca.fitted_components,
        },
    )


def _write_embedding_artifacts(
    *,
    embeddings_root: Path,
    condition: str,
    provider_name: str,
    embedding: np.ndarray,
    embedding_2d: np.ndarray,
    cells: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    provider_root = embeddings_root / condition / provider_name
    provider_root.mkdir(parents=True, exist_ok=True)
    np.save(provider_root / "embedding.npy", embedding)
    np.save(provider_root / "embedding_2d.npy", embedding_2d)

    embedding_cells = pd.DataFrame(
        {
            "row_index": np.arange(len(cells)),
            "cell_id": cells["cell_id"].astype(str),
            "domain": cells["domain"].astype(str),
            "condition_id": condition,
        }
    )
    assert_embedding_row_identity(embedding_cells, cells)
    embedding_cells.to_csv(
        provider_root / "embedding_cells.csv", index=False, columns=EMBEDDING_CELLS_COLUMNS
    )
    write_manifest(
        provider_root / "manifest.json",
        Manifest(
            stage=STAGE,
            artifacts={
                "embedding": str(provider_root / "embedding.npy"),
                "embedding_2d": str(provider_root / "embedding_2d.npy"),
                "embedding_cells": str(provider_root / "embedding_cells.csv"),
            },
            metadata=metadata,
        ),
    )


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise EmbeddingRunnerError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_int(config: dict[str, Any], path: tuple[str, ...]) -> int:
    value = _required_value(config, path)
    if not isinstance(value, int):
        dotted = ".".join(path)
        raise EmbeddingRunnerError(f"Expected integer config value: {dotted}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise EmbeddingRunnerError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
