from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import IncrementalPCA


@dataclass(frozen=True)
class IncrementalPCAModel:
    selected_gene_indices: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray


def fit_incremental_pca(
    counts: sparse.spmatrix,
    *,
    n_components: int,
    n_top_genes: int,
    chunk_size: int = 2048,
    target_sum: float = 10_000.0,
) -> IncrementalPCAModel:
    matrix = sparse.csr_matrix(counts)
    if n_components <= 0 or chunk_size < n_components:
        raise ValueError("incremental PCA requires 0 < n_components <= chunk_size")
    transformed = _normalize_log1p(matrix, target_sum)
    mean = np.asarray(transformed.mean(axis=0)).ravel()
    second = np.asarray(transformed.multiply(transformed).mean(axis=0)).ravel()
    variance = np.maximum(second - mean * mean, 0.0)
    n_selected = min(n_top_genes, matrix.shape[1])
    selected = np.sort(np.argsort(-variance, kind="mergesort")[:n_selected])
    feature_mean = mean[selected]
    feature_scale = np.sqrt(variance[selected])
    feature_scale[feature_scale == 0.0] = 1.0
    fitted_components = min(n_components, matrix.shape[0], n_selected)
    model = IncrementalPCA(n_components=fitted_components, batch_size=chunk_size)
    for start, stop in _complete_chunks(matrix.shape[0], chunk_size, fitted_components):
        model.partial_fit(
            _standardized_chunk(transformed[start:stop, selected], feature_mean, feature_scale)
        )
    return IncrementalPCAModel(
        selected_gene_indices=selected,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        pca_mean=model.mean_.copy(),
        components=model.components_.copy(),
    )


def transform_incremental_pca(
    counts: sparse.spmatrix,
    model: IncrementalPCAModel,
    *,
    n_components: int,
    chunk_size: int = 2048,
    target_sum: float = 10_000.0,
) -> np.ndarray:
    transformed = _normalize_log1p(sparse.csr_matrix(counts), target_sum)
    output = np.zeros((transformed.shape[0], n_components), dtype=float)
    for start in range(0, transformed.shape[0], chunk_size):
        stop = min(start + chunk_size, transformed.shape[0])
        standardized = _standardized_chunk(
            transformed[start:stop, model.selected_gene_indices],
            model.feature_mean,
            model.feature_scale,
        )
        output[start:stop, : model.components.shape[0]] = (
            standardized - model.pca_mean
        ) @ model.components.T
    return output


def _normalize_log1p(matrix: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
    output = matrix.astype(float, copy=True)
    row_sum = np.asarray(output.sum(axis=1)).ravel()
    factors = np.divide(target_sum, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    output = sparse.diags(factors) @ output
    output.data = np.log1p(output.data)
    return output.tocsr()


def _standardized_chunk(
    matrix: sparse.spmatrix, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (matrix.toarray() - mean) / scale


def _complete_chunks(n_rows: int, chunk_size: int, minimum: int):
    start = 0
    while start < n_rows:
        stop = min(start + chunk_size, n_rows)
        if n_rows - stop and n_rows - stop < minimum:
            stop = n_rows
        yield start, stop
        start = stop
