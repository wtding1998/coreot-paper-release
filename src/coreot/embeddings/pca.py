from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


class PCAEmbeddingError(ValueError):
    """Raised when PCA embedding cannot be computed."""


@dataclass(frozen=True)
class PCAEmbedding:
    embedding: np.ndarray
    embedding_2d: np.ndarray
    requested_components: int
    fitted_components: int


def compute_pca_embedding(matrix: sparse.spmatrix, n_components: int, random_state: int = 0) -> PCAEmbedding:
    if n_components <= 0:
        raise PCAEmbeddingError("PCA n_components must be positive")

    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    if dense.ndim != 2:
        raise PCAEmbeddingError("PCA input matrix must be two-dimensional")
    n_cells, n_features = dense.shape
    if n_cells == 0 or n_features == 0:
        raise PCAEmbeddingError("PCA input matrix must have nonzero cells and features")

    fitted_components = min(n_components, n_cells, n_features)
    fitted = PCA(n_components=fitted_components, svd_solver="full", random_state=random_state).fit_transform(
        dense
    )
    embedding = _pad_columns(fitted, n_components)
    embedding_2d = _pad_columns(fitted[:, : min(2, fitted.shape[1])], 2)
    return PCAEmbedding(
        embedding=embedding,
        embedding_2d=embedding_2d,
        requested_components=n_components,
        fitted_components=fitted_components,
    )


def _pad_columns(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.shape[1] == width:
        return matrix
    padded = np.zeros((matrix.shape[0], width), dtype=matrix.dtype)
    padded[:, : matrix.shape[1]] = matrix
    return padded
