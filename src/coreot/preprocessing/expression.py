from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


class ExpressionPreprocessingError(ValueError):
    """Raised when expression preprocessing configuration is invalid."""


@dataclass(frozen=True)
class PreprocessedExpression:
    matrix: sparse.csr_matrix
    selected_gene_indices: np.ndarray


def preprocess_expression(matrix: Any, config: dict[str, Any]) -> PreprocessedExpression:
    working = _to_dense_float(matrix)

    if bool(config.get("normalize_total", False)):
        target_sum = float(config.get("target_sum", 10000))
        if target_sum <= 0:
            raise ExpressionPreprocessingError("preprocessing.target_sum must be positive")
        row_sums = working.sum(axis=1)
        scale = np.divide(target_sum, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
        working = working * scale[:, np.newaxis]

    if bool(config.get("log1p", False)):
        if np.any(working < 0):
            raise ExpressionPreprocessingError("log1p preprocessing requires nonnegative expression")
        working = np.log1p(working)

    selected = _select_hvg_indices(working, config)
    working = working[:, selected]

    if bool(config.get("scale", False)):
        mean = working.mean(axis=0)
        std = working.std(axis=0)
        std[std == 0] = 1.0
        working = (working - mean) / std

    return PreprocessedExpression(matrix=sparse.csr_matrix(working), selected_gene_indices=selected)


def _to_dense_float(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(float, copy=False)
    return np.asarray(matrix, dtype=float)


def _select_hvg_indices(matrix: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    hvg = config.get("hvg", {})
    if not isinstance(hvg, dict) or not bool(hvg.get("enabled", False)):
        return np.arange(matrix.shape[1])

    n_top = int(hvg.get("n_top_genes", matrix.shape[1]))
    if n_top <= 0:
        raise ExpressionPreprocessingError("preprocessing.hvg.n_top_genes must be positive")
    n_selected = min(n_top, matrix.shape[1])
    variances = matrix.var(axis=0)
    ordered = np.argsort(-variances, kind="mergesort")[:n_selected]
    return np.sort(ordered)
