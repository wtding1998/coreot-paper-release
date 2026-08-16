from __future__ import annotations

import numpy as np
import pandas as pd


def one_hot_label_probabilities(
    cell_ids: pd.Series,
    forced_labels: pd.Series,
    label_order: list[str],
) -> dict[str, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(label_order)}
    probabilities = np.zeros((len(cell_ids), len(label_order)), dtype=float)
    for row_index, label in enumerate(forced_labels.astype(str)):
        probabilities[row_index, label_to_index[label]] = 1.0
    return {
        "cell_ids": cell_ids.astype(str).to_numpy(),
        "labels": np.asarray(label_order, dtype=object),
        "probabilities": probabilities,
    }


def coupling_label_probabilities(
    *,
    cell_ids: pd.Series,
    source_index: np.ndarray,
    target_index: np.ndarray,
    coupling: np.ndarray,
    source_marginal: np.ndarray,
    target_labels: pd.Series,
    eta: float,
) -> dict[str, np.ndarray]:
    label_order = sorted(target_labels.astype(str).unique())
    label_to_index = {label: index for index, label in enumerate(label_order)}
    probabilities = np.zeros((len(cell_ids), len(label_order)), dtype=float)
    label_indices = target_labels.astype(str).map(label_to_index).to_numpy(dtype=int)

    for edge_index, mass in enumerate(coupling):
        row = source_index[edge_index]
        denominator = source_marginal[row]
        if denominator > eta:
            probabilities[row, label_indices[target_index[edge_index]]] += mass / denominator

    return {
        "cell_ids": cell_ids.astype(str).to_numpy(),
        "labels": np.asarray(label_order, dtype=object),
        "probabilities": probabilities,
    }
