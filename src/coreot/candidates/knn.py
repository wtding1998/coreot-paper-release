from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class CandidateKNNError(ValueError):
    """Raised when candidate nearest-neighbor construction is invalid."""


def build_candidate_edges(
    embedding: np.ndarray,
    embedding_cells: pd.DataFrame,
    *,
    k_source_to_target: int,
    add_reverse_edges: bool,
    distance: str,
    k_target_to_source: int | None = None,
    target_broad_labels: pd.Series | None = None,
    k_per_target_broad_class: int = 0,
) -> pd.DataFrame:
    if distance != "euclidean":
        raise CandidateKNNError(f"Only euclidean candidate distance is implemented; got {distance!r}")
    if k_source_to_target <= 0:
        raise CandidateKNNError("candidate_graph.k_source_to_target must be positive")

    source_rows = embedding_cells.index[embedding_cells["domain"].astype(str) == "query"].to_numpy()
    target_rows = embedding_cells.index[embedding_cells["domain"].astype(str) == "reference"].to_numpy()
    if len(source_rows) == 0 or len(target_rows) == 0:
        raise CandidateKNNError("candidate graph requires nonempty query and reference rows")

    edge_map: dict[tuple[int, int], dict[str, object]] = {}
    k_forward = min(k_source_to_target, len(target_rows))
    forward_distances, forward_indices = _nearest(
        query=embedding[source_rows],
        fit=embedding[target_rows],
        k=k_forward,
    )
    for source_pos, source_row in enumerate(source_rows):
        for neighbor_pos, distance_value in zip(
            forward_indices[source_pos], forward_distances[source_pos], strict=True
        ):
            target_row = int(target_rows[neighbor_pos])
            edge_map[(int(source_row), target_row)] = {
                "source_row": int(source_row),
                "target_row": target_row,
                "distance": float(distance_value),
                "is_reverse_edge": False,
            }

    if add_reverse_edges:
        k_reverse = min(k_target_to_source or k_source_to_target, len(source_rows))
        reverse_distances, reverse_indices = _nearest(
            query=embedding[target_rows],
            fit=embedding[source_rows],
            k=k_reverse,
        )
        for target_pos, target_row in enumerate(target_rows):
            for neighbor_pos, distance_value in zip(
                reverse_indices[target_pos], reverse_distances[target_pos], strict=True
            ):
                source_row = int(source_rows[neighbor_pos])
                key = (source_row, int(target_row))
                edge_map.setdefault(
                    key,
                    {
                        "source_row": source_row,
                        "target_row": int(target_row),
                        "distance": float(distance_value),
                        "is_reverse_edge": True,
                    },
                )

    if k_per_target_broad_class > 0:
        if target_broad_labels is None:
            raise CandidateKNNError("per-broad-class candidates require target broad labels")
        broad_by_row = target_broad_labels.astype(str)
        for broad_class in sorted(broad_by_row.unique()):
            class_rows = target_rows[
                embedding_cells.loc[target_rows, "cell_id"].astype(str).map(broad_by_row).to_numpy()
                == broad_class
            ]
            if not len(class_rows):
                continue
            distances, indices = _nearest(
                query=embedding[source_rows],
                fit=embedding[class_rows],
                k=min(k_per_target_broad_class, len(class_rows)),
            )
            for source_pos, source_row in enumerate(source_rows):
                for neighbor_pos, distance_value in zip(indices[source_pos], distances[source_pos], strict=True):
                    target_row = int(class_rows[neighbor_pos])
                    edge_map.setdefault(
                        (int(source_row), target_row),
                        {"source_row": int(source_row), "target_row": target_row, "distance": float(distance_value), "is_reverse_edge": False},
                    )

    edges = pd.DataFrame(edge_map.values())
    edges = edges.sort_values(["source_row", "target_row"]).reset_index(drop=True)
    edges.insert(0, "source_cell_id", embedding_cells.loc[edges["source_row"], "cell_id"].to_numpy())
    edges.insert(1, "target_cell_id", embedding_cells.loc[edges["target_row"], "cell_id"].to_numpy())
    return edges


def _nearest(query: np.ndarray, fit: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    model.fit(fit)
    return model.kneighbors(query, return_distance=True)
