from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.candidates.knn import build_candidate_edges
from coreot.candidates.scaling import scale_distances
from coreot.config.load import load_yaml
from coreot.data.schemas import EMBEDDING_CELLS_COLUMNS
from coreot.data.validation import validate_required_columns, validate_stage_can_read


STAGE = "candidate-cost"


class CandidateCostRunnerError(ValueError):
    """Raised when candidate-cost artifacts cannot be constructed."""


@dataclass(frozen=True)
class CandidateCostResult:
    candidates_root: Path
    conditions: tuple[str, ...]
    providers: tuple[str, ...]


def run_candidate_cost(config_path: str | Path) -> CandidateCostResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    output_root = _required_path(config, ("outputs", "root"))
    conditions = tuple(config.get("conditions", ()))
    providers = tuple(config.get("providers", ()))
    if not conditions:
        raise CandidateCostRunnerError("candidate-cost requires conditions")
    if not providers:
        raise CandidateCostRunnerError("candidate-cost requires providers")

    graph_config = _required_mapping(config, ("candidate_graph",))
    scaling_config = _required_mapping(config, ("cost_scaling",))
    run_root = output_root / run_id
    candidates_root = run_root / "candidates"

    for condition in conditions:
        for provider in providers:
            if not isinstance(provider, str) or not provider:
                raise CandidateCostRunnerError("candidate-cost providers must be provider-name strings")
            _run_provider(run_root, candidates_root, condition, provider, graph_config, scaling_config)

    return CandidateCostResult(candidates_root=candidates_root, conditions=conditions, providers=providers)


def _run_provider(
    run_root: Path,
    candidates_root: Path,
    condition: str,
    provider: str,
    graph_config: dict[str, Any],
    scaling_config: dict[str, Any],
) -> None:
    embedding_root = run_root / "embeddings" / condition / provider
    embedding_path = embedding_root / "embedding.npy"
    cells_path = embedding_root / "embedding_cells.csv"
    for path in (embedding_path, cells_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required candidate-cost input does not exist: {path}")

    embedding = np.load(embedding_path)
    embedding_cells = pd.read_csv(cells_path)
    validate_required_columns(embedding_cells, EMBEDDING_CELLS_COLUMNS, str(cells_path))
    if embedding.shape[0] != len(embedding_cells):
        raise CandidateCostRunnerError("embedding.npy rows must match embedding_cells.csv rows")

    k = _required_int(graph_config, ("k_source_to_target",))
    target_broad_labels = None
    k_per_class = int(graph_config.get("k_per_target_broad_class", 0))
    if k_per_class > 0:
        labels_path = run_root / "benchmark" / condition / "model_visible" / "target_labels.csv"
        labels = pd.read_csv(labels_path)
        target_broad_labels = labels.set_index(labels["cell_id"].astype(str))["broad_label"]
    edges = build_candidate_edges(
        embedding,
        embedding_cells,
        k_source_to_target=k,
        add_reverse_edges=bool(graph_config.get("add_reverse_edges", False)),
        distance=_required_str(graph_config, ("distance",)),
        k_target_to_source=int(graph_config.get("k_target_to_source", k)),
        target_broad_labels=target_broad_labels,
        k_per_target_broad_class=k_per_class,
    )
    distances = edges["distance"].to_numpy(dtype=float)
    reference_condition = scaling_config.get("reference_condition")
    if reference_condition is None:
        edges["scaled_distance"] = scale_distances(
            distances,
            scale=_required_str(scaling_config, ("scale",)),
            clip_quantile=_required_float(scaling_config, ("clip_quantile",)),
            delta=_required_float(scaling_config, ("delta",)),
        )
    else:
        scale_path = candidates_root / f"{provider}_distance_scale.txt"
        if condition == str(reference_condition):
            denominator = float(np.median(distances)) + _required_float(scaling_config, ("delta",))
            scale_path.parent.mkdir(parents=True, exist_ok=True)
            scale_path.write_text(f"{denominator:.17g}\n", encoding="utf-8")
        elif scale_path.is_file():
            denominator = float(scale_path.read_text(encoding="utf-8"))
        else:
            raise CandidateCostRunnerError(
                "cost scaling reference condition must be processed before paired conditions"
            )
        scaled = distances / denominator
        clip_quantile = _required_float(scaling_config, ("clip_quantile",))
        edges["scaled_distance"] = np.minimum(scaled, np.quantile(scaled, clip_quantile))
    edges = edges[
        [
            "source_cell_id",
            "target_cell_id",
            "source_row",
            "target_row",
            "distance",
            "scaled_distance",
            "is_reverse_edge",
        ]
    ]

    output_root = candidates_root / condition / f"{provider}_k{k}"
    output_root.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(output_root / "candidate_edges.parquet", index=False)
    np.savez_compressed(
        output_root / "cost_arrays.npz",
        source_row=edges["source_row"].to_numpy(dtype=int),
        target_row=edges["target_row"].to_numpy(dtype=int),
        distance=edges["distance"].to_numpy(dtype=float),
        scaled_distance=edges["scaled_distance"].to_numpy(dtype=float),
        is_reverse_edge=edges["is_reverse_edge"].to_numpy(dtype=bool),
    )
    write_manifest(
        output_root / "candidate_manifest.yaml",
        Manifest(
            stage=STAGE,
            artifacts={
                "candidate_edges": str(output_root / "candidate_edges.parquet"),
                "cost_arrays": str(output_root / "cost_arrays.npz"),
            },
            metadata={
                "condition": condition,
                "provider": provider,
                "k_source_to_target": k,
                "add_reverse_edges": bool(graph_config.get("add_reverse_edges", False)),
                "distance": graph_config["distance"],
                "n_edges": int(len(edges)),
            },
        ),
    )


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise CandidateCostRunnerError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_int(config: dict[str, Any], path: tuple[str, ...]) -> int:
    value = _required_value(config, path)
    if not isinstance(value, int):
        dotted = ".".join(path)
        raise CandidateCostRunnerError(f"Expected integer config value: {dotted}")
    return value


def _required_float(config: dict[str, Any], path: tuple[str, ...]) -> float:
    value = _required_value(config, path)
    if not isinstance(value, int | float):
        dotted = ".".join(path)
        raise CandidateCostRunnerError(f"Expected numeric config value: {dotted}")
    return float(value)


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_mapping(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value = _required_value(config, path)
    if not isinstance(value, dict):
        dotted = ".".join(path)
        raise CandidateCostRunnerError(f"Expected mapping config value: {dotted}")
    return value


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise CandidateCostRunnerError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
