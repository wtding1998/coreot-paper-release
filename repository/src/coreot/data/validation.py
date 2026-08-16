from __future__ import annotations

from pathlib import Path

import pandas as pd

from coreot.data.hidden import is_evaluation_truth_path
from coreot.data.schemas import EMBEDDING_CELLS_COLUMNS, MODEL_VISIBLE_CELLS_COLUMNS

MODEL_FACING_STAGES = frozenset(
    {
        "model-visible-derivation",
        "embedding",
        "candidate-cost",
        "transport",
        "score-abstain",
        "correction",
    }
)


class HiddenTruthAccessError(PermissionError):
    """Raised when a model-facing stage attempts to read hidden truth."""


def validate_stage_can_read(path: str | Path, stage: str) -> None:
    if stage in MODEL_FACING_STAGES and is_evaluation_truth_path(path):
        raise HiddenTruthAccessError(
            f"Stage {stage!r} cannot read evaluation-truth artifact: {path}"
        )


def validate_required_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def validate_model_visible_cells(frame: pd.DataFrame) -> None:
    validate_required_columns(frame, MODEL_VISIBLE_CELLS_COLUMNS, "model_visible/cells.csv")


def validate_embedding_cells(frame: pd.DataFrame, model_visible_cells: pd.DataFrame) -> None:
    validate_required_columns(frame, EMBEDDING_CELLS_COLUMNS, "embedding_cells.csv")
    validate_model_visible_cells(model_visible_cells)

    if not frame["row_index"].is_unique:
        raise ValueError("embedding_cells.csv row_index values must be unique")
    expected_row_index = list(range(len(frame)))
    if frame["row_index"].tolist() != expected_row_index:
        raise ValueError("embedding_cells.csv row_index values must be contiguous from zero")

    visible_ids = set(model_visible_cells["cell_id"])
    embedding_ids = set(frame["cell_id"])
    if embedding_ids != visible_ids:
        missing = sorted(visible_ids - embedding_ids)
        extra = sorted(embedding_ids - visible_ids)
        raise ValueError(
            "embedding_cells.csv cell IDs must match model_visible/cells.csv; "
            f"missing={missing}, extra={extra}"
        )
