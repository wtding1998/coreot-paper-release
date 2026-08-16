from __future__ import annotations

import pandas as pd

from coreot.data.validation import validate_required_columns

EXTERNAL_BASELINE_METHODS: tuple[str, ...] = (
    "seurat_anchor",
    "singleR",
    "celltypist_l3",
    "scmap_cell",
    "scmap_cluster",
    "chetah",
)

DEFAULT_EXTERNAL_BASELINE_METHODS: tuple[str, ...] = (
    "seurat_anchor",
    "singleR",
    "celltypist_l3",
    "scmap_cell",
    "scmap_cluster",
    "chetah",
)

EXTERNAL_BASELINE_PREDICTION_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "method",
    "heldout_label",
    "repeat",
    "is_full_reference_control",
    "pred_label",
    "native_pred_label",
    "z_absent_score",
    "native_abstain",
    "max_confidence_or_similarity",
)


def validate_external_baseline_predictions(
    frame: pd.DataFrame, *, method: str, source: str
) -> pd.DataFrame:
    validate_required_columns(frame, EXTERNAL_BASELINE_PREDICTION_COLUMNS, source)
    if method not in EXTERNAL_BASELINE_METHODS:
        raise ValueError(f"Unsupported external baseline method: {method}")
    out = frame.loc[:, EXTERNAL_BASELINE_PREDICTION_COLUMNS].copy()
    out["cell_id"] = out["cell_id"].astype(str)
    out["method"] = out["method"].astype(str)
    out["pred_label"] = out["pred_label"].fillna("").astype(str)
    out["native_pred_label"] = out["native_pred_label"].fillna("").astype(str)
    if set(out["method"]) != {method}:
        raise ValueError(f"{source} must contain only method={method!r}")
    if out["cell_id"].duplicated().any():
        duplicated = sorted(out.loc[out["cell_id"].duplicated(), "cell_id"].unique())
        raise ValueError(f"{source} contains duplicated cell_id values: {duplicated[:5]}")
    out["z_absent_score"] = pd.to_numeric(out["z_absent_score"], errors="coerce")
    if out["z_absent_score"].isna().any():
        raise ValueError(f"{source} contains non-numeric z_absent_score values")
    out["max_confidence_or_similarity"] = pd.to_numeric(
        out["max_confidence_or_similarity"], errors="coerce"
    )
    out["native_abstain"] = out["native_abstain"].astype(bool)
    out["is_full_reference_control"] = out["is_full_reference_control"].astype(bool)
    return out
