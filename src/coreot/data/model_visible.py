from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from coreot.data.schemas import MODEL_VISIBLE_CELLS_COLUMNS
from coreot.data.validation import validate_required_columns


class ModelVisibleMatrixError(ValueError):
    """Raised when a model-visible count matrix cannot be aligned to its cell manifest."""


def read_condition_adata(counts_path: str | Path, cells: pd.DataFrame) -> ad.AnnData:
    """Read and order active condition rows from a dedicated or shared AnnData matrix."""

    path = Path(counts_path)
    validate_required_columns(cells, MODEL_VISIBLE_CELLS_COLUMNS, "model-visible cells")
    adata = ad.read_h5ad(path, backed="r")
    try:
        if "cell_id" not in adata.obs.columns:
            raise ModelVisibleMatrixError(f"{path} is missing .obs['cell_id']")
        matrix_ids = adata.obs["cell_id"].astype(str)
        if matrix_ids.duplicated().any():
            raise ModelVisibleMatrixError(f"{path} contains duplicated cell_id values")
        lookup = pd.Series(range(len(matrix_ids)), index=matrix_ids)
        requested = cells["cell_id"].astype(str)
        rows = requested.map(lookup)
        if rows.isna().any():
            missing = requested.loc[rows.isna()].tolist()
            raise ModelVisibleMatrixError(
                f"{path} does not contain manifest cell(s): {missing[:5]}"
            )
        out = adata[rows.to_numpy(dtype=int)].to_memory()
    finally:
        adata.file.close()
    out.obs = cells.reset_index(drop=True).copy()
    out.obs_names = out.obs["cell_id"].astype(str)
    return out
