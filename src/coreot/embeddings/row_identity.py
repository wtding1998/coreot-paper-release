from __future__ import annotations

import pandas as pd

from coreot.data.validation import validate_embedding_cells


def assert_embedding_row_identity(
    embedding_cells: pd.DataFrame, model_visible_cells: pd.DataFrame
) -> None:
    validate_embedding_cells(embedding_cells, model_visible_cells)
