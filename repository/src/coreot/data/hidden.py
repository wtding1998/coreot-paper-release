from __future__ import annotations

from pathlib import Path

EVALUATION_TRUTH_DIRNAME = "evaluation_truth"


def is_evaluation_truth_path(path: str | Path) -> bool:
    return EVALUATION_TRUTH_DIRNAME in Path(path).parts
