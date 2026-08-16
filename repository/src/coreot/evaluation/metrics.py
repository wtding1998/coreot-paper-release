from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auroc(y_true: pd.Series, score: pd.Series) -> float:
    y, values = _finite_binary_score_arrays(y_true, score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, values))


def safe_auprc(y_true: pd.Series, score: pd.Series) -> float:
    y, values = _finite_binary_score_arrays(y_true, score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, values))


def safe_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return float("nan")
    return float(finite.median())


def _finite_binary_score_arrays(y_true: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"y_true": y_true.astype(bool), "score": pd.to_numeric(score, errors="coerce")})
    frame = frame.dropna(subset=["score"])
    return frame["y_true"].astype(int).to_numpy(), frame["score"].astype(float).to_numpy()
