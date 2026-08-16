"""Prior-adjusted deficit scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BIOLOGICAL_FOLD_COLUMNS = ("donor_id", "sample_id", "batch_id")


@dataclass(frozen=True)
class PriorAdjustedDeficit:
    expected_u: pd.Series
    u_tilde: pd.Series
    percentile_residual: pd.Series
    metadata: dict[str, Any]


def compute_prior_adjusted_deficit(
    scores: pd.DataFrame,
    *,
    n_folds: int = 5,
    stratify_by_anchor: bool = False,
    min_anchor_size: int = 20,
    compute_percentile_residual: bool = False,
) -> PriorAdjustedDeficit:
    """Estimate expected raw deficit from prior risk and return residuals."""
    if not isinstance(stratify_by_anchor, bool):
        raise ValueError("prior adjustment stratify_by_anchor must be boolean")
    if not isinstance(min_anchor_size, int) or min_anchor_size < 2:
        raise ValueError("prior adjustment min_anchor_size must be an integer >= 2")
    if not isinstance(compute_percentile_residual, bool):
        raise ValueError("prior adjustment compute_percentile_residual must be boolean")
    if not stratify_by_anchor:
        return _compute_global_prior_adjusted_deficit(
            scores,
            n_folds=n_folds,
            compute_percentile_residual=compute_percentile_residual,
        )
    if "anchor_class_pred" not in scores.columns:
        result = _compute_global_prior_adjusted_deficit(
            scores,
            n_folds=n_folds,
            compute_percentile_residual=compute_percentile_residual,
        )
        return PriorAdjustedDeficit(
            expected_u=result.expected_u,
            u_tilde=result.u_tilde,
            percentile_residual=result.percentile_residual,
            metadata=result.metadata
            | {
                "anchor_stratification": "global_fallback",
                "fallback": "missing_anchor_metadata",
                "min_anchor_size": min_anchor_size,
            },
        )

    global_result = _compute_global_prior_adjusted_deficit(
        scores,
        n_folds=n_folds,
        compute_percentile_residual=compute_percentile_residual,
    )
    frame = scores.loc[:, ["cell_id", "u", "prior_risk", "anchor_class_pred"]].copy()
    u = pd.to_numeric(frame["u"], errors="coerce")
    prior_risk = pd.to_numeric(frame["prior_risk"], errors="coerce")
    anchor_class = frame["anchor_class_pred"].astype("string")
    valid_u_prior = u.notna() & prior_risk.notna()
    valid = valid_u_prior & anchor_class.notna() & anchor_class.str.len().gt(0)
    class_counts = anchor_class.loc[valid].value_counts()
    large_classes = sorted(class_counts[class_counts >= min_anchor_size].index.astype(str).tolist())
    if not large_classes:
        return PriorAdjustedDeficit(
            expected_u=global_result.expected_u,
            u_tilde=global_result.u_tilde,
            percentile_residual=global_result.percentile_residual,
            metadata=global_result.metadata
            | {
                "anchor_stratification": "global_fallback",
                "fallback": "anchor_classes_below_min_size",
                "min_anchor_size": min_anchor_size,
            },
        )

    expected = global_result.expected_u.copy()
    percentile = global_result.percentile_residual.copy()
    anchor_class_folds: dict[str, dict[str, Any]] = {}
    for anchor in large_classes:
        anchor_mask = valid & anchor_class.eq(anchor)
        anchor_result = _compute_global_prior_adjusted_deficit(
            scores.loc[anchor_mask].copy(),
            n_folds=n_folds,
            compute_percentile_residual=compute_percentile_residual,
        )
        expected.loc[anchor_result.expected_u.index] = anchor_result.expected_u
        percentile.loc[anchor_result.percentile_residual.index] = (
            anchor_result.percentile_residual
        )
        anchor_class_folds[anchor] = _fold_provenance(anchor_result.metadata)

    fallback = "none"
    if int(valid_u_prior.sum()) != int(class_counts.loc[large_classes].sum()):
        fallback = "global_for_small_anchor_classes"
    return PriorAdjustedDeficit(
        expected_u=expected,
        u_tilde=u - expected,
        percentile_residual=percentile,
        metadata={
            "model": "cross_fitted_isotonic",
            "anchor_stratification": "anchor_stratified",
            "fallback": fallback,
            "min_anchor_size": min_anchor_size,
            "anchor_classes": large_classes,
            "anchor_class_folds": anchor_class_folds,
        }
        | _summarize_anchor_fold_provenance(anchor_class_folds),
    )


def _compute_global_prior_adjusted_deficit(
    scores: pd.DataFrame,
    *,
    n_folds: int,
    compute_percentile_residual: bool,
) -> PriorAdjustedDeficit:
    """Estimate expected raw deficit globally from prior risk and return residuals."""
    required = {"cell_id", "u", "prior_risk"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"prior adjustment scores missing required columns: {missing}")
    if n_folds < 2:
        raise ValueError("prior adjustment requires at least 2 folds")

    columns = [
        "cell_id",
        "u",
        "prior_risk",
        *[col for col in BIOLOGICAL_FOLD_COLUMNS if col in scores.columns],
    ]
    frame = scores.loc[:, columns].copy()
    u = pd.to_numeric(frame["u"], errors="coerce")
    prior_risk = pd.to_numeric(frame["prior_risk"], errors="coerce")
    valid = u.notna() & prior_risk.notna()
    expected = pd.Series(np.nan, index=frame.index, dtype=float)
    percentile = pd.Series(np.nan, index=frame.index, dtype=float)
    valid_positions = np.flatnonzero(valid.to_numpy())
    if len(valid_positions) < 2:
        return PriorAdjustedDeficit(
            expected_u=expected,
            u_tilde=u - expected,
            percentile_residual=percentile,
            metadata={
                "model": "cross_fitted_isotonic",
                "cross_fitting": "not_enough_cells",
                "n_folds": 0,
                "fold_source": "none",
                "anchor_stratification": "global",
                "fallback": "all_missing",
            },
        )

    folds, metadata = _make_folds(frame, valid_positions, int(n_folds))
    for test_positions in folds:
        train_positions = np.setdiff1d(valid_positions, test_positions, assume_unique=True)
        model = _fit_isotonic(prior_risk.iloc[train_positions], u.iloc[train_positions])
        expected.iloc[test_positions] = model.predict(prior_risk.iloc[test_positions].to_numpy())
        if compute_percentile_residual:
            percentile.iloc[test_positions] = _nearest_prior_percentiles(
                u=u,
                prior_risk=prior_risk,
                train_positions=train_positions,
                test_positions=test_positions,
            )

    return PriorAdjustedDeficit(
        expected_u=expected,
        u_tilde=u - expected,
        percentile_residual=percentile,
        metadata={
            "model": "cross_fitted_isotonic",
            "anchor_stratification": "global",
        }
        | metadata,
    )


def _make_folds(
    frame: pd.DataFrame, valid_positions: np.ndarray, n_folds: int
) -> tuple[list[np.ndarray], dict[str, Any]]:
    for column in BIOLOGICAL_FOLD_COLUMNS:
        if column not in frame.columns:
            continue
        group_values = frame.iloc[valid_positions][column].astype("string")
        if group_values.isna().any() or group_values.str.len().eq(0).any():
            continue
        groups = sorted(group_values.unique().tolist())
        if len(groups) < 2:
            continue
        fold_count = min(n_folds, len(groups))
        group_to_fold = {group: group_index % fold_count for group_index, group in enumerate(groups)}
        fold_ids = group_values.map(group_to_fold).to_numpy(dtype=int)
        folds = [valid_positions[fold_ids == fold_id] for fold_id in range(fold_count)]
        return folds, {
            "cross_fitting": "group",
            "n_folds": fold_count,
            "fold_source": column,
            "fallback": "none",
        }

    fold_count = min(n_folds, len(valid_positions))
    folds = [
        valid_positions[np.arange(len(valid_positions)) % fold_count == fold_id]
        for fold_id in range(fold_count)
    ]
    return folds, {
        "cross_fitting": "cell",
        "n_folds": fold_count,
        "fold_source": "deterministic_cell",
        "fallback": "none",
    }


def _fold_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in ("cross_fitting", "n_folds", "fold_source", "fallback")
        if key in metadata
    }


def _summarize_anchor_fold_provenance(
    anchor_class_folds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fold_metadata = list(anchor_class_folds.values())
    first = fold_metadata[0]
    first_fold_mode = (
        first.get("cross_fitting"),
        first.get("n_folds"),
        first.get("fold_source"),
    )
    if all(
        (
            metadata.get("cross_fitting"),
            metadata.get("n_folds"),
            metadata.get("fold_source"),
        )
        == first_fold_mode
        for metadata in fold_metadata
    ):
        return {
            "cross_fitting": first.get("cross_fitting"),
            "n_folds": first.get("n_folds"),
            "fold_source": first.get("fold_source"),
        }
    return {
        "cross_fitting": "mixed",
        "n_folds": None,
        "fold_source": "mixed",
    }


def _fit_isotonic(prior_risk: pd.Series, u: pd.Series) -> IsotonicRegression:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    return model.fit(prior_risk.to_numpy(dtype=float), u.to_numpy(dtype=float))


def _nearest_prior_percentiles(
    *,
    u: pd.Series,
    prior_risk: pd.Series,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
) -> np.ndarray:
    train_prior = prior_risk.iloc[train_positions].to_numpy(dtype=float)
    train_u = u.iloc[train_positions].to_numpy(dtype=float)
    percentiles = []
    for position in test_positions:
        test_prior = float(prior_risk.iloc[position])
        distances = np.abs(train_prior - test_prior)
        nearest = train_u[distances == distances.min()]
        percentiles.append(float(np.mean(nearest <= float(u.iloc[position]))))
    return np.array(percentiles, dtype=float)
