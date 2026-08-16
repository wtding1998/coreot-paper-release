from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold


FloatArray = NDArray[np.float64]


class ProviderReliabilityError(ValueError):
    """Raised when a calibrated provider-reliability prior cannot be constructed."""


@dataclass(frozen=True)
class ProviderReliabilityResult:
    source: pd.DataFrame
    metadata: dict[str, Any]
    reliability_table: pd.DataFrame


class _ConstantCalibrator:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, score: FloatArray) -> FloatArray:
        return np.full(np.asarray(score).shape, self.value, dtype=float)


class _PlattCalibrator:
    def __init__(self, model: LogisticRegression) -> None:
        self.model = model

    def predict(self, score: FloatArray) -> FloatArray:
        return self.model.predict_proba(_score_logit(score))[:, 1]


def construct_provider_reliability(
    *,
    cell_ids: pd.Series,
    embedding: FloatArray,
    is_reference: NDArray[np.bool_],
    reference_labels: pd.Series,
    reference_donor_ids: pd.Series,
    clip_eps: float = 1.0e-6,
    classifier_c: float = 1.0,
    min_correct_isotonic: int = 25,
    min_error_isotonic: int = 25,
    min_distinct_isotonic: int = 10,
) -> ProviderReliabilityResult:
    ids = cell_ids.astype(str).reset_index(drop=True)
    matrix = np.asarray(embedding, dtype=float)
    reference = np.asarray(is_reference, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ProviderReliabilityError("embedding rows must match cell_ids")
    if reference.shape != (len(ids),):
        raise ProviderReliabilityError("is_reference must align with cell_ids")
    if not np.all(np.isfinite(matrix)):
        raise ProviderReliabilityError("embedding contains non-finite values")
    if not 0.0 < clip_eps < 0.5:
        raise ProviderReliabilityError("clip_eps must lie in (0, 0.5)")
    if classifier_c <= 0.0:
        raise ProviderReliabilityError("classifier_c must be positive")

    labels = reference_labels.astype(str).reset_index(drop=True)
    donors = reference_donor_ids.astype(str).reset_index(drop=True)
    x_reference = matrix[reference]
    if len(labels) != len(x_reference) or len(donors) != len(x_reference):
        raise ProviderReliabilityError("reference labels and donors must align with reference rows")
    if labels.nunique() < 2:
        raise ProviderReliabilityError("provider requires at least two reference broad classes")

    splits = _class_covering_group_splits(x_reference, labels.to_numpy(), donors.to_numpy())
    classes = np.sort(labels.unique())
    oof_score = np.empty(len(labels), dtype=float)
    oof_correct = np.empty(len(labels), dtype=int)
    for train, test in splits:
        model = _fit_provider(x_reference[train], labels.iloc[train], classifier_c)
        probabilities = _aligned_probabilities(model, x_reference[test], classes)
        predictions = classes[np.argmax(probabilities, axis=1)]
        oof_score[test] = probabilities.max(axis=1)
        oof_correct[test] = (predictions == labels.iloc[test].to_numpy()).astype(int)

    calibrator, calibration_metadata = fit_scalar_reliability_calibrator(
        reference_score_oof=oof_score,
        reference_correct_oof=oof_correct,
        reference_donor_id=donors.to_numpy(),
        min_correct_isotonic=min_correct_isotonic,
        min_error_isotonic=min_error_isotonic,
        min_distinct_isotonic=min_distinct_isotonic,
    )
    calibrated_reference = np.asarray(calibrator.predict(oof_score), dtype=float)
    weights = donor_equal_weights(donors.to_numpy())
    diagnostics = _calibration_diagnostics(
        score=oof_score,
        correct=oof_correct,
        calibrated=calibrated_reference,
        weights=weights,
    )

    final_provider = _fit_provider(x_reference, labels, classifier_c)
    source = ~reference
    source_score = _aligned_probabilities(final_provider, matrix[source], classes).max(axis=1)
    rho_raw = np.asarray(calibrator.predict(source_score), dtype=float)
    rho = np.clip(rho_raw, clip_eps, 1.0 - clip_eps)
    source_ids = ids[source].reset_index(drop=True)
    source_frame = pd.DataFrame(
        {
            "cell_id": source_ids,
            "coreot_provider_score_raw": source_score,
            "coreot_provider_reliability_calibrated": rho_raw,
            "coreot_rho_raw": rho_raw,
            "coreot_rho": rho,
        }
    )
    cell_id_hash = _cell_value_hash(source_frame["cell_id"], source_frame["coreot_rho"])
    metadata: dict[str, Any] = {
        "rho_mode": "calibrate_provider_score",
        "provider": "pca_broad_lineage_logistic",
        "provider_classifier_c": float(classifier_c),
        "provider_classes": classes.tolist(),
        "calibration_donor_ids": sorted(donors.unique().tolist()),
        "n_group_folds": len(splits),
        "numerical_clip_eps": float(clip_eps),
        "rho_mean": float(rho.mean()),
        "rho_sd": float(rho.std()),
        "rho_iqr": float(np.quantile(rho, 0.75) - np.quantile(rho, 0.25)),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_effectively_constant": bool(rho.std() < 0.02),
        "cell_id_rho_sha256": cell_id_hash,
        "random_seed": 0,
        "scikit_learn_version": sklearn.__version__,
        **calibration_metadata,
        **diagnostics,
    }
    return ProviderReliabilityResult(
        source=source_frame,
        metadata=metadata,
        reliability_table=_reliability_table(
            score=oof_score,
            correct=oof_correct,
            calibrated=calibrated_reference,
            weights=weights,
        ),
    )


def write_provider_reliability_artifacts(
    output_root: str | Path,
    result: ProviderReliabilityResult,
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    result.source.to_csv(root / "source_rho.csv", index=False)
    result.reliability_table.to_csv(root / "calibration_reliability.csv", index=False)
    (root / "rho_metadata.yaml").write_text(
        yaml.safe_dump(result.metadata, sort_keys=True),
        encoding="utf-8",
    )


def fit_scalar_reliability_calibrator(
    reference_score_oof: FloatArray,
    reference_correct_oof: NDArray[np.int_],
    reference_donor_id: NDArray[Any],
    *,
    min_correct_isotonic: int = 25,
    min_error_isotonic: int = 25,
    min_distinct_isotonic: int = 10,
) -> tuple[Any, dict[str, Any]]:
    score = _probability_vector("reference_score_oof", reference_score_oof)
    correct = np.asarray(reference_correct_oof, dtype=int)
    donors = np.asarray(reference_donor_id)
    if correct.shape != score.shape or donors.shape != score.shape:
        raise ProviderReliabilityError("calibration arrays must have identical shapes")
    if not np.isin(correct, [0, 1]).all():
        raise ProviderReliabilityError("reference_correct_oof must contain only 0 and 1")
    weights = donor_equal_weights(donors)
    n_correct = int(correct.sum())
    n_error = int(len(correct) - n_correct)
    n_distinct = int(np.unique(score).size)
    metadata: dict[str, Any] = {
        "n_reference": int(len(correct)),
        "n_correct": n_correct,
        "n_error": n_error,
        "n_distinct_score": n_distinct,
        "donor_equal_weight": True,
    }
    if (
        n_correct >= min_correct_isotonic
        and n_error >= min_error_isotonic
        and n_distinct >= min_distinct_isotonic
    ):
        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        ).fit(score, correct, sample_weight=weights)
        metadata.update(
            calibrator="isotonic",
            rho_informative=True,
            rho_fallback=None,
            fallback_reason=None,
        )
        return calibrator, metadata
    if n_correct > 0 and n_error > 0 and n_distinct > 1:
        model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=0)
        model.fit(_score_logit(score), correct, sample_weight=weights)
        if float(model.coef_[0, 0]) < 0.0:
            posterior = (n_correct + 1.0) / (len(correct) + 2.0)
            metadata.update(
                calibrator="constant_beta_binomial",
                rho_informative=False,
                rho_fallback="constant_beta_binomial",
                fallback_reason="nonmonotone_platt_fit",
                constant_value=float(posterior),
            )
            return _ConstantCalibrator(posterior), metadata
        metadata.update(
            calibrator="platt_logistic",
            rho_informative=True,
            rho_fallback=None,
            fallback_reason="insufficient_data_for_isotonic",
        )
        return _PlattCalibrator(model), metadata
    posterior = (n_correct + 1.0) / (len(correct) + 2.0)
    metadata.update(
        calibrator="constant_beta_binomial",
        rho_informative=False,
        rho_fallback="constant_beta_binomial",
        fallback_reason="constant_score_or_single_correctness_outcome",
        constant_value=float(posterior),
    )
    return _ConstantCalibrator(posterior), metadata


def donor_equal_weights(donor_id: NDArray[Any]) -> FloatArray:
    donors = np.asarray(donor_id)
    if donors.ndim != 1 or donors.size == 0:
        raise ProviderReliabilityError("donor_id must be a nonempty one-dimensional array")
    _, inverse, counts = np.unique(donors, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(float)
    return weights / weights.mean()


def _class_covering_group_splits(
    x: FloatArray,
    y: NDArray[np.str_],
    groups: NDArray[np.str_],
) -> list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    classes = set(y)
    for n_splits in range(min(5, len(np.unique(groups))), 1, -1):
        splits = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
        if all(set(y[train]) == classes and set(y[test]) == classes for train, test in splits):
            return splits
    raise ProviderReliabilityError("fewer than two class-covering donor folds are available")


def _fit_provider(x: FloatArray, y: pd.Series, classifier_c: float) -> LogisticRegression:
    return LogisticRegression(
        C=classifier_c,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    ).fit(x, y)


def _aligned_probabilities(
    model: LogisticRegression,
    x: FloatArray,
    classes: NDArray[np.str_],
) -> FloatArray:
    probabilities = model.predict_proba(x)
    lookup = {str(label): index for index, label in enumerate(model.classes_)}
    return np.column_stack([probabilities[:, lookup[str(label)]] for label in classes])


def _probability_vector(name: str, values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ProviderReliabilityError(f"{name} must be a finite one-dimensional array")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ProviderReliabilityError(f"{name} must lie in [0, 1]")
    return array


def _score_logit(score: FloatArray) -> FloatArray:
    safe = np.clip(_probability_vector("score", score), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(safe / (1.0 - safe)).reshape(-1, 1)


def _calibration_diagnostics(
    *, score: FloatArray, correct: NDArray[np.int_], calibrated: FloatArray, weights: FloatArray
) -> dict[str, float]:
    del score
    safe = np.clip(calibrated, 1.0e-15, 1.0 - 1.0e-15)
    return {
        "weighted_brier_score": float(brier_score_loss(correct, calibrated, sample_weight=weights)),
        "weighted_log_loss": float(log_loss(correct, safe, sample_weight=weights, labels=[0, 1])),
    }


def _reliability_table(
    *, score: FloatArray, correct: NDArray[np.int_], calibrated: FloatArray, weights: FloatArray
) -> pd.DataFrame:
    bins = np.minimum((np.asarray(score) * 10).astype(int), 9)
    rows: list[dict[str, float | int]] = []
    for bin_index in sorted(np.unique(bins)):
        mask = bins == bin_index
        bin_weights = weights[mask]
        rows.append(
            {
                "score_bin": int(bin_index),
                "n": int(mask.sum()),
                "weight_sum": float(bin_weights.sum()),
                "mean_raw_score": float(np.average(score[mask], weights=bin_weights)),
                "mean_calibrated_reliability": float(
                    np.average(calibrated[mask], weights=bin_weights)
                ),
                "observed_correct_rate": float(np.average(correct[mask], weights=bin_weights)),
            }
        )
    return pd.DataFrame(rows)


def _cell_value_hash(cell_ids: pd.Series, values: pd.Series) -> str:
    payload = "\n".join(
        f"{cell_id}\t{float(value).hex()}" for cell_id, value in zip(cell_ids, values, strict=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
