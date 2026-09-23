from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from coreot.preprocessing.provider_reliability import (
    construct_provider_reliability,
    donor_equal_weights,
    fit_scalar_reliability_calibrator,
)


def test_donor_equal_weights_give_each_donor_equal_total_weight() -> None:
    donors = np.array(["a", "a", "a", "b"])

    weights = donor_equal_weights(donors)

    assert weights.mean() == pytest.approx(1.0)
    assert weights[donors == "a"].sum() == pytest.approx(weights[donors == "b"].sum())


def test_constant_calibration_input_uses_beta_binomial_fallback() -> None:
    calibrator, metadata = fit_scalar_reliability_calibrator(
        reference_score_oof=np.full(6, 0.9),
        reference_correct_oof=np.ones(6, dtype=int),
        reference_donor_id=np.array(["a", "a", "b", "b", "c", "c"]),
    )

    expected = 7.0 / 8.0
    assert np.allclose(calibrator.predict(np.array([0.1, 0.9])), expected)
    assert metadata["calibrator"] == "constant_beta_binomial"
    assert metadata["rho_informative"] is False
    assert metadata["rho_fallback"] == "constant_beta_binomial"


def test_platt_calibration_is_monotone() -> None:
    score = np.linspace(0.1, 0.9, 12)
    correct = np.array([0, 1] * 6)
    calibrator, metadata = fit_scalar_reliability_calibrator(
        reference_score_oof=score,
        reference_correct_oof=correct,
        reference_donor_id=np.repeat(["a", "b", "c"], 4),
        min_correct_isotonic=25,
        min_error_isotonic=25,
        min_distinct_isotonic=10,
    )

    predicted = calibrator.predict(score)
    assert metadata["calibrator"] == "platt_logistic"
    assert np.all(np.diff(predicted) >= 0.0)


def test_nonmonotone_platt_fit_uses_constant_fallback() -> None:
    score = np.linspace(0.1, 0.9, 12)
    correct = np.array([1] * 6 + [0] * 6)

    calibrator, metadata = fit_scalar_reliability_calibrator(
        reference_score_oof=score,
        reference_correct_oof=correct,
        reference_donor_id=np.repeat(["a", "b", "c"], 4),
    )

    predicted = calibrator.predict(score)
    assert metadata["calibrator"] == "constant_beta_binomial"
    assert metadata["fallback_reason"] == "nonmonotone_platt_fit"
    assert np.allclose(predicted, predicted[0])


def test_provider_reliability_is_constructed_without_source_labels() -> None:
    rng = np.random.default_rng(4)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    donors: list[str] = []
    for donor in ("d1", "d2", "d3", "d4"):
        rows.extend(rng.normal(loc=(-1.0, 0.0), scale=0.8, size=(8, 2)))
        rows.extend(rng.normal(loc=(1.0, 0.0), scale=0.8, size=(8, 2)))
        labels.extend(["A"] * 8 + ["B"] * 8)
        donors.extend([donor] * 16)
    source = rng.normal(size=(10, 2))
    embedding = np.vstack([np.asarray(rows), source])
    n_reference = len(rows)
    ids = pd.Series([f"r{i}" for i in range(n_reference)] + [f"q{i}" for i in range(10)])
    is_reference = np.arange(len(ids)) < n_reference

    result = construct_provider_reliability(
        cell_ids=ids,
        embedding=embedding,
        is_reference=is_reference,
        reference_labels=pd.Series(labels),
        reference_donor_ids=pd.Series(donors),
    )

    assert result.source["cell_id"].tolist() == [f"q{i}" for i in range(10)]
    assert result.source["coreot_rho"].between(1.0e-6, 1.0 - 1.0e-6).all()
    assert len(result.metadata["cell_id_rho_sha256"]) == 64
    assert not result.reliability_table.empty
