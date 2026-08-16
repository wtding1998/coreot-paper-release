from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from coreot.benchmarks.missing_state import run_missing_state_benchmark_build
from coreot.candidates.runner import run_candidate_cost
from coreot.data.raw_import import run_raw_import
from coreot.embeddings.runner import run_embedding
from coreot.preprocessing.provider_reliability import (
    construct_provider_reliability,
    donor_equal_weights,
    fit_scalar_reliability_calibrator,
)
from coreot.preprocessing.runner import run_model_visible_derivation
from coreot.transport.runner import run_transport


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


def test_canonical_pbmc_provider_rho_is_reused_across_paired_targets(tmp_path) -> None:
    raw_path = Path("data/raw/kang_2018.h5ad")
    if not raw_path.is_file():
        pytest.skip(f"{raw_path} is unavailable")
    base = Path("experiments/pbmc_state/configs")
    run_id = "pbmc_provider_rho_integration"
    output_root = tmp_path / "runs"
    configs: dict[str, Path] = {}
    for name in ("raw_import", "benchmark", "derivation", "embedding", "candidates", "transport"):
        payload = yaml.safe_load((base / f"{name}.yaml").read_text(encoding="utf-8"))
        payload["run_id"] = run_id
        payload["outputs"]["root"] = str(output_root)
        if name == "raw_import":
            payload["input"]["path"] = str(raw_path.resolve())
        elif name == "benchmark":
            payload["split"]["min_query_positives"] = 5
            payload["subsample"]["max_query_per_cell_type"] = 30
            payload["subsample"]["max_reference_per_cell_type"] = 50
        elif name == "embedding":
            payload["providers"][0]["n_components"] = 10
        elif name == "candidates":
            payload["candidate_graph"]["k_source_to_target"] = 5
        elif name == "transport":
            payload["candidate_sets"][0]["name"] = "pca30_k5"
            payload["methods"] = [
                method
                for method in payload["methods"]
                if method["name"] in {"prior_only", "coreot_constant_tau", "coreot_full"}
            ]
            for method in payload["methods"]:
                method["max_iter"] = 500
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        configs[name] = path

    run_raw_import(configs["raw_import"])
    run_missing_state_benchmark_build(configs["benchmark"])
    run_model_visible_derivation(configs["derivation"])
    run_embedding(configs["embedding"])
    run_candidate_cost(configs["candidates"])
    run_transport(configs["transport"])

    run_root = output_root / run_id
    rho = pd.read_csv(run_root / "transport/provider_reliability/pca30/source_rho.csv")
    assert rho["coreot_rho"].between(1.0e-6, 1.0 - 1.0e-6).all()
    transport_root = run_root / "transport"
    hashes = []
    for condition in ("incomplete_reference", "full_reference_control"):
        root = transport_root / condition / "pca30_k5"
        full = pd.read_parquet(root / "coreot_full/cell_transport_scores.parquet")
        constant = pd.read_parquet(root / "coreot_constant_tau/cell_transport_scores.parquet")
        expected_full_tau = 0.05 + (1.0 - 0.05) * full["rho"]
        assert np.allclose(full["tau_source"], expected_full_tau)
        assert np.allclose(constant["tau_source"], 1.0)
        assert full["tau_source"].nunique() > 1 or rho["coreot_rho"].nunique() == 1
        manifest = yaml.safe_load(
            (root / "coreot_full/transport_manifest.yaml").read_text(encoding="utf-8")
        )
        hashes.append(manifest["metadata"]["rho_cell_id_hash"])
    assert len(set(hashes)) == 1
