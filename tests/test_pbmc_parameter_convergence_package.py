from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from coreot.results.pbmc_supplement import ENDPOINTS, SEEDS
from experiments.submission.build_pbmc_parameter_and_calibration_unit import (
    validate_sensitivity_lineage,
)
from submission.reproduction.pbmc import (
    PBMCReleaseRegenerationError,
    _validate_parameter_convergence,
)


def _number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _write_retained_sensitivity_evidence(repository_root: Path) -> pd.DataFrame:
    records = []
    alpha_by_endpoint = {
        "B cells": 4.0,
        "NK cells": 2.0,
        "Dendritic cells": 3.0,
    }
    for endpoint in ENDPOINTS:
        for seed in SEEDS:
            for tau_min in (0.5, 0.75, 1.0):
                for tau_max in (1.0, 1.25, 1.5):
                    alpha = alpha_by_endpoint[endpoint]
                    run_id = (
                        f"{endpoint.lower().replace(' ', '_')}-seed{seed}-"
                        f"taumin{_number_token(tau_min)}-"
                        f"taumax{_number_token(tau_max)}"
                    )
                    relative_root = Path(
                        f"runs/{run_id}/transport/incomplete_reference/"
                        "pca30_k100/coreot_full"
                    )
                    fit_root = repository_root / relative_root
                    fit_root.mkdir(parents=True)
                    params_path = fit_root / "method_params.yaml"
                    params_path.write_text(
                        yaml.safe_dump(
                            {
                                "name": "coreot_full",
                                "tau_min": tau_min,
                                "tau_max": tau_max,
                                "tau_target": 1.0,
                                "alpha": alpha,
                                "epsilon": 0.05,
                                "max_iter": 5000,
                                "tol": 1.0e-6,
                                "numerical_floor": 1.0e-300,
                            },
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    manifest_path = fit_root / "transport_manifest.yaml"
                    manifest_path.write_text(
                        yaml.safe_dump(
                            {
                                "stage": "transport",
                                "artifacts": {
                                    "method_params": (
                                        relative_root / "method_params.yaml"
                                    ).as_posix()
                                },
                                "metadata": {
                                    "method": "coreot_full",
                                    "alpha": alpha,
                                    "epsilon": 0.05,
                                    "converged": True,
                                    "n_iter": 721,
                                },
                            },
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    records.append(
                        {
                            "held_out_label": endpoint,
                            "seed": seed,
                            "run_id": run_id,
                            "tau_min": tau_min,
                            "tau_max": tau_max,
                            "alpha": alpha,
                            "method": "coreot_full",
                            "condition": "incomplete_reference",
                            "candidate_set": "pca30_k100",
                            "convergence_evidence": "verified_converged",
                            "converged": True,
                            "n_iter": 721,
                            "max_iter": 5000,
                            "tol": 1.0e-6,
                            "transport_manifest": (
                                relative_root / "transport_manifest.yaml"
                            ).as_posix(),
                            "method_params": (
                                relative_root / "method_params.yaml"
                            ).as_posix(),
                        }
                    )
    frame = pd.DataFrame.from_records(records)
    table = (
        repository_root
        / "results/PBMC/manuscript/supplement/pbmc_sensitivity_by_seed.csv"
    )
    table.parent.mkdir(parents=True)
    frame.to_csv(table, index=False)
    return frame


def test_parameter_package_lineage_requires_all_135_verified_fits(
    tmp_path: Path,
) -> None:
    expected = _write_retained_sensitivity_evidence(tmp_path)

    observed, evidence_paths = validate_sensitivity_lineage(tmp_path)

    pd.testing.assert_frame_equal(observed, expected)
    assert len(evidence_paths) == 2 * len(ENDPOINTS) * len(SEEDS) * 9
    assert all(path.is_file() for path in evidence_paths)


def test_parameter_package_lineage_rejects_table_to_fit_cap_drift(
    tmp_path: Path,
) -> None:
    frame = _write_retained_sensitivity_evidence(tmp_path)
    frame.loc[0, "max_iter"] = 2000
    frame.to_csv(
        tmp_path
        / "results/PBMC/manuscript/supplement/pbmc_sensitivity_by_seed.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="max_iter disagrees"):
        validate_sensitivity_lineage(tmp_path)


def test_clean_room_parameter_adapter_rejects_missing_packaged_fit_evidence(
    tmp_path: Path,
) -> None:
    unit_root = tmp_path / "unit"
    unit_root.mkdir()
    frame = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "condition": ["incomplete_reference"],
            "candidate_set": ["pca30_k100"],
            "method": ["coreot_full"],
            "convergence_evidence": ["verified_converged"],
            "converged": [True],
            "n_iter": [721],
            "max_iter": [5000],
            "tol": [1.0e-6],
            "transport_manifest": [
                "runs/run-1/transport/incomplete_reference/pca30_k100/"
                "coreot_full/transport_manifest.yaml"
            ],
            "method_params": [
                "runs/run-1/transport/incomplete_reference/pca30_k100/"
                "coreot_full/method_params.yaml"
            ],
        }
    )

    with pytest.raises(PBMCReleaseRegenerationError, match="incomplete for run-1"):
        _validate_parameter_convergence(unit_root, frame)
