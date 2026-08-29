from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from coreot.results.pbmc_supplement import (
    PBMCSupplementError,
    _read_sensitivity_fit_convergence,
)


def _write_fit_evidence(
    runs_root: Path,
    *,
    converged: bool = True,
    max_iter: int = 5000,
    n_iter: int = 721,
) -> tuple[str, Path]:
    run_id = "pbmc-sensitivity-run"
    fit_root = (
        runs_root
        / run_id
        / "transport/incomplete_reference/pca30_k100/coreot_full"
    )
    fit_root.mkdir(parents=True)
    method_params = fit_root / "method_params.yaml"
    method_params.write_text(
        yaml.safe_dump(
            {
                "name": "coreot_full",
                "tau_min": 0.5,
                "tau_max": 1.25,
                "tau_target": 1.0,
                "alpha": 4.0,
                "epsilon": 0.05,
                "max_iter": max_iter,
                "tol": 1.0e-6,
                "numerical_floor": 1.0e-300,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = fit_root / "transport_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "stage": "transport",
                "artifacts": {
                    "method_params": method_params.relative_to(runs_root.parent).as_posix()
                },
                "metadata": {
                    "method": "coreot_full",
                    "alpha": 4.0,
                    "epsilon": 0.05,
                    "converged": converged,
                    "n_iter": n_iter,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_id, fit_root


def test_sensitivity_fit_convergence_reads_the_observed_method_cap(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_id, _ = _write_fit_evidence(runs_root)

    record = _read_sensitivity_fit_convergence(
        runs_root=runs_root,
        run_id=run_id,
        candidate_set="pca30_k100",
        tau_min=0.5,
        tau_max=1.25,
        alpha=4.0,
    )

    assert record == {
        "condition": "incomplete_reference",
        "candidate_set": "pca30_k100",
        "convergence_evidence": "verified_converged",
        "converged": True,
        "n_iter": 721,
        "max_iter": 5000,
        "tol": 1.0e-6,
        "transport_manifest": (
            "runs/pbmc-sensitivity-run/transport/incomplete_reference/"
            "pca30_k100/coreot_full/transport_manifest.yaml"
        ),
        "method_params": (
            "runs/pbmc-sensitivity-run/transport/incomplete_reference/"
            "pca30_k100/coreot_full/method_params.yaml"
        ),
    }


@pytest.mark.parametrize(
    ("converged", "max_iter", "n_iter", "message"),
    [
        (False, 5000, 5000, "did not satisfy"),
        (True, 5000, 5001, "iteration evidence is invalid"),
    ],
)
def test_sensitivity_fit_convergence_rejects_terminal_or_inconsistent_records(
    tmp_path: Path,
    converged: bool,
    max_iter: int,
    n_iter: int,
    message: str,
) -> None:
    runs_root = tmp_path / "runs"
    run_id, _ = _write_fit_evidence(
        runs_root,
        converged=converged,
        max_iter=max_iter,
        n_iter=n_iter,
    )

    with pytest.raises(PBMCSupplementError, match=message):
        _read_sensitivity_fit_convergence(
            runs_root=runs_root,
            run_id=run_id,
            candidate_set="pca30_k100",
            tau_min=0.5,
            tau_max=1.25,
            alpha=4.0,
        )


def test_sensitivity_fit_convergence_requires_same_directory_method_params(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_id, fit_root = _write_fit_evidence(runs_root)
    manifest_path = fit_root / "transport_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["method_params"] = "runs/another-fit/method_params.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(PBMCSupplementError, match="outside its fit directory"):
        _read_sensitivity_fit_convergence(
            runs_root=runs_root,
            run_id=run_id,
            candidate_set="pca30_k100",
            tau_min=0.5,
            tau_max=1.25,
            alpha=4.0,
        )
