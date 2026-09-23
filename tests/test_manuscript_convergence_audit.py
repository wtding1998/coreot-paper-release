from pathlib import Path

import pytest
import yaml

from experiments.generate_manuscript_convergence_audit import (
    EVIDENCE_RECORD_UNAVAILABLE,
    SETTINGS_CONFIGURED_ONLY,
    SETTINGS_FIT_METHOD_PARAMS,
    _manifest_row,
)


def _write_manifest_fixture(
    project_root: Path,
    *,
    manifest_method: str = "coreot_full",
    params_method: str = "coreot_full",
    params_artifact: str | None = None,
    write_params: bool = True,
) -> Path:
    fit_directory = (
        project_root
        / "runs/example/transport/incomplete_reference/candidates/coreot_full"
    )
    fit_directory.mkdir(parents=True)
    method_params_path = fit_directory / "method_params.yaml"
    recorded_artifact = params_artifact or str(
        method_params_path.relative_to(project_root)
    )
    (fit_directory / "transport_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "artifacts": {"method_params": recorded_artifact},
                "metadata": {
                    "converged": True,
                    "method": manifest_method,
                    "n_iter": 123,
                },
            }
        ),
        encoding="utf-8",
    )
    if write_params:
        method_params_path.write_text(
            yaml.safe_dump(
                {
                    "name": params_method,
                    "max_iter": 5000,
                    "tol": 2.0e-7,
                }
            ),
            encoding="utf-8",
        )
    return fit_directory


def _fixture_manifest_row(project_root: Path) -> dict[str, object]:
    return _manifest_row(
        project_root=project_root,
        experiment="fixture",
        analysis_family="fixture",
        endpoint="fixture",
        seed=1,
        condition="incomplete_reference",
        run_id="example",
        candidate_set="candidates",
        method="coreot_full",
        max_iter=2000,
        tol=1.0e-6,
    )


def test_manifest_row_uses_observed_fit_settings_not_index_fallback(
    tmp_path: Path,
) -> None:
    _write_manifest_fixture(tmp_path)
    row = _fixture_manifest_row(tmp_path)
    assert row["max_iter"] == 5000
    assert row["tol"] == 2.0e-7
    assert row["solver_settings_evidence"] == SETTINGS_FIT_METHOD_PARAMS


def test_manifest_row_labels_fallback_only_when_fit_files_are_absent(
    tmp_path: Path,
) -> None:
    row = _fixture_manifest_row(tmp_path)
    assert row["convergence_evidence"] == EVIDENCE_RECORD_UNAVAILABLE
    assert row["max_iter"] == 2000
    assert row["tol"] == 1.0e-6
    assert row["solver_settings_evidence"] == SETTINGS_CONFIGURED_ONLY
    assert row["method_params_artifact"] == ""


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"manifest_method": "uniform_uot"}, "manifest method mismatch"),
        ({"params_method": "uniform_uot"}, "identity mismatch"),
        (
            {"params_artifact": "runs/other/method_params.yaml"},
            "same fit directory",
        ),
        ({"write_params": False}, "incomplete manifest/method-parameter pair"),
    ],
)
def test_manifest_row_rejects_inconsistent_fit_lineage(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    message: str,
) -> None:
    _write_manifest_fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(ValueError, match=message):
        _fixture_manifest_row(tmp_path)
