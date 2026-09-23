from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from coreot.results.hiha_parameter_sensitivity import (
    HLA_TAU_MAX,
    HLA_TAU_MIN,
    ISG_TAU_MAX,
    ISG_TAU_MIN,
    HIHAParameterSensitivityError,
    collect_hiha_parameter_sensitivity_from_h1,
    render_hiha_parameter_sensitivity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_checksums(
    checksums_path: Path,
    *,
    hla_path: Path,
    verification_path: Path,
) -> None:
    checksums_path.write_text(
        f"{sha256(verification_path.read_bytes()).hexdigest()}  "
        "audit/h1_verification_report.json\n"
        f"{sha256(hla_path.read_bytes()).hexdigest()}  "
        "tables/h1_split_metrics.csv\n",
        encoding="utf-8",
    )


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    hla_rows = []
    for seed in range(1, 6):
        for tau_min in HLA_TAU_MIN:
            for tau_max in HLA_TAU_MAX:
                if tau_min > tau_max:
                    continue
                hla_rows.append(
                    {
                        "endpoint": "HLA-DRhi cDC2",
                        "condition": "incomplete_reference",
                        "seed": seed,
                        "tau_min": tau_min,
                        "tau_max": tau_max,
                        "run_id": f"hla_{seed}_{tau_min}_{tau_max}",
                        "is_selected_cell": tau_min == 2.5 and tau_max == 3.0,
                        "within_cdc2_ap": 0.78 + 0.001 * seed + 0.002 * tau_min,
                        "represented_state_forced_macro_f1": 0.95 + 0.001 * seed,
                    }
                )
    isg_rows = []
    for seed in range(1, 6):
        for tau_min in ISG_TAU_MIN:
            for tau_max in ISG_TAU_MAX:
                isg_rows.append(
                    {
                        "run_id": f"isg_{seed}_{tau_min}_{tau_max}",
                        "seed": seed,
                        "tau_min": tau_min,
                        "tau_max": tau_max,
                        "tau_reference": 1.0,
                        "alpha": 0.25,
                        "coreot_auprc": 0.75 + 0.001 * seed + 0.002 * tau_min,
                        "coreot_forced_macro_f1": 0.94 + 0.001 * seed,
                    }
                )
    verification = {
        "unit_id": "phase2_h1_hiha_hladrhi",
        "execution_version": "2026-08-21-execution-v4",
        "status": "verified",
        "passed": True,
        "expected_case_count": 40,
        "observed_case_count": 40,
        "valid_case_count": 40,
        "selection_performed": False,
        "selected_cell_unchanged": True,
        "selected_cell": {"tau_max": 3.0, "tau_min": 2.5},
    }
    hla_path = tmp_path / "h1_split_metrics.csv"
    verification_path = tmp_path / "h1_verification_report.json"
    checksums_path = tmp_path / "checksums.sha256"
    isg_path = tmp_path / "isg.csv"
    pd.DataFrame(hla_rows).to_csv(hla_path, index=False)
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    _write_checksums(
        checksums_path,
        hla_path=hla_path,
        verification_path=verification_path,
    )
    pd.DataFrame(isg_rows).to_csv(isg_path, index=False)
    return hla_path, verification_path, checksums_path, isg_path


def _collect_sources(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hla_path, verification_path, checksums_path, isg_path = _write_sources(tmp_path)
    return collect_hiha_parameter_sensitivity_from_h1(
        hla_split_metrics_path=hla_path,
        hla_verification_path=verification_path,
        hla_checksums_path=checksums_path,
        isg_path=isg_path,
    )


def test_collects_complete_endpoint_specific_grids(tmp_path: Path) -> None:
    by_split, summary = _collect_sources(tmp_path)

    assert len(by_split) == 85
    assert len(summary) == 17
    assert summary["n_splits"].eq(5).all()
    assert set(summary["held_out_label"]) == {"HLA-DRhi cDC2", "ISG+ cDC2"}
    hla = summary.loc[summary["held_out_label"].eq("HLA-DRhi cDC2")]
    assert not ((hla["tau_min"] == 3.0) & (hla["tau_max"] == 2.5)).any()


def test_rejects_missing_valid_hla_cell(tmp_path: Path) -> None:
    hla_path, verification_path, checksums_path, isg_path = _write_sources(tmp_path)
    hla = pd.read_csv(hla_path)
    hla = hla.loc[~((hla["seed"] == 5) & (hla["tau_min"] == 3.0) & (hla["tau_max"] == 3.5))]
    hla.to_csv(hla_path, index=False)
    _write_checksums(
        checksums_path,
        hla_path=hla_path,
        verification_path=verification_path,
    )

    with pytest.raises(HIHAParameterSensitivityError, match="incorrect parameter-grid keys"):
        collect_hiha_parameter_sensitivity_from_h1(
            hla_split_metrics_path=hla_path,
            hla_verification_path=verification_path,
            hla_checksums_path=checksums_path,
            isg_path=isg_path,
        )


def test_rejects_h1_table_that_disagrees_with_immutable_checksum(tmp_path: Path) -> None:
    hla_path, verification_path, checksums_path, isg_path = _write_sources(tmp_path)
    hla = pd.read_csv(hla_path)
    hla.loc[0, "within_cdc2_ap"] += 0.01
    hla.to_csv(hla_path, index=False)

    with pytest.raises(HIHAParameterSensitivityError, match="immutable-root checksums"):
        collect_hiha_parameter_sensitivity_from_h1(
            hla_split_metrics_path=hla_path,
            hla_verification_path=verification_path,
            hla_checksums_path=checksums_path,
            isg_path=isg_path,
        )


def test_rejects_h1_report_that_performed_selection(tmp_path: Path) -> None:
    hla_path, verification_path, checksums_path, isg_path = _write_sources(tmp_path)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["selection_performed"] = True
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    _write_checksums(
        checksums_path,
        hla_path=hla_path,
        verification_path=verification_path,
    )

    with pytest.raises(HIHAParameterSensitivityError, match="does not authorize"):
        collect_hiha_parameter_sensitivity_from_h1(
            hla_split_metrics_path=hla_path,
            hla_verification_path=verification_path,
            hla_checksums_path=checksums_path,
            isg_path=isg_path,
        )


def test_renders_all_formats_without_selected_setting_markers(tmp_path: Path) -> None:
    _, summary = _collect_sources(tmp_path)
    outputs = {suffix: tmp_path / f"figure.{suffix}" for suffix in ("png", "pdf", "svg")}

    render_hiha_parameter_sensitivity(summary, outputs)

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    first_hashes = {
        suffix: sha256(path.read_bytes()).hexdigest()
        for suffix, path in outputs.items()
    }

    render_hiha_parameter_sensitivity(summary, outputs)

    assert {
        suffix: sha256(path.read_bytes()).hexdigest()
        for suffix, path in outputs.items()
    } == first_hashes
    assert "<dc:date>" not in outputs["svg"].read_text(encoding="utf-8")
