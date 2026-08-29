from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
import yaml

from coreot.results.hiha_parameter_sensitivity import (
    H1_CHECKSUMS_RELATIVE,
    H1_SPLIT_METRICS_RELATIVE,
    H1_VERIFICATION_RELATIVE,
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


def test_generated_artifacts_match_verified_sources() -> None:
    isg_path = (
        PROJECT_ROOT / "results/HIHA_DC/sensitivity/isg_tau_range_target1_alpha025_discovery/"
        "tables/discovery_by_split.csv"
    )
    _, expected = collect_hiha_parameter_sensitivity_from_h1(
        hla_split_metrics_path=PROJECT_ROOT / H1_SPLIT_METRICS_RELATIVE,
        hla_verification_path=PROJECT_ROOT / H1_VERIFICATION_RELATIVE,
        hla_checksums_path=PROJECT_ROOT / H1_CHECKSUMS_RELATIVE,
        isg_path=isg_path,
    )
    observed = pd.read_csv(
        PROJECT_ROOT / "results/HIHA_DC/figures/data/hiha_parameter_sensitivity_summary.csv"
    )
    pd.testing.assert_frame_equal(observed, expected, check_exact=False, atol=1.0e-15)

    manifest = yaml.safe_load(
        (
            PROJECT_ROOT / "results/HIHA_DC/figures/hiha_parameter_sensitivity_manifest.yaml"
        ).read_text(encoding="utf-8")
    )
    assert manifest["sources"]["hla_h1_split_metrics"]["path"] == str(H1_SPLIT_METRICS_RELATIVE)
    assert manifest["sources"]["hla_h1_verification_report"]["path"] == str(
        H1_VERIFICATION_RELATIVE
    )
    assert manifest["sources"]["hla_h1_checksums"]["path"] == str(H1_CHECKSUMS_RELATIVE)
    assert manifest["selected_settings"]["HLA-DRhi cDC2"] == [2.5, 3.0]
    assert "alt_text" not in manifest["artifacts"]
    assert not (
        PROJECT_ROOT / "docs/figs/manuscript_fig_hiha_supp_parameter_sensitivity_alt.txt"
    ).exists()
    assert manifest["rendering_metadata_policy"] == {
        "svg_hash_salt": "coreot-hiha-parameter-sensitivity",
        "svg_date_removed": True,
        "pdf_creation_and_modification_dates_removed": True,
    }
    assert manifest["interpretation"]["hla_selection_performed"] is False
    assert (
        manifest["interpretation"]["cross_endpoint_sensitivity_magnitude_comparison_supported"]
        is False
    )

    supplement = (PROJECT_ROOT / "docs/manuscript_supp.md").read_text(encoding="utf-8")
    section = supplement.split(
        "#### S2.6.3. Query-penalty sensitivity",
        maxsplit=1,
    )[1].split("### S2.7.", maxsplit=1)[0]
    assert "Stars and white outlines" not in section
    assert "larger empirical range" in section
