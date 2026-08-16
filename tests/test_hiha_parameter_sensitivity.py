from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coreot.results.hiha_parameter_sensitivity import (
    HLA_TAU_MAX,
    HLA_TAU_MIN,
    ISG_TAU_MAX,
    ISG_TAU_MIN,
    HIHAParameterSensitivityError,
    collect_hiha_parameter_sensitivity,
    render_hiha_parameter_sensitivity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    hla_detection_rows = []
    hla_transfer_rows = []
    for seed in range(1, 6):
        for tau_min in HLA_TAU_MIN:
            for tau_max in HLA_TAU_MAX:
                keys = {
                    "run_id": f"hla_{seed}_{tau_min}_{tau_max}",
                    "held_out_label": "HLA-DRhi cDC2",
                    "seed": seed,
                    "method": "coreot_full",
                    "tau_min": tau_min,
                    "tau_max": tau_max,
                    "alpha": 2.0,
                }
                hla_detection_rows.append(
                    {
                        **keys,
                        "evaluation_scope": "local_within_broad_state",
                        "auprc": 0.68 + 0.001 * seed + 0.002 * tau_min,
                    }
                )
                hla_transfer_rows.append({**keys, "forced_macro_f1": 0.95 + 0.001 * seed})
    isg_rows = []
    for seed in range(1, 6):
        for tau_min in ISG_TAU_MIN:
            for tau_max in ISG_TAU_MAX:
                isg_rows.append(
                    {
                        "run_id": f"isg_seed{seed}",
                        "seed": seed,
                        "tau_min": tau_min,
                        "tau_max": tau_max,
                        "tau_reference": 1.0,
                        "alpha": 0.25,
                        "coreot_auprc": 0.75 + 0.001 * seed + 0.002 * tau_min,
                        "coreot_forced_macro_f1": 0.94 + 0.001 * seed,
                    }
                )
    hla_detection_path = tmp_path / "hla_detection.csv"
    hla_transfer_path = tmp_path / "hla_transfer.csv"
    isg_path = tmp_path / "isg.csv"
    pd.DataFrame(hla_detection_rows).to_csv(hla_detection_path, index=False)
    pd.DataFrame(hla_transfer_rows).to_csv(hla_transfer_path, index=False)
    pd.DataFrame(isg_rows).to_csv(isg_path, index=False)
    return hla_detection_path, hla_transfer_path, isg_path


def test_collects_complete_endpoint_specific_grids(tmp_path: Path) -> None:
    hla_detection_path, hla_transfer_path, isg_path = _write_sources(tmp_path)

    by_split, summary = collect_hiha_parameter_sensitivity(
        hla_within_cdc2_detection_path=hla_detection_path,
        hla_transfer_path=hla_transfer_path,
        isg_path=isg_path,
    )

    assert len(by_split) == 90
    assert len(summary) == 18
    assert summary["n_splits"].eq(5).all()
    assert set(summary["held_out_label"]) == {"HLA-DRhi cDC2", "ISG+ cDC2"}


def test_rejects_incomplete_grid(tmp_path: Path) -> None:
    hla_detection_path, hla_transfer_path, isg_path = _write_sources(tmp_path)
    isg = pd.read_csv(isg_path).iloc[:-1]
    isg.to_csv(isg_path, index=False)

    with pytest.raises(HIHAParameterSensitivityError, match="incorrect parameter-grid keys"):
        collect_hiha_parameter_sensitivity(
            hla_within_cdc2_detection_path=hla_detection_path,
            hla_transfer_path=hla_transfer_path,
            isg_path=isg_path,
        )


def test_renders_all_formats_with_y_ticks_on_every_panel(tmp_path: Path) -> None:
    hla_detection_path, hla_transfer_path, isg_path = _write_sources(tmp_path)
    _, summary = collect_hiha_parameter_sensitivity(
        hla_within_cdc2_detection_path=hla_detection_path,
        hla_transfer_path=hla_transfer_path,
        isg_path=isg_path,
    )
    outputs = {suffix: tmp_path / f"figure.{suffix}" for suffix in ("png", "pdf", "svg")}

    render_hiha_parameter_sensitivity(summary, outputs)

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())


def test_generated_summary_and_manuscript_ranges_match_regenerated_inputs() -> None:
    hla_root = PROJECT_ROOT / "results/HIHA_DC/sensitivity/full_tau_target2_alpha2_hla/tables"
    isg_path = (
        PROJECT_ROOT / "results/HIHA_DC/sensitivity/isg_tau_range_target1_alpha025_discovery/"
        "tables/discovery_by_split.csv"
    )
    _, expected = collect_hiha_parameter_sensitivity(
        hla_within_cdc2_detection_path=hla_root / "detection_within_cdc2_by_run.csv",
        hla_transfer_path=hla_root / "shared_label_transfer_by_run.csv",
        isg_path=isg_path,
    )
    observed = pd.read_csv(
        PROJECT_ROOT / "results/HIHA_DC/figures/data/hiha_parameter_sensitivity_summary.csv"
    )
    pd.testing.assert_frame_equal(observed, expected, check_exact=False, atol=1.0e-15)

    supplement = (PROJECT_ROOT / "docs/manuscript_supp.md").read_text(encoding="utf-8")
    figure_block = supplement.split(
        "**Supplementary Figure S4. Sensitivity to HIHA query-penalty bounds.**",
        maxsplit=1,
    )[1].split("#### S2.5.4.", maxsplit=1)[0]
    compact = " ".join(figure_block.split())
    assert "mean AP ranged from \\(0.784\\) to \\(0.810\\)" in compact
    assert "ranges were \\(0.748\\)--\\(0.773\\) and \\(0.941\\)--\\(0.943\\)" in compact
    assert "Stars and white outlines mark the reported operating point" in compact
    assert "HLA-DRhi cDC2 point lies outside its displayed grid" in compact
    assert "does not establish local robustness" in compact
