from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.pbmc_state import generate_pbmc_compare_baseline
from experiments.pbmc_state.generate_pbmc_compare_baseline import (
    DEFAULT_SELECTED_TAU_RANGES,
    PBMC_INTERNAL_BASELINES,
    _add_internal_baselines,
    _add_match_only_method,
    _selected_coreot_run_ids,
)


def test_main_retires_u_tilde_from_pbmc_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    selected_grid = tmp_path / "selected"
    output_root = tmp_path / "comparison"

    monkeypatch.setattr(
        generate_pbmc_compare_baseline,
        "sync_selected_coreot_grid",
        lambda **_: selected_grid,
    )

    def fake_write_compare_baselines_results(**kwargs: object) -> SimpleNamespace:
        calls.update(kwargs)
        return SimpleNamespace(output_root=output_root)

    monkeypatch.setattr(
        generate_pbmc_compare_baseline,
        "write_compare_baselines_results",
        fake_write_compare_baselines_results,
    )

    assert generate_pbmc_compare_baseline.main([]) == 0
    assert calls["internal_score_overrides"] == {}


def test_add_match_only_method_changes_only_name_and_alpha(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    full = {
        "name": "coreot_full",
        "epsilon": 0.05,
        "tau_min": 0.5,
        "tau_max": 1.0,
        "tau_target": 1.0,
        "alpha": 4.0,
    }
    payloads = {
        "transport.yaml": {"methods": [full, {"name": "prior_only"}]},
        "scoring.yaml": {"methods": ["coreot_full", "prior_only"]},
        "evaluation.yaml": {"methods": ["coreot_full", "prior_only"]},
    }
    for name, payload in payloads.items():
        (run_dir / name).write_text(yaml.safe_dump(payload), encoding="utf-8")

    _add_match_only_method(run_dir)

    transport = yaml.safe_load((run_dir / "transport.yaml").read_text())
    assert transport["methods"][1] == full | {
        "name": "coreot_match_only",
        "alpha": 0.0,
    }
    for name in ("scoring.yaml", "evaluation.yaml"):
        payload = yaml.safe_load((run_dir / name).read_text())
        assert payload["methods"] == [
            "coreot_full",
            "coreot_match_only",
            "prior_only",
        ]


def test_selected_pbmc_coreot_grid_uses_label_specific_tau_ranges() -> None:
    run_ids = _selected_coreot_run_ids(tau_ranges=DEFAULT_SELECTED_TAU_RANGES)

    assert len(run_ids) == 20
    assert len(set(run_ids)) == 20
    assert any("b_cells_stim_seed1_taumin0p5_taumax1_" in run_id for run_id in run_ids)
    assert any("nk_cells_stim_seed1_taumin0p5_taumax1_" in run_id for run_id in run_ids)
    assert any(
        "dendritic_cells_stim_seed1_taumin0p75_taumax1_" in run_id
        for run_id in run_ids
    )
    assert any(
        "cd8_t_cells_stim_seed1_taumin0p5_taumax1p5_" in run_id
        for run_id in run_ids
    )


def test_add_internal_baselines_uses_prespecified_pbmc_settings(tmp_path: Path) -> None:
    uniform_uot = next(
        method for method in PBMC_INTERNAL_BASELINES if method["name"] == "uniform_uot"
    )
    assert uniform_uot["tau_source"] == 1.0
    assert uniform_uot["tau_target"] == 1.0

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payloads = {
        "transport.yaml": {
            "methods": [{"name": "coreot_full"}, {"name": "prior_only"}]
        },
        "scoring.yaml": {"methods": ["coreot_full", "prior_only"]},
        "evaluation.yaml": {"methods": ["coreot_full", "prior_only"]},
    }
    for name, payload in payloads.items():
        (run_dir / name).write_text(yaml.safe_dump(payload), encoding="utf-8")

    _add_internal_baselines(run_dir)

    expected_names = [method["name"] for method in PBMC_INTERNAL_BASELINES]
    transport = yaml.safe_load((run_dir / "transport.yaml").read_text())
    assert transport["methods"] == [{"name": "coreot_full"}, *PBMC_INTERNAL_BASELINES]
    for name in ("scoring.yaml", "evaluation.yaml"):
        payload = yaml.safe_load((run_dir / name).read_text())
        assert payload["methods"] == ["coreot_full", *expected_names]
