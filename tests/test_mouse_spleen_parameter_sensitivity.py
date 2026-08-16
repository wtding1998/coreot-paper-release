from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coreot.results.mouse_spleen_parameter_sensitivity import (
    DISPLAY_TAU_MAX,
    DISPLAY_TAU_MIN,
    MouseSpleenParameterSensitivityError,
    TAU_VALUES,
    collect_mouse_spleen_parameter_sensitivity,
    render_mouse_spleen_parameter_sensitivity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source_frame() -> pd.DataFrame:
    rows = []
    for tau_min in TAU_VALUES:
        for tau_max in TAU_VALUES:
            if tau_min > tau_max:
                continue
            rows.append(
                {
                    "run_id": f"tau_{tau_min}_{tau_max}",
                    "natural_endpoint": "Proliferating",
                    "condition": "natural_mismatch",
                    "method": "coreot_full",
                    "tau_min": tau_min,
                    "tau_max": tau_max,
                    "tau_target": 8.0,
                    "alpha": 40.0,
                    "converged": True,
                    "evaluation_scope": "global_all_query",
                    "n_query": 4333,
                    "n_positive": 62,
                    "auprc": 0.4 + 0.001 * tau_min,
                    "shared_forced_macro_f1": 0.65 + 0.001 * tau_max,
                    "is_canonical": tau_min == 3.0 and tau_max == 5.0,
                    "source_run_id": "main" if tau_min == 3.0 and tau_max == 5.0 else "grid",
                }
            )
    return pd.DataFrame(rows)


def test_collects_complete_converged_triangular_grid(tmp_path: Path) -> None:
    path = tmp_path / "grid.csv"
    _source_frame().to_csv(path, index=False)

    result = collect_mouse_spleen_parameter_sensitivity(path)

    assert len(result) == 28
    assert result["converged"].all()
    assert result["is_canonical"].sum() == 1
    assert {"ap", "represented_state_forced_macro_f1"} <= set(result.columns)


def test_rejects_incomplete_grid(tmp_path: Path) -> None:
    path = tmp_path / "grid.csv"
    _source_frame().iloc[:-1].to_csv(path, index=False)

    with pytest.raises(MouseSpleenParameterSensitivityError, match="exact triangular grid"):
        collect_mouse_spleen_parameter_sensitivity(path)


def test_renders_all_formats_with_y_ticks_on_both_panels(tmp_path: Path) -> None:
    path = tmp_path / "grid.csv"
    _source_frame().to_csv(path, index=False)
    frame = collect_mouse_spleen_parameter_sensitivity(path)
    outputs = {suffix: tmp_path / f"figure.{suffix}" for suffix in ("png", "pdf", "svg")}

    render_mouse_spleen_parameter_sensitivity(frame, outputs)

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    assert DISPLAY_TAU_MIN == (2.0, 3.0, 4.0)
    assert DISPLAY_TAU_MAX == (4.0, 5.0, 6.0)


def test_generated_source_matches_regenerated_grid() -> None:
    grid_path = (
        PROJECT_ROOT
        / "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "figure4_full_tau_alpha_40_tau_target_8/tables/metrics_by_grid.csv"
    )
    expected = collect_mouse_spleen_parameter_sensitivity(grid_path)
    observed = pd.read_csv(
        PROJECT_ROOT
        / "results/mouse_spleen_core_ot/manuscript/parameter_sensitivity/"
        "mouse_spleen_parameter_sensitivity.csv"
    )
    pd.testing.assert_frame_equal(observed, expected, check_exact=False, atol=1.0e-15)

    supplement = (PROJECT_ROOT / "docs/manuscript_supp.md").read_text(
        encoding="utf-8"
    )
    section = supplement.split("#### S4.5.3. Parameter sensitivity", maxsplit=1)[
        1
    ].split("#### S4.5.4.", maxsplit=1)[0]
    compact = " ".join(section.split())
    assert "figs/manuscript_fig_mouse_spleen_supp_parameter_sensitivity.png" in section
    assert "AP ranged from \\(0.440\\) to \\(0.458\\)" in compact
    assert "forced macro-F1 ranged from \\(0.657\\) to \\(0.664\\)" in compact
    assert "\\tau_{\\min}\\in\\{2,3,4\\}" in compact
    assert "\\tau_{\\max}\\in\\{4,5,6\\}" in compact
    assert "rather than independent robustness validation" in compact
