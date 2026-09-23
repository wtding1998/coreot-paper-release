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
