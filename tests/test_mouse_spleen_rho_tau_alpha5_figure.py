from __future__ import annotations

from itertools import combinations_with_replacement

import pandas as pd
import pytest

from experiments.mouse_spleen.generate_rho_tau_alpha5_figure import (
    ABSOLUTE_EFFECT_COLUMNS,
    METRICS,
    RELATIVE_EFFECT_COLUMNS,
    SOURCE_COLUMNS,
    STORED_RELATIVE_AP_COLUMN,
    TAU_VALUES,
    absolute_effect_column,
    focused_surface,
    relative_effect_column,
)


def _synthetic_summary() -> pd.DataFrame:
    rows = []
    for tau_min, tau_max in combinations_with_replacement(TAU_VALUES, 2):
        delta = 0.0 if tau_min == tau_max else 0.01
        row = {
            "analysis_stage": "coarse",
            "endpoint": "Proliferating",
            "n_splits": 1,
            "alpha": 5.0,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "mean_matched_tau_mean": (tau_min + tau_max) / 2.0,
            "all_heterogeneous_converged": True,
            "all_uniform_converged": True,
        }
        for metric, _ in METRICS:
            uniform = 0.2
            row[f"heterogeneous_{metric}_mean"] = uniform + delta
            row[f"uniform_{metric}_mean"] = uniform
            row[absolute_effect_column(metric)] = delta
        row[STORED_RELATIVE_AP_COLUMN] = 100.0 * delta / 0.2
        rows.append(row)
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def test_focused_surface_requires_complete_triangle_and_exact_diagonal() -> None:
    selected = focused_surface(_synthetic_summary())
    assert len(selected) == 15
    assert selected[relative_effect_column("auprc")].max() == pytest.approx(5.0)
    diagonal = selected["tau_min"].eq(selected["tau_max"])
    for column in (*ABSOLUTE_EFFECT_COLUMNS, *RELATIVE_EFFECT_COLUMNS):
        assert selected.loc[diagonal, column].eq(0.0).all()


def test_focused_surface_rejects_missing_cell() -> None:
    with pytest.raises(ValueError, match="incorrect triangular coverage"):
        focused_surface(_synthetic_summary().iloc[:-1])
