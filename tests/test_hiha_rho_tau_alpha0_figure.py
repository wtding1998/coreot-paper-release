from __future__ import annotations

from itertools import combinations_with_replacement

import pandas as pd
import pytest

from experiments.missing_celltype.generate_hiha_rho_tau_alpha0_figure import (
    EFFECT_COLUMNS,
    ENDPOINTS,
    METRICS,
    RELATIVE_EFFECT_COLUMNS,
    SOURCE_COLUMNS,
    TAU_VALUES,
    effect_column,
    focused_surface,
    relative_effect_column,
    relative_effect_sd_column,
)


def _synthetic_summary() -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        for tau_min, tau_max in combinations_with_replacement(TAU_VALUES, 2):
            delta = 0.0 if tau_min == tau_max else -0.01
            row = {
                "experiment": "hiha",
                "analysis_stage": "alpha0_focused025_125",
                "cohort": "discovery",
                "endpoint": endpoint,
                "alpha": 0.0,
                "alpha_ratio": 0.0,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "n_splits": 5,
                "all_heterogeneous_converged": True,
                "all_uniform_converged": True,
                "mean_matched_tau_mean": (tau_min + tau_max) / 2.0,
                "mean_matched_tau_sd": 0.01,
                "discovery_support": False,
            }
            for metric, _ in METRICS:
                row.update(
                    {
                        f"heterogeneous_{metric}_mean": 0.5 + delta,
                        f"heterogeneous_{metric}_sd": 0.02,
                        f"uniform_{metric}_mean": 0.5,
                        f"uniform_{metric}_sd": 0.02,
                        effect_column(metric): delta,
                        f"delta_{metric}_heterogeneous_minus_uniform_sd": 0.001,
                        f"delta_{metric}_heterogeneous_minus_uniform_n_positive": 0,
                        relative_effect_column(metric): 100.0 * delta / 0.5,
                        relative_effect_sd_column(metric): 0.1,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def test_focused_surface_requires_both_complete_triangles() -> None:
    selected = focused_surface(_synthetic_summary())
    assert len(selected) == 2 * 15
    assert selected.groupby("endpoint").size().eq(15).all()
    diagonal = selected["tau_min"].eq(selected["tau_max"])
    assert selected.loc[
        diagonal, (*EFFECT_COLUMNS, *RELATIVE_EFFECT_COLUMNS)
    ].eq(0.0).all(axis=None)


def test_focused_surface_rejects_missing_cell() -> None:
    with pytest.raises(ValueError, match="incorrect triangular coverage"):
        focused_surface(_synthetic_summary().iloc[:-1])
