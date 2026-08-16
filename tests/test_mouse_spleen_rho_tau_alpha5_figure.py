from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

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


def test_manuscript_uses_focused_surface_and_preserves_selected_fit_result() -> None:
    manuscript = Path("docs/manuscript_supp.md").read_text(encoding="utf-8")
    section = manuscript.split(
        "#### S4.5.2. Mean-matched matchability-penalty attribution",
        maxsplit=1,
    )[1].split(
        "#### S4.5.3. Parameter sensitivity",
        maxsplit=1,
    )[0]
    source = pd.read_csv(
        "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "rho_attribution_tau_alpha_search/tables/"
        "rho_tau_surface_alpha5_range025_4.csv"
    )
    off_diagonal = source.loc[
        ~np.isclose(source["tau_min"], source["tau_max"])
    ]

    assert "manuscript_supp_rho_attribution_mouse_spleen_alpha5.png" in section
    assert "manuscript_fig_mouse_rho_attribution_alpha_search.png" not in section
    assert off_diagonal[
        "delta_auprc_heterogeneous_minus_uniform_mean"
    ].gt(0.0).all()
    assert set(RELATIVE_EFFECT_COLUMNS) <= set(source.columns)
    manifest = yaml.safe_load(
        Path(
            "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
            "rho_attribution_tau_alpha_search/design/"
            "rho_tau_surface_alpha5_range025_4.yaml"
        ).read_text(encoding="utf-8")
    )
    assert manifest["effect_units"] == "percent"
    assert manifest["layout"] == "2_by_2_row_major"
    assert "at the selected \\(\\alpha=40\\), it was \\(-0.0055\\)" not in (
        " ".join(section.split())
    )
