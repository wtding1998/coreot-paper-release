from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.mouse_spleen.generate_prior_dependence_figure import (
    PARAMETERS_RELATIVE,
    PRIORS_RELATIVE,
    SCORES_RELATIVE,
    TRANSPORT_MANIFEST_RELATIVE,
    TRUTH_RELATIVE,
    correlation_summary,
    equal_frequency_bins,
    generate,
)


def test_equal_frequency_bins_preserve_prior_risk_ties() -> None:
    prior_risk = np.concatenate(
        [np.full(36, 0.45), np.linspace(0.46, 0.55, 4)]
    )
    frame = pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(40)],
            "u": np.linspace(0.0, 0.9, 40),
            "prior_risk": prior_risk,
        }
    )

    bins = equal_frequency_bins(frame)

    assert bins["n_cells"].sum() == len(frame)
    assert len(bins) < 20
    tied_bin = bins.loc[np.isclose(bins["mean_prior_risk"], 0.45)].iloc[0]
    assert tied_bin["n_cells"] == 36


def test_correlation_summary_uses_average_ranks() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "u": [0.1, 0.2, 0.3, 0.4],
            "prior_risk": [0.45, 0.45, 0.50, 0.55],
        }
    )

    summary = correlation_summary(frame).iloc[0]

    expected = frame["u"].rank(method="average").corr(
        frame["prior_risk"].rank(method="average")
    )
    assert summary["spearman"] == pytest.approx(expected)
    assert summary["n_cells"] == 4


def test_generate_writes_source_tables_artwork_and_manifest(
    tmp_path: Path,
) -> None:
    cell_ids = [f"cell_{index}" for index in range(40)]
    prior_risk = np.concatenate(
        [np.full(36, 0.45), np.linspace(0.46, 0.55, 4)]
    )
    scores = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "condition_id": "natural_mismatch",
            "method": "coreot_full",
            "u": np.linspace(0.0, 0.9, 40),
            "prior_risk": prior_risk,
        }
    )
    priors = pd.DataFrame(
        {"cell_id": cell_ids, "prior_risk": prior_risk}
    )
    truth = pd.DataFrame({"cell_id": cell_ids})
    parameters = {
        "tau_min": 3.0,
        "tau_max": 5.0,
        "tau_target": 8.0,
        "alpha": 40.0,
        "epsilon": 0.05,
    }
    for relative in (
        SCORES_RELATIVE,
        PRIORS_RELATIVE,
        TRUTH_RELATIVE,
        PARAMETERS_RELATIVE,
        TRANSPORT_MANIFEST_RELATIVE,
    ):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(tmp_path / SCORES_RELATIVE, index=False)
    priors.to_csv(tmp_path / PRIORS_RELATIVE, index=False)
    truth.to_csv(tmp_path / TRUTH_RELATIVE, index=False)
    (tmp_path / PARAMETERS_RELATIVE).write_text(
        yaml.safe_dump(parameters),
        encoding="utf-8",
    )
    (tmp_path / TRANSPORT_MANIFEST_RELATIVE).write_text(
        yaml.safe_dump({"metadata": {"converged": True}}),
        encoding="utf-8",
    )

    outputs = generate(tmp_path)

    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    manifest = yaml.safe_load(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["cohort"] == "all_query_cells"
    assert manifest["truth_stratification"] is False
    assert manifest["binning"]["n_occupied_intervals"] < 20
    svg = outputs["figure_svg"].read_text(encoding="utf-8").lower()
    for color in ("#d55e00", "#b1aaa4", "#6f655e", "#ece6e1"):
        assert color in svg
    assert "Orange points show means" in outputs["caption"].read_text(
        encoding="utf-8"
    )


def test_manuscript_prior_section_matches_regenerated_sources() -> None:
    manuscript = Path("docs/manuscript_supp.md").read_text(encoding="utf-8")
    bins = pd.read_csv(
        "results/mouse_spleen_core_ot/manuscript/prior_dependence/"
        "prior_bins.csv"
    )
    correlation = pd.read_csv(
        "results/mouse_spleen_core_ot/manuscript/prior_dependence/"
        "prior_correlation.csv"
    ).iloc[0]

    assert (
        "#### S4.5.4. Association of query-marginal deficit with prior risk"
        in manuscript
    )
    assert "figs/manuscript_fig_mouse_spleen_prior_dependence.png" in manuscript
    assert f"{correlation.spearman:.3f}" in manuscript
    for row in bins.itertuples(index=False):
        assert f"{row.mean_prior_risk:.3f}" in manuscript
        assert f"{row.mean_u:.3f}" in manuscript
        assert f"{row.n_cells:,}" in manuscript
