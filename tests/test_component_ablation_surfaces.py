from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.component_ablation_surfaces import (
    _display_metric,
    _retained_aggregate_rows,
    alpha_values,
    evaluate_scores,
    tau_pairs,
)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["p1", "p2", "n1", "n2", "s1"],
            "u": [0.9, 0.8, 0.2, 0.1, 0.05],
            "forced_label": ["other", "other", "B cells", "B cells", "NK cells"],
        }
    )


def test_component_ablation_grids_have_agreed_sizes() -> None:
    assert len(tau_pairs((2, 3, 4, 5, 6))) == 15
    assert (3.0, 5.0) in tau_pairs((2, 3, 4, 5, 6))
    assert alpha_values(4.0) == (0.0, 2.0, 4.0, 6.0, 8.0)


def test_mouse_spleen_component_figures_use_two_by_two_layout() -> None:
    manifest = yaml.safe_load(
        Path(
            "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
            "component_ablation_proliferating/manifest.yaml"
        ).read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["heatmap_layout"] == "2_by_2_row_major"
    assert manifest["metadata"]["heatmap_metrics"] == [
        "auprc",
        "auroc",
        "forced_accuracy",
        "forced_macro_f1",
    ]


def test_pbmc_detection_is_within_celltype_and_transfer_uses_shared_cells() -> None:
    truth = pd.DataFrame(
        {
            "cell_id": ["p1", "p2", "n1", "n2", "s1"],
            "true_label": ["B cells", "B cells", "B cells", "B cells", "NK cells"],
            "is_absent_state": [True, True, False, False, False],
            "is_shared_state": [False, False, True, True, True],
        }
    )
    condition = pd.Series(
        {"p1": "stim", "p2": "stim", "n1": "ctrl", "n2": "ctrl", "s1": "ctrl"}
    )
    metrics = evaluate_scores(
        experiment="pbmc",
        endpoint="B cells",
        truth=truth,
        scores=_scores(),
        pbmc_condition=condition,
    )
    assert metrics["evaluation_scope"] == "within_celltype"
    assert metrics["n_detection"] == 4
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["forced_accuracy"] == 1.0
    assert metrics["forced_macro_f1"] == 1.0


def test_hiha_detection_is_restricted_to_cdc2() -> None:
    truth = pd.DataFrame(
        {
            "cell_id": ["p1", "p2", "n1", "n2", "s1"],
            "true_label": [
                "HLA-DRhi cDC2",
                "HLA-DRhi cDC2",
                "CD14+ cDC2",
                "ISG+ cDC2",
                "pDC",
            ],
            "is_absent_state": [True, True, False, False, False],
            "is_shared_state": [False, False, True, True, True],
        }
    )
    metrics = evaluate_scores(
        experiment="hiha",
        endpoint="HLA-DRhi cDC2",
        truth=truth,
        scores=_scores(),
    )
    assert metrics["evaluation_scope"] == "within_cdc2"
    assert metrics["n_detection"] == 4
    assert np.isclose(metrics["auroc"], 1.0)


def test_generated_pbmc_surfaces_use_calibrated_provider_rho() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "results/PBMC/sensitivity/component_ablation/tables/"
        "component_ablation_by_seed.csv"
    )
    frame = pd.read_csv(path)
    assert len(frame) == 600
    assert frame["source_priors_sha256"].astype(str).str.count(r"\+").eq(2).all()
    assert frame["converged"].all()


def test_hiha_retained_aggregate_excludes_only_archived_compatibility_tau() -> None:
    frame = pd.DataFrame(
        {
            "variant": [
                "compatibility_only",
                "compatibility_only",
                "compatibility_only",
                "match_only",
            ],
            "tau_source": [1.0, 2.5, 5.0, np.nan],
        }
    )

    retained = _retained_aggregate_rows(frame, experiment="hiha")

    assert retained.index.tolist() == [0, 2, 3]
    assert _retained_aggregate_rows(frame, experiment="pbmc").equals(frame)


def test_display_metric_limits_ignore_hidden_rows_and_variants() -> None:
    summary = pd.DataFrame(
        {
            "endpoint": ["HLA", "HLA", "HLA", "HLA"],
            "variant": [
                "compatibility_only",
                "compatibility_only",
                "compatibility_only",
                "match_only",
            ],
            "tau_source": [1.0, 5.0, 2.5, np.nan],
            "alpha": [0.0, 0.0, 0.0, np.nan],
            "auprc_mean": [0.72, 0.81, 0.10, 0.05],
        }
    )

    pivot, limits = _display_metric(
        summary=summary,
        endpoint="HLA",
        variant="compatibility_only",
        row_name="tau_source",
        column_name="alpha",
        row_values=[1.0, 5.0],
        column_values=[0.0],
        metric="auprc",
    )

    assert pivot.to_numpy().tolist() == [[0.72], [0.81]]
    assert limits == (0.72, 0.81)
