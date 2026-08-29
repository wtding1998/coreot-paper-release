from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import component_ablation_surfaces as component
from experiments.component_ablation_surfaces import (
    RETAINED_FIT_FILES,
    _display_metric,
    _retained_aggregate_rows,
    build_parser,
    alpha_values,
    evaluate_scores,
    selected_compatibility_tau_values,
    selected_specs,
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


def test_retained_only_compatibility_grid_excludes_archived_hiha_tau() -> None:
    hiha = selected_specs("hiha", ("HLA-DRhi cDC2",))[0]
    pbmc = selected_specs("pbmc", ("B cells",))[0]

    assert selected_compatibility_tau_values(hiha, retained_only=True) == (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    )
    assert len(selected_compatibility_tau_values(hiha, retained_only=False)) == 6
    assert selected_compatibility_tau_values(
        pbmc, retained_only=True
    ) == selected_compatibility_tau_values(pbmc, retained_only=False)


def test_pbmc_specs_resolve_selected_primary_run_ids() -> None:
    observed = {
        spec.endpoint: spec.run_prefix
        for spec in selected_specs("pbmc", None)
    }

    assert observed == {
        "B cells": "pbmc_ifnb_b_cells_stim_seed{seed}_taumin0p5_taumax1_alpha4_coreot_full",
        "NK cells": "pbmc_ifnb_nk_cells_stim_seed{seed}_taumin0p5_taumax1_alpha2_coreot_full",
        "Dendritic cells": (
            "pbmc_ifnb_dendritic_cells_stim_seed{seed}_"
            "taumin0p75_taumax1_alpha3_coreot_full"
        ),
    }


def test_component_ablation_cli_selects_one_retained_variant() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--experiment",
            "hiha",
            "--variant",
            "compatibility_only",
            "--retained-only",
            "--retain-fit-artifacts",
            "--jobs",
            "1",
        ]
    )

    assert args.variant == ["compatibility_only"]
    assert args.retained_only is True
    assert args.retain_fit_artifacts is True
    assert args.jobs == 1


def test_component_ablation_retains_complete_fit_and_reruns_incomplete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = component.EndpointSpec(
        experiment="hiha",
        endpoint="HLA-DRhi cDC2",
        run_prefix="run_seed{seed}",
        candidate_set="candidate",
        tau_values=(4.0,),
        selected_tau=(4.0, 4.0),
        selected_alpha=2.0,
        tau_target=2.0,
    )
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
    calls: list[Path] = []

    def fake_runner(method_root: Path, *_args: object) -> None:
        calls.append(method_root)
        _scores().to_parquet(method_root / "cell_transport_scores.parquet", index=False)
        pd.DataFrame({"mass": [1.0]}).to_parquet(
            method_root / "sparse_coupling.parquet", index=False
        )
        (method_root / "label_probabilities.npz").write_bytes(b"test")
        (method_root / "transport_manifest.yaml").write_text(
            yaml.safe_dump({"metadata": {"converged": True, "n_iter": 7}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(component, "_run_coreot_constant_tau", fake_runner)
    monkeypatch.setattr(
        component,
        "selected_compatibility_tau_values",
        lambda _spec, *, retained_only: (4.0,),
    )
    monkeypatch.setattr(component, "alpha_values", lambda _selected: (0.0,))
    arguments = {
        "spec": spec,
        "seed": 1,
        "run_root": tmp_path / "run",
        "output_root": tmp_path / "output",
        "variant": "compatibility_only",
        "candidates": pd.DataFrame(),
        "source_priors": pd.DataFrame(),
        "target_priors": pd.DataFrame(),
        "truth": truth,
        "pbmc_condition": None,
        "candidate_hash": "candidate-hash",
        "source_hash": "source-hash",
        "retained_only": True,
        "retain_fit_artifacts": True,
    }

    component._run_variant(**arguments)

    fit_root = (
        tmp_path
        / "output/fits/hladrhi_cdc2/seed1/compatibility_only/"
        "tau_source_4_alpha_0"
    )
    assert all((fit_root / filename).is_file() for filename in RETAINED_FIT_FILES)
    assert yaml.safe_load((fit_root / "method_params.yaml").read_text()) == {
        "alpha": 0.0,
        "epsilon": 0.05,
        "eta": 1.0e-12,
        "max_iter": 5000,
        "name": "coreot_constant_tau",
        "numerical_floor": 1.0e-300,
        "tau_source": 4.0,
        "tau_target": 2.0,
        "tol": 1.0e-6,
    }

    component._run_variant(**arguments)
    assert calls == [fit_root]

    (fit_root / "method_params.yaml").unlink()
    component._run_variant(**arguments)
    assert calls == [fit_root, fit_root]
    assert all((fit_root / filename).is_file() for filename in RETAINED_FIT_FILES)


def test_component_ablation_cli_dispatches_retention_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_surfaces(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(component, "run_surfaces", fake_run_surfaces)
    arguments = [
        "run",
        "--experiment",
        "hiha",
        "--endpoint",
        "HLA-DRhi cDC2",
        "--seed",
        "1",
        "--variant",
        "compatibility_only",
        "--retain-fit-artifacts",
        "--output-root",
        str(tmp_path / "output"),
    ]

    assert component.main(arguments) == 0
    assert len(calls) == 1
    assert calls[0]["variants"] == ("compatibility_only",)
    assert calls[0]["retain_fit_artifacts"] is True
    assert calls[0]["retained_only"] is False


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
