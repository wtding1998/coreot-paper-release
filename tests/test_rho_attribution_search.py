from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments import rho_attribution_search
from experiments.rho_attribution_search import (
    empirical_mass,
    mean_matched_tau,
    summarize_results,
)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (
            "heterogeneous",
            {"name": "coreot_full", "tau_min": 1.0, "tau_max": 2.0},
        ),
        (
            "mean_matched_uniform",
            {
                "name": "coreot_constant_tau",
                "tau_source": "matched_coreot_mean",
                "matched_tau_min": 1.0,
                "matched_tau_max": 2.0,
            },
        ),
    ],
)
def test_run_method_materializes_exact_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected: dict[str, object],
) -> None:
    def runner(
        method_root: Path,
        condition: str,
        candidates: pd.DataFrame,
        source_priors: pd.DataFrame,
        target_priors: pd.DataFrame,
        config: dict[str, object],
    ) -> None:
        del condition, candidates, source_priors, target_priors
        assert yaml.safe_load((method_root / "method_params.yaml").read_text()) == config
        pd.DataFrame({"cell_id": ["q"]}).to_parquet(
            method_root / "cell_transport_scores.parquet",
            index=False,
        )
        (method_root / "transport_manifest.yaml").write_text(
            yaml.safe_dump({"metadata": {"converged": True, "n_iter": 7}})
        )

    monkeypatch.setattr(rho_attribution_search, "_run_coreot_full", runner)
    monkeypatch.setattr(rho_attribution_search, "_run_coreot_constant_tau", runner)
    spec = rho_attribution_search.EndpointSpec(
        experiment="mouse",
        endpoint="Proliferating",
        run_prefix="",
        candidate_set="candidate",
        tau_values=(),
        selected_tau=(1.0, 2.0),
        selected_alpha=5.0,
        tau_target=8.0,
    )

    rho_attribution_search._run_method(
        variant=variant,
        method_root=tmp_path / variant,
        condition="natural_mismatch",
        candidates=pd.DataFrame(),
        source_priors=pd.DataFrame(),
        target_priors=pd.DataFrame(),
        spec=spec,
        alpha=5.0,
    )

    params = yaml.safe_load((tmp_path / variant / "method_params.yaml").read_text())
    assert {key: params[key] for key in expected} == expected


def test_mean_matched_tau_uses_normalized_empirical_query_mass() -> None:
    priors = pd.DataFrame(
        {
            "rho": [0.0, 0.5, 1.0],
            "empirical_mass": [1.0, 2.0, 1.0],
        }
    )
    assert mean_matched_tau(priors, 2.0, 6.0) == pytest.approx(4.0)
    assert empirical_mass(priors).sum() == pytest.approx(1.0)


def test_summary_requires_four_positive_splits_and_positive_mean_ap() -> None:
    rows = []
    for seed, delta in enumerate((0.1, 0.1, 0.1, 0.1, -0.05), start=1):
        row: dict[str, object] = {
            "experiment": "hiha",
            "endpoint": "endpoint",
            "replicate": f"seed{seed}",
            "alpha": 2.0,
            "alpha_ratio": 1.0,
            "heterogeneous_converged": True,
            "uniform_converged": True,
            "heterogeneous_n_iterations": 10,
            "tau_min": 1.0,
            "tau_max": 2.0,
            "mean_matched_tau": 1.5,
        }
        for metric in ("auroc", "auprc", "forced_accuracy", "forced_macro_f1"):
            metric_delta = delta if metric == "auprc" else 0.0
            row[f"heterogeneous_{metric}"] = 0.5 + metric_delta
            row[f"uniform_{metric}"] = 0.5
            row[f"delta_{metric}_heterogeneous_minus_uniform"] = metric_delta
        rows.append(row)
    summary = summarize_results(pd.DataFrame(rows))
    assert bool(summary.loc[0, "replicated_ap_support"])
    assert (
        summary.loc[
            0, "delta_auprc_heterogeneous_minus_uniform_n_positive"
        ]
        == 4
    )


def test_single_dataset_never_counts_as_replicated_support() -> None:
    row: dict[str, object] = {
        "experiment": "mouse",
        "endpoint": "Proliferating",
        "replicate": "single_dataset",
        "alpha": 40.0,
        "alpha_ratio": 1.0,
        "heterogeneous_converged": True,
        "uniform_converged": True,
        "heterogeneous_n_iterations": 10,
        "tau_min": 3.0,
        "tau_max": 5.0,
        "mean_matched_tau": 4.0,
    }
    for metric in ("auroc", "auprc", "forced_accuracy", "forced_macro_f1"):
        row[f"heterogeneous_{metric}"] = 0.6
        row[f"uniform_{metric}"] = 0.5
        row[f"delta_{metric}_heterogeneous_minus_uniform"] = 0.1
    summary = summarize_results(pd.DataFrame([row]))
    assert not bool(summary.loc[0, "replicated_ap_support"])
    assert bool(summary.loc[0, "exploratory_point_ap_advantage"])


def test_generated_search_has_complete_coverage_and_calibrated_pbmc_rho() -> None:
    project_root = Path(__file__).resolve().parents[1]
    expected = {
        "mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "rho_attribution_alpha_search": (1, 5, 1),
        "HIHA_DC/sensitivity/rho_attribution_alpha_search": (2, 10, 5),
        "PBMC/sensitivity/rho_attribution_alpha_search": (3, 15, 5),
    }
    for relative, (n_endpoints, n_rows, n_replicates) in expected.items():
        root = project_root / "results" / relative / "tables"
        summary = pd.read_csv(root / "rho_attribution_summary.csv")
        assert summary["endpoint"].nunique() == n_endpoints
        assert len(summary) == n_rows
        assert summary["n_replicates"].eq(n_replicates).all()
        assert summary["all_heterogeneous_converged"].all()
        assert summary["all_uniform_converged"].all()

    pbmc = pd.read_csv(
        project_root
        / "results/PBMC/sensitivity/rho_attribution_alpha_search/tables/"
        "rho_attribution_by_replicate.csv"
    )
    assert pbmc["source_priors_sha256"].astype(str).str.count(r"\+").eq(2).all()
    assert pbmc["mean_matched_tau"].lt(pbmc["tau_max"]).all()
    assert pbmc["replicate"].nunique() == 5
    manifest = yaml.safe_load(
        (
            project_root
            / "results/PBMC/sensitivity/rho_attribution_alpha_search/"
            "manifest.yaml"
        ).read_text(encoding="utf-8")
    )
    assert manifest["visualization"] == {
        "endpoint_order": ["B cells", "NK cells", "Dendritic cells"],
        "primary_metric": "absolute_AP_difference",
        "replicate_curves": "gray",
        "aggregate_curve": "orange",
        "aggregate_label": "Donor-equal mean",
        "replicated_support_marker": "blue_star",
        "overall_title": False,
    }
