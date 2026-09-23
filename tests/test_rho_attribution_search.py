from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments import rho_attribution_search
from experiments.rho_attribution_search import (
    _pbmc_provider_rho_path,
    empirical_mass,
    mean_matched_tau,
    summarize_results,
)


def test_provider_rho_path_uses_resolved_selected_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "pbmc_ifnb_b_cells_stim_seed1_taumin0p5_taumax1_alpha4_coreot_full"

    assert _pbmc_provider_rho_path(run_root) == (
        run_root / "transport/provider_reliability/pca30/source_rho.csv"
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


def test_retained_alpha_pairs_require_complete_bundles_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    inputs = {
        "candidates": tmp_path / "candidate_edges.parquet",
        "source_priors": tmp_path / "source_priors.csv",
        "target_priors": tmp_path / "target_priors.csv",
        "truth": tmp_path / "query_truth.csv",
    }
    pd.DataFrame({"source_cell_id": ["q"], "target_cell_id": ["r"]}).to_parquet(
        inputs["candidates"], index=False
    )
    pd.DataFrame({"cell_id": ["q"], "rho": [0.5]}).to_csv(
        inputs["source_priors"], index=False
    )
    pd.DataFrame({"cell_id": ["r"]}).to_csv(inputs["target_priors"], index=False)
    pd.DataFrame({"cell_id": ["q"]}).to_csv(inputs["truth"], index=False)
    monkeypatch.setattr(
        rho_attribution_search,
        "_mouse_input_paths",
        lambda _: (run_root, inputs),
    )
    monkeypatch.setattr(
        rho_attribution_search,
        "alpha_values",
        lambda _: (1.0,),
    )
    calls: list[Path] = []

    def fake_run_method(*, variant: str, method_root: Path, **_: object):
        calls.append(method_root)
        method_root.mkdir(parents=True, exist_ok=True)
        for filename in rho_attribution_search.RETAINED_FIT_FILES:
            (method_root / filename).write_bytes(b"complete")
        return (
            pd.DataFrame(),
            {"converged": True, "n_iterations": 7},
            0.1,
        )

    def fake_evaluate_variant(**_: object) -> dict[str, object]:
        return {
            "evaluation_scope": "global_all_query",
            "n_detection": 1,
            "n_positive": 1,
            **{metric: 0.5 for metric in rho_attribution_search.METRIC_NAMES},
        }

    monkeypatch.setattr(rho_attribution_search, "_run_method", fake_run_method)
    monkeypatch.setattr(
        rho_attribution_search, "_evaluate_variant", fake_evaluate_variant
    )
    spec = rho_attribution_search.EndpointSpec(
        experiment="mouse",
        endpoint="Proliferating",
        run_prefix="",
        candidate_set="candidate",
        tau_values=(),
        selected_tau=(1.0, 2.0),
        selected_alpha=1.0,
        tau_target=8.0,
    )

    paths = rho_attribution_search._search_paths(tmp_path, "mouse")
    arguments = {
        "project_root": tmp_path,
        "runs_root": tmp_path / "runs",
        "paths": paths,
        "spec": spec,
        "seed": 0,
        "pbmc_condition": None,
        "retain_fit_artifacts": True,
    }
    rho_attribution_search._run_replicate(**arguments)
    pair_root = paths.root / "fits/proliferating/single_dataset/alpha_1"
    assert all(
        (pair_root / variant / filename).is_file()
        for variant in rho_attribution_search.VARIANTS
        for filename in rho_attribution_search.RETAINED_FIT_FILES
    )
    assert not (paths.root / "tmp").exists()

    rho_attribution_search._run_replicate(**arguments)
    assert len(calls) == 2

    (pair_root / "heterogeneous/method_params.yaml").unlink()
    rho_attribution_search._run_replicate(**arguments)
    assert len(calls) == 4


def test_alpha_pairs_keep_existing_tmp_cleanup_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    inputs = {
        "candidates": tmp_path / "candidate_edges.parquet",
        "source_priors": tmp_path / "source_priors.csv",
        "target_priors": tmp_path / "target_priors.csv",
        "truth": tmp_path / "query_truth.csv",
    }
    pd.DataFrame({"source_cell_id": ["q"], "target_cell_id": ["r"]}).to_parquet(
        inputs["candidates"], index=False
    )
    pd.DataFrame({"cell_id": ["q"], "rho": [0.5]}).to_csv(
        inputs["source_priors"], index=False
    )
    pd.DataFrame({"cell_id": ["r"]}).to_csv(inputs["target_priors"], index=False)
    pd.DataFrame({"cell_id": ["q"]}).to_csv(inputs["truth"], index=False)
    monkeypatch.setattr(
        rho_attribution_search,
        "_mouse_input_paths",
        lambda _: (run_root, inputs),
    )
    monkeypatch.setattr(
        rho_attribution_search,
        "alpha_values",
        lambda _: (1.0,),
    )

    def fake_run_method(*, method_root: Path, **_: object):
        method_root.mkdir(parents=True, exist_ok=True)
        return (
            pd.DataFrame(),
            {"converged": True, "n_iterations": 7},
            0.1,
        )

    monkeypatch.setattr(rho_attribution_search, "_run_method", fake_run_method)
    monkeypatch.setattr(
        rho_attribution_search,
        "_evaluate_variant",
        lambda **_: {
            "evaluation_scope": "global_all_query",
            "n_detection": 1,
            "n_positive": 1,
            **{metric: 0.5 for metric in rho_attribution_search.METRIC_NAMES},
        },
    )
    spec = rho_attribution_search.EndpointSpec(
        experiment="mouse",
        endpoint="Proliferating",
        run_prefix="",
        candidate_set="candidate",
        tau_values=(),
        selected_tau=(1.0, 2.0),
        selected_alpha=1.0,
        tau_target=8.0,
    )
    paths = rho_attribution_search._search_paths(tmp_path / "default", "mouse")
    rho_attribution_search._run_replicate(
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
        paths=paths,
        spec=spec,
        seed=0,
        pbmc_condition=None,
    )
    assert not (paths.root / "tmp/proliferating/single_dataset/alpha_1").exists()
    assert not (paths.root / "fits").exists()


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
