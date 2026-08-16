from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import rho_attribution_discovery_confirmation as attribution
from experiments.rho_attribution_discovery_confirmation import (
    EndpointDesign,
    HIHA_ALPHA0_FOCUSED_TAU_VALUES,
    HIHA_ALPHA0_FINE_TAU_VALUES,
    SurfaceConfiguration,
    _base_input_paths,
    _build_base_run,
    coarse_configurations,
    hiha_alpha0_fine_configurations,
    hiha_alpha0_focused_configurations,
    refinement_configurations,
    relative_percent,
    summarize_surface,
    tau_pairs,
)


def test_mouse_surface_retains_complete_pairs_and_resumes(
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
        inputs["candidates"],
        index=False,
    )
    pd.DataFrame({"cell_id": ["q"], "rho": [0.5]}).to_csv(
        inputs["source_priors"],
        index=False,
    )
    pd.DataFrame({"cell_id": ["r"]}).to_csv(
        inputs["target_priors"],
        index=False,
    )
    pd.DataFrame({"cell_id": ["q"]}).to_csv(inputs["truth"], index=False)
    monkeypatch.setattr(
        attribution,
        "_mouse_input_paths",
        lambda _: (run_root, inputs),
    )
    calls: list[Path] = []

    def fake_pair(*, method_root: Path, **_: object) -> dict[str, object]:
        calls.append(method_root)
        for variant in attribution.VARIANTS:
            variant_root = method_root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"cell_id": ["q"]}).to_parquet(
                variant_root / "cell_transport_scores.parquet",
                index=False,
            )
            pd.DataFrame({"mass": [1.0]}).to_parquet(
                variant_root / "sparse_coupling.parquet",
                index=False,
            )
            (variant_root / "label_probabilities.npz").write_bytes(b"test")
            (variant_root / "method_params.yaml").write_text("name: test\n")
            (variant_root / "transport_manifest.yaml").write_text(
                yaml.safe_dump({"metadata": {"converged": True, "n_iter": 7}})
            )
        row: dict[str, object] = {
            "heterogeneous_converged": True,
            "heterogeneous_n_iterations": 7,
            "heterogeneous_runtime_seconds": 0.1,
            "uniform_converged": True,
            "uniform_n_iterations": 7,
            "uniform_runtime_seconds": 0.1,
            "evaluation_scope": "global_all_query",
            "n_detection": 1,
            "n_positive": 1,
        }
        for metric in attribution.METRIC_NAMES:
            row[f"heterogeneous_{metric}"] = 0.6
            row[f"uniform_{metric}"] = 0.5
            row[f"delta_{metric}_heterogeneous_minus_uniform"] = 0.1
        row["relative_auprc_heterogeneous_minus_uniform_percent"] = 20.0
        return row

    monkeypatch.setattr(attribution, "_run_pair", fake_pair)
    design = tmp_path / "design.yaml"
    design.write_text("schema_version: 1\n")
    output = tmp_path / "output"
    configuration = SurfaceConfiguration(alpha=5.0, tau_min=1.0, tau_max=2.0)
    arguments = {
        "project_root": tmp_path,
        "design_manifest": design,
        "output_root": output,
        "analysis_stage": "coarse",
        "configurations": (configuration,),
        "jobs": 1,
        "retain_fit_artifacts": True,
    }

    attribution._run_mouse_surface(**arguments)

    pair_root = output / "fits/coarse/alpha_5_tau_min_1_tau_max_2"
    for variant in attribution.VARIANTS:
        for filename in attribution.RETAINED_FIT_FILES:
            assert (pair_root / variant / filename).is_file()

    attribution._run_mouse_surface(**arguments)
    assert calls == [pair_root]

    (pair_root / "heterogeneous/method_params.yaml").unlink()
    attribution._run_mouse_surface(**arguments)
    assert calls == [pair_root, pair_root]


def test_hiha_alpha0_fine_grid_is_locked_to_small_penalty_spreads() -> None:
    configurations = hiha_alpha0_fine_configurations()
    assert len(HIHA_ALPHA0_FINE_TAU_VALUES) == 9
    assert len(configurations) == 35
    assert {configuration.alpha for configuration in configurations} == {0.0}
    assert all(
        configuration.tau_max / configuration.tau_min <= 2.0 + 1.0e-12
        for configuration in configurations
    )


def test_hiha_alpha0_focused_grid_uses_quarter_step_triangle() -> None:
    configurations = hiha_alpha0_focused_configurations()
    assert HIHA_ALPHA0_FOCUSED_TAU_VALUES == (0.25, 0.5, 0.75, 1.0, 1.25)
    assert len(configurations) == 15
    assert {configuration.alpha for configuration in configurations} == {0.0}
    assert (
        configurations[-1]
        == SurfaceConfiguration(alpha=0.0, tau_min=1.25, tau_max=1.25)
    )


def test_focused_discovery_result_manifest_hashes_current_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    analysis_root = (
        project_root
        / "results/HIHA_DC/sensitivity/rho_attribution_discovery_confirmation"
    )
    result = yaml.safe_load(
        (
            analysis_root / "design/alpha0_focused025_125_result.yaml"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "no_discovery_supported_configuration"
    assert result["n_supported_cells"] == 0
    for artifact in ("design", "by_split", "summary", "figure"):
        path = Path(result["artifacts"][artifact])
        assert path.is_file()
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == result["artifacts"][f"{artifact}_sha256"]
        )


def test_locked_coarse_grid_sizes_and_relative_effect() -> None:
    assert len(tau_pairs((0.125, 0.25, 0.5, 1, 2, 4, 8, 16))) == 36
    assert len(tau_pairs((0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32))) == 45
    assert (
        len(
            coarse_configurations(
                (0.125, 0.25, 0.5, 1, 2, 4, 8, 16),
                (0, 0.5, 1, 2, 4, 8),
            )
        )
        == 216
    )
    assert relative_percent(0.02, 0.5) == pytest.approx(4.0)


def test_refinement_uses_locked_midpoint_rules_and_excludes_coarse_cells() -> None:
    tau_values = (0.5, 1.0, 2.0, 4.0)
    alpha_values = (0.0, 1.0, 2.0)
    coarse = coarse_configurations(tau_values, alpha_values)
    center = SurfaceConfiguration(alpha=1.0, tau_min=1.0, tau_max=2.0)
    refined = refinement_configurations(
        center=center,
        tau_values=tau_values,
        alpha_values=alpha_values,
        coarse=coarse,
    )
    assert refined
    assert set(refined).isdisjoint(coarse)
    assert {configuration.alpha for configuration in refined} <= {0.5, 1.0, 1.5}
    assert any(
        configuration.tau_min == pytest.approx(np.sqrt(0.5))
        for configuration in refined
    )
    assert any(
        configuration.tau_max == pytest.approx(np.sqrt(8.0))
        for configuration in refined
    )


def test_summary_support_requires_four_positive_splits_and_positive_mean() -> None:
    rows = []
    for seed, delta in enumerate((0.02, 0.01, 0.03, 0.01, -0.005), start=1):
        row: dict[str, object] = {
            "experiment": "hiha",
            "analysis_stage": "coarse",
            "cohort": "discovery",
            "endpoint": "ISG+ cDC2",
            "seed": seed,
            "alpha": 0.5,
            "alpha_ratio": 0.5,
            "tau_min": 0.5,
            "tau_max": 4.0,
            "mean_matched_tau": 3.0,
            "heterogeneous_converged": True,
            "uniform_converged": True,
            "heterogeneous_n_iterations": 10,
            "relative_auprc_heterogeneous_minus_uniform_percent": (
                100 * delta / 0.5
            ),
        }
        for metric in ("auroc", "auprc", "forced_accuracy", "forced_macro_f1"):
            metric_delta = delta if metric == "auprc" else 0.0
            row[f"heterogeneous_{metric}"] = 0.5 + metric_delta
            row[f"uniform_{metric}"] = 0.5
            row[f"delta_{metric}_heterogeneous_minus_uniform"] = metric_delta
        rows.append(row)
    summary = summarize_surface(pd.DataFrame(rows), expected_splits=5)
    assert bool(summary.loc[0, "discovery_support"])
    assert (
        summary.loc[
            0, "delta_auprc_heterogeneous_minus_uniform_n_positive"
        ]
        == 4
    )
    for metric in ("auroc", "auprc", "forced_accuracy", "forced_macro_f1"):
        assert (
            f"relative_{metric}_heterogeneous_minus_uniform_percent_mean"
            in summary
        )


def test_synthetic_base_builder_produces_complete_paired_inputs(
    tmp_path: Path,
) -> None:
    endpoint = EndpointDesign(
        endpoint="HLA-DRhi cDC2",
        slug="hladrhi_cdc2",
        selected_alpha=2.0,
        tau_target=2.0,
    )
    rows = []
    for donor_index in range(54):
        donor = f"BR1{donor_index:03d}"
        for cell_index in range(6):
            held_out = cell_index < 3
            rows.append(
                {
                    "cell_id": f"{donor}_cell{cell_index}",
                    "cell_type": (
                        "HLA-DRhi cDC2" if held_out else "CD14+ cDC2"
                    ),
                    "broad_label": "cDC2",
                    "sample_id": donor,
                    "donor_id": donor,
                    "batch_id": "batch",
                    "rho": 0.2 if held_out else 0.8,
                    "prior_risk": 0.8 if held_out else 0.2,
                }
            )
    obs = pd.DataFrame(rows).set_index("cell_id", drop=False)
    embedding = np.random.default_rng(7).normal(size=(len(obs), 30))
    analysis_root = tmp_path / "results" / "analysis"
    manifest = _build_base_run(
        obs=obs,
        full_embedding=embedding,
        source_sha256="source",
        cohort_manifest_sha256="design",
        cohort="discovery",
        cohort_donors=set(obs["donor_id"]),
        endpoint=endpoint,
        seed=101,
        analysis_root=analysis_root,
    )
    assert manifest.is_file()
    run_root = manifest.parent
    paths = _base_input_paths(run_root)
    assert all(path.is_file() for path in paths.values())
    candidates = pd.read_parquet(paths["candidates"])
    source = pd.read_csv(paths["source_priors"])
    target = pd.read_csv(paths["target_priors"])
    truth = pd.read_csv(paths["truth"])
    assert set(candidates["source_cell_id"]) == set(source["cell_id"])
    assert set(candidates["target_cell_id"]) == set(target["cell_id"])
    assert set(truth["cell_id"]) == set(source["cell_id"])
    assert truth["is_absent_state"].sum() >= 30
    assert target["target_label_visible"].ne(endpoint.endpoint).all()
