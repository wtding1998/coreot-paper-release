from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from coreot.artifacts.manifests import read_manifest
from experiments.mouse_spleen.generate_natural_reports import _write_sensitivity_report
from experiments.mouse_spleen.pipeline import (
    MouseSpleenConfigError,
    _fit_source_priors,
    _label_transfer_metrics,
    _transport_method_configs,
    run_stage,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rna_path = tmp_path / "rna.h5ad"
    atac_genes_path = tmp_path / "atac-genes.h5ad"
    atac_peaks_path = tmp_path / "atac-peaks.h5ad"
    rna_labels = [
        "B follicular",
        "B transitional",
        "Marginal zone B",
        "DC",
        "Macrophage",
        "Granulocyte",
        "NK",
        "T CD4",
        "T CD4 reg",
        "T CD8",
        "Unknown",
        "Proliferating",
    ]
    atac_labels = [
        "B follicular",
        "B transitional",
        "Marginal zone B",
        "DC",
        "Macrophage",
        "Granulocyte",
        "NK CD27+",
        "NK CD27-",
        "T CD4 naive",
        "T CD4 reg",
        "T CD8 Memory",
        "T CD8 naive",
    ]
    genes = ["g1", "g2", "g3", "g4"]
    ad.AnnData(
        X=np.ones((len(rna_labels), len(genes))),
        obs=pd.DataFrame(
            {"cell_type": rna_labels, "source": "RNA"},
            index=[f"r{i}" for i in range(len(rna_labels))],
        ),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(rna_path)
    ad.AnnData(
        X=np.ones((len(atac_labels), len(genes))),
        obs=pd.DataFrame(
            {"cell_type": atac_labels, "source": "ATAC"},
            index=[f"a{i}" for i in range(len(atac_labels))],
        ),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(atac_genes_path)
    ad.AnnData(
        X=np.ones((len(atac_labels), 5)),
        obs=pd.DataFrame(
            {"cell_type": atac_labels, "source": "ATAC"},
            index=[f"a{i}" for i in range(len(atac_labels))],
        ),
        var=pd.DataFrame(index=[f"p{i}" for i in range(5)]),
    ).write_h5ad(atac_peaks_path)
    return rna_path, atac_genes_path, atac_peaks_path


def _write_config(tmp_path: Path, *, aliases: dict[str, list[str]] | None = None) -> Path:
    rna, atac_genes, atac_peaks = _write_inputs(tmp_path)
    config = {
        "experiment": {"name": "mouse_spleen_test", "output_dir": str(tmp_path / "results")},
        "data": {
            "rna_h5ad": str(rna),
            "atac_gene_activity_h5ad": str(atac_genes),
            "atac_peaks_h5ad": str(atac_peaks),
            "allow_shape_mismatch": True,
            "label_key": "cell_type",
            "source_key": "source",
        },
        "labels": {
            "allow_case_insensitive_exact_match": True,
            "allow_fuzzy_match": False,
            "aliases": aliases
            or {
                "Follicular B": ["B follicular"],
                "Transitional B": ["B transitional"],
                "Marginal zone B": ["Marginal zone B"],
                "DC": ["DC"],
                "Macrophage": ["Macrophage"],
                "Granulocyte": ["Granulocyte"],
                "NK": ["NK", "NK CD27+", "NK CD27-"],
                "CD4 T": ["T CD4", "T CD4 naive"],
                "Treg": ["T CD4 reg"],
                "CD8 T": ["T CD8", "T CD8 Memory", "T CD8 naive"],
                "Ifit B": ["Unknown"],
                "Proliferating": ["Proliferating"],
            },
            "shared_labels": [
                "Follicular B",
                "Transitional B",
                "Marginal zone B",
                "DC",
                "Macrophage",
                "Granulocyte",
                "NK",
                "CD4 T",
                "Treg",
                "CD8 T",
            ],
            "rna_only_labels": ["Ifit B", "Proliferating"],
            "broad_map": {
                "B": ["Follicular B", "Transitional B", "Marginal zone B", "Ifit B"],
                "T_NK_lymphoid": ["CD4 T", "Treg", "CD8 T", "NK"],
                "Myeloid": ["DC", "Macrophage", "Granulocyte"],
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_validate_data_resolves_explicit_many_to_one_labels(tmp_path: Path) -> None:
    result = run_stage(_write_config(tmp_path), "validate_data")

    assert result.stage == "validate_data"
    composition = pd.read_csv(result.artifacts["cell_type_composition"])
    nk = composition.loc[composition["fine_label"] == "NK"].iloc[0]
    assert nk["rna_count"] == 1
    assert nk["atac_count"] == 2

    resolved = yaml.safe_load(Path(result.artifacts["resolved_label_map"]).read_text())
    assert resolved["raw_to_canonical"]["atac"]["NK CD27+"] == "NK"
    assert resolved["raw_to_canonical"]["rna"]["Unknown"] == "Ifit B"

    manifest = read_manifest(result.manifest_path)
    assert manifest.stage == "validate_data"
    assert manifest.metadata["shapes"]["rna"] == [12, 4]
    assert manifest.metadata["shapes"]["atac_gene_activity"] == [12, 4]


def test_validate_data_rejects_alias_used_by_two_canonical_labels(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["labels"]["aliases"]["DC"].append("Macrophage")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(MouseSpleenConfigError, match="maps to multiple canonical labels"):
        run_stage(config_path, "validate_data")


def test_prepare_embedding_caches_aligned_precomputed_provider(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    provider_path = tmp_path / "provider.npz"
    np.savez_compressed(
        provider_path,
        rna_cell_ids=np.array([f"r{i}" for i in range(12)]),
        atac_cell_ids=np.array([f"a{i}" for i in range(12)]),
        rna_embedding=np.arange(36, dtype=float).reshape(12, 3),
        atac_embedding=np.arange(36, 72, dtype=float).reshape(12, 3),
    )
    config["embedding"] = {
        "mode": "precomputed",
        "precomputed_path": str(provider_path),
        "freeze_for_primary_paired_runs": True,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_stage(config_path, "prepare_embedding")

    coordinates = np.load(result.artifacts["embedding"])
    cells = pd.read_csv(result.artifacts["embedding_cells"])
    assert coordinates.shape == (24, 3)
    assert cells["cell_id"].tolist() == [
        *(f"rna::r{i}" for i in range(12)),
        *(f"atac::a{i}" for i in range(12)),
    ]
    assert cells["modality"].tolist() == ["rna"] * 12 + ["atac"] * 12
    manifest = read_manifest(result.manifest_path)
    assert manifest.metadata["mode"] == "precomputed"
    assert manifest.metadata["frozen"] is True


def test_prepare_embedding_reuses_combined_cached_provider(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)
    original = run_stage(config_path, "prepare_embedding")
    config = yaml.safe_load(config_path.read_text())
    config["experiment"]["output_dir"] = str(tmp_path / "reused_results")
    config["embedding"] = {
        "mode": "precomputed_combined",
        "precomputed_embedding_path": str(original.artifacts["embedding"]),
        "precomputed_cells_path": str(original.artifacts["embedding_cells"]),
        "freeze_for_primary_paired_runs": True,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    reused = run_stage(config_path, "prepare_embedding")

    np.testing.assert_array_equal(
        np.load(reused.artifacts["embedding"]),
        np.load(original.artifacts["embedding"]),
    )
    manifest = read_manifest(reused.manifest_path)
    assert manifest.metadata["mode"] == "precomputed_combined"


def _configure_precomputed_embedding(config_path: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    provider_path = tmp_path / "provider.npz"
    np.savez_compressed(
        provider_path,
        rna_cell_ids=np.array([f"r{i}" for i in range(12)]),
        atac_cell_ids=np.array([f"a{i}" for i in range(12)]),
        rna_embedding=np.array([[i, i % 3, (i * 2) % 5] for i in range(12)], dtype=float),
        atac_embedding=np.array(
            [[i + 0.1, i % 3 + 0.1, (i * 2) % 5 + 0.1] for i in range(12)],
            dtype=float,
        ),
    )
    config["embedding"] = {
        "mode": "precomputed",
        "precomputed_path": str(provider_path),
        "freeze_for_primary_paired_runs": True,
    }
    config["priors"] = {
        "broad_classifier": {"C": 1.0, "max_iter": 5000, "random_state": 7},
        "matchability": {
            "query_neighbor_k": 3,
            "confidence_weight": 0.5,
            "neighbor_weight": 0.5,
            "clip": [0.0, 1.0],
        },
    }
    config["experiments"] = {
        "controlled": {
            "primary_holdouts": ["Marginal zone B"],
            "run_all_shared_holdouts": False,
            "main_min_query_positive_cells": 1,
            "supplementary_min_query_positive_cells": 1,
        }
    }
    config["candidate_graph"] = {
        "k_source_to_target": 100,
        "add_reverse_edges": True,
        "distance": "euclidean",
    }
    config["cost_scaling"] = {"scale": "median", "clip_quantile": 0.99, "delta": 1e-8}
    config["transport"] = {
        "alpha": 0.25,
        "epsilon": 0.05,
        "tau_min": 0.05,
        "tau_max": 1.0,
        "tau_source_constant": 1.0,
        "tau_target": 1.0,
        "eta": 1e-12,
        "tolerance": 1e-6,
        "max_iterations": 2000,
    }
    config["methods"] = {"transport": ["coreot_full"]}
    config["scoring"] = {
        "primary_coreot_score": "u",
        "prior_adjustment": {"enabled": False},
        "thresholds": {
            "theta_u": {
                "source_condition": "full_reference_control",
                "source_method": "method",
                "quantile": 0.95,
            },
            "theta_H": {"value": 0.8},
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_prepare_task_family_freezes_query_prior_across_paired_conditions(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)

    result = run_stage(config_path, "prepare_task_families")

    task_manifest = read_manifest(result.artifacts["marginal_zone_b_manifest"])
    assert task_manifest.metadata["holdout_label"] == "Marginal zone B"
    assert task_manifest.metadata["classifier_training_fine_labels"].get("Marginal zone B", 0) == 0
    run_root = Path(task_manifest.artifacts["run_root"])
    full_source = pd.read_csv(
        run_root / "derived/full_reference_control/prior_profiles/default/source_priors.csv"
    )
    incomplete_source = pd.read_csv(
        run_root / "derived/incomplete_reference/prior_profiles/default/source_priors.csv"
    )
    pd.testing.assert_frame_equal(full_source, incomplete_source)
    assert full_source["rho"].between(0.0, 1.0).all()

    incomplete_targets = pd.read_csv(
        run_root / "benchmark/incomplete_reference/model_visible/target_labels.csv"
    )
    full_targets = pd.read_csv(
        run_root / "benchmark/full_reference_control/model_visible/target_labels.csv"
    )
    assert "Marginal zone B" not in set(incomplete_targets["target_label"])
    assert "Marginal zone B" in set(full_targets["target_label"])


def test_natural_prior_decouples_anchor_and_matchability_classifiers() -> None:
    reference = pd.DataFrame(
        {
            "cell_id": [f"reference::{index}" for index in range(6)],
            "broad_label": ["B", "B", "B", "Myeloid", "Myeloid", "Myeloid"],
        }
    )
    query = pd.DataFrame({"cell_id": [f"query::{index}" for index in range(4)]})
    coordinates = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, -0.1],
            [1.0, 1.0],
            [0.9, 1.1],
            [1.1, 0.9],
            [0.25, 0.15],
            [0.45, 0.40],
            [0.65, 0.60],
            [0.85, 0.80],
        ]
    )
    embedding_by_id = dict(
        zip([*reference["cell_id"], *query["cell_id"]], coordinates, strict=True)
    )
    global_priors = {
        "broad_classifier": {"C": 1.0, "max_iter": 5000, "random_state": 7},
        "matchability": {
            "query_neighbor_k": 3,
            "confidence_weight": 0.5,
            "neighbor_weight": 0.5,
            "clip": [0.0, 1.0],
        },
    }
    config = {"priors": global_priors}
    baseline = _fit_source_priors(config, query, reference, embedding_by_id)
    configured = _fit_source_priors(
        config,
        query,
        reference,
        embedding_by_id,
        priors_config={
            "anchor_classifier": global_priors["broad_classifier"],
            "matchability_classifier": {
                "C": 10.0,
                "max_iter": 5000,
                "random_state": 20260713,
            },
            "matchability": {
                "query_neighbor_k": 10,
                "confidence_weight": 0.5,
                "neighbor_weight": 0.5,
                "clip": [0.45, 0.55],
            },
        },
    )

    anchor_columns = [column for column in configured if column.startswith("anchor_probability::")]
    pd.testing.assert_frame_equal(configured[anchor_columns], baseline[anchor_columns])
    assert configured["rho"].between(0.45, 0.55).all()
    assert set(configured["anchor_classifier_C"]) == {1.0}
    assert set(configured["matchability_classifier_C"]) == {10.0}
    assert set(configured["rho_recipe"]) == {
        "rho=clip(0.5*matchability_classifier_entropy_confidence+"
        "0.5*query_neighbor_agreement,0.45,0.55)"
    }


def test_canonical_natural_prior_config_uses_conservative_dynamic_range() -> None:
    config = yaml.safe_load(
        Path("experiments/mouse_spleen/configs/mouse_spleen_core_ot.yaml").read_text(
            encoding="utf-8"
        )
    )
    natural_priors = config["experiments"]["natural_mismatch"]["priors"]
    assert natural_priors == {
        "anchor_classifier": {
            "C": 1.0,
            "max_iter": 5000,
            "random_state": 20260713,
        },
        "matchability_classifier": {
            "C": 10.0,
            "max_iter": 5000,
            "random_state": 20260713,
        },
        "matchability": {
            "query_neighbor_k": 10,
            "confidence_weight": 0.5,
            "neighbor_weight": 0.5,
            "clip": [0.45, 0.55],
        },
    }
    assert config["priors"]["broad_classifier"]["C"] == 1.0
    assert config["priors"]["matchability"]["query_neighbor_k"] == 15
    assert config["experiments"]["natural_mismatch"]["coreot_full"] == {
        "tau_min": 3,
        "tau_max": 5,
        "tau_target": 8,
        "alpha": 40,
        "max_iterations": 5000,
    }
    assert config["experiments"]["natural_mismatch"]["mean_matched_primary_factorial"] is True
    assert config["experiments"]["natural_mismatch"]["endpoints"]["Proliferating"] == {
        "uniform_uot_tau_source": 4.0
    }
    assert config["experiments"]["natural_mismatch"]["methods"] == [
        "nn",
        "uniform_uot",
        "coreot_constant_tau",
        "coreot_match_only",
        "coreot_full",
        "prior_only",
    ]


def test_proliferating_uniform_tau_override_preserves_mean_matched_ablation() -> None:
    config = yaml.safe_load(
        Path("experiments/mouse_spleen/configs/mouse_spleen_core_ot.yaml").read_text(
            encoding="utf-8"
        )
    )
    natural = config["experiments"]["natural_mismatch"]
    methods = _transport_method_configs(
        config,
        natural["methods"],
        coreot_full_override=natural["coreot_full"],
        mean_matched_factorial=natural["mean_matched_primary_factorial"],
        uniform_uot_tau_source=natural["endpoints"]["Proliferating"][
            "uniform_uot_tau_source"
        ],
    )
    uniform = next(item for item in methods if item["name"] == "uniform_uot")
    compatibility = next(
        item for item in methods if item["name"] == "coreot_constant_tau"
    )

    assert uniform["tau_source"] == 4.0
    assert "matched_tau_min" not in uniform
    assert "matched_tau_max" not in uniform
    assert compatibility["tau_source"] == "matched_coreot_mean"
    assert compatibility["matched_tau_min"] == 3.0
    assert compatibility["matched_tau_max"] == 5.0


def test_prepare_task_family_writes_bounded_deterministic_smoke_sample(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["experiments"]["smoke"] = {
        "enabled": True,
        "holdout_label": "Marginal zone B",
        "query_total": 10,
        "query_positive": 1,
        "reference_total": 10,
        "reference_holdout": 1,
        "seed": 17,
    }
    config["experiments"]["support_dose"] = {
        "fractions": [0.0, 1.0],
        "n_seeds_for_partial_fractions": 0,
    }
    config["experiments"]["random_deletion"] = {"n_seeds": 0, "controls": []}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    first = run_stage(config_path, "prepare_task_families")
    second = run_stage(config_path, "prepare_task_families")

    first_manifest = read_manifest(first.artifacts["marginal_zone_b_manifest"])
    second_manifest = read_manifest(second.artifacts["marginal_zone_b_manifest"])
    assert first_manifest.metadata["smoke"] is True
    assert first_manifest.metadata["n_query"] == 10
    assert first_manifest.metadata["n_query_positive"] == 1
    assert first_manifest.metadata["n_reference_full"] == 10
    assert first_manifest.metadata["n_reference_incomplete"] == 9
    assert (
        first_manifest.metadata["smoke_query_cell_id_sha256"]
        == second_manifest.metadata["smoke_query_cell_id_sha256"]
    )
    run_root = Path(first_manifest.artifacts["run_root"])
    truth = pd.read_csv(
        run_root / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
    )
    same_broad_negatives = truth.loc[
        truth["true_broad_label"].eq("B") & ~truth["true_label"].eq("Marginal zone B")
    ]
    assert not same_broad_negatives.empty


def test_prepare_task_family_writes_deterministic_dose_and_deletion_conditions(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["experiments"]["support_dose"] = {
        "fractions": [0.0, 0.5, 1.0],
        "n_seeds_for_partial_fractions": 2,
    }
    config["experiments"]["random_deletion"] = {
        "n_seeds": 2,
        "controls": ["global_stratified", "same_broad_stratified"],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_stage(config_path, "prepare_task_families")

    task_manifest = read_manifest(result.artifacts["marginal_zone_b_manifest"])
    conditions = task_manifest.metadata["conditions"]
    assert {item["condition_id"] for item in conditions} == {
        "full_reference_control",
        "incomplete_reference",
        "support_0p5_seed_0",
        "support_0p5_seed_1",
        "global_stratified_seed_0",
        "global_stratified_seed_1",
        "same_broad_stratified_seed_0",
        "same_broad_stratified_seed_1",
    }
    assert all(item["retained_reference_sha256"] for item in conditions)
    run_root = Path(task_manifest.artifacts["run_root"])
    baseline = pd.read_csv(
        run_root / "derived/incomplete_reference/prior_profiles/default/source_priors.csv"
    )
    for item in conditions:
        current = pd.read_csv(
            run_root / "derived" / item["condition_id"] / "prior_profiles/default/source_priors.csv"
        )
        pd.testing.assert_frame_equal(baseline, current)


def test_run_controlled_uses_native_transport_scoring_and_evaluation(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)

    result = run_stage(config_path, "run_controlled")

    manifest = read_manifest(result.artifacts["marginal_zone_b_manifest"])
    run_root = Path(manifest.artifacts["run_root"])
    candidate_set = "mouse_spleen_provider_k100"
    for condition in ("full_reference_control", "incomplete_reference"):
        method_root = run_root / "transport" / condition / candidate_set / "coreot_full"
        assert (method_root / "sparse_coupling.parquet").is_file()
        scores = pd.read_parquet(method_root / "cell_transport_scores.parquet")
        assert scores["u"].between(0.0, 1.0).all()
        assert (run_root / "scoring" / condition / candidate_set / "cell_scores.parquet").is_file()
        prior_only_root = run_root / "transport" / condition / candidate_set / "prior_only"
        assert (prior_only_root / "cell_transport_scores.parquet").is_file()
        assert not (prior_only_root / "sparse_coupling.parquet").exists()
    metrics = pd.read_csv(run_root / "evaluation/metrics.csv")
    incomplete = metrics.loc[metrics["condition_id"] == "incomplete_reference"]
    assert {"auroc", "auprc", "median_absent", "median_shared"} <= set(incomplete["metric"])


def test_aggregate_metrics_pairs_deficits_by_stable_query_id(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)
    run_stage(config_path, "run_controlled")

    result = run_stage(config_path, "aggregate_metrics")

    paired = pd.read_parquet(result.artifacts["paired_cells"])
    assert paired["query_cell_id"].is_unique
    assert np.allclose(paired["delta_u"], paired["u_incomplete"] - paired["u_full"])
    assert np.allclose(paired["one_minus_rho_incomplete"], paired["one_minus_rho_full"])
    assert np.allclose(paired["one_minus_rho"], paired["one_minus_rho_full"])
    detection = pd.read_csv(result.artifacts["controlled_detection"])
    assert set(detection["evaluation_scope"]) == {"within_broad", "global_all_query"}
    assert set(detection["score"]) == {"u"}
    prior = pd.read_csv(result.artifacts["prior_only_detection"])
    assert set(prior["evaluation_scope"]) == {"within_broad", "global_all_query"}
    assert set(prior["method"]) == {"prior_only"}
    assert set(prior["score"]) == {"one_minus_rho"}
    restoration = pd.read_csv(result.artifacts["restoration"])
    assert {"median_delta_u_positive", "median_delta_u_shared"} <= set(restoration.columns)


def test_aggregate_metrics_summarizes_support_dose_and_deletion_controls(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["experiments"]["support_dose"] = {
        "fractions": [0.0, 0.5, 1.0],
        "n_seeds_for_partial_fractions": 2,
    }
    config["experiments"]["random_deletion"] = {
        "n_seeds": 1,
        "controls": ["global_stratified", "same_broad_stratified"],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_stage(config_path, "run_controlled")

    result = run_stage(config_path, "aggregate_metrics")

    dose = pd.read_csv(result.artifacts["support_dose"])
    assert set(dose["support_fraction"]) == {0.0, 0.5, 1.0}
    assert {"mean", "sd", "median", "min", "max", "n_seeds"} <= set(dose.columns)
    deletion = pd.read_csv(result.artifacts["random_deletion"])
    assert set(deletion["deletion_control_type"]) == {
        "global_stratified",
        "same_broad_stratified",
    }


def test_run_natural_mismatch_reverses_mapping_and_marks_restoration_undefined(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["experiments"]["natural_mismatch"] = {
        "enabled": True,
        "priors": {
            "anchor_classifier": {
                "C": 1.0,
                "max_iter": 5000,
                "random_state": 7,
            },
            "matchability_classifier": {
                "C": 10.0,
                "max_iter": 5000,
                "random_state": 20260713,
            },
            "matchability": {
                "query_neighbor_k": 10,
                "confidence_weight": 0.5,
                "neighbor_weight": 0.5,
                "clip": [0.45, 0.55],
            },
        },
        "endpoints": {
            "Ifit B": {},
            "Proliferating": {"uniform_uot_tau_source": 4.0},
        },
        "methods": [
            "uniform_uot",
            "coreot_constant_tau",
            "coreot_match_only",
            "coreot_full",
            "prior_only",
        ],
        "mean_matched_primary_factorial": True,
        "coreot_full": {
            "tau_min": 3.0,
            "tau_max": 5.0,
            "tau_target": 8.0,
            "alpha": 40.0,
            "max_iterations": 5000,
        },
        "sensitivity": {
            "alpha_values": [0, 1],
            "tau_values": [1, 2],
            "max_iterations_by_endpoint": {"Proliferating": 3500},
        },
        "match_only_sensitivity": {
            "tau_values": [0.05, 1.0],
            "tau_target": 1.0,
            "max_iterations": 3500,
        },
        "full_tau_range_sensitivity": {
            "tau_values": [0.05, 1.0],
            "alpha": 20.0,
            "tau_target": 8.0,
            "max_iterations": 5000,
        },
        "component_ablation": {
            "endpoint": "Proliferating",
            "match_tau_min_values": [3.0, 4.0],
            "match_tau_max_values": [4.0, 5.0],
            "compatibility_tau_values": [4.0, 5.0],
            "alpha_values": [40.0],
            "tau_target": 8.0,
            "max_iterations": 5000,
        },
    }
    config["methods"]["transport"] = [
        "coreot_constant_tau",
        "coreot_full",
    ]
    config["baselines"] = {"internal": ["prior_only"], "external": []}
    config["scoring"]["prior_adjustment"]["enabled"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_stage(config_path, "run_natural_mismatch")

    endpoint_manifest = read_manifest(result.artifacts["ifit_b"])
    assert endpoint_manifest.metadata["restoration_metric_applicability"] == "undefined"
    run_root = Path(endpoint_manifest.artifacts["run_root"])
    source_priors = pd.read_csv(
        run_root / "derived/natural_mismatch/prior_profiles/default/source_priors.csv"
    )
    assert source_priors["rho"].between(0.45, 0.55).all()
    assert set(source_priors["anchor_classifier_C"]) == {1.0}
    assert set(source_priors["matchability_classifier_C"]) == {10.0}
    transport_config = yaml.safe_load(
        (run_root / "generated_configs/transport.yaml").read_text(encoding="utf-8")
    )
    full_config = next(
        method for method in transport_config["methods"] if method["name"] == "coreot_full"
    )
    assert full_config["tau_min"] == 3.0
    assert full_config["tau_max"] == 5.0
    assert full_config["tau_target"] == 8.0
    assert full_config["alpha"] == 40.0
    assert full_config["max_iter"] == 5000
    compatibility_config = next(
        method for method in transport_config["methods"] if method["name"] == "coreot_constant_tau"
    )
    uniform_config = next(
        method for method in transport_config["methods"] if method["name"] == "uniform_uot"
    )
    match_config = next(
        method for method in transport_config["methods"] if method["name"] == "coreot_match_only"
    )
    for method_config in (compatibility_config, uniform_config):
        assert method_config["tau_source"] == "matched_coreot_mean"
        assert method_config["matched_tau_min"] == 3.0
        assert method_config["matched_tau_max"] == 5.0
        assert method_config["tau_target"] == 8.0
        assert method_config["max_iter"] == 5000
    assert compatibility_config["alpha"] == 40.0
    assert match_config["tau_min"] == 3.0
    assert match_config["tau_max"] == 5.0
    assert match_config["tau_target"] == 8.0
    assert match_config["alpha"] == 0.0
    assert match_config["max_iter"] == 5000
    proliferating_manifest = read_manifest(result.artifacts["proliferating"])
    proliferating_run_root = Path(proliferating_manifest.artifacts["run_root"])
    proliferating_transport_config = yaml.safe_load(
        (
            proliferating_run_root / "generated_configs/transport.yaml"
        ).read_text(encoding="utf-8")
    )
    proliferating_uniform_config = next(
        method
        for method in proliferating_transport_config["methods"]
        if method["name"] == "uniform_uot"
    )
    assert proliferating_uniform_config["tau_source"] == 4.0
    assert "matched_tau_min" not in proliferating_uniform_config
    assert "matched_tau_max" not in proliferating_uniform_config
    proliferating_compatibility_config = next(
        method
        for method in proliferating_transport_config["methods"]
        if method["name"] == "coreot_constant_tau"
    )
    assert proliferating_compatibility_config["tau_source"] == "matched_coreot_mean"
    cells = pd.read_csv(run_root / "benchmark/natural_mismatch/model_visible/cells.csv")
    assert cells.loc[cells["domain"] == "query", "cell_id"].str.startswith("rna::").all()
    assert cells.loc[cells["domain"] == "reference", "cell_id"].str.startswith("atac::").all()
    scores = pd.read_parquet(
        run_root / "scoring/natural_mismatch/mouse_spleen_provider_k100/cell_scores.parquet"
    )
    assert {
        "coreot_full",
        "coreot_constant_tau",
        "coreot_match_only",
        "uniform_uot",
    } <= set(scores["method"])
    aggregate = run_stage(config_path, "aggregate_natural_mismatch")
    metrics = pd.read_csv(aggregate.artifacts["metrics"])
    required_scores = {
        "u",
        "nearest_reference_distance",
        "mean_k_reference_distance",
        "one_minus_max_broad_probability",
        "broad_probability_entropy",
        "one_minus_rho",
        "one_minus_max_conditional_transport_label_probability",
        "conditional_transport_label_entropy",
    }
    assert required_scores <= set(metrics["score"])
    ifit = metrics.loc[metrics["natural_endpoint"].eq("Ifit B")]
    assert set(ifit["evaluation_scope"]) == {"within_broad", "global_all_query"}
    proliferating = metrics.loc[metrics["natural_endpoint"].eq("Proliferating")]
    assert set(proliferating["evaluation_scope"]) == {"global_all_query"}
    broad = pd.read_csv(aggregate.artifacts["broad_predictions"])
    assert set(broad["natural_endpoint"]) == {"Ifit B", "Proliferating"}
    assert np.allclose(broad.groupby("natural_endpoint")["predicted_fraction"].sum(), 1.0)
    baseline_comparison = run_stage(config_path, "aggregate_natural_baselines")
    baseline_detection = pd.read_csv(baseline_comparison.artifacts["detection"])
    assert set(baseline_detection["method"]) == {
        "coreot_constant_tau",
        "coreot_match_only",
        "coreot_full",
        "prior_only",
        "uniform_uot",
    }
    baseline_manifest = read_manifest(baseline_comparison.manifest_path)
    assert baseline_manifest.metadata["coreot_full_operating_point"] == {
        "tau_min": 3.0,
        "tau_max": 5.0,
        "tau_target": 8.0,
        "alpha": 40.0,
        "max_iterations": 5000,
    }
    assert baseline_manifest.metadata["coreot_full_convergence_required"] is True
    assert baseline_manifest.metadata["required_converged_internal_methods"] == [
        "coreot_constant_tau",
        "coreot_full",
        "coreot_match_only",
        "uniform_uot",
    ]
    assert baseline_manifest.metadata["internal_transport_convergence_required"] is True
    assert baseline_manifest.metadata["uniform_uot_tau_source_by_endpoint"] == {
        "Ifit B": "matched_coreot_mean",
        "Proliferating": 4.0,
    }
    report_path = Path(baseline_manifest.artifacts["report"])
    report = report_path.read_text(encoding="utf-8")
    expected_header = (
        "| Method | AUROC | AUPRC | Median endpoint | Median shared | "
        "Forced accuracy | Forced macro-F1 | Coverage | "
        "Shared false-abstention rate |"
    )
    assert report.count(expected_header) == 2
    assert "## Ifit B" in report
    assert "## Proliferating" in report
    assert "| Family |" not in report
    assert "| Score |" not in report
    assert "AUPRC baseline" not in report
    assert "prior_only |" in report
    assert "| NA | NA | NA | NA |" in report
    assert "tau_min=3" in report
    assert "replicate-based uncertainty" in report
    full_manifest_path = (
        run_root
        / "transport/natural_mismatch/mouse_spleen_provider_k100/coreot_full"
        / "transport_manifest.yaml"
    )
    full_manifest = yaml.safe_load(full_manifest_path.read_text(encoding="utf-8"))
    full_manifest["metadata"]["converged"] = False
    full_manifest_path.write_text(yaml.safe_dump(full_manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(MouseSpleenConfigError, match="did not converge"):
        run_stage(config_path, "aggregate_natural_baselines")
    figure = run_stage(config_path, "make_natural_mismatch_figure")
    assert figure.artifacts["natural_mismatch"].is_file()

    sensitivity = run_stage(config_path, "aggregate_natural_constant_tau_sensitivity")
    sensitivity_rows = pd.read_csv(sensitivity.artifacts["detection_by_run"])
    assert len(sensitivity_rows) == 8
    assert set(sensitivity_rows["alpha"]) == {0.0, 1.0}
    assert set(sensitivity_rows["tau"]) == {1.0, 2.0}
    assert sensitivity_rows["converged"].all()
    assert (
        sensitivity_rows[["u_tilde_auroc", "u_tilde_auprc", "u_tilde_median_absent"]]
        .notna()
        .all()
        .all()
    )
    assert set(sensitivity_rows.groupby("natural_endpoint")["max_iterations"].first().items()) == {
        ("Ifit B", 2000),
        ("Proliferating", 3500),
    }
    assert sensitivity.artifacts["heatmaps"].is_file()

    match_only = run_stage(config_path, "aggregate_natural_match_only_sensitivity")
    match_rows = pd.read_csv(match_only.artifacts["metrics_by_grid"])
    assert len(match_rows) == 6
    assert (match_rows["method"] == "coreot_match_only").all()
    assert (match_rows["alpha"] == 0.0).all()
    assert (match_rows["tau_min"] <= match_rows["tau_max"]).all()
    assert match_rows["converged"].all()
    assert match_rows[["u_tilde_auroc", "u_tilde_auprc"]].notna().all().all()
    assert match_only.artifacts["ifit_b_report"].is_file()
    assert match_only.artifacts["proliferating_report"].is_file()

    original_constant = sensitivity_rows.copy()
    target8_constant = run_stage(config_path, "aggregate_natural_constant_tau_target8_sensitivity")
    target8_constant_rows = pd.read_csv(target8_constant.artifacts["detection_by_run"])
    assert len(target8_constant_rows) == 8
    assert (target8_constant_rows["tau_target"] == 8.0).all()
    assert (target8_constant_rows["max_iterations"] == 5000).all()
    pd.testing.assert_frame_equal(
        pd.read_csv(sensitivity.artifacts["detection_by_run"]),
        original_constant,
    )

    original_match = match_rows.copy()
    target8_match = run_stage(config_path, "aggregate_natural_match_only_target8_sensitivity")
    target8_match_rows = pd.read_csv(target8_match.artifacts["metrics_by_grid"])
    assert len(target8_match_rows) == 6
    assert (target8_match_rows["tau_target"] == 8.0).all()
    assert (target8_match_rows["max_iterations"] == 5000).all()
    pd.testing.assert_frame_equal(
        pd.read_csv(match_only.artifacts["metrics_by_grid"]),
        original_match,
    )
    constant_report_path = (
        Path(config["experiment"]["output_dir"])
        / "natural_mismatch/reports/ifit_b/constant_tau_alpha_metrics.md"
    )
    _write_sensitivity_report(
        endpoint="Ifit B",
        report_path=constant_report_path,
        sensitivity_path=sensitivity.artifacts["detection_by_run"],
        target8_path=target8_constant.artifacts["detection_by_run"],
    )
    constant_report = constant_report_path.read_text(encoding="utf-8")
    assert "$u$-based detection metrics — tau target = 8" in constant_report
    match_report = match_only.artifacts["ifit_b_report"].read_text(encoding="utf-8")
    assert "$u$-based detection metrics — tau target = 8" in match_report

    default_prior_path = (
        run_root / "derived/natural_mismatch/prior_profiles/default/source_priors.csv"
    )
    default_prior_bytes = default_prior_path.read_bytes()
    full_tau_range = run_stage(config_path, "aggregate_natural_full_tau_range_sensitivity")
    full_tau_range_rows = pd.read_csv(full_tau_range.artifacts["metrics_by_grid"])
    assert len(full_tau_range_rows) == 6
    assert set(full_tau_range_rows["method"]) == {"coreot_full"}
    assert set(full_tau_range_rows["alpha"]) == {20.0}
    assert set(full_tau_range_rows["tau_target"]) == {8.0}
    assert set(full_tau_range_rows["max_iterations"]) == {5000}
    assert full_tau_range_rows["anchor_prior_reuse_verified"].all()
    assert full_tau_range_rows["source_priors_sha256"].nunique() == 2
    assert default_prior_path.read_bytes() == default_prior_bytes
    full_tau_range_manifest = read_manifest(full_tau_range.manifest_path)
    assert full_tau_range_manifest.metadata["n_completed"] == 6
    assert full_tau_range_manifest.metadata["n_expected"] == 6
    assert full_tau_range_manifest.metadata["anchor_prior_reuse_verified"] is True

    component = run_stage(config_path, "aggregate_natural_component_ablation")
    match_component = pd.read_csv(component.artifacts["match_only_metrics"])
    compatibility_component = pd.read_csv(component.artifacts["compatibility_only_metrics"])
    assert len(match_component) == 4
    assert len(compatibility_component) == 2
    assert set(match_component["natural_endpoint"]) == {"Proliferating"}
    assert set(compatibility_component["natural_endpoint"]) == {"Proliferating"}
    assert set(match_component["method"]) == {"coreot_match_only"}
    assert set(compatibility_component["method"]) == {"coreot_constant_tau"}
    assert (match_component["tau_target"] == 8.0).all()
    assert (compatibility_component["tau_target"] == 8.0).all()
    report_metrics = {
        "auroc",
        "auprc",
        "forced_accuracy",
        "forced_macro_f1",
        "post_abstention_accuracy",
        "post_abstention_macro_f1",
        "coverage",
        "shared_false_abstention_rate",
    }
    assert match_component[list(report_metrics)].notna().all().all()
    assert compatibility_component[list(report_metrics)].notna().all().all()
    assert component.artifacts["compatibility_only_figure"].is_file()
    assert (
        component.artifacts["compatibility_only_figure"].name
        == "manuscript_fig_mouse_spleen_component_minus_m.png"
    )
    assert "compatibility_only_report" not in component.artifacts
    component_manifest = read_manifest(component.manifest_path)
    assert component_manifest.metadata["n_match_only_completed"] == 4
    assert component_manifest.metadata["n_compatibility_only_completed"] == 2
    assert component_manifest.metadata["point_estimates_only"] is True
    assert component_manifest.metadata["heatmap_metrics"] == [
        "auprc",
        "auroc",
        "forced_accuracy",
        "forced_macro_f1",
    ]

    config = yaml.safe_load(config_path.read_text())
    config["experiments"]["natural_mismatch"]["component_ablation"]["compatibility_tau_values"] = [
        5.0
    ]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    pruned_component = run_stage(config_path, "aggregate_natural_component_ablation")
    pruned_compatibility = pd.read_csv(pruned_component.artifacts["compatibility_only_metrics"])
    assert len(pruned_compatibility) == 1
    assert set(pruned_compatibility["tau_source"]) == {5.0}


def test_prepare_baselines_reconstructs_explicit_pseudocount_input(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _configure_precomputed_embedding(config_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    for key in ("rna_h5ad", "atac_gene_activity_h5ad"):
        path = Path(config["data"][key])
        adata = ad.read_h5ad(path)
        adata.X = np.full(adata.shape, np.log1p(2500.0))
        adata.write_h5ad(path)
    config["baselines"] = {
        "internal": ["prior_only", "nn", "uniform_uot"],
        "external": ["celltypist_l3"],
        "threshold_quantile": 0.95,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_stage(config_path, "prepare_baselines")

    pseudocount_manifest = read_manifest(result.artifacts["pseudocount_manifest"])
    assert pseudocount_manifest.metadata["n_common_genes"] == 4
    assert pseudocount_manifest.metadata["raw_library_sizes_recoverable"] is False
    counts = ad.read_h5ad(pseudocount_manifest.artifacts["model_visible_pseudocounts"])
    assert np.all(np.asarray(counts.layers["counts"].sum(axis=1)).ravel() == 10000)
    configs = [path for key, path in result.artifacts.items() if key.endswith("external_config")]
    assert len(configs) == 1
    payload = yaml.safe_load(configs[0].read_text())
    assert payload["conditions"] == ["incomplete_reference", "full_reference_control"]
    assert payload["methods"] == ["celltypist_l3"]
    task = read_manifest(
        Path(config["experiment"]["output_dir"])
        / "task_families/marginal_zone_b/task_manifest.yaml"
    )
    assert {item["condition_id"] for item in task.metadata["conditions"]} == {
        "incomplete_reference",
        "full_reference_control",
    }


def test_detection_only_method_has_no_label_transfer_metrics() -> None:
    metrics = _label_transfer_metrics(pd.DataFrame(), applicable=False)

    assert metrics
    assert all(np.isnan(value) for value in metrics.values())
