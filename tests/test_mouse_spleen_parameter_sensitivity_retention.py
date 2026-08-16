from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from coreot.artifacts.hashes import sha256_file
from experiments.mouse_spleen import pipeline


def _exercise_variant(
    tmp_path: Path,
    monkeypatch,
    *,
    retain_fit_artifacts: bool,
    candidate_distances: tuple[float, float] = (0.1, 0.2),
    retention_exempt_fit_roots: dict[tuple[float, float], Path] | None = None,
) -> tuple[Path, Path, list[Path]]:
    output_root = tmp_path / "output"
    run_root = tmp_path / "run"
    candidate_root = run_root / "candidates/natural_mismatch/candidate"
    profile_root = run_root / "derived/natural_mismatch/prior_profiles/default"
    truth_root = run_root / "benchmark/natural_mismatch/evaluation_truth"
    candidate_root.mkdir(parents=True, exist_ok=True)
    profile_root.mkdir(parents=True, exist_ok=True)
    truth_root.mkdir(parents=True, exist_ok=True)

    candidates = pd.DataFrame(
        {
            "source_cell_id": ["query_1", "query_2"],
            "target_cell_id": ["reference_1", "reference_1"],
            "scaled_distance": list(candidate_distances),
        }
    )
    candidate_path = candidate_root / "candidate_edges.parquet"
    candidates.to_parquet(candidate_path, index=False)
    pd.DataFrame({"cell_id": ["reference_1"]}).to_csv(
        profile_root / "target_priors.csv", index=False
    )
    pd.DataFrame(
        {
            "cell_id": ["query_1", "query_2"],
            "true_label": ["Proliferating", "B cell"],
            "is_shared_state": [False, True],
        }
    ).to_csv(truth_root / "query_truth.csv", index=False)
    source_priors = pd.DataFrame(
        {
            "cell_id": ["query_1", "query_2"],
            "rho": [0.45, 0.55],
            "prior_risk": [0.55, 0.45],
            "anchor_class_pred": ["other", "B"],
            "rho_recipe": ["test", "test"],
            "anchor_classifier_C": [1.0, 1.0],
            "anchor_classifier_max_iter": [5000, 5000],
            "anchor_classifier_random_state": [7, 7],
            "matchability_classifier_C": [10.0, 10.0],
            "matchability_classifier_max_iter": [5000, 5000],
            "matchability_classifier_random_state": [20260713, 20260713],
        }
    )

    monkeypatch.setattr(
        pipeline,
        "_natural_endpoint_manifests",
        lambda _: [
            {
                "metadata": {
                    "natural_endpoint": "Proliferating",
                    "candidate_set": "candidate",
                },
                "artifacts": {"run_root": str(run_root)},
            }
        ],
    )

    def write_fixed_priors(
        config: dict[str, object],
        *,
        run_root: Path,
        output_path: Path,
    ) -> pd.DataFrame:
        del config, run_root
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_priors.to_csv(output_path, index=False)
        return source_priors

    monkeypatch.setattr(pipeline, "_write_fixed_natural_source_priors", write_fixed_priors)
    monkeypatch.setattr(
        pipeline,
        "compute_prior_adjusted_deficit",
        lambda *_args, **_kwargs: SimpleNamespace(
            u_tilde=np.array([0.8, 0.2]),
            metadata={
                "model": "test",
                "cross_fitting": "cell",
                "n_folds": 2,
                "fold_source": "test",
                "anchor_stratification": "global",
                "fallback": "none",
            },
        ),
    )
    calls: list[Path] = []

    def run_method(
        method_root: Path,
        condition: str,
        candidate_edges: pd.DataFrame,
        source: pd.DataFrame,
        target: pd.DataFrame,
        method_config: dict[str, object],
    ) -> None:
        del condition, candidate_edges, source, target
        calls.append(method_root)
        assert yaml.safe_load((method_root / "method_params.yaml").read_text()) == method_config
        pd.DataFrame(
            {
                "cell_id": ["query_1", "query_2"],
                "u": [0.9, 0.1],
                "forced_label": ["B cell", "B cell"],
            }
        ).to_parquet(method_root / "cell_transport_scores.parquet", index=False)
        pd.DataFrame({"coupling": [1.0]}).to_parquet(
            method_root / "sparse_coupling.parquet", index=False
        )
        (method_root / "label_probabilities.npz").write_bytes(b"test")
        (method_root / "transport_manifest.yaml").write_text(
            yaml.safe_dump({"metadata": {"converged": True, "n_iter": 7}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline, "_run_coreot_full", run_method)
    config = {
        "experiment": {"output_dir": str(output_root)},
        "transport": {
            "epsilon": 0.05,
            "tolerance": 1.0e-6,
            "eta": 1.0e-12,
        },
        "scoring": {
            "prior_adjustment": {
                "enabled": True,
                "n_folds": 2,
                "stratify_by_anchor": False,
                "min_anchor_size": 1,
            }
        },
        "experiments": {
            "natural_mismatch": {
                "match_only_sensitivity": {
                    "tau_values": [1.0, 2.0],
                    "tau_target": 8.0,
                    "max_iterations": 5000,
                }
            }
        },
    }
    result = pipeline._run_natural_match_only_sensitivity_variant(
        config,
        root_name="parameter_grid",
        tau_target_override=8.0,
        max_iterations_override=5000,
        method_name="coreot_full",
        alpha=40.0,
        fixed_default_rho=True,
        pairs_override=[(1.0, 2.0)],
        endpoints={"Proliferating"},
        retain_fit_artifacts=retain_fit_artifacts,
        retention_exempt_fit_roots=retention_exempt_fit_roots,
    )
    return result.root, candidate_path, calls


def test_full_tau_variant_retains_complete_fit_artifacts_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, _, calls = _exercise_variant(
        tmp_path, monkeypatch, retain_fit_artifacts=True
    )
    fit_root = root / "fits/run/taumin_1_taumax_2"
    required = {
        "cell_transport_scores.parquet",
        "label_probabilities.npz",
        "method_params.yaml",
        "sparse_coupling.parquet",
        "transport_manifest.yaml",
    }
    assert required <= {path.name for path in fit_root.iterdir() if path.is_file()}
    checkpoint_path = root / "checkpoints/proliferating.csv"
    manifest_path = root / "run_manifest.yaml"
    checkpoint_before_resume = checkpoint_path.read_bytes()
    manifest_before_resume = manifest_path.read_bytes()

    _, _, resume_calls = _exercise_variant(
        tmp_path, monkeypatch, retain_fit_artifacts=True
    )
    assert calls == [fit_root]
    assert resume_calls == []
    assert checkpoint_path.read_bytes() == checkpoint_before_resume
    assert manifest_path.read_bytes() == manifest_before_resume

    external_fit_root = tmp_path / "canonical_fit"
    shutil.copytree(fit_root, external_fit_root)
    shutil.rmtree(fit_root)
    _, _, exempt_calls = _exercise_variant(
        tmp_path,
        monkeypatch,
        retain_fit_artifacts=True,
        retention_exempt_fit_roots={(1.0, 2.0): external_fit_root},
    )
    assert exempt_calls == []

    _, _, resumed_calls = _exercise_variant(
        tmp_path, monkeypatch, retain_fit_artifacts=True
    )
    assert resumed_calls == [fit_root]

    with pytest.raises(
        pipeline.MouseSpleenConfigError,
        match="candidate_edges_sha256",
    ):
        _exercise_variant(
            tmp_path,
            monkeypatch,
            retain_fit_artifacts=True,
            candidate_distances=(0.3, 0.4),
        )


def test_full_tau_variant_manifest_binds_candidate_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, candidate_path, _ = _exercise_variant(
        tmp_path, monkeypatch, retain_fit_artifacts=False
    )
    manifest = yaml.safe_load((root / "run_manifest.yaml").read_text())

    assert manifest["metadata"]["candidate_edges"]["proliferating"] == {
        "path": str(candidate_path),
        "sha256": sha256_file(candidate_path),
    }
