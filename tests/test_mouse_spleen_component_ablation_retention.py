from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from experiments.mouse_spleen import component_ablation


def test_run_grid_retains_complete_fit_artifacts_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[Path] = []

    def runner(
        method_root: Path,
        condition: str,
        candidates: pd.DataFrame,
        source_priors: pd.DataFrame,
        target_priors: pd.DataFrame,
        method_config: dict[str, object],
    ) -> None:
        del condition, candidates, source_priors, target_priors, method_config
        calls.append(method_root)
        pd.DataFrame({"cell_id": ["query_1"]}).to_parquet(
            method_root / "cell_transport_scores.parquet",
            index=False,
        )
        pd.DataFrame({"source_index": [0], "target_index": [0], "mass": [1.0]}).to_parquet(
            method_root / "sparse_coupling.parquet",
            index=False,
        )
        (method_root / "label_probabilities.npz").write_bytes(b"test")
        (method_root / "transport_manifest.yaml").write_text(
            yaml.safe_dump({"metadata": {"converged": True, "n_iter": 7}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        component_ablation,
        "_evaluate",
        lambda **_: {
            "evaluation_scope": "global_all_query",
            "n_query": 1,
            "n_positive": 1,
            "auroc": 1.0,
            "auprc": 1.0,
            "forced_accuracy": 1.0,
            "forced_macro_f1": 1.0,
            "post_abstention_accuracy": 1.0,
            "post_abstention_macro_f1": 1.0,
            "coverage": 1.0,
            "shared_false_abstention_rate": 0.0,
            "threshold_applicability": "undefined_without_matched_full_reference",
        },
    )
    settings = component_ablation._settings(
        {
            "experiments": {
                "natural_mismatch": {
                    "component_ablation": {
                        "endpoint": "Proliferating",
                        "match_tau_min_values": [3.0],
                        "match_tau_max_values": [5.0],
                        "compatibility_tau_values": [4.0],
                        "alpha_values": [0.0],
                        "tau_target": 8.0,
                        "max_iterations": 5000,
                        "retain_fit_artifacts": ["compatibility_only"],
                    }
                }
            }
        }
    )
    assert settings.retained_fit_variants == ("compatibility_only",)
    assert set(settings.execute_variants) == {"compatibility_only", "match_only"}
    arguments = {
        "variant": "compatibility_only",
        "combinations": [(4.0, 0.0)],
        "parameter_names": ("tau_source", "alpha"),
        "method_name": "coreot_constant_tau",
        "runner": runner,
        "root": tmp_path,
        "condition": "natural_mismatch",
        "candidate": "candidate",
        "run_root": Path("run"),
        "candidates": pd.DataFrame(),
        "source_priors": pd.DataFrame(),
        "target_priors": pd.DataFrame(),
        "truth": pd.DataFrame(),
        "settings": settings,
        "transport": {},
        "source_priors_hash": "source-hash",
        "candidate_edges_hash": "candidate-hash",
        "retain_fit_artifacts": True,
    }

    component_ablation._run_grid(**arguments)

    fit_root = (
        tmp_path
        / "fits/compatibility_only/tau_source_4_alpha_0"
    )
    assert (fit_root / "cell_transport_scores.parquet").is_file()
    assert (fit_root / "sparse_coupling.parquet").is_file()
    assert (fit_root / "label_probabilities.npz").is_file()
    assert (fit_root / "method_params.yaml").is_file()
    assert (fit_root / "transport_manifest.yaml").is_file()
    assert yaml.safe_load((fit_root / "method_params.yaml").read_text()) == {
        "alpha": 0.0,
        "epsilon": 0.05,
        "eta": 1.0e-12,
        "max_iter": 5000,
        "name": "coreot_constant_tau",
        "tau_source": 4.0,
        "tau_target": 8.0,
        "tol": 1.0e-6,
    }

    component_ablation._run_grid(**arguments)

    assert calls == [fit_root]


def test_component_ablation_execution_variants_can_select_compatibility_only() -> None:
    settings = component_ablation._settings(
        {
            "experiments": {
                "natural_mismatch": {
                    "component_ablation": {
                        "endpoint": "Proliferating",
                        "match_tau_min_values": [3.0],
                        "match_tau_max_values": [5.0],
                        "compatibility_tau_values": [1.0, 2.0],
                        "alpha_values": [0.0, 1.0],
                        "tau_target": 8.0,
                        "max_iterations": 5000,
                        "retain_fit_artifacts": ["compatibility_only"],
                        "execute_variants": ["compatibility_only"],
                    }
                }
            }
        }
    )

    assert settings.execute_variants == ("compatibility_only",)
