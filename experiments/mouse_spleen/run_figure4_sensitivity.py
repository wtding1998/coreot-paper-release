from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import yaml

from coreot.artifacts.hashes import sha256_file
from coreot.config.load import load_yaml
from coreot.evaluation.metrics import safe_auprc, safe_auroc, safe_median
from experiments.mouse_spleen.pipeline import (
    _natural_endpoint_manifests,
    _run_natural_constant_tau_sensitivity,
    _run_natural_match_only_sensitivity_variant,
    _write_fixed_natural_source_priors,
)


ENDPOINT = "Proliferating"
F1_ALPHA_VALUES = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
F1_TAU_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
F2_TAU_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
TAU_TARGET = 8.0
ALPHA_ADAPTIVE = 40.0
MAX_ITERATIONS = 5_000


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.lower().str.strip()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("Convergence column contains non-Boolean values")
    return normalized.eq("true")


def _endpoint_manifest(config: dict[str, object]) -> dict[str, object]:
    matches = [
        manifest
        for manifest in _natural_endpoint_manifests(config)
        if str(manifest["metadata"]["natural_endpoint"]) == ENDPOINT
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {ENDPOINT} endpoint manifest, found {len(matches)}")
    return matches[0]


def _seed_f1_checkpoint(output_root: Path, destination: Path) -> int:
    source = (
        output_root
        / "natural_mismatch/sensitivity/constant_tau_alpha_tau_target_8/tables/detection_by_run.csv"
    )
    frame = pd.read_csv(source)
    selected = frame.loc[
        frame["natural_endpoint"].eq(ENDPOINT)
        & frame["alpha"].astype(float).isin({0.0, 5.0, 10.0})
        & frame["tau"].astype(float).isin(F1_TAU_VALUES)
        & np.isclose(frame["tau_target"].astype(float), TAU_TARGET)
    ].copy()
    if len(selected) != 21 or not _as_bool(selected["converged"]).all():
        raise ValueError("The 21 reusable F1 configurations are incomplete or nonconverged")
    selected["converged"] = _as_bool(selected["converged"])
    selected["n_iterations"] = np.nan
    selected = selected.drop(columns=["valid_for_interpretation"], errors="ignore")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False)
    return len(selected)


def _canonical_f2_row(
    config: dict[str, object],
    *,
    output_root: Path,
    sensitivity_root: Path,
) -> pd.DataFrame:
    manifest = _endpoint_manifest(config)
    candidate = str(manifest["metadata"]["candidate_set"])
    run_root = Path(str(manifest["artifacts"]["run_root"]))
    fixed_prior_path = sensitivity_root / "priors/proliferating/source_priors.csv"
    source_priors = _write_fixed_natural_source_priors(
        config, run_root=run_root, output_path=fixed_prior_path
    )
    source_hash = sha256_file(fixed_prior_path)
    candidate_edges_path = (
        run_root
        / f"candidates/natural_mismatch/{candidate}/candidate_edges.parquet"
    )
    truth = pd.read_csv(
        run_root / "benchmark/natural_mismatch/evaluation_truth/query_truth.csv"
    )
    all_scores = pd.read_parquet(
        run_root
        / "scoring/natural_mismatch/mouse_spleen_provider_k100/cell_scores.parquet"
    )
    scores = all_scores.loc[all_scores["method"].astype(str).eq("coreot_full")].copy()
    joined = truth.merge(scores, on="cell_id", validate="one_to_one")
    positive = joined["true_label"].astype(str).eq(ENDPOINT)
    shared = joined.loc[joined["is_shared_state"].astype(bool)]
    transport_manifest_path = (
        run_root
        / "transport/natural_mismatch/mouse_spleen_provider_k100/coreot_full/transport_manifest.yaml"
    )
    transport_manifest = yaml.safe_load(transport_manifest_path.read_text(encoding="utf-8"))
    metadata = transport_manifest["metadata"]
    expected = {
        "alpha": ALPHA_ADAPTIVE,
        "tau_target": TAU_TARGET,
        "tau_source_min": 3.9,
        "tau_source_max": 4.1,
    }
    for key, value in expected.items():
        if not np.isclose(float(metadata[key]), value, rtol=0, atol=1e-12):
            raise ValueError(f"Canonical transport manifest has unexpected {key}")
    if not bool(metadata["converged"]):
        raise ValueError("Canonical CoRe-OT run did not converge")
    if not np.array_equal(
        np.sort(source_priors["cell_id"].astype(str).to_numpy()),
        np.sort(truth["cell_id"].astype(str).to_numpy()),
    ):
        raise ValueError("Canonical source priors and evaluation truth are misaligned")
    return pd.DataFrame(
        [
            {
                "run_id": run_root.name,
                "natural_endpoint": ENDPOINT,
                "condition": "natural_mismatch",
                "candidate_set": candidate,
                "method": "coreot_full",
                "tau_min": 3.0,
                "tau_max": 5.0,
                "tau_target": TAU_TARGET,
                "alpha": ALPHA_ADAPTIVE,
                "max_iterations": MAX_ITERATIONS,
                "converged": True,
                "n_iterations": int(metadata["n_iter"]),
                "runtime_seconds": np.nan,
                "evaluation_scope": "global_all_query",
                "n_query": int(len(joined)),
                "n_positive": int(positive.sum()),
                "auroc": safe_auroc(positive, joined["u"]),
                "auprc": safe_auprc(positive, joined["u"]),
                "auprc_baseline": float(positive.mean()),
                "median_absent": safe_median(joined.loc[positive, "u"]),
                "median_shared": safe_median(joined.loc[~positive, "u"]),
                "u_tilde_auroc": safe_auroc(positive, joined["u_tilde"]),
                "u_tilde_auprc": safe_auprc(positive, joined["u_tilde"]),
                "u_tilde_auprc_baseline": float(positive.mean()),
                "u_tilde_median_absent": safe_median(
                    joined.loc[positive, "u_tilde"]
                ),
                "u_tilde_median_shared": safe_median(
                    joined.loc[~positive, "u_tilde"]
                ),
                "u_tilde_model": "cross_fitted_isotonic",
                "u_tilde_cross_fitting": "cell",
                "u_tilde_n_folds": 5,
                "u_tilde_fold_source": "deterministic_cell",
                "u_tilde_anchor_stratification": "global",
                "u_tilde_fallback": "none",
                "shared_forced_accuracy": float(
                    accuracy_score(
                        shared["true_label"].astype(str),
                        shared["forced_label"].fillna("").astype(str),
                    )
                ),
                "shared_forced_macro_f1": float(
                    f1_score(
                        shared["true_label"].astype(str),
                        shared["forced_label"].fillna("").astype(str),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "rho_recipe": str(source_priors["rho_recipe"].iloc[0]),
                "source_priors_sha256": source_hash,
                "candidate_edges_sha256": sha256_file(candidate_edges_path),
                "anchor_prior_reuse_verified": True,
                "anchor_classifier_C": float(source_priors["anchor_classifier_C"].iloc[0]),
                "anchor_classifier_max_iter": int(
                    source_priors["anchor_classifier_max_iter"].iloc[0]
                ),
                "anchor_classifier_random_state": int(
                    source_priors["anchor_classifier_random_state"].iloc[0]
                ),
                "matchability_classifier_C": float(
                    source_priors["matchability_classifier_C"].iloc[0]
                ),
                "matchability_classifier_max_iter": int(
                    source_priors["matchability_classifier_max_iter"].iloc[0]
                ),
                "matchability_classifier_random_state": int(
                    source_priors["matchability_classifier_random_state"].iloc[0]
                ),
            }
        ]
    )


def _standardize_f1(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["source_run_id"] = result["run_id"]
    result["run_id"] = [
        f"figure4_f1_alpha_{alpha:g}_tau_{tau:g}"
        for alpha, tau in zip(result["alpha"], result["tau"], strict=True)
    ]
    result["tau_min"] = result["tau"].astype(float)
    result["tau_max"] = result["tau"].astype(float)
    result["is_constant_tau"] = True
    result["is_canonical"] = False
    result["is_exact_uniform_uot"] = False
    result["iterations"] = result["n_iterations"]
    result["reused_existing_artifact"] = result["alpha"].astype(float).isin(
        {0.0, 5.0, 10.0}
    )
    result["converged"] = _as_bool(result["converged"])
    return result.sort_values(["tau", "alpha"]).reset_index(drop=True)


def _standardize_f2(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["source_run_id"] = result["run_id"]
    result["run_id"] = [
        f"figure4_f2_taumin_{tau_min:g}_taumax_{tau_max:g}"
        for tau_min, tau_max in zip(result["tau_min"], result["tau_max"], strict=True)
    ]
    result["is_constant_tau"] = np.isclose(result["tau_min"], result["tau_max"])
    result["is_canonical"] = np.isclose(result["tau_min"], 3.0) & np.isclose(
        result["tau_max"], 5.0
    )
    result["is_exact_uniform_uot"] = False
    result["iterations"] = result["n_iterations"]
    result["reused_existing_artifact"] = result["is_canonical"]
    result["converged"] = _as_bool(result["converged"])
    return result.sort_values(["tau_max", "tau_min"]).reset_index(drop=True)


def _write_result(
    *, root: Path, frame: pd.DataFrame, stage: str, metadata: dict[str, object]
) -> Path:
    tables_root = root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    table_path = tables_root / "metrics_by_grid.csv"
    frame.to_csv(table_path, index=False)
    manifest = {
        "stage": stage,
        "artifacts": {"metrics_by_grid": str(table_path)},
        "metadata": metadata,
    }
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return table_path


def run_figure4_sensitivity(
    config_path: Path,
    *,
    retain_f2_fit_artifacts: bool = False,
) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    output_root = Path(str(config["experiment"]["output_dir"]))

    f1_root = (
        output_root
        / "natural_mismatch/sensitivity/figure4_constant_tau_alpha_tau_target_8"
    )
    f1_checkpoint = f1_root / "checkpoints/proliferating.csv"
    if not f1_checkpoint.is_file():
        reused_f1 = _seed_f1_checkpoint(output_root, f1_checkpoint)
    else:
        reused_f1 = int(
            pd.read_csv(f1_checkpoint)["alpha"].astype(float).isin({0.0, 5.0, 10.0}).sum()
        )
    f1_config = copy.deepcopy(config)
    sensitivity = f1_config["experiments"]["natural_mismatch"]["sensitivity"]
    sensitivity["alpha_values"] = F1_ALPHA_VALUES
    sensitivity["tau_values"] = F1_TAU_VALUES
    _run_natural_constant_tau_sensitivity(
        f1_config,
        root_name=f1_root.name,
        tau_target_override=TAU_TARGET,
        max_iterations_override=MAX_ITERATIONS,
        endpoints={ENDPOINT},
    )
    f1 = _standardize_f1(pd.read_csv(f1_checkpoint))
    if len(f1) != len(F1_ALPHA_VALUES) * len(F1_TAU_VALUES):
        raise ValueError("F1 checkpoint is incomplete")
    f1_path = _write_result(
        root=f1_root,
        frame=f1,
        stage="run_mouse_spleen_figure4_f1_sensitivity",
        metadata={
            "endpoint": ENDPOINT,
            "alpha_values": F1_ALPHA_VALUES,
            "tau_values": F1_TAU_VALUES,
            "tau_target": TAU_TARGET,
            "reused_existing_configurations": reused_f1,
            "actual_iterations_unavailable_for_reused_legacy_rows": True,
            "nonconverged_policy": "retained_with_flag_and_masked_in_figure",
        },
    )

    f2_root = (
        output_root
        / "natural_mismatch/sensitivity/figure4_full_tau_alpha_40_tau_target_8"
    )
    f2_checkpoint = f2_root / "checkpoints/proliferating.csv"
    if not f2_checkpoint.is_file():
        canonical = _canonical_f2_row(
            config, output_root=output_root, sensitivity_root=f2_root
        )
        f2_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_csv(f2_checkpoint, index=False)
    pairs = [
        (tau_min, tau_max)
        for tau_min in F2_TAU_VALUES
        for tau_max in F2_TAU_VALUES
        if tau_min <= tau_max
    ]
    canonical_manifest = _endpoint_manifest(config)
    canonical_candidate = str(canonical_manifest["metadata"]["candidate_set"])
    canonical_run_root = Path(str(canonical_manifest["artifacts"]["run_root"]))
    canonical_fit_root = (
        canonical_run_root
        / f"transport/natural_mismatch/{canonical_candidate}/coreot_full"
    )
    _run_natural_match_only_sensitivity_variant(
        config,
        root_name=f2_root.name,
        tau_target_override=TAU_TARGET,
        max_iterations_override=MAX_ITERATIONS,
        method_name="coreot_full",
        alpha=ALPHA_ADAPTIVE,
        fixed_default_rho=True,
        pairs_override=pairs,
        endpoints={ENDPOINT},
        retain_fit_artifacts=retain_f2_fit_artifacts,
        retention_exempt_fit_roots=(
            {(3.0, 5.0): canonical_fit_root}
            if retain_f2_fit_artifacts
            else None
        ),
    )
    f2 = _standardize_f2(pd.read_csv(f2_checkpoint))
    if len(f2) != len(pairs) or int(f2["is_canonical"].sum()) != 1:
        raise ValueError("F2 checkpoint is incomplete or lacks a unique canonical cell")
    f2_path = _write_result(
        root=f2_root,
        frame=f2,
        stage="run_mouse_spleen_figure4_f2_sensitivity",
        metadata={
            "endpoint": ENDPOINT,
            "tau_values": F2_TAU_VALUES,
            "alpha": ALPHA_ADAPTIVE,
            "tau_target": TAU_TARGET,
            "canonical": [3.0, 5.0, TAU_TARGET, ALPHA_ADAPTIVE],
            "canonical_reused_from_main_run": True,
            "nonconverged_policy": "retained_with_flag_and_masked_in_figure",
        },
    )
    return f1_path, f2_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Proliferating-only Figure 4 sensitivity grids.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/mouse_spleen/configs/mouse_spleen_core_ot.yaml"),
    )
    parser.add_argument(
        "--retain-f2-fit-artifacts",
        action="store_true",
        help="Retain complete noncanonical F2 transport artifacts and resume only complete fits.",
    )
    args = parser.parse_args(argv)
    for path in run_figure4_sensitivity(
        args.config,
        retain_f2_fit_artifacts=args.retain_f2_fit_artifacts,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
