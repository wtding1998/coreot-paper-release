from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow
import sklearn
import yaml

REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from coreot.transport.runner import _apply_provider_rho  # noqa: E402
from experiments.component_ablation_surfaces import (  # noqa: E402
    CONDITION,
    _input_paths,
    _pbmc_condition_map,
    alpha_values,
    selected_specs,
)
from experiments.rho_attribution_search import (  # noqa: E402
    _evaluate_variant,
    _pbmc_provider_rho_path,
    _run_method,
    mean_matched_tau,
)


UNIT = Path(__file__).resolve().parents[2]
FITS = UNIT / "regenerated/fits"
TEMP_ROOT = UNIT / "regenerated/tmp"
STATUS = UNIT / "audit/reconstruction_status.csv"
FOCUSED = (
    REPO
    / "results/PBMC/sensitivity/rho_attribution_tau_surface_alpha0_range075_175"
    / "tables/rho_tau_surface_range075_175_by_seed.csv"
)
ALPHA = (
    REPO
    / "results/PBMC/sensitivity/rho_attribution_alpha_search/tables"
    / "rho_attribution_by_replicate.csv"
)
VARIANTS = ("heterogeneous", "mean_matched_uniform")
METRICS = ("auroc", "auprc", "forced_accuracy", "forced_macro_f1")
SCORE_COLUMNS = ("cell_id", "u", "forced_label")
TOLERANCE = 1.0e-12
FOCUSED_TAU_VALUES = (0.75, 1.0, 1.25, 1.5, 1.75)
CODE_PATHS = {
    "pbmc_rho_tau_heatmap": REPO / "experiments/pbmc_rho_tau_heatmap.py",
    "rho_attribution_search": REPO / "experiments/rho_attribution_search.py",
    "component_ablation_surfaces": REPO / "experiments/component_ablation_surfaces.py",
    "mouse_component_ablation": REPO / "experiments/mouse_spleen/component_ablation.py",
    "artifact_hashes": REPO / "src/coreot/artifacts/hashes.py",
    "artifact_manifests": REPO / "src/coreot/artifacts/manifests.py",
    "config_load": REPO / "src/coreot/config/load.py",
    "data_schemas": REPO / "src/coreot/data/schemas.py",
    "data_validation": REPO / "src/coreot/data/validation.py",
    "evaluation_metrics": REPO / "src/coreot/evaluation/metrics.py",
    "provider_reliability": REPO / "src/coreot/preprocessing/provider_reliability.py",
    "label_transfer": REPO / "src/coreot/transport/label_transfer.py",
    "transport_runner": REPO / "src/coreot/transport/runner.py",
    "sinkhorn_solver": REPO / "src/coreot/transport/sinkhorn.py",
    "reconstruction_worker": Path(__file__).resolve(),
}


def value_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def endpoint_slug(value: str) -> str:
    return value.lower().replace("+", "").replace("-", "").replace(" ", "_")


def case_slug(case: object) -> str:
    if str(case.analysis_family) == "focused_tau":
        return (
            f"tau_min_{value_slug(case.tau_min)}_"
            f"tau_max_{value_slug(case.tau_max)}"
        )
    return f"alpha_{value_slug(case.alpha)}"


def fit_id(case: object, variant: str) -> str:
    return (
        f"{case.analysis_family}_{endpoint_slug(str(case.endpoint))}_"
        f"seed{int(case.seed)}_{case_slug(case)}_{variant}"
    )


def fit_root(case: object, variant: str) -> Path:
    return (
        FITS
        / str(case.analysis_family)
        / endpoint_slug(str(case.endpoint))
        / f"seed{int(case.seed)}"
        / case_slug(case)
        / variant
    )


def load_cases() -> pd.DataFrame:
    focused = pd.read_csv(FOCUSED)
    alpha = pd.read_csv(ALPHA)
    if len(focused) != 225 or focused.duplicated(
        ["endpoint", "seed", "tau_min", "tau_max"]
    ).any():
        raise RuntimeError("Focused attribution table is not the expected 225 unique pairs")
    if len(alpha) != 75 or alpha.duplicated(["endpoint", "seed", "alpha"]).any():
        raise RuntimeError("Alpha-search table is not the expected 75 unique pairs")
    focused["analysis_family"] = "focused_tau"
    focused["epsilon"] = 0.05
    focused["max_iterations"] = 5000
    focused["tolerance"] = 1.0e-6
    focused["numerical_floor"] = 1.0e-300
    focused["evaluation_scope"] = "within_celltype"
    focused["n_detection"] = np.nan
    focused["n_positive"] = np.nan
    focused["heterogeneous_runtime_seconds"] = np.nan
    focused["uniform_runtime_seconds"] = np.nan
    alpha["analysis_family"] = "alpha_search"
    specs = {spec.endpoint: spec for spec in selected_specs("pbmc")}
    expected_endpoints = set(specs)
    expected_seeds = set(range(1, 6))
    focused_pairs = {
        (tau_min, tau_max)
        for tau_min in FOCUSED_TAU_VALUES
        for tau_max in FOCUSED_TAU_VALUES
        if tau_min <= tau_max
    }
    expected_focused = {
        (endpoint, seed, tau_min, tau_max)
        for endpoint in expected_endpoints
        for seed in expected_seeds
        for tau_min, tau_max in focused_pairs
    }
    observed_focused = {
        (str(row.endpoint), int(row.seed), float(row.tau_min), float(row.tau_max))
        for row in focused.itertuples(index=False)
    }
    if observed_focused != expected_focused:
        raise RuntimeError("Focused attribution parameter-key set is incomplete or stale")
    if not focused["alpha"].astype(float).eq(0.0).all():
        raise RuntimeError("Focused attribution rows must all use alpha=0")
    expected_alpha = {
        (endpoint, seed, float(value), float(value / specs[endpoint].selected_alpha))
        for endpoint in expected_endpoints
        for seed in expected_seeds
        for value in alpha_values(specs[endpoint].selected_alpha)
    }
    observed_alpha = {
        (str(row.endpoint), int(row.seed), float(row.alpha), float(row.alpha_ratio))
        for row in alpha.itertuples(index=False)
    }
    if observed_alpha != expected_alpha:
        raise RuntimeError("Alpha-search parameter-key set is incomplete or stale")
    for endpoint, spec in specs.items():
        focused_local = focused.loc[focused["endpoint"].eq(endpoint)]
        alpha_local = alpha.loc[alpha["endpoint"].eq(endpoint)]
        expected_run_ids = {
            spec.run_prefix.format(seed=seed) for seed in expected_seeds
        }
        if set(focused_local["run_id"].astype(str)) != expected_run_ids:
            raise RuntimeError(f"Focused run-ID drift for {endpoint}")
        if set(alpha_local["run_id"].astype(str)) != expected_run_ids:
            raise RuntimeError(f"Alpha-search run-ID drift for {endpoint}")
        if not np.isclose(focused_local["tau_target"], spec.tau_target).all():
            raise RuntimeError(f"Focused tau_target drift for {endpoint}")
        if not np.isclose(alpha_local["tau_min"], spec.selected_tau[0]).all():
            raise RuntimeError(f"Alpha-search tau_min drift for {endpoint}")
        if not np.isclose(alpha_local["tau_max"], spec.selected_tau[1]).all():
            raise RuntimeError(f"Alpha-search tau_max drift for {endpoint}")
        if not np.isclose(alpha_local["tau_target"], spec.tau_target).all():
            raise RuntimeError(f"Alpha-search tau_target drift for {endpoint}")
    if not np.isclose(alpha["epsilon"], 0.05).all():
        raise RuntimeError("Alpha-search epsilon drift")
    if not alpha["max_iterations"].astype(int).eq(5000).all():
        raise RuntimeError("Alpha-search iteration-cap drift")
    if not np.isclose(alpha["tolerance"], 1.0e-6).all():
        raise RuntimeError("Alpha-search tolerance drift")
    if not np.isclose(alpha["numerical_floor"], 1.0e-300, rtol=0.0, atol=0.0).all():
        raise RuntimeError("Alpha-search numerical-floor drift")
    if not alpha["evaluation_scope"].astype(str).eq("within_celltype").all():
        raise RuntimeError("Alpha-search evaluation-scope drift")
    cases = pd.concat([focused, alpha], ignore_index=True, sort=False)
    cases = cases.sort_values(
        ["endpoint", "seed", "analysis_family", "tau_min", "tau_max", "alpha"]
    ).reset_index(drop=True)
    if len(cases) != 300:
        raise RuntimeError(f"Expected 300 paired cases, found {len(cases)}")
    return cases


def environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in CODE_PATHS.items()}


def prepared_inputs(
    *, case: object, runs_root: Path
) -> tuple[object, dict[str, Path], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = {spec.endpoint: spec for spec in selected_specs("pbmc")}
    spec = specs[str(case.endpoint)]
    run_root = runs_root / str(case.run_id)
    paths = _input_paths(run_root, spec.candidate_set)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attribution inputs: {missing}")
    candidates = pd.read_parquet(paths["candidates"])
    source_priors = pd.read_csv(paths["source_priors"])
    target_priors = pd.read_csv(paths["target_priors"])
    truth = pd.read_csv(paths["truth"])
    provider_path = _pbmc_provider_rho_path(runs_root, run_root, spec)
    provider_metadata_path = provider_path.with_name("rho_metadata.yaml")
    for path in (provider_path, provider_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    provider_rho = pd.read_csv(provider_path)
    provider_metadata = yaml.safe_load(
        provider_metadata_path.read_text(encoding="utf-8")
    )
    provider_rho["coreot_rho_cell_id_hash"] = str(
        provider_metadata["cell_id_rho_sha256"]
    )
    source_priors = _apply_provider_rho(source_priors, provider_rho, CONDITION)
    paths = paths | {
        "provider_rho": provider_path,
        "provider_metadata": provider_metadata_path,
    }
    return spec, paths, candidates, source_priors, target_priors, truth


def verify_input_hashes(case: object, paths: dict[str, Path]) -> dict[str, str]:
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    combined_source = "+".join(
        [
            hashes["source_priors"],
            hashes["provider_rho"],
            hashes["provider_metadata"],
        ]
    )
    if hashes["candidates"] != str(case.candidate_edges_sha256):
        raise RuntimeError(f"Candidate hash mismatch for {case.endpoint} seed {case.seed}")
    if combined_source != str(case.source_priors_sha256):
        raise RuntimeError(f"Source-prior hash mismatch for {case.endpoint} seed {case.seed}")
    hashes["combined_source_priors"] = combined_source
    return hashes


def method_config(case: object, variant: str) -> dict[str, object]:
    common: dict[str, object] = {
        "tau_target": float(case.tau_target),
        "epsilon": float(case.epsilon),
        "alpha": float(case.alpha),
        "max_iter": int(case.max_iterations),
        "tol": float(case.tolerance),
        "numerical_floor": float(case.numerical_floor),
        "eta": 1.0e-12,
    }
    if variant == "heterogeneous":
        return {
            **common,
            "name": "coreot_full",
            "tau_min": float(case.tau_min),
            "tau_max": float(case.tau_max),
        }
    return {
        **common,
        "name": "coreot_constant_tau",
        "tau_source": "matched_coreot_mean",
        "matched_tau_min": float(case.tau_min),
        "matched_tau_max": float(case.tau_max),
    }


def historical_values(case: object, variant: str) -> tuple[dict[str, float], bool, int, float | None]:
    prefix = "heterogeneous" if variant == "heterogeneous" else "uniform"
    metrics = {metric: float(getattr(case, f"{prefix}_{metric}")) for metric in METRICS}
    converged = bool(getattr(case, f"{prefix}_converged"))
    iterations = int(getattr(case, f"{prefix}_n_iterations"))
    runtime_value = getattr(case, f"{prefix}_runtime_seconds")
    runtime = None if pd.isna(runtime_value) else float(runtime_value)
    return metrics, converged, iterations, runtime


def string_sequence_hash(values: pd.Series) -> str:
    payload = "\n".join(values.astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def schema_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{column}:{frame[column].dtype}" for column in frame.columns
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_compact_scores(scores: pd.DataFrame, truth: pd.DataFrame) -> None:
    if tuple(scores.columns) != SCORE_COLUMNS:
        raise RuntimeError(f"Unexpected compact score columns: {list(scores.columns)}")
    if scores["cell_id"].isna().any() or scores["cell_id"].duplicated().any():
        raise RuntimeError("Compact scores contain null or duplicate cell IDs")
    if scores["cell_id"].astype(str).tolist() != truth["cell_id"].astype(str).tolist():
        raise RuntimeError("Compact score cell IDs/order differ from evaluation truth")
    values = pd.to_numeric(scores["u"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Compact scores contain non-finite u")
    if scores["forced_label"].isna().any():
        raise RuntimeError("Compact scores contain null forced labels")
    forced = set(scores["forced_label"].astype(str))
    allowed = set(truth["true_label"].dropna().astype(str))
    if not forced or not forced.issubset(allowed):
        raise RuntimeError("Compact scores contain an invalid forced-label domain")


def existing_fit_is_valid(
    *,
    root: Path,
    case: object,
    variant: str,
    input_hashes: dict[str, str],
    truth: pd.DataFrame,
) -> bool:
    required = ("resolved_config.yaml", "fit_manifest.yaml", "cell_scores.parquet")
    if not all((root / name).is_file() for name in required):
        return False
    try:
        config_path = root / "resolved_config.yaml"
        manifest = yaml.safe_load((root / "fit_manifest.yaml").read_text(encoding="utf-8"))
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        score_path = root / "cell_scores.parquet"
        scores = pd.read_parquet(score_path)
        artifacts = manifest["artifacts"]
        metadata = manifest["metadata"]
        if metadata["comparison_status"] != "pass":
            return False
        if resolved["fit_id"] != fit_id(case, variant) or resolved["variant"] != variant:
            return False
        if resolved["method"] != method_config(case, variant):
            return False
        if resolved["code_sha256"] != code_hashes():
            return False
        if metadata["code_sha256"] != code_hashes():
            return False
        if artifacts["resolved_config"]["sha256"] != sha256_file(config_path):
            return False
        score_record = artifacts["cell_scores"]
        if score_record["sha256"] != sha256_file(score_path):
            return False
        if int(score_record["bytes"]) != score_path.stat().st_size:
            return False
        if int(score_record["rows"]) != len(scores):
            return False
        if list(score_record["columns"]) != list(scores.columns):
            return False
        if tuple(scores.columns) != SCORE_COLUMNS:
            return False
        validate_compact_scores(scores, truth)
        if score_record["cell_id_order_sha256"] != string_sequence_hash(scores["cell_id"]):
            return False
        if score_record["schema_sha256"] != schema_hash(scores):
            return False
        for name, record in resolved["inputs"].items():
            path = REPO / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                return False
            if path.stat().st_size != int(record["bytes"]):
                return False
            if input_hashes[name] != record["sha256"]:
                return False
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError):
        return False
    return True


def reconstruct_one(
    *,
    case: object,
    variant: str,
    spec: object,
    paths: dict[str, Path],
    input_hashes: dict[str, str],
    candidates: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    truth: pd.DataFrame,
    condition_map: pd.Series,
) -> dict[str, object]:
    root = fit_root(case, variant)
    if existing_fit_is_valid(
        root=root,
        case=case,
        variant=variant,
        input_hashes=input_hashes,
        truth=truth,
    ):
        manifest = yaml.safe_load((root / "fit_manifest.yaml").read_text(encoding="utf-8"))
        return dict(manifest["metadata"]["status_row"])
    local_spec = replace(
        spec,
        selected_tau=(float(case.tau_min), float(case.tau_max)),
    )
    expected_mean = mean_matched_tau(
        source_priors, float(case.tau_min), float(case.tau_max)
    )
    if abs(expected_mean - float(case.mean_matched_tau)) > TOLERANCE:
        raise RuntimeError(
            f"Mean-matched tau differs for {fit_id(case, variant)}: "
            f"{expected_mean} vs {case.mean_matched_tau}"
        )
    historical_metrics, historical_converged, historical_iterations, historical_runtime = (
        historical_values(case, variant)
    )
    current_code_hashes = code_hashes()
    current_environment = environment_metadata()
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pbmc_matchability_fit_", dir=TEMP_ROOT) as directory:
        temporary = Path(directory)
        scores, metadata, runtime = _run_method(
            variant=variant,
            method_root=temporary,
            condition=CONDITION,
            candidates=candidates,
            source_priors=source_priors,
            target_priors=target_priors,
            spec=local_spec,
            alpha=float(case.alpha),
        )
        metrics = _evaluate_variant(
            experiment="pbmc",
            endpoint=str(case.endpoint),
            truth=truth,
            scores=scores,
            pbmc_condition=condition_map,
        )
        differences = {
            metric: abs(float(metrics[metric]) - historical_metrics[metric])
            for metric in METRICS
        }
        comparison_status = "pass" if max(differences.values()) <= TOLERANCE else "fail"
        if bool(metadata["converged"]) != historical_converged:
            comparison_status = "fail"
        if int(metadata["n_iterations"]) != historical_iterations:
            comparison_status = "fail"
        if str(metrics["evaluation_scope"]) != str(case.evaluation_scope):
            comparison_status = "fail"
        if not pd.isna(case.n_detection) and int(metrics["n_detection"]) != int(case.n_detection):
            comparison_status = "fail"
        if not pd.isna(case.n_positive) and int(metrics["n_positive"]) != int(case.n_positive):
            comparison_status = "fail"
        if comparison_status != "pass":
            raise RuntimeError(
                f"Reconstructed fit differs for {fit_id(case, variant)}: "
                f"iterations={metadata['n_iterations']}/{historical_iterations}, "
                f"differences={differences}"
            )
        root.mkdir(parents=True, exist_ok=True)
        score_path = root / "cell_scores.parquet"
        compact_scores = scores.loc[:, list(SCORE_COLUMNS)].copy()
        validate_compact_scores(compact_scores, truth)
        compact_scores.to_parquet(score_path, index=False)
        resolved = {
            "fit_id": fit_id(case, variant),
            "analysis_family": str(case.analysis_family),
            "experiment": "pbmc",
            "endpoint": str(case.endpoint),
            "seed": int(case.seed),
            "base_run_id": str(case.run_id),
            "condition": CONDITION,
            "candidate_set": "pca30_k100",
            "variant": variant,
            "method": method_config(case, variant),
            "mean_matched_tau": expected_mean,
            "inputs": {
                name: {
                    "path": str(path.resolve().relative_to(REPO)),
                    "sha256": input_hashes[name],
                    "bytes": path.stat().st_size,
                    "visibility": (
                        "evaluation_only"
                        if name in {"truth", "pbmc_raw_h5ad"}
                        else "model_visible"
                    ),
                }
                for name, path in paths.items()
            },
            "combined_source_priors_sha256": input_hashes["combined_source_priors"],
            "code_sha256": current_code_hashes,
            "environment": current_environment,
            "evaluation_scope": str(metrics["evaluation_scope"]),
        }
        config_path = root / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        status_row = {
            "fit_id": fit_id(case, variant),
            "analysis_family": str(case.analysis_family),
            "endpoint": str(case.endpoint),
            "seed": int(case.seed),
            "tau_min": float(case.tau_min),
            "tau_max": float(case.tau_max),
            "alpha": float(case.alpha),
            "variant": variant,
            "converged": bool(metadata["converged"]),
            "n_iterations": int(metadata["n_iterations"]),
            "historical_runtime_seconds": historical_runtime,
            "reconstruction_runtime_seconds": runtime,
            "n_detection": int(metrics["n_detection"]),
            "n_positive": int(metrics["n_positive"]),
            "max_metric_abs_difference": max(differences.values()),
            "comparison_status": comparison_status,
            "cell_scores_sha256": sha256_file(score_path),
            "cell_scores_bytes": score_path.stat().st_size,
            "cell_scores_rows": len(compact_scores),
        }
        transport_manifest = yaml.safe_load(
            (temporary / "transport_manifest.yaml").read_text(encoding="utf-8")
        )
        manifest = {
            "stage": "isolated-rho-attribution-fit-reconstruction",
            "artifacts": {
                "resolved_config": {
                    "path": str(config_path.relative_to(REPO)),
                    "sha256": sha256_file(config_path),
                    "bytes": config_path.stat().st_size,
                },
                "cell_scores": {
                    "path": str(score_path.relative_to(REPO)),
                    "sha256": sha256_file(score_path),
                    "bytes": score_path.stat().st_size,
                    "rows": len(compact_scores),
                    "columns": list(compact_scores.columns),
                    "cell_id_order_sha256": string_sequence_hash(
                        compact_scores["cell_id"]
                    ),
                    "schema_sha256": schema_hash(compact_scores),
                },
            },
            "metadata": {
                "comparison_status": comparison_status,
                "tolerance": TOLERANCE,
                "historical_metrics": historical_metrics,
                "reconstructed_metrics": {
                    metric: float(metrics[metric]) for metric in METRICS
                },
                "metric_abs_differences": differences,
                "transport_metadata": transport_manifest["metadata"],
                "temporary_sparse_coupling_retained": False,
                "temporary_label_probabilities_retained": False,
                "score_projection": list(SCORE_COLUMNS),
                "code_sha256": current_code_hashes,
                "environment": current_environment,
                "status_row": status_row,
            },
        }
        (root / "fit_manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
    return status_row


def write_status(rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows).drop_duplicates("fit_id", keep="last")
    frame = frame.sort_values(
        ["endpoint", "seed", "analysis_family", "tau_min", "tau_max", "alpha", "variant"]
    )
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(STATUS, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--pbmc-raw-path", type=Path, default=Path("data/raw/kang_2018.h5ad"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fit-id", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases()
    selected: list[tuple[object, str]] = [
        (case, variant)
        for case in cases.itertuples(index=False)
        for variant in VARIANTS
        if not args.fit_id or fit_id(case, variant) in set(args.fit_id)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    selected_ids = [fit_id(case, variant) for case, variant in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Selected reconstruction fit IDs are not unique")
    if not args.fit_id and args.limit is None and len(selected_ids) != 600:
        raise RuntimeError(f"Expected exactly 600 fits, found {len(selected_ids)}")
    raw_path = args.pbmc_raw_path.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    condition_map = _pbmc_condition_map(raw_path)
    status_rows = pd.read_csv(STATUS).to_dict("records") if STATUS.is_file() else []
    failures = 0
    cached_key: tuple[str, int] | None = None
    prepared: tuple[object, dict[str, Path], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    input_hashes: dict[str, str] | None = None
    for index, (case, variant) in enumerate(selected, start=1):
        key = (str(case.endpoint), int(case.seed))
        try:
            if key != cached_key:
                prepared = prepared_inputs(case=case, runs_root=args.runs_root)
                spec, paths, _, _, _, _ = prepared
                paths["pbmc_raw_h5ad"] = raw_path
                input_hashes = verify_input_hashes(case, paths)
                cached_key = key
            if prepared is None or input_hashes is None:
                raise RuntimeError("Prepared input cache was not initialized")
            spec, paths, candidates, source_priors, target_priors, truth = prepared
            status_row = reconstruct_one(
                case=case,
                variant=variant,
                spec=spec,
                paths=paths,
                input_hashes=input_hashes,
                candidates=candidates,
                source_priors=source_priors,
                target_priors=target_priors,
                truth=truth,
                condition_map=condition_map,
            )
        except Exception as error:  # continue independent fits unattended
            failures += 1
            status_row = {
                "fit_id": fit_id(case, variant),
                "analysis_family": str(case.analysis_family),
                "endpoint": str(case.endpoint),
                "seed": int(case.seed),
                "tau_min": float(case.tau_min),
                "tau_max": float(case.tau_max),
                "alpha": float(case.alpha),
                "variant": variant,
                "comparison_status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        status_rows.append(status_row)
        write_status(status_rows)
        print(
            f"completed {index}/{len(selected)} {fit_id(case, variant)} "
            f"status={status_row['comparison_status']}",
            flush=True,
        )
    print(
        f"environment python={platform.python_version()} numpy={np.__version__} "
        f"pandas={pd.__version__}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
