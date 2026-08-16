from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow
import yaml

REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from coreot.transport.runner import (  # noqa: E402
    _apply_provider_rho,
    _run_coreot_constant_tau,
)
from experiments.component_ablation_surfaces import (  # noqa: E402
    CONDITION,
    _input_paths,
    _pbmc_condition_map,
    _value_slug,
    evaluate_scores,
    selected_specs,
)


UNIT = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results/PBMC/sensitivity/component_ablation/tables/component_ablation_by_seed.csv"
FITS = UNIT / "regenerated/fits"
TEMP_ROOT = UNIT / "regenerated/tmp"
STATUS = UNIT / "audit/reconstruction_status.csv"
METRICS = ("auroc", "auprc", "forced_accuracy", "forced_macro_f1")
TOLERANCE = 1.0e-12
SCORE_COLUMNS = ("cell_id", "u", "forced_label")
CODE_PATHS = {
    "component_ablation_surfaces": REPO / "experiments/component_ablation_surfaces.py",
    "artifact_hashes": REPO / "src/coreot/artifacts/hashes.py",
    "artifact_manifests": REPO / "src/coreot/artifacts/manifests.py",
    "config_load": REPO / "src/coreot/config/load.py",
    "data_hidden": REPO / "src/coreot/data/hidden.py",
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


def fit_id(row: object) -> str:
    return (
        f"{endpoint_slug(str(row.endpoint))}_seed{int(row.seed)}_"
        f"tau{value_slug(row.tau_source)}_alpha{value_slug(row.alpha)}"
    )


def fit_root(row: object) -> Path:
    return (
        FITS
        / endpoint_slug(str(row.endpoint))
        / f"seed{int(row.seed)}"
        / f"tau{value_slug(row.tau_source)}_alpha{value_slug(row.alpha)}"
    )


def selected_run_id(spec: object, base_run_id: str) -> str:
    return (
        f"{base_run_id}_taumin{_value_slug(spec.selected_tau[0])}"
        f"_taumax{_value_slug(spec.selected_tau[1])}"
        f"_alpha{_value_slug(spec.selected_alpha)}_coreot_full"
    )


def load_rows() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    frame = frame.loc[
        frame["experiment"].eq("pbmc")
        & frame["variant"].eq("compatibility_only")
    ].copy()
    frame = frame.sort_values(["endpoint", "seed", "tau_source", "alpha"])
    if len(frame) != 375:
        raise RuntimeError(f"Expected 375 compatibility fits, found {len(frame)}")
    if frame.duplicated(["endpoint", "seed", "tau_source", "alpha"]).any():
        raise RuntimeError("Compatibility fit keys are not unique")
    return frame.reset_index(drop=True)


def prepared_inputs(
    *, row: object, runs_root: Path
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = {spec.endpoint: spec for spec in selected_specs("pbmc")}
    spec = specs[str(row.endpoint)]
    run_root = runs_root / str(row.run_id)
    paths = _input_paths(run_root, spec.candidate_set)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing focused-fit inputs: {missing}")

    candidates = pd.read_parquet(paths["candidates"])
    source_priors = pd.read_csv(paths["source_priors"])
    target_priors = pd.read_csv(paths["target_priors"])
    truth = pd.read_csv(paths["truth"])
    selected_id = selected_run_id(spec, str(row.run_id))
    provider_root = runs_root / selected_id / "transport/provider_reliability/pca30"
    provider_path = provider_root / "source_rho.csv"
    provider_metadata_path = provider_root / "rho_metadata.yaml"
    for path in (provider_path, provider_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    provider_rho = pd.read_csv(provider_path)
    provider_metadata = yaml.safe_load(provider_metadata_path.read_text(encoding="utf-8"))
    provider_rho["coreot_rho_cell_id_hash"] = str(
        provider_metadata["cell_id_rho_sha256"]
    )
    source_priors = _apply_provider_rho(source_priors, provider_rho, CONDITION)
    paths = paths | {
        "provider_rho": provider_path,
        "provider_metadata": provider_metadata_path,
    }
    return paths, candidates, source_priors, target_priors, truth


def verify_input_hashes(row: object, paths: dict[str, Path]) -> dict[str, str]:
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    combined_source = "+".join(
        [
            hashes["source_priors"],
            hashes["provider_rho"],
            hashes["provider_metadata"],
        ]
    )
    if hashes["candidates"] != str(row.candidate_edges_sha256):
        raise RuntimeError(f"Candidate hash mismatch for {fit_id(row)}")
    if combined_source != str(row.source_priors_sha256):
        raise RuntimeError(f"Source-prior hash mismatch for {fit_id(row)}")
    hashes["combined_source_priors"] = combined_source
    return hashes


def method_config(row: object) -> dict[str, object]:
    return {
        "name": "coreot_constant_tau",
        "tau_source": float(row.tau_source),
        "tau_target": float(row.tau_target),
        "alpha": float(row.alpha),
        "epsilon": float(row.epsilon),
        "max_iter": int(row.max_iterations),
        "tol": float(row.tolerance),
        "numerical_floor": float(row.numerical_floor),
        "eta": 1.0e-12,
    }


def environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
    }


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in CODE_PATHS.items()}


def existing_fit_is_valid(root: Path, row: object) -> bool:
    required = ("resolved_config.yaml", "fit_manifest.yaml", "cell_scores.parquet")
    if not all((root / name).is_file() for name in required):
        return False
    try:
        config_path = root / "resolved_config.yaml"
        manifest = yaml.safe_load(
            (root / "fit_manifest.yaml").read_text(encoding="utf-8")
        )
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        score_path = root / "cell_scores.parquet"
        score_frame = pd.read_parquet(score_path)
        artifacts = manifest["artifacts"]
        metadata = manifest["metadata"]
        if metadata["comparison_status"] != "pass":
            return False
        if resolved["fit_id"] != fit_id(row):
            return False
        if artifacts["resolved_config"]["sha256"] != sha256_file(config_path):
            return False
        score_record = artifacts["cell_scores"]
        if score_record["sha256"] != sha256_file(score_path):
            return False
        if int(score_record["bytes"]) != score_path.stat().st_size:
            return False
        if int(score_record["rows"]) != len(score_frame):
            return False
        if list(score_record["columns"]) != list(score_frame.columns):
            return False
        if tuple(score_frame.columns) != SCORE_COLUMNS:
            return False
        if metadata["code_sha256"] != code_hashes():
            return False
        for record in resolved["inputs"].values():
            path = REPO / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                return False
            if path.stat().st_size != int(record["bytes"]):
                return False
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
        return False
    return True


def reconstruct_one(
    *,
    row: object,
    runs_root: Path,
    condition_map: pd.Series,
    pbmc_raw_path: Path,
) -> dict[str, object]:
    root = fit_root(row)
    if existing_fit_is_valid(root, row):
        manifest = yaml.safe_load((root / "fit_manifest.yaml").read_text(encoding="utf-8"))
        return dict(manifest["metadata"]["status_row"])

    paths, candidates, source_priors, target_priors, truth = prepared_inputs(
        row=row, runs_root=runs_root
    )
    hashes = verify_input_hashes(row, paths)
    raw_path = pbmc_raw_path.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    paths["pbmc_raw_h5ad"] = raw_path
    hashes["pbmc_raw_h5ad"] = sha256_file(raw_path)
    config = method_config(row)
    current_code_hashes = code_hashes()
    current_environment = environment_metadata()

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pbmc_compatibility_fit_", dir=TEMP_ROOT
    ) as directory:
        temporary = Path(directory)
        (temporary / "method_params.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )
        started = time.perf_counter()
        _run_coreot_constant_tau(
            temporary,
            CONDITION,
            candidates,
            source_priors,
            target_priors,
            config,
        )
        runtime = time.perf_counter() - started
        scores = pd.read_parquet(temporary / "cell_transport_scores.parquet")
        transport_manifest = yaml.safe_load(
            (temporary / "transport_manifest.yaml").read_text(encoding="utf-8")
        )
        metrics = evaluate_scores(
            experiment="pbmc",
            endpoint=str(row.endpoint),
            truth=truth,
            scores=scores,
            pbmc_condition=condition_map,
        )
        metadata = transport_manifest["metadata"]
        differences = {
            metric: abs(float(metrics[metric]) - float(getattr(row, metric)))
            for metric in METRICS
        }
        comparison_status = "pass" if max(differences.values()) <= TOLERANCE else "fail"
        if bool(metadata["converged"]) != bool(row.converged):
            comparison_status = "fail"
        if int(metadata["n_iter"]) != int(row.n_iterations):
            comparison_status = "fail"
        if comparison_status != "pass":
            raise RuntimeError(
                f"Reconstructed fit differs for {fit_id(row)}: "
                f"n_iter={metadata['n_iter']}/{row.n_iterations}, differences={differences}"
            )

        root.mkdir(parents=True, exist_ok=True)
        score_path = root / "cell_scores.parquet"
        compact_scores = scores.loc[:, list(SCORE_COLUMNS)].copy()
        compact_scores.to_parquet(score_path, index=False)
        resolved = {
            "fit_id": fit_id(row),
            "experiment": "pbmc",
            "endpoint": str(row.endpoint),
            "seed": int(row.seed),
            "base_run_id": str(row.run_id),
            "condition": CONDITION,
            "candidate_set": "pca30_k100",
            "variant": "compatibility_only",
            "method": config,
            "inputs": {
                name: {
                    "path": str(path.resolve().relative_to(REPO)),
                    "sha256": hashes[name],
                    "bytes": path.stat().st_size,
                }
                for name, path in paths.items()
            },
            "combined_source_priors_sha256": hashes["combined_source_priors"],
            "code_sha256": current_code_hashes,
            "environment": current_environment,
            "evaluation_scope": "within_celltype",
        }
        config_path = root / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        status_row = {
            "fit_id": fit_id(row),
            "endpoint": str(row.endpoint),
            "seed": int(row.seed),
            "tau_source": float(row.tau_source),
            "alpha": float(row.alpha),
            "converged": bool(metadata["converged"]),
            "n_iterations": int(metadata["n_iter"]),
            "historical_runtime_seconds": float(row.runtime_seconds),
            "reconstruction_runtime_seconds": runtime,
            "max_metric_abs_difference": max(differences.values()),
            "comparison_status": comparison_status,
            "cell_scores_sha256": sha256_file(score_path),
            "cell_scores_bytes": score_path.stat().st_size,
            "cell_scores_rows": len(compact_scores),
            }
        manifest = {
            "stage": "isolated-component-fit-reconstruction",
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
                },
            },
            "metadata": {
                "comparison_status": comparison_status,
                "tolerance": TOLERANCE,
                "historical_metrics": {
                    metric: float(getattr(row, metric)) for metric in METRICS
                },
                "reconstructed_metrics": {
                    metric: float(metrics[metric]) for metric in METRICS
                },
                "metric_abs_differences": differences,
                "transport_metadata": metadata,
                "temporary_sparse_coupling_retained": False,
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
    frame = frame.sort_values(["endpoint", "seed", "tau_source", "alpha"])
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(STATUS, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--pbmc-raw-path", type=Path, default=Path("data/raw/kang_2018.h5ad")
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fit-id", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows()
    if args.fit_id:
        requested = set(args.fit_id)
        rows = rows.loc[[fit_id(row) in requested for row in rows.itertuples(index=False)]]
    if args.limit is not None:
        rows = rows.head(args.limit)
    condition_map = _pbmc_condition_map(args.pbmc_raw_path)
    status_rows = (
        pd.read_csv(STATUS).to_dict("records") if STATUS.is_file() else []
    )
    failures = 0
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        try:
            status_row = reconstruct_one(
                row=row,
                runs_root=args.runs_root,
                condition_map=condition_map,
                pbmc_raw_path=args.pbmc_raw_path,
            )
        except Exception as error:  # continue independent fits unattended
            failures += 1
            status_row = {
                "fit_id": fit_id(row),
                "endpoint": str(row.endpoint),
                "seed": int(row.seed),
                "tau_source": float(row.tau_source),
                "alpha": float(row.alpha),
                "historical_runtime_seconds": float(row.runtime_seconds),
                "comparison_status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        status_rows.append(status_row)
        write_status(status_rows)
        print(
            f"completed {index}/{len(rows)} {fit_id(row)} "
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
