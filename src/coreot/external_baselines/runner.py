from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as scipy_io
from scipy import sparse

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.benchmarks.missing_state import run_missing_state_benchmark_build
from coreot.config.load import load_yaml
from coreot.data.schemas import ABSTENTION_CALLS_COLUMNS, CELL_SCORES_COLUMNS, TARGET_LABELS_COLUMNS
from coreot.data.validation import validate_required_columns, validate_stage_can_read
from coreot.data.model_visible import read_condition_adata
from coreot.data.raw_import import run_raw_import
from coreot.external_baselines.schema import (
    DEFAULT_EXTERNAL_BASELINE_METHODS,
    EXTERNAL_BASELINE_METHODS,
    EXTERNAL_BASELINE_PREDICTION_COLUMNS,
    validate_external_baseline_predictions,
)

STAGE = "external-baseline"
DEFAULT_CANDIDATE_SET = "external_reference_mapping"


class ExternalBaselineError(ValueError):
    """Raised when external baseline artifacts cannot be generated."""


@dataclass(frozen=True)
class ExternalBaselineResult:
    run_root: Path
    conditions: tuple[str, ...]
    candidate_set: str
    methods: tuple[str, ...]


def run_external_baselines(
    config_path: str | Path,
    *,
    force: bool | None = None,
    keep_intermediates: bool | None = None,
) -> ExternalBaselineResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    runs_root = _required_path(config, ("outputs", "root"))
    run_root = runs_root / run_id
    conditions = tuple(str(c) for c in config.get("conditions", ()))
    methods = tuple(str(m) for m in config.get("methods", DEFAULT_EXTERNAL_BASELINE_METHODS))
    candidate_set = str(config.get("candidate_set", DEFAULT_CANDIDATE_SET))
    if not conditions:
        raise ExternalBaselineError("external baselines require conditions")
    _validate_methods(methods)

    heldout_label = str(config.get("heldout_label") or _read_heldout_label(run_root))
    repeat = int(config.get("repeat") or _read_split_seed(run_root))
    scripts_dir = Path(
        config.get("r_scripts_dir", "experiments/missing_celltype/external_baselines")
    )
    threshold_quantile = float(config.get("threshold_quantile", 0.95))
    threshold_mode = str(config.get("threshold_mode", "full_reference_quantile"))
    force = bool(config.get("force", False)) if force is None else force
    keep_intermediates = (
        bool(config.get("keep_intermediate_artifacts", False))
        if keep_intermediates is None
        else keep_intermediates
    )
    chetah_config = _chetah_config(config)
    celltypist_parameters = _celltypist_config(config, repeat=repeat)
    shared_counts_source = _optional_path(config, "model_visible_counts_source")

    methods_to_run = methods if force else _methods_missing_from_scores(
        run_root=run_root,
        conditions=conditions,
        candidate_set=candidate_set,
        methods=methods,
    )
    if not methods_to_run:
        return ExternalBaselineResult(
            run_root=run_root,
            conditions=conditions,
            candidate_set=candidate_set,
            methods=methods,
        )

    _restore_external_inputs_if_needed(
        config=config,
        run_root=run_root,
        conditions=conditions,
        shared_counts_source=shared_counts_source,
    )

    for condition in conditions:
        is_full_reference = condition == "full_reference_control"
        for method in methods_to_run:
            _ensure_model_visible_counts_link(
                run_root=run_root,
                condition=condition,
                source=shared_counts_source,
            )
            output_path = _prediction_path(run_root, condition, method)
            if output_path.is_file() and not force:
                try:
                    _read_predictions(run_root, condition, method)
                    continue
                except (ValueError, FileNotFoundError):
                    pass
            if method == "celltypist_l3":
                frame = run_celltypist_l3(
                    run_root=run_root,
                    condition=condition,
                    heldout_label=heldout_label,
                    repeat=repeat,
                    is_full_reference_control=is_full_reference,
                    training_parameters=celltypist_parameters,
                )
                _write_predictions(
                    run_root,
                    condition,
                    method,
                    frame,
                    parameters=celltypist_parameters,
                )
            else:
                _run_r_baseline(
                    run_root=run_root,
                    condition=condition,
                    method=method,
                    heldout_label=heldout_label,
                    repeat=repeat,
                    is_full_reference_control=is_full_reference,
                    scripts_dir=scripts_dir,
                    keep_intermediates=keep_intermediates,
                    chetah_max_reference_per_label=chetah_config[
                        "max_reference_per_label"
                    ],
                )

    write_external_baseline_cell_scores(
        run_root=run_root,
        conditions=conditions,
        candidate_set=candidate_set,
        methods=methods_to_run,
        threshold_quantile=threshold_quantile,
        threshold_mode=threshold_mode,
        method_parameters={
            **(
                {"celltypist_l3": celltypist_parameters}
                if "celltypist_l3" in methods_to_run
                else {}
            ),
            "chetah": {
                "native_threshold": 0.1,
                "forced_threshold": 0.0,
                "n_genes": 200,
                "correlation": "spearman",
                "max_reference_per_label": chetah_config[
                    "max_reference_per_label"
                ],
            },
            "scmap_cluster": {
                "n_features": 500,
                "native_threshold": 0.7,
                "forced_rule": "maximum_cosine_cluster_centroid_similarity",
            },
        },
    )
    if not keep_intermediates:
        _cleanup_external_baseline_artifacts(run_root, conditions, methods_to_run)
    return ExternalBaselineResult(
        run_root=run_root,
        conditions=conditions,
        candidate_set=candidate_set,
        methods=methods,
    )


def write_external_baseline_cell_scores(
    *,
    run_root: Path,
    conditions: tuple[str, ...],
    candidate_set: str = DEFAULT_CANDIDATE_SET,
    methods: tuple[str, ...] = DEFAULT_EXTERNAL_BASELINE_METHODS,
    threshold_quantile: float = 0.95,
    threshold_mode: str = "full_reference_quantile",
    method_parameters: dict[str, dict[str, object]] | None = None,
) -> None:
    _validate_methods(methods)
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ExternalBaselineError("threshold_quantile must lie in [0, 1]")
    if threshold_mode == "full_reference_quantile":
        thresholds = {
            method: _calibrate_threshold(run_root, method, threshold_quantile)
            for method in methods
        }
        inherited_thresholds = _thresholds_from_existing_full_reference_scores(
            run_root=run_root,
            candidate_set=candidate_set,
            replacing_methods=methods,
            quantile=threshold_quantile,
        )
    elif threshold_mode == "none":
        thresholds = {method: float("inf") for method in methods}
        inherited_thresholds = {}
    else:
        raise ExternalBaselineError(
            "threshold_mode must be 'full_reference_quantile' or 'none'"
        )
    for condition in conditions:
        frames = []
        for method in methods:
            predictions = _read_predictions(run_root, condition, method)
            frames.append(_predictions_to_cell_scores(predictions, condition, thresholds[method]))
        new_scores = pd.concat(frames, ignore_index=True)
        output_root = run_root / "scoring" / condition / candidate_set
        scores_path = output_root / "cell_scores.parquet"
        manifest_path = output_root / "scoring_manifest.yaml"
        previous_thresholds: dict[str, float] = {}
        previous_method_parameters: dict[str, object] = {}
        if manifest_path.is_file():
            previous_manifest = load_yaml(manifest_path)
            previous_metadata = previous_manifest.get("metadata", {})
            if isinstance(previous_metadata, dict):
                raw_thresholds = previous_metadata.get("theta_by_method", {})
                if isinstance(raw_thresholds, dict):
                    previous_thresholds = {
                        str(key): float(value) for key, value in raw_thresholds.items()
                    }
                raw_parameters = previous_metadata.get("method_parameters", {})
                if isinstance(raw_parameters, dict):
                    previous_method_parameters = dict(raw_parameters)
        if scores_path.is_file():
            existing = pd.read_parquet(scores_path)
            validate_required_columns(existing, CELL_SCORES_COLUMNS, str(scores_path))
            existing = existing.loc[~existing["method"].astype(str).isin(methods)]
            cell_scores = pd.concat([existing, new_scores], ignore_index=True)
        else:
            cell_scores = new_scores
        cell_scores = cell_scores.loc[:, CELL_SCORES_COLUMNS]

        output_root.mkdir(parents=True, exist_ok=True)
        cell_scores.to_parquet(output_root / "cell_scores.parquet", index=False)
        cell_scores.loc[:, ABSTENTION_CALLS_COLUMNS].to_csv(
            output_root / "abstention_calls.csv", index=False
        )
        np.savez_compressed(output_root / "label_probabilities.npz")
        write_manifest(
            manifest_path,
            Manifest(
                stage=STAGE,
                artifacts={
                    "cell_scores": str(output_root / "cell_scores.parquet"),
                    "abstention_calls": str(output_root / "abstention_calls.csv"),
                    "label_probabilities": str(output_root / "label_probabilities.npz"),
                },
                metadata={
                    "candidate_set": candidate_set,
                    "condition": condition,
                    "methods": list(dict.fromkeys(cell_scores["method"].astype(str))),
                    "score_mapping": "u stores the method-specific z_absent_score",
                    "threshold_mode": threshold_mode,
                    "threshold_quantile": (
                        threshold_quantile
                        if threshold_mode == "full_reference_quantile"
                        else None
                    ),
                    "theta_by_method": {
                        **inherited_thresholds,
                        **previous_thresholds,
                        **thresholds,
                    },
                    "method_parameters": {
                        **previous_method_parameters,
                        **(method_parameters or {}),
                    },
                },
            ),
        )


def run_celltypist_l3(
    *,
    run_root: Path,
    condition: str,
    heldout_label: str,
    repeat: int,
    is_full_reference_control: bool,
    training_parameters: dict[str, object] | None = None,
) -> pd.DataFrame:
    import celltypist

    ref, qry, labels = _load_reference_query_adata(run_root, condition)
    ref = _prepare_celltypist_adata(ref)
    qry = _prepare_celltypist_adata(qry)
    common = ref.var_names.intersection(qry.var_names)
    if len(common) == 0:
        raise ExternalBaselineError("celltypist_l3 requires at least one shared gene")
    ref = ref[:, common].copy()
    qry = qry[:, common].copy()
    ref.obs["AIFI_L3"] = labels.loc[ref.obs_names, "target_label"].astype(str)

    parameters = training_parameters or _celltypist_config({}, repeat=repeat)
    profile = str(parameters["profile"])
    random_state = int(parameters["random_state"])
    train_kwargs = {
        key: parameters[key]
        for key in (
            "use_SGD",
            "with_mean",
            "mini_batch",
            "batch_number",
            "batch_size",
            "epochs",
            "max_iter",
            "n_jobs",
            "random_state",
        )
        if key in parameters
    }

    numpy_random_state = np.random.get_state()
    np.random.seed(random_state)
    try:
        model = celltypist.train(
            ref,
            labels="AIFI_L3",
            **train_kwargs,
        )
    finally:
        np.random.set_state(numpy_random_state)
    if profile == "centered_full_sgd":
        pred = celltypist.annotate(
            qry,
            model=model,
            mode=str(parameters.get("annotation_mode", "best match")),
            majority_voting=bool(parameters.get("majority_voting", False)),
        )
        probabilities = pred.probability_matrix.copy().loc[qry.obs_names]
    else:
        if "annotation_mode" in parameters or "probability_threshold" in parameters:
            probabilities = _celltypist_sparse_probabilities(
                qry,
                model,
                mode=str(parameters.get("annotation_mode", "best match")),
                p_thres=float(parameters.get("probability_threshold", 0.5)),
            )
        else:
            # Keep the existing helper call shape for integrations that patch
            # the sparse probability extractor.
            probabilities = _celltypist_sparse_probabilities(qry, model)
    max_score = probabilities.max(axis=1)
    return pd.DataFrame(
        {
            "cell_id": qry.obs_names.astype(str),
            "method": "celltypist_l3",
            "heldout_label": heldout_label,
            "repeat": repeat,
            "is_full_reference_control": is_full_reference_control,
            "pred_label": probabilities.idxmax(axis=1).astype(str).to_numpy(),
            "native_pred_label": probabilities.idxmax(axis=1).astype(str).to_numpy(),
            "z_absent_score": (1.0 - max_score).astype(float).to_numpy(),
            "native_abstain": False,
            "max_confidence_or_similarity": max_score.astype(float).to_numpy(),
        }
    )


def _celltypist_sparse_probabilities(
    qry: ad.AnnData,
    model: Any,
    *,
    mode: str = "best match",
    p_thres: float = 0.5,
) -> pd.DataFrame:
    model_features = pd.Index(np.asarray(model.classifier.features).astype(str))
    query_features = pd.Index(qry.var_names.astype(str))
    query_indices = query_features.get_indexer(model_features)
    if (query_indices < 0).any():
        missing = model_features[query_indices < 0].tolist()
        raise ExternalBaselineError(
            f"CellTypist query is missing trained feature(s): {missing[:5]}"
        )

    query_matrix = sparse.csr_matrix(qry.X[:, query_indices])
    scaled = sparse.csr_matrix(model.scaler.transform(query_matrix))
    scaled.data[scaled.data > 10] = 10
    _, probability_matrix, _ = model.predict_labels_and_prob(
        scaled,
        mode=mode,
        p_thres=p_thres,
    )
    return pd.DataFrame(
        probability_matrix,
        index=qry.obs_names,
        columns=np.asarray(model.classifier.classes_).astype(str),
    )


def _run_r_baseline(
    *,
    run_root: Path,
    condition: str,
    method: str,
    heldout_label: str,
    repeat: int,
    is_full_reference_control: bool,
    scripts_dir: Path,
    keep_intermediates: bool = False,
    chetah_max_reference_per_label: int | None = None,
) -> None:
    script_name = {
        "seurat_anchor": "run_seurat_anchor.R",
        "singleR": "run_singleR.R",
        "scmap_cell": "run_scmap_cell.R",
        "scmap_cluster": "run_scmap_cluster.R",
        "chetah": "run_chetah.R",
    }[method]
    script = scripts_dir / script_name
    if not script.is_file():
        raise FileNotFoundError(f"External baseline R script does not exist: {script}")
    if shutil.which("Rscript") is None:
        raise ExternalBaselineError(f"{method} requires Rscript on PATH")
    method_root = run_root / "external_baselines" / condition / method
    input_dir = method_root / "r_input"
    _write_r_input(
        run_root,
        condition,
        input_dir,
        max_reference_per_label=(
            chetah_max_reference_per_label if method == "chetah" else None
        ),
        repeat=repeat,
    )
    output_path = _prediction_path(run_root, condition, method)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "Rscript",
        str(script),
        "--input-dir",
        str(input_dir),
        "--output",
        str(output_path),
        "--method",
        method,
        "--heldout-label",
        heldout_label,
        "--repeat",
        str(repeat),
        "--is-full-reference-control",
        "true" if is_full_reference_control else "false",
    ]
    method_parameters: dict[str, object] = {}
    if method == "chetah" and chetah_max_reference_per_label is not None:
        cmd.extend(
            ["--max-reference-per-label", str(chetah_max_reference_per_label)]
        )
        method_parameters["max_reference_per_label"] = chetah_max_reference_per_label
    try:
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise ExternalBaselineError(
                f"{method} R runner failed with exit code {completed.returncode}"
            )
        frame = pd.read_csv(output_path)
        validate_external_baseline_predictions(
            frame, method=method, source=str(output_path)
        )
        _write_method_manifest(
            run_root,
            condition,
            method,
            output_path,
            parameters=method_parameters,
        )
    finally:
        if not keep_intermediates:
            shutil.rmtree(input_dir, ignore_errors=True)


def _write_r_input(
    run_root: Path,
    condition: str,
    input_dir: Path,
    *,
    max_reference_per_label: int | None = None,
    repeat: int = 0,
) -> None:
    ref, qry, labels = _load_reference_query_adata(run_root, condition)
    if max_reference_per_label is not None:
        ref = _subsample_reference_by_label(
            ref,
            labels,
            maximum=max_reference_per_label,
            repeat=repeat,
        )
    input_dir.mkdir(parents=True, exist_ok=True)
    ref_x = sparse.csr_matrix(_counts_matrix(ref)).transpose().tocsr()
    qry_x = sparse.csr_matrix(_counts_matrix(qry)).transpose().tocsr()
    scipy_io.mmwrite(input_dir / "ref_counts.mtx", ref_x)
    scipy_io.mmwrite(input_dir / "query_counts.mtx", qry_x)
    pd.Series(ref.var_names.astype(str)).to_csv(
        input_dir / "ref_genes.tsv", index=False, header=False
    )
    pd.Series(qry.var_names.astype(str)).to_csv(
        input_dir / "query_genes.tsv", index=False, header=False
    )
    labels.loc[ref.obs_names, ["target_label"]].rename(columns={"target_label": "AIFI_L3"}).assign(
        cell_id=ref.obs_names.astype(str)
    ).loc[:, ["cell_id", "AIFI_L3"]].to_csv(input_dir / "ref_meta.csv", index=False)
    pd.DataFrame({"cell_id": qry.obs_names.astype(str)}).to_csv(
        input_dir / "query_meta.csv", index=False
    )


def _subsample_reference_by_label(
    ref: ad.AnnData,
    labels: pd.DataFrame,
    *,
    maximum: int,
    repeat: int,
) -> ad.AnnData:
    selected: list[str] = []
    ref_labels = labels.loc[ref.obs_names, "target_label"].astype(str)
    for label in sorted(ref_labels.unique()):
        cell_ids = ref_labels.index[ref_labels.eq(label)].astype(str).tolist()
        if len(cell_ids) > maximum:
            cell_ids = sorted(
                cell_ids,
                key=lambda cell_id: hashlib.sha256(
                    f"{repeat}:{label}:{cell_id}".encode()
                ).digest(),
            )[:maximum]
        selected.extend(cell_ids)
    return ref[selected].copy()


def _load_reference_query_adata(
    run_root: Path, condition: str
) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    model_visible = run_root / "benchmark" / condition / "model_visible"
    counts_path = model_visible / "counts.h5ad"
    cells_path = model_visible / "cells.csv"
    labels_path = model_visible / "target_labels.csv"
    validate_stage_can_read(counts_path, STAGE)
    validate_stage_can_read(labels_path, STAGE)
    if not counts_path.is_file():
        raise FileNotFoundError(f"Model-visible counts do not exist: {counts_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Target labels do not exist: {labels_path}")
    if cells_path.is_file():
        cells = pd.read_csv(cells_path)
        adata = read_condition_adata(counts_path, cells)
    else:
        # Backward compatibility for dedicated condition matrices written before
        # shared matrix manifests were introduced.
        adata = ad.read_h5ad(counts_path)
    if "domain" not in adata.obs.columns:
        raise ExternalBaselineError(f"{counts_path} is missing .obs['domain']")
    _attach_counts_layer(adata, counts_path)
    labels = pd.read_csv(labels_path)
    validate_required_columns(labels, TARGET_LABELS_COLUMNS, str(labels_path))
    labels = labels.assign(cell_id=labels["cell_id"].astype(str)).set_index("cell_id", drop=False)
    adata.obs_names = adata.obs["cell_id"].astype(str)
    ref = adata[adata.obs["domain"].astype(str).eq("reference")].copy()
    qry = adata[adata.obs["domain"].astype(str).eq("query")].copy()
    missing = set(ref.obs_names.astype(str)) - set(labels.index.astype(str))
    if missing:
        raise ExternalBaselineError(
            f"Missing target labels for reference cells: {sorted(missing)[:5]}"
        )
    if ref.n_obs == 0 or qry.n_obs == 0:
        raise ExternalBaselineError(f"{condition} requires nonempty query and reference sets")
    return ref, qry, labels


def _ensure_model_visible_counts_link(
    *, run_root: Path, condition: str, source: Path | None
) -> None:
    if source is None:
        return
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Shared model-visible counts do not exist: {source}")
    destination = run_root / "benchmark" / condition / "model_visible" / "counts.h5ad"
    if destination.is_file():
        return
    if destination.is_symlink():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def _restore_external_inputs_if_needed(
    *,
    config: dict[str, Any],
    run_root: Path,
    conditions: tuple[str, ...],
    shared_counts_source: Path | None,
) -> None:
    if shared_counts_source is not None:
        return
    counts_paths = tuple(
        run_root / "benchmark" / condition / "model_visible" / "counts.h5ad"
        for condition in conditions
    )
    if all(path.is_file() for path in counts_paths):
        return
    restore = config.get("input_restore")
    if restore is None:
        return
    if not isinstance(restore, dict):
        raise ExternalBaselineError("input_restore config must be a mapping")
    raw_config = Path(_mapping_required_str(restore, "raw_import_config", "input_restore"))
    benchmark_config = Path(
        _mapping_required_str(restore, "benchmark_config", "input_restore")
    )
    if not (run_root / "raw" / "input.h5ad").is_file():
        run_raw_import(raw_config)
    run_missing_state_benchmark_build(benchmark_config)
    missing = [str(path) for path in counts_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Input restoration did not create model-visible counts: {missing}"
        )


def _prepare_celltypist_adata(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    out.X = _counts_matrix(out).copy()
    out.uns.pop("log1p", None)
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def _attach_counts_layer(adata: ad.AnnData, source: Path) -> None:
    if "counts" in adata.layers:
        return
    if adata.raw is not None:
        try:
            raw_counts = adata.raw[:, adata.var_names].X
        except (KeyError, ValueError) as exc:
            raise ExternalBaselineError(
                f"{source} has .raw, but raw genes cannot be aligned to .var_names"
            ) from exc
        if raw_counts.shape != adata.X.shape:
            raise ExternalBaselineError(
                f"{source} raw-count shape {raw_counts.shape} does not match .X shape {adata.X.shape}"
            )
        if not _looks_like_count_matrix(raw_counts):
            raise ExternalBaselineError(
                f"{source} .raw.X does not look like a nonnegative integer count matrix"
            )
        adata.layers["counts"] = sparse.csr_matrix(raw_counts).copy()
        return
    if _looks_like_count_matrix(adata.X):
        adata.layers["counts"] = sparse.csr_matrix(adata.X).copy()
        return
    raise ExternalBaselineError(
        f"{source} does not contain a usable raw-count matrix. External count-based "
        "baselines require .layers['counts'], count-like .raw.X, or count-like .X."
    )


def _counts_matrix(adata: ad.AnnData) -> Any:
    if "counts" not in adata.layers:
        raise ExternalBaselineError("AnnData is missing required .layers['counts']")
    return adata.layers["counts"]


def _looks_like_count_matrix(matrix: Any) -> bool:
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if data.size == 0:
        return True
    data = np.asarray(data)
    if not np.isfinite(data).all():
        return False
    nonzero = data[data != 0]
    if nonzero.size == 0:
        return True
    sample = nonzero[: min(nonzero.size, 100_000)]
    return bool((sample >= 0).all() and np.allclose(sample, np.round(sample)))


def _predictions_to_cell_scores(
    predictions: pd.DataFrame, condition: str, theta: float
) -> pd.DataFrame:
    z = predictions["z_absent_score"].astype(float)
    abstain = z > theta
    forced_label = predictions["pred_label"].fillna("").astype(str)
    return pd.DataFrame(
        {
            "cell_id": predictions["cell_id"].astype(str),
            "condition_id": condition,
            "method": predictions["method"].astype(str),
            "u": z,
            "u_tilde": np.nan,
            "prior_risk": np.nan,
            "e": np.nan,
            "hub_exposure": np.nan,
            "max_label_probability": predictions["max_confidence_or_similarity"].astype(float),
            "label_uncertainty": z,
            "label_entropy": np.nan,
            "forced_label": forced_label,
            "nn_distance": np.nan,
            "abstain_u": abstain,
            "abstain_u_or_entropy": abstain,
            "final_label_abstention_aware": forced_label.mask(abstain, ""),
        }
    )


def _calibrate_threshold(run_root: Path, method: str, quantile: float) -> float:
    predictions = _read_predictions(run_root, "full_reference_control", method)
    return float(np.nanquantile(predictions["z_absent_score"].astype(float).to_numpy(), quantile))


def _thresholds_from_existing_full_reference_scores(
    *,
    run_root: Path,
    candidate_set: str,
    replacing_methods: tuple[str, ...],
    quantile: float,
) -> dict[str, float]:
    path = (
        run_root
        / "scoring"
        / "full_reference_control"
        / candidate_set
        / "cell_scores.parquet"
    )
    if not path.is_file():
        return {}
    scores = pd.read_parquet(path)
    validate_required_columns(scores, CELL_SCORES_COLUMNS, str(path))
    retained = scores.loc[~scores["method"].astype(str).isin(replacing_methods)]
    return {
        str(method): float(np.nanquantile(group["u"].astype(float), quantile))
        for method, group in retained.groupby("method", sort=False)
    }


def _prediction_path(run_root: Path, condition: str, method: str) -> Path:
    return run_root / "external_baselines" / condition / method / "predictions.csv"


def _read_predictions(run_root: Path, condition: str, method: str) -> pd.DataFrame:
    path = _prediction_path(run_root, condition, method)
    validate_stage_can_read(path, STAGE)
    if not path.is_file():
        raise FileNotFoundError(f"External baseline predictions do not exist: {path}")
    return validate_external_baseline_predictions(
        pd.read_csv(path), method=method, source=str(path)
    )


def _external_baseline_scores_are_complete(
    *,
    run_root: Path,
    conditions: tuple[str, ...],
    candidate_set: str,
    methods: tuple[str, ...],
) -> bool:
    expected_methods = set(methods)
    for condition in conditions:
        output_root = run_root / "scoring" / condition / candidate_set
        scores_path = output_root / "cell_scores.parquet"
        abstention_path = output_root / "abstention_calls.csv"
        if not scores_path.is_file() or not abstention_path.is_file():
            return False
        try:
            scores = pd.read_parquet(scores_path)
            validate_required_columns(scores, CELL_SCORES_COLUMNS, str(scores_path))
            if not expected_methods.issubset(set(scores["method"].astype(str))):
                return False
        except Exception:
            return False
    return True


def _methods_missing_from_scores(
    *,
    run_root: Path,
    conditions: tuple[str, ...],
    candidate_set: str,
    methods: tuple[str, ...],
) -> tuple[str, ...]:
    missing = []
    for method in methods:
        if not _external_baseline_scores_are_complete(
            run_root=run_root,
            conditions=conditions,
            candidate_set=candidate_set,
            methods=(method,),
        ):
            missing.append(method)
    return tuple(missing)


def _cleanup_external_baseline_artifacts(
    run_root: Path, conditions: tuple[str, ...], methods: tuple[str, ...]
) -> None:
    _cleanup_external_baseline_input_h5ad(run_root, conditions)
    for condition in conditions:
        for method in methods:
            method_root = run_root / "external_baselines" / condition / method
            if not method_root.exists():
                continue
            shutil.rmtree(method_root / "r_input", ignore_errors=True)
            shutil.rmtree(method_root / "model", ignore_errors=True)
            for filename in ("predictions.csv", "external_baseline_manifest.yaml"):
                path = method_root / filename
                if path.exists():
                    path.unlink()
            _remove_empty_parents(
                method_root,
                stop_at=run_root / "external_baselines",
            )


def _cleanup_external_baseline_input_h5ad(run_root: Path, conditions: tuple[str, ...]) -> None:
    for path in (
        run_root / "raw" / "input.h5ad",
        run_root / "benchmark" / "shared" / "model_visible" / "counts.h5ad",
        *(
            run_root / "benchmark" / condition / "model_visible" / "counts.h5ad"
            for condition in conditions
        ),
    ):
        if path.exists():
            path.unlink()


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    stop_at = stop_at.resolve()
    while True:
        try:
            current.rmdir()
        except OSError:
            return
        if current.resolve() == stop_at:
            return
        current = current.parent


def _write_predictions(
    run_root: Path,
    condition: str,
    method: str,
    frame: pd.DataFrame,
    *,
    parameters: dict[str, object] | None = None,
) -> None:
    path = _prediction_path(run_root, condition, method)
    validated = validate_external_baseline_predictions(frame, method=method, source=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False, columns=EXTERNAL_BASELINE_PREDICTION_COLUMNS)
    _write_method_manifest(
        run_root,
        condition,
        method,
        path,
        parameters=parameters,
    )


def _write_method_manifest(
    run_root: Path,
    condition: str,
    method: str,
    predictions_path: Path,
    *,
    parameters: dict[str, object] | None = None,
) -> None:
    method_root = run_root / "external_baselines" / condition / method
    write_manifest(
        method_root / "external_baseline_manifest.yaml",
        Manifest(
            stage=STAGE,
            artifacts={"predictions": str(predictions_path)},
            metadata={
                "condition": condition,
                "method": method,
                "prediction_columns": list(EXTERNAL_BASELINE_PREDICTION_COLUMNS),
                "parameters": parameters or {},
            },
        ),
    )


def _read_heldout_label(run_root: Path) -> str:
    path = run_root / "benchmark" / "condition_manifest.csv"
    validate_stage_can_read(path, STAGE)
    frame = pd.read_csv(path)
    if "removed_state" not in frame.columns or frame.empty:
        raise ExternalBaselineError(f"Cannot infer held-out label from {path}")
    return str(frame["removed_state"].iloc[0])


def _read_split_seed(run_root: Path) -> int:
    path = run_root / "benchmark" / "split_manifest.csv"
    validate_stage_can_read(path, STAGE)
    frame = pd.read_csv(path)
    if "split_seed" not in frame.columns or frame.empty:
        raise ExternalBaselineError(f"Cannot infer split seed from {path}")
    return int(frame["split_seed"].iloc[0])


def _validate_methods(methods: tuple[str, ...]) -> None:
    unknown = sorted(set(methods) - set(EXTERNAL_BASELINE_METHODS))
    if unknown:
        raise ExternalBaselineError(f"Unsupported external baseline method(s): {unknown}")


def _celltypist_config(config: dict[str, Any], *, repeat: int) -> dict[str, object]:
    raw = config.get("celltypist", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ExternalBaselineError("celltypist config must be a mapping")

    profile = str(raw.get("profile", "sparse_minibatch"))
    if profile not in {"sparse_minibatch", "centered_full_sgd"}:
        raise ExternalBaselineError(
            "celltypist.profile must be 'sparse_minibatch' or 'centered_full_sgd'"
        )
    random_state = int(raw.get("random_state", repeat))
    common: dict[str, object] = {
        "profile": profile,
        "random_state": random_state,
        "numpy_random_seed": random_state,
        "use_SGD": True,
    }
    if profile == "centered_full_sgd":
        return {
            **common,
            "with_mean": True,
            "mini_batch": False,
            "max_iter": 1000,
            "n_jobs": 1,
            "probability_path": "celltypist.annotate",
            "annotation_mode": "best match",
            "majority_voting": False,
        }
    resolved = {
        **common,
        "with_mean": False,
        "mini_batch": True,
        "batch_size": 1000,
        "epochs": 10,
        "n_jobs": -1,
        "probability_path": "sparse_model_probability",
    }
    # The historical sparse profile intentionally keeps its -1 worker default.
    # Provenance-certified callers may opt into a stricter single-worker
    # contract without changing unrelated external-baseline workflows.
    if "n_jobs" in raw:
        resolved["n_jobs"] = int(raw["n_jobs"])
    elif "worker_count" in raw:
        resolved["n_jobs"] = int(raw["worker_count"])
    for key in (
        "batch_number",
        "max_iter",
        "annotation_mode",
        "probability_threshold",
        "majority_voting",
    ):
        if key in raw:
            resolved[key] = raw[key]
    return resolved


def _chetah_config(config: dict[str, Any]) -> dict[str, int | None]:
    raw = config.get("chetah", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ExternalBaselineError("chetah config must be a mapping")
    value = raw.get("max_reference_per_label")
    if value is None:
        return {"max_reference_per_label": None}
    maximum = int(value)
    if maximum < 1:
        raise ExternalBaselineError("chetah.max_reference_per_label must be positive")
    return {"max_reference_per_label": maximum}


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        raise ExternalBaselineError(f"Expected nonempty string config value: {'.'.join(path)}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _optional_path(config: dict[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ExternalBaselineError(f"Expected nonempty string config value: {key}")
    return Path(value)


def _mapping_required_str(mapping: dict[str, Any], key: str, prefix: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ExternalBaselineError(
            f"Expected nonempty string config value: {prefix}.{key}"
        )
    return value


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ExternalBaselineError(f"Missing required config value: {'.'.join(path)}")
        current = current[key]
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run external reference-mapping baselines.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate predictions even when predictions.csv already exists.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep per-method predictions, R inputs, and saved model directories.",
    )
    args = parser.parse_args(argv)
    result = run_external_baselines(
        args.config,
        force=args.force,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"external-baseline: root={result.run_root / 'external_baselines'}")
    print(f"external-baseline: candidate_set={result.candidate_set}")
    print(f"external-baseline: methods={','.join(result.methods)}")
    return 0
