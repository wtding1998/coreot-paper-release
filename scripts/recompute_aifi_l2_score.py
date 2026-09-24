#!/usr/bin/env python3
"""Recompute AIFI Level-2 CellTypist prediction scores for a HIHA AnnData file."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import celltypist
from celltypist.models import Model
import numpy as np
import pandas as pd
import scanpy as sc


GENE_SYMBOL_COLUMNS = ("gene_name", "gene_symbols", "feature_name", "symbol")
PROVENANCE_KEY = "AIFI_L2_score_recomputed_provenance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute AIFI Level-2 CellTypist prediction scores."
    )
    parser.add_argument("--input-h5ad", required=True, type=Path)
    parser.add_argument("--model-pkl", required=True, type=Path)
    parser.add_argument("--output-h5ad", required=True, type=Path)
    parser.add_argument("--score-col", default="AIFI_L2_score_recomputed")
    parser.add_argument("--label-col", default="predicted_AIFI_L2_recomputed")
    parser.add_argument("--fill-missing-official-name", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-feature-overlap", default=1000, type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_features(model_path: Path) -> tuple[Model, pd.Index]:
    model = Model.load(str(model_path))
    features = getattr(model, "features", None)
    if features is None:
        classifier = getattr(model, "classifier", None)
        features = getattr(classifier, "features", None)
    if features is None:
        raise ValueError("Could not inspect required features from the CellTypist model.")
    return model, pd.Index(pd.Series(features, dtype="string").dropna().astype(str))


def set_best_gene_symbols(adata_ct: ad.AnnData, model_features: pd.Index) -> str:
    current_overlap = len(model_features.intersection(adata_ct.var_names.astype(str)))
    best_column = ""
    best_overlap = current_overlap

    for column in GENE_SYMBOL_COLUMNS:
        if column not in adata_ct.var:
            continue
        candidate = adata_ct.var[column].dropna().astype(str)
        if candidate.empty:
            continue
        overlap = len(model_features.intersection(pd.Index(candidate)))
        if overlap > best_overlap:
            best_column = column
            best_overlap = overlap

    if best_column:
        adata_ct.var_names = adata_ct.var[best_column].astype(str)
    adata_ct.var_names_make_unique()
    return best_column or "var_names"


def looks_log1p_normalized(adata_ct: ad.AnnData, n_cells: int = 1000, n_genes: int = 1000) -> bool:
    rows = min(n_cells, adata_ct.n_obs)
    cols = min(n_genes, adata_ct.n_vars)
    sample = adata_ct.X[:rows, :cols]
    arr = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    max_value = float(np.max(finite))
    has_fractional_values = bool(np.any(np.abs(finite - np.rint(finite)) > 1e-6))
    return max_value <= 30.0 and has_fractional_values


def prepare_for_celltypist(adata: ad.AnnData) -> tuple[ad.AnnData, str, bool, bool]:
    if adata.raw is not None:
        adata_ct = adata.raw.to_adata()
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
        return adata_ct, "raw.to_adata", True, True

    adata_ct = adata.copy()
    if looks_log1p_normalized(adata_ct):
        return adata_ct, "X", False, False

    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
    return adata_ct, "X", True, True


def predicted_label_series(labels: pd.DataFrame) -> pd.Series:
    if "predicted_labels" in labels.columns:
        pred_col = "predicted_labels"
    elif labels.shape[1] == 1:
        pred_col = labels.columns[0]
    else:
        raise ValueError(f"Cannot identify predicted-label column: {labels.columns.tolist()}")
    return labels[pred_col].astype(str)


def extract_scores(predicted_l2: pd.Series, probability_matrix: pd.DataFrame) -> np.ndarray:
    prob = probability_matrix.reindex(predicted_l2.index)
    if prob.isna().all(axis=None):
        raise ValueError("Could not align predicted labels with the probability matrix index.")

    col_indexer = prob.columns.get_indexer(predicted_l2)
    if (col_indexer < 0).any():
        missing = sorted(set(predicted_l2[col_indexer < 0]))
        raise ValueError(f"Predicted labels missing from probability matrix columns: {missing}")

    row_indexer = np.arange(prob.shape[0])
    scores = prob.to_numpy()[row_indexer, col_indexer].astype(float)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite AIFI Level-2 scores detected.")
    if scores.min() < 0 or scores.max() > 1:
        raise ValueError("AIFI Level-2 scores are outside [0, 1].")
    return scores


def report_path(output_h5ad: Path) -> Path:
    return output_h5ad.with_name(f"{output_h5ad.stem}.AIFI_L2_score_recomputed_report.tsv")


def write_report(
    path: Path,
    adata: ad.AnnData,
    original_obs: pd.DataFrame,
    label_col: str,
    score_col: str,
) -> None:
    report = pd.DataFrame(index=adata.obs_names)
    report.index.name = "obs_name"
    report[label_col] = adata.obs[label_col].astype(str)
    report[score_col] = adata.obs[score_col].astype(float)
    for column in ("AIFI_L2", "predicted_AIFI_L2", "AIFI_L2_score"):
        if column in original_obs:
            report[column] = original_obs[column].reindex(adata.obs_names)
    report.to_csv(path, sep="\t")


def summarize_qc(
    adata: ad.AnnData,
    original_obs: pd.DataFrame,
    label_col: str,
    score_col: str,
    used_raw: bool,
    n_feature_overlap: int,
) -> dict[str, Any]:
    scores = adata.obs[score_col].astype(float)
    summary: dict[str, Any] = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "used_raw": bool(used_raw),
        "n_feature_overlap": int(n_feature_overlap),
        "score_min": float(scores.min()),
        "score_median": float(scores.median()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "n_nonfinite_scores": int((~np.isfinite(scores.to_numpy())).sum()),
        "predicted_value_counts": adata.obs[label_col].astype(str).value_counts().to_dict(),
    }

    if "AIFI_L2" in original_obs:
        summary["final_aifi_l2_vs_recomputed"] = pd.crosstab(
            original_obs["AIFI_L2"], adata.obs[label_col]
        ).to_dict()
    if "predicted_AIFI_L2" in original_obs:
        original_pred = original_obs["predicted_AIFI_L2"].astype(str)
        recomputed_pred = adata.obs[label_col].astype(str)
        summary["predicted_aifi_l2_agreement"] = float((original_pred == recomputed_pred).mean())
    if "AIFI_L2_score" in original_obs:
        original_score = pd.to_numeric(original_obs["AIFI_L2_score"], errors="coerce")
        recomputed_score = scores.reindex(original_score.index)
        valid = original_score.notna() & recomputed_score.notna()
        if valid.any():
            summary["official_score_pearson"] = float(
                original_score[valid].corr(recomputed_score[valid], method="pearson")
            )
            summary["official_score_spearman"] = float(
                original_score[valid].corr(recomputed_score[valid], method="spearman")
            )
            summary["official_score_max_abs_diff"] = float(
                (original_score[valid] - recomputed_score[valid]).abs().max()
            )
    return summary


def print_qc(summary: dict[str, Any]) -> None:
    print("Quality-control summary")
    for key in (
        "n_obs",
        "n_vars",
        "used_raw",
        "n_feature_overlap",
        "score_min",
        "score_median",
        "score_mean",
        "score_max",
        "n_nonfinite_scores",
    ):
        print(f"{key}: {summary[key]}")
    print("predicted value counts:")
    for label, count in summary["predicted_value_counts"].items():
        print(f"  {label}: {count}")
    if "predicted_aifi_l2_agreement" in summary:
        print(f"predicted_AIFI_L2 agreement: {summary['predicted_aifi_l2_agreement']}")
    if "official_score_pearson" in summary:
        print(f"official score Pearson: {summary['official_score_pearson']}")
        print(f"official score Spearman: {summary['official_score_spearman']}")
        print(f"official score max abs diff: {summary['official_score_max_abs_diff']}")


def main() -> None:
    args = parse_args()
    input_h5ad = args.input_h5ad
    model_pkl = args.model_pkl
    output_h5ad = args.output_h5ad

    if input_h5ad.resolve() == output_h5ad.resolve():
        raise ValueError("Refusing to overwrite the input AnnData file.")
    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Input AnnData file not found: {input_h5ad}")
    if not model_pkl.is_file():
        raise FileNotFoundError(f"CellTypist model file not found: {model_pkl}")

    model, model_features = load_model_features(model_pkl)
    adata = sc.read_h5ad(input_h5ad)
    original_obs_names = adata.obs_names.copy()
    original_obs = adata.obs.copy()

    adata_ct, preprocessing_source, normalized, log1p_applied = prepare_for_celltypist(adata)
    if not np.array_equal(adata_ct.obs_names, original_obs_names):
        raise ValueError("CellTypist input does not preserve the original cell order.")

    gene_source = set_best_gene_symbols(adata_ct, model_features)
    n_feature_overlap = len(model_features.intersection(adata_ct.var_names.astype(str)))
    print(f"Feature overlap: {n_feature_overlap} / {len(model_features)}")
    print(f"Gene identifier source: {gene_source}")
    if n_feature_overlap < args.min_feature_overlap:
        raise ValueError(
            "Feature overlap is below --min-feature-overlap; gene identifiers probably "
            f"do not match the model ({n_feature_overlap} < {args.min_feature_overlap})."
        )

    predictions = celltypist.annotate(
        adata_ct,
        model=model,
        majority_voting=False,
    )
    labels = predictions.predicted_labels.copy()
    predicted_l2 = predicted_label_series(labels)
    if not predicted_l2.index.equals(adata_ct.obs_names):
        raise ValueError("Predicted-label index does not match CellTypist input obs_names.")

    scores = extract_scores(predicted_l2, predictions.probability_matrix)
    if not np.array_equal(adata_ct.obs_names, original_obs_names):
        raise ValueError("Cell order changed during prediction.")

    adata.obs[args.label_col] = pd.Categorical(
        pd.Series(predicted_l2.to_numpy(), index=original_obs_names)
    )
    adata.obs[args.score_col] = pd.Series(scores, index=original_obs_names).astype(float)

    created_columns = [args.label_col, args.score_col]
    if args.fill_missing_official_name:
        if "AIFI_L2_score" not in adata.obs or args.force:
            adata.obs["AIFI_L2_score"] = adata.obs[args.score_col]
            created_columns.append("AIFI_L2_score")
        if "predicted_AIFI_L2" not in adata.obs or args.force:
            adata.obs["predicted_AIFI_L2"] = adata.obs[args.label_col]
            created_columns.append("predicted_AIFI_L2")

    provenance = {
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "model_pkl": str(model_pkl),
        "model_sha256": sha256_file(model_pkl),
        "celltypist_version": celltypist.__version__,
        "scanpy_version": sc.__version__,
        "anndata_version": ad.__version__,
        "preprocessing_source": preprocessing_source,
        "normalized_total_target_sum": 10000 if normalized else None,
        "log1p_applied": bool(log1p_applied),
        "majority_voting": False,
        "n_cells": int(adata_ct.n_obs),
        "n_genes_input_to_celltypist": int(adata_ct.n_vars),
        "n_model_features": int(len(model_features)),
        "n_feature_overlap": int(n_feature_overlap),
        "score_definition": "probability_matrix[cell, predicted_AIFI_L2_recomputed]",
        "created_columns": created_columns,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    adata.uns[PROVENANCE_KEY] = provenance

    qc_summary = summarize_qc(
        adata=adata,
        original_obs=original_obs,
        label_col=args.label_col,
        score_col=args.score_col,
        used_raw=preprocessing_source == "raw.to_adata",
        n_feature_overlap=n_feature_overlap,
    )
    print_qc(qc_summary)
    adata.uns[f"{PROVENANCE_KEY}_qc"] = json.loads(json.dumps(qc_summary, default=str))

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    report = report_path(output_h5ad)
    adata.write_h5ad(output_h5ad)
    write_report(report, adata, original_obs, args.label_col, args.score_col)
    print(f"Wrote AnnData: {output_h5ad}")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
