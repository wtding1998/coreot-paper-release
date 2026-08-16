from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from coreot.artifacts.hashes import sha256_file


ENDPOINT = "Proliferating"
OTHER_RNA_ONLY_ENDPOINT = "Ifit B"
EXCLUSION_REASON = "other_rna_only_endpoint_not_in_proliferating_analysis"
EXPECTED_COHORT_COUNTS = {
    "processed": 4_382,
    "retained": 4_333,
    "excluded": 49,
    "positive": 62,
    "negative": 4_271,
}
N_BOOTSTRAP = 2_000
BOOTSTRAP_SEED = 20260713
MAIN_METHOD_ORDER = [
    "coreot_full",
    "prior_only",
    "uniform_uot",
    "nn",
    "seurat_anchor",
    "singleR",
    "celltypist_l3",
    "scmap_cell",
    "scmap_cluster",
    "chetah",
]
SUPPLEMENT_METHOD_ORDER = MAIN_METHOD_ORDER
# Backward-compatible name for the curated main-report method set.
METHOD_ORDER = MAIN_METHOD_ORDER
METHOD_LABELS = {
    "coreot_full": "CoRe-OT",
    "coreot_constant_tau": "Constant-tau CoRe-OT",
    "coreot_match_only": "CoRe-OT (−C)",
    "prior_only": "Prior only",
    "uniform_uot": "Uniform UOT",
    "nn": "Nearest neighbor",
    "seurat_anchor": "Seurat",
    "singleR": "SingleR",
    "celltypist_l3": "CellTypist",
    "scmap_cell": "scmap-cell",
    "scmap_cluster": "scmap-cluster",
    "chetah": "CHETAH",
}


def _relative_link(report_path: Path, source_path: Path) -> str:
    return os.path.relpath(source_path, start=report_path.parent)


def _metric_pair(truth: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(truth, score)),
        float(average_precision_score(truth, score)),
    )


def _load_detection_scores(
    *, run_root: Path, detection_path: Path
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    truth_path = run_root / "benchmark/natural_mismatch/evaluation_truth/query_truth.csv"
    truth = pd.read_csv(truth_path).sort_values("cell_id").reset_index(drop=True)
    if len(truth) != truth["cell_id"].nunique():
        raise ValueError("Proliferating truth contains duplicate cell IDs")
    positive = truth["true_label"].astype(str).eq(ENDPOINT).to_numpy()
    if positive.sum() == 0 or positive.all():
        raise ValueError("Proliferating truth must contain both classes")

    candidate_paths = {
        "mouse_spleen_provider_k100": run_root
        / "scoring/natural_mismatch/mouse_spleen_provider_k100/cell_scores.parquet",
        "external_reference_mapping": run_root
        / "scoring/natural_mismatch/external_reference_mapping/cell_scores.parquet",
    }
    candidate_frames = {
        candidate: pd.read_parquet(path) for candidate, path in candidate_paths.items()
    }
    detection = pd.read_csv(detection_path)
    detection = detection.loc[
        detection["holdout_label"].eq(ENDPOINT) & detection["method"].isin(SUPPLEMENT_METHOD_ORDER)
    ].copy()
    if set(detection["method"]) != set(SUPPLEMENT_METHOD_ORDER):
        missing = sorted(set(SUPPLEMENT_METHOD_ORDER) - set(detection["method"]))
        raise ValueError(f"Detection table omits report methods: {missing}")

    scores: dict[str, np.ndarray] = {}
    for method in SUPPLEMENT_METHOD_ORDER:
        row = detection.loc[detection["method"].eq(method)].iloc[0]
        frame = candidate_frames[str(row["candidate_set"])]
        method_frame = frame.loc[frame["method"].astype(str).eq(method)].copy()
        score_column = {"prior_only": "prior_risk"}.get(method, "u")
        joined = truth[["cell_id"]].merge(
            method_frame[["cell_id", score_column]],
            on="cell_id",
            how="left",
            validate="one_to_one",
        )
        if joined[score_column].isna().any():
            raise ValueError(f"Incomplete per-cell score for method={method}")
        scores[method] = joined[score_column].to_numpy(dtype=float)
        computed = _metric_pair(positive, scores[method])
        recorded = (float(row["auroc"]), float(row["auprc"]))
        if not np.allclose(computed, recorded, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Per-cell metrics disagree with regenerated table for method={method}"
            )
    return truth, scores, detection


def _build_processed_to_retained_cohort(
    *,
    embedding_cells: pd.DataFrame,
    proliferating_truth: pd.DataFrame,
    ifit_b_truth: pd.DataFrame,
    composition: pd.DataFrame,
    expected_n_rna: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    embedding_columns = {"row_index", "cell_id", "modality"}
    truth_columns = {"cell_id", "true_label"}
    composition_columns = {"fine_label", "rna_count"}
    if not embedding_columns.issubset(embedding_cells.columns):
        raise ValueError("Embedding cell inventory omits required columns")
    if not truth_columns.issubset(proliferating_truth.columns):
        raise ValueError("Proliferating truth omits required columns")
    if not truth_columns.issubset(ifit_b_truth.columns):
        raise ValueError("Ifit B truth omits required columns")
    if not composition_columns.issubset(composition.columns):
        raise ValueError("Cell-type composition omits required columns")

    if embedding_cells["row_index"].duplicated().any():
        raise ValueError("Embedding cell inventory contains duplicate row indices")
    if embedding_cells["cell_id"].duplicated().any():
        raise ValueError("Embedding cell inventory contains duplicate cell IDs")
    rna_cells = embedding_cells.loc[
        embedding_cells["modality"].astype(str).eq("rna"),
        ["row_index", "cell_id"],
    ].copy()
    if len(rna_cells) != expected_n_rna:
        raise ValueError(
            f"Embedding inventory has {len(rna_cells)} RNA cells; expected {expected_n_rna}"
        )

    truths: dict[str, pd.DataFrame] = {}
    for name, frame in (
        ("Proliferating", proliferating_truth),
        ("Ifit B", ifit_b_truth),
    ):
        truth = frame.loc[:, ["cell_id", "true_label"]].copy()
        truth["cell_id"] = truth["cell_id"].astype(str)
        truth["true_label"] = truth["true_label"].astype(str)
        if truth["cell_id"].duplicated().any():
            raise ValueError(f"{name} truth contains duplicate cell IDs")
        truths[name] = truth

    combined = truths["Proliferating"].merge(
        truths["Ifit B"],
        on="cell_id",
        how="outer",
        suffixes=("_proliferating", "_ifit_b"),
        indicator=True,
        validate="one_to_one",
    )
    overlap = combined["_merge"].eq("both")
    disagrees = overlap & combined["true_label_proliferating"].ne(combined["true_label_ifit_b"])
    if disagrees.any():
        raise ValueError("Endpoint truth tables disagree on shared cell labels")
    if (
        combined.loc[overlap, "true_label_proliferating"]
        .isin([ENDPOINT, OTHER_RNA_ONLY_ENDPOINT])
        .any()
    ):
        raise ValueError("Endpoint-specific cells unexpectedly overlap between truth tables")
    if set(combined.loc[combined["_merge"].eq("left_only"), "true_label_proliferating"]) != {
        ENDPOINT
    }:
        raise ValueError("Proliferating-only truth rows do not exactly define the endpoint")
    if set(combined.loc[combined["_merge"].eq("right_only"), "true_label_ifit_b"]) != {
        OTHER_RNA_ONLY_ENDPOINT
    }:
        raise ValueError("Ifit B-only truth rows do not exactly define the other RNA endpoint")

    combined["canonical_fine_label"] = combined["true_label_proliferating"].combine_first(
        combined["true_label_ifit_b"]
    )
    processed_ids = set(rna_cells["cell_id"].astype(str))
    if set(combined["cell_id"]) != processed_ids:
        raise ValueError("Endpoint truth union does not exactly match processed RNA cell IDs")

    if composition["fine_label"].duplicated().any():
        raise ValueError("Cell-type composition contains duplicate fine labels")
    composition_counts = composition.set_index("fine_label")["rna_count"].astype(int)
    if int(composition_counts.sum()) != expected_n_rna:
        raise ValueError("Cell-type composition RNA count disagrees with embedding manifest")
    observed_counts = combined["canonical_fine_label"].value_counts().sort_index()
    expected_counts = composition_counts.sort_index()
    if not observed_counts.equals(expected_counts):
        raise ValueError("Endpoint truth labels disagree with canonical cell-type composition")

    retained_ids = set(truths["Proliferating"]["cell_id"])
    cohort = rna_cells.rename(columns={"row_index": "embedding_row_index"}).merge(
        combined[["cell_id", "canonical_fine_label"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    cohort["retained_for_proliferating_analysis"] = cohort["cell_id"].isin(retained_ids)
    cohort["analysis_role"] = "represented_state_negative"
    cohort.loc[cohort["canonical_fine_label"].eq(ENDPOINT), "analysis_role"] = "endpoint_positive"
    cohort.loc[~cohort["retained_for_proliferating_analysis"], "analysis_role"] = (
        "excluded_other_rna_only_endpoint"
    )
    cohort["exclusion_reason"] = ""
    cohort.loc[~cohort["retained_for_proliferating_analysis"], "exclusion_reason"] = (
        EXCLUSION_REASON
    )
    cohort = cohort.sort_values("embedding_row_index").reset_index(drop=True)

    excluded = cohort.loc[~cohort["retained_for_proliferating_analysis"]]
    if set(excluded["canonical_fine_label"]) != {OTHER_RNA_ONLY_ENDPOINT}:
        raise ValueError("Excluded cells are not exactly the Ifit B endpoint")
    if set(cohort.loc[cohort["retained_for_proliferating_analysis"], "cell_id"]) != retained_ids:
        raise ValueError("Retained cell IDs disagree with Proliferating query truth")

    counts = {
        "processed": len(cohort),
        "retained": int(cohort["retained_for_proliferating_analysis"].sum()),
        "excluded": len(excluded),
        "positive": int(cohort["analysis_role"].eq("endpoint_positive").sum()),
        "negative": int(cohort["analysis_role"].eq("represented_state_negative").sum()),
    }
    return cohort, counts


def _write_cohort_transition_provenance(
    *, project_root: Path, output_root: Path
) -> tuple[Path, Path]:
    embedding_cells_path = output_root / "embedding/embedding_cells.csv"
    embedding_manifest_path = output_root / "embedding/embedding_manifest.yaml"
    composition_path = output_root / "manifest/cell_type_composition.csv"
    proliferating_truth_path = (
        output_root / "runs/mouse_spleen_natural_proliferating/benchmark/natural_mismatch/"
        "evaluation_truth/query_truth.csv"
    )
    ifit_b_truth_path = (
        output_root / "runs/mouse_spleen_natural_ifit_b/benchmark/natural_mismatch/"
        "evaluation_truth/query_truth.csv"
    )
    provenance_root = output_root / "manuscript/provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    cohort_path = provenance_root / "processed_to_retained_cohort.csv"
    transition_manifest_path = provenance_root / "cohort_transition_manifest.yaml"

    embedding_manifest = yaml.safe_load(embedding_manifest_path.read_text(encoding="utf-8"))
    metadata = embedding_manifest["metadata"]
    embedding_cells = pd.read_csv(embedding_cells_path)
    if len(embedding_cells) != int(metadata["n_rna"]) + int(metadata["n_atac"]):
        raise ValueError("Complete embedding inventory count disagrees with embedding manifest")
    cohort, counts = _build_processed_to_retained_cohort(
        embedding_cells=embedding_cells,
        proliferating_truth=pd.read_csv(proliferating_truth_path),
        ifit_b_truth=pd.read_csv(ifit_b_truth_path),
        composition=pd.read_csv(composition_path),
        expected_n_rna=int(metadata["n_rna"]),
    )
    if counts != EXPECTED_COHORT_COUNTS:
        raise ValueError(
            f"Processed-to-retained cohort counts changed: {counts}; "
            f"expected {EXPECTED_COHORT_COUNTS}"
        )
    cohort.to_csv(cohort_path, index=False)

    source_paths = {
        "embedding_cells": embedding_cells_path,
        "embedding_manifest": embedding_manifest_path,
        "cell_type_composition": composition_path,
        "proliferating_query_truth": proliferating_truth_path,
        "ifit_b_query_truth": ifit_b_truth_path,
    }
    manifest = {
        "stage": "generate_processed_to_retained_cohort_provenance",
        "artifact": {
            "path": str(cohort_path.relative_to(project_root)),
            "sha256": sha256_file(cohort_path),
        },
        "sources": {
            name: {
                "path": str(source_path.relative_to(project_root)),
                "sha256": sha256_file(source_path),
            }
            for name, source_path in source_paths.items()
        },
        "counts": counts,
        "fixed_embedding_scope": {
            "data_scope": metadata["data_scope"],
            "mode": metadata["mode"],
            "dimension": int(metadata["dimension"]),
            "frozen": bool(metadata["frozen"]),
            "n_rna": int(metadata["n_rna"]),
            "n_atac": int(metadata["n_atac"]),
        },
        "retention_rule": (
            "Retain exactly the processed RNA cell IDs in the Proliferating "
            "endpoint-specific query truth."
        ),
        "exclusion_reason": EXCLUSION_REASON,
        "derivation": {
            "mode": "artifact_only_no_refit",
            "model_fits_run": False,
            "hdf5_inputs_read": False,
            "method": (
                "Join the fixed complete embedding cell inventory to the union "
                "of Proliferating and Ifit B endpoint-specific query truth tables."
            ),
        },
    }
    transition_manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return cohort_path, transition_manifest_path


def _paired_stratified_bootstrap(
    *,
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_indices = np.flatnonzero(truth)
    negative_indices = np.flatnonzero(~truth)
    rng = np.random.default_rng(seed)
    method_metrics = {
        method: np.empty((n_bootstrap, 2), dtype=float) for method in SUPPLEMENT_METHOD_ORDER
    }
    for replicate in range(n_bootstrap):
        sampled = np.concatenate(
            [
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
            ]
        )
        sampled_truth = truth[sampled]
        for method in SUPPLEMENT_METHOD_ORDER:
            method_metrics[method][replicate] = _metric_pair(sampled_truth, scores[method][sampled])

    summary_rows: list[dict[str, object]] = []
    point_metrics: dict[str, tuple[float, float]] = {}
    for method in SUPPLEMENT_METHOD_ORDER:
        point_metrics[method] = _metric_pair(truth, scores[method])
        for metric_index, metric in enumerate(("auroc", "auprc")):
            values = method_metrics[method][:, metric_index]
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": point_metrics[method][metric_index],
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "n_bootstrap": n_bootstrap,
                    "bootstrap_seed": seed,
                }
            )

    contrast_rows: list[dict[str, object]] = []
    for baseline in MAIN_METHOD_ORDER:
        if baseline == "coreot_full":
            continue
        for metric_index, metric in enumerate(("auroc", "auprc")):
            values = (
                method_metrics["coreot_full"][:, metric_index]
                - method_metrics[baseline][:, metric_index]
            )
            contrast_rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "estimate": (
                        point_metrics["coreot_full"][metric_index]
                        - point_metrics[baseline][metric_index]
                    ),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "n_bootstrap": n_bootstrap,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def _interval_text(row: pd.Series) -> str:
    return (
        f"{float(row['estimate']):.3f} [{float(row['ci_lower']):.3f}, {float(row['ci_upper']):.3f}]"
    )


def _contrast_interval_text(row: pd.Series) -> str:
    values = [
        float(row["estimate"]),
        float(row["ci_lower"]),
        float(row["ci_upper"]),
    ]
    digits = 6 if max(map(abs, values)) < 0.02 else 3
    return f"{values[0]:.{digits}f} [{values[1]:.{digits}f}, {values[2]:.{digits}f}]"


def _value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def _combined_results_table(
    detection: pd.DataFrame,
    bootstrap: pd.DataFrame,
    transfer: pd.DataFrame,
) -> str:
    return _supplementary_results_table(detection, bootstrap, transfer)


def _supplementary_results_table(
    detection: pd.DataFrame,
    bootstrap: pd.DataFrame,
    transfer: pd.DataFrame,
) -> str:
    lookup = bootstrap.set_index(["method", "metric"])
    detection = detection.loc[detection["method"].isin(SUPPLEMENT_METHOD_ORDER)]
    if set(detection["method"]) != set(SUPPLEMENT_METHOD_ORDER):
        raise ValueError("Detection rows do not match the supplementary method set")
    transfer = transfer.loc[
        transfer["holdout_label"].eq(ENDPOINT) & transfer["method"].isin(SUPPLEMENT_METHOD_ORDER)
    ].set_index("method")
    if set(transfer.index) != set(SUPPLEMENT_METHOD_ORDER):
        raise ValueError("Transfer rows do not match the supplementary method set")

    rows = [
        "| Method | AUROC [95% interval] | AP [95% interval] | Forced accuracy | Forced macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in SUPPLEMENT_METHOD_ORDER:
        transfer_row = transfer.loc[method]
        rows.append(
            "| "
            + " | ".join(
                [
                    METHOD_LABELS[method],
                    _interval_text(lookup.loc[(method, "auroc")]),
                    _interval_text(lookup.loc[(method, "auprc")]),
                    _value(transfer_row["forced_accuracy"]),
                    _value(transfer_row["forced_macro_f1"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _contrast_table(contrasts: pd.DataFrame) -> str:
    lookup = contrasts.set_index(["baseline", "metric"])
    rows = [
        "| Baseline | Delta AUROC [95% interval] | Delta AP [95% interval] |",
        "| --- | ---: | ---: |",
    ]
    for baseline in MAIN_METHOD_ORDER:
        if baseline == "coreot_full":
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    METHOD_LABELS[baseline],
                    _contrast_interval_text(lookup.loc[(baseline, "auroc")]),
                    _contrast_interval_text(lookup.loc[(baseline, "auprc")]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _write_report(
    *,
    docs_root: Path,
    run_root: Path,
    detection_path: Path,
    transfer_path: Path,
    bootstrap_path: Path,
    contrast_path: Path,
    truth: pd.DataFrame,
    detection: pd.DataFrame,
    bootstrap: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> Path:
    docs_root.mkdir(parents=True, exist_ok=True)
    main_path = docs_root / "manuscript_results_mouse_spleen.md"
    supplement_path = docs_root / "manuscript_supp.md"
    transfer = pd.read_csv(transfer_path)
    n_query = len(truth)
    n_positive = int(truth["true_label"].astype(str).eq(ENDPOINT).sum())
    prevalence = n_positive / n_query

    coreot_transfer = transfer.loc[
        transfer["holdout_label"].astype(str).eq(ENDPOINT)
        & transfer["method"].astype(str).eq("coreot_full")
    ]
    if len(coreot_transfer) != 1:
        raise ValueError("Expected exactly one Proliferating CoRe-OT label-transfer row")
    coreot_forced_accuracy = float(coreot_transfer["forced_accuracy"].iloc[0])
    coreot_forced_macro_f1 = float(coreot_transfer["forced_macro_f1"].iloc[0])

    main = f"""# Mouse-spleen Proliferating motivating application

## Objective

We evaluated whether CoRe-OT assigns high fitted deficit to `Proliferating`
RNA cells and separately characterized forced label-transfer performance on
represented states. The analysis is restricted to this endpoint; pooled
natural-mismatch endpoints are outside the report scope.

## Dataset and evaluation task

The query contains {n_query} RNA cells, including {n_positive} `Proliferating` cells. The remaining {n_query - n_positive} cells belong to states represented in the ATAC reference. Detection is evaluated over all query cells, giving a positive-prevalence reference value of {prevalence:.3f} for AP. The primary detection score for CoRe-OT is the query-marginal deficit $u$. AUROC and AP assess endpoint ranking; AP is emphasized because the endpoint is rare.

The canonical CoRe-OT run uses $(\\tau_{{\\min}},\\tau_{{\\max}},\\tau_{{\\mathrm{{target}}}},\\alpha)=(3,5,8,40)$ and entered the comparison only after convergence was verified. Uniform UOT uses $(\\tau_q,\\tau_r,\\alpha)=(4,8,0)$. The remaining comparators are the prior-only score, nearest-neighbor mapping, Seurat, SingleR, CellTypist, scmap-cell, scmap-cluster, and CHETAH. CoRe-OT component ablations are reported separately in the [supplement]({_relative_link(main_path, supplement_path)}).

## Detection and represented-state label-transfer results

{_combined_results_table(detection, bootstrap, transfer)}

AUROC and AP values are point estimates followed by descriptive 95% cell-bootstrap intervals in brackets. The intervals use 2,000 paired stratified replicates with seed `20260713`; they quantify cell-resampling variability within this dataset, not biological-replicate or population-level uncertainty. `Prior only` is a detection-only control and therefore has no label-transfer metrics. No matched full-reference threshold calibration is available for the natural-mismatch task. Consequently, post-abstention accuracy, post-abstention macro-F1, and represented-state coverage are not applicable; represented-state label transfer is evaluated only through forced-label metrics.

On represented-state cells, the fitted CoRe-OT model had forced accuracy
and forced macro-F1 point estimates of {coreot_forced_accuracy:.3f} and
{coreot_forced_macro_f1:.3f}, respectively.

### Paired detection contrasts

The paired differences below are defined as CoRe-OT minus the named baseline.

{_contrast_table(contrasts)}

## Interpretation and limitations

On represented-state cells, CoRe-OT had forced accuracy and forced
macro-F1 of {coreot_forced_accuracy:.3f} and
{coreot_forced_macro_f1:.3f}, respectively. The ranking results support
weak-reference-correspondence detection under the fitted models; they do not
by themselves prove biological absence, novelty, or population-level
generalization.

The matchability-prior construction and operating point were selected after
examining the same `Proliferating` endpoint. The comparative results are
therefore exploratory, may be optimistic, and are not an independent
confirmatory evaluation.

The experiment contains no donor or sample replicates. The bootstrap intervals are descriptive cell-resampling summaries and do not replace biological replication. Detection scores are method-specific and should be compared through ranking metrics rather than their raw numerical scales. Threshold-dependent selective-prediction claims are not made because a matched full-reference calibration condition is unavailable.

## Data lineage

The report was generated from the current per-cell artifacts after regenerating the natural baseline summaries. Detection results trace from the internal and external per-cell score tables under `{run_root}` through [{detection_path.name}]({_relative_link(main_path, detection_path)}). Represented-state metrics come from [{transfer_path.name}]({_relative_link(main_path, transfer_path)}). Bootstrap summaries and paired contrasts are stored in [{bootstrap_path.name}]({_relative_link(main_path, bootstrap_path)}) and [{contrast_path.name}]({_relative_link(main_path, contrast_path)}), respectively.
"""
    main_path.write_text(main, encoding="utf-8")

    return main_path


def generate_report(project_root: Path) -> tuple[Path, Path]:
    output_root = project_root / "results/mouse_spleen_core_ot"
    cohort_path, cohort_transition_manifest_path = _write_cohort_transition_provenance(
        project_root=project_root,
        output_root=output_root,
    )
    run_root = output_root / "runs/mouse_spleen_natural_proliferating"
    baseline_root = output_root / "natural_mismatch/compare_baselines"
    detection_path = baseline_root / "tables/baseline_detection_by_run.csv"
    transfer_path = baseline_root / "tables/baseline_shared_label_transfer_by_run.csv"
    report_tables = output_root / "manuscript/tables"
    report_tables.mkdir(parents=True, exist_ok=True)
    bootstrap_path = report_tables / "proliferating_detection_bootstrap.csv"
    contrast_path = report_tables / "proliferating_detection_contrasts.csv"
    component_root = output_root / "natural_mismatch/sensitivity/component_ablation_proliferating"
    compatibility_ablation_path = component_root / "tables/compatibility_only_metrics.csv"
    docs_root = project_root / "docs"
    compatibility_ablation_report_path = (
        docs_root / "manuscript_supp_mouse_spleen_proliferating_compatibility_only.md"
    )
    compatibility_ablation_figure_path = (
        docs_root / "figs" / "manuscript_fig_mouse_spleen_component_minus_m.png"
    )
    rho_tau_root = (
        output_root
        / "natural_mismatch/sensitivity/rho_attribution_tau_alpha_search"
    )
    rho_tau_alpha5_summary_path = (
        rho_tau_root / "tables/rho_tau_surface_alpha5_range025_4.csv"
    )
    rho_tau_alpha5_manifest_path = (
        rho_tau_root / "design/rho_tau_surface_alpha5_range025_4.yaml"
    )
    rho_tau_alpha5_figure_path = (
        docs_root
        / "figs/manuscript_supp_rho_attribution_mouse_spleen_alpha5.png"
    )
    prior_dependence_root = output_root / "manuscript/prior_dependence"
    prior_dependence_figure_path = (
        docs_root / "figs/manuscript_fig_mouse_spleen_prior_dependence.png"
    )
    prior_dependence_summary_path = (
        prior_dependence_root / "prior_correlation.csv"
    )
    prior_dependence_manifest_path = prior_dependence_root / "manifest.yaml"
    required_supplement_artifacts = (
        compatibility_ablation_path,
        compatibility_ablation_report_path,
        compatibility_ablation_figure_path,
        rho_tau_alpha5_summary_path,
        rho_tau_alpha5_manifest_path,
        rho_tau_alpha5_figure_path,
        prior_dependence_figure_path,
        prior_dependence_summary_path,
        prior_dependence_manifest_path,
    )
    missing_supplement_artifacts = [
        str(path) for path in required_supplement_artifacts if not path.is_file()
    ]
    if missing_supplement_artifacts:
        raise FileNotFoundError(
            "Supplementary artifacts are required before regenerating the "
            f"Proliferating report: {missing_supplement_artifacts}"
        )
    truth, scores, detection = _load_detection_scores(
        run_root=run_root, detection_path=detection_path
    )
    positive = truth["true_label"].astype(str).eq(ENDPOINT).to_numpy()
    bootstrap, contrasts = _paired_stratified_bootstrap(
        truth=positive,
        scores=scores,
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )
    bootstrap.to_csv(bootstrap_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    main_path = _write_report(
        docs_root=project_root / "docs",
        run_root=run_root,
        detection_path=detection_path,
        transfer_path=transfer_path,
        bootstrap_path=bootstrap_path,
        contrast_path=contrast_path,
        truth=truth,
        detection=detection,
        bootstrap=bootstrap,
        contrasts=contrasts,
    )
    manifest_path = output_root / "manuscript/proliferating_report_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "generate_proliferating_report",
                "artifacts": {
                    "main_report": str(main_path.relative_to(project_root)),
                    "bootstrap": str(bootstrap_path.relative_to(project_root)),
                    "contrasts": str(contrast_path.relative_to(project_root)),
                    "compatibility_ablation_figure": str(
                        compatibility_ablation_figure_path.relative_to(project_root)
                    ),
                    "rho_attribution_alpha5_figure": str(
                        rho_tau_alpha5_figure_path.relative_to(project_root)
                    ),
                    "rho_attribution_alpha5_summary": str(
                        rho_tau_alpha5_summary_path.relative_to(project_root)
                    ),
                    "rho_attribution_alpha5_manifest": str(
                        rho_tau_alpha5_manifest_path.relative_to(project_root)
                    ),
                    "prior_dependence_figure": str(
                        prior_dependence_figure_path.relative_to(project_root)
                    ),
                    "prior_dependence_summary": str(
                        prior_dependence_summary_path.relative_to(project_root)
                    ),
                    "prior_dependence_manifest": str(
                        prior_dependence_manifest_path.relative_to(project_root)
                    ),
                    "processed_to_retained_cohort": str(cohort_path.relative_to(project_root)),
                    "cohort_transition_manifest": str(
                        cohort_transition_manifest_path.relative_to(project_root)
                    ),
                    "cell_type_composition": str(
                        (output_root / "manifest/cell_type_composition.csv").relative_to(
                            project_root
                        )
                    ),
                    "embedding_manifest": str(
                        (output_root / "embedding/embedding_manifest.yaml").relative_to(
                            project_root
                        )
                    ),
                },
                "metadata": {
                    "endpoint": ENDPOINT,
                    "excluded_endpoints": ["Ifit B", "pooled"],
                    "main_excluded_methods": [
                        "coreot_constant_tau",
                        "coreot_match_only",
                    ],
                    "retained_methods": SUPPLEMENT_METHOD_ORDER,
                    "n_bootstrap": N_BOOTSTRAP,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "interval": "percentile_2.5_97.5",
                    "uncertainty_scope": "descriptive_cell_resampling",
                    "figures": "supplementary_component_artifacts",
                    "reported_coreot_scores": ["u"],
                    "excluded_manuscript_scores": ["u_tilde"],
                    "uniform_uot_operating_point": {
                        "tau_q": 4.0,
                        "tau_r": 8.0,
                        "alpha": 0.0,
                    },
                    "figure_construction_guide": ("docs/manuscript_mouse_spleen_main_figure.md"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return main_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for output_path in generate_report(args.project_root.resolve()):
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
