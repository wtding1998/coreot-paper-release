from __future__ import annotations

import hashlib
import math
from pathlib import Path

import anndata
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from coreot.config.load import load_yaml
from coreot.results.grid_common import ResultsGridError, discover_run_descriptors

INTERNAL_CANDIDATE_SET = "hiha_harmony30_k100"
EXTERNAL_CANDIDATE_SET = "external_reference_mapping"
EXPECTED_INPUT_SHA256 = "64b48e211170ac921165e4402300dce0ffdf5f6fae559ec1d6b890da8c9aaa13"
SCORES = {
    "prior_only": ("prior_risk",),
    "nn": ("nn_distance",),
    "uniform_uot": ("u",),
    "coreot_full": ("u",),
    "seurat_anchor": ("u",),
    "singleR": ("u",),
    "celltypist_l3": ("u",),
    "scmap_cell": ("u",),
    "scmap_cluster": ("u",),
    "chetah": ("u",),
}
PRIMARY = {method: scores[-1] for method, scores in SCORES.items()}
METHOD_DISPLAY = {
    "coreot_full": "CoRe-OT",
    "uniform_uot": "Uniform UOT",
    "nn": "Nearest neighbor",
    "prior_only": "Prior only",
    "seurat_anchor": "Seurat",
    "singleR": "SingleR",
    "celltypist_l3": "CellTypist",
    "scmap_cell": "scmap-cell",
    "scmap_cluster": "scmap-cluster",
    "chetah": "CHETAH",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_path(grid_dir: Path, run_id: str) -> Path:
    config = load_yaml(grid_dir / run_id / "raw_import.yaml")
    return Path(config["input"]["path"])


def _truth(run_root: Path, condition: str) -> pd.DataFrame:
    path = run_root / "benchmark" / condition / "evaluation_truth" / "query_truth.csv"
    frame = pd.read_csv(path)
    if frame["cell_id"].duplicated().any():
        raise ResultsGridError(f"Duplicate query truth cell IDs: {path}")
    return frame


def _scores(run_root: Path, condition: str, candidate_set: str) -> pd.DataFrame:
    path = run_root / "scoring" / condition / candidate_set / "cell_scores.parquet"
    frame = pd.read_parquet(path)
    if frame.duplicated(["cell_id", "method"]).any():
        raise ResultsGridError(f"Duplicate score cell/method rows: {path}")
    return frame


def _metadata(input_path: Path) -> pd.DataFrame:
    obs = anndata.read_h5ad(input_path, backed="r").obs[["AIFI_L2", "AIFI_L3"]].copy()
    obs["cell_id"] = obs.index.astype(str)
    return obs.reset_index(drop=True)


def _method_scores(run_root: Path, condition: str) -> pd.DataFrame:
    frames = []
    for candidate_set in (INTERNAL_CANDIDATE_SET, EXTERNAL_CANDIDATE_SET):
        frame = _scores(run_root, condition, candidate_set)
        frame["candidate_set"] = candidate_set
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined.loc[combined["method"].isin(SCORES)].copy()


def build_hiha_detection_by_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    method_grid_dirs: dict[str, Path] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    descriptors = discover_run_descriptors(grid_dir)
    descriptor_keys = {
        (descriptor.held_out_label, descriptor.seed) for descriptor in descriptors
    }
    replacement_roots: dict[str, dict[tuple[str, int], Path]] = {}
    for method, method_grid_dir in (method_grid_dirs or {}).items():
        method_descriptors = discover_run_descriptors(method_grid_dir)
        method_keys = {
            (descriptor.held_out_label, descriptor.seed)
            for descriptor in method_descriptors
        }
        if method_keys != descriptor_keys:
            raise ResultsGridError(
                f"Replacement grid for {method} has state/seed keys "
                f"{sorted(method_keys)}; expected {sorted(descriptor_keys)}"
            )
        replacement_roots[method] = {
            (descriptor.held_out_label, descriptor.seed): runs_root / descriptor.run_id
            for descriptor in method_descriptors
        }
    metadata = _metadata(_input_path(grid_dir, descriptors[0].run_id))
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _truth(run_root, "incomplete_reference")
        score_frame = _method_scores(run_root, "incomplete_reference")
        score_frame["score_run_id"] = descriptor.run_id
        for method, roots_by_key in replacement_roots.items():
            replacement_root = roots_by_key[
                (descriptor.held_out_label, descriptor.seed)
            ]
            replacement = _scores(
                replacement_root,
                "incomplete_reference",
                candidate_set_for(method),
            )
            replacement = replacement.loc[replacement["method"].eq(method)].copy()
            if replacement.empty:
                raise ResultsGridError(
                    f"Missing {method} scores in replacement run "
                    f"{replacement_root.name}"
                )
            replacement["candidate_set"] = candidate_set_for(method)
            replacement["score_run_id"] = replacement_root.name
            score_frame = pd.concat(
                [
                    score_frame.loc[~score_frame["method"].eq(method)],
                    replacement,
                ],
                ignore_index=True,
            )
        joined = score_frame.merge(truth, on="cell_id", validate="many_to_one")
        joined = joined.merge(metadata, on="cell_id", validate="many_to_one")
        if joined[["true_label", "AIFI_L2"]].isna().any().any():
            raise ResultsGridError(f"Incomplete truth join for {descriptor.run_id}")
        for method, score_names in SCORES.items():
            method_rows = joined.loc[joined["method"].eq(method)]
            if method_rows.empty:
                raise ResultsGridError(f"Missing {method} scores for {descriptor.run_id}")
            score_run_ids = method_rows["score_run_id"].astype(str).unique()
            if len(score_run_ids) != 1:
                raise ResultsGridError(
                    f"Multiple score-run IDs for {descriptor.run_id}/{method}: "
                    f"{sorted(score_run_ids)}"
                )
            for score in score_names:
                for scope, eligible in (
                    ("global_all_query", pd.Series(True, index=method_rows.index)),
                    ("local_within_broad_state", method_rows["AIFI_L2"].astype(str).eq("cDC2")),
                ):
                    subset = method_rows.loc[eligible].copy()
                    values = pd.to_numeric(subset[score], errors="coerce")
                    finite = values.notna() & values.map(math.isfinite)
                    subset = subset.loc[finite]
                    values = values.loc[finite]
                    positive = subset["is_absent_state"].astype(bool)
                    n_positive = int(positive.sum())
                    n_negative = int((~positive).sum())
                    if not n_positive or not n_negative:
                        raise ResultsGridError(
                            f"Degenerate {scope} cohort for {descriptor.run_id}/{method}/{score}"
                        )
                    rows.append(
                        {
                            "run_id": descriptor.run_id,
                            "evaluation_run_id": descriptor.run_id,
                            "score_run_id": score_run_ids[0],
                            "held_out_label": descriptor.held_out_label,
                            "seed": descriptor.seed,
                            "condition_id": "incomplete_reference",
                            "candidate_set": str(method_rows["candidate_set"].iloc[0]),
                            "method": method,
                            "method_group": "external_mapping"
                            if candidate_set_for(method) == EXTERNAL_CANDIDATE_SET
                            else "internal_control",
                            "primary_score": PRIMARY[method],
                            "score": score,
                            "score_role": "primary" if score == PRIMARY[method] else "secondary",
                            "evaluation_scope": scope,
                            "n_query": len(subset),
                            "n_positive": n_positive,
                            "n_negative": n_negative,
                            "prevalence": n_positive / len(subset),
                            "auroc": roc_auc_score(positive, values),
                            "auprc": average_precision_score(positive, values),
                            "auprc_baseline": n_positive / len(subset),
                        }
                    )
    frame = pd.DataFrame(rows)
    comparator = frame.loc[
        frame["method"].eq("uniform_uot"),
        ["held_out_label", "seed", "evaluation_scope", "auroc", "auprc"],
    ].rename(columns={"auroc": "comparator_auroc", "auprc": "comparator_auprc"})
    frame = frame.merge(comparator, on=["held_out_label", "seed", "evaluation_scope"], how="left")
    is_coreot = frame["method"].eq("coreot_full")
    frame["comparator_method"] = pd.NA
    frame.loc[is_coreot, "comparator_method"] = "uniform_uot"
    frame["delta_auroc_vs_comparator"] = math.nan
    frame["delta_auprc_vs_comparator"] = math.nan
    frame.loc[is_coreot, "delta_auroc_vs_comparator"] = (
        frame.loc[is_coreot, "auroc"] - frame.loc[is_coreot, "comparator_auroc"]
    )
    frame.loc[is_coreot, "delta_auprc_vs_comparator"] = (
        frame.loc[is_coreot, "auprc"] - frame.loc[is_coreot, "comparator_auprc"]
    )
    return frame.drop(columns=["comparator_auroc", "comparator_auprc"])


def candidate_set_for(method: str) -> str:
    return (
        EXTERNAL_CANDIDATE_SET
        if method
        in {"seurat_anchor", "singleR", "celltypist_l3", "scmap_cell", "scmap_cluster", "chetah"}
        else INTERNAL_CANDIDATE_SET
    )


def build_hiha_rescue(
    *,
    runs_root: Path,
    grid_dir: Path,
    method_grid_dirs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_frames = []
    descriptors = discover_run_descriptors(grid_dir)
    descriptor_keys = {
        (descriptor.held_out_label, descriptor.seed) for descriptor in descriptors
    }
    replacement_roots: dict[str, dict[tuple[str, int], Path]] = {}
    for method, method_grid_dir in (method_grid_dirs or {}).items():
        method_descriptors = discover_run_descriptors(method_grid_dir)
        method_keys = {
            (descriptor.held_out_label, descriptor.seed)
            for descriptor in method_descriptors
        }
        if method_keys != descriptor_keys:
            raise ResultsGridError(
                f"Replacement grid for {method} has state/seed keys "
                f"{sorted(method_keys)}; expected {sorted(descriptor_keys)}"
            )
        replacement_roots[method] = {
            (descriptor.held_out_label, descriptor.seed): runs_root / descriptor.run_id
            for descriptor in method_descriptors
        }
    metadata = _metadata(_input_path(grid_dir, descriptors[0].run_id))
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _truth(run_root, "incomplete_reference")
        for method in ("coreot_full", "uniform_uot"):
            score_root = replacement_roots.get(method, {}).get(
                (descriptor.held_out_label, descriptor.seed),
                run_root,
            )
            inc = _scores(score_root, "incomplete_reference", INTERNAL_CANDIDATE_SET)
            full = _scores(score_root, "full_reference_control", INTERNAL_CANDIDATE_SET)
            score_names = SCORES[method]
            left = inc.loc[inc["method"].eq(method), ["cell_id", *score_names]].copy()
            right = full.loc[full["method"].eq(method), ["cell_id", *score_names]].copy()
            paired = left.merge(
                right, on="cell_id", validate="one_to_one", suffixes=("_ablated", "_full")
            )
            if len(paired) != len(left) or len(left) != len(right):
                raise ResultsGridError(f"Unpaired rescue cells for {descriptor.run_id}/{method}")
            paired = paired.merge(
                truth[["cell_id", "is_absent_state", "is_shared_state"]],
                on="cell_id",
                validate="one_to_one",
            )
            paired = paired.merge(
                metadata[["cell_id", "AIFI_L2"]], on="cell_id", validate="one_to_one"
            )
            paired["truth_group"] = "other_shared"
            paired.loc[paired["AIFI_L2"].astype(str).eq("cDC2"), "truth_group"] = "shared_cDC2"
            paired.loc[paired["is_absent_state"].astype(bool), "truth_group"] = "held_out_positive"
            paired.insert(0, "run_id", descriptor.run_id)
            paired.insert(1, "evaluation_run_id", descriptor.run_id)
            paired.insert(2, "score_run_id", score_root.name)
            paired.insert(3, "held_out_label", descriptor.held_out_label)
            paired.insert(4, "seed", descriptor.seed)
            paired.insert(5, "method", method)
            for score in score_names:
                paired[f"delta_{score}_rescue"] = (
                    paired[f"{score}_ablated"] - paired[f"{score}_full"]
                )
            cell_frames.append(paired)
    cells = pd.concat(cell_frames, ignore_index=True)
    delta_columns = ["delta_u_rescue"]
    summary = cells.groupby(
        [
            "run_id",
            "evaluation_run_id",
            "score_run_id",
            "held_out_label",
            "seed",
            "method",
            "truth_group",
        ],
        as_index=False,
    )[delta_columns].agg(["count", "mean", "median"])
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return cells, summary


def summarize_detection(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["held_out_label", "method", "method_group", "score", "score_role", "evaluation_scope"]
    values = [
        "n_query",
        "n_positive",
        "n_negative",
        "prevalence",
        "auprc",
        "auroc",
        "delta_auprc_vs_comparator",
        "delta_auroc_vs_comparator",
    ]
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        base = dict(zip(keys, key, strict=True))
        for value in values:
            series = pd.to_numeric(group[value], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "quantity": value,
                    "mean": series.mean() if len(series) else math.nan,
                    "std": series.std(ddof=1) if len(series) > 1 else math.nan,
                    "n_runs": len(series),
                }
            )
    return pd.DataFrame(rows)


def write_hiha_reformulation(
    *,
    runs_root: Path,
    grid_dir: Path,
    output_root: Path,
    method_grid_dirs: dict[str, Path] | None = None,
) -> dict[str, Path]:
    descriptors = discover_run_descriptors(grid_dir)
    expected = {(label, seed) for label in ("HLA-DRhi cDC2", "ISG+ cDC2") for seed in range(1, 6)}
    observed = {(item.held_out_label, item.seed) for item in descriptors}
    if observed != expected:
        raise ResultsGridError(
            f"Expected state/seed pairs {sorted(expected)}; got {sorted(observed)}"
        )
    input_path = _input_path(grid_dir, descriptors[0].run_id)
    checksum = _sha256(input_path)
    if checksum != EXPECTED_INPUT_SHA256:
        raise ResultsGridError(f"HIHA input checksum mismatch: {checksum}")
    tables = output_root / "main" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    detection = build_hiha_detection_by_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        method_grid_dirs=method_grid_dirs,
    )
    summary = summarize_detection(detection)
    cells, rescue = build_hiha_rescue(
        runs_root=runs_root,
        grid_dir=grid_dir,
        method_grid_dirs=method_grid_dirs,
    )
    legacy_path = output_root / "compare_baselines" / "tables" / "compare_detection_by_run.csv"
    legacy = pd.read_csv(legacy_path)
    keys = ["held_out_label", "seed", "method", "score"]
    audit = detection.loc[detection["evaluation_scope"].eq("global_all_query")].merge(
        legacy[keys + ["auroc", "auprc"]],
        on=keys,
        validate="one_to_one",
        suffixes=("_regenerated", "_preserved"),
    )
    audit["auroc_absolute_difference"] = (
        audit["auroc_regenerated"] - audit["auroc_preserved"]
    ).abs()
    audit["auprc_absolute_difference"] = (
        audit["auprc_regenerated"] - audit["auprc_preserved"]
    ).abs()
    if (
        len(audit) != len(detection) // 2
        or audit[["auroc_absolute_difference", "auprc_absolute_difference"]].max().max() > 1e-6
    ):
        raise ResultsGridError(
            "Global detection reproduction failed; canonical files were not replaced"
        )
    composition = detection.loc[
        detection["method"].eq("coreot_full") & detection["score"].eq("u"),
        [
            "run_id",
            "evaluation_run_id",
            "score_run_id",
            "held_out_label",
            "seed",
            "evaluation_scope",
            "n_query",
            "n_positive",
            "n_negative",
            "prevalence",
        ],
    ].copy()
    paths = {
        "detection": tables / "main_detection_by_run.csv",
        "detection_summary": tables / "main_detection_summary.csv",
        "rescue_cells": tables / "rescue_by_cell.parquet",
        "rescue": tables / "rescue_by_run.csv",
        "qc": output_root / "reproducibility" / "qc_report.csv",
        "discrepancies": output_root / "reproducibility" / "reproduction_discrepancies.csv",
        "composition": tables / "benchmark_composition_by_run.csv",
        "manifest": output_root / "main" / "manifest.yaml",
    }
    paths["qc"].parent.mkdir(parents=True, exist_ok=True)
    detection.to_csv(paths["detection"], index=False)
    summary.to_csv(paths["detection_summary"], index=False)
    cells.to_parquet(paths["rescue_cells"], index=False)
    rescue.to_csv(paths["rescue"], index=False)
    composition.to_csv(paths["composition"], index=False)
    audit.to_csv(paths["discrepancies"], index=False)
    pd.DataFrame(
        [
            {"check": "input_sha256", "status": "pass", "value": checksum},
            {"check": "state_seed_pairs", "status": "pass", "value": len(observed)},
            {"check": "model_rerun", "status": "pass", "value": False},
        ]
    ).to_csv(paths["qc"], index=False)
    paths["manifest"].write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-results",
                "artifacts": {key: str(value) for key, value in paths.items() if key != "manifest"},
                "metadata": {
                    "experiment": "HIHA_DC",
                    "input_path": str(input_path),
                    "input_sha256": checksum,
                    "grid_dir": str(grid_dir),
                    "method_grid_dirs": {
                        method: str(path)
                        for method, path in (method_grid_dirs or {}).items()
                    },
                    "runs_root": str(runs_root),
                    "model_rerun": False,
                    "evaluation_scopes": ["global_all_query", "local_within_broad_state"],
                    "variability": "descriptive_seed_level_mean_std_ddof1",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return paths
