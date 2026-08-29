from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

from coreot.artifacts.run_artifacts import (
    ArtifactBatchError,
    ArtifactContractError,
    ArtifactFailure,
    RunArtifacts,
)

STAGE = "manuscript-results"

PRIMARY_SCORE_BY_METHOD: dict[str, str] = {
    "nn": "nn_distance",
    "uniform_uot": "u",
    "coreot_constant_tau": "u",
    "coreot_full": "u",
    "coreot_match_only": "u",
    "prior_only": "prior_risk",
}

EXTERNAL_PRIMARY_SCORE_BY_METHOD: dict[str, str] = {
    "seurat_anchor": "u",
    "singleR": "u",
    "celltypist_l3": "u",
    "scmap_cell": "u",
    "scmap_cluster": "u",
    "chetah": "u",
}

METHOD_GROUP_BY_METHOD: dict[str, str] = {
    "nn": "baseline",
    "uniform_uot": "baseline",
    "coreot_constant_tau": "ablation",
    "coreot_full": "coreot",
    "coreot_match_only": "ablation",
    "prior_only": "prior",
    "seurat_anchor": "external_baseline",
    "singleR": "external_baseline",
    "celltypist_l3": "external_baseline",
    "scmap_cell": "external_baseline",
    "scmap_cluster": "external_baseline",
    "chetah": "external_baseline",
}

LABEL_TRANSFER_METHODS = tuple(
    method for method in PRIMARY_SCORE_BY_METHOD if method != "prior_only"
) + tuple(EXTERNAL_PRIMARY_SCORE_BY_METHOD)

DISPLAY_NAME_BY_METHOD: dict[str, str] = {
    "prior_only": "Prior-only",
    "nn": "Nearest neighbor",
    "uniform_uot": "Uniform UOT",
    "coreot_constant_tau": "Constant-tau CoRe-OT",
    "coreot_full": "CoRe-OT",
    "coreot_match_only": "CoRe-OT (−C)",
    "seurat_anchor": "Seurat",
    "singleR": "SingleR",
    "celltypist_l3": "CellTypist",
    "scmap_cell": "scmap-cell",
    "scmap_cluster": "scmap-cluster",
    "chetah": "CHETAH",
}

TABLE_ROW_ORDER: tuple[str, ...] = (
    "prior_only",
    "nn",
    "uniform_uot",
    "coreot_constant_tau",
    "coreot_full",
    "coreot_match_only",
)

EXTERNAL_TABLE_ROW_ORDER: tuple[str, ...] = (
    "seurat_anchor",
    "singleR",
    "celltypist_l3",
    "scmap_cell",
    "scmap_cluster",
    "chetah",
)

MAIN_TABLE_1_QUANTITIES: tuple[str, ...] = (
    "auroc",
    "auprc",
    "auprc_baseline",
    "absent_abstention_rate",
    "shared_false_abstention_rate",
)

MAIN_TABLE_1_COLUMN_HEADERS: dict[str, str] = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "auprc_baseline": "AUPRC bl",
    "absent_abstention_rate": "AbsAb",
    "shared_false_abstention_rate": "ShFA",
}

MAIN_TABLE_2_QUANTITIES: tuple[str, ...] = (
    "forced_accuracy",
    "forced_macro_f1",
    "post_abstention_accuracy",
    "post_abstention_macro_f1",
    "coverage",
    "shared_false_abstention_rate",
)

MAIN_TABLE_2_COLUMN_HEADERS: dict[str, str] = {
    "forced_accuracy": "ForcedAcc",
    "forced_macro_f1": "ForcedF1",
    "post_abstention_accuracy": "PostAcc",
    "post_abstention_macro_f1": "PostF1",
    "coverage": "Coverage",
    "shared_false_abstention_rate": "FalseAb",
}


class ResultsGridError(ValueError):
    """Raised when manuscript-results artifacts cannot be generated."""


@dataclass(frozen=True)
class RunDescriptor:
    run_id: str
    held_out_label: str
    seed: int

def discover_run_descriptors(grid_dir: str | Path) -> tuple[RunDescriptor, ...]:
    root = Path(grid_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Generated grid directory does not exist: {root}")
    descriptors = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        benchmark_path = run_dir / "benchmark.yaml"
        payload = _read_yaml(benchmark_path)
        held_out_label = _required_str(payload, "removed_state", benchmark_path)
        split = payload.get("split", {})
        if not isinstance(split, dict) or "seed" not in split:
            raise ResultsGridError(f"Missing split.seed in {benchmark_path}")
        descriptors.append(
            RunDescriptor(
                run_id=run_dir.name,
                held_out_label=held_out_label,
                seed=int(payload.get("repeat", split["seed"])),
            )
        )
    return tuple(descriptors)

def build_detection_by_run(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
    methods: tuple[str, ...],
    grid_dir: Path | None = None,
    score_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build per-run detection metrics.

    When *score_overrides* maps a method to an alternative score name
    (e.g. ``{"coreot_full": "u_tilde"}``), that score is used for AUROC,
    AUPRC and median metrics. When the effective score is ``u_tilde`` and
    *grid_dir* is supplied, abstention rates use the corresponding
    full-reference-calibrated ``u_tilde`` policy.
    """
    overrides = score_overrides or {}
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        metrics = _read_metrics(run_root)
        truth = _read_query_truth(run_root, condition)
        score_frame = _read_cell_scores(run_root, condition, candidate_set)
        auprc_baseline = float(truth["is_absent_state"].astype(bool).mean())
        for method in methods:
            primary_score = primary_score_for_method(method)
            effective_score = overrides.get(method, primary_score)
            method_metrics = metrics.loc[
                (metrics["condition_id"] == condition)
                & (metrics["candidate_set"] == candidate_set)
                & (metrics["method"] == method)
            ]
            metric_values = _metric_values(method_metrics, effective_score)
            if effective_score == "u_tilde" and grid_dir is not None:
                joined = _joined_method_scores(score_frame, truth, method)
                abstention = _u_tilde_abstention(
                    joined=joined,
                    runs_root=runs_root,
                    grid_dir=grid_dir,
                    descriptor=descriptor,
                    candidate_set=candidate_set,
                    method=method,
                )
                absent_mask = joined["is_absent_state"].astype(bool)
                shared_mask = joined["is_shared_state"].astype(bool)
                absent_abstention = float(abstention.loc[absent_mask].mean())
                shared_false_abstention = float(abstention.loc[shared_mask].mean())
            else:
                absent_abstention = _single_metric(
                    method_metrics,
                    score="abstain_u_or_entropy",
                    metric="absent_abstention_rate",
                )
                shared_false_abstention = _shared_false_abstention_rate(
                    score_frame, truth, method
                )
            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(method, "unknown"),
                    "primary_score": primary_score,
                    "score": effective_score,
                    "auroc": metric_values["auroc"],
                    "auprc": metric_values["auprc"],
                    "auprc_baseline": auprc_baseline,
                    "absent_abstention_rate": absent_abstention,
                    "shared_false_abstention_rate": shared_false_abstention,
                    "median_absent": metric_values["median_absent"],
                    "median_shared": metric_values["median_shared"],
                    "absent_minus_shared_median": metric_values["absent_minus_shared_median"],
                }
            )
    return pd.DataFrame(rows)


def primary_score_for_method(method: str) -> str:
    try:
        return PRIMARY_SCORE_BY_METHOD[method]
    except KeyError:
        try:
            return EXTERNAL_PRIMARY_SCORE_BY_METHOD[method]
        except KeyError as exc:
            known = sorted(set(PRIMARY_SCORE_BY_METHOD) | set(EXTERNAL_PRIMARY_SCORE_BY_METHOD))
            raise ResultsGridError(
                f"Unknown method {method!r}; known manuscript result methods: {known}"
            ) from exc


def build_forced_label_summary_by_run(
    *, runs_root: Path, descriptors: tuple[RunDescriptor, ...], methods: tuple[str, ...]
) -> pd.DataFrame:
    frames = []
    for descriptor in descriptors:
        frame = _read_forced_label_summary(runs_root / descriptor.run_id)
        frame.insert(0, "seed", descriptor.seed)
        frame.insert(0, "held_out_label", descriptor.held_out_label)
        frame.insert(0, "run_id", descriptor.run_id)
        frame = frame.loc[frame["method"].isin(methods)]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_shared_label_transfer_by_run(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
    methods: tuple[str, ...],
    grid_dir: Path | None = None,
    score_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    overrides = score_overrides or {}
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _read_query_truth(run_root, condition)
        score_frame = _read_cell_scores(run_root, condition, candidate_set)
        for method in methods:
            effective_score = overrides.get(method, None)
            joined = _joined_method_scores(score_frame, truth, method)
            shared = joined.loc[joined["is_shared_state"].astype(bool)].copy()
            forced_labeled = shared.loc[_nonempty_string(shared["forced_label"])]
            if effective_score == "u_tilde":
                if grid_dir is not None:
                    u_tilde_abstain = _u_tilde_abstention(
                        joined=shared,
                        runs_root=runs_root,
                        grid_dir=grid_dir,
                        descriptor=descriptor,
                        candidate_set=candidate_set,
                        method=method,
                    )
                else:
                    u_tilde_abstain = shared["abstain_u_or_entropy"].astype(bool)
                non_abstained = shared.loc[~u_tilde_abstain]
                abstention_col = u_tilde_abstain
                post_label_column = "forced_label"
            else:
                non_abstained = shared.loc[~shared["abstain_u_or_entropy"].astype(bool)]
                abstention_col = shared["abstain_u_or_entropy"].astype(bool)
                post_label_column = "final_label_abstention_aware"
            post_labeled = non_abstained.loc[
                _nonempty_string(non_abstained[post_label_column])
            ]
            labels = _represented_truth_labels(shared)
            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(method, "unknown"),
                    "score": effective_score or primary_score_for_method(method),
                    "n_shared": int(len(shared)),
                    "n_forced_labeled": int(len(forced_labeled)),
                    "n_non_abstained": int(len(non_abstained)),
                    "forced_accuracy": _accuracy(
                        forced_labeled["true_label"], forced_labeled["forced_label"]
                    ),
                    "forced_macro_f1": _macro_f1(
                        forced_labeled["true_label"], forced_labeled["forced_label"], labels
                    ),
                    "post_abstention_accuracy": _accuracy(
                        post_labeled["true_label"],
                        post_labeled[post_label_column],
                    ),
                    "post_abstention_macro_f1": _macro_f1(
                        post_labeled["true_label"],
                        post_labeled[post_label_column],
                        labels,
                    ),
                    "coverage": float(len(non_abstained) / len(shared)) if len(shared) else math.nan,
                    "shared_false_abstention_rate": float(
                        abstention_col.astype(bool).mean()
                    )
                    if len(shared)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _u_tilde_abstention(
    *,
    joined: pd.DataFrame,
    runs_root: Path,
    grid_dir: Path,
    descriptor: RunDescriptor,
    candidate_set: str,
    method: str,
) -> pd.Series:
    theta_u_tilde, theta_h = _compute_u_tilde_thresholds(
        runs_root=runs_root,
        grid_dir=grid_dir,
        descriptor=descriptor,
        candidate_set=candidate_set,
        method=method,
    )
    u_tilde = pd.to_numeric(joined["u_tilde"], errors="coerce")
    entropy = pd.to_numeric(joined["label_entropy"], errors="coerce")
    if u_tilde.isna().any() or entropy.isna().any():
        raise ResultsGridError(
            f"Missing u_tilde or label_entropy values for run_id={descriptor.run_id}, "
            f"method={method}"
        )
    return (u_tilde > theta_u_tilde) | (entropy > theta_h)


def _compute_u_tilde_thresholds(
    *,
    runs_root: Path,
    grid_dir: Path,
    descriptor: RunDescriptor,
    candidate_set: str,
    method: str,
) -> tuple[float, float]:
    """Return (theta_u_tilde, theta_h) for counterfactual u_tilde abstention."""
    config_dir = grid_dir / descriptor.run_id
    scoring_path = config_dir / "scoring.yaml"
    scoring_config = _read_yaml(scoring_path)
    thresholds = scoring_config.get("thresholds") or {}
    theta_u_cfg = thresholds.get("theta_u") or {}
    source_condition = str(theta_u_cfg.get("source_condition", "full_reference_control"))
    quantile = float(theta_u_cfg.get("quantile", 0.95))
    theta_h_cfg = thresholds.get("theta_H") or {}
    theta_h = float(theta_h_cfg.get("value", float("inf")))
    # Read full-reference scores to calibrate theta_u_tilde
    ref_root = runs_root / descriptor.run_id
    ref_scores = _read_cell_scores(ref_root, source_condition, candidate_set)
    method_rows = ref_scores.loc[ref_scores["method"].astype(str).eq(method)]
    if method_rows.empty:
        raise ResultsGridError(
            f"No {method} full-reference cell scores for run_id={descriptor.run_id}"
        )
    u_tilde_vals = pd.to_numeric(method_rows["u_tilde"], errors="coerce").dropna()
    if u_tilde_vals.empty:
        raise ResultsGridError(
            f"No finite u_tilde values for run_id={descriptor.run_id}"
        )
    theta_u_tilde = float(u_tilde_vals.quantile(quantile))
    return theta_u_tilde, theta_h


def build_full_reference_false_abstention_by_run(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
    methods: tuple[str, ...],
    grid_dir: Path | None = None,
    score_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    overrides = score_overrides or {}
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _read_query_truth(run_root, condition)
        score_frame = _read_cell_scores(run_root, condition, candidate_set)
        for method in methods:
            joined = _joined_method_scores(score_frame, truth, method)
            shared = joined.loc[joined["is_shared_state"].astype(bool)]
            forced_labeled = shared.loc[_nonempty_string(shared["forced_label"])]
            score = overrides.get(method, primary_score_for_method(method))
            if score == "u_tilde" and grid_dir is not None:
                abstention = _u_tilde_abstention(
                    joined=shared,
                    runs_root=runs_root,
                    grid_dir=grid_dir,
                    descriptor=descriptor,
                    candidate_set=candidate_set,
                    method=method,
                )
                post_label_column = "forced_label"
            else:
                abstention = shared["abstain_u_or_entropy"].astype(bool)
                post_label_column = "final_label_abstention_aware"
            non_abstained = shared.loc[~abstention]
            post_labeled = non_abstained.loc[
                _nonempty_string(non_abstained[post_label_column])
            ]
            score_values = pd.to_numeric(joined[score], errors="coerce").dropna()
            labels = sorted(str(label) for label in shared["true_label"].dropna().unique())
            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(method, "unknown"),
                    "score": score,
                    "n_shared": int(len(shared)),
                    "full_reference_false_abstention_rate": float(
                        abstention.mean()
                    )
                    if len(shared)
                    else math.nan,
                    "score_median": float(score_values.median())
                    if len(score_values)
                    else math.nan,
                    "score_p95": float(score_values.quantile(0.95))
                    if len(score_values)
                    else math.nan,
                    "forced_accuracy": _accuracy(
                        forced_labeled["true_label"], forced_labeled["forced_label"]
                    ),
                    "forced_macro_f1": _macro_f1(
                        forced_labeled["true_label"],
                        forced_labeled["forced_label"],
                        labels,
                    ),
                    "post_abstention_accuracy": _accuracy(
                        post_labeled["true_label"],
                        post_labeled[post_label_column],
                    ),
                    "post_abstention_macro_f1": _macro_f1(
                        post_labeled["true_label"],
                        post_labeled[post_label_column],
                        labels,
                    ),
                    "coverage": float(len(non_abstained) / len(shared))
                    if len(shared)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_run_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
    include_overall: bool,
) -> pd.DataFrame:
    summaries = [_summarize(frame, group_columns, value_columns)]
    if include_overall:
        overall = frame.copy()
        overall["held_out_label"] = "overall"
        summaries.append(_summarize(overall, group_columns, value_columns))
    return pd.concat(summaries, ignore_index=True)

def _format_table_value(cell: object, quantity: str) -> str:
    """Format a cell value as 'mean ± std' with appropriate precision."""
    if cell is None:
        return "—"
    if isinstance(cell, (dict, pd.Series)):
        mean_val = cell.get("mean", math.nan) if hasattr(cell, "get") else math.nan
        std_val = cell.get("std", math.nan) if hasattr(cell, "get") else math.nan
    else:
        return "—"
    try:
        mean_val = float(mean_val)
        std_val = float(std_val)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(mean_val):
        return "—"

    if quantity in ("auroc", "auprc", "auprc_baseline"):
        dp = 3
    elif quantity in ("absent_abstention_rate", "shared_false_abstention_rate"):
        dp = 2
    elif quantity in ("forced_accuracy", "forced_macro_f1", "post_abstention_accuracy", "post_abstention_macro_f1", "coverage"):
        dp = 3
    else:
        dp = 3

    if math.isnan(std_val):
        return f"{mean_val:.{dp}f}"
    return f"{mean_val:.{dp}f} ± {std_val:.{dp}f}"


def _source_note() -> str:
    return "results/HIHA_DC/main/tables/main_detection_summary.csv"


def _read_metrics(run_root: Path) -> pd.DataFrame:
    return RunArtifacts(run_root, STAGE).evaluation().metrics().read()


def _read_forced_label_summary(run_root: Path) -> pd.DataFrame:
    return RunArtifacts(run_root, STAGE).evaluation().forced_label_summary().read()


def _read_query_truth(run_root: Path, condition: str) -> pd.DataFrame:
    return RunArtifacts(run_root, STAGE).evaluation_truth(condition).query_truth().read()


def _read_cell_scores(run_root: Path, condition: str, candidate_set: str) -> pd.DataFrame:
    return RunArtifacts(run_root, STAGE).scoring(condition, candidate_set).cell_scores().read()


def _validate_main_grid_artifacts(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    conditions: tuple[str, ...],
) -> None:
    _, failures = _partition_main_grid_runs(
        runs_root=runs_root,
        descriptors=descriptors,
        candidate_set=candidate_set,
        conditions=conditions,
    )
    if failures:
        raise ArtifactBatchError(failures)


def _partition_main_grid_runs(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    conditions: tuple[str, ...],
) -> tuple[tuple[RunDescriptor, ...], list[ArtifactFailure]]:
    completed: list[RunDescriptor] = []
    failures: list[ArtifactFailure] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        artifacts = RunArtifacts(run_root, STAGE)
        checks = [artifacts.evaluation().metrics(), artifacts.evaluation().forced_label_summary()]
        run_failures: list[ArtifactFailure] = []
        for condition in conditions:
            checks.append(artifacts.evaluation_truth(condition).query_truth())
            checks.append(artifacts.scoring(condition, candidate_set).cell_scores())
        for handle in checks:
            try:
                handle.read()
            except ArtifactContractError as exc:
                run_failures.append(exc.to_failure())
        if run_failures:
            failures.extend(run_failures)
            continue
        completed.append(descriptor)
    return tuple(completed), failures


def _joined_method_scores(scores: pd.DataFrame, truth: pd.DataFrame, method: str) -> pd.DataFrame:
    method_scores = scores.loc[scores["method"] == method].copy()
    if method_scores.empty:
        raise ResultsGridError(f"No cell scores found for method={method}")
    joined = method_scores.merge(truth, on="cell_id", how="left", validate="many_to_one")
    if joined["true_label"].isna().any():
        raise ResultsGridError(f"Missing query truth for method={method}")
    return joined


def _shared_false_abstention_rate(scores: pd.DataFrame, truth: pd.DataFrame, method: str) -> float:
    joined = _joined_method_scores(scores, truth, method)
    shared = joined.loc[joined["is_shared_state"].astype(bool)]
    if shared.empty:
        return math.nan
    return float(shared["abstain_u_or_entropy"].astype(bool).mean())


def _metric_values(metrics: pd.DataFrame, primary_score: str) -> dict[str, float]:
    return {
        metric: _single_metric(metrics, score=primary_score, metric=metric)
        for metric in (
            "auroc",
            "auprc",
            "median_absent",
            "median_shared",
            "absent_minus_shared_median",
        )
    }


def _single_metric(metrics: pd.DataFrame, *, score: str, metric: str) -> float:
    values = metrics.loc[(metrics["score"] == score) & (metrics["metric"] == metric), "value"]
    if len(values) != 1:
        raise ResultsGridError(
            f"Expected exactly one metric value for score={score}, metric={metric}; found {len(values)}"
        )
    return float(values.iloc[0])


def _summarize(
    frame: pd.DataFrame, group_columns: tuple[str, ...], value_columns: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in frame.groupby(list(group_columns), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(group_columns, group_values, strict=True))
        for value_column in value_columns:
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            n = int(len(values))
            std = float(values.std(ddof=1)) if n > 1 else math.nan
            rows.append(
                base
                | {
                    "quantity": value_column,
                    "mean": float(values.mean()) if n else math.nan,
                    "std": std,
                    "sem": float(std / math.sqrt(n)) if n > 1 else math.nan,
                    "n_runs": n,
                }
            )
    return pd.DataFrame(rows)


def _accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(accuracy_score(y_true.astype(str), y_pred.astype(str)))


def _represented_truth_labels(shared: pd.DataFrame) -> list[str]:
    """Return every truth label represented in the shared evaluation cohort."""
    return sorted(str(label) for label in shared["true_label"].dropna().unique())


def _macro_f1(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    if len(y_true) == 0 or not labels:
        return math.nan
    return float(
        f1_score(
            y_true.astype(str),
            y_pred.astype(str),
            labels=labels,
            average="macro",
            zero_division=0.0,
        )
    )


def _nonempty_string(values: pd.Series) -> pd.Series:
    return values.astype(str).str.len() > 0


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    formatted = frame.map(_format_value)
    columns = list(formatted.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.astype(str).to_numpy()]
    return [header, separator, *rows]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6g}"
    return str(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected YAML file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ResultsGridError(f"Expected YAML mapping in {path}")
    return payload


def _required_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ResultsGridError(f"Expected nonempty string {key} in {path}")
    return value
