from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.artifacts.run_artifacts import ArtifactMissing, RunArtifacts
from coreot.config.load import load_yaml
from coreot.data.schemas import (
    EVALUATION_METRICS_COLUMNS,
    FORCED_LABEL_SUMMARY_COLUMNS,
)
from coreot.evaluation.metrics import safe_auprc, safe_auroc, safe_median


STAGE = "evaluation"


class EvaluationRunnerError(ValueError):
    """Raised when evaluation artifacts cannot be constructed."""


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_root: Path
    conditions: tuple[str, ...]
    candidate_sets: tuple[str, ...]
    methods: tuple[str, ...]


def run_evaluation(config_path: str | Path) -> EvaluationResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    output_root = _required_path(config, ("outputs", "root"))
    conditions = tuple(config.get("conditions", ()))
    candidate_sets = tuple(config.get("candidate_sets", ()))
    methods = tuple(config.get("methods", ()))
    primary_negative_set = str(config.get("primary_negative_set", "all_shared"))
    if not conditions:
        raise EvaluationRunnerError("evaluation requires conditions")
    if not candidate_sets:
        raise EvaluationRunnerError("evaluation requires candidate_sets")
    if not methods:
        raise EvaluationRunnerError("evaluation requires methods")

    run_root = output_root / run_id
    evaluation_root = run_root / "evaluation"
    artifacts = RunArtifacts(run_root, STAGE)
    all_metrics: list[pd.DataFrame] = []
    all_forced_label_summaries: list[pd.DataFrame] = []
    for condition in conditions:
        query_truth_artifact = artifacts.evaluation_truth(condition).query_truth()
        try:
            query_truth = query_truth_artifact.read()
        except ArtifactMissing as exc:
            raise FileNotFoundError(
                f"Evaluation truth does not exist: {query_truth_artifact.path}"
            ) from exc

        for candidate_set in candidate_sets:
            if not isinstance(candidate_set, str) or not candidate_set:
                raise EvaluationRunnerError("evaluation candidate_sets must be strings")
            cell_scores_artifact = artifacts.scoring(condition, candidate_set).cell_scores()
            try:
                cell_scores = cell_scores_artifact.read()
            except ArtifactMissing as exc:
                raise FileNotFoundError(
                    f"Scoring artifact does not exist: {cell_scores_artifact.path}"
                ) from exc

            for method in methods:
                method_scores = cell_scores.loc[cell_scores["method"] == method].copy()
                if method_scores.empty:
                    continue
                joined = method_scores.merge(query_truth, on="cell_id", how="left", validate="many_to_one")
                if joined["is_absent_state"].isna().any():
                    raise EvaluationRunnerError(f"Missing query truth for condition={condition}, method={method}")
                evaluation_joined = joined
                if primary_negative_set == "same_broad":
                    if "true_broad_label" not in joined.columns:
                        raise EvaluationRunnerError(
                            "same_broad evaluation requires query truth true_broad_label"
                        )
                    removed_state_rows = joined["true_label"].astype(str).eq(
                        joined["removed_state"].astype(str)
                    )
                    positive_broad = joined.loc[
                        removed_state_rows, "true_broad_label"
                    ].astype(str).unique()
                    if len(positive_broad) != 1:
                        raise EvaluationRunnerError("absent state must have one broad label")
                    evaluation_joined = joined.loc[
                        joined["true_broad_label"].astype(str).eq(positive_broad[0])
                    ]
                elif primary_negative_set != "all_shared":
                    raise EvaluationRunnerError(
                        "primary_negative_set must be 'all_shared' or 'same_broad'"
                    )
                all_metrics.append(
                    _evaluate_method(condition, candidate_set, str(method), evaluation_joined)
                )
                forced = _forced_label_summary(condition, candidate_set, str(method), joined)
                if not forced.empty:
                    all_forced_label_summaries.append(forced)

    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else _empty_metrics()
    forced_label_summary = (
        pd.concat(all_forced_label_summaries, ignore_index=True)
        if all_forced_label_summaries
        else _empty_forced_label_summary()
    )

    evaluation_root.mkdir(parents=True, exist_ok=True)
    artifacts.evaluation().metrics().write(metrics.loc[:, EVALUATION_METRICS_COLUMNS])
    artifacts.evaluation().forced_label_summary().write(
        forced_label_summary.loc[:, FORCED_LABEL_SUMMARY_COLUMNS]
    )
    write_manifest(
        evaluation_root / "evaluation_manifest.yaml",
        Manifest(
            stage=STAGE,
            artifacts={
                "metrics": str(evaluation_root / "metrics.csv"),
                "forced_label_summary": str(evaluation_root / "forced_label_summary.csv"),
            },
            metadata={
                "metrics_scope": "baseline_missing_state_detection",
                "implemented_metrics": [
                    "auroc",
                    "auprc",
                    "median_absent",
                    "median_shared",
                    "absent_minus_shared_median",
                    "absent_abstention_rate",
                ],
            },
        ),
    )

    return EvaluationResult(
        evaluation_root=evaluation_root,
        conditions=conditions,
        candidate_sets=candidate_sets,
        methods=methods,
    )


def _evaluate_method(
    condition: str, candidate_set: str, method: str, joined: pd.DataFrame
) -> pd.DataFrame:
    score_columns = [
        "u",
        "u_tilde",
        "prior_risk",
        "label_uncertainty",
        "label_entropy",
        "nn_distance",
    ]
    rows: list[dict[str, object]] = []
    absent = joined["is_absent_state"].astype(bool)
    for score in score_columns:
        if score not in joined.columns:
            continue
        values = joined[score]
        median_absent = safe_median(values.loc[absent])
        median_shared = safe_median(values.loc[~absent])
        metrics = {
            "auroc": safe_auroc(absent, values),
            "auprc": safe_auprc(absent, values),
            "median_absent": median_absent,
            "median_shared": median_shared,
            "absent_minus_shared_median": median_absent - median_shared,
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": method,
                    "score": score,
                    "metric": metric,
                    "value": value,
                }
            )

    rows.append(
        {
            "condition_id": condition,
            "candidate_set": candidate_set,
            "method": method,
            "score": "abstain_u_or_entropy",
            "metric": "absent_abstention_rate",
            "value": float(joined.loc[absent, "abstain_u_or_entropy"].astype(bool).mean())
            if absent.any()
            else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def _forced_label_summary(
    condition: str, candidate_set: str, method: str, joined: pd.DataFrame
) -> pd.DataFrame:
    if "forced_label" not in joined.columns or joined["forced_label"].replace("", pd.NA).isna().all():
        return _empty_forced_label_summary()
    rows = []
    for subset, subset_frame in (
        ("absent_state", joined.loc[joined["is_absent_state"].astype(bool)]),
        ("shared_state", joined.loc[joined["is_shared_state"].astype(bool)]),
    ):
        if subset_frame.empty:
            continue
        grouped = subset_frame.groupby("forced_label", dropna=False)
        for forced_label, group in grouped:
            rows.append(
                {
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": method,
                    "subset": subset,
                    "forced_label": forced_label,
                    "n_cells": int(len(group)),
                    "mean_max_label_probability": float(
                        pd.to_numeric(group["max_label_probability"], errors="coerce").mean()
                    ),
                    "abstention_rate": float(group["abstain_u_or_entropy"].astype(bool).mean()),
                }
            )
    return pd.DataFrame(rows)


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=EVALUATION_METRICS_COLUMNS)


def _empty_forced_label_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FORCED_LABEL_SUMMARY_COLUMNS)


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise EvaluationRunnerError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise EvaluationRunnerError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
