from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.artifacts.run_artifacts import RunArtifacts
from coreot.config.load import load_yaml
from coreot.data.schemas import (
    ABSTENTION_CALLS_COLUMNS,
    BROAD_ANCHOR_PRIORS_COLUMNS,
    CELL_SCORES_COLUMNS,
    CELL_TRANSPORT_SCORES_COLUMNS,
    INITIAL_PRIORS_COLUMNS,
    MODEL_VISIBLE_CELLS_COLUMNS,
    PRIOR_ONLY_SCORES_COLUMNS,
)
from coreot.data.validation import validate_required_columns, validate_stage_can_read
from coreot.scoring.prior_adjusted import BIOLOGICAL_FOLD_COLUMNS, compute_prior_adjusted_deficit


STAGE = "score-abstain"


class ScoringRunnerError(ValueError):
    """Raised when scoring artifacts cannot be constructed."""


@dataclass(frozen=True)
class ScoringResult:
    scoring_root: Path
    conditions: tuple[str, ...]
    candidate_sets: tuple[str, ...]
    methods: tuple[str, ...]


def run_scoring(config_path: str | Path) -> ScoringResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    output_root = _required_path(config, ("outputs", "root"))
    conditions = tuple(config.get("conditions", ()))
    candidate_sets = tuple(config.get("candidate_sets", ()))
    methods = tuple(config.get("methods", ()))
    if not conditions:
        raise ScoringRunnerError("score-abstain requires conditions")
    if not candidate_sets:
        raise ScoringRunnerError("score-abstain requires candidate_sets")
    if not methods:
        raise ScoringRunnerError("score-abstain requires methods")
    thresholds = config.get("thresholds", {})
    if thresholds is not None and not isinstance(thresholds, dict):
        raise ScoringRunnerError("score-abstain thresholds must be a mapping")
    prior_adjustment = config.get("prior_adjustment", {})
    if prior_adjustment is None:
        prior_adjustment = {}
    if not isinstance(prior_adjustment, dict):
        raise ScoringRunnerError("score-abstain prior_adjustment must be a mapping")

    run_root = output_root / run_id
    scoring_root = run_root / "scoring"
    for condition in conditions:
        for candidate_set in candidate_sets:
            if not isinstance(candidate_set, str) or not candidate_set:
                raise ScoringRunnerError("score-abstain candidate_sets must be strings")
            _score_candidate_set(
                run_root,
                scoring_root,
                condition,
                candidate_set,
                methods,
                conditions,
                thresholds or {},
                prior_adjustment,
            )

    return ScoringResult(
        scoring_root=scoring_root,
        conditions=conditions,
        candidate_sets=candidate_sets,
        methods=methods,
    )


def _score_candidate_set(
    run_root: Path,
    scoring_root: Path,
    condition: str,
    candidate_set: str,
    methods: tuple[object, ...],
    conditions: tuple[object, ...],
    thresholds: dict[str, Any],
    prior_adjustment: dict[str, Any],
) -> None:
    transport_methods = {
        "nn",
        "balanced_ot",
        "uniform_uot",
        "coreot_constant_tau",
        "coreot_full",
        "coreot_match_only",
    }
    method_names = [method for method in methods if method in transport_methods | {"prior_only"}]
    if not method_names:
        raise ScoringRunnerError(
            "score-abstain currently supports nn, balanced_ot, uniform_uot, "
            "coreot_constant_tau, coreot_full, coreot_match_only, and prior_only"
        )

    prior_risk = _read_prior_risk(run_root, condition, candidate_set)
    source_fold_metadata = _read_source_fold_metadata(run_root, condition)
    theta_h = _label_entropy_threshold(thresholds)
    percentile_residual_enabled = _prior_adjustment_percentile_residual(prior_adjustment)
    score_frames = []
    prior_adjustment_metadata: dict[str, dict[str, Any]] = {}
    label_probability_source: Path | None = None
    for method in method_names:
        transport_method_root = run_root / "transport" / condition / candidate_set / str(method)
        scores_path = transport_method_root / "cell_transport_scores.parquet"
        validate_stage_can_read(scores_path, STAGE)
        if not scores_path.is_file():
            raise FileNotFoundError(f"Required scoring input does not exist: {scores_path}")

        if method in transport_methods:
            method_name = str(method)
            theta_u = _u_threshold(
                run_root=run_root,
                candidate_set=candidate_set,
                method=method_name,
                declared_conditions=conditions,
                thresholds=thresholds,
            )
            frame = _score_transport_scores(
                pd.read_parquet(scores_path),
                prior_risk,
                source_fold_metadata,
                method_name,
                theta_u=theta_u,
                theta_h=theta_h,
                prior_adjustment_enabled=bool(prior_adjustment.get("enabled", False)),
                prior_adjustment_n_folds=_prior_adjustment_n_folds(prior_adjustment),
                prior_adjustment_stratify_by_anchor=_prior_adjustment_stratify_by_anchor(
                    prior_adjustment
                ),
                prior_adjustment_min_anchor_size=_prior_adjustment_min_anchor_size(
                    prior_adjustment
                ),
                prior_adjustment_percentile_residual=percentile_residual_enabled,
            )
            if "_prior_adjustment_metadata" in frame.attrs:
                prior_adjustment_metadata[method_name] = frame.attrs["_prior_adjustment_metadata"]
            candidate_label_probabilities = transport_method_root / "label_probabilities.npz"
            if candidate_label_probabilities.is_file():
                label_probability_source = candidate_label_probabilities
        elif method == "prior_only":
            frame = _score_prior_only(pd.read_parquet(scores_path))
        else:
            raise ScoringRunnerError(f"Unsupported scoring method: {method}")
        score_frames.append(frame)

    cell_scores = pd.concat(score_frames, ignore_index=True)
    cell_score_columns = list(CELL_SCORES_COLUMNS)
    if percentile_residual_enabled:
        cell_score_columns.append("u_percentile_residual")
    cell_scores = cell_scores.loc[:, cell_score_columns]

    output_root = scoring_root / condition / candidate_set
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(run_root, STAGE)
    artifacts.scoring(condition, candidate_set).cell_scores().write(cell_scores)
    cell_scores.loc[:, ABSTENTION_CALLS_COLUMNS].to_csv(
        output_root / "abstention_calls.csv", index=False
    )
    if label_probability_source is not None:
        shutil.copy2(label_probability_source, output_root / "label_probabilities.npz")
    else:
        np.savez_compressed(output_root / "label_probabilities.npz")

    write_manifest(
        output_root / "scoring_manifest.yaml",
        Manifest(
            stage=STAGE,
            artifacts={
                "cell_scores": str(output_root / "cell_scores.parquet"),
                "abstention_calls": str(output_root / "abstention_calls.csv"),
                "label_probabilities": str(output_root / "label_probabilities.npz"),
            },
            metadata={
                "condition": condition,
                "candidate_set": candidate_set,
                "threshold_mode": _threshold_mode(thresholds, conditions),
                "thresholds": thresholds,
                "prior_adjustment": _prior_adjustment_manifest_metadata(
                    prior_adjustment, prior_adjustment_metadata
                ),
            },
        ),
    )


def _read_prior_risk(run_root: Path, condition: str, candidate_set: str) -> pd.DataFrame:
    prior_only_path = (
        run_root / "transport" / condition / candidate_set / "prior_only" / "cell_transport_scores.parquet"
    )
    validate_stage_can_read(prior_only_path, STAGE)
    if prior_only_path.is_file():
        prior_only = pd.read_parquet(prior_only_path)
        validate_required_columns(prior_only, PRIOR_ONLY_SCORES_COLUMNS, str(prior_only_path))
        return prior_only.loc[:, ["cell_id", "prior_risk"]]

    model_visible = run_root / "benchmark" / condition / "model_visible"
    initial_priors_path = model_visible / "initial_priors.csv"
    cells_path = model_visible / "cells.csv"
    for path in (initial_priors_path, cells_path):
        validate_stage_can_read(path, STAGE)
        if not path.is_file():
            raise FileNotFoundError(f"Required scoring prior input does not exist: {path}")
    initial_priors = pd.read_csv(initial_priors_path)
    cells = pd.read_csv(cells_path)
    validate_required_columns(initial_priors, INITIAL_PRIORS_COLUMNS, str(initial_priors_path))
    validate_required_columns(cells, MODEL_VISIBLE_CELLS_COLUMNS, str(cells_path))
    query_ids = cells.loc[cells["domain"].astype(str).eq("query"), ["cell_id"]].copy()
    query_ids["cell_id"] = query_ids["cell_id"].astype(str)
    initial_priors["cell_id"] = initial_priors["cell_id"].astype(str)
    prior_risk = query_ids.merge(
        initial_priors.loc[:, ["cell_id", "prior_risk"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    if prior_risk["prior_risk"].isna().any():
        raise ScoringRunnerError("initial_priors.csv must contain prior_risk for every source cell")
    return prior_risk


def _read_source_fold_metadata(run_root: Path, condition: str) -> pd.DataFrame:
    cells_path = run_root / "benchmark" / condition / "model_visible" / "cells.csv"
    validate_stage_can_read(cells_path, STAGE)
    if not cells_path.is_file():
        return pd.DataFrame(columns=("cell_id", *BIOLOGICAL_FOLD_COLUMNS, "anchor_class_pred"))
    cells = pd.read_csv(cells_path)
    validate_required_columns(cells, MODEL_VISIBLE_CELLS_COLUMNS, str(cells_path))
    cells["cell_id"] = cells["cell_id"].astype(str)
    source_cells = cells.loc[cells["domain"].astype(str).eq("query")].copy()
    metadata = source_cells.loc[:, ("cell_id", *BIOLOGICAL_FOLD_COLUMNS)]
    anchors_path = run_root / "benchmark" / condition / "model_visible" / "broad_anchor_priors.csv"
    validate_stage_can_read(anchors_path, STAGE)
    if not anchors_path.is_file():
        return metadata
    anchors = pd.read_csv(anchors_path)
    validate_required_columns(anchors, BROAD_ANCHOR_PRIORS_COLUMNS, str(anchors_path))
    anchors["cell_id"] = anchors["cell_id"].astype(str)
    anchors = anchors.loc[:, ["cell_id", "broad_anchor_class"]].rename(
        columns={"broad_anchor_class": "anchor_class_pred"}
    )
    return metadata.merge(anchors, on="cell_id", how="left", validate="one_to_one")


def _score_transport_scores(
    transport_scores: pd.DataFrame,
    prior_risk: pd.DataFrame,
    source_fold_metadata: pd.DataFrame,
    method: str,
    *,
    theta_u: float | None,
    theta_h: float,
    prior_adjustment_enabled: bool,
    prior_adjustment_n_folds: int,
    prior_adjustment_stratify_by_anchor: bool,
    prior_adjustment_min_anchor_size: int,
    prior_adjustment_percentile_residual: bool,
) -> pd.DataFrame:
    validate_required_columns(
        transport_scores, CELL_TRANSPORT_SCORES_COLUMNS, f"{method} transport scores"
    )
    merged = transport_scores.merge(prior_risk, on="cell_id", how="left", validate="one_to_one")
    if merged["prior_risk"].isna().any():
        raise ScoringRunnerError(f"{method} scoring requires prior_only prior_risk for every source cell")
    u = merged["u"].astype(float)
    label_entropy = merged["label_entropy"].astype(float)
    max_label_probability = merged["max_label_probability"].astype(float)
    label_uncertainty = 1.0 - max_label_probability
    abstain_u = pd.Series(False, index=merged.index)
    if theta_u is not None:
        abstain_u = u > theta_u
    if method == "balanced_ot":
        abstain_u = pd.Series(False, index=merged.index)
    abstain_u_or_entropy = abstain_u | (label_entropy > theta_h)
    forced_label = merged["forced_label"].astype(str)
    final_label = forced_label.mask(abstain_u_or_entropy, "")
    u_tilde = u
    u_percentile_residual = pd.Series(np.nan, index=merged.index, dtype=float)
    if prior_adjustment_enabled:
        calibration_input = pd.DataFrame(
            {
                "cell_id": merged["cell_id"].astype(str),
                "u": u,
                "prior_risk": merged["prior_risk"].astype(float),
            },
            index=merged.index,
        ).merge(source_fold_metadata, on="cell_id", how="left", validate="one_to_one")
        adjusted = compute_prior_adjusted_deficit(
            calibration_input,
            n_folds=prior_adjustment_n_folds,
            stratify_by_anchor=prior_adjustment_stratify_by_anchor,
            min_anchor_size=prior_adjustment_min_anchor_size,
            compute_percentile_residual=prior_adjustment_percentile_residual,
        )
        u_tilde = adjusted.u_tilde
        if prior_adjustment_percentile_residual:
            u_percentile_residual = adjusted.percentile_residual
    output = pd.DataFrame(
        {
            "cell_id": merged["cell_id"].astype(str),
            "condition_id": merged["condition_id"].astype(str),
            "method": method,
            "u": u,
            "u_tilde": u_tilde,
            "u_percentile_residual": u_percentile_residual,
            "prior_risk": merged["prior_risk"].astype(float),
            "e": merged["e"].astype(float),
            "hub_exposure": merged["hub_exposure"].astype(float),
            "max_label_probability": max_label_probability,
            "label_uncertainty": label_uncertainty,
            "label_entropy": label_entropy,
            "forced_label": forced_label,
            "nn_distance": merged["nn_distance"].astype(float),
            "abstain_u": abstain_u,
            "abstain_u_or_entropy": abstain_u_or_entropy,
            "final_label_abstention_aware": final_label,
        }
    )
    if prior_adjustment_enabled:
        output.attrs["_prior_adjustment_metadata"] = adjusted.metadata
    return output


def _prior_adjustment_manifest_metadata(
    config: dict[str, Any], method_metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not bool(config.get("enabled", False)):
        return {"enabled": False, "percentile_residual_enabled": False}
    percentile_residual_enabled = _prior_adjustment_percentile_residual(config)
    selected = _selected_prior_adjustment_metadata(method_metadata)
    return {
        "enabled": True,
        "percentile_residual_enabled": percentile_residual_enabled,
        **selected,
        "methods": method_metadata,
    }


def _selected_prior_adjustment_metadata(
    method_metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    for metadata in method_metadata.values():
        if metadata.get("fallback") != "all_missing":
            return metadata
    if method_metadata:
        return next(iter(method_metadata.values()))
    return {
        "model": "cross_fitted_isotonic",
        "cross_fitting": "not_run",
        "n_folds": 0,
        "fold_source": "none",
        "anchor_stratification": "global",
        "fallback": "no_supported_methods",
    }


def _prior_adjustment_n_folds(config: dict[str, Any]) -> int:
    value = config.get("n_folds", 5)
    if not isinstance(value, int):
        raise ScoringRunnerError("prior_adjustment.n_folds must be an integer")
    return value


def _prior_adjustment_stratify_by_anchor(config: dict[str, Any]) -> bool:
    value = config.get("stratify_by_anchor", False)
    if not isinstance(value, bool):
        raise ScoringRunnerError("prior_adjustment.stratify_by_anchor must be a boolean")
    fallback = config.get("fallback", "global")
    if value and fallback != "global":
        raise ScoringRunnerError("prior_adjustment.fallback currently supports only 'global'")
    return value


def _prior_adjustment_min_anchor_size(config: dict[str, Any]) -> int:
    value = config.get("min_anchor_size", 20)
    if not isinstance(value, int):
        raise ScoringRunnerError("prior_adjustment.min_anchor_size must be an integer")
    if value < 2:
        raise ScoringRunnerError("prior_adjustment.min_anchor_size must be at least 2")
    return value


def _prior_adjustment_percentile_residual(config: dict[str, Any]) -> bool:
    value = config.get("percentile_residual", False)
    if not isinstance(value, bool):
        raise ScoringRunnerError("prior_adjustment.percentile_residual must be a boolean")
    return bool(config.get("enabled", False)) and value


def _score_prior_only(transport_scores: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(transport_scores, PRIOR_ONLY_SCORES_COLUMNS, "prior_only transport scores")
    return pd.DataFrame(
        {
            "cell_id": transport_scores["cell_id"].astype(str),
            "condition_id": transport_scores["condition_id"].astype(str),
            "method": "prior_only",
            "u": np.nan,
            "u_tilde": np.nan,
            "u_percentile_residual": np.nan,
            "prior_risk": transport_scores["prior_risk"].astype(float),
            "e": np.nan,
            "hub_exposure": np.nan,
            "max_label_probability": np.nan,
            "label_uncertainty": np.nan,
            "label_entropy": np.nan,
            "forced_label": "",
            "nn_distance": np.nan,
            "abstain_u": False,
            "abstain_u_or_entropy": False,
            "final_label_abstention_aware": "",
        }
    )


def _u_threshold(
    *,
    run_root: Path,
    candidate_set: str,
    method: str,
    declared_conditions: tuple[object, ...],
    thresholds: dict[str, Any],
) -> float | None:
    theta_config = thresholds.get("theta_u")
    if theta_config is None:
        return None
    if not isinstance(theta_config, dict):
        raise ScoringRunnerError("thresholds.theta_u must be a mapping")
    source_condition = str(theta_config.get("source_condition", "full_reference_control"))
    if source_condition not in declared_conditions:
        return None
    source_method = str(theta_config.get("source_method", method))
    if source_method == "method":
        source_method = method
    quantile = _numeric_config(theta_config, "quantile", 0.95)
    if not 0.0 <= quantile <= 1.0:
        raise ScoringRunnerError("thresholds.theta_u.quantile must be between 0 and 1")
    scores_path = (
        run_root
        / "transport"
        / source_condition
        / candidate_set
        / source_method
        / "cell_transport_scores.parquet"
    )
    validate_stage_can_read(scores_path, STAGE)
    if not scores_path.is_file():
        raise FileNotFoundError(f"Threshold calibration transport scores do not exist: {scores_path}")
    scores = pd.read_parquet(scores_path)
    validate_required_columns(scores, CELL_TRANSPORT_SCORES_COLUMNS, str(scores_path))
    values = pd.to_numeric(scores["u"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.quantile(quantile))


def _threshold_mode(thresholds: dict[str, Any], conditions: tuple[object, ...]) -> str:
    theta_config = thresholds.get("theta_u")
    if not isinstance(theta_config, dict):
        return "placeholder_no_abstention"
    source_condition = str(theta_config.get("source_condition", "full_reference_control"))
    if source_condition not in conditions:
        return "unavailable_undeclared_source_condition"
    return "full_reference_quantile"


def _label_entropy_threshold(thresholds: dict[str, Any]) -> float:
    theta_config = thresholds.get("theta_H", {})
    if theta_config is None:
        return float("inf")
    if isinstance(theta_config, dict):
        return _numeric_config(theta_config, "value", float("inf"))
    if isinstance(theta_config, int | float):
        return float(theta_config)
    raise ScoringRunnerError("thresholds.theta_H must be numeric or a mapping")


def _numeric_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float):
        raise ScoringRunnerError(f"Expected numeric threshold config value: {key}")
    return float(value)


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise ScoringRunnerError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise ScoringRunnerError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
