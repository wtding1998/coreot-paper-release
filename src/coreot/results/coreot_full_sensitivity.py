from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.results.grid_common import (
    DISPLAY_NAME_BY_METHOD,
    MAIN_TABLE_1_COLUMN_HEADERS,
    MAIN_TABLE_1_QUANTITIES,
    MAIN_TABLE_2_COLUMN_HEADERS,
    MAIN_TABLE_2_QUANTITIES,
    ResultsGridError,
    RunDescriptor,
    _accuracy,
    _format_table_value,
    _macro_f1,
    _nonempty_string,
    _read_cell_scores,
    _read_metrics,
    _read_query_truth,
    _single_metric,
    _read_yaml,
    build_detection_by_run,
    build_shared_label_transfer_by_run,
    discover_run_descriptors,
    summarize_run_metrics,
)
from coreot.results.sweep_common import (
    DEFAULT_FIXED_REFERENCE_ROOT,
    FIXED_REFERENCE_METHODS,
    _failed_run_ids,
    _is_missing_tau,
    _partition_completed_sweep_runs,
    read_fixed_reference_bundle,
)

HELD_OUT_LABELS: tuple[str, ...] = ("CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2")
COREOT_FULL_METHOD = "coreot_full"
DEFAULT_TAU_MIN_VALUES: tuple[float, ...] = (5.0, 6.0, 7.0)
DEFAULT_TAU_MAX_VALUES: tuple[float, ...] = (7.0, 8.0, 9.0)
DEFAULT_ALPHA = 5.0


@dataclass(frozen=True)
class CoreotFullTauRangeArtifactPaths:
    output_root: Path
    detection_by_run: Path
    detection_summary: Path
    u_vs_u_tilde_detection_by_run: Path
    u_vs_u_tilde_detection_summary: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    u_tilde_counterfactual_shared_label_transfer_by_run: Path
    u_tilde_counterfactual_shared_label_transfer_summary: Path
    table_1: Path
    table_2: Path
    table_3: Path
    table_bundle: Path | None
    report: Path
    manifest: Path
    figures: dict[str, Path]


def parse_coreot_full_tau_range_params(run_config_dir: Path) -> tuple[float, float, float]:
    transport_path = run_config_dir / "transport.yaml"
    config = _read_yaml(transport_path)
    methods = config.get("methods", ())
    for method in methods:
        if not isinstance(method, dict):
            continue
        if method.get("name") != COREOT_FULL_METHOD:
            continue
        tau_min = float(method["tau_min"])
        tau_max = float(method["tau_max"])
        alpha = float(method.get("alpha", float("nan")))
        if tau_min <= 0:
            raise ResultsGridError(f"tau_min must be positive in {transport_path}")
        if tau_max < tau_min:
            raise ResultsGridError(f"tau_max must be >= tau_min in {transport_path}")
        if alpha <= 0:
            raise ResultsGridError(f"alpha must be positive in {transport_path}")
        return tau_min, tau_max, alpha
    raise ResultsGridError(f"No coreot_full method found in {transport_path}")


def _build_param_maps(
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    tau_min_map: dict[str, float] = {}
    tau_max_map: dict[str, float] = {}
    alpha_map: dict[str, float] = {}
    for desc in descriptors:
        tau_min, tau_max, alpha = parse_coreot_full_tau_range_params(grid_dir / desc.run_id)
        tau_min_map[desc.run_id] = tau_min
        tau_max_map[desc.run_id] = tau_max
        alpha_map[desc.run_id] = alpha
    return tau_min_map, tau_max_map, alpha_map


def _numeric_threshold(config: object, key: str, default: float) -> float:
    if not isinstance(config, dict):
        return default
    value = config.get(key, default)
    if not isinstance(value, int | float):
        raise ResultsGridError(f"Expected numeric scoring threshold value: {key}")
    return float(value)


def _counterfactual_threshold_settings(run_config_dir: Path) -> tuple[str, float, float]:
    scoring_path = run_config_dir / "scoring.yaml"
    config = _read_yaml(scoring_path)
    thresholds = config.get("thresholds", {})
    if thresholds is None:
        thresholds = {}
    if not isinstance(thresholds, dict):
        raise ResultsGridError(f"scoring thresholds must be a mapping in {scoring_path}")
    theta_u = thresholds.get("theta_u", {})
    if theta_u is None:
        theta_u = {}
    if theta_u and not isinstance(theta_u, dict):
        raise ResultsGridError(f"thresholds.theta_u must be a mapping in {scoring_path}")
    source_condition = (
        str(theta_u.get("source_condition", "full_reference_control"))
        if isinstance(theta_u, dict)
        else "full_reference_control"
    )
    quantile = _numeric_threshold(theta_u, "quantile", 0.95)
    if not 0.0 <= quantile <= 1.0:
        raise ResultsGridError(f"thresholds.theta_u.quantile must be between 0 and 1 in {scoring_path}")
    theta_h_config = thresholds.get("theta_H", {})
    if isinstance(theta_h_config, int | float):
        theta_h = float(theta_h_config)
    else:
        theta_h = _numeric_threshold(theta_h_config, "value", float("inf"))
    return source_condition, quantile, theta_h


def _u_tilde_threshold(
    *,
    runs_root: Path,
    grid_dir: Path,
    descriptor: RunDescriptor,
    candidate_set: str,
) -> tuple[float, float]:
    source_condition, quantile, theta_h = _counterfactual_threshold_settings(
        grid_dir / descriptor.run_id
    )
    source_scores = _read_cell_scores(
        runs_root / descriptor.run_id,
        source_condition,
        candidate_set,
    )
    method_scores = source_scores.loc[source_scores["method"].astype(str).eq(COREOT_FULL_METHOD)]
    if method_scores.empty:
        raise ResultsGridError(
            f"No {COREOT_FULL_METHOD} full-reference cell scores found for run_id={descriptor.run_id}"
        )
    values = pd.to_numeric(method_scores["u_tilde"], errors="coerce").dropna()
    if values.empty:
        raise ResultsGridError(
            f"No finite u_tilde values available for threshold calibration in run_id={descriptor.run_id}"
        )
    return float(values.quantile(quantile)), theta_h


def _counterfactual_joined_scores(
    *,
    runs_root: Path,
    grid_dir: Path,
    descriptor: RunDescriptor,
    candidate_set: str,
    condition: str,
) -> pd.DataFrame:
    truth = _read_query_truth(runs_root / descriptor.run_id, condition)
    scores = _read_cell_scores(runs_root / descriptor.run_id, condition, candidate_set)
    method_scores = scores.loc[scores["method"].astype(str).eq(COREOT_FULL_METHOD)].copy()
    if method_scores.empty:
        raise ResultsGridError(f"No {COREOT_FULL_METHOD} cell scores found for run_id={descriptor.run_id}")
    theta_u_tilde, theta_h = _u_tilde_threshold(
        runs_root=runs_root,
        grid_dir=grid_dir,
        descriptor=descriptor,
        candidate_set=candidate_set,
    )
    joined = method_scores.merge(truth, on="cell_id", how="left", validate="many_to_one")
    if joined["true_label"].isna().any():
        raise ResultsGridError(f"Missing query truth for run_id={descriptor.run_id}")
    u_tilde = pd.to_numeric(joined["u_tilde"], errors="coerce")
    if u_tilde.isna().any():
        raise ResultsGridError(f"Missing u_tilde values for run_id={descriptor.run_id}")
    label_entropy = pd.to_numeric(joined["label_entropy"], errors="coerce")
    if label_entropy.isna().any():
        raise ResultsGridError(f"Missing label_entropy values for run_id={descriptor.run_id}")
    abstain = (u_tilde > theta_u_tilde) | (label_entropy > theta_h)
    result = joined.copy()
    result["theta_u_tilde"] = theta_u_tilde
    result["theta_h"] = theta_h
    result["abstain_u_tilde_or_entropy"] = abstain
    result["final_label_u_tilde_abstention_aware"] = result["forced_label"].astype(str).mask(
        abstain,
        "",
    )
    return result


def build_u_vs_u_tilde_detection_by_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        metrics = _read_metrics(run_root)
        truth = _read_query_truth(run_root, condition)
        primary_scores = _read_cell_scores(run_root, condition, candidate_set)
        counterfactual = _counterfactual_joined_scores(
            runs_root=runs_root,
            grid_dir=grid_dir,
            descriptor=descriptor,
            candidate_set=candidate_set,
            condition=condition,
        )
        primary_joined = primary_scores.loc[
            primary_scores["method"].astype(str).eq(COREOT_FULL_METHOD)
        ].merge(truth, on="cell_id", how="left", validate="many_to_one")
        if primary_joined["true_label"].isna().any():
            raise ResultsGridError(f"Missing query truth for run_id={descriptor.run_id}")
        auprc_baseline = float(truth["is_absent_state"].astype(bool).mean())
        method_metrics = metrics.loc[
            (metrics["condition_id"] == condition)
            & (metrics["candidate_set"] == candidate_set)
            & (metrics["method"] == COREOT_FULL_METHOD)
        ]
        for score, score_role, joined, abstention_column in (
            ("u", "primary", primary_joined, "abstain_u_or_entropy"),
            (
                "u_tilde",
                "secondary_counterfactual",
                counterfactual,
                "abstain_u_tilde_or_entropy",
            ),
        ):
            metric_values = {
                metric: _single_metric(method_metrics, score=score, metric=metric)
                for metric in (
                    "auroc",
                    "auprc",
                    "median_absent",
                    "median_shared",
                    "absent_minus_shared_median",
                )
            }
            absent = joined.loc[joined["is_absent_state"].astype(bool)]
            shared = joined.loc[joined["is_shared_state"].astype(bool)]
            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": candidate_set,
                    "method": COREOT_FULL_METHOD,
                    "method_group": "coreot",
                    "score": score,
                    "score_role": score_role,
                    "auroc": metric_values["auroc"],
                    "auprc": metric_values["auprc"],
                    "auprc_baseline": auprc_baseline,
                    "absent_abstention_rate": float(absent[abstention_column].astype(bool).mean())
                    if len(absent)
                    else float("nan"),
                    "shared_false_abstention_rate": float(
                        shared[abstention_column].astype(bool).mean()
                    )
                    if len(shared)
                    else float("nan"),
                    "median_absent": metric_values["median_absent"],
                    "median_shared": metric_values["median_shared"],
                    "absent_minus_shared_median": metric_values[
                        "absent_minus_shared_median"
                    ],
                }
            )
    return pd.DataFrame(rows)


def build_u_tilde_counterfactual_shared_label_transfer_by_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        joined = _counterfactual_joined_scores(
            runs_root=runs_root,
            grid_dir=grid_dir,
            descriptor=descriptor,
            candidate_set=candidate_set,
            condition=condition,
        )
        shared = joined.loc[joined["is_shared_state"].astype(bool)].copy()
        forced_labeled = shared.loc[_nonempty_string(shared["forced_label"])]
        non_abstained = shared.loc[~shared["abstain_u_tilde_or_entropy"].astype(bool)]
        post_labeled = non_abstained.loc[
            _nonempty_string(non_abstained["final_label_u_tilde_abstention_aware"])
        ]
        labels = sorted(
            str(label)
            for label in shared["true_label"].dropna().unique()
            if str(label) != descriptor.held_out_label
        )
        rows.append(
            {
                "run_id": descriptor.run_id,
                "held_out_label": descriptor.held_out_label,
                "seed": descriptor.seed,
                "condition_id": condition,
                "candidate_set": candidate_set,
                "method": COREOT_FULL_METHOD,
                "method_group": "coreot",
                "score": "u_tilde",
                "score_role": "secondary_counterfactual",
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
                    post_labeled["final_label_u_tilde_abstention_aware"],
                ),
                "post_abstention_macro_f1": _macro_f1(
                    post_labeled["true_label"],
                    post_labeled["final_label_u_tilde_abstention_aware"],
                    labels,
                ),
                "coverage": float(len(non_abstained) / len(shared))
                if len(shared)
                else float("nan"),
                "shared_false_abstention_rate": float(
                    shared["abstain_u_tilde_or_entropy"].astype(bool).mean()
                )
                if len(shared)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _annotate_params(
    frame: pd.DataFrame,
    *,
    tau_min_map: dict[str, float],
    tau_max_map: dict[str, float],
    alpha_map: dict[str, float],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["tau_min"] = result["run_id"].map(tau_min_map)
    result["tau_max"] = result["run_id"].map(tau_max_map)
    result["alpha"] = result["run_id"].map(alpha_map)
    result["sensitivity_role"] = "swept"
    return result


def _build_summaries(
    *,
    runs_root: Path,
    grid_dir: Path,
    candidate_set: str,
    condition: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    int,
]:
    descriptors = discover_run_descriptors(grid_dir)
    if not descriptors:
        raise ResultsGridError(f"No run descriptors found in {grid_dir}")
    completed, failures = _partition_completed_sweep_runs(
        runs_root=runs_root,
        descriptors=descriptors,
        candidate_set=candidate_set,
        condition=condition,
    )
    if not completed:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            _failed_run_ids(failures),
            len(descriptors),
        )

    tau_min_map, tau_max_map, alpha_map = _build_param_maps(grid_dir, completed)
    detection_by_run = build_detection_by_run(
        runs_root=runs_root,
        descriptors=completed,
        candidate_set=candidate_set,
        condition=condition,
        methods=(COREOT_FULL_METHOD,),
    )
    shared_by_run = build_shared_label_transfer_by_run(
        runs_root=runs_root,
        descriptors=completed,
        candidate_set=candidate_set,
        condition=condition,
        methods=(COREOT_FULL_METHOD,),
    )
    u_vs_u_tilde_by_run = build_u_vs_u_tilde_detection_by_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        descriptors=completed,
        candidate_set=candidate_set,
        condition=condition,
    )
    u_tilde_counterfactual_shared_by_run = (
        build_u_tilde_counterfactual_shared_label_transfer_by_run(
            runs_root=runs_root,
            grid_dir=grid_dir,
            descriptors=completed,
            candidate_set=candidate_set,
            condition=condition,
        )
    )
    detection_by_run = _annotate_params(
        detection_by_run,
        tau_min_map=tau_min_map,
        tau_max_map=tau_max_map,
        alpha_map=alpha_map,
    )
    u_vs_u_tilde_by_run = _annotate_params(
        u_vs_u_tilde_by_run,
        tau_min_map=tau_min_map,
        tau_max_map=tau_max_map,
        alpha_map=alpha_map,
    )
    u_tilde_counterfactual_shared_by_run = _annotate_params(
        u_tilde_counterfactual_shared_by_run,
        tau_min_map=tau_min_map,
        tau_max_map=tau_max_map,
        alpha_map=alpha_map,
    )
    shared_by_run = _annotate_params(
        shared_by_run,
        tau_min_map=tau_min_map,
        tau_max_map=tau_max_map,
        alpha_map=alpha_map,
    )

    detection_summary = summarize_run_metrics(
        detection_by_run,
        group_columns=(
            "held_out_label",
            "method",
            "method_group",
            "primary_score",
            "tau_min",
            "tau_max",
            "alpha",
        ),
        value_columns=(
            "auroc",
            "auprc",
            "auprc_baseline",
            "absent_abstention_rate",
            "shared_false_abstention_rate",
            "median_absent",
            "median_shared",
            "absent_minus_shared_median",
        ),
        include_overall=True,
    )
    shared_summary = summarize_run_metrics(
        shared_by_run,
        group_columns=("held_out_label", "method", "method_group", "tau_min", "tau_max", "alpha"),
        value_columns=(
            "forced_accuracy",
            "forced_macro_f1",
            "post_abstention_accuracy",
            "post_abstention_macro_f1",
            "coverage",
            "shared_false_abstention_rate",
        ),
        include_overall=True,
    )
    u_vs_u_tilde_summary = summarize_run_metrics(
        u_vs_u_tilde_by_run,
        group_columns=(
            "held_out_label",
            "method",
            "method_group",
            "score",
            "score_role",
            "tau_min",
            "tau_max",
            "alpha",
        ),
        value_columns=(
            "auroc",
            "auprc",
            "auprc_baseline",
            "absent_abstention_rate",
            "shared_false_abstention_rate",
            "median_absent",
            "median_shared",
            "absent_minus_shared_median",
        ),
        include_overall=True,
    )
    u_tilde_counterfactual_shared_summary = summarize_run_metrics(
        u_tilde_counterfactual_shared_by_run,
        group_columns=(
            "held_out_label",
            "method",
            "method_group",
            "score",
            "score_role",
            "tau_min",
            "tau_max",
            "alpha",
        ),
        value_columns=(
            "forced_accuracy",
            "forced_macro_f1",
            "post_abstention_accuracy",
            "post_abstention_macro_f1",
            "coverage",
            "shared_false_abstention_rate",
        ),
        include_overall=True,
    )
    detection_summary["sensitivity_role"] = "swept"
    shared_summary["sensitivity_role"] = "swept"
    u_vs_u_tilde_summary["sensitivity_role"] = "swept"
    u_tilde_counterfactual_shared_summary["sensitivity_role"] = "swept"
    return (
        detection_by_run,
        detection_summary,
        shared_by_run,
        shared_summary,
        u_vs_u_tilde_by_run,
        u_vs_u_tilde_summary,
        u_tilde_counterfactual_shared_by_run,
        u_tilde_counterfactual_shared_summary,
        _failed_run_ids(failures),
        len(descriptors),
    )


def _format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _display_name(method: str, tau_min: float | None, tau_max: float | None, alpha: float | None) -> str:
    if tau_min is None or tau_max is None:
        return DISPLAY_NAME_BY_METHOD.get(method, method)
    return (
        "CoRe-OT full, "
        f"τ_min={_format_number(tau_min)}, "
        f"τ_max={_format_number(tau_max)}, "
        f"α={_format_number(alpha or DEFAULT_ALPHA)}"
    )


def _sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ref_order = {"prior_only": 0, "nn": 1, "balanced_ot": 2}
    ordered["_method_order"] = ordered["method"].map(ref_order).fillna(10)
    ordered["_tau_min"] = ordered.get("tau_min", pd.Series(dtype=float)).fillna(-1)
    ordered["_tau_max"] = ordered.get("tau_max", pd.Series(dtype=float)).fillna(-1)
    ordered["_alpha"] = ordered.get("alpha", pd.Series(dtype=float)).fillna(-1)
    return ordered.sort_values(
        ["_method_order", "_tau_min", "_tau_max", "_alpha", "held_out_label"]
    ).reset_index(drop=True)


def _lookup_cell(
    summary: pd.DataFrame,
    *,
    method: str,
    held_out_label: str,
    quantity: str,
    tau_min: float | None,
    tau_max: float | None,
    alpha: float | None,
) -> object:
    mask = (
        (summary["method"] == method)
        & (summary["held_out_label"] == held_out_label)
        & (summary["quantity"] == quantity)
    )
    if tau_min is None:
        mask = mask & summary.get("tau_min", pd.Series(float("nan"), index=summary.index)).isna()
    else:
        mask = mask & (summary["tau_min"] == tau_min)
    if tau_max is None:
        mask = mask & summary.get("tau_max", pd.Series(float("nan"), index=summary.index)).isna()
    else:
        mask = mask & (summary["tau_max"] == tau_max)
    if alpha is None:
        mask = mask & summary.get("alpha", pd.Series(float("nan"), index=summary.index)).isna()
    else:
        mask = mask & (summary["alpha"] == alpha)
    matches = summary.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def _render_table(
    *,
    summary: pd.DataFrame,
    quantities: tuple[str, ...],
    headers: dict[str, str],
    title: str,
    intro_line: str,
    include_prior_only: bool,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
    include_overall_section: bool = True,
) -> str:
    combined = _sort_rows(summary)
    header = ["Method"] + [headers[quantity] for quantity in quantities]
    separator = ["---"] * len(header)
    lines = [title, "", intro_line, "", "Values are mean ± std across donor-split seeds.", ""]

    labels = (*held_out_labels, "overall") if include_overall_section else held_out_labels
    for held_out_label in labels:
        section_title = (
            "Mean across held-out labels"
            if held_out_label == "overall"
            else held_out_label
        )
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(separator) + " |")
        label_rows = combined.loc[combined["held_out_label"] == held_out_label]
        row_keys = (
            label_rows.loc[:, ["method", "tau_min", "tau_max", "alpha"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for _, row_key in row_keys.iterrows():
            method = str(row_key["method"])
            if method == "prior_only" and not include_prior_only:
                continue
            tau_min = None if _is_missing_tau(row_key.get("tau_min")) else float(row_key["tau_min"])
            tau_max = None if _is_missing_tau(row_key.get("tau_max")) else float(row_key["tau_max"])
            alpha = None if _is_missing_tau(row_key.get("alpha")) else float(row_key["alpha"])
            row = [_display_name(method, tau_min, tau_max, alpha)]
            for quantity in quantities:
                cell = _lookup_cell(
                    combined,
                    method=method,
                    held_out_label=held_out_label,
                    quantity=quantity,
                    tau_min=tau_min,
                    tau_max=tau_max,
                    alpha=alpha,
                )
                row.append(_format_table_value(cell, quantity))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def render_coreot_full_tau_range_table_1(
    detection_summary: pd.DataFrame,
    fixed_detection: pd.DataFrame,
    *,
    alpha: float = 5.0,
    alpha_label: str | None = None,
    include_fixed_references: bool = True,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
    include_overall_section: bool = True,
) -> str:
    combined = (
        detection_summary.copy()
        if fixed_detection.empty
        else pd.concat([fixed_detection, detection_summary], ignore_index=True)
    )
    alpha_text = alpha_label or f"fixed α={alpha}"
    if include_fixed_references:
        intro = (
            "Fixed reference rows from the tuned-baseline bundle. Swept rows: "
            f"`coreot_full` with {alpha_text} and τ_min/τ_max varied."
        )
    else:
        intro = (
            f"Swept rows: `coreot_full` with {alpha_text} and τ_min/τ_max varied. "
            "No fixed-reference methods included."
        )
    return _render_table(
        summary=combined,
        quantities=MAIN_TABLE_1_QUANTITIES,
        headers=MAIN_TABLE_1_COLUMN_HEADERS,
        title="# Main Table 1 — coreot_full τ_min/τ_max sensitivity",
        intro_line=intro,
        include_prior_only=True,
        held_out_labels=held_out_labels,
        include_overall_section=include_overall_section,
    )


def render_coreot_full_tau_range_table_2(
    shared_summary: pd.DataFrame,
    fixed_shared: pd.DataFrame,
    *,
    alpha: float = 5.0,
    alpha_label: str | None = None,
    include_fixed_references: bool = True,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
    include_overall_section: bool = True,
) -> str:
    combined = (
        shared_summary.copy()
        if fixed_shared.empty
        else pd.concat([fixed_shared, shared_summary], ignore_index=True)
    )
    alpha_text = alpha_label or f"fixed α={alpha}"
    if include_fixed_references:
        intro = (
            "Fixed reference rows from the tuned-baseline bundle (prior-only excluded). "
            f"Swept rows: `coreot_full` with {alpha_text} and τ_min/τ_max varied."
        )
    else:
        intro = (
            f"Swept rows: `coreot_full` with {alpha_text} and τ_min/τ_max varied. "
            "No fixed-reference methods included."
        )
    return _render_table(
        summary=combined,
        quantities=MAIN_TABLE_2_QUANTITIES,
        headers=MAIN_TABLE_2_COLUMN_HEADERS,
        title="# Main Table 2 — coreot_full τ_min/τ_max sensitivity",
        intro_line=intro,
        include_prior_only=False,
        held_out_labels=held_out_labels,
        include_overall_section=include_overall_section,
    )


def _score_display_name(score: str) -> str:
    if score == "u":
        return "raw u"
    if score == "u_tilde":
        return "prior-adjusted u_tilde"
    return score


def _sort_u_vs_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_tau_min"] = ordered.get("tau_min", pd.Series(dtype=float)).fillna(-1)
    ordered["_tau_max"] = ordered.get("tau_max", pd.Series(dtype=float)).fillna(-1)
    ordered["_alpha"] = ordered.get("alpha", pd.Series(dtype=float)).fillna(-1)
    ordered["_score_order"] = ordered["score"].map({"u": 0, "u_tilde": 1}).fillna(99)
    return ordered.sort_values(
        ["_tau_min", "_tau_max", "_alpha", "_score_order", "held_out_label"]
    ).reset_index(drop=True)


def _lookup_u_vs_cell(
    summary: pd.DataFrame,
    *,
    held_out_label: str,
    score: str,
    quantity: str,
    tau_min: float,
    tau_max: float,
    alpha: float,
) -> object:
    matches = summary.loc[
        (summary["held_out_label"] == held_out_label)
        & (summary["score"] == score)
        & (summary["quantity"] == quantity)
        & (summary["tau_min"] == tau_min)
        & (summary["tau_max"] == tau_max)
        & (summary["alpha"] == alpha)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_coreot_full_u_vs_u_tilde_table(
    u_vs_u_tilde_summary: pd.DataFrame,
    *,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
    include_overall_section: bool = True,
) -> str:
    if u_vs_u_tilde_summary.empty:
        return "\n".join(
            [
                "# Table 3 — raw u versus prior-adjusted u_tilde",
                "",
                "No completed swept `coreot_full` runs were available for this comparison.",
            ]
        )
    quantities = (
        "auroc",
        "auprc",
        "absent_abstention_rate",
        "shared_false_abstention_rate",
        "absent_minus_shared_median",
    )
    headers = {
        **MAIN_TABLE_1_COLUMN_HEADERS,
        "absent_minus_shared_median": "Δ median",
    }
    combined = _sort_u_vs_rows(u_vs_u_tilde_summary)
    header = ["Score"] + [headers[quantity] for quantity in quantities]
    separator = ["---"] * len(header)
    lines = [
        "# Table 3 — raw u versus prior-adjusted u_tilde",
        "",
        (
            "Raw `u` is the primary score. `u_tilde` rows use the secondary "
            "counterfactual abstention policy with the same entropy clause."
        ),
        "",
        "Values are mean ± std across donor-split seeds.",
        "",
    ]
    labels = (*held_out_labels, "overall") if include_overall_section else held_out_labels
    for held_out_label in labels:
        section_title = (
            "Mean across held-out labels"
            if held_out_label == "overall"
            else held_out_label
        )
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(separator) + " |")
        label_rows = combined.loc[combined["held_out_label"] == held_out_label]
        row_keys = (
            label_rows.loc[:, ["score", "tau_min", "tau_max", "alpha"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for _, row_key in row_keys.iterrows():
            score = str(row_key["score"])
            tau_min = float(row_key["tau_min"])
            tau_max = float(row_key["tau_max"])
            alpha = float(row_key["alpha"])
            row = [
                (
                    f"{_score_display_name(score)}, "
                    f"τ_min={_format_number(tau_min)}, "
                    f"τ_max={_format_number(tau_max)}, "
                    f"α={_format_number(alpha)}"
                )
            ]
            for quantity in quantities:
                cell = _lookup_u_vs_cell(
                    combined,
                    held_out_label=held_out_label,
                    score=score,
                    quantity=quantity,
                    tau_min=tau_min,
                    tau_max=tau_max,
                    alpha=alpha,
                )
                row.append(_format_table_value(cell, quantity))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def _table_body(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[2:] if len(lines) > 1 and lines[1] == "" else lines[1:]
    return "\n".join(lines).strip()


def render_coreot_full_tau_range_report(
    *,
    title: str = "HIHA DC coreot_full τ_min/τ_max Sensitivity Report",
    fixed_reference_root: str,
    grid_dir: str,
    tau_min_values: tuple[float, ...],
    tau_max_values: tuple[float, ...],
    alpha_label: str,
    n_runs_completed: int,
    n_runs_expected: int,
    table_1_markdown: str,
    table_2_markdown: str,
    table_3_markdown: str,
    heatmap_pdf: str | None = None,
    u_tilde_heatmap_pdf: str | None = None,
    failed_runs: list[str] | None = None,
    include_fixed_references: bool = True,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Task Definition",
        "",
        "Parameter-sensitivity diagnostic for provisional `coreot_full`, varying the "
        "matchability-dependent source-penalty range while keeping the broad-anchor "
        "cost weight fixed.",
        "",
        "## Sweep Grid",
        "",
        "- **Method:** `coreot_full`",
        f"- **α setting:** {alpha_label}",
        f"- **τ_min values:** {{{', '.join(_format_number(v) for v in tau_min_values)}}}",
        f"- **τ_max values:** {{{', '.join(_format_number(v) for v in tau_max_values)}}}",
        "- **Constraint:** τ_min ≤ τ_max",
    ]
    if include_fixed_references:
        lines.extend([
            "",
            "## Fixed Reference Rows",
            "",
            "- **Fixed methods:** `prior_only`, `nn`, `balanced_ot`",
            f"- **Fixed reference source:** `{fixed_reference_root}`",
        ])
    lines.extend([
        "",
        "## Run Coverage",
        "",
        f"- **Completed runs:** {n_runs_completed} / {n_runs_expected}",
        f"- **Grid dir:** `{grid_dir}`",
    ])
    if failed_runs:
        lines.append(f"- **Failed runs:** {', '.join(failed_runs)}")
    if heatmap_pdf is not None:
        lines.extend(["", "## Heatmaps", "", f"- **Heatmap PDF:** `{heatmap_pdf}`"])
    if u_tilde_heatmap_pdf is not None:
        if heatmap_pdf is None:
            lines.extend(["", "## Heatmaps", ""])
        lines.append(f"- **u_tilde heatmap PDF:** `{u_tilde_heatmap_pdf}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "_Fill in after inspecting results. Separate detection behavior, abstention "
            "tradeoffs, shared-label-transfer preservation, and the secondary u_tilde "
            "counterfactual policy. Do not claim optimality unless a selection rule is "
            "documented._",
            "",
            "## Main Table 1",
            "",
            _table_body(table_1_markdown),
            "",
            "## Main Table 2",
            "",
            _table_body(table_2_markdown),
            "",
            "## Table 3",
            "",
            _table_body(table_3_markdown),
            "",
        ]
    )
    return "\n".join(lines)


FIXED_REF_DETECTION_COLUMNS = [
    "held_out_label", "method", "method_group", "primary_score",
    "tau_min", "tau_max", "alpha",
    "quantity", "mean", "std", "sem", "n_runs", "sensitivity_role",
]
FIXED_REF_SHARED_COLUMNS = [
    "held_out_label", "method", "method_group",
    "tau_min", "tau_max", "alpha",
    "quantity", "mean", "std", "sem", "n_runs", "sensitivity_role",
]


def _empty_fixed_reference_detection() -> pd.DataFrame:
    return pd.DataFrame(columns=FIXED_REF_DETECTION_COLUMNS)


def _empty_fixed_reference_shared() -> pd.DataFrame:
    return pd.DataFrame(columns=FIXED_REF_SHARED_COLUMNS)


def write_coreot_full_tau_range_results(
    *,
    runs_root: str | Path = Path("runs"),
    grid_dir: str | Path = Path(
        "experiments/missing_celltype/generated_configs/"
        "hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid"
    ),
    output_root: str | Path = Path("results/hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid"),
    main_grid_root: str | Path = DEFAULT_FIXED_REFERENCE_ROOT,
    candidate_set: str = "hiha_harmony30_k100",
    condition: str = "incomplete_reference",
    write_figures: bool = False,
    include_fixed_references: bool = True,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
    report_title: str = "HIHA DC coreot_full τ_min/τ_max Sensitivity Report",
    alpha_label: str | None = None,
    write_separate_tables: bool = True,
    table_bundle_filename: str | None = None,
    include_overall_table_sections: bool = True,
) -> CoreotFullTauRangeArtifactPaths:
    runs_root = Path(runs_root)
    grid_dir = Path(grid_dir)
    output_root = Path(output_root)
    main_grid_root = Path(main_grid_root)

    tables_root = output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    (
        detection_by_run,
        detection_summary,
        shared_by_run,
        shared_summary,
        u_vs_u_tilde_by_run,
        u_vs_u_tilde_summary,
        u_tilde_counterfactual_shared_by_run,
        u_tilde_counterfactual_shared_summary,
        failed_runs,
        n_expected,
    ) = _build_summaries(
        runs_root=runs_root,
        grid_dir=grid_dir,
        candidate_set=candidate_set,
        condition=condition,
    )

    if include_fixed_references:
        fixed_detection, fixed_shared = read_fixed_reference_bundle(main_grid_root, held_out_labels)
        fixed_detection["tau_min"] = float("nan")
        fixed_detection["tau_max"] = float("nan")
        fixed_detection["alpha"] = float("nan")
        fixed_shared["tau_min"] = float("nan")
        fixed_shared["tau_max"] = float("nan")
        fixed_shared["alpha"] = float("nan")
    else:
        fixed_detection = _empty_fixed_reference_detection()
        fixed_shared = _empty_fixed_reference_shared()

    detection_by_run_path = tables_root / "detection_by_run.csv"
    detection_summary_path = tables_root / "detection_summary.csv"
    u_vs_u_tilde_by_run_path = tables_root / "u_vs_u_tilde_detection_by_run.csv"
    u_vs_u_tilde_summary_path = tables_root / "u_vs_u_tilde_detection_summary.csv"
    shared_by_run_path = tables_root / "shared_label_transfer_by_run.csv"
    shared_summary_path = tables_root / "shared_label_transfer_summary.csv"
    u_tilde_counterfactual_shared_by_run_path = (
        tables_root / "u_tilde_counterfactual_shared_label_transfer_by_run.csv"
    )
    u_tilde_counterfactual_shared_summary_path = (
        tables_root / "u_tilde_counterfactual_shared_label_transfer_summary.csv"
    )
    table_1_path = tables_root / "table_1_coreot_full_tau_range.md"
    table_2_path = tables_root / "table_2_coreot_full_tau_range.md"
    table_3_path = tables_root / "table_3_u_vs_u_tilde.md"

    detection_by_run.to_csv(detection_by_run_path, index=False)
    detection_summary.to_csv(detection_summary_path, index=False)
    u_vs_u_tilde_by_run.to_csv(u_vs_u_tilde_by_run_path, index=False)
    u_vs_u_tilde_summary.to_csv(u_vs_u_tilde_summary_path, index=False)
    shared_by_run.to_csv(shared_by_run_path, index=False)
    shared_summary.to_csv(shared_summary_path, index=False)
    u_tilde_counterfactual_shared_by_run.to_csv(
        u_tilde_counterfactual_shared_by_run_path, index=False
    )
    u_tilde_counterfactual_shared_summary.to_csv(
        u_tilde_counterfactual_shared_summary_path, index=False
    )

    tau_min_values = tuple(sorted(detection_summary["tau_min"].dropna().unique().tolist()))
    tau_max_values = tuple(sorted(detection_summary["tau_max"].dropna().unique().tolist()))
    alpha_values = tuple(sorted(detection_summary["alpha"].dropna().unique().tolist()))
    alpha = float(alpha_values[0]) if alpha_values else DEFAULT_ALPHA
    resolved_alpha_label = (
        alpha_label
        or (
            f"fixed α={_format_number(alpha)}"
            if len(alpha_values) <= 1
            else "label-specific α"
        )
    )

    table_1_markdown = render_coreot_full_tau_range_table_1(
        detection_summary,
        fixed_detection,
        alpha=alpha,
        alpha_label=resolved_alpha_label,
        include_fixed_references=include_fixed_references,
        held_out_labels=held_out_labels,
        include_overall_section=include_overall_table_sections,
    )
    table_2_markdown = render_coreot_full_tau_range_table_2(
        shared_summary,
        fixed_shared,
        alpha=alpha,
        alpha_label=resolved_alpha_label,
        include_fixed_references=include_fixed_references,
        held_out_labels=held_out_labels,
        include_overall_section=include_overall_table_sections,
    )
    table_3_markdown = render_coreot_full_u_vs_u_tilde_table(
        u_vs_u_tilde_summary,
        held_out_labels=held_out_labels,
        include_overall_section=include_overall_table_sections,
    )
    table_bundle_path = tables_root / table_bundle_filename if table_bundle_filename else None
    if write_separate_tables:
        table_1_path.write_text(table_1_markdown, encoding="utf-8")
        table_2_path.write_text(table_2_markdown, encoding="utf-8")
        table_3_path.write_text(table_3_markdown, encoding="utf-8")
    if table_bundle_path is not None:
        table_bundle_path.write_text(
            "\n\n".join([table_1_markdown, table_2_markdown, table_3_markdown]),
            encoding="utf-8",
        )

    figure_paths: dict[str, Path] = {}
    heatmap_pdf_path: str | None = None
    u_tilde_heatmap_pdf_path: str | None = None
    if write_figures and not detection_summary.empty:
        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            from coreot.results.figures import write_coreot_full_tau_range_heatmaps

            figure_paths = write_coreot_full_tau_range_heatmaps(
                detection_summary=detection_summary,
                shared_summary=shared_summary,
                output_dir=figures_dir,
            )
            u_tilde_detection_summary = u_vs_u_tilde_summary.loc[
                u_vs_u_tilde_summary["score"].astype(str).eq("u_tilde")
            ].copy()
            u_tilde_figure_paths = write_coreot_full_tau_range_heatmaps(
                detection_summary=u_tilde_detection_summary,
                shared_summary=u_tilde_counterfactual_shared_summary,
                output_dir=figures_dir,
                filename="u_tilde_heatmaps.pdf",
                title_suffix="u_tilde counterfactual",
            )
            figure_paths.update(u_tilde_figure_paths)
            heatmap_pdf = figure_paths.get("heatmaps_pdf")
            heatmap_pdf_path = str(heatmap_pdf) if heatmap_pdf is not None else None
            u_tilde_heatmap_pdf = figure_paths.get("u_tilde_heatmaps_pdf")
            u_tilde_heatmap_pdf_path = (
                str(u_tilde_heatmap_pdf) if u_tilde_heatmap_pdf is not None else None
            )
        except ImportError:
            pass

    n_completed = int(detection_by_run["run_id"].nunique()) if not detection_by_run.empty else 0

    report_path = output_root / "report.md"
    report_path.write_text(
        render_coreot_full_tau_range_report(
            title=report_title,
            fixed_reference_root=str(main_grid_root),
            grid_dir=str(grid_dir),
            tau_min_values=tau_min_values,
            tau_max_values=tau_max_values,
            alpha_label=resolved_alpha_label,
            n_runs_completed=n_completed,
            n_runs_expected=n_expected,
            table_1_markdown=table_1_markdown,
            table_2_markdown=table_2_markdown,
            table_3_markdown=table_3_markdown,
            heatmap_pdf=heatmap_pdf_path,
            u_tilde_heatmap_pdf=u_tilde_heatmap_pdf_path,
            failed_runs=failed_runs or None,
            include_fixed_references=include_fixed_references,
        ),
        encoding="utf-8",
    )

    manifest_path = output_root / "manifest.yaml"
    artifacts: dict[str, str] = {
        "detection_by_run": str(detection_by_run_path),
        "detection_summary": str(detection_summary_path),
        "u_vs_u_tilde_detection_by_run": str(u_vs_u_tilde_by_run_path),
        "u_vs_u_tilde_detection_summary": str(u_vs_u_tilde_summary_path),
        "shared_label_transfer_by_run": str(shared_by_run_path),
        "shared_label_transfer_summary": str(shared_summary_path),
        "u_tilde_counterfactual_shared_label_transfer_by_run": str(
            u_tilde_counterfactual_shared_by_run_path
        ),
        "u_tilde_counterfactual_shared_label_transfer_summary": str(
            u_tilde_counterfactual_shared_summary_path
        ),
        "report": str(report_path),
    }
    if write_separate_tables:
        artifacts["table_1_coreot_full_tau_range"] = str(table_1_path)
        artifacts["table_2_coreot_full_tau_range"] = str(table_2_path)
        artifacts["table_3_u_vs_u_tilde"] = str(table_3_path)
    if table_bundle_path is not None:
        artifacts["table_bundle"] = str(table_bundle_path)
    for name, path in figure_paths.items():
        artifacts[f"figure_{name}"] = str(path)

    write_manifest(
        manifest_path,
        Manifest(
            stage="manuscript-results-coreot-full-tau-range",
            artifacts=artifacts,
            metadata={
                "runs_root": str(runs_root),
                "grid_dir": str(grid_dir),
                "condition": condition,
                "candidate_set": candidate_set,
                "method": COREOT_FULL_METHOD,
                "tau_min_values": [float(v) for v in tau_min_values],
                "tau_max_values": [float(v) for v in tau_max_values],
                "alpha_values": [float(v) for v in alpha_values],
                "alpha_label": resolved_alpha_label,
                "fixed_reference_methods": list(FIXED_REFERENCE_METHODS),
                "fixed_reference_root": str(main_grid_root),
                "n_runs_completed": n_completed,
                "n_runs_expected": n_expected,
                "failed_runs": failed_runs,
                "u_tilde_policy": "secondary_counterfactual_full_reference_quantile_with_entropy",
                "variability": "descriptive_seed_level_mean_std_sem",
            },
        ),
    )

    return CoreotFullTauRangeArtifactPaths(
        output_root=output_root,
        detection_by_run=detection_by_run_path,
        detection_summary=detection_summary_path,
        u_vs_u_tilde_detection_by_run=u_vs_u_tilde_by_run_path,
        u_vs_u_tilde_detection_summary=u_vs_u_tilde_summary_path,
        shared_label_transfer_by_run=shared_by_run_path,
        shared_label_transfer_summary=shared_summary_path,
        u_tilde_counterfactual_shared_label_transfer_by_run=(
            u_tilde_counterfactual_shared_by_run_path
        ),
        u_tilde_counterfactual_shared_label_transfer_summary=(
            u_tilde_counterfactual_shared_summary_path
        ),
        table_1=table_1_path,
        table_2=table_2_path,
        table_3=table_3_path,
        table_bundle=table_bundle_path,
        report=report_path,
        manifest=manifest_path,
        figures=figure_paths,
    )
