from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.artifacts.run_artifacts import (
    ArtifactBatchError,
    ArtifactFailure,
    RunArtifacts,
)
from coreot.evaluation.metrics import safe_auprc, safe_auroc
from coreot.results.grid_common import (
    DISPLAY_NAME_BY_METHOD,
    EXTERNAL_TABLE_ROW_ORDER,
    MAIN_TABLE_1_COLUMN_HEADERS,
    MAIN_TABLE_1_QUANTITIES,
    MAIN_TABLE_2_COLUMN_HEADERS,
    MAIN_TABLE_2_QUANTITIES,
    METHOD_GROUP_BY_METHOD,
    ResultsGridError,
    RunDescriptor,
    _format_table_value,
    _markdown_table,
    _represented_truth_labels,
    discover_run_descriptors,
    summarize_run_metrics,
)
from sklearn.metrics import accuracy_score, f1_score

STAGE = "manuscript-results"
CANDIDATE_SET = "external_reference_mapping"
CONDITIONS = ("incomplete_reference", "full_reference_control")


@dataclass(frozen=True)
class ExternalBaselineArtifactPaths:
    output_root: Path
    tables_root: Path
    detection_by_run: Path
    detection_summary: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    full_reference_by_run: Path
    full_reference_summary: Path
    table_1_detection: Path
    table_2_shared: Path
    table_s1_full_reference: Path
    markdown_summary: Path
    manifest: Path


def write_external_baseline_results(
    *,
    runs_root: str | Path,
    grid_dir: str | Path,
    output_root: str | Path,
    expected_held_out_labels: tuple[str, ...] | None = None,
    expected_seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    report_title: str = "External Baselines Report - leave-one HIHA DC",
    report_description: str = (
        "External reference-mapping baselines evaluated on the selected "
        "report_leave_one_HIHA_DC run roots."
    ),
    source_note: str | None = None,
    methods: tuple[str, ...] | None = None,
) -> ExternalBaselineArtifactPaths:
    if source_note is None:
        source_note = (
            "results/HIHA_DC/compare_baselines/external_baselines/"
            "tables/external_detection_summary.csv"
        )

    descriptors = discover_run_descriptors(grid_dir)
    if not descriptors:
        raise ResultsGridError(f"No run descriptors found in {grid_dir}")
    if expected_held_out_labels is None:
        expected_held_out_labels = ("HLA-DRhi cDC2", "ISG+ cDC2")
    descriptors = _select_descriptors(
        descriptors,
        expected_held_out_labels=expected_held_out_labels,
        expected_seeds=expected_seeds,
    )
    _validate_descriptors(
        descriptors,
        expected_held_out_labels=expected_held_out_labels,
        expected_seeds=expected_seeds,
    )

    completed_descriptors, failures = _validate_external_artifacts(
        runs_root=Path(runs_root), descriptors=descriptors
    )
    if failures:
        raise ArtifactBatchError(failures)

    methods = methods or _external_methods()
    if not methods:
        raise ResultsGridError("No external baseline methods configured")

    output = Path(output_root)
    tables_root = output / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    # Detection (incomplete_reference)
    detection_by_run = _build_external_detection_by_run(
        runs_root=Path(runs_root),
        descriptors=completed_descriptors,
        methods=methods,
        expected_seeds=expected_seeds,
    )
    detection_summary = summarize_run_metrics(
        detection_by_run,
        group_columns=("held_out_label", "method", "method_group"),
        value_columns=(
            "auroc",
            "auprc",
            "auprc_baseline",
            "absent_abstention_rate",
            "shared_false_abstention_rate",
        ),
        include_overall=True,
    )

    # Shared label transfer (incomplete_reference)
    shared_by_run = _build_external_shared_label_transfer_by_run(
        runs_root=Path(runs_root),
        descriptors=completed_descriptors,
        methods=methods,
    )
    shared_summary = summarize_run_metrics(
        shared_by_run,
        group_columns=("held_out_label", "method", "method_group"),
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

    # Full-reference false abstention
    full_ref_by_run = _build_external_full_reference_by_run(
        runs_root=Path(runs_root),
        descriptors=completed_descriptors,
        methods=methods,
        expected_seeds=expected_seeds,
    )
    full_ref_summary = summarize_run_metrics(
        full_ref_by_run,
        group_columns=("held_out_label", "method", "method_group"),
        value_columns=(
            "full_reference_false_abstention_rate",
            "score_median",
            "score_p95",
            "forced_accuracy",
            "forced_macro_f1",
            "post_abstention_accuracy",
            "post_abstention_macro_f1",
            "coverage",
        ),
        include_overall=True,
    )

    paths = ExternalBaselineArtifactPaths(
        output_root=output,
        tables_root=tables_root,
        detection_by_run=tables_root / "external_detection_by_run.csv",
        detection_summary=tables_root / "external_detection_summary.csv",
        shared_label_transfer_by_run=tables_root
        / "external_shared_label_transfer_by_run.csv",
        shared_label_transfer_summary=tables_root
        / "external_shared_label_transfer_summary.csv",
        full_reference_by_run=tables_root / "external_full_reference_by_run.csv",
        full_reference_summary=tables_root
        / "external_full_reference_summary.csv",
        table_1_detection=tables_root / "table_1_external_detection.md",
        table_2_shared=tables_root
        / "table_2_external_shared_label_transfer.md",
        table_s1_full_reference=tables_root
        / "table_s1_external_full_reference.md",
        markdown_summary=output / "summary.md",
        manifest=output / "manifest.yaml",
    )

    detection_by_run.to_csv(paths.detection_by_run, index=False)
    detection_summary.to_csv(paths.detection_summary, index=False)
    shared_by_run.to_csv(paths.shared_label_transfer_by_run, index=False)
    shared_summary.to_csv(paths.shared_label_transfer_summary, index=False)
    full_ref_by_run.to_csv(paths.full_reference_by_run, index=False)
    full_ref_summary.to_csv(paths.full_reference_summary, index=False)

    paths.table_1_detection.write_text(
        _render_external_table_1(detection_summary, source_note=source_note),
        encoding="utf-8",
    )
    paths.table_2_shared.write_text(
        _render_external_table_2(shared_summary, source_note=source_note),
        encoding="utf-8",
    )
    paths.table_s1_full_reference.write_text(
        _render_external_table_s1(full_ref_summary, source_note=source_note),
        encoding="utf-8",
    )
    paths.markdown_summary.write_text(
        _render_external_markdown_summary(
            detection_summary,
            shared_summary,
            full_ref_summary,
            report_title=report_title,
            report_description=report_description,
            expected_held_out_labels=expected_held_out_labels,
            expected_seeds=expected_seeds,
        ),
        encoding="utf-8",
    )

    write_manifest(
        paths.manifest,
        Manifest(
            stage=STAGE,
            artifacts={
                "external_detection_by_run": str(paths.detection_by_run),
                "external_detection_summary": str(paths.detection_summary),
                "external_shared_label_transfer_by_run": str(
                    paths.shared_label_transfer_by_run
                ),
                "external_shared_label_transfer_summary": str(
                    paths.shared_label_transfer_summary
                ),
                "external_full_reference_by_run": str(
                    paths.full_reference_by_run
                ),
                "external_full_reference_summary": str(
                    paths.full_reference_summary
                ),
                "table_1_external_detection": str(paths.table_1_detection),
                "table_2_external_shared_label_transfer": str(
                    paths.table_2_shared
                ),
                "table_s1_external_full_reference": str(
                    paths.table_s1_full_reference
                ),
                "markdown_summary": str(paths.markdown_summary),
            },
            metadata={
                "runs_root": str(runs_root),
                "grid_dir": str(grid_dir),
                "candidate_set": CANDIDATE_SET,
                "n_runs": len(completed_descriptors),
                "n_runs_expected": len(descriptors),
                "held_out_labels": list(expected_held_out_labels),
                "seeds": list(expected_seeds),
                "methods": list(methods),
                "variability": "descriptive_seed_level_mean_std_sem",
                "evaluation_artifacts_untouched": True,
            },
        ),
    )
    return paths


def _external_methods() -> tuple[str, ...]:
    from coreot.external_baselines.schema import DEFAULT_EXTERNAL_BASELINE_METHODS

    return DEFAULT_EXTERNAL_BASELINE_METHODS


def _select_descriptors(
    descriptors: tuple[RunDescriptor, ...],
    *,
    expected_held_out_labels: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> tuple[RunDescriptor, ...]:
    label_set = set(expected_held_out_labels)
    seed_set = set(expected_seeds)
    return tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.held_out_label in label_set and descriptor.seed in seed_set
    )


def _validate_descriptors(
    descriptors: tuple[RunDescriptor, ...],
    *,
    expected_held_out_labels: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> None:
    expected_n = len(expected_held_out_labels) * len(expected_seeds)
    if len(descriptors) != expected_n:
        raise ResultsGridError(
            f"Expected exactly {expected_n} run descriptors; found {len(descriptors)}"
        )

    expected_labels = set(expected_held_out_labels)
    expected_seed_set = set(expected_seeds)
    actual_labels = {d.held_out_label for d in descriptors}
    if actual_labels != expected_labels:
        raise ResultsGridError(
            f"Held-out labels must be exactly {sorted(expected_labels)}; "
            f"got {sorted(actual_labels)}"
        )

    for label in sorted(expected_labels):
        label_seeds = {
            d.seed for d in descriptors if d.held_out_label == label
        }
        if label_seeds != expected_seed_set:
            raise ResultsGridError(
                f"Seeds for {label!r} must be exactly {sorted(expected_seed_set)}; "
                f"got {sorted(label_seeds)}"
            )


def _validate_external_artifacts(
    *, runs_root: Path, descriptors: tuple[RunDescriptor, ...]
) -> tuple[tuple[RunDescriptor, ...], list[ArtifactFailure]]:
    completed: list[RunDescriptor] = []
    failures: list[ArtifactFailure] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        artifacts = RunArtifacts(run_root, STAGE)
        run_failures: list[ArtifactFailure] = []
        for condition in CONDITIONS:
            try:
                artifacts.evaluation_truth(condition).query_truth().read()
            except Exception as exc:
                run_failures.append(
                    ArtifactFailure(
                        run_id=descriptor.run_id,
                        artifact_kind="query_truth",
                        state="error",
                        path=artifacts.evaluation_truth(condition)
                        .query_truth()
                        .path,
                        condition=condition,
                        detail=str(exc),
                    )
                )
            try:
                artifacts.scoring(
                    condition, CANDIDATE_SET
                ).cell_scores().read()
            except Exception as exc:
                run_failures.append(
                    ArtifactFailure(
                        run_id=descriptor.run_id,
                        artifact_kind="cell_scores",
                        state="error",
                        path=artifacts.scoring(condition, CANDIDATE_SET)
                        .cell_scores()
                        .path,
                        condition=condition,
                        candidate_set=CANDIDATE_SET,
                        detail=str(exc),
                    )
                )
        if run_failures:
            failures.extend(run_failures)
        else:
            completed.append(descriptor)
    return tuple(completed), failures


def _read_external_cell_scores(
    run_root: Path, condition: str
) -> pd.DataFrame:
    return (
        RunArtifacts(run_root, STAGE)
        .scoring(condition, CANDIDATE_SET)
        .cell_scores()
        .read()
    )


def _read_query_truth(run_root: Path, condition: str) -> pd.DataFrame:
    return (
        RunArtifacts(run_root, STAGE)
        .evaluation_truth(condition)
        .query_truth()
        .read()
    )


def _build_external_detection_by_run(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    methods: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> pd.DataFrame:
    condition = "incomplete_reference"
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _read_query_truth(run_root, condition)
        scores = _read_external_cell_scores(run_root, condition)
        for method in methods:
            joined = _joined_method_scores(scores, truth, method)
            if "true_broad_label" in joined.columns:
                removed_broad = joined.loc[
                    joined["true_label"].astype(str).eq(descriptor.held_out_label),
                    "true_broad_label",
                ].astype(str).unique()
                if len(removed_broad) != 1:
                    raise ResultsGridError(
                        f"Expected one evaluation broad class for {descriptor.run_id}"
                    )
                joined = joined.loc[
                    joined["true_broad_label"].astype(str).eq(removed_broad[0])
                ].copy()
            absent = joined["is_absent_state"].astype(bool)
            z = joined["u"]
            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": CANDIDATE_SET,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(
                        method, "unknown"
                    ),
                    "auroc": safe_auroc(absent, z),
                    "auprc": safe_auprc(absent, z),
                    "auprc_baseline": float(absent.mean()),
                    "absent_abstention_rate": float(
                        joined.loc[absent, "abstain_u_or_entropy"]
                        .astype(bool)
                        .mean()
                    )
                    if absent.any()
                    else float("nan"),
                    "shared_false_abstention_rate": float(
                        joined.loc[~absent, "abstain_u_or_entropy"]
                        .astype(bool)
                        .mean()
                    )
                    if (~absent).any()
                    else float("nan"),
                }
            )
    frame = pd.DataFrame(rows)
    _validate_detection_rows(frame, methods=methods, expected_seeds=expected_seeds)
    return frame


def _validate_detection_rows(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> None:
    incomplete = frame.loc[frame["condition_id"] == "incomplete_reference"]
    labels = set(incomplete["held_out_label"].astype(str))
    expected_seed_set = set(expected_seeds)
    n_expected = len(labels) * len(methods) * len(expected_seeds)
    if len(incomplete) != n_expected:
        raise ResultsGridError(
            f"Expected {n_expected} incomplete-reference detection rows; "
            f"found {len(incomplete)}"
        )
    for (label, method), group in incomplete.groupby(
        ["held_out_label", "method"]
    ):
        seeds = set(group["seed"])
        if seeds != expected_seed_set:
            raise ResultsGridError(
                f"Expected seeds {sorted(expected_seed_set)} for "
                f"label={label!r} method={method!r}; "
                f"got {sorted(seeds)}"
            )


def _build_external_shared_label_transfer_by_run(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    methods: tuple[str, ...],
) -> pd.DataFrame:
    condition = "incomplete_reference"
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _read_query_truth(run_root, condition)
        scores = _read_external_cell_scores(run_root, condition)
        for method in methods:
            joined = _joined_method_scores(scores, truth, method)
            shared = joined.loc[joined["is_shared_state"].astype(bool)].copy()
            if shared.empty:
                continue
            forced_labeled = shared.loc[
                shared["forced_label"].astype(str).str.len() > 0
            ]
            abstention = shared["abstain_u_or_entropy"].astype(bool)
            non_abstained = shared.loc[~abstention]
            post_labeled = non_abstained.loc[
                non_abstained["final_label_abstention_aware"]
                .astype(str)
                .str.len()
                > 0
            ]
            labels = _represented_truth_labels(shared)
            rows.append(
                {
                    "run_id": descriptor.run_id,
                    "held_out_label": descriptor.held_out_label,
                    "seed": descriptor.seed,
                    "condition_id": condition,
                    "candidate_set": CANDIDATE_SET,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(
                        method, "unknown"
                    ),
                    "n_shared": int(len(shared)),
                    "n_forced_labeled": int(len(forced_labeled)),
                    "n_non_abstained": int(len(non_abstained)),
                    "forced_accuracy": _accuracy(
                        forced_labeled["true_label"],
                        forced_labeled["forced_label"],
                    ),
                    "forced_macro_f1": _macro_f1(
                        forced_labeled["true_label"],
                        forced_labeled["forced_label"],
                        labels,
                    ),
                    "post_abstention_accuracy": _accuracy(
                        post_labeled["true_label"],
                        post_labeled["final_label_abstention_aware"],
                    ),
                    "post_abstention_macro_f1": _macro_f1(
                        post_labeled["true_label"],
                        post_labeled["final_label_abstention_aware"],
                        labels,
                    ),
                    "coverage": float(len(non_abstained) / len(shared))
                    if len(shared)
                    else math.nan,
                    "shared_false_abstention_rate": float(
                        abstention.astype(bool).mean()
                    )
                    if len(shared)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_external_full_reference_by_run(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    methods: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> pd.DataFrame:
    condition = "full_reference_control"
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth = _read_query_truth(run_root, condition)
        scores = _read_external_cell_scores(run_root, condition)
        for method in methods:
            joined = _joined_method_scores(scores, truth, method)
            shared = joined.loc[joined["is_shared_state"].astype(bool)].copy()
            if shared.empty:
                continue
            z = pd.to_numeric(joined["u"], errors="coerce").dropna()
            forced_labeled = shared.loc[
                shared["forced_label"].astype(str).str.len() > 0
            ]
            abstention = shared["abstain_u_or_entropy"].astype(bool)
            non_abstained = shared.loc[~abstention]
            post_labeled = non_abstained.loc[
                non_abstained["final_label_abstention_aware"]
                .astype(str)
                .str.len()
                > 0
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
                    "candidate_set": CANDIDATE_SET,
                    "method": method,
                    "method_group": METHOD_GROUP_BY_METHOD.get(
                        method, "unknown"
                    ),
                    "n_shared": int(len(shared)),
                    "full_reference_false_abstention_rate": float(
                        abstention.astype(bool).mean()
                    )
                    if len(shared)
                    else math.nan,
                    "score_median": float(z.median()) if len(z) else math.nan,
                    "score_p95": float(z.quantile(0.95))
                    if len(z)
                    else math.nan,
                    "forced_accuracy": _accuracy(
                        forced_labeled["true_label"],
                        forced_labeled["forced_label"],
                    ),
                    "forced_macro_f1": _macro_f1(
                        forced_labeled["true_label"],
                        forced_labeled["forced_label"],
                        labels,
                    ),
                    "post_abstention_accuracy": _accuracy(
                        post_labeled["true_label"],
                        post_labeled["final_label_abstention_aware"],
                    ),
                    "post_abstention_macro_f1": _macro_f1(
                        post_labeled["true_label"],
                        post_labeled["final_label_abstention_aware"],
                        labels,
                    ),
                    "coverage": float(len(non_abstained) / len(shared))
                    if len(shared)
                    else math.nan,
                }
            )
    frame = pd.DataFrame(rows)
    _validate_full_reference_rows(frame, methods=methods, expected_seeds=expected_seeds)
    return frame


def _validate_full_reference_rows(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> None:
    labels = set(frame["held_out_label"].astype(str))
    expected_seed_set = set(expected_seeds)
    n_expected = len(labels) * len(methods) * len(expected_seeds)
    if len(frame) != n_expected:
        raise ResultsGridError(
            f"Expected {n_expected} full-reference rows; found {len(frame)}"
        )
    for (label, method), group in frame.groupby(
        ["held_out_label", "method"]
    ):
        seeds = set(group["seed"])
        if seeds != expected_seed_set:
            raise ResultsGridError(
                f"Expected seeds {sorted(expected_seed_set)} for "
                f"full-reference label={label!r} "
                f"method={method!r}; got {sorted(seeds)}"
            )


def _joined_method_scores(
    scores: pd.DataFrame, truth: pd.DataFrame, method: str
) -> pd.DataFrame:
    method_scores = scores.loc[scores["method"] == method].copy()
    if method_scores.empty:
        raise ResultsGridError(
            f"No cell scores found for method={method}"
        )
    joined = method_scores.merge(
        truth, on="cell_id", how="left", validate="many_to_one"
    )
    if joined["true_label"].isna().any():
        raise ResultsGridError(
            f"Missing query truth for method={method}"
        )
    return joined


def _accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(
        accuracy_score(y_true.astype(str), y_pred.astype(str))
    )


def _macro_f1(
    y_true: pd.Series, y_pred: pd.Series, labels: list[str]
) -> float:
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


def _render_external_table_1(
    detection_summary: pd.DataFrame, *, source_note: str
) -> str:
    held_out_labels = [
        lbl
        for lbl in detection_summary["held_out_label"].unique()
        if lbl != "overall"
    ]
    metric_header = [
        MAIN_TABLE_1_COLUMN_HEADERS[q] for q in MAIN_TABLE_1_QUANTITIES
    ]
    header = ["Method"] + metric_header
    sep = ["---"] * len(header)

    lines = [
        "# Table 1: External baselines — absent-state detection",
        "",
        "One sub-table per held-out label. Values are mean ± std across n=5 donor-split seeds.",
        "",
    ]

    method_order = [
        m
        for m in EXTERNAL_TABLE_ROW_ORDER
        if m in detection_summary["method"].values
    ]

    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in method_order:
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            row = [display]
            for qty in MAIN_TABLE_1_QUANTITIES:
                cell = _lookup_summary(
                    detection_summary, method, lbl, qty
                )
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    n_labels = len(held_out_labels)
    lines.append(f"## Mean across {n_labels} held-out labels")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for method in method_order:
        display = DISPLAY_NAME_BY_METHOD.get(method, method)
        row = [display]
        for qty in MAIN_TABLE_1_QUANTITIES:
            cell = _lookup_summary(
                detection_summary, method, "overall", qty
            )
            row.append(_format_table_value(cell, qty))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", f"_Generated from `{source_note}`._", ""])
    return "\n".join(lines)


def _render_external_table_2(
    shared_summary: pd.DataFrame, *, source_note: str
) -> str:
    held_out_labels = [
        lbl
        for lbl in shared_summary["held_out_label"].unique()
        if lbl != "overall"
    ]
    metric_header = [
        MAIN_TABLE_2_COLUMN_HEADERS[q] for q in MAIN_TABLE_2_QUANTITIES
    ]
    header = ["Method"] + metric_header
    sep = ["---"] * len(header)

    lines = [
        "# Table 2: External baselines — shared-cell label transfer",
        "",
        "One sub-table per held-out label. Values are mean ± std across n=5 donor-split seeds.",
        "",
    ]

    method_order = [
        m
        for m in EXTERNAL_TABLE_ROW_ORDER
        if m in shared_summary["method"].values
    ]

    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in method_order:
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            row = [display]
            for qty in MAIN_TABLE_2_QUANTITIES:
                cell = _lookup_summary(
                    shared_summary, method, lbl, qty
                )
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    n_labels = len(held_out_labels)
    lines.append(f"## Mean across {n_labels} held-out labels")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for method in method_order:
        display = DISPLAY_NAME_BY_METHOD.get(method, method)
        row = [display]
        for qty in MAIN_TABLE_2_QUANTITIES:
            cell = _lookup_summary(
                shared_summary, method, "overall", qty
            )
            row.append(_format_table_value(cell, qty))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", f"_Generated from `{source_note}`._", ""])
    return "\n".join(lines)


def _render_external_table_s1(
    full_ref_summary: pd.DataFrame, *, source_note: str
) -> str:
    held_out_labels = [
        lbl
        for lbl in full_ref_summary["held_out_label"].unique()
        if lbl != "overall"
    ]
    quantities = (
        "full_reference_false_abstention_rate",
        "score_median",
        "score_p95",
        "forced_accuracy",
        "forced_macro_f1",
        "post_abstention_accuracy",
        "post_abstention_macro_f1",
        "coverage",
    )
    column_headers = {
        "full_reference_false_abstention_rate": "FalseAb",
        "score_median": "ScoreMed",
        "score_p95": "ScoreP95",
        "forced_accuracy": "ForcedAcc",
        "forced_macro_f1": "ForcedF1",
        "post_abstention_accuracy": "PostAcc",
        "post_abstention_macro_f1": "PostF1",
        "coverage": "Coverage",
    }
    metric_header = [column_headers[q] for q in quantities]
    header = ["Method"] + metric_header
    sep = ["---"] * len(header)

    lines = [
        "# Table S1: External baselines — full-reference negative control",
        "",
        "One sub-table per held-out label. Values are mean ± std across n=5 donor-split seeds.",
        "",
    ]

    method_order = [
        m
        for m in EXTERNAL_TABLE_ROW_ORDER
        if m in full_ref_summary["method"].values
    ]

    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in method_order:
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            row = [display]
            for qty in quantities:
                cell = _lookup_summary(
                    full_ref_summary, method, lbl, qty
                )
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    n_labels = len(held_out_labels)
    lines.append(f"## Mean across {n_labels} held-out labels")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for method in method_order:
        display = DISPLAY_NAME_BY_METHOD.get(method, method)
        row = [display]
        for qty in quantities:
            cell = _lookup_summary(
                full_ref_summary, method, "overall", qty
            )
            row.append(_format_table_value(cell, qty))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", f"_Generated from `{source_note}`._", ""])
    return "\n".join(lines)


def _render_external_markdown_summary(
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    full_ref_summary: pd.DataFrame,
    *,
    report_title: str,
    report_description: str,
    expected_held_out_labels: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> str:
    label_count = len(expected_held_out_labels)
    seed_count = len(expected_seeds)
    lines = [
        f"# {report_title}",
        "",
        report_description,
        "",
        "Default external reference-mapping baselines (Seurat, "
        "SingleR, CellTypist, scmap-cell) across "
        f"{label_count} held-out labels x {seed_count} donor-split seeds.",
        "",
        "Variability summaries are descriptive across donor-split seeds.",
        "",
        "## Detection Summary (incomplete reference)",
        "",
        *_markdown_table(detection_summary),
        "",
        "## Shared-Cell Label Transfer Summary (incomplete reference)",
        "",
        *_markdown_table(shared_summary),
        "",
        "## Full-Reference Negative Control Summary",
        "",
        *_markdown_table(full_ref_summary),
        "",
    ]
    return "\n".join(lines)


def _lookup_summary(
    summary: pd.DataFrame,
    method: str,
    held_out_label: str,
    quantity: str,
) -> object:
    mask = (
        (summary["method"] == method)
        & (summary["held_out_label"] == held_out_label)
        & (summary["quantity"] == quantity)
    )
    matches = summary.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]
