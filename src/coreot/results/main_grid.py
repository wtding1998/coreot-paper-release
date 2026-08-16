from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.artifacts.run_artifacts import ArtifactBatchError, ArtifactFailure
from coreot.results.figures import Figure1Paths, Figure2Paths, write_figure_1, write_figure_2
from coreot.results.grid_common import (
    DISPLAY_NAME_BY_METHOD,
    EXTERNAL_TABLE_ROW_ORDER,
    LABEL_TRANSFER_METHODS,
    MAIN_TABLE_1_COLUMN_HEADERS,
    MAIN_TABLE_1_QUANTITIES,
    MAIN_TABLE_2_COLUMN_HEADERS,
    MAIN_TABLE_2_QUANTITIES,
    METHOD_GROUP_BY_METHOD,
    PRIMARY_SCORE_BY_METHOD,
    STAGE,
    TABLE_ROW_ORDER,
    ResultsGridError,
    _format_table_value,
    _markdown_table,
    _partition_main_grid_runs,
    build_detection_by_run,
    build_forced_label_summary_by_run,
    build_full_reference_false_abstention_by_run,
    build_shared_label_transfer_by_run,
    discover_run_descriptors,
    primary_score_for_method,
    summarize_run_metrics,
)

TUNED_BASELINE_METHODS: tuple[str, ...] = ("prior_only", "nn", "balanced_ot")


@dataclass(frozen=True)
class ResultsArtifactPaths:
    output_root: Path
    detection_by_run: Path
    detection_summary: Path
    forced_label_summary_by_run: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    full_reference_false_abstention_by_run: Path
    full_reference_false_abstention_summary: Path
    main_tables: Path
    figure_1: Figure1Paths | None
    figure_2: Figure2Paths | None
    markdown_summary: Path
    manifest: Path


@dataclass(frozen=True)
class TunedBaselineArtifactPaths:
    output_root: Path
    tables_root: Path
    detection_by_run: Path
    detection_summary: Path
    forced_label_summary_by_run: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    full_reference_false_abstention_by_run: Path
    full_reference_false_abstention_summary: Path
    manifest: Path


def write_tuned_baseline_bundle(
    *,
    source_root: str | Path,
    output_root: str | Path,
    methods: tuple[str, ...] = TUNED_BASELINE_METHODS,
) -> TunedBaselineArtifactPaths:
    source = Path(source_root)
    source_tables = source / "tables"
    output = Path(output_root)
    tables_root = output / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    paths = TunedBaselineArtifactPaths(
        output_root=output,
        tables_root=tables_root,
        detection_by_run=tables_root / "main_detection_by_run.csv",
        detection_summary=tables_root / "main_detection_summary.csv",
        forced_label_summary_by_run=tables_root / "forced_label_summary_by_run.csv",
        shared_label_transfer_by_run=tables_root / "shared_label_transfer_by_run.csv",
        shared_label_transfer_summary=tables_root / "shared_label_transfer_summary.csv",
        full_reference_false_abstention_by_run=tables_root
        / "full_reference_false_abstention_by_run.csv",
        full_reference_false_abstention_summary=tables_root
        / "full_reference_false_abstention_summary.csv",
        manifest=output / "manifest.yaml",
    )
    table_paths = {
        "main_detection_by_run": paths.detection_by_run,
        "main_detection_summary": paths.detection_summary,
        "forced_label_summary_by_run": paths.forced_label_summary_by_run,
        "shared_label_transfer_by_run": paths.shared_label_transfer_by_run,
        "shared_label_transfer_summary": paths.shared_label_transfer_summary,
        "full_reference_false_abstention_by_run": paths.full_reference_false_abstention_by_run,
        "full_reference_false_abstention_summary": paths.full_reference_false_abstention_summary,
    }
    for destination in table_paths.values():
        source_path = source_tables / destination.name
        filtered = _read_filtered_methods_table(source_path=source_path, methods=methods)
        filtered.to_csv(destination, index=False)

    write_manifest(
        paths.manifest,
        Manifest(
            stage=STAGE,
            artifacts={name: str(path) for name, path in table_paths.items()},
            metadata={
                "bundle_name": "tuned_baseline",
                "source_results_root": str(source),
                "source_manifest": str(source / "manifest.yaml"),
                "methods": list(methods),
            },
        ),
    )
    return paths


def _read_filtered_methods_table(*, source_path: Path, methods: tuple[str, ...]) -> pd.DataFrame:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source manuscript-results table not found: {source_path}. "
            "Run generate_hiha_dc_results.py first."
        )
    frame = pd.read_csv(source_path)
    if "method" not in frame.columns:
        raise ResultsGridError(f"Expected a 'method' column in {source_path}")
    return frame.loc[frame["method"].isin(methods)].copy()


def write_grid_results(
    *,
    runs_root: str | Path,
    grid_dir: str | Path,
    output_root: str | Path,
    candidate_set: str,
    condition: str = "incomplete_reference",
    embedding_name: str = "hiha_harmony30",
    write_figures: bool = True,
    methods: tuple[str, ...] = tuple(PRIMARY_SCORE_BY_METHOD),
    skip_incomplete: bool = False,
    source_note: str | None = None,
    score_overrides: dict[str, str | tuple[str, ...]] | None = None,
) -> ResultsArtifactPaths:
    if source_note is None:
        source_note = f"{output_root}/tables/main_detection_summary.csv"
    descriptors = discover_run_descriptors(grid_dir)
    if not descriptors:
        raise ResultsGridError(f"No run descriptors found in {grid_dir}")
    completed_descriptors, failures = _partition_main_grid_runs(
        runs_root=Path(runs_root),
        descriptors=descriptors,
        candidate_set=candidate_set,
        conditions=(condition, "full_reference_control"),
    )
    if failures and not skip_incomplete:
        raise ArtifactBatchError(failures)
    descriptors_to_use = completed_descriptors
    if not descriptors_to_use:
        raise ResultsGridError("No completed runs available for manuscript-results generation")

    output = Path(output_root)
    tables_root = output / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    detection_by_run = build_detection_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        candidate_set=candidate_set,
        condition=condition,
        methods=methods,
        grid_dir=Path(grid_dir),
    )
    detection_by_run["score"] = detection_by_run.get(
        "score", detection_by_run["primary_score"]
    )
    if score_overrides:
        extra_frames: list[pd.DataFrame] = []
        for method, alt_scores in score_overrides.items():
            if isinstance(alt_scores, str):
                alt_scores = (alt_scores,)
            for alt_score in alt_scores:
                extra = build_detection_by_run(
                    runs_root=Path(runs_root),
                    descriptors=descriptors_to_use,
                    candidate_set=candidate_set,
                    condition=condition,
                    methods=(method,),
                    grid_dir=Path(grid_dir),
                    score_overrides={method: alt_score},
                )
                extra_frames.append(extra)
        if extra_frames:
            detection_by_run = pd.concat(
                [detection_by_run] + extra_frames, ignore_index=True
            )
    detection_summary = summarize_run_metrics(
        detection_by_run,
        group_columns=("held_out_label", "method", "method_group", "score"),
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
    forced_label_summary = build_forced_label_summary_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        methods=tuple(method for method in methods if method in LABEL_TRANSFER_METHODS),
    )
    label_transfer_methods = tuple(
        method for method in methods if method in LABEL_TRANSFER_METHODS
    )
    shared_by_run = build_shared_label_transfer_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        candidate_set=candidate_set,
        condition=condition,
        methods=label_transfer_methods,
        grid_dir=Path(grid_dir),
    )
    shared_by_run["score"] = shared_by_run.get(
        "score", shared_by_run["method"].map(primary_score_for_method)
    )
    if score_overrides:
        extra_shared_frames: list[pd.DataFrame] = []
        for method, alt_scores in score_overrides.items():
            if method not in label_transfer_methods:
                continue
            if isinstance(alt_scores, str):
                alt_scores = (alt_scores,)
            for alt_score in alt_scores:
                extra = build_shared_label_transfer_by_run(
                    runs_root=Path(runs_root),
                    descriptors=descriptors_to_use,
                    candidate_set=candidate_set,
                    condition=condition,
                    methods=(method,),
                    grid_dir=Path(grid_dir),
                    score_overrides={method: alt_score},
                )
                extra_shared_frames.append(extra)
        if extra_shared_frames:
            shared_by_run = pd.concat(
                [shared_by_run] + extra_shared_frames, ignore_index=True
            )
    shared_summary = summarize_run_metrics(
        shared_by_run,
        group_columns=("held_out_label", "method", "method_group", "score"),
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
    full_reference_false_abstention = build_full_reference_false_abstention_by_run(
        runs_root=Path(runs_root),
        descriptors=descriptors_to_use,
        candidate_set=candidate_set,
        condition="full_reference_control",
        methods=methods,
        grid_dir=Path(grid_dir),
    )
    if score_overrides:
        extra_full_reference_frames: list[pd.DataFrame] = []
        for method, alt_scores in score_overrides.items():
            if isinstance(alt_scores, str):
                alt_scores = (alt_scores,)
            for alt_score in alt_scores:
                extra = build_full_reference_false_abstention_by_run(
                    runs_root=Path(runs_root),
                    descriptors=descriptors_to_use,
                    candidate_set=candidate_set,
                    condition="full_reference_control",
                    methods=(method,),
                    grid_dir=Path(grid_dir),
                    score_overrides={method: alt_score},
                )
                extra_full_reference_frames.append(extra)
        if extra_full_reference_frames:
            full_reference_false_abstention = pd.concat(
                [full_reference_false_abstention] + extra_full_reference_frames,
                ignore_index=True,
            )
    full_reference_false_abstention_summary = summarize_run_metrics(
        full_reference_false_abstention,
        group_columns=("held_out_label", "method", "method_group", "score"),
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

    paths = ResultsArtifactPaths(
        output_root=output,
        detection_by_run=tables_root / "main_detection_by_run.csv",
        detection_summary=tables_root / "main_detection_summary.csv",
        forced_label_summary_by_run=tables_root / "forced_label_summary_by_run.csv",
        shared_label_transfer_by_run=tables_root / "shared_label_transfer_by_run.csv",
        shared_label_transfer_summary=tables_root / "shared_label_transfer_summary.csv",
        full_reference_false_abstention_by_run=tables_root
        / "full_reference_false_abstention_by_run.csv",
        full_reference_false_abstention_summary=tables_root
        / "full_reference_false_abstention_summary.csv",
        main_tables=tables_root / "main_tables.md",
        figure_1=None,
        figure_2=None,
        markdown_summary=output / "summary.md",
        manifest=output / "manifest.yaml",
    )
    detection_by_run.to_csv(paths.detection_by_run, index=False)
    detection_summary.to_csv(paths.detection_summary, index=False)
    forced_label_summary.to_csv(paths.forced_label_summary_by_run, index=False)
    shared_by_run.to_csv(paths.shared_label_transfer_by_run, index=False)
    shared_summary.to_csv(paths.shared_label_transfer_summary, index=False)
    full_reference_false_abstention.to_csv(
        paths.full_reference_false_abstention_by_run, index=False
    )
    full_reference_false_abstention_summary.to_csv(
        paths.full_reference_false_abstention_summary, index=False
    )
    paths.main_tables.write_text(
        render_main_tables(
            detection_summary,
            shared_summary,
            full_reference_false_abstention_summary,
            source_note=source_note,
        ),
        encoding="utf-8",
    )
    for stale_markdown in (tables_root / "main_table_1.md", tables_root / "main_table_2.md"):
        if stale_markdown.exists():
            stale_markdown.unlink()
    paths.markdown_summary.write_text(
        render_markdown_summary(
            detection_summary,
            shared_summary,
            skipped_failures=tuple(failures),
        ),
        encoding="utf-8",
    )
    figure_1_paths = None
    figure_2_paths = None
    if write_figures:
        figure_1_paths = write_figure_1(
            runs_root=runs_root,
            output_root=output,
            detection_summary_path=paths.detection_summary,
            forced_label_summary_path=paths.forced_label_summary_by_run,
            condition=condition,
            candidate_set=candidate_set,
            embedding_name=embedding_name,
        )
        figure_2_paths = write_figure_2(
            output_root=output,
            detection_summary_path=paths.detection_summary,
            full_reference_summary_path=paths.full_reference_false_abstention_summary,
        )
        paths = ResultsArtifactPaths(
            output_root=paths.output_root,
            detection_by_run=paths.detection_by_run,
            detection_summary=paths.detection_summary,
            forced_label_summary_by_run=paths.forced_label_summary_by_run,
            shared_label_transfer_by_run=paths.shared_label_transfer_by_run,
            shared_label_transfer_summary=paths.shared_label_transfer_summary,
            full_reference_false_abstention_by_run=paths.full_reference_false_abstention_by_run,
            full_reference_false_abstention_summary=paths.full_reference_false_abstention_summary,
            main_tables=paths.main_tables,
            figure_1=figure_1_paths,
            figure_2=figure_2_paths,
            markdown_summary=paths.markdown_summary,
            manifest=paths.manifest,
        )
    artifacts = {
        "main_detection_by_run": str(paths.detection_by_run),
        "main_detection_summary": str(paths.detection_summary),
        "forced_label_summary_by_run": str(paths.forced_label_summary_by_run),
        "shared_label_transfer_by_run": str(paths.shared_label_transfer_by_run),
        "shared_label_transfer_summary": str(paths.shared_label_transfer_summary),
        "full_reference_false_abstention_by_run": str(
            paths.full_reference_false_abstention_by_run
        ),
        "full_reference_false_abstention_summary": str(
            paths.full_reference_false_abstention_summary
        ),
        "main_tables": str(paths.main_tables),
        "markdown_summary": str(paths.markdown_summary),
    }
    if figure_1_paths is not None:
        artifacts["figure_1_png"] = str(figure_1_paths.png)
        artifacts["figure_1_pdf"] = str(figure_1_paths.pdf)
    if figure_2_paths is not None:
        artifacts["figure_2_png"] = str(figure_2_paths.png)
        artifacts["figure_2_pdf"] = str(figure_2_paths.pdf)
    write_manifest(
        paths.manifest,
        Manifest(
            stage=STAGE,
            artifacts=artifacts,
            metadata={
                "runs_root": str(runs_root),
                "grid_dir": str(grid_dir),
                "condition": condition,
                "candidate_set": candidate_set,
                "embedding_name": embedding_name,
                "skip_incomplete": skip_incomplete,
                "n_runs": len(descriptors_to_use),
                "n_runs_expected": len(descriptors),
                "n_runs_completed": len(descriptors_to_use),
                "skipped_runs": [_artifact_failure_metadata(failure) for failure in failures],
                "primary_score_by_method": PRIMARY_SCORE_BY_METHOD,
                "method_group_by_method": METHOD_GROUP_BY_METHOD,
                "variability": "descriptive_seed_level_mean_std_sem",
                "figure_1": {
                    "held_out_label": "ISG+ cDC2",
                    "run_id": "hiha_dc_isg_cdc2_seed1",
                    "seed": 1,
                    "method": "coreot_full",
                }
                if figure_1_paths is not None
                else None,
                "figure_2": {
                    "held_out_labels": ["CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2"],
                    "methods": [
                        "prior_only",
                        "uniform_uot",
                        "coreot_constant_tau",
                        "coreot_full",
                    ],
                    "full_reference_condition": "full_reference_control",
                }
                if figure_2_paths is not None
                else None,
            },
        ),
    )
    return paths

def render_markdown_summary(
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    *,
    skipped_failures: tuple[ArtifactFailure, ...] = (),
) -> str:
    lines = [
        "# HIHA DC Main Grid Manuscript Results",
        "",
        "These tables are generated from completed run artifacts. Variability summaries are descriptive across donor-split seeds.",
        "",
        "## Main Detection Summary",
        "",
        *_markdown_table(detection_summary),
        "",
        "## Shared-Cell Label Transfer Summary",
        "",
        *_markdown_table(shared_summary),
        "",
    ]
    if skipped_failures:
        lines.extend(
            [
                "## Skipped Runs",
                "",
                "Skip-incomplete mode excluded the following runs because required artifacts were missing, invalid, or access-denied.",
                "",
                *_markdown_table(
                    pd.DataFrame(
                        [
                            _artifact_failure_metadata(failure)
                            for failure in skipped_failures
                        ]
                    )
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _artifact_failure_metadata(failure: ArtifactFailure) -> dict[str, object]:
    return {
        "run_id": failure.run_id,
        "artifact_kind": failure.artifact_kind,
        "state": failure.state,
        "condition": failure.condition,
        "candidate_set": failure.candidate_set,
        "method": failure.method,
        "path": str(failure.path),
        "detail": failure.detail,
    }


def render_main_table_1(
    detection_summary: pd.DataFrame,
    *,
    source_note: str | None = None,
) -> str:
    """Render Main Table 1: absent-state detection.

    One sub-table per held-out label. Each sub-table: rows = methods present,
    columns = AUROC, AUPRC, AUPRC bl, AbsAb, ShFA.
    """
    held_out_labels = [
        lbl
        for lbl in detection_summary["held_out_label"].unique()
        if lbl != "overall"
    ]
    if source_note is None:
        source_note = "results/HIHA_DC/main/tables/main_detection_summary.csv"
    lines = [
        "# Main Table 1: absent-state detection",
        "",
        "One sub-table per held-out label. Rows: methods. Columns: AUROC, AUPRC, AUPRC baseline, absent-cell abstention rate, shared-cell false-abstention rate.",
        "",
        "Values are mean ± std across n=5 donor-split seeds.",
        "",
    ]

    metric_header = [MAIN_TABLE_1_COLUMN_HEADERS[q] for q in MAIN_TABLE_1_QUANTITIES]
    header = ["Method"] + metric_header
    sep = ["---"] * len(header)

    method_order = _method_order_from_summary(detection_summary, include_prior=True)
    scores_present = sorted(
        set(detection_summary.get("score", pd.Series(dtype=str)).dropna().unique())
    )
    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in method_order:
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            method_scores = [
                s for s in scores_present
                if _lookup_summary(detection_summary, method, lbl, MAIN_TABLE_1_QUANTITIES[0], score=s) is not None
            ]
            for score in method_scores:
                label = f"{display} ({score})" if len(method_scores) > 1 else display
                row = [label]
                for qty in MAIN_TABLE_1_QUANTITIES:
                    cell = _lookup_summary(detection_summary, method, lbl, qty, score=score)
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
        method_scores = [
            s for s in scores_present
            if _lookup_summary(detection_summary, method, "overall", MAIN_TABLE_1_QUANTITIES[0], score=s) is not None
        ]
        for score in method_scores:
            label = f"{display} ({score})" if len(method_scores) > 1 else display
            row = [label]
            for qty in MAIN_TABLE_1_QUANTITIES:
                cell = _lookup_summary(detection_summary, method, "overall", qty, score=score)
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", f"_Generated from `{source_note}`._", ""])
    return "\n".join(lines)


def _lookup_summary(
    summary: pd.DataFrame, method: str, held_out_label: str, quantity: str, score: str | None = None
) -> object:
    """Look up a single (method, held_out_label, quantity, score) row from a summary DataFrame."""
    mask = (
        (summary["method"] == method)
        & (summary["held_out_label"] == held_out_label)
        & (summary["quantity"] == quantity)
    )
    if score is not None and "score" in summary.columns:
        mask = mask & (summary["score"] == score)
    matches = summary.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_main_tables(
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    full_reference_summary: pd.DataFrame,
    *,
    source_note: str | None = None,
) -> str:
    """Render the selected-report manuscript tables in one markdown artifact."""
    sections = [
        render_main_table_1(detection_summary, source_note=source_note),
        render_main_table_2(shared_summary, source_note=source_note),
        render_full_reference_table(full_reference_summary, source_note=source_note),
    ]
    return "\n\n".join(section.strip() for section in sections) + "\n"


def render_main_table_2(
    shared_summary: pd.DataFrame,
    *,
    source_note: str | None = None,
) -> str:
    """Render Main Table 2: shared-cell label transfer.

    One sub-table per held-out label, then aggregate. Rows: label-transfer
    methods. Columns: forced accuracy, forced macro-F1, post-abstention
    accuracy, post-abstention macro-F1, coverage, shared false-abstention rate.
    """
    held_out_labels = [
        lbl
        for lbl in shared_summary["held_out_label"].unique()
        if lbl != "overall"
    ]
    n_labels = len(held_out_labels)
    if source_note is None:
        source_note = "results/HIHA_DC/main/tables/main_detection_summary.csv"

    metric_header = [MAIN_TABLE_2_COLUMN_HEADERS[q] for q in MAIN_TABLE_2_QUANTITIES]
    header = ["Method"] + metric_header
    sep = ["---"] * len(header)

    methods = _method_order_from_summary(shared_summary, include_prior=False)
    scores_present = sorted(
        set(shared_summary.get("score", pd.Series(dtype=str)).dropna().unique())
    )

    lines = [
        "# Main Table 2: shared-cell label transfer",
        "",
        "Rows: label-transfer methods (prior-only excluded). Columns: forced and post-abstention accuracy/F1, coverage, false-abstention rate.",
        "",
        "Values are mean ± std across n=5 donor-split seeds.",
        "",
    ]

    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in methods:
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            method_scores = [
                s for s in scores_present
                if _lookup_summary(shared_summary, method, lbl, MAIN_TABLE_2_QUANTITIES[0], score=s) is not None
            ]
            for score in method_scores:
                label = f"{display} ({score})" if len(method_scores) > 1 else display
                row = [label]
                for qty in MAIN_TABLE_2_QUANTITIES:
                    cell = _lookup_summary(shared_summary, method, lbl, qty, score=score)
                    row.append(_format_table_value(cell, qty))
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Aggregate across all held-out labels
    lines.append(f"## Mean across {n_labels} held-out labels")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for method in methods:
        display = DISPLAY_NAME_BY_METHOD.get(method, method)
        method_scores = [
            s for s in scores_present
            if _lookup_summary(shared_summary, method, "overall", MAIN_TABLE_2_QUANTITIES[0], score=s) is not None
        ]
        for score in method_scores:
            label = f"{display} ({score})" if len(method_scores) > 1 else display
            row = [label]
            for qty in MAIN_TABLE_2_QUANTITIES:
                cell = _lookup_summary(shared_summary, method, "overall", qty, score=score)
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", f"_Generated from `{source_note}`._", ""])
    return "\n".join(lines)


def render_full_reference_table(
    full_reference_summary: pd.DataFrame,
    *,
    source_note: str | None = None,
) -> str:
    held_out_labels = [
        lbl
        for lbl in full_reference_summary["held_out_label"].unique()
        if lbl != "overall"
    ]
    n_labels = len(held_out_labels)
    if source_note is None:
        source_note = "results/HIHA_DC/main/tables/main_detection_summary.csv"

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
    header = ["Method"] + [column_headers[q] for q in quantities]
    sep = ["---"] * len(header)
    methods = _method_order_from_summary(full_reference_summary, include_prior=True)
    scores_present = sorted(
        set(full_reference_summary.get("score", pd.Series(dtype=str)).dropna().unique())
    )

    lines = [
        "# Main Table S1: full-reference negative control",
        "",
        "Rows: methods. Columns: full-reference false-abstention rate, primary-score summaries, forced and post-abstention accuracy/F1, and coverage.",
        "",
        "Values are mean ± std across n=5 donor-split seeds.",
        "",
    ]

    for lbl in held_out_labels:
        lines.append(f"## {lbl}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for method in methods:
            display = DISPLAY_NAME_BY_METHOD.get(method, method)
            method_scores = [
                s
                for s in scores_present
                if _lookup_summary(
                    full_reference_summary, method, lbl, quantities[0], score=s
                )
                is not None
            ]
            for score in method_scores:
                label = f"{display} ({score})" if len(method_scores) > 1 else display
                row = [label]
                for qty in quantities:
                    cell = _lookup_summary(full_reference_summary, method, lbl, qty, score=score)
                    row.append(_format_table_value(cell, qty))
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append(f"## Mean across {n_labels} held-out labels")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for method in methods:
        display = DISPLAY_NAME_BY_METHOD.get(method, method)
        method_scores = [
            s
            for s in scores_present
            if _lookup_summary(
                full_reference_summary, method, "overall", quantities[0], score=s
            )
            is not None
        ]
        for score in method_scores:
            label = f"{display} ({score})" if len(method_scores) > 1 else display
            row = [label]
            for qty in quantities:
                cell = _lookup_summary(full_reference_summary, method, "overall", qty, score=score)
                row.append(_format_table_value(cell, qty))
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", f"_Generated from `{source_note}`._", ""])
    return "\n".join(lines)


def _method_order_from_summary(summary: pd.DataFrame, *, include_prior: bool) -> list[str]:
    present = [str(method) for method in summary.get("method", pd.Series(dtype=str)).dropna().unique()]
    preferred = list(TABLE_ROW_ORDER + EXTERNAL_TABLE_ROW_ORDER)
    if not include_prior:
        preferred = [method for method in preferred if method != "prior_only"]
    ordered = [method for method in preferred if method in present]
    extras = sorted(method for method in present if method not in set(ordered))
    return ordered + extras


def _source_note() -> str:
    return "results/HIHA_DC/main/tables/main_detection_summary.csv"
