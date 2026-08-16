from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.results.external_baselines import write_external_baseline_results
from coreot.results.grid_common import (
    DISPLAY_NAME_BY_METHOD,
    MAIN_TABLE_1_COLUMN_HEADERS,
    MAIN_TABLE_1_QUANTITIES,
    MAIN_TABLE_2_COLUMN_HEADERS,
    MAIN_TABLE_2_QUANTITIES,
    STAGE,
    _format_table_value,
    primary_score_for_method,
)
from coreot.results.main_grid import write_grid_results
from coreot.results.main_grid import (
    ResultsArtifactPaths,
    render_main_tables,
    render_markdown_summary,
)

COMPARE_INTERNAL_METHODS: tuple[str, ...] = (
    "prior_only",
    "nn",
    "balanced_ot",
    "uniform_uot",
    "coreot_full",
    "coreot_match_only",
)


def select_primary_comparison_rows(
    frame: pd.DataFrame,
    *,
    source: str | Path = "comparison table",
) -> pd.DataFrame:
    """Select each method's declared primary score from a comparison table."""
    missing_columns = sorted({"method", "score"} - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{source} is missing columns {missing_columns}.")
    if frame.empty:
        raise ValueError(f"{source} is empty.")

    methods = frame["method"].astype(str)
    expected_scores = methods.map(primary_score_for_method)
    selected = frame.loc[frame["score"].astype(str).eq(expected_scores)].copy()
    missing_methods = sorted(set(methods) - set(selected["method"].astype(str)))
    if missing_methods:
        raise ValueError(
            f"{source} is missing primary-score rows for methods {missing_methods}."
        )
    return selected


@dataclass(frozen=True)
class CompareBaselinesArtifactPaths:
    output_root: Path
    internal_root: Path
    external_root: Path
    tables_root: Path
    detection_by_run: Path
    detection_summary: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    full_reference_by_run: Path
    full_reference_summary: Path
    main_tables: Path
    markdown_summary: Path
    manifest: Path


def write_compare_baselines_results(
    *,
    runs_root: str | Path,
    grid_dir: str | Path,
    output_root: str | Path,
    external_grid_dir: str | Path | None = None,
    condition: str = "incomplete_reference",
    internal_candidate_set: str = "hiha_harmony30_k100",
    embedding_name: str = "hiha_harmony30",
    internal_methods: tuple[str, ...] = COMPARE_INTERNAL_METHODS,
    internal_method_grid_dirs: dict[str, str | Path] | None = None,
    internal_score_overrides: dict[str, str | tuple[str, ...]] | None = None,
    expected_external_held_out_labels: tuple[str, ...] | None = None,
    expected_external_seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    external_methods: tuple[str, ...] | None = None,
    external_report_title: str = "External Baselines Report - leave-one HIHA DC",
    external_report_description: str = (
        "External reference-mapping baselines evaluated on the selected "
        "report_leave_one_HIHA_DC run roots."
    ),
    summary_title: str = "HIHA DC Baseline Comparison",
    summary_description: str = (
        "This bundle regenerates the selected internal CoRe-OT report and the "
        "external reference-mapping baseline report from run artifacts, then "
        "writes merged comparison tables."
    ),
    skip_incomplete: bool = False,
) -> CompareBaselinesArtifactPaths:
    resolved_score_overrides = (
        {"coreot_full": "u_tilde"}
        if internal_score_overrides is None
        else internal_score_overrides
    )
    replacement_grid_dirs = dict(internal_method_grid_dirs or {})
    unknown_replacements = sorted(set(replacement_grid_dirs) - set(internal_methods))
    if unknown_replacements:
        raise ValueError(
            "Replacement grid supplied for method(s) outside internal_methods: "
            + ", ".join(unknown_replacements)
        )
    output = Path(output_root)
    internal_root = output / "internal"
    external_root = output / "external_baselines"
    tables_root = output / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    source_note = f"{output}/tables"
    if replacement_grid_dirs:
        selected_methods = tuple(
            method for method in internal_methods if method not in replacement_grid_dirs
        )
        internal_sources = [
            write_grid_results(
                runs_root=runs_root,
                grid_dir=grid_dir,
                output_root=internal_root / "selected_methods",
                condition=condition,
                candidate_set=internal_candidate_set,
                embedding_name=embedding_name,
                methods=selected_methods,
                write_figures=False,
                skip_incomplete=skip_incomplete,
                source_note=source_note,
                score_overrides={
                    method: scores
                    for method, scores in resolved_score_overrides.items()
                    if method in selected_methods
                },
            )
        ]
        for method in internal_methods:
            replacement_grid_dir = replacement_grid_dirs.get(method)
            if replacement_grid_dir is None:
                continue
            internal_sources.append(
                write_grid_results(
                    runs_root=runs_root,
                    grid_dir=replacement_grid_dir,
                    output_root=internal_root / method,
                    condition=condition,
                    candidate_set=internal_candidate_set,
                    embedding_name=embedding_name,
                    methods=(method,),
                    write_figures=False,
                    skip_incomplete=skip_incomplete,
                    source_note=source_note,
                    score_overrides={
                        method: resolved_score_overrides[method]
                    }
                    if method in resolved_score_overrides
                    else None,
                )
            )
        internal_paths = _write_combined_internal_results(
            output_root=internal_root,
            sources=tuple(internal_sources),
            source_note=source_note,
        )
    else:
        internal_paths = write_grid_results(
            runs_root=runs_root,
            grid_dir=grid_dir,
            output_root=internal_root,
            condition=condition,
            candidate_set=internal_candidate_set,
            embedding_name=embedding_name,
            methods=internal_methods,
            write_figures=False,
            skip_incomplete=skip_incomplete,
            source_note=source_note,
            score_overrides=resolved_score_overrides,
        )
    external_paths = write_external_baseline_results(
        runs_root=runs_root,
        grid_dir=external_grid_dir or grid_dir,
        output_root=external_root,
        expected_held_out_labels=expected_external_held_out_labels,
        expected_seeds=expected_external_seeds,
        report_title=external_report_title,
        report_description=external_report_description,
        source_note=source_note,
        methods=external_methods,
    )
    realized_external_methods = tuple(
        pd.read_csv(external_paths.detection_by_run)["method"].drop_duplicates()
    )

    paths = CompareBaselinesArtifactPaths(
        output_root=output,
        internal_root=internal_root,
        external_root=external_root,
        tables_root=tables_root,
        detection_by_run=tables_root / "compare_detection_by_run.csv",
        detection_summary=tables_root / "compare_detection_summary.csv",
        shared_label_transfer_by_run=tables_root
        / "compare_shared_label_transfer_by_run.csv",
        shared_label_transfer_summary=tables_root
        / "compare_shared_label_transfer_summary.csv",
        full_reference_by_run=tables_root / "compare_full_reference_by_run.csv",
        full_reference_summary=tables_root
        / "compare_full_reference_summary.csv",
        main_tables=tables_root / "compare_tables.md",
        markdown_summary=output / "summary.md",
        manifest=output / "manifest.yaml",
    )

    detection_by_run = _combine_detection_by_run(
        internal_paths.detection_by_run, external_paths.detection_by_run
    )
    detection_summary = _combine_summary(
        internal_paths.detection_summary, external_paths.detection_summary
    )
    shared_by_run = _combine_shared_by_run(
        internal_paths.shared_label_transfer_by_run,
        external_paths.shared_label_transfer_by_run,
    )
    shared_summary = _combine_summary(
        internal_paths.shared_label_transfer_summary,
        external_paths.shared_label_transfer_summary,
    )
    full_reference_by_run = _combine_full_reference_by_run(
        internal_paths.full_reference_false_abstention_by_run,
        external_paths.full_reference_by_run,
    )
    full_reference_summary = _combine_summary(
        internal_paths.full_reference_false_abstention_summary,
        external_paths.full_reference_summary,
    )

    primary_detection_summary = select_primary_comparison_rows(
        detection_summary, source=paths.detection_summary
    )
    primary_shared_summary = select_primary_comparison_rows(
        shared_summary, source=paths.shared_label_transfer_summary
    )
    primary_full_reference_summary = select_primary_comparison_rows(
        full_reference_summary, source=paths.full_reference_summary
    )

    detection_by_run.to_csv(paths.detection_by_run, index=False)
    detection_summary.to_csv(paths.detection_summary, index=False)
    shared_by_run.to_csv(paths.shared_label_transfer_by_run, index=False)
    shared_summary.to_csv(paths.shared_label_transfer_summary, index=False)
    full_reference_by_run.to_csv(paths.full_reference_by_run, index=False)
    full_reference_summary.to_csv(paths.full_reference_summary, index=False)

    paths.main_tables.write_text(
        render_compare_baselines_tables(
            detection_summary=primary_detection_summary,
            shared_summary=primary_shared_summary,
            full_reference_summary=primary_full_reference_summary,
            source_note=source_note,
        ),
        encoding="utf-8",
    )
    paths.markdown_summary.write_text(
        render_compare_baselines_summary(
            detection_summary=primary_detection_summary,
            shared_summary=primary_shared_summary,
            full_reference_summary=primary_full_reference_summary,
            title=summary_title,
            description=summary_description,
        ),
        encoding="utf-8",
    )

    write_manifest(
        paths.manifest,
        Manifest(
            stage=STAGE,
            artifacts={
                "internal_results_root": str(paths.internal_root),
                "external_baseline_results_root": str(paths.external_root),
                "compare_detection_by_run": str(paths.detection_by_run),
                "compare_detection_summary": str(paths.detection_summary),
                "compare_shared_label_transfer_by_run": str(
                    paths.shared_label_transfer_by_run
                ),
                "compare_shared_label_transfer_summary": str(
                    paths.shared_label_transfer_summary
                ),
                "compare_full_reference_by_run": str(paths.full_reference_by_run),
                "compare_full_reference_summary": str(paths.full_reference_summary),
                "compare_tables": str(paths.main_tables),
                "markdown_summary": str(paths.markdown_summary),
            },
            metadata={
                "runs_root": str(runs_root),
                "grid_dir": str(grid_dir),
                "external_grid_dir": str(external_grid_dir or grid_dir),
                "condition": condition,
                "internal_candidate_set": internal_candidate_set,
                "external_candidate_set": "external_reference_mapping",
                "embedding_name": embedding_name,
                "skip_incomplete": skip_incomplete,
                "internal_methods": list(internal_methods),
                "internal_method_grid_dirs": {
                    method: str(path)
                    for method, path in replacement_grid_dirs.items()
                },
                "internal_score_overrides": resolved_score_overrides,
                "external_methods": list(realized_external_methods),
                "variability": "descriptive_seed_level_mean_std_sem",
            },
        ),
    )
    return paths


def _write_combined_internal_results(
    *,
    output_root: Path,
    sources: tuple[ResultsArtifactPaths, ...],
    source_note: str,
) -> ResultsArtifactPaths:
    tables_root = output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    paths = ResultsArtifactPaths(
        output_root=output_root,
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
        markdown_summary=output_root / "summary.md",
        manifest=output_root / "manifest.yaml",
    )
    table_attributes = (
        "detection_by_run",
        "detection_summary",
        "forced_label_summary_by_run",
        "shared_label_transfer_by_run",
        "shared_label_transfer_summary",
        "full_reference_false_abstention_by_run",
        "full_reference_false_abstention_summary",
    )
    combined: dict[str, pd.DataFrame] = {}
    for attribute in table_attributes:
        frame = pd.concat(
            [pd.read_csv(getattr(source, attribute)) for source in sources],
            ignore_index=True,
        )
        frame.to_csv(getattr(paths, attribute), index=False)
        combined[attribute] = frame

    paths.main_tables.write_text(
        render_main_tables(
            combined["detection_summary"],
            combined["shared_label_transfer_summary"],
            combined["full_reference_false_abstention_summary"],
            source_note=source_note,
        ),
        encoding="utf-8",
    )
    paths.markdown_summary.write_text(
        render_markdown_summary(
            combined["detection_summary"],
            combined["shared_label_transfer_summary"],
        ),
        encoding="utf-8",
    )
    write_manifest(
        paths.manifest,
        Manifest(
            stage=STAGE,
            artifacts={
                attribute: str(getattr(paths, attribute))
                for attribute in table_attributes
            }
            | {
                "main_tables": str(paths.main_tables),
                "markdown_summary": str(paths.markdown_summary),
            },
            metadata={
                "source_results_roots": [str(source.output_root) for source in sources],
                "variability": "descriptive_seed_level_mean_std_sem",
            },
        ),
    )
    return paths


def _combine_detection_by_run(internal_path: Path, external_path: Path) -> pd.DataFrame:
    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)
    external = external.assign(primary_score="u", score="u")
    return _concat_with_family(internal, external)


def _combine_shared_by_run(internal_path: Path, external_path: Path) -> pd.DataFrame:
    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)
    external = external.assign(score="u")
    return _concat_with_family(internal, external)


def _combine_full_reference_by_run(internal_path: Path, external_path: Path) -> pd.DataFrame:
    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)
    external = external.assign(score="u")
    return _concat_with_family(internal, external)


def _combine_summary(internal_path: Path, external_path: Path) -> pd.DataFrame:
    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)
    external = external.assign(score="u")
    return _concat_with_family(internal, external)


def _concat_with_family(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    internal = internal.copy()
    external = external.copy()
    internal.insert(0, "result_family", "internal")
    external.insert(0, "result_family", "external_baseline")
    all_columns = list(dict.fromkeys([*internal.columns, *external.columns]))
    return pd.concat(
        [internal.reindex(columns=all_columns), external.reindex(columns=all_columns)],
        ignore_index=True,
    )


def render_compare_baselines_tables(
    *,
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    full_reference_summary: pd.DataFrame,
    source_note: str,
) -> str:
    sections = [
        _render_detection_table(detection_summary, source_note=source_note),
        _render_shared_table(shared_summary, source_note=source_note),
        _render_full_reference_table(full_reference_summary, source_note=source_note),
    ]
    return "\n\n".join(section.strip() for section in sections) + "\n"


def render_compare_baselines_summary(
    *,
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    full_reference_summary: pd.DataFrame,
    title: str = "HIHA DC Baseline Comparison",
    description: str = (
        "This bundle regenerates the selected internal CoRe-OT report and the "
        "external reference-mapping baseline report from run artifacts, then "
        "writes merged comparison tables."
    ),
) -> str:
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "Internal CoRe-OT methods and external reference-mapping baselines are shown together for metric comparison, but their score definitions are method-specific and should not be described as the same diagnostic.",
        "",
        "## Generated Tables",
        "",
        "- `tables/compare_detection_by_run.csv`",
        "- `tables/compare_detection_summary.csv`",
        "- `tables/compare_shared_label_transfer_by_run.csv`",
        "- `tables/compare_shared_label_transfer_summary.csv`",
        "- `tables/compare_full_reference_by_run.csv`",
        "- `tables/compare_full_reference_summary.csv`",
        "- `tables/compare_tables.md`",
        "",
        "## Included Methods",
        "",
        *_method_lines(detection_summary),
        "",
        "## Metric Families",
        "",
        f"- Detection summary rows: {len(detection_summary)}",
        f"- Shared-cell label-transfer summary rows: {len(shared_summary)}",
        f"- Full-reference-control summary rows: {len(full_reference_summary)}",
        "",
    ]
    return "\n".join(lines)


def _render_detection_table(summary: pd.DataFrame, *, source_note: str) -> str:
    return _render_metric_table(
        title="Table 1: absent-state detection across internal and external baselines",
        description=(
            "Rows are methods and method-specific scores. Columns are AUROC, AUPRC, "
            "AUPRC baseline, absent-cell abstention rate, and shared-cell "
            "false-abstention rate."
        ),
        summary=summary,
        quantities=MAIN_TABLE_1_QUANTITIES,
        headers=MAIN_TABLE_1_COLUMN_HEADERS,
        include_prior=True,
        source_note=f"{source_note}/compare_detection_summary.csv",
    )


def _render_shared_table(summary: pd.DataFrame, *, source_note: str) -> str:
    return _render_metric_table(
        title="Table 2: shared-cell label transfer across internal and external baselines",
        description=(
            "Rows exclude prior-only because it does not emit transferred labels. "
            "Coverage and false-abstention rate are reported separately from "
            "post-abstention accuracy and macro-F1."
        ),
        summary=summary,
        quantities=MAIN_TABLE_2_QUANTITIES,
        headers=MAIN_TABLE_2_COLUMN_HEADERS,
        include_prior=False,
        source_note=f"{source_note}/compare_shared_label_transfer_summary.csv",
    )


def _render_full_reference_table(summary: pd.DataFrame, *, source_note: str) -> str:
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
    headers = {
        "full_reference_false_abstention_rate": "FalseAb",
        "score_median": "ScoreMed",
        "score_p95": "ScoreP95",
        "forced_accuracy": "ForcedAcc",
        "forced_macro_f1": "ForcedF1",
        "post_abstention_accuracy": "PostAcc",
        "post_abstention_macro_f1": "PostF1",
        "coverage": "Coverage",
    }
    return _render_metric_table(
        title="Table S1: full-reference control across internal and external baselines",
        description=(
            "Rows are methods and score definitions. Full-reference controls support "
            "threshold calibration and false-abstention checks; they do not contain "
            "absent-positive detection labels."
        ),
        summary=summary,
        quantities=quantities,
        headers=headers,
        include_prior=True,
        source_note=f"{source_note}/compare_full_reference_summary.csv",
    )


def _render_metric_table(
    *,
    title: str,
    description: str,
    summary: pd.DataFrame,
    quantities: tuple[str, ...],
    headers: dict[str, str],
    include_prior: bool,
    source_note: str,
) -> str:
    labels = [label for label in summary["held_out_label"].unique() if label != "overall"]
    method_keys = _ordered_method_keys(summary, include_prior=include_prior)
    table_header = ["Method", "Family"] + [headers[q] for q in quantities]
    sep = ["---"] * len(table_header)
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "Values are mean +/- std across donor-split seeds.",
        "",
    ]
    for label in [*labels, "overall"]:
        heading = "Mean across held-out labels" if label == "overall" else str(label)
        lines.extend([f"## {heading}", "", "| " + " | ".join(table_header) + " |"])
        lines.append("| " + " | ".join(sep) + " |")
        for result_family, method, score in method_keys:
            if not include_prior and method == "prior_only":
                continue
            first = _lookup(summary, result_family, method, score, label, quantities[0])
            if first is None:
                continue
            row = [_display_method(method, score), result_family]
            for quantity in quantities:
                row.append(
                    _format_table_value(
                        _lookup(summary, result_family, method, score, label, quantity),
                        quantity,
                    )
                )
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.extend([f"_Generated from `{source_note}`._", ""])
    return "\n".join(lines)


def _ordered_method_keys(
    summary: pd.DataFrame, *, include_prior: bool
) -> list[tuple[str, str, str]]:
    order = {
        method: idx
        for idx, method in enumerate(
            (
                "prior_only",
                "nn",
                "balanced_ot",
                "uniform_uot",
                "coreot_full",
                "coreot_match_only",
                "seurat_anchor",
                "singleR",
                "celltypist_l3",
                "scmap_cell",
            )
        )
    }
    keys = (
        summary[["result_family", "method", "score"]]
        .drop_duplicates()
        .dropna(subset=["method", "score"])
    )
    records = [
        (str(row.result_family), str(row.method), str(row.score))
        for row in keys.itertuples(index=False)
        if include_prior or row.method != "prior_only"
    ]
    return sorted(records, key=lambda key: (order.get(key[1], 999), key[2], key[0]))


def _lookup(
    summary: pd.DataFrame,
    result_family: str,
    method: str,
    score: str,
    held_out_label: str,
    quantity: str,
) -> pd.Series | None:
    mask = (
        (summary["result_family"] == result_family)
        & (summary["method"] == method)
        & (summary["score"] == score)
        & (summary["held_out_label"] == held_out_label)
        & (summary["quantity"] == quantity)
    )
    matches = summary.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def _display_method(method: str, score: str) -> str:
    display = DISPLAY_NAME_BY_METHOD.get(method, method)
    return f"{display} ({score})"


def _method_lines(summary: pd.DataFrame) -> list[str]:
    lines = []
    for result_family, method, score in _ordered_method_keys(summary, include_prior=True):
        lines.append(f"- {result_family}: {_display_method(method, score)}")
    return lines
