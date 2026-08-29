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
    _format_table_value,
    build_detection_by_run,
    build_shared_label_transfer_by_run,
    discover_run_descriptors,
    summarize_run_metrics,
)
from coreot.results.sweep_common import (
    DEFAULT_FIXED_REFERENCE_ROOT,
    FIXED_REFERENCE_METHODS,
    _build_labelwise_param_maps,
    _failed_run_ids,
    _is_missing_tau,
    _labelwise_swept_display_name,
    _lookup_labelwise_cell,
    _partition_completed_sweep_runs,
    _tau_or_none,
    read_fixed_reference_bundle,
    read_fixed_reference_rows,
)
from coreot.results.tau_sensitivity import (
    build_tau_detection_by_run,
    build_tau_shared_label_transfer_by_run,
    parse_tau_from_transport_config,
)

DEFAULT_EXCLUSION_FILTERS: dict[str, float] = {
    "shared_false_abstention_rate_max": 0.15,
    "coverage_min": 0.70,
    "post_abstention_macro_f1_min": 0.70,
}

HELD_OUT_LABELS: tuple[str, ...] = ("HLA-DRhi cDC2", "ISG+ cDC2")
COREOT_METHOD = "coreot_constant_tau"
UNIFORM_METHOD = "uniform_uot"
DEFAULT_BROAD_TAU_VALUES: tuple[float, ...] = tuple(float(v) for v in range(1, 10))
DEFAULT_BROAD_ALPHA_VALUES: tuple[float, ...] = tuple(float(v) for v in range(1, 11))
PRESENTED_ALPHA_VALUES: tuple[float, ...] = (0.0, *DEFAULT_BROAD_ALPHA_VALUES)


@dataclass(frozen=True)
class BroadGridArtifactPaths:
    output_root: Path
    detection_by_run: Path
    detection_summary: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    pareto_table: Path
    table_1: Path
    table_2: Path
    report: Path
    manifest: Path
    figures: dict[str, Path]


def _is_grid_value(value: float, allowed_values: tuple[float, ...]) -> bool:
    return any(abs(value - allowed) < 1e-9 for allowed in allowed_values)


def _filter_coreot_descriptors(
    *,
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> tuple[RunDescriptor, ...]:
    tau_map, alpha_map = _build_labelwise_param_maps(grid_dir, descriptors)
    return tuple(
        desc
        for desc in descriptors
        if desc.held_out_label in held_out_labels
        and _is_grid_value(tau_map[desc.run_id], DEFAULT_BROAD_TAU_VALUES)
        and _is_grid_value(alpha_map[desc.run_id], DEFAULT_BROAD_ALPHA_VALUES)
    )


def _filter_uniform_descriptors(
    *,
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> tuple[RunDescriptor, ...]:
    return tuple(
        desc
        for desc in descriptors
        if desc.held_out_label in held_out_labels
        and _is_grid_value(
            parse_tau_from_transport_config(grid_dir / desc.run_id),
            DEFAULT_BROAD_TAU_VALUES,
        )
    )


def build_pareto_table(
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    *,
    exclusion_filters: dict[str, float] | None = None,
) -> pd.DataFrame:
    filters = exclusion_filters or DEFAULT_EXCLUSION_FILTERS

    det_wide = detection_summary.loc[
        (detection_summary["held_out_label"] != "overall")
        & (detection_summary["method"] == COREOT_METHOD)
    ].pivot_table(
        index=["held_out_label", "method", "tau", "alpha"],
        columns="quantity",
        values="mean",
    ).reset_index()

    shared_wide = shared_summary.loc[
        (shared_summary["held_out_label"] != "overall")
        & (shared_summary["method"] == COREOT_METHOD)
    ].pivot_table(
        index=["held_out_label", "method", "tau", "alpha"],
        columns="quantity",
        values="mean",
    ).reset_index()

    merged = det_wide.merge(
        shared_wide,
        on=["held_out_label", "method", "tau", "alpha"],
        how="outer",
    )

    reasons = []
    for _, row in merged.iterrows():
        fails = []
        shfa = float(row.get("shared_false_abstention_rate", 0))
        cov = float(row.get("coverage", 1))
        pf1 = float(row.get("post_abstention_macro_f1", 1))
        if shfa > filters.get("shared_false_abstention_rate_max", 0.15):
            fails.append(f"shfa={shfa:.3f}>{filters['shared_false_abstention_rate_max']}")
        if cov < filters.get("coverage_min", 0.70):
            fails.append(f"cov={cov:.3f}<{filters['coverage_min']}")
        if pf1 < filters.get("post_abstention_macro_f1_min", 0.70):
            fails.append(f"postF1={pf1:.3f}<{filters['post_abstention_macro_f1_min']}")
        reasons.append("; ".join(fails) if fails else "")

    merged["exclusion_reason"] = reasons
    merged["passes_exclusion_filters"] = [reason == "" for reason in reasons]
    for column in (
        "auroc",
        "auprc",
        "absent_abstention_rate",
        "shared_false_abstention_rate",
        "coverage",
        "post_abstention_macro_f1",
    ):
        if column not in merged.columns:
            merged[column] = float("nan")
    return merged


def _method_display_name(method: str, tau: float | None, alpha: float | None) -> str:
    if tau is None:
        return DISPLAY_NAME_BY_METHOD.get(method, method)
    if method == UNIFORM_METHOD:
        tau_str = str(int(tau)) if tau == int(tau) else str(tau)
        return f"Uniform UOT, τ={tau_str}, α=0"
    return _labelwise_swept_display_name(method, tau, alpha or 0.0)


def _sorted_rows(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.copy()
    ordered["_tau"] = ordered["tau"].fillna(-1)
    ordered["_alpha"] = ordered["alpha"].fillna(-1)
    method_order = {
        "prior_only": 0,
        "nn": 1,
        "uniform_uot": 2,
        "coreot_constant_tau": 3,
    }
    ordered["_method_order"] = ordered["method"].map(method_order).fillna(99)
    ordered = ordered.sort_values(
        ["_method_order", "_tau", "_alpha", "held_out_label"]
    ).reset_index(drop=True)
    return ordered


def _render_combined_table(
    *,
    summary: pd.DataFrame,
    quantities: tuple[str, ...],
    headers: dict[str, str],
    title: str,
    intro_line: str,
    include_prior_only: bool,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> str:
    combined = _sorted_rows(summary)
    header = ["Method"] + [headers[quantity] for quantity in quantities]
    separator = ["---"] * len(header)
    lines = [title, "", intro_line, "", "Values are mean ± std across donor-split seeds.", ""]

    for held_out_label in (*held_out_labels, "overall"):
        section_title = (
            f"Mean across {len(held_out_labels)} held-out labels"
            if held_out_label == "overall"
            else held_out_label
        )
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(separator) + " |")
        label_rows = combined.loc[combined["held_out_label"] == held_out_label]
        unique = (
            label_rows.loc[:, ["method", "tau", "alpha"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for _, row_data in unique.iterrows():
            method = str(row_data["method"])
            if not include_prior_only and method == "prior_only":
                continue
            tau_val = _tau_or_none(row_data.get("tau", float("nan")))
            alpha_val = None
            if "alpha" in row_data.index:
                alpha_raw = row_data.get("alpha", float("nan"))
                alpha_val = None if _is_missing_tau(alpha_raw) else float(alpha_raw)
            row = [_method_display_name(method, tau_val, alpha_val)]
            for quantity in quantities:
                cell = _lookup_labelwise_cell(
                    combined,
                    method,
                    held_out_label,
                    quantity,
                    tau_val,
                    alpha_val,
                )
                row.append(_format_table_value(cell, quantity))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def render_broad_grid_table_1(
    broad_detection_summary: pd.DataFrame,
    fixed_detection: pd.DataFrame,
    *,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> str:
    combined = pd.concat([fixed_detection, broad_detection_summary], ignore_index=True)
    return _render_combined_table(
        summary=combined,
        quantities=MAIN_TABLE_1_QUANTITIES,
        headers=MAIN_TABLE_1_COLUMN_HEADERS,
        title="# Main Table 1 — broad τ/α tuning",
        intro_line=(
            "Fixed reference rows from the tuned-baseline bundle. Swept rows: "
            "`uniform_uot` at α=0 and `coreot_constant_tau` on the rectangular τ×α grid."
        ),
        include_prior_only=True,
        held_out_labels=held_out_labels,
    )


def render_broad_grid_table_2(
    broad_shared_summary: pd.DataFrame,
    fixed_shared: pd.DataFrame,
    *,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> str:
    combined = pd.concat([fixed_shared, broad_shared_summary], ignore_index=True)
    return _render_combined_table(
        summary=combined,
        quantities=MAIN_TABLE_2_QUANTITIES,
        headers=MAIN_TABLE_2_COLUMN_HEADERS,
        title="# Main Table 2 — broad τ/α tuning",
        intro_line=(
            "Fixed reference rows from the tuned-baseline bundle (prior-only excluded). "
            "Swept rows: `uniform_uot` at α=0 and `coreot_constant_tau` on the rectangular τ×α grid."
        ),
        include_prior_only=False,
        held_out_labels=held_out_labels,
    )


def _render_uniform_baseline_table(
    *,
    summary: pd.DataFrame,
    quantities: tuple[str, ...],
    headers: dict[str, str],
    title: str,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> str:
    frame = _sorted_rows(
        summary.loc[summary["method"] == UNIFORM_METHOD].copy()
    )
    header = ["Method"] + [headers[quantity] for quantity in quantities]
    separator = ["---"] * len(header)
    tau_vals = sorted(frame["tau"].dropna().unique().tolist())
    tau_str = ", ".join(str(int(v)) if v == int(v) else str(v) for v in tau_vals)
    lines = [
        title,
        "",
        f"Geometry-only baseline used as the α=0 column, with τ ∈ {{{tau_str}}}.",
        "",
    ]
    for held_out_label in (*held_out_labels, "overall"):
        section_title = (
            f"Mean across {len(held_out_labels)} held-out labels"
            if held_out_label == "overall"
            else held_out_label
        )
        lines.append(f"### {section_title}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(separator) + " |")
        label_rows = (
            frame.loc[frame["held_out_label"] == held_out_label, ["tau", "alpha"]]
            .drop_duplicates()
            .sort_values(["tau", "alpha"])
        )
        for _, row_data in label_rows.iterrows():
            tau_val = float(row_data["tau"])
            alpha_val = float(row_data["alpha"])
            row = [_method_display_name(UNIFORM_METHOD, tau_val, alpha_val)]
            for quantity in quantities:
                cell = _lookup_labelwise_cell(
                    frame,
                    UNIFORM_METHOD,
                    held_out_label,
                    quantity,
                    tau_val,
                    alpha_val,
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


def render_broad_grid_report(
    *,
    fixed_reference_root: str,
    coreot_grid_dir: str,
    uniform_grid_dir: str,
    tau_values: tuple[float, ...],
    alpha_values: tuple[float, ...],
    uniform_tau_values: tuple[float, ...],
    coreot_n_runs_completed: int,
    coreot_n_runs_expected: int,
    uniform_n_runs_completed: int,
    uniform_n_runs_expected: int,
    pareto_summary: str,
    uniform_table_1_markdown: str,
    uniform_table_2_markdown: str,
    table_1_markdown: str,
    table_2_markdown: str,
    heatmap_pdf: str | None,
    heatmap_panel_pdf: str | None,
    failed_runs: list[str] | None = None,
) -> str:
    lines = [
        "# HIHA DC Broad τ/α Tuning Report",
        "",
        "## Task Definition",
        "",
        "Fixed-τ α-sensitivity report combining `coreot_constant_tau` for α>0 with "
        "the geometry-only `uniform_uot` α=0 baseline in one τ×α grid.",
        "",
        "## Sweep Grids",
        "",
        f"- **Coreot constant τ:** τ ∈ {{{', '.join(str(int(v)) for v in tau_values)}}}, "
        f"α ∈ {{{', '.join(str(int(v)) for v in alpha_values)}}}",
        f"- **Uniform UOT baseline:** α=0, τ ∈ {{{', '.join(str(int(v)) for v in uniform_tau_values)}}}",
        "- **Held-out labels:** HLA-DRhi cDC2 and ISG+ cDC2",
        "",
        "## Fixed Reference Rows",
        "",
        "- **Fixed methods:** `prior_only`, `nn`",
        f"- **Fixed reference source:** `{fixed_reference_root}`",
        "",
        "## Run Coverage",
        "",
        f"- **Coreot constant τ completed runs:** {coreot_n_runs_completed} / {coreot_n_runs_expected}",
        f"- **Uniform UOT completed runs:** {uniform_n_runs_completed} / {uniform_n_runs_expected}",
        f"- **Coreot grid dir:** `{coreot_grid_dir}`",
        f"- **Uniform grid dir:** `{uniform_grid_dir}`",
    ]
    if failed_runs:
        lines.append(f"- **Failed runs:** {', '.join(failed_runs)}")
    lines.extend(
        [
            "",
            "## Pareto Summary",
            "",
            pareto_summary or "_See pareto CSV for details._",
            "",
            "## Heatmaps",
            "",
            "The τ×α heatmap artifacts combine `uniform_uot` at α=0 with "
            "`coreot_constant_tau` at α>0.",
        ]
    )
    if heatmap_pdf is not None:
        lines.append(f"- **Heatmap PDF:** `{heatmap_pdf}`")
    if heatmap_panel_pdf is not None:
        lines.append(f"- **Full-panel heatmap PDF:** `{heatmap_panel_pdf}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "_Fill in after inspecting results. Describe detection gains, abstention costs, and "
            "shared-label-transfer preservation. Do not claim optimality._",
            "",
        "## Uniform UOT α=0 Rows",
            "",
            _table_body(uniform_table_1_markdown),
            "",
            _table_body(uniform_table_2_markdown),
            "",
            "## Main Table 1",
            "",
            _table_body(table_1_markdown),
            "",
            "## Main Table 2",
            "",
            _table_body(table_2_markdown),
            "",
        ]
    )
    return "\n".join(lines)


def _build_coreot_summaries(
    *,
    runs_root: Path,
    grid_dir: Path,
    candidate_set: str,
    condition: str,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    descriptors = discover_run_descriptors(grid_dir)
    descriptors = _filter_coreot_descriptors(
        grid_dir=grid_dir, descriptors=descriptors, held_out_labels=held_out_labels
    )
    if not descriptors:
        raise ResultsGridError(f"No run descriptors found in {grid_dir}")
    completed, failures = _partition_completed_sweep_runs(
        runs_root=runs_root,
        descriptors=descriptors,
        candidate_set=candidate_set,
        condition=condition,
    )
    failed_run_ids = _failed_run_ids(failures)
    if not completed:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            failed_run_ids,
        )
    tau_map, alpha_map = _build_labelwise_param_maps(grid_dir, completed)
    detection_by_run = build_detection_by_run(
        runs_root=runs_root,
        descriptors=completed,
        candidate_set=candidate_set,
        condition=condition,
        methods=(COREOT_METHOD,),
    )
    shared_by_run = build_shared_label_transfer_by_run(
        runs_root=runs_root,
        descriptors=completed,
        candidate_set=candidate_set,
        condition=condition,
        methods=(COREOT_METHOD,),
    )
    detection_by_run["tau"] = detection_by_run["run_id"].map(tau_map)
    detection_by_run["alpha"] = detection_by_run["run_id"].map(alpha_map)
    shared_by_run["tau"] = shared_by_run["run_id"].map(tau_map)
    shared_by_run["alpha"] = shared_by_run["run_id"].map(alpha_map)
    detection_summary = summarize_run_metrics(
        detection_by_run,
        group_columns=("held_out_label", "method", "method_group", "primary_score", "tau", "alpha"),
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
        group_columns=("held_out_label", "method", "method_group", "tau", "alpha"),
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

    return (
        detection_by_run,
        detection_summary,
        shared_by_run,
        shared_summary,
        failed_run_ids,
    )


def _build_uniform_summaries(
    *,
    runs_root: Path,
    grid_dir: Path,
    candidate_set: str,
    condition: str,
    held_out_labels: tuple[str, ...] = HELD_OUT_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    descriptors = discover_run_descriptors(grid_dir)
    descriptors = _filter_uniform_descriptors(
        grid_dir=grid_dir, descriptors=descriptors, held_out_labels=held_out_labels
    )
    if not descriptors:
        raise ResultsGridError(f"No run descriptors found in {grid_dir}")
    detection_by_run = build_tau_detection_by_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        descriptors=descriptors,
        candidate_set=candidate_set,
        condition=condition,
        methods=(UNIFORM_METHOD,),
    )
    shared_by_run = build_tau_shared_label_transfer_by_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        descriptors=descriptors,
        candidate_set=candidate_set,
        condition=condition,
        methods=(UNIFORM_METHOD,),
    )
    detection_by_run["alpha"] = 0.0
    shared_by_run["alpha"] = 0.0

    detection_summary = summarize_run_metrics(
        detection_by_run,
        group_columns=("held_out_label", "method", "method_group", "primary_score", "tau", "alpha"),
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
        group_columns=("held_out_label", "method", "method_group", "tau", "alpha"),
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

    completed, failures = _partition_completed_sweep_runs(
        runs_root=runs_root,
        descriptors=descriptors,
        candidate_set=candidate_set,
        condition=condition,
    )
    return (
        detection_by_run,
        detection_summary,
        shared_by_run,
        shared_summary,
        _failed_run_ids(failures),
    )


def write_broad_grid_results(
    *,
    runs_root: str | Path = Path("runs"),
    coreot_grid_dir: str | Path = Path(
        "experiments/missing_celltype/generated_configs/hiha_dc_coreot_constant_tau_alpha_broad_grid/coreot_constant_tau"
    ),
    uniform_grid_dir: str | Path = Path(
        "experiments/missing_celltype/generated_configs/hiha_dc_coreot_constant_tau_alpha_broad_grid/uniform_uot_tau"
    ),
    output_root: str | Path = Path("results/HIHA_DC/sensitivity/constant_tau_alpha"),
    main_grid_root: str | Path = DEFAULT_FIXED_REFERENCE_ROOT,
    candidate_set: str = "hiha_harmony30_k100",
    condition: str = "incomplete_reference",
    write_figures: bool = False,
    exclusion_filters: dict[str, float] | None = None,
) -> BroadGridArtifactPaths:
    import sys as _sys

    runs_root = Path(runs_root)
    coreot_grid_dir = Path(coreot_grid_dir)
    uniform_grid_dir = Path(uniform_grid_dir)
    output_root = Path(output_root)
    main_grid_root = Path(main_grid_root)

    tables_root = output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    coreot_descriptors = _filter_coreot_descriptors(
        grid_dir=coreot_grid_dir,
        descriptors=discover_run_descriptors(coreot_grid_dir),
    )
    uniform_descriptors = _filter_uniform_descriptors(
        grid_dir=uniform_grid_dir,
        descriptors=discover_run_descriptors(uniform_grid_dir),
    )

    (
        coreot_detection_by_run,
        coreot_detection_summary,
        coreot_shared_by_run,
        coreot_shared_summary,
        coreot_failed_runs,
    ) = _build_coreot_summaries(
        runs_root=runs_root,
        grid_dir=coreot_grid_dir,
        candidate_set=candidate_set,
        condition=condition,
    )
    (
        uniform_detection_by_run,
        uniform_detection_summary,
        uniform_shared_by_run,
        uniform_shared_summary,
        uniform_failed_runs,
    ) = _build_uniform_summaries(
        runs_root=runs_root,
        grid_dir=uniform_grid_dir,
        candidate_set=candidate_set,
        condition=condition,
    )

    fixed_detection, fixed_shared = read_fixed_reference_bundle(main_grid_root, HELD_OUT_LABELS)
    fixed_detection["alpha"] = float("nan")
    fixed_shared["alpha"] = float("nan")

    combined_detection_by_run = pd.concat(
        [uniform_detection_by_run, coreot_detection_by_run], ignore_index=True
    )
    combined_shared_by_run = pd.concat(
        [uniform_shared_by_run, coreot_shared_by_run], ignore_index=True
    )
    combined_detection_summary = pd.concat(
        [uniform_detection_summary, coreot_detection_summary], ignore_index=True
    )
    combined_shared_summary = pd.concat(
        [uniform_shared_summary, coreot_shared_summary], ignore_index=True
    )

    pareto = build_pareto_table(
        coreot_detection_summary,
        coreot_shared_summary,
        exclusion_filters=exclusion_filters,
    )
    coreot_tau_values = tuple(
        sorted(
            float(value)
            for value in coreot_detection_summary["tau"].dropna().unique().tolist()
        )
    )
    coreot_alpha_values = tuple(
        sorted(
            float(value)
            for value in coreot_detection_summary["alpha"].dropna().unique().tolist()
        )
    )
    uniform_tau_values = tuple(
        sorted(
            float(value)
            for value in uniform_detection_summary["tau"].dropna().unique().tolist()
        )
    )

    detection_by_run_path = tables_root / "detection_by_run.csv"
    detection_summary_path = tables_root / "detection_summary.csv"
    shared_by_run_path = tables_root / "shared_label_transfer_by_run.csv"
    shared_summary_path = tables_root / "shared_label_transfer_summary.csv"
    pareto_path = tables_root / "pareto.csv"

    combined_detection_by_run.to_csv(detection_by_run_path, index=False)
    combined_detection_summary.to_csv(detection_summary_path, index=False)
    combined_shared_by_run.to_csv(shared_by_run_path, index=False)
    combined_shared_summary.to_csv(shared_summary_path, index=False)
    pareto.to_csv(pareto_path, index=False)

    table_1_markdown = render_broad_grid_table_1(combined_detection_summary, fixed_detection)
    table_2_markdown = render_broad_grid_table_2(combined_shared_summary, fixed_shared)
    uniform_table_1_markdown = _render_uniform_baseline_table(
        summary=uniform_detection_summary,
        quantities=MAIN_TABLE_1_QUANTITIES,
        headers=MAIN_TABLE_1_COLUMN_HEADERS,
        title="# Uniform UOT baseline — Main Table 1",
    )
    uniform_table_2_markdown = _render_uniform_baseline_table(
        summary=uniform_shared_summary,
        quantities=MAIN_TABLE_2_QUANTITIES,
        headers=MAIN_TABLE_2_COLUMN_HEADERS,
        title="# Uniform UOT baseline — Main Table 2",
    )

    table_1_path = tables_root / "table_1_broad_grid.md"
    table_2_path = tables_root / "table_2_broad_grid.md"
    table_1_path.write_text(table_1_markdown, encoding="utf-8")
    table_2_path.write_text(table_2_markdown, encoding="utf-8")

    figure_paths: dict[str, Path] = {}
    heatmap_pdf_path: str | None = None
    heatmap_panel_pdf_path: str | None = None
    if write_figures:
        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            from coreot.results.figures import (
                write_broad_grid_alpha_sensitivity_figure,
                write_broad_grid_heatmaps,
            )

            figure_paths = write_broad_grid_heatmaps(
                detection_summary=combined_detection_summary,
                shared_summary=combined_shared_summary,
                output_dir=figures_dir,
            )
            figure_paths.update(
                write_broad_grid_alpha_sensitivity_figure(
                    detection_summary=combined_detection_summary,
                    shared_summary=combined_shared_summary,
                    output_dir=figures_dir,
                )
            )
            heatmap_pdf = figure_paths.get("heatmaps_pdf")
            heatmap_pdf_path = str(heatmap_pdf) if heatmap_pdf is not None else None
            heatmap_panel_pdf = figure_paths.get("heatmaps_panel_pdf")
            heatmap_panel_pdf_path = (
                str(heatmap_panel_pdf) if heatmap_panel_pdf is not None else None
            )
        except ImportError:
            _sys.stderr.write("Skipping figures: not available\n")

    report_path = output_root / "report.md"
    n_pareto_passing = int(pareto["passes_exclusion_filters"].sum()) if not pareto.empty else 0
    report_path.write_text(
        render_broad_grid_report(
            fixed_reference_root=str(main_grid_root),
            coreot_grid_dir=str(coreot_grid_dir),
            uniform_grid_dir=str(uniform_grid_dir),
            tau_values=coreot_tau_values,
            alpha_values=coreot_alpha_values,
            uniform_tau_values=uniform_tau_values,
            coreot_n_runs_completed=len(coreot_detection_by_run["run_id"].unique()),
            coreot_n_runs_expected=len(coreot_descriptors),
            uniform_n_runs_completed=len(uniform_detection_by_run["run_id"].unique()),
            uniform_n_runs_expected=len(uniform_descriptors),
            pareto_summary=f"{n_pareto_passing} / {len(pareto)} coreot settings pass exclusion filters.",
            uniform_table_1_markdown=uniform_table_1_markdown,
            uniform_table_2_markdown=uniform_table_2_markdown,
            table_1_markdown=table_1_markdown,
            table_2_markdown=table_2_markdown,
            heatmap_pdf=heatmap_pdf_path,
            heatmap_panel_pdf=heatmap_panel_pdf_path,
            failed_runs=sorted(set(coreot_failed_runs + uniform_failed_runs)) or None,
        ),
        encoding="utf-8",
    )

    manifest_path = output_root / "manifest.yaml"
    artifacts: dict[str, str] = {
        "detection_by_run": str(detection_by_run_path),
        "detection_summary": str(detection_summary_path),
        "shared_label_transfer_by_run": str(shared_by_run_path),
        "shared_label_transfer_summary": str(shared_summary_path),
        "pareto": str(pareto_path),
        "table_1_broad_grid": str(table_1_path),
        "table_2_broad_grid": str(table_2_path),
        "report": str(report_path),
    }
    for name, path in figure_paths.items():
        artifacts[f"figure_{name}"] = str(path)

    write_manifest(
        manifest_path,
        Manifest(
            stage="broad-grid-full",
            artifacts=artifacts,
            metadata={
                "runs_root": str(runs_root),
                "coreot_grid_dir": str(coreot_grid_dir),
                "uniform_grid_dir": str(uniform_grid_dir),
                "fixed_reference_root": str(main_grid_root),
                "candidate_set": candidate_set,
                "condition": condition,
                "coreot_tau_values": [float(v) for v in coreot_tau_values],
                "coreot_alpha_values": [float(v) for v in coreot_alpha_values],
                "presented_alpha_values": [float(v) for v in PRESENTED_ALPHA_VALUES],
                "uniform_tau_values": [float(v) for v in uniform_tau_values],
                "fixed_reference_methods": list(FIXED_REFERENCE_METHODS),
                "n_coreot_runs_completed": len(coreot_detection_by_run["run_id"].unique()),
                "n_coreot_runs_expected": len(coreot_descriptors),
                "n_uniform_runs_completed": len(uniform_detection_by_run["run_id"].unique()),
                "n_uniform_runs_expected": len(uniform_descriptors),
                "n_pareto_passing": n_pareto_passing,
                "exclusion_filters": exclusion_filters or DEFAULT_EXCLUSION_FILTERS,
                "failed_runs": sorted(set(coreot_failed_runs + uniform_failed_runs)),
            },
        ),
    )

    return BroadGridArtifactPaths(
        output_root=output_root,
        detection_by_run=detection_by_run_path,
        detection_summary=detection_summary_path,
        shared_label_transfer_by_run=shared_by_run_path,
        shared_label_transfer_summary=shared_summary_path,
        pareto_table=pareto_path,
        table_1=table_1_path,
        table_2=table_2_path,
        report=report_path,
        manifest=manifest_path,
        figures=figure_paths,
    )


# ---------------------------------------------------------------------------
# PBMC broad grid
# ---------------------------------------------------------------------------

PBMC_HELD_OUT_LABELS: tuple[str, ...] = (
    "CD14+ Monocytes",
    "CD4 T cells",
    "B cells",
    "NK cells",
    "Dendritic cells",
    "CD8 T cells",
    "FCGR3A+ Monocytes",
)
PBMC_CANDIDATE_SET = "pca30_k100"
PBMC_FIXED_REFERENCE_ROOT = Path("results/PBMC/main")


@dataclass(frozen=True)
class PbmcBroadGridArtifactPaths:
    output_root: Path
    detection_by_run: Path
    detection_summary: Path
    shared_label_transfer_by_run: Path
    shared_label_transfer_summary: Path
    pareto_table: Path
    table_1: Path
    table_2: Path
    report: Path
    manifest: Path
    figures: dict[str, Path]


def render_pbmc_broad_grid_report(
    *,
    fixed_reference_root: str,
    coreot_grid_dir: str,
    uniform_grid_dir: str,
    tau_values: tuple[float, ...],
    alpha_values: tuple[float, ...],
    uniform_tau_values: tuple[float, ...],
    coreot_n_runs_completed: int,
    coreot_n_runs_expected: int,
    uniform_n_runs_completed: int,
    uniform_n_runs_expected: int,
    pareto_summary: str,
    uniform_table_1_markdown: str,
    uniform_table_2_markdown: str,
    table_1_markdown: str,
    table_2_markdown: str,
    heatmap_pdf: str | None,
    heatmap_panel_pdf: str | None,
    failed_runs: list[str] | None = None,
) -> str:
    held_out_str = ", ".join(PBMC_HELD_OUT_LABELS)
    lines = [
        "# PBMC Broad τ/α Tuning Report",
        "",
        "## Task Definition",
        "",
        "Fixed-τ α-sensitivity report for the PBMC stimulated-state experiment, "
        "combining `coreot_constant_tau` for α>0 with the geometry-only `uniform_uot` "
        "α=0 baseline in one τ×α grid.",
        "",
        "## Sweep Grids",
        "",
        f"- **Coreot constant τ:** τ ∈ {{{', '.join(str(int(v)) for v in tau_values)}}}, "
        f"α ∈ {{{', '.join(str(int(v)) for v in alpha_values)}}}",
        f"- **Uniform UOT baseline:** α=0, τ ∈ {{{', '.join(str(int(v)) for v in uniform_tau_values)}}}",
        f"- **Held-out cell types:** {held_out_str}",
        "",
        "## Fixed Reference Rows",
        "",
        "- **Fixed methods:** `prior_only`, `nn`",
        f"- **Fixed reference source:** `{fixed_reference_root}`",
        "",
        "## Run Coverage",
        "",
        f"- **Coreot constant τ completed runs:** {coreot_n_runs_completed} / {coreot_n_runs_expected}",
        f"- **Uniform UOT completed runs:** {uniform_n_runs_completed} / {uniform_n_runs_expected}",
        f"- **Coreot grid dir:** `{coreot_grid_dir}`",
        f"- **Uniform grid dir:** `{uniform_grid_dir}`",
    ]
    if failed_runs:
        lines.append(f"- **Failed runs:** {', '.join(failed_runs)}")
    lines.extend(
        [
            "",
            "## Pareto Summary",
            "",
            pareto_summary or "_See pareto CSV for details._",
            "",
            "## Heatmaps",
            "",
            "The τ×α heatmap artifacts combine `uniform_uot` at α=0 with "
            "`coreot_constant_tau` at α>0.",
        ]
    )
    if heatmap_pdf is not None:
        lines.append(f"- **Heatmap PDF:** `{heatmap_pdf}`")
    if heatmap_panel_pdf is not None:
        lines.append(f"- **Full-panel heatmap PDF:** `{heatmap_panel_pdf}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "_Fill in after inspecting results. Describe detection gains, abstention costs, and "
            "shared-label-transfer preservation. Do not claim optimality._",
            "",
            "## Uniform UOT α=0 Rows",
            "",
            _table_body(uniform_table_1_markdown),
            "",
            _table_body(uniform_table_2_markdown),
            "",
            "## Main Table 1",
            "",
            _table_body(table_1_markdown),
            "",
            "## Main Table 2",
            "",
            _table_body(table_2_markdown),
            "",
        ]
    )
    return "\n".join(lines)


def write_pbmc_broad_grid_results(
    *,
    runs_root: str | Path = Path("runs"),
    coreot_grid_dir: str | Path = Path(
        "experiments/pbmc_state/generated_configs/pbmc_coreot_constant_tau_alpha_broad_grid/coreot_constant_tau"
    ),
    uniform_grid_dir: str | Path = Path(
        "experiments/pbmc_state/generated_configs/pbmc_coreot_constant_tau_alpha_broad_grid/uniform_uot_tau"
    ),
    output_root: str | Path = Path("results/PBMC/sensitivity/constant_tau_alpha"),
    fixed_reference_root: str | Path = PBMC_FIXED_REFERENCE_ROOT,
    candidate_set: str = PBMC_CANDIDATE_SET,
    condition: str = "incomplete_reference",
    write_figures: bool = False,
    exclusion_filters: dict[str, float] | None = None,
) -> PbmcBroadGridArtifactPaths:
    import sys as _sys

    runs_root = Path(runs_root)
    coreot_grid_dir = Path(coreot_grid_dir)
    uniform_grid_dir = Path(uniform_grid_dir)
    output_root = Path(output_root)
    fixed_reference_root = Path(fixed_reference_root)

    tables_root = output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    coreot_descriptors = _filter_coreot_descriptors(
        grid_dir=coreot_grid_dir,
        descriptors=discover_run_descriptors(coreot_grid_dir),
        held_out_labels=PBMC_HELD_OUT_LABELS,
    )
    uniform_descriptors = _filter_uniform_descriptors(
        grid_dir=uniform_grid_dir,
        descriptors=discover_run_descriptors(uniform_grid_dir),
        held_out_labels=PBMC_HELD_OUT_LABELS,
    )

    (
        coreot_detection_by_run,
        coreot_detection_summary,
        coreot_shared_by_run,
        coreot_shared_summary,
        coreot_failed_runs,
    ) = _build_coreot_summaries(
        runs_root=runs_root,
        grid_dir=coreot_grid_dir,
        candidate_set=candidate_set,
        condition=condition,
        held_out_labels=PBMC_HELD_OUT_LABELS,
    )
    (
        uniform_detection_by_run,
        uniform_detection_summary,
        uniform_shared_by_run,
        uniform_shared_summary,
        uniform_failed_runs,
    ) = _build_uniform_summaries(
        runs_root=runs_root,
        grid_dir=uniform_grid_dir,
        candidate_set=candidate_set,
        condition=condition,
        held_out_labels=PBMC_HELD_OUT_LABELS,
    )

    fixed_detection, fixed_shared = read_fixed_reference_rows(
        detection_summary_path=fixed_reference_root / "tables" / "global_detection_summary.csv",
        shared_summary_path=fixed_reference_root / "tables" / "shared_label_transfer_summary.csv",
        held_out_labels=PBMC_HELD_OUT_LABELS,
    )
    fixed_detection["alpha"] = float("nan")
    fixed_shared["alpha"] = float("nan")

    combined_detection_by_run = pd.concat(
        [uniform_detection_by_run, coreot_detection_by_run], ignore_index=True
    )
    combined_shared_by_run = pd.concat(
        [uniform_shared_by_run, coreot_shared_by_run], ignore_index=True
    )
    combined_detection_summary = pd.concat(
        [uniform_detection_summary, coreot_detection_summary], ignore_index=True
    )
    combined_shared_summary = pd.concat(
        [uniform_shared_summary, coreot_shared_summary], ignore_index=True
    )

    pareto = build_pareto_table(
        coreot_detection_summary,
        coreot_shared_summary,
        exclusion_filters=exclusion_filters,
    )
    coreot_tau_values = tuple(
        sorted(
            float(value)
            for value in coreot_detection_summary["tau"].dropna().unique().tolist()
        )
    )
    coreot_alpha_values = tuple(
        sorted(
            float(value)
            for value in coreot_detection_summary["alpha"].dropna().unique().tolist()
        )
    )
    uniform_tau_values = tuple(
        sorted(
            float(value)
            for value in uniform_detection_summary["tau"].dropna().unique().tolist()
        )
    )

    detection_by_run_path = tables_root / "detection_by_run.csv"
    detection_summary_path = tables_root / "detection_summary.csv"
    shared_by_run_path = tables_root / "shared_label_transfer_by_run.csv"
    shared_summary_path = tables_root / "shared_label_transfer_summary.csv"
    pareto_path = tables_root / "pareto.csv"

    combined_detection_by_run.to_csv(detection_by_run_path, index=False)
    combined_detection_summary.to_csv(detection_summary_path, index=False)
    combined_shared_by_run.to_csv(shared_by_run_path, index=False)
    combined_shared_summary.to_csv(shared_summary_path, index=False)
    pareto.to_csv(pareto_path, index=False)

    table_1_markdown = render_broad_grid_table_1(
        combined_detection_summary, fixed_detection, held_out_labels=PBMC_HELD_OUT_LABELS
    )
    table_2_markdown = render_broad_grid_table_2(
        combined_shared_summary, fixed_shared, held_out_labels=PBMC_HELD_OUT_LABELS
    )
    uniform_table_1_markdown = _render_uniform_baseline_table(
        summary=uniform_detection_summary,
        quantities=MAIN_TABLE_1_QUANTITIES,
        headers=MAIN_TABLE_1_COLUMN_HEADERS,
        title="# Uniform UOT baseline — Main Table 1",
        held_out_labels=PBMC_HELD_OUT_LABELS,
    )
    uniform_table_2_markdown = _render_uniform_baseline_table(
        summary=uniform_shared_summary,
        quantities=MAIN_TABLE_2_QUANTITIES,
        headers=MAIN_TABLE_2_COLUMN_HEADERS,
        title="# Uniform UOT baseline — Main Table 2",
        held_out_labels=PBMC_HELD_OUT_LABELS,
    )

    table_1_path = tables_root / "table_1_broad_grid.md"
    table_2_path = tables_root / "table_2_broad_grid.md"
    table_1_path.write_text(table_1_markdown, encoding="utf-8")
    table_2_path.write_text(table_2_markdown, encoding="utf-8")

    figure_paths: dict[str, Path] = {}
    heatmap_pdf_path: str | None = None
    heatmap_panel_pdf_path: str | None = None
    if write_figures:
        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            from coreot.results.figures import write_broad_grid_heatmaps

            figure_paths = write_broad_grid_heatmaps(
                detection_summary=combined_detection_summary,
                shared_summary=combined_shared_summary,
                output_dir=figures_dir,
                held_out_labels=PBMC_HELD_OUT_LABELS,
            )
            heatmap_pdf = figure_paths.get("heatmaps_pdf")
            heatmap_pdf_path = str(heatmap_pdf) if heatmap_pdf is not None else None
            heatmap_panel_pdf = figure_paths.get("heatmaps_panel_pdf")
            heatmap_panel_pdf_path = (
                str(heatmap_panel_pdf) if heatmap_panel_pdf is not None else None
            )
        except ImportError:
            _sys.stderr.write("Skipping figures: not available\n")

    report_path = output_root / "report.md"
    n_pareto_passing = int(pareto["passes_exclusion_filters"].sum()) if not pareto.empty else 0
    report_path.write_text(
        render_pbmc_broad_grid_report(
            fixed_reference_root=str(fixed_reference_root),
            coreot_grid_dir=str(coreot_grid_dir),
            uniform_grid_dir=str(uniform_grid_dir),
            tau_values=coreot_tau_values,
            alpha_values=coreot_alpha_values,
            uniform_tau_values=uniform_tau_values,
            coreot_n_runs_completed=len(coreot_detection_by_run["run_id"].unique()),
            coreot_n_runs_expected=len(coreot_descriptors),
            uniform_n_runs_completed=len(uniform_detection_by_run["run_id"].unique()),
            uniform_n_runs_expected=len(uniform_descriptors),
            pareto_summary=f"{n_pareto_passing} / {len(pareto)} coreot settings pass exclusion filters.",
            uniform_table_1_markdown=uniform_table_1_markdown,
            uniform_table_2_markdown=uniform_table_2_markdown,
            table_1_markdown=table_1_markdown,
            table_2_markdown=table_2_markdown,
            heatmap_pdf=heatmap_pdf_path,
            heatmap_panel_pdf=heatmap_panel_pdf_path,
            failed_runs=sorted(set(coreot_failed_runs + uniform_failed_runs)) or None,
        ),
        encoding="utf-8",
    )

    manifest_path = output_root / "manifest.yaml"
    artifacts: dict[str, str] = {
        "detection_by_run": str(detection_by_run_path),
        "detection_summary": str(detection_summary_path),
        "shared_label_transfer_by_run": str(shared_by_run_path),
        "shared_label_transfer_summary": str(shared_summary_path),
        "pareto": str(pareto_path),
        "table_1_broad_grid": str(table_1_path),
        "table_2_broad_grid": str(table_2_path),
        "report": str(report_path),
    }
    for name, path in figure_paths.items():
        artifacts[f"figure_{name}"] = str(path)

    write_manifest(
        manifest_path,
        Manifest(
            stage="broad-grid-full",
            artifacts=artifacts,
            metadata={
                "experiment": "pbmc_stimulated_state",
                "runs_root": str(runs_root),
                "coreot_grid_dir": str(coreot_grid_dir),
                "uniform_grid_dir": str(uniform_grid_dir),
                "fixed_reference_root": str(fixed_reference_root),
                "candidate_set": candidate_set,
                "condition": condition,
                "coreot_tau_values": [float(v) for v in coreot_tau_values],
                "coreot_alpha_values": [float(v) for v in coreot_alpha_values],
                "presented_alpha_values": [float(v) for v in PRESENTED_ALPHA_VALUES],
                "uniform_tau_values": [float(v) for v in uniform_tau_values],
                "fixed_reference_methods": list(FIXED_REFERENCE_METHODS),
                "n_coreot_runs_completed": len(coreot_detection_by_run["run_id"].unique()),
                "n_coreot_runs_expected": len(coreot_descriptors),
                "n_uniform_runs_completed": len(uniform_detection_by_run["run_id"].unique()),
                "n_uniform_runs_expected": len(uniform_descriptors),
                "n_pareto_passing": n_pareto_passing,
                "exclusion_filters": exclusion_filters or DEFAULT_EXCLUSION_FILTERS,
                "failed_runs": sorted(set(coreot_failed_runs + uniform_failed_runs)),
                "held_out_labels": list(PBMC_HELD_OUT_LABELS),
            },
        ),
    )

    return PbmcBroadGridArtifactPaths(
        output_root=output_root,
        detection_by_run=detection_by_run_path,
        detection_summary=detection_summary_path,
        shared_label_transfer_by_run=shared_by_run_path,
        shared_label_transfer_summary=shared_summary_path,
        pareto_table=pareto_path,
        table_1=table_1_path,
        table_2=table_2_path,
        report=report_path,
        manifest=manifest_path,
        figures=figure_paths,
    )
