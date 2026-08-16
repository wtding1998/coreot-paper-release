from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from coreot.results.grid_common import discover_run_descriptors


LABELS = ("HLA-DRhi cDC2", "ISG+ cDC2")
PERCENTILES = (0.90, 0.95, 0.975)
POLICIES = (
    ("coreot_full", "u", r"CoRe-OT ($u$)"),
    ("uniform_uot", "u", r"Uniform UOT ($u$)"),
)
PRIMARY_POLICIES = (
    ("coreot_full", "u", "CoRe-OT"),
    ("uniform_uot", "u", "Uniform UOT"),
    ("seurat_anchor", "u", "Seurat"),
    ("singleR", "u", "SingleR"),
    ("celltypist_l3", "u", "CellTypist"),
    ("scmap_cell", "u", "scmap-cell"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
METRICS = (
    "full_reference_false_abstention_rate",
    "absent_abstention_rate",
    "shared_false_abstention_rate",
    "coverage",
    "post_abstention_macro_f1",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate HIHA DC abstention-threshold sensitivity artifacts."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/"
            "report_leave_one_HIHA_DC"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/HIHA_DC/diagnostics/threshold_sensitivity"),
    )
    parser.add_argument(
        "--uniform-grid-dir",
        type=Path,
        default=Path(
            "experiments/missing_celltype/generated_configs/"
            "hiha_dc_uniform_uot_tau05"
        ),
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=Path("results/HIHA_DC/compare_baselines/tables"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("results/HIHA_DC/figures"),
    )
    parser.add_argument(
        "--manuscript-table-root",
        type=Path,
        default=Path("results/HIHA_DC/manuscript/tables"),
    )
    return parser


def compute_run_threshold_sensitivity(
    *,
    full_reference_scores: pd.DataFrame,
    incomplete_scores: pd.DataFrame,
    truth: pd.DataFrame,
    run_id: str,
    held_out_label: str,
    seed: int,
    entropy_threshold: float,
    percentiles: tuple[float, ...] = PERCENTILES,
    policies: tuple[tuple[str, str, str], ...] = POLICIES,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, score, display_name in policies:
        reference = full_reference_scores.loc[
            full_reference_scores["method"].astype(str).eq(method)
        ].copy()
        incomplete = incomplete_scores.loc[
            incomplete_scores["method"].astype(str).eq(method)
        ].merge(truth, on="cell_id", how="left", validate="one_to_one")
        if reference.empty or incomplete.empty:
            raise ValueError(f"Missing score rows for run={run_id}, method={method}.")
        if incomplete["true_label"].isna().any():
            raise ValueError(f"Missing query truth after joining run={run_id}, method={method}.")
        _validate_numeric(reference, (score, "label_entropy"), run_id, method)
        _validate_numeric(incomplete, (score, "label_entropy"), run_id, method)

        reference_score = pd.to_numeric(reference[score], errors="raise")
        reference_entropy = pd.to_numeric(
            reference["label_entropy"], errors="raise"
        )
        incomplete_score = pd.to_numeric(incomplete[score], errors="raise")
        incomplete_entropy = pd.to_numeric(
            incomplete["label_entropy"], errors="raise"
        )
        absent = incomplete["is_absent_state"].astype(bool)
        shared = incomplete["is_shared_state"].astype(bool)
        labels = sorted(incomplete.loc[shared, "true_label"].astype(str).unique())

        for percentile in percentiles:
            threshold = float(reference_score.quantile(percentile))
            reference_abstention = (reference_score > threshold) | (
                reference_entropy > entropy_threshold
            )
            abstention = (incomplete_score > threshold) | (
                incomplete_entropy > entropy_threshold
            )
            retained_shared = shared & ~abstention
            post = incomplete.loc[retained_shared]
            post = post.loc[post["forced_label"].astype(str).str.len() > 0]
            rows.append(
                {
                    "run_id": run_id,
                    "held_out_label": held_out_label,
                    "seed": seed,
                    "method": method,
                    "score": score,
                    "display_name": display_name,
                    "threshold_percentile": percentile,
                    "threshold": threshold,
                    "entropy_threshold": entropy_threshold,
                    "n_absent": int(absent.sum()),
                    "n_shared": int(shared.sum()),
                    "full_reference_false_abstention_rate": float(
                        reference_abstention.mean()
                    ),
                    "absent_abstention_rate": float(abstention.loc[absent].mean()),
                    "shared_false_abstention_rate": float(
                        abstention.loc[shared].mean()
                    ),
                    "coverage": float((~abstention.loc[shared]).mean()),
                    "post_abstention_macro_f1": _macro_f1(
                        post["true_label"], post["forced_label"], labels
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_threshold_sensitivity(by_seed: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "held_out_label",
        "method",
        "score",
        "display_name",
        "threshold_percentile",
    ]
    summary = by_seed.groupby(group_columns, sort=False)[list(METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    counts = (
        by_seed.groupby(group_columns, sort=False)["seed"]
        .nunique()
        .rename("n_runs")
        .reset_index()
    )
    return summary.merge(counts, on=group_columns, validate="one_to_one")


def render_primary_policy_table(
    detection_summary: pd.DataFrame,
    transfer_summary: pd.DataFrame,
) -> str:
    headers = (
        "Endpoint",
        "Method",
        "Held-out-state abstention rate",
        "Represented-state coverage",
        "Post-abstention macro-F1",
    )
    lines = [
        "# Supplementary Table S8: calibrated held-out-state abstention and "
        "represented-state label transfer",
        "",
        "Values are arithmetic means \\(\\pm\\) sample standard deviations across "
        "five donor splits.",
        "",
        "The primary 95th-percentile operational policy is reported. Methods "
        "without an applicable operational abstention policy are omitted.",
        "Held-out-state abstention and represented-state coverage are evaluated "
        "on disjoint cell sets and are therefore not complements; post-abstention "
        "macro-F1 is conditional on retained represented-state cells.",
        "",
        "| " + " | ".join(headers) + " |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        for method, score, display_name in PRIMARY_POLICIES:
            lines.append(
                "| "
                + " | ".join(
                    (
                        label,
                        display_name,
                        _comparison_summary(
                            detection_summary,
                            held_out_label=label,
                            method=method,
                            score=score,
                            quantity="absent_abstention_rate",
                        ),
                        _comparison_summary(
                            transfer_summary,
                            held_out_label=label,
                            method=method,
                            score=score,
                            quantity="coverage",
                        ),
                        _comparison_summary(
                            transfer_summary,
                            held_out_label=label,
                            method=method,
                            score=score,
                            quantity="post_abstention_macro_f1",
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "_Sources: `results/HIHA_DC/compare_baselines/tables/"
            "compare_detection_summary.csv` and `results/HIHA_DC/compare_baselines/"
            "tables/compare_shared_label_transfer_summary.csv`._",
            "",
        ]
    )
    return "\n".join(lines)


def render_threshold_sensitivity_table(summary: pd.DataFrame) -> str:
    headers = (
        "Endpoint",
        "Method",
        "Calibration percentile",
        "Held-out-state abstention rate",
        "Represented-state coverage",
        "Post-abstention macro-F1",
    )
    lines = [
        "# Supplementary Table S9: calibration-percentile sensitivity",
        "",
        "Values are arithmetic means \\(\\pm\\) sample standard deviations across "
        "five donor splits.",
        "For each donor split and method, the query-marginal-deficit cutoff was "
        "estimated at the indicated percentile from all finite scores in the "
        "matched full-reference control and applied to the corresponding "
        "incomplete-reference scores. The fitted transports were unchanged, and "
        "the normalized label-entropy cutoff remained \\(0.8\\).",
        "",
        "| " + " | ".join(headers) + " |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        for method, score, display_name in POLICIES:
            for percentile in PERCENTILES:
                rows = summary.loc[
                    summary["held_out_label"].eq(label)
                    & summary["method"].eq(method)
                    & summary["score"].eq(score)
                    & np.isclose(summary["threshold_percentile"], percentile)
                ]
                if len(rows) != 1:
                    raise ValueError(
                        f"Expected one summary row for {label}, {display_name}, {percentile}."
                    )
                row = rows.iloc[0]
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            label,
                            display_name.removesuffix(r" ($u$)"),
                            f"{100 * percentile:g}%",
                            _format_summary(row, "absent_abstention_rate"),
                            _format_summary(row, "coverage"),
                            _format_summary(row, "post_abstention_macro_f1"),
                        )
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "_Source: `results/HIHA_DC/diagnostics/threshold_sensitivity/tables/"
            "threshold_sensitivity_summary.csv`._",
            "",
        ]
    )
    return "\n".join(lines)


def write_threshold_sensitivity_figure(
    summary: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "supplementary_figure_s4_threshold_sensitivity.png"
    pdf_path = output_dir / "supplementary_figure_s4_threshold_sensitivity.pdf"
    panels = (
        ("absent_abstention_rate_mean", "Absent-cell abstention"),
        ("shared_false_abstention_rate_mean", "Shared-cell false abstention"),
        ("coverage_mean", "Shared-cell coverage"),
        ("post_abstention_macro_f1_mean", "Post-abstention macro-F1"),
    )
    colors = {
        "coreot_full/u": "#7B3294",
        "uniform_uot/u": "#1F78B4",
    }
    linestyles = {"HLA-DRhi cDC2": "-", "ISG+ cDC2": "--"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    for panel_index, (axis, (column, title)) in enumerate(
        zip(axes.ravel(), panels, strict=True)
    ):
        for label in LABELS:
            for method, score, display_name in POLICIES:
                rows = summary.loc[
                    summary["held_out_label"].eq(label)
                    & summary["method"].eq(method)
                    & summary["score"].eq(score)
                ].sort_values("threshold_percentile")
                key = f"{method}/{score}"
                axis.plot(
                    100 * rows["threshold_percentile"],
                    rows[column],
                    color=colors[key],
                    linestyle=linestyles[label],
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    label=f"{display_name}; {label}",
                )
        axis.set_xticks([90.0, 95.0, 97.5])
        axis.set_xlabel("Full-reference calibration percentile (%)")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_index),
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames: list[pd.DataFrame] = []
    expected_runs = {(label, seed) for label in LABELS for seed in range(1, 6)}
    for grid_dir, policy in (
        (args.grid_dir, POLICIES[0]),
        (args.uniform_grid_dir, POLICIES[1]),
    ):
        descriptors = tuple(
            descriptor
            for descriptor in discover_run_descriptors(grid_dir)
            if descriptor.held_out_label in LABELS
        )
        selected_runs = {
            (descriptor.held_out_label, descriptor.seed) for descriptor in descriptors
        }
        if selected_runs != expected_runs or len(descriptors) != len(expected_runs):
            raise ValueError(
                f"Incomplete selected run set for {policy[0]}: "
                f"{sorted(selected_runs)}"
            )
        for descriptor in descriptors:
            config_dir = grid_dir / descriptor.run_id
            scoring = yaml.safe_load((config_dir / "scoring.yaml").read_text())
            entropy_threshold = float(scoring["thresholds"]["theta_H"]["value"])
            run_root = args.runs_root / descriptor.run_id
            frames.append(
                compute_run_threshold_sensitivity(
                    full_reference_scores=pd.read_parquet(
                        run_root
                        / "scoring/full_reference_control/hiha_harmony30_k100/"
                        "cell_scores.parquet"
                    ),
                    incomplete_scores=pd.read_parquet(
                        run_root
                        / "scoring/incomplete_reference/hiha_harmony30_k100/"
                        "cell_scores.parquet"
                    ),
                    truth=pd.read_csv(
                        run_root
                        / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
                    ),
                    run_id=descriptor.run_id,
                    held_out_label=descriptor.held_out_label,
                    seed=descriptor.seed,
                    entropy_threshold=entropy_threshold,
                    policies=(policy,),
                )
            )

    by_seed = pd.concat(frames, ignore_index=True)
    summary = summarize_threshold_sensitivity(by_seed)
    tables_root = args.output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    by_seed_path = tables_root / "threshold_sensitivity_by_seed.csv"
    summary_path = tables_root / "threshold_sensitivity_summary.csv"
    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    args.manuscript_table_root.mkdir(parents=True, exist_ok=True)
    primary_manuscript_path = (
        args.manuscript_table_root
        / "supplementary_table_s8_calibrated_label_transfer.md"
    )
    primary_manuscript_path.write_text(
        render_primary_policy_table(
            pd.read_csv(args.comparison_root / "compare_detection_summary.csv"),
            pd.read_csv(
                args.comparison_root / "compare_shared_label_transfer_summary.csv"
            ),
        ),
        encoding="utf-8",
    )
    sensitivity_manuscript_path = (
        args.manuscript_table_root
        / "supplementary_table_s9_calibration_percentile_sensitivity.md"
    )
    sensitivity_manuscript_path.write_text(
        render_threshold_sensitivity_table(summary), encoding="utf-8"
    )
    figure_paths = write_threshold_sensitivity_figure(summary, args.figure_root)
    for path in (
        by_seed_path,
        summary_path,
        primary_manuscript_path,
        sensitivity_manuscript_path,
        *figure_paths.values(),
    ):
        print(path)
    return 0


def _validate_numeric(
    frame: pd.DataFrame, columns: tuple[str, ...], run_id: str, method: str
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(
                f"Non-finite {column} values for run={run_id}, method={method}."
            )


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


def _format_summary(row: pd.Series, metric: str) -> str:
    return (
        f"{float(row[f'{metric}_mean']):.3f} "
        f"\\(\\pm\\) {float(row[f'{metric}_std']):.3f}"
    )


def _comparison_summary(
    summary: pd.DataFrame,
    *,
    held_out_label: str,
    method: str,
    score: str,
    quantity: str,
) -> str:
    rows = summary.loc[
        summary["held_out_label"].eq(held_out_label)
        & summary["method"].eq(method)
        & summary["score"].eq(score)
        & summary["quantity"].eq(quantity)
    ]
    if len(rows) != 1:
        raise ValueError(
            "Expected one comparison-summary row for "
            f"{held_out_label}, {method}/{score}, {quantity}; found {len(rows)}."
        )
    row = rows.iloc[0]
    return f"{float(row['mean']):.3f} \\(\\pm\\) {float(row['std']):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
