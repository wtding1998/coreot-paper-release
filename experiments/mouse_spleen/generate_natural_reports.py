from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ENDPOINT_SLUGS = {
    "Ifit B": "ifit_b",
    "Proliferating": "proliferating",
}

SENSITIVITY_COLUMNS = [
    ("configuration", "Method / configuration"),
    ("auroc", "$u$-based AUROC"),
    ("auprc", "$u$-based AUPRC"),
    ("auprc_baseline", "$u$-based AUPRC baseline"),
    ("median_absent", "$u$-based median endpoint"),
    ("median_shared", "$u$-based median shared"),
    ("u_tilde_auroc", "Tilde-u AUROC"),
    ("u_tilde_auprc", "Tilde-u AUPRC"),
    ("u_tilde_auprc_baseline", "Tilde-u AUPRC baseline"),
    ("u_tilde_median_absent", "Tilde-u median endpoint"),
    ("u_tilde_median_shared", "Tilde-u median shared"),
    ("shared_forced_accuracy", "Shared forced accuracy"),
    ("shared_forced_macro_f1", "Shared forced macro-F1"),
]

def _format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = row._asdict()
        lines.append(
            "| "
            + " | ".join(_format_value(values[column]) for column, _ in columns)
            + " |"
        )
    return "\n".join(lines)


def _relative_link(report_path: Path, source_path: Path) -> str:
    return os.path.relpath(source_path, start=report_path.parent)


def _write_baseline_report(
    *,
    endpoint: str,
    report_path: Path,
    detection_path: Path,
    transfer_path: Path,
) -> None:
    detection = pd.read_csv(detection_path)
    transfer = pd.read_csv(transfer_path)
    detection = detection.loc[detection["holdout_label"].eq(endpoint)].copy()
    transfer = transfer.loc[transfer["holdout_label"].eq(endpoint)].copy()
    detection["method_family"] = np.where(
        detection["candidate_set"].eq("external_reference_mapping"),
        "external",
        "internal",
    )
    transfer["method_family"] = np.where(
        transfer["candidate_set"].eq("external_reference_mapping"),
        "external",
        "internal",
    )
    order = ["internal", "external"]
    detection["method_family"] = pd.Categorical(
        detection["method_family"], categories=order, ordered=True
    )
    transfer["method_family"] = pd.Categorical(
        transfer["method_family"], categories=order, ordered=True
    )
    detection = detection.sort_values(["method_family", "method"])
    transfer = transfer.sort_values(["method_family", "method"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# {endpoint}: baseline metrics\n\n"
        f"Detection source: [{detection_path.name}]"
        f"({_relative_link(report_path, detection_path)})  \n"
        f"Shared-cell source: [{transfer_path.name}]"
        f"({_relative_link(report_path, transfer_path)})\n\n"
        "External methods have no matched full-reference threshold calibration; "
        "their abstention-derived metrics are descriptive outputs, not calibrated "
        "selective-prediction comparisons.\n\n"
        "## Detection metrics\n\n"
        + _markdown_table(
            detection,
            [
                ("method_family", "Family"),
                ("method", "Method"),
                ("score", "Score"),
                ("candidate_set", "Candidate set"),
                ("evaluation_scope", "Scope"),
                ("n_query", "N query"),
                ("n_positive", "N positive"),
                ("auroc", "AUROC"),
                ("auprc", "AUPRC"),
                ("auprc_baseline", "AUPRC baseline"),
                ("median_absent", "Median endpoint"),
                ("median_shared", "Median shared"),
                ("threshold_applicability", "Threshold applicability"),
            ],
        )
        + "\n\n## Shared-cell label-transfer metrics\n\n"
        + _markdown_table(
            transfer,
            [
                ("method_family", "Family"),
                ("method", "Method"),
                ("candidate_set", "Candidate set"),
                ("label_transfer_applicability", "Applicability"),
                ("forced_accuracy", "Forced accuracy"),
                ("forced_macro_f1", "Forced macro-F1"),
                ("post_abstention_accuracy", "Post-abstention accuracy"),
                ("post_abstention_macro_f1", "Post-abstention macro-F1"),
                ("coverage", "Coverage"),
                ("shared_false_abstention_rate", "Shared false-abstention rate"),
            ],
        )
        + "\n",
        encoding="utf-8",
    )


def _write_sensitivity_report(
    *,
    endpoint: str,
    report_path: Path,
    sensitivity_path: Path,
    target8_path: Path | None = None,
    baseline_detection_path: Path | None = None,
    baseline_transfer_path: Path | None = None,
) -> None:
    frame = pd.read_csv(sensitivity_path)
    frame = frame.loc[frame["natural_endpoint"].eq(endpoint)].sort_values(
        ["tau", "alpha"]
    )
    target8 = None
    if target8_path is not None and target8_path.is_file():
        target8 = pd.read_csv(target8_path)
        target8 = target8.loc[target8["natural_endpoint"].eq(endpoint)].sort_values(
            ["tau", "alpha"]
        )
    detection_columns = [
        ("tau", "Tau"),
        ("alpha", "Alpha"),
        ("converged", "Converged"),
        ("evaluation_scope", "Scope"),
        ("n_query", "N query"),
        ("n_positive", "N positive"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("auprc_baseline", "AUPRC baseline"),
        ("median_absent", "Median endpoint"),
        ("median_shared", "Median shared"),
    ]
    tilde_columns = [
        ("tau", "Tau"),
        ("alpha", "Alpha"),
        ("converged", "Converged"),
        ("evaluation_scope", "Scope"),
        ("n_query", "N query"),
        ("n_positive", "N positive"),
        ("u_tilde_auroc", "AUROC"),
        ("u_tilde_auprc", "AUPRC"),
        ("u_tilde_auprc_baseline", "AUPRC baseline"),
        ("u_tilde_median_absent", "Median endpoint"),
        ("u_tilde_median_shared", "Median shared"),
    ]
    shared_columns = [
        ("tau", "Tau"),
        ("alpha", "Alpha"),
        ("converged", "Converged"),
        ("shared_forced_accuracy", "Forced accuracy"),
        ("shared_forced_macro_f1", "Forced macro-F1"),
    ]
    content = (
        f"# {endpoint}: constant-tau CoRe-OT alpha/tau metrics\n\n"
        f"Source: [{sensitivity_path.name}]"
        f"({_relative_link(report_path, sensitivity_path)})\n\n"
        "Computed terminal-iterate metrics are retained for every combination, "
        "including fits marked `converged = no`.\n\n"
        "## $u$-based detection metrics — original target policy\n\n"
        + _markdown_table(frame, detection_columns)
    )
    if target8 is not None:
        content += (
            "\n\n## $u$-based detection metrics — tau target = 8\n\n"
            + _markdown_table(target8, detection_columns)
        )
    content += (
        "\n\n## Tilde-u detection metrics — original target policy\n\n"
        + _markdown_table(frame, tilde_columns)
    )
    if target8 is not None:
        content += (
            "\n\n## Tilde-u detection metrics — tau target = 8\n\n"
            + _markdown_table(target8, tilde_columns)
        )
    content += (
        "\n\n## Shared-cell label-transfer metrics — original target policy\n\n"
        + _markdown_table(frame, shared_columns)
    )
    if target8 is not None:
        content += (
            "\n\n## Shared-cell label-transfer metrics — tau target = 8\n\n"
            + _markdown_table(target8, shared_columns)
        )
    if baseline_detection_path is not None:
        baseline_detection = pd.read_csv(baseline_detection_path)
        baseline_detection = baseline_detection.loc[
            baseline_detection["holdout_label"].eq(endpoint)
        ].copy()
        baseline_detection["method_family"] = np.where(
            baseline_detection["candidate_set"].eq("external_reference_mapping"),
            "external",
            "internal",
        )
        baseline_detection["method_family"] = pd.Categorical(
            baseline_detection["method_family"], categories=["internal", "external"], ordered=True
        )
        baseline_detection = baseline_detection.sort_values(
            ["method_family", "method"]
        )
        content += (
            "\n\n## Baseline detection comparison\n\n"
            f"Source: [{baseline_detection_path.name}]"
            f"({_relative_link(report_path, baseline_detection_path)})\n\n"
            + _markdown_table(
                baseline_detection,
                [
                    ("method_family", "Family"),
                    ("method", "Method"),
                    ("score", "Score"),
                    ("candidate_set", "Candidate set"),
                    ("evaluation_scope", "Scope"),
                    ("n_query", "N query"),
                    ("n_positive", "N positive"),
                    ("auroc", "AUROC"),
                    ("auprc", "AUPRC"),
                    ("auprc_baseline", "AUPRC baseline"),
                    ("median_absent", "Median endpoint"),
                    ("median_shared", "Median shared"),
                    ("threshold_applicability", "Threshold applicability"),
                ],
            )
        )
    if baseline_transfer_path is not None:
        baseline_transfer = pd.read_csv(baseline_transfer_path)
        baseline_transfer = baseline_transfer.loc[
            baseline_transfer["holdout_label"].eq(endpoint)
        ].copy()
        baseline_transfer["method_family"] = np.where(
            baseline_transfer["candidate_set"].eq("external_reference_mapping"),
            "external",
            "internal",
        )
        baseline_transfer["method_family"] = pd.Categorical(
            baseline_transfer["method_family"], categories=["internal", "external"], ordered=True
        )
        baseline_transfer = baseline_transfer.sort_values(["method_family", "method"])
        content += (
            "\n\n## Baseline shared-cell label-transfer comparison\n\n"
            f"Source: [{baseline_transfer_path.name}]"
            f"({_relative_link(report_path, baseline_transfer_path)})\n\n"
            + _markdown_table(
                baseline_transfer,
                [
                    ("method_family", "Family"),
                    ("method", "Method"),
                    ("candidate_set", "Candidate set"),
                    ("label_transfer_applicability", "Applicability"),
                    ("forced_accuracy", "Forced accuracy"),
                    ("forced_macro_f1", "Forced macro-F1"),
                    ("post_abstention_accuracy", "Post-abstention accuracy"),
                    ("post_abstention_macro_f1", "Post-abstention macro-F1"),
                    ("coverage", "Coverage"),
                    ("shared_false_abstention_rate", "Shared false-abstention rate"),
                ],
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content + "\n", encoding="utf-8")


def _configuration_label(row: pd.Series, *, family: str) -> str:
    if family == "constant_tau":
        return (
            "coreot_constant_tau "
            f"(tau={float(row['tau']):g}, alpha={float(row['alpha']):g}, "
            f"tau_target={float(row['tau_target']):g})"
        )
    if family == "match_only":
        return (
            "coreot_match_only "
            f"(tau_min={float(row['tau_min']):g}, "
            f"tau_max={float(row['tau_max']):g}, "
            f"tau_target={float(row['tau_target']):g})"
        )
    if family == "full_tau_range":
        return (
            "coreot_full "
            f"(tau_min={float(row['tau_min']):g}, "
            f"tau_max={float(row['tau_max']):g}, "
            f"alpha={float(row['alpha']):g}, "
            f"tau_target={float(row['tau_target']):g})"
        )
    raise ValueError(f"Unknown sensitivity family: {family}")


def _sensitivity_table(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    family: str,
) -> pd.DataFrame:
    endpoint_rows = frame.loc[frame["natural_endpoint"].eq(endpoint)].copy()
    if family == "constant_tau":
        order = ["tau", "alpha"]
    elif family in {"match_only", "full_tau_range"}:
        order = ["tau_min", "tau_max"]
    else:
        raise ValueError(f"Unknown sensitivity family: {family}")
    endpoint_rows = endpoint_rows.sort_values(order)
    endpoint_rows["configuration"] = endpoint_rows.apply(
        _configuration_label, axis=1, family=family
    )
    nonconverged = ~endpoint_rows["converged"].astype(bool)
    endpoint_rows.loc[nonconverged, "configuration"] += (
        " [nonconverged terminal iterate]"
    )
    return endpoint_rows


def _baseline_table(
    detection: pd.DataFrame,
    transfer: pd.DataFrame,
    *,
    endpoint: str,
) -> pd.DataFrame:
    detection = detection.loc[detection["holdout_label"].eq(endpoint)].copy()
    transfer = transfer.loc[transfer["holdout_label"].eq(endpoint)].copy()
    merged = detection.merge(
        transfer.drop(columns=["candidate_set"]),
        on=["run_id", "holdout_label", "method"],
        how="left",
        validate="one_to_one",
    )
    merged["family"] = np.where(
        merged["candidate_set"].eq("external_reference_mapping"),
        "external",
        "internal",
    )
    merged["configuration"] = (
        merged["method"].astype(str) + " (score=" + merged["score"].astype(str) + ")"
    )
    merged["family"] = pd.Categorical(
        merged["family"], categories=["internal", "external"], ordered=True
    )
    return merged.sort_values(["family", "method"])


def _source_link(report_path: Path, source_path: Path) -> str:
    return (
        f"[Source CSV: {source_path.name}]"
        f"({_relative_link(report_path, source_path)})"
    )


def _write_full_report(
    *,
    report_path: Path,
    detection_path: Path,
    transfer_path: Path,
    constant_path: Path,
    match_only_path: Path,
    constant_target8_path: Path,
    match_only_target8_path: Path,
    full_tau_range_path: Path,
    constant_alpha11_20_path: Path | None = None,
    constant_alpha21_40_path: Path | None = None,
) -> None:
    detection = pd.read_csv(detection_path)
    transfer = pd.read_csv(transfer_path)
    sensitivity_inputs = {
        "constant": (pd.read_csv(constant_path), "constant_tau", constant_path),
        "match": (pd.read_csv(match_only_path), "match_only", match_only_path),
        "constant_target8": (
            pd.read_csv(constant_target8_path),
            "constant_tau",
            constant_target8_path,
        ),
        "match_target8": (
            pd.read_csv(match_only_target8_path),
            "match_only",
            match_only_target8_path,
        ),
        "full_tau_range": (
            pd.read_csv(full_tau_range_path),
            "full_tau_range",
            full_tau_range_path,
        ),
    }
    if constant_alpha11_20_path is not None and constant_alpha11_20_path.is_file():
        sensitivity_inputs["constant_alpha11_20"] = (
            pd.read_csv(constant_alpha11_20_path),
            "constant_tau",
            constant_alpha11_20_path,
        )
    if constant_alpha21_40_path is not None and constant_alpha21_40_path.is_file():
        sensitivity_inputs["constant_alpha21_40"] = (
            pd.read_csv(constant_alpha21_40_path),
            "constant_tau",
            constant_alpha21_40_path,
        )
    baseline_columns = [
        ("family", "Family"),
        ("configuration", "Method / configuration"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("auprc_baseline", "AUPRC baseline"),
        ("median_absent", "Median endpoint"),
        ("median_shared", "Median shared"),
        ("forced_accuracy", "Shared forced accuracy"),
        ("forced_macro_f1", "Shared forced macro-F1"),
        ("post_abstention_accuracy", "Post-abstention accuracy"),
        ("post_abstention_macro_f1", "Post-abstention macro-F1"),
        ("coverage", "Coverage"),
        ("shared_false_abstention_rate", "Shared false-abstention rate"),
    ]
    table_specs = [
        ("Constant-tau CoRe-OT", "constant"),
        ("Match-only CoRe-OT", "match"),
        ("Constant-tau CoRe-OT with tau target = 8", "constant_target8"),
        ("Match-only CoRe-OT with tau target = 8", "match_target8"),
        (
            "Full CoRe-OT with default rho (alpha=20, tau target=8)",
            "full_tau_range",
        ),
    ]
    if "constant_alpha11_20" in sensitivity_inputs:
        table_specs.append(
            (
                "Constant-tau CoRe-OT (tau=1, alpha=11-20)",
                "constant_alpha11_20",
            )
        )
    if "constant_alpha21_40" in sensitivity_inputs:
        table_specs.append(
            (
                "Constant-tau CoRe-OT (tau=1, alpha=21-40)",
                "constant_alpha21_40",
            )
        )
    sections: list[str] = ["# Mouse-spleen natural-mismatch results"]
    for endpoint in ENDPOINT_SLUGS:
        baseline = _baseline_table(detection, transfer, endpoint=endpoint)
        endpoint_content = [
            f"## {endpoint}",
            "### Internal and external baselines",
            _source_link(report_path, detection_path)
            + " · "
            + _source_link(report_path, transfer_path),
            _markdown_table(baseline, baseline_columns),
        ]
        for title, key in table_specs:
            frame, family, source_path = sensitivity_inputs[key]
            included = _sensitivity_table(frame, endpoint=endpoint, family=family)
            endpoint_content.extend(
                [
                    f"### {title}",
                    _source_link(report_path, source_path),
                ]
            )
            endpoint_content.append(_markdown_table(included, SENSITIVITY_COLUMNS))
        sections.append("\n\n".join(endpoint_content))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def generate_reports(results_root: Path) -> list[Path]:
    natural_root = results_root / "natural_mismatch"
    detection_path = (
        natural_root / "compare_baselines/tables/baseline_detection_by_run.csv"
    )
    transfer_path = (
        natural_root
        / "compare_baselines/tables/baseline_shared_label_transfer_by_run.csv"
    )
    sensitivity_path = (
        natural_root
        / "sensitivity/constant_tau_alpha/tables/detection_by_run.csv"
    )
    target8_path = (
        natural_root
        / "sensitivity/constant_tau_alpha_tau_target_8/tables/detection_by_run.csv"
    )
    match_only_path = (
        natural_root / "sensitivity/match_only_tau_range/tables/metrics_by_grid.csv"
    )
    match_only_target8_path = (
        natural_root
        / "sensitivity/match_only_tau_range_tau_target_8/tables/metrics_by_grid.csv"
    )
    full_tau_range_path = (
        natural_root
        / "sensitivity/full_tau_range_alpha_20_tau_target_8"
        / "tables/metrics_by_grid.csv"
    )
    constant_alpha11_20_path = (
        natural_root
        / "sensitivity/constant_tau_alpha_alpha11_20/tables/detection_by_run.csv"
    )
    constant_alpha21_40_path = (
        natural_root
        / "sensitivity/constant_tau_alpha_alpha21_40/tables/detection_by_run.csv"
    )
    for path in (
        detection_path,
        transfer_path,
        sensitivity_path,
        target8_path,
        match_only_path,
        match_only_target8_path,
        full_tau_range_path,
        constant_alpha11_20_path,
        constant_alpha21_40_path,
    ):
        if not path.is_file():
            if path in {constant_alpha11_20_path, constant_alpha21_40_path}:
                continue
            raise FileNotFoundError(f"Natural report input does not exist: {path}")
    reports: list[Path] = []
    for endpoint, slug in ENDPOINT_SLUGS.items():
        report_root = natural_root / "reports" / slug
        baseline_report = report_root / "baseline_metrics.md"
        sensitivity_report = report_root / "constant_tau_alpha_metrics.md"
        _write_baseline_report(
            endpoint=endpoint,
            report_path=baseline_report,
            detection_path=detection_path,
            transfer_path=transfer_path,
        )
        _write_sensitivity_report(
            endpoint=endpoint,
            report_path=sensitivity_report,
            sensitivity_path=sensitivity_path,
            target8_path=target8_path,
            baseline_detection_path=detection_path,
            baseline_transfer_path=transfer_path,
        )
        reports.extend([baseline_report, sensitivity_report])
    full_report = natural_root / "reports/natural_mismatch_full_report.md"
    _write_full_report(
        report_path=full_report,
        detection_path=detection_path,
        transfer_path=transfer_path,
        constant_path=sensitivity_path,
        match_only_path=match_only_path,
        constant_target8_path=target8_path,
        match_only_target8_path=match_only_target8_path,
        full_tau_range_path=full_tau_range_path,
        constant_alpha11_20_path=constant_alpha11_20_path
        if constant_alpha11_20_path.is_file()
        else None,
        constant_alpha21_40_path=constant_alpha21_40_path
        if constant_alpha21_40_path.is_file()
        else None,
    )
    reports.append(full_report)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/mouse_spleen_core_ot")
    )
    args = parser.parse_args()
    for path in generate_reports(args.results_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
