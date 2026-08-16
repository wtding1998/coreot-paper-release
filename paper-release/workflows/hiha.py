"""Regenerate manuscript-facing HIHA artifacts from a submission release."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from coreot.results.hiha_parameter_sensitivity import (
    HIHAParameterSensitivityError,
    collect_hiha_parameter_sensitivity,
    render_hiha_parameter_sensitivity,
)


ENDPOINTS = ("HLA-DRhi cDC2", "ISG+ cDC2")
DETECTION_METHODS = (
    ("coreot_full", "u", "CoRe-OT"),
    ("uniform_uot", "u", "Uniform UOT"),
    ("nn", "nn_distance", "Nearest neighbor"),
    ("prior_only", "prior_risk", "Prior only"),
    ("seurat_anchor", "u", "Seurat"),
    ("singleR", "u", "SingleR"),
    ("celltypist_l3", "u", "CellTypist"),
    ("scmap_cell", "u", "scmap-cell"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
TRANSFER_METHODS = tuple(row for row in DETECTION_METHODS if row[0] != "prior_only")
OPERATIONAL_METHODS = tuple(row for row in DETECTION_METHODS if row[0] not in {"nn", "prior_only"})
THRESHOLD_METHODS = (
    ("coreot_full", "u", "CoRe-OT"),
    ("uniform_uot", "u", "Uniform UOT"),
)
PERCENTILES = (0.9, 0.95, 0.975)

Unit = Literal["primary", "parameter-calibration"]


class HIHAReleaseArtifactError(ValueError):
    """Raised when packaged HIHA evidence violates the release contract."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAReleaseArtifactError(f"{source.name} is missing columns {missing}.")


def _validate_endpoint_values(
    frame: pd.DataFrame, source: Path, *, allow_overall: bool = False
) -> None:
    observed = set(frame["held_out_label"].dropna().astype(str))
    allowed = set(ENDPOINTS) | ({"overall"} if allow_overall else set())
    unexpected = observed - allowed
    if unexpected:
        raise HIHAReleaseArtifactError(
            f"{source.name} contains unexpected HIHA endpoints {sorted(unexpected)}."
        )
    missing = set(ENDPOINTS) - observed
    if missing:
        raise HIHAReleaseArtifactError(
            f"{source.name} is missing HIHA endpoints {sorted(missing)}."
        )


def _validate_scores(frame: pd.DataFrame, source: Path) -> None:
    scores = set(frame["score"].dropna().astype(str))
    forbidden = sorted(score for score in scores if score == "u_tilde" or "raw" in score.casefold())
    if forbidden:
        raise HIHAReleaseArtifactError(
            f"{source.name} contains excluded manuscript score labels {forbidden}."
        )


def _one_summary(
    frame: pd.DataFrame,
    *,
    source: Path,
    endpoint: str,
    method: str,
    score: str,
    quantity: str,
    scope: str | None = None,
) -> pd.Series:
    selected = frame.loc[
        frame["held_out_label"].eq(endpoint)
        & frame["method"].eq(method)
        & frame["score"].eq(score)
        & frame["quantity"].eq(quantity)
    ]
    if scope is not None:
        selected = selected.loc[selected["evaluation_scope"].eq(scope)]
    if len(selected) != 1:
        scope_text = f", scope={scope}" if scope is not None else ""
        raise HIHAReleaseArtifactError(
            f"{source.name} must contain exactly one row for endpoint={endpoint}, "
            f"method={method}, score={score}, quantity={quantity}{scope_text}; "
            f"found {len(selected)}."
        )
    row = selected.iloc[0]
    for column in ("mean", "std"):
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            raise HIHAReleaseArtifactError(
                f"{source.name} contains a nonnumeric {column} for {endpoint}/{method}/{quantity}."
            )
    if int(row["n_runs"]) != 5:
        raise HIHAReleaseArtifactError(
            f"{source.name} must summarize five donor splits for {endpoint}/{method}/{quantity}."
        )
    return row


def _mean_std(row: pd.Series) -> str:
    return f"{float(row['mean']):.3f} \\(\\pm\\) {float(row['std']):.3f}"


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _validate_manuscript_text(text: str, artifact_name: str) -> None:
    normalized = text.casefold().replace("-", " ").replace("_", " ")
    forbidden = [label for label in ("u tilde", "raw deficit") if label in normalized]
    suffixed = [label for label in ("CoRe-OT", "Uniform UOT") if f"{label} (" in text]
    if forbidden or suffixed:
        raise HIHAReleaseArtifactError(
            f"{artifact_name} contains excluded manuscript labels: "
            f"{sorted([*forbidden, *suffixed])}."
        )


def _render_detection_table(
    summary: pd.DataFrame,
    *,
    source: Path,
    table_number: int,
    scope: str,
) -> str:
    title = (
        "Held-out-state ranking within the cDC2 query cohort"
        if scope == "local_within_broad_state"
        else "Secondary global all-query detection audit"
    )
    rows: list[tuple[str, ...]] = []
    for method, score, display in DETECTION_METHODS:
        values: list[str] = []
        for endpoint in ENDPOINTS:
            for quantity in ("auprc", "auroc"):
                values.append(
                    _mean_std(
                        _one_summary(
                            summary,
                            source=source,
                            endpoint=endpoint,
                            method=method,
                            score=score,
                            quantity=quantity,
                            scope=scope,
                        )
                    )
                )
        rows.append((display, *values))
    headers = (
        "Method",
        "HLA-DRhi cDC2 AP",
        "HLA-DRhi cDC2 AUROC",
        "ISG+ cDC2 AP",
        "ISG+ cDC2 AUROC",
    )
    return "\n\n".join(
        (
            f"# Supplementary Table S{table_number}. {title}",
            "Values are mean \\(\\pm\\) sample standard deviation across five donor splits. "
            "AP denotes average precision.",
            _markdown_table(headers, rows),
            "_Source: `units/hiha_primary/verified_results/main/tables/"
            f"main_detection_summary.csv`, `evaluation_scope={scope}`._\n",
        )
    )


def _render_transfer_table(summary: pd.DataFrame, *, source: Path) -> str:
    rows: list[tuple[str, ...]] = []
    for method, score, display in TRANSFER_METHODS:
        values: list[str] = []
        for endpoint in ENDPOINTS:
            for quantity in ("forced_accuracy", "forced_macro_f1"):
                values.append(
                    _mean_std(
                        _one_summary(
                            summary,
                            source=source,
                            endpoint=endpoint,
                            method=method,
                            score=score,
                            quantity=quantity,
                        )
                    )
                )
        rows.append((display, *values))
    headers = (
        "Method",
        "HLA-DRhi cDC2 accuracy",
        "HLA-DRhi cDC2 macro-F1",
        "ISG+ cDC2 accuracy",
        "ISG+ cDC2 macro-F1",
    )
    return "\n\n".join(
        (
            "# Supplementary Table S7. Represented-state forced label transfer",
            "Values are mean \\(\\pm\\) sample standard deviation across five donor splits.",
            _markdown_table(headers, rows),
            "_Source: `units/hiha_primary/verified_results/compare_baselines/tables/"
            "compare_shared_label_transfer_summary.csv`._\n",
        )
    )


def _write_manifest(
    *,
    release_root: Path,
    output_root: Path,
    unit: Unit,
    sources: tuple[Path, ...],
    artifacts: tuple[Path, ...],
) -> Path:
    manifest_path = output_root / "manifest.yaml"
    manifest = {
        "schema_version": 1,
        "unit": unit,
        "sources": [
            {
                "path": source.relative_to(release_root).as_posix(),
                "sha256": _sha256(source),
            }
            for source in sources
        ],
        "artifacts": [
            {
                "path": artifact.relative_to(output_root).as_posix(),
                "sha256": _sha256(artifact),
            }
            for artifact in artifacts
        ],
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _generate_primary(release_root: Path, output_root: Path) -> tuple[Path, ...]:
    unit_root = release_root / "units/hiha_primary/verified_results"
    detection_path = unit_root / "main/tables/main_detection_summary.csv"
    transfer_path = unit_root / "compare_baselines/tables/compare_shared_label_transfer_summary.csv"
    detection = pd.read_csv(detection_path)
    transfer = pd.read_csv(transfer_path)
    for frame, path, columns, allow_overall in (
        (
            detection,
            detection_path,
            {
                "held_out_label",
                "method",
                "score",
                "evaluation_scope",
                "quantity",
                "mean",
                "std",
                "n_runs",
            },
            False,
        ),
        (
            transfer,
            transfer_path,
            {"held_out_label", "method", "score", "quantity", "mean", "std", "n_runs"},
            True,
        ),
    ):
        _require_columns(frame, columns, path)
        _validate_endpoint_values(frame, path, allow_overall=allow_overall)
        _validate_scores(frame, path)

    output_root.mkdir(parents=True)
    contents = {
        "supplementary_table_s5.md": _render_detection_table(
            detection,
            source=detection_path,
            table_number=5,
            scope="local_within_broad_state",
        ),
        "supplementary_table_s6.md": _render_detection_table(
            detection,
            source=detection_path,
            table_number=6,
            scope="global_all_query",
        ),
        "supplementary_table_s7.md": _render_transfer_table(
            transfer,
            source=transfer_path,
        ),
    }
    artifacts = tuple(output_root / name for name in contents)
    for path, content in zip(artifacts, contents.values(), strict=True):
        _validate_manuscript_text(content, path.name)
        path.write_text(content, encoding="utf-8")
    manifest = _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit="primary",
        sources=(detection_path, transfer_path),
        artifacts=artifacts,
    )
    return (*artifacts, manifest)


def _render_primary_policy_table(
    detection: pd.DataFrame,
    transfer: pd.DataFrame,
    *,
    detection_path: Path,
    transfer_path: Path,
) -> str:
    rows: list[tuple[str, ...]] = []
    for endpoint in ENDPOINTS:
        for method, score, display in OPERATIONAL_METHODS:
            values = (
                _mean_std(
                    _one_summary(
                        detection,
                        source=detection_path,
                        endpoint=endpoint,
                        method=method,
                        score=score,
                        quantity="absent_abstention_rate",
                    )
                ),
                _mean_std(
                    _one_summary(
                        transfer,
                        source=transfer_path,
                        endpoint=endpoint,
                        method=method,
                        score=score,
                        quantity="coverage",
                    )
                ),
                _mean_std(
                    _one_summary(
                        transfer,
                        source=transfer_path,
                        endpoint=endpoint,
                        method=method,
                        score=score,
                        quantity="post_abstention_macro_f1",
                    )
                ),
            )
            rows.append((endpoint, display, *values))
    headers = (
        "Endpoint",
        "Method",
        "Held-out-state abstention rate",
        "Represented-state coverage",
        "Post-abstention macro-F1",
    )
    return "\n\n".join(
        (
            "# Supplementary Table S8. Calibrated held-out-state abstention and "
            "represented-state label transfer under the primary operational policy",
            "Values are arithmetic means \\(\\pm\\) sample standard deviations across "
            "five donor splits. The score cutoff is the method-specific 95th percentile "
            "estimated from the corresponding matched full-reference control. "
            "Held-out-state abstention and represented-state coverage are evaluated on "
            "disjoint cell sets and are therefore not complements; post-abstention "
            "macro-F1 is conditional on retained represented-state cells. Methods without "
            "an applicable operational abstention policy are omitted.",
            _markdown_table(headers, rows),
            "_Sources: `units/hiha_parameter_and_calibration/data/comparison/"
            "compare_detection_summary.csv` and `units/hiha_parameter_and_calibration/"
            "data/comparison/compare_shared_label_transfer_summary.csv`._\n",
        )
    )


def _threshold_row(
    summary: pd.DataFrame,
    *,
    source: Path,
    endpoint: str,
    method: str,
    score: str,
    percentile: float,
) -> pd.Series:
    percentile_values = pd.to_numeric(summary["threshold_percentile"], errors="coerce")
    selected = summary.loc[
        summary["held_out_label"].eq(endpoint)
        & summary["method"].eq(method)
        & summary["score"].eq(score)
        & percentile_values.map(lambda value: math.isclose(value, percentile))
    ]
    if len(selected) != 1:
        raise HIHAReleaseArtifactError(
            f"{source.name} must contain exactly one row for endpoint={endpoint}, "
            f"method={method}, score={score}, percentile={percentile}; found {len(selected)}."
        )
    row = selected.iloc[0]
    metric_columns = tuple(
        f"{metric}_{stat}"
        for metric in (
            "absent_abstention_rate",
            "coverage",
            "post_abstention_macro_f1",
        )
        for stat in ("mean", "std")
    )
    if pd.to_numeric(row[list(metric_columns)], errors="coerce").isna().any():
        raise HIHAReleaseArtifactError(
            f"{source.name} contains nonnumeric metrics for {endpoint}/{method}/{percentile}."
        )
    if int(row["n_runs"]) != 5:
        raise HIHAReleaseArtifactError(
            f"{source.name} must summarize five donor splits for {endpoint}/{method}/{percentile}."
        )
    return row


def _threshold_mean_std(row: pd.Series, metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.3f} \\(\\pm\\) {float(row[f'{metric}_std']):.3f}"


def _render_threshold_table(summary: pd.DataFrame, *, source: Path) -> str:
    rows: list[tuple[str, ...]] = []
    for endpoint in ENDPOINTS:
        for method, score, display in THRESHOLD_METHODS:
            for percentile in PERCENTILES:
                row = _threshold_row(
                    summary,
                    source=source,
                    endpoint=endpoint,
                    method=method,
                    score=score,
                    percentile=percentile,
                )
                rows.append(
                    (
                        endpoint,
                        display,
                        f"{100 * percentile:g}%",
                        _threshold_mean_std(row, "absent_abstention_rate"),
                        _threshold_mean_std(row, "coverage"),
                        _threshold_mean_std(row, "post_abstention_macro_f1"),
                    )
                )
    headers = (
        "Endpoint",
        "Method",
        "Calibration percentile",
        "Held-out-state abstention rate",
        "Represented-state coverage",
        "Post-abstention macro-F1",
    )
    return "\n\n".join(
        (
            "# Supplementary Table S9. Calibration-percentile sensitivity for "
            "CoRe-OT and Uniform UOT",
            "Values are arithmetic means \\(\\pm\\) sample standard deviations across "
            "five donor splits. For each donor split and method, the query-marginal-deficit "
            "cutoff was estimated at the indicated percentile from all finite scores in the "
            "matched full-reference control and applied to the corresponding incomplete-"
            "reference scores. The fitted transports were unchanged, and the normalized "
            "label-entropy cutoff remained \\(0.8\\).",
            _markdown_table(headers, rows),
            "_Source: `units/hiha_parameter_and_calibration/verified_results/"
            "threshold_sensitivity_summary.csv`._\n",
        )
    )


def _assert_primary_threshold_agreement(
    detection: pd.DataFrame,
    transfer: pd.DataFrame,
    threshold: pd.DataFrame,
    *,
    detection_path: Path,
    transfer_path: Path,
    threshold_path: Path,
) -> None:
    for endpoint in ENDPOINTS:
        for method, score, _ in THRESHOLD_METHODS:
            threshold_row = _threshold_row(
                threshold,
                source=threshold_path,
                endpoint=endpoint,
                method=method,
                score=score,
                percentile=0.95,
            )
            for metric, frame, source in (
                ("absent_abstention_rate", detection, detection_path),
                ("coverage", transfer, transfer_path),
                ("post_abstention_macro_f1", transfer, transfer_path),
            ):
                primary_row = _one_summary(
                    frame,
                    source=source,
                    endpoint=endpoint,
                    method=method,
                    score=score,
                    quantity=metric,
                )
                for stat in ("mean", "std"):
                    first = float(primary_row[stat])
                    second = float(threshold_row[f"{metric}_{stat}"])
                    if not math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-12):
                        raise HIHAReleaseArtifactError(
                            "S8 and S9 disagree at the 95% calibration percentile for "
                            f"{endpoint}/{method}/{metric}/{stat}: {first} != {second}."
                        )


def _validate_parameter_manifest(unit_root: Path) -> tuple[tuple[Path, ...], str]:
    manifest_path = unit_root / "manifests/hiha_parameter_sensitivity_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    source_paths = {
        "hla_within_cdc2_detection_by_run": (
            unit_root / "data/parameter_sources/hla_detection_within_cdc2_by_run.csv"
        ),
        "hla_transfer_by_run": (
            unit_root / "data/parameter_sources/hla_shared_label_transfer_by_run.csv"
        ),
        "isg_discovery_by_split": (unit_root / "data/parameter_sources/isg_discovery_by_split.csv"),
    }
    for name, path in source_paths.items():
        try:
            expected = manifest["sources"][name]["sha256"]
        except (KeyError, TypeError) as error:
            raise HIHAReleaseArtifactError(
                f"{manifest_path.name} is missing the {name} source hash."
            ) from error
        observed = _sha256(path)
        if observed != expected:
            raise HIHAReleaseArtifactError(
                f"Packaged Figure S4 source hash mismatch for {path.name}: "
                f"expected {expected}, observed {observed}."
            )
    try:
        expected_png_hash = str(manifest["artifacts"]["figure_png"]["sha256"])
    except (KeyError, TypeError) as error:
        raise HIHAReleaseArtifactError(
            f"{manifest_path.name} is missing the Figure S4 PNG hash."
        ) from error
    return tuple(source_paths.values()), expected_png_hash


def _generate_parameter_calibration(release_root: Path, output_root: Path) -> tuple[Path, ...]:
    unit_root = release_root / "units/hiha_parameter_and_calibration"
    detection_path = unit_root / "data/comparison/compare_detection_summary.csv"
    transfer_path = unit_root / "data/comparison/compare_shared_label_transfer_summary.csv"
    threshold_path = unit_root / "verified_results/threshold_sensitivity_summary.csv"
    detection = pd.read_csv(detection_path)
    transfer = pd.read_csv(transfer_path)
    threshold = pd.read_csv(threshold_path)
    comparison_columns = {
        "held_out_label",
        "method",
        "score",
        "quantity",
        "mean",
        "std",
        "n_runs",
    }
    for frame, path in ((detection, detection_path), (transfer, transfer_path)):
        _require_columns(frame, comparison_columns, path)
        _validate_endpoint_values(frame, path, allow_overall=True)
        _validate_scores(frame, path)
    threshold_columns = {
        "held_out_label",
        "method",
        "score",
        "display_name",
        "threshold_percentile",
        "absent_abstention_rate_mean",
        "absent_abstention_rate_std",
        "coverage_mean",
        "coverage_std",
        "post_abstention_macro_f1_mean",
        "post_abstention_macro_f1_std",
        "n_runs",
    }
    _require_columns(threshold, threshold_columns, threshold_path)
    _validate_endpoint_values(threshold, threshold_path)
    _validate_scores(threshold, threshold_path)
    _assert_primary_threshold_agreement(
        detection,
        transfer,
        threshold,
        detection_path=detection_path,
        transfer_path=transfer_path,
        threshold_path=threshold_path,
    )

    parameter_sources, expected_png_hash = _validate_parameter_manifest(unit_root)
    try:
        by_split, figure_summary = collect_hiha_parameter_sensitivity(
            hla_within_cdc2_detection_path=parameter_sources[0],
            hla_transfer_path=parameter_sources[1],
            isg_path=parameter_sources[2],
        )
    except HIHAParameterSensitivityError as error:
        raise HIHAReleaseArtifactError(f"Figure S4 parameter grid is invalid: {error}") from error
    expected_by_split = pd.read_csv(
        unit_root / "verified_results/hiha_parameter_sensitivity_by_split.csv"
    )
    expected_summary = pd.read_csv(
        unit_root / "verified_results/hiha_parameter_sensitivity_summary.csv"
    )
    try:
        pd.testing.assert_frame_equal(by_split, expected_by_split, check_exact=False, atol=1.0e-15)
        pd.testing.assert_frame_equal(
            figure_summary,
            expected_summary,
            check_exact=False,
            atol=1.0e-15,
        )
    except AssertionError as error:
        raise HIHAReleaseArtifactError(
            "Regenerated Figure S4 source summaries disagree with verified release results."
        ) from error

    output_root.mkdir(parents=True)
    table_s8 = output_root / "supplementary_table_s8.md"
    table_s9 = output_root / "supplementary_table_s9.md"
    table_s8_text = _render_primary_policy_table(
        detection,
        transfer,
        detection_path=detection_path,
        transfer_path=transfer_path,
    )
    table_s9_text = _render_threshold_table(threshold, source=threshold_path)
    _validate_manuscript_text(table_s8_text, table_s8.name)
    _validate_manuscript_text(table_s9_text, table_s9.name)
    table_s8.write_text(table_s8_text, encoding="utf-8")
    table_s9.write_text(table_s9_text, encoding="utf-8")
    figures = {
        suffix: output_root / f"supplementary_figure_s4.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    render_hiha_parameter_sensitivity(figure_summary, figures)
    observed_png_hash = _sha256(figures["png"])
    if observed_png_hash != expected_png_hash:
        raise HIHAReleaseArtifactError(
            "The rendered Figure S4 PNG disagrees with its packaged reference hash: "
            f"expected {expected_png_hash}, observed {observed_png_hash}."
        )
    alt_path = output_root / "supplementary_figure_s4_alt.txt"
    alt_path.write_text(
        "Four-panel HIHA query-penalty-bound sensitivity figure. Panels A and B "
        "show within-cDC2 average precision, and Panels C and D show "
        "represented-state forced macro-F1 for the HLA-DRhi cDC2 and ISG+ cDC2 "
        "endpoints, respectively. Stars mark the reported operating point in the "
        "ISG+ cDC2 panels; the reported HLA-DRhi cDC2 point is outside its "
        "displayed grid.\n",
        encoding="utf-8",
    )
    artifacts = (table_s8, table_s9, *figures.values(), alt_path)
    source_paths = (
        detection_path,
        transfer_path,
        threshold_path,
        unit_root / "manifests/hiha_parameter_sensitivity_manifest.yaml",
        *parameter_sources,
        unit_root / "verified_results/hiha_parameter_sensitivity_by_split.csv",
        unit_root / "verified_results/hiha_parameter_sensitivity_summary.csv",
    )
    manifest = _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit="parameter-calibration",
        sources=source_paths,
        artifacts=artifacts,
    )
    return (*artifacts, manifest)


def generate_hiha_release_artifacts(
    *,
    release_root: Path,
    output_root: Path,
    unit: Unit,
) -> tuple[Path, ...]:
    """Generate one HIHA release artifact family without fitting any model."""
    release_root = release_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise HIHAReleaseArtifactError(f"Output root already exists: {output_root}")
    if unit == "primary":
        return _generate_primary(release_root, output_root)
    if unit == "parameter-calibration":
        return _generate_parameter_calibration(release_root, output_root)
    raise HIHAReleaseArtifactError(f"Unsupported HIHA unit: {unit!r}")
