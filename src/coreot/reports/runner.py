from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coreot.artifacts.manifests import Manifest, read_manifest, write_manifest
from coreot.artifacts.run_artifacts import ArtifactInvalid, ArtifactMissing, RunArtifacts
from coreot.config.load import load_yaml
from coreot.reports.markdown import render_baseline_markdown_report


STAGE = "report"


class ReportRunnerError(ValueError):
    """Raised when report generation cannot proceed."""


@dataclass(frozen=True)
class ReportResult:
    reports_root: Path
    markdown_path: Path


def run_report(config_path: str | Path) -> ReportResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    outputs = _required_mapping(config, ("outputs",))
    if not bool(outputs.get("markdown", False)):
        raise ReportRunnerError("report currently requires outputs.markdown=true")
    if bool(outputs.get("html", False)):
        raise ReportRunnerError("HTML report output is not implemented yet")

    output_root = _required_path(config, ("outputs", "root"))
    run_root = output_root / run_id
    evaluation_root = run_root / "evaluation"
    artifacts = RunArtifacts(run_root, STAGE)
    metrics_artifact = artifacts.evaluation().metrics()
    forced_summary_artifact = artifacts.evaluation().forced_label_summary()
    metrics_path = metrics_artifact.path
    forced_path = forced_summary_artifact.path
    evaluation_manifest_path = evaluation_root / "evaluation_manifest.yaml"
    try:
        metrics = metrics_artifact.read()
    except ArtifactMissing as exc:
        raise FileNotFoundError(f"Evaluation metrics do not exist: {metrics_path}") from exc
    except ArtifactInvalid as exc:
        if metrics_path == exc.path and "must not be empty" in str(exc):
            raise ReportRunnerError(f"Evaluation metrics are empty: {metrics_path}") from exc
        raise ReportRunnerError(f"Evaluation metrics are invalid: {metrics_path}") from exc

    try:
        forced_label_summary = forced_summary_artifact.read()
    except ArtifactMissing as exc:
        raise FileNotFoundError(f"Forced-label summary does not exist: {forced_path}") from exc
    except ArtifactInvalid as exc:
        raise ReportRunnerError(f"Forced-label summary is invalid: {forced_path}") from exc

    if not evaluation_manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest does not exist: {evaluation_manifest_path}")

    evaluation_manifest = read_manifest(evaluation_manifest_path)

    reports_root = run_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    markdown_path = reports_root / "benchmark_report.md"
    report_manifest_path = reports_root / "report_manifest.yaml"
    markdown_path.write_text(
        render_baseline_markdown_report(
            metrics,
            forced_label_summary,
            provenance={
                "run_id": run_id,
                "evaluation_metrics": str(metrics_path),
                "forced_label_summary": str(forced_path),
                "evaluation_manifest": str(evaluation_manifest_path),
                "evaluation_scope": evaluation_manifest.metadata.get("metrics_scope", ""),
                "report_manifest": str(report_manifest_path),
                "report_scope": "baseline_markdown_only",
            },
        ),
        encoding="utf-8",
    )
    write_manifest(
        report_manifest_path,
        Manifest(
            stage=STAGE,
            artifacts={"markdown": str(markdown_path)},
            metadata={"report_scope": "baseline_markdown_only"},
        ),
    )
    return ReportResult(reports_root=reports_root, markdown_path=markdown_path)


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise ReportRunnerError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_mapping(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value = _required_value(config, path)
    if not isinstance(value, dict):
        dotted = ".".join(path)
        raise ReportRunnerError(f"Expected mapping config value: {dotted}")
    return value


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise ReportRunnerError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
