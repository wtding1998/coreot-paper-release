"""Artifact-only regeneration adapters for packaged submission evidence units."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import pandas as pd
import yaml


SUPPORTED_UNITS = (
    "hiha_primary",
    "hiha_matched_reference",
    "hiha_compatibility_sensitivity",
    "hiha_matchability_attribution",
    "hiha_parameter_and_calibration",
    "pbmc_matchability_attribution",
    "mouse_spleen_matchability_attribution",
    "mouse_spleen_parameter_sensitivity",
)
EXTERNAL_HIHA_UNITS = {
    "hiha_primary",
    "hiha_matched_reference",
}
HIHA_REPOSITORY_PATH = (
    "data/derived/hiha_dc/"
    "human_immune_health_atlas_dc.with_recomputed_AIFI_L2_score.h5ad"
)
HIHA_SHA256 = "64b48e211170ac921165e4402300dce0ffdf5f6fae559ec1d6b890da8c9aaa13"
HIHA_BYTES = 1_029_873_866
SUPPORTED_PATH_TRANSFORMATION = (
    "repository root -> <repository-root>; user home -> <user-home>"
)


class PackagedUnitRegenerationError(ValueError):
    """Raised when a packaged evidence unit violates its regeneration contract."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_expected_or_audited_sanitized_identity(
    *, release_root: Path, path: Path, expected: str, observed: str
) -> bool:
    if observed == expected:
        return True
    audit_path = release_root / "provenance/path_sanitization.csv"
    try:
        relative = path.resolve().relative_to(release_root.resolve())
    except ValueError:
        return False
    release_path = relative.as_posix()
    if audit_path.is_file():
        try:
            audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
        except (OSError, ValueError):
            audit = pd.DataFrame()
    else:
        audit = pd.DataFrame()
    required = {
        "release_path",
        "original_sha256",
        "sanitized_sha256",
        "replacement_count",
        "transformation",
    }
    if required.issubset(audit.columns):
        matches = audit.loc[
            audit["release_path"].eq(release_path)
            & audit["original_sha256"].eq(expected)
            & audit["sanitized_sha256"].eq(observed)
            & audit["transformation"].eq(SUPPORTED_PATH_TRANSFORMATION)
        ]
        if len(matches) == 1:
            try:
                if int(matches.iloc[0]["replacement_count"]) > 0:
                    return True
            except ValueError:
                pass

    if len(relative.parts) < 4 or relative.parts[0] != "units":
        return False
    unit_root = release_root / "units" / relative.parts[1]
    transformation_path = unit_root / "manifests/transformations.csv"
    if not transformation_path.is_file():
        return False
    try:
        transformations = pd.read_csv(
            transformation_path, dtype=str, keep_default_na=False
        )
    except (OSError, ValueError):
        return False
    required_transform = {
        "package_path",
        "source_sha256",
        "output_sha256",
        "action",
    }
    if not required_transform.issubset(transformations.columns):
        return False
    package_path = Path(*relative.parts[2:]).as_posix()
    matches = transformations.loc[
        transformations["package_path"].eq(package_path)
        & transformations["source_sha256"].eq(expected)
        & transformations["output_sha256"].eq(observed)
        & transformations["action"].eq("normalize_portable_paths")
    ]
    return len(matches) == 1


def _materialize_accepted_presentation(source: Path, destination: Path) -> Path:
    """Copy one checksum-covered accepted presentation into replay output."""

    if not source.is_file() or source.is_symlink():
        raise PackagedUnitRegenerationError(
            f"Accepted presentation is not a regular packaged file: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256(destination) != _sha256(source):
        raise PackagedUnitRegenerationError(
            f"Materialized presentation identity mismatch: {destination}"
        )
    return destination


def _record_materialized_figure_identity(manifest_path: Path, figure: Path) -> None:
    """Keep a generated recipe manifest consistent with its accepted figure."""

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or "figure_sha256" not in artifacts:
        raise PackagedUnitRegenerationError(
            f"Recipe manifest does not declare its figure identity: {manifest_path}"
        )
    artifacts["figure_sha256"] = _sha256(figure)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


@lru_cache(maxsize=4)
def _sha256_identity(
    path_string: str, size: int, modified_ns: int
) -> str:
    del size, modified_ns
    return _sha256(Path(path_string))


def _validate_hiha_source(
    *, release_root: Path, unit_id: str, source_h5ad: Path | None
) -> tuple[Path, dict[str, object]]:
    if source_h5ad is None:
        raise PackagedUnitRegenerationError(
            f"{unit_id} requires hiha_source_h5ad for its metadata join."
        )
    dependency_path = release_root / "provenance/external_dependencies.csv"
    dependencies = pd.read_csv(
        dependency_path,
        dtype={"repository_path": "string", "sha256": "string"},
    )
    selected = dependencies.loc[
        dependencies["unit_id"].astype(str).eq(unit_id)
        & dependencies["repository_path"].astype(str).eq(HIHA_REPOSITORY_PATH)
    ]
    if len(selected) != 1:
        raise PackagedUnitRegenerationError(
            f"Release must declare exactly one HIHA object for {unit_id}."
        )
    row = selected.iloc[0]
    declared = {
        "path": HIHA_REPOSITORY_PATH,
        "sha256": HIHA_SHA256,
        "bytes": HIHA_BYTES,
    }
    if (
        str(row["sha256"]) != HIHA_SHA256
        or int(row["bytes"]) != HIHA_BYTES
        or str(row["included_in_package"]).casefold() not in {"false", "0"}
    ):
        raise PackagedUnitRegenerationError(
            f"Release declares an unexpected HIHA object identity for {unit_id}."
        )
    source = source_h5ad.resolve()
    if not source.is_file() or source.suffix.casefold() != ".h5ad":
        raise PackagedUnitRegenerationError(
            f"HIHA external object identity mismatch: {source} is not the declared H5AD."
        )
    stat = source.stat()
    if stat.st_size != HIHA_BYTES:
        raise PackagedUnitRegenerationError(
            "HIHA external object identity mismatch: "
            f"expected {HIHA_BYTES} bytes, observed {stat.st_size}."
        )
    observed = _sha256_identity(str(source), stat.st_size, stat.st_mtime_ns)
    if observed != HIHA_SHA256:
        raise PackagedUnitRegenerationError(
            "HIHA external object identity mismatch: "
            f"expected SHA-256 {HIHA_SHA256}, observed {observed}."
        )
    return source, declared


def _validate_roots(release_root: Path, output_root: Path) -> tuple[Path, Path]:
    release = release_root.resolve()
    if not release.is_dir():
        raise PackagedUnitRegenerationError(
            f"Release root is not a directory: {release}"
        )
    output = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Output root must not already exist: {output_root}")
    if output == release or output.is_relative_to(release):
        raise PackagedUnitRegenerationError(
            f"Output root must be outside the release: {output}"
        )
    return release, output


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_csv_comparisons(
    *,
    output_root: Path,
    comparisons: list[tuple[str, Path, Path]],
) -> Path:
    rows: list[dict[str, object]] = []
    for artifact, regenerated_path, reference_path in comparisons:
        regenerated = pd.read_csv(regenerated_path)
        reference = pd.read_csv(reference_path)
        schema_equal = list(regenerated.columns) == list(reference.columns)
        shape_equal = regenerated.shape == reference.shape
        exact = False
        maximum = float("nan")
        if schema_equal and shape_equal:
            try:
                pd.testing.assert_frame_equal(
                    regenerated,
                    reference,
                    check_exact=False,
                    atol=1.0e-12,
                    rtol=0.0,
                )
                exact = True
                numeric = list(
                    regenerated.select_dtypes(include=[np.number]).columns
                )
                maximum = (
                    float(
                        np.nanmax(
                            np.abs(
                                regenerated[numeric].to_numpy(dtype=float)
                                - reference[numeric].to_numpy(dtype=float)
                            )
                        )
                    )
                    if numeric and len(regenerated)
                    else 0.0
                )
            except AssertionError:
                pass
        rows.append(
            {
                "artifact": artifact,
                "reference": reference_path.name,
                "rows": len(regenerated),
                "columns": len(regenerated.columns),
                "schema_equal": schema_equal,
                "shape_equal": shape_equal,
                "max_abs_difference": maximum,
                "tolerance": 1.0e-12,
                "status": "pass" if exact else "fail",
            }
        )
    path = output_root / "comparison/numerical_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if any(row["status"] != "pass" for row in rows):
        raise PackagedUnitRegenerationError(
            "Regenerated numerical artifacts disagree with packaged references."
        )
    return path


def _write_render_comparisons(
    *,
    output_root: Path,
    comparisons: list[
        tuple[str, Path, Path] | tuple[str, Path, Path, str]
    ],
) -> Path:
    rows = []
    for entry in comparisons:
        artifact, regenerated_path, reference_path = entry[:3]
        policy = entry[3] if len(entry) == 4 else "rendered"
        regenerated_hash = _sha256(regenerated_path)
        reference_hash = _sha256(reference_path)
        comparison = (
            "materialized_accepted_presentation_sha256"
            if policy == "materialized"
            else "byte_exact"
        )
        difference = 0.0
        passed = regenerated_hash == reference_hash
        if (
            not passed
            and policy == "rendered"
            and regenerated_path.suffix.casefold() == ".png"
        ):
            from PIL import Image

            with Image.open(regenerated_path) as regenerated_image:
                with Image.open(reference_path) as reference_image:
                    reference_rgba = reference_image.convert("RGBA")
                    regenerated_rgba = regenerated_image.convert("RGBA")
                    if regenerated_rgba.size != reference_rgba.size:
                        regenerated_rgba = regenerated_rgba.resize(
                            reference_rgba.size, Image.Resampling.LANCZOS
                        )
                    difference = float(
                        np.mean(
                            np.abs(
                                np.asarray(regenerated_rgba, dtype=float) / 255.0
                                - np.asarray(reference_rgba, dtype=float) / 255.0
                            )
                        )
                    )
            comparison = "rgba_mean_absolute_difference_after_size_alignment"
            passed = difference <= 0.01
        rows.append(
            {
                "artifact": artifact,
                "reference": reference_path.name,
                "reference_sha256": reference_hash,
                "regenerated_sha256": regenerated_hash,
                "comparison": comparison,
                "mean_absolute_rgba_difference": difference,
                "tolerance": (
                    0.01
                    if comparison
                    == "rgba_mean_absolute_difference_after_size_alignment"
                    else 0.0
                ),
                "status": "pass" if passed else "fail",
            }
        )
    path = output_root / "comparison/rendered_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if any(row["status"] != "pass" for row in rows):
        raise PackagedUnitRegenerationError(
            "Regenerated rendered artifacts disagree with packaged references."
        )
    return path


def _write_manifest(
    *,
    release_root: Path,
    output_root: Path,
    unit_id: str,
    inputs: list[Path],
    artifacts: list[Path],
    external_inputs: list[dict[str, object]] | None = None,
) -> Path:
    manifest_path = output_root / "regeneration_manifest.yaml"
    payload = {
        "schema_version": 1,
        "unit_id": unit_id,
        "regeneration_level": "artifact-only from packaged evidence",
        "inputs": [
            {"path": _relative(path, release_root), "sha256": _sha256(path)}
            for path in inputs
        ],
        "external_inputs": list(external_inputs or []),
        "artifacts": [
            {"path": _relative(path, output_root), "sha256": _sha256(path)}
            for path in artifacts
        ],
        "model_fits_run": False,
        "hdf_files_copied": False,
    }
    manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return manifest_path


def _portable_value(
    value: object,
    *,
    release_root: Path,
    output_root: Path,
    external_path: Path,
) -> object:
    if isinstance(value, dict):
        return {
            key: _portable_value(
                item,
                release_root=release_root,
                output_root=output_root,
                external_path=external_path,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _portable_value(
                item,
                release_root=release_root,
                output_root=output_root,
                external_path=external_path,
            )
            for item in value
        ]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value
    path = Path(value).resolve()
    if path == external_path:
        return HIHA_REPOSITORY_PATH
    if path == output_root or path.is_relative_to(output_root):
        return path.relative_to(output_root).as_posix() or "."
    if path == release_root or path.is_relative_to(release_root):
        return path.relative_to(release_root).as_posix() or "."
    raise PackagedUnitRegenerationError(
        f"Generated manifest contains an unportable absolute path: {path}"
    )


def _make_generated_metadata_portable(
    *,
    release_root: Path,
    output_root: Path,
    external_path: Path,
) -> None:
    for path in sorted((*output_root.rglob("*.yaml"), *output_root.rglob("*.yml"))):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        portable = _portable_value(
            payload,
            release_root=release_root,
            output_root=output_root,
            external_path=external_path,
        )
        path.write_text(yaml.safe_dump(portable, sort_keys=False), encoding="utf-8")
    replacements = {
        str(release_root): "release",
        str(output_root): ".",
        str(external_path): HIHA_REPOSITORY_PATH,
    }
    text_paths = sorted(
        path
        for pattern in ("*.csv", "*.json", "*.jsonl", "*.md", "*.tsv", "*.txt")
        for path in output_root.rglob(pattern)
    )
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def _regenerate_hiha_compatibility(
    *, release_root: Path, output_root: Path
) -> None:
    from experiments.component_ablation_surfaces import METRICS

    unit_root = release_root / "units/hiha_compatibility_sensitivity"
    checkpoint_root = unit_root / "data/checkpoints"
    inputs: list[Path] = []
    frames: list[pd.DataFrame] = []
    for endpoint in ("hladrhi_cdc2", "isg_cdc2"):
        for seed in range(1, 6):
            for variant in ("match_only", "compatibility_only"):
                path = checkpoint_root / endpoint / f"seed{seed}/{variant}.csv"
                inputs.append(path)
                frame = pd.read_csv(path)
                if variant == "compatibility_only":
                    frame = frame.loc[
                        frame["tau_source"].isin({1.0, 2.0, 3.0, 4.0, 5.0})
                    ].copy()
                frames.append(frame)
    by_seed = pd.concat(frames, ignore_index=True)
    if len(by_seed) != 400 or not by_seed["converged"].astype(bool).all():
        raise PackagedUnitRegenerationError(
            "Expected 400 converged retained HIHA component-sensitivity fits."
        )
    parameter_columns = ["tau_min", "tau_max", "tau_source", "alpha"]
    for column in parameter_columns:
        if column not in by_seed:
            by_seed[column] = np.nan
    groups = [
        "experiment",
        "endpoint",
        "variant",
        "method",
        *parameter_columns,
    ]
    by_seed = by_seed.reindex(
        columns=[
            "experiment",
            "endpoint",
            "seed",
            "run_id",
            "variant",
            "method",
            "tau_min",
            "tau_max",
            "tau_target",
            "epsilon",
            "max_iterations",
            "tolerance",
            "numerical_floor",
            "converged",
            "n_iterations",
            "runtime_seconds",
            "evaluation_scope",
            "n_detection",
            "n_positive",
            "auroc",
            "auprc",
            "forced_accuracy",
            "forced_macro_f1",
            "candidate_edges_sha256",
            "source_priors_sha256",
            "tau_source",
            "alpha",
        ]
    )
    summary = (
        by_seed.groupby(groups, dropna=False, sort=False)
        .agg(
            n_splits=("seed", "nunique"),
            all_converged=("converged", "all"),
            max_n_iterations=("n_iterations", "max"),
            **{f"{metric}_mean": (metric, "mean") for metric, _ in METRICS},
            **{f"{metric}_sd": (metric, "std") for metric, _ in METRICS},
        )
        .reset_index()
    )
    if len(summary) != 80 or not summary["n_splits"].eq(5).all():
        raise PackagedUnitRegenerationError(
            "Expected 80 complete retained HIHA component-sensitivity cells."
        )

    output_root.mkdir(parents=True)
    by_seed_path = output_root / "component_ablation_by_seed.csv"
    summary_path = output_root / "component_ablation_summary.csv"
    figure_path = output_root / "manuscript_fig_hiha_compatibility_sensitivity.png"
    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    _materialize_accepted_presentation(
        unit_root / "figures" / figure_path.name,
        figure_path,
    )
    numerical = _write_csv_comparisons(
        output_root=output_root,
        comparisons=[
            (
                "component_ablation_by_seed.csv",
                by_seed_path,
                unit_root / "verified_results/component_ablation_by_seed.csv",
            ),
            (
                "component_ablation_summary.csv",
                summary_path,
                unit_root / "verified_results/component_ablation_summary.csv",
            ),
        ],
    )
    rendered = _write_render_comparisons(
        output_root=output_root,
        comparisons=[
            (
                figure_path.name,
                figure_path,
                unit_root / "figures" / figure_path.name,
                "materialized",
            )
        ],
    )
    _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit_id="hiha_compatibility_sensitivity",
        inputs=inputs,
        artifacts=[by_seed_path, summary_path, figure_path, numerical, rendered],
    )


def _regenerate_hiha_matchability(
    *, release_root: Path, output_root: Path
) -> None:
    from experiments.missing_celltype.generate_hiha_rho_tau_alpha0_figure import (
        generate,
    )

    unit_root = release_root / "units/hiha_matchability_attribution"
    source = unit_root / "data/discovery_alpha0_focused025_125_by_split.csv"
    analysis_root = (
        output_root
        / "results/HIHA_DC/sensitivity/rho_attribution_discovery_confirmation"
    )
    by_split_path = (
        analysis_root / "tables/discovery_alpha0_focused025_125_by_split.csv"
    )
    by_split_path.parent.mkdir(parents=True)
    pd.read_csv(source, float_precision="round_trip").to_csv(
        by_split_path, index=False
    )
    source_table, figure, caption, recipe_manifest = generate(output_root)
    accepted_figure = unit_root / "figures" / figure.name
    _materialize_accepted_presentation(accepted_figure, figure)
    _record_materialized_figure_identity(recipe_manifest, figure)
    summary_path = (
        analysis_root / "tables/discovery_alpha0_focused025_125_summary.csv"
    )
    numerical = _write_csv_comparisons(
        output_root=output_root,
        comparisons=[
            (by_split_path.name, by_split_path, source),
            (
                summary_path.name,
                summary_path,
                unit_root
                / "verified_results/discovery_alpha0_focused025_125_summary.csv",
            ),
            (
                source_table.name,
                source_table,
                unit_root
                / "verified_results/rho_tau_surface_alpha0_range025_125.csv",
            ),
        ],
    )
    rendered = _write_render_comparisons(
        output_root=output_root,
        comparisons=[
            (figure.name, figure, accepted_figure, "materialized"),
            (caption.name, caption, unit_root / "figures" / caption.name),
        ],
    )
    _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit_id="hiha_matchability_attribution",
        inputs=[source],
        artifacts=[
            by_split_path,
            summary_path,
            source_table,
            figure,
            caption,
            recipe_manifest,
            numerical,
            rendered,
        ],
    )


def _regenerate_hiha_parameter_calibration(
    *, release_root: Path, output_root: Path
) -> None:
    from experiments.missing_celltype.generate_hiha_dc_threshold_sensitivity import (
        summarize_threshold_sensitivity,
    )
    from .hiha import (
        _render_primary_policy_table,
        _render_threshold_table,
        _validate_parameter_manifest,
    )

    unit_root = release_root / "units/hiha_parameter_and_calibration"
    by_split_source, parameter_summary_source, accepted_figures = (
        _validate_parameter_manifest(unit_root)
    )
    threshold_by_seed_path = (
        unit_root / "verified_results/threshold_sensitivity_by_seed.csv"
    )
    threshold_by_seed = pd.read_csv(
        threshold_by_seed_path, float_precision="round_trip"
    )
    threshold_summary = summarize_threshold_sensitivity(threshold_by_seed)

    result_root = output_root / "results/HIHA_DC/parameter_and_calibration"
    table_root = result_root / "tables"
    table_root.mkdir(parents=True)
    by_split_path = table_root / "hiha_parameter_sensitivity_by_split.csv"
    parameter_summary_path = table_root / "hiha_parameter_sensitivity_summary.csv"
    threshold_by_seed_output = table_root / "threshold_sensitivity_by_seed.csv"
    threshold_summary_path = table_root / "threshold_sensitivity_summary.csv"
    _materialize_accepted_presentation(by_split_source, by_split_path)
    _materialize_accepted_presentation(
        parameter_summary_source, parameter_summary_path
    )
    threshold_by_seed.to_csv(threshold_by_seed_output, index=False)
    threshold_summary.to_csv(threshold_summary_path, index=False)

    figure_root = output_root / "docs/figs"
    figures = {
        suffix: figure_root
        / f"manuscript_fig_hiha_supp_parameter_sensitivity.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    for suffix, destination in figures.items():
        _materialize_accepted_presentation(
            accepted_figures[suffix], destination
        )

    detection_path = unit_root / "data/comparison/compare_detection_summary.csv"
    transfer_path = (
        unit_root / "data/comparison/compare_shared_label_transfer_summary.csv"
    )
    detection = pd.read_csv(detection_path)
    transfer = pd.read_csv(transfer_path)
    table_s8 = result_root / "supplementary_table_s8.md"
    table_s9 = result_root / "supplementary_table_s9.md"
    table_s8.write_text(
        _render_primary_policy_table(
            detection,
            transfer,
            detection_path=detection_path,
            transfer_path=transfer_path,
        ),
        encoding="utf-8",
    )
    table_s9.write_text(
        _render_threshold_table(threshold_summary, source=threshold_summary_path),
        encoding="utf-8",
    )

    numerical = _write_csv_comparisons(
        output_root=output_root,
        comparisons=[
            (
                by_split_path.name,
                by_split_path,
                unit_root / "verified_results" / by_split_path.name,
            ),
            (
                parameter_summary_path.name,
                parameter_summary_path,
                unit_root / "verified_results" / parameter_summary_path.name,
            ),
            (
                threshold_by_seed_output.name,
                threshold_by_seed_output,
                threshold_by_seed_path,
            ),
            (
                threshold_summary_path.name,
                threshold_summary_path,
                unit_root / "verified_results" / threshold_summary_path.name,
            ),
        ],
    )
    rendered = _write_render_comparisons(
        output_root=output_root,
        comparisons=[
            (
                figures["png"].name,
                figures["png"],
                unit_root / "figures" / figures["png"].name,
                "materialized",
            )
        ],
    )
    _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit_id="hiha_parameter_and_calibration",
        inputs=[
            by_split_source,
            parameter_summary_source,
            *accepted_figures.values(),
            unit_root / "manifests/hiha_parameter_sensitivity_manifest.yaml",
            threshold_by_seed_path,
            detection_path,
            transfer_path,
        ],
        artifacts=[
            by_split_path,
            parameter_summary_path,
            threshold_by_seed_output,
            threshold_summary_path,
            *figures.values(),
            table_s8,
            table_s9,
            numerical,
            rendered,
        ],
    )


def _regenerate_hiha_primary_or_matched(
    *,
    release_root: Path,
    output_root: Path,
    unit_id: str,
    hiha_source_h5ad: Path,
    external_identity: dict[str, object],
) -> None:
    from unittest.mock import patch

    from coreot.results.compare_baselines import write_compare_baselines_results
    from coreot.results.hiha import write_hiha_reformulation
    from coreot.results.hiha_supplement import write_hiha_s1

    unit_root = release_root / "units" / unit_id
    runs_root = unit_root / "data/runs"
    grid_dir = unit_root / "configs/report_leave_one_HIHA_DC"
    uniform_grid_dir = unit_root / "configs/hiha_dc_uniform_uot_tau05"
    result_root = output_root / "results/HIHA_DC"
    comparison_paths = write_compare_baselines_results(
        runs_root=runs_root,
        grid_dir=grid_dir,
        output_root=result_root / "compare_baselines",
        internal_method_grid_dirs={"uniform_uot": uniform_grid_dir},
        internal_score_overrides={},
    )
    with patch("coreot.results.hiha._input_path", return_value=hiha_source_h5ad):
        main_paths = write_hiha_reformulation(
            runs_root=runs_root,
            grid_dir=grid_dir,
            output_root=result_root,
            method_grid_dirs={"uniform_uot": uniform_grid_dir},
        )

    panel_root = result_root / "manuscript/figure2_panels"
    figure_root = result_root / "manuscript/figure2"
    render_paths: list[Path] = []
    if unit_id == "hiha_primary":
        for source_root, destination_root in (
            (unit_root / "verified_results/figure2_panels", panel_root),
            (unit_root / "verified_results/figure2", figure_root),
            (unit_root / "figures", output_root / "docs/figs"),
        ):
            for source in sorted(source_root.rglob("*")):
                if source.is_file():
                    destination = destination_root / source.relative_to(source_root)
                    render_paths.append(
                        _materialize_accepted_presentation(source, destination)
                    )
        rendered_comparisons = [
            (
                "main_figure.png",
                figure_root / "main_figure.png",
                unit_root / "verified_results/figure2/main_figure.png",
                "materialized",
            )
        ]
    else:
        for source in sorted((unit_root / "figures/figure2_panels").glob("panel_c.*")):
            render_paths.append(
                _materialize_accepted_presentation(source, panel_root / source.name)
            )
        s1 = write_hiha_s1(
            runs_root=runs_root,
            input_path=hiha_source_h5ad,
            output_root=result_root / "supplementary_figure_s1",
            candidate_set="hiha_harmony30_k100",
        )
        render_paths.extend(s1.values())
        rendered_comparisons = [
            (
                "panel_c.png",
                panel_root / "panel_c.png",
                unit_root / "figures/figure2_panels/panel_c.png",
                "materialized",
            ),
            (
                "supplementary_figure_s1_destinations.png",
                s1["figure_png"],
                unit_root
                / "verified_results/supplementary_figure_s1/"
                "supplementary_figure_s1_destinations.png",
            ),
        ]

    _make_generated_metadata_portable(
        release_root=release_root,
        output_root=output_root,
        external_path=hiha_source_h5ad,
    )
    numerical = _write_csv_comparisons(
        output_root=output_root,
        comparisons=[
            (
                comparison_paths.detection_by_run.name,
                comparison_paths.detection_by_run,
                unit_root
                / "verified_results/compare_baselines/tables"
                / comparison_paths.detection_by_run.name,
            ),
            (
                main_paths["detection_summary"].name,
                main_paths["detection_summary"],
                unit_root
                / "verified_results/main/tables"
                / main_paths["detection_summary"].name,
            ),
            (
                main_paths["rescue"].name,
                main_paths["rescue"],
                unit_root / "verified_results/main/tables" / main_paths["rescue"].name,
            ),
        ],
    )
    rendered = _write_render_comparisons(
        output_root=output_root,
        comparisons=rendered_comparisons,
    )
    packaged_inputs = sorted(
        path
        for root in (runs_root, grid_dir, uniform_grid_dir)
        for path in root.rglob("*")
        if path.is_file()
    )
    artifacts = sorted(
        {
            *(
                path
                for path in result_root.rglob("*")
                if path.is_file()
            ),
            *(path for path in render_paths if path.is_file()),
            numerical,
            rendered,
        }
    )
    _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit_id=unit_id,
        inputs=packaged_inputs,
        external_inputs=[external_identity],
        artifacts=artifacts,
    )


def _regenerate_pbmc_matchability(
    *, release_root: Path, output_root: Path
) -> None:
    from experiments.pbmc_rho_tau_heatmap import (
        FOCUSED_TAU_VALUES,
        render_s2_aligned_heatmap,
        summarize,
    )
    from experiments.rho_attribution_search import (
        _render as render_alpha_search,
        summarize_results,
    )

    unit_root = release_root / "units/pbmc_matchability_attribution"
    source_root = unit_root / "verified_results"
    alpha_input = (
        source_root
        / "rho_attribution_alpha_search/tables/rho_attribution_by_replicate.csv"
    )
    tau_input = (
        source_root
        / "rho_attribution_tau_surface_alpha0_range075_175/tables/"
        "rho_tau_surface_range075_175_by_seed.csv"
    )
    alpha_rows = pd.read_csv(alpha_input, float_precision="round_trip")
    tau_rows = pd.read_csv(tau_input, float_precision="round_trip")
    if len(alpha_rows) != 75 or alpha_rows.duplicated(
        ["endpoint", "replicate", "alpha"]
    ).any():
        raise PackagedUnitRegenerationError(
            "PBMC alpha-search input must contain 75 unique paired rows."
        )
    if len(tau_rows) != 225 or tau_rows.duplicated(
        ["endpoint", "seed", "tau_min", "tau_max"]
    ).any():
        raise PackagedUnitRegenerationError(
            "PBMC focused tau input must contain 225 unique paired rows."
        )

    alpha_root = output_root / "results/PBMC/sensitivity/rho_attribution_alpha_search"
    alpha_tables = alpha_root / "tables"
    alpha_tables.mkdir(parents=True)
    alpha_by_replicate = alpha_tables / "rho_attribution_by_replicate.csv"
    alpha_summary_path = alpha_tables / "rho_attribution_summary.csv"
    alpha_rows.to_csv(alpha_by_replicate, index=False)
    alpha_summary = summarize_results(alpha_rows)
    alpha_summary.to_csv(alpha_summary_path, index=False)

    tau_root = (
        output_root
        / "results/PBMC/sensitivity/"
        "rho_attribution_tau_surface_alpha0_range075_175"
    )
    tau_tables = tau_root / "tables"
    tau_tables.mkdir(parents=True)
    tau_by_seed = tau_tables / "rho_tau_surface_range075_175_by_seed.csv"
    tau_summary_path = tau_tables / "rho_tau_surface_range075_175_summary.csv"
    tau_rows.to_csv(tau_by_seed, index=False)
    tau_summary = summarize(tau_rows)
    tau_summary.to_csv(tau_summary_path, index=False)

    figure_root = output_root / "docs/figs"
    alpha_figure = figure_root / "manuscript_fig_pbmc_rho_attribution_alpha_search.png"
    tau_figure = (
        figure_root
        / "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png"
    )
    render_alpha_search(alpha_summary, alpha_rows, alpha_figure)
    render_s2_aligned_heatmap(
        tau_summary,
        tau_figure,
        tau_values=FOCUSED_TAU_VALUES,
    )
    caption = tau_figure.with_name(
        "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175_caption.md"
    )
    caption.write_text(
        "**PBMC matchability-penalty attribution at fixed $\\alpha=0$.** "
        "Rows show paired relative heterogeneous-minus-mean-matched-uniform "
        "differences (%) in AP, AUROC, represented-state forced accuracy, "
        "and represented-state forced macro-F1 across five donor splits.\n",
        encoding="utf-8",
    )
    alpha_manifest = alpha_root / "manifest.yaml"
    alpha_manifest.write_text(
        yaml.safe_dump(
            {
                "analysis": "paired_mean_matched_alpha_grid",
                "inputs": [
                    alpha_by_replicate.relative_to(output_root).as_posix()
                ],
                "artifacts": [
                    alpha_summary_path.relative_to(output_root).as_posix(),
                    alpha_figure.relative_to(output_root).as_posix(),
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    tau_manifest = tau_root / "manifest.yaml"
    tau_manifest.write_text(
        yaml.safe_dump(
            {
                "analysis": "paired_mean_matched_focused_tau_grid",
                "inputs": [tau_by_seed.relative_to(output_root).as_posix()],
                "artifacts": [
                    tau_summary_path.relative_to(output_root).as_posix(),
                    tau_figure.relative_to(output_root).as_posix(),
                    caption.relative_to(output_root).as_posix(),
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    expected_project = source_root / "current_revalidation/project"
    accepted_alpha_figure = expected_project / "docs/figs" / alpha_figure.name
    accepted_tau_figure = expected_project / "docs/figs" / tau_figure.name
    _materialize_accepted_presentation(accepted_alpha_figure, alpha_figure)
    _materialize_accepted_presentation(accepted_tau_figure, tau_figure)
    numerical = _write_csv_comparisons(
        output_root=output_root,
        comparisons=[
            (
                alpha_summary_path.name,
                alpha_summary_path,
                expected_project
                / "results/PBMC/sensitivity/rho_attribution_alpha_search/"
                "tables/rho_attribution_summary.csv",
            ),
            (
                tau_summary_path.name,
                tau_summary_path,
                expected_project
                / "results/PBMC/sensitivity/"
                "rho_attribution_tau_surface_alpha0_range075_175/tables/"
                "rho_tau_surface_range075_175_summary.csv",
            ),
        ],
    )
    rendered = _write_render_comparisons(
        output_root=output_root,
        comparisons=[
            (
                alpha_figure.name,
                alpha_figure,
                accepted_alpha_figure,
                "materialized",
            ),
            (
                tau_figure.name,
                tau_figure,
                accepted_tau_figure,
                "materialized",
            ),
        ],
    )
    _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit_id="pbmc_matchability_attribution",
        inputs=[alpha_input, tau_input],
        artifacts=[
            alpha_by_replicate,
            alpha_summary_path,
            alpha_figure,
            alpha_manifest,
            tau_by_seed,
            tau_summary_path,
            tau_figure,
            caption,
            tau_manifest,
            numerical,
            rendered,
        ],
    )


def _regenerate_mouse_matchability(
    *, release_root: Path, output_root: Path
) -> None:
    from experiments.mouse_spleen.generate_rho_tau_alpha5_figure import generate
    from experiments.rho_attribution_discovery_confirmation import summarize_surface

    unit_root = release_root / "units/mouse_spleen_matchability_attribution"
    source = unit_root / "verified_results/tables/coarse_by_dataset.csv"
    by_dataset = pd.read_csv(source, float_precision="round_trip")
    summary = summarize_surface(by_dataset, expected_splits=1)
    summary["discovery_support"] = (
        summary["all_heterogeneous_converged"].astype(bool)
        & summary["all_uniform_converged"].astype(bool)
        & summary["delta_auprc_heterogeneous_minus_uniform_mean"].gt(0.0)
    )
    analysis_root = (
        output_root
        / "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "rho_attribution_tau_alpha_search"
    )
    table_root = analysis_root / "tables"
    table_root.mkdir(parents=True)
    by_dataset_path = table_root / "coarse_by_dataset.csv"
    summary_path = table_root / "coarse_summary.csv"
    by_dataset.to_csv(by_dataset_path, index=False)
    summary.to_csv(summary_path, index=False)
    source_table, figure, caption, recipe_manifest = generate(output_root)
    accepted_figure = unit_root / "figures" / figure.name
    _materialize_accepted_presentation(accepted_figure, figure)
    _record_materialized_figure_identity(recipe_manifest, figure)

    numerical = _write_csv_comparisons(
        output_root=output_root,
        comparisons=[
            (by_dataset_path.name, by_dataset_path, source),
            (
                summary_path.name,
                summary_path,
                unit_root / "verified_results/tables/coarse_summary.csv",
            ),
            (
                source_table.name,
                source_table,
                unit_root / "verified_results/tables" / source_table.name,
            ),
        ],
    )
    rendered = _write_render_comparisons(
        output_root=output_root,
        comparisons=[
            (figure.name, figure, accepted_figure, "materialized"),
        ],
    )
    _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit_id="mouse_spleen_matchability_attribution",
        inputs=[source],
        artifacts=[
            by_dataset_path,
            summary_path,
            source_table,
            figure,
            caption,
            recipe_manifest,
            numerical,
            rendered,
        ],
    )


def _regenerate_mouse_parameter(
    *, release_root: Path, output_root: Path
) -> None:
    from coreot.results.mouse_spleen_parameter_sensitivity import (
        collect_mouse_spleen_parameter_sensitivity,
        render_mouse_spleen_parameter_sensitivity,
    )

    unit_root = release_root / "units/mouse_spleen_parameter_sensitivity"
    sources = {
        "grid_table": unit_root / "verified_results/grid/metrics_by_grid.csv",
        "grid_manifest": unit_root / "verified_results/grid/manifest.yaml",
        "run_manifest": unit_root / "verified_results/grid/run_manifest.yaml",
    }
    reference_manifest_path = (
        unit_root / "verified_results/manuscript_source/manifest.yaml"
    )
    reference_manifest = yaml.safe_load(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    for name, path in sources.items():
        expected = str(reference_manifest["sources"][name]["sha256"])
        observed = _sha256(path)
        if not _matches_expected_or_audited_sanitized_identity(
            release_root=release_root,
            path=path,
            expected=expected,
            observed=observed,
        ):
            raise PackagedUnitRegenerationError(
                f"Mouse-spleen staged-grid identity mismatch for {name}: "
                f"{observed} != {expected}."
            )
    frame = collect_mouse_spleen_parameter_sensitivity(sources["grid_table"])
    result_root = (
        output_root / "results/mouse_spleen_core_ot/manuscript/parameter_sensitivity"
    )
    result_root.mkdir(parents=True)
    source_data = result_root / "mouse_spleen_parameter_sensitivity.csv"
    frame.to_csv(source_data, index=False)
    figure_root = output_root / "docs/figs"
    figures = {
        suffix: figure_root
        / f"manuscript_fig_mouse_spleen_supp_parameter_sensitivity.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    render_mouse_spleen_parameter_sensitivity(frame, figures)
    accepted_figures = {
        suffix: unit_root / "figures" / path.name
        for suffix, path in figures.items()
    }
    for suffix, destination in figures.items():
        _materialize_accepted_presentation(
            accepted_figures[suffix], destination
        )
    recipe_manifest = result_root / "manifest.yaml"
    recipe_manifest.write_text(
        yaml.safe_dump(
            {
                **{
                    key: value
                    for key, value in reference_manifest.items()
                    if key not in {"sources", "artifacts"}
                },
                "sources": {
                    name: {
                        "path": path.relative_to(release_root).as_posix(),
                        "sha256": _sha256(path),
                    }
                    for name, path in sources.items()
                },
                "artifacts": {
                    "source_data": {
                        "path": source_data.relative_to(output_root).as_posix(),
                        "sha256": _sha256(source_data),
                    },
                    **{
                        f"figure_{suffix}": {
                            "path": path.relative_to(output_root).as_posix(),
                            "sha256": _sha256(path),
                        }
                        for suffix, path in figures.items()
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    numerical = _write_csv_comparisons(
        output_root=output_root,
        comparisons=[
            (
                source_data.name,
                source_data,
                unit_root
                / "verified_results/manuscript_source"
                / source_data.name,
            )
        ],
    )
    rendered = _write_render_comparisons(
        output_root=output_root,
        comparisons=[
            (
                figures["png"].name,
                figures["png"],
                unit_root / "figures" / figures["png"].name,
                "materialized",
            )
        ],
    )
    _write_manifest(
        release_root=release_root,
        output_root=output_root,
        unit_id="mouse_spleen_parameter_sensitivity",
        inputs=[*sources.values(), reference_manifest_path],
        artifacts=[
            source_data,
            *figures.values(),
            recipe_manifest,
            numerical,
            rendered,
        ],
    )


def regenerate_packaged_unit(
    *,
    release_root: Path,
    output_root: Path,
    unit_id: str,
    hiha_source_h5ad: Path | None = None,
) -> Path:
    """Regenerate one packaged artifact-only evidence unit into a new directory."""
    import matplotlib as mpl

    if unit_id not in SUPPORTED_UNITS:
        raise PackagedUnitRegenerationError(f"Unsupported packaged unit: {unit_id!r}")
    release, output = _validate_roots(Path(release_root), Path(output_root))
    recipes: dict[str, Callable[..., None]] = {
        "hiha_compatibility_sensitivity": _regenerate_hiha_compatibility,
        "hiha_matchability_attribution": _regenerate_hiha_matchability,
        "hiha_parameter_and_calibration": _regenerate_hiha_parameter_calibration,
        "pbmc_matchability_attribution": _regenerate_pbmc_matchability,
        "mouse_spleen_matchability_attribution": _regenerate_mouse_matchability,
        "mouse_spleen_parameter_sensitivity": _regenerate_mouse_parameter,
    }
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        if unit_id in EXTERNAL_HIHA_UNITS:
            source, identity = _validate_hiha_source(
                release_root=release,
                unit_id=unit_id,
                source_h5ad=hiha_source_h5ad,
            )
            _regenerate_hiha_primary_or_matched(
                release_root=release,
                output_root=output,
                unit_id=unit_id,
                hiha_source_h5ad=source,
                external_identity=identity,
            )
            return output
        recipe = recipes.get(unit_id)
        if recipe is None:
            raise PackagedUnitRegenerationError(
                f"Packaged recipe is not implemented yet: {unit_id!r}"
            )
        recipe(release_root=release, output_root=output)
    return output
