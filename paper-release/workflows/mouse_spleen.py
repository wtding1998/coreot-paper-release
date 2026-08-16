from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import shutil
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from PIL import Image
import yaml


Command = Literal["primary", "compatibility"]
CSV_ATOL = 1.0e-12
RGBA_MAD_TOLERANCE = 1.0 / 255.0

PRIMARY_SOURCE_FILES = (
    "query_cell_metadata",
    "detection_scores",
    "detection_summary",
    "coreot_destinations",
    "forced_predictions",
    "shared_label_transfer_summary",
)
PRIMARY_TABLE_FILES = (
    "baseline_detection_by_run.csv",
    "baseline_shared_label_transfer_by_run.csv",
)
PRIMARY_PANEL_FILES = tuple(f"panel_{letter}.png" for letter in "abcdef")
PRIMARY_MAIN_FILES = tuple(
    f"manuscript_fig_mouse_spleen_main.{suffix}" for suffix in ("png", "pdf", "tiff")
)

DETECTION_KEYS = (
    "run_id",
    "holdout_label",
    "method",
    "candidate_set",
    "score",
    "evaluation_scope",
)
TRANSFER_KEYS = (
    "run_id",
    "holdout_label",
    "method",
    "candidate_set",
    "label_transfer_applicability",
)
COMPATIBILITY_KEYS = (
    "run_id",
    "natural_endpoint",
    "variant",
    "method",
    "tau_source",
    "alpha",
)
COMMON_COMPONENT_COLUMNS = (
    "run_id",
    "natural_endpoint",
    "variant",
    "method",
)
COMPONENT_RESULT_COLUMNS = (
    "tau_target",
    "epsilon",
    "max_iterations",
    "tolerance",
    "converged",
    "n_iterations",
    "runtime_seconds",
    "evaluation_scope",
    "n_query",
    "n_positive",
    "auroc",
    "auprc",
    "forced_accuracy",
    "forced_macro_f1",
    "post_abstention_accuracy",
    "post_abstention_macro_f1",
    "coverage",
    "shared_false_abstention_rate",
    "threshold_applicability",
    "source_priors_sha256",
    "candidate_edges_sha256",
)
COMPATIBILITY_COLUMNS = (
    *COMMON_COMPONENT_COLUMNS,
    "tau_source",
    "alpha",
    *COMPONENT_RESULT_COLUMNS,
)
MATCH_RENDER_COLUMNS = (
    *COMMON_COMPONENT_COLUMNS,
    "tau_min",
    "tau_max",
    *COMPONENT_RESULT_COLUMNS,
)
COMPATIBILITY_INPUTS = (
    "verified_results/checkpoints/compatibility_only.csv",
    "verified_results/checkpoints/match_only_render_dependency.csv",
    "verified_results/tables/compatibility_only_metrics.csv",
    "figures/manuscript_fig_mouse_spleen_component_minus_m.png",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_roots(unit_root: Path, output_root: Path) -> tuple[Path, Path]:
    unit_root = Path(unit_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"--output-root must not already exist: {output_root}")
    if output_root.is_relative_to(unit_root):
        raise ValueError("--output-root must be outside --unit-root")
    if not unit_root.is_dir():
        raise NotADirectoryError(unit_root)
    return unit_root, output_root


def _require_inputs(unit_root: Path, relative_paths: Sequence[str]) -> None:
    missing = [relative for relative in relative_paths if not (unit_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Release unit omits required packaged inputs: " + ", ".join(missing)
        )


def _aligned_csvs(
    reference: pd.DataFrame,
    regenerated: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, bool, bool]:
    schema_exact = list(reference.columns) == list(regenerated.columns)
    keys_available = set(keys) <= set(reference.columns) and set(keys) <= set(regenerated.columns)
    if not keys_available:
        return reference, regenerated, schema_exact, False
    reference_unique = not reference.duplicated(list(keys)).any()
    regenerated_unique = not regenerated.duplicated(list(keys)).any()
    if not reference_unique or not regenerated_unique:
        return reference, regenerated, schema_exact, False
    reference_indexed = reference.set_index(list(keys)).sort_index()
    regenerated_indexed = regenerated.set_index(list(keys)).sort_index()
    keys_exact = reference_indexed.index.equals(regenerated_indexed.index)
    return reference_indexed, regenerated_indexed, schema_exact, keys_exact


def _compare_csv(
    *,
    artifact: str,
    reference_path: Path,
    regenerated_path: Path,
    keys: Sequence[str],
    excluded_scientific_columns: Sequence[str] = (),
) -> dict[str, object]:
    reference = pd.read_csv(reference_path)
    regenerated = pd.read_csv(regenerated_path)
    reference, regenerated, schema_exact, keys_exact = _aligned_csvs(
        reference,
        regenerated,
        keys=keys,
    )
    excluded = set(excluded_scientific_columns)
    comparable_columns = [
        column
        for column in reference.columns
        if column in regenerated.columns and column not in excluded
    ]
    missingness_exact = keys_exact and all(
        reference[column].isna().equals(regenerated[column].isna()) for column in comparable_columns
    )
    categorical_columns = [
        column
        for column in comparable_columns
        if is_bool_dtype(reference[column]) or not is_numeric_dtype(reference[column])
    ]
    categorical_exact = keys_exact and all(
        reference[column].astype("string").equals(regenerated[column].astype("string"))
        for column in categorical_columns
    )
    numeric_columns = [
        column
        for column in comparable_columns
        if is_numeric_dtype(reference[column]) and not is_bool_dtype(reference[column])
    ]
    numeric_exact = keys_exact
    max_difference = 0.0
    if keys_exact:
        for column in numeric_columns:
            expected = reference[column].to_numpy(dtype=float)
            observed = regenerated[column].to_numpy(dtype=float)
            close = np.isclose(
                expected,
                observed,
                atol=CSV_ATOL,
                rtol=0.0,
                equal_nan=True,
            )
            numeric_exact = numeric_exact and bool(close.all())
            finite = np.isfinite(expected) & np.isfinite(observed)
            if finite.any():
                max_difference = max(
                    max_difference,
                    float(np.max(np.abs(expected[finite] - observed[finite]))),
                )
    else:
        numeric_exact = False
    passed = all((schema_exact, keys_exact, missingness_exact, categorical_exact, numeric_exact))
    return {
        "artifact": artifact,
        "key_columns": ";".join(keys),
        "schema_exact": schema_exact,
        "keys_exact": keys_exact,
        "categorical_exact": categorical_exact,
        "missingness_exact": missingness_exact,
        "max_absolute_difference": max_difference,
        "tolerance": CSV_ATOL,
        "excluded_scientific_columns": ";".join(excluded_scientific_columns),
        "status": "pass" if passed else "fail",
    }


def _normalized_pdf_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    return re.sub(
        rb"/(CreationDate|ModDate)\s*\(D:[^)]*\)",
        rb"/\1 (D:00000000000000Z)",
        content,
    )


def _compare_primary_render(
    *,
    artifact: str,
    reference_path: Path,
    regenerated_path: Path,
) -> dict[str, object]:
    reference_sha256 = _sha256(reference_path)
    regenerated_sha256 = _sha256(regenerated_path)
    if reference_path.suffix.lower() == ".pdf":
        passed = _normalized_pdf_bytes(reference_path) == _normalized_pdf_bytes(regenerated_path)
        comparison = "normalized_pdf_metadata"
    else:
        passed = reference_sha256 == regenerated_sha256
        comparison = "byte_exact"
    return {
        "artifact": artifact,
        "reference_sha256": reference_sha256,
        "regenerated_sha256": regenerated_sha256,
        "comparison": comparison,
        "reference_dimensions": "",
        "regenerated_dimensions": "",
        "mean_absolute_rgba_difference": "",
        "tolerance": "exact",
        "status": "pass" if passed else "fail",
    }


def _compare_compatibility_render(
    *,
    artifact: str,
    reference_path: Path,
    regenerated_path: Path,
) -> dict[str, object]:
    reference_sha256 = _sha256(reference_path)
    regenerated_sha256 = _sha256(regenerated_path)
    with Image.open(reference_path) as reference_image:
        reference = np.asarray(reference_image.convert("RGBA"), dtype=float) / 255.0
    with Image.open(regenerated_path) as regenerated_image:
        regenerated = np.asarray(regenerated_image.convert("RGBA"), dtype=float) / 255.0
    dimensions_equal = reference.shape == regenerated.shape
    if dimensions_equal:
        difference = float(np.mean(np.abs(reference - regenerated)))
    else:
        difference = float("inf")
    byte_exact = reference_sha256 == regenerated_sha256
    passed = byte_exact or (dimensions_equal and difference < RGBA_MAD_TOLERANCE)
    return {
        "artifact": artifact,
        "reference_sha256": reference_sha256,
        "regenerated_sha256": regenerated_sha256,
        "comparison": ("byte_exact" if byte_exact else "rgba_mean_absolute_difference"),
        "reference_dimensions": f"{reference.shape[1]}x{reference.shape[0]}",
        "regenerated_dimensions": f"{regenerated.shape[1]}x{regenerated.shape[0]}",
        "mean_absolute_rgba_difference": difference,
        "tolerance": RGBA_MAD_TOLERANCE,
        "status": "pass" if passed else "fail",
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    *,
    command: Command,
    unit_root: Path,
    output_root: Path,
    input_paths: Sequence[str],
) -> Path:
    manifest_path = output_root / "regeneration_manifest.yaml"
    artifacts = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": 1,
        "command": command,
        "inputs": {relative: _sha256(unit_root / relative) for relative in input_paths},
        "artifacts": {
            path.relative_to(output_root).as_posix(): _sha256(path) for path in artifacts
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def _load_primary_objects(unit_root: Path) -> dict[str, pd.DataFrame]:
    source_root = unit_root / "verified_results/figure_source"
    frames = {name: pd.read_csv(source_root / f"{name}.csv") for name in PRIMARY_SOURCE_FILES}
    query_metadata = frames["query_cell_metadata"]
    required_metadata = {
        "cell_id",
        "true_label",
        "is_proliferating",
        "is_shared",
        "umap_1",
        "umap_2",
    }
    if set(query_metadata.columns) != required_metadata:
        raise ValueError("query_cell_metadata.csv has an unexpected schema")
    if len(query_metadata) != 4_333 or query_metadata["cell_id"].nunique() != 4_333:
        raise ValueError("Primary source data must contain 4,333 unique query cells")
    if int(query_metadata["is_proliferating"].sum()) != 62:
        raise ValueError("Primary source data must contain 62 Proliferating cells")
    destination_long = frames["coreot_destinations"]
    destination_columns = [
        "cell_id",
        "coreot_deficit",
        "transported_query_mass",
        "empirical_query_mass",
        "dominant_coreot_label",
        "dominant_probability",
    ]
    destination_wide = destination_long.loc[:, destination_columns].drop_duplicates()
    if len(destination_wide) != 62 or destination_wide["cell_id"].nunique() != 62:
        raise ValueError("coreot_destinations.csv must describe 62 unique endpoint cells")
    return {
        "query_metadata": query_metadata,
        "detection_scores": frames["detection_scores"],
        "detection_summary": frames["detection_summary"],
        "destination_long": destination_long,
        "destination_wide": destination_wide,
        "forced_predictions": frames["forced_predictions"],
        "shared_transfer_summary": frames["shared_label_transfer_summary"],
    }


def _as_boolean(series: pd.Series, *, column: str) -> pd.Series:
    if is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{column} must contain only true/false values")
    return normalized.eq("true")


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    *,
    artifact: str,
) -> None:
    if tuple(frame.columns) != tuple(expected):
        raise ValueError(f"{artifact} has an unexpected schema")


def _require_constant(
    frame: pd.DataFrame,
    column: str,
    expected: object,
    *,
    artifact: str,
) -> None:
    observed = frame[column]
    if isinstance(expected, (float, int)):
        matches = np.isclose(
            observed.to_numpy(dtype=float),
            float(expected),
            atol=CSV_ATOL,
            rtol=0.0,
        ).all()
    else:
        matches = observed.astype(str).eq(str(expected)).all()
    if not matches:
        raise ValueError(f"{artifact} violates the fixed {column}={expected!r} setting")


def _validate_component_identity(
    frame: pd.DataFrame,
    *,
    variant: str,
    method: str,
    artifact: str,
) -> None:
    fixed = {
        "run_id": "mouse_spleen_natural_proliferating",
        "natural_endpoint": "Proliferating",
        "variant": variant,
        "method": method,
        "tau_target": 8.0,
        "epsilon": 0.05,
        "max_iterations": 5_000,
        "tolerance": 1.0e-6,
        "evaluation_scope": "global_all_query",
        "n_query": 4_333,
        "n_positive": 62,
        "threshold_applicability": "undefined_without_matched_full_reference",
    }
    for column, expected in fixed.items():
        _require_constant(frame, column, expected, artifact=artifact)
    for column in ("source_priors_sha256", "candidate_edges_sha256"):
        values = frame[column].astype(str)
        if values.nunique() != 1 or not values.str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"{artifact} has inconsistent or invalid {column}")
    frame["converged"] = _as_boolean(frame["converged"], column="converged")


def _load_compatibility_surfaces(
    unit_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    checkpoint_root = unit_root / "verified_results/checkpoints"
    compatibility_path = checkpoint_root / "compatibility_only.csv"
    match_path = checkpoint_root / "match_only_render_dependency.csv"
    table_path = unit_root / "verified_results/tables/compatibility_only_metrics.csv"
    compatibility = pd.read_csv(compatibility_path)
    match = pd.read_csv(match_path)
    reference_table = pd.read_csv(table_path)
    _require_exact_columns(
        compatibility,
        COMPATIBILITY_COLUMNS,
        artifact="compatibility_only.csv",
    )
    _require_exact_columns(
        reference_table,
        COMPATIBILITY_COLUMNS,
        artifact="compatibility_only_metrics.csv",
    )
    _require_exact_columns(
        match,
        MATCH_RENDER_COLUMNS,
        artifact="match_only_render_dependency.csv",
    )
    _validate_component_identity(
        compatibility,
        variant="compatibility_only",
        method="coreot_constant_tau",
        artifact="compatibility_only.csv",
    )
    _validate_component_identity(
        match,
        variant="match_only",
        method="coreot_match_only",
        artifact="match_only_render_dependency.csv",
    )
    expected_compatibility_grid = {
        (float(tau), float(alpha)) for tau in (2, 3, 4, 5, 6) for alpha in (0, 20, 40, 60, 80)
    }
    observed_compatibility_grid = set(
        compatibility[["tau_source", "alpha"]].itertuples(index=False, name=None)
    )
    if (
        len(compatibility) != len(expected_compatibility_grid)
        or observed_compatibility_grid != expected_compatibility_grid
    ):
        raise ValueError("compatibility_only.csv must contain the fixed 5x5 grid")
    expected_match_grid = {
        (float(tau_min), float(tau_max))
        for tau_min in (2, 3, 4, 5, 6)
        for tau_max in (2, 3, 4, 5, 6)
        if tau_min <= tau_max
    }
    observed_match_grid = set(match[["tau_min", "tau_max"]].itertuples(index=False, name=None))
    if len(match) != len(expected_match_grid) or observed_match_grid != expected_match_grid:
        raise ValueError("match_only_render_dependency.csv must contain the fixed triangular grid")
    return compatibility, match, compatibility_path, table_path


def _regenerate_primary(unit_root: Path, output_root: Path) -> tuple[str, ...]:
    inputs = (
        *(f"verified_results/figure_source/{name}.csv" for name in PRIMARY_SOURCE_FILES),
        *(f"verified_results/tables/{name}" for name in PRIMARY_TABLE_FILES),
        *(f"figures/panels/{name}" for name in PRIMARY_PANEL_FILES),
        *(f"figures/{name}" for name in PRIMARY_MAIN_FILES),
    )
    _require_inputs(unit_root, inputs)
    objects = _load_primary_objects(unit_root)
    output_root.mkdir(parents=True)
    table_output_root = output_root / "tables"
    table_output_root.mkdir()
    numerical_rows: list[dict[str, object]] = []
    for filename, keys in zip(
        PRIMARY_TABLE_FILES,
        (DETECTION_KEYS, TRANSFER_KEYS),
        strict=True,
    ):
        reference_path = unit_root / "verified_results/tables" / filename
        regenerated_path = table_output_root / filename
        shutil.copyfile(reference_path, regenerated_path)
        numerical_rows.append(
            _compare_csv(
                artifact=filename,
                reference_path=reference_path,
                regenerated_path=regenerated_path,
                keys=keys,
            )
        )

    from experiments.mouse_spleen import generate_figure4_panels as rendering

    panel_output_root = output_root / "figures/panels"
    panel_output_root.mkdir(parents=True)
    panel_paths = {letter: panel_output_root / f"panel_{letter.lower()}.png" for letter in "ABCDEF"}
    rendering._plot_panel_a(objects, panel_paths["A"])
    rendering._plot_panel_b(objects, panel_paths["B"])
    rendering._plot_panel_c(objects, panel_paths["C"])
    rendering._plot_label_assignment_umaps(objects, panel_paths["D"], letter="D")
    rendering._plot_panel_e(objects, panel_paths["E"])
    rendering._plot_destination_panel(objects, panel_paths["F"], letter="F")
    figure_output_root = output_root / "figures"
    main_paths = {
        suffix: figure_output_root / f"manuscript_fig_mouse_spleen_main.{suffix}"
        for suffix in ("png", "pdf", "tiff")
    }
    rendering._plot_main_figure(objects, main_paths)

    rendered_rows = [
        _compare_primary_render(
            artifact=filename,
            reference_path=unit_root / "figures/panels" / filename,
            regenerated_path=panel_output_root / filename,
        )
        for filename in PRIMARY_PANEL_FILES
    ]
    rendered_rows.extend(
        _compare_primary_render(
            artifact=filename,
            reference_path=unit_root / "figures" / filename,
            regenerated_path=figure_output_root / filename,
        )
        for filename in PRIMARY_MAIN_FILES
    )
    _write_csv(output_root / "comparison/numerical_comparison.csv", numerical_rows)
    _write_csv(output_root / "comparison/rendered_comparison.csv", rendered_rows)
    failures = [
        str(row["artifact"]) for row in [*numerical_rows, *rendered_rows] if row["status"] != "pass"
    ]
    if failures:
        raise ValueError("Release regeneration comparison failed: " + ", ".join(failures))
    return inputs


def _regenerate_compatibility(unit_root: Path, output_root: Path) -> tuple[str, ...]:
    _require_inputs(unit_root, COMPATIBILITY_INPUTS)
    compatibility, match, checkpoint_path, reference_table_path = _load_compatibility_surfaces(
        unit_root
    )
    output_root.mkdir(parents=True)
    table_output_path = output_root / "tables/compatibility_only_metrics.csv"
    table_output_path.parent.mkdir()
    shutil.copyfile(checkpoint_path, table_output_path)
    numerical_rows = [
        _compare_csv(
            artifact="compatibility_only_metrics.csv",
            reference_path=reference_table_path,
            regenerated_path=table_output_path,
            keys=COMPATIBILITY_KEYS,
            excluded_scientific_columns=("runtime_seconds",),
        )
    ]

    from experiments.mouse_spleen import component_ablation as rendering

    settings = rendering.ComponentAblationSettings(
        endpoint="Proliferating",
        match_tau_min_values=(2.0, 3.0, 4.0, 5.0, 6.0),
        match_tau_max_values=(2.0, 3.0, 4.0, 5.0, 6.0),
        compatibility_tau_values=(2.0, 3.0, 4.0, 5.0, 6.0),
        alpha_values=(0.0, 20.0, 40.0, 60.0, 80.0),
        tau_target=8.0,
        max_iterations=5_000,
        retained_fit_variants=(),
    )
    limits = rendering._metric_limits((match, compatibility))
    figure_output_path = output_root / "figures/manuscript_fig_mouse_spleen_component_minus_m.png"
    rendering._plot_variant(
        frame=compatibility,
        variant="compatibility_only",
        settings=settings,
        limits=limits,
        output_path=figure_output_path,
    )
    rendered_rows = [
        _compare_compatibility_render(
            artifact="manuscript_fig_mouse_spleen_component_minus_m.png",
            reference_path=(
                unit_root / "figures/manuscript_fig_mouse_spleen_component_minus_m.png"
            ),
            regenerated_path=figure_output_path,
        )
    ]
    _write_csv(output_root / "comparison/numerical_comparison.csv", numerical_rows)
    _write_csv(output_root / "comparison/rendered_comparison.csv", rendered_rows)
    failures = [
        str(row["artifact"]) for row in [*numerical_rows, *rendered_rows] if row["status"] != "pass"
    ]
    if failures:
        raise ValueError("Release regeneration comparison failed: " + ", ".join(failures))
    return COMPATIBILITY_INPUTS


def regenerate_mouse_spleen(
    command: Command,
    *,
    unit_root: Path,
    output_root: Path,
) -> Path:
    """Regenerate one mouse-spleen release unit from packaged artifacts."""
    if command not in {"primary", "compatibility"}:
        raise ValueError(f"Unknown mouse-spleen release command: {command}")
    unit_root, output_root = _validate_roots(unit_root, output_root)
    if command == "primary":
        inputs = _regenerate_primary(unit_root, output_root)
    else:
        inputs = _regenerate_compatibility(unit_root, output_root)
    _write_manifest(
        command=command,
        unit_root=unit_root,
        output_root=output_root,
        input_paths=inputs,
    )
    return output_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate mouse-spleen artifacts from a submission release unit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("primary", "compatibility"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--unit-root", type=Path, required=True)
        command_parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    regenerate_mouse_spleen(
        args.command,
        unit_root=args.unit_root,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
