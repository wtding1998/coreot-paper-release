"""Regenerate the cross-analysis convergence and environment evidence unit."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import yaml

from .release_validation import validate_submission_release


CSV_ATOL = 1.0e-12
UNAVAILABLE = "per_fit_convergence_record_unavailable"
CONVERGENCE_FILES = (
    "retained_analysis_coverage.csv",
    "transport_convergence_by_fit.csv",
    "transport_convergence_summary.csv",
)
ENVIRONMENT_CSV_FILES = (
    "external_software_and_models.csv",
    "python_packages.csv",
    "r_packages.csv",
)
RESOURCE_DATASET_FILES = (
    "HIHA_DC_compare_baselines_by_method.csv",
    "PBMC_compare_baselines_by_method.csv",
    "mouse_spleen_core_ot_compare_baselines_by_method.csv",
)
RESOURCE_DERIVED_FILES = (
    "compare_baselines_by_method.csv",
    "resource_measurement_status_summary.csv",
)


class ConvergenceReleaseError(ValueError):
    """Raised when packaged convergence evidence violates its release contract."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output(release: Path, output_root: str | Path) -> Path:
    raw = Path(output_root)
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"Output root must not already exist: {raw}")
    output = raw.resolve()
    if output == release or output.is_relative_to(release):
        raise ConvergenceReleaseError(
            f"Output root must be outside the read-only release: {output}"
        )
    return output


def _validate_complete_fit_fields(
    by_fit: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    if not coverage["n_without_per_fit_record"].astype(int).eq(0).all():
        return
    required = {
        "experiment",
        "analysis_family",
        "endpoint",
        "seed",
        "condition",
        "run_id",
        "method",
        "convergence_evidence",
        "converged",
        "n_iter",
        "max_iter",
        "tol",
        "source_artifact",
    }
    missing = sorted(required - set(by_fit.columns))
    if missing:
        raise ConvergenceReleaseError(
            f"Complete convergence evidence omits required fields {missing}."
        )
    numeric = by_fit.loc[:, ["n_iter", "max_iter", "tol"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        not by_fit["convergence_evidence"].eq("verified_converged").all()
        or not by_fit["converged"].astype(str).str.casefold().eq("true").all()
        or not numeric.notna().all().all()
        or not numeric["n_iter"].gt(0).all()
        or not numeric["max_iter"].gt(0).all()
        or not numeric["n_iter"].le(numeric["max_iter"]).all()
        or not numeric["tol"].gt(0).all()
        or by_fit["source_artifact"].isna().any()
        or by_fit["source_artifact"].astype(str).str.strip().eq("").any()
    ):
        raise ConvergenceReleaseError(
            "Complete convergence evidence contains an invalid retained fit record."
        )


def _recompute_convergence(reference_root: Path, output_root: Path) -> None:
    source_by_fit = reference_root / "transport_convergence_by_fit.csv"
    output_by_fit = output_root / "transport_convergence_by_fit.csv"
    shutil.copyfile(source_by_fit, output_by_fit)
    by_fit = pd.read_csv(source_by_fit)

    summary = (
        by_fit.groupby(
            ["experiment", "analysis_family", "convergence_evidence"],
            sort=True,
            dropna=False,
        )
        .size()
        .rename("n_fits")
        .reset_index()
    )
    summary.to_csv(output_root / "transport_convergence_summary.csv", index=False)

    reference_coverage = pd.read_csv(reference_root / "retained_analysis_coverage.csv")
    _validate_complete_fit_fields(by_fit, reference_coverage)
    counts = (
        by_fit.assign(_without=by_fit["convergence_evidence"].eq(UNAVAILABLE))
        .groupby(["experiment", "analysis_family"], sort=False, dropna=False)
        .agg(n_indexed_fits=("run_id", "size"), n_without_per_fit_record=("_without", "sum"))
        .reset_index()
    )
    counts["n_without_per_fit_record"] = counts[
        "n_without_per_fit_record"
    ].astype(int)
    counts["n_with_per_fit_record"] = (
        counts["n_indexed_fits"] - counts["n_without_per_fit_record"]
    )
    counts["coverage_status"] = np.where(
        counts["n_without_per_fit_record"].eq(0),
        "indexed_with_complete_per_fit_records",
        "indexed_with_record_gaps",
    )
    descriptive = reference_coverage.loc[
        :,
        [
            "experiment",
            "analysis_family",
            "manuscript_scope",
            "convergence_evidence_source",
        ],
    ]
    coverage = descriptive.merge(
        counts,
        on=["experiment", "analysis_family"],
        how="left",
        validate="one_to_one",
    )
    coverage = coverage.loc[:, reference_coverage.columns]
    coverage.to_csv(output_root / "retained_analysis_coverage.csv", index=False)


def _recompute_resources(reference_root: Path, output_root: Path) -> None:
    frames_by_dataset: dict[str, pd.DataFrame] = {}
    for filename in RESOURCE_DATASET_FILES:
        source = reference_root / filename
        shutil.copyfile(source, output_root / filename)
        frame = pd.read_csv(source)
        datasets = frame["dataset"].drop_duplicates().astype(str).tolist()
        if len(datasets) != 1 or datasets[0] in frames_by_dataset:
            raise ConvergenceReleaseError(
                f"Resource input {filename} must contain one unique dataset."
            )
        frames_by_dataset[datasets[0]] = frame
    reference_combined = pd.read_csv(reference_root / "compare_baselines_by_method.csv")
    dataset_order = (
        reference_combined["dataset"].drop_duplicates().astype(str).tolist()
    )
    if set(dataset_order) != set(frames_by_dataset):
        raise ConvergenceReleaseError(
            "Resource aggregate dataset membership differs from its dataset inputs."
        )
    combined = pd.concat(
        [frames_by_dataset[dataset] for dataset in dataset_order],
        ignore_index=True,
    )
    combined.to_csv(output_root / "compare_baselines_by_method.csv", index=False)
    summary = (
        combined.groupby(["dataset", "measurement_status"], sort=True, dropna=False)
        .size()
        .rename("n_methods")
        .reset_index()
    )
    summary.to_csv(output_root / "resource_measurement_status_summary.csv", index=False)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _write_current_runtime(path: Path) -> None:
    payload = {
        "scope": "artifact-only clean-room regeneration",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_basename": Path(sys.executable).name,
        "packages": {
            name: _package_version(name)
            for name in ("coreot", "numpy", "pandas", "scipy", "matplotlib", "Pillow")
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _compare_csv(artifact: str, reference: Path, regenerated: Path) -> dict[str, object]:
    expected = pd.read_csv(reference)
    observed = pd.read_csv(regenerated)
    schema_exact = list(expected.columns) == list(observed.columns)
    rows_exact = len(expected) == len(observed)
    missingness_exact = schema_exact and rows_exact and expected.isna().equals(observed.isna())
    categorical_exact = schema_exact and rows_exact
    maximum = 0.0
    if schema_exact and rows_exact:
        for column in expected.columns:
            if (
                is_numeric_dtype(expected[column])
                and is_numeric_dtype(observed[column])
                and not is_bool_dtype(expected[column])
            ):
                left = expected[column].to_numpy(dtype=float)
                right = observed[column].to_numpy(dtype=float)
                finite = np.isfinite(left) & np.isfinite(right)
                if finite.any():
                    maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
            else:
                categorical_exact = categorical_exact and expected[column].astype(
                    "string"
                ).equals(observed[column].astype("string"))
    else:
        categorical_exact = False
        maximum = float("inf")
    passed = all(
        (
            schema_exact,
            rows_exact,
            missingness_exact,
            categorical_exact,
            maximum <= CSV_ATOL,
        )
    )
    return {
        "artifact": artifact,
        "rows": len(observed),
        "schema_exact": schema_exact,
        "missingness_exact": missingness_exact,
        "categorical_exact": categorical_exact,
        "max_absolute_difference": maximum,
        "tolerance": CSV_ATOL,
        "status": "pass" if passed else "fail",
    }


def regenerate_convergence_unit(
    *, release_root: str | Path, output_root: str | Path
) -> Path:
    """Recompute derived convergence/resource audits from the verified release."""

    release = Path(release_root).resolve()
    try:
        validate_submission_release(release)
    except Exception as exc:
        raise ConvergenceReleaseError(f"Invalid submission release: {release}") from exc
    output = _validate_output(release, output_root)
    unit_root = release / "units/convergence_and_environment/verified_results"
    reference_convergence = unit_root / "convergence"
    reference_environment = unit_root / "environment"
    reference_resources = unit_root / "resources"

    for path in (reference_convergence, reference_environment, reference_resources):
        if not path.is_dir():
            raise ConvergenceReleaseError(f"Release omits convergence input root: {path}")

    convergence_output = output / "convergence"
    environment_output = output / "environment"
    resources_output = output / "resources"
    comparison_output = output / "comparison"
    for path in (
        convergence_output,
        environment_output,
        resources_output,
        comparison_output,
    ):
        path.mkdir(parents=True, exist_ok=True)

    _recompute_convergence(reference_convergence, convergence_output)
    for filename in ENVIRONMENT_CSV_FILES:
        shutil.copyfile(
            reference_environment / filename,
            environment_output / filename,
        )
    shutil.copyfile(
        reference_environment / "runtime.yaml",
        environment_output / "retained_runtime.yaml",
    )
    _write_current_runtime(environment_output / "current_runtime.yaml")
    _recompute_resources(reference_resources, resources_output)

    comparisons: list[dict[str, object]] = []
    for group, filenames in (
        ("convergence", CONVERGENCE_FILES),
        ("environment", ENVIRONMENT_CSV_FILES),
        ("resources", (*RESOURCE_DATASET_FILES, *RESOURCE_DERIVED_FILES)),
    ):
        reference_group = unit_root / group
        output_group = output / group
        comparisons.extend(
            _compare_csv(
                f"{group}/{filename}",
                reference_group / filename,
                output_group / filename,
            )
            for filename in filenames
        )
    comparison_path = comparison_output / "numerical_comparison.csv"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    failures = [row["artifact"] for row in comparisons if row["status"] != "pass"]
    if failures:
        raise ConvergenceReleaseError(
            "Convergence regeneration comparison failed: " + ", ".join(failures)
        )

    source_paths = (
        *(reference_convergence / filename for filename in CONVERGENCE_FILES),
        *(reference_environment / filename for filename in ENVIRONMENT_CSV_FILES),
        reference_environment / "runtime.yaml",
        *(reference_resources / filename for filename in RESOURCE_DATASET_FILES),
        *(reference_resources / filename for filename in RESOURCE_DERIVED_FILES),
    )
    manifest_path = output / "regeneration_manifest.yaml"
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "unit_id": "convergence_and_environment",
        "regeneration_level": "aggregate and environment audit",
        "inputs": {
            path.relative_to(release).as_posix(): _sha256(path) for path in source_paths
        },
        "artifacts": {
            path.relative_to(output).as_posix(): _sha256(path) for path in artifacts
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return output
