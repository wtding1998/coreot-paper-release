from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yaml

MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from experiments.generate_manuscript_convergence_audit import generate_audit  # noqa: E402


UNIT_ID = "convergence_and_environment"
RESOURCE_COLUMNS = (
    "dataset",
    "evidence_unit",
    "method",
    "condition",
    "seed_or_split",
    "stage",
    "measurement_status",
    "wall_seconds",
    "peak_rss_bytes",
    "peak_rss_gib",
    "shared_input_bytes",
    "retained_output_bytes",
    "command",
    "measurement_tool",
    "measurement_scope",
    "parallel_jobs",
    "software_environment",
    "raw_log_path",
    "notes",
)
CURRENT_RESOURCE_REQUIRED_COLUMNS = frozenset(
    {
        "experiment",
        "evidence_unit",
        "method",
        "condition",
        "endpoint",
        "seed",
        "repetition",
        "verification_status",
        "wall_seconds",
        "peak_rss_bytes",
        "peak_rss_gib",
        "resource_tool",
        "raw_log_path",
        "environment_identity",
    }
)
CURRENT_RESOURCE_DATASET_NAMES = {
    "hiha": "HIHA_DC",
    "pbmc": "PBMC",
    "mouse_spleen": "mouse_spleen_core_ot",
}
R_PACKAGES = (
    "Seurat",
    "SingleR",
    "scmap",
    "CHETAH",
    "Matrix",
    "SingleCellExperiment",
    "SummarizedExperiment",
)
RETAINED_FIT_COLUMNS = (
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
)
PERSONAL_HOME_PREFIX = re.compile(r"(?:/Users|/home)/[^/]+/")


@dataclass(frozen=True)
class IncludedFile:
    package_path: str
    source_path: str
    action: str
    source_bytes: int
    package_bytes: int
    source_sha256: str
    package_sha256: str
    note: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    path.mkdir(parents=True)


def _validate_retained_fit_evidence(convergence_root: Path) -> None:
    by_fit_path = convergence_root / "transport_convergence_by_fit.csv"
    coverage_path = convergence_root / "retained_analysis_coverage.csv"
    by_fit = pd.read_csv(by_fit_path)
    coverage = pd.read_csv(coverage_path)
    missing = sorted(set(RETAINED_FIT_COLUMNS) - set(by_fit.columns))
    if missing:
        raise ValueError(f"Retained fit evidence omits required columns {missing}: {by_fit_path}")
    if (
        by_fit.empty
        or by_fit.duplicated(
            [
                "experiment",
                "analysis_family",
                "endpoint",
                "seed",
                "condition",
                "run_id",
                "method",
            ]
        ).any()
    ):
        raise ValueError("Retained fit evidence is empty or has duplicate fit keys.")
    unavailable = by_fit["convergence_evidence"].eq("per_fit_convergence_record_unavailable")
    nonconverged = by_fit["convergence_evidence"].eq("verified_nonconverged")
    if unavailable.any() or nonconverged.any():
        raise ValueError(
            "Cannot build the convergence release unit while retained fit evidence "
            f"has unavailable={int(unavailable.sum())} or "
            f"nonconverged={int(nonconverged.sum())} rows."
        )
    if not by_fit["convergence_evidence"].eq("verified_converged").all():
        unexpected = sorted(set(by_fit["convergence_evidence"].astype(str)))
        raise ValueError(f"Unexpected retained convergence evidence: {unexpected}")
    converged = by_fit["converged"].astype(str).str.casefold().eq("true")
    numeric = by_fit.loc[:, ["n_iter", "max_iter", "tol"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        not converged.all()
        or not numeric.notna().all().all()
        or not numeric["n_iter"].gt(0).all()
        or not numeric["max_iter"].gt(0).all()
        or not numeric["n_iter"].le(numeric["max_iter"]).all()
        or not numeric["tol"].gt(0).all()
        or by_fit["source_artifact"].isna().any()
        or by_fit["source_artifact"].astype(str).str.strip().eq("").any()
        or by_fit["source_artifact"].map(lambda value: Path(str(value)).is_absolute()).any()
    ):
        raise ValueError(
            "Retained fit evidence has invalid status, iteration, tolerance, or source fields."
        )
    if (
        coverage["n_without_per_fit_record"].astype(int).ne(0).any()
        or not coverage["coverage_status"].eq("indexed_with_complete_per_fit_records").all()
    ):
        raise ValueError("Retained analysis coverage still reports a fit-level gap.")


def _convergence_source_artifacts(
    repository_root: Path,
    by_fit: pd.DataFrame,
    auxiliary_inputs: list[str],
) -> list[Path]:
    relative_paths = set(by_fit["source_artifact"].dropna().astype(str))
    relative_paths.update(auxiliary_inputs)
    for relative_raw in tuple(relative_paths):
        source = repository_root / relative_raw
        if source.name != "transport_manifest.yaml" or not source.is_file():
            continue
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        method_params = artifacts.get("method_params") if isinstance(artifacts, dict) else None
        if not isinstance(method_params, str) or not method_params:
            raise ValueError(f"Transport manifest does not declare method_params: {relative_raw}")
        relative_params = Path(method_params)
        if relative_params.is_absolute():
            raise ValueError(
                f"Transport manifest uses an absolute method_params path: {relative_raw}"
            )
        if not (repository_root / relative_params).is_file():
            raise FileNotFoundError(repository_root / relative_params)
        relative_paths.add(relative_params.as_posix())
    return [Path(path) for path in sorted(relative_paths) if path]


def _capture_python_environment(output_root: Path) -> Path:
    packages = sorted(
        (
            {
                "package": distribution.metadata.get("Name", distribution.name),
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda row: str(row["package"]).lower(),
    )
    path = output_root / "python_packages.csv"
    _write_csv(path, packages)
    return path


def _capture_r_environment(output_root: Path) -> tuple[Path, str]:
    expression = (
        'cat("R_VERSION\\t", R.version.string, "\\n", sep=""); '
        f"pkgs <- c({', '.join(repr(name) for name in R_PACKAGES)}); "
        "for (p in pkgs) { v <- if (requireNamespace(p, quietly=TRUE)) "
        'as.character(packageVersion(p)) else "unavailable"; '
        'cat(p, "\\t", v, "\\n", sep="") }'
    )
    completed = subprocess.run(
        ["Rscript", "-e", expression],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        rows = [
            {
                "package": name,
                "version": "unavailable",
                "status": "Rscript_failed",
            }
            for name in R_PACKAGES
        ]
        r_version = "unavailable"
    else:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        r_version = lines[0].split("\t", 1)[1]
        rows = [
            {
                "package": line.split("\t", 1)[0],
                "version": line.split("\t", 1)[1],
                "status": (
                    "available" if line.split("\t", 1)[1] != "unavailable" else "unavailable"
                ),
            }
            for line in lines[1:]
        ]
    path = output_root / "r_packages.csv"
    _write_csv(path, rows)
    return path, r_version


def _capture_runtime(output_root: Path, *, r_version: str) -> Path:
    uv = subprocess.run(
        ["uv", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "build": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "uv": uv.stdout.strip() if uv.returncode == 0 else "unavailable",
        "r": r_version,
        "environment_scope": "host-specific verification snapshot",
    }
    path = output_root / "runtime.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _external_software_rows(
    repository_root: Path,
    *,
    python_packages: Path,
    r_packages: Path,
) -> list[dict[str, object]]:
    python = pd.read_csv(python_packages).set_index("package")["version"].to_dict()
    r = pd.read_csv(r_packages).set_index("package")["version"].to_dict()
    model = repository_root / (
        "data/models/celltypist/ref_pbmc_clean_celltypist_model_AIFI_L2_2024-04-19.pkl"
    )
    embedding = repository_root / "results/mouse_spleen_core_ot/embedding/embedding.npy"
    rows = [
        {
            "name": "AIFI Level-2 CellTypist model",
            "category": "external_trained_model",
            "version_or_identifier": "ref_pbmc_clean_celltypist_model_AIFI_L2_2024-04-19.pkl",
            "source_or_accession": "Allen Institute Human Immune Health Atlas model download",
            "repository_path": model.relative_to(repository_root).as_posix(),
            "sha256": sha256(model),
            "bytes": model.stat().st_size,
            "role": "recompute the provisional HIHA AIFI Level-2 reliability score",
            "provenance_source": "docs/manuscript_submission_hiha_data_code.md",
            "availability": "external; not packaged",
        },
        {
            "name": "MultiMAP",
            "category": "external_program",
            "version_or_identifier": "0.0.1; commit 681e608c45cdb6b139dfb6700e40c7520bc6096d",
            "source_or_accession": "https://github.com/Teichlab/MultiMAP.git",
            "repository_path": "results/mouse_spleen_core_ot/embedding/embedding.npy",
            "sha256": sha256(embedding),
            "bytes": embedding.stat().st_size,
            "role": "generate the fixed mouse-spleen input embedding geometry",
            "provenance_source": "results/mouse_spleen_core_ot/embedding/embedding_manifest.yaml",
            "availability": "generated embedding external to this unit",
        },
        {
            "name": "CellTypist",
            "category": "python_program",
            "version_or_identifier": str(python.get("celltypist", "unavailable")),
            "source_or_accession": "Python environment",
            "repository_path": "",
            "sha256": "",
            "bytes": "",
            "role": "HIHA score preparation and trained-in-run external baseline",
            "provenance_source": "python_packages.csv",
            "availability": "environment dependency",
        },
    ]
    for package in ("Seurat", "SingleR", "scmap", "CHETAH"):
        rows.append(
            {
                "name": package,
                "category": "r_program",
                "version_or_identifier": str(r.get(package, "unavailable")),
                "source_or_accession": "R/Bioconductor environment",
                "repository_path": "",
                "sha256": "",
                "bytes": "",
                "role": "retained external reference-mapping baseline",
                "provenance_source": "r_packages.csv",
                "availability": (
                    "environment dependency" if r.get(package) != "unavailable" else "unavailable"
                ),
            }
        )
    return rows


def _sanitize_resource_table(
    source: Path,
    destination: Path,
    *,
    repository_root: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(source, escapechar="\\")
    missing = sorted(set(RESOURCE_COLUMNS) - set(frame.columns))
    if missing and not CURRENT_RESOURCE_REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError(f"Resource table is missing columns {missing}: {source}")
    if missing:
        unknown_experiments = sorted(
            set(frame["experiment"].astype(str)) - set(CURRENT_RESOURCE_DATASET_NAMES)
        )
        if unknown_experiments:
            raise ValueError(
                f"Resource table has unknown experiments {unknown_experiments}: {source}"
            )
        commands = frame.get("exact_command", pd.Series("", index=frame.index))
        commands = commands.fillna("").astype(str)
        fallback_commands = frame.get("command_json", pd.Series("", index=frame.index))
        commands = commands.mask(commands.eq(""), fallback_commands.fillna("").astype(str))
        workers = frame.get("effective_workers", pd.Series("", index=frame.index))
        workers = workers.fillna("")
        requested = frame.get("requested_workers", pd.Series("", index=frame.index))
        workers = workers.mask(workers.eq(""), requested.fillna(""))
        frame = pd.DataFrame(
            {
                "dataset": frame["experiment"].map(CURRENT_RESOURCE_DATASET_NAMES),
                "evidence_unit": frame["evidence_unit"],
                "method": frame["method"],
                "condition": frame["condition"].astype(str) + ":" + frame["endpoint"].astype(str),
                "seed_or_split": frame["endpoint"].astype(str)
                + " seed"
                + frame["seed"].astype(str)
                + " repeat"
                + frame["repetition"].astype(str),
                "stage": "representative_mapping",
                "measurement_status": frame["verification_status"].map(
                    lambda status: "measured" if status == "passed" else "unavailable"
                ),
                "wall_seconds": frame["wall_seconds"],
                "peak_rss_bytes": frame["peak_rss_bytes"],
                "peak_rss_gib": frame["peak_rss_gib"],
                "shared_input_bytes": "",
                "retained_output_bytes": "",
                "command": commands,
                "measurement_tool": frame["resource_tool"],
                "measurement_scope": "representative_mapping_repetition",
                "parallel_jobs": workers,
                "software_environment": frame["environment_identity"],
                "raw_log_path": frame["raw_log_path"],
                "notes": frame.get("verification_path", pd.Series("", index=frame.index)).fillna(
                    ""
                ),
            }
        )
    else:
        frame = frame.loc[:, RESOURCE_COLUMNS].copy()
    prefix = repository_root.as_posix() + "/"
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.replace(prefix, "", regex=False)
            .str.replace(PERSONAL_HOME_PREFIX, "<user-home>/", regex=True)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def _consolidate_resources(
    repository_root: Path,
    resource_paths: tuple[Path, ...],
    output_root: Path,
) -> tuple[list[Path], Path, Path]:
    frames = []
    copied_paths = []
    for source in resource_paths:
        source = (repository_root / source).resolve()
        columns = set(pd.read_csv(source, nrows=0, escapechar="\\").columns)
        canonical_current = CURRENT_RESOURCE_REQUIRED_COLUMNS.issubset(columns)
        compatibility_destination = output_root / (
            "resources_by_repeat_compatibility.csv"
            if canonical_current
            else "resource_compatibility.csv"
        )
        sanitized = _sanitize_resource_table(
            source,
            compatibility_destination,
            repository_root=repository_root,
        )
        if canonical_current:
            canonical = pd.read_csv(source, escapechar="\\")
            prefix = repository_root.as_posix() + "/"
            for column in canonical.select_dtypes(include="object").columns:
                canonical[column] = (
                    canonical[column]
                    .fillna("")
                    .astype(str)
                    .str.replace(prefix, "", regex=False)
                    .str.replace(PERSONAL_HOME_PREFIX, "<user-home>/", regex=True)
                )
            canonical_path = output_root / source.name
            canonical.to_csv(canonical_path, index=False)
            copied_paths.append(canonical_path)
        for dataset, frame in sanitized.groupby("dataset", sort=True, dropna=False):
            destination = output_root / f"{dataset}_compare_baselines_by_method.csv"
            frame.to_csv(destination, index=False)
            copied_paths.append(destination)
        frames.append(sanitized)
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_root / "compare_baselines_by_method.csv"
    combined.to_csv(combined_path, index=False)
    summary = (
        combined.groupby(["dataset", "measurement_status"], dropna=False)
        .size()
        .rename("n_methods")
        .reset_index()
        .sort_values(["dataset", "measurement_status"])
    )
    summary_path = output_root / "resource_measurement_status_summary.csv"
    summary.to_csv(summary_path, index=False)
    return copied_paths, combined_path, summary_path


class PackageBuilder:
    def __init__(self, *, repository_root: Path, output_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.output_root = output_root.resolve()
        self.included: list[IncludedFile] = []

    def build(
        self,
        *,
        convergence_root: Path,
        resource_root: Path,
        environment_root: Path,
        audit_root: Path,
    ) -> None:
        _require_new_directory(self.output_root)
        for relative in (
            Path("experiments/generate_manuscript_convergence_audit.py"),
            Path("experiments/submission/build_convergence_environment_unit.py"),
            Path("tests/test_manuscript_convergence_audit.py"),
        ):
            self.copy_file(
                self.repository_root / relative,
                Path("code") / relative,
                source_label=relative.as_posix(),
            )
        for relative in (Path("pyproject.toml"), Path("uv.lock")):
            self.copy_file(
                self.repository_root / relative,
                Path("configs/environment") / relative,
                source_label=relative.as_posix(),
            )
        self.copy_tree(convergence_root, Path("verified_results/convergence"))
        self.copy_tree(resource_root, Path("verified_results/resources"))
        self.copy_tree(environment_root, Path("verified_results/environment"))
        self.copy_tree(audit_root, Path("audits"))
        self._write_readme()
        self._write_external_dependencies(convergence_root)
        self._write_exclusions()
        self._write_transformations()
        self._write_included_files()
        self._write_checksums()

    def copy_file(
        self,
        source: Path,
        destination: Path,
        *,
        source_label: str | None = None,
        note: str = "copied byte for byte",
    ) -> None:
        target = self.output_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.included.append(
            IncludedFile(
                package_path=destination.as_posix(),
                source_path=source_label or source.relative_to(self.repository_root).as_posix(),
                action="copy",
                source_bytes=source.stat().st_size,
                package_bytes=target.stat().st_size,
                source_sha256=sha256(source),
                package_sha256=sha256(target),
                note=note,
            )
        )

    def copy_tree(self, source: Path, destination: Path) -> None:
        for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
            relative = path.relative_to(source)
            self.copy_file(path, destination / relative)

    def _record_generated(self, relative: Path, note: str) -> None:
        path = self.output_root / relative
        self.included.append(
            IncludedFile(
                package_path=relative.as_posix(),
                source_path="generated_by:experiments/submission/build_convergence_environment_unit.py",
                action="generate",
                source_bytes=path.stat().st_size,
                package_bytes=path.stat().st_size,
                source_sha256=sha256(path),
                package_sha256=sha256(path),
                note=note,
            )
        )

    def _write_readme(self) -> None:
        path = self.output_root / "README.md"
        path.write_text(
            "# Convergence and environment evidence unit\n\n"
            "This package records the current cross-experiment convergence audit, "
            "the exact Python and R environment snapshot, external software/model "
            "provenance, and the three dataset resource audits. Scaling-iterate "
            "convergence is an iteration-stability diagnostic, not an exact "
            "optimization certificate. Missing fit-level convergence records and "
            "method-isolated resource measurements remain explicitly unavailable; "
            "no values are inferred.\n\n"
            "Regenerate the convergence tables with:\n\n"
            "```bash\n"
            "uv run python code/experiments/generate_manuscript_convergence_audit.py "
            "--project-root <repository-root> --output-root <new-output-root>\n"
            "```\n\n"
            "The declared source artifacts must be supplied at the repository-relative "
            "paths recorded in `manifests/external_dependencies.csv`.\n",
            encoding="utf-8",
        )
        self._record_generated(Path("README.md"), "unit scope and regeneration instructions")

    def _write_external_dependencies(self, convergence_root: Path) -> None:
        by_fit = pd.read_csv(convergence_root / "transport_convergence_by_fit.csv")
        convergence_manifest = yaml.safe_load(
            (convergence_root / "manifest.yaml").read_text(encoding="utf-8")
        )
        auxiliary_inputs = convergence_manifest.get(
            "auxiliary_index_inputs",
            [],
        )
        if not isinstance(auxiliary_inputs, list) or not all(
            isinstance(path, str) for path in auxiliary_inputs
        ):
            raise ValueError(
                "Convergence manifest auxiliary_index_inputs must be a list "
                "of repository-relative paths"
            )
        rows: list[dict[str, object]] = []
        source_artifacts = _convergence_source_artifacts(
            self.repository_root,
            by_fit,
            auxiliary_inputs,
        )
        for relative in source_artifacts:
            source = self.repository_root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            rows.append(
                {
                    "repository_path": relative.as_posix(),
                    "accession": "",
                    "sha256": sha256(source),
                    "bytes": source.stat().st_size,
                    "included_in_package": False,
                    "role": (
                        "retained per-fit or table-level convergence evidence "
                        "source or auxiliary family-index input"
                    ),
                    "prepared_by": "retained experiment pipeline; indexed by generate_manuscript_convergence_audit.py",
                }
            )
        model_relative = Path(
            "data/models/celltypist/ref_pbmc_clean_celltypist_model_AIFI_L2_2024-04-19.pkl"
        )
        model = self.repository_root / model_relative
        rows.append(
            {
                "repository_path": model_relative.as_posix(),
                "accession": "Allen Institute Human Immune Health Atlas model download",
                "sha256": sha256(model),
                "bytes": model.stat().st_size,
                "included_in_package": False,
                "role": "external trained model used to prepare the retained HIHA reliability score",
                "prepared_by": "external model; consumed by scripts/recompute_aifi_l2_score.py",
            }
        )
        relative = Path("manifests/external_dependencies.csv")
        _write_csv(self.output_root / relative, rows)
        self._record_generated(relative, "declared package-external evidence and model artifacts")

    def _write_exclusions(self) -> None:
        relative = Path("manifests/excluded_files.csv")
        _write_csv(
            self.output_root / relative,
            [
                {
                    "source_path": "results/submission_verification/*/baseline_resource_audit/raw/",
                    "exclusion_type": "raw_resource_logs",
                    "reason": "Dataset audits retain raw logs; this unit consolidates their method-level records without duplication.",
                    "replacement": "verified_results/resources/compare_baselines_by_method.csv",
                },
                {
                    "source_path": "**/*.h5; **/*.h5ad; **/*.hdf5",
                    "exclusion_type": "forbidden_large_analysis_object",
                    "reason": "HDF5-family analysis objects remain external to submission evidence packages.",
                    "replacement": "manifests/external_dependencies.csv or dataset-unit manifests",
                },
            ],
        )
        self._record_generated(relative, "explicit package exclusions")

    def _write_transformations(self) -> None:
        relative = Path("manifests/transformations.csv")
        _write_csv(
            self.output_root / relative,
            [
                {
                    "package_path": "verified_results/resources/*.csv",
                    "source_path": "dataset baseline_resource_audit/compare_baselines_by_method.csv",
                    "transformation": "replace repository-root absolute prefix in recorded commands with a repository-relative command",
                    "scientific_fields_changed": False,
                    "reason": "remove author-specific filesystem paths without changing measurements",
                }
            ],
        )
        self._record_generated(relative, "documented path-only resource-table transformation")

    def _write_included_files(self) -> None:
        relative = Path("manifests/included_files.csv")
        path = self.output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(IncludedFile.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(
                {field: getattr(item, field) for field in IncludedFile.__dataclass_fields__}
                for item in self.included
            )

    def _write_checksums(self) -> None:
        path = self.output_root / "manifests/checksums.sha256"
        files = sorted(
            candidate
            for candidate in self.output_root.rglob("*")
            if candidate.is_file() and candidate != path
        )
        path.write_text(
            "\n".join(
                f"{sha256(file)}  {file.relative_to(self.output_root).as_posix()}" for file in files
            )
            + "\n",
            encoding="utf-8",
        )


def _write_audits(
    snapshot_root: Path,
    *,
    convergence_root: Path,
    resource_paths: tuple[Path, ...],
) -> Path:
    audit_root = snapshot_root / "audit"
    audit_root.mkdir(parents=True)
    coverage = pd.read_csv(convergence_root / "retained_analysis_coverage.csv")
    gaps = coverage.loc[coverage["n_without_per_fit_record"].gt(0)]
    (audit_root / "scope.md").write_text(
        "# Scope\n\n"
        "Semantic unit: `convergence_and_environment`. The scope is every retained "
        "transport-fit family, the Python/R software snapshot, external programs and "
        "trained models, and the three dataset method-resource audits. Historical "
        "experiments and non-retained endpoints are excluded.\n",
        encoding="utf-8",
    )
    lineage_rows = [
        {
            "artifact": "verified_results/convergence/transport_convergence_by_fit.csv",
            "source": "retained transport manifests and convergence-bearing sensitivity tables",
            "generator": "experiments/generate_manuscript_convergence_audit.py",
            "status": "complete retained-family index with fit-level convergence evidence",
        },
        {
            "artifact": "verified_results/resources/compare_baselines_by_method.csv",
            "source": "; ".join(path.as_posix() for path in resource_paths),
            "generator": "experiments/submission/build_convergence_environment_unit.py",
            "status": "consolidated without inferred measurements",
        },
        {
            "artifact": "verified_results/environment/runtime.yaml",
            "source": "active verification environment",
            "generator": "experiments/submission/build_convergence_environment_unit.py",
            "status": "host-specific snapshot",
        },
    ]
    _write_csv(audit_root / "lineage.csv", lineage_rows)
    (audit_root / "code_config.md").write_text(
        "# Code and configuration\n\n"
        "The package copies the convergence generator, package builder, focused test, "
        "`pyproject.toml`, and `uv.lock` with source and staged SHA-256 identities. "
        "Environment capture records installed Python and selected R packages.\n",
        encoding="utf-8",
    )
    if gaps.empty:
        coverage_text = (
            "Every retained analysis family has complete fit-level convergence "
            "evidence; no retained record is unavailable or nonconverged."
        )
    else:
        gap_lines = "\n".join(
            f"- `{row.experiment}` / `{row.analysis_family}`: "
            f"{int(row.n_without_per_fit_record)} fit-level records unavailable."
            for row in gaps.itertuples(index=False)
        )
        coverage_text = (
            "The following retained families lack per-fit convergence records:\n\n" + gap_lines
        )
    (audit_root / "limitations.md").write_text(
        "# Limitations\n\n"
        "Scaling-iterate convergence is an iteration-stability diagnostic, not an "
        "exact optimization certificate. "
        + coverage_text
        + "\n\nMethod-isolated runtime or memory remains `unavailable` wherever the "
        "dataset resource table says so; no values are inferred. Resource measurements "
        "are host-specific provenance rather than portable efficiency rankings.\n",
        encoding="utf-8",
    )
    return audit_root


def build_unit(
    *,
    repository_root: Path,
    snapshot_root: Path,
    resource_paths: tuple[Path, ...],
) -> Path:
    repository_root = repository_root.resolve()
    snapshot_root = snapshot_root.resolve()
    _require_new_directory(snapshot_root)
    convergence_root = snapshot_root / "regenerated/convergence"
    convergence_root.mkdir(parents=True)
    generate_audit(repository_root, output_root=convergence_root)
    _validate_retained_fit_evidence(convergence_root)

    environment_root = snapshot_root / "regenerated/environment"
    environment_root.mkdir(parents=True)
    python_packages = _capture_python_environment(environment_root)
    r_packages, r_version = _capture_r_environment(environment_root)
    _capture_runtime(environment_root, r_version=r_version)
    _write_csv(
        environment_root / "external_software_and_models.csv",
        _external_software_rows(
            repository_root,
            python_packages=python_packages,
            r_packages=r_packages,
        ),
    )

    resource_root = snapshot_root / "regenerated/resources"
    resource_root.mkdir(parents=True)
    _consolidate_resources(
        repository_root,
        resource_paths,
        resource_root,
    )
    audit_root = _write_audits(
        snapshot_root,
        convergence_root=convergence_root,
        resource_paths=resource_paths,
    )
    package_root = snapshot_root / f"package_staging/{UNIT_ID}"
    PackageBuilder(
        repository_root=repository_root,
        output_root=package_root,
    ).build(
        convergence_root=convergence_root,
        resource_root=resource_root,
        environment_root=environment_root,
        audit_root=audit_root,
    )
    validation_report = snapshot_root / "package_validation/validation_report.md"
    validation_report.parent.mkdir(parents=True)
    retained_fit_count = len(pd.read_csv(convergence_root / "transport_convergence_by_fit.csv"))
    validation_report.write_text(
        "# Convergence and environment package validation\n\n"
        "- Overall status: **pass**\n"
        f"- Retained transport fits: {retained_fit_count}\n"
        "- Unavailable fit records: 0\n"
        "- Nonconverged fits: 0\n"
        "- Fit-level cap and tolerance agreement: **pass**\n",
        encoding="utf-8",
    )
    return package_root


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--resource-table", type=Path, action="append", required=True)
    args = parser.parse_args()
    package = build_unit(
        repository_root=args.repository_root,
        snapshot_root=args.snapshot_root,
        resource_paths=tuple(args.resource_table),
    )
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
