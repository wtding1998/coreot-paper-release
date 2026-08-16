from __future__ import annotations

import csv
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yaml


SPEC_SCHEMA_VERSION = 1
UNIT_STATUSES = {"pending", "audited", "verified"}
REQUIRED_UNIT_MANIFESTS = (
    Path("manifests/checksums.sha256"),
    Path("manifests/included_files.csv"),
    Path("manifests/external_dependencies.csv"),
)
EXTERNAL_ONLY_SUFFIXES = {".h5", ".h5ad", ".hdf5"}
EXTERNAL_DEPENDENCY_COLUMNS = {
    "repository_path",
    "accession",
    "sha256",
    "bytes",
    "included_in_package",
    "role",
    "prepared_by",
}


class SubmissionReleaseError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceUnitSpec:
    unit_id: str
    title: str
    status: str
    required: bool
    package_root: Path | None
    validation_report: Path | None


@dataclass(frozen=True)
class SubmissionReleaseSpec:
    release_id: str
    title: str
    units: tuple[EvidenceUnitSpec, ...]


@dataclass(frozen=True)
class UnitAudit:
    unit_id: str
    status: str
    required: bool
    package_valid: bool
    package_files: int
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseAudit:
    release_id: str
    ready: bool
    units: tuple[UnitAudit, ...]

    @property
    def verified_units(self) -> int:
        return sum(
            unit.required and unit.status == "verified" and unit.package_valid
            for unit in self.units
        )

    @property
    def required_units(self) -> int:
        return sum(unit.required for unit in self.units)

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(issue for unit in self.units for issue in unit.issues)


@dataclass(frozen=True)
class ReleaseBuild:
    package_root: Path
    units: int
    files: int
    bytes: int


@dataclass(frozen=True)
class ReleaseValidation:
    package_root: Path
    units: int
    files: int
    bytes: int


@dataclass(frozen=True)
class _CopyItem:
    source: Path
    destination: Path
    unit_id: str
    source_package_path: str
    sha256: str
    bytes: int


def audit_submission_release(
    *,
    spec_path: str | Path,
    repository_root: str | Path,
) -> ReleaseAudit:
    repository = Path(repository_root).resolve()
    spec = _load_spec(Path(spec_path), repository)
    audits = tuple(_audit_unit(unit, repository) for unit in spec.units)
    ready = all(
        unit.status == "verified" and unit.package_valid
        for unit in audits
        if unit.required
    ) and not any(unit.issues for unit in audits if unit.status == "verified")
    return ReleaseAudit(release_id=spec.release_id, ready=ready, units=audits)


def build_submission_release(
    *,
    spec_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> ReleaseBuild:
    repository = Path(repository_root).resolve()
    spec_path = Path(spec_path).resolve()
    spec = _load_spec(spec_path, repository)
    audit = audit_submission_release(
        spec_path=spec_path,
        repository_root=repository,
    )
    if not audit.ready:
        details = []
        for unit in audit.units:
            if unit.required and unit.status != "verified":
                details.append(f"{unit.unit_id}: status={unit.status}")
            details.extend(f"{unit.unit_id}: {issue}" for issue in unit.issues)
        raise SubmissionReleaseError(
            "Release is not ready; every required evidence unit must be verified. "
            + "; ".join(details)
        )

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty release root: {output}")

    copy_items = _release_copy_plan(spec, repository)
    output.mkdir(parents=True, exist_ok=True)
    copied_rows = []
    for item in copy_items:
        destination = output / item.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, destination)
        copied_rows.append(
            {
                "release_path": item.destination.as_posix(),
                "unit_id": item.unit_id,
                "source_package_path": item.source_package_path,
                "sha256": item.sha256,
                "bytes": item.bytes,
            }
        )

    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec_path, provenance / "release.yaml")
    _write_csv(
        provenance / "unit_registry.csv",
        [
            {
                "unit_id": unit.unit_id,
                "title": unit.title,
                "status": unit.status,
                "required": unit.required,
                "included": unit.status == "verified",
                "package_root": _relative_to_repository(unit.package_root, repository),
                "validation_report": _relative_to_repository(
                    unit.validation_report, repository
                ),
            }
            for unit in spec.units
        ],
    )
    _write_csv(provenance / "release_files.csv", copied_rows)
    _write_csv(
        provenance / "unit_code_requirements.csv",
        _collect_unit_code_requirements(spec),
    )
    _write_csv(
        provenance / "external_dependencies.csv",
        _collect_external_dependencies(spec),
    )
    (output / "README.md").write_text(_release_readme(spec), encoding="utf-8")
    _write_checksums(output, provenance / "checksums.sha256")
    validation = validate_submission_release(output)
    return ReleaseBuild(
        package_root=output,
        units=validation.units,
        files=validation.files,
        bytes=validation.bytes,
    )


def validate_submission_release(package_root: str | Path) -> ReleaseValidation:
    package = Path(package_root).resolve()
    if not package.is_dir():
        raise SubmissionReleaseError(f"Release root does not exist: {package}")
    checksum_path = package / "provenance/checksums.sha256"
    declared = _verify_checksum_manifest(package, checksum_path)
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual != declared:
        missing = sorted(declared - actual)
        undeclared = sorted(actual - declared)
        raise SubmissionReleaseError(
            f"Release checksum inventory mismatch: missing={missing}, undeclared={undeclared}"
        )
    if any(path.is_symlink() for path in package.rglob("*")):
        raise SubmissionReleaseError("Release packages must not contain symbolic links")
    packaged_external_objects = sorted(
        relative
        for relative in actual
        if Path(relative).suffix.lower() in EXTERNAL_ONLY_SUFFIXES
    )
    if packaged_external_objects:
        raise SubmissionReleaseError(
            "Release contains HDF5 objects that must remain external: "
            + ", ".join(packaged_external_objects)
        )

    registry_path = package / "provenance/unit_registry.csv"
    registry = pd.read_csv(registry_path)
    required_columns = {"unit_id", "status", "required", "included"}
    missing_columns = sorted(required_columns - set(registry.columns))
    if missing_columns:
        raise SubmissionReleaseError(
            f"Release unit registry is missing columns {missing_columns}: {registry_path}"
        )
    required = registry["required"].map(_as_bool)
    included = registry["included"].map(_as_bool)
    invalid_required = registry.loc[
        required & (~included | registry["status"].astype(str).ne("verified"))
    ]
    if not invalid_required.empty:
        raise SubmissionReleaseError(
            "Release contains an unverified required unit: "
            + ", ".join(invalid_required["unit_id"].astype(str))
        )
    if not (package / "code").is_dir():
        raise SubmissionReleaseError("Release is missing the shared code directory")
    expected_units = set(registry.loc[included, "unit_id"].astype(str))
    units_root = package / "units"
    actual_units = (
        {path.name for path in units_root.iterdir() if path.is_dir()}
        if units_root.is_dir()
        else set()
    )
    if actual_units != expected_units:
        raise SubmissionReleaseError(
            "Release unit directories do not match the included registry: "
            f"expected={sorted(expected_units)}, actual={sorted(actual_units)}"
        )

    external_dependencies_path = package / "provenance/external_dependencies.csv"
    external_dependency_issues = _external_dependency_issues(
        external_dependencies_path,
        expected_unit_ids=expected_units,
        allow_none_sentinel=True,
    )
    if external_dependency_issues:
        raise SubmissionReleaseError(
            "Release external-dependency manifest is invalid: "
            + "; ".join(external_dependency_issues)
        )

    release_files_path = package / "provenance/release_files.csv"
    release_files = pd.read_csv(release_files_path)
    file_columns = {"release_path", "unit_id", "sha256", "bytes"}
    missing_file_columns = sorted(file_columns - set(release_files.columns))
    if missing_file_columns:
        raise SubmissionReleaseError(
            f"Release file manifest is missing columns {missing_file_columns}: "
            f"{release_files_path}"
        )
    if release_files["release_path"].astype(str).duplicated().any():
        raise SubmissionReleaseError("Release file manifest contains duplicate paths")
    declared_release_files = set(release_files["release_path"].astype(str))
    generated_paths = {
        "README.md",
        "provenance/release.yaml",
        "provenance/unit_registry.csv",
        "provenance/release_files.csv",
        "provenance/unit_code_requirements.csv",
        "provenance/external_dependencies.csv",
    }
    copied_files = actual - generated_paths
    if declared_release_files != copied_files:
        raise SubmissionReleaseError(
            "Release file manifest does not cover exactly the copied files: "
            f"missing={sorted(declared_release_files - copied_files)}, "
            f"undeclared={sorted(copied_files - declared_release_files)}"
        )
    for row in release_files.itertuples(index=False):
        relative = _safe_relative(str(row.release_path))
        released = package / relative
        if sha256(released) != str(row.sha256):
            raise SubmissionReleaseError(f"Release file manifest hash mismatch: {relative}")
        if released.stat().st_size != int(row.bytes):
            raise SubmissionReleaseError(f"Release file manifest size mismatch: {relative}")
        if str(row.unit_id) not in expected_units:
            raise SubmissionReleaseError(
                f"Release file names an unregistered included unit: {row.unit_id}"
            )

    code_requirements_path = package / "provenance/unit_code_requirements.csv"
    code_requirements = pd.read_csv(code_requirements_path)
    code_columns = {"unit_id", "release_path", "package_sha256"}
    missing_code_columns = sorted(code_columns - set(code_requirements.columns))
    if missing_code_columns:
        raise SubmissionReleaseError(
            f"Unit code-requirements manifest is missing columns {missing_code_columns}: "
            f"{code_requirements_path}"
        )
    code_units = set(code_requirements["unit_id"].astype(str))
    if code_units != expected_units:
        raise SubmissionReleaseError(
            "Unit code requirements do not cover every included unit: "
            f"expected={sorted(expected_units)}, actual={sorted(code_units)}"
        )
    for row in code_requirements.itertuples(index=False):
        relative = _safe_relative(str(row.release_path))
        released = package / relative
        if not released.is_file() or sha256(released) != str(row.package_sha256):
            raise SubmissionReleaseError(
                f"Shared code does not satisfy {row.unit_id}: {relative}"
            )
    included_units = int(included.sum())
    total_bytes = sum(path.stat().st_size for path in package.rglob("*") if path.is_file())
    return ReleaseValidation(
        package_root=package,
        units=included_units,
        files=len(actual) + 1,
        bytes=total_bytes,
    )


def format_release_audit(audit: ReleaseAudit) -> str:
    lines = [
        f"release_id={audit.release_id}",
        f"ready={str(audit.ready).lower()}",
        f"verified_units={audit.verified_units}/{audit.required_units}",
    ]
    for unit in audit.units:
        state = "valid" if unit.package_valid else "not_validated"
        lines.append(
            f"{unit.unit_id}: status={unit.status}, package={state}, files={unit.package_files}"
        )
        lines.extend(f"  issue: {issue}" for issue in unit.issues)
    return "\n".join(lines)


def _load_spec(path: Path, repository: Path) -> SubmissionReleaseSpec:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SubmissionReleaseError(f"Release specification must be a mapping: {path}")
    if raw.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise SubmissionReleaseError(
            f"Expected schema_version={SPEC_SCHEMA_VERSION}: {path}"
        )
    release = raw.get("release")
    if not isinstance(release, Mapping):
        raise SubmissionReleaseError(f"Missing release mapping: {path}")
    release_id = _nonempty_string(release.get("id"), "release.id")
    title = _nonempty_string(release.get("title"), "release.title")
    raw_units = raw.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise SubmissionReleaseError(f"Release must register at least one unit: {path}")

    units = []
    seen: set[str] = set()
    for index, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, Mapping):
            raise SubmissionReleaseError(f"units[{index}] must be a mapping")
        unit_id = _nonempty_string(raw_unit.get("id"), f"units[{index}].id")
        if re.fullmatch(r"[a-z0-9][a-z0-9_]*", unit_id) is None:
            raise SubmissionReleaseError(
                f"Evidence-unit id must use lowercase letters, digits, and underscores: "
                f"{unit_id!r}"
            )
        if unit_id in seen:
            raise SubmissionReleaseError(f"Duplicate evidence-unit id: {unit_id}")
        seen.add(unit_id)
        status = _nonempty_string(raw_unit.get("status"), f"units[{index}].status")
        if status not in UNIT_STATUSES:
            raise SubmissionReleaseError(
                f"Unknown status {status!r} for {unit_id}; expected {sorted(UNIT_STATUSES)}"
            )
        package_root = _repository_path(
            repository, raw_unit.get("package_root"), f"{unit_id}.package_root"
        )
        validation_report = _repository_path(
            repository,
            raw_unit.get("validation_report"),
            f"{unit_id}.validation_report",
        )
        if status == "verified" and (package_root is None or validation_report is None):
            raise SubmissionReleaseError(
                f"Verified unit {unit_id} requires package_root and validation_report"
            )
        units.append(
            EvidenceUnitSpec(
                unit_id=unit_id,
                title=_nonempty_string(raw_unit.get("title"), f"{unit_id}.title"),
                status=status,
                required=bool(raw_unit.get("required", True)),
                package_root=package_root,
                validation_report=validation_report,
            )
        )
    return SubmissionReleaseSpec(release_id=release_id, title=title, units=tuple(units))


def _audit_unit(unit: EvidenceUnitSpec, repository: Path) -> UnitAudit:
    if unit.status != "verified":
        return UnitAudit(
            unit_id=unit.unit_id,
            status=unit.status,
            required=unit.required,
            package_valid=False,
            package_files=0,
            issues=(),
        )
    issues: list[str] = []
    package = unit.package_root
    if package is None or not package.is_dir():
        issues.append(f"package root is missing: {package}")
        return UnitAudit(unit.unit_id, unit.status, unit.required, False, 0, tuple(issues))
    if unit.validation_report is None or not unit.validation_report.is_file():
        issues.append(f"validation report is missing: {unit.validation_report}")
    for relative in REQUIRED_UNIT_MANIFESTS:
        if not (package / relative).is_file():
            issues.append(f"required unit manifest is missing: {relative}")
    package_files = 0
    if not issues:
        try:
            declared = _verify_checksum_manifest(
                package, package / "manifests/checksums.sha256"
            )
            package_files = len(declared) + 1
            actual = {
                path.relative_to(package).as_posix()
                for path in package.rglob("*")
                if path.is_file() and path != package / "manifests/checksums.sha256"
            }
            if actual != declared:
                issues.append("unit checksum manifest does not cover exactly the package files")
            packaged_external_objects = sorted(
                relative
                for relative in actual
                if Path(relative).suffix.lower() in EXTERNAL_ONLY_SUFFIXES
            )
            if packaged_external_objects:
                issues.append(
                    "large HDF5 objects must be declared as external dependencies, "
                    "not copied into an evidence package: "
                    + ", ".join(packaged_external_objects)
                )
            if any(path.is_symlink() for path in package.rglob("*")):
                issues.append("unit package contains a symbolic link")
            issues.extend(
                _external_dependency_issues(
                    package / "manifests/external_dependencies.csv"
                )
            )
            issues.extend(_code_snapshot_issues(package, repository))
        except (OSError, SubmissionReleaseError, ValueError) as exc:
            issues.append(str(exc))
    return UnitAudit(
        unit_id=unit.unit_id,
        status=unit.status,
        required=unit.required,
        package_valid=not issues,
        package_files=package_files,
        issues=tuple(issues),
    )


def _code_snapshot_issues(package: Path, repository: Path) -> list[str]:
    manifest_path = package / "manifests/included_files.csv"
    frame = pd.read_csv(manifest_path)
    required = {"package_path", "source_path", "source_sha256", "package_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return [f"included-files manifest is missing columns {missing}"]
    code = frame.loc[frame["package_path"].astype(str).str.startswith("code/")]
    if code.empty:
        return ["unit package has no declared code snapshot"]
    issues = []
    for row in code.itertuples(index=False):
        source_relative = _safe_relative(str(row.source_path))
        package_relative = _safe_relative(str(row.package_path))
        source = repository / source_relative
        staged = package / package_relative
        if not source.is_file():
            issues.append(f"code source is missing: {source_relative}")
            continue
        source_digest = sha256(source)
        if source_digest != str(row.source_sha256):
            issues.append(f"code source changed after unit validation: {source_relative}")
        if not staged.is_file() or sha256(staged) != str(row.package_sha256):
            issues.append(f"staged code differs from its manifest: {package_relative}")
    return issues


def _release_copy_plan(
    spec: SubmissionReleaseSpec,
    repository: Path,
) -> tuple[_CopyItem, ...]:
    by_destination: dict[str, _CopyItem] = {}
    for unit in spec.units:
        if unit.status != "verified" or unit.package_root is None:
            continue
        package = unit.package_root
        included = pd.read_csv(package / "manifests/included_files.csv")
        code_rows = included.loc[
            included["package_path"].astype(str).str.startswith("code/")
        ]
        for row in code_rows.itertuples(index=False):
            relative = _safe_relative(str(row.package_path))
            source = package / relative
            item = _CopyItem(
                source=source,
                destination=relative,
                unit_id=unit.unit_id,
                source_package_path=relative.as_posix(),
                sha256=sha256(source),
                bytes=source.stat().st_size,
            )
            _merge_copy_item(by_destination, item)

        checksum_path = package / "manifests/checksums.sha256"
        for source in sorted(path for path in package.rglob("*") if path.is_file()):
            relative = source.relative_to(package)
            if relative.parts[0] == "code" or source == checksum_path:
                continue
            destination = Path("units") / unit.unit_id / relative
            item = _CopyItem(
                source=source,
                destination=destination,
                unit_id=unit.unit_id,
                source_package_path=relative.as_posix(),
                sha256=sha256(source),
                bytes=source.stat().st_size,
            )
            _merge_copy_item(by_destination, item)
        if unit.validation_report is None:
            raise SubmissionReleaseError(
                f"Verified unit lacks a validation report: {unit.unit_id}"
            )
        validation_destination = (
            Path("units") / unit.unit_id / "audits/independent_validation.md"
        )
        validation_item = _CopyItem(
            source=unit.validation_report,
            destination=validation_destination,
            unit_id=unit.unit_id,
            source_package_path=(
                "validation_report:"
                + _relative_to_repository(unit.validation_report, repository)
            ),
            sha256=sha256(unit.validation_report),
            bytes=unit.validation_report.stat().st_size,
        )
        _merge_copy_item(by_destination, validation_item)
    if not any(path.startswith("code/") for path in by_destination):
        raise SubmissionReleaseError("Release copy plan contains no shared code")
    return tuple(by_destination[key] for key in sorted(by_destination))


def _merge_copy_item(by_destination: dict[str, _CopyItem], item: _CopyItem) -> None:
    key = item.destination.as_posix()
    existing = by_destination.get(key)
    if existing is not None and existing.sha256 != item.sha256:
        raise SubmissionReleaseError(
            f"Verified units contain conflicting shared code at {key}: "
            f"{existing.unit_id} != {item.unit_id}"
        )
    if existing is None:
        by_destination[key] = item


def _collect_external_dependencies(spec: SubmissionReleaseSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for unit in spec.units:
        if unit.status != "verified" or unit.package_root is None:
            continue
        path = unit.package_root / "manifests/external_dependencies.csv"
        frame = pd.read_csv(path)
        for record in frame.to_dict(orient="records"):
            rows.append({"unit_id": unit.unit_id, **record})
    if not rows:
        rows.append(
            {
                "unit_id": "none",
                "repository_path": "none",
                "accession": "",
                "sha256": "not_applicable",
                "bytes": 0,
                "included_in_package": False,
                "role": "No external dependencies declared.",
                "prepared_by": "not_applicable",
            }
        )
    return rows


def _external_dependency_issues(
    path: Path,
    *,
    expected_unit_ids: set[str] | None = None,
    allow_none_sentinel: bool = False,
) -> list[str]:
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return [f"cannot read external-dependency manifest {path}: {exc}"]

    required = set(EXTERNAL_DEPENDENCY_COLUMNS)
    if expected_unit_ids is not None:
        required.add("unit_id")
    missing = sorted(required - set(frame.columns))
    if missing:
        return [f"external-dependency manifest is missing columns {missing}"]

    issues: list[str] = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        unit_id = str(row.get("unit_id", "")).strip()
        repository_path = str(row["repository_path"]).strip()
        accession = str(row["accession"]).strip()
        digest = str(row["sha256"]).strip()
        role = str(row["role"]).strip()
        prepared_by = str(row["prepared_by"]).strip()

        is_none_sentinel = (
            allow_none_sentinel
            and unit_id == "none"
            and repository_path == "none"
            and digest == "not_applicable"
        )
        if is_none_sentinel:
            continue

        if expected_unit_ids is not None and unit_id not in expected_unit_ids:
            issues.append(f"row {row_number} names unknown unit_id {unit_id!r}")
        if not repository_path and not accession:
            issues.append(
                f"row {row_number} requires repository_path or accession"
            )
        if repository_path:
            try:
                _safe_relative(repository_path)
            except SubmissionReleaseError as exc:
                issues.append(f"row {row_number} has invalid repository_path: {exc}")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            issues.append(f"row {row_number} has invalid sha256 {digest!r}")
        try:
            byte_count = int(row["bytes"])
            if byte_count < 0:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(f"row {row_number} has invalid bytes {row['bytes']!r}")
        included = str(row["included_in_package"]).strip().lower()
        if included not in {"false", "0", "no"}:
            issues.append(
                f"row {row_number} must set included_in_package=false"
            )
        if not role:
            issues.append(f"row {row_number} requires a nonempty role")
        if not prepared_by:
            issues.append(f"row {row_number} requires nonempty preparation lineage")
    return issues


def _collect_unit_code_requirements(
    spec: SubmissionReleaseSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for unit in spec.units:
        if unit.status != "verified" or unit.package_root is None:
            continue
        included = pd.read_csv(unit.package_root / "manifests/included_files.csv")
        code = included.loc[
            included["package_path"].astype(str).str.startswith("code/")
        ]
        for record in code.to_dict(orient="records"):
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "release_path": record["package_path"],
                    "source_path": record["source_path"],
                    "source_sha256": record["source_sha256"],
                    "package_sha256": record["package_sha256"],
                    "bytes": record.get("package_bytes", ""),
                }
            )
    return rows


def _verify_checksum_manifest(package: Path, manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        raise SubmissionReleaseError(f"Checksum manifest is missing: {manifest_path}")
    declared: set[str] = set()
    for line_number, raw in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        digest, separator, relative_raw = raw.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise SubmissionReleaseError(
                f"Malformed checksum entry at {manifest_path}:{line_number}"
            )
        relative = _safe_relative(relative_raw).as_posix()
        if relative in declared:
            raise SubmissionReleaseError(f"Duplicate checksum path: {relative}")
        path = package / relative
        if not path.is_file():
            raise SubmissionReleaseError(f"Checksummed file is missing: {relative}")
        if sha256(path) != digest:
            raise SubmissionReleaseError(f"Checksum mismatch: {relative}")
        declared.add(relative)
    if not declared:
        raise SubmissionReleaseError(f"Checksum manifest is empty: {manifest_path}")
    return declared


def _write_checksums(package: Path, path: Path) -> None:
    files = sorted(
        candidate
        for candidate in package.rglob("*")
        if candidate.is_file() and candidate != path
    )
    path.write_text(
        "\n".join(
            f"{sha256(file)}  {file.relative_to(package).as_posix()}" for file in files
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise SubmissionReleaseError(f"Refusing to write empty release manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _release_readme(spec: SubmissionReleaseSpec) -> str:
    unit_lines = "\n".join(
        f"- `{unit.unit_id}`: {unit.title}"
        for unit in spec.units
        if unit.status == "verified"
    )
    return f"""# {spec.title}

This release package was assembled only from verified submission evidence
units. Shared code appears once under `code/`; unit-specific configurations,
data, results, figures, audits, and provenance appear under `units/`.

## Included evidence units

{unit_lines}

## Integrity

Run `shasum -a 256 -c provenance/checksums.sha256` from this directory.
`provenance/unit_registry.csv` records the source package and validation report
for each unit. `provenance/release_files.csv` records the lineage of every
copied file. External datasets or models that are not redistributed are listed
in `provenance/external_dependencies.csv` with checksums.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(
    repository: Path,
    raw: object,
    field: str,
) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise SubmissionReleaseError(f"{field} must be a nonempty relative path or null")
    relative = _safe_relative(raw)
    return repository / relative


def _safe_relative(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SubmissionReleaseError(f"Package paths must be safe and relative: {raw!r}")
    return path


def _relative_to_repository(path: Path | None, repository: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(repository).as_posix()
    except ValueError as exc:
        raise SubmissionReleaseError(f"Path lies outside repository: {path}") from exc


def _nonempty_string(raw: object, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise SubmissionReleaseError(f"{field} must be a nonempty string")
    return raw.strip()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
