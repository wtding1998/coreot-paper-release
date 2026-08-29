"""Validation contracts for deterministic CellTypist supplement candidates.

The validator intentionally works from files exposed by a candidate package.
It does not import a runner or regenerate a fit.  This keeps the promotion
boundary useful for both PBMC and HIHA candidates, including candidates whose
older row formats do not repeat endpoint and condition identifiers in every
CSV/Parquet file.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONDITIONS = ("incomplete_reference", "full_reference_control")
DEFAULT_METHOD = "celltypist_l3"
DEFAULT_PROFILE = "centered_full_sgd"
PROFILE_PARAMETERS: Mapping[str, object] = {
    "use_SGD": True,
    "with_mean": True,
    "mini_batch": False,
    "max_iter": 1000,
    "n_jobs": 1,
    "probability_path": "celltypist.annotate",
    "annotation_mode": "best match",
    "majority_voting": False,
}
SPARSE_PROFILE_PARAMETERS: Mapping[str, object] = {
    "use_SGD": True,
    "with_mean": False,
    "mini_batch": True,
    "batch_size": 1000,
    "epochs": 10,
    "n_jobs": -1,
    "probability_path": "sparse_model_probability",
}
PROFILE_PARAMETERS_BY_PROFILE: Mapping[str, Mapping[str, object]] = {
    "centered_full_sgd": PROFILE_PARAMETERS,
    "sparse_minibatch": SPARSE_PROFILE_PARAMETERS,
}
DEFAULT_MODEL_VISIBLE_FILES = (
    "counts.h5ad",
    "cells.csv",
    "genes.csv",
    "target_labels.csv",
    "broad_anchor_priors.csv",
    "initial_priors.csv",
    "dataset_config.yaml",
)


class CellTypistPromotionValidationError(ValueError):
    """Raised when a CellTypist candidate violates its declared contract."""

    def __init__(
        self,
        message: str,
        *,
        report: "ValidationReport | None" = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.issues = report.issues if report is not None else (message,)


class CellTypistCandidateRootError(FileExistsError):
    """Raised when a candidate workflow would reuse an existing output root."""


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    candidate_root: Path
    expected_units: tuple[tuple[str, int, str], ...]
    observed_units: tuple[tuple[str, int, str], ...]
    checks: tuple[ValidationCheck, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def passed(self) -> bool:
        """Boolean alias useful to callers that compare audit reports."""

        return self.valid

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(
            f"{check.name}: {check.detail}" for check in self.checks if not check.passed
        )


@dataclass(frozen=True)
class TableContract:
    """A generated table and its stable scientific-key contract."""

    path: str | Path
    key_columns: tuple[str, ...] = ()
    expected_keys: tuple[tuple[object, ...], ...] = ()
    require_full_coverage: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "key_columns", tuple(self.key_columns))
        object.__setattr__(
            self,
            "expected_keys",
            tuple(tuple(key) for key in self.expected_keys),
        )


@dataclass(frozen=True)
class ProtectedOutputContract:
    """A file or table that must remain unchanged during promotion.

    For a table, ``key_columns`` identify rows independent of row order.  The
    CellTypist method is excluded only when ``whole_file`` is false; all other
    rows and columns are compared exactly.
    """

    path: str | Path
    key_columns: tuple[str, ...] = ()
    whole_file: bool = False
    method_column: str = "method"
    celltypist_method: str = DEFAULT_METHOD

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "key_columns", tuple(self.key_columns))


@dataclass(frozen=True)
class ProtectedComparison:
    path: Path
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ProtectedOutputReport:
    candidate_root: Path
    reference_root: Path
    comparisons: tuple[ProtectedComparison, ...]

    @property
    def passed(self) -> bool:
        return all(comparison.passed for comparison in self.comparisons)

    @property
    def valid(self) -> bool:
        return self.passed

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(
            f"{comparison.path}: {comparison.detail}"
            for comparison in self.comparisons
            if not comparison.passed
        )


@dataclass(frozen=True)
class DatasetContract:
    """Observable candidate contract shared by PBMC and HIHA.

    ``source_root`` points to an authorized retained execution root.  It may
    be omitted for a deliberately self-contained candidate, but when the
    candidate manifest declares a source root it is always checked.
    """

    dataset: str
    endpoints: tuple[str, ...]
    seeds: tuple[int, ...]
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS
    candidate_root: Path | None = None
    source_root: Path | None = None
    source_truth_root: Path | None = None
    source_truth_columns: tuple[str, ...] = (
        "cell_id",
        "true_label",
        "removed_state",
        "is_absent_state",
        "is_shared_state",
    )
    source_identity: str = "semantic"
    method: str = DEFAULT_METHOD
    profile: str = DEFAULT_PROFILE
    model_visible_files: tuple[str, ...] = DEFAULT_MODEL_VISIBLE_FILES
    table_contracts: tuple[TableContract, ...] = ()
    aggregate_tables: tuple[TableContract, ...] = ()
    comparison_tables: tuple[TableContract, ...] = ()
    protected_outputs: tuple[ProtectedOutputContract, ...] = ()
    protected_reference_root: Path | None = None
    declared_files: tuple[str | Path, ...] = ()
    checksum_path: str | Path = "checksums.sha256"
    require_checksums: bool = True
    profile_parameters: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", str(self.dataset))
        object.__setattr__(self, "endpoints", tuple(str(value) for value in self.endpoints))
        object.__setattr__(self, "seeds", tuple(int(value) for value in self.seeds))
        object.__setattr__(self, "conditions", tuple(str(value) for value in self.conditions))
        if self.candidate_root is not None:
            object.__setattr__(self, "candidate_root", Path(self.candidate_root))
        if self.source_root is not None:
            object.__setattr__(self, "source_root", Path(self.source_root))
        if self.source_truth_root is not None:
            object.__setattr__(self, "source_truth_root", Path(self.source_truth_root))
        object.__setattr__(self, "source_truth_columns", tuple(self.source_truth_columns))
        object.__setattr__(self, "model_visible_files", tuple(self.model_visible_files))
        object.__setattr__(self, "table_contracts", tuple(self.table_contracts))
        object.__setattr__(self, "aggregate_tables", tuple(self.aggregate_tables))
        object.__setattr__(self, "comparison_tables", tuple(self.comparison_tables))
        object.__setattr__(self, "protected_outputs", tuple(self.protected_outputs))
        if self.protected_reference_root is not None:
            object.__setattr__(self, "protected_reference_root", Path(self.protected_reference_root))
        object.__setattr__(self, "declared_files", tuple(self.declared_files))
        object.__setattr__(self, "checksum_path", Path(self.checksum_path))
        if self.profile_parameters is not None:
            object.__setattr__(self, "profile_parameters", dict(self.profile_parameters))

    @property
    def expected_units(self) -> tuple[tuple[str, int, str], ...]:
        return tuple(
            (endpoint, seed, condition)
            for endpoint in self.endpoints
            for seed in self.seeds
            for condition in self.conditions
        )

    @property
    def tables(self) -> tuple[TableContract, ...]:
        return tuple(
            dict.fromkeys(
                (*self.table_contracts, *self.aggregate_tables, *self.comparison_tables)
            )
        )


@dataclass
class _Unit:
    run_root: Path
    run_id: str
    condition: str
    method: str
    endpoint: str | None = None
    seed: int | None = None
    prediction_manifest_path: Path | None = None
    prediction_manifest: dict[str, Any] = field(default_factory=dict)
    prediction_path: Path | None = None
    scoring_manifest_path: Path | None = None
    scoring_manifest: dict[str, Any] = field(default_factory=dict)
    score_path: Path | None = None
    truth_path: Path | None = None
    config_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    benchmark_path: Path | None = None
    benchmark: dict[str, Any] = field(default_factory=dict)
    prediction_ids: frozenset[str] = frozenset()
    truth_ids: frozenset[str] = frozenset()
    score_ids: frozenset[str] = frozenset()

    @property
    def key(self) -> tuple[str, int, str] | None:
        if self.endpoint is None or self.seed is None:
            return None
        return (self.endpoint, self.seed, self.condition)


class _ValidationState:
    def __init__(self, dataset: str, root: Path, expected: tuple[tuple[str, int, str], ...]):
        self.dataset = dataset
        self.root = root
        self.expected = expected
        self.checks: list[ValidationCheck] = []
        self.units: list[_Unit] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(ValidationCheck(name, bool(passed), detail))

    def issue(self, name: str, detail: str) -> None:
        self.check(name, False, detail)

    def report(self) -> ValidationReport:
        observed_keys = [unit.key for unit in self.units if unit.key is not None]
        expected_order = {key: index for index, key in enumerate(self.expected)}
        observed = tuple(
            sorted(observed_keys, key=lambda key: (expected_order.get(key, len(expected_order)), key))
        )
        return ValidationReport(
            dataset=self.dataset,
            candidate_root=self.root,
            expected_units=self.expected,
            observed_units=tuple(observed),
            checks=tuple(self.checks),
        )


def validate_celltypist_candidate(
    candidate_root: str | Path | None = None,
    contract: DatasetContract | None = None,
    *,
    dataset_contract: DatasetContract | None = None,
) -> ValidationReport:
    """Validate one complete CellTypist candidate without mutating it.

    A failed validation raises :class:`CellTypistPromotionValidationError`.
    The exception's ``report`` attribute retains every observed failure.
    """

    if isinstance(candidate_root, DatasetContract):
        if contract is not None or dataset_contract is not None:
            raise TypeError("a positional DatasetContract cannot be combined with contract keywords")
        contract = candidate_root
        candidate_root = None
    if contract is not None and dataset_contract is not None and contract != dataset_contract:
        raise TypeError("pass either contract or dataset_contract, not both")
    contract = contract or dataset_contract
    root = Path(candidate_root or (contract.candidate_root if contract else "")).resolve()
    if contract is None:
        contract = _infer_contract(root)
    elif candidate_root is None and contract.candidate_root is None:
        raise ValueError("candidate_root is required when contract.candidate_root is unset")

    state = _ValidationState(contract.dataset, root, contract.expected_units)
    manifest_path = root / "manifest.yaml"
    manifest = _read_yaml(manifest_path, state, "top manifest")
    if manifest is None:
        report = state.report()
        raise CellTypistPromotionValidationError(
            f"CellTypist candidate validation failed for {root}", report=report
        )

    _validate_top_manifest(state, manifest, contract)
    _validate_declared_artifacts(state, manifest, root, contract)
    _validate_checksums(state, root, contract)
    _discover_units(state, manifest, contract)
    _validate_matrix(state, contract)
    for unit in state.units:
        _validate_unit(state, unit, contract)
    _validate_paired_units(state, contract)
    _validate_source_lineage(state, manifest, contract)
    _validate_tables(state, root, manifest, contract)
    _validate_protected_contract(state, root, contract)

    report = state.report()
    if not report.valid:
        raise CellTypistPromotionValidationError(
            f"CellTypist candidate validation failed for {root}: "
            + "; ".join(report.issues),
            report=report,
        )
    return report


def validate_candidate(
    candidate_root: str | Path | None = None,
    contract: DatasetContract | None = None,
    **kwargs: Any,
) -> ValidationReport:
    """Short alias for :func:`validate_celltypist_candidate`."""

    return validate_celltypist_candidate(candidate_root, contract, **kwargs)


validate_dataset_candidate = validate_celltypist_candidate


def compare_protected_outputs(
    candidate_root: str | Path,
    reference_root: str | Path,
    contracts: Iterable[ProtectedOutputContract] = (),
    *,
    protected_outputs: Iterable[ProtectedOutputContract] | None = None,
) -> ProtectedOutputReport:
    """Compare protected files and non-CellTypist rows without writing files."""

    if protected_outputs is not None:
        if tuple(contracts):
            raise TypeError("pass either contracts or protected_outputs, not both")
        contracts = protected_outputs
    candidate = Path(candidate_root).resolve()
    reference = Path(reference_root).resolve()
    comparisons: list[ProtectedComparison] = []
    for raw_contract in contracts:
        current = raw_contract if isinstance(raw_contract, ProtectedOutputContract) else ProtectedOutputContract(raw_contract)
        candidate_path = _resolve_path(current.path, candidate)
        reference_path = _resolve_path(current.path, reference)
        if not candidate_path.is_file() or not reference_path.is_file():
            missing = []
            if not candidate_path.is_file():
                missing.append(f"candidate missing {candidate_path}")
            if not reference_path.is_file():
                missing.append(f"reference missing {reference_path}")
            comparisons.append(ProtectedComparison(current.path, False, "; ".join(missing)))
            continue
        if current.whole_file or current.path.suffix.lower() not in {".csv", ".parquet"}:
            passed = candidate_path.read_bytes() == reference_path.read_bytes()
            comparisons.append(
                ProtectedComparison(
                    current.path,
                    passed,
                    "" if passed else "protected file bytes differ",
                )
            )
            continue
        try:
            candidate_frame = _read_table(candidate_path)
            reference_frame = _read_table(reference_path)
            _compare_protected_tables(current, candidate_frame, reference_frame)
        except (KeyError, ValueError, AssertionError) as exc:
            comparisons.append(ProtectedComparison(current.path, False, str(exc)))
        else:
            comparisons.append(ProtectedComparison(current.path, True))
    return ProtectedOutputReport(candidate, reference, tuple(comparisons))


def assert_protected_outputs(
    candidate_root: str | Path,
    reference_root: str | Path,
    contracts: Iterable[ProtectedOutputContract] = (),
    **kwargs: Any,
) -> ProtectedOutputReport:
    """Compare protected outputs and raise the shared domain error on mismatch."""

    report = compare_protected_outputs(candidate_root, reference_root, contracts, **kwargs)
    if not report.passed:
        checks = tuple(
            ValidationCheck("protected_output", False, issue) for issue in report.issues
        )
        validation = ValidationReport(
            dataset="protected_outputs",
            candidate_root=Path(candidate_root).resolve(),
            expected_units=(),
            observed_units=(),
            checks=checks,
        )
        raise CellTypistPromotionValidationError(
            "Protected-output comparison failed: " + "; ".join(report.issues),
            report=validation,
        )
    return report


def _infer_contract(root: Path) -> DatasetContract:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8")) or {}
    metadata = manifest.get("metadata", {})
    endpoints = tuple(str(value) for value in metadata.get("endpoints", ()))
    seeds = tuple(int(value) for value in metadata.get("seeds", ()))
    if not endpoints or not seeds:
        raise ValueError("candidate manifest must declare endpoints and seeds when no contract is supplied")
    dataset = str(metadata.get("dataset") or metadata.get("dataset_id") or root.name)
    source = metadata.get("source_root") or metadata.get("source_v3_root")
    return DatasetContract(
        dataset=dataset,
        endpoints=endpoints,
        seeds=seeds,
        conditions=tuple(metadata.get("conditions", DEFAULT_CONDITIONS)),
        candidate_root=root,
        source_root=Path(source) if source else None,
    )


def _validate_top_manifest(
    state: _ValidationState,
    manifest: Mapping[str, Any],
    contract: DatasetContract,
) -> None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        state.issue("manifest.metadata", "top manifest metadata must be a mapping")
        return
    declared_dataset = metadata.get("dataset") or metadata.get("dataset_id")
    if declared_dataset is not None:
        state.check(
            "manifest.dataset",
            str(declared_dataset) == contract.dataset,
            f"expected {contract.dataset!r}, got {declared_dataset!r}",
        )
    else:
        state.check("manifest.dataset", True, "dataset supplied by DatasetContract")

    endpoints = tuple(str(value) for value in metadata.get("endpoints", ()))
    seeds = tuple(_as_int(value) for value in metadata.get("seeds", ()))
    conditions = tuple(str(value) for value in metadata.get("conditions", ()))
    state.check(
        "manifest.endpoints",
        endpoints == contract.endpoints,
        f"expected {contract.endpoints!r}, got {endpoints!r}",
    )
    state.check(
        "manifest.seeds",
        seeds == contract.seeds,
        f"expected {contract.seeds!r}, got {seeds!r}",
    )
    state.check(
        "manifest.conditions",
        conditions == contract.conditions,
        f"expected {contract.conditions!r}, got {conditions!r}",
    )
    execution_policy = metadata.get("execution_policy") or metadata.get("resource_policy")
    state.check(
        "manifest.execution_policy",
        execution_policy == "global_serial",
        f"expected 'global_serial', got {execution_policy!r}",
    )
    source_root = metadata.get("source_root") or metadata.get("source_v3_root")
    declared_source = _resolve_path(source_root, state.root) if source_root is not None else None
    fallback_source = contract.source_root or contract.source_truth_root
    if source_root is None and fallback_source is None:
        state.issue("manifest.source_root", "no authorized source root is declared")
    else:
        resolved = (
            declared_source
            if declared_source is not None and declared_source.is_dir()
            else _resolve_path(fallback_source, state.root)
        )
        state.check(
            "manifest.source_root",
            resolved.is_dir(),
            f"source root does not exist: {resolved}",
        )

    profiles = _profile_blocks(metadata)
    if not profiles:
        state.issue("manifest.profile", "no per-seed CellTypist profile block is declared")
    for seed in contract.seeds:
        block = profiles.get(seed)
        if block is None:
            state.issue("manifest.profile", f"missing profile block for seed {seed}")
            continue
        _validate_profile_block(state, f"manifest.profile[{seed}]", block, seed, contract)


def _validate_profile_block(
    state: _ValidationState,
    name: str,
    block: Mapping[str, Any],
    seed: int,
    contract: DatasetContract,
) -> None:
    profile = block.get("profile")
    state.check(name + ".name", profile == contract.profile, f"expected {contract.profile!r}, got {profile!r}")
    expected_parameters = contract.profile_parameters
    if expected_parameters is None:
        expected_parameters = PROFILE_PARAMETERS_BY_PROFILE.get(
            contract.profile, PROFILE_PARAMETERS
        )
    for key, expected in expected_parameters.items():
        actual = block.get(key)
        state.check(
            f"{name}.{key}",
            actual == expected,
            f"expected {expected!r}, got {actual!r}",
        )
    for key in ("random_state", "numpy_random_seed"):
        actual = _as_int(block.get(key))
        state.check(
            f"{name}.{key}",
            actual == seed,
            f"expected split seed {seed}, got {actual!r}",
        )


def _profile_blocks(metadata: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = metadata.get("celltypist_parameters_by_seed")
    if raw is None:
        raw = metadata.get("profile_by_seed")
    if raw is None:
        profile = metadata.get("celltypist_profile") or metadata.get("profile")
        if isinstance(profile, Mapping):
            raw = {seed: profile for seed in metadata.get("seeds", ())}
    if not isinstance(raw, Mapping):
        return {}
    out: dict[int, Mapping[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping) and _as_int(key) is not None:
            out[int(key)] = value
    return out


def _validate_declared_artifacts(
    state: _ValidationState,
    manifest: Mapping[str, Any],
    root: Path,
    contract: DatasetContract,
) -> None:
    paths: list[Path] = [_resolve_path(path, root) for path in contract.declared_files]
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        paths.extend(
            _resolve_path(value, root)
            for value in artifacts.values()
            if isinstance(value, (str, Path))
        )
    # Manifest artifact maps are part of the public contract too.  Reading
    # them here catches a stale prediction or score path even when a caller
    # did not list that path separately in DatasetContract.declared_files.
    for nested_manifest in root.rglob("*.yaml"):
        if nested_manifest == root / "manifest.yaml":
            continue
        try:
            payload = yaml.safe_load(nested_manifest.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(payload, Mapping) or not isinstance(payload.get("artifacts"), Mapping):
            continue
        for value in payload["artifacts"].values():
            if isinstance(value, (str, Path)):
                paths.append(_resolve_path(value, root))
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        state.check("declared_artifact", path.exists(), f"missing declared artifact: {path}")


def _validate_checksums(state: _ValidationState, root: Path, contract: DatasetContract) -> None:
    checksum_path = _resolve_path(contract.checksum_path, root)
    if not checksum_path.is_file():
        if contract.require_checksums:
            state.issue("checksums", f"checksum manifest does not exist: {checksum_path}")
        return
    declared: set[Path] = set()
    malformed = 0
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+[* ]?(.*)$", line)
        if match is None:
            malformed += 1
            state.issue("checksums", f"malformed checksum line {line_number}")
            continue
        expected, raw_path = match.groups()
        path = _resolve_path(raw_path, root)
        declared.add(path)
        if not path.is_file():
            state.issue("checksums", f"declared file does not exist: {raw_path}")
            continue
        actual = _sha256(path)
        state.check(
            "checksums",
            actual.lower() == expected.lower(),
            f"checksum mismatch for {raw_path}: expected {expected}, got {actual}",
        )
    if malformed == 0 and declared:
        state.check("checksums.inventory", True)


def _discover_units(
    state: _ValidationState,
    top_manifest: Mapping[str, Any],
    contract: DatasetContract,
) -> None:
    runs_root = _artifact_value(top_manifest, "runs")
    runs = _resolve_path(runs_root, state.root) if runs_root is not None else state.root / "runs"
    if not runs.is_dir():
        state.issue("units.discovery", f"runs root does not exist: {runs}")
        return
    manifest_paths = sorted(
        {
            *runs.rglob("external_baseline_manifest.yaml"),
            *runs.rglob("prediction_manifest.yaml"),
        }
    )
    prediction_dirs = {
        path.parent
        for path in runs.rglob("predictions.csv")
        if path.parent.name == contract.method
    }
    manifest_dirs = {path.parent for path in manifest_paths}
    for missing in sorted(prediction_dirs - manifest_dirs):
        state.issue("units.discovery", f"prediction directory has no manifest: {missing}")
    for manifest_path in manifest_paths:
        method = _method_from_path(manifest_path, contract.method)
        run_root = _run_root_from_path(manifest_path, runs)
        condition = _condition_from_path(manifest_path, contract.conditions)
        payload = _read_yaml(manifest_path, state, f"prediction manifest {manifest_path}") or {}
        metadata = payload.get("metadata", {})
        if isinstance(metadata, Mapping):
            method = str(metadata.get("method") or method)
            condition = str(metadata.get("condition") or condition)
        unit = _Unit(
            run_root=run_root,
            run_id=run_root.name,
            condition=condition,
            method=method,
            prediction_manifest_path=manifest_path,
            prediction_manifest=payload,
        )
        unit.prediction_path = _artifact_path(
            payload,
            "predictions",
            manifest_path.parent / "predictions.csv",
            state.root,
        )
        unit.config_path = _find_config(state.root, run_root.name)
        unit.config = _read_yaml(unit.config_path, state, f"resolved config {unit.config_path}") or {}
        unit.benchmark_path = _find_benchmark(state.root, run_root.name)
        unit.benchmark = _read_yaml(unit.benchmark_path, state, f"benchmark config {unit.benchmark_path}") or {}
        unit.endpoint = _first_text(
            _mapping_value(unit.config, "heldout_label", "endpoint", "removed_state"),
            _mapping_value(unit.benchmark, "heldout_label", "endpoint", "removed_state"),
            _mapping_value(metadata, "heldout_label", "endpoint", "removed_state"),
        )
        unit.seed = _first_int(
            _mapping_value(unit.benchmark, "split.seed", "seed", "repeat"),
            _mapping_value(unit.config, "repeat", "split.seed", "seed"),
            _mapping_value(metadata, "repeat", "seed"),
        )
        unit.truth_path = _find_truth(unit.run_root, unit.condition, state.root)
        unit.scoring_manifest_path = _find_scoring_manifest(unit.run_root, unit.condition)
        if unit.scoring_manifest_path is not None:
            unit.scoring_manifest = _read_yaml(
                unit.scoring_manifest_path,
                state,
                f"scoring manifest {unit.scoring_manifest_path}",
            ) or {}
            unit.score_path = _artifact_path(
                unit.scoring_manifest,
                "cell_scores",
                unit.scoring_manifest_path.parent / "cell_scores.parquet",
                state.root,
            )
        state.units.append(unit)


def _validate_matrix(state: _ValidationState, contract: DatasetContract) -> None:
    expected = set(contract.expected_units)
    observed = [unit.key for unit in state.units if unit.key is not None]
    observed_set = set(observed)
    duplicates = sorted({key for key in observed if observed.count(key) > 1})
    missing = sorted(expected - observed_set)
    extra = sorted(observed_set - expected)
    state.check("unit_matrix.duplicates", not duplicates, f"duplicate units: {duplicates}")
    state.check("unit_matrix.missing", not missing, f"missing units: {missing}")
    state.check("unit_matrix.extras", not extra, f"undeclared units: {extra}")
    state.check(
        "unit_matrix.exact",
        not duplicates and not missing and not extra and len(observed) == len(expected),
        f"expected {len(expected)} units, observed {len(observed)}",
    )


def _validate_unit(state: _ValidationState, unit: _Unit, contract: DatasetContract) -> None:
    label = f"unit[{unit.run_id}/{unit.condition}]"
    if unit.method != contract.method:
        state.issue(label, f"expected method {contract.method!r}, got {unit.method!r}")
    if unit.endpoint is None:
        state.issue(label, "endpoint identity is unavailable")
    if unit.seed is None:
        state.issue(label, "split seed identity is unavailable")
        return
    metadata = unit.prediction_manifest.get("metadata", {})
    parameters = metadata.get("parameters", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(parameters, Mapping):
        parameters = {}
    _validate_parameter_identity(state, label + ".prediction_manifest", parameters, unit.seed, contract)
    if isinstance(metadata, Mapping):
        state.check(
            label + ".prediction_manifest.condition",
            str(metadata.get("condition", unit.condition)) == unit.condition,
            f"condition mismatch: expected {unit.condition!r}",
        )
        state.check(
            label + ".prediction_manifest.method",
            str(metadata.get("method", unit.method)) == contract.method,
            f"method mismatch: expected {contract.method!r}",
        )
    _validate_config_identity(state, label + ".config", unit, contract)
    _validate_scoring_identity(state, label + ".scoring_manifest", unit, contract)
    prediction = _read_table_checked(state, unit.prediction_path, label + ".predictions")
    truth = _read_table_checked(state, unit.truth_path, label + ".truth")
    scores = _read_table_checked(state, unit.score_path, label + ".scores")
    if prediction is not None:
        _validate_prediction_rows(state, label + ".predictions", prediction, unit, contract)
        unit.prediction_ids = frozenset(prediction["cell_id"].astype(str)) if "cell_id" in prediction else frozenset()
    if truth is not None:
        _validate_truth_rows(state, label + ".truth", truth, unit, contract)
        unit.truth_ids = frozenset(truth["cell_id"].astype(str)) if "cell_id" in truth else frozenset()
    if scores is not None:
        _validate_score_rows(state, label + ".scores", scores, unit, contract)
        unit.score_ids = frozenset(scores.loc[scores["method"].astype(str).eq(contract.method), "cell_id"].astype(str)) if "cell_id" in scores and "method" in scores else frozenset()
    if prediction is not None and truth is not None:
        state.check(
            label + ".prediction_truth_membership",
            unit.prediction_ids == unit.truth_ids,
            f"prediction/truth IDs differ: prediction-only={sorted(unit.prediction_ids - unit.truth_ids)[:5]}, truth-only={sorted(unit.truth_ids - unit.prediction_ids)[:5]}",
        )
    if scores is not None and prediction is not None:
        state.check(
            label + ".prediction_score_membership",
            unit.score_ids == unit.prediction_ids,
            f"prediction/score IDs differ: prediction-only={sorted(unit.prediction_ids - unit.score_ids)[:5]}, score-only={sorted(unit.score_ids - unit.prediction_ids)[:5]}",
        )


def _validate_parameter_identity(
    state: _ValidationState,
    label: str,
    parameters: Mapping[str, Any],
    seed: int,
    contract: DatasetContract,
) -> None:
    _validate_profile_block(state, label, parameters, seed, contract)


def _validate_config_identity(
    state: _ValidationState,
    label: str,
    unit: _Unit,
    contract: DatasetContract,
) -> None:
    config = unit.config
    if not config:
        return
    state.check(
        label + ".endpoint",
        _first_text(_mapping_value(config, "heldout_label", "endpoint", "removed_state")) == unit.endpoint,
        f"expected endpoint {unit.endpoint!r}",
    )
    state.check(
        label + ".seed",
        _first_int(_mapping_value(config, "repeat", "split.seed", "seed")) == unit.seed,
        f"expected seed {unit.seed}",
    )
    celltypist = config.get("celltypist", {})
    if not isinstance(celltypist, Mapping):
        state.issue(label + ".celltypist", "celltypist block is not a mapping")
        return
    _validate_partial_parameter_identity(
        state, label + ".celltypist", celltypist, unit.seed or -1, contract
    )
    methods = tuple(str(method) for method in config.get("methods", ()))
    if methods:
        state.check(label + ".methods", contract.method in methods, f"expected {contract.method!r} in {methods!r}")


def _validate_scoring_identity(
    state: _ValidationState,
    label: str,
    unit: _Unit,
    contract: DatasetContract,
) -> None:
    manifest = unit.scoring_manifest
    if not manifest:
        state.issue(label, "scoring manifest is unavailable")
        return
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        state.issue(label, "metadata is not a mapping")
        return
    state.check(label + ".condition", str(metadata.get("condition", unit.condition)) == unit.condition, f"expected {unit.condition!r}")
    methods = tuple(str(method) for method in metadata.get("methods", ()))
    state.check(label + ".method", contract.method in methods, f"expected {contract.method!r} in {methods!r}")
    parameters = metadata.get("method_parameters", {})
    if isinstance(parameters, Mapping) and isinstance(parameters.get(contract.method), Mapping):
        _validate_parameter_identity(
            state,
            label + ".parameters",
            parameters[contract.method],
            unit.seed or -1,
            contract,
        )
    else:
        state.issue(label + ".parameters", f"missing method parameters for {contract.method}")


def _validate_partial_parameter_identity(
    state: _ValidationState,
    label: str,
    parameters: Mapping[str, Any],
    seed: int,
    contract: DatasetContract,
) -> None:
    """Validate legacy resolved configs without inventing omitted defaults."""

    profile = parameters.get("profile")
    if profile is not None:
        state.check(label + ".profile", profile == contract.profile, f"expected {contract.profile!r}")
    for key in ("random_state", "numpy_random_seed"):
        if key in parameters:
            state.check(label + f".{key}", _as_int(parameters[key]) == seed, f"expected split seed {seed}")
    expected_parameters = contract.profile_parameters
    if expected_parameters is None:
        expected_parameters = PROFILE_PARAMETERS_BY_PROFILE.get(
            contract.profile, PROFILE_PARAMETERS
        )
    for key, expected in expected_parameters.items():
        if key in parameters:
            state.check(label + f".{key}", parameters[key] == expected, f"expected {expected!r}")


def _validate_prediction_rows(
    state: _ValidationState,
    label: str,
    frame: pd.DataFrame,
    unit: _Unit,
    contract: DatasetContract,
) -> None:
    _require_columns(state, label, frame, ("cell_id", "method", "heldout_label", "repeat"))
    if not {"cell_id", "method", "heldout_label", "repeat"} <= set(frame.columns):
        return
    state.check(label + ".cell_ids_unique", not frame["cell_id"].astype(str).duplicated().any(), "duplicate cell_id values")
    state.check(label + ".method", set(frame["method"].astype(str)) == {contract.method}, f"expected only {contract.method!r}")
    state.check(label + ".endpoint", set(frame["heldout_label"].astype(str)) == {str(unit.endpoint)}, f"expected endpoint {unit.endpoint!r}")
    state.check(label + ".seed", set(_numeric_ints(frame["repeat"])) == {unit.seed}, f"expected seed {unit.seed}")
    if "is_full_reference_control" in frame.columns:
        expected = unit.condition == "full_reference_control"
        observed = frame["is_full_reference_control"].map(_as_bool)
        state.check(label + ".condition", set(observed) == {expected}, f"expected is_full_reference_control={expected}")
    _validate_optional_identity_columns(state, label, frame, unit, contract)


def _validate_truth_rows(
    state: _ValidationState,
    label: str,
    frame: pd.DataFrame,
    unit: _Unit,
    contract: DatasetContract,
) -> None:
    _require_columns(state, label, frame, ("cell_id",))
    if "cell_id" not in frame.columns:
        return
    state.check(label + ".cell_ids_unique", not frame["cell_id"].astype(str).duplicated().any(), "duplicate cell_id values")
    _validate_optional_identity_columns(state, label, frame, unit, contract)


def _validate_score_rows(
    state: _ValidationState,
    label: str,
    frame: pd.DataFrame,
    unit: _Unit,
    contract: DatasetContract,
) -> None:
    _require_columns(state, label, frame, ("cell_id", "method"))
    if not {"cell_id", "method"} <= set(frame.columns):
        return
    method_frame = frame.loc[frame["method"].astype(str).eq(contract.method)].copy()
    state.check(label + ".method", set(frame["method"].astype(str)) == {contract.method}, f"expected only {contract.method!r}")
    state.check(
        label + ".cell_ids_unique",
        not method_frame["cell_id"].astype(str).duplicated().any(),
        "duplicate cell_id values",
    )
    _validate_optional_identity_columns(state, label, frame, unit, contract)


def _validate_optional_identity_columns(
    state: _ValidationState,
    label: str,
    frame: pd.DataFrame,
    unit: _Unit,
    contract: DatasetContract,
) -> None:
    aliases = {
        "endpoint": ("endpoint", "heldout_label", "removed_state"),
        "seed": ("seed", "repeat", "split_seed"),
        "condition": ("condition", "condition_id"),
        "method": ("method",),
    }
    expected = {
        "endpoint": unit.endpoint,
        "seed": unit.seed,
        "condition": unit.condition,
        "method": contract.method,
    }
    for identity, columns in aliases.items():
        present = next((column for column in columns if column in frame.columns), None)
        if present is None:
            continue
        observed = frame[present]
        if identity == "seed":
            values: set[object] = set(_numeric_ints(observed))
        else:
            values = set(observed.astype(str))
        state.check(label + f".{identity}", values == {expected[identity]}, f"expected {expected[identity]!r}")


def _validate_paired_units(state: _ValidationState, contract: DatasetContract) -> None:
    by_pair: dict[tuple[str, int], list[_Unit]] = {}
    for unit in state.units:
        if unit.endpoint is not None and unit.seed is not None:
            by_pair.setdefault((unit.endpoint, unit.seed), []).append(unit)
    for pair, units in sorted(by_pair.items()):
        by_condition = {unit.condition: unit for unit in units}
        if set(by_condition) != set(contract.conditions):
            continue
        first = by_condition[contract.conditions[0]]
        for condition in contract.conditions[1:]:
            other = by_condition[condition]
            state.check(
                f"paired_membership[{pair[0]}/{pair[1]}]",
                first.prediction_ids == other.prediction_ids
                and first.truth_ids == other.truth_ids,
                f"query membership differs between {first.condition} and {other.condition}",
            )


def _validate_source_lineage(
    state: _ValidationState,
    top_manifest: Mapping[str, Any],
    contract: DatasetContract,
) -> None:
    metadata = top_manifest.get("metadata", {})
    declared = metadata.get("source_root") or metadata.get("source_v3_root") if isinstance(metadata, Mapping) else None
    declared_path = _resolve_path(declared, state.root) if declared else None
    source_value = (
        declared_path
        if declared_path is not None and declared_path.exists()
        else contract.source_root
    )
    source_root = _resolve_path(source_value, state.root) if source_value is not None else None
    truth_root = (
        _resolve_path(contract.source_truth_root, state.root)
        if contract.source_truth_root is not None
        else source_root
    )
    if source_root is None and truth_root is None:
        return
    if source_root is not None and not source_root.exists():
        source_root = None
    if truth_root is not None and not truth_root.exists():
        truth_root = None
    if source_root is None and truth_root is None:
        return
    for pair in sorted({(unit.endpoint, unit.seed) for unit in state.units if unit.endpoint and unit.seed is not None}):
        endpoint, seed = pair
        matching = [unit for unit in state.units if unit.endpoint == endpoint and unit.seed == seed]
        for unit in matching:
            source_truth_run = _source_run_root(truth_root, unit.run_id) if truth_root else None
            source_truth = (
                source_truth_run / "benchmark" / unit.condition / "evaluation_truth" / "query_truth.csv"
                if source_truth_run
                else state.root / "__missing__"
            )
            candidate_truth = _read_table(unit.truth_path) if unit.truth_path and unit.truth_path.is_file() else None
            source_frame = _read_table(source_truth) if source_truth.is_file() else None
            if candidate_truth is None or source_frame is None:
                state.issue(
                    f"source_truth[{endpoint}/{seed}/{unit.condition}]",
                    f"missing candidate or authorized source truth ({candidate_truth is None=}, {source_truth})",
                )
            else:
                try:
                    _compare_truth_frames(candidate_truth, source_frame, contract)
                except (AssertionError, KeyError, ValueError) as exc:
                    state.issue(f"source_truth[{endpoint}/{seed}/{unit.condition}]", str(exc))
                else:
                    state.check(f"source_truth[{endpoint}/{seed}/{unit.condition}]", True)

        if source_root is None:
            continue
        source_cells: dict[str, set[str]] = {}
        source_references: dict[str, set[str]] = {}
        for condition in contract.conditions:
            cells_path = _source_run_root(source_root, matching[0].run_id) / "benchmark" / condition / "model_visible" / "cells.csv" if matching else None
            if cells_path is None:
                continue
            model_visible = cells_path.parent
            for filename in contract.model_visible_files:
                state.check(
                    f"source_model_visible[{endpoint}/{seed}/{condition}]",
                    (model_visible / filename).is_file(),
                    f"missing model-visible artifact: {model_visible / filename}",
                )
            if cells_path.is_file():
                cells = _read_table(cells_path)
                if cells is not None and {"cell_id", "domain"} <= set(cells.columns):
                    ids = cells["cell_id"].astype(str)
                    state.check(
                        f"source_model_visible[{endpoint}/{seed}/{condition}].unique",
                        not ids.duplicated().any(),
                        "duplicate model-visible cell IDs",
                    )
                    source_cells[condition] = set(ids.loc[cells["domain"].astype(str).eq("query")])
                    source_references[condition] = set(ids.loc[cells["domain"].astype(str).eq("reference")])
        if len(source_cells) == len(contract.conditions):
            query_sets = tuple(source_cells[condition] for condition in contract.conditions)
            state.check(
                f"source_pairing[{endpoint}/{seed}].query",
                all(query_set == query_sets[0] for query_set in query_sets[1:]),
                "paired source conditions have different query membership",
            )
            if "full_reference_control" in source_references and "incomplete_reference" in source_references:
                state.check(
                    f"source_pairing[{endpoint}/{seed}].reference",
                    source_references["incomplete_reference"] <= source_references["full_reference_control"],
                    "incomplete reference membership is not a subset of restored reference membership",
                )


def _validate_tables(
    state: _ValidationState,
    root: Path,
    top_manifest: Mapping[str, Any],
    contract: DatasetContract,
) -> None:
    contracts = contract.tables
    if not contracts:
        tables_root_value = _artifact_value(top_manifest, "tables")
        tables_root = _resolve_path(tables_root_value, root) if tables_root_value else root / "tables"
        discovered = sorted(tables_root.glob("*.csv")) if tables_root.is_dir() else []
        contracts = tuple(TableContract(path.relative_to(root)) for path in discovered)
        if not contracts:
            state.issue("tables", f"no aggregate/comparison CSV artifacts found under {tables_root}")
    for table_contract in contracts:
        path = _resolve_path(table_contract.path, root)
        if not path.is_file():
            state.issue("table", f"missing declared table: {path}")
            continue
        try:
            frame = _read_table(path)
            _validate_table_contract(frame, table_contract, contract)
        except (KeyError, ValueError, AssertionError) as exc:
            state.issue(f"table[{table_contract.path}]", str(exc))
        else:
            state.check(f"table[{table_contract.path}]", True)


def _validate_protected_contract(
    state: _ValidationState,
    candidate_root: Path,
    contract: DatasetContract,
) -> None:
    if not contract.protected_outputs:
        return
    if contract.protected_reference_root is None:
        state.issue(
            "protected_outputs",
            "protected outputs are declared but protected_reference_root is unset",
        )
        return
    report = compare_protected_outputs(
        candidate_root,
        _resolve_path(contract.protected_reference_root, candidate_root),
        contract.protected_outputs,
    )
    if report.passed:
        state.check("protected_outputs", True)
    else:
        for issue in report.issues:
            state.issue("protected_outputs", issue)


def _validate_table_contract(
    frame: pd.DataFrame,
    table_contract: TableContract,
    contract: DatasetContract,
) -> None:
    columns = table_contract.key_columns or _infer_key_columns(frame)
    if not columns:
        raise ValueError("no stable key columns declared or inferable")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing stable key columns {missing}")
    key_frame = frame.loc[:, columns].copy()
    if key_frame.duplicated().any():
        raise ValueError("stable scientific keys are not unique")
    if table_contract.expected_keys:
        expected = pd.DataFrame(table_contract.expected_keys, columns=columns)
        observed = key_frame.reset_index(drop=True).sort_values(list(columns), kind="mergesort").reset_index(drop=True)
        expected = expected.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_exact=True,
                check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError("stable scientific keys do not have the declared full coverage") from exc
    elif table_contract.require_full_coverage:
        expected = {
            (endpoint, seed, condition)
            for endpoint in contract.endpoints
            for seed in contract.seeds
            for condition in contract.conditions
        }
        aliases = {"endpoint": _column_alias(columns, ("endpoint", "held_out_label", "removed_state")), "seed": _column_alias(columns, ("seed", "repeat")), "condition": _column_alias(columns, ("condition", "condition_id"))}
        if all(aliases.values()):
            observed = {
                (str(row[aliases["endpoint"]]), int(row[aliases["seed"]]), str(row[aliases["condition"]]))
                for _, row in frame.iterrows()
            }
            if observed != expected:
                raise ValueError("stable scientific keys do not cover the complete endpoint/seed/condition matrix")


def _compare_protected_tables(
    contract: ProtectedOutputContract,
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    for frame in (candidate, reference):
        if contract.method_column in frame.columns:
            frame.drop(frame.index[frame[contract.method_column].astype(str).eq(contract.celltypist_method)], inplace=True)
    columns = contract.key_columns or _infer_key_columns(candidate)
    if not columns:
        columns = tuple(candidate.columns)
    missing = [column for column in columns if column not in candidate.columns or column not in reference.columns]
    if missing:
        raise KeyError(f"missing protected stable key columns {missing}")
    if candidate.loc[:, columns].duplicated().any() or reference.loc[:, columns].duplicated().any():
        raise ValueError("protected stable keys are not unique")
    candidate = candidate.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    reference = reference.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    if list(candidate.columns) != list(reference.columns):
        raise ValueError("protected non-CellTypist columns differ")
    pd.testing.assert_frame_equal(candidate, reference, check_exact=True, check_dtype=True)


def _read_table_checked(state: _ValidationState, path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        state.issue(label, f"artifact does not exist: {path}")
        return None
    try:
        return _read_table(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        state.issue(label, f"cannot read artifact: {exc}")
        return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"unsupported table format: {path}")


def _read_yaml(path: Path | None, state: _ValidationState, label: str) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        state.issue(label, f"artifact does not exist: {path}")
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        state.issue(label, f"cannot read YAML: {exc}")
        return None
    if not isinstance(payload, dict):
        state.issue(label, "YAML root must be a mapping")
        return None
    return payload


def _artifact_value(manifest: Mapping[str, Any], name: str) -> object | None:
    artifacts = manifest.get("artifacts", {})
    return artifacts.get(name) if isinstance(artifacts, Mapping) else None


def _artifact_path(
    manifest: Mapping[str, Any],
    name: str,
    fallback: Path,
    root: Path,
) -> Path:
    value = _artifact_value(manifest, name)
    return _resolve_path(value, root) if value is not None else fallback


def _find_config(root: Path, run_id: str) -> Path | None:
    candidates = (
        root / "resolved_configs" / run_id / "external_baselines.yaml",
        root / "resolved_configs" / f"{run_id}.yaml",
        root / "runs" / run_id / "external_baselines.yaml",
        root / "runs" / run_id / "resolved_config.yaml",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _find_benchmark(root: Path, run_id: str) -> Path | None:
    candidates = (
        root / "grid" / run_id / "benchmark.yaml",
        root / "runs" / run_id / "benchmark.yaml",
        root / "runs" / run_id / "benchmark" / "benchmark.yaml",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _find_truth(run_root: Path, condition: str, root: Path) -> Path:
    candidates = (
        run_root / "benchmark" / condition / "evaluation_truth" / "query_truth.csv",
        run_root / "evaluation_truth" / condition / "query_truth.csv",
        run_root / "truth" / condition / "query_truth.csv",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _find_scoring_manifest(run_root: Path, condition: str) -> Path | None:
    candidates = sorted((run_root / "scoring" / condition).glob("*/scoring_manifest.yaml"))
    if candidates:
        return candidates[0]
    candidates = sorted(run_root.rglob("scoring_manifest.yaml"))
    return next((path for path in candidates if condition in path.parts), None)


def _run_root_from_path(path: Path, runs_root: Path) -> Path:
    try:
        relative = path.relative_to(runs_root)
        return runs_root / relative.parts[0]
    except (ValueError, IndexError):
        return path.parent


def _source_run_root(source_root: Path, run_id: str) -> Path:
    if (source_root / "runs" / run_id).exists():
        return source_root / "runs" / run_id
    if (source_root / run_id).exists():
        return source_root / run_id
    return source_root / "runs" / run_id


def _condition_from_path(path: Path, conditions: Sequence[str]) -> str:
    for condition in conditions:
        if condition in path.parts:
            return condition
    return ""


def _method_from_path(path: Path, default: str) -> str:
    if path.parent.name:
        return path.parent.name
    return default


def _resolve_path(value: str | Path | object | None, root: Path) -> Path:
    if value is None:
        return root / "__missing__"
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_value(mapping: Mapping[str, Any], *keys: str) -> object | None:
    for key in keys:
        if "." in key:
            value: object = mapping
            for part in key.split("."):
                if not isinstance(value, Mapping):
                    value = None
                    break
                value = value.get(part)
            if value is not None:
                return value
        elif key in mapping:
            return mapping[key]
    return None


def _first_text(*values: object | None) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def _first_int(*values: object | None) -> int | None:
    for value in values:
        converted = _as_int(value)
        if converted is not None:
            return converted
    return None


def _as_int(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_ints(values: Iterable[object]) -> list[int | None]:
    return [_as_int(value) for value in values]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _require_columns(
    state: _ValidationState,
    label: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        state.issue(label + ".columns", f"missing required columns {missing}")


def _column_alias(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    return next((alias for alias in aliases if alias in columns), None)


def _infer_key_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    candidates = (
        ("table", "held_out_label", "method", "score", "metric"),
        ("held_out_label", "seed", "condition_id", "method"),
        ("held_out_label", "seed", "method"),
        ("held_out_label", "method", "score"),
        ("held_out_label", "method", "quantity"),
        ("method", "score"),
        ("run_id", "condition_id", "method"),
        ("run_id", "method"),
    )
    for candidate in candidates:
        if set(candidate) <= set(frame.columns):
            return candidate
    return ()


def _compare_truth_frames(
    candidate: pd.DataFrame,
    source: pd.DataFrame,
    contract: DatasetContract,
) -> None:
    if contract.source_identity not in {"semantic", "exact"}:
        raise ValueError("source_identity must be 'semantic' or 'exact'")
    columns = (
        tuple(candidate.columns)
        if contract.source_identity == "exact"
        else tuple(column for column in contract.source_truth_columns if column in candidate.columns and column in source.columns)
    )
    if not columns:
        raise ValueError("authorized truth has no declared identity columns")
    missing = [column for column in contract.source_truth_columns if column not in candidate.columns or column not in source.columns]
    if contract.source_identity == "exact" and missing:
        raise KeyError(f"truth columns differ; missing {missing}")
    left = candidate.loc[:, columns].copy()
    right = source.loc[:, columns].copy()
    if "cell_id" not in columns:
        raise KeyError("truth identity must include cell_id")
    if left["cell_id"].astype(str).duplicated().any() or right["cell_id"].astype(str).duplicated().any():
        raise ValueError("truth identity contains duplicate cell_id values")
    left["cell_id"] = left["cell_id"].astype(str)
    right["cell_id"] = right["cell_id"].astype(str)
    left = left.sort_values("cell_id", kind="mergesort").reset_index(drop=True)
    right = right.sort_values("cell_id", kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)


# Mouse-spleen uses the current sparse profile, but its retained scientific
# scope is deliberately narrower than the reusable PBMC/HIHA contract above.
MOUSE_SPLEEN_DATASET = "mouse_spleen"
MOUSE_SPLEEN_ENDPOINT = "Proliferating"
MOUSE_SPLEEN_CONDITION = "natural_mismatch"
MOUSE_SPLEEN_METHOD = "celltypist_l3"
MOUSE_SPLEEN_PROFILE = "sparse_minibatch"
MOUSE_SPLEEN_REQUIRED_PARAMETERS = (
    "profile",
    "use_SGD",
    "with_mean",
    "mini_batch",
    "batch_number",
    "batch_size",
    "epochs",
    "n_jobs",
    "probability_path",
    "annotation_mode",
    "majority_voting",
    "probability_threshold",
    "random_state",
    "numpy_random_seed",
)
MOUSE_SPLEEN_CONTINUOUS_COLUMNS = (
    "z_absent_score",
    "max_confidence_or_similarity",
    "u",
    "max_label_probability",
    "score",
)


@dataclass(frozen=True)
class MouseSpleenCellTypistContract:
    """Strict observable contract for the mouse-spleen sentinel execution."""

    candidate_root: Path
    seed: int
    expected_query_cells: tuple[str, ...] = ()
    class_domain: tuple[str, ...] = ()
    profile_parameters: Mapping[str, object] = field(
        default_factory=lambda: {
            "profile": MOUSE_SPLEEN_PROFILE,
            "use_SGD": True,
            "with_mean": False,
            "mini_batch": True,
            "batch_number": 100,
            "batch_size": 1000,
            "epochs": 10,
            "n_jobs": 1,
            "probability_path": "sparse_model_probability",
            "annotation_mode": "best match",
            "majority_voting": False,
            "probability_threshold": 0.5,
        }
    )
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 0.0
    checksum_path: str | Path = "checksums.sha256"
    require_checksums: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_root", Path(self.candidate_root))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "expected_query_cells",
            tuple(str(value) for value in self.expected_query_cells),
        )
        object.__setattr__(
            self,
            "class_domain",
            tuple(str(value) for value in self.class_domain),
        )
        object.__setattr__(self, "profile_parameters", dict(self.profile_parameters))
        object.__setattr__(self, "checksum_path", Path(self.checksum_path))
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("repeatability tolerances must be nonnegative")

    @property
    def expected_unit(self) -> tuple[str, int, str]:
        return (MOUSE_SPLEEN_ENDPOINT, self.seed, MOUSE_SPLEEN_CONDITION)

    @property
    def parameters(self) -> dict[str, object]:
        return {
            **dict(self.profile_parameters),
            "random_state": self.seed,
            "numpy_random_seed": self.seed,
        }


def validate_mouse_spleen_celltypist_provenance(
    candidate_root: str | Path | MouseSpleenCellTypistContract,
    contract: MouseSpleenCellTypistContract | None = None,
) -> ValidationReport:
    """Validate one isolated Proliferating/natural-mismatch certificate.

    The function intentionally reads only observable candidate artifacts.  It
    never fits a model, writes a report, or modifies the candidate root.
    ``CellTypistPromotionValidationError.report`` contains every failed check
    so callers can distinguish scope, identity, membership, and repeatability
    failures.
    """

    if isinstance(candidate_root, MouseSpleenCellTypistContract):
        if contract is not None:
            raise TypeError("a positional contract cannot be combined with contract")
        contract = candidate_root
        root = contract.candidate_root.resolve()
    else:
        root = Path(candidate_root).resolve()
    if contract is None:
        contract = _infer_mouse_spleen_contract(root)
    elif root != contract.candidate_root.resolve():
        contract = replace_mouse_spleen_contract(contract, candidate_root=root)

    state = _ValidationState(MOUSE_SPLEEN_DATASET, root, (contract.expected_unit,))
    manifest_path = root / "manifest.yaml"
    manifest = _read_yaml(manifest_path, state, "mouse_spleen.top_manifest")
    if manifest is None:
        _raise_mouse_validation(state)
        raise AssertionError("unreachable")

    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        state.issue("mouse_spleen.manifest.metadata", "metadata must be a mapping")
        metadata = {}
    _validate_mouse_scope(state, metadata, contract)
    _validate_mouse_environment(state, metadata)
    _validate_mouse_nonmutation(state, metadata)
    config_path, config = _mouse_config(root, manifest, metadata, state)
    _validate_mouse_config(state, config, contract)
    _validate_mouse_top_profile(state, metadata, contract)
    _validate_mouse_checksums(state, root, contract)

    run_root, prediction_manifest_path, prediction_manifest = _mouse_prediction_unit(
        root, manifest, metadata, state
    )
    if prediction_manifest_path is not None:
        state.units.append(
            _Unit(
                run_root=run_root,
                run_id=run_root.name,
                condition=MOUSE_SPLEEN_CONDITION,
                method=MOUSE_SPLEEN_METHOD,
                endpoint=MOUSE_SPLEEN_ENDPOINT,
                seed=contract.seed,
                prediction_manifest_path=prediction_manifest_path,
                prediction_manifest=dict(prediction_manifest or {}),
            )
        )
    score_manifest_path, score_manifest = _mouse_scoring_unit(
        run_root, manifest, state
    )
    if prediction_manifest is not None:
        _validate_mouse_prediction_manifest(
            state,
            prediction_manifest,
            prediction_manifest_path,
            score_manifest,
            contract,
            root,
        )
    if score_manifest is not None:
        _validate_mouse_scoring_manifest(
            state, score_manifest, score_manifest_path, prediction_manifest, contract, root
        )

    query_ids = _mouse_query_ids(root, manifest, metadata, contract, state)
    prediction = _mouse_table_from_manifest(
        prediction_manifest, "predictions", prediction_manifest_path, root, state
    )
    scores = _mouse_table_from_manifest(
        score_manifest, "cell_scores", score_manifest_path, root, state
    )
    if prediction is not None:
        _validate_mouse_predictions(state, prediction, query_ids, contract)
    if scores is not None:
        _validate_mouse_scores(state, scores, query_ids, prediction, contract)
    if prediction is not None and scores is not None:
        _validate_mouse_prediction_score_identity(state, prediction, scores, contract)
    _validate_mouse_artifact_identities(
        state,
        root,
        manifest,
        metadata,
        prediction_manifest,
        prediction_manifest_path,
        score_manifest,
        score_manifest_path,
        config_path,
    )
    _validate_mouse_repeatability(state, root, manifest, metadata, prediction, contract)

    report = state.report()
    if not report.valid:
        raise CellTypistPromotionValidationError(
            f"Mouse-spleen CellTypist provenance validation failed for {root}: "
            + "; ".join(report.issues),
            report=report,
        )
    return report


def replace_mouse_spleen_contract(
    contract: MouseSpleenCellTypistContract, **changes: object
) -> MouseSpleenCellTypistContract:
    """Small local replacement helper avoiding a public dataclasses import."""

    values = {
        "candidate_root": contract.candidate_root,
        "seed": contract.seed,
        "expected_query_cells": contract.expected_query_cells,
        "class_domain": contract.class_domain,
        "profile_parameters": contract.profile_parameters,
        "absolute_tolerance": contract.absolute_tolerance,
        "relative_tolerance": contract.relative_tolerance,
        "checksum_path": contract.checksum_path,
        "require_checksums": contract.require_checksums,
    }
    values.update(changes)
    return MouseSpleenCellTypistContract(**values)


def _infer_mouse_spleen_contract(root: Path) -> MouseSpleenCellTypistContract:
    path = root / "manifest.yaml"
    if not path.is_file():
        raise ValueError(f"mouse-spleen provenance manifest does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("mouse-spleen provenance metadata must be a mapping")
    seeds = metadata.get("seeds", ())
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or len(seeds) != 1:
        raise ValueError("mouse-spleen provenance must declare exactly one seed")
    try:
        seed = int(seeds[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("mouse-spleen provenance seed must be an integer") from exc
    class_domain = metadata.get("class_domain", ())
    if not isinstance(class_domain, Sequence) or isinstance(class_domain, (str, bytes)):
        raise ValueError("mouse-spleen provenance must declare class_domain")
    tolerance = metadata.get("repeatability_tolerance", {})
    if not isinstance(tolerance, Mapping):
        tolerance = {}
    return MouseSpleenCellTypistContract(
        candidate_root=root,
        seed=seed,
        expected_query_cells=tuple(str(value) for value in metadata.get("query_cell_ids", ())),
        class_domain=tuple(str(value) for value in class_domain),
        absolute_tolerance=float(tolerance.get("absolute", 1e-12)),
        relative_tolerance=float(tolerance.get("relative", 0.0)),
    )


def _raise_mouse_validation(state: _ValidationState) -> None:
    report = state.report()
    raise CellTypistPromotionValidationError(
        f"Mouse-spleen CellTypist provenance validation failed for {state.root}",
        report=report,
    )


def _mouse_scope(
    state: _ValidationState,
    label: str,
    observed: object,
    expected: object,
) -> None:
    state.check(label, observed == expected, f"expected {expected!r}, got {observed!r}")


def _validate_mouse_scope(
    state: _ValidationState,
    metadata: Mapping[str, Any],
    contract: MouseSpleenCellTypistContract,
) -> None:
    _mouse_scope(state, "mouse_spleen.scope.dataset", str(metadata.get("dataset", "")), MOUSE_SPLEEN_DATASET)
    _mouse_scope(state, "mouse_spleen.scope.endpoint", str(metadata.get("endpoint", "")), MOUSE_SPLEEN_ENDPOINT)
    _mouse_scope(state, "mouse_spleen.scope.condition", str(metadata.get("condition", "")), MOUSE_SPLEEN_CONDITION)
    _mouse_scope(state, "mouse_spleen.scope.method", str(metadata.get("method", "")), MOUSE_SPLEEN_METHOD)
    _mouse_scope(state, "mouse_spleen.scope.profile", str(metadata.get("profile", "")), MOUSE_SPLEEN_PROFILE)
    for key, expected in (
        ("endpoints", (MOUSE_SPLEEN_ENDPOINT,)),
        ("conditions", (MOUSE_SPLEEN_CONDITION,)),
        ("methods", (MOUSE_SPLEEN_METHOD,)),
        ("seeds", (contract.seed,)),
    ):
        observed = metadata.get(key, ())
        try:
            observed_value = tuple(str(value) for value in observed) if key != "seeds" else tuple(int(value) for value in observed)
        except (TypeError, ValueError):
            observed_value = ()
        _mouse_scope(state, f"mouse_spleen.scope.{key}", observed_value, expected)
    state.check(
        "mouse_spleen.execution_policy",
        metadata.get("execution_policy") == "global_serial",
        "expected execution_policy='global_serial'",
    )


def _validate_mouse_top_profile(
    state: _ValidationState,
    metadata: Mapping[str, Any],
    contract: MouseSpleenCellTypistContract,
) -> None:
    raw = metadata.get("profile_by_seed") or metadata.get("celltypist_parameters_by_seed")
    block: Mapping[str, Any] | None = None
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if _as_int(key) == contract.seed and isinstance(value, Mapping):
                block = value
                break
    if block is None:
        state.issue("mouse_spleen.profile", "missing per-seed sparse-minibatch profile")
        return
    _validate_mouse_parameters(state, "mouse_spleen.profile", block, contract)


def _validate_mouse_parameters(
    state: _ValidationState,
    label: str,
    parameters: Mapping[str, Any],
    contract: MouseSpleenCellTypistContract,
) -> None:
    expected = contract.parameters
    for key in MOUSE_SPLEEN_REQUIRED_PARAMETERS:
        actual = parameters.get(key)
        wanted = expected.get(key)
        if key in {"random_state", "numpy_random_seed", "batch_number", "batch_size", "epochs", "n_jobs"}:
            actual = _as_int(actual)
            wanted = _as_int(wanted)
        elif key == "probability_threshold":
            try:
                actual = float(actual)
                wanted = float(wanted)
            except (TypeError, ValueError):
                pass
        state.check(
            f"{label}.{key}",
            actual == wanted,
            f"expected {wanted!r}, got {actual!r}",
        )


def _mouse_config(
    root: Path,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    state: _ValidationState,
) -> tuple[Path | None, Mapping[str, Any]]:
    artifacts = manifest.get("artifacts", {})
    value = artifacts.get("resolved_config") if isinstance(artifacts, Mapping) else None
    value = value or metadata.get("resolved_config")
    candidates = []
    if value is not None:
        candidates.append(_resolve_path(value, root))
    candidates.extend((root / "resolved_config.yaml", root / "config" / "resolved_config.yaml"))
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0] if candidates else None)
    if path is None or not path.is_file():
        state.issue("mouse_spleen.resolved_config", f"resolved config does not exist: {path}")
        return path, {}
    payload = _read_yaml(path, state, "mouse_spleen.resolved_config") or {}
    return path, payload


def _validate_mouse_config(
    state: _ValidationState,
    config: Mapping[str, Any],
    contract: MouseSpleenCellTypistContract,
) -> None:
    if not config:
        return
    _mouse_scope(state, "mouse_spleen.config.endpoint", config.get("heldout_label", config.get("endpoint")), MOUSE_SPLEEN_ENDPOINT)
    conditions = tuple(str(value) for value in config.get("conditions", ()))
    _mouse_scope(state, "mouse_spleen.config.conditions", conditions, (MOUSE_SPLEEN_CONDITION,))
    methods = tuple(str(value) for value in config.get("methods", ()))
    _mouse_scope(state, "mouse_spleen.config.methods", methods, (MOUSE_SPLEEN_METHOD,))
    observed_seed = _first_int(config.get("repeat"), _mapping_value(config, "split.seed", "seed"))
    _mouse_scope(state, "mouse_spleen.config.seed", observed_seed, contract.seed)
    celltypist = config.get("celltypist", {})
    if not isinstance(celltypist, Mapping):
        state.issue("mouse_spleen.config.celltypist", "celltypist must be a mapping")
        return
    _validate_mouse_parameters(state, "mouse_spleen.config.celltypist", celltypist, contract)


def _validate_mouse_environment(state: _ValidationState, metadata: Mapping[str, Any]) -> None:
    environment = metadata.get("environment")
    if not isinstance(environment, Mapping):
        state.issue("mouse_spleen.environment", "environment must be a mapping")
        return
    for key in ("python", "celltypist", "numpy", "platform"):
        value = environment.get(key)
        state.check(
            f"mouse_spleen.environment.{key}",
            value is not None and str(value).strip() != "",
            "material environment identity is missing",
        )
    packages = environment.get("packages")
    state.check(
        "mouse_spleen.environment.packages",
        isinstance(packages, Mapping) and bool(packages),
        "material package versions are missing",
    )
    workers = environment.get("workers", environment.get("worker_count"))
    state.check("mouse_spleen.environment.workers", _as_int(workers) == 1, "expected one fit worker")
    threads = environment.get("threads", environment.get("thread_settings"))
    state.check(
        "mouse_spleen.environment.threads",
        isinstance(threads, Mapping) and bool(threads),
        "effective thread settings are missing",
    )


def _validate_mouse_nonmutation(state: _ValidationState, metadata: Mapping[str, Any]) -> None:
    state.check("mouse_spleen.isolated_output", metadata.get("isolated_output") is True, "candidate output must be isolated")
    nonmutation = metadata.get("nonmutation")
    if not isinstance(nonmutation, Mapping):
        state.issue("mouse_spleen.nonmutation", "nonmutation record is missing")
        return
    for key in ("canonical_results", "manuscript_sources", "historical_evidence", "external_data"):
        state.check(
            f"mouse_spleen.nonmutation.{key}",
            nonmutation.get(key) is False,
            "isolated validation must not mutate this root",
        )


def _mouse_prediction_unit(
    root: Path,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    state: _ValidationState,
) -> tuple[Path, Path | None, Mapping[str, Any] | None]:
    runs_value = _artifact_value(manifest, "runs")
    runs = _resolve_path(runs_value, root) if runs_value is not None else root / "runs"
    if not runs.is_dir():
        state.issue("mouse_spleen.units", f"runs root does not exist: {runs}")
        return runs, None, None
    paths = sorted({*runs.rglob("external_baseline_manifest.yaml"), *runs.rglob("prediction_manifest.yaml")})
    state.check("mouse_spleen.units.count", len(paths) == 1, f"expected one prediction manifest, got {len(paths)}")
    if not paths:
        return runs, None, None
    path = paths[0]
    run_root = runs / path.relative_to(runs).parts[0]
    payload = _read_yaml(path, state, "mouse_spleen.prediction_manifest") or {}
    return run_root, path, payload


def _mouse_scoring_unit(
    run_root: Path,
    manifest: Mapping[str, Any],
    state: _ValidationState,
) -> tuple[Path | None, Mapping[str, Any] | None]:
    value = _artifact_value(manifest, "scoring_manifest")
    candidates = [_resolve_path(value, state.root)] if value is not None else []
    if run_root.is_dir():
        candidates.extend(sorted(run_root.rglob("scoring_manifest.yaml")))
    candidates = list(dict.fromkeys(path for path in candidates if path.is_file()))
    state.check("mouse_spleen.scoring_manifest.count", len(candidates) == 1, f"expected one scoring manifest, got {len(candidates)}")
    if not candidates:
        return None, None
    path = candidates[0]
    payload = _read_yaml(path, state, "mouse_spleen.scoring_manifest") or {}
    return path, payload


def _validate_mouse_prediction_manifest(
    state: _ValidationState,
    manifest: Mapping[str, Any],
    path: Path | None,
    scoring_manifest: Mapping[str, Any] | None,
    contract: MouseSpleenCellTypistContract,
    root: Path,
) -> None:
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        state.issue("mouse_spleen.prediction_manifest.metadata", "metadata must be a mapping")
        return
    for key, expected in (("condition", MOUSE_SPLEEN_CONDITION), ("method", MOUSE_SPLEEN_METHOD), ("endpoint", MOUSE_SPLEEN_ENDPOINT), ("heldout_label", MOUSE_SPLEEN_ENDPOINT)):
        _mouse_scope(state, f"mouse_spleen.prediction_manifest.{key}", metadata.get(key), expected)
    _mouse_scope(state, "mouse_spleen.prediction_manifest.seed", _first_int(metadata.get("seed"), metadata.get("repeat")), contract.seed)
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        state.issue("mouse_spleen.prediction_manifest.parameters", "resolved parameters are missing")
    else:
        _validate_mouse_parameters(state, "mouse_spleen.prediction_manifest.parameters", parameters, contract)
    class_domain = metadata.get("class_domain")
    _mouse_scope(state, "mouse_spleen.prediction_manifest.class_domain", tuple(str(v) for v in class_domain) if isinstance(class_domain, Sequence) and not isinstance(class_domain, (str, bytes)) else (), contract.class_domain)
    preprocessing = metadata.get("preprocessing")
    state.check("mouse_spleen.prediction_manifest.preprocessing", isinstance(preprocessing, Mapping) and bool(preprocessing), "preprocessing settings are missing")
    prediction_settings = metadata.get("prediction")
    state.check("mouse_spleen.prediction_manifest.prediction", isinstance(prediction_settings, Mapping) and bool(prediction_settings), "prediction settings are missing")
    if scoring_manifest is not None:
        score_meta = scoring_manifest.get("metadata", {})
        if isinstance(score_meta, Mapping):
            state.check(
                "mouse_spleen.manifest.parameter_agreement",
                score_meta.get("method_parameters", {}).get(MOUSE_SPLEEN_METHOD) == parameters
                if isinstance(score_meta.get("method_parameters"), Mapping)
                else False,
                "prediction and scoring parameters differ",
            )


def _validate_mouse_scoring_manifest(
    state: _ValidationState,
    manifest: Mapping[str, Any],
    path: Path | None,
    prediction_manifest: Mapping[str, Any] | None,
    contract: MouseSpleenCellTypistContract,
    root: Path,
) -> None:
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        state.issue("mouse_spleen.scoring_manifest.metadata", "metadata must be a mapping")
        return
    _mouse_scope(state, "mouse_spleen.scoring_manifest.condition", metadata.get("condition"), MOUSE_SPLEEN_CONDITION)
    _mouse_scope(state, "mouse_spleen.scoring_manifest.endpoint", metadata.get("endpoint"), MOUSE_SPLEEN_ENDPOINT)
    _mouse_scope(state, "mouse_spleen.scoring_manifest.method", tuple(str(v) for v in metadata.get("methods", ())), (MOUSE_SPLEEN_METHOD,))
    _mouse_scope(state, "mouse_spleen.scoring_manifest.seed", _first_int(metadata.get("seed"), metadata.get("repeat")), contract.seed)
    parameters = metadata.get("method_parameters", {})
    score_parameters = parameters.get(MOUSE_SPLEEN_METHOD) if isinstance(parameters, Mapping) else None
    if not isinstance(score_parameters, Mapping):
        state.issue("mouse_spleen.scoring_manifest.parameters", "resolved method parameters are missing")
    else:
        _validate_mouse_parameters(state, "mouse_spleen.scoring_manifest.parameters", score_parameters, contract)
    for key in ("preprocessing", "prediction"):
        state.check(
            f"mouse_spleen.scoring_manifest.{key}",
            isinstance(metadata.get(key), Mapping) and bool(metadata.get(key)),
            f"{key} settings are missing",
        )


def _mouse_table_from_manifest(
    manifest: Mapping[str, Any] | None,
    name: str,
    manifest_path: Path | None,
    root: Path,
    state: _ValidationState,
) -> pd.DataFrame | None:
    if manifest is None or manifest_path is None:
        state.issue(f"mouse_spleen.{name}", "manifest is unavailable")
        return None
    path = _artifact_value(manifest, name)
    fallback = manifest_path.parent / ("predictions.csv" if name == "predictions" else "cell_scores.parquet")
    artifact = _resolve_path(path, root) if path is not None else fallback
    if not artifact.is_file():
        state.issue(f"mouse_spleen.{name}", f"artifact does not exist: {artifact}")
        return None
    try:
        return _read_table(artifact)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        state.issue(f"mouse_spleen.{name}", f"cannot read artifact: {exc}")
        return None


def _mouse_query_ids(
    root: Path,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    contract: MouseSpleenCellTypistContract,
    state: _ValidationState,
) -> tuple[str, ...]:
    if contract.expected_query_cells:
        ids = contract.expected_query_cells
    else:
        raw = metadata.get("query_cell_ids", ())
        ids = tuple(str(value) for value in raw) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ()
    if not ids:
        value = _artifact_value(manifest, "model_visible_cells") or _artifact_value(manifest, "input")
        path = _resolve_path(value, root) if value is not None else root / "inputs" / "model_visible" / "cells.csv"
        if path.is_dir():
            path = path / "cells.csv"
        try:
            frame = _read_table(path)
            if {"cell_id", "domain"} <= set(frame.columns):
                ids = tuple(frame.loc[frame["domain"].astype(str).eq("query"), "cell_id"].astype(str))
            elif "cell_id" in frame.columns:
                ids = tuple(frame["cell_id"].astype(str))
        except (OSError, ValueError, pd.errors.ParserError):
            ids = ()
    state.check("mouse_spleen.query_cells.present", bool(ids), "query-cell membership is unavailable")
    state.check("mouse_spleen.query_cells.unique", len(set(ids)) == len(ids), "query-cell membership has duplicates")
    return ids


def _validate_mouse_predictions(
    state: _ValidationState,
    frame: pd.DataFrame,
    query_ids: tuple[str, ...],
    contract: MouseSpleenCellTypistContract,
) -> None:
    required = ("cell_id", "method", "heldout_label", "repeat", "pred_label", "native_pred_label")
    _require_columns(state, "mouse_spleen.predictions", frame, required)
    if not set(required) <= set(frame.columns):
        return
    ids = tuple(frame["cell_id"].astype(str))
    state.check("mouse_spleen.predictions.cell_ids_unique", len(set(ids)) == len(ids), "duplicate cell_id values")
    state.check("mouse_spleen.predictions.row_order", ids == query_ids, f"expected query order {query_ids!r}, got {ids!r}")
    state.check("mouse_spleen.predictions.method", set(frame["method"].astype(str)) == {MOUSE_SPLEEN_METHOD}, "method drift")
    state.check("mouse_spleen.predictions.endpoint", set(frame["heldout_label"].astype(str)) == {MOUSE_SPLEEN_ENDPOINT}, "endpoint drift")
    state.check("mouse_spleen.predictions.seed", set(_numeric_ints(frame["repeat"])) == {contract.seed}, "seed drift")
    if "condition" in frame.columns:
        state.check("mouse_spleen.predictions.condition", set(frame["condition"].astype(str)) == {MOUSE_SPLEEN_CONDITION}, "condition drift")
    if "is_full_reference_control" in frame.columns:
        state.check("mouse_spleen.predictions.condition_flag", set(frame["is_full_reference_control"].map(_as_bool)) == {False}, "natural-mismatch condition flag drift")
    domain = set(contract.class_domain)
    for column in ("pred_label", "native_pred_label"):
        state.check(
            f"mouse_spleen.predictions.{column}.class_domain",
            set(frame[column].astype(str)) <= domain,
            f"{column} contains labels outside declared class domain",
        )
    for column in MOUSE_SPLEEN_CONTINUOUS_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            state.check(
                f"mouse_spleen.predictions.{column}.finite",
                bool(np.isfinite(values).all()),
                f"{column} contains nonfinite scores",
            )


def _validate_mouse_scores(
    state: _ValidationState,
    frame: pd.DataFrame,
    query_ids: tuple[str, ...],
    prediction: pd.DataFrame | None,
    contract: MouseSpleenCellTypistContract,
) -> None:
    required = ("cell_id", "method")
    _require_columns(state, "mouse_spleen.scores", frame, required)
    if not set(required) <= set(frame.columns):
        return
    ids = tuple(frame["cell_id"].astype(str))
    state.check("mouse_spleen.scores.cell_ids_unique", len(set(ids)) == len(ids), "duplicate cell_id values")
    state.check("mouse_spleen.scores.row_order", ids == query_ids, f"expected query order {query_ids!r}, got {ids!r}")
    state.check("mouse_spleen.scores.method", set(frame["method"].astype(str)) == {MOUSE_SPLEEN_METHOD}, "method drift")
    for identity, expected, aliases in (
        ("endpoint", MOUSE_SPLEEN_ENDPOINT, ("endpoint", "heldout_label", "removed_state")),
        ("condition", MOUSE_SPLEEN_CONDITION, ("condition", "condition_id")),
    ):
        column = next((candidate for candidate in aliases if candidate in frame.columns), None)
        if column is not None:
            state.check(f"mouse_spleen.scores.{identity}", set(frame[column].astype(str)) == {expected}, f"{identity} drift")
    numeric_columns = [column for column in MOUSE_SPLEEN_CONTINUOUS_COLUMNS if column in frame.columns]
    state.check("mouse_spleen.scores.score_column", bool(numeric_columns), "no continuous score column is declared")
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        state.check(
            f"mouse_spleen.scores.{column}.finite",
            bool(np.isfinite(values).all()),
            f"{column} contains nonfinite scores",
        )
    if prediction is not None and "u" in frame.columns and "z_absent_score" in prediction.columns:
        left = pd.to_numeric(frame["u"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(prediction["z_absent_score"], errors="coerce").to_numpy(dtype=float)
        state.check("mouse_spleen.prediction_score.values", bool(np.array_equal(left, right)), "prediction and score values differ")


def _validate_mouse_prediction_score_identity(
    state: _ValidationState,
    prediction: pd.DataFrame,
    scores: pd.DataFrame,
    contract: MouseSpleenCellTypistContract,
) -> None:
    state.check(
        "mouse_spleen.prediction_score.membership",
        tuple(prediction["cell_id"].astype(str)) == tuple(scores["cell_id"].astype(str)),
        "prediction and score cell membership/order differ",
    )


def _validate_mouse_repeatability(
    state: _ValidationState,
    root: Path,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    prediction: pd.DataFrame | None,
    contract: MouseSpleenCellTypistContract,
) -> None:
    value = _artifact_value(manifest, "repeatability_report") or metadata.get("repeatability_report")
    report_path = _resolve_path(value, root) if value is not None else root / "repeatability" / "report.yaml"
    report = _read_yaml(report_path, state, "mouse_spleen.repeatability.report")
    if report is None:
        return
    tolerance = report.get("tolerance", {})
    if not isinstance(tolerance, Mapping):
        tolerance = {}
    try:
        absolute = float(tolerance.get("absolute"))
        relative = float(tolerance.get("relative"))
    except (TypeError, ValueError):
        absolute = relative = math.nan
    state.check("mouse_spleen.repeatability.tolerance.absolute", absolute == contract.absolute_tolerance, "absolute tolerance drift")
    state.check("mouse_spleen.repeatability.tolerance.relative", relative == contract.relative_tolerance, "relative tolerance drift")
    state.check("mouse_spleen.repeatability.status", report.get("status") == "passed" and report.get("passed") is True, "repeatability certificate did not pass")
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    value = artifacts.get("repeated_predictions") or _artifact_value(manifest, "repeated_predictions")
    repeated_path = _resolve_path(value, root) if value is not None else root / "repeatability" / "repeated_predictions.csv"
    repeated = None
    if not repeated_path.is_file():
        state.issue("mouse_spleen.repeatability.predictions", f"repeated predictions do not exist: {repeated_path}")
    else:
        try:
            repeated = _read_table(repeated_path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            state.issue("mouse_spleen.repeatability.predictions", f"cannot read repeated predictions: {exc}")
    if repeated is None or prediction is None:
        return
    repeated_ids = tuple(repeated.get("cell_id", ()))
    prediction_ids = tuple(prediction.get("cell_id", ()))
    state.check(
        "mouse_spleen.repeatability.membership",
        repeated_ids == prediction_ids,
        "repeated prediction cell membership/order differs",
    )
    state.check("mouse_spleen.repeatability.labels", tuple(repeated.get("pred_label", ())) == tuple(prediction.get("pred_label", ())), "predicted labels differ across clean processes")
    for column in ("z_absent_score", "max_confidence_or_similarity"):
        if column not in repeated.columns or column not in prediction.columns:
            state.issue(f"mouse_spleen.repeatability.{column}", "continuous comparison column is missing")
            continue
        left = pd.to_numeric(prediction[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(repeated[column], errors="coerce").to_numpy(dtype=float)
        if left.shape != right.shape:
            state.issue(f"mouse_spleen.repeatability.{column}", "continuous comparison membership differs")
            continue
        difference = np.abs(left - right)
        denominator = np.maximum(np.abs(left), np.finfo(float).tiny)
        relative_difference = difference / denominator
        max_absolute = float(np.max(difference)) if difference.size else 0.0
        max_relative = float(np.max(relative_difference)) if difference.size else 0.0
        state.check(
            f"mouse_spleen.repeatability.{column}",
            bool(np.allclose(left, right, atol=contract.absolute_tolerance, rtol=contract.relative_tolerance, equal_nan=False)),
            f"max absolute={max_absolute}, max relative={max_relative}",
        )
        observed = report.get("comparison", {})
        observed = observed.get(column, {}) if isinstance(observed, Mapping) else {}
        if isinstance(observed, Mapping):
            state.check(
                f"mouse_spleen.repeatability.{column}.report",
                float(observed.get("max_absolute_difference", math.nan)) == max_absolute
                and float(observed.get("max_relative_difference", math.nan)) == max_relative,
                "repeatability report does not record observed maximum differences",
            )


def _validate_mouse_artifact_identities(
    state: _ValidationState,
    root: Path,
    top_manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any] | None,
    prediction_manifest_path: Path | None,
    score_manifest: Mapping[str, Any] | None,
    score_manifest_path: Path | None,
    config_path: Path | None,
) -> None:
    identities = metadata.get("identities")
    if not isinstance(identities, Mapping):
        state.issue("mouse_spleen.identities", "artifact identity mapping is missing")
        return
    for name in ("input", "runner", "prediction", "score", "trained_model"):
        value = identities.get(name)
        if not isinstance(value, Mapping):
            state.issue(f"mouse_spleen.identity.{name}", "identity is missing")
            continue
        retained = value.get("retained", True)
        if name == "trained_model" and retained is False:
            state.check("mouse_spleen.identity.trained_model", True)
            continue
        path_value = value.get("path")
        digest = str(value.get("sha256", ""))
        path = _resolve_path(path_value, root) if path_value is not None else root / "__missing__"
        actual = _sha256(path) if path.is_file() else ""
        state.check(f"mouse_spleen.identity.{name}.path", path.is_file(), f"artifact does not exist: {path}")
        state.check(f"mouse_spleen.identity.{name}.sha256", bool(re.fullmatch(r"[0-9a-fA-F]{64}", digest)) and actual.lower() == digest.lower(), "artifact checksum identity mismatch")
    artifact_paths = {
        "prediction": _artifact_path_from_nested(prediction_manifest, "predictions", prediction_manifest_path, root),
        "score": _artifact_path_from_nested(score_manifest, "cell_scores", score_manifest_path, root),
    }
    for name, expected in artifact_paths.items():
        value = identities.get(name)
        if isinstance(value, Mapping) and expected is not None:
            state.check(
                f"mouse_spleen.identity.{name}.path_agreement",
                _resolve_path(value.get("path"), root) == expected,
                "identity path differs from manifest artifact",
            )
    if config_path is not None:
        value = top_manifest.get("artifacts", {})
        if isinstance(value, Mapping) and value.get("resolved_config") is not None:
            state.check(
                "mouse_spleen.identity.resolved_config",
                _resolve_path(value.get("resolved_config"), root) == config_path,
                "resolved config identity differs from top manifest",
            )


def _artifact_path_from_nested(
    manifest: Mapping[str, Any] | None,
    name: str,
    manifest_path: Path | None,
    root: Path,
) -> Path | None:
    if manifest is None or manifest_path is None:
        return None
    value = _artifact_value(manifest, name)
    fallback = manifest_path.parent / ("predictions.csv" if name == "predictions" else "cell_scores.parquet")
    return _resolve_path(value, root) if value is not None else fallback


def _validate_mouse_checksums(
    state: _ValidationState,
    root: Path,
    contract: MouseSpleenCellTypistContract,
) -> None:
    path = _resolve_path(contract.checksum_path, root)
    if not path.is_file():
        if contract.require_checksums:
            state.issue("mouse_spleen.checksums", f"checksum manifest does not exist: {path}")
        return
    declared: dict[Path, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+[* ]?(.+)", line)
        if match is None:
            state.issue("mouse_spleen.checksums", f"malformed checksum line {line_number}")
            continue
        digest, raw = match.groups()
        member = _resolve_path(raw, root)
        if member in declared:
            state.issue("mouse_spleen.checksums", f"duplicate checksum path: {raw}")
        declared[member] = digest
        state.check("mouse_spleen.checksums." + raw, member.is_file(), f"declared file does not exist: {raw}")
        if member.is_file():
            state.check("mouse_spleen.checksums." + raw + ".digest", _sha256(member).lower() == digest.lower(), "checksum mismatch")
    actual = {item for item in root.rglob("*") if item.is_file() and item != path}
    state.check("mouse_spleen.checksums.inventory", set(declared) == actual, f"checksum inventory differs: missing={sorted(str(item.relative_to(root)) for item in actual - set(declared))}, extra={sorted(str(item.relative_to(root)) for item in set(declared) - actual)}")


__all__ = [
    "CandidateValidationError",
    "CandidateValidationReport",
    "CellTypistPromotionValidationError",
    "MouseSpleenCellTypistContract",
    "MOUSE_SPLEEN_CONDITION",
    "MOUSE_SPLEEN_DATASET",
    "MOUSE_SPLEEN_ENDPOINT",
    "MOUSE_SPLEEN_METHOD",
    "MOUSE_SPLEEN_PROFILE",
    "DatasetCandidateContract",
    "DatasetContract",
    "ProtectedComparison",
    "ProtectedOutputContract",
    "ProtectedOutputReport",
    "TableContract",
    "ValidationCheck",
    "ValidationReport",
    "assert_protected_outputs",
    "compare_protected_outputs",
    "validate_candidate",
    "validate_celltypist_candidate",
    "validate_dataset_candidate",
    "validate_mouse_spleen_celltypist_provenance",
]

# Compatibility names keep the domain terminology discoverable without
# duplicating implementations in dataset-specific callers.
CandidateValidationError = CellTypistPromotionValidationError
CandidateValidationReport = ValidationReport
DatasetCandidateContract = DatasetContract
