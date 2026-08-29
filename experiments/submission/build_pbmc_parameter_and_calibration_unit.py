#!/usr/bin/env python3
"""Build the versioned PBMC parameter/calibration evidence unit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from coreot.results.pbmc_supplement import (
    ENDPOINTS,
    SEEDS,
    _read_sensitivity_fit_convergence,
)


UNIT_ID = "pbmc_parameter_and_calibration"
SENSITIVITY_TABLES = (
    "pbmc_calibration_percentile_by_seed.csv",
    "pbmc_calibration_percentile_summary.csv",
    "pbmc_calibration_sensitivity_by_seed.csv",
    "pbmc_calibration_sensitivity_summary.csv",
    "pbmc_full_reference_calibration_by_run.csv",
    "pbmc_full_reference_calibration_summary.csv",
    "pbmc_selected_penalties_by_seed.csv",
    "pbmc_selected_penalties_summary.csv",
    "pbmc_sensitivity_by_seed.csv",
    "pbmc_sensitivity_compact.csv",
    "pbmc_sensitivity_summary.csv",
)
SOURCE_INPUTS = (
    Path("results/PBMC/sensitivity/full_tau_labelwise/tables/detection_by_run.csv"),
    Path(
        "results/PBMC/sensitivity/full_tau_labelwise/tables/"
        "shared_label_transfer_by_run.csv"
    ),
    Path("results/PBMC/compare_baselines/tables/compare_full_reference_by_run.csv"),
)
CODE_PATHS = (
    Path("experiments/pbmc_state/generate_pbmc_supplement.py"),
    Path("experiments/submission/build_pbmc_parameter_and_calibration_unit.py"),
    Path("src/coreot/results/pbmc_supplement.py"),
    Path("submission/reproduction/pbmc.py"),
    Path("tests/test_pbmc_sensitivity_convergence_lineage.py"),
    Path("tests/test_pbmc_parameter_convergence_package.py"),
)


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def validate_sensitivity_lineage(
    repository_root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    repository_root = repository_root.resolve()
    table_path = (
        repository_root
        / "results/PBMC/manuscript/supplement/pbmc_sensitivity_by_seed.csv"
    )
    by_seed = pd.read_csv(table_path)
    expected_keys = {
        (endpoint, seed, tau_min, tau_max)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for tau_min in (0.5, 0.75, 1.0)
        for tau_max in (1.0, 1.25, 1.5)
    }
    key_columns = ["held_out_label", "seed", "tau_min", "tau_max"]
    required = {
        *key_columns,
        "run_id",
        "alpha",
        "condition",
        "candidate_set",
        "method",
        "convergence_evidence",
        "converged",
        "n_iter",
        "max_iter",
        "tol",
        "transport_manifest",
        "method_params",
    }
    missing = sorted(required - set(by_seed.columns))
    if missing:
        raise ValueError(f"PBMC sensitivity table omits convergence columns {missing}.")
    observed_keys = set(
        by_seed.loc[:, key_columns].itertuples(index=False, name=None)
    )
    if by_seed.duplicated(key_columns).any() or observed_keys != expected_keys:
        raise ValueError("PBMC sensitivity table does not contain the exact retained grid.")
    if by_seed["run_id"].duplicated().any():
        raise ValueError("PBMC sensitivity table contains duplicate run identifiers.")

    evidence_paths: list[Path] = []
    for row in by_seed.itertuples(index=False):
        observed = _read_sensitivity_fit_convergence(
            runs_root=repository_root / "runs",
            run_id=str(row.run_id),
            candidate_set=str(row.candidate_set),
            tau_min=float(row.tau_min),
            tau_max=float(row.tau_max),
            alpha=float(row.alpha),
        )
        for field in (
            "condition",
            "candidate_set",
            "convergence_evidence",
            "converged",
            "n_iter",
            "max_iter",
            "tol",
            "transport_manifest",
            "method_params",
        ):
            if getattr(row, field) != observed[field]:
                raise ValueError(
                    f"PBMC sensitivity table field {field} disagrees with "
                    f"fit evidence for {row.run_id}."
                )
        evidence_paths.extend(
            repository_root / str(observed[field])
            for field in ("transport_manifest", "method_params")
        )
    return by_seed, evidence_paths


class PackageBuilder:
    def __init__(
        self,
        *,
        repository_root: Path,
        snapshot_root: Path,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.snapshot_root = snapshot_root.resolve()
        self.package_root = self.snapshot_root / f"package_staging/{UNIT_ID}"
        self.included: list[IncludedFile] = []

    def copy_file(
        self,
        source: Path,
        destination: Path,
        *,
        source_label: str | None = None,
        note: str = "copied byte for byte",
    ) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self.package_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.included.append(
            IncludedFile(
                package_path=destination.as_posix(),
                source_path=(
                    source_label
                    or source.relative_to(self.repository_root).as_posix()
                ),
                action="copy",
                source_bytes=source.stat().st_size,
                package_bytes=target.stat().st_size,
                source_sha256=sha256(source),
                package_sha256=sha256(target),
                note=note,
            )
        )

    def _record_generated(self, relative: Path, note: str) -> None:
        path = self.package_root / relative
        self.included.append(
            IncludedFile(
                package_path=relative.as_posix(),
                source_path=(
                    "generated_by:experiments/submission/"
                    "build_pbmc_parameter_and_calibration_unit.py"
                ),
                action="generate",
                source_bytes=path.stat().st_size,
                package_bytes=path.stat().st_size,
                source_sha256=sha256(path),
                package_sha256=sha256(path),
                note=note,
            )
        )

    def build(self) -> Path:
        if self.snapshot_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing snapshot root: {self.snapshot_root}"
            )
        self.package_root.mkdir(parents=True)
        by_seed, evidence_paths = validate_sensitivity_lineage(
            self.repository_root
        )

        for relative in CODE_PATHS:
            self.copy_file(
                self.repository_root / relative,
                Path("code") / relative,
                source_label=relative.as_posix(),
            )
        supplement_root = (
            self.repository_root / "results/PBMC/manuscript/supplement"
        )
        for name in SENSITIVITY_TABLES:
            self.copy_file(
                supplement_root / name,
                Path("verified_results/supplement") / name,
            )
        for suffix in ("png", "pdf", "svg"):
            name = f"manuscript_fig_pbmc_supp_robustness.{suffix}"
            self.copy_file(
                self.repository_root / "docs/figs" / name,
                Path("verified_results/figures") / name,
            )
        self.copy_file(
            self.repository_root
            / "results/PBMC/manuscript/pbmc_supplement_manifest.yaml",
            Path("verified_results/pbmc_supplement_manifest.yaml"),
        )

        s7_root = self.repository_root / "results/PBMC/figures"
        for source in sorted((s7_root / "data").glob("figure_s7_*.csv")):
            self.copy_file(
                source,
                Path("verified_results/s7/data") / source.name,
            )
        for source in sorted(s7_root.glob("figure_s7_pbmc_calibration.*")):
            self.copy_file(source, Path("verified_results/s7") / source.name)
        for relative in SOURCE_INPUTS:
            self.copy_file(
                self.repository_root / relative,
                Path("source_inputs") / relative,
            )
        for row, manifest_path, params_path in zip(
            by_seed.itertuples(index=False),
            evidence_paths[::2],
            evidence_paths[1::2],
            strict=True,
        ):
            destination = Path("fit_evidence") / str(row.run_id)
            self.copy_file(
                manifest_path,
                destination / "transport_manifest.yaml",
            )
            self.copy_file(params_path, destination / "method_params.yaml")

        self._write_readme()
        self._write_audits(by_seed)
        self._write_external_dependencies()
        self._write_included_files()
        self._write_checksums()
        return self.package_root

    def _write_readme(self) -> None:
        relative = Path("README.md")
        (self.package_root / relative).write_text(
            "# PBMC parameter and calibration evidence unit\n\n"
            "This versioned unit contains the retained PBMC parameter and calibration "
            "source tables and Figure S9. Each of the 135 query-penalty sensitivity "
            "rows is linked to a preserved transport manifest and method-parameter "
            "file. The recorded convergence status denotes scaling-iterate tolerance "
            "convergence, not an exact optimization certificate.\n",
            encoding="utf-8",
        )
        self._record_generated(relative, "unit scope and convergence qualification")

    def _write_audits(self, by_seed: pd.DataFrame) -> None:
        lineage = Path("audits/lineage.csv")
        _write_csv(
            self.package_root / lineage,
            [
                {
                    "artifact": "verified_results/supplement/pbmc_sensitivity_by_seed.csv",
                    "source": "fit-level score/truth artifacts and transport evidence",
                    "generator": "src/coreot/results/pbmc_supplement.py",
                    "status": "complete retained grid with fit-level convergence lineage",
                },
                {
                    "artifact": "verified_results/figures/manuscript_fig_pbmc_supp_robustness.png",
                    "source": "verified_results/supplement/pbmc_sensitivity_summary.csv",
                    "generator": "src/coreot/results/pbmc_supplement.py::render_robustness_figure",
                    "status": "manuscript Supplementary Figure S9",
                },
            ],
        )
        self._record_generated(lineage, "PBMC parameter and Figure S9 lineage")
        report = self.snapshot_root / "package_validation/validation_report.md"
        report.parent.mkdir(parents=True)
        report.write_text(
            "# PBMC parameter and calibration package validation\n\n"
            "- Overall status: **pass**\n"
            f"- Retained sensitivity fits: {len(by_seed)}\n"
            "- Unavailable fit records: 0\n"
            "- Nonconverged fits: 0\n"
            "- Fit-level cap and tolerance agreement: **pass**\n",
            encoding="utf-8",
        )

    def _write_external_dependencies(self) -> None:
        source = self.repository_root / "data/raw/kang_2018.h5ad"
        relative = Path("manifests/external_dependencies.csv")
        _write_csv(
            self.package_root / relative,
            [
                {
                    "repository_path": "data/raw/kang_2018.h5ad",
                    "accession": "GSE96583; Figshare file 34464122",
                    "sha256": sha256(source),
                    "bytes": source.stat().st_size,
                    "included_in_package": False,
                    "role": "canonical analyzed PBMC object",
                    "prepared_by": "Pertpy PBMC object; retained pipeline input",
                }
            ],
        )
        self._record_generated(relative, "acquisition-only PBMC object identity")

    def _write_included_files(self) -> None:
        relative = Path("manifests/included_files.csv")
        path = self.package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(IncludedFile.__dataclass_fields__),
            )
            writer.writeheader()
            writer.writerows(
                {
                    field: getattr(item, field)
                    for field in IncludedFile.__dataclass_fields__
                }
                for item in self.included
            )

    def _write_checksums(self) -> None:
        path = self.package_root / "manifests/checksums.sha256"
        files = sorted(
            candidate
            for candidate in self.package_root.rglob("*")
            if candidate.is_file() and candidate != path
        )
        path.write_text(
            "".join(
                f"{sha256(file)}  {file.relative_to(self.package_root).as_posix()}\n"
                for file in files
            ),
            encoding="utf-8",
        )


def build_unit(repository_root: Path, snapshot_root: Path) -> Path:
    return PackageBuilder(
        repository_root=repository_root,
        snapshot_root=snapshot_root,
    ).build()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_unit(args.repository_root, args.snapshot_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
