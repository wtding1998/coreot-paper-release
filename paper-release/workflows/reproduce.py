from __future__ import annotations

import argparse
import csv
from pathlib import Path
import traceback
from typing import Sequence

from coreot.submission import validate_submission_release

from .convergence import regenerate_convergence_unit
from .hiha import generate_hiha_release_artifacts
from .mouse_spleen import regenerate_mouse_spleen
from .packaged_units import (
    SUPPORTED_UNITS as PACKAGED_UNITS,
    regenerate_packaged_unit,
)
from .pbmc import (
    SUPPORTED_UNITS as PBMC_UNITS,
    export_pbmc_metadata,
    regenerate_pbmc_release_unit,
)


ORDERED_RELEASE_UNITS = (
    "hiha_primary",
    "hiha_matched_reference",
    "hiha_compatibility_sensitivity",
    "hiha_matchability_attribution",
    "hiha_parameter_and_calibration",
    "hiha_prior_dependence",
    "pbmc_primary",
    "pbmc_matched_reference",
    "pbmc_compatibility_sensitivity",
    "pbmc_matchability_attribution",
    "pbmc_parameter_and_calibration",
    "pbmc_prior_dependence",
    "mouse_spleen_primary",
    "mouse_spleen_compatibility_sensitivity",
    "mouse_spleen_matchability_attribution",
    "mouse_spleen_parameter_sensitivity",
    "mouse_spleen_prior_dependence",
    "convergence_and_environment",
)


class ReproductionError(ValueError):
    """Raised when the release reproduction contract is not satisfied."""


def ordered_release_units() -> tuple[str, ...]:
    """Return the required clean-room execution order."""

    return ORDERED_RELEASE_UNITS


def _dispatch_unit(
    *,
    release: Path,
    output: Path,
    unit: str,
    pbmc_metadata_root: Path | None,
    hiha_source_h5ad: Path | None,
) -> Path:
    unit_output = output / unit
    mouse_spleen_commands = {
        "mouse_spleen_primary": "primary",
        "mouse_spleen_compatibility_sensitivity": "compatibility",
    }
    if unit in mouse_spleen_commands:
        regenerate_mouse_spleen(
            mouse_spleen_commands[unit],
            unit_root=release / "units" / unit,
            output_root=unit_output,
        )
        return unit_output
    if unit in PBMC_UNITS:
        if pbmc_metadata_root is None:
            raise ReproductionError(f"PBMC metadata is unavailable for {unit}.")
        regenerate_pbmc_release_unit(
            release,
            unit,
            pbmc_metadata_root,
            unit_output,
        )
        return unit_output
    if unit in PACKAGED_UNITS:
        regenerate_packaged_unit(
            release_root=release,
            output_root=unit_output,
            unit_id=unit,
            hiha_source_h5ad=hiha_source_h5ad,
        )
        hiha_release_units = {
            "hiha_primary": "primary",
            "hiha_parameter_and_calibration": "parameter-calibration",
        }
        if unit in hiha_release_units:
            generate_hiha_release_artifacts(
                release_root=release,
                output_root=unit_output / "release_artifacts",
                unit=hiha_release_units[unit],
            )
        return unit_output
    if unit == "convergence_and_environment":
        regenerate_convergence_unit(
            release_root=release,
            output_root=unit_output,
        )
        return unit_output
    raise ReproductionError(f"Release-unit dispatch is not available: {unit}")


def _write_session_status(
    output: Path, rows: list[dict[str, str]]
) -> tuple[Path, Path]:
    status_path = output / "unit_status.csv"
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("unit_id", "status", "output", "error_log"),
        )
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row["status"] == "pass" for row in rows)
    failed = len(rows) - passed
    report_path = output / "session_report.md"
    table = [
        "| Unit | Status | Output | Error log |",
        "|---|---|---|---|",
        *(
            f"| `{row['unit_id']}` | {row['status']} | "
            f"`{row['output'] or 'not produced'}` | "
            f"`{row['error_log'] or 'none'}` |"
            for row in rows
        ),
    ]
    report_path.write_text(
        "\n".join(
            (
                "# Release clean-room reproduction session",
                "",
                f"Status: {'pass' if failed == 0 else 'fail'}",
                "",
                "The submission release passed integrity validation before any "
                "unit command executed. Units were attempted serially in the "
                "registry order; a failed unit did not prevent later independent "
                "units from running.",
                "",
                f"Passed units: {passed}/{len(rows)}. Failed units: {failed}.",
                "",
                *table,
                "",
                "No model fit is performed by this artifact-regeneration workflow.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return status_path, report_path


def run_release_reproduction(
    *,
    release_root: str | Path,
    output_root: str | Path,
    unit: str = "all",
    pbmc_metadata_root: str | Path | None = None,
    pbmc_source_h5ad: str | Path | None = None,
    hiha_source_h5ad: str | Path | None = None,
) -> tuple[Path, ...]:
    """Validate a release before dispatching artifact reproduction."""

    release = Path(release_root).resolve()
    try:
        validate_submission_release(release)
    except Exception as exc:
        raise ReproductionError(f"Invalid submission release: {release}") from exc
    output = Path(output_root).resolve()
    if output == release or release in output.parents:
        raise ReproductionError(
            f"Reproduction output must be outside the submission release: {output}"
        )
    if output.exists() or output.is_symlink():
        raise ReproductionError(
            f"Reproduction output root must not already exist: {output}"
        )
    if unit != "all" and unit not in ORDERED_RELEASE_UNITS:
        raise ReproductionError(f"Unknown release unit: {unit}")
    if pbmc_metadata_root is not None and pbmc_source_h5ad is not None:
        raise ReproductionError(
            "Specify either --pbmc-metadata-root or --pbmc-source-h5ad, not both."
        )
    if unit == "all" and hiha_source_h5ad is None:
        raise ReproductionError("all-unit reproduction requires --hiha-source-h5ad.")
    if (
        unit == "all"
        and pbmc_metadata_root is None
        and pbmc_source_h5ad is None
    ):
        raise ReproductionError(
            "all-unit reproduction requires --pbmc-source-h5ad or "
            "--pbmc-metadata-root."
        )
    hiha_source = (
        Path(hiha_source_h5ad).resolve()
        if hiha_source_h5ad is not None
        else None
    )
    metadata_root = (
        Path(pbmc_metadata_root).resolve()
        if pbmc_metadata_root is not None
        else None
    )
    if unit != "all":
        if unit in PBMC_UNITS and metadata_root is None:
            if pbmc_source_h5ad is None:
                raise ReproductionError(
                    f"{unit} requires --pbmc-source-h5ad or --pbmc-metadata-root."
                )
            metadata_root = output / "external_inputs/pbmc_metadata"
            export_pbmc_metadata(release, pbmc_source_h5ad, metadata_root)
        return (
            _dispatch_unit(
                release=release,
                output=output,
                unit=unit,
                pbmc_metadata_root=metadata_root,
                hiha_source_h5ad=hiha_source,
            ),
        )

    output.mkdir(parents=True)
    metadata_error: Exception | None = None
    if metadata_root is None:
        metadata_root = output / "external_inputs/pbmc_metadata"
        try:
            export_pbmc_metadata(release, pbmc_source_h5ad, metadata_root)
        except Exception as exc:
            metadata_error = exc
            metadata_root = None
            error_path = output / "errors/pbmc_metadata_export.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")

    outputs: list[Path] = []
    status_rows: list[dict[str, str]] = []
    for unit_id in ORDERED_RELEASE_UNITS:
        error_relative = ""
        try:
            if unit_id in PBMC_UNITS and metadata_error is not None:
                raise ReproductionError(
                    "PBMC metadata export failed; see "
                    "errors/pbmc_metadata_export.txt."
                ) from metadata_error
            unit_output = _dispatch_unit(
                release=release,
                output=output,
                unit=unit_id,
                pbmc_metadata_root=metadata_root,
                hiha_source_h5ad=hiha_source,
            )
            outputs.append(unit_output)
            status = "pass"
            output_relative = unit_output.relative_to(output).as_posix()
        except Exception:
            status = "fail"
            output_relative = ""
            error_path = output / "errors" / f"{unit_id}.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            error_relative = error_path.relative_to(output).as_posix()
        status_rows.append(
            {
                "unit_id": unit_id,
                "status": status,
                "output": output_relative,
                "error_log": error_relative,
            }
        )
        _write_session_status(output, status_rows)
    failures = [row["unit_id"] for row in status_rows if row["status"] != "pass"]
    if failures:
        raise ReproductionError(
            "Release reproduction failed for "
            f"{len(failures)} unit(s); see {output / 'session_report.md'}."
        )
    return tuple(outputs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate CoRe-OT manuscript artifacts from a validated release."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--unit",
        choices=("all", *ORDERED_RELEASE_UNITS),
        default="all",
    )
    parser.add_argument("--pbmc-metadata-root", type=Path)
    parser.add_argument("--pbmc-source-h5ad", type=Path)
    parser.add_argument("--hiha-source-h5ad", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = run_release_reproduction(
        release_root=args.release_root,
        output_root=args.output_root,
        unit=args.unit,
        pbmc_metadata_root=args.pbmc_metadata_root,
        pbmc_source_h5ad=args.pbmc_source_h5ad,
        hiha_source_h5ad=args.hiha_source_h5ad,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
