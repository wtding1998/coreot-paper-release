from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from coreot.submission import audit_submission_release


REPO = Path(__file__).resolve().parents[7]
SNAPSHOT = Path(__file__).resolve().parents[3]
HDF5_SUFFIXES = {".h5", ".h5ad", ".hdf5"}
TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    base_unit: Path
    base_package: Path
    base_validation_report: Path


def _old(relative: str) -> Path:
    return REPO / "results/submission_verification/PBMC" / relative


UNITS = {
    "pbmc_primary": UnitSpec(
        "pbmc_primary",
        _old("2026-08-03/pbmc_primary_u_tilde_retirement"),
        _old(
            "2026-08-03/pbmc_primary_u_tilde_retirement/"
            "package_staging/pbmc_primary"
        ),
        _old(
            "2026-08-03/pbmc_primary_u_tilde_retirement/"
            "package_validation/validation_report.md"
        ),
    ),
    "pbmc_matched_reference": UnitSpec(
        "pbmc_matched_reference",
        _old("2026-08-03/pbmc_matched_reference_estimand_fix"),
        _old(
            "2026-08-03/pbmc_matched_reference_estimand_fix/"
            "package_staging/pbmc_matched_reference"
        ),
        _old(
            "2026-08-03/pbmc_matched_reference_estimand_fix/"
            "package_validation/validation_report.md"
        ),
    ),
    "pbmc_compatibility_sensitivity": UnitSpec(
        "pbmc_compatibility_sensitivity",
        _old("2026-08-03/pbmc_compatibility_sensitivity_lineage_reconstruction"),
        _old(
            "2026-08-03/pbmc_compatibility_sensitivity_lineage_reconstruction/"
            "package_staging/pbmc_compatibility_sensitivity"
        ),
        _old(
            "2026-08-03/pbmc_compatibility_sensitivity_lineage_reconstruction/"
            "package_validation/validation_report.md"
        ),
    ),
    "pbmc_matchability_attribution": UnitSpec(
        "pbmc_matchability_attribution",
        _old("2026-08-05/pbmc_matchability_attribution_lineage_reconstruction"),
        _old(
            "2026-08-05/pbmc_matchability_attribution_lineage_reconstruction/"
            "package_staging/pbmc_matchability_attribution"
        ),
        _old(
            "2026-08-05/pbmc_matchability_attribution_lineage_reconstruction/"
            "package_validation/validation_report.md"
        ),
    ),
    "pbmc_parameter_and_calibration": UnitSpec(
        "pbmc_parameter_and_calibration",
        _old("2026-08-03/pbmc_parameter_and_calibration_s7_fix"),
        _old(
            "2026-08-03/pbmc_parameter_and_calibration_s7_fix/"
            "package_staging/pbmc_parameter_and_calibration"
        ),
        _old(
            "2026-08-03/pbmc_parameter_and_calibration_s7_fix/"
            "package_validation/validation_report.md"
        ),
    ),
    "pbmc_prior_dependence": UnitSpec(
        "pbmc_prior_dependence",
        _old("2026-08-03/pbmc_prior_dependence"),
        _old(
            "2026-08-03/pbmc_prior_dependence/"
            "package_staging/pbmc_prior_dependence"
        ),
        _old(
            "2026-08-03/pbmc_prior_dependence/"
            "package_validation/validation_report.md"
        ),
    ),
}


CURRENT_CODE = {
    "pbmc_primary": (
        "experiments/pbmc_state/generate_pbmc_compare_baseline.py",
        "experiments/pbmc_state/generate_pbmc_figure3_panels.py",
        "experiments/pbmc_state/generate_pbmc_supplement.py",
        "src/coreot/results/grid.py",
        "src/coreot/results/pbmc_figure3.py",
        "src/coreot/results/pbmc_supplement.py",
    ),
    "pbmc_matched_reference": (
        "experiments/pbmc_state/generate_pbmc_s6_matched_reference_rescue.py",
        "experiments/pbmc_state/generate_pbmc_figure3_panels.py",
        "experiments/pbmc_state/generate_pbmc_supplement.py",
        "src/coreot/results/pbmc_s6_matched_reference_rescue.py",
        "src/coreot/results/pbmc_figure3.py",
        "src/coreot/results/pbmc_supplement.py",
    ),
    "pbmc_compatibility_sensitivity": (
        "experiments/component_ablation_surfaces.py",
    ),
    "pbmc_matchability_attribution": (
        "experiments/pbmc_rho_tau_heatmap.py",
        "experiments/rho_attribution_search.py",
        "experiments/component_ablation_surfaces.py",
    ),
    "pbmc_parameter_and_calibration": (
        "experiments/pbmc_state/generate_pbmc_s7_calibration.py",
        "experiments/pbmc_state/generate_pbmc_supplement.py",
        "src/coreot/results/pbmc_s7_calibration.py",
        "src/coreot/results/pbmc_supplement.py",
    ),
    "pbmc_prior_dependence": (
        "experiments/pbmc_state/generate_pbmc_supplement.py",
        "src/coreot/results/pbmc_supplement.py",
    ),
}


def unit_root(unit_id: str) -> Path:
    return SNAPSHOT / unit_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_new(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in HDF5_SUFFIXES:
        raise RuntimeError(f"Refusing to copy HDF5-family object: {source}")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_new(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to overwrite: {destination}")
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            continue
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            copy_new(path, destination / relative)


def prepare() -> None:
    resource_source = _old("2026-08-03/baseline_resource_audit")
    resource_target = SNAPSHOT / "baseline_resource_audit"
    for name in ("README.md", "compare_baselines_by_method.csv"):
        copy_new(resource_source / name, resource_target / name)
    for source in sorted((resource_source / "raw").glob("*.log")):
        copy_new(source, resource_target / "raw" / source.name)

    compatibility_source = (
        UNITS["pbmc_compatibility_sensitivity"].base_unit
        / "regenerated/component_ablation"
    )
    compatibility_target = (
        unit_root("pbmc_compatibility_sensitivity")
        / "regenerated/component_ablation"
    )
    compatibility_target.mkdir(parents=True, exist_ok=True)
    for endpoint in ("b_cells", "nk_cells", "dendritic_cells"):
        copy_tree_new(
            compatibility_source / endpoint,
            compatibility_target / endpoint,
        )

    match_source = UNITS["pbmc_matchability_attribution"].base_unit / "regenerated"
    match_project = (
        unit_root("pbmc_matchability_attribution") / "regenerated/project"
    )
    for family, table in (
        (
            "rho_attribution_tau_surface_alpha0_range075_175",
            "rho_tau_surface_range075_175_by_seed.csv",
        ),
        (
            "rho_attribution_alpha_search",
            "rho_attribution_by_replicate.csv",
        ),
    ):
        source_root = match_source / family
        destination_root = match_project / "results/PBMC/sensitivity" / family
        copy_new(source_root / "manifest.yaml", destination_root / "manifest.yaml")
        copy_new(
            source_root / "tables" / table,
            destination_root / "tables" / table,
        )


def _link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def setup_supplement(unit_id: str) -> None:
    project = unit_root(unit_id) / "regenerated/supplement_project"
    if project.exists() and any(project.iterdir()):
        raise RuntimeError(f"Refusing nonempty isolated project: {project}")
    project.mkdir(parents=True, exist_ok=True)
    _link(REPO / "runs", project / "runs")
    _link(REPO / "data", project / "data")
    comparison = REPO / "results/PBMC/compare_baselines"
    if unit_id == "pbmc_primary":
        comparison = unit_root(unit_id) / "regenerated/PBMC/compare_baselines"
    _link(comparison, project / "results/PBMC/compare_baselines")
    for name in ("main", "sensitivity"):
        _link(REPO / "results/PBMC" / name, project / "results/PBMC" / name)
    _link(
        REPO / "results/PBMC/manuscript/supp_destination_support",
        project / "results/PBMC/manuscript/supp_destination_support",
    )

    source_data = REPO / "results/PBMC/figures/data"
    target_data = project / "results/PBMC/figures/data"
    target_data.mkdir(parents=True)
    for source in sorted(source_data.iterdir()):
        if source.is_file():
            copy_new(source, target_data / source.name)
    replacement_root = None
    if unit_id == "pbmc_matched_reference":
        replacement_root = unit_root(unit_id) / "regenerated/figures_s6/data"
    elif unit_id == "pbmc_parameter_and_calibration":
        replacement_root = unit_root(unit_id) / "regenerated/figures_s7/data"
    if replacement_root is not None:
        for source in sorted(replacement_root.glob("*.csv")):
            destination = target_data / source.name
            if destination.exists():
                destination.unlink()
            copy_new(source, destination)

    for name in (
        "manuscript_fig_pbmc_component_minus_m.png",
        "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png",
        "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175_caption.md",
    ):
        _link(REPO / "docs/figs" / name, project / "docs/figs" / name)


def _compare_frame(actual_path: Path, expected_path: Path) -> tuple[int, float, str]:
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(expected_path)
    filtered_variant = False
    if "variant" in actual.columns and "variant" in expected.columns:
        actual_variants = set(actual["variant"].astype(str))
        expected_variants = set(expected["variant"].astype(str))
        if actual_variants < expected_variants:
            expected = expected.loc[
                expected["variant"].astype(str).isin(actual_variants)
            ].reset_index(drop=True)
            filtered_variant = True
    preferred_keys = (
        "experiment",
        "endpoint",
        "seed",
        "replicate",
        "run_id",
        "variant",
        "method",
        "score",
        "tau_min",
        "tau_max",
        "tau_source",
        "alpha",
        "quantile",
        "statistic",
    )
    keys = [key for key in preferred_keys if key in actual.columns]
    if keys and (filtered_variant or len(actual) == len(expected)):
        actual = actual.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
    if actual.shape != expected.shape or list(actual.columns) != list(expected.columns):
        return len(actual), float("inf"), "schema_or_shape_mismatch"
    maximum = 0.0
    for column in actual.columns:
        if column.endswith("runtime_seconds"):
            continue
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
                return len(actual), float("inf"), f"missingness_mismatch:{column}"
            difference = np.abs(left_values - right_values)
            finite = difference[np.isfinite(difference)]
            if finite.size:
                maximum = max(maximum, float(finite.max()))
        elif not left.astype("string").equals(right.astype("string")):
            return len(actual), float("inf"), f"categorical_mismatch:{column}"
    return len(actual), maximum, "pass" if maximum <= TOLERANCE else "numeric_mismatch"


def _csv_pairs(unit_id: str) -> list[tuple[str, Path, Path]]:
    root = unit_root(unit_id) / "regenerated"
    pairs: list[tuple[str, Path, Path]] = []
    if unit_id == "pbmc_primary":
        fresh = root / "PBMC/compare_baselines"
        expected = UNITS[unit_id].base_package / "verified_results/compare_baselines"
        for path in sorted(fresh.rglob("*.csv")):
            pairs.append(("primary_comparison", path, expected / path.relative_to(fresh)))
        figure = root / "PBMC/manuscript/figure3"
        canonical = REPO / "results/PBMC/manuscript/figure3"
        for path in sorted(figure.rglob("*.csv")):
            pairs.append(("figure3_source", path, canonical / path.relative_to(figure)))
    elif unit_id == "pbmc_matched_reference":
        fresh = root / "figures_s6/data"
        canonical = REPO / "results/PBMC/figures/data"
        for path in sorted(fresh.glob("*.csv")):
            pairs.append(("matched_reference_s6", path, canonical / path.name))
        figure = root / "figure3"
        canonical_figure = REPO / "results/PBMC/manuscript/figure3"
        for path in sorted(figure.rglob("*.csv")):
            pairs.append(
                ("matched_reference_figure3", path, canonical_figure / path.relative_to(figure))
            )
    elif unit_id == "pbmc_compatibility_sensitivity":
        fresh = root / "component_ablation/tables"
        expected = REPO / "results/PBMC/sensitivity/component_ablation/tables"
        for path in sorted(fresh.glob("*.csv")):
            pairs.append(("compatibility", path, expected / path.name))
    elif unit_id == "pbmc_matchability_attribution":
        project = root / "project/results/PBMC/sensitivity"
        canonical = REPO / "results/PBMC/sensitivity"
        for family in (
            "rho_attribution_tau_surface_alpha0_range075_175",
            "rho_attribution_alpha_search",
        ):
            for path in sorted((project / family / "tables").glob("*.csv")):
                pairs.append(("matchability", path, canonical / family / "tables" / path.name))
    elif unit_id == "pbmc_parameter_and_calibration":
        fresh = root / "figures_s7/data"
        canonical = REPO / "results/PBMC/figures/data"
        for path in sorted(fresh.glob("*.csv")):
            pairs.append(("calibration_s7", path, canonical / path.name))

    if (root / "supplement_project/results/PBMC/manuscript/supplement").is_dir():
        fresh = root / "supplement_project/results/PBMC/manuscript/supplement"
        canonical = REPO / "results/PBMC/manuscript/supplement"
        for path in sorted(fresh.glob("*.csv")):
            pairs.append(("supplement", path, canonical / path.name))
    return pairs


def _png_pairs(unit_id: str) -> list[tuple[str, Path, Path]]:
    root = unit_root(unit_id) / "regenerated"
    pairs: list[tuple[str, Path, Path]] = []
    if unit_id == "pbmc_primary":
        fresh = root / "PBMC/manuscript/figure3"
        canonical = REPO / "results/PBMC/manuscript/figure3"
        for path in sorted(fresh.rglob("*.png")):
            pairs.append(("figure3", path, canonical / path.relative_to(fresh)))
        fresh_panels = root / "PBMC/manuscript/figure3_panels"
        canonical_panels = REPO / "results/PBMC/manuscript/figure3_panels"
        for path in sorted(fresh_panels.glob("*.png")):
            pairs.append(("figure3_panel", path, canonical_panels / path.name))
    elif unit_id == "pbmc_matched_reference":
        pairs.append(
            (
                "figure_s6",
                root / "figures_s6/figure_s6_pbmc_matched_reference_rescue.png",
                REPO / "results/PBMC/figures/figure_s6_pbmc_matched_reference_rescue.png",
            )
        )
        pairs.append(
            (
                "figure3_panel_c",
                root / "figure3_composite_panels/figure_3_pbmc_panel_c.png",
                REPO / "results/PBMC/manuscript/figure3_panels/figure_3_pbmc_panel_c.png",
            )
        )
    elif unit_id == "pbmc_compatibility_sensitivity":
        pairs.append(
            (
                "compatibility_current_code",
                root / "figures/manuscript_fig_pbmc_component_minus_m.png",
                REPO / "docs/figs/manuscript_fig_pbmc_component_minus_m.png",
            )
        )
    elif unit_id == "pbmc_matchability_attribution":
        pairs.append(
            (
                "matchability_focused",
                root
                / "project/docs/figs/"
                "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png",
                REPO
                / "docs/figs/"
                "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png",
            )
        )
        pairs.append(
            (
                "matchability_alpha",
                root / "project/docs/figs/manuscript_fig_pbmc_rho_attribution_alpha_search.png",
                REPO / "docs/figs/manuscript_fig_pbmc_rho_attribution_alpha_search.png",
            )
        )
    elif unit_id == "pbmc_parameter_and_calibration":
        pairs.append(
            (
                "calibration_s7",
                root / "figures_s7/figure_s7_pbmc_calibration.png",
                REPO / "results/PBMC/figures/figure_s7_pbmc_calibration.png",
            )
        )
    supplement_docs = root / "supplement_project/docs/figs"
    if supplement_docs.is_dir():
        names = {
            "pbmc_matched_reference": ("manuscript_fig_pbmc_supp_mechanistic.png",),
            "pbmc_parameter_and_calibration": ("manuscript_fig_pbmc_supp_robustness.png",),
            "pbmc_prior_dependence": ("manuscript_fig_pbmc_supp_prior_dependence.png",),
        }.get(unit_id, ())
        for name in names:
            pairs.append(("supplement", supplement_docs / name, REPO / "docs/figs" / name))
    return pairs


def compare(unit_id: str) -> None:
    output = unit_root(unit_id) / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, actual, expected in _csv_pairs(unit_id):
        if not actual.is_file() or not expected.is_file():
            rows.append(
                {
                    "family": family,
                    "artifact": str(actual.relative_to(REPO)),
                    "canonical": str(expected.relative_to(REPO)),
                    "rows": "",
                    "max_abs_difference": "",
                    "status": "missing",
                }
            )
            continue
        count, maximum, status = _compare_frame(actual, expected)
        rows.append(
            {
                "family": family,
                "artifact": str(actual.relative_to(REPO)),
                "canonical": str(expected.relative_to(REPO)),
                "rows": count,
                "max_abs_difference": maximum,
                "status": status,
            }
        )
    pd.DataFrame(rows).to_csv(output / "numerical_comparison.csv", index=False)

    rendered = []
    for family, actual, expected in _png_pairs(unit_id):
        if not actual.is_file() or not expected.is_file():
            rendered.append(
                {
                    "family": family,
                    "artifact": str(actual.relative_to(REPO)),
                    "canonical": str(expected.relative_to(REPO)),
                    "dimensions_match": False,
                    "byte_exact": False,
                    "pixel_exact": False,
                    "status": "missing",
                }
            )
            continue
        with Image.open(actual) as left, Image.open(expected) as right:
            dimensions = left.size == right.size
            pixels = dimensions and np.array_equal(np.asarray(left), np.asarray(right))
        expected_difference = unit_id == "pbmc_compatibility_sensitivity"
        rendered.append(
            {
                "family": family,
                "artifact": str(actual.relative_to(REPO)),
                "canonical": str(expected.relative_to(REPO)),
                "dimensions_match": dimensions,
                "byte_exact": actual.read_bytes() == expected.read_bytes(),
                "pixel_exact": pixels,
                "status": (
                    "expected_current_code_render_difference"
                    if expected_difference and dimensions
                    else "pass"
                    if pixels
                    else "render_mismatch"
                ),
            }
        )
    pd.DataFrame(rendered).to_csv(output / "rendered_comparison.csv", index=False)

    if unit_id == "pbmc_primary":
        tables = unit_root(unit_id) / "regenerated/PBMC/compare_baselines"
        for path in tables.rglob("*.csv"):
            frame = pd.read_csv(path)
            if "score" in frame and frame["score"].astype(str).eq("u_tilde").any():
                raise RuntimeError(f"Retired score found in {path}")
    if unit_id == "pbmc_parameter_and_calibration":
        path = (
            unit_root(unit_id)
            / "regenerated/figures_s7/data/figure_s7_calibration_by_seed.csv"
        )
        quantiles = sorted(pd.read_csv(path)["quantile"].drop_duplicates().tolist())
        expected = [*[round(0.85 + 0.01 * index, 2) for index in range(15)], 0.975, 0.995]
        if sorted(quantiles) != sorted(expected):
            raise RuntimeError(f"Calibration grid mismatch: {quantiles}")

    failures = [row for row in rows if row["status"] != "pass"]
    render_failures = [
        row
        for row in rendered
        if row["status"] not in {"pass", "expected_current_code_render_difference"}
    ]
    if failures or render_failures:
        raise RuntimeError(
            f"Comparison failures for {unit_id}: numerical={failures}, rendered={render_failures}"
        )
    print(
        json.dumps(
            {
                "unit_id": unit_id,
                "numerical_artifacts": len(rows),
                "rendered_artifacts": len(rendered),
                "maximum_numeric_difference": max(
                    (float(row["max_abs_difference"]) for row in rows), default=0.0
                ),
                "status": "pass",
            },
            sort_keys=True,
        )
    )


def _source_for_base_code(spec: UnitSpec, relative: Path) -> Path:
    code_relative = relative.relative_to("code")
    candidates = []
    if code_relative.parts[0] == "repository":
        candidates.append(REPO / Path(*code_relative.parts[1:]))
    elif code_relative.parts[0] == "environment":
        candidates.append(REPO / code_relative.name)
    elif code_relative.parts[0] == "verification":
        candidates.append(spec.base_unit / "audit/workers" / code_relative.name)
    candidates.extend((REPO / code_relative, spec.base_package / relative))
    return next((path for path in candidates if path.is_file()), candidates[-1])


def build_package(unit_id: str) -> None:
    spec = UNITS[unit_id]
    unit = unit_root(unit_id)
    package = unit / "package_staging" / unit_id
    if package.exists() and any(package.iterdir()):
        raise RuntimeError(f"Refusing nonempty package: {package}")
    package.mkdir(parents=True, exist_ok=True)
    source_map: dict[str, Path] = {}

    for source in sorted(spec.base_package.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        relative = source.relative_to(spec.base_package)
        if relative in {
            Path("manifests/included_files.csv"),
            Path("manifests/checksums.sha256"),
        }:
            continue
        actual_source = source
        if relative.parts[0] == "code":
            actual_source = _source_for_base_code(spec, relative)
        destination_relative = relative
        if relative.parts[:2] == ("code", "verification"):
            destination_relative = (
                Path("code/verification") / unit_id / Path(*relative.parts[2:])
            )
        destination = package / destination_relative
        copy_new(actual_source, destination)
        if relative.parts[0] == "code":
            source_map[destination_relative.as_posix()] = actual_source

    for relative_text in CURRENT_CODE[unit_id]:
        source = REPO / relative_text
        relative = Path("code/repository") / relative_text
        destination = package / relative
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Unexpected staged code collision: {relative}")
        else:
            copy_new(source, destination)
        source_map[relative.as_posix()] = source

    worker = Path(__file__).resolve()
    worker_relative = (
        Path("code/verification") / unit_id / "revalidate_pbmc_snapshot.py"
    )
    worker_destination = package / worker_relative
    if not worker_destination.exists():
        copy_new(worker, worker_destination)
    source_map[worker_relative.as_posix()] = worker

    for source_root, destination_root in (
        (unit / "audit", package / "audits/current_revalidation"),
        (unit / "comparison", package / "audits/current_revalidation/comparison"),
        (unit / "regenerated", package / "verified_results/current_revalidation"),
    ):
        for source in sorted(source_root.rglob("*")):
            if source.is_symlink() or not source.is_file():
                continue
            if source.suffix.lower() in HDF5_SUFFIXES:
                raise RuntimeError(f"Forbidden generated HDF5-family object: {source}")
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            if destination.exists():
                continue
            copy_new(source, destination)

    readme = package / "README.md"
    readme.write_text(
        "# " + unit_id + " evidence package\n\n"
        "Fresh 2026-08-11 revalidation of the retained PBMC evidence unit. "
        "Canonical HDF5-family inputs remain external and are declared in "
        "`manifests/external_dependencies.csv`. Current isolated regeneration "
        "and comparison evidence is under `verified_results/current_revalidation/` "
        "and `audits/current_revalidation/`.\n",
        encoding="utf-8",
    )

    symlinks = [path for path in package.rglob("*") if path.is_symlink()]
    hdf5 = [
        path
        for path in package.rglob("*")
        if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
    ]
    if symlinks or hdf5:
        raise RuntimeError(f"Forbidden package objects: symlinks={symlinks}, hdf5={hdf5}")

    included = package / "manifests/included_files.csv"
    checksums = package / "manifests/checksums.sha256"
    payloads = [
        path
        for path in sorted(package.rglob("*"))
        if path.is_file() and path not in {included, checksums}
    ]
    with included.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "package_path",
                "source_path",
                "source_sha256",
                "package_sha256",
                "sha256",
                "bytes",
                "role",
            ),
        )
        writer.writeheader()
        for path in payloads:
            relative = path.relative_to(package).as_posix()
            source = source_map.get(relative)
            writer.writerow(
                {
                    "package_path": relative,
                    "source_path": (
                        source.relative_to(REPO).as_posix() if source is not None else ""
                    ),
                    "source_sha256": sha256(source) if source is not None else "",
                    "package_sha256": sha256(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "role": path.relative_to(package).parts[0],
                }
            )
    checksum_targets = [
        path for path in sorted(package.rglob("*")) if path.is_file() and path != checksums
    ]
    checksums.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(package).as_posix()}\n"
            for path in checksum_targets
        ),
        encoding="utf-8",
    )
    print(package.relative_to(REPO))


def validate_package(unit_id: str) -> None:
    spec = UNITS[unit_id]
    unit = unit_root(unit_id)
    package = unit / "package_staging" / unit_id
    external = pd.read_csv(package / "manifests/external_dependencies.csv")
    required_external = {
        "repository_path",
        "accession",
        "sha256",
        "bytes",
        "included_in_package",
        "role",
        "prepared_by",
    }
    if not required_external.issubset(external.columns):
        raise RuntimeError("External-dependency manifest schema is incomplete")
    if external["included_in_package"].astype(str).str.lower().ne("false").any():
        raise RuntimeError("External dependency marked included")
    canonical = external.loc[
        external["repository_path"].astype(str).eq("data/raw/kang_2018.h5ad")
    ]
    if len(canonical) != 1:
        raise RuntimeError("Canonical PBMC object declaration is not unique")
    canonical_path = REPO / "data/raw/kang_2018.h5ad"
    record = canonical.iloc[0]
    if sha256(canonical_path) != str(record["sha256"]) or canonical_path.stat().st_size != int(
        record["bytes"]
    ):
        raise RuntimeError("Canonical PBMC object checksum or size mismatch")

    temporary_spec = unit / "audit/release_audit_spec.yaml"
    temporary_spec.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release": {"id": f"{unit_id}-isolated-audit", "title": unit_id},
                "units": [
                    {
                        "id": unit_id,
                        "title": unit_id,
                        "status": "verified",
                        "required": True,
                        "package_root": str(package.relative_to(REPO)),
                        "validation_report": str(spec.base_validation_report.relative_to(REPO)),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    audit = audit_submission_release(
        spec_path=temporary_spec,
        repository_root=REPO,
    )
    unit_audit = audit.units[0]
    if not unit_audit.package_valid or unit_audit.issues:
        raise RuntimeError(f"Public release audit failed: {unit_audit.issues}")

    numerical = pd.read_csv(unit / "comparison/numerical_comparison.csv")
    rendered = pd.read_csv(unit / "comparison/rendered_comparison.csv")
    if not numerical["status"].astype(str).eq("pass").all():
        raise RuntimeError("Numerical comparison contains failures")
    accepted_render = {"pass", "expected_current_code_render_difference"}
    if not rendered["status"].astype(str).isin(accepted_render).all():
        raise RuntimeError("Rendered comparison contains failures")

    package_files = [path for path in package.rglob("*") if path.is_file()]
    result = {
        "unit_id": unit_id,
        "package_files": len(package_files),
        "external_dependencies": len(external),
        "numerical_artifacts": len(numerical),
        "max_abs_difference": float(numerical["max_abs_difference"].max()),
        "rendered_artifacts": len(rendered),
        "public_release_audit_ready": audit.ready,
        "public_release_audit_verified_units": audit.verified_units,
        "symlinks": sum(path.is_symlink() for path in package.rglob("*")),
        "hdf5_files": sum(
            path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
            for path in package.rglob("*")
        ),
        "status": "pass",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    setup = subparsers.add_parser("setup-supplement")
    setup.add_argument("--unit", choices=sorted(UNITS), required=True)
    for name in ("compare", "build-package", "validate-package"):
        command = subparsers.add_parser(name)
        command.add_argument("--unit", choices=sorted(UNITS), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "setup-supplement":
        setup_supplement(args.unit)
    elif args.command == "compare":
        compare(args.unit)
    elif args.command == "build-package":
        build_package(args.unit)
    else:
        validate_package(args.unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
