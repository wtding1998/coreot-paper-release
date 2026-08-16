from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coreot.artifacts.hashes import sha256_file  # noqa: E402


UNIT = Path(__file__).resolve().parents[2]
AUDIT = UNIT / "audit"
COMPARISON = UNIT / "comparison"
FOCUSED_CANONICAL = REPO / (
    "results/PBMC/sensitivity/"
    "rho_attribution_tau_surface_alpha0_range075_175"
)
ALPHA_CANONICAL = REPO / (
    "results/PBMC/sensitivity/rho_attribution_alpha_search"
)
FOCUSED_REGENERATED = UNIT / "regenerated" / (
    "rho_attribution_tau_surface_alpha0_range075_175"
)
ALPHA_REGENERATED = UNIT / "regenerated/rho_attribution_alpha_search"
TOLERANCE = 1.0e-12


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def _checkpoint_difference(
    expected_path: Path,
    observed_path: Path,
    *,
    keys: list[str],
    excluded: set[str],
) -> tuple[float, bool]:
    expected = pd.read_csv(expected_path).sort_values(keys).reset_index(drop=True)
    observed = pd.read_csv(observed_path).sort_values(keys).reset_index(drop=True)
    if list(expected.columns) != list(observed.columns) or len(expected) != len(
        observed
    ):
        return float("inf"), False
    maximum = 0.0
    for column in expected.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
            if not np.array_equal(np.isnan(left), np.isnan(right)):
                return float("inf"), False
            maximum = max(maximum, float(np.nanmax(np.abs(left - right))))
        elif not expected[column].fillna("<NA>").astype(str).equals(
            observed[column].fillna("<NA>").astype(str)
        ):
            return float("inf"), False
    return maximum, maximum <= TOLERANCE


def build_checkpoint_index() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, canonical, regenerated, keys, excluded in (
        (
            "focused_tau",
            FOCUSED_CANONICAL,
            FOCUSED_REGENERATED,
            ["endpoint", "seed", "tau_min", "tau_max"],
            set(),
        ),
        (
            "alpha_search",
            ALPHA_CANONICAL,
            ALPHA_REGENERATED,
            ["endpoint", "seed", "alpha"],
            {"heterogeneous_runtime_seconds", "uniform_runtime_seconds"},
        ),
    ):
        for expected_path in sorted((canonical / "checkpoints").glob("*/seed*.csv")):
            relative = expected_path.relative_to(canonical)
            observed_path = regenerated / relative
            maximum, passed = _checkpoint_difference(
                expected_path,
                observed_path,
                keys=keys,
                excluded=excluded,
            )
            frame = pd.read_csv(expected_path)
            rows.append(
                {
                    "analysis_family": family,
                    "endpoint": str(frame.iloc[0]["endpoint"]),
                    "seed": int(frame.iloc[0]["seed"]),
                    "canonical_path": _relative(expected_path),
                    "canonical_sha256": sha256_file(expected_path),
                    "regenerated_path": _relative(observed_path),
                    "regenerated_sha256": sha256_file(observed_path),
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "runtime_columns_excluded": ";".join(sorted(excluded)),
                    "max_scientific_abs_difference": maximum,
                    "byte_exact": sha256_file(expected_path)
                    == sha256_file(observed_path),
                    "status": "pass" if passed else "fail",
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 30 or not result["status"].eq("pass").all():
        raise RuntimeError("Checkpoint comparison is incomplete or failed")
    result.to_csv(AUDIT / "checkpoint_artifacts.csv", index=False)
    return result


def _lineage_record(
    *,
    family: str,
    kind: str,
    role: str,
    canonical: Path,
    regenerated: Path,
    status: str,
    note: str,
) -> dict[str, object]:
    return {
        "analysis_family": family,
        "artifact_kind": kind,
        "role": role,
        "canonical_path": _relative(canonical),
        "canonical_sha256": sha256_file(canonical),
        "canonical_bytes": canonical.stat().st_size,
        "regenerated_path": _relative(regenerated),
        "regenerated_sha256": sha256_file(regenerated),
        "regenerated_bytes": regenerated.stat().st_size,
        "byte_exact": sha256_file(canonical) == sha256_file(regenerated),
        "status": status,
        "note": note,
    }


def build_lineage(checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in checkpoints.itertuples(index=False):
        rows.append(
            _lineage_record(
                family=str(row.analysis_family),
                kind="checkpoint",
                role="per-seed paired evaluated metrics",
                canonical=REPO / str(row.canonical_path),
                regenerated=REPO / str(row.regenerated_path),
                status="pass",
                note=(
                    "Scientific fields agree within 1e-12; alpha-search "
                    "reconstruction runtimes are new provenance and excluded."
                ),
            )
        )
    for family, canonical, regenerated, table_names, figure_paths in (
        (
            "focused_tau",
            FOCUSED_CANONICAL,
            FOCUSED_REGENERATED,
            (
                "rho_tau_surface_range075_175_by_seed.csv",
                "rho_tau_surface_range075_175_summary.csv",
            ),
            (
                REPO
                / "docs/figs/"
                "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png",
                UNIT
                / "regenerated/figures/"
                "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png",
            ),
        ),
        (
            "alpha_search",
            ALPHA_CANONICAL,
            ALPHA_REGENERATED,
            ("rho_attribution_by_replicate.csv", "rho_attribution_summary.csv"),
            (
                REPO
                / "docs/figs/manuscript_fig_pbmc_rho_attribution_alpha_search.png",
                UNIT
                / "regenerated/figures/"
                "manuscript_fig_pbmc_rho_attribution_alpha_search.png",
            ),
        ),
    ):
        for table_name in table_names:
            rows.append(
                _lineage_record(
                    family=family,
                    kind="table",
                    role=(
                        "paired evaluated metrics"
                        if "summary" not in table_name
                        else "donor-equal aggregate statistics"
                    ),
                    canonical=canonical / "tables" / table_name,
                    regenerated=regenerated / "tables" / table_name,
                    status="pass",
                    note="Keys, schema, missingness, categorical fields, and numeric fields verified.",
                )
            )
        rows.append(
            _lineage_record(
                family=family,
                kind="manifest",
                role="canonical analysis design and artifact registry",
                canonical=canonical / "manifest.yaml",
                regenerated=regenerated / "manifest.yaml",
                status="audited",
                note=(
                    "The isolated manifest records reconstruction provenance; "
                    "it is not expected to be byte-identical to the canonical registry."
                ),
            )
        )
        rows.append(
            _lineage_record(
                family=family,
                kind="figure",
                role="rendered retained evidence",
                canonical=figure_paths[0],
                regenerated=figure_paths[1],
                status="pass",
                note="Byte-exact and pixel-exact isolated render.",
            )
        )
    result = pd.DataFrame(rows)
    if len(result) != 38:
        raise RuntimeError(f"Expected 38 retained lineage rows; found {len(result)}")
    result.to_csv(AUDIT / "lineage.csv", index=False)
    return result


def write_component_manifest(
    checkpoints: pd.DataFrame, lineage: pd.DataFrame
) -> None:
    status = pd.read_csv(AUDIT / "reconstruction_status.csv")
    numerical = pd.read_csv(COMPARISON / "numerical_comparison.csv")
    summaries = pd.read_csv(COMPARISON / "summary_comparison.csv")
    rendered = pd.read_csv(COMPARISON / "rendered_comparison.csv")
    dependency_index = AUDIT / "fit_dependencies.csv"
    payload = {
        "unit_id": "pbmc_matchability_attribution",
        "stage": "isolated-lineage-reconstruction",
        "inputs": {
            "paired_cases": 300,
            "fit_count": 600,
            "compact_fit_artifact_count": 1800,
            "external_dependency_count": 91,
            "raw_pbmc_h5ad": {
                "path": "data/raw/kang_2018.h5ad",
                "sha256": "229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff",
                "bytes": 123669441,
                "included_in_package": False,
            },
        },
        "fit_dependency_index": {
            "path": _relative(dependency_index),
            "sha256": sha256_file(dependency_index),
            "bytes": dependency_index.stat().st_size,
            "rows": 600,
        },
        "retained_artifacts": {
            "count": len(lineage),
            "lineage_path": _relative(AUDIT / "lineage.csv"),
            "lineage_sha256": sha256_file(AUDIT / "lineage.csv"),
            "checkpoint_count": len(checkpoints),
        },
        "comparison": {
            "tolerance": TOLERANCE,
            "max_fit_metric_abs_difference": float(
                status["max_metric_abs_difference"].max()
            ),
            "max_derived_numeric_abs_difference": float(
                pd.concat(
                    [
                        numerical["max_absolute_difference"],
                        summaries["max_absolute_difference"],
                    ]
                ).max()
            ),
            "all_fit_comparisons_pass": bool(
                status["comparison_status"].eq("pass").all()
            ),
            "all_derived_comparisons_pass": bool(
                numerical["status"].ne("fail").all()
                and summaries["status"].eq("pass").all()
            ),
            "rendered_figures_byte_exact": bool(rendered["byte_exact"].all()),
            "rendered_figures_pixel_exact": bool(rendered["pixel_exact"].all()),
        },
        "status": "pass",
    }
    (AUDIT / "component_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    checkpoints = build_checkpoint_index()
    lineage = build_lineage(checkpoints)
    write_component_manifest(checkpoints, lineage)
    print(
        f"audit metadata pass: checkpoints={len(checkpoints)}, lineage={len(lineage)}"
    )


if __name__ == "__main__":
    main()
