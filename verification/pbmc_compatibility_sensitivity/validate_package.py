from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[7]
UNIT = Path(__file__).resolve().parents[2]
PACKAGE = UNIT / "package_staging/pbmc_compatibility_sensitivity"
STAGED_REPOSITORY = PACKAGE / "code/repository"
for import_root in (REPO, STAGED_REPOSITORY, STAGED_REPOSITORY / "src"):
    if import_root.is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from experiments.component_ablation_surfaces import (  # noqa: E402
    _pbmc_condition_map,
    evaluate_scores,
)


VALIDATION = UNIT / "package_validation/validation_report.md"
SOURCE_TABLE = (
    REPO
    / "results/PBMC/sensitivity/component_ablation/tables/component_ablation_by_seed.csv"
)
METRICS = ("auroc", "auprc", "forced_accuracy", "forced_macro_f1")
KEYS = ("endpoint", "seed", "tau_source", "alpha")
TOLERANCE = 1.0e-12
HDF5_SUFFIXES = {".h5", ".hdf5", ".h5ad", ".hdf", ".he5"}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def endpoint_slug(value: str) -> str:
    return value.lower().replace("+", "").replace("-", "").replace(" ", "_")


def value_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def expected_fit_root(row: object) -> Path:
    return (
        PACKAGE
        / "fit_evidence"
        / endpoint_slug(str(row.endpoint))
        / f"seed{int(row.seed)}"
        / f"tau{value_slug(row.tau_source)}_alpha{value_slug(row.alpha)}"
    )


def load_historical() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_TABLE)
    frame = frame.loc[
        frame["experiment"].eq("pbmc")
        & frame["variant"].eq("compatibility_only")
    ].sort_values(list(KEYS)).reset_index(drop=True)
    require(len(frame) == 375, f"Canonical focused grid has {len(frame)} rows")
    require(not frame.duplicated(list(KEYS)).any(), "Canonical focused keys duplicate")
    return frame


def validate_inventory() -> tuple[int, int]:
    files = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    require(files, "Package is empty")
    for path in PACKAGE.rglob("*"):
        require(not path.is_symlink(), f"Symlink forbidden: {path}")
        if path.is_file():
            require(os.stat(path).st_nlink == 1, f"Hard link forbidden: {path}")
            require(path.suffix.lower() not in HDF5_SUFFIXES, f"HDF5 suffix: {path}")
            with path.open("rb") as handle:
                require(handle.read(8) != HDF5_MAGIC, f"HDF5 magic bytes: {path}")
    included_path = PACKAGE / "manifests/included_files.csv"
    included = pd.read_csv(included_path)
    require(not included["package_path"].duplicated().any(), "Included inventory duplicates")
    expected_included = {
        str(path.relative_to(PACKAGE))
        for path in files
        if path
        not in (included_path, PACKAGE / "manifests/checksums.sha256")
    }
    require(set(included["package_path"]) == expected_included, "Included coverage gap")
    for row in included.itertuples(index=False):
        path = PACKAGE / str(row.package_path)
        require(sha256_file(path) == str(row.sha256), f"Included digest mismatch: {path}")
        require(path.stat().st_size == int(row.bytes), f"Included size mismatch: {path}")
    checksums_path = PACKAGE / "manifests/checksums.sha256"
    checksum_records: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        checksum_records[relative] = digest
    expected_checksums = {
        str(path.relative_to(PACKAGE)) for path in files if path != checksums_path
    }
    require(set(checksum_records) == expected_checksums, "Checksum coverage gap")
    for relative, digest in checksum_records.items():
        require(sha256_file(PACKAGE / relative) == digest, f"Checksum mismatch: {relative}")
    return len(files), sum(path.stat().st_size for path in files)


def validate_code() -> dict[str, str]:
    records = pd.read_csv(PACKAGE / "manifests/code_dependencies.csv")
    require(len(records) == 13, f"Expected 13 scientific code files, found {len(records)}")
    require(not records["code_name"].duplicated().any(), "Code names duplicate")
    for row in records.itertuples(index=False):
        source = REPO / str(row.repository_path)
        staged = PACKAGE / str(row.package_path)
        require(source.is_file() and staged.is_file(), f"Missing code dependency: {row.code_name}")
        require(sha256_file(source) == str(row.sha256), f"Source code drift: {source}")
        require(sha256_file(staged) == str(row.sha256), f"Staged code drift: {staged}")
        require(staged.stat().st_size == int(row.bytes), f"Staged code size: {staged}")
        compile(staged.read_text(encoding="utf-8"), str(staged), "exec")
    support = pd.read_csv(PACKAGE / "manifests/runtime_support.csv")
    require(len(support) == 7, f"Expected 7 runtime-support files, found {len(support)}")
    for row in support.itertuples(index=False):
        source = REPO / str(row.repository_path)
        staged = PACKAGE / str(row.package_path)
        require(sha256_file(source) == str(row.sha256), f"Runtime source drift: {source}")
        require(sha256_file(staged) == str(row.sha256), f"Runtime staged drift: {staged}")
        require(staged.stat().st_size == int(row.bytes), f"Runtime support size: {staged}")
        compile(staged.read_text(encoding="utf-8"), str(staged), "exec")
    imported_component = Path(sys.modules["experiments.component_ablation_surfaces"].__file__)
    imported_hashes = Path(sys.modules["coreot.artifacts.hashes"].__file__)
    require(imported_component.is_relative_to(PACKAGE), "Component code was not imported from package")
    require(imported_hashes.is_relative_to(PACKAGE), "Hash code was not imported from package")
    return dict(zip(records["code_name"], records["sha256"], strict=True))


def validate_dependencies() -> tuple[pd.DataFrame, set[str]]:
    external = pd.read_csv(PACKAGE / "manifests/external_dependencies.csv")
    require(len(external) == 91, f"Expected 91 dependencies, found {len(external)}")
    require(not external["repository_path"].duplicated().any(), "Dependency duplicates")
    require(external["included_in_package"].astype(str).str.lower().eq("false").all(), "Dependency inclusion flag")
    for row in external.itertuples(index=False):
        path = REPO / str(row.repository_path)
        require(path.is_file(), f"Missing external dependency: {path}")
        require(sha256_file(path) == str(row.sha256), f"Dependency digest drift: {path}")
        require(path.stat().st_size == int(row.bytes), f"Dependency size drift: {path}")
    raw = external.loc[external["repository_path"].eq("data/raw/kang_2018.h5ad")]
    require(len(raw) == 1, "Raw PBMC dependency declaration missing")
    require(
        str(raw.iloc[0]["sha256"])
        == "229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff",
        "Raw PBMC dependency digest mismatch",
    )
    return external, set(external["repository_path"])


def validate_fits(
    historical: pd.DataFrame,
    external_paths: set[str],
    expected_code_hashes: dict[str, str],
) -> tuple[float, int, set[str]]:
    actual_roots = sorted(path.parent for path in PACKAGE.glob("fit_evidence/*/seed*/tau*_alpha*/cell_scores.parquet"))
    require(len(actual_roots) == 375, f"Expected 375 packaged fits, found {len(actual_roots)}")
    require(len(set(actual_roots)) == 375, "Packaged fit roots duplicate")
    condition_map = _pbmc_condition_map(REPO / "data/raw/kang_2018.h5ad")
    maximum_difference = 0.0
    total_score_rows = 0
    declared_inputs: set[str] = set()
    for row in historical.itertuples(index=False):
        root = expected_fit_root(row)
        require(root in actual_roots, f"Missing expected fit root: {root}")
        entries = {path.name for path in root.iterdir() if path.is_file()}
        require(
            entries == {"resolved_config.yaml", "fit_manifest.yaml", "cell_scores.parquet"},
            f"Unexpected fit files at {root}: {entries}",
        )
        config_path = root / "resolved_config.yaml"
        manifest_path = root / "fit_manifest.yaml"
        score_path = root / "cell_scores.parquet"
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        expected_id = (
            f"{endpoint_slug(str(row.endpoint))}_seed{int(row.seed)}_"
            f"tau{value_slug(row.tau_source)}_alpha{value_slug(row.alpha)}"
        )
        require(resolved["fit_id"] == expected_id, f"Resolved fit ID mismatch: {root}")
        require(resolved["experiment"] == "pbmc", f"Experiment mismatch: {root}")
        require(resolved["endpoint"] == str(row.endpoint), f"Endpoint mismatch: {root}")
        require(int(resolved["seed"]) == int(row.seed), f"Seed mismatch: {root}")
        require(resolved["base_run_id"] == str(row.run_id), f"Run ID mismatch: {root}")
        require(resolved["variant"] == "compatibility_only", f"Variant mismatch: {root}")
        require(resolved["condition"] == "incomplete_reference", f"Condition mismatch: {root}")
        require(resolved["candidate_set"] == "pca30_k100", f"Candidate set mismatch: {root}")
        require(resolved["evaluation_scope"] == "within_celltype", f"Scope mismatch: {root}")
        method = resolved["method"]
        expected_method = {
            "name": "coreot_constant_tau",
            "tau_source": float(row.tau_source),
            "tau_target": 1.0,
            "alpha": float(row.alpha),
            "epsilon": 0.05,
            "max_iter": 5000,
            "tol": 1.0e-6,
            "numerical_floor": 1.0e-300,
            "eta": 1.0e-12,
        }
        require(method == expected_method, f"Method configuration mismatch: {root}")
        require(resolved["code_sha256"] == expected_code_hashes, f"Resolved code lock: {root}")
        for record in resolved["inputs"].values():
            relative = str(record["path"])
            declared_inputs.add(relative)
            path = REPO / relative
            require(relative in external_paths, f"Undeclared fit dependency: {relative}")
            require(sha256_file(path) == str(record["sha256"]), f"Fit input digest: {path}")
            require(path.stat().st_size == int(record["bytes"]), f"Fit input size: {path}")
        artifacts = manifest["artifacts"]
        require(manifest["metadata"]["comparison_status"] == "pass", f"Fit status: {root}")
        require(
            manifest["metadata"]["code_sha256"] == expected_code_hashes,
            f"Manifest code lock: {root}",
        )
        require(
            manifest["metadata"]["environment"] == resolved["environment"],
            f"Environment declaration mismatch: {root}",
        )
        config_artifact = artifacts["resolved_config"]
        score_artifact = artifacts["cell_scores"]
        require(PACKAGE / config_artifact["path"] == config_path, f"Config path rebase: {root}")
        require(sha256_file(config_path) == str(config_artifact["sha256"]), f"Config digest: {root}")
        require(config_path.stat().st_size == int(config_artifact["bytes"]), f"Config size: {root}")
        require(PACKAGE / score_artifact["path"] == score_path, f"Score path rebase: {root}")
        require(sha256_file(score_path) == str(score_artifact["sha256"]), f"Score digest: {root}")
        require(score_path.stat().st_size == int(score_artifact["bytes"]), f"Score size: {root}")
        source_manifest = REPO / manifest["packaging"]["source_fit_manifest_path"]
        require(
            sha256_file(source_manifest) == manifest["packaging"]["source_fit_manifest_sha256"],
            f"Source manifest drift: {root}",
        )
        scores = pd.read_parquet(score_path)
        require(list(scores.columns) == ["cell_id", "u", "forced_label"], f"Score schema: {root}")
        require(len(scores) == int(score_artifact["rows"]), f"Score row count: {root}")
        require(list(score_artifact["columns"]) == list(scores.columns), f"Manifest schema: {root}")
        require(scores["cell_id"].notna().all(), f"Null cell ID: {root}")
        require(not scores["cell_id"].duplicated().any(), f"Duplicate score cell ID: {root}")
        require(np.issubdtype(scores["u"].dtype, np.number), f"Non-numeric u: {root}")
        require(np.isfinite(scores["u"].to_numpy(dtype=float)).all(), f"Non-finite u: {root}")
        require(scores["forced_label"].notna().all(), f"Null forced label: {root}")
        truth_path = REPO / resolved["inputs"]["truth"]["path"]
        truth = pd.read_csv(truth_path)
        require(set(scores["cell_id"]) == set(truth["cell_id"]), f"Truth cell IDs differ: {root}")
        metrics = evaluate_scores(
            experiment="pbmc",
            endpoint=str(row.endpoint),
            truth=truth,
            scores=scores,
            pbmc_condition=condition_map,
        )
        require(int(metrics["n_detection"]) == int(row.n_detection), f"n_detection: {root}")
        require(int(metrics["n_positive"]) == int(row.n_positive), f"n_positive: {root}")
        for metric in METRICS:
            difference = abs(float(metrics[metric]) - float(getattr(row, metric)))
            maximum_difference = max(maximum_difference, difference)
            require(difference <= TOLERANCE, f"{metric} differs at {root}: {difference}")
        transport = manifest["metadata"]["transport_metadata"]
        require(bool(transport["converged"]) == bool(row.converged), f"Convergence: {root}")
        require(int(transport["n_iter"]) == int(row.n_iterations), f"Iterations: {root}")
        total_score_rows += len(scores)
    require(declared_inputs == external_paths, "Fit dependency union is not exact")
    status = pd.read_csv(PACKAGE / "audits/reconstruction_status.csv")
    require(len(status) == 375 and status["fit_id"].nunique() == 375, "Status grid incomplete")
    require(status["comparison_status"].eq("pass").all(), "Non-passing reconstruction status")
    return maximum_difference, total_score_rows, declared_inputs


def validate_outputs() -> tuple[float, bool]:
    numerical = pd.read_csv(PACKAGE / "audits/comparison/numerical_comparison.csv")
    summary = pd.read_csv(PACKAGE / "audits/comparison/summary_comparison.csv")
    rendered = pd.read_csv(PACKAGE / "audits/comparison/rendered_comparison.csv")
    require(len(numerical) == 375 and numerical["status"].eq("pass").all(), "Numerical comparison")
    require(len(summary) == 75 and summary["status"].eq("pass").all(), "Summary comparison")
    require(
        len(rendered) == 1
        and rendered["status"].isin(["pass", "expected_change"]).all()
        and rendered["visual_layout_pass"].all(),
        "Rendered comparison",
    )
    component = yaml.safe_load((PACKAGE / "audits/component_manifest.yaml").read_text(encoding="utf-8"))
    require(component["inputs"]["fit_count"] == 375, "Component fit count")
    require(component["metadata"]["all_fit_comparisons_pass"], "Component fit comparison")
    require(component["metadata"]["all_summary_comparisons_pass"], "Component summary comparison")
    require(
        component["metadata"]["rendered_figure_pixel_exact"]
        or component["metadata"]["rendered_figure_expected_change"],
        "Component figure comparison",
    )
    require(
        component["metadata"]["rendered_figure_visual_layout_pass"],
        "Component figure layout",
    )
    checkpoints = sorted(PACKAGE.glob("verified_results/checkpoints/*/seed*/compatibility_only.csv"))
    require(len(checkpoints) == 15, f"Checkpoint count: {len(checkpoints)}")
    require(sum(len(pd.read_csv(path)) for path in checkpoints) == 375, "Checkpoint row count")
    by_seed = pd.read_csv(PACKAGE / "verified_results/tables/component_ablation_by_seed.csv")
    aggregate = pd.concat([pd.read_csv(path) for path in checkpoints], ignore_index=True)
    pd.testing.assert_frame_equal(
        by_seed.sort_values(list(KEYS)).reset_index(drop=True),
        aggregate.sort_values(list(KEYS)).reset_index(drop=True),
        check_exact=True,
    )
    packaged_figure = PACKAGE / "verified_results/figures/pbmc_compatibility_sensitivity.png"
    regenerated_figure = UNIT / "regenerated/figures/manuscript_fig_pbmc_component_minus_m.png"
    require(sha256_file(packaged_figure) == sha256_file(regenerated_figure), "Packaged figure drift")
    maximum_difference = max(
        float(numerical["max_metric_abs_difference"].max()),
        float(summary["max_statistic_abs_difference"].max()),
    )
    return maximum_difference, bool(rendered.loc[0, "pixel_exact"])


def validate_resources() -> tuple[int, float, float]:
    packaged = PACKAGE / "audits/baseline_resource_audit/compare_baselines_by_method.csv"
    source = REPO / "results/submission_verification/PBMC/2026-08-03/baseline_resource_audit/compare_baselines_by_method.csv"
    require(sha256_file(packaged) == sha256_file(source), "Resource audit byte drift")
    frame = pd.read_csv(packaged)
    require(len(frame) == 12 and frame["method"].nunique() == 12, "Resource method coverage")
    require(frame["measurement_status"].eq("measured").all(), "Resource status")
    require(frame["measurement_scope"].eq("representative_run").all(), "Resource scope")
    require(frame["parallel_jobs"].eq(1).all(), "Resource parallelism")
    require(frame["seed_or_split"].eq("B cells seed1").all(), "Resource representative")
    return len(frame), float(frame["wall_seconds"].max()), float(frame["peak_rss_gib"].max())


def main() -> int:
    try:
        file_count, package_bytes = validate_inventory()
        code_hashes = validate_code()
        external, external_paths = validate_dependencies()
        historical = load_historical()
        fit_max, score_rows, declared_inputs = validate_fits(
            historical, external_paths, code_hashes
        )
        table_max, pixel_exact = validate_outputs()
        resource_methods, max_wall, max_rss = validate_resources()
    except Exception as error:
        VALIDATION.parent.mkdir(parents=True, exist_ok=True)
        VALIDATION.write_text(
            "# PBMC compatibility-sensitivity package validation\n\n"
            "Status: **FAIL**\n\n"
            f"Failure: `{type(error).__name__}: {error}`\n",
            encoding="utf-8",
        )
        raise
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(
        "# PBMC compatibility-sensitivity package validation\n\n"
        "Status: **PASS**\n\n"
        f"- Package files: {file_count:,}; bytes: {package_bytes:,}.\n"
        "- Required fit evidence: 375 fits and 1,125 per-fit files; all configurations, manifests, and compact scores passed.\n"
        f"- Compact score rows re-evaluated: {score_rows:,}; maximum fit-metric absolute difference: {fit_max:.3e}.\n"
        f"- Declared external dependencies: {len(external):,}; exact fit-input union: {len(declared_inputs):,}.\n"
        f"- Scientific code closure: {len(code_hashes)} byte-identical, syntax-valid files.\n"
        f"- Reconstructed checkpoint/table maximum metric difference: {table_max:.3e}; figure pixel exact: {pixel_exact}. "
        "The expected render change is current-code variant-local color normalization; dimensions and visual layout pass.\n"
        f"- Shared resource audit: {resource_methods} measured methods; maximum wall time {max_wall:.2f} s; maximum peak RSS {max_rss:.6f} GiB.\n"
        "- No symlinks, hard links, HDF5-family suffixes, or HDF5 magic bytes were found.\n"
        "- Runtime was not compared because reconstruction runtime is new provenance.\n",
        encoding="utf-8",
    )
    print(
        f"PASS files={file_count} fits={len(historical)} score_rows={score_rows} "
        f"max_metric_difference={fit_max:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
