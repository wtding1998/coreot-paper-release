from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[7]
UNIT = Path(__file__).resolve().parents[2]
PACKAGE = UNIT / "package_staging/pbmc_matchability_attribution"
STAGED_REPOSITORY = PACKAGE / "code/repository"
sys.path.insert(0, str(STAGED_REPOSITORY))
sys.path.insert(0, str(STAGED_REPOSITORY / "src"))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from experiments.component_ablation_surfaces import (  # noqa: E402
    _pbmc_condition_map,
    evaluate_scores,
)
from experiments.pbmc_rho_tau_heatmap import (  # noqa: E402
    FOCUSED_TAU_VALUES,
    render_s2_aligned_heatmap,
    summarize,
)
from experiments.rho_attribution_search import (  # noqa: E402
    _render,
    summarize_results,
)


VALIDATION = UNIT / "package_validation/validation_report.md"
FOCUSED_SOURCE = REPO / (
    "results/PBMC/sensitivity/"
    "rho_attribution_tau_surface_alpha0_range075_175/tables/"
    "rho_tau_surface_range075_175_by_seed.csv"
)
ALPHA_SOURCE = REPO / (
    "results/PBMC/sensitivity/rho_attribution_alpha_search/tables/"
    "rho_attribution_by_replicate.csv"
)
RESOURCE_SOURCE = REPO / (
    "results/submission_verification/PBMC/2026-08-03/"
    "baseline_resource_audit/compare_baselines_by_method.csv"
)
RAW = REPO / "data/raw/kang_2018.h5ad"
METRICS = ("auroc", "auprc", "forced_accuracy", "forced_macro_f1")
VARIANTS = ("heterogeneous", "mean_matched_uniform")
TOLERANCE = 1.0e-12
HDF5_SUFFIXES = {".h5", ".hdf5", ".h5ad", ".hdf", ".he5"}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
CONFIG_CODE_PATHS = {
    "pbmc_rho_tau_heatmap": "experiments/pbmc_rho_tau_heatmap.py",
    "rho_attribution_search": "experiments/rho_attribution_search.py",
    "component_ablation_surfaces": "experiments/component_ablation_surfaces.py",
    "mouse_component_ablation": "experiments/mouse_spleen/component_ablation.py",
    "artifact_hashes": "src/coreot/artifacts/hashes.py",
    "artifact_manifests": "src/coreot/artifacts/manifests.py",
    "config_load": "src/coreot/config/load.py",
    "data_schemas": "src/coreot/data/schemas.py",
    "data_validation": "src/coreot/data/validation.py",
    "evaluation_metrics": "src/coreot/evaluation/metrics.py",
    "provider_reliability": "src/coreot/preprocessing/provider_reliability.py",
    "label_transfer": "src/coreot/transport/label_transfer.py",
    "transport_runner": "src/coreot/transport/runner.py",
    "sinkhorn_solver": "src/coreot/transport/sinkhorn.py",
    "reconstruction_worker": (
        "results/submission_verification/PBMC/2026-08-05/"
        "pbmc_matchability_attribution_lineage_reconstruction/"
        "audit/workers/reconstruct_fits.py"
    ),
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def endpoint_slug(value: str) -> str:
    return value.lower().replace("+", "").replace("-", "").replace(" ", "_")


def value_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def string_sequence_hash(values: pd.Series) -> str:
    payload = "\n".join(values.astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def schema_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{column}:{frame[column].dtype}" for column in frame.columns
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_cases() -> pd.DataFrame:
    focused = pd.read_csv(FOCUSED_SOURCE)
    alpha = pd.read_csv(ALPHA_SOURCE)
    require(len(focused) == 225, f"Focused source rows: {len(focused)}")
    require(len(alpha) == 75, f"Alpha source rows: {len(alpha)}")
    require(
        not focused.duplicated(["endpoint", "seed", "tau_min", "tau_max"]).any(),
        "Focused source keys duplicate",
    )
    require(
        not alpha.duplicated(["endpoint", "seed", "alpha"]).any(),
        "Alpha source keys duplicate",
    )
    focused["analysis_family"] = "focused_tau"
    alpha["analysis_family"] = "alpha_search"
    return pd.concat([focused, alpha], ignore_index=True, sort=False)


def case_slug(case: object) -> str:
    if str(case.analysis_family) == "focused_tau":
        return (
            f"tau_min_{value_slug(case.tau_min)}_"
            f"tau_max_{value_slug(case.tau_max)}"
        )
    return f"alpha_{value_slug(case.alpha)}"


def fit_id(case: object, variant: str) -> str:
    return (
        f"{case.analysis_family}_{endpoint_slug(str(case.endpoint))}_"
        f"seed{int(case.seed)}_{case_slug(case)}_{variant}"
    )


def expected_fit_root(case: object, variant: str) -> Path:
    return (
        PACKAGE
        / "fit_evidence"
        / str(case.analysis_family)
        / endpoint_slug(str(case.endpoint))
        / f"seed{int(case.seed)}"
        / case_slug(case)
        / variant
    )


def expected_method(case: object, variant: str) -> dict[str, object]:
    common: dict[str, object] = {
        "tau_target": float(case.tau_target),
        "epsilon": 0.05,
        "alpha": float(case.alpha),
        "max_iter": 5000,
        "tol": 1.0e-6,
        "numerical_floor": 1.0e-300,
        "eta": 1.0e-12,
    }
    if variant == "heterogeneous":
        return {
            **common,
            "name": "coreot_full",
            "tau_min": float(case.tau_min),
            "tau_max": float(case.tau_max),
        }
    return {
        **common,
        "name": "coreot_constant_tau",
        "tau_source": "matched_coreot_mean",
        "matched_tau_min": float(case.tau_min),
        "matched_tau_max": float(case.tau_max),
    }


def validate_inventory() -> tuple[int, int]:
    files = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    require(files, "Package is empty")
    for path in PACKAGE.rglob("*"):
        require(not path.is_symlink(), f"Symlink forbidden: {path}")
        require("__pycache__" not in path.parts, f"Bytecode cache forbidden: {path}")
        if path.is_file():
            require(os.stat(path).st_nlink == 1, f"Hard link forbidden: {path}")
            require(path.suffix != ".pyc", f"Python bytecode forbidden: {path}")
            require(
                path.suffix.lower() not in HDF5_SUFFIXES,
                f"HDF5 suffix forbidden: {path}",
            )
            with path.open("rb") as handle:
                require(handle.read(8) != HDF5_MAGIC, f"HDF5 content: {path}")

    included_path = PACKAGE / "manifests/included_files.csv"
    checksums_path = PACKAGE / "manifests/checksums.sha256"
    included = pd.read_csv(included_path)
    require(not included["package_path"].duplicated().any(), "Inventory duplicates")
    expected_included = {
        str(path.relative_to(PACKAGE))
        for path in files
        if path not in (included_path, checksums_path)
    }
    require(set(included["package_path"]) == expected_included, "Inventory gap")
    for row in included.itertuples(index=False):
        path = PACKAGE / str(row.package_path)
        require(sha256_file(path) == str(row.sha256), f"Inventory hash: {path}")
        require(path.stat().st_size == int(row.bytes), f"Inventory bytes: {path}")

    checksum_records: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        require(relative not in checksum_records, f"Checksum duplicate: {relative}")
        checksum_records[relative] = digest
    expected_checksums = {
        str(path.relative_to(PACKAGE)) for path in files if path != checksums_path
    }
    require(set(checksum_records) == expected_checksums, "Checksum coverage gap")
    for relative, digest in checksum_records.items():
        require(
            sha256_file(PACKAGE / relative) == digest,
            f"Checksum mismatch: {relative}",
        )
    return len(files), sum(path.stat().st_size for path in files)


def validate_code() -> tuple[dict[str, str], int]:
    records = pd.read_csv(PACKAGE / "manifests/code_dependencies.csv")
    require(not records["package_path"].duplicated().any(), "Code paths duplicate")
    require(
        {"scientific_or_derivation_code", "verification_code"}.issubset(
            set(records["role"])
        ),
        "Code roles incomplete",
    )
    for row in records.itertuples(index=False):
        source = REPO / str(row.repository_path)
        staged = PACKAGE / str(row.package_path)
        require(source.is_file() and staged.is_file(), f"Missing code: {row.package_path}")
        require(sha256_file(source) == str(row.sha256), f"Source code drift: {source}")
        require(sha256_file(staged) == str(row.sha256), f"Staged code drift: {staged}")
        require(staged.stat().st_size == int(row.bytes), f"Code size: {staged}")
        compile(staged.read_text(encoding="utf-8"), str(staged), "exec")
    support = pd.read_csv(PACKAGE / "manifests/runtime_support.csv")
    require(len(support) == 7, f"Runtime support rows: {len(support)}")
    for row in support.itertuples(index=False):
        source = REPO / str(row.repository_path)
        staged = PACKAGE / str(row.package_path)
        require(sha256_file(source) == str(row.sha256), f"Runtime source: {source}")
        require(sha256_file(staged) == str(row.sha256), f"Runtime staged: {staged}")
        compile(staged.read_text(encoding="utf-8"), str(staged), "exec")
    for name in ("pyproject.toml", "uv.lock"):
        require(
            sha256_file(PACKAGE / "code/environment" / name)
            == sha256_file(REPO / name),
            f"Environment lock drift: {name}",
        )
    imported_component = Path(sys.modules["experiments.component_ablation_surfaces"].__file__)
    imported_hashes = Path(sys.modules["coreot.artifacts.hashes"].__file__)
    require(imported_component.is_relative_to(PACKAGE), "Component import not staged")
    require(imported_hashes.is_relative_to(PACKAGE), "Hash import not staged")
    source_hashes = {
        key: sha256_file(REPO / relative)
        for key, relative in CONFIG_CODE_PATHS.items()
    }
    return source_hashes, len(records)


def validate_external_dependencies() -> tuple[pd.DataFrame, set[str]]:
    external = pd.read_csv(PACKAGE / "manifests/external_dependencies.csv")
    require(len(external) == 91, f"External dependencies: {len(external)}")
    require(
        not external["repository_path"].duplicated().any(),
        "External dependency duplicates",
    )
    require(
        external["included_in_package"].astype(str).str.lower().eq("false").all(),
        "External inclusion flag",
    )
    for row in external.itertuples(index=False):
        path = REPO / str(row.repository_path)
        require(path.is_file(), f"Missing external dependency: {path}")
        require(sha256_file(path) == str(row.sha256), f"External hash: {path}")
        require(path.stat().st_size == int(row.bytes), f"External bytes: {path}")
    raw = external.loc[external["repository_path"].eq("data/raw/kang_2018.h5ad")]
    require(len(raw) == 1, "Raw PBMC object is not declared once")
    require(
        str(raw.iloc[0]["sha256"])
        == "229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff",
        "Raw PBMC checksum",
    )
    require(int(raw.iloc[0]["bytes"]) == 123669441, "Raw PBMC bytes")
    return external, set(external["repository_path"])


def validate_dependency_index(expected_ids: set[str]) -> None:
    index = pd.read_csv(PACKAGE / "manifests/fit_dependencies.csv")
    require(len(index) == 600, f"Fit dependency rows: {len(index)}")
    require(not index["fit_id"].duplicated().any(), "Fit dependency IDs duplicate")
    require(set(index["fit_id"]) == expected_ids, "Fit dependency key set")
    for row in index.itertuples(index=False):
        config_path = PACKAGE / str(row.resolved_config)
        manifest_path = PACKAGE / str(row.fit_manifest)
        score_path = PACKAGE / str(row.cell_scores)
        require(config_path.is_file(), f"Dependency config missing: {row.fit_id}")
        require(manifest_path.is_file(), f"Dependency manifest missing: {row.fit_id}")
        require(score_path.is_file(), f"Dependency scores missing: {row.fit_id}")
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(str(resolved["fit_id"]) == str(row.fit_id), f"Dependency ID: {row.fit_id}")
        require(
            str(resolved["analysis_family"]) == str(row.analysis_family),
            f"Dependency family: {row.fit_id}",
        )
        require(str(resolved["variant"]) == str(row.variant), f"Dependency variant: {row.fit_id}")

    audit_index = PACKAGE / "audits/fit_dependencies.csv"
    require(audit_index.is_file(), "Audit fit-dependency index missing")
    audit = pd.read_csv(audit_index)
    require(len(audit) == 600 and audit["fit_id"].nunique() == 600, "Audit dependency index")
    require(set(audit["fit_id"]) == expected_ids, "Audit dependency keys")
    for family in (
        "rho_attribution_tau_surface_alpha0_range075_175",
        "rho_attribution_alpha_search",
    ):
        manifest = yaml.safe_load(
            (
                PACKAGE / "verified_results" / family / "manifest.yaml"
            ).read_text(encoding="utf-8")
        )
        require(
            manifest["fit_dependency_index_sha256"] == sha256_file(audit_index),
            f"Derived dependency-index hash: {family}",
        )
        require(
            int(manifest["fit_dependency_index_bytes"]) == audit_index.stat().st_size,
            f"Derived dependency-index bytes: {family}",
        )
        code = manifest["code_sha256"]
        derivation_worker = PACKAGE / "code/verification/build_outputs.py"
        require(
            code["derivation_worker"] == sha256_file(derivation_worker),
            f"Derivation worker lock: {family}",
        )
        environment = manifest["environment"]
        require(
            environment["pyproject_sha256"]
            == sha256_file(PACKAGE / "code/environment/pyproject.toml"),
            f"Pyproject lock: {family}",
        )
        require(
            environment["uv_lock_sha256"]
            == sha256_file(PACKAGE / "code/environment/uv.lock"),
            f"uv.lock lock: {family}",
        )


def validate_fits(
    cases: pd.DataFrame,
    external_paths: set[str],
    expected_code_hashes: dict[str, str],
) -> tuple[float, int, set[str]]:
    actual_roots = sorted(
        path.parent for path in PACKAGE.glob("fit_evidence/**/cell_scores.parquet")
    )
    require(len(actual_roots) == 600, f"Packaged fits: {len(actual_roots)}")
    require(len(set(actual_roots)) == 600, "Packaged fit roots duplicate")
    expected_roots = {
        expected_fit_root(case, variant)
        for case in cases.itertuples(index=False)
        for variant in VARIANTS
    }
    require(set(actual_roots) == expected_roots, "Exact 600-fit key/root set differs")
    condition_map = _pbmc_condition_map(RAW)
    maximum_difference = 0.0
    total_score_rows = 0
    declared_inputs: set[str] = set()
    truth_cache: dict[str, pd.DataFrame] = {}

    for case in cases.itertuples(index=False):
        for variant in VARIANTS:
            root = expected_fit_root(case, variant)
            entries = {path.name for path in root.iterdir() if path.is_file()}
            require(
                entries
                == {"resolved_config.yaml", "fit_manifest.yaml", "cell_scores.parquet"},
                f"Unexpected compact fit files: {root}: {entries}",
            )
            config_path = root / "resolved_config.yaml"
            manifest_path = root / "fit_manifest.yaml"
            score_path = root / "cell_scores.parquet"
            resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            require(resolved["fit_id"] == fit_id(case, variant), f"Fit ID: {root}")
            require(resolved["analysis_family"] == case.analysis_family, f"Family: {root}")
            require(resolved["experiment"] == "pbmc", f"Experiment: {root}")
            require(resolved["endpoint"] == str(case.endpoint), f"Endpoint: {root}")
            require(int(resolved["seed"]) == int(case.seed), f"Seed: {root}")
            require(resolved["base_run_id"] == str(case.run_id), f"Run ID: {root}")
            require(resolved["condition"] == "incomplete_reference", f"Condition: {root}")
            require(resolved["candidate_set"] == "pca30_k100", f"Candidates: {root}")
            require(resolved["variant"] == variant, f"Variant: {root}")
            require(resolved["evaluation_scope"] == "within_celltype", f"Scope: {root}")
            require(resolved["method"] == expected_method(case, variant), f"Method: {root}")
            require(
                abs(float(resolved["mean_matched_tau"]) - float(case.mean_matched_tau))
                <= TOLERANCE,
                f"Mean-matched tau: {root}",
            )
            require(resolved["code_sha256"] == expected_code_hashes, f"Code lock: {root}")
            require(
                manifest["metadata"]["code_sha256"] == expected_code_hashes,
                f"Manifest code lock: {root}",
            )
            require(
                manifest["metadata"]["comparison_status"] == "pass",
                f"Fit comparison status: {root}",
            )
            require(
                manifest["metadata"]["environment"] == resolved["environment"],
                f"Environment declaration: {root}",
            )
            require(len(resolved["inputs"]) == 7, f"Input count: {root}")
            for record in resolved["inputs"].values():
                relative = str(record["path"])
                declared_inputs.add(relative)
                path = REPO / relative
                require(relative in external_paths, f"Undeclared dependency: {relative}")
                require(sha256_file(path) == str(record["sha256"]), f"Input hash: {path}")
                require(path.stat().st_size == int(record["bytes"]), f"Input bytes: {path}")

            artifacts = manifest["artifacts"]
            config_artifact = artifacts["resolved_config"]
            score_artifact = artifacts["cell_scores"]
            require(PACKAGE / config_artifact["path"] == config_path, f"Config rebase: {root}")
            require(sha256_file(config_path) == config_artifact["sha256"], f"Config hash: {root}")
            require(config_path.stat().st_size == int(config_artifact["bytes"]), f"Config bytes: {root}")
            require(PACKAGE / score_artifact["path"] == score_path, f"Score rebase: {root}")
            require(sha256_file(score_path) == score_artifact["sha256"], f"Score hash: {root}")
            require(score_path.stat().st_size == int(score_artifact["bytes"]), f"Score bytes: {root}")
            source_manifest = REPO / manifest["packaging"]["source_fit_manifest_path"]
            require(
                sha256_file(source_manifest)
                == manifest["packaging"]["source_fit_manifest_sha256"],
                f"Source manifest drift: {root}",
            )

            scores = pd.read_parquet(score_path)
            require(list(scores.columns) == ["cell_id", "u", "forced_label"], f"Score schema: {root}")
            require(len(scores) == int(score_artifact["rows"]), f"Score rows: {root}")
            require(list(score_artifact["columns"]) == list(scores.columns), f"Score columns manifest: {root}")
            require(scores["cell_id"].notna().all(), f"Null cell ID: {root}")
            require(not scores["cell_id"].duplicated().any(), f"Duplicate cell ID: {root}")
            values = pd.to_numeric(scores["u"], errors="coerce").to_numpy(float)
            require(np.isfinite(values).all(), f"Non-finite u: {root}")
            require(scores["forced_label"].notna().all(), f"Null forced label: {root}")
            truth_relative = str(resolved["inputs"]["truth"]["path"])
            if truth_relative not in truth_cache:
                truth_cache[truth_relative] = pd.read_csv(REPO / truth_relative)
            truth = truth_cache[truth_relative]
            require(
                scores["cell_id"].astype(str).tolist()
                == truth["cell_id"].astype(str).tolist(),
                f"Truth cell ID/order: {root}",
            )
            require(
                set(scores["forced_label"].astype(str)).issubset(
                    set(truth["true_label"].dropna().astype(str))
                ),
                f"Forced-label domain: {root}",
            )
            require(
                score_artifact["cell_id_order_sha256"]
                == string_sequence_hash(scores["cell_id"]),
                f"Cell-ID order hash: {root}",
            )
            require(
                score_artifact["schema_sha256"] == schema_hash(scores),
                f"Schema hash: {root}",
            )
            metrics = evaluate_scores(
                experiment="pbmc",
                endpoint=str(case.endpoint),
                truth=truth,
                scores=scores,
                pbmc_condition=condition_map,
            )
            prefix = "heterogeneous" if variant == "heterogeneous" else "uniform"
            for metric in METRICS:
                difference = abs(
                    float(metrics[metric]) - float(getattr(case, f"{prefix}_{metric}"))
                )
                maximum_difference = max(maximum_difference, difference)
                require(difference <= TOLERANCE, f"{metric} differs at {root}: {difference}")
            transport = manifest["metadata"]["transport_metadata"]
            require(
                bool(transport["converged"])
                == bool(getattr(case, f"{prefix}_converged")),
                f"Convergence: {root}",
            )
            require(
                int(transport["n_iter"])
                == int(getattr(case, f"{prefix}_n_iterations")),
                f"Iterations: {root}",
            )
            if str(case.analysis_family) == "alpha_search":
                require(int(metrics["n_detection"]) == int(case.n_detection), f"n_detection: {root}")
                require(int(metrics["n_positive"]) == int(case.n_positive), f"n_positive: {root}")
            total_score_rows += len(scores)
    require(declared_inputs == external_paths, "Fit-input union is not exact")
    status = pd.read_csv(PACKAGE / "audits/reconstruction_status.csv")
    require(len(status) == 600 and status["fit_id"].nunique() == 600, "Status grid")
    require(status["comparison_status"].eq("pass").all(), "Status failures")
    return maximum_difference, total_score_rows, declared_inputs


def assert_frames_close(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    keys: list[str],
    excluded: set[str] | None = None,
) -> float:
    excluded = excluded or set()
    expected = expected.sort_values(keys).reset_index(drop=True)
    observed = observed.sort_values(keys).reset_index(drop=True)
    require(list(expected.columns) == list(observed.columns), "Frame columns differ")
    require(len(expected) == len(observed), "Frame rows differ")
    maximum = 0.0
    for column in expected.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
            require(np.array_equal(np.isnan(left), np.isnan(right)), f"Missingness: {column}")
            difference = float(np.nanmax(np.abs(left - right))) if len(left) else 0.0
            maximum = max(maximum, difference)
            require(difference <= TOLERANCE, f"Numeric column differs: {column}: {difference}")
        else:
            require(
                expected[column].fillna("<NA>").astype(str).equals(
                    observed[column].fillna("<NA>").astype(str)
                ),
                f"Categorical column differs: {column}",
            )
    return maximum


def validate_outputs() -> tuple[float, bool]:
    focused_root = PACKAGE / "verified_results" / (
        "rho_attribution_tau_surface_alpha0_range075_175"
    )
    alpha_root = PACKAGE / "verified_results/rho_attribution_alpha_search"
    focused = pd.read_csv(
        focused_root / "tables/rho_tau_surface_range075_175_by_seed.csv"
    )
    focused_summary = pd.read_csv(
        focused_root / "tables/rho_tau_surface_range075_175_summary.csv"
    )
    alpha = pd.read_csv(alpha_root / "tables/rho_attribution_by_replicate.csv")
    alpha_summary = pd.read_csv(alpha_root / "tables/rho_attribution_summary.csv")
    maximum = assert_frames_close(
        pd.read_csv(FOCUSED_SOURCE),
        focused,
        keys=["endpoint", "seed", "tau_min", "tau_max"],
    )
    maximum = max(
        maximum,
        assert_frames_close(
            pd.read_csv(ALPHA_SOURCE),
            alpha,
            keys=["endpoint", "seed", "alpha"],
            excluded={"heterogeneous_runtime_seconds", "uniform_runtime_seconds"},
        ),
    )
    recomputed_focused = summarize(focused)
    recomputed_alpha = summarize_results(alpha)
    maximum = max(
        maximum,
        assert_frames_close(
            focused_summary,
            recomputed_focused,
            keys=["endpoint", "tau_min", "tau_max"],
        ),
        assert_frames_close(
            alpha_summary,
            recomputed_alpha,
            keys=["endpoint", "alpha"],
        ),
    )
    focused_checkpoints = sorted((focused_root / "checkpoints").glob("*/seed*.csv"))
    alpha_checkpoints = sorted((alpha_root / "checkpoints").glob("*/seed*.csv"))
    require(len(focused_checkpoints) == 15, "Focused checkpoint count")
    require(len(alpha_checkpoints) == 15, "Alpha checkpoint count")
    require(sum(len(pd.read_csv(path)) for path in focused_checkpoints) == 225, "Focused checkpoint rows")
    require(sum(len(pd.read_csv(path)) for path in alpha_checkpoints) == 75, "Alpha checkpoint rows")

    with tempfile.TemporaryDirectory(prefix="pbmc-matchability-validate-") as directory:
        temporary = Path(directory)
        focused_render = temporary / "focused.png"
        alpha_render = temporary / "alpha.png"
        render_s2_aligned_heatmap(
            focused_summary,
            focused_render,
            tau_values=FOCUSED_TAU_VALUES,
        )
        _render(alpha_summary, alpha, alpha_render)
        packaged_focused = PACKAGE / "verified_results/figures/" / (
            "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png"
        )
        packaged_alpha = PACKAGE / "verified_results/figures/" / (
            "manuscript_fig_pbmc_rho_attribution_alpha_search.png"
        )
        require(
            sha256_file(focused_render) == sha256_file(packaged_focused),
            "Focused staged-code render differs",
        )
        require(
            sha256_file(alpha_render) == sha256_file(packaged_alpha),
            "Alpha staged-code render differs",
        )
    comparisons = PACKAGE / "audits/comparison"
    numerical = pd.read_csv(comparisons / "numerical_comparison.csv")
    summaries = pd.read_csv(comparisons / "summary_comparison.csv")
    rendered = pd.read_csv(comparisons / "rendered_comparison.csv")
    require(not numerical["status"].eq("fail").any(), "Numerical comparison failure")
    require(summaries["status"].eq("pass").all(), "Summary comparison failure")
    require(len(rendered) == 2 and rendered["status"].eq("pass").all(), "Figure comparison")
    require(rendered["byte_exact"].all() and rendered["pixel_exact"].all(), "Figure exactness")
    return maximum, True


def validate_resources() -> tuple[int, float, float]:
    packaged = PACKAGE / "audits/baseline_resource_audit/compare_baselines_by_method.csv"
    reference = yaml.safe_load(
        (PACKAGE / "manifests/resource_audit_reference.yaml").read_text(
            encoding="utf-8"
        )
    )
    require(sha256_file(packaged) == sha256_file(RESOURCE_SOURCE), "Resource CSV drift")
    require(reference["source_summary_sha256"] == sha256_file(RESOURCE_SOURCE), "Resource reference hash")
    require(int(reference["source_summary_bytes"]) == RESOURCE_SOURCE.stat().st_size, "Resource reference bytes")
    frame = pd.read_csv(packaged)
    require(len(frame) == 12 and frame["method"].nunique() == 12, "Resource method coverage")
    require(frame["measurement_status"].isin(["measured", "unavailable"]).all(), "Resource completion")
    require(frame["measurement_status"].eq("measured").all(), "Unexpected unavailable resource row")
    require(frame["measurement_scope"].eq("representative_run").all(), "Resource scope")
    require(frame["parallel_jobs"].eq(1).all(), "Resource parallel jobs")
    require(frame["seed_or_split"].eq("B cells seed1").all(), "Resource representative split")
    return len(frame), float(frame["wall_seconds"].max()), float(frame["peak_rss_gib"].max())


def main() -> None:
    try:
        cases = load_cases()
        expected_ids = {
            fit_id(case, variant)
            for case in cases.itertuples(index=False)
            for variant in VARIANTS
        }
        require(len(expected_ids) == 600, f"Expected fit IDs: {len(expected_ids)}")
        file_count, package_bytes = validate_inventory()
        code_hashes, code_count = validate_code()
        external, external_paths = validate_external_dependencies()
        validate_dependency_index(expected_ids)
        fit_max, score_rows, declared_inputs = validate_fits(
            cases, external_paths, code_hashes
        )
        table_max, figures_exact = validate_outputs()
        resource_methods, max_wall, max_rss = validate_resources()
    except Exception as error:
        VALIDATION.parent.mkdir(parents=True, exist_ok=True)
        VALIDATION.write_text(
            "# PBMC matchability-attribution package validation\n\n"
            "Status: **FAIL**\n\n"
            f"Failure: `{type(error).__name__}: {error}`\n",
            encoding="utf-8",
        )
        raise
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(
        "# PBMC matchability-attribution package validation\n\n"
        "Status: **PASS**\n\n"
        f"- Package files: {file_count:,}; bytes: {package_bytes:,}.\n"
        "- Exact retained design: 300 paired cases, 600 fits, and 1,800 compact per-fit files.\n"
        f"- Compact score rows independently re-evaluated from staged code: {score_rows:,}; maximum fit-metric absolute difference: {fit_max:.3e}.\n"
        f"- External dependencies: {len(external):,}; exact fit-input union: {len(declared_inputs):,}. The canonical HDF5 object remains external.\n"
        f"- Code closure: {code_count} checksum-locked code files plus runtime support and byte-exact `pyproject.toml`/`uv.lock`.\n"
        f"- Recomputed tables/summaries maximum scientific difference: {table_max:.3e}; both staged-code figure renders byte exact: {figures_exact}.\n"
        f"- Dataset-level resource audit: `results/submission_verification/PBMC/2026-08-03/baseline_resource_audit/compare_baselines_by_method.csv`; {resource_methods} measured methods, maximum wall time {max_wall:.2f} s, maximum peak RSS {max_rss:.6f} GiB.\n"
        "- The validator found no checksum gaps, undeclared dependencies, stale staged code, symlinks, hard links, Python bytecode caches, or packaged HDF5 objects.\n"
        "- Historical fit bytes were not recovered; this package is an isolated exact scientific reconstruction from hash-matching retained inputs. Reconstruction runtime is new provenance and was not compared as a scientific result.\n",
        encoding="utf-8",
    )
    print(
        f"PASS files={file_count} fits=600 score_rows={score_rows} "
        f"fit_max={fit_max:.3e} table_max={table_max:.3e}"
    )


if __name__ == "__main__":
    main()
