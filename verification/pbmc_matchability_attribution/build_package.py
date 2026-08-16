from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coreot.artifacts.hashes import sha256_file  # noqa: E402


UNIT = Path(__file__).resolve().parents[2]
PACKAGE = UNIT / "package_staging/pbmc_matchability_attribution"
FITS = UNIT / "regenerated/fits"
RESOURCE_ROOT = (
    REPO
    / "results/submission_verification/PBMC/2026-08-03/baseline_resource_audit"
)
REPOSITORY_CODE_PATHS = (
    REPO / "experiments/pbmc_rho_tau_heatmap.py",
    REPO / "experiments/rho_attribution_search.py",
    REPO / "experiments/component_ablation_surfaces.py",
    REPO / "experiments/mouse_spleen/component_ablation.py",
    REPO / "src/coreot/artifacts/hashes.py",
    REPO / "src/coreot/artifacts/manifests.py",
    REPO / "src/coreot/config/load.py",
    REPO / "src/coreot/data/hidden.py",
    REPO / "src/coreot/data/schemas.py",
    REPO / "src/coreot/data/validation.py",
    REPO / "src/coreot/evaluation/metrics.py",
    REPO / "src/coreot/preprocessing/provider_reliability.py",
    REPO / "src/coreot/transport/label_transfer.py",
    REPO / "src/coreot/transport/runner.py",
    REPO / "src/coreot/transport/sinkhorn.py",
)
RUNTIME_SUPPORT_PATHS = (
    REPO / "src/coreot/__init__.py",
    REPO / "src/coreot/artifacts/__init__.py",
    REPO / "src/coreot/config/__init__.py",
    REPO / "src/coreot/data/__init__.py",
    REPO / "src/coreot/evaluation/__init__.py",
    REPO / "src/coreot/preprocessing/__init__.py",
    REPO / "src/coreot/transport/__init__.py",
)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_role(path: Path) -> str:
    return path.relative_to(PACKAGE).parts[0]


def stage_fit_evidence() -> dict[str, dict[str, object]]:
    score_paths = sorted(FITS.rglob("cell_scores.parquet"))
    if len(score_paths) != 600:
        raise RuntimeError(f"Expected 600 compact score files; found {len(score_paths)}")
    inputs: dict[str, dict[str, object]] = {}
    for score_source in score_paths:
        source_root = score_source.parent
        entries = {path.name for path in source_root.iterdir() if path.is_file()}
        required = {"resolved_config.yaml", "fit_manifest.yaml", "cell_scores.parquet"}
        if entries != required:
            raise RuntimeError(f"Unexpected source fit files at {source_root}: {entries}")
        relative_root = source_root.relative_to(FITS)
        destination_root = PACKAGE / "fit_evidence" / relative_root
        config_source = source_root / "resolved_config.yaml"
        manifest_source = source_root / "fit_manifest.yaml"
        config_destination = destination_root / "resolved_config.yaml"
        score_destination = destination_root / "cell_scores.parquet"
        manifest_destination = destination_root / "fit_manifest.yaml"
        copy_file(config_source, config_destination)
        copy_file(score_source, score_destination)
        resolved = yaml.safe_load(config_source.read_text(encoding="utf-8"))
        for name, record in resolved["inputs"].items():
            relative = str(record["path"])
            candidate = {
                "repository_path": relative,
                "sha256": str(record["sha256"]),
                "bytes": int(record["bytes"]),
                "visibility": str(record["visibility"]),
                "roles": {str(name)},
            }
            if relative in inputs:
                existing = inputs[relative]
                for field in ("sha256", "bytes", "visibility"):
                    if existing[field] != candidate[field]:
                        raise RuntimeError(f"Inconsistent input declaration: {relative}")
                existing["roles"].add(str(name))
            else:
                inputs[relative] = candidate
        manifest = yaml.safe_load(manifest_source.read_text(encoding="utf-8"))
        score_record = manifest["artifacts"]["cell_scores"]
        manifest["artifacts"] = {
            "resolved_config": {
                "path": str(config_destination.relative_to(PACKAGE)),
                "sha256": sha256_file(config_destination),
                "bytes": config_destination.stat().st_size,
            },
            "cell_scores": {
                "path": str(score_destination.relative_to(PACKAGE)),
                "sha256": sha256_file(score_destination),
                "bytes": score_destination.stat().st_size,
                "rows": int(score_record["rows"]),
                "columns": list(score_record["columns"]),
                "cell_id_order_sha256": str(score_record["cell_id_order_sha256"]),
                "schema_sha256": str(score_record["schema_sha256"]),
            },
        }
        manifest["packaging"] = {
            "source_fit_manifest_path": str(manifest_source.relative_to(REPO)),
            "source_fit_manifest_sha256": sha256_file(manifest_source),
            "artifact_paths_rebased_to_package": True,
        }
        manifest_destination.parent.mkdir(parents=True, exist_ok=True)
        manifest_destination.write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
    if len(inputs) != 91:
        raise RuntimeError(f"Expected 91 unique external inputs; found {len(inputs)}")
    return inputs


def stage_external_dependencies(inputs: dict[str, dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for relative, record in sorted(inputs.items()):
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if (
            sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise RuntimeError(f"External dependency drift: {path}")
        rows.append(
            {
                "repository_path": relative,
                "accession": (
                    "GSE96583; Figshare file 34464122"
                    if relative == "data/raw/kang_2018.h5ad"
                    else ""
                ),
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "included_in_package": False,
                "role": (
                    f"{record['visibility']}:" + "+".join(sorted(record["roles"]))
                ),
                "prepared_by": (
                    "external analyzed object declared in docs/exp_pbmc.md"
                    if relative == "data/raw/kang_2018.h5ad"
                    else "upstream PBMC run; declared by per-fit resolved configuration"
                ),
            }
        )
    destination = PACKAGE / "manifests/external_dependencies.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)


def stage_code() -> None:
    records: list[dict[str, object]] = []
    for source in REPOSITORY_CODE_PATHS:
        relative = Path("code/repository") / source.relative_to(REPO)
        copy_file(source, PACKAGE / relative)
        records.append(
            {
                "code_name": source.stem,
                "role": "scientific_or_derivation_code",
                "repository_path": str(source.relative_to(REPO)),
                "package_path": str(relative),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    for source in sorted((UNIT / "audit/workers").glob("*.py")):
        relative = Path("code/verification") / source.name
        copy_file(source, PACKAGE / relative)
        records.append(
            {
                "code_name": source.stem,
                "role": "verification_code",
                "repository_path": str(source.relative_to(REPO)),
                "package_path": str(relative),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    records_frame = pd.DataFrame(records)
    if records_frame["package_path"].duplicated().any():
        raise RuntimeError("Code package paths duplicate")
    records_frame.to_csv(PACKAGE / "manifests/code_dependencies.csv", index=False)

    support_rows = []
    for source in RUNTIME_SUPPORT_PATHS:
        relative = Path("code/repository") / source.relative_to(REPO)
        copy_file(source, PACKAGE / relative)
        support_rows.append(
            {
                "repository_path": str(source.relative_to(REPO)),
                "package_path": str(relative),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "role": "package import support; no fit-level scientific logic",
            }
        )
    pd.DataFrame(support_rows).to_csv(
        PACKAGE / "manifests/runtime_support.csv", index=False
    )
    for source in (REPO / "pyproject.toml", REPO / "uv.lock"):
        copy_file(source, PACKAGE / "code/environment" / source.name)


def stage_audits_and_outputs() -> None:
    for source in sorted((UNIT / "audit").iterdir()):
        if source.is_file():
            copy_file(source, PACKAGE / "audits" / source.name)
    for source in sorted((UNIT / "comparison").glob("*")):
        if source.is_file():
            copy_file(source, PACKAGE / "audits/comparison" / source.name)
    for name in ("README.md", "compare_baselines_by_method.csv"):
        copy_file(
            RESOURCE_ROOT / name,
            PACKAGE / "audits/baseline_resource_audit" / name,
        )
    for family in (
        "rho_attribution_tau_surface_alpha0_range075_175",
        "rho_attribution_alpha_search",
    ):
        source_root = UNIT / "regenerated" / family
        for source in sorted(source_root.rglob("*")):
            if source.is_file():
                copy_file(
                    source,
                    PACKAGE
                    / "verified_results"
                    / family
                    / source.relative_to(source_root),
                )
    for source in sorted((UNIT / "regenerated/figures").glob("*.png")):
        copy_file(source, PACKAGE / "verified_results/figures" / source.name)


def write_package_manifests() -> None:
    dependency_rows = []
    for config_path in sorted(PACKAGE.glob("fit_evidence/**/resolved_config.yaml")):
        root = config_path.parent
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dependency_rows.append(
            {
                "fit_id": str(resolved["fit_id"]),
                "analysis_family": str(resolved["analysis_family"]),
                "endpoint": str(resolved["endpoint"]),
                "seed": int(resolved["seed"]),
                "variant": str(resolved["variant"]),
                "resolved_config": str(config_path.relative_to(PACKAGE)),
                "fit_manifest": str(
                    (root / "fit_manifest.yaml").relative_to(PACKAGE)
                ),
                "cell_scores": str(
                    (root / "cell_scores.parquet").relative_to(PACKAGE)
                ),
            }
        )
    dependencies = pd.DataFrame(dependency_rows).sort_values("fit_id")
    if len(dependencies) != 600 or dependencies["fit_id"].duplicated().any():
        raise RuntimeError("Packaged fit-dependency index is incomplete")
    dependencies.to_csv(PACKAGE / "manifests/fit_dependencies.csv", index=False)

    exclusions = [
        (
            "temporary sparse coupling and label-probability intermediates",
            "excluded",
            "Compact metric-sufficient cell evidence is retained per fit.",
        ),
        (
            "full cell_transport_scores.parquet",
            "projected",
            "Retained exactly cell_id, u, and forced_label; declared per manifest.",
        ),
        (
            "data/raw/kang_2018.h5ad",
            "external dependency",
            "Canonical HDF5 object was read in place and not copied or deleted.",
        ),
        (
            "baseline_resource_audit/resource_runs and raw logs",
            "excluded",
            "Locked 12-method README and summary CSV are retained.",
        ),
        (
            "historical runtime_seconds",
            "not compared",
            "Reconstruction runtime is new execution provenance.",
        ),
        (
            "rho_attribution_alpha_search",
            "retained as exploratory",
            "Exact reproduction does not change inferential status.",
        ),
        (
            "focused figure caption",
            "excluded",
            "Manuscript prose is outside this verification unit.",
        ),
    ]
    pd.DataFrame(exclusions, columns=["item", "action", "reason"]).to_csv(
        PACKAGE / "manifests/exclusions_and_transformations.csv", index=False
    )
    resource_csv = RESOURCE_ROOT / "compare_baselines_by_method.csv"
    reference = {
        "dataset_level_resource_audit": str(RESOURCE_ROOT.relative_to(REPO)),
        "source_summary_sha256": sha256_file(resource_csv),
        "source_summary_bytes": resource_csv.stat().st_size,
        "packaged_summary": "audits/baseline_resource_audit/compare_baselines_by_method.csv",
        "method_count": 12,
        "measurement_scope": "representative_run",
    }
    (PACKAGE / "manifests/resource_audit_reference.yaml").write_text(
        yaml.safe_dump(reference, sort_keys=False), encoding="utf-8"
    )


def write_inventory() -> None:
    manifests = PACKAGE / "manifests"
    included = manifests / "included_files.csv"
    checksums = manifests / "checksums.sha256"
    payloads = [
        path
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path not in (included, checksums)
    ]
    with included.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("package_path", "sha256", "bytes", "role")
        )
        writer.writeheader()
        for path in payloads:
            writer.writerow(
                {
                    "package_path": str(path.relative_to(PACKAGE)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "role": package_role(path),
                }
            )
    targets = [path for path in sorted(PACKAGE.rglob("*")) if path.is_file()]
    checksums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(PACKAGE)}\n"
            for path in targets
            if path != checksums
        ),
        encoding="utf-8",
    )


def main() -> None:
    if PACKAGE.exists() and any(PACKAGE.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty package: {PACKAGE}")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    inputs = stage_fit_evidence()
    stage_external_dependencies(inputs)
    stage_code()
    stage_audits_and_outputs()
    write_package_manifests()
    write_inventory()
    files = sum(path.is_file() for path in PACKAGE.rglob("*"))
    size = sum(path.stat().st_size for path in PACKAGE.rglob("*") if path.is_file())
    print(f"package_files={files} package_bytes={size}")


if __name__ == "__main__":
    main()
