from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coreot.artifacts.hashes import sha256_file  # noqa: E402


UNIT = Path(__file__).resolve().parents[2]
PACKAGE = UNIT / "package_staging/pbmc_compatibility_sensitivity"
FITS = UNIT / "regenerated/fits"
OLD_EXTERNAL = (
    REPO
    / "results/submission_verification/PBMC/2026-08-03"
    / "pbmc_compatibility_sensitivity/package_staging"
    / "pbmc_compatibility_sensitivity/manifests/external_dependencies.csv"
)
RESOURCE_ROOT = (
    REPO
    / "results/submission_verification/PBMC/2026-08-03/baseline_resource_audit"
)
CODE_PATHS = {
    "component_ablation_surfaces": REPO / "experiments/component_ablation_surfaces.py",
    "artifact_hashes": REPO / "src/coreot/artifacts/hashes.py",
    "artifact_manifests": REPO / "src/coreot/artifacts/manifests.py",
    "config_load": REPO / "src/coreot/config/load.py",
    "data_hidden": REPO / "src/coreot/data/hidden.py",
    "data_schemas": REPO / "src/coreot/data/schemas.py",
    "data_validation": REPO / "src/coreot/data/validation.py",
    "evaluation_metrics": REPO / "src/coreot/evaluation/metrics.py",
    "provider_reliability": REPO / "src/coreot/preprocessing/provider_reliability.py",
    "label_transfer": REPO / "src/coreot/transport/label_transfer.py",
    "transport_runner": REPO / "src/coreot/transport/runner.py",
    "sinkhorn_solver": REPO / "src/coreot/transport/sinkhorn.py",
    "reconstruction_worker": UNIT / "audit/workers/reconstruct_fits.py",
}
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


def role_for(path: Path) -> str:
    relative = path.relative_to(PACKAGE)
    return relative.parts[0]


def stage_fit_evidence() -> set[str]:
    score_paths = sorted(FITS.glob("*/seed*/tau*_alpha*/cell_scores.parquet"))
    if len(score_paths) != 375:
        raise RuntimeError(f"Expected 375 fit score files, found {len(score_paths)}")
    declared_inputs: set[str] = set()
    for score_source in score_paths:
        source_root = score_source.parent
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
        declared_inputs.update(
            str(record["path"]) for record in resolved["inputs"].values()
        )
        manifest = yaml.safe_load(manifest_source.read_text(encoding="utf-8"))
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
                "rows": int(manifest["artifacts"]["cell_scores"]["rows"]),
                "columns": list(manifest["artifacts"]["cell_scores"]["columns"]),
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
    return declared_inputs


def stage_external_dependencies(declared_inputs: set[str]) -> None:
    source = pd.read_csv(OLD_EXTERNAL, dtype={"included_in_package": str})
    if len(source) != 91 or source["repository_path"].duplicated().any():
        raise RuntimeError("Existing dependency inventory is not the expected 91 unique rows")
    if set(source["repository_path"]) != declared_inputs:
        missing = sorted(declared_inputs - set(source["repository_path"]))
        extra = sorted(set(source["repository_path"]) - declared_inputs)
        raise RuntimeError(f"Dependency inventory mismatch: missing={missing}, extra={extra}")
    for row in source.itertuples(index=False):
        path = REPO / str(row.repository_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise RuntimeError(f"External dependency drift: {path}")
    external = PACKAGE / "manifests/external_dependencies.csv"
    external.parent.mkdir(parents=True, exist_ok=True)
    source.sort_values("repository_path").to_csv(external, index=False)


def stage_code() -> None:
    records: list[dict[str, object]] = []
    for name, source in CODE_PATHS.items():
        if source.is_relative_to(REPO / "results/submission_verification"):
            package_relative = Path("code/verification/reconstruct_fits.py")
        else:
            package_relative = Path("code/repository") / source.relative_to(REPO)
        destination = PACKAGE / package_relative
        copy_file(source, destination)
        records.append(
            {
                "code_name": name,
                "repository_path": str(source.relative_to(REPO)),
                "package_path": str(package_relative),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    for source in sorted((UNIT / "audit/workers").glob("*.py")):
        if source.name == "reconstruct_fits.py":
            continue
        copy_file(source, PACKAGE / "code/verification" / source.name)
    pd.DataFrame(records).to_csv(PACKAGE / "manifests/code_dependencies.csv", index=False)
    support_records: list[dict[str, object]] = []
    for source in RUNTIME_SUPPORT_PATHS:
        package_relative = Path("code/repository") / source.relative_to(REPO)
        destination = PACKAGE / package_relative
        copy_file(source, destination)
        support_records.append(
            {
                "repository_path": str(source.relative_to(REPO)),
                "package_path": str(package_relative),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "role": "package import support; no fit-level scientific logic",
            }
        )
    pd.DataFrame(support_records).to_csv(
        PACKAGE / "manifests/runtime_support.csv", index=False
    )


def stage_audits_and_outputs() -> None:
    audit_names = (
        "scope.md",
        "lineage.csv",
        "code_config.md",
        "exclusions_transformations.md",
        "hypothesis_audit.md",
        "reconstruction_status.csv",
        "fit_dependencies.csv",
        "checkpoint_artifacts.csv",
        "component_manifest.yaml",
    )
    for name in audit_names:
        copy_file(UNIT / "audit" / name, PACKAGE / "audits" / name)
    for name in ("README.md", "compare_baselines_by_method.csv"):
        copy_file(
            RESOURCE_ROOT / name,
            PACKAGE / "audits/baseline_resource_audit" / name,
        )
    for source in sorted((UNIT / "comparison").glob("*")):
        copy_file(source, PACKAGE / "audits/comparison" / source.name)
    checkpoints = sorted(
        (UNIT / "regenerated/component_ablation").glob(
            "*/seed*/compatibility_only.csv"
        )
    )
    if len(checkpoints) != 15:
        raise RuntimeError(f"Expected 15 reconstructed checkpoints, found {len(checkpoints)}")
    for source in checkpoints:
        relative = source.relative_to(UNIT / "regenerated/component_ablation")
        copy_file(source, PACKAGE / "verified_results/checkpoints" / relative)
    for source in sorted((UNIT / "regenerated/component_ablation/tables").glob("*.csv")):
        copy_file(source, PACKAGE / "verified_results/tables" / source.name)
    copy_file(
        UNIT / "regenerated/figures/manuscript_fig_pbmc_component_minus_m.png",
        PACKAGE / "verified_results/figures/pbmc_compatibility_sensitivity.png",
    )


def write_exclusions() -> None:
    records = [
        {
            "item": "temporary sparse coupling and label-probability intermediates",
            "action": "excluded",
            "reason": "Metric-sufficient compact per-cell evidence retained; task-owned transients deleted after each fit.",
        },
        {
            "item": "full 15-column cell_transport_scores.parquet",
            "action": "projected",
            "reason": "Retained exactly cell_id, u, and forced_label; projection is declared in each manifest.",
        },
        {
            "item": "data/raw/kang_2018.h5ad",
            "action": "external dependency",
            "reason": "Canonical HDF5 object was read in place and was not copied or deleted.",
        },
        {
            "item": "baseline_resource_audit/resource_runs",
            "action": "excluded",
            "reason": "Large working tree contains symlinks and intermediates; locked 12-method summary evidence retained.",
        },
        {
            "item": "manuscript_fig_pbmc_component_minus_m.png",
            "action": "package-local semantic rename",
            "reason": "Packaged as pbmc_compatibility_sensitivity.png; canonical manuscript artifact was not overwritten.",
        },
        {
            "item": "historical cross-variant color normalization",
            "action": "replaced in isolated render",
            "reason": "Current code derives each compatibility-panel color range from compatibility rows only; dimensions, labels, displayed values, and layout are unchanged.",
        },
        {
            "item": "historical runtime_seconds",
            "action": "not compared",
            "reason": "Reconstruction runtime is new execution provenance, not a recoverable historical scientific result.",
        },
    ]
    path = PACKAGE / "manifests/exclusions_and_transformations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


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
                    "role": role_for(path),
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


def main() -> int:
    if PACKAGE.exists() and any(PACKAGE.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty package: {PACKAGE}")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    declared_inputs = stage_fit_evidence()
    stage_external_dependencies(declared_inputs)
    stage_code()
    stage_audits_and_outputs()
    write_exclusions()
    write_inventory()
    files = sum(path.is_file() for path in PACKAGE.rglob("*"))
    size = sum(path.stat().st_size for path in PACKAGE.rglob("*") if path.is_file())
    print(f"package_files={files} package_bytes={size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
