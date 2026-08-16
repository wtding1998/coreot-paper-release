from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from coreot.artifacts.hashes import sha256_file
from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.config.load import load_yaml


class RawImportConfigError(ValueError):
    """Raised when raw-import configuration is invalid."""


@dataclass(frozen=True)
class RawImportResult:
    run_root: Path
    raw_path: Path
    manifest_path: Path


def run_raw_import(config_path: str | Path) -> RawImportResult:
    config = load_yaml(config_path)
    source = _required_path(config, ("input", "path"))
    if not source.is_file():
        raise FileNotFoundError(f"Raw input file does not exist: {source}")
    if source.suffix != ".h5ad":
        raise RawImportConfigError(f"Raw input must be an .h5ad file: {source}")

    run_id = _required_str(config, ("run_id",))
    output_root = _required_path(config, ("outputs", "root"))
    run_root = output_root / run_id
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    destination = raw_dir / "input.h5ad"
    source_hash = sha256_file(source)
    adapter = config.get("adapter")
    if adapter is None:
        if str(config.get("input", {}).get("mode", "copy")) == "symlink":
            destination.symlink_to(source.resolve())
            adapter_name = "symlink"
        else:
            shutil.copy2(source, destination)
            adapter_name = "copy"
    elif adapter == "hiha_dc":
        _write_hiha_dc_standardized_input(source, destination, config.get("hiha_dc", {}))
        adapter_name = "hiha_dc"
    elif adapter == "pbmc_ifnb":
        _write_pbmc_ifnb_standardized_input(source, destination, config.get("pbmc_ifnb", {}))
        adapter_name = "pbmc_ifnb"
    else:
        raise RawImportConfigError(f"Unsupported raw-import adapter: {adapter!r}")

    destination_hash = sha256_file(destination)
    if adapter is None and destination_hash != source_hash:
        raise OSError(f"Copied raw input hash mismatch: {source} -> {destination}")

    manifest_path = raw_dir / "raw_manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="raw-import",
            artifacts={"input_h5ad": str(destination)},
            metadata={
                "dataset_name": _required_str(config, ("dataset_name",)),
                "run_id": run_id,
                "source_path": str(source),
                "source_sha256": source_hash,
                "sha256": destination_hash,
                "size_bytes": destination.stat().st_size,
                "adapter": adapter_name,
            },
        ),
    )

    return RawImportResult(
        run_root=run_root,
        raw_path=destination,
        manifest_path=manifest_path,
    )


def _write_hiha_dc_standardized_input(
    source: Path, destination: Path, adapter_config: object
) -> None:
    if adapter_config is not None and not isinstance(adapter_config, dict):
        raise RawImportConfigError("hiha_dc adapter config must be a mapping")
    config = adapter_config if isinstance(adapter_config, dict) else {}

    adata = ad.read_h5ad(source)
    obs = adata.obs.copy()
    _require_obs_columns(
        obs,
        (
            "subject.subjectGuid",
            "sample.sampleKitGuid",
            "AIFI_L2",
            "AIFI_L3",
        ),
    )
    batch_column = str(config.get("batch_column", "cohort.cohortGuid"))
    if batch_column not in obs.columns:
        batch_column = "batch_id"
    _require_obs_columns(obs, (batch_column,))

    cell_ids = _cell_ids(adata, obs, str(config.get("cell_id_source", "obs_names")))
    standardized_obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": obs["AIFI_L3"].astype(str).to_numpy(),
            "broad_label": obs["AIFI_L2"].astype(str).to_numpy(),
            "sample_id": obs["sample.sampleKitGuid"].astype(str).to_numpy(),
            "donor_id": obs["subject.subjectGuid"].astype(str).to_numpy(),
            "batch_id": obs[batch_column].astype(str).to_numpy(),
        },
        index=pd.Index(cell_ids, name=adata.obs_names.name),
    )
    matchability_column = config.get("matchability_score_column")
    if matchability_column is not None:
        matchability_column = str(matchability_column)
        _require_obs_columns(obs, (matchability_column,))
        rho = pd.to_numeric(obs[matchability_column], errors="coerce")
        if rho.isna().any():
            raise RawImportConfigError(
                f"Configured matchability_score_column contains nonnumeric value(s): {matchability_column}"
            )
        clip_min = _float_config(config, "matchability_clip_min", 0.05)
        clip_max = _float_config(config, "matchability_clip_max", 0.95)
        if not 0.0 <= clip_min <= clip_max <= 1.0:
            raise RawImportConfigError("matchability clip bounds must satisfy 0 <= min <= max <= 1")
        standardized_obs["rho"] = rho.clip(clip_min, clip_max).to_numpy()
        standardized_obs["prior_risk"] = 1.0 - standardized_obs["rho"].astype(float)
        standardized_obs["prior_source"] = str(
            config.get("matchability_prior_source", matchability_column)
        )
    if standardized_obs["cell_id"].duplicated().any():
        raise RawImportConfigError("HIHA adapter produced duplicate cell_id values")

    standardized = adata.copy()
    standardized.obs = standardized_obs
    standardized.write_h5ad(destination)


def _write_pbmc_ifnb_standardized_input(
    source: Path, destination: Path, adapter_config: object
) -> None:
    if adapter_config is not None and not isinstance(adapter_config, dict):
        raise RawImportConfigError("pbmc_ifnb adapter config must be a mapping")
    config = adapter_config if isinstance(adapter_config, dict) else {}

    adata = ad.read_h5ad(source)
    obs = adata.obs.copy()
    celltype_column = str(config.get("cell_type_column", "cell_type"))
    condition_column = str(config.get("condition_column", "label"))
    donor_column = str(config.get("donor_column", "replicate"))
    sample_column = str(config.get("sample_column", donor_column))
    batch_column = str(config.get("batch_column", donor_column))
    _require_obs_columns(
        obs,
        (
            celltype_column,
            condition_column,
            donor_column,
            sample_column,
            batch_column,
        ),
    )

    condition = _normalize_pbmc_condition(obs[condition_column], config.get("condition_values", {}))
    cell_ids = _cell_ids(adata, obs, str(config.get("cell_id_source", "obs_names")))
    broad_label = obs[celltype_column].astype(str).map(_pbmc_broad_lineage).astype(str)
    standardized_obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": obs[celltype_column].astype(str).to_numpy(),
            "broad_label": broad_label.to_numpy(),
            "sample_id": obs[sample_column].astype(str).to_numpy(),
            "donor_id": obs[donor_column].astype(str).to_numpy(),
            "batch_id": obs[batch_column].astype(str).to_numpy(),
            "state_condition": condition.to_numpy(),
        },
        index=pd.Index(cell_ids, name=adata.obs_names.name),
    )
    if "constant_matchability" in config:
        constant_rho = _float_config(config, "constant_matchability", 0.8)
        if not 0.0 <= constant_rho <= 1.0:
            raise RawImportConfigError("pbmc_ifnb.constant_matchability must lie in [0, 1]")
        standardized_obs["rho"] = constant_rho
        standardized_obs["prior_risk"] = 1.0 - constant_rho
        standardized_obs["prior_source"] = str(
            config.get(
                "matchability_prior_source",
                f"constant_condition_blind_rho_{constant_rho:g}",
            )
        )
    if standardized_obs["cell_id"].duplicated().any():
        raise RawImportConfigError("PBMC IFN-beta adapter produced duplicate cell_id values")

    standardized = adata.copy()
    standardized.obs = standardized_obs
    standardized.write_h5ad(destination)


def _normalize_pbmc_condition(
    values: pd.Series, condition_values: object
) -> pd.Series:
    aliases = {
        "ctrl": {"ctrl", "control", "unstim", "unstimulated", "untreated"},
        "stim": {"stim", "stimulated", "ifnb", "ifn-beta", "ifn_beta", "ifn beta"},
    }
    if isinstance(condition_values, dict):
        for canonical in ("ctrl", "stim"):
            configured = condition_values.get(canonical)
            if configured is None:
                continue
            if not isinstance(configured, list | tuple):
                raise RawImportConfigError(
                    f"pbmc_ifnb.condition_values.{canonical} must be a list"
                )
            aliases[canonical] = {str(value).strip().lower() for value in configured}

    normalized = []
    unknown = set()
    for value in values.astype(str):
        key = value.strip().lower()
        if key in aliases["ctrl"]:
            normalized.append("ctrl")
        elif key in aliases["stim"]:
            normalized.append("stim")
        else:
            unknown.add(value)
            normalized.append("")
    if unknown:
        examples = ", ".join(sorted(unknown)[:5])
        raise RawImportConfigError(f"Unrecognized PBMC condition value(s): {examples}")
    return pd.Series(normalized, index=values.index)


def _pbmc_broad_lineage(cell_type: str) -> str:
    label = cell_type.lower()
    if "doublet" in label:
        return "other/rare"
    if "mono" in label or "monocyte" in label:
        return "monocyte lineage"
    if "cd4" in label or "cd8" in label or "t cell" in label or label.startswith("t "):
        return "T lineage"
    if "b cell" in label or label.startswith("b ") or "activated b" in label:
        return "B lineage"
    if "nk" in label:
        return "NK lineage"
    if "dendritic" in label or label == "dc" or "pdc" in label:
        return "DC lineage"
    if "mega" in label or "platelet" in label:
        return "platelet/megakaryocyte lineage"
    return "other/rare"


def _cell_ids(adata: ad.AnnData, obs: pd.DataFrame, source: str) -> pd.Index:
    if source == "obs_names":
        return pd.Index(adata.obs_names.astype(str))
    if source not in obs.columns:
        raise RawImportConfigError(f"Configured cell_id_source is absent from .obs: {source}")
    return pd.Index(obs[source].astype(str))


def _require_obs_columns(obs: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise RawImportConfigError(
            f"HIHA adapter input .obs is missing required column(s): {', '.join(missing)}"
        )


def _float_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float):
        raise RawImportConfigError(f"Expected numeric hiha_dc config value: {key}")
    return float(value)


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise RawImportConfigError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise RawImportConfigError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
