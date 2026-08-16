from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.benchmarks.conditions import (
    DEFAULT_CONDITIONS,
    FULL_REFERENCE_CONTROL,
    INCOMPLETE_REFERENCE,
)
from coreot.benchmarks.splits import make_query_mask
from coreot.benchmarks.validation import validate_missing_state_benchmark_artifacts
from coreot.config.load import load_yaml
from coreot.data.schemas import (
    BENCHMARK_OBS_REQUIRED_COLUMNS,
    MODEL_VISIBLE_CELLS_COLUMNS,
    QUERY_TRUTH_COLUMNS,
    TARGET_LABELS_COLUMNS,
)
from coreot.data.validation import validate_required_columns


class BenchmarkBuildError(ValueError):
    """Raised when missing-state benchmark construction is invalid."""


@dataclass(frozen=True)
class BenchmarkBuildResult:
    benchmark_root: Path
    conditions: tuple[str, ...]


def run_missing_state_benchmark_build(config_path: str | Path) -> BenchmarkBuildResult:
    config = load_yaml(config_path)
    run_id = _required_str(config, ("run_id",))
    dataset_name = _required_str(config, ("dataset_name",))
    removed_state = _required_str(config, ("removed_state",))
    removed_state_condition = config.get("removed_state_condition")
    if removed_state_condition is not None:
        removed_state_condition = str(removed_state_condition)
    output_root = _required_path(config, ("outputs", "root"))
    run_root = output_root / run_id
    raw_input = run_root / "raw" / "input.h5ad"
    if not raw_input.is_file():
        raise FileNotFoundError(f"Raw-import artifact does not exist: {raw_input}")

    split_config = _required_mapping(config, ("split",))
    split_strategy = _required_str(split_config, ("strategy",))
    query_fraction = _required_float(split_config, ("query_fraction",))
    seed = _required_int(split_config, ("seed",))
    min_query_positives = int(split_config.get("min_query_positives", 1))
    max_resamples = int(split_config.get("max_resamples", 1))
    conditions = tuple(config.get("conditions", DEFAULT_CONDITIONS))
    _validate_conditions(conditions)
    subsample_config = config.get("subsample")
    if subsample_config is not None and not isinstance(subsample_config, dict):
        raise BenchmarkBuildError("subsample config must be a mapping")
    storage_config = config.get("storage", {})
    if not isinstance(storage_config, dict):
        raise BenchmarkBuildError("storage config must be a mapping")
    shared_counts = bool(storage_config.get("shared_counts", False))
    shared_counts_source = storage_config.get("shared_counts_source")
    truth_config = config.get("evaluation_truth", {})
    if not isinstance(truth_config, dict):
        raise BenchmarkBuildError("evaluation_truth config must be a mapping")

    adata = ad.read_h5ad(raw_input)
    obs = adata.obs.copy()
    validate_required_columns(obs, BENCHMARK_OBS_REQUIRED_COLUMNS, "raw input .obs")
    _validate_obs_values(obs, removed_state, removed_state_condition)

    query_mask, split_seed = _make_valid_split(
        obs=obs,
        split_strategy=split_strategy,
        query_fraction=query_fraction,
        seed=seed,
        max_resamples=max_resamples,
        removed_state=removed_state,
        removed_state_condition=removed_state_condition,
        min_query_positives=min_query_positives,
        eligibility=split_config,
    )
    reference_mask = ~query_mask
    _validate_split(
        obs,
        query_mask,
        reference_mask,
        removed_state,
        removed_state_condition,
        min_query_positives,
        split_config,
    )
    eligible_query_mask = query_mask.copy()
    eligible_reference_mask = reference_mask.copy()
    query_mask, reference_mask = _apply_subsampling(
        obs=obs,
        query_mask=query_mask,
        reference_mask=reference_mask,
        removed_state=removed_state,
        removed_state_condition=removed_state_condition,
        subsample_config=subsample_config,
        seed=split_seed,
    )
    _validate_split(
        obs,
        query_mask,
        reference_mask,
        removed_state,
        removed_state_condition,
        min_query_positives,
        split_config,
    )

    benchmark_root = run_root / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    _write_split_manifest(benchmark_root, obs, query_mask, reference_mask, split_seed)
    _write_condition_manifest(benchmark_root, conditions, removed_state, removed_state_condition)

    shared_counts_path = None
    if shared_counts:
        shared_root = benchmark_root / "shared" / "model_visible"
        shared_root.mkdir(parents=True, exist_ok=True)
        shared_counts_path = shared_root / "counts.h5ad"
        shared_mask = eligible_query_mask | eligible_reference_mask
        if shared_counts_source is not None:
            source_path = Path(str(shared_counts_source))
            if not source_path.is_file():
                raise FileNotFoundError(f"shared model-visible counts do not exist: {source_path}")
            if not shared_counts_path.is_file():
                if shared_counts_path.is_symlink():
                    shared_counts_path.unlink()
                shared_counts_path.symlink_to(source_path.resolve())
        else:
            shared_adata = adata[shared_mask.to_numpy()].copy()
            shared_adata.obs = pd.DataFrame(
                {"cell_id": obs.loc[shared_mask, "cell_id"].astype(str).to_numpy()},
                index=shared_adata.obs_names,
            )
            shared_adata.write_h5ad(shared_counts_path)
        pca_fit_mask = eligible_query_mask | (
            eligible_reference_mask
            & ~_removed_state_mask(obs, removed_state, removed_state_condition)
        )
        pd.DataFrame(
            {"cell_id": obs.loc[pca_fit_mask, "cell_id"].astype(str).to_numpy()}
        ).to_csv(shared_root / "pca_fit_cells.csv", index=False)

    for condition in conditions:
        _write_condition(
            adata=adata,
            obs=obs,
            query_mask=query_mask,
            reference_mask=reference_mask,
            condition=condition,
            benchmark_root=benchmark_root,
            dataset_name=dataset_name,
            removed_state=removed_state,
            removed_state_condition=removed_state_condition,
            seed=split_seed,
            shared_counts_path=shared_counts_path,
            include_query_broad_truth=bool(truth_config.get("include_broad_label", False)),
        )

    validate_missing_state_benchmark_artifacts(
        benchmark_root,
        removed_state,
        conditions,
        removed_state_condition=removed_state_condition,
    )
    return BenchmarkBuildResult(benchmark_root=benchmark_root, conditions=conditions)


def _write_condition(
    *,
    adata: ad.AnnData,
    obs: pd.DataFrame,
    query_mask: pd.Series,
    reference_mask: pd.Series,
    condition: str,
    benchmark_root: Path,
    dataset_name: str,
    removed_state: str,
    removed_state_condition: str | None,
    seed: int,
    shared_counts_path: Path | None,
    include_query_broad_truth: bool,
) -> None:
    condition_root = benchmark_root / condition
    model_visible = condition_root / "model_visible"
    evaluation_truth = condition_root / "evaluation_truth"
    model_visible.mkdir(parents=True, exist_ok=True)
    evaluation_truth.mkdir(parents=True, exist_ok=True)

    if condition == INCOMPLETE_REFERENCE:
        condition_reference_mask = reference_mask & ~_removed_state_mask(
            obs, removed_state, removed_state_condition
        )
    elif condition == FULL_REFERENCE_CONTROL:
        condition_reference_mask = reference_mask
    else:
        raise BenchmarkBuildError(f"Unsupported condition: {condition}")

    condition_mask = query_mask | condition_reference_mask
    visible_obs = obs.loc[condition_mask].copy()
    visible_obs["domain"] = "reference"
    visible_obs.loc[query_mask.loc[condition_mask], "domain"] = "query"
    cells = pd.DataFrame(
        {
            "cell_id": visible_obs["cell_id"].astype(str),
            "domain": visible_obs["domain"],
            "sample_id": visible_obs["sample_id"].astype(str),
            "donor_id": visible_obs["donor_id"].astype(str),
            "batch_id": visible_obs["batch_id"].astype(str),
            "condition_id": condition,
            "split_seed": seed,
        }
    )
    cells.to_csv(model_visible / "cells.csv", index=False, columns=MODEL_VISIBLE_CELLS_COLUMNS)

    counts_output = model_visible / "counts.h5ad"
    if shared_counts_path is not None:
        if not counts_output.is_file():
            if counts_output.is_symlink():
                counts_output.unlink()
            counts_output.symlink_to(shared_counts_path.resolve())
    else:
        condition_adata = adata[condition_mask.to_numpy()].copy()
        condition_adata.obs = cells.set_index("cell_id", drop=False)
        condition_adata.write_h5ad(counts_output)

    target_obs = obs.loc[condition_reference_mask].copy()
    target_labels = pd.DataFrame(
        {
            "cell_id": target_obs["cell_id"].astype(str),
            "target_label": target_obs["cell_type"].astype(str),
            "broad_label": _broad_labels(target_obs),
        }
    )
    target_labels.to_csv(
        model_visible / "target_labels.csv", index=False, columns=TARGET_LABELS_COLUMNS
    )
    _write_genes(model_visible / "genes.csv", adata)
    _write_model_visible_priors(model_visible, visible_obs)
    _write_dataset_config(
        model_visible / "dataset_config.yaml",
        dataset_name=dataset_name,
        condition=condition,
        removed_state=removed_state,
        removed_state_condition=removed_state_condition,
        seed=seed,
        n_query=int(query_mask.sum()),
        n_reference=int(condition_reference_mask.sum()),
    )

    query_obs = obs.loc[query_mask].copy()
    if condition == INCOMPLETE_REFERENCE:
        is_absent_state = _removed_state_mask(query_obs, removed_state, removed_state_condition)
    elif condition == FULL_REFERENCE_CONTROL:
        is_absent_state = pd.Series(False, index=query_obs.index)
    else:
        raise BenchmarkBuildError(f"Unsupported condition: {condition}")
    query_truth = pd.DataFrame(
        {
            "cell_id": query_obs["cell_id"].astype(str),
            "true_label": query_obs["cell_type"].astype(str),
            "removed_state": removed_state,
            "is_absent_state": is_absent_state,
            "is_shared_state": ~is_absent_state,
        }
    )
    query_truth_columns = list(QUERY_TRUTH_COLUMNS)
    if include_query_broad_truth:
        query_truth["true_broad_label"] = _broad_labels(query_obs).to_numpy()
        query_truth_columns.append("true_broad_label")
    query_truth.to_csv(evaluation_truth / "query_truth.csv", index=False, columns=query_truth_columns)
    _write_state_presence(
        evaluation_truth / "state_presence.csv", obs, query_mask, condition_reference_mask
    )
    write_manifest(
        evaluation_truth / "benchmark_truth_manifest.yaml",
        Manifest(
            stage="benchmark-build",
            artifacts={
                "query_truth": str(evaluation_truth / "query_truth.csv"),
                "state_presence": str(evaluation_truth / "state_presence.csv"),
            },
            metadata={
                "condition": condition,
                "removed_state": removed_state,
                "evaluation_truth_only": True,
            },
        ),
    )


def _write_genes(path: Path, adata: ad.AnnData) -> None:
    genes = pd.DataFrame(
        {
            "gene_id": adata.var_names.astype(str),
            "gene_name": adata.var_names.astype(str),
        }
    )
    genes.to_csv(path, index=False)


def _write_model_visible_priors(model_visible: Path, visible_obs: pd.DataFrame) -> None:
    broad_anchor_priors = pd.DataFrame(
        {
            "cell_id": visible_obs["cell_id"].astype(str),
            "broad_anchor_class": _broad_labels(visible_obs).astype(str).to_numpy(),
            "anchor_confidence": 1.0,
        }
    )
    broad_anchor_priors.to_csv(model_visible / "broad_anchor_priors.csv", index=False)

    if {"rho", "prior_risk", "prior_source"} <= set(visible_obs.columns):
        rho = pd.to_numeric(visible_obs["rho"], errors="coerce")
        prior_risk = pd.to_numeric(visible_obs["prior_risk"], errors="coerce")
        if rho.isna().any() or prior_risk.isna().any():
            raise BenchmarkBuildError("model-visible rho and prior_risk must be numeric")
        if not rho.between(0.0, 1.0).all() or not prior_risk.between(0.0, 1.0).all():
            raise BenchmarkBuildError("model-visible rho and prior_risk must lie in [0, 1]")
        initial_priors = pd.DataFrame(
            {
                "cell_id": visible_obs["cell_id"].astype(str),
                "rho": rho.astype(float),
                "prior_risk": prior_risk.astype(float),
                "prior_source": visible_obs["prior_source"].astype(str),
            }
        )
    else:
        initial_priors = pd.DataFrame(
            {
                "cell_id": visible_obs["cell_id"].astype(str),
                "rho": 1.0,
                "prior_risk": 0.0,
                "prior_source": "neutral_benchmark_default",
            }
        )
    initial_priors.to_csv(model_visible / "initial_priors.csv", index=False)


def _write_dataset_config(
    path: Path,
    *,
    dataset_name: str,
    condition: str,
    removed_state: str,
    removed_state_condition: str | None,
    seed: int,
    n_query: int,
    n_reference: int,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "condition_id": condition,
        "removed_state": removed_state,
        "removed_state_condition": removed_state_condition,
        "split_seed": seed,
        "n_query": n_query,
        "n_reference": n_reference,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _write_state_presence(
    path: Path, obs: pd.DataFrame, query_mask: pd.Series, reference_mask: pd.Series
) -> None:
    states = sorted(obs["cell_type"].astype(str).unique())
    rows = []
    for state in states:
        state_mask = obs["cell_type"].astype(str) == state
        rows.append(
            {
                "cell_type": state,
                "present_in_query": bool((query_mask & state_mask).any()),
                "present_in_reference": bool((reference_mask & state_mask).any()),
            }
        )
    if "state_condition" in obs.columns:
        for (state, state_condition), _ in obs.groupby(
            ["cell_type", "state_condition"],
            sort=True,
            observed=False,
        ):
            state_mask = (obs["cell_type"].astype(str) == str(state)) & (
                obs["state_condition"].astype(str) == str(state_condition)
            )
            rows.append(
                {
                    "cell_type": str(state),
                    "state_condition": str(state_condition),
                    "benchmark_state": f"{state}::{state_condition}",
                    "present_in_query": bool((query_mask & state_mask).any()),
                    "present_in_reference": bool((reference_mask & state_mask).any()),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_split_manifest(
    benchmark_root: Path,
    obs: pd.DataFrame,
    query_mask: pd.Series,
    reference_mask: pd.Series,
    seed: int,
) -> None:
    split_manifest = pd.DataFrame(
        {
            "cell_id": obs["cell_id"].astype(str),
            "split_domain": "reference",
            "split_seed": seed,
        }
    )
    split_manifest.loc[query_mask.to_numpy(), "split_domain"] = "query"
    split_manifest.loc[reference_mask.to_numpy(), "split_domain"] = "reference"
    split_manifest.to_csv(benchmark_root / "split_manifest.csv", index=False)


def _write_condition_manifest(
    benchmark_root: Path,
    conditions: tuple[str, ...],
    removed_state: str,
    removed_state_condition: str | None,
) -> None:
    rows = [
        {
            "condition_id": condition,
            "removed_state": removed_state,
            "removed_state_condition": removed_state_condition,
            "removes_state_from_reference": condition == INCOMPLETE_REFERENCE,
        }
        for condition in conditions
    ]
    pd.DataFrame(rows).to_csv(benchmark_root / "condition_manifest.csv", index=False)


def _validate_conditions(conditions: tuple[str, ...]) -> None:
    if conditions != DEFAULT_CONDITIONS:
        raise BenchmarkBuildError(
            "benchmark-build currently requires conditions "
            f"{DEFAULT_CONDITIONS}; got {conditions}"
        )


def _validate_obs_values(
    obs: pd.DataFrame, removed_state: str, removed_state_condition: str | None
) -> None:
    if obs["cell_id"].astype(str).duplicated().any():
        raise BenchmarkBuildError("raw input .obs cell_id values must be unique")
    if removed_state not in set(obs["cell_type"].astype(str)):
        raise BenchmarkBuildError(f"removed_state is absent from raw input .obs: {removed_state}")
    if removed_state_condition is not None:
        if "state_condition" not in obs.columns:
            raise BenchmarkBuildError(
                "removed_state_condition requires raw input .obs state_condition"
            )
        if removed_state_condition not in set(obs["state_condition"].astype(str)):
            raise BenchmarkBuildError(
                "removed_state_condition is absent from raw input .obs: "
                f"{removed_state_condition}"
            )
        if not _removed_state_mask(obs, removed_state, removed_state_condition).any():
            raise BenchmarkBuildError(
                "removed_state and removed_state_condition identify no raw input cells"
            )


def _make_valid_split(
    *,
    obs: pd.DataFrame,
    split_strategy: str,
    query_fraction: float,
    seed: int,
    max_resamples: int,
    removed_state: str,
    removed_state_condition: str | None,
    min_query_positives: int,
    eligibility: dict[str, Any],
) -> tuple[pd.Series, int]:
    attempts = max(1, max_resamples)
    last_error: BenchmarkBuildError | None = None
    for offset in range(attempts):
        split_seed = seed + offset
        query_mask = make_query_mask(obs, split_strategy, query_fraction, split_seed)
        reference_mask = ~query_mask
        try:
            _validate_split(
                obs,
                query_mask,
                reference_mask,
                removed_state,
                removed_state_condition,
                min_query_positives,
                eligibility,
            )
        except BenchmarkBuildError as exc:
            last_error = exc
            continue
        return query_mask, split_seed

    if last_error is not None:
        raise BenchmarkBuildError(
            f"Could not produce a valid split after {attempts} attempt(s): {last_error}"
        ) from last_error
    raise BenchmarkBuildError("Could not produce a valid benchmark split")


def _apply_subsampling(
    *,
    obs: pd.DataFrame,
    query_mask: pd.Series,
    reference_mask: pd.Series,
    removed_state: str,
    removed_state_condition: str | None,
    subsample_config: dict[str, Any] | None,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    if not subsample_config:
        return query_mask, reference_mask

    query_limit = subsample_config.get("max_query_per_cell_type")
    reference_limit = subsample_config.get("max_reference_per_cell_type")
    preserve_donor_balance = bool(subsample_config.get("preserve_donor_balance", False))
    if query_limit is not None:
        query_mask = _subsample_mask_by_cell_type(
            obs, query_mask, int(query_limit), seed, preserve_donor_balance
        )
    if reference_limit is not None:
        reference_mask = _subsample_mask_by_cell_type(
            obs, reference_mask, int(reference_limit), seed + 100_000, preserve_donor_balance
        )
    if not _removed_state_mask(obs.loc[query_mask], removed_state, removed_state_condition).any():
        raise BenchmarkBuildError(
            f"subsampling removed all held-out query cells for removed_state {removed_state!r}"
        )
    return query_mask, reference_mask


def _subsample_mask_by_cell_type(
    obs: pd.DataFrame,
    mask: pd.Series,
    limit: int,
    seed: int,
    preserve_donor_balance: bool = False,
) -> pd.Series:
    if limit <= 0:
        raise BenchmarkBuildError("subsample limits must be positive")
    selected: list[object] = []
    for label, group in obs.loc[mask].groupby("cell_type", sort=True, observed=False):
        if len(group) <= limit:
            selected.extend(group.index.tolist())
            continue
        random_state = seed + _stable_label_offset(str(label))
        if not preserve_donor_balance:
            sampled = group.sample(n=limit, random_state=random_state)
            selected.extend(sampled.index.tolist())
            continue
        donor_groups = [donor_group for _, donor_group in group.groupby("donor_id", sort=True)]
        rng = np.random.default_rng(random_state)
        donor_rows = [list(rng.permutation(donor_group.index.to_numpy())) for donor_group in donor_groups]
        positions = [0] * len(donor_rows)
        while sum(positions) < limit:
            progressed = False
            for donor_index, rows in enumerate(donor_rows):
                if positions[donor_index] < len(rows) and sum(positions) < limit:
                    selected.append(rows[positions[donor_index]])
                    positions[donor_index] += 1
                    progressed = True
            if not progressed:
                break
    return pd.Series(obs.index.isin(selected), index=obs.index)


def _stable_label_offset(label: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(label))


def _broad_labels(obs: pd.DataFrame) -> pd.Series:
    if "broad_label" in obs.columns:
        return obs["broad_label"].astype(str)
    return obs["cell_type"].astype(str)


def _validate_split(
    obs: pd.DataFrame,
    query_mask: pd.Series,
    reference_mask: pd.Series,
    removed_state: str,
    removed_state_condition: str | None = None,
    min_query_positives: int = 1,
    eligibility: dict[str, Any] | None = None,
) -> None:
    eligibility = eligibility or {}
    if not query_mask.any():
        raise BenchmarkBuildError("benchmark split produced no query cells")
    if not reference_mask.any():
        raise BenchmarkBuildError("benchmark split produced no reference cells")
    query_types = set(obs.loc[query_mask, "cell_type"].astype(str))
    reference_types = set(obs.loc[reference_mask, "cell_type"].astype(str))
    query_positives = int(
        _removed_state_mask(obs.loc[query_mask], removed_state, removed_state_condition).sum()
    )
    if query_positives < min_query_positives:
        raise BenchmarkBuildError(
            "benchmark query split has too few removed_state positives "
            f"for {removed_state!r}: {query_positives} < {min_query_positives}"
        )
    reference_positives = int(
        _removed_state_mask(obs.loc[reference_mask], removed_state, removed_state_condition).sum()
    )
    if reference_positives < 1:
        raise BenchmarkBuildError(
            "benchmark reference split has no removed_state cells for the full-reference control"
        )
    query_positive_mask = query_mask & _removed_state_mask(
        obs, removed_state, removed_state_condition
    )
    reference_positive_mask = reference_mask & _removed_state_mask(
        obs, removed_state, removed_state_condition
    )
    _require_minimum(
        int(obs.loc[query_positive_mask, "donor_id"].astype(str).nunique()),
        int(eligibility.get("min_query_positive_donors", 1)),
        "query positive donors",
    )
    _require_minimum(
        reference_positives,
        int(eligibility.get("min_reference_positives", 1)),
        "reference positives",
    )
    _require_minimum(
        int(obs.loc[reference_positive_mask, "donor_id"].astype(str).nunique()),
        int(eligibility.get("min_reference_positive_donors", 1)),
        "reference positive donors",
    )
    broad = _broad_labels(obs)
    removed_broad_values = broad.loc[_removed_state_mask(obs, removed_state, removed_state_condition)].unique()
    if len(removed_broad_values) != 1:
        raise BenchmarkBuildError("removed state must map to exactly one broad compartment")
    removed_broad = str(removed_broad_values[0])
    same_broad_negatives = query_mask & broad.eq(removed_broad) & ~_removed_state_mask(
        obs, removed_state, removed_state_condition
    )
    _require_minimum(
        int(same_broad_negatives.sum()),
        int(eligibility.get("min_same_broad_negatives", 0)),
        "same-broad query negatives",
    )
    min_broad_cells = int(eligibility.get("min_reference_broad_cells", 0))
    min_broad_donors = int(eligibility.get("min_reference_broad_donors", 0))
    for broad_class in sorted(broad.loc[query_mask].unique()):
        reference_broad = reference_mask & broad.eq(str(broad_class))
        _require_minimum(int(reference_broad.sum()), min_broad_cells, f"reference cells for {broad_class}")
        _require_minimum(
            int(obs.loc[reference_broad, "donor_id"].astype(str).nunique()),
            min_broad_donors,
            f"reference donors for {broad_class}",
        )
    if removed_state_condition is not None:
        retained_same_cell_type = (
            (obs.loc[reference_mask, "cell_type"].astype(str) == removed_state)
            & (obs.loc[reference_mask, "state_condition"].astype(str) != removed_state_condition)
        )
        if not retained_same_cell_type.any():
            raise BenchmarkBuildError(
                "condition-specific benchmark requires retained reference cells with "
                f"cell_type={removed_state!r} and state_condition!={removed_state_condition!r}"
            )
    shared = query_types & reference_types
    if not shared:
        raise BenchmarkBuildError("benchmark split has no shared cell types across domains")


def _removed_state_mask(
    obs: pd.DataFrame, removed_state: str, removed_state_condition: str | None
) -> pd.Series:
    mask = obs["cell_type"].astype(str) == removed_state
    if removed_state_condition is not None:
        mask = mask & (obs["state_condition"].astype(str) == removed_state_condition)
    return mask


def _require_minimum(actual: int, required: int, label: str) -> None:
    if actual < required:
        raise BenchmarkBuildError(f"benchmark split has too few {label}: {actual} < {required}")


def _required_str(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise BenchmarkBuildError(f"Expected nonempty string config value: {dotted}")
    return value


def _required_int(config: dict[str, Any], path: tuple[str, ...]) -> int:
    value = _required_value(config, path)
    if not isinstance(value, int):
        dotted = ".".join(path)
        raise BenchmarkBuildError(f"Expected integer config value: {dotted}")
    return value


def _required_float(config: dict[str, Any], path: tuple[str, ...]) -> float:
    value = _required_value(config, path)
    if not isinstance(value, int | float):
        dotted = ".".join(path)
        raise BenchmarkBuildError(f"Expected numeric config value: {dotted}")
    return float(value)


def _required_path(config: dict[str, Any], path: tuple[str, ...]) -> Path:
    return Path(_required_str(config, path))


def _required_mapping(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value = _required_value(config, path)
    if not isinstance(value, dict):
        dotted = ".".join(path)
        raise BenchmarkBuildError(f"Expected mapping config value: {dotted}")
    return value


def _required_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise BenchmarkBuildError(f"Missing required config value: {dotted}")
        current = current[key]
    return current
