from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable, Sequence

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from coreot.benchmarks.missing_state import (  # noqa: E402
    BenchmarkBuildError,
    _apply_subsampling,
    _validate_split,
)
from coreot.benchmarks.splits import make_query_mask  # noqa: E402
from coreot.candidates.knn import build_candidate_edges  # noqa: E402
from coreot.candidates.scaling import scale_distances  # noqa: E402
from experiments.component_ablation_surfaces import (  # noqa: E402
    CONDITION,
    EndpointSpec,
)
from experiments.rho_attribution_search import (  # noqa: E402
    METRIC_NAMES,
    RETAINED_FIT_FILES,
    VARIANTS,
    _evaluate_variant,
    _mouse_input_paths,
    _run_method,
    mean_matched_tau,
)


DEFAULT_DESIGN_MANIFEST = Path(
    "results/HIHA_DC/sensitivity/"
    "rho_attribution_discovery_confirmation/design/cohort_manifest.yaml"
)
DEFAULT_HIHA_ROOT = Path(
    "results/HIHA_DC/sensitivity/rho_attribution_discovery_confirmation"
)
DEFAULT_MOUSE_ROOT = Path(
    "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
    "rho_attribution_tau_alpha_search"
)
HIHA_CANDIDATE_SET = "hiha_harmony30_k100"
HIHA_PROVIDER = "hiha_harmony30"
HIHA_MATCHABILITY_SOURCE = (
    "recomputed_AIFI_L2_score_celltypist_2024-04-19_provisional"
)
MIN_QUERY_POSITIVES = 30
MAX_SPLIT_RESAMPLES = 100
QUERY_CELL_TYPE_LIMIT = 500
REFERENCE_CELL_TYPE_LIMIT = 1000


@dataclass(frozen=True)
class EndpointDesign:
    endpoint: str
    slug: str
    selected_alpha: float
    tau_target: float


HIHA_ENDPOINTS = (
    EndpointDesign(
        endpoint="HLA-DRhi cDC2",
        slug="hladrhi_cdc2",
        selected_alpha=2.0,
        tau_target=2.0,
    ),
    EndpointDesign(
        endpoint="ISG+ cDC2",
        slug="isg_cdc2",
        selected_alpha=1.0,
        tau_target=3.0,
    ),
)
HIHA_ALPHA0_FINE_TAU_VALUES = tuple(
    2.0**exponent
    for exponent in (-3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5)
)
HIHA_ALPHA0_FOCUSED_TAU_VALUES = (0.25, 0.5, 0.75, 1.0, 1.25)
MOUSE_ENDPOINT = EndpointDesign(
    endpoint="Proliferating",
    slug="proliferating",
    selected_alpha=40.0,
    tau_target=8.0,
)


@dataclass(frozen=True, order=True)
class SurfaceConfiguration:
    alpha: float
    tau_min: float
    tau_max: float


def tau_pairs(values: Sequence[float]) -> tuple[tuple[float, float], ...]:
    ordered = tuple(sorted({float(value) for value in values}))
    if not ordered or any(value <= 0.0 for value in ordered):
        raise ValueError("tau values must be nonempty and positive")
    return tuple(
        (tau_min, tau_max)
        for tau_min in ordered
        for tau_max in ordered
        if tau_min <= tau_max
    )


def coarse_configurations(
    tau_values: Sequence[float], alpha_values: Sequence[float]
) -> tuple[SurfaceConfiguration, ...]:
    alphas = tuple(sorted({float(value) for value in alpha_values}))
    if not alphas or any(value < 0.0 for value in alphas):
        raise ValueError("alpha values must be nonempty and nonnegative")
    return tuple(
        SurfaceConfiguration(alpha, tau_min, tau_max)
        for alpha in alphas
        for tau_min, tau_max in tau_pairs(tau_values)
    )


def hiha_alpha0_fine_configurations() -> tuple[SurfaceConfiguration, ...]:
    return tuple(
        SurfaceConfiguration(alpha=0.0, tau_min=tau_min, tau_max=tau_max)
        for tau_min, tau_max in tau_pairs(HIHA_ALPHA0_FINE_TAU_VALUES)
        if tau_max / tau_min <= 2.0 + 1.0e-12
    )


def hiha_alpha0_focused_configurations() -> tuple[SurfaceConfiguration, ...]:
    return coarse_configurations(HIHA_ALPHA0_FOCUSED_TAU_VALUES, (0.0,))


def relative_percent(delta: float, comparator: float) -> float:
    if comparator <= 0.0:
        raise ValueError("relative metric difference requires a positive comparator")
    return 100.0 * delta / comparator


def _neighbor_refinement_values(
    values: Sequence[float], center: float, *, geometric: bool
) -> tuple[float, ...]:
    ordered = tuple(sorted({float(value) for value in values}))
    try:
        index = next(
            i for i, value in enumerate(ordered) if math.isclose(value, center)
        )
    except StopIteration as exc:
        raise ValueError(f"refinement center {center:g} is absent from the axis") from exc
    refined = {float(center)}
    if index > 0:
        left = ordered[index - 1]
        refined.add(math.sqrt(left * center) if geometric else 0.5 * (left + center))
    if index + 1 < len(ordered):
        right = ordered[index + 1]
        refined.add(math.sqrt(center * right) if geometric else 0.5 * (center + right))
    return tuple(sorted(refined))


def refinement_configurations(
    *,
    center: SurfaceConfiguration,
    tau_values: Sequence[float],
    alpha_values: Sequence[float],
    coarse: Iterable[SurfaceConfiguration],
) -> tuple[SurfaceConfiguration, ...]:
    if math.isclose(center.tau_min, center.tau_max):
        raise ValueError("refinement center must be off diagonal")
    tau_min_values = _neighbor_refinement_values(
        tau_values, center.tau_min, geometric=True
    )
    tau_max_values = _neighbor_refinement_values(
        tau_values, center.tau_max, geometric=True
    )
    local_alpha_values = _neighbor_refinement_values(
        alpha_values, center.alpha, geometric=False
    )
    coarse_set = set(coarse)
    configurations = {
        SurfaceConfiguration(alpha, tau_min, tau_max)
        for alpha in local_alpha_values
        for tau_min in tau_min_values
        for tau_max in tau_max_values
        if tau_min <= tau_max
    }
    return tuple(sorted(configurations - coarse_set))


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _design_hash(path: Path) -> str:
    return sha256_file(path)


def validate_design_manifest(path: Path, *, project_root: Path) -> dict[str, object]:
    design = _load_yaml_mapping(path)
    discovery = tuple(str(value) for value in design.get("discovery_donors", ()))
    confirmation = tuple(
        str(value) for value in design.get("confirmation_donors", ())
    )
    if len(discovery) != 54 or len(confirmation) != 54:
        raise ValueError("design manifest must contain 54 donors in each cohort")
    if len(set(discovery)) != 54 or len(set(confirmation)) != 54:
        raise ValueError("design manifest contains duplicate donor identifiers")
    if set(discovery) & set(confirmation):
        raise ValueError("discovery and confirmation donors must be disjoint")
    source = design.get("source")
    if not isinstance(source, dict):
        raise ValueError("design manifest lacks source metadata")
    source_path = project_root / str(source.get("path", ""))
    if not source_path.is_file():
        raise FileNotFoundError(f"HIHA source data are missing: {source_path}")
    data = ad.read_h5ad(source_path, backed="r")
    try:
        donor_column = str(source.get("donor_column", ""))
        observed = set(data.obs[donor_column].astype(str))
    finally:
        data.file.close()
    if set(discovery) | set(confirmation) != observed:
        raise ValueError("cohort donor lists do not exactly partition the source donors")
    return design


def _endpoint_design(endpoint: str) -> EndpointDesign:
    for design in HIHA_ENDPOINTS:
        if design.endpoint == endpoint:
            return design
    raise ValueError(f"Unknown HIHA endpoint: {endpoint}")


def _selected_endpoints(endpoints: Sequence[str] | None) -> tuple[EndpointDesign, ...]:
    if not endpoints:
        return HIHA_ENDPOINTS
    requested = set(endpoints)
    selected = tuple(
        endpoint for endpoint in HIHA_ENDPOINTS if endpoint.endpoint in requested
    )
    unknown = requested - {endpoint.endpoint for endpoint in selected}
    if unknown:
        raise ValueError(f"Unknown HIHA endpoint(s): {sorted(unknown)}")
    return selected


def _source_prefix(donor: str) -> str:
    if donor.startswith("BR1"):
        return "BR10"
    if donor.startswith("BR2"):
        return "BR20"
    if donor.startswith("UP"):
        return "UP"
    raise ValueError(f"Unrecognized HIHA donor prefix: {donor}")


def _standardized_hiha_obs(data: ad.AnnData) -> pd.DataFrame:
    required = (
        "subject.subjectGuid",
        "sample.sampleKitGuid",
        "cohort.cohortGuid",
        "AIFI_L2",
        "AIFI_L3",
        "AIFI_L2_score_recomputed",
    )
    missing = [column for column in required if column not in data.obs]
    if missing:
        raise ValueError(f"HIHA source data lack columns: {missing}")
    rho = pd.to_numeric(
        data.obs["AIFI_L2_score_recomputed"], errors="coerce"
    ).clip(0.05, 0.95)
    if rho.isna().any():
        raise ValueError("HIHA matchability source contains nonnumeric values")
    cell_ids = data.obs_names.astype(str)
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": data.obs["AIFI_L3"].astype(str).to_numpy(),
            "broad_label": data.obs["AIFI_L2"].astype(str).to_numpy(),
            "sample_id": data.obs["sample.sampleKitGuid"].astype(str).to_numpy(),
            "donor_id": data.obs["subject.subjectGuid"].astype(str).to_numpy(),
            "batch_id": data.obs["cohort.cohortGuid"].astype(str).to_numpy(),
            "rho": rho.to_numpy(dtype=float),
            "prior_risk": 1.0 - rho.to_numpy(dtype=float),
        },
        index=pd.Index(cell_ids, name=data.obs_names.name),
    )


def _valid_split(
    obs: pd.DataFrame, endpoint: str, seed: int
) -> tuple[pd.Series, pd.Series, int]:
    last_error: Exception | None = None
    for offset in range(MAX_SPLIT_RESAMPLES):
        actual_seed = seed + offset
        query = make_query_mask(obs, "donor_aware", 0.2, actual_seed)
        reference = ~query
        try:
            _validate_split(
                obs,
                query,
                reference,
                endpoint,
                min_query_positives=MIN_QUERY_POSITIVES,
            )
            query, reference = _apply_subsampling(
                obs=obs,
                query_mask=query,
                reference_mask=reference,
                removed_state=endpoint,
                removed_state_condition=None,
                subsample_config={
                    "max_query_per_cell_type": QUERY_CELL_TYPE_LIMIT,
                    "max_reference_per_cell_type": REFERENCE_CELL_TYPE_LIMIT,
                },
                seed=actual_seed,
            )
            _validate_split(
                obs,
                query,
                reference,
                endpoint,
                min_query_positives=MIN_QUERY_POSITIVES,
            )
            return query, reference, actual_seed
        except BenchmarkBuildError as exc:
            last_error = exc
    raise BenchmarkBuildError(
        f"Could not produce a valid cohort split for {endpoint}, seed {seed}: "
        f"{last_error}"
    )


def _base_run_id(endpoint: EndpointDesign, cohort: str, seed: int) -> str:
    return (
        f"hiha_dc_{endpoint.slug}_{cohort}_seed{seed}_"
        "rho_attribution_confirmation"
    )


def _base_run_root(
    analysis_root: Path, endpoint: EndpointDesign, cohort: str, seed: int
) -> Path:
    return analysis_root / "base_runs" / cohort / _base_run_id(endpoint, cohort, seed)


def _base_input_paths(run_root: Path) -> dict[str, Path]:
    profile = run_root / f"derived/{CONDITION}/prior_profiles/default"
    return {
        "candidates": run_root
        / f"candidates/{CONDITION}/{HIHA_CANDIDATE_SET}/candidate_edges.parquet",
        "source_priors": profile / "source_priors.csv",
        "target_priors": profile / "target_priors.csv",
        "truth": run_root
        / f"benchmark/{CONDITION}/evaluation_truth/query_truth.csv",
        "split": run_root / "benchmark/split_manifest.csv",
        "manifest": run_root / "base_manifest.yaml",
    }


def _build_base_run(
    *,
    obs: pd.DataFrame,
    full_embedding: np.ndarray,
    source_sha256: str,
    cohort_manifest_sha256: str,
    cohort: str,
    cohort_donors: set[str],
    endpoint: EndpointDesign,
    seed: int,
    analysis_root: Path,
) -> Path:
    run_root = _base_run_root(analysis_root, endpoint, cohort, seed)
    paths = _base_input_paths(run_root)
    if paths["manifest"].is_file():
        manifest = _load_yaml_mapping(paths["manifest"])
        metadata = manifest.get("metadata", {})
        if (
            isinstance(metadata, dict)
            and str(metadata.get("cohort_manifest_sha256"))
            == cohort_manifest_sha256
            and str(metadata.get("source_sha256")) == source_sha256
            and all(path.is_file() for key, path in paths.items() if key != "manifest")
        ):
            return paths["manifest"]
        raise ValueError(f"Incompatible existing base run: {run_root}")

    local = obs.loc[obs["donor_id"].isin(cohort_donors)].copy()
    if local["donor_id"].nunique() != 54:
        raise ValueError(f"{cohort} base data do not contain 54 donors")
    query_mask, reference_mask, actual_seed = _valid_split(
        local, endpoint.endpoint, seed
    )
    incomplete_reference = reference_mask & ~local["cell_type"].eq(endpoint.endpoint)
    condition_mask = query_mask | incomplete_reference
    condition_obs = local.loc[condition_mask].copy()
    domains = pd.Series("reference", index=condition_obs.index)
    domains.loc[query_mask.loc[condition_mask]] = "query"
    positions = obs.index.get_indexer(condition_obs.index)
    if np.any(positions < 0):
        raise ValueError("condition cells are absent from the frozen source embedding")
    embedding = np.asarray(full_embedding[positions, :30], dtype=float)
    embedding_cells = pd.DataFrame(
        {
            "row_index": np.arange(len(condition_obs), dtype=int),
            "cell_id": condition_obs["cell_id"].astype(str).to_numpy(),
            "domain": domains.to_numpy(),
            "condition_id": CONDITION,
        }
    )
    edges = build_candidate_edges(
        embedding,
        embedding_cells,
        k_source_to_target=100,
        add_reverse_edges=True,
        distance="euclidean",
        k_target_to_source=100,
    )
    edges["scaled_distance"] = scale_distances(
        edges["distance"].to_numpy(dtype=float),
        scale="median",
        clip_quantile=0.99,
        delta=1.0e-8,
    )
    edges = edges[
        [
            "source_cell_id",
            "target_cell_id",
            "source_row",
            "target_row",
            "distance",
            "scaled_distance",
            "is_reverse_edge",
        ]
    ]

    query = local.loc[query_mask].copy()
    target = local.loc[incomplete_reference].copy()
    source_priors = pd.DataFrame(
        {
            "cell_id": query["cell_id"].astype(str).to_numpy(),
            "rho": query["rho"].to_numpy(dtype=float),
            "rho_source": HIHA_MATCHABILITY_SOURCE,
            "rho_recipe": HIHA_MATCHABILITY_SOURCE,
            "prior_risk": query["prior_risk"].to_numpy(dtype=float),
            "pmax_reference_classifier": 1.0,
            "anchor_class_pred": query["broad_label"].astype(str).to_numpy(),
            "anchor_confidence": 1.0,
        }
    )
    target_priors = pd.DataFrame(
        {
            "cell_id": target["cell_id"].astype(str).to_numpy(),
            "target_label_visible": target["cell_type"].astype(str).to_numpy(),
            "broad_anchor_class": target["broad_label"].astype(str).to_numpy(),
            "rho_target": 1.0,
        }
    )
    truth = pd.DataFrame(
        {
            "cell_id": query["cell_id"].astype(str).to_numpy(),
            "true_label": query["cell_type"].astype(str).to_numpy(),
            "removed_state": endpoint.endpoint,
            "is_absent_state": query["cell_type"].eq(endpoint.endpoint).to_numpy(),
            "is_shared_state": ~query["cell_type"].eq(endpoint.endpoint).to_numpy(),
        }
    )
    split = pd.DataFrame(
        {
            "cell_id": pd.concat(
                [
                    local.loc[query_mask, "cell_id"],
                    local.loc[reference_mask, "cell_id"],
                ],
                ignore_index=True,
            ).astype(str),
            "split_domain": (
                ["query"] * int(query_mask.sum())
                + ["reference"] * int(reference_mask.sum())
            ),
            "split_seed": actual_seed,
        }
    )

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(paths["candidates"], index=False)
    source_priors.to_csv(paths["source_priors"], index=False)
    target_priors.to_csv(paths["target_priors"], index=False)
    truth.to_csv(paths["truth"], index=False)
    split.to_csv(paths["split"], index=False)
    artifact_hashes = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key not in {"manifest"}
    }
    _write_yaml(
        paths["manifest"],
        {
            "stage": "rho_attribution_base_artifacts",
            "artifacts": {
                key: str(path.relative_to(analysis_root.parent.parent.parent.parent))
                if path.is_relative_to(analysis_root.parent.parent.parent.parent)
                else str(path)
                for key, path in paths.items()
                if key != "manifest"
            },
            "sha256": artifact_hashes,
            "metadata": {
                "cohort": cohort,
                "endpoint": endpoint.endpoint,
                "requested_seed": seed,
                "actual_seed": actual_seed,
                "n_cohort_donors": int(local["donor_id"].nunique()),
                "n_query": int(len(query)),
                "n_reference": int(len(target)),
                "n_positive": int(truth["is_absent_state"].sum()),
                "candidate_edges": int(len(edges)),
                "source_sha256": source_sha256,
                "cohort_manifest_sha256": cohort_manifest_sha256,
                "provider": HIHA_PROVIDER,
                "embedding_key": "X_pca_harmony",
                "embedding_dimensions": 30,
                "candidate_k": 100,
                "reverse_edges": True,
                "cost_scale": "condition_specific_median",
            },
        },
    )
    return paths["manifest"]


def prepare_hiha_bases(
    *,
    project_root: Path,
    design_manifest: Path,
    analysis_root: Path,
    cohorts: Sequence[str],
    endpoints: Sequence[str] | None,
    seeds: Sequence[int] | None,
    jobs: int,
) -> list[Path]:
    design = validate_design_manifest(design_manifest, project_root=project_root)
    source = design["source"]
    if not isinstance(source, dict):
        raise ValueError("design source must be a mapping")
    source_path = project_root / str(source["path"])
    source_sha256 = sha256_file(source_path)
    cohort_manifest_sha256 = _design_hash(design_manifest)
    selected_endpoints = _selected_endpoints(endpoints)
    selected_cohorts = tuple(cohorts or ("discovery", "confirmation"))
    unknown = set(selected_cohorts) - {"discovery", "confirmation"}
    if unknown:
        raise ValueError(f"Unknown cohort(s): {sorted(unknown)}")

    data = ad.read_h5ad(source_path, backed="r")
    try:
        obs = _standardized_hiha_obs(data)
        full_embedding = np.asarray(data.obsm["X_pca_harmony"])
    finally:
        data.file.close()
    tasks: list[tuple[str, set[str], EndpointDesign, int]] = []
    split_config = design.get("splits")
    if not isinstance(split_config, dict):
        raise ValueError("design splits must be a mapping")
    for cohort in selected_cohorts:
        cohort_donors = set(str(value) for value in design[f"{cohort}_donors"])
        configured_seeds = tuple(
            int(value) for value in split_config[f"{cohort}_seeds"]
        )
        active_seeds = tuple(seeds or configured_seeds)
        if not set(active_seeds) <= set(configured_seeds):
            raise ValueError(f"Requested seeds are outside the locked {cohort} seeds")
        for endpoint in selected_endpoints:
            for seed in active_seeds:
                tasks.append((cohort, cohort_donors, endpoint, seed))

    manifests: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _build_base_run,
                obs=obs,
                full_embedding=full_embedding,
                source_sha256=source_sha256,
                cohort_manifest_sha256=cohort_manifest_sha256,
                cohort=cohort,
                cohort_donors=cohort_donors,
                endpoint=endpoint,
                seed=seed,
                analysis_root=analysis_root,
            ): (cohort, endpoint.endpoint, seed)
            for cohort, cohort_donors, endpoint, seed in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            cohort, endpoint, seed = futures[future]
            manifest = future.result()
            manifests.append(manifest)
            print(f"prepared {cohort}: {endpoint}, seed {seed}", flush=True)
    return sorted(manifests)


def _checkpoint_columns() -> list[str]:
    columns = [
        "experiment",
        "analysis_stage",
        "cohort",
        "endpoint",
        "seed",
        "run_id",
        "alpha",
        "alpha_ratio",
        "tau_min",
        "tau_max",
        "mean_matched_tau",
        "tau_target",
        "epsilon",
        "max_iterations",
        "tolerance",
        "numerical_floor",
        "candidate_edges_sha256",
        "source_priors_sha256",
        "cohort_manifest_sha256",
        "heterogeneous_converged",
        "heterogeneous_n_iterations",
        "heterogeneous_runtime_seconds",
        "uniform_converged",
        "uniform_n_iterations",
        "uniform_runtime_seconds",
        "evaluation_scope",
        "n_detection",
        "n_positive",
    ]
    for metric in METRIC_NAMES:
        columns.extend(
            (
                f"heterogeneous_{metric}",
                f"uniform_{metric}",
                f"delta_{metric}_heterogeneous_minus_uniform",
            )
        )
    columns.append("relative_auprc_heterogeneous_minus_uniform_percent")
    return columns


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _read_checkpoint(
    path: Path,
    *,
    identity: dict[str, object],
    configured: set[SurfaceConfiguration],
) -> pd.DataFrame:
    columns = _checkpoint_columns()
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, float_precision="round_trip")
    if missing := sorted(set(columns) - set(frame)):
        raise ValueError(f"{path} lacks checkpoint columns: {missing}")
    for column, expected in identity.items():
        observed = frame[column]
        if isinstance(expected, int | float):
            matches = np.isclose(observed.astype(float), float(expected)).all()
        else:
            matches = observed.astype(str).eq(str(expected)).all()
        if not matches:
            raise ValueError(f"{path} has incompatible {column}")
    for column in ("heterogeneous_converged", "uniform_converged"):
        frame[column] = frame[column].map(_as_bool)
    keep = [
        SurfaceConfiguration(
            alpha=float(row.alpha),
            tau_min=float(row.tau_min),
            tau_max=float(row.tau_max),
        )
        in configured
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[keep, columns].copy()


def _configuration_slug(configuration: SurfaceConfiguration) -> str:
    return (
        f"alpha_{configuration.alpha:g}_"
        f"tau_min_{configuration.tau_min:g}_"
        f"tau_max_{configuration.tau_max:g}"
    ).replace(".", "p")


def _run_pair(
    *,
    experiment: str,
    endpoint: EndpointDesign,
    configuration: SurfaceConfiguration,
    condition: str,
    candidates: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    truth: pd.DataFrame,
    method_root: Path,
) -> dict[str, object]:
    spec = EndpointSpec(
        experiment=experiment,
        endpoint=endpoint.endpoint,
        run_prefix="",
        candidate_set=HIHA_CANDIDATE_SET,
        tau_values=(),
        selected_tau=(configuration.tau_min, configuration.tau_max),
        selected_alpha=endpoint.selected_alpha,
        tau_target=endpoint.tau_target,
    )
    metrics: dict[str, dict[str, object]] = {}
    metadata: dict[str, dict[str, object]] = {}
    runtimes: dict[str, float] = {}
    for variant in VARIANTS:
        scores, metadata[variant], runtimes[variant] = _run_method(
            variant=variant,
            method_root=method_root / variant,
            condition=condition,
            candidates=candidates,
            source_priors=source_priors,
            target_priors=target_priors,
            spec=spec,
            alpha=configuration.alpha,
        )
        metrics[variant] = _evaluate_variant(
            experiment=experiment,
            endpoint=endpoint.endpoint,
            truth=truth,
            scores=scores,
            pbmc_condition=None,
        )
    row: dict[str, object] = {
        "heterogeneous_converged": metadata["heterogeneous"]["converged"],
        "heterogeneous_n_iterations": metadata["heterogeneous"]["n_iterations"],
        "heterogeneous_runtime_seconds": runtimes["heterogeneous"],
        "uniform_converged": metadata["mean_matched_uniform"]["converged"],
        "uniform_n_iterations": metadata["mean_matched_uniform"]["n_iterations"],
        "uniform_runtime_seconds": runtimes["mean_matched_uniform"],
        "evaluation_scope": metrics["heterogeneous"]["evaluation_scope"],
        "n_detection": metrics["heterogeneous"]["n_detection"],
        "n_positive": metrics["heterogeneous"]["n_positive"],
    }
    for metric in METRIC_NAMES:
        heterogeneous = float(metrics["heterogeneous"][metric])
        uniform = float(metrics["mean_matched_uniform"][metric])
        row[f"heterogeneous_{metric}"] = heterogeneous
        row[f"uniform_{metric}"] = uniform
        row[f"delta_{metric}_heterogeneous_minus_uniform"] = (
            heterogeneous - uniform
        )
    row["relative_auprc_heterogeneous_minus_uniform_percent"] = relative_percent(
        float(row["delta_auprc_heterogeneous_minus_uniform"]),
        float(row["uniform_auprc"]),
    )
    if math.isclose(configuration.tau_min, configuration.tau_max):
        diagonal_delta = abs(
            float(row["delta_auprc_heterogeneous_minus_uniform"])
        )
        if diagonal_delta > 1.0e-12:
            raise ValueError(
                "Diagonal heterogeneous and mean-matched uniform fits differ: "
                f"{endpoint.endpoint}, {configuration}, delta={diagonal_delta}"
            )
    return row


def _run_hiha_split(
    *,
    analysis_root: Path,
    design_manifest_sha256: str,
    cohort: str,
    analysis_stage: str,
    endpoint: EndpointDesign,
    seed: int,
    configurations: Sequence[SurfaceConfiguration],
) -> Path:
    run_root = _base_run_root(analysis_root, endpoint, cohort, seed)
    inputs = _base_input_paths(run_root)
    missing = [path for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing base inputs for {run_root}: {', '.join(map(str, missing))}"
        )
    candidates = pd.read_parquet(inputs["candidates"])
    source_priors = pd.read_csv(inputs["source_priors"])
    target_priors = pd.read_csv(inputs["target_priors"])
    truth = pd.read_csv(inputs["truth"])
    candidate_hash = sha256_file(inputs["candidates"])
    source_hash = sha256_file(inputs["source_priors"])
    configured = set(configurations)
    checkpoint = (
        analysis_root
        / "checkpoints"
        / cohort
        / analysis_stage
        / endpoint.slug
        / f"seed{seed}.csv"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "experiment": "hiha",
        "analysis_stage": analysis_stage,
        "cohort": cohort,
        "endpoint": endpoint.endpoint,
        "seed": seed,
        "run_id": run_root.name,
        "tau_target": endpoint.tau_target,
        "epsilon": 0.05,
        "max_iterations": 5000,
        "tolerance": 1.0e-6,
        "numerical_floor": 1.0e-300,
        "candidate_edges_sha256": candidate_hash,
        "source_priors_sha256": source_hash,
        "cohort_manifest_sha256": design_manifest_sha256,
    }
    frame = _read_checkpoint(
        checkpoint, identity=identity, configured=configured
    )
    frame.to_csv(checkpoint, index=False)
    completed = {
        SurfaceConfiguration(
            alpha=float(row.alpha),
            tau_min=float(row.tau_min),
            tau_max=float(row.tau_max),
        )
        for row in frame.itertuples(index=False)
    }
    for configuration in sorted(configured - completed):
        tau_mean = mean_matched_tau(
            source_priors, configuration.tau_min, configuration.tau_max
        )
        temporary = (
            analysis_root
            / "tmp"
            / cohort
            / analysis_stage
            / endpoint.slug
            / f"seed{seed}"
            / _configuration_slug(configuration)
        )
        pair = _run_pair(
            experiment="hiha",
            endpoint=endpoint,
            configuration=configuration,
            condition=CONDITION,
            candidates=candidates,
            source_priors=source_priors,
            target_priors=target_priors,
            truth=truth,
            method_root=temporary,
        )
        row = {
            **identity,
            "alpha": configuration.alpha,
            "alpha_ratio": configuration.alpha / endpoint.selected_alpha,
            "tau_min": configuration.tau_min,
            "tau_max": configuration.tau_max,
            "mean_matched_tau": tau_mean,
            **pair,
        }
        new_row = pd.DataFrame([row], columns=_checkpoint_columns())
        frame = (
            new_row
            if frame.empty
            else pd.concat([frame, new_row], ignore_index=True)
        ).sort_values(["alpha", "tau_min", "tau_max"])
        frame.to_csv(checkpoint, index=False)
        shutil.rmtree(temporary)
        print(
            f"completed {cohort}/{analysis_stage}: {endpoint.endpoint}, "
            f"seed {seed}, {configuration}",
            flush=True,
        )
    return checkpoint


def summarize_surface(by_split: pd.DataFrame, *, expected_splits: int) -> pd.DataFrame:
    by_split = by_split.copy()
    group_columns = [
        "experiment",
        "analysis_stage",
        "cohort",
        "endpoint",
        "alpha",
        "alpha_ratio",
        "tau_min",
        "tau_max",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "n_splits": ("seed", "nunique"),
        "all_heterogeneous_converged": ("heterogeneous_converged", "all"),
        "all_uniform_converged": ("uniform_converged", "all"),
        "max_n_iterations": ("heterogeneous_n_iterations", "max"),
        "mean_matched_tau_mean": ("mean_matched_tau", "mean"),
        "mean_matched_tau_sd": ("mean_matched_tau", "std"),
    }
    for metric in METRIC_NAMES:
        for prefix in ("heterogeneous", "uniform"):
            aggregations[f"{prefix}_{metric}_mean"] = (
                f"{prefix}_{metric}",
                "mean",
            )
            aggregations[f"{prefix}_{metric}_sd"] = (
                f"{prefix}_{metric}",
                "std",
            )
        delta = f"delta_{metric}_heterogeneous_minus_uniform"
        aggregations[f"{delta}_mean"] = (delta, "mean")
        aggregations[f"{delta}_sd"] = (delta, "std")
        aggregations[f"{delta}_n_positive"] = (
            delta,
            lambda values: int((values > 0.0).sum()),
        )
        relative = f"relative_{metric}_heterogeneous_minus_uniform_percent"
        comparator = by_split[f"uniform_{metric}"].astype(float)
        if not comparator.gt(0.0).all():
            raise ValueError(
                f"relative {metric} difference requires positive comparators"
            )
        computed_relative = 100.0 * by_split[delta].astype(float) / comparator
        if relative in by_split and not np.allclose(
            by_split[relative].astype(float),
            computed_relative,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError(f"stored relative {metric} differences are inconsistent")
        by_split[relative] = computed_relative
        aggregations[f"{relative}_mean"] = (relative, "mean")
        aggregations[f"{relative}_sd"] = (relative, "std")
    summary = (
        by_split.groupby(group_columns, sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    ap_delta = "delta_auprc_heterogeneous_minus_uniform"
    summary["discovery_support"] = (
        summary["n_splits"].eq(expected_splits)
        & summary["all_heterogeneous_converged"].astype(bool)
        & summary["all_uniform_converged"].astype(bool)
        & summary[f"{ap_delta}_mean"].gt(0.0)
        & summary[f"{ap_delta}_n_positive"].ge(4)
    )
    return summary


def _surface_table_paths(
    analysis_root: Path, cohort: str, analysis_stage: str
) -> tuple[Path, Path]:
    tables = analysis_root / "tables"
    return (
        tables / f"{cohort}_{analysis_stage}_by_split.csv",
        tables / f"{cohort}_{analysis_stage}_summary.csv",
    )


def _aggregate_hiha(
    *,
    analysis_root: Path,
    cohort: str,
    analysis_stage: str,
    endpoints: Sequence[EndpointDesign],
    seeds: Sequence[int],
    configurations: dict[str, tuple[SurfaceConfiguration, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for endpoint in endpoints:
        expected = set(configurations[endpoint.endpoint])
        for seed in seeds:
            checkpoint = (
                analysis_root
                / "checkpoints"
                / cohort
                / analysis_stage
                / endpoint.slug
                / f"seed{seed}.csv"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            frame = pd.read_csv(checkpoint, float_precision="round_trip")
            observed = {
                SurfaceConfiguration(
                    alpha=float(row.alpha),
                    tau_min=float(row.tau_min),
                    tau_max=float(row.tau_max),
                )
                for row in frame.itertuples(index=False)
            }
            if observed != expected:
                raise ValueError(
                    f"Incomplete checkpoint {checkpoint}: "
                    f"{len(observed)} of {len(expected)} configurations"
                )
            frames.append(frame)
    by_split = pd.concat(frames, ignore_index=True)
    summary = summarize_surface(by_split, expected_splits=len(seeds))
    by_split_path, summary_path = _surface_table_paths(
        analysis_root, cohort, analysis_stage
    )
    by_split_path.parent.mkdir(parents=True, exist_ok=True)
    by_split.to_csv(by_split_path, index=False)
    summary.to_csv(summary_path, index=False)
    return by_split, summary


def _render_relative_heatmaps(
    *,
    summary: pd.DataFrame,
    endpoints: Sequence[EndpointDesign],
    tau_values: Sequence[float],
    alpha_values: dict[str, Sequence[float]],
    path: Path,
) -> None:
    n_columns = max(len(tuple(alpha_values[endpoint.endpoint])) for endpoint in endpoints)
    figure, axes = plt.subplots(
        len(endpoints),
        n_columns,
        figsize=(3.1 * n_columns + 0.9, 3.4 * len(endpoints) + 0.5),
        squeeze=False,
        layout="constrained",
    )
    effect = "relative_auprc_heterogeneous_minus_uniform_percent_mean"
    finite = summary[effect].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit = max(limit, 1.0e-12)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None
    ordered_tau = tuple(sorted({float(value) for value in tau_values}))
    for row_index, endpoint in enumerate(endpoints):
        endpoint_alphas = tuple(alpha_values[endpoint.endpoint])
        for column_index in range(n_columns):
            axis = axes[row_index, column_index]
            if column_index >= len(endpoint_alphas):
                axis.axis("off")
                continue
            alpha = endpoint_alphas[column_index]
            local = summary.loc[
                summary["endpoint"].eq(endpoint.endpoint)
                & np.isclose(summary["alpha"].astype(float), alpha)
            ]
            matrix = local.pivot(
                index="tau_min", columns="tau_max", values=effect
            ).reindex(index=ordered_tau, columns=ordered_tau)
            image = axis.imshow(
                matrix.to_numpy(dtype=float),
                origin="lower",
                cmap="RdBu_r",
                norm=norm,
                aspect="equal",
            )
            axis.set_title(
                f"{endpoint.endpoint}\n"
                rf"$\alpha={alpha:g}$ ($\alpha/\alpha^\star="
                f"{alpha / endpoint.selected_alpha:g}$)",
                fontsize=9,
            )
            axis.set_xticks(range(len(ordered_tau)), [f"{v:g}" for v in ordered_tau])
            axis.set_yticks(range(len(ordered_tau)), [f"{v:g}" for v in ordered_tau])
            axis.tick_params(axis="x", rotation=55, labelsize=7)
            axis.tick_params(axis="y", labelsize=7)
            for i in range(len(ordered_tau)):
                for j in range(len(ordered_tau)):
                    value = matrix.iloc[i, j]
                    if pd.isna(value):
                        continue
                    normalized = float(norm(float(value)))
                    axis.text(
                        j,
                        i,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if abs(normalized - 0.5) > 0.32 else "black",
                    )
    figure.supxlabel(r"$\tau_{\max}$", fontsize=10)
    figure.supylabel(r"$\tau_{\min}$", fontsize=10)
    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            location="right",
            shrink=0.82,
            pad=0.025,
        )
        colorbar.set_label("Relative AP difference (%)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _coarse_hiha_configurations(
    design: dict[str, object], endpoints: Sequence[EndpointDesign]
) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]], dict[str, tuple[SurfaceConfiguration, ...]]]:
    discovery_grid = design.get("discovery_grid")
    if not isinstance(discovery_grid, dict):
        raise ValueError("design lacks discovery_grid")
    hiha = discovery_grid.get("hiha")
    if not isinstance(hiha, dict):
        raise ValueError("design lacks discovery_grid.hiha")
    tau_values = tuple(float(value) for value in hiha["tau_values"])
    ratios = tuple(float(value) for value in hiha["alpha_ratios"])
    alpha_values = {
        endpoint.endpoint: tuple(endpoint.selected_alpha * ratio for ratio in ratios)
        for endpoint in endpoints
    }
    configurations = {
        endpoint.endpoint: coarse_configurations(
            tau_values, alpha_values[endpoint.endpoint]
        )
        for endpoint in endpoints
    }
    return tau_values, alpha_values, configurations


def run_hiha_coarse(
    *,
    project_root: Path,
    design_manifest: Path,
    analysis_root: Path,
    endpoints: Sequence[str] | None,
    seeds: Sequence[int] | None,
    jobs: int,
    only_configuration: SurfaceConfiguration | None = None,
) -> tuple[Path, Path] | None:
    design = validate_design_manifest(design_manifest, project_root=project_root)
    selected_endpoints = _selected_endpoints(endpoints)
    split_config = design["splits"]
    if not isinstance(split_config, dict):
        raise ValueError("design splits must be a mapping")
    configured_seeds = tuple(
        int(value) for value in split_config["discovery_seeds"]
    )
    active_seeds = tuple(seeds or configured_seeds)
    if not set(active_seeds) <= set(configured_seeds):
        raise ValueError("coarse discovery seeds are outside the locked design")
    tau_values, alpha_values, all_configurations = _coarse_hiha_configurations(
        design, selected_endpoints
    )
    configurations = {
        endpoint.endpoint: (
            (only_configuration,)
            if only_configuration is not None
            else all_configurations[endpoint.endpoint]
        )
        for endpoint in selected_endpoints
    }
    if only_configuration is not None:
        for endpoint in selected_endpoints:
            if only_configuration not in set(all_configurations[endpoint.endpoint]):
                raise ValueError(
                    f"Pilot configuration is outside the locked {endpoint.endpoint} grid"
                )
    design_sha256 = _design_hash(design_manifest)
    tasks = [
        (endpoint, seed)
        for endpoint in selected_endpoints
        for seed in active_seeds
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_hiha_split,
                analysis_root=analysis_root,
                design_manifest_sha256=design_sha256,
                cohort="discovery",
                analysis_stage="coarse",
                endpoint=endpoint,
                seed=seed,
                configurations=configurations[endpoint.endpoint],
            ): (endpoint.endpoint, seed)
            for endpoint, seed in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if only_configuration is not None or set(active_seeds) != set(configured_seeds):
        return None
    by_split, summary = _aggregate_hiha(
        analysis_root=analysis_root,
        cohort="discovery",
        analysis_stage="coarse",
        endpoints=selected_endpoints,
        seeds=active_seeds,
        configurations=configurations,
    )
    figure = analysis_root / "figures/discovery_coarse_relative_ap.png"
    _render_relative_heatmaps(
        summary=summary,
        endpoints=selected_endpoints,
        tau_values=tau_values,
        alpha_values=alpha_values,
        path=figure,
    )
    return _surface_table_paths(analysis_root, "discovery", "coarse")


def _run_hiha_alpha0_grid(
    *,
    project_root: Path,
    design_manifest: Path,
    analysis_root: Path,
    endpoints: Sequence[str] | None,
    jobs: int,
    tau_values: Sequence[float],
    configurations: Sequence[SurfaceConfiguration],
    analysis_stage: str,
    design_stage: str,
    design_filename: str,
    result_filename: str,
    figure_filename: str,
    tau_pair_rule: str,
    selection_provenance: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, Path]:
    design = validate_design_manifest(design_manifest, project_root=project_root)
    selected_endpoints = _selected_endpoints(endpoints)
    split_config = design["splits"]
    if not isinstance(split_config, dict):
        raise ValueError("design splits must be a mapping")
    seeds = tuple(int(value) for value in split_config["discovery_seeds"])
    configurations = tuple(configurations)
    endpoint_configurations = {
        endpoint.endpoint: configurations for endpoint in selected_endpoints
    }
    grid_design: dict[str, object] = {
        "stage": design_stage,
        "status": "locked_before_execution",
        "cohort_manifest_sha256": _design_hash(design_manifest),
        "endpoints": [endpoint.endpoint for endpoint in selected_endpoints],
        "alpha": 0.0,
        "tau_values": [float(value) for value in tau_values],
        "tau_pair_rule": tau_pair_rule,
        "n_tau_pairs": len(configurations),
        "discovery_seeds": list(seeds),
        "comparator": "empirical_mass_mean_matched_uniform_query_penalty",
        "primary_effect": "heterogeneous_ap_minus_mean_matched_uniform_ap",
        "support_rule": {
            "require_positive_mean": True,
            "minimum_positive_splits": 4,
            "n_splits": len(seeds),
        },
        "selection_rule": (
            "largest mean AP difference in AP units among supported cells"
        ),
        "confirmation_policy": (
            "do not inspect confirmation outcomes unless a discovery cell "
            "satisfies the locked support rule"
        ),
    }
    if selection_provenance is not None:
        grid_design["selection_provenance"] = selection_provenance
    grid_design_path = analysis_root / f"design/{design_filename}"
    if grid_design_path.is_file():
        if _load_yaml_mapping(grid_design_path) != grid_design:
            raise ValueError(
                f"Refusing to change locked alpha-zero grid: {grid_design_path}"
            )
    else:
        _write_yaml(grid_design_path, grid_design)

    design_sha256 = _design_hash(design_manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _run_hiha_split,
                analysis_root=analysis_root,
                design_manifest_sha256=design_sha256,
                cohort="discovery",
                analysis_stage=analysis_stage,
                endpoint=endpoint,
                seed=seed,
                configurations=configurations,
            )
            for endpoint in selected_endpoints
            for seed in seeds
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    by_split, summary = _aggregate_hiha(
        analysis_root=analysis_root,
        cohort="discovery",
        analysis_stage=analysis_stage,
        endpoints=selected_endpoints,
        seeds=seeds,
        configurations=endpoint_configurations,
    )
    figure_path = analysis_root / f"figures/{figure_filename}"
    _render_relative_heatmaps(
        summary=summary,
        endpoints=selected_endpoints,
        tau_values=tau_values,
        alpha_values={
            endpoint.endpoint: (0.0,) for endpoint in selected_endpoints
        },
        path=figure_path,
    )

    supported = summary.loc[summary["discovery_support"].map(_as_bool)].copy()
    off_diagonal = summary.loc[
        ~np.isclose(summary["tau_min"], summary["tau_max"])
    ]
    result: dict[str, object] = {
        "status": (
            "discovery_supported_configuration"
            if not supported.empty
            else "no_discovery_supported_configuration"
        ),
        "alpha": 0.0,
        "n_parameter_cells": int(len(summary)),
        "n_paired_split_outcomes": int(len(by_split)),
        "n_supported_cells": int(len(supported)),
        "all_heterogeneous_converged": bool(
            summary["all_heterogeneous_converged"].all()
        ),
        "all_uniform_converged": bool(
            summary["all_uniform_converged"].all()
        ),
        "best_off_diagonal_by_endpoint": {},
    }
    best_by_endpoint = result["best_off_diagonal_by_endpoint"]
    if not isinstance(best_by_endpoint, dict):
        raise TypeError("best_off_diagonal_by_endpoint must be a mapping")
    for endpoint in selected_endpoints:
        row = (
            off_diagonal.loc[
                off_diagonal["endpoint"].eq(endpoint.endpoint)
            ]
            .sort_values(
                [
                    "delta_auprc_heterogeneous_minus_uniform_mean",
                    "tau_min",
                    "tau_max",
                ],
                ascending=[False, True, True],
            )
            .iloc[0]
        )
        best_by_endpoint[endpoint.endpoint] = {
            "tau_min": float(row["tau_min"]),
            "tau_max": float(row["tau_max"]),
            "heterogeneous_ap": float(row["heterogeneous_auprc_mean"]),
            "uniform_ap": float(row["uniform_auprc_mean"]),
            "delta_ap": float(
                row["delta_auprc_heterogeneous_minus_uniform_mean"]
            ),
            "n_positive_splits": int(
                row[
                    "delta_auprc_heterogeneous_minus_uniform_n_positive"
                ]
            ),
            "relative_delta_ap_percent": float(
                row[
                    "relative_auprc_heterogeneous_minus_uniform_percent_mean"
                ]
            ),
        }
    if not supported.empty:
        selected = supported.sort_values(
            [
                "delta_auprc_heterogeneous_minus_uniform_mean",
                "endpoint",
                "tau_min",
                "tau_max",
            ],
            ascending=[False, True, True, True],
        ).iloc[0]
        result["locked_candidate"] = {
            "endpoint": str(selected["endpoint"]),
            "tau_min": float(selected["tau_min"]),
            "tau_max": float(selected["tau_max"]),
            "mean_delta_ap": float(
                selected[
                    "delta_auprc_heterogeneous_minus_uniform_mean"
                ]
            ),
            "n_positive_splits": int(
                selected[
                    "delta_auprc_heterogeneous_minus_uniform_n_positive"
                ]
            ),
        }

    by_split_path, summary_path = _surface_table_paths(
        analysis_root, "discovery", analysis_stage
    )
    result["artifacts"] = {
        "design": str(grid_design_path),
        "design_sha256": sha256_file(grid_design_path),
        "by_split": str(by_split_path),
        "by_split_sha256": sha256_file(by_split_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "figure": str(figure_path),
        "figure_sha256": sha256_file(figure_path),
    }
    result_path = analysis_root / f"design/{result_filename}"
    if result_path.is_file():
        if _load_yaml_mapping(result_path) != result:
            raise ValueError(
                f"Refusing to change alpha-zero grid result: {result_path}"
            )
    else:
        _write_yaml(result_path, result)
    return by_split_path, summary_path, figure_path, result_path


def run_hiha_alpha0_fine(
    *,
    project_root: Path,
    design_manifest: Path,
    analysis_root: Path,
    endpoints: Sequence[str] | None,
    jobs: int,
) -> tuple[Path, Path, Path, Path]:
    return _run_hiha_alpha0_grid(
        project_root=project_root,
        design_manifest=design_manifest,
        analysis_root=analysis_root,
        endpoints=endpoints,
        jobs=jobs,
        tau_values=HIHA_ALPHA0_FINE_TAU_VALUES,
        configurations=hiha_alpha0_fine_configurations(),
        analysis_stage="alpha0_fine",
        design_stage="discovery_alpha0_fine",
        design_filename="alpha0_fine_grid.yaml",
        result_filename="alpha0_fine_result.yaml",
        figure_filename="discovery_alpha0_fine_relative_ap.png",
        tau_pair_rule="tau_min <= tau_max and tau_max / tau_min <= 2",
    )


def run_hiha_alpha0_focused(
    *,
    project_root: Path,
    design_manifest: Path,
    analysis_root: Path,
    endpoints: Sequence[str] | None,
    jobs: int,
) -> tuple[Path, Path, Path, Path]:
    return _run_hiha_alpha0_grid(
        project_root=project_root,
        design_manifest=design_manifest,
        analysis_root=analysis_root,
        endpoints=endpoints,
        jobs=jobs,
        tau_values=HIHA_ALPHA0_FOCUSED_TAU_VALUES,
        configurations=hiha_alpha0_focused_configurations(),
        analysis_stage="alpha0_focused025_125",
        design_stage="discovery_alpha0_focused025_125",
        design_filename="alpha0_focused025_125_grid.yaml",
        result_filename="alpha0_focused025_125_result.yaml",
        figure_filename="discovery_alpha0_focused025_125_relative_ap.png",
        tau_pair_rule="all tau_min <= tau_max pairs",
        selection_provenance={
            "status": "post_hoc_exploratory",
            "requested_after_inspecting": (
                "the broader fixed-alpha-zero tau surface"
            ),
            "purpose": "narrower manuscript-facing descriptive display",
            "confirmatory_interpretation": False,
        },
    )


def _best_off_diagonal(summary: pd.DataFrame, endpoint: str) -> SurfaceConfiguration:
    local = summary.loc[
        summary["endpoint"].eq(endpoint)
        & ~np.isclose(summary["tau_min"], summary["tau_max"])
        & summary["all_heterogeneous_converged"].astype(bool)
        & summary["all_uniform_converged"].astype(bool)
    ].copy()
    if local.empty:
        raise ValueError(f"No converged off-diagonal coarse cells for {endpoint}")
    local = local.sort_values(
        [
            "delta_auprc_heterogeneous_minus_uniform_mean",
            "alpha",
            "tau_min",
            "tau_max",
        ],
        ascending=[False, True, True, True],
    )
    row = local.iloc[0]
    return SurfaceConfiguration(
        alpha=float(row["alpha"]),
        tau_min=float(row["tau_min"]),
        tau_max=float(row["tau_max"]),
    )


def prepare_hiha_refinement(
    *,
    design_manifest: Path,
    analysis_root: Path,
    endpoints: Sequence[str] | None,
) -> Path:
    design = _load_yaml_mapping(design_manifest)
    selected_endpoints = _selected_endpoints(endpoints)
    _, coarse_summary_path = _surface_table_paths(
        analysis_root, "discovery", "coarse"
    )
    if not coarse_summary_path.is_file():
        raise FileNotFoundError(f"Missing coarse summary: {coarse_summary_path}")
    summary = pd.read_csv(coarse_summary_path, float_precision="round_trip")
    tau_values, alpha_values, coarse = _coarse_hiha_configurations(
        design, selected_endpoints
    )
    endpoint_payload: dict[str, object] = {}
    for endpoint in selected_endpoints:
        center = _best_off_diagonal(summary, endpoint.endpoint)
        refined = refinement_configurations(
            center=center,
            tau_values=tau_values,
            alpha_values=alpha_values[endpoint.endpoint],
            coarse=coarse[endpoint.endpoint],
        )
        endpoint_payload[endpoint.endpoint] = {
            "center": {
                "alpha": center.alpha,
                "tau_min": center.tau_min,
                "tau_max": center.tau_max,
            },
            "configurations": [
                {
                    "alpha": configuration.alpha,
                    "tau_min": configuration.tau_min,
                    "tau_max": configuration.tau_max,
                }
                for configuration in refined
            ],
        }
    path = analysis_root / "design/refinement_grid.yaml"
    payload = {
        "stage": "discovery_local_refinement",
        "locked_before_refinement_execution": True,
        "cohort_manifest_sha256": _design_hash(design_manifest),
        "coarse_summary_sha256": sha256_file(coarse_summary_path),
        "endpoints": endpoint_payload,
    }
    if path.is_file():
        existing = _load_yaml_mapping(path)
        if existing != payload:
            existing_design = dict(existing)
            current_design = dict(payload)
            existing_design.pop("coarse_summary_sha256", None)
            current_design.pop("coarse_summary_sha256", None)
            if existing_design != current_design:
                raise ValueError(
                    f"Refusing to overwrite incompatible refinement design: {path}"
                )
        return path
    _write_yaml(path, payload)
    return path


def _load_refinement_configurations(
    path: Path, endpoint: str
) -> tuple[SurfaceConfiguration, ...]:
    payload = _load_yaml_mapping(path)
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, dict) or endpoint not in endpoints:
        raise ValueError(f"Refinement design lacks endpoint {endpoint}")
    local = endpoints[endpoint]
    if not isinstance(local, dict):
        raise ValueError(f"Invalid refinement design for {endpoint}")
    configurations = local.get("configurations")
    if not isinstance(configurations, list):
        raise ValueError(f"Refinement configurations must be a list for {endpoint}")
    return tuple(
        SurfaceConfiguration(
            alpha=float(configuration["alpha"]),
            tau_min=float(configuration["tau_min"]),
            tau_max=float(configuration["tau_max"]),
        )
        for configuration in configurations
        if isinstance(configuration, dict)
    )


def run_hiha_refinement(
    *,
    project_root: Path,
    design_manifest: Path,
    analysis_root: Path,
    endpoints: Sequence[str] | None,
    jobs: int,
) -> tuple[Path, Path]:
    design = validate_design_manifest(design_manifest, project_root=project_root)
    selected_endpoints = _selected_endpoints(endpoints)
    refinement_design = prepare_hiha_refinement(
        design_manifest=design_manifest,
        analysis_root=analysis_root,
        endpoints=endpoints,
    )
    split_config = design["splits"]
    if not isinstance(split_config, dict):
        raise ValueError("design splits must be a mapping")
    seeds = tuple(int(value) for value in split_config["discovery_seeds"])
    configurations = {
        endpoint.endpoint: _load_refinement_configurations(
            refinement_design, endpoint.endpoint
        )
        for endpoint in selected_endpoints
    }
    design_sha256 = _design_hash(design_manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _run_hiha_split,
                analysis_root=analysis_root,
                design_manifest_sha256=design_sha256,
                cohort="discovery",
                analysis_stage="refinement",
                endpoint=endpoint,
                seed=seed,
                configurations=configurations[endpoint.endpoint],
            )
            for endpoint in selected_endpoints
            for seed in seeds
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    _aggregate_hiha(
        analysis_root=analysis_root,
        cohort="discovery",
        analysis_stage="refinement",
        endpoints=selected_endpoints,
        seeds=seeds,
        configurations=configurations,
    )
    return _surface_table_paths(analysis_root, "discovery", "refinement")


def _combined_discovery_tables(analysis_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_split_frames = []
    summary_frames = []
    for stage in ("coarse", "refinement"):
        by_split_path, summary_path = _surface_table_paths(
            analysis_root, "discovery", stage
        )
        if not by_split_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing discovery {stage} tables")
        by_split_frames.append(
            pd.read_csv(by_split_path, float_precision="round_trip")
        )
        summary_frames.append(
            pd.read_csv(summary_path, float_precision="round_trip")
        )
    by_split = pd.concat(by_split_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    return by_split, summary


def lock_hiha_selection(
    *, design_manifest: Path, analysis_root: Path
) -> Path:
    by_split, summary = _combined_discovery_tables(analysis_root)
    candidates = summary.loc[summary["discovery_support"].map(_as_bool)].copy()
    if candidates.empty:
        failure = analysis_root / "design/discovery_no_supported_configuration.yaml"
        _write_yaml(
            failure,
            {
                "status": "no_discovery_supported_configuration",
                "cohort_manifest_sha256": _design_hash(design_manifest),
                "coarse_summary_sha256": sha256_file(
                    _surface_table_paths(
                        analysis_root, "discovery", "coarse"
                    )[1]
                ),
                "refinement_summary_sha256": sha256_file(
                    _surface_table_paths(
                        analysis_root, "discovery", "refinement"
                    )[1]
                ),
            },
        )
        raise ValueError(
            "Discovery produced no configuration meeting the locked support rule"
        )
    candidates = candidates.sort_values(
        [
            "delta_auprc_heterogeneous_minus_uniform_mean",
            "endpoint",
            "alpha",
            "tau_min",
            "tau_max",
        ],
        ascending=[False, True, True, True, True],
    )
    selected = candidates.iloc[0]
    endpoint = str(selected["endpoint"])
    configuration = SurfaceConfiguration(
        alpha=float(selected["alpha"]),
        tau_min=float(selected["tau_min"]),
        tau_max=float(selected["tau_max"]),
    )
    selected_splits = by_split.loc[
        by_split["endpoint"].eq(endpoint)
        & np.isclose(by_split["alpha"], configuration.alpha)
        & np.isclose(by_split["tau_min"], configuration.tau_min)
        & np.isclose(by_split["tau_max"], configuration.tau_max)
    ].sort_values("seed")
    payload = {
        "status": "locked_before_confirmation_execution",
        "endpoint": endpoint,
        "configuration": {
            "alpha": configuration.alpha,
            "tau_min": configuration.tau_min,
            "tau_max": configuration.tau_max,
            "tau_target": _endpoint_design(endpoint).tau_target,
        },
        "selection_rule": (
            "largest mean absolute AP difference among cells with positive "
            "differences on at least four of five discovery splits"
        ),
        "discovery": {
            "mean_delta_ap": float(
                selected["delta_auprc_heterogeneous_minus_uniform_mean"]
            ),
            "n_positive_splits": int(
                selected[
                    "delta_auprc_heterogeneous_minus_uniform_n_positive"
                ]
            ),
            "mean_relative_delta_ap_percent": float(
                selected[
                    "relative_auprc_heterogeneous_minus_uniform_percent_mean"
                ]
            ),
            "split_delta_ap": [
                {
                    "seed": int(row.seed),
                    "delta_ap": float(
                        row.delta_auprc_heterogeneous_minus_uniform
                    ),
                }
                for row in selected_splits.itertuples(index=False)
            ],
        },
        "cohort_manifest_sha256": _design_hash(design_manifest),
        "coarse_summary_sha256": sha256_file(
            _surface_table_paths(analysis_root, "discovery", "coarse")[1]
        ),
        "refinement_summary_sha256": sha256_file(
            _surface_table_paths(analysis_root, "discovery", "refinement")[1]
        ),
    }
    path = analysis_root / "design/locked_selection.yaml"
    if path.is_file():
        existing = _load_yaml_mapping(path)
        if existing != payload:
            raise ValueError(f"Refusing to change the locked selection: {path}")
        return path
    _write_yaml(path, payload)
    return path


def run_hiha_confirmation(
    *,
    project_root: Path,
    design_manifest: Path,
    analysis_root: Path,
    jobs: int,
) -> tuple[Path, Path]:
    design = validate_design_manifest(design_manifest, project_root=project_root)
    locked_path = analysis_root / "design/locked_selection.yaml"
    if not locked_path.is_file():
        raise FileNotFoundError(
            "Confirmation cannot run before design/locked_selection.yaml exists"
        )
    locked = _load_yaml_mapping(locked_path)
    if str(locked.get("status")) != "locked_before_confirmation_execution":
        raise ValueError("Locked selection has an invalid status")
    if str(locked.get("cohort_manifest_sha256")) != _design_hash(design_manifest):
        raise ValueError("Locked selection does not match the cohort manifest")
    endpoint = _endpoint_design(str(locked["endpoint"]))
    raw_configuration = locked["configuration"]
    if not isinstance(raw_configuration, dict):
        raise ValueError("Locked configuration must be a mapping")
    configuration = SurfaceConfiguration(
        alpha=float(raw_configuration["alpha"]),
        tau_min=float(raw_configuration["tau_min"]),
        tau_max=float(raw_configuration["tau_max"]),
    )
    split_config = design["splits"]
    if not isinstance(split_config, dict):
        raise ValueError("design splits must be a mapping")
    seeds = tuple(int(value) for value in split_config["confirmation_seeds"])
    design_sha256 = _design_hash(design_manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _run_hiha_split,
                analysis_root=analysis_root,
                design_manifest_sha256=design_sha256,
                cohort="confirmation",
                analysis_stage="locked",
                endpoint=endpoint,
                seed=seed,
                configurations=(configuration,),
            )
            for seed in seeds
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    configurations = {endpoint.endpoint: (configuration,)}
    by_split, summary = _aggregate_hiha(
        analysis_root=analysis_root,
        cohort="confirmation",
        analysis_stage="locked",
        endpoints=(endpoint,),
        seeds=seeds,
        configurations=configurations,
    )
    row = summary.iloc[0]
    support = bool(row["discovery_support"])
    result = {
        "status": (
            "confirmed_endpoint_specific_advantage"
            if support
            else "confirmation_criterion_not_met"
        ),
        "endpoint": endpoint.endpoint,
        "configuration": {
            "alpha": configuration.alpha,
            "tau_min": configuration.tau_min,
            "tau_max": configuration.tau_max,
            "tau_target": endpoint.tau_target,
        },
        "confirmation": {
            "mean_delta_ap": float(
                row["delta_auprc_heterogeneous_minus_uniform_mean"]
            ),
            "n_positive_splits": int(
                row["delta_auprc_heterogeneous_minus_uniform_n_positive"]
            ),
            "mean_relative_delta_ap_percent": float(
                row[
                    "relative_auprc_heterogeneous_minus_uniform_percent_mean"
                ]
            ),
            "criterion_met": support,
        },
        "locked_selection_sha256": sha256_file(locked_path),
        "confirmation_by_split_sha256": sha256_file(
            _surface_table_paths(analysis_root, "confirmation", "locked")[0]
        ),
        "confirmation_summary_sha256": sha256_file(
            _surface_table_paths(analysis_root, "confirmation", "locked")[1]
        ),
    }
    _write_yaml(analysis_root / "confirmation_result.yaml", result)
    return _surface_table_paths(analysis_root, "confirmation", "locked")


def _mouse_configurations(
    design_manifest: Path,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[SurfaceConfiguration, ...],
]:
    design = _load_yaml_mapping(design_manifest)
    discovery_grid = design.get("discovery_grid")
    if not isinstance(discovery_grid, dict):
        raise ValueError("design lacks discovery grid")
    mouse = discovery_grid.get("mouse_spleen")
    if not isinstance(mouse, dict):
        raise ValueError("design lacks mouse-spleen discovery grid")
    tau_values = tuple(float(value) for value in mouse["tau_values"])
    alpha_values = tuple(float(value) for value in mouse["alpha_values"])
    return (
        tau_values,
        alpha_values,
        coarse_configurations(tau_values, alpha_values),
    )


def _run_mouse_surface(
    *,
    project_root: Path,
    design_manifest: Path,
    output_root: Path,
    analysis_stage: str,
    configurations: Sequence[SurfaceConfiguration],
    jobs: int,
    retain_fit_artifacts: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root, inputs = _mouse_input_paths(project_root)
    missing = [path for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing mouse inputs: {missing}")
    candidates = pd.read_parquet(inputs["candidates"])
    source_priors = pd.read_csv(inputs["source_priors"])
    target_priors = pd.read_csv(inputs["target_priors"])
    truth = pd.read_csv(inputs["truth"])
    candidate_hash = sha256_file(inputs["candidates"])
    source_hash = sha256_file(inputs["source_priors"])
    design_sha256 = _design_hash(design_manifest)
    checkpoint = output_root / f"checkpoints/{analysis_stage}/proliferating.csv"
    identity = {
        "experiment": "mouse",
        "analysis_stage": analysis_stage,
        "cohort": "single_dataset",
        "endpoint": MOUSE_ENDPOINT.endpoint,
        "seed": 0,
        "run_id": run_root.name,
        "tau_target": MOUSE_ENDPOINT.tau_target,
        "epsilon": 0.05,
        "max_iterations": 5000,
        "tolerance": 1.0e-6,
        "numerical_floor": 1.0e-300,
        "candidate_edges_sha256": candidate_hash,
        "source_priors_sha256": source_hash,
        "cohort_manifest_sha256": design_sha256,
    }
    configured = set(configurations)
    frame = _read_checkpoint(
        checkpoint, identity=identity, configured=configured
    )

    def pair_root(configuration: SurfaceConfiguration) -> Path:
        storage = "fits" if retain_fit_artifacts else "tmp"
        return (
            output_root
            / storage
            / analysis_stage
            / _configuration_slug(configuration)
        )

    def complete_pair_artifacts(path: Path) -> bool:
        return all(
            (path / variant / filename).is_file()
            for variant in VARIANTS
            for filename in RETAINED_FIT_FILES
        )

    if retain_fit_artifacts:
        keep = [
            complete_pair_artifacts(
                pair_root(
                    SurfaceConfiguration(
                        alpha=float(row.alpha),
                        tau_min=float(row.tau_min),
                        tau_max=float(row.tau_max),
                    )
                )
            )
            for row in frame.itertuples(index=False)
        ]
        frame = frame.loc[keep].copy()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checkpoint, index=False)
    completed = {
        SurfaceConfiguration(
            alpha=float(row.alpha),
            tau_min=float(row.tau_min),
            tau_max=float(row.tau_max),
        )
        for row in frame.itertuples(index=False)
    }

    def execute(configuration: SurfaceConfiguration) -> dict[str, object]:
        temporary = pair_root(configuration)
        if temporary.exists():
            shutil.rmtree(temporary)
        started = time.perf_counter()
        pair = _run_pair(
            experiment="mouse",
            endpoint=MOUSE_ENDPOINT,
            configuration=configuration,
            condition="natural_mismatch",
            candidates=candidates,
            source_priors=source_priors,
            target_priors=target_priors,
            truth=truth,
            method_root=temporary,
        )
        if retain_fit_artifacts and not complete_pair_artifacts(temporary):
            missing = [
                f"{variant}/{filename}"
                for variant in VARIANTS
                for filename in RETAINED_FIT_FILES
                if not (temporary / variant / filename).is_file()
            ]
            raise ValueError(
                f"Retained rho-attribution pair lacks artifacts {missing}: "
                f"{temporary}"
            )
        pair["paired_runtime_seconds"] = time.perf_counter() - started
        return {
            **identity,
            "alpha": configuration.alpha,
            "alpha_ratio": configuration.alpha / MOUSE_ENDPOINT.selected_alpha,
            "tau_min": configuration.tau_min,
            "tau_max": configuration.tau_max,
            "mean_matched_tau": mean_matched_tau(
                source_priors, configuration.tau_min, configuration.tau_max
            ),
            **pair,
            "_temporary": temporary,
        }

    remaining = sorted(configured - completed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(execute, configuration): configuration for configuration in remaining}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            temporary = Path(row.pop("_temporary"))
            new_row = pd.DataFrame([row], columns=_checkpoint_columns())
            frame = (
                new_row
                if frame.empty
                else pd.concat([frame, new_row], ignore_index=True)
            ).sort_values(["alpha", "tau_min", "tau_max"])
            frame.to_csv(checkpoint, index=False)
            if not retain_fit_artifacts:
                shutil.rmtree(temporary)
            print(
                f"completed mouse/{analysis_stage}: {futures[future]}",
                flush=True,
            )
    summary = summarize_surface(frame, expected_splits=1)
    summary["discovery_support"] = (
        summary["all_heterogeneous_converged"].astype(bool)
        & summary["all_uniform_converged"].astype(bool)
        & summary["delta_auprc_heterogeneous_minus_uniform_mean"].gt(0.0)
    )
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    by_split_path = tables / f"{analysis_stage}_by_dataset.csv"
    summary_path = tables / f"{analysis_stage}_summary.csv"
    frame.to_csv(by_split_path, index=False)
    summary.to_csv(summary_path, index=False)
    return frame, summary


def run_mouse_coarse(
    *,
    project_root: Path,
    design_manifest: Path,
    output_root: Path,
    jobs: int,
    configurations: Sequence[SurfaceConfiguration] | None = None,
    retain_fit_artifacts: bool = False,
) -> tuple[Path, Path]:
    tau_values, alpha_values, full_configurations = _mouse_configurations(
        design_manifest
    )
    if configurations is None:
        active_configurations = full_configurations
        active_tau_values = tau_values
        active_alpha_values = alpha_values
    else:
        active_configurations = tuple(sorted(set(configurations)))
        if not active_configurations:
            raise ValueError("Mouse coarse configurations must be nonempty")
        if not set(active_configurations) <= set(full_configurations):
            raise ValueError("Mouse coarse configurations are outside the locked design")
        active_tau_values = tuple(
            sorted(
                {
                    value
                    for configuration in active_configurations
                    for value in (configuration.tau_min, configuration.tau_max)
                }
            )
        )
        active_alpha_values = tuple(
            sorted({configuration.alpha for configuration in active_configurations})
        )
    _, summary = _run_mouse_surface(
        project_root=project_root,
        design_manifest=design_manifest,
        output_root=output_root,
        analysis_stage="coarse",
        configurations=active_configurations,
        jobs=jobs,
        retain_fit_artifacts=retain_fit_artifacts,
    )
    _render_relative_heatmaps(
        summary=summary,
        endpoints=(MOUSE_ENDPOINT,),
        tau_values=active_tau_values,
        alpha_values={MOUSE_ENDPOINT.endpoint: active_alpha_values},
        path=output_root / "figures/coarse_relative_ap.png",
    )
    return (
        output_root / "tables/coarse_by_dataset.csv",
        output_root / "tables/coarse_summary.csv",
    )


def prepare_mouse_refinement(
    *, design_manifest: Path, output_root: Path
) -> Path:
    coarse_summary = output_root / "tables/coarse_summary.csv"
    if not coarse_summary.is_file():
        raise FileNotFoundError(f"Missing mouse coarse summary: {coarse_summary}")
    summary = pd.read_csv(coarse_summary, float_precision="round_trip")
    center = _best_off_diagonal(summary, MOUSE_ENDPOINT.endpoint)
    tau_values, alpha_values, coarse = _mouse_configurations(design_manifest)
    refined = refinement_configurations(
        center=center,
        tau_values=tau_values,
        alpha_values=alpha_values,
        coarse=coarse,
    )
    payload = {
        "stage": "mouse_exploratory_local_refinement",
        "locked_before_refinement_execution": True,
        "cohort_manifest_sha256": _design_hash(design_manifest),
        "coarse_summary_sha256": sha256_file(coarse_summary),
        "center": {
            "alpha": center.alpha,
            "tau_min": center.tau_min,
            "tau_max": center.tau_max,
        },
        "configurations": [
            {
                "alpha": configuration.alpha,
                "tau_min": configuration.tau_min,
                "tau_max": configuration.tau_max,
            }
            for configuration in refined
        ],
    }
    path = output_root / "design/refinement_grid.yaml"
    if path.is_file():
        existing = _load_yaml_mapping(path)
        if existing != payload:
            existing_design = dict(existing)
            current_design = dict(payload)
            existing_design.pop("coarse_summary_sha256", None)
            current_design.pop("coarse_summary_sha256", None)
            if existing_design != current_design:
                raise ValueError(
                    f"Refusing to overwrite mouse refinement design: {path}"
                )
        return path
    _write_yaml(path, payload)
    return path


def write_mouse_exploratory_selection(*, output_root: Path) -> Path:
    coarse_path = output_root / "tables/coarse_summary.csv"
    refinement_path = output_root / "tables/refinement_summary.csv"
    refinement_design_path = output_root / "design/refinement_grid.yaml"
    missing = [
        path
        for path in (coarse_path, refinement_path, refinement_design_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing mouse selection inputs: {missing}")
    coarse = pd.read_csv(coarse_path, float_precision="round_trip")
    refinement = pd.read_csv(refinement_path, float_precision="round_trip")
    combined = pd.concat((coarse, refinement), ignore_index=True)
    converged = combined.loc[
        ~np.isclose(combined["tau_min"], combined["tau_max"])
        & combined["all_heterogeneous_converged"].map(_as_bool)
        & combined["all_uniform_converged"].map(_as_bool)
    ].copy()
    if converged.empty:
        raise ValueError("Mouse coarse/refinement union has no converged off-diagonal cell")
    metric = "delta_auprc_heterogeneous_minus_uniform_mean"
    selected = converged.sort_values(
        [metric, "analysis_stage", "alpha", "tau_min", "tau_max"],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    coarse_off_diagonal = coarse.loc[
        ~np.isclose(coarse["tau_min"], coarse["tau_max"])
    ]
    refinement_off_diagonal = refinement.loc[
        ~np.isclose(refinement["tau_min"], refinement["tau_max"])
    ]
    refinement_design = _load_yaml_mapping(refinement_design_path)
    delta = float(selected[metric])
    payload: dict[str, object] = {
        "status": (
            "positive_exploratory_point_estimate"
            if delta > 0.0
            else "no_positive_exploratory_point_estimate"
        ),
        "scope": "single_dataset_mouse_spleen_natural_mismatch",
        "claim_limit": (
            "Exploratory point estimate only; no biological replicates or "
            "donor-separated confirmation."
        ),
        "selection_rule": (
            "Largest converged off-diagonal absolute AP difference over the "
            "locked coarse/refinement union."
        ),
        "source_tables": {
            "coarse": str(coarse_path),
            "coarse_sha256": sha256_file(coarse_path),
            "refinement": str(refinement_path),
            "refinement_sha256": sha256_file(refinement_path),
        },
        "locked_refinement_design": {
            "path": str(refinement_design_path),
            "sha256": sha256_file(refinement_design_path),
            "pre_refinement_coarse_summary_sha256": str(
                refinement_design["coarse_summary_sha256"]
            ),
            "regeneration_note": (
                "The locked raw coarse-summary hash is retained. When a "
                "regenerated CSV hash differs after checkpoint round-tripping, "
                "the recomputed center and configuration list must match the "
                "locked design exactly; the design file is not overwritten."
            ),
        },
        "surface_counts": {
            "coarse_off_diagonal": int(len(coarse_off_diagonal)),
            "coarse_positive": int(
                (coarse_off_diagonal[metric] > 0.0).sum()
            ),
            "refinement_off_diagonal": int(len(refinement_off_diagonal)),
            "refinement_positive": int(
                (refinement_off_diagonal[metric] > 0.0).sum()
            ),
        },
        "selected": {
            "analysis_stage": str(selected["analysis_stage"]),
            "endpoint": str(selected["endpoint"]),
            "alpha": float(selected["alpha"]),
            "tau_min": float(selected["tau_min"]),
            "tau_max": float(selected["tau_max"]),
            "heterogeneous_auprc": float(selected["heterogeneous_auprc_mean"]),
            "uniform_auprc": float(selected["uniform_auprc_mean"]),
            "delta_auprc_heterogeneous_minus_uniform": delta,
            "relative_auprc_heterogeneous_minus_uniform_percent": float(
                selected[
                    "relative_auprc_heterogeneous_minus_uniform_percent_mean"
                ]
            ),
        },
    }
    path = output_root / "design/exploratory_selection.yaml"
    if path.is_file() and _load_yaml_mapping(path) != payload:
        raise ValueError(f"Refusing to overwrite mouse exploratory selection: {path}")
    _write_yaml(path, payload)
    return path


def run_mouse_refinement(
    *,
    project_root: Path,
    design_manifest: Path,
    output_root: Path,
    jobs: int,
) -> tuple[Path, Path]:
    refinement = prepare_mouse_refinement(
        design_manifest=design_manifest, output_root=output_root
    )
    payload = _load_yaml_mapping(refinement)
    raw_configurations = payload.get("configurations")
    if not isinstance(raw_configurations, list):
        raise ValueError("Mouse refinement configurations must be a list")
    configurations = tuple(
        SurfaceConfiguration(
            alpha=float(configuration["alpha"]),
            tau_min=float(configuration["tau_min"]),
            tau_max=float(configuration["tau_max"]),
        )
        for configuration in raw_configurations
        if isinstance(configuration, dict)
    )
    _run_mouse_surface(
        project_root=project_root,
        design_manifest=design_manifest,
        output_root=output_root,
        analysis_stage="refinement",
        configurations=configurations,
        jobs=jobs,
    )
    write_mouse_exploratory_selection(output_root=output_root)
    return (
        output_root / "tables/refinement_by_dataset.csv",
        output_root / "tables/refinement_summary.csv",
    )


def _parse_configuration(args: argparse.Namespace) -> SurfaceConfiguration | None:
    supplied = (args.alpha, args.tau_min, args.tau_max)
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise ValueError("--alpha, --tau-min, and --tau-max must be supplied together")
    return SurfaceConfiguration(
        alpha=float(args.alpha),
        tau_min=float(args.tau_min),
        tau_max=float(args.tau_max),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run donor-separated discovery and locked confirmation for the "
            "paired matchability-informed versus mean-matched-uniform comparison."
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "validate-design",
            "prepare-hiha",
            "run-hiha-coarse",
            "run-hiha-alpha0-fine",
            "run-hiha-alpha0-focused",
            "run-hiha-refinement",
            "lock-hiha",
            "run-hiha-confirmation",
            "run-mouse-coarse",
            "run-mouse-refinement",
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--design-manifest", type=Path, default=DEFAULT_DESIGN_MANIFEST
    )
    parser.add_argument("--hiha-root", type=Path, default=DEFAULT_HIHA_ROOT)
    parser.add_argument("--mouse-root", type=Path, default=DEFAULT_MOUSE_ROOT)
    parser.add_argument("--cohort", action="append", default=None)
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--tau-min", type=float, default=None)
    parser.add_argument("--tau-max", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    project_root = args.project_root.resolve()
    design_manifest = (
        args.design_manifest
        if args.design_manifest.is_absolute()
        else project_root / args.design_manifest
    )
    hiha_root = (
        args.hiha_root
        if args.hiha_root.is_absolute()
        else project_root / args.hiha_root
    )
    mouse_root = (
        args.mouse_root
        if args.mouse_root.is_absolute()
        else project_root / args.mouse_root
    )

    if args.command == "validate-design":
        validate_design_manifest(design_manifest, project_root=project_root)
        print(design_manifest)
    elif args.command == "prepare-hiha":
        for path in prepare_hiha_bases(
            project_root=project_root,
            design_manifest=design_manifest,
            analysis_root=hiha_root,
            cohorts=tuple(args.cohort or ("discovery", "confirmation")),
            endpoints=args.endpoint,
            seeds=args.seed,
            jobs=args.jobs,
        ):
            print(path)
    elif args.command == "run-hiha-coarse":
        result = run_hiha_coarse(
            project_root=project_root,
            design_manifest=design_manifest,
            analysis_root=hiha_root,
            endpoints=args.endpoint,
            seeds=args.seed,
            jobs=args.jobs,
            only_configuration=_parse_configuration(args),
        )
        if result is not None:
            for path in result:
                print(path)
    elif args.command == "run-hiha-refinement":
        for path in run_hiha_refinement(
            project_root=project_root,
            design_manifest=design_manifest,
            analysis_root=hiha_root,
            endpoints=args.endpoint,
            jobs=args.jobs,
        ):
            print(path)
    elif args.command == "run-hiha-alpha0-fine":
        for path in run_hiha_alpha0_fine(
            project_root=project_root,
            design_manifest=design_manifest,
            analysis_root=hiha_root,
            endpoints=args.endpoint,
            jobs=args.jobs,
        ):
            print(path)
    elif args.command == "run-hiha-alpha0-focused":
        for path in run_hiha_alpha0_focused(
            project_root=project_root,
            design_manifest=design_manifest,
            analysis_root=hiha_root,
            endpoints=args.endpoint,
            jobs=args.jobs,
        ):
            print(path)
    elif args.command == "lock-hiha":
        print(
            lock_hiha_selection(
                design_manifest=design_manifest, analysis_root=hiha_root
            )
        )
    elif args.command == "run-hiha-confirmation":
        for path in run_hiha_confirmation(
            project_root=project_root,
            design_manifest=design_manifest,
            analysis_root=hiha_root,
            jobs=args.jobs,
        ):
            print(path)
    elif args.command == "run-mouse-coarse":
        for path in run_mouse_coarse(
            project_root=project_root,
            design_manifest=design_manifest,
            output_root=mouse_root,
            jobs=args.jobs,
        ):
            print(path)
    elif args.command == "run-mouse-refinement":
        for path in run_mouse_refinement(
            project_root=project_root,
            design_manifest=design_manifest,
            output_root=mouse_root,
            jobs=args.jobs,
        ):
            print(path)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
