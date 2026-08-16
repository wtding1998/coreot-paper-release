from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from coreot.transport.runner import (  # noqa: E402
    _apply_provider_rho,
    _run_coreot_constant_tau,
    _run_coreot_full,
)
from experiments.component_ablation_surfaces import (  # noqa: E402
    CONDITION,
    METRICS,
    SEEDS,
    EndpointSpec,
    _input_paths,
    _pbmc_condition_map,
    alpha_values,
    evaluate_scores,
    selected_specs,
)
from experiments.mouse_spleen.component_ablation import (  # noqa: E402
    _evaluate as _evaluate_mouse,
)


METRIC_NAMES = tuple(metric for metric, _ in METRICS)
VARIANTS = ("heterogeneous", "mean_matched_uniform")
RETAINED_FIT_FILES = (
    "cell_transport_scores.parquet",
    "label_probabilities.npz",
    "method_params.yaml",
    "sparse_coupling.parquet",
    "transport_manifest.yaml",
)
MOUSE_SPEC = EndpointSpec(
    experiment="mouse",
    endpoint="Proliferating",
    run_prefix="mouse_spleen_natural_proliferating",
    candidate_set="mouse_spleen_provider_k100",
    tau_values=(2.0, 3.0, 4.0, 5.0, 6.0),
    selected_tau=(3.0, 5.0),
    selected_alpha=40.0,
    tau_target=8.0,
)


@dataclass(frozen=True)
class SearchPaths:
    root: Path
    by_replicate: Path
    summary: Path
    figure: Path
    manifest: Path


def _search_paths(project_root: Path, experiment: str) -> SearchPaths:
    result_root = {
        "mouse": project_root
        / "results/mouse_spleen_core_ot/natural_mismatch/sensitivity",
        "hiha": project_root / "results/HIHA_DC/sensitivity",
        "pbmc": project_root / "results/PBMC/sensitivity",
    }[experiment]
    root = result_root / "rho_attribution_alpha_search"
    return SearchPaths(
        root=root,
        by_replicate=root / "tables/rho_attribution_by_replicate.csv",
        summary=root / "tables/rho_attribution_summary.csv",
        figure=project_root
        / f"docs/figs/manuscript_fig_{experiment}_rho_attribution_alpha_search.png",
        manifest=root / "manifest.yaml",
    )


def _specs(experiment: str, endpoints: Sequence[str] | None = None):
    if experiment == "mouse":
        if endpoints and tuple(endpoints) != (MOUSE_SPEC.endpoint,):
            raise ValueError("Mouse search supports only Proliferating")
        return (MOUSE_SPEC,)
    return selected_specs(experiment, endpoints)


def _mouse_input_paths(project_root: Path) -> tuple[Path, dict[str, Path]]:
    run_root = (
        project_root
        / "results/mouse_spleen_core_ot/runs/mouse_spleen_natural_proliferating"
    )
    profile = run_root / "derived/natural_mismatch/prior_profiles/default"
    return run_root, {
        "candidates": run_root
        / "candidates/natural_mismatch/mouse_spleen_provider_k100/"
        "candidate_edges.parquet",
        "source_priors": profile / "source_priors.csv",
        "target_priors": profile / "target_priors.csv",
        "truth": run_root
        / "benchmark/natural_mismatch/evaluation_truth/query_truth.csv",
    }


def empirical_mass(priors: pd.DataFrame) -> np.ndarray:
    if "empirical_mass" not in priors:
        return np.full(len(priors), 1.0 / len(priors))
    mass = pd.to_numeric(priors["empirical_mass"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("empirical_mass must contain finite positive values")
    return mass / mass.sum()


def mean_matched_tau(
    source_priors: pd.DataFrame, tau_min: float, tau_max: float
) -> float:
    rho = pd.to_numeric(source_priors["rho"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(rho).all() or np.any((rho < 0.0) | (rho > 1.0)):
        raise ValueError("rho must contain finite values in [0, 1]")
    tau = tau_min + (tau_max - tau_min) * rho
    return float(np.sum(empirical_mass(source_priors) * tau))


def _checkpoint_columns() -> list[str]:
    columns = [
        "experiment",
        "endpoint",
        "replicate",
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
            [
                f"heterogeneous_{metric}",
                f"uniform_{metric}",
                f"delta_{metric}_heterogeneous_minus_uniform",
            ]
        )
    return columns


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _read_checkpoint(
    path: Path,
    identity: dict[str, object],
    configured_alpha: set[float],
    *,
    reset_on_identity_mismatch: bool = False,
) -> pd.DataFrame:
    columns = _checkpoint_columns()
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    if missing := sorted(set(columns) - set(frame.columns)):
        raise ValueError(f"{path} lacks checkpoint columns: {missing}")
    for column, expected in identity.items():
        observed = frame[column]
        if isinstance(expected, int | float):
            matches = np.isclose(observed.astype(float), float(expected)).all()
        else:
            matches = observed.astype(str).eq(str(expected)).all()
        if not matches:
            if reset_on_identity_mismatch:
                return pd.DataFrame(columns=columns)
            raise ValueError(f"{path} has incompatible {column}")
    frame = frame.loc[
        frame["alpha"].astype(float).isin(configured_alpha), columns
    ].copy()
    for column in ("heterogeneous_converged", "uniform_converged"):
        frame[column] = frame[column].map(_as_bool)
    return frame


def _value_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _pbmc_provider_rho_path(
    runs_root: Path, run_root: Path, spec: EndpointSpec
) -> Path:
    selected_run = (
        f"{run_root.name}_taumin{_value_slug(spec.selected_tau[0])}"
        f"_taumax{_value_slug(spec.selected_tau[1])}"
        f"_alpha{_value_slug(spec.selected_alpha)}_coreot_full"
    )
    return (
        runs_root
        / selected_run
        / "transport/provider_reliability/pca30/source_rho.csv"
    )


def _method_metadata(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    return {
        "converged": bool(metadata.get("converged", False)),
        "n_iterations": int(metadata.get("n_iter", 0)),
    }


def _run_method(
    *,
    variant: str,
    method_root: Path,
    condition: str,
    candidates: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    spec: EndpointSpec,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, object], float]:
    common: dict[str, object] = {
        "tau_target": spec.tau_target,
        "epsilon": 0.05,
        "alpha": alpha,
        "max_iter": 5000,
        "tol": 1.0e-6,
        "numerical_floor": 1.0e-300,
        "eta": 1.0e-12,
    }
    if variant == "heterogeneous":
        config = {
            **common,
            "name": "coreot_full",
            "tau_min": spec.selected_tau[0],
            "tau_max": spec.selected_tau[1],
        }
        runner = _run_coreot_full
    elif variant == "mean_matched_uniform":
        config = {
            **common,
            "name": "coreot_constant_tau",
            "tau_source": "matched_coreot_mean",
            "matched_tau_min": spec.selected_tau[0],
            "matched_tau_max": spec.selected_tau[1],
        }
        runner = _run_coreot_constant_tau
    else:
        raise ValueError(f"Unknown variant: {variant}")
    method_root.mkdir(parents=True, exist_ok=True)
    (method_root / "method_params.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    started = time.perf_counter()
    runner(
        method_root,
        condition,
        candidates,
        source_priors,
        target_priors,
        config,
    )
    runtime = time.perf_counter() - started
    scores = pd.read_parquet(method_root / "cell_transport_scores.parquet")
    metadata = _method_metadata(method_root / "transport_manifest.yaml")
    return scores, metadata, runtime


def _evaluate_variant(
    *,
    experiment: str,
    endpoint: str,
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    pbmc_condition: pd.Series | None,
) -> dict[str, object]:
    if experiment == "mouse":
        metrics = _evaluate_mouse(scores=scores, truth=truth, endpoint=endpoint)
        return {
            "evaluation_scope": metrics["evaluation_scope"],
            "n_detection": metrics["n_query"],
            "n_positive": metrics["n_positive"],
            **{metric: metrics[metric] for metric in METRIC_NAMES},
        }
    return evaluate_scores(
        experiment=experiment,
        endpoint=endpoint,
        truth=truth,
        scores=scores,
        pbmc_condition=pbmc_condition,
    )


def _run_replicate(
    *,
    project_root: Path,
    runs_root: Path,
    paths: SearchPaths,
    spec: EndpointSpec,
    seed: int,
    pbmc_condition: pd.Series | None,
) -> Path:
    if spec.experiment == "mouse":
        run_root, inputs = _mouse_input_paths(project_root)
        condition = "natural_mismatch"
        replicate = "single_dataset"
    else:
        run_root = runs_root / spec.run_prefix.format(seed=seed)
        inputs = _input_paths(run_root, spec.candidate_set)
        condition = CONDITION
        replicate = f"seed{seed}"
    missing = [path for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing rho-attribution inputs: {', '.join(map(str, missing))}"
        )
    candidates = pd.read_parquet(inputs["candidates"])
    source_priors = pd.read_csv(inputs["source_priors"])
    target_priors = pd.read_csv(inputs["target_priors"])
    truth = pd.read_csv(inputs["truth"])
    source_hash = sha256_file(inputs["source_priors"])
    if spec.experiment == "pbmc":
        provider_rho_path = _pbmc_provider_rho_path(runs_root, run_root, spec)
        if not provider_rho_path.is_file():
            raise FileNotFoundError(
                f"Missing calibrated PBMC provider reliability: {provider_rho_path}"
            )
        provider_rho = pd.read_csv(provider_rho_path)
        rho_metadata_path = provider_rho_path.with_name("rho_metadata.yaml")
        rho_metadata = yaml.safe_load(
            rho_metadata_path.read_text(encoding="utf-8")
        )
        provider_rho["coreot_rho_cell_id_hash"] = str(
            rho_metadata["cell_id_rho_sha256"]
        )
        source_priors = _apply_provider_rho(
            source_priors, provider_rho, condition
        )
        source_hash = (
            f"{source_hash}+{sha256_file(provider_rho_path)}"
            f"+{sha256_file(rho_metadata_path)}"
        )
    tau_mean = mean_matched_tau(
        source_priors, spec.selected_tau[0], spec.selected_tau[1]
    )
    candidate_hash = sha256_file(inputs["candidates"])
    configured_alpha = set(alpha_values(spec.selected_alpha))
    checkpoint = paths.root / "checkpoints" / spec.slug / f"{replicate}.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "experiment": spec.experiment,
        "endpoint": spec.endpoint,
        "replicate": replicate,
        "seed": seed,
        "run_id": run_root.name,
        "tau_min": spec.selected_tau[0],
        "tau_max": spec.selected_tau[1],
        "mean_matched_tau": tau_mean,
        "tau_target": spec.tau_target,
        "epsilon": 0.05,
        "max_iterations": 5000,
        "tolerance": 1.0e-6,
        "numerical_floor": 1.0e-300,
        "candidate_edges_sha256": candidate_hash,
        "source_priors_sha256": source_hash,
    }
    frame = _read_checkpoint(
        checkpoint,
        identity,
        configured_alpha,
        reset_on_identity_mismatch=spec.experiment == "pbmc",
    )
    frame.to_csv(checkpoint, index=False)
    completed = set(frame["alpha"].astype(float))
    for alpha in sorted(configured_alpha):
        if alpha in completed:
            continue
        metrics: dict[str, dict[str, object]] = {}
        metadata: dict[str, dict[str, object]] = {}
        runtime: dict[str, float] = {}
        for variant in VARIANTS:
            method_root = (
                paths.root
                / "tmp"
                / spec.slug
                / replicate
                / f"alpha_{alpha:g}"
                / variant
            )
            scores, metadata[variant], runtime[variant] = _run_method(
                variant=variant,
                method_root=method_root,
                condition=condition,
                candidates=candidates,
                source_priors=source_priors,
                target_priors=target_priors,
                spec=spec,
                alpha=alpha,
            )
            metrics[variant] = _evaluate_variant(
                experiment=spec.experiment,
                endpoint=spec.endpoint,
                truth=truth,
                scores=scores,
                pbmc_condition=pbmc_condition,
            )
        row: dict[str, object] = {
            **identity,
            "alpha": alpha,
            "alpha_ratio": (
                alpha / spec.selected_alpha
                if spec.selected_alpha > 0
                else float("nan")
            ),
            "heterogeneous_converged": metadata["heterogeneous"]["converged"],
            "heterogeneous_n_iterations": metadata["heterogeneous"][
                "n_iterations"
            ],
            "heterogeneous_runtime_seconds": runtime["heterogeneous"],
            "uniform_converged": metadata["mean_matched_uniform"]["converged"],
            "uniform_n_iterations": metadata["mean_matched_uniform"][
                "n_iterations"
            ],
            "uniform_runtime_seconds": runtime["mean_matched_uniform"],
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
        new_row = pd.DataFrame([row], columns=_checkpoint_columns())
        frame = (
            new_row
            if frame.empty
            else pd.concat([frame, new_row], ignore_index=True)
        ).sort_values("alpha")
        frame.to_csv(checkpoint, index=False)
        shutil.rmtree(
            paths.root
            / "tmp"
            / spec.slug
            / replicate
            / f"alpha_{alpha:g}"
        )
    return checkpoint


def summarize_results(by_replicate: pd.DataFrame) -> pd.DataFrame:
    group = ["experiment", "endpoint", "alpha", "alpha_ratio"]
    aggregations: dict[str, tuple[str, str]] = {
        "n_replicates": ("replicate", "nunique"),
        "all_heterogeneous_converged": ("heterogeneous_converged", "all"),
        "all_uniform_converged": ("uniform_converged", "all"),
        "max_n_iterations": ("heterogeneous_n_iterations", "max"),
        "tau_min": ("tau_min", "first"),
        "tau_max": ("tau_max", "first"),
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
    summary = (
        by_replicate.groupby(group, sort=False, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    summary["replicated_ap_support"] = (
        summary["n_replicates"].eq(5)
        & summary[
            "delta_auprc_heterogeneous_minus_uniform_n_positive"
        ].ge(4)
        & summary["delta_auprc_heterogeneous_minus_uniform_mean"].gt(0.0)
    )
    summary["exploratory_point_ap_advantage"] = (
        summary["delta_auprc_heterogeneous_minus_uniform_mean"].gt(0.0)
    )
    return summary


def _render(summary: pd.DataFrame, by_replicate: pd.DataFrame, path: Path) -> None:
    available_endpoints = set(summary["endpoint"].astype(str))
    pbmc_order = ("B cells", "NK cells", "Dendritic cells")
    endpoints = (
        [endpoint for endpoint in pbmc_order if endpoint in available_endpoints]
        if summary["experiment"].astype(str).eq("pbmc").all()
        else list(summary["endpoint"].drop_duplicates())
    )
    is_mouse_single_endpoint = (
        len(endpoints) == 1
        and summary["experiment"].astype(str).eq("mouse").all()
    )
    figure, axes = plt.subplots(
        1, len(endpoints), figsize=(4.7 * len(endpoints), 3.8), squeeze=False
    )
    for index, endpoint in enumerate(endpoints):
        axis = axes[0, index]
        local_summary = summary.loc[summary["endpoint"].eq(endpoint)].sort_values(
            "alpha_ratio"
        )
        local = by_replicate.loc[by_replicate["endpoint"].eq(endpoint)]
        for replicate_index, (_, replicate) in enumerate(
            local.groupby("replicate", sort=True)
        ):
            axis.plot(
                replicate["alpha_ratio"],
                replicate["delta_auprc_heterogeneous_minus_uniform"],
                color="#999999",
                alpha=0.45,
                linewidth=1.0,
                label=(
                    "Donor split"
                    if replicate_index == 0 and len(local["replicate"].unique()) > 1
                    else None
                ),
            )
        axis.plot(
            local_summary["alpha_ratio"],
            local_summary[
                "delta_auprc_heterogeneous_minus_uniform_mean"
            ],
            color="#D55E00",
            marker="o",
            linewidth=2.0,
            label=(
                "Donor-equal mean"
                if len(local["replicate"].unique()) > 1
                else "Point estimate"
            ),
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        supported = local_summary["replicated_ap_support"]
        if supported.any():
            axis.scatter(
                local_summary.loc[supported, "alpha_ratio"],
                local_summary.loc[
                    supported,
                    "delta_auprc_heterogeneous_minus_uniform_mean",
                ],
                marker="*",
                s=100,
                color="#2166ac",
                zorder=5,
                label="Replicated support",
            )
        if not is_mouse_single_endpoint:
            axis.set_title(endpoint)
        axis.set_xlabel(r"$\alpha/\alpha^\star$")
        if index == 0:
            axis.set_ylabel("AP difference")
        axis.grid(alpha=0.2, linewidth=0.6)
        if not is_mouse_single_endpoint:
            axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_derived_artifacts(
    *, paths: SearchPaths, by_replicate: pd.DataFrame
) -> None:
    summary = summarize_results(by_replicate)
    summary.to_csv(paths.summary, index=False)
    _render(summary, by_replicate, paths.figure)
    available_endpoints = set(summary["endpoint"].astype(str))
    pbmc_order = ("B cells", "NK cells", "Dendritic cells")
    endpoint_order = (
        [endpoint for endpoint in pbmc_order if endpoint in available_endpoints]
        if summary["experiment"].astype(str).eq("pbmc").all()
        else list(summary["endpoint"].drop_duplicates())
    )
    manifest = {
        "design": "paired_mean_matched_alpha_grid",
        "comparison": "heterogeneous_minus_mean_matched_uniform",
        "alpha_ratios": [0.0, 0.5, 1.0, 1.5, 2.0],
        "support_rule": (
            "positive AP difference on at least four of five splits and "
            "positive arithmetic-mean AP difference"
        ),
        "exploratory": True,
        "visualization": {
            "endpoint_order": endpoint_order,
            "primary_metric": "absolute_AP_difference",
            "replicate_curves": "gray",
            "aggregate_curve": "orange",
            "aggregate_label": (
                "Donor-equal mean"
                if by_replicate["replicate"].nunique() > 1
                else "Point estimate"
            ),
            "replicated_support_marker": "blue_star",
            "overall_title": False,
        },
        "artifacts": {
            "by_replicate": str(paths.by_replicate),
            "summary": str(paths.summary),
            "figure": str(paths.figure),
        },
    }
    paths.manifest.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def aggregate_existing_search(
    *, project_root: Path, experiment: str
) -> SearchPaths:
    """Regenerate search summaries and displays from saved replicate results."""
    paths = _search_paths(project_root, experiment)
    if not paths.by_replicate.is_file():
        raise FileNotFoundError(
            f"Missing attribution search table: {paths.by_replicate}"
        )
    by_replicate = pd.read_csv(paths.by_replicate)
    if not by_replicate["experiment"].astype(str).eq(experiment).all():
        raise ValueError(
            f"{paths.by_replicate} contains rows outside {experiment}"
        )
    _write_derived_artifacts(paths=paths, by_replicate=by_replicate)
    return paths


def run_search(
    *,
    project_root: Path,
    experiment: str,
    endpoints: Sequence[str] | None,
    seeds: Sequence[int],
    jobs: int,
    runs_root: Path,
    pbmc_raw_path: Path,
) -> SearchPaths:
    specs = _specs(experiment, endpoints)
    paths = _search_paths(project_root, experiment)
    pbmc_condition = (
        _pbmc_condition_map(pbmc_raw_path) if experiment == "pbmc" else None
    )
    tasks = [
        (spec, 0 if experiment == "mouse" else int(seed))
        for spec in specs
        for seed in ((0,) if experiment == "mouse" else seeds)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_replicate,
                project_root=project_root,
                runs_root=runs_root,
                paths=paths,
                spec=spec,
                seed=seed,
                pbmc_condition=pbmc_condition,
            ): (spec.endpoint, seed)
            for spec, seed in tasks
        }
        checkpoints = []
        for future in concurrent.futures.as_completed(futures):
            endpoint, seed = futures[future]
            checkpoints.append(future.result())
            print(f"completed {experiment}: {endpoint}, replicate {seed}", flush=True)
    frames = [pd.read_csv(path) for path in sorted(checkpoints)]
    by_replicate = pd.concat(frames, ignore_index=True)
    expected = len(tasks) * 5
    if len(by_replicate) != expected:
        raise ValueError(
            f"{experiment} search has {len(by_replicate)} rows; expected {expected}"
        )
    paths.by_replicate.parent.mkdir(parents=True, exist_ok=True)
    by_replicate.to_csv(paths.by_replicate, index=False)
    _write_derived_artifacts(paths=paths, by_replicate=by_replicate)
    if (paths.root / "tmp").is_dir() and not any((paths.root / "tmp").rglob("*")):
        shutil.rmtree(paths.root / "tmp")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired heterogeneous-versus-mean-matched-uniform query "
            "penalty comparisons over the predefined alpha grid."
        )
    )
    parser.add_argument(
        "--experiment", choices=("mouse", "hiha", "pbmc", "all"), default="all"
    )
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--pbmc-raw-path",
        type=Path,
        default=Path("data/raw/kang_2018.h5ad"),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Regenerate summaries, figures, and manifests from existing "
            "per-replicate tables without running model fits."
        ),
    )
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
        os.environ.setdefault(variable, "1")
    experiments = (
        ("mouse", "hiha", "pbmc")
        if args.experiment == "all"
        else (args.experiment,)
    )
    if args.endpoint and len(experiments) > 1:
        raise ValueError("--endpoint requires one --experiment")
    if args.aggregate_only and (args.endpoint or args.seed):
        raise ValueError("--aggregate-only does not accept --endpoint or --seed")
    for experiment in experiments:
        if args.aggregate_only:
            paths = aggregate_existing_search(
                project_root=args.project_root.resolve(),
                experiment=experiment,
            )
        else:
            paths = run_search(
                project_root=args.project_root.resolve(),
                experiment=experiment,
                endpoints=args.endpoint,
                seeds=tuple(args.seed or SEEDS),
                jobs=args.jobs,
                runs_root=args.runs_root.resolve(),
                pbmc_raw_path=args.pbmc_raw_path.resolve(),
            )
        print(paths.summary)
        print(paths.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
