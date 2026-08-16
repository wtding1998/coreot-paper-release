from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Callable, Sequence

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from coreot.artifacts.hashes import sha256_file
from coreot.evaluation.metrics import safe_auprc, safe_auroc
from coreot.transport.runner import (
    _apply_provider_rho,
    _run_coreot_constant_tau,
    _run_coreot_match_only,
)


CONDITION = "incomplete_reference"
SEEDS = (1, 2, 3, 4, 5)
METRICS = (
    ("auroc", "$u$-based AUROC"),
    ("auprc", "$u$-based AP"),
    ("forced_accuracy", "Forced accuracy"),
    ("forced_macro_f1", "Forced macro-F1"),
)
HIHA_COMPATIBILITY_METRICS = (
    ("auprc", "AP"),
    ("auroc", "AUROC"),
    ("forced_accuracy", "Forced accuracy"),
    ("forced_macro_f1", "Forced macro-F1"),
)
HIHA_COMPATIBILITY_TAU_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)
VARIANTS = ("match_only", "compatibility_only")
VARIANT_METHOD = {
    "match_only": "coreot_match_only",
    "compatibility_only": "coreot_constant_tau",
}
VARIANT_DISPLAY = {
    "match_only": r"Heterogeneous-query-penalty sensitivity at $\alpha=0$",
    "compatibility_only": "Constant-$\\tau_q$ compatibility sensitivity",
}


@dataclass(frozen=True)
class EndpointSpec:
    experiment: str
    endpoint: str
    run_prefix: str
    candidate_set: str
    tau_values: tuple[float, ...]
    selected_tau: tuple[float, float]
    selected_alpha: float
    tau_target: float

    @property
    def slug(self) -> str:
        return self.endpoint.lower().replace("+", "").replace("-", "").replace(" ", "_")


SPECS = (
    EndpointSpec(
        "hiha",
        "HLA-DRhi cDC2",
        "hiha_dc_hladrhi_cdc2_seed{seed}_report_leave_one_HIHA_DC",
        "hiha_harmony30_k100",
        (2.0, 2.5, 3.0, 4.0, 5.0),
        (2.5, 3.0),
        2.0,
        2.0,
    ),
    EndpointSpec(
        "hiha",
        "ISG+ cDC2",
        "hiha_dc_isg_cdc2_seed{seed}_report_leave_one_HIHA_DC",
        "hiha_harmony30_k100",
        (2.0, 2.5, 3.0, 4.0, 5.0),
        (2.5, 4.0),
        1.0,
        3.0,
    ),
    EndpointSpec(
        "pbmc",
        "B cells",
        "pbmc_ifnb_b_cells_stim_seed{seed}",
        "pca30_k100",
        (0.5, 0.75, 1.0, 1.25, 1.5),
        (0.5, 1.0),
        4.0,
        1.0,
    ),
    EndpointSpec(
        "pbmc",
        "NK cells",
        "pbmc_ifnb_nk_cells_stim_seed{seed}",
        "pca30_k100",
        (0.5, 0.75, 1.0, 1.25, 1.5),
        (0.5, 1.0),
        2.0,
        1.0,
    ),
    EndpointSpec(
        "pbmc",
        "Dendritic cells",
        "pbmc_ifnb_dendritic_cells_stim_seed{seed}",
        "pca30_k100",
        (0.5, 0.75, 1.0, 1.25, 1.5),
        (0.75, 1.0),
        3.0,
        1.0,
    ),
)


def tau_pairs(values: Sequence[float]) -> tuple[tuple[float, float], ...]:
    ordered = tuple(sorted({float(value) for value in values}))
    return tuple((low, high) for low in ordered for high in ordered if low <= high)


def alpha_values(selected_alpha: float) -> tuple[float, ...]:
    return tuple(selected_alpha * multiplier for multiplier in (0.0, 0.5, 1.0, 1.5, 2.0))


def compatibility_tau_values(spec: EndpointSpec) -> tuple[float, ...]:
    if spec.experiment == "hiha":
        return tuple(sorted(set(spec.tau_values) | set(HIHA_COMPATIBILITY_TAU_VALUES)))
    return spec.tau_values


def _value_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _forced_metrics(joined: pd.DataFrame) -> tuple[float, float]:
    shared = joined.loc[joined["is_shared_state"].astype(bool)].copy()
    shared = shared.loc[shared["forced_label"].notna() & shared["forced_label"].astype(str).ne("")]
    if shared.empty:
        return float("nan"), float("nan")
    labels = sorted(shared["true_label"].astype(str).unique())
    return (
        float(accuracy_score(shared["true_label"], shared["forced_label"])),
        float(
            f1_score(
                shared["true_label"],
                shared["forced_label"],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    )


def evaluate_scores(
    *,
    experiment: str,
    endpoint: str,
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    pbmc_condition: pd.Series | None = None,
) -> dict[str, float | int | str]:
    required_truth = {
        "cell_id",
        "true_label",
        "is_absent_state",
        "is_shared_state",
    }
    required_scores = {"cell_id", "u", "forced_label"}
    if missing := sorted(required_truth - set(truth.columns)):
        raise ValueError(f"Truth lacks required columns: {missing}")
    if missing := sorted(required_scores - set(scores.columns)):
        raise ValueError(f"Scores lack required columns: {missing}")
    joined = truth.loc[:, sorted(required_truth)].merge(
        scores.loc[:, sorted(required_scores)],
        on="cell_id",
        validate="one_to_one",
    )
    if len(joined) != len(truth) or len(joined) != len(scores):
        raise ValueError("Truth and score cell sets do not match exactly")

    if experiment == "hiha":
        detection = joined["true_label"].astype(str).str.endswith("cDC2")
        scope = "within_cdc2"
    elif experiment == "pbmc":
        if pbmc_condition is None:
            raise ValueError("PBMC evaluation requires cell-level condition metadata")
        condition = joined["cell_id"].map(pbmc_condition)
        if condition.isna().any():
            raise ValueError("PBMC condition metadata does not cover every query cell")
        detection = joined["true_label"].astype(str).eq(endpoint) & condition.isin(("ctrl", "stim"))
        scope = "within_celltype"
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    local = joined.loc[detection]
    positive = local["is_absent_state"].astype(bool)
    if positive.nunique() != 2:
        raise ValueError(f"Detection cohort lacks both classes for {endpoint}")
    forced_accuracy, forced_macro_f1 = _forced_metrics(joined)
    return {
        "evaluation_scope": scope,
        "n_detection": int(len(local)),
        "n_positive": int(positive.sum()),
        "auroc": safe_auroc(positive, local["u"]),
        "auprc": safe_auprc(positive, local["u"]),
        "forced_accuracy": forced_accuracy,
        "forced_macro_f1": forced_macro_f1,
    }


def _pbmc_condition_map(path: Path) -> pd.Series:
    data = ad.read_h5ad(path, backed="r")
    try:
        column = data.obs["label"].astype(str)
        normalized = column.str.lower().map(
            {
                "ctrl": "ctrl",
                "control": "ctrl",
                "unstim": "ctrl",
                "unstimulated": "ctrl",
                "untreated": "ctrl",
                "stim": "stim",
                "stimulated": "stim",
                "ifn-beta": "stim",
                "ifnb": "stim",
                "ifn_beta": "stim",
            }
        )
        if normalized.isna().any():
            unknown = sorted(column.loc[normalized.isna()].unique())
            raise ValueError(f"Unknown PBMC condition values: {unknown}")
        return pd.Series(normalized.to_numpy(), index=data.obs_names.astype(str))
    finally:
        data.file.close()


def _input_paths(run_root: Path, candidate_set: str) -> dict[str, Path]:
    profile = run_root / f"derived/{CONDITION}/prior_profiles/default"
    return {
        "candidates": run_root / f"candidates/{CONDITION}/{candidate_set}/candidate_edges.parquet",
        "source_priors": profile / "source_priors.csv",
        "target_priors": profile / "target_priors.csv",
        "truth": run_root / f"benchmark/{CONDITION}/evaluation_truth/query_truth.csv",
    }


def _checkpoint_columns(variant: str) -> list[str]:
    parameters = ["tau_min", "tau_max"] if variant == "match_only" else ["tau_source", "alpha"]
    return [
        "experiment",
        "endpoint",
        "seed",
        "run_id",
        "variant",
        "method",
        *parameters,
        "tau_target",
        "epsilon",
        "max_iterations",
        "tolerance",
        "numerical_floor",
        "converged",
        "n_iterations",
        "runtime_seconds",
        "evaluation_scope",
        "n_detection",
        "n_positive",
        *[metric for metric, _ in METRICS],
        "candidate_edges_sha256",
        "source_priors_sha256",
    ]


def _read_checkpoint(
    path: Path,
    *,
    columns: list[str],
    identity: dict[str, object],
    reset_on_identity_mismatch: bool = False,
) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    if missing := sorted(set(columns) - set(frame.columns)):
        raise ValueError(f"{path} lacks checkpoint columns: {missing}")
    for column, expected in identity.items():
        observed = frame[column]
        if isinstance(expected, (int, float)):
            matched = np.isclose(observed.astype(float), float(expected)).all()
        else:
            matched = observed.astype(str).eq(str(expected)).all()
        if not matched:
            if reset_on_identity_mismatch:
                return pd.DataFrame(columns=columns)
            raise ValueError(f"{path} has incompatible {column}")
    frame["converged"] = (
        frame["converged"]
        if pd.api.types.is_bool_dtype(frame["converged"])
        else frame["converged"].astype(str).str.lower().eq("true")
    )
    return frame.loc[:, columns]


def _run_variant(
    *,
    spec: EndpointSpec,
    seed: int,
    run_root: Path,
    output_root: Path,
    variant: str,
    candidates: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    truth: pd.DataFrame,
    pbmc_condition: pd.Series | None,
    candidate_hash: str,
    source_hash: str,
) -> Path:
    method = VARIANT_METHOD[variant]
    if variant == "match_only":
        parameter_names = ("tau_min", "tau_max")
        combinations = tau_pairs(spec.tau_values)
        runner: Callable[..., None] = _run_coreot_match_only
    else:
        parameter_names = ("tau_source", "alpha")
        combinations = tuple(
            (tau, alpha)
            for tau in compatibility_tau_values(spec)
            for alpha in alpha_values(spec.selected_alpha)
        )
        runner = _run_coreot_constant_tau

    checkpoint_path = output_root / spec.slug / f"seed{seed}" / f"{variant}.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _checkpoint_columns(variant)
    identity = {
        "experiment": spec.experiment,
        "endpoint": spec.endpoint,
        "seed": seed,
        "run_id": run_root.name,
        "variant": variant,
        "method": method,
        "tau_target": spec.tau_target,
        "epsilon": 0.05,
        "max_iterations": 5000,
        "tolerance": 1.0e-6,
        "numerical_floor": 1.0e-300,
        "candidate_edges_sha256": candidate_hash,
        "source_priors_sha256": source_hash,
    }
    frame = _read_checkpoint(
        checkpoint_path,
        columns=columns,
        identity=identity,
        reset_on_identity_mismatch=spec.experiment == "pbmc",
    )
    configured = set(combinations)
    first, second = parameter_names
    frame = frame.loc[
        [
            (float(getattr(row, first)), float(getattr(row, second))) in configured
            for row in frame.itertuples(index=False)
        ]
    ].copy()
    frame.to_csv(checkpoint_path, index=False)
    completed = {
        (float(getattr(row, first)), float(getattr(row, second)))
        for row in frame.itertuples(index=False)
    }

    for first_value, second_value in combinations:
        if (first_value, second_value) in completed:
            continue
        temporary = (
            output_root
            / "tmp"
            / spec.slug
            / f"seed{seed}"
            / variant
            / f"{first}_{first_value:g}_{second}_{second_value:g}"
        )
        temporary.mkdir(parents=True, exist_ok=True)
        method_config: dict[str, object] = {
            "name": method,
            "tau_target": spec.tau_target,
            "epsilon": 0.05,
            "max_iter": 5000,
            "tol": 1.0e-6,
            "numerical_floor": 1.0e-300,
            "eta": 1.0e-12,
        }
        if variant == "match_only":
            method_config.update(
                {
                    "tau_min": first_value,
                    "tau_max": second_value,
                    "alpha": 0.0,
                }
            )
        else:
            method_config.update({"tau_source": first_value, "alpha": second_value})
        started = time.perf_counter()
        runner(
            temporary,
            CONDITION,
            candidates,
            source_priors,
            target_priors,
            method_config,
        )
        runtime = time.perf_counter() - started
        scores = pd.read_parquet(temporary / "cell_transport_scores.parquet")
        manifest = yaml.safe_load(
            (temporary / "transport_manifest.yaml").read_text(encoding="utf-8")
        )
        metadata = manifest.get("metadata", {})
        row = {
            **identity,
            first: first_value,
            second: second_value,
            "converged": bool(metadata.get("converged", False)),
            "n_iterations": int(metadata.get("n_iter", 0)),
            "runtime_seconds": runtime,
            **evaluate_scores(
                experiment=spec.experiment,
                endpoint=spec.endpoint,
                truth=truth,
                scores=scores,
                pbmc_condition=pbmc_condition,
            ),
        }
        new_row = pd.DataFrame([row], columns=columns)
        frame = (
            new_row if frame.empty else pd.concat([frame, new_row], ignore_index=True)
        ).sort_values([first, second])
        frame.to_csv(checkpoint_path, index=False)
        shutil.rmtree(temporary)
    return checkpoint_path


def _run_endpoint_seed(
    *,
    spec: EndpointSpec,
    seed: int,
    runs_root: Path,
    output_root: Path,
    pbmc_condition: pd.Series | None,
) -> tuple[Path, Path]:
    run_root = runs_root / spec.run_prefix.format(seed=seed)
    paths = _input_paths(run_root, spec.candidate_set)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Component-ablation inputs are missing for "
            f"{run_root}: {', '.join(str(path) for path in missing)}"
        )
    candidates = pd.read_parquet(paths["candidates"])
    source_priors = pd.read_csv(paths["source_priors"])
    target_priors = pd.read_csv(paths["target_priors"])
    truth = pd.read_csv(paths["truth"])
    candidate_hash = sha256_file(paths["candidates"])
    source_hash = sha256_file(paths["source_priors"])
    if spec.experiment == "pbmc":
        selected_run = (
            f"{run_root.name}_taumin{_value_slug(spec.selected_tau[0])}"
            f"_taumax{_value_slug(spec.selected_tau[1])}"
            f"_alpha{_value_slug(spec.selected_alpha)}_coreot_full"
        )
        provider_root = runs_root / selected_run / "transport/provider_reliability/pca30"
        provider_path = provider_root / "source_rho.csv"
        metadata_path = provider_root / "rho_metadata.yaml"
        if not provider_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing calibrated PBMC provider reliability under {provider_root}"
            )
        provider_rho = pd.read_csv(provider_path)
        provider_metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        provider_rho["coreot_rho_cell_id_hash"] = str(provider_metadata["cell_id_rho_sha256"])
        source_priors = _apply_provider_rho(source_priors, provider_rho, CONDITION)
        source_hash = f"{source_hash}+{sha256_file(provider_path)}+{sha256_file(metadata_path)}"
    return tuple(
        _run_variant(
            spec=spec,
            seed=seed,
            run_root=run_root,
            output_root=output_root,
            variant=variant,
            candidates=candidates,
            source_priors=source_priors,
            target_priors=target_priors,
            truth=truth,
            pbmc_condition=pbmc_condition,
            candidate_hash=candidate_hash,
            source_hash=source_hash,
        )
        for variant in VARIANTS
    )


def selected_specs(
    experiment: str, endpoints: Sequence[str] | None = None
) -> tuple[EndpointSpec, ...]:
    requested = set(endpoints or ())
    specs = tuple(
        spec
        for spec in SPECS
        if spec.experiment == experiment and (not requested or spec.endpoint in requested)
    )
    if not specs:
        known = [spec.endpoint for spec in SPECS if spec.experiment == experiment]
        raise ValueError(f"No endpoints selected; known {experiment} endpoints: {known}")
    return specs


def run_surfaces(
    *,
    experiment: str,
    endpoints: Sequence[str] | None,
    seeds: Sequence[int],
    jobs: int,
    runs_root: Path,
    output_root: Path,
    pbmc_raw_path: Path,
) -> None:
    specs = selected_specs(experiment, endpoints)
    pbmc_condition = _pbmc_condition_map(pbmc_raw_path) if experiment == "pbmc" else None
    tasks = [(spec, int(seed)) for spec in specs for seed in seeds]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_endpoint_seed,
                spec=spec,
                seed=seed,
                runs_root=runs_root,
                output_root=output_root,
                pbmc_condition=pbmc_condition,
            ): (spec.endpoint, seed)
            for spec, seed in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            endpoint, seed = futures[future]
            future.result()
            print(f"completed {experiment}: {endpoint}, seed {seed}", flush=True)


def collect_results(
    *,
    experiment: str,
    endpoints: Sequence[str] | None,
    seeds: Sequence[int],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for spec in selected_specs(experiment, endpoints):
        for seed in seeds:
            for variant in VARIANTS:
                path = output_root / spec.slug / f"seed{seed}" / f"{variant}.csv"
                frame = pd.read_csv(path)
                expected = (
                    len(tau_pairs(spec.tau_values))
                    if variant == "match_only"
                    else len(compatibility_tau_values(spec))
                    * len(alpha_values(spec.selected_alpha))
                )
                if len(frame) != expected:
                    raise ValueError(f"{path} has {len(frame)} rows; expected {expected}")
                frames.append(frame)
    by_seed = _retained_aggregate_rows(
        pd.concat(frames, ignore_index=True),
        experiment=experiment,
    )
    parameter_columns = ["tau_min", "tau_max", "tau_source", "alpha"]
    for column in parameter_columns:
        if column not in by_seed:
            by_seed[column] = np.nan
    group_columns = [
        "experiment",
        "endpoint",
        "variant",
        "method",
        *parameter_columns,
    ]
    summary = (
        by_seed.groupby(group_columns, dropna=False, sort=False)
        .agg(
            n_splits=("seed", "nunique"),
            all_converged=("converged", "all"),
            max_n_iterations=("n_iterations", "max"),
            **{f"{metric}_mean": (metric, "mean") for metric, _ in METRICS},
            **{f"{metric}_sd": (metric, "std") for metric, _ in METRICS},
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(len(tuple(seeds))).all():
        raise ValueError("Component-ablation summary has incomplete split coverage")
    return by_seed, summary


def _retained_aggregate_rows(
    frame: pd.DataFrame,
    *,
    experiment: str,
) -> pd.DataFrame:
    if experiment != "hiha":
        return frame.copy()
    archived_compatibility = frame["variant"].eq("compatibility_only") & ~frame[
        "tau_source"
    ].isin(HIHA_COMPATIBILITY_TAU_VALUES)
    return frame.loc[~archived_compatibility].copy()


def _limits(values: np.ndarray) -> tuple[float, float]:
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    if np.isclose(lower, upper):
        return 0.0, 1.0
    return lower, upper


def _display_metric(
    *,
    summary: pd.DataFrame,
    endpoint: str,
    variant: str,
    row_name: str,
    column_name: str,
    row_values: Sequence[float],
    column_values: Sequence[float],
    metric: str,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    scoped = summary.loc[
        summary["endpoint"].eq(endpoint) & summary["variant"].eq(variant)
    ]
    pivot = scoped.pivot(
        index=row_name,
        columns=column_name,
        values=f"{metric}_mean",
    ).reindex(index=row_values, columns=column_values)
    return pivot, _limits(pivot.to_numpy(dtype=float))


def _render_variant(
    *,
    experiment: str,
    specs: tuple[EndpointSpec, ...],
    summary: pd.DataFrame,
    variant: str,
    output_path: Path,
) -> None:
    display_metrics = HIHA_COMPATIBILITY_METRICS if variant == "compatibility_only" else METRICS
    figure, axes = plt.subplots(
        len(display_metrics),
        len(specs),
        figsize=(4.8 * len(specs), 3.6 * len(display_metrics)),
        squeeze=False,
    )
    for column_index, spec in enumerate(specs):
        endpoint = summary.loc[
            summary["endpoint"].eq(spec.endpoint) & summary["variant"].eq(variant)
        ]
        if variant == "match_only":
            row_name, column_name = "tau_min", "tau_max"
            row_values = column_values = list(spec.tau_values)
        else:
            row_name, column_name = "tau_source", "alpha"
            row_values = list(
                HIHA_COMPATIBILITY_TAU_VALUES
                if experiment == "hiha"
                else spec.tau_values
            )
            column_values = list(alpha_values(spec.selected_alpha))
        convergence = endpoint.pivot(
            index=row_name, columns=column_name, values="all_converged"
        ).reindex(index=row_values, columns=column_values)
        for row_index, (metric, metric_label) in enumerate(display_metrics):
            axis = axes[row_index, column_index]
            pivot, (lower, upper) = _display_metric(
                summary=summary,
                endpoint=spec.endpoint,
                variant=variant,
                row_name=row_name,
                column_name=column_name,
                row_values=row_values,
                column_values=column_values,
                metric=metric,
            )
            image = axis.imshow(
                pivot.to_numpy(dtype=float),
                origin="lower",
                aspect="auto",
                cmap="RdBu_r" if variant == "compatibility_only" else "viridis",
                vmin=lower,
                vmax=upper,
            )
            axis.set_xticks(
                range(len(column_values)),
                [f"{value:g}" for value in column_values],
            )
            axis.set_yticks(
                range(len(row_values)),
                [f"{value:g}" for value in row_values],
            )
            if variant == "match_only":
                axis.set_xlabel(r"$\tau_{\max}$")
            else:
                axis.set_xlabel(r"$\alpha$")
            parameter_label = r"$\tau_{\min}$" if variant == "match_only" else r"$\tau_q$"
            axis.set_ylabel(
                f"{metric_label}\n{parameter_label}" if column_index == 0 else parameter_label
            )
            if row_index == 0:
                axis.set_title(spec.endpoint)
            threshold = (lower + upper) / 2.0
            for matrix_row, tau_value in enumerate(row_values):
                for matrix_column, parameter_value in enumerate(column_values):
                    value = pivot.loc[tau_value, parameter_value]
                    if pd.isna(value):
                        continue
                    if variant == "compatibility_only":
                        red, green, blue, _ = image.cmap(image.norm(float(value)))
                        text_color = (
                            "white"
                            if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                            else "black"
                        )
                    else:
                        text_color = "white" if float(value) < threshold else "black"
                    axis.text(
                        matrix_column,
                        matrix_row,
                        f"{float(value):.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=text_color,
                    )
                    converged = convergence.loc[tau_value, parameter_value]
                    if not pd.isna(converged) and not bool(converged):
                        axis.text(
                            matrix_column,
                            matrix_row + 0.27,
                            "×",
                            color="red",
                            fontsize=11,
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )
            if variant == "match_only":
                selected_row = row_values.index(spec.selected_tau[0])
                selected_column = column_values.index(spec.selected_tau[1])
                axis.add_patch(
                    Rectangle(
                        (selected_column - 0.48, selected_row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                    )
                )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if variant == "match_only":
        figure.suptitle(
            VARIANT_DISPLAY[variant],
            fontsize=14,
            y=0.995,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.965))
    else:
        figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def aggregate_surfaces(
    *,
    experiment: str,
    endpoints: Sequence[str] | None,
    seeds: Sequence[int],
    output_root: Path,
    docs_fig_root: Path,
) -> tuple[Path, Path, Path, Path]:
    specs = selected_specs(experiment, endpoints)
    by_seed, summary = collect_results(
        experiment=experiment,
        endpoints=endpoints,
        seeds=seeds,
        output_root=output_root,
    )
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    by_seed_path = tables / "component_ablation_by_seed.csv"
    summary_path = tables / "component_ablation_summary.csv"
    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    match_path = docs_fig_root / f"manuscript_fig_{experiment}_component_minus_c.png"
    compatibility_filename = (
        "manuscript_fig_hiha_compatibility_sensitivity.png"
        if experiment == "hiha"
        else f"manuscript_fig_{experiment}_component_minus_m.png"
    )
    compatibility_path = docs_fig_root / compatibility_filename
    _render_variant(
        experiment=experiment,
        specs=specs,
        summary=summary,
        variant="match_only",
        output_path=match_path,
    )
    _render_variant(
        experiment=experiment,
        specs=specs,
        summary=summary,
        variant="compatibility_only",
        output_path=compatibility_path,
    )
    return by_seed_path, summary_path, match_path, compatibility_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and aggregate cross-experiment CoRe-OT component-ablation surfaces."
    )
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--experiment", choices=("hiha", "pbmc"), required=True)
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--pbmc-raw-path", type=Path, default=Path("data/raw/kang_2018.h5ad"))
    parser.add_argument("--docs-fig-root", type=Path, default=Path("docs/figs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root or Path(
        f"results/{'HIHA_DC' if args.experiment == 'hiha' else 'PBMC'}"
        "/sensitivity/component_ablation"
    )
    seeds = tuple(args.seed or SEEDS)
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if args.command == "run":
        run_surfaces(
            experiment=args.experiment,
            endpoints=args.endpoint,
            seeds=seeds,
            jobs=args.jobs,
            runs_root=args.runs_root,
            output_root=output_root,
            pbmc_raw_path=args.pbmc_raw_path,
        )
    else:
        artifacts = aggregate_surfaces(
            experiment=args.experiment,
            endpoints=args.endpoint,
            seeds=seeds,
            output_root=output_root,
            docs_fig_root=args.docs_fig_root,
        )
        for artifact in artifacts:
            print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
