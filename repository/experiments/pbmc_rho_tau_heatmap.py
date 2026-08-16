from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import replace
from pathlib import Path
import shutil
import sys
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from coreot.transport.runner import _apply_provider_rho  # noqa: E402
from experiments.component_ablation_surfaces import (  # noqa: E402
    CONDITION,
    METRICS,
    SEEDS,
    _input_paths,
    _pbmc_condition_map,
    selected_specs,
    tau_pairs,
)
from experiments.rho_attribution_search import (  # noqa: E402
    _evaluate_variant,
    _pbmc_provider_rho_path,
    _run_method,
    mean_matched_tau,
)


TAU_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5)
EXPANDED_TAU_VALUES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FOCUSED_TAU_VALUES = (0.75, 1.0, 1.25, 1.5, 1.75)
DENDRITIC_FINE_TAU_VALUES = (
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
    1.625,
    1.75,
)
METRIC_NAMES = tuple(metric for metric, _ in METRICS)
S2_ALIGNED_METRICS = (
    ("auprc", "AP"),
    ("auroc", "AUROC"),
    ("forced_accuracy", "Forced accuracy"),
    ("forced_macro_f1", "Forced macro-F1"),
)


def relative_percent(delta: float, comparator: float) -> float:
    if comparator <= 0.0:
        raise ValueError("Relative effect requires a positive comparator")
    return 100.0 * delta / comparator


def _load_inputs(
    *,
    runs_root: Path,
    spec,
    seed: int,
) -> tuple[
    Path,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    str,
]:
    run_root = runs_root / spec.run_prefix.format(seed=seed)
    paths = _input_paths(run_root, spec.candidate_set)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing PBMC tau-surface inputs: {', '.join(map(str, missing))}"
        )
    candidates = pd.read_parquet(paths["candidates"])
    source_priors = pd.read_csv(paths["source_priors"])
    target_priors = pd.read_csv(paths["target_priors"])
    truth = pd.read_csv(paths["truth"])
    provider_path = _pbmc_provider_rho_path(runs_root, run_root, spec)
    metadata_path = provider_path.with_name("rho_metadata.yaml")
    provider_rho = pd.read_csv(provider_path)
    provider_metadata = yaml.safe_load(
        metadata_path.read_text(encoding="utf-8")
    )
    provider_rho["coreot_rho_cell_id_hash"] = str(
        provider_metadata["cell_id_rho_sha256"]
    )
    source_priors = _apply_provider_rho(
        source_priors, provider_rho, CONDITION
    )
    source_hash = (
        f"{sha256_file(paths['source_priors'])}+{sha256_file(provider_path)}"
        f"+{sha256_file(metadata_path)}"
    )
    return (
        run_root,
        candidates,
        source_priors,
        target_priors,
        truth,
        sha256_file(paths["candidates"]),
        source_hash,
    )


def _columns() -> list[str]:
    columns = [
        "endpoint",
        "seed",
        "run_id",
        "tau_min",
        "tau_max",
        "mean_matched_tau",
        "alpha",
        "tau_target",
        "candidate_edges_sha256",
        "source_priors_sha256",
        "uniform_converged",
        "uniform_n_iterations",
    ]
    columns.extend(f"uniform_{metric}" for metric in METRIC_NAMES)
    return columns


def _read_checkpoint(
    path: Path, identity: dict[str, object], configured: set[tuple[float, float]]
) -> pd.DataFrame:
    columns = _columns()
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
            raise ValueError(f"{path} has incompatible {column}")
    keep = [
        (float(row.tau_min), float(row.tau_max)) in configured
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[keep, columns].copy()


def _run_seed(
    *,
    runs_root: Path,
    output_root: Path,
    spec,
    seed: int,
    pbmc_condition: pd.Series,
) -> Path:
    (
        run_root,
        candidates,
        source_priors,
        target_priors,
        truth,
        candidate_hash,
        source_hash,
    ) = _load_inputs(runs_root=runs_root, spec=spec, seed=seed)
    configured = set(tau_pairs(TAU_VALUES))
    checkpoint = output_root / "checkpoints" / spec.slug / f"seed{seed}.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "endpoint": spec.endpoint,
        "seed": seed,
        "run_id": run_root.name,
        "alpha": 0.0,
        "tau_target": spec.tau_target,
        "candidate_edges_sha256": candidate_hash,
        "source_priors_sha256": source_hash,
    }
    frame = _read_checkpoint(checkpoint, identity, configured)
    frame.to_csv(checkpoint, index=False)
    completed = {
        (float(row.tau_min), float(row.tau_max))
        for row in frame.itertuples(index=False)
    }
    for tau_min, tau_max in sorted(configured):
        if (tau_min, tau_max) in completed:
            continue
        local_spec = replace(spec, selected_tau=(tau_min, tau_max))
        method_root = (
            output_root
            / "tmp"
            / spec.slug
            / f"seed{seed}"
            / f"tau_min_{tau_min:g}_tau_max_{tau_max:g}"
        )
        scores, metadata, _ = _run_method(
            variant="mean_matched_uniform",
            method_root=method_root,
            condition=CONDITION,
            candidates=candidates,
            source_priors=source_priors,
            target_priors=target_priors,
            spec=local_spec,
            alpha=0.0,
        )
        metrics = _evaluate_variant(
            experiment="pbmc",
            endpoint=spec.endpoint,
            truth=truth,
            scores=scores,
            pbmc_condition=pbmc_condition,
        )
        row = {
            **identity,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "mean_matched_tau": mean_matched_tau(
                source_priors, tau_min, tau_max
            ),
            "uniform_converged": metadata["converged"],
            "uniform_n_iterations": metadata["n_iterations"],
            **{
                f"uniform_{metric}": metrics[metric]
                for metric in METRIC_NAMES
            },
        }
        new_row = pd.DataFrame([row], columns=_columns())
        frame = (
            new_row
            if frame.empty
            else pd.concat([frame, new_row], ignore_index=True)
        ).sort_values(["tau_min", "tau_max"])
        frame.to_csv(checkpoint, index=False)
        shutil.rmtree(method_root)
    return checkpoint


def pair_with_heterogeneous(
    uniform: pd.DataFrame, heterogeneous: pd.DataFrame
) -> pd.DataFrame:
    selected = heterogeneous.loc[
        heterogeneous["variant"].eq("match_only"),
        [
            "endpoint",
            "seed",
            "tau_min",
            "tau_max",
            "converged",
            "n_iterations",
            *METRIC_NAMES,
        ],
    ].rename(
        columns={
            "converged": "heterogeneous_converged",
            "n_iterations": "heterogeneous_n_iterations",
            **{
                metric: f"heterogeneous_{metric}"
                for metric in METRIC_NAMES
            },
        }
    )
    paired = uniform.merge(
        selected,
        on=["endpoint", "seed", "tau_min", "tau_max"],
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(uniform):
        raise ValueError("Heterogeneous surface does not cover every uniform fit")
    for metric in METRIC_NAMES:
        delta = (
            paired[f"heterogeneous_{metric}"]
            - paired[f"uniform_{metric}"]
        )
        paired[f"delta_{metric}_heterogeneous_minus_uniform"] = delta
        paired[
            f"relative_{metric}_heterogeneous_minus_uniform_percent"
        ] = [
            relative_percent(value, comparator)
            for value, comparator in zip(
                delta, paired[f"uniform_{metric}"], strict=True
            )
        ]
    diagonal = np.isclose(paired["tau_min"], paired["tau_max"])
    diagonal_metrics = [
        f"delta_{metric}_heterogeneous_minus_uniform"
        for metric in METRIC_NAMES
    ]
    maximum = paired.loc[diagonal, diagonal_metrics].abs().to_numpy().max()
    if maximum > 1.0e-12:
        raise ValueError(
            f"Diagonal heterogeneous/uniform identity failed: {maximum}"
        )
    paired["diagonal_identity_verified"] = diagonal
    return paired


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_splits": ("seed", "nunique"),
        "all_heterogeneous_converged": (
            "heterogeneous_converged",
            "all",
        ),
        "all_uniform_converged": ("uniform_converged", "all"),
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
        relative = (
            f"relative_{metric}_heterogeneous_minus_uniform_percent"
        )
        aggregations[f"{delta}_mean"] = (delta, "mean")
        aggregations[f"{delta}_sd"] = (delta, "std")
        aggregations[f"{delta}_n_positive"] = (
            delta,
            lambda values: int((values > 0.0).sum()),
        )
        aggregations[f"{relative}_mean"] = (relative, "mean")
        aggregations[f"{relative}_sd"] = (relative, "std")
    return (
        paired.groupby(
            ["endpoint", "tau_min", "tau_max"],
            sort=False,
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
    )


def _symmetric_limit(values: np.ndarray) -> float:
    limit = float(np.nanmax(np.abs(values)))
    return limit if limit > 0.0 else 1.0


def render_heatmap(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    tau_values: Sequence[float] = TAU_VALUES,
) -> None:
    tau_values = tuple(float(value) for value in tau_values)
    specs = selected_specs("pbmc")
    available = set(summary["endpoint"].astype(str))
    endpoints = [spec.endpoint for spec in specs if spec.endpoint in available]
    selected_tau = {spec.endpoint: spec.selected_tau for spec in specs}
    dense_grid = len(tau_values) >= 9
    absolute_scale = 1000.0 if dense_grid else 1.0
    rows = (
        (
            "delta_auprc_heterogeneous_minus_uniform_mean",
            (
                r"Absolute AP difference ($\times 10^{-3}$)"
                if dense_grid
                else "Absolute AP difference"
            ),
            ".2f" if dense_grid else ".3f",
            absolute_scale,
        ),
        (
            "relative_auprc_heterogeneous_minus_uniform_percent_mean",
            "Relative AP difference (%)",
            ".2f",
            1.0,
        ),
    )
    limits = {
        column: _symmetric_limit(
            summary[column].to_numpy(dtype=float) * scale
        )
        for column, _, _, scale in rows
    }
    figure, axes = plt.subplots(
        2,
        len(endpoints),
        figsize=(
            max(10.0 if len(tau_values) >= 9 else 7.0, 4.8 * len(endpoints)),
            8.0,
        ),
        squeeze=False,
    )
    for row_index, (column, label, number_format, scale) in enumerate(rows):
        norm = TwoSlopeNorm(
            vmin=-limits[column], vcenter=0.0, vmax=limits[column]
        )
        for column_index, endpoint in enumerate(endpoints):
            axis = axes[row_index, column_index]
            local = summary.loc[summary["endpoint"].eq(endpoint)]
            pivot = local.pivot(
                index="tau_min", columns="tau_max", values=column
            ).reindex(index=tau_values, columns=tau_values) * scale
            image = axis.imshow(
                pivot.to_numpy(dtype=float),
                origin="lower",
                aspect="equal",
                cmap="RdBu_r",
                norm=norm,
            )
            for i, tau_min in enumerate(tau_values):
                for j, tau_max in enumerate(tau_values):
                    value = pivot.loc[tau_min, tau_max]
                    if pd.isna(value):
                        continue
                    axis.text(
                        j,
                        i,
                        format(float(value), number_format),
                        ha="center",
                        va="center",
                        fontsize=7 if dense_grid else 8,
                        color=(
                            "white"
                            if abs(float(value)) > 0.55 * limits[column]
                            else "black"
                        ),
                    )
            if (
                selected_tau[endpoint][0] in tau_values
                and selected_tau[endpoint][1] in tau_values
            ):
                selected_row = tau_values.index(selected_tau[endpoint][0])
                selected_column = tau_values.index(selected_tau[endpoint][1])
                axis.add_patch(
                    Rectangle(
                        (selected_column - 0.48, selected_row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#542788",
                        linewidth=2.0,
                    )
                )
            axis.set_xticks(
                range(len(tau_values)),
                [f"{value:g}" for value in tau_values],
                rotation=45 if len(tau_values) >= 9 else 0,
                ha="right" if len(tau_values) >= 9 else "center",
            )
            axis.set_yticks(
                range(len(tau_values)),
                [f"{value:g}" for value in tau_values],
            )
            axis.set_xlabel(r"$\tau_{\max}$")
            axis.set_ylabel(f"{label}\n" + r"$\tau_{\min}$")
            if row_index == 0:
                axis.set_title(endpoint)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        r"PBMC paired $\rho_i$ attribution at $\alpha=0$: "
        r"heterogeneous $\tau_{q,i}$ minus mean-matched uniform $\tau_q$",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def render_s2_aligned_heatmap(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    tau_values: Sequence[float],
) -> None:
    tau_values = tuple(float(value) for value in tau_values)
    endpoints = [spec.endpoint for spec in selected_specs("pbmc")]
    matrices = {
        (metric, endpoint): (
            summary.loc[summary["endpoint"].eq(endpoint)]
            .pivot(
                index="tau_min",
                columns="tau_max",
                values=(
                    f"relative_{metric}_heterogeneous_minus_uniform_percent_mean"
                ),
            )
            .reindex(index=tau_values, columns=tau_values)
            .astype(float)
        )
        for metric, _ in S2_ALIGNED_METRICS
        for endpoint in endpoints
    }
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    figure, axes = plt.subplots(
        len(S2_ALIGNED_METRICS),
        len(endpoints),
        figsize=(4.2 * len(endpoints), 12.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    labels = [f"{value:g}" for value in tau_values]
    for row_index, (metric, metric_label) in enumerate(S2_ALIGNED_METRICS):
        limit = max(
            _symmetric_limit(matrices[(metric, endpoint)].to_numpy(dtype=float))
            for endpoint in endpoints
        )
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = None
        for column_index, endpoint in enumerate(endpoints):
            axis = axes[row_index, column_index]
            matrix = matrices[(metric, endpoint)]
            image = axis.imshow(
                matrix.to_numpy(dtype=float),
                origin="lower",
                aspect="equal",
                cmap=cmap,
                norm=norm,
            )
            for i, tau_min in enumerate(tau_values):
                for j, tau_max in enumerate(tau_values):
                    value = matrix.loc[tau_min, tau_max]
                    if pd.isna(value):
                        continue
                    red, green, blue, _ = image.cmap(image.norm(float(value)))
                    axis.text(
                        j,
                        i,
                        f"{float(value):.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=(
                            "white"
                            if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                            else "black"
                        ),
                    )
            axis.set_xticks(range(len(tau_values)), labels)
            axis.tick_params(axis="x", labelbottom=True)
            axis.set_yticks(range(len(tau_values)), labels)
            axis.set_xlabel(r"$\tau_{\max}$")
            axis.set_ylabel(
                f"Relative {metric_label} difference (%)\n$\\tau_{{\\min}}$"
                if column_index == 0
                else r"$\tau_{\min}$"
            )
            if row_index == 0:
                axis.set_title(endpoint, fontsize=10, fontweight="bold")
        if image is not None:
            figure.colorbar(
                image,
                ax=axes[row_index, :],
                location="right",
                fraction=0.046,
                pad=0.03,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def _paired_columns() -> list[str]:
    columns = [
        "endpoint",
        "seed",
        "run_id",
        "tau_min",
        "tau_max",
        "mean_matched_tau",
        "alpha",
        "tau_target",
        "candidate_edges_sha256",
        "source_priors_sha256",
        "heterogeneous_converged",
        "heterogeneous_n_iterations",
        "uniform_converged",
        "uniform_n_iterations",
    ]
    for metric in METRIC_NAMES:
        columns.extend(
            [
                f"heterogeneous_{metric}",
                f"uniform_{metric}",
                f"delta_{metric}_heterogeneous_minus_uniform",
                f"relative_{metric}_heterogeneous_minus_uniform_percent",
            ]
        )
    return columns


def _read_paired_checkpoint(
    path: Path,
    identity: dict[str, object],
    configured: set[tuple[float, float]],
) -> pd.DataFrame:
    columns = _paired_columns()
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    if missing := sorted(set(columns) - set(frame.columns)):
        raise ValueError(f"{path} lacks paired checkpoint columns: {missing}")
    for column, expected in identity.items():
        observed = frame[column]
        if isinstance(expected, int | float):
            matches = np.isclose(observed.astype(float), float(expected)).all()
        else:
            matches = observed.astype(str).eq(str(expected)).all()
        if not matches:
            raise ValueError(f"{path} has incompatible {column}")
    keep = [
        (float(row.tau_min), float(row.tau_max)) in configured
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[keep, columns].copy()


def _run_direct_seed(
    *,
    runs_root: Path,
    output_root: Path,
    spec,
    seed: int,
    pbmc_condition: pd.Series,
    tau_values: Sequence[float],
) -> Path:
    (
        run_root,
        candidates,
        source_priors,
        target_priors,
        truth,
        candidate_hash,
        source_hash,
    ) = _load_inputs(runs_root=runs_root, spec=spec, seed=seed)
    configured = set(tau_pairs(tau_values))
    checkpoint = output_root / "checkpoints" / spec.slug / f"seed{seed}.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "endpoint": spec.endpoint,
        "seed": seed,
        "run_id": run_root.name,
        "alpha": 0.0,
        "tau_target": spec.tau_target,
        "candidate_edges_sha256": candidate_hash,
        "source_priors_sha256": source_hash,
    }
    frame = _read_paired_checkpoint(checkpoint, identity, configured)
    frame.to_csv(checkpoint, index=False)
    completed = {
        (float(row.tau_min), float(row.tau_max))
        for row in frame.itertuples(index=False)
    }
    for tau_min, tau_max in sorted(configured):
        if (tau_min, tau_max) in completed:
            continue
        local_spec = replace(spec, selected_tau=(tau_min, tau_max))
        result: dict[str, dict[str, object]] = {}
        metadata: dict[str, dict[str, object]] = {}
        for variant in ("heterogeneous", "mean_matched_uniform"):
            method_root = (
                output_root
                / "tmp"
                / spec.slug
                / f"seed{seed}"
                / f"tau_min_{tau_min:g}_tau_max_{tau_max:g}"
                / variant
            )
            scores, metadata[variant], _ = _run_method(
                variant=variant,
                method_root=method_root,
                condition=CONDITION,
                candidates=candidates,
                source_priors=source_priors,
                target_priors=target_priors,
                spec=local_spec,
                alpha=0.0,
            )
            result[variant] = _evaluate_variant(
                experiment="pbmc",
                endpoint=spec.endpoint,
                truth=truth,
                scores=scores,
                pbmc_condition=pbmc_condition,
            )
        row: dict[str, object] = {
            **identity,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "mean_matched_tau": mean_matched_tau(
                source_priors, tau_min, tau_max
            ),
            "heterogeneous_converged": metadata["heterogeneous"][
                "converged"
            ],
            "heterogeneous_n_iterations": metadata["heterogeneous"][
                "n_iterations"
            ],
            "uniform_converged": metadata["mean_matched_uniform"][
                "converged"
            ],
            "uniform_n_iterations": metadata["mean_matched_uniform"][
                "n_iterations"
            ],
        }
        for metric in METRIC_NAMES:
            heterogeneous = float(result["heterogeneous"][metric])
            uniform = float(result["mean_matched_uniform"][metric])
            delta = heterogeneous - uniform
            row[f"heterogeneous_{metric}"] = heterogeneous
            row[f"uniform_{metric}"] = uniform
            row[f"delta_{metric}_heterogeneous_minus_uniform"] = delta
            row[
                f"relative_{metric}_heterogeneous_minus_uniform_percent"
            ] = relative_percent(delta, uniform)
        new_row = pd.DataFrame([row], columns=_paired_columns())
        frame = (
            new_row
            if frame.empty
            else pd.concat([frame, new_row], ignore_index=True)
        ).sort_values(["tau_min", "tau_max"])
        frame.to_csv(checkpoint, index=False)
        shutil.rmtree(
            output_root
            / "tmp"
            / spec.slug
            / f"seed{seed}"
            / f"tau_min_{tau_min:g}_tau_max_{tau_max:g}"
        )
    return checkpoint


def _run_direct_grid(
    *,
    project_root: Path,
    runs_root: Path,
    jobs: int,
    pbmc_raw_path: Path,
    specs: Sequence,
    tau_values: Sequence[float],
    output_name: str,
    table_stem: str,
    figure_name: str,
    progress_label: str,
) -> tuple[Path, Path, Path]:
    output_root = (
        project_root
        / "results/PBMC/sensitivity"
        / output_name
    )
    condition = _pbmc_condition_map(pbmc_raw_path)
    tasks = [
        (spec, seed)
        for spec in specs
        for seed in SEEDS
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_direct_seed,
                runs_root=runs_root,
                output_root=output_root,
                spec=spec,
                seed=seed,
                pbmc_condition=condition,
                tau_values=tau_values,
            ): (spec.endpoint, seed)
            for spec, seed in tasks
        }
        checkpoints = []
        for future in concurrent.futures.as_completed(futures):
            endpoint, seed = futures[future]
            checkpoints.append(future.result())
            print(
                f"completed {progress_label}: {endpoint}, seed {seed}",
                flush=True,
            )
    paired = pd.concat(
        [pd.read_csv(path) for path in sorted(checkpoints)],
        ignore_index=True,
    )
    expected = len(tasks) * len(tau_pairs(tau_values))
    if len(paired) != expected:
        raise ValueError(
            f"Direct paired grid has {len(paired)} rows; expected {expected}"
        )
    diagonal = np.isclose(paired["tau_min"], paired["tau_max"])
    delta_columns = [
        f"delta_{metric}_heterogeneous_minus_uniform"
        for metric in METRIC_NAMES
    ]
    maximum = paired.loc[diagonal, delta_columns].abs().to_numpy().max()
    if maximum > 1.0e-12:
        raise ValueError(f"Direct-grid diagonal identity failed: {maximum}")
    paired["diagonal_identity_verified"] = diagonal
    summary = summarize(paired)
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paired_path = tables / f"{table_stem}_by_seed.csv"
    summary_path = tables / f"{table_stem}_summary.csv"
    figure_path = project_root / "docs/figs" / figure_name
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    render_heatmap(
        summary,
        figure_path,
        tau_values=tau_values,
    )
    manifest = {
        "alpha": 0.0,
        "tau_values": list(tau_values),
        "endpoints": [spec.endpoint for spec in specs],
        "n_pairs_per_endpoint": len(tau_pairs(tau_values)),
        "comparison": (
            "heterogeneous query penalty minus split-specific "
            "empirical-mass-mean-matched uniform query penalty"
        ),
        "selection_status": "no_revised_operating_point_selected",
        "artifacts": {
            "by_seed": str(paired_path),
            "summary": str(summary_path),
            "figure": str(figure_path),
        },
    }
    (output_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return paired_path, summary_path, figure_path


def run_expanded(
    *,
    project_root: Path,
    runs_root: Path,
    jobs: int,
    pbmc_raw_path: Path,
) -> tuple[Path, Path, Path]:
    return _run_direct_grid(
        project_root=project_root,
        runs_root=runs_root,
        jobs=jobs,
        pbmc_raw_path=pbmc_raw_path,
        specs=selected_specs("pbmc"),
        tau_values=EXPANDED_TAU_VALUES,
        output_name="rho_attribution_tau_surface_alpha0_expanded",
        table_stem="rho_tau_surface_expanded",
        figure_name="manuscript_fig_pbmc_rho_tau_surface_alpha0_expanded.png",
        progress_label="expanded PBMC",
    )


def run_focused(
    *,
    project_root: Path,
    runs_root: Path,
    jobs: int,
    pbmc_raw_path: Path,
) -> tuple[Path, Path, Path, Path]:
    output_name = "rho_attribution_tau_surface_alpha0_range075_175"
    _run_direct_grid(
        project_root=project_root,
        runs_root=runs_root,
        jobs=jobs,
        pbmc_raw_path=pbmc_raw_path,
        specs=selected_specs("pbmc"),
        tau_values=FOCUSED_TAU_VALUES,
        output_name=output_name,
        table_stem="rho_tau_surface_range075_175",
        figure_name=(
            "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png"
        ),
        progress_label="focused PBMC",
    )
    return aggregate_focused(project_root=project_root)


def aggregate_focused(
    *, project_root: Path
) -> tuple[Path, Path, Path, Path]:
    """Regenerate the focused display from the existing paired results."""
    output_name = "rho_attribution_tau_surface_alpha0_range075_175"
    output_root = project_root / "results/PBMC/sensitivity" / output_name
    paired_path = (
        output_root / "tables/rho_tau_surface_range075_175_by_seed.csv"
    )
    summary_path = (
        output_root / "tables/rho_tau_surface_range075_175_summary.csv"
    )
    figure_path = (
        project_root
        / "docs/figs/"
        "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png"
    )
    if not paired_path.is_file():
        raise FileNotFoundError(f"Missing focused paired table: {paired_path}")
    paired = pd.read_csv(paired_path)
    expected_rows = (
        len(selected_specs("pbmc"))
        * len(SEEDS)
        * len(tau_pairs(FOCUSED_TAU_VALUES))
    )
    if len(paired) != expected_rows:
        raise ValueError(
            f"Focused paired table has {len(paired)} rows; "
            f"expected {expected_rows}"
        )
    diagonal = np.isclose(paired["tau_min"], paired["tau_max"])
    delta_columns = [
        f"delta_{metric}_heterogeneous_minus_uniform"
        for metric in METRIC_NAMES
    ]
    maximum = paired.loc[diagonal, delta_columns].abs().to_numpy().max()
    if maximum > 1.0e-12:
        raise ValueError(f"Focused-grid diagonal identity failed: {maximum}")
    summary = summarize(paired)
    summary.to_csv(summary_path, index=False)
    summary = pd.read_csv(summary_path)
    render_s2_aligned_heatmap(
        summary,
        figure_path,
        tau_values=FOCUSED_TAU_VALUES,
    )
    caption_path = (
        project_root
        / "docs/figs/"
        "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175_caption.md"
    )
    caption_path.write_text(
        (
            "**PBMC matchability-penalty attribution at fixed "
            "$\\alpha=0$.** Rows show paired relative heterogeneous-minus-mean-"
            "matched-uniform differences (%) in AP, AUROC, represented-state "
            "forced accuracy, and represented-state forced macro-F1; columns "
            "show the B-cell, NK-cell, and dendritic-cell endpoints. For metric "
            "$M$, each split-level value is "
            "$100(M_{\\mathrm{heterogeneous}}-M_{\\mathrm{uniform}})/"
            "M_{\\mathrm{uniform}}$, and each displayed cell is the arithmetic "
            "mean across five fixed donor splits. Positive "
            "values favor the heterogeneous query penalty, and negative "
            "values favor the mean-matched uniform comparator. The vertical "
            "and horizontal axes show $\\tau_{\\min}$ and $\\tau_{\\max}$, "
            "respectively, with "
            "$\\tau_{\\min}\\leq\\tau_{\\max}$. Diagonal cells are zero "
            "because the two penalties coincide. Color scales are centered at "
            "zero and shared across endpoints within each metric but differ "
            "across metrics. Corresponding absolute differences are retained "
            "as scale context in Supplementary Data 4.\n"
        ),
        encoding="utf-8",
    )
    manifest_path = (
        project_root
        / "results/PBMC/sensitivity"
        / output_name
        / "manifest.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["visualization"] = {
        "metrics": [metric for metric, _ in S2_ALIGNED_METRICS],
        "effects": "relative_heterogeneous_minus_mean_matched_uniform_percent",
        "aggregation": "mean_of_split_level_relative_differences",
        "row_order": [label for _, label in S2_ALIGNED_METRICS],
        "color_scales": "zero_centered_shared_across_endpoints_within_metric",
        "overall_title": False,
        "selected_parameter_border": False,
    }
    manifest["artifacts"]["caption"] = str(caption_path)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return paired_path, summary_path, figure_path, caption_path


def run_dendritic_fine(
    *,
    project_root: Path,
    runs_root: Path,
    jobs: int,
    pbmc_raw_path: Path,
) -> tuple[Path, Path, Path]:
    return _run_direct_grid(
        project_root=project_root,
        runs_root=runs_root,
        jobs=jobs,
        pbmc_raw_path=pbmc_raw_path,
        specs=selected_specs("pbmc", ("Dendritic cells",)),
        tau_values=DENDRITIC_FINE_TAU_VALUES,
        output_name="rho_attribution_tau_surface_alpha0_dendritic_fine",
        table_stem="rho_tau_surface_dendritic_fine",
        figure_name=(
            "manuscript_fig_pbmc_rho_tau_surface_alpha0_dendritic_fine.png"
        ),
        progress_label="dendritic-fine PBMC",
    )


def run(
    *,
    project_root: Path,
    runs_root: Path,
    jobs: int,
    pbmc_raw_path: Path,
) -> tuple[Path, Path, Path]:
    output_root = (
        project_root
        / "results/PBMC/sensitivity/rho_attribution_tau_surface_alpha0"
    )
    condition = _pbmc_condition_map(pbmc_raw_path)
    tasks = [
        (spec, seed)
        for spec in selected_specs("pbmc")
        for seed in SEEDS
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_seed,
                runs_root=runs_root,
                output_root=output_root,
                spec=spec,
                seed=seed,
                pbmc_condition=condition,
            ): (spec.endpoint, seed)
            for spec, seed in tasks
        }
        checkpoints = []
        for future in concurrent.futures.as_completed(futures):
            endpoint, seed = futures[future]
            checkpoints.append(future.result())
            print(f"completed PBMC: {endpoint}, seed {seed}", flush=True)
    uniform = pd.concat(
        [pd.read_csv(path) for path in sorted(checkpoints)],
        ignore_index=True,
    )
    if len(uniform) != 225:
        raise ValueError(f"Uniform grid has {len(uniform)} rows; expected 225")
    heterogeneous_path = (
        project_root
        / "results/PBMC/sensitivity/component_ablation/tables/"
        "component_ablation_by_seed.csv"
    )
    paired = pair_with_heterogeneous(
        uniform, pd.read_csv(heterogeneous_path)
    )
    summary = summarize(paired)
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paired_path = tables / "rho_tau_surface_by_seed.csv"
    summary_path = tables / "rho_tau_surface_summary.csv"
    figure_path = (
        project_root
        / "docs/figs/manuscript_fig_pbmc_rho_tau_surface_alpha0.png"
    )
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    render_heatmap(summary, figure_path)
    manifest = {
        "alpha": 0.0,
        "tau_values": list(TAU_VALUES),
        "comparison": (
            "heterogeneous query penalty minus split-specific "
            "empirical-mass-mean-matched uniform query penalty"
        ),
        "primary_effect": "absolute_average_precision_difference",
        "secondary_effect": (
            "100 * absolute_average_precision_difference / uniform_AP"
        ),
        "artifacts": {
            "by_seed": str(paired_path),
            "summary": str(summary_path),
            "figure": str(figure_path),
        },
    }
    (output_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return paired_path, summary_path, figure_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the PBMC fixed-alpha paired rho-attribution "
            "tau-min/tau-max heatmap."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument(
        "--grid",
        choices=("standard", "expanded", "focused", "dendritic-fine"),
        default="focused",
    )
    parser.add_argument(
        "--pbmc-raw-path",
        type=Path,
        default=Path("data/raw/kang_2018.h5ad"),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Regenerate the focused summary, figure, caption, and manifest "
            "from the existing paired table without running model fits."
        ),
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.aggregate_only:
        if args.grid != "focused":
            raise ValueError("--aggregate-only currently requires --grid focused")
        for path in aggregate_focused(
            project_root=args.project_root.resolve()
        ):
            print(path)
        return 0
    runners = {
        "standard": run,
        "expanded": run_expanded,
        "focused": run_focused,
        "dendritic-fine": run_dendritic_fine,
    }
    runner = runners[args.grid]
    for path in runner(
        project_root=args.project_root.resolve(),
        runs_root=args.runs_root.resolve(),
        jobs=args.jobs,
        pbmc_raw_path=args.pbmc_raw_path.resolve(),
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
