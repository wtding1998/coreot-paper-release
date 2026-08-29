from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


EXPECTED_LABELS = ("HLA-DRhi cDC2", "ISG+ cDC2")
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
RECOVERY_GRID = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class MethodScore:
    display_name: str
    method: str
    score: str
    source: str


METHOD_SCORES = (
    MethodScore(r"CoRe-OT $\widetilde{u}$", "coreot_full", "u_tilde", "internal"),
    MethodScore(r"CoRe-OT $u$", "coreot_full", "u", "internal"),
    MethodScore("Uniform UOT", "uniform_uot", "u", "internal"),
    MethodScore("Seurat", "seurat_anchor", "u", "external"),
    MethodScore("SingleR", "singleR", "u", "external"),
    MethodScore("CellTypist", "celltypist_l3", "u", "external"),
    MethodScore("scmap-cell", "scmap_cell", "u", "external"),
    MethodScore("Nearest neighbor", "nn", "nn_distance", "internal"),
    MethodScore("Prior only", "prior_only", "prior_risk", "internal"),
)


@dataclass(frozen=True)
class RunDescriptor:
    run_id: str
    held_out_label: str
    seed: int


@dataclass(frozen=True)
class RankRecoveryPaths:
    output_root: Path
    data_root: Path
    deciles_by_seed: Path
    deciles_summary: Path
    recovery_by_seed: Path
    recovery_summary: Path
    png: Path
    pdf: Path
    description: Path


class RankRecoveryError(ValueError):
    pass


def tie_aware_cumulative_positives(
    scores: Iterable[float], labels: Iterable[bool], positions: Iterable[float]
) -> np.ndarray:
    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels), dtype=bool)
    position_array = np.asarray(list(positions), dtype=float)
    if score_array.ndim != 1 or label_array.ndim != 1 or len(score_array) != len(label_array):
        raise RankRecoveryError("Scores and labels must be one-dimensional and equally sized.")
    if len(score_array) == 0:
        raise RankRecoveryError("Cannot rank an empty score vector.")
    if not np.isfinite(score_array).all():
        raise RankRecoveryError("Ranked scores must all be finite.")
    if np.any(position_array < 0.0) or np.any(position_array > len(score_array)):
        raise RankRecoveryError("Rank positions must lie between zero and the cell count.")

    grouped = (
        pd.DataFrame({"score": score_array, "positive": label_array.astype(float)})
        .groupby("score", sort=True, as_index=False)
        .agg(group_size=("positive", "size"), positives=("positive", "sum"))
        .sort_values("score", ascending=False, ignore_index=True)
    )
    sizes = grouped["group_size"].to_numpy(dtype=float)
    positives = grouped["positives"].to_numpy(dtype=float)
    starts = np.concatenate(([0.0], np.cumsum(sizes)[:-1]))
    overlap = np.clip(position_array[:, None] - starts[None, :], 0.0, sizes[None, :])
    return np.sum(overlap * (positives / sizes)[None, :], axis=1)


def rank_decile_composition(scores: Iterable[float], labels: Iterable[bool]) -> pd.DataFrame:
    score_array = np.asarray(list(scores), dtype=float)
    boundaries = np.linspace(0.0, float(len(score_array)), 11)
    cumulative = tie_aware_cumulative_positives(scores, labels, boundaries)
    positive_counts = np.diff(cumulative)
    bin_sizes = np.diff(boundaries)
    return pd.DataFrame(
        {
            "rank_decile": np.arange(1, 11),
            "q_start": np.arange(0.0, 1.0, 0.1),
            "q_end": np.arange(0.1, 1.01, 0.1),
            "expected_held_out_count": positive_counts,
            "bin_size": bin_sizes,
            "held_out_percent": 100.0 * positive_counts / bin_sizes,
        }
    )


def recovery_curve(
    scores: Iterable[float], labels: Iterable[bool], q_grid: Iterable[float] = RECOVERY_GRID
) -> pd.DataFrame:
    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels), dtype=bool)
    positives = int(label_array.sum())
    if positives == 0:
        raise RankRecoveryError("Recovery is undefined without held-out cells.")
    q_array = np.asarray(list(q_grid), dtype=float)
    recovered = tie_aware_cumulative_positives(
        score_array, label_array, q_array * len(score_array)
    )
    return pd.DataFrame(
        {
            "q": q_array,
            "expected_held_out_recovered": recovered,
            "recovery": recovered / positives,
        }
    )


def discover_runs(grid_dir: Path) -> list[RunDescriptor]:
    descriptors: list[RunDescriptor] = []
    for config_dir in sorted(path for path in grid_dir.iterdir() if path.is_dir()):
        config = yaml.safe_load((config_dir / "benchmark.yaml").read_text(encoding="utf-8"))
        descriptors.append(
            RunDescriptor(
                run_id=str(config["run_id"]),
                held_out_label=str(config["removed_state"]),
                seed=int(config["split"]["seed"]),
            )
        )
    actual = {(item.held_out_label, item.seed) for item in descriptors}
    expected = {(label, seed) for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS}
    if actual != expected or len(descriptors) != len(expected):
        raise RankRecoveryError(
            "Selected run grid must contain exactly both held-out labels and seeds 1--5; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )
    return descriptors


def _read_and_validate_method_scores(
    *, run_root: Path, truth: pd.DataFrame, method_score: MethodScore, condition: str,
    internal_candidate_set: str, external_candidate_set: str
) -> pd.DataFrame:
    candidate_set = (
        internal_candidate_set if method_score.source == "internal" else external_candidate_set
    )
    path = run_root / "scoring" / condition / candidate_set / "cell_scores.parquet"
    scores = pd.read_parquet(path)
    required = {"cell_id", "method", method_score.score}
    missing = required - set(scores.columns)
    if missing:
        raise RankRecoveryError(f"{path} is missing columns {sorted(missing)}.")
    selected = scores.loc[scores["method"] == method_score.method, ["cell_id", method_score.score]]
    if selected["cell_id"].duplicated().any():
        raise RankRecoveryError(f"{path} has duplicate cells for method {method_score.method}.")
    truth_ids = set(truth["cell_id"])
    score_ids = set(selected["cell_id"])
    if len(selected) != len(truth) or score_ids != truth_ids:
        raise RankRecoveryError(
            f"Cell set mismatch for {method_score.method}/{method_score.score} in {path}."
        )
    selected = truth[["cell_id", "is_absent_state"]].merge(
        selected, on="cell_id", how="left", validate="one_to_one"
    )
    values = selected[method_score.score].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RankRecoveryError(
            f"Non-finite {method_score.score} values for {method_score.method} in {path}."
        )
    return selected


def _validate_metrics(
    *, descriptor: RunDescriptor, method_score: MethodScore, scores: pd.DataFrame,
    comparison: pd.DataFrame, tolerance: float
) -> None:
    expected = comparison.loc[
        (comparison["run_id"] == descriptor.run_id)
        & (comparison["held_out_label"] == descriptor.held_out_label)
        & (comparison["seed"] == descriptor.seed)
        & (comparison["method"] == method_score.method)
        & (comparison["score"] == method_score.score)
    ]
    if len(expected) != 1:
        raise RankRecoveryError(
            "Expected exactly one comparison row for "
            f"{descriptor.run_id}, {method_score.method}/{method_score.score}; found {len(expected)}."
        )
    y_true = scores["is_absent_state"].astype(bool).to_numpy()
    y_score = scores[method_score.score].to_numpy(dtype=float)
    actual_auroc = roc_auc_score(y_true, y_score)
    actual_auprc = average_precision_score(y_true, y_score)
    row = expected.iloc[0]
    if not np.isclose(actual_auroc, float(row["auroc"]), rtol=0.0, atol=tolerance):
        raise RankRecoveryError(
            f"AUROC mismatch for {descriptor.run_id}, {method_score.method}/{method_score.score}."
        )
    if not np.isclose(actual_auprc, float(row["auprc"]), rtol=0.0, atol=tolerance):
        raise RankRecoveryError(
            f"AUPRC mismatch for {descriptor.run_id}, {method_score.method}/{method_score.score}."
        )


def _validate_rank_outputs(scores: pd.DataFrame, method_score: MethodScore) -> None:
    values = scores[method_score.score].to_numpy(dtype=float)
    labels = scores["is_absent_state"].astype(bool).to_numpy()
    curve = recovery_curve(values, labels)
    recovery = curve["recovery"].to_numpy()
    if not np.isclose(recovery[0], 0.0) or not np.isclose(recovery[-1], 1.0):
        raise RankRecoveryError("Recovery curve endpoints must equal zero and one.")
    if np.any(np.diff(recovery) < -1e-12):
        raise RankRecoveryError("Recovery curves must be nondecreasing.")

    reversed_curve = recovery_curve(values[::-1], labels[::-1])
    order = np.lexsort((scores["cell_id"].astype(str).to_numpy()[::-1], -values))
    tied_reordered_curve = recovery_curve(values[order], labels[order])
    if not np.allclose(curve["recovery"], reversed_curve["recovery"], atol=1e-12):
        raise RankRecoveryError("Recovery changed after permuting input rows.")
    if not np.allclose(curve["recovery"], tied_reordered_curve["recovery"], atol=1e-12):
        raise RankRecoveryError("Recovery changed after reordering tied-score rows.")


def _summarize_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["held_out_label", "display_name", "method", "score", "rank_decile", "q_start", "q_end"]
    summary = (
        frame.groupby(keys, sort=False)["held_out_percent"]
        .agg(mean="mean", std="std", n_seeds="count")
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n_seeds"])
    return summary


def _summarize_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["held_out_label", "display_name", "method", "score", "q"]
    summary = (
        frame.groupby(keys, sort=False)["recovery"]
        .agg(mean="mean", std="std", n_seeds="count")
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n_seeds"])
    return summary


def _method_styles() -> dict[tuple[str, str], dict[str, object]]:
    colors = [
        "#005A9C", "#005A9C", "#B35C00", "#6A3D9A", "#5F6B6D",
        "#7A6F59", "#607D3B", "#8A5A6B", "#397D8C", "#777777",
    ]
    linestyles = ["-", "--", "-", "-.", "-", "--", "-.", ":", "--", ":"]
    return {
        (item.method, item.score): {
            "color": colors[index],
            "linestyle": linestyles[index],
            "linewidth": 2.8 if index == 0 else 2.0,
            "alpha": 1.0 if index < 2 else 0.82,
        }
        for index, item in enumerate(METHOD_SCORES)
    }


def _draw_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.98, "A", fontsize=13, fontweight="bold", va="top")
    boxes = (
        (0.05, 0.72, 0.38, 0.18, "Query", "held-out state\nretained"),
        (0.57, 0.72, 0.38, 0.18, "Reference", "held-out state\nremoved"),
        (0.16, 0.40, 0.68, 0.16, "Method-specific score", "higher = weaker support"),
        (0.16, 0.12, 0.68, 0.17, "Descending rank", "flag highest-scoring cells first"),
    )
    for x, y, width, height, title, subtitle in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.015", facecolor="#F5F5F5",
                edgecolor="#555555", linewidth=1.0
            )
        )
        ax.text(x + width / 2, y + height * 0.68, title, ha="center", fontweight="bold")
        ax.text(
            x + width / 2, y + height * 0.25, subtitle, ha="center", va="center", fontsize=7.5
        )
    for x, color in zip((0.15, 0.23, 0.31, 0.65, 0.73, 0.81),
                        ("#C43C39", "#4C78A8", "#4C78A8", "#4C78A8", "#4C78A8", "#4C78A8")):
        ax.scatter(x, 0.80, s=52, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.add_patch(FancyArrowPatch((0.24, 0.71), (0.48, 0.57), arrowstyle="-|>", mutation_scale=12))
    ax.add_patch(FancyArrowPatch((0.76, 0.71), (0.52, 0.57), arrowstyle="-|>", mutation_scale=12))
    ax.add_patch(FancyArrowPatch((0.50, 0.39), (0.50, 0.30), arrowstyle="-|>", mutation_scale=12))
    for index, color in enumerate(("#C43C39", "#D97A45", "#A9A9A9", "#4C78A8")):
        ax.scatter(0.32 + 0.12 * index, 0.145, s=55 - 6 * index, color=color, edgecolor="white")
    ax.text(0.50, 0.05, "inspection budget $q$", ha="center", fontsize=8)


def _draw_heatmap(
    ax: plt.Axes, frame: pd.DataFrame, held_out_label: str, panel: str, vmax: float
) -> None:
    subset = frame.loc[frame["held_out_label"] == held_out_label]
    matrix = np.vstack(
        [
            subset.loc[
                (subset["method"] == item.method) & (subset["score"] == item.score)
            ].sort_values("rank_decile")["mean"].to_numpy()
            for item in METHOD_SCORES
        ]
    )
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
    threshold = 0.55 * vmax
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column, row, f"{value:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if value > threshold else "#222222"
            )
    ax.set_yticks(range(len(METHOD_SCORES)), [item.display_name for item in METHOD_SCORES])
    ax.set_xticks(range(10), [str(index) for index in range(1, 11)])
    ax.set_xlabel("Score-rank decile (1 = highest score)")
    ax.set_title(held_out_label, fontsize=10)
    ax.text(-0.18, 1.08, panel, transform=ax.transAxes, fontsize=13, fontweight="bold")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Held-out cells (%)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def _draw_recovery(
    ax: plt.Axes, frame: pd.DataFrame, held_out_label: str, panel: str
) -> None:
    subset = frame.loc[frame["held_out_label"] == held_out_label]
    styles = _method_styles()
    for item in METHOD_SCORES:
        curve = subset.loc[
            (subset["method"] == item.method) & (subset["score"] == item.score)
        ].sort_values("q")
        ax.plot(curve["q"], curve["mean"], **styles[(item.method, item.score)])
    ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1.2, zorder=0)
    ax.axvline(0.10, color="#555555", linestyle=":", linewidth=1.0)
    ax.text(
        0.105, 0.04, "10% budget", rotation=90, fontsize=7, va="bottom", color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fraction of query cells flagged")
    ax.set_ylabel("Fraction of held-out cells recovered")
    ax.set_title(held_out_label, fontsize=10)
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.text(-0.16, 1.08, panel, transform=ax.transAxes, fontsize=13, fontweight="bold")


def _write_figure(deciles: pd.DataFrame, recovery: pd.DataFrame, png: Path, pdf: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(15.5, 9.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(0.82, 1.55, 1.25))
    schematic_ax = figure.add_subplot(grid[:, 0])
    hla_heatmap_ax = figure.add_subplot(grid[0, 1])
    hla_recovery_ax = figure.add_subplot(grid[0, 2])
    isg_heatmap_ax = figure.add_subplot(grid[1, 1])
    isg_recovery_ax = figure.add_subplot(grid[1, 2])
    _draw_schematic(schematic_ax)
    vmax = float(deciles["mean"].max())
    _draw_heatmap(hla_heatmap_ax, deciles, EXPECTED_LABELS[0], "B", vmax)
    _draw_recovery(hla_recovery_ax, recovery, EXPECTED_LABELS[0], "C")
    _draw_heatmap(isg_heatmap_ax, deciles, EXPECTED_LABELS[1], "D", vmax)
    _draw_recovery(isg_recovery_ax, recovery, EXPECTED_LABELS[1], "E")
    styles = _method_styles()
    handles = [
        Line2D([0], [0], label=item.display_name, **styles[(item.method, item.score)])
        for item in METHOD_SCORES
    ]
    handles.append(Line2D([0], [0], color="#AAAAAA", linestyle="--", label="Random ranking"))
    figure.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=8)
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def _render_description(paths: RankRecoveryPaths) -> str:
    return f"""# Figure 1: Prioritization of query cells from states missing in the reference

## Files

- PNG: `{paths.png}`
- PDF: `{paths.pdf}`

## Source data

- Seed-level rank-decile composition: `{paths.deciles_by_seed}`
- Rank-decile composition summary: `{paths.deciles_summary}`
- Seed-level recovery curves: `{paths.recovery_by_seed}`
- Recovery-curve summary: `{paths.recovery_summary}`

## Construction and interpretation

Cells are ranked separately for each method-score pair, with higher scores
denoting weaker inferred reference support. Tied-score groups are allocated
proportionally at rank boundaries, so the derived quantities do not depend on
row order or hidden evaluation labels. Rank-decile tiles show the mean
held-out-cell percentage across five donor-split seeds. Recovery curves show
the five-seed mean on the common grid `q = 0, 0.01, ..., 1`; individual seed
curves are retained in the source CSV but omitted from the main panel.

The generator validates cell alignment, finite scores, recovery invariants,
row-order invariance, and per-run AUROC/AUPRC against
`results/HIHA_DC/compare_baselines/tables/compare_detection_by_run.csv`.
The figure re-expresses the same cell-level rankings used by AUROC and AUPRC
and is not independent validation evidence.
"""


def write_rank_recovery_figure(
    *, runs_root: Path = Path("runs"),
    grid_dir: Path = Path("experiments/missing_celltype/generated_configs/report_leave_one_HIHA_DC"),
    comparison_path: Path = Path("results/HIHA_DC/compare_baselines/tables/compare_detection_by_run.csv"),
    output_root: Path = Path("results/HIHA_DC/figures"),
    condition: str = "incomplete_reference",
    internal_candidate_set: str = "hiha_harmony30_k100",
    external_candidate_set: str = "external_reference_mapping",
    metric_tolerance: float = 1e-10,
) -> RankRecoveryPaths:
    descriptors = discover_runs(grid_dir)
    comparison = pd.read_csv(comparison_path)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    paths = RankRecoveryPaths(
        output_root=output_root,
        data_root=data_root,
        deciles_by_seed=data_root / "figure_1_rank_deciles_by_seed.csv",
        deciles_summary=data_root / "figure_1_rank_deciles_summary.csv",
        recovery_by_seed=data_root / "figure_1_recovery_curves_by_seed.csv",
        recovery_summary=data_root / "figure_1_recovery_curves_summary.csv",
        png=output_root / "figure_1_rank_recovery.png",
        pdf=output_root / "figure_1_rank_recovery.pdf",
        description=output_root / "figure_1_rank_recovery.md",
    )

    decile_frames: list[pd.DataFrame] = []
    recovery_frames: list[pd.DataFrame] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        truth_path = run_root / "benchmark" / condition / "evaluation_truth" / "query_truth.csv"
        truth = pd.read_csv(truth_path)
        required_truth = {"cell_id", "is_absent_state"}
        if required_truth - set(truth.columns) or truth["cell_id"].duplicated().any():
            raise RankRecoveryError(f"Invalid query truth artifact: {truth_path}.")
        for method_score in METHOD_SCORES:
            scores = _read_and_validate_method_scores(
                run_root=run_root, truth=truth, method_score=method_score, condition=condition,
                internal_candidate_set=internal_candidate_set,
                external_candidate_set=external_candidate_set,
            )
            _validate_metrics(
                descriptor=descriptor, method_score=method_score, scores=scores,
                comparison=comparison, tolerance=metric_tolerance,
            )
            _validate_rank_outputs(scores, method_score)
            metadata = {
                "run_id": descriptor.run_id,
                "held_out_label": descriptor.held_out_label,
                "seed": descriptor.seed,
                "display_name": method_score.display_name,
                "method": method_score.method,
                "score": method_score.score,
            }
            decile_frames.append(rank_decile_composition(
                scores[method_score.score], scores["is_absent_state"]
            ).assign(**metadata))
            recovery_frames.append(recovery_curve(
                scores[method_score.score], scores["is_absent_state"]
            ).assign(**metadata))

    deciles_by_seed = pd.concat(decile_frames, ignore_index=True)
    recovery_by_seed = pd.concat(recovery_frames, ignore_index=True)
    deciles_summary = _summarize_deciles(deciles_by_seed)
    recovery_summary = _summarize_recovery(recovery_by_seed)
    deciles_by_seed.to_csv(paths.deciles_by_seed, index=False)
    deciles_summary.to_csv(paths.deciles_summary, index=False)
    recovery_by_seed.to_csv(paths.recovery_by_seed, index=False)
    recovery_summary.to_csv(paths.recovery_summary, index=False)
    _write_figure(deciles_summary, recovery_summary, paths.png, paths.pdf)
    paths.description.write_text(_render_description(paths), encoding="utf-8")
    return paths
