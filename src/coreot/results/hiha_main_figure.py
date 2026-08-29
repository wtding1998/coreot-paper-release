from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from PIL import Image


ROOT = Path("results/HIHA_DC")
RUNS = Path("runs")
GRID = Path("experiments/missing_celltype/generated_configs/report_leave_one_HIHA_DC")
FIGURES = ROOT / "figures"
DATA = FIGURES / "data"
FIGURE_SIZE_INCHES = (178 / 25.4, 118 / 25.4)
RASTER_DPI = 351
LABELS = ("HLA-DRhi cDC2", "ISG+ cDC2")
METHODS = (
    ("coreot_full", "u", r"CoRe-OT ($u$)"),
    ("uniform_uot", "u", "Uniform UOT"),
    ("nn", "nn_distance", "Nearest neighbor"),
    ("prior_only", "prior_risk", "Prior only"),
    ("seurat_anchor", "u", "Seurat"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
OPERATIONAL = (
    ("coreot_full", "u", r"CoRe-OT ($u$)"),
    ("uniform_uot", "u", "Uniform UOT"),
    ("seurat_anchor", "u", "Seurat"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
COLORS = {
    "coreot_full": "#0072B2",
    "uniform_uot": "#3F3F3F",
    "nn": "#A0A0A0",
    "prior_only": "#BDBDBD",
    "seurat_anchor": "#E69F00",
    "scmap_cluster": "#6F58A8",
    "chetah": "#CC79A7",
}
RETAINED_FILL = "#D5D5D5"
ABSTAINED_FILL = "#009E73"
TRUTH_EDGE = "#111111"
BACKGROUND_SIZE = 2.0
ABSTAINED_SIZE = 4.0
HELD_OUT_SIZE = 4.5
DISPLAY = {method: name for method, _, name in METHODS}


class HIHAMainFigureError(ValueError):
    """Raised when retained HIHA artifacts cannot support the main figure."""


def _require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        raise HIHAMainFigureError(f"{name} has duplicate keys: {keys}")


def _finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise HIHAMainFigureError(f"{name}.{column} contains nonfinite values")


def _selected_detection() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "main/tables/main_detection_by_run.csv")
    selected = {(method, score) for method, score, _ in METHODS}
    frame = frame.loc[
        frame["evaluation_scope"].eq("local_within_broad_state")
        & frame.apply(lambda row: (row["method"], row["score"]) in selected, axis=1)
    ].copy()
    expected = {
        (label, seed, method, score)
        for label in LABELS
        for seed in range(1, 6)
        for method, score, _ in METHODS
    }
    observed = set(
        zip(frame["held_out_label"], frame["seed"], frame["method"], frame["score"], strict=False)
    )
    if observed != expected:
        raise HIHAMainFigureError(
            "Panel C does not contain the approved state, seed, method, and score keys"
        )
    _require_unique(frame, ["held_out_label", "seed", "method", "score"], "detection")
    _finite(frame, ["auprc", "prevalence"], "detection")
    return frame


def _selected_operational(detection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    score_by_method = {method: score for method, score, _ in OPERATIONAL}
    run_rows = detection.loc[
        detection["method"].eq("coreot_full") & detection["score"].eq("u")
    ]
    for run in run_rows.itertuples(index=False):
        run_root = RUNS / str(run.run_id)
        truth = pd.read_csv(
            run_root / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
        )
        _require_unique(truth, ["cell_id"], f"{run.run_id} truth")
        for method, score, _ in OPERATIONAL:
            if method == "coreot_full":
                calls = _u_abstention(run_root)
            else:
                method_scores = _method_scores(run_root, method, "incomplete_reference")
                calls = method_scores[["cell_id", "abstain_u_or_entropy"]].rename(
                    columns={"abstain_u_or_entropy": "abstained"}
                )
            joined = truth[["cell_id", "is_absent_state", "is_shared_state"]].merge(
                calls, on="cell_id", validate="one_to_one"
            )
            absent = joined.loc[joined["is_absent_state"].astype(bool), "abstained"]
            shared = joined.loc[joined["is_shared_state"].astype(bool), "abstained"]
            rows.append(
                {
                    "run_id": run.run_id,
                    "held_out_label": run.held_out_label,
                    "seed": run.seed,
                    "method": method,
                    "score": score_by_method[method],
                    "absent_abstention_rate": float(absent.astype(bool).mean()),
                    "coverage": float(1.0 - shared.astype(bool).mean()),
                }
            )
    result = pd.DataFrame(rows)
    _require_unique(
        result, ["run_id", "held_out_label", "seed", "method", "score"], "operational"
    )
    expected = {(label, seed) for label in LABELS for seed in range(1, 6)}
    if set(zip(result["held_out_label"], result["seed"], strict=False)) != expected:
        raise HIHAMainFigureError("Panel E does not contain both states and seeds 1 through 5")
    _finite(result, ["absent_abstention_rate", "coverage"], "operational")
    return result


def _selected_rescue() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "main/tables/rescue_by_run.csv")
    frame = frame.loc[
        frame["method"].eq("coreot_full")
        & frame["truth_group"].isin(["held_out_positive", "shared_cDC2"])
    ].copy()
    _require_unique(frame, ["held_out_label", "seed", "truth_group"], "rescue")
    expected = {
        (label, seed, group)
        for label in LABELS
        for seed in range(1, 6)
        for group in ("held_out_positive", "shared_cDC2")
    }
    observed = set(
        zip(frame["held_out_label"], frame["seed"], frame["truth_group"], strict=False)
    )
    if observed != expected:
        raise HIHAMainFigureError(
            "Panel D does not contain both rescue groups for all five seeds"
        )
    _finite(frame, ["delta_u_rescue_mean"], "rescue")
    return frame


def _run_id(detection: pd.DataFrame, label: str) -> str:
    rows = detection.loc[
        detection["held_out_label"].eq(label)
        & detection["seed"].eq(1)
        & detection["method"].eq("coreot_full")
        & detection["score"].eq("u"),
        "run_id",
    ]
    if len(rows) != 1:
        raise HIHAMainFigureError(f"Expected one seed-1 run for {label}; found {len(rows)}")
    return str(rows.iloc[0])


def _method_scores(run_root: Path, method: str, condition: str) -> pd.DataFrame:
    family = (
        "hiha_harmony30_k100"
        if method in {"coreot_full", "uniform_uot"}
        else "external_reference_mapping"
    )
    frame = pd.read_parquet(run_root / f"scoring/{condition}/{family}/cell_scores.parquet")
    frame = frame.loc[frame["method"].eq(method)].copy()
    _require_unique(frame, ["cell_id"], f"{method} {condition} scores")
    return frame


def _u_abstention(run_root: Path) -> pd.DataFrame:
    scoring = yaml.safe_load((GRID / run_root.name / "scoring.yaml").read_text(encoding="utf-8"))
    thresholds = scoring.get("thresholds") or {}
    u_config = thresholds.get("theta_u") or {}
    quantile = float(u_config.get("quantile", 0.95))
    source_condition = str(u_config.get("source_condition", "full_reference_control"))
    entropy_threshold = float((thresholds.get("theta_H") or {}).get("value", float("inf")))
    full = _method_scores(run_root, "coreot_full", source_condition)
    incomplete = _method_scores(run_root, "coreot_full", "incomplete_reference")
    threshold = float(pd.to_numeric(full["u"], errors="coerce").quantile(quantile))
    if not math.isfinite(threshold):
        raise HIHAMainFigureError(f"Nonfinite u threshold for {run_root.name}")
    u = pd.to_numeric(incomplete["u"], errors="coerce")
    entropy = pd.to_numeric(incomplete["label_entropy"], errors="coerce")
    if not np.isfinite(u).all() or not np.isfinite(entropy).all():
        raise HIHAMainFigureError(f"Nonfinite CoRe-OT scores for {run_root.name}")
    return pd.DataFrame(
        {
            "cell_id": incomplete["cell_id"].astype(str),
            "abstained": (u > threshold) | (entropy > entropy_threshold),
        }
    )


def _query_umap(run_root: Path) -> pd.DataFrame:
    embedding_root = run_root / "embeddings/incomplete_reference/hiha_harmony30"
    cells = pd.read_csv(embedding_root / "embedding_cells.csv")
    matrix = np.load(embedding_root / "embedding.npy")
    if matrix.shape != (len(cells), 30) or not np.isfinite(matrix).all():
        raise HIHAMainFigureError(
            f"Expected finite ({len(cells)}, 30) embedding for {run_root.name}; got {matrix.shape}"
        )
    query_mask = cells["domain"].astype(str).eq("query").to_numpy()
    query_cells = cells.loc[query_mask, "cell_id"].astype(str).reset_index(drop=True)
    query = ad.AnnData(X=matrix[query_mask].copy())
    sc.pp.neighbors(query, n_neighbors=15, use_rep="X", metric="euclidean", random_state=1)
    sc.tl.umap(query, min_dist=0.3, random_state=1)
    return pd.DataFrame(
        {
            "cell_id": query_cells,
            "umap_1": query.obsm["X_umap"][:, 0],
            "umap_2": query.obsm["X_umap"][:, 1],
        }
    )


def _embedding_data(detection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    map_methods = ("coreot_full", "uniform_uot", "seurat_anchor", "scmap_cluster", "chetah")
    for label in LABELS:
        run_root = RUNS / _run_id(detection, label)
        truth = pd.read_csv(
            run_root / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
        )
        _require_unique(truth, ["cell_id"], f"{label} truth")
        coordinates = _query_umap(run_root)
        if set(coordinates["cell_id"]) != set(truth["cell_id"].astype(str)):
            raise HIHAMainFigureError(f"Embedding/truth query-cell mismatch for {label}")
        base = coordinates.merge(
            truth[["cell_id", "is_absent_state"]], on="cell_id", validate="one_to_one"
        )
        base.insert(0, "held_out_label", label)
        truth_view = base.copy()
        truth_view["method"] = "truth"
        truth_view["abstained"] = False
        rows.append(truth_view)
        for method in map_methods:
            if method == "coreot_full":
                calls = _u_abstention(run_root)
            else:
                scores = _method_scores(run_root, method, "incomplete_reference")
                calls = scores[["cell_id", "abstain_u_or_entropy"]].rename(
                    columns={"abstain_u_or_entropy": "abstained"}
                )
            if set(calls["cell_id"].astype(str)) != set(base["cell_id"]):
                raise HIHAMainFigureError(f"{method} query-cell mismatch for {label}")
            view = base.merge(calls, on="cell_id", validate="one_to_one")
            view["method"] = method
            rows.append(view)
    result = pd.concat(rows, ignore_index=True)
    coordinates_by_cell = result.groupby(["held_out_label", "cell_id"], sort=False)[
        ["umap_1", "umap_2"]
    ].nunique()
    if coordinates_by_cell.gt(1).any().any():
        raise HIHAMainFigureError(
            "Panel B coordinates differ across columns within a held-out-state row"
        )
    return result


def _validate_embedding_operating_rates(
    embedding: pd.DataFrame, operational: pd.DataFrame
) -> None:
    for label in LABELS:
        for method, score, _ in OPERATIONAL:
            frame = embedding.loc[
                embedding["held_out_label"].eq(label) & embedding["method"].eq(method)
            ]
            held_out = frame["is_absent_state"].astype(bool)
            abstained = frame["abstained"].astype(bool)
            observed_absent = float(abstained.loc[held_out].mean())
            observed_shared_false = float(abstained.loc[~held_out].mean())
            expected = operational.loc[
                operational["held_out_label"].eq(label)
                & operational["seed"].eq(1)
                & operational["method"].eq(method)
                & operational["score"].eq(score)
            ]
            if len(expected) != 1:
                raise HIHAMainFigureError(
                    f"Expected one seed-1 operating row for {label}/{method}/{score}"
                )
            expected_absent = float(expected.iloc[0]["absent_abstention_rate"])
            expected_shared_false = 1.0 - float(expected.iloc[0]["coverage"])
            if not np.isclose(observed_absent, expected_absent, atol=1e-12):
                raise HIHAMainFigureError(
                    f"Panel B held-out abstention mismatch for {label}/{method}"
                )
            if not np.isclose(observed_shared_false, expected_shared_false, atol=1e-12):
                raise HIHAMainFigureError(
                    f"Panel B shared false-abstention mismatch for {label}/{method}"
                )


def _outward_limits(values: pd.Series, *, lower_bound: float = 0.0, step: float = 0.02) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, step)
    low = max(lower_bound, math.floor((low - 0.08 * span) / step) * step)
    high = min(1.0, math.ceil((high + 0.08 * span) / step) * step)
    return low, high


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.04, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def _draw_cluster(ax: plt.Axes, x: float, y: float, color: str, *, empty: bool = False) -> None:
    offsets = ((0, 0), (-0.035, 0.025), (0.035, 0.025), (-0.025, -0.035), (0.03, -0.035))
    for dx, dy in offsets:
        ax.add_patch(
            Circle(
                (x + dx, y + dy), 0.014, facecolor="white" if empty else color,
                edgecolor=color, linewidth=0.8, linestyle="--" if empty else "-",
            )
        )


def _draw_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    _panel_letter(ax, "A")
    represented = "#56B4E9"
    held_out = "#D55E00"
    other = "#BDBDBD"
    _draw_cluster(ax, 0.18, 0.66, held_out)
    _draw_cluster(ax, 0.18, 0.54, represented)
    _draw_cluster(ax, 0.18, 0.42, other)
    _draw_cluster(ax, 0.70, 0.66, held_out, empty=True)
    _draw_cluster(ax, 0.70, 0.54, represented)
    _draw_cluster(ax, 0.70, 0.42, other)
    _draw_cluster(ax, 0.70, 0.24, held_out)
    _draw_cluster(ax, 0.70, 0.12, represented)
    _draw_cluster(ax, 0.70, 0.00, other)
    ax.text(0.18, 0.93, "Query", ha="center", va="center", fontsize=6.5)
    ax.text(0.18, 0.87, "all L3 states", ha="center", va="center", fontsize=6.5)
    ax.text(0.70, 0.93, "Ablated reference", ha="center", va="center", fontsize=6.5)
    ax.text(0.70, 0.37, "Full ref.\nstate restored", ha="center", va="center", fontsize=6.5)
    ax.add_patch(FancyArrowPatch((0.31, 0.65), (0.57, 0.65), arrowstyle="->", mutation_scale=8, lw=0.7))
    ax.add_patch(FancyArrowPatch((0.31, 0.52), (0.57, 0.24), arrowstyle="->", mutation_scale=8, lw=0.7))
    ax.text(0.46, 0.77, "held-out AIFI_L3 state\nHLA-DRhi cDC2 / ISG+ cDC2", ha="center", va="center", fontsize=6.5)
    ax.plot((0.09, 0.79), (-0.07, -0.07), color="#555555", lw=0.6)
    ax.plot((0.09, 0.09), (-0.07, -0.02), color="#555555", lw=0.6)
    ax.plot((0.79, 0.79), (-0.07, -0.02), color="#555555", lw=0.6)
    ax.text(0.44, -0.09, "broad anchor retained: AIFI_L2 = cDC2", ha="center", va="top", fontsize=6.5)
    ax.set(xlim=(0, 1), ylim=(-0.12, 1))


def _draw_embedding_grid(container, data: pd.DataFrame) -> None:
    grid = container.subgridspec(
        2,
        7,
        width_ratios=(1.0, 0.14, 1.0, 1.0, 1.0, 1.0, 1.0),
        wspace=0.04,
        hspace=0.10,
    )
    methods = ("truth", "coreot_full", "uniform_uot", "seurat_anchor", "scmap_cluster", "chetah")
    grid_columns = (0, 2, 3, 4, 5, 6)
    titles = (
        "Evaluation\ntruth",
        "CoRe-OT\n" + r"$u$",
        "Uniform UOT\n" + r"$u$",
        "Seurat",
        "scmap\ncluster",
        "CHETAH",
    )
    for row, label in enumerate(LABELS):
        label_data = data.loc[data["held_out_label"].eq(label)]
        xlim = tuple(label_data["umap_1"].quantile([0.005, 0.995]))
        ylim = tuple(label_data["umap_2"].quantile([0.005, 0.995]))
        for col, (method, title, grid_column) in enumerate(
            zip(methods, titles, grid_columns, strict=True)
        ):
            ax = plt.subplot(grid[row, grid_column])
            frame = label_data.loc[label_data["method"].eq(method)]
            held = frame["is_absent_state"].astype(bool)
            if method == "truth":
                ax.scatter(
                    frame.loc[~held, "umap_1"],
                    frame.loc[~held, "umap_2"],
                    s=BACKGROUND_SIZE,
                    color=RETAINED_FILL,
                    alpha=0.45,
                    linewidth=0,
                    rasterized=True,
                    zorder=1,
                )
                ax.scatter(
                    frame.loc[held, "umap_1"],
                    frame.loc[held, "umap_2"],
                    s=4.2,
                    color=TRUTH_EDGE,
                    alpha=0.95,
                    linewidth=0,
                    rasterized=True,
                    zorder=2,
                )
            else:
                abstained = frame["abstained"].astype(bool)
                shared = ~held
                retained = ~abstained
                categories = (
                    (shared & retained, BACKGROUND_SIZE, RETAINED_FILL, "none", 0.45, 0.0, 1),
                    (shared & abstained, ABSTAINED_SIZE, ABSTAINED_FILL, "none", 0.90, 0.0, 2),
                    (held & retained, HELD_OUT_SIZE, RETAINED_FILL, TRUTH_EDGE, 0.95, 0.30, 3),
                    (held & abstained, HELD_OUT_SIZE, ABSTAINED_FILL, TRUTH_EDGE, 1.0, 0.30, 4),
                )
                for mask, size, fill, edge, alpha, linewidth, zorder in categories:
                    ax.scatter(
                        frame.loc[mask, "umap_1"],
                        frame.loc[mask, "umap_2"],
                        s=size,
                        facecolors=fill,
                        edgecolors=edge,
                        linewidths=linewidth,
                        alpha=alpha,
                        rasterized=True,
                        zorder=zorder,
                    )
            ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=6.5, pad=1, linespacing=0.9)
                if method != "truth":
                    ax.plot(
                        (0.18, 0.82),
                        (1.015, 1.015),
                        transform=ax.transAxes,
                        color=COLORS[method],
                        lw=1.1,
                        solid_capstyle="round",
                        clip_on=False,
                    )
            if method == "truth":
                ax.plot(
                    (1.10, 1.10),
                    (0.0, 1.0),
                    transform=ax.transAxes,
                    color="#B8B8B8",
                    lw=0.6,
                    clip_on=False,
                )
            if row == 0 and col == 0:
                _panel_letter(ax, "B")


def _draw_detection(container, detection: pd.DataFrame) -> None:
    grid = container.subgridspec(1, 2, wspace=0.10)
    order = [method for method, _, _ in METHODS]
    for col, label in enumerate(LABELS):
        ax = plt.subplot(grid[0, col])
        frame = detection.loc[detection["held_out_label"].eq(label)]
        for y, method in enumerate(order):
            score = dict((m, s) for m, s, _ in METHODS)[method]
            values = frame.loc[frame["method"].eq(method) & frame["score"].eq(score), "auprc"]
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            ax.scatter(values, np.full(len(values), y), s=10, alpha=0.35, color=COLORS[method], linewidth=0)
            ax.errorbar(mean, y, xerr=sd, fmt="o", ms=5, color=COLORS[method], mec="#222222", mew=0.5, capsize=2, lw=0.8)
        prevalence = float(frame.loc[frame["method"].eq("coreot_full") & frame["score"].eq("u"), "prevalence"].mean())
        ax.axvline(prevalence, color="#777777", linestyle="--", lw=0.8)
        ax.text(
            prevalence + 0.01,
            len(order) - 0.7,
            f"prevalence = {prevalence:.3f}",
            rotation=90,
            fontsize=6.5,
            color="#555555",
            va="bottom",
            ha="left",
        )
        ax.axhline(4.5, color="#D0D0D0", lw=0.7)
        ax.set(xlim=(0, 1), ylim=(len(order) - 0.5, -0.8), xlabel="Within-cDC2 AUPRC", title=label)
        ax.title.set_fontsize(7)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.grid(axis="x", color="#E5E5E5", lw=0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([DISPLAY[m] for m in order] if col == 0 else [])
        if col == 0:
            _panel_letter(ax, "C")


def _draw_rescue(container, rescue: pd.DataFrame) -> None:
    grid = container.subgridspec(1, 2, wspace=0.42)
    maximum = float(rescue["delta_u_rescue_mean"].abs().max()) * 1.15
    groups = (("held_out_positive", "Held-out"), ("shared_cDC2", "Represented\ncDC2"))
    for col, label in enumerate(LABELS):
        ax = plt.subplot(grid[0, col])
        frame = rescue.loc[rescue["held_out_label"].eq(label)]
        seeds = sorted(frame["seed"].astype(int).unique())
        offsets = np.linspace(-0.14, 0.14, len(seeds))
        by_seed = frame.set_index(["seed", "truth_group"])["delta_u_rescue_mean"]
        for offset, seed in zip(offsets, seeds, strict=True):
            held_value = float(by_seed.loc[(seed, "held_out_positive")])
            shared_value = float(by_seed.loc[(seed, "shared_cDC2")])
            ax.plot(
                (offset, 1 + offset),
                (held_value, shared_value),
                color="#999999",
                alpha=0.45,
                lw=0.55,
                zorder=1,
            )
        for x, (group, _) in enumerate(groups):
            values = pd.Series([by_seed.loc[(seed, group)] for seed in seeds], dtype=float)
            x_values = offsets if x == 0 else 1 + offsets
            ax.scatter(x_values, values.to_numpy(), s=13, alpha=0.55, color=COLORS["coreot_full"], linewidth=0, zorder=2)
            ax.scatter(x, values.mean(), s=36, color=COLORS["coreot_full"], edgecolor="#222222", linewidth=0.7, zorder=3)
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set(xlim=(-0.45, 1.45), ylim=(-maximum, maximum), title=label)
        if col == 0:
            ax.set_xticks((0, 1), [name for _, name in groups], fontsize=6.0)
        else:
            ax.set_xticks((0, 1), ("", ""))
        ax.tick_params(axis="y", labelsize=6.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.03,
            0.96,
            "positive = lower deficit",
            transform=ax.transAxes,
            fontsize=6.5,
            color="#555555",
            ha="left",
            va="top",
        )
        if col == 0:
            ax.set_ylabel("Deficit decrease\n" + r"$\Delta u=u^{\mathrm{ablated}}-u^{\mathrm{full}}$", fontsize=6.5)
            _panel_letter(ax, "D")
        else:
            ax.set_yticklabels([])


def _draw_operational(container, operational: pd.DataFrame) -> None:
    grid = container.subgridspec(1, 2, wspace=0.12)
    xlim = _outward_limits(operational["coverage"])
    ylim = _outward_limits(operational["absent_abstention_rate"])
    for col, label in enumerate(LABELS):
        ax = plt.subplot(grid[0, col])
        frame = operational.loc[operational["held_out_label"].eq(label)]
        for method, score, _ in OPERATIONAL:
            values = frame.loc[frame["method"].eq(method) & frame["score"].eq(score)]
            ax.scatter(values["coverage"], values["absent_abstention_rate"], s=11, alpha=0.35, color=COLORS[method], linewidth=0)
            ax.scatter(values["coverage"].mean(), values["absent_abstention_rate"].mean(), s=35, color=COLORS[method], edgecolor="#222222", linewidth=0.6)
        ax.set(xlim=xlim, ylim=ylim, title=label)
        ax.title.set_fontsize(7)
        ax.tick_params(axis="both", labelsize=6.2)
        ax.grid(color="#E5E5E5", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(
            "higher coverage\nand abstention ↗",
            xy=(0.98, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=6.5,
            color="#555555",
        )
        if col == 0:
            ax.set_ylabel("Held-out-state\nabstention", fontsize=7)
            _panel_letter(ax, "E")
        else:
            ax.set_yticklabels([])


def _render(embedding: pd.DataFrame, detection: pd.DataFrame, rescue: pd.DataFrame, operational: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7})
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=False)
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 1, height_ratios=(1.04, 0.96), hspace=0.43)
    top = outer[0].subgridspec(1, 2, width_ratios=(22, 78), wspace=0.08)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(46, 27, 27), wspace=0.18)
    _draw_schematic(fig.add_subplot(top[0, 0]))
    _draw_embedding_grid(top[0, 1], embedding)
    _draw_detection(bottom[0, 0], detection)
    _draw_rescue(bottom[0, 1], rescue)
    _draw_operational(bottom[0, 2], operational)
    fig.text(0.17, 0.973, "Benchmark design", ha="center", fontsize=8.5)
    fig.text(0.61, 0.973, "Calibrated abstention on fixed query-only embeddings", ha="center", fontsize=8.0)
    fig.text(0.61, 0.947, "Seed 1 fixed in advance; coordinates reused within each row", ha="center", fontsize=6.2, color="#666666")
    fig.text(0.345, 0.79, "Held out:\nHLA-DRhi cDC2", ha="right", va="center", fontsize=6.5)
    fig.text(0.345, 0.66, "Held out:\nISG+ cDC2", ha="right", va="center", fontsize=6.5)
    fig.text(0.275, 0.505, "Within-cDC2 AUPRC", ha="center", fontsize=8.5)
    fig.text(0.655, 0.505, "Deficit decrease", ha="center", fontsize=8.5)
    fig.text(0.875, 0.505, "Abstention vs coverage", ha="center", fontsize=8.5)
    status_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=RETAINED_FILL, markeredgecolor="none", markersize=4.5, label="shared, retained"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=ABSTAINED_FILL, markeredgecolor="none", markersize=4.5, label="shared, abstained"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=RETAINED_FILL, markeredgecolor=TRUTH_EDGE, markeredgewidth=0.7, markersize=4.5, label="held-out, retained"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=ABSTAINED_FILL, markeredgecolor=TRUTH_EDGE, markeredgewidth=0.7, markersize=4.5, label="held-out, abstained"),
    ]
    fig.legend(status_handles, [handle.get_label() for handle in status_handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.68, 0.575), fontsize=6.1, handletextpad=0.22, columnspacing=0.55)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS[m], markeredgecolor="#222222", markersize=5, label=name)
        for m, _, name in OPERATIONAL
    ]
    fig.text(0.875, 0.145, "Shared-cell coverage", ha="center", fontsize=6.5)
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.77, 0.018), fontsize=6.5)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.91, bottom=0.15)
    for suffix in ("png", "tiff", "pdf", "svg"):
        path = FIGURES / f"hiha_main_results.{suffix}"
        fig.savefig(path, dpi=RASTER_DPI if suffix in {"png", "tiff"} else None, facecolor="white")
    plt.close(fig)


def _validate_rendered_outputs() -> None:
    expected_minimum = (2453, 1626)
    for suffix in ("png", "tiff"):
        path = FIGURES / f"hiha_main_results.{suffix}"
        with Image.open(path) as image:
            if image.width < expected_minimum[0] or image.height < expected_minimum[1]:
                raise HIHAMainFigureError(
                    f"{path} is below the 178 mm by 118 mm, 350-dpi target: {image.size}"
                )
            if image.width / image.height < 1.45 or image.width / image.height > 1.58:
                raise HIHAMainFigureError(f"Unexpected publication aspect ratio for {path}: {image.size}")


def generate_hiha_main_figure() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    detection = _selected_detection()
    rescue = _selected_rescue()
    operational = _selected_operational(detection)
    embedding = _embedding_data(detection)
    _validate_embedding_operating_rates(embedding, operational)
    embedding.to_csv(DATA / "hiha_main_results_embedding_abstention.csv", index=False)
    detection.to_csv(DATA / "hiha_main_results_detection.csv", index=False)
    rescue.to_csv(DATA / "hiha_main_results_rescue.csv", index=False)
    operational.to_csv(DATA / "hiha_main_results_operational.csv", index=False)
    _render(embedding, detection, rescue, operational)
    for path in (
        FIGURES / "hiha_main_results.png",
        FIGURES / "hiha_main_results.tiff",
        FIGURES / "hiha_main_results.pdf",
        FIGURES / "hiha_main_results.svg",
        DATA / "hiha_main_results_embedding_abstention.csv",
        DATA / "hiha_main_results_detection.csv",
        DATA / "hiha_main_results_rescue.csv",
        DATA / "hiha_main_results_operational.csv",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAMainFigureError(f"Missing or empty figure artifact: {path}")
    _validate_rendered_outputs()
    return 0
