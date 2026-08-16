from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from coreot.results.pbmc_figure3 import (
    CELL_TYPE_COLORS,
    DOCS_ROOT,
    ENDPOINTS,
    ENDPOINT_SHORT,
    INK,
    LIGHT_EDGE,
    METHOD_COLORS,
    PANEL_B_METHODS,
    PANEL_B_AP_DISPLAY_LIMITS,
    PANEL_B_AP_TICKS,
    PANEL_C_PROBABILITY_DISPLAY_LIMITS,
    PANEL_D_METRICS,
    PANEL_D_SEED,
    PANEL_E_CONTEXT_GRAY,
    PANEL_E_ELIGIBLE_GRAY,
    PANEL_E_MAPS,
    PANEL_E_METHODS,
    PANEL_E_TRUTH_COLOR,
    PANEL_F_MAPS,
    PANEL_F_METHODS,
    PANEL_RASTER_DPI,
    PBMCFigure3Error,
    QUERY_FILL,
    REFERENCE_FILL,
    RESTORED,
    RESULT_ROOT,
    REVIEW_ROOT,
    SEEDS,
    SOURCE_DATA_ROOT,
    STIMULATED,
    _file_sha256,
    _git_dirty,
    _git_head,
    _software_versions,
)


MAIN_FIGURE_SIZE_INCHES = (178 / 25.4, 230 / 25.4)
MAIN_FIGURE_MIN_FONT_SIZE = 6.5
PANEL_SET = ("A", "B", "C", "D", "E", "F")
PANEL_BOUNDS = {
    "A": (0.015, 0.775, 0.305, 0.215),
    "B": (0.305, 0.775, 0.685, 0.215),
    "C": (0.015, 0.475, 0.485, 0.285),
    "D": (0.505, 0.455, 0.485, 0.305),
    "E": (0.005, 0.255, 0.990, 0.210),
    "F": (0.005, 0.030, 0.990, 0.215),
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    panel: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PBMCFigure3Error(
            f"Composite Panel {panel} source is missing columns {missing}."
        )
    if frame.empty:
        raise PBMCFigure3Error(f"Composite Panel {panel} source is empty.")


def _require_exact_keys(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    expected: set[tuple[object, ...]],
    panel: str,
) -> None:
    if frame.duplicated(columns).any():
        raise PBMCFigure3Error(
            f"Composite Panel {panel} source has duplicate keys {columns}."
        )
    observed = set(frame[columns].itertuples(index=False, name=None))
    if observed != expected:
        raise PBMCFigure3Error(
            f"Composite Panel {panel} keys differ from the figure contract; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}."
        )


def read_composite_sources(source_root: Path = SOURCE_DATA_ROOT) -> dict[str, object]:
    paths = {
        "A": source_root / "panel_a_design.csv",
        "B": source_root / "figure_3_within_celltype_detection_by_seed.csv",
        "C": source_root / "figure_3_matched_reference_by_seed.csv",
        "D": source_root / "figure_3_represented_celltype_transfer_by_seed.csv",
        "E": source_root / "figure_3_weak_support_umap_cells.parquet",
        "F": source_root / "figure_3_label_assignment_umap_cells.parquet",
        "F_summary": source_root / "figure_3_label_assignment_umap_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise PBMCFigure3Error(
            "Generate all Figure 3 panel sources before rendering; "
            f"missing={missing}."
        )
    panel_a = pd.read_csv(paths["A"])
    panel_b = pd.read_csv(paths["B"])
    panel_c = pd.read_csv(paths["C"])
    panel_d = pd.read_csv(paths["D"])
    panel_e = pd.read_parquet(paths["E"])
    panel_f = pd.read_parquet(paths["F"])
    panel_f_summary = pd.read_csv(paths["F_summary"])

    _require_exact_keys(
        panel_a,
        columns=["field"],
        expected={
            ("displayed_endpoints",),
            ("split",),
            ("ablated_reference",),
            ("full_reference",),
            ("primary_evaluation",),
        },
        panel="A",
    )
    _require_columns(
        panel_b,
        {"endpoint", "seed", "method", "score", "prevalence", "auprc"},
        panel="B",
    )
    _require_exact_keys(
        panel_b,
        columns=["endpoint", "seed", "method", "score"],
        expected={
            (endpoint, seed, method, score)
            for endpoint in ENDPOINTS
            for seed in SEEDS
            for method, score, _ in PANEL_B_METHODS
        },
        panel="B",
    )
    _require_columns(
        panel_c,
        {
            "endpoint",
            "seed",
            "heldout_u_ablated",
            "heldout_u_full",
            "control_u_ablated",
            "control_u_full",
            "restoration_specificity",
            "restored_state_conditional_probability",
        },
        panel="C",
    )
    _require_exact_keys(
        panel_c,
        columns=["endpoint", "seed"],
        expected={(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS},
        panel="C",
    )
    _require_columns(
        panel_d,
        {
            "endpoint",
            "seed",
            "method",
            "score",
            "forced_macro_f1",
            "forced_accuracy",
        },
        panel="D",
    )
    _require_exact_keys(
        panel_d,
        columns=["endpoint", "seed", "method", "score"],
        expected={
            (endpoint, seed, method, score)
            for endpoint in ENDPOINTS
            for seed in SEEDS
            for method, score, _ in PANEL_F_METHODS
        },
        panel="D",
    )
    _require_columns(
        panel_e,
        {
            "endpoint",
            "seed",
            "cell_id",
            "map_id",
            "is_evaluation_cohort",
            "is_selected",
            "umap_1",
            "umap_2",
        },
        panel="E",
    )
    expected_e_maps = {
        (endpoint, PANEL_D_SEED, map_id)
        for endpoint in ENDPOINTS
        for map_id, _ in PANEL_E_MAPS
    }
    observed_e_maps = set(
        panel_e[["endpoint", "seed", "map_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_e_maps != expected_e_maps:
        raise PBMCFigure3Error("Composite Panel E maps differ from the contract.")
    if panel_e.duplicated(["endpoint", "map_id", "cell_id"]).any():
        raise PBMCFigure3Error("Composite Panel E contains duplicate map-cell rows.")
    _require_columns(
        panel_f,
        {
            "endpoint",
            "seed",
            "cell_id",
            "map_id",
            "is_represented",
            "is_held_out",
            "displayed_assignment",
            "umap_1",
            "umap_2",
        },
        panel="F",
    )
    expected_f_maps = {
        (endpoint, PANEL_D_SEED, map_id)
        for endpoint in ENDPOINTS
        for map_id, _ in PANEL_F_MAPS
    }
    observed_f_maps = set(
        panel_f[["endpoint", "seed", "map_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_f_maps != expected_f_maps:
        raise PBMCFigure3Error("Composite Panel F maps differ from the contract.")
    if panel_f.duplicated(["endpoint", "map_id", "cell_id"]).any():
        raise PBMCFigure3Error("Composite Panel F contains duplicate map-cell rows.")
    _require_columns(
        panel_f_summary,
        {
            "endpoint",
            "seed",
            "method",
            "forced_accuracy",
            "forced_macro_f1",
            "panel_d_forced_accuracy",
            "panel_d_forced_macro_f1",
        },
        panel="F summary",
    )
    _require_exact_keys(
        panel_f_summary,
        columns=["endpoint", "seed", "method"],
        expected={
            (endpoint, PANEL_D_SEED, method)
            for endpoint in ENDPOINTS
            for method, _, _ in PANEL_F_METHODS
        },
        panel="F summary",
    )
    panel_d_seed = (
        panel_d.loc[panel_d["seed"].eq(PANEL_D_SEED)]
        .sort_values(["endpoint", "method"])
        .reset_index(drop=True)
    )
    panel_f_summary = panel_f_summary.sort_values(
        ["endpoint", "method"]
    ).reset_index(drop=True)
    if not panel_d_seed[["endpoint", "method"]].equals(
        panel_f_summary[["endpoint", "method"]]
    ):
        raise PBMCFigure3Error(
            "Panel F assignment summaries are not aligned with Panel D seed-1 rows."
        )
    for metric in ("forced_accuracy", "forced_macro_f1"):
        panel_d_values = panel_d_seed[metric].to_numpy(dtype=float)
        panel_f_values = panel_f_summary[metric].to_numpy(dtype=float)
        panel_f_lineage = panel_f_summary[f"panel_d_{metric}"].to_numpy(dtype=float)
        if not (
            np.allclose(panel_f_values, panel_d_values, rtol=0.0, atol=1e-12)
            and np.allclose(panel_f_lineage, panel_d_values, rtol=0.0, atol=1e-12)
        ):
            raise PBMCFigure3Error(
                f"Panel F {metric} values do not reproduce Panel D seed-1 values."
            )

    e_truth = (
        panel_e.loc[panel_e["map_id"].eq("truth")]
        .sort_values(["endpoint", "cell_id"])
        [["endpoint", "cell_id", "umap_1", "umap_2"]]
        .reset_index(drop=True)
    )
    f_truth = (
        panel_f.loc[panel_f["map_id"].eq("truth")]
        .sort_values(["endpoint", "cell_id"])
        [["endpoint", "cell_id", "umap_1", "umap_2"]]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(e_truth, f_truth, check_dtype=False)
    return {
        "A": panel_a.set_index("field")["value"].astype(str).to_dict(),
        "B": panel_b,
        "C": panel_c,
        "D": panel_d,
        "E": panel_e,
        "F": panel_f,
        "F_summary": panel_f_summary,
        "paths": paths,
    }


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = LIGHT_EDGE,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.8,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        color=INK,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.8,
            color="#666666",
        )
    )


def _draw_panel_a(fig: plt.Figure, design: dict[str, str]) -> None:
    ax = fig.add_axes((0.025, 0.785, 0.275, 0.20))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.0, 0.99, "A", fontsize=10, fontweight="bold", ha="left", va="top")
    _box(
        ax,
        0.16,
        0.80,
        0.68,
        0.13,
        "IFN-β PBMCs\nB · NK · DC endpoints",
        facecolor=REFERENCE_FILL,
    )
    _box(
        ax,
        0.16,
        0.62,
        0.68,
        0.12,
        "Donor-aware split",
        facecolor=REFERENCE_FILL,
    )
    _arrow(ax, (0.50, 0.80), (0.50, 0.745))
    _box(
        ax,
        0.03,
        0.31,
        0.43,
        0.25,
        "Complete query\nall states retained",
        facecolor=QUERY_FILL,
    )
    _box(
        ax,
        0.54,
        0.31,
        0.43,
        0.25,
        "",
        facecolor=REFERENCE_FILL,
    )
    ax.text(
        0.755,
        0.49,
        "Reference",
        ha="center",
        va="center",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        color=INK,
    )
    ax.text(
        0.755,
        0.415,
        "Ablated: − Stim",
        ha="center",
        va="center",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        color=STIMULATED,
    )
    ax.text(
        0.755,
        0.345,
        "Full: + Stim",
        ha="center",
        va="center",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        color=RESTORED,
    )
    _arrow(ax, (0.43, 0.62), (0.25, 0.565))
    _arrow(ax, (0.57, 0.62), (0.75, 0.565))
    _box(
        ax,
        0.12,
        0.03,
        0.76,
        0.14,
        "Identical query\nwithin-type ranking · response",
        facecolor="#F7F7F7",
        edgecolor=INK,
    )
    _arrow(ax, (0.25, 0.31), (0.40, 0.175))
    _arrow(ax, (0.75, 0.31), (0.60, 0.175))
    if "same query" not in design["split"].lower():
        raise PBMCFigure3Error("Panel A source must record the identical query.")


def _style_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=MAIN_FIGURE_MIN_FONT_SIZE, length=2, pad=1)


def _draw_panel_b(fig: plt.Figure, panel_b: pd.DataFrame) -> None:
    fig.text(0.32, 0.985, "B", fontsize=10, fontweight="bold", ha="left", va="top")
    methods = [method for method, _, _ in PANEL_B_METHODS]
    labels = {
        method: ("Seurat" if method == "seurat_anchor" else display)
        for method, _, display in PANEL_B_METHODS
    }
    positions = np.arange(len(methods), dtype=float)
    x_positions = (0.405, 0.600, 0.795)
    for index, (x0, endpoint) in enumerate(zip(x_positions, ENDPOINTS, strict=True)):
        ax = fig.add_axes((x0, 0.835, 0.160, 0.115))
        local = panel_b.loc[panel_b["endpoint"].eq(endpoint)]
        for y, method in enumerate(methods):
            values = (
                local.loc[local["method"].eq(method)]
                .sort_values("seed")["auprc"]
                .to_numpy(dtype=float)
            )
            offsets = np.linspace(-0.13, 0.13, len(values))
            ax.scatter(
                values,
                y + offsets,
                s=9,
                color=METHOD_COLORS[method],
                alpha=0.45,
                linewidth=0,
                zorder=3,
            )
            ax.errorbar(
                float(values.mean()),
                y,
                xerr=float(values.std(ddof=1)),
                fmt="o",
                markersize=3.8,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor=INK,
                markeredgewidth=0.45,
                ecolor=METHOD_COLORS[method],
                elinewidth=0.8,
                capsize=2,
                zorder=4,
            )
        ax.axvline(
            float(local.groupby("seed")["prevalence"].first().mean()),
            color="#777777",
            linestyle=(0, (2.5, 2.0)),
            linewidth=0.7,
        )
        ax.set(
            xlim=PANEL_B_AP_DISPLAY_LIMITS,
            ylim=(len(methods) - 0.5, -0.5),
            xticks=PANEL_B_AP_TICKS,
            yticks=positions,
        )
        ax.set_yticklabels(
            [labels[method] for method in methods] if index == 0 else []
        )
        ax.set_title(
            f"{ENDPOINT_SHORT[endpoint]} held out",
            fontsize=7.0,
            pad=3,
        )
        _style_axis(ax)
    fig.text(
        0.69,
        0.812,
        "Average precision (AP) for held-out-state ranking",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        ha="center",
        va="top",
        color=INK,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#777777",
                markeredgecolor="none",
                alpha=0.45,
                markersize=3.5,
                label="Donor split",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                color="#777777",
                markerfacecolor="#777777",
                markeredgecolor=INK,
                markeredgewidth=0.4,
                markersize=4,
                linewidth=0.8,
                label="Mean ± sample SD",
            ),
            Line2D(
                [0],
                [0],
                linestyle=(0, (2.5, 2.0)),
                color="#777777",
                linewidth=0.7,
                label="Mean prevalence",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.69, 0.777),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def _draw_panel_c(fig: plt.Figure, panel_c: pd.DataFrame) -> None:
    fig.text(0.025, 0.755, "C", fontsize=10, fontweight="bold", ha="left", va="top")
    data = panel_c.copy()
    data["heldout_delta_u"] = (
        data["heldout_u_ablated"] - data["heldout_u_full"]
    )
    data["control_delta_u"] = (
        data["control_u_ablated"] - data["control_u_full"]
    )
    rescue_ax = fig.add_axes((0.19, 0.515, 0.20, 0.205))
    probability_ax = fig.add_axes((0.425, 0.515, 0.055, 0.205), sharey=rescue_ax)
    rows = (
        ("heldout_delta_u", "Held-out cells", METHOD_COLORS["coreot_full"]),
        ("control_delta_u", "Same-type controls", "#777777"),
        (
            "restoration_specificity",
            "Control-adjusted, $\\Delta^{\\mathrm{CA}}$",
            STIMULATED,
        ),
    )
    positions = {
        ENDPOINTS[0]: np.array([8.4, 7.4, 6.4]),
        ENDPOINTS[1]: np.array([4.8, 3.8, 2.8]),
        ENDPOINTS[2]: np.array([1.2, 0.2, -0.8]),
    }
    offsets = np.linspace(-0.14, 0.14, len(SEEDS))
    plotted = data[[column for column, _, _ in rows]].to_numpy(dtype=float)
    lower = min(-0.01, float(np.floor((plotted.min() - 0.01) / 0.02) * 0.02))
    upper = max(0.02, float(np.ceil((plotted.max() + 0.01) / 0.02) * 0.02))
    for endpoint in ENDPOINTS:
        local = data.loc[data["endpoint"].eq(endpoint)].sort_values("seed")
        center = float(positions[endpoint].mean())
        for position, (column, _, color) in zip(
            positions[endpoint], rows, strict=True
        ):
            values = local[column].to_numpy(dtype=float)
            rescue_ax.scatter(
                values,
                position + offsets,
                s=9,
                color=color,
                alpha=0.45,
                linewidth=0,
            )
            rescue_ax.errorbar(
                float(values.mean()),
                position,
                xerr=float(values.std(ddof=1)),
                fmt="o",
                markersize=3.8,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.45,
                ecolor=color,
                elinewidth=0.8,
                capsize=2,
            )
        probabilities = local[
            "restored_state_conditional_probability"
        ].to_numpy(dtype=float)
        probability_ax.scatter(
            probabilities,
            center + offsets,
            s=9,
            color=RESTORED,
            alpha=0.45,
            linewidth=0,
        )
        probability_ax.errorbar(
            float(probabilities.mean()),
            center,
            xerr=float(probabilities.std(ddof=1)),
            fmt="o",
            markersize=3.8,
            markerfacecolor=RESTORED,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            ecolor=RESTORED,
            elinewidth=0.8,
            capsize=2,
        )
        y_fraction = (center + 1.0) / 10.0
        fig.text(
            0.025,
            0.515 + 0.205 * y_fraction,
            f"{ENDPOINT_SHORT[endpoint]}\nheld out",
            fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
            fontweight="bold",
            linespacing=0.9,
            rotation=90,
            ha="center",
            va="center",
            color=INK,
        )
    row_positions = np.concatenate([positions[endpoint] for endpoint in ENDPOINTS])
    rescue_ax.axvline(0, color=INK, linewidth=0.7)
    rescue_ax.axhline(5.6, color="#B8B8B8", linewidth=0.7)
    rescue_ax.axhline(2.0, color="#B8B8B8", linewidth=0.7)
    rescue_ax.set(
        xlim=(lower, upper),
        ylim=(-1.3, 8.9),
        yticks=row_positions,
        yticklabels=[label for _ in ENDPOINTS for _, label, _ in rows],
    )
    rescue_ax.set_xlabel(
        "Median-deficit decrease",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        labelpad=3,
    )
    _style_axis(rescue_ax)
    for boundary in (5.6, 2.0):
        probability_ax.axhline(boundary, color="#B8B8B8", linewidth=0.7)
    probability_ax.set(
        xlim=PANEL_C_PROBABILITY_DISPLAY_LIMITS,
        ylim=(-1.3, 8.9),
        xticks=(0, 0.5, 1),
    )
    probability_ax.tick_params(
        axis="y", left=False, labelleft=False
    )
    probability_ax.set_xlabel(
        "Restored-state\nprobability",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        labelpad=3,
    )
    _style_axis(probability_ax)
    probability_ax.spines["left"].set_visible(False)


def _draw_panel_d(fig: plt.Figure, panel_d: pd.DataFrame) -> None:
    fig.text(0.515, 0.755, "D", fontsize=10, fontweight="bold", ha="left", va="top")
    methods = [method for method, _, _ in PANEL_F_METHODS]
    labels = {
        method: ("Seurat" if method == "seurat_anchor" else display)
        for method, _, display in PANEL_F_METHODS
    }
    positions = np.arange(len(methods), dtype=float)
    offsets = (-0.15, 0.15)
    y_positions = (0.655, 0.575, 0.495)
    for index, (y0, endpoint) in enumerate(zip(y_positions, ENDPOINTS, strict=True)):
        ax = fig.add_axes((0.635, y0, 0.325, 0.062))
        local = panel_d.loc[panel_d["endpoint"].eq(endpoint)]
        for method_index, method in enumerate(methods):
            method_rows = local.loc[local["method"].eq(method)].sort_values("seed")
            for metric_index, (metric, _, color) in enumerate(PANEL_D_METRICS):
                values = method_rows[metric].to_numpy(dtype=float)
                mean = float(values.mean())
                center = positions[method_index] + offsets[metric_index]
                ax.barh(
                    center,
                    mean - 0.70,
                    left=0.70,
                    height=0.18,
                    color=color,
                    alpha=0.48,
                    edgecolor=color,
                    linewidth=0.55,
                )
                ax.errorbar(
                    mean,
                    center,
                    xerr=float(values.std(ddof=1)),
                    fmt="none",
                    ecolor=INK,
                    elinewidth=0.7,
                    capsize=1.8,
                    capthick=0.7,
                )
        ax.set(
            xlim=(0.70, 1.0),
            ylim=(len(methods) - 0.5, -0.5),
            xticks=np.arange(0.70, 1.001, 0.10),
            yticks=positions,
        )
        ax.set_yticklabels([labels[method] for method in methods])
        ax.set_title(
            f"{ENDPOINT_SHORT[endpoint]} held out",
            fontsize=7.0,
            pad=3,
        )
        if index < len(ENDPOINTS) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", length=0, pad=2)
        _style_axis(ax)
        ax.spines["left"].set_visible(False)
    fig.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor=color,
                linewidth=0.55,
                alpha=0.55,
                label=display,
            )
            for _, display, color in PANEL_D_METRICS
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.75, 0.752),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.text(
        0.80,
        0.477,
        "Metric value (bars start at 0.70)",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        ha="center",
        va="top",
        color=INK,
    )


def _umap_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_span = max(float(frame["umap_1"].max() - frame["umap_1"].min()), 1e-6)
    y_span = max(float(frame["umap_2"].max() - frame["umap_2"].min()), 1e-6)
    return (
        (
            float(frame["umap_1"].min() - 0.025 * x_span),
            float(frame["umap_1"].max() + 0.025 * x_span),
        ),
        (
            float(frame["umap_2"].min() - 0.025 * y_span),
            float(frame["umap_2"].max() + 0.025 * y_span),
        ),
    )


def _draw_panel_e(fig: plt.Figure, panel_e: pd.DataFrame) -> None:
    fig.text(0.012, 0.46, "E", fontsize=10, fontweight="bold", ha="left", va="top")
    x_positions = np.linspace(0.105, 0.87, len(PANEL_E_MAPS))
    y_positions = (0.390, 0.345, 0.300)
    width = 0.11
    height = 0.035
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_data = panel_e.loc[panel_e["endpoint"].eq(endpoint)]
        truth_rows = endpoint_data.loc[endpoint_data["map_id"].eq("truth")]
        xlim, ylim = _umap_limits(truth_rows)
        for column, (map_id, display) in enumerate(PANEL_E_MAPS):
            ax = fig.add_axes(
                (float(x_positions[column]), y_positions[row], width, height)
            )
            frame = endpoint_data.loc[endpoint_data["map_id"].eq(map_id)]
            outside = frame.loc[~frame["is_evaluation_cohort"]]
            eligible = frame.loc[
                frame["is_evaluation_cohort"] & ~frame["is_selected"]
            ]
            selected = frame.loc[frame["is_selected"]]
            ax.scatter(
                outside["umap_1"],
                outside["umap_2"],
                s=0.8,
                color=PANEL_E_CONTEXT_GRAY,
                alpha=0.65,
                linewidth=0,
                rasterized=True,
            )
            ax.scatter(
                eligible["umap_1"],
                eligible["umap_2"],
                s=1.1,
                color=PANEL_E_ELIGIBLE_GRAY,
                alpha=0.78,
                linewidth=0,
                rasterized=True,
            )
            highlight = (
                PANEL_E_TRUTH_COLOR if map_id == "truth" else METHOD_COLORS[map_id]
            )
            ax.scatter(
                selected["umap_1"],
                selected["umap_2"],
                s=1.8,
                color=highlight,
                alpha=0.9,
                linewidth=0,
                rasterized=True,
            )
            ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                title = "Seurat" if map_id == "seurat_anchor" else display
                ax.set_title(
                    title,
                    fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
                    pad=2,
                )
        fig.text(
            0.095,
            y_positions[row] + height / 2,
            f"{ENDPOINT_SHORT[endpoint]}\nheld out",
            fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
            fontweight="bold",
            linespacing=0.9,
            ha="right",
            va="center",
            color=INK,
        )
    method_handles = tuple(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="none",
            markersize=3.5,
        )
        for method, _, _ in PANEL_E_METHODS
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_TRUTH_COLOR,
                markeredgecolor="none",
                markersize=3.5,
            ),
            method_handles,
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_ELIGIBLE_GRAY,
                markeredgecolor="none",
                markersize=3.5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_CONTEXT_GRAY,
                markeredgecolor="none",
                markersize=3.5,
            ),
        ],
        labels=[
            "Held-out truth",
            "Method-specific top-ranked sets",
            "Within-type, not selected",
            "Outside within-type cohort",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.535, 0.270),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def _draw_panel_f(fig: plt.Figure, panel_f: pd.DataFrame) -> None:
    fig.text(0.012, 0.245, "F", fontsize=10, fontweight="bold", ha="left", va="top")
    x_positions = np.linspace(0.17, 0.805, len(PANEL_F_MAPS))
    y_positions = (0.185, 0.140, 0.095)
    width = 0.11
    height = 0.035
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_data = panel_f.loc[panel_f["endpoint"].eq(endpoint)]
        truth_rows = endpoint_data.loc[endpoint_data["map_id"].eq("truth")]
        xlim, ylim = _umap_limits(truth_rows)
        for column, (map_id, display) in enumerate(PANEL_F_MAPS):
            ax = fig.add_axes(
                (float(x_positions[column]), y_positions[row], width, height)
            )
            frame = endpoint_data.loc[endpoint_data["map_id"].eq(map_id)]
            represented = frame.loc[frame["is_represented"]].sort_values("cell_id")
            held_out = frame.loc[frame["is_held_out"]].sort_values("cell_id")
            colors = represented["displayed_assignment"].map(CELL_TYPE_COLORS)
            if colors.isna().any():
                raise PBMCFigure3Error(
                    "Composite Panel F contains unmapped biological labels."
                )
            ax.scatter(
                represented["umap_1"],
                represented["umap_2"],
                c=colors,
                s=1.5,
                alpha=0.80,
                linewidth=0,
                rasterized=True,
            )
            ax.scatter(
                held_out["umap_1"],
                held_out["umap_2"],
                color=CELL_TYPE_COLORS["Held-out state (not evaluated)"],
                s=2.5,
                alpha=1.0,
                linewidth=0,
                rasterized=True,
            )
            ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                title = "Seurat" if map_id == "seurat_anchor" else display
                ax.set_title(
                    title,
                    fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
                    pad=2,
                )
        fig.text(
            0.15,
            y_positions[row] + height / 2,
            f"{ENDPOINT_SHORT[endpoint]}\nheld out",
            fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
            fontweight="bold",
            linespacing=0.9,
            ha="right",
            va="center",
            color=INK,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=3.5,
            label=label,
        )
        for label, color in CELL_TYPE_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.535, 0.030),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.25,
        columnspacing=0.65,
    )


def draw_main_figure(sources: dict[str, object]) -> plt.Figure:
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
    _draw_panel_a(fig, sources["A"])
    _draw_panel_b(fig, sources["B"])
    _draw_panel_c(fig, sources["C"])
    _draw_panel_d(fig, sources["D"])
    _draw_panel_e(fig, sources["E"])
    _draw_panel_f(fig, sources["F"])
    return fig


def draw_panel_figure(sources: dict[str, object], letter: str) -> plt.Figure:
    drawers = {
        "A": _draw_panel_a,
        "B": _draw_panel_b,
        "C": _draw_panel_c,
        "D": _draw_panel_d,
        "E": _draw_panel_e,
        "F": _draw_panel_f,
    }
    if letter not in drawers:
        raise PBMCFigure3Error(f"Unknown Figure 3 panel: {letter}.")
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
    drawers[letter](fig, sources[letter])
    return fig


def _validate_fonts(fig: plt.Figure) -> None:
    sizes = [
        float(text.get_fontsize())
        for text in fig.findobj(match=Text)
        if text.get_text()
    ]
    if not sizes or min(sizes) < MAIN_FIGURE_MIN_FONT_SIZE:
        raise PBMCFigure3Error(
            "Main Figure 3 contains text below the final-size font floor; "
            f"minimum={min(sizes) if sizes else None}, "
            f"required={MAIN_FIGURE_MIN_FONT_SIZE}."
        )


def _validate_artist_bounds(fig: plt.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if (
            box.x0 < figure_box.x0 - 2
            or box.y0 < figure_box.y0 - 2
            or box.x1 > figure_box.x1 + 2
            or box.y1 > figure_box.y1 + 2
        ):
            raise PBMCFigure3Error(
                f"Figure 3 text lies outside the canvas: {text.get_text()!r}."
            )


def _save(fig: plt.Figure, outputs: dict[str, Path], *, bbox: Bbox | None = None) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
            bbox_inches=bbox,
        )


def _panel_bbox(fig: plt.Figure, letter: str) -> Bbox:
    x, y, width, height = PANEL_BOUNDS[letter]
    return Bbox.from_bounds(
        x * fig.get_figwidth(),
        y * fig.get_figheight(),
        width * fig.get_figwidth(),
        height * fig.get_figheight(),
    )


def generate_panel_images(
    *,
    letters: Iterable[str] = PANEL_SET,
    panel_root: Path = REVIEW_ROOT,
    source_root: Path = SOURCE_DATA_ROOT,
) -> dict[str, Path]:
    sources = read_composite_sources(source_root)
    outputs: dict[str, Path] = {}
    for raw_letter in letters:
        letter = raw_letter.upper()
        if letter not in PANEL_SET:
            raise PBMCFigure3Error(f"Unknown Figure 3 panel: {raw_letter}.")
        fig = draw_panel_figure(sources, letter)
        _validate_fonts(fig)
        _validate_artist_bounds(fig)
        panel_outputs = {
            suffix: panel_root
            / f"figure_3_pbmc_panel_{letter.lower()}.{suffix}"
            for suffix in ("png", "pdf", "svg")
        }
        _save(fig, panel_outputs, bbox=_panel_bbox(fig, letter))
        outputs[f"panel_{letter.lower()}"] = panel_outputs["png"]
        plt.close(fig)
    return outputs


def _validate_main_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected = tuple(
            round(value * PANEL_RASTER_DPI) for value in MAIN_FIGURE_SIZE_INCHES
        )
        if image.width < expected[0] - 4 or image.height < expected[1] - 4:
            raise PBMCFigure3Error(
                f"Main Figure 3 raster is below target: {path} has {image.size}, "
                f"expected {expected}."
            )
        aspect = image.width / image.height
        if not 0.75 <= aspect <= 0.80:
            raise PBMCFigure3Error(
                f"Unexpected Main Figure 3 aspect ratio: {path} has {image.size}."
            )


def generate_main_figure(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    source_root: Path = SOURCE_DATA_ROOT,
) -> dict[str, Path]:
    sources = read_composite_sources(source_root)
    fig = draw_main_figure(sources)
    _validate_fonts(fig)
    _validate_artist_bounds(fig)
    docs_figure_root = docs_root / "figs"
    docs_figure_root.mkdir(parents=True, exist_ok=True)
    docs_outputs = {
        suffix: docs_figure_root / f"manuscript_fig_pbmc_main.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    result_outputs = {
        suffix: result_root / f"figure_3_pbmc_main.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    _save(fig, docs_outputs)
    _save(fig, result_outputs)
    plt.close(fig)
    _validate_main_raster(docs_outputs["png"])

    source_paths = list(sources["paths"].values())
    manifest_path = result_root / "figure_3_pbmc_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "PBMC condition-specific weak correspondence Figure 3",
                "panel_set": list(PANEL_SET),
                "panel_contract": {
                    "D": "represented_celltype_label_transfer",
                    "E": "weak_support_umap",
                },
                "generator": (
                    "experiments/pbmc_state/"
                    "generate_pbmc_figure3_panels.py --panel main"
                ),
                "artifacts": {
                    "manuscript": {
                        suffix: str(path) for suffix, path in docs_outputs.items()
                    },
                    "result": {
                        suffix: str(path) for suffix, path in result_outputs.items()
                    },
                },
                "sources": [str(path) for path in source_paths],
                "source_data_sha256": {
                    str(path): _file_sha256(path) for path in source_paths
                },
                "parameters": {
                    "canvas_inches": list(MAIN_FIGURE_SIZE_INCHES),
                    "canvas_millimeters": [178, 230],
                    "raster_dpi": PANEL_RASTER_DPI,
                    "minimum_font_size_points": MAIN_FIGURE_MIN_FONT_SIZE,
                    "panel_d_axis": [0.70, 1.0],
                    "panel_e_representative_seed": PANEL_D_SEED,
                    "panel_f_representative_seed": PANEL_D_SEED,
                    "dense_umap_points_rasterized": True,
                },
                "software_versions": _software_versions(),
                "plotting_script": {
                    "path": str(Path(__file__)),
                    "git_commit": _git_head(),
                    "repository_worktree_dirty": _git_dirty(),
                    "sha256": _file_sha256(Path(__file__)),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        **{f"docs_{key}": value for key, value in docs_outputs.items()},
        **{f"result_{key}": value for key, value in result_outputs.items()},
        "manifest": manifest_path,
    }


def generate_review_panels(
    *,
    review_root: Path = REVIEW_ROOT,
    source_root: Path = SOURCE_DATA_ROOT,
) -> dict[str, Path]:
    readme = review_root / "README.md"
    if not readme.is_file():
        raise PBMCFigure3Error(f"Missing durable review log: {readme}.")
    outputs = generate_panel_images(
        letters=PANEL_SET,
        panel_root=review_root,
        source_root=source_root,
    )
    return {**outputs, "readme": readme}
