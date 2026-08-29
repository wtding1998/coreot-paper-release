from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    INK,
    LIGHT_EDGE,
    METHOD_COLORS,
    PANEL_B_METHODS,
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
    RESTORED,
    RESULT_ROOT,
    REVIEW_ROOT,
    SEEDS,
    SOURCE_DATA_ROOT,
    STIMULATED,
    _file_sha256,
    _software_versions,
)
from coreot.results.main_metric_bars import (
    FillMetricSpec,
    draw_endpoint_metric_facet,
    fill_metric_handles,
)


MAIN_FIGURE_SIZE_INCHES = (178 / 25.4, 230 / 25.4)
MAIN_FIGURE_MIN_FONT_SIZE = 6.5
PANEL_A_CONDITION_FONT_SIZE = 5.2
PANEL_A_MIN_FONT_SIZE = 4.5
PANEL_A_SMALL_TEXT_GIDS = frozenset(
    {
        "panel-a-scenario-title",
        "panel-a-scenario-states",
        "panel-a-split-label",
        "panel-a-query-title",
        "panel-a-query-detail",
        "panel-a-reference-donor-label",
        "panel-a-reference-omitted-label",
        "panel-a-reference-omitted-detail",
        "panel-a-restored-reference-label",
        "panel-a-restored-reference-detail",
    }
)
PANEL_B_ENDPOINT_FONT_SIZE = 5.0
PANEL_B_ENDPOINT_GIDS = frozenset(
    {
        "panel-b-endpoint-label-stimulated-b",
        "panel-b-endpoint-label-stimulated-nk",
        "panel-b-endpoint-label-stimulated-dc",
    }
)
PANEL_SPATIAL_ENDPOINT_FONT_SIZE = 5.5
PANEL_SPATIAL_ENDPOINT_GIDS = frozenset(
    f"panel-{panel}-endpoint-{endpoint}"
    for panel in ("e", "f")
    for endpoint in ("stimulated-b", "stimulated-nk", "stimulated-dc")
)
PANEL_SET = ("A", "B", "C", "D", "E", "F")
PANEL_C_METRICS = (
    FillMetricSpec("auprc", "AP", True),
    FillMetricSpec("auroc", "AUROC", False),
)
PANEL_D_BAR_METRICS = (
    FillMetricSpec("forced_accuracy", "Forced accuracy", True),
    FillMetricSpec("forced_macro_f1", "Forced macro-F1", False),
)
ENDPOINT_LABELS = {
    "B cells": "Stimulated B",
    "NK cells": "Stimulated NK",
    "Dendritic cells": "Stimulated DC",
}
PANEL_BOUNDS = {
    "A": (0.015, 0.775, 0.305, 0.215),
    "B": (0.305, 0.775, 0.685, 0.215),
    "C": (0.005, 0.455, 0.545, 0.305),
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
        raise PBMCFigure3Error(f"Composite Panel {panel} source is missing columns {missing}.")
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
        raise PBMCFigure3Error(f"Composite Panel {panel} source has duplicate keys {columns}.")
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
            f"Generate all Figure 3 panel sources before rendering; missing={missing}."
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
        {"endpoint", "seed", "method", "score", "prevalence", "auprc", "auroc"},
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
        (endpoint, PANEL_D_SEED, map_id) for endpoint in ENDPOINTS for map_id, _ in PANEL_E_MAPS
    }
    observed_e_maps = set(
        panel_e[["endpoint", "seed", "map_id"]].drop_duplicates().itertuples(index=False, name=None)
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
        (endpoint, PANEL_D_SEED, map_id) for endpoint in ENDPOINTS for map_id, _ in PANEL_F_MAPS
    }
    observed_f_maps = set(
        panel_f[["endpoint", "seed", "map_id"]].drop_duplicates().itertuples(index=False, name=None)
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
    panel_f_summary = panel_f_summary.sort_values(["endpoint", "method"]).reset_index(drop=True)
    if not panel_d_seed[["endpoint", "method"]].equals(panel_f_summary[["endpoint", "method"]]):
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
        .sort_values(["endpoint", "cell_id"])[["endpoint", "cell_id", "umap_1", "umap_2"]]
        .reset_index(drop=True)
    )
    f_truth = (
        panel_f.loc[panel_f["map_id"].eq("truth")]
        .sort_values(["endpoint", "cell_id"])[["endpoint", "cell_id", "umap_1", "umap_2"]]
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
    gid: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    patch.set_gid(gid)
    ax.add_patch(patch)
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
    if "same query" not in design["split"].lower():
        raise PBMCFigure3Error("Panel A source must record the identical query.")

    ax = fig.add_axes((0.025, 0.785, 0.275, 0.20))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    neutral_fill = "#F2F2F2"
    query_fill = "#EAF2F7"
    query_edge = "#6B9FBC"
    omitted_fill = "#FFF1EA"
    restored_fill = "#E9F5EF"
    gray_arrow = "#5E5E5E"

    def add_box(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        facecolor: str,
        edgecolor: str,
        gid: str,
        radius: float = 0.018,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0.85,
            edgecolor=edgecolor,
            facecolor=facecolor,
            zorder=3,
        )
        patch.set_gid(gid)
        ax.add_patch(patch)

    def add_line(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = gray_arrow,
        linewidth: float = 0.95,
    ) -> None:
        ax.plot(
            (start[0], end[0]),
            (start[1], end[1]),
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            clip_on=False,
            zorder=1,
        )

    def add_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = gray_arrow,
        linewidth: float = 0.95,
        mutation_scale: float = 6.5,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=mutation_scale,
                linewidth=linewidth,
                color=color,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0",
                clip_on=False,
                zorder=2,
            )
        )

    ax.text(
        0.026,
        0.962,
        "A",
        fontsize=9,
        fontweight="bold",
        fontfamily="Arial",
        ha="left",
        va="top",
    )

    scenario_x, scenario_y, scenario_w, scenario_h = 0.180, 0.835, 0.660, 0.122
    split_x, split_y, split_w, split_h = 0.180, 0.680, 0.660, 0.086
    query_x, query_y, query_w, query_h = 0.020, 0.240, 0.400, 0.245
    omit_x, omit_y, omit_w, omit_h = 0.515, 0.370, 0.415, 0.215
    restore_x, restore_y, restore_w, restore_h = 0.515, 0.075, 0.415, 0.215

    add_box(
        scenario_x,
        scenario_y,
        scenario_w,
        scenario_h,
        facecolor=neutral_fill,
        edgecolor=LIGHT_EDGE,
        gid="panel-a-source-box",
    )
    ax.text(
        scenario_x + scenario_w / 2,
        scenario_y + scenario_h * 0.66,
        "Separate scenarios",
        ha="center",
        va="center",
        fontsize=5.8,
        fontfamily="Arial",
        color=INK,
        gid="panel-a-scenario-title",
        zorder=5,
    )
    ax.text(
        scenario_x + scenario_w / 2,
        scenario_y + scenario_h * 0.34,
        "Stimulated B | NK | DC",
        ha="center",
        va="center",
        fontsize=5.45,
        fontfamily="Arial",
        color=INK,
        gid="panel-a-scenario-states",
        zorder=5,
    )

    add_box(
        split_x,
        split_y,
        split_w,
        split_h,
        facecolor=neutral_fill,
        edgecolor=LIGHT_EDGE,
        gid="panel-a-split-box",
        radius=0.014,
    )
    ax.text(
        split_x + split_w / 2,
        split_y + split_h / 2,
        "Five donor splits",
        ha="center",
        va="center",
        fontsize=6.0,
        fontfamily="Arial",
        color=INK,
        gid="panel-a-split-label",
        zorder=5,
    )
    add_arrow(
        (scenario_x + scenario_w / 2, scenario_y),
        (split_x + split_w / 2, split_y + split_h),
    )

    add_box(
        query_x,
        query_y,
        query_w,
        query_h,
        facecolor=query_fill,
        edgecolor=query_edge,
        gid="panel-a-query-box",
    )
    ax.text(
        query_x + query_w / 2,
        query_y + query_h * 0.69,
        "Same query cells",
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="semibold",
        fontfamily="Arial",
        color=INK,
        gid="panel-a-query-title",
        zorder=5,
    )
    ax.text(
        query_x + query_w / 2,
        query_y + query_h * 0.36,
        "All states\nretained",
        ha="center",
        va="center",
        fontsize=5.3,
        fontfamily="Arial",
        color=INK,
        linespacing=1.18,
        gid="panel-a-query-detail",
        zorder=5,
    )

    add_box(
        omit_x,
        omit_y,
        omit_w,
        omit_h,
        facecolor=omitted_fill,
        edgecolor=STIMULATED,
        gid="panel-a-reference-omitted-box",
    )
    ax.text(
        omit_x + omit_w / 2,
        omit_y + omit_h * 0.70,
        "Reference-omitted\ncondition",
        ha="center",
        va="center",
        fontsize=PANEL_A_CONDITION_FONT_SIZE,
        fontweight="semibold",
        fontfamily="Arial",
        color=STIMULATED,
        linespacing=1.06,
        gid="panel-a-reference-omitted-label",
        zorder=5,
    )
    ax.text(
        omit_x + omit_w / 2,
        omit_y + omit_h * 0.25,
        "Stimulated cells omitted\nSame-type controls retained",
        ha="center",
        va="center",
        fontsize=4.55,
        fontfamily="Arial",
        color=INK,
        linespacing=1.12,
        gid="panel-a-reference-omitted-detail",
        zorder=5,
    )

    add_box(
        restore_x,
        restore_y,
        restore_w,
        restore_h,
        facecolor=restored_fill,
        edgecolor=RESTORED,
        gid="panel-a-restored-reference-box",
    )
    ax.text(
        restore_x + restore_w / 2,
        restore_y + restore_h * 0.70,
        "Restored-reference\ncondition",
        ha="center",
        va="center",
        fontsize=PANEL_A_CONDITION_FONT_SIZE,
        fontweight="semibold",
        fontfamily="Arial",
        color=RESTORED,
        linespacing=1.06,
        gid="panel-a-restored-reference-label",
        zorder=5,
    )
    ax.text(
        restore_x + restore_w / 2,
        restore_y + restore_h * 0.25,
        "Stimulated cells restored\nSame-type controls retained",
        ha="center",
        va="center",
        fontsize=4.55,
        fontfamily="Arial",
        color=INK,
        linespacing=1.12,
        gid="panel-a-restored-reference-detail",
        zorder=5,
    )

    split_center_x = split_x + split_w / 2
    branch_y = 0.615
    query_branch_x = query_x + query_w / 2
    reference_trunk_x = 0.980
    omit_center_y = omit_y + omit_h / 2
    restore_center_y = restore_y + restore_h / 2

    add_line((split_center_x, split_y), (split_center_x, branch_y))
    add_line((split_center_x, branch_y), (query_branch_x, branch_y))
    add_arrow((query_branch_x, branch_y), (query_branch_x, query_y + query_h))
    add_line((split_center_x, branch_y), (reference_trunk_x, branch_y))
    ax.text(
        0.720,
        branch_y,
        "Same reference donors",
        ha="center",
        va="center",
        fontsize=PANEL_A_MIN_FONT_SIZE,
        fontfamily="Arial",
        color=INK,
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none"},
        gid="panel-a-reference-donor-label",
        zorder=7,
    )
    add_line((reference_trunk_x, branch_y), (reference_trunk_x, restore_center_y))
    add_arrow((reference_trunk_x, omit_center_y), (omit_x + omit_w, omit_center_y))
    add_arrow((reference_trunk_x, restore_center_y), (restore_x + restore_w, restore_center_y))

    add_arrow(
        (query_x + query_w, query_y + query_h * 0.66),
        (omit_x, omit_y + omit_h * 0.56),
        color=query_edge,
        linewidth=1.15,
        mutation_scale=7.0,
    )
    add_arrow(
        (query_x + query_w, query_y + query_h * 0.31),
        (restore_x, restore_y + restore_h * 0.54),
        color=query_edge,
        linewidth=1.15,
        mutation_scale=7.0,
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=MAIN_FIGURE_MIN_FONT_SIZE, length=2, pad=1)


def _draw_panel_c_metric_bars(fig: plt.Figure, panel_c: pd.DataFrame) -> None:
    fig.text(0.025, 0.755, "C", fontsize=10, fontweight="bold", ha="left", va="top")
    methods = tuple(method for method, _, _ in PANEL_B_METHODS)
    labels = {
        method: "scmap-\ncluster" if display == "scmap-cluster" else display
        for method, _, display in PANEL_B_METHODS
    }
    x_positions = (0.12, 0.275, 0.43)
    for index, (x0, endpoint) in enumerate(zip(x_positions, ENDPOINTS, strict=True)):
        axis = fig.add_axes((x0, 0.515, 0.105, 0.185))
        draw_endpoint_metric_facet(
            axis,
            panel_c,
            endpoint_column="endpoint",
            endpoint=endpoint,
            endpoint_label=ENDPOINT_LABELS[endpoint],
            method_order=methods,
            method_labels=labels,
            method_colors=METHOD_COLORS,
            metrics=PANEL_C_METRICS,
            ink=INK,
            grid_color="#E8E8E8",
            show_method_labels=index == 0,
            prevalence_column="prevalence",
            font_size=MAIN_FIGURE_MIN_FONT_SIZE,
        )
    fig.legend(
        handles=[
            *fill_metric_handles(PANEL_C_METRICS, color=INK),
            Line2D(
                [0],
                [0],
                color="#666666",
                linestyle=(0, (3, 2)),
                linewidth=0.9,
                label="Mean prevalence (AP)",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.30, 0.755),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handlelength=1.3,
        handletextpad=0.3,
        columnspacing=0.6,
    )
    fig.text(
        0.32,
        0.49,
        "Metric value",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        ha="center",
        va="top",
        color=INK,
    )


def _draw_panel_b_restoration(fig: plt.Figure, panel_b: pd.DataFrame) -> None:
    fig.text(0.32, 0.985, "B", fontsize=10, fontweight="bold", ha="left", va="top")
    data = panel_b.copy()
    data["heldout_delta_u"] = data["heldout_u_ablated"] - data["heldout_u_full"]
    data["control_delta_u"] = data["control_u_ablated"] - data["control_u_full"]
    axis = fig.add_axes((0.54, 0.805, 0.37, 0.15))
    rows = (
        (
            "heldout_delta_u",
            "Reference-omitted cells",
            METHOD_COLORS["coreot_full"],
        ),
        ("control_delta_u", "Same-type controls", "#777777"),
        (
            "restoration_specificity",
            "Control-adjusted $\\Delta^{\\mathrm{CA}}$",
            STIMULATED,
        ),
        (
            "restored_state_conditional_probability",
            "Restored-state destination fraction",
            RESTORED,
        ),
    )
    positions = {
        ENDPOINTS[0]: np.array([11.0, 10.2, 9.4, 8.6]),
        ENDPOINTS[1]: np.array([7.0, 6.2, 5.4, 4.6]),
        ENDPOINTS[2]: np.array([3.0, 2.2, 1.4, 0.6]),
    }
    for endpoint in ENDPOINTS:
        local = data.loc[data["endpoint"].eq(endpoint)].sort_values("seed")
        for position, (column, _, color) in zip(positions[endpoint], rows, strict=True):
            values = local[column].to_numpy(dtype=float)
            bars = axis.barh(
                position,
                float(values.mean()),
                xerr=float(values.std(ddof=1)),
                height=0.48,
                facecolor=color,
                edgecolor=color,
                linewidth=0.9,
                error_kw={
                    "ecolor": INK,
                    "elinewidth": 0.65,
                    "capsize": 1.5,
                    "capthick": 0.65,
                },
                zorder=2,
            )
            bars.patches[0].set_gid(
                f"panel-b-bar-{endpoint}-{column}".replace(" ", "_")
            )
    endpoint_labels = (
        (ENDPOINTS[0], 0.942, "panel-b-endpoint-label-stimulated-b"),
        (ENDPOINTS[1], 0.880, "panel-b-endpoint-label-stimulated-nk"),
        (ENDPOINTS[2], 0.818, "panel-b-endpoint-label-stimulated-dc"),
    )
    for endpoint, y_position, gid in endpoint_labels:
        fig.text(
            0.975,
            y_position,
            ENDPOINT_LABELS[endpoint],
            fontsize=PANEL_B_ENDPOINT_FONT_SIZE,
            fontweight="normal",
            rotation=270,
            ha="center",
            va="center",
            color=INK,
            gid=gid,
        )
    row_positions = np.concatenate([positions[endpoint] for endpoint in ENDPOINTS])
    axis.axhline(7.8, color="#B8B8B8", linewidth=0.7)
    axis.axhline(3.8, color="#B8B8B8", linewidth=0.7)
    axis.set(
        xlim=(0.0, 1.02),
        ylim=(0.1, 11.5),
        xticks=(0.0, 0.5, 1.0),
        yticks=row_positions,
        yticklabels=[label for _ in ENDPOINTS for _, label, _ in rows],
    )
    axis.tick_params(axis="x", labelsize=MAIN_FIGURE_MIN_FONT_SIZE, length=2, pad=1)
    axis.tick_params(axis="y", labelsize=MAIN_FIGURE_MIN_FONT_SIZE, length=0, pad=2.5)
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_xlabel(
        "Metric value",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        labelpad=2,
    )


def _draw_panel_d_metric_bars(fig: plt.Figure, panel_d: pd.DataFrame) -> None:
    fig.text(0.515, 0.755, "D", fontsize=10, fontweight="bold", ha="left", va="top")
    methods = tuple(method for method, _, _ in PANEL_F_METHODS)
    labels = {
        method: "scmap-\ncluster" if display == "scmap-cluster" else display
        for method, _, display in PANEL_F_METHODS
    }
    x_positions = (0.61, 0.745, 0.88)
    for index, (x0, endpoint) in enumerate(zip(x_positions, ENDPOINTS, strict=True)):
        axis = fig.add_axes((x0, 0.515, 0.10, 0.185))
        draw_endpoint_metric_facet(
            axis,
            panel_d,
            endpoint_column="endpoint",
            endpoint=endpoint,
            endpoint_label=ENDPOINT_LABELS[endpoint],
            method_order=methods,
            method_labels=labels,
            method_colors=METHOD_COLORS,
            metrics=PANEL_D_BAR_METRICS,
            ink=INK,
            grid_color="#E8E8E8",
            show_method_labels=index == 0,
            font_size=MAIN_FIGURE_MIN_FONT_SIZE,
        )
    fig.legend(
        handles=fill_metric_handles(PANEL_D_BAR_METRICS, color=INK),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.75, 0.755),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handlelength=1.3,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    fig.text(
        0.80,
        0.49,
        "Metric value",
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
    x_positions = np.linspace(0.070, 0.860, len(PANEL_E_MAPS))
    y_positions = (0.405, 0.3425, 0.280)
    width = 0.128
    height = 0.045
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_data = panel_e.loc[panel_e["endpoint"].eq(endpoint)]
        truth_rows = endpoint_data.loc[endpoint_data["map_id"].eq("truth")]
        xlim, ylim = _umap_limits(truth_rows)
        for column, (map_id, display) in enumerate(PANEL_E_MAPS):
            ax = fig.add_axes((float(x_positions[column]), y_positions[row], width, height))
            frame = endpoint_data.loc[endpoint_data["map_id"].eq(map_id)]
            outside = frame.loc[~frame["is_evaluation_cohort"]]
            eligible = frame.loc[frame["is_evaluation_cohort"] & ~frame["is_selected"]]
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
            highlight = PANEL_E_TRUTH_COLOR if map_id == "truth" else METHOD_COLORS[map_id]
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
        endpoint_label = {
            "B cells": "Stimulated B",
            "NK cells": "Stimulated NK",
            "Dendritic cells": "Stimulated DC",
        }[endpoint]
        fig.text(
            0.032,
            y_positions[row] + height / 2,
            endpoint_label,
            fontsize=PANEL_SPATIAL_ENDPOINT_FONT_SIZE,
            fontweight="normal",
            rotation=270,
            ha="center",
            va="center",
            color=INK,
            gid=f"panel-e-endpoint-{endpoint_label.lower().replace(' ', '-')}",
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
            "Reference-omitted cells",
            "Method-specific top-ranked $N_+$ cells",
            "Other ranking-cohort cells",
            "Outside ranking cohort",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.50, 0.242),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def _draw_panel_f(fig: plt.Figure, panel_f: pd.DataFrame) -> None:
    fig.text(0.012, 0.245, "F", fontsize=10, fontweight="bold", ha="left", va="top")
    x_positions = np.linspace(0.070, 0.860, len(PANEL_F_MAPS))
    y_positions = (0.180, 0.1125, 0.045)
    width = 0.128
    height = 0.050
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_data = panel_f.loc[panel_f["endpoint"].eq(endpoint)]
        truth_rows = endpoint_data.loc[endpoint_data["map_id"].eq("truth")]
        xlim, ylim = _umap_limits(truth_rows)
        for column, (map_id, display) in enumerate(PANEL_F_MAPS):
            ax = fig.add_axes((float(x_positions[column]), y_positions[row], width, height))
            frame = endpoint_data.loc[endpoint_data["map_id"].eq(map_id)]
            represented = frame.loc[frame["is_represented"]].sort_values("cell_id")
            held_out = frame.loc[frame["is_held_out"]].sort_values("cell_id")
            colors = represented["displayed_assignment"].map(CELL_TYPE_COLORS)
            if colors.isna().any():
                raise PBMCFigure3Error("Composite Panel F contains unmapped biological labels.")
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
        endpoint_label = {
            "B cells": "Stimulated B",
            "NK cells": "Stimulated NK",
            "Dendritic cells": "Stimulated DC",
        }[endpoint]
        fig.text(
            0.032,
            y_positions[row] + height / 2,
            endpoint_label,
            fontsize=PANEL_SPATIAL_ENDPOINT_FONT_SIZE,
            fontweight="normal",
            rotation=270,
            ha="center",
            va="center",
            color=INK,
            gid=f"panel-f-endpoint-{endpoint_label.lower().replace(' ', '-')}",
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
            label=(
                "Reference-omitted cells (not evaluated)"
                if label == "Held-out state (not evaluated)"
                else label
            ),
        )
        for label, color in CELL_TYPE_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.535, 0.005),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.25,
        columnspacing=0.65,
    )


def draw_main_figure(sources: dict[str, object]) -> plt.Figure:
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
    _draw_panel_a(fig, sources["A"])
    _draw_panel_b_restoration(fig, sources["C"])
    _draw_panel_c_metric_bars(fig, sources["B"])
    _draw_panel_d_metric_bars(fig, sources["D"])
    _draw_panel_e(fig, sources["E"])
    _draw_panel_f(fig, sources["F"])
    return fig


def draw_panel_figure(sources: dict[str, object], letter: str) -> plt.Figure:
    drawers = {
        "A": lambda fig: _draw_panel_a(fig, sources["A"]),
        "B": lambda fig: _draw_panel_b_restoration(fig, sources["C"]),
        "C": lambda fig: _draw_panel_c_metric_bars(fig, sources["B"]),
        "D": lambda fig: _draw_panel_d_metric_bars(fig, sources["D"]),
        "E": lambda fig: _draw_panel_e(fig, sources["E"]),
        "F": lambda fig: _draw_panel_f(fig, sources["F"]),
    }
    if letter not in drawers:
        raise PBMCFigure3Error(f"Unknown Figure 3 panel: {letter}.")
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
    drawers[letter](fig)
    return fig


def _validate_fonts(fig: plt.Figure) -> None:
    sizes = [
        float(text.get_fontsize())
        for text in fig.findobj(match=Text)
        if text.get_text()
        and text.get_gid() not in PANEL_A_SMALL_TEXT_GIDS
        and text.get_gid() not in PANEL_B_ENDPOINT_GIDS
        and text.get_gid() not in PANEL_SPATIAL_ENDPOINT_GIDS
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
            raise PBMCFigure3Error(f"Figure 3 text lies outside the canvas: {text.get_text()!r}.")


def _save(fig: plt.Figure, outputs: dict[str, Path], *, bbox: Bbox | None = None) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "dpi": PANEL_RASTER_DPI if suffix in {"png", "tiff"} else None,
            "facecolor": "white",
            "bbox_inches": bbox,
        }
        if suffix == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **save_kwargs)


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
            suffix: panel_root / f"figure_3_pbmc_panel_{letter.lower()}.{suffix}"
            for suffix in ("png", "pdf", "tiff")
        }
        legacy_svg = panel_root / f"figure_3_pbmc_panel_{letter.lower()}.svg"
        if legacy_svg.is_file():
            legacy_svg.unlink()
        _save(fig, panel_outputs, bbox=_panel_bbox(fig, letter))
        outputs[f"panel_{letter.lower()}"] = panel_outputs["png"]
        plt.close(fig)
    return outputs


def _validate_main_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected = tuple(round(value * PANEL_RASTER_DPI) for value in MAIN_FIGURE_SIZE_INCHES)
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
    project_root: Path | None = None,
    generation_provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    sources = read_composite_sources(source_root)
    fig = draw_main_figure(sources)
    _validate_fonts(fig)
    _validate_artist_bounds(fig)
    docs_figure_root = docs_root / "figs"
    docs_figure_root.mkdir(parents=True, exist_ok=True)
    docs_outputs = {
        suffix: docs_figure_root / f"manuscript_fig_pbmc_main.{suffix}"
        for suffix in ("png", "pdf", "tiff")
    }
    result_outputs = {
        suffix: result_root / f"figure_3_pbmc_main.{suffix}" for suffix in ("png", "pdf", "tiff")
    }
    for legacy_svg in (
        docs_figure_root / "manuscript_fig_pbmc_main.svg",
        result_root / "figure_3_pbmc_main.svg",
    ):
        if legacy_svg.is_file():
            legacy_svg.unlink()
    _save(fig, docs_outputs)
    _save(fig, result_outputs)
    plt.close(fig)
    _validate_main_raster(docs_outputs["png"])

    source_paths = list(sources["paths"].values())
    manifest_path = result_root / "figure_3_pbmc_manifest.yaml"
    manifest_root = project_root.resolve() if project_root is not None else None

    def manifest_path_value(path: Path) -> str:
        if manifest_root is None:
            return str(path)
        try:
            return str(path.resolve().relative_to(manifest_root))
        except ValueError as error:
            raise PBMCFigure3Error(
                f"Canonical manifest path is outside the repository: {path}"
            ) from error

    renderer_source = Path(__file__).resolve()
    provenance = dict(generation_provenance or {})
    if generation_provenance is None and manifest_path.is_file():
        previous_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        previous_provenance = (
            previous_manifest.get("provenance", {})
            if isinstance(previous_manifest, Mapping)
            else {}
        )
        if isinstance(previous_provenance, Mapping):
            provenance.update(previous_provenance)
    provenance["renderer_source"] = {
        "path": "src/coreot/results/pbmc_figure3_composite.py",
        "sha256": _file_sha256(renderer_source),
    }
    deterministic_outputs = [
        path
        for outputs in (docs_outputs, result_outputs)
        for suffix, path in outputs.items()
        if suffix in {"png", "tiff"}
    ]
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "PBMC condition-specific weak correspondence Figure 3",
                "panel_set": list(PANEL_SET),
                "panel_contract": {
                    "B": "paired_reference_restoration_response",
                    "C": "within_celltype_omitted_state_ranking",
                    "D": "represented_celltype_label_transfer",
                    "E": "weak_support_umap",
                },
                "generator": (
                    "experiments/pbmc_state/generate_pbmc_figure3_panels.py --panel main"
                ),
                "artifacts": {
                    "manuscript": {
                        suffix: manifest_path_value(path) for suffix, path in docs_outputs.items()
                    },
                    "result": {
                        suffix: manifest_path_value(path) for suffix, path in result_outputs.items()
                    },
                },
                "sources": [manifest_path_value(path) for path in source_paths],
                "source_data_sha256": {
                    manifest_path_value(path): _file_sha256(path) for path in source_paths
                },
                "output_sha256": {
                    manifest_path_value(path): _file_sha256(path) for path in deterministic_outputs
                },
                "container_metadata_policy": {
                    "authoritative_hashed_formats": ["png", "tiff"],
                    "pdf_hashes_recorded": False,
                    "pdf_reason": (
                        "PDF container timestamps may vary; PDF is an untracked "
                        "delivery format and is excluded from deterministic identity."
                    ),
                },
                "parameters": {
                    "canvas_inches": list(MAIN_FIGURE_SIZE_INCHES),
                    "canvas_millimeters": [178, 230],
                    "raster_dpi": PANEL_RASTER_DPI,
                    "minimum_font_size_points": PANEL_A_MIN_FONT_SIZE,
                    "standard_minimum_font_size_points": MAIN_FIGURE_MIN_FONT_SIZE,
                    "panel_a_condition_font_size_points": PANEL_A_CONDITION_FONT_SIZE,
                    "panel_b_endpoint_label_alignment": "right_edge_rotated_270",
                    "panel_b_endpoint_font_size_points": PANEL_B_ENDPOINT_FONT_SIZE,
                    "panel_b_axis_bounds_fraction": [0.54, 0.805, 0.37, 0.15],
                    "panel_b_encoding": {
                        "rows": [
                            "reference_omitted_cell_median_deficit_decrease",
                            "same_type_control_median_deficit_decrease",
                            "control_adjusted_median_deficit_decrease",
                            "median_restored_state_destination_fraction",
                        ],
                        "bar": "arithmetic_mean_across_five_donor_splits",
                        "whisker": "sample_sd_ddof1",
                        "points": "not_rendered",
                        "axis": [0.0, 1.0],
                        "axis_label": "Metric value",
                    },
                    "panel_e_representative_seed": PANEL_D_SEED,
                    "panel_f_representative_seed": PANEL_D_SEED,
                    "spatial_panel_layout": {
                        "panels": ["E", "F"],
                        "method_columns": "aligned",
                        "axis_width_fraction": 0.128,
                        "panel_e_axis_height_fraction": 0.045,
                        "panel_f_axis_height_fraction": 0.050,
                        "endpoint_labels": "left_border_rotated_270",
                        "endpoint_label_x_fraction": 0.032,
                        "endpoint_label_weight": "normal",
                        "endpoint_label_font_size_points": (
                            PANEL_SPATIAL_ENDPOINT_FONT_SIZE
                        ),
                        "panel_e_legend_y_fraction": 0.242,
                    },
                    "metric_panel_vertical_alignment": {
                        "panels": ["C", "D"],
                        "axis_y_fraction": 0.515,
                        "axis_height_fraction": 0.185,
                        "axis_label_y_fraction": 0.49,
                        "legend_y_fraction": 0.755,
                    },
                    "method_label_layout": {
                        "panels": ["C", "D"],
                        "scmap_cluster_lines": 2,
                    },
                    "panel_c_encoding": {
                        "solid": "average_precision",
                        "hollow": "auroc",
                        "reference": "mean_positive_prevalence",
                        "bar": "arithmetic_mean_across_five_donor_splits",
                        "whisker": "sample_sd_ddof1",
                        "points": "not_rendered",
                        "axis": [0.0, 1.0],
                    },
                    "panel_d_encoding": {
                        "solid": "forced_accuracy",
                        "hollow": "forced_macro_f1",
                        "bar": "arithmetic_mean_across_five_donor_splits",
                        "whisker": "sample_sd_ddof1",
                        "points": "not_rendered",
                        "axis": [0.0, 1.0],
                    },
                    "dense_umap_points_rasterized": True,
                },
                "software_versions": _software_versions(),
                "provenance": provenance,
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
