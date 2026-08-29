from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from coreot.results.hiha_figure2 import (
    BROAD_DESTINATION_COLORS,
    BROAD_DESTINATION_ORDER,
    DOCS_ROOT,
    ENDPOINTS,
    HELD_OUT,
    INK,
    LIGHT_EDGE,
    METHOD_COLORS,
    MUTED_INK,
    PANEL_B_METHODS,
    PANEL_C_COMPARATORS,
    PANEL_C_GROUPS,
    PANEL_D_LIM,
    PANEL_D_METHODS,
    PANEL_D_XLABEL,
    PANEL_D_YLABEL,
    PANEL_E_CONTEXT_GRAY,
    PANEL_E_ELIGIBLE_GRAY,
    PANEL_E_LEGEND_LABELS,
    PANEL_E_MAPS,
    PANEL_A_REFERENCE_OMITTED_LABEL,
    PANEL_A_RESTORED_LABEL,
    PANEL_E_TRUTH_COLOR,
    REPRESENTATIVE_SEED,
    PANEL_F_HELD_OUT_COLOR,
    PANEL_F_HELD_OUT_POINT_SIZE,
    PANEL_F_LABEL_COLORS,
    PANEL_F_MAPS,
    PANEL_F_REPRESENTED_POINT_SIZE,
    PANEL_ROOT,
    PANEL_RASTER_DPI,
    QUERY_FILL,
    REFERENCE_FILL,
    RESTORED,
    RESULT_ROOT,
    SOURCE_DATA_ROOT,
    HIHAFigure2Error,
)
from coreot.results.main_metric_bars import (
    FillMetricSpec,
    draw_endpoint_metric_facet,
    fill_metric_handles,
)


MAIN_FIGURE_SIZE_INCHES = (178 / 25.4, 230 / 25.4)
MAIN_FIGURE_PANEL_SET = ("A", "B", "C", "D", "E", "F")
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
        "panel-a-reference-omitted-title",
        "panel-a-reference-omitted-detail",
        "panel-a-restored-reference-title",
        "panel-a-restored-reference-detail",
    }
)
PANEL_B_ENDPOINT_FONT_SIZE = 5.8
PANEL_B_ENDPOINT_GIDS = frozenset(
    {
        "panel-b-endpoint-label-hla-drhi-cdc2",
        "panel-b-endpoint-label-isg-cdc2",
    }
)
PANEL_C_METRICS = (
    FillMetricSpec("average_precision", "AP", True),
    FillMetricSpec("auroc", "AUROC", False),
)
PANEL_D_BAR_METRICS = (
    FillMetricSpec("forced_accuracy", "Forced accuracy", True),
    FillMetricSpec("forced_macro_f1", "Forced macro-F1", False),
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, required: set[str], panel: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(
            f"Composite Panel {panel} source is missing columns {missing}: {path}"
        )
    if frame.empty:
        raise HIHAFigure2Error(f"Composite Panel {panel} source is empty: {path}")
    return frame


def _resolve_source_path(
    source_root: Path,
    preferred_name: str,
    legacy_name: str,
) -> Path:
    legacy = source_root / legacy_name
    if legacy.is_file():
        return legacy
    return source_root / preferred_name


def _require_exact_keys(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    expected: set[tuple[object, ...]],
    panel: str,
) -> None:
    if frame.duplicated(columns).any():
        raise HIHAFigure2Error(f"Composite Panel {panel} source has duplicate keys {columns}")
    observed = set(frame[columns].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise HIHAFigure2Error(
            f"Composite Panel {panel} keys differ from the figure contract; "
            f"missing={missing}, extra={extra}"
        )


def _read_composite_sources(
    source_root: Path,
) -> tuple[
    dict[str, str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    design_path = source_root / "panel_a_design.csv"
    design_frame = _read_csv(
        design_path,
        {"field", "value", "source_path"},
        "A",
    )
    if design_frame["field"].duplicated().any():
        raise HIHAFigure2Error("Composite Panel A design fields are duplicated")
    design = design_frame.set_index("field")["value"].astype(str).to_dict()
    required_design = {
        "number_of_subjects",
        "donor_split",
        "displayed_endpoints",
        "primary_evaluation",
    }
    if set(design) != required_design:
        raise HIHAFigure2Error("Composite Panel A design fields differ from the standalone panel")
    if design["number_of_subjects"] != "108":
        raise HIHAFigure2Error("Composite Panel A requires the recorded 108 subjects")
    if "subject-level" not in design["donor_split"]:
        raise HIHAFigure2Error("Composite Panel A requires a subject-level split")

    panel_b = _read_csv(
        source_root / "panel_b_detection.csv",
        {
            "held_out_label",
            "seed",
            "method",
            "score",
            "average_precision",
            "auroc",
            "positive_prevalence",
        },
        "B",
    )
    expected_b = {
        (endpoint, seed, method, score)
        for endpoint in ENDPOINTS
        for seed in range(1, 6)
        for method, score, _ in PANEL_B_METHODS
    }
    _require_exact_keys(
        panel_b,
        columns=["held_out_label", "seed", "method", "score"],
        expected=expected_b,
        panel="B",
    )

    panel_c_summary = _read_csv(
        source_root / "panel_c_deficit_by_seed.csv",
        {
            "held_out_label",
            "seed",
            "truth_group",
            "median_u_ablated",
            "median_u_full",
        },
        "C",
    )
    expected_c_summary = {
        (endpoint, seed, group)
        for endpoint in ENDPOINTS
        for seed in range(1, 6)
        for group, _ in PANEL_C_GROUPS
    }
    _require_exact_keys(
        panel_c_summary,
        columns=["held_out_label", "seed", "truth_group"],
        expected=expected_c_summary,
        panel="C",
    )
    panel_c_restoration = _read_csv(
        source_root / "panel_c_restoration_by_seed.csv",
        {
            "held_out_label",
            "seed",
            "restoration_specificity",
            "median_restored_state_probability",
        },
        "C",
    )
    expected_c_restoration = {(endpoint, seed) for endpoint in ENDPOINTS for seed in range(1, 6)}
    _require_exact_keys(
        panel_c_restoration,
        columns=["held_out_label", "seed"],
        expected=expected_c_restoration,
        panel="C",
    )

    panel_d = _read_csv(
        _resolve_source_path(
            source_root,
            "panel_d_label_transfer.csv",
            "panel_e_label_transfer.csv",
        ),
        {
            "held_out_label",
            "seed",
            "method",
            "score",
            "forced_macro_f1",
            "forced_accuracy",
        },
        "D",
    )
    expected_d = {
        (endpoint, seed, method, score)
        for endpoint in ENDPOINTS
        for seed in range(1, 6)
        for method, score, _ in PANEL_D_METHODS
    }
    _require_exact_keys(
        panel_d,
        columns=["held_out_label", "seed", "method", "score"],
        expected=expected_d,
        panel="D",
    )

    panel_e = _read_csv(
        _resolve_source_path(
            source_root,
            "panel_e_umap_cells.csv",
            "panel_d_umap_cells.csv",
        ),
        {
            "held_out_label",
            "seed",
            "map_id",
            "cell_id",
            "is_held_out_truth",
            "is_within_cdc2",
            "is_selected",
            "umap_1",
            "umap_2",
        },
        "E",
    )
    if set(panel_e["held_out_label"]) != set(ENDPOINTS):
        raise HIHAFigure2Error("Composite Panel E requires exactly the two endpoints")
    if set(panel_e["seed"]) != {REPRESENTATIVE_SEED}:
        raise HIHAFigure2Error(
            f"Composite Panel E requires representative seed {REPRESENTATIVE_SEED}"
        )
    if panel_e.duplicated(["held_out_label", "cell_id"]).any():
        if panel_e.duplicated(["held_out_label", "map_id", "cell_id"]).any():
            raise HIHAFigure2Error("Composite Panel E has duplicate endpoint/map/cell IDs")
    expected_e_maps = {
        (endpoint, map_id) for endpoint in ENDPOINTS for map_id, _, _ in PANEL_E_MAPS
    }
    observed_e_maps = set(
        panel_e[["held_out_label", "map_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_e_maps != expected_e_maps:
        raise HIHAFigure2Error("Composite Panel E map keys differ from its contract")

    panel_f = _read_csv(
        source_root / "panel_f_label_assignment_cells.csv",
        {
            "held_out_label",
            "seed",
            "cell_id",
            "umap_1",
            "umap_2",
            "is_represented_state",
            "is_held_out_state",
            "map_id",
            "displayed_assignment",
        },
        "F",
    )
    expected_f_maps = {(endpoint, map_id) for endpoint in ENDPOINTS for map_id, _ in PANEL_F_MAPS}
    observed_f_maps = set(
        panel_f[["held_out_label", "map_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_f_maps != expected_f_maps:
        raise HIHAFigure2Error("Composite Panel F map keys differ from its contract")
    if set(panel_f["seed"]) != {REPRESENTATIVE_SEED}:
        raise HIHAFigure2Error(
            f"Composite Panel F requires representative seed {REPRESENTATIVE_SEED}"
        )
    if panel_f.duplicated(["held_out_label", "map_id", "cell_id"]).any():
        raise HIHAFigure2Error("Composite Panel F has duplicate endpoint/map/cell IDs")

    numeric_frames = (
        (panel_b, ("average_precision", "auroc", "positive_prevalence"), "C"),
        (
            panel_c_summary,
            ("median_u_ablated", "median_u_full"),
            "C",
        ),
        (
            panel_c_restoration,
            ("restoration_specificity", "median_restored_state_probability"),
            "C",
        ),
        (panel_d, ("forced_macro_f1", "forced_accuracy"), "D"),
        (panel_e, ("umap_1", "umap_2"), "E"),
        (panel_f, ("umap_1", "umap_2"), "F"),
    )
    for frame, columns, panel in numeric_frames:
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise HIHAFigure2Error(f"Composite Panel {panel} contains nonfinite plotted values")
        frame.loc[:, columns] = values

    for column in ("is_held_out_truth", "is_within_cdc2", "is_selected"):
        normalized = panel_e[column].astype(str).str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise HIHAFigure2Error(f"Composite Panel E {column} is not a Boolean field")
        panel_e[column] = normalized.eq("true")

    for column in ("is_represented_state", "is_held_out_state"):
        normalized = panel_f[column].astype(str).str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise HIHAFigure2Error(f"Composite Panel F {column} is not a Boolean field")
        panel_f[column] = normalized.eq("true")

    for column in ("average_precision", "auroc"):
        if not panel_b[column].between(0, 1).all():
            raise HIHAFigure2Error(f"Composite Panel C {column} falls outside [0, 1]")
    for column in ("forced_macro_f1", "forced_accuracy"):
        if not panel_d[column].between(0, 1).all():
            raise HIHAFigure2Error(f"Composite Panel D {column} falls outside [0, 1]")

    return (
        design,
        panel_b,
        panel_c_summary,
        panel_c_restoration,
        panel_d,
        panel_e,
        panel_f,
    )


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
    fontsize: float = MAIN_FIGURE_MIN_FONT_SIZE,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=0.7,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.15,
        zorder=3,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.7,
            color=MUTED_INK,
            shrinkA=1.5,
            shrinkB=2.5,
            zorder=2,
        )
    )


def _draw_panel_a(fig: plt.Figure, design: dict[str, str]) -> None:
    ax = fig.add_axes((0.025, 0.865, 0.95, 0.125))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.0, 0.98, "A", fontsize=10, fontweight="bold", ha="left", va="top")

    _box(
        ax,
        0.01,
        0.24,
        0.105,
        0.45,
        f"HIHA DCs\n{design['number_of_subjects']} subjects",
        facecolor=REFERENCE_FILL,
    )
    _box(
        ax,
        0.15,
        0.24,
        0.115,
        0.45,
        "Donor-level split\n80% reference / 20% query",
        facecolor=REFERENCE_FILL,
    )
    _arrow(ax, (0.118, 0.465), (0.147, 0.465))

    _box(
        ax,
        0.35,
        0.58,
        0.16,
        0.25,
        "Query cells\nall states retained",
        facecolor=QUERY_FILL,
    )
    _box(
        ax,
        0.35,
        0.16,
        0.16,
        0.25,
        "Reference 80%",
        facecolor=REFERENCE_FILL,
    )
    _arrow(ax, (0.268, 0.50), (0.307, 0.705))
    _arrow(ax, (0.268, 0.42), (0.307, 0.285))

    _box(
        ax,
        0.52,
        0.42,
        0.17,
        0.25,
        PANEL_A_REFERENCE_OMITTED_LABEL,
        facecolor="#FCE7DF",
        edgecolor=HELD_OUT,
    )
    _box(
        ax,
        0.52,
        0.11,
        0.17,
        0.25,
        PANEL_A_RESTORED_LABEL,
        facecolor="#E3F3EC",
        edgecolor=RESTORED,
    )
    _arrow(ax, (0.473, 0.285), (0.516, 0.545))
    _arrow(ax, (0.473, 0.285), (0.516, 0.235))

    _box(
        ax,
        0.76,
        0.18,
        0.225,
        0.55,
        (
            "HIHA dendritic-cell benchmark\n"
            "Evaluated fine states: HLA-DRhi cDC2, ISG+ cDC2\n"
            "within-cDC2 omitted-state ranking\n"
            "Reference-omitted condition: omitted-state ranking\n"
            "Restored-reference condition: paired reference-restoration response\n"
            "Same query cells"
        ),
        facecolor="#F7F7F7",
        edgecolor=INK,
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
    )
    _arrow(ax, (0.473, 0.705), (0.756, 0.68))
    _arrow(ax, (0.693, 0.545), (0.756, 0.53))
    _arrow(ax, (0.693, 0.235), (0.756, 0.37))


def _draw_horizontal_summary(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    method_column: str,
    value_column: str,
    order: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    xlim: tuple[float, float],
    xticks: list[float],
    show_ylabels: bool,
) -> None:
    positions = np.arange(len(order), dtype=float)
    offsets = np.linspace(-0.10, 0.10, 5)
    for y, method in zip(positions, order, strict=True):
        values = (
            frame.loc[frame[method_column].eq(method)]
            .sort_values("seed")[value_column]
            .to_numpy(dtype=float)
        )
        mean = float(values.mean())
        sample_sd = float(values.std(ddof=1))
        color = colors[method]
        ax.scatter(
            values,
            y + offsets,
            s=8,
            color=color,
            alpha=0.42,
            linewidth=0,
            zorder=2,
        )
        ax.errorbar(
            mean,
            y,
            xerr=sample_sd,
            fmt="o",
            markersize=3.8,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            capthick=0.7,
            zorder=3,
        )
    ax.set(xlim=xlim, ylim=(len(order) - 0.5, -0.5))
    ax.set_xticks(xticks)
    ax.tick_params(
        axis="x",
        labelsize=MAIN_FIGURE_MIN_FONT_SIZE,
        length=2.5,
        pad=2,
    )
    ax.tick_params(axis="y", length=0, pad=2)
    if show_ylabels:
        ax.set_yticks(
            positions,
            [labels[method] for method in order],
            fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        )
    else:
        ax.set_yticks(positions, [])
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _draw_panels_b_c(
    fig: plt.Figure,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
) -> None:
    fig.text(0.025, 0.842, "B", fontsize=10, fontweight="bold", ha="left", va="top")
    fig.text(
        0.052,
        0.842,
        "External detection benchmark",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )
    fig.text(
        0.255,
        0.818,
        "AP for omitted-state ranking • dashed = mean prevalence",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        ha="center",
        va="top",
        color=MUTED_INK,
    )
    fig.text(0.515, 0.842, "C", fontsize=10, fontweight="bold", ha="left", va="top")
    fig.text(
        0.542,
        0.842,
        "Transport and prior controls",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )
    fig.text(
        0.755,
        0.818,
        r"Paired $\Delta$AP = CoRe-OT − comparator",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        ha="center",
        va="top",
        color=MUTED_INK,
    )

    b_order = [method for method, _, _ in PANEL_B_METHODS]
    b_labels = {method: display for method, _, display in PANEL_B_METHODS}
    c_order = [method for method, _, _ in PANEL_C_COMPARATORS]
    c_labels = {
        "coreot_match_only": "CoRe-OT\n(−C)",
        "uniform_uot": "Uniform UOT",
        "prior_only": "Prior only",
    }
    b_axes = (
        fig.add_axes((0.115, 0.665, 0.17, 0.13)),
        fig.add_axes((0.315, 0.665, 0.17, 0.13)),
    )
    c_axes = (
        fig.add_axes((0.615, 0.665, 0.16, 0.13)),
        fig.add_axes((0.815, 0.665, 0.16, 0.13)),
    )
    for index, endpoint in enumerate(ENDPOINTS):
        b_endpoint = panel_b.loc[panel_b["held_out_label"].eq(endpoint)]
        _draw_horizontal_summary(
            b_axes[index],
            b_endpoint,
            method_column="method",
            value_column="average_precision",
            order=b_order,
            colors=METHOD_COLORS,
            labels=b_labels,
            xlim=(0, 1),
            xticks=[0, 0.5, 1],
            show_ylabels=index == 0,
        )
        prevalence = float(b_endpoint.groupby("seed")["positive_prevalence"].first().mean())
        b_axes[index].axvline(
            prevalence,
            color="#777777",
            linewidth=0.65,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        b_axes[index].set_title(endpoint, fontsize=6.7, pad=3)

        c_endpoint = panel_c.loc[panel_c["held_out_label"].eq(endpoint)]
        _draw_horizontal_summary(
            c_axes[index],
            c_endpoint,
            method_column="comparator_method",
            value_column="delta_average_precision",
            order=c_order,
            colors=METHOD_COLORS,
            labels=c_labels,
            xlim=(-0.08, 0.65),
            xticks=[0, 0.3, 0.6],
            show_ylabels=index == 0,
        )
        c_axes[index].axvline(0, color=INK, linewidth=0.65, zorder=1)
        c_axes[index].set_title(endpoint, fontsize=6.7, pad=3)

    fig.text(
        0.33,
        0.64,
        "AP",
        fontsize=6.4,
        ha="center",
        va="center",
        color=INK,
    )
    fig.text(
        0.80,
        0.64,
        r"Paired $\Delta$AP (positive favors restored-reference condition)",
        fontsize=6.4,
        ha="center",
        va="center",
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
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.50, 0.608),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.35,
        columnspacing=1.2,
    )


def _style_panel_d_axis(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        labelsize=MAIN_FIGURE_MIN_FONT_SIZE,
        length=2,
        pad=1.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.4)
    ax.set_axisbelow(True)


def _draw_panel_b_main(fig: plt.Figure, panel_b: pd.DataFrame) -> None:
    fig.text(0.025, 0.842, "B", fontsize=10, fontweight="bold", ha="left", va="top")
    order = [method for method, _, _ in PANEL_B_METHODS]
    labels = {method: display for method, _, display in PANEL_B_METHODS}
    axes = (
        fig.add_axes((0.13, 0.685, 0.36, 0.125)),
        fig.add_axes((0.59, 0.685, 0.36, 0.125)),
    )
    for index, (axis, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        endpoint_data = panel_b.loc[panel_b["held_out_label"].eq(endpoint)]
        _draw_horizontal_summary(
            axis,
            endpoint_data,
            method_column="method",
            value_column="average_precision",
            order=order,
            colors=METHOD_COLORS,
            labels=labels,
            xlim=(0, 1),
            xticks=[0, 0.5, 1],
            show_ylabels=index == 0,
        )
        prevalence = float(endpoint_data["positive_prevalence"].mean())
        axis.axvline(
            prevalence,
            color="#777777",
            linestyle=(0, (2.5, 2.0)),
            linewidth=0.7,
            zorder=0,
        )
        axis.set_title(endpoint, fontsize=6.8, pad=3)
    fig.text(
        0.54,
        0.657,
        "AP for omitted-state ranking",
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
        bbox_to_anchor=(0.54, 0.625),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.35,
        columnspacing=1.0,
    )


def _draw_panel_d(
    fig: plt.Figure,
    summary: pd.DataFrame,
    restoration: pd.DataFrame,
) -> None:
    fig.text(0.025, 0.605, "C", fontsize=10, fontweight="bold", ha="left", va="top")

    x_positions = (0.085, 0.315, 0.555, 0.79)
    widths = (0.18, 0.18, 0.17, 0.18)
    y_positions = (0.515, 0.425)
    height = 0.07
    axes = [
        [fig.add_axes((x_positions[column], y, widths[column], height)) for column in range(4)]
        for y in y_positions
    ]
    column_titles = (
        "Reference-omitted cells",
        "Represented cDC2",
        r"Rescue specificity $R$",
        "Median restored-state destination fraction",
    )
    for column, title in enumerate(column_titles):
        axes[0][column].set_title(title, fontsize=6.4, pad=3)

    for row, endpoint in enumerate(ENDPOINTS):
        fig.text(
            0.035,
            y_positions[row] + height / 2,
            endpoint,
            fontsize=6.6,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
            color=INK,
        )
        endpoint_summary = summary.loc[summary["held_out_label"].eq(endpoint)]
        for column, group in enumerate(("held_out", "represented_cdc2")):
            ax = axes[row][column]
            group_frame = endpoint_summary.loc[
                endpoint_summary["truth_group"].eq(group)
            ].sort_values("seed")
            for _, record in group_frame.iterrows():
                ax.plot(
                    [0, 1],
                    [record["median_u_ablated"], record["median_u_full"]],
                    color="#8A8A8A",
                    linewidth=0.65,
                    alpha=0.65,
                    zorder=1,
                )
                ax.scatter(
                    [0, 1],
                    [record["median_u_ablated"], record["median_u_full"]],
                    s=9,
                    color=[HELD_OUT, RESTORED],
                    edgecolor=INK,
                    linewidth=0.25,
                    zorder=2,
                )
            ax.set(xlim=(-0.2, 1.2), ylim=(0, 0.25))
            ax.set_xticks([0, 1], ["Reference-omitted", "Restored-reference"])
            if column == 0:
                ax.set_ylabel(
                    "Median raw $u$",
                    fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
                    labelpad=1,
                )
            else:
                ax.set_yticklabels([])
            _style_panel_d_axis(ax)

        endpoint_restoration = restoration.loc[
            restoration["held_out_label"].eq(endpoint)
        ].sort_values("seed")
        for column, value_column in enumerate(
            ("restoration_specificity", "median_restored_state_probability"),
            start=2,
        ):
            ax = axes[row][column]
            values = endpoint_restoration[value_column].to_numpy(dtype=float)
            offsets = np.linspace(-0.08, 0.08, len(values))
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            color = METHOD_COLORS["coreot_full"]
            ax.scatter(
                0.5 + offsets,
                values,
                s=9,
                color=color,
                alpha=0.45,
                linewidth=0,
                zorder=2,
            )
            ax.errorbar(
                0.5,
                mean,
                yerr=sample_sd,
                fmt="o",
                markersize=3.8,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.4,
                ecolor=color,
                elinewidth=0.8,
                capsize=2,
                zorder=3,
            )
            ax.set_xlim((0.28, 0.72))
            ax.set_xticks([])
            if column == 2:
                ax.set_ylim((0, 0.13))
                ax.axhline(0, color=INK, linewidth=0.6)
                ax.set_ylabel(
                    r"$R$",
                    fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
                    labelpad=1,
                )
            else:
                ax.set_ylim((0, 1.02))
            _style_panel_d_axis(ax)


def _draw_panel_e(fig: plt.Figure, cells: pd.DataFrame) -> None:
    fig.text(0.025, 0.395, "D", fontsize=10, fontweight="bold", ha="left", va="top")

    x_positions = (0.09, 0.54)
    y_positions = (0.315, 0.25)
    width = 0.41
    height = 0.05
    axes = [
        [fig.add_axes((x_positions[column], y, width, height)) for column in range(2)]
        for y in y_positions
    ]
    deficit_cmap = plt.get_cmap("viridis")
    deficit_norm = Normalize(vmin=0, vmax=1)

    for row, endpoint in enumerate(ENDPOINTS):
        frame = cells.loc[cells["held_out_label"].eq(endpoint)].copy()
        x_span = max(float(frame["umap_1"].max() - frame["umap_1"].min()), 1.0e-6)
        y_span = max(float(frame["umap_2"].max() - frame["umap_2"].min()), 1.0e-6)
        xlim = (
            float(frame["umap_1"].min() - 0.025 * x_span),
            float(frame["umap_1"].max() + 0.025 * x_span),
        )
        ylim = (
            float(frame["umap_2"].min() - 0.025 * y_span),
            float(frame["umap_2"].max() + 0.025 * y_span),
        )
        deficit_axis, destination_axis = axes[row]
        ordered = frame.sort_values(["raw_u", "cell_id"], kind="mergesort")
        deficit_axis.scatter(
            ordered["umap_1"],
            ordered["umap_2"],
            c=ordered["raw_u"],
            cmap=deficit_cmap,
            norm=deficit_norm,
            s=2.2,
            alpha=0.82,
            linewidth=0,
            rasterized=True,
            zorder=1,
        )
        held = frame.loc[frame["is_held_out_truth"]]
        deficit_axis.scatter(
            held["umap_1"],
            held["umap_2"],
            s=3.7,
            facecolors="none",
            edgecolors=INK,
            linewidths=0.18,
            alpha=0.75,
            rasterized=True,
            zorder=3,
        )
        for broad_label in BROAD_DESTINATION_ORDER:
            subset = frame.loc[frame["broad_destination"].eq(broad_label)]
            if subset.empty:
                continue
            destination_axis.scatter(
                subset["umap_1"],
                subset["umap_2"],
                s=2.2,
                color=BROAD_DESTINATION_COLORS[broad_label],
                alpha=0.77,
                linewidth=0,
                rasterized=True,
                zorder=1,
            )
        high = frame.loc[frame["high_deficit"]]
        destination_axis.scatter(
            high["umap_1"],
            high["umap_2"],
            s=4.0,
            facecolors="none",
            edgecolors=INK,
            linewidths=0.22,
            alpha=0.8,
            rasterized=True,
            zorder=3,
        )
        for axis in (deficit_axis, destination_axis):
            axis.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
        if row == 0:
            deficit_axis.set_title("Query-marginal deficit $u$", fontsize=6.5, pad=3)
            destination_axis.set_title(
                "Conditional broad destination",
                fontsize=6.5,
                pad=3,
            )
        fig.text(
            0.035,
            y_positions[row] + height / 2,
            endpoint,
            fontsize=6.6,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
            color=INK,
        )

    colorbar_axis = fig.add_axes((0.12, 0.218, 0.26, 0.006))
    colorbar = fig.colorbar(
        ScalarMappable(norm=deficit_norm, cmap=deficit_cmap),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_label(
        "Query-marginal deficit $u$ (0–1)",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        labelpad=1,
    )
    colorbar.ax.tick_params(
        labelsize=MAIN_FIGURE_MIN_FONT_SIZE,
        length=1.8,
        pad=1,
    )

    destination_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=BROAD_DESTINATION_COLORS[label],
            markeredgecolor="none",
            markersize=3.6,
            label=label,
        )
        for label in BROAD_DESTINATION_ORDER
    ]
    fig.legend(
        handles=destination_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.72, 0.215),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        title="Broad destination",
        title_fontsize=6.3,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor=INK,
                markeredgewidth=0.55,
                markersize=3.8,
                label="Reference-omitted cells (left)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor=INK,
                markeredgewidth=0.65,
                markersize=4,
                label="Above restored-reference $u$ 95th percentile (right)",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.70, 0.19),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def _draw_panel_d_label_transfer_main(
    fig: plt.Figure,
    panel_d: pd.DataFrame,
) -> None:
    fig.text(0.025, 0.18, "D", fontsize=10, fontweight="bold", ha="left", va="top")
    display = {method: name for method, _, name in PANEL_D_METHODS}
    axes = (
        fig.add_axes((0.12, 0.085, 0.36, 0.075)),
        fig.add_axes((0.59, 0.085, 0.36, 0.075)),
    )
    ticks = np.linspace(PANEL_D_LIM[0], PANEL_D_LIM[1], 4)
    for index, (axis, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        endpoint_data = panel_d.loc[panel_d["held_out_label"].eq(endpoint)]
        for method, _, _ in PANEL_D_METHODS:
            method_data = endpoint_data.loc[endpoint_data["method"].eq(method)].sort_values("seed")
            macro_f1 = method_data["forced_macro_f1"].to_numpy(dtype=float)
            accuracy = method_data["forced_accuracy"].to_numpy(dtype=float)
            mean_macro_f1 = float(macro_f1.mean())
            mean_accuracy = float(accuracy.mean())
            sd_macro_f1 = float(macro_f1.std(ddof=1))
            sd_accuracy = float(accuracy.std(ddof=1))
            color = METHOD_COLORS[method]
            axis.scatter(
                macro_f1,
                accuracy,
                s=8,
                color=color,
                alpha=0.42,
                linewidth=0,
                zorder=2,
            )
            axis.errorbar(
                mean_macro_f1,
                mean_accuracy,
                xerr=sd_macro_f1,
                yerr=sd_accuracy,
                fmt="o",
                markersize=3.8,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.45,
                ecolor=color,
                elinewidth=0.8,
                capsize=2,
                capthick=0.7,
                zorder=3,
            )
        axis.set(xlim=PANEL_D_LIM, ylim=PANEL_D_LIM)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.set_title(endpoint, fontsize=6.8, pad=3)
        axis.tick_params(
            axis="both",
            labelsize=MAIN_FIGURE_MIN_FONT_SIZE,
            length=2,
            pad=1.5,
        )
        axis.grid(color="#E7E7E7", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            axis.set_ylabel(
                PANEL_D_YLABEL,
                fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
                labelpad=1,
            )
        else:
            axis.tick_params(axis="y", labelleft=False)
    fig.text(
        0.535,
        0.065,
        PANEL_D_XLABEL,
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        ha="center",
        va="top",
        color=INK,
    )
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=INK,
            markeredgewidth=0.4,
            markersize=4,
            label=display[method],
        )
        for method, _, _ in PANEL_D_METHODS
    ]
    method_legend = fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.535, 0.025),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.add_artist(method_legend)
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
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.535, 0.001),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=1.0,
    )


def _draw_panel_d_main(fig: plt.Figure, panel_d: pd.DataFrame) -> None:
    fig.text(0.025, 0.395, "D", fontsize=10, fontweight="bold", ha="left", va="top")
    x_positions = np.linspace(0.075, 0.865, len(PANEL_E_MAPS))
    y_positions = (0.315, 0.235)
    width = 0.115
    height = 0.055
    axes = [
        [
            fig.add_axes((float(x_positions[column]), y, width, height))
            for column in range(len(PANEL_E_MAPS))
        ]
        for y in y_positions
    ]
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_data = panel_d.loc[panel_d["held_out_label"].eq(endpoint)]
        truth_rows = endpoint_data.loc[endpoint_data["map_id"].eq("truth")]
        x_span = max(
            float(truth_rows["umap_1"].max() - truth_rows["umap_1"].min()),
            1.0e-6,
        )
        y_span = max(
            float(truth_rows["umap_2"].max() - truth_rows["umap_2"].min()),
            1.0e-6,
        )
        xlim = (
            float(truth_rows["umap_1"].min() - 0.025 * x_span),
            float(truth_rows["umap_1"].max() + 0.025 * x_span),
        )
        ylim = (
            float(truth_rows["umap_2"].min() - 0.025 * y_span),
            float(truth_rows["umap_2"].max() + 0.025 * y_span),
        )
        for column, (map_id, _, display) in enumerate(PANEL_E_MAPS):
            axis = axes[row][column]
            frame = endpoint_data.loc[endpoint_data["map_id"].eq(map_id)]
            outside = frame.loc[~frame["is_within_cdc2"]]
            eligible = frame.loc[frame["is_within_cdc2"] & ~frame["is_selected"]]
            selected = frame.loc[frame["is_selected"]]
            axis.scatter(
                outside["umap_1"],
                outside["umap_2"],
                s=0.8,
                color=PANEL_E_CONTEXT_GRAY,
                alpha=0.65,
                linewidth=0,
                rasterized=True,
            )
            axis.scatter(
                eligible["umap_1"],
                eligible["umap_2"],
                s=1.1,
                color=PANEL_E_ELIGIBLE_GRAY,
                alpha=0.78,
                linewidth=0,
                rasterized=True,
            )
            highlight = PANEL_E_TRUTH_COLOR if map_id == "truth" else METHOD_COLORS[map_id]
            axis.scatter(
                selected["umap_1"],
                selected["umap_2"],
                s=1.8,
                color=highlight,
                alpha=0.9,
                linewidth=0,
                rasterized=True,
            )
            axis.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(display, fontsize=MAIN_FIGURE_MIN_FONT_SIZE, pad=2)
        fig.text(
            0.035,
            y_positions[row] + height / 2,
            endpoint,
            fontsize=6.6,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
            color=INK,
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
                label="Reference-omitted cells",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=METHOD_COLORS["coreot_full"],
                markeredgecolor="none",
                markersize=3.5,
                label="Top-ranked set",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_ELIGIBLE_GRAY,
                markeredgecolor="none",
                markersize=3.5,
                label="Other ranking-cohort cells",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_CONTEXT_GRAY,
                markeredgecolor="none",
                markersize=3.5,
                label="Outside ranking cohort",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.46, 0.20),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.5,
    )


def _draw_panel_a_narrow(fig: plt.Figure, design: dict[str, str]) -> None:
    ax = fig.add_axes((0.025, 0.785, 0.275, 0.20))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    del design  # The compact panel intentionally omits cohort-size metadata.

    neutral_fill = "#F2F2F2"
    query_fill = "#EAF2F7"
    query_edge = "#6B9FBC"
    omitted_fill = "#FFF1EA"
    restored_fill = "#E9F5EF"

    def add_box(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        facecolor: str,
        edgecolor: str,
        radius: float = 0.018,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                linewidth=0.85,
                edgecolor=edgecolor,
                facecolor=facecolor,
                zorder=3,
            )
        )

    def add_line(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = MUTED_INK,
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
        color: str = MUTED_INK,
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
        "HLA-DRhi cDC2 | ISG+ cDC2",
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
        "Evaluated state\nretained",
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
        edgecolor=HELD_OUT,
    )
    ax.text(
        omit_x + omit_w / 2,
        omit_y + omit_h * 0.70,
        PANEL_A_REFERENCE_OMITTED_LABEL,
        ha="center",
        va="center",
        fontsize=PANEL_A_CONDITION_FONT_SIZE,
        fontweight="semibold",
        fontfamily="Arial",
        color=HELD_OUT,
        linespacing=1.06,
        gid="panel-a-reference-omitted-title",
        zorder=5,
    )
    ax.text(
        omit_x + omit_w / 2,
        omit_y + omit_h * 0.28,
        "Evaluated-state cells\nomitted",
        ha="center",
        va="center",
        fontsize=4.9,
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
    )
    ax.text(
        restore_x + restore_w / 2,
        restore_y + restore_h * 0.70,
        PANEL_A_RESTORED_LABEL,
        ha="center",
        va="center",
        fontsize=PANEL_A_CONDITION_FONT_SIZE,
        fontweight="semibold",
        fontfamily="Arial",
        color=RESTORED,
        linespacing=1.06,
        gid="panel-a-restored-reference-title",
        zorder=5,
    )
    ax.text(
        restore_x + restore_w / 2,
        restore_y + restore_h * 0.28,
        "Omitted-state cells\nrestored",
        ha="center",
        va="center",
        fontsize=4.9,
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


def _draw_panel_b_wide(fig: plt.Figure, panel_b: pd.DataFrame) -> None:
    fig.text(0.32, 0.985, "B", fontsize=10, fontweight="bold", ha="left", va="top")
    order = [method for method, _, _ in PANEL_B_METHODS]
    labels = {method: display for method, _, display in PANEL_B_METHODS}
    axes = (
        fig.add_axes((0.405, 0.835, 0.255, 0.115)),
        fig.add_axes((0.705, 0.835, 0.255, 0.115)),
    )
    for index, (axis, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        endpoint_data = panel_b.loc[panel_b["held_out_label"].eq(endpoint)]
        _draw_horizontal_summary(
            axis,
            endpoint_data,
            method_column="method",
            value_column="average_precision",
            order=order,
            colors=METHOD_COLORS,
            labels=labels,
            xlim=(0, 1),
            xticks=[0, 0.5, 1],
            show_ylabels=index == 0,
        )
        axis.axvline(
            float(endpoint_data["positive_prevalence"].mean()),
            color="#777777",
            linestyle=(0, (2.5, 2.0)),
            linewidth=0.7,
            zorder=0,
        )
        axis.set_title(endpoint, fontsize=7.0, pad=3)
    fig.text(
        0.68,
        0.812,
        "AP for omitted-state ranking",
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
        bbox_to_anchor=(0.68, 0.777),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def _rescue_source(
    summary: pd.DataFrame,
    restoration: pd.DataFrame,
) -> pd.DataFrame:
    pivot = summary.pivot(
        index=["held_out_label", "seed"],
        columns="truth_group",
        values=["median_u_ablated", "median_u_full"],
    )
    pivot.columns = ["_".join(column) for column in pivot.columns]
    result = pivot.reset_index()
    result["held_out_delta_median_u"] = (
        result["median_u_ablated_held_out"] - result["median_u_full_held_out"]
    )
    result["control_delta_median_u"] = (
        result["median_u_ablated_represented_cdc2"] - result["median_u_full_represented_cdc2"]
    )
    return result.merge(
        restoration[
            [
                "held_out_label",
                "seed",
                "restoration_specificity",
                "median_restored_state_probability",
            ]
        ],
        on=["held_out_label", "seed"],
        validate="one_to_one",
    )


def _draw_panel_b_restoration(
    fig: plt.Figure,
    summary: pd.DataFrame,
    restoration: pd.DataFrame,
) -> None:
    fig.text(0.32, 0.985, "B", fontsize=10, fontweight="bold", ha="left", va="top")
    data = _rescue_source(summary, restoration)
    axis = fig.add_axes((0.54, 0.82, 0.37, 0.135))
    rows = (
        (
            "held_out_delta_median_u",
            "Reference-omitted cells",
            METHOD_COLORS["coreot_full"],
        ),
        ("control_delta_median_u", "Represented cDC2 controls", "#777777"),
        (
            "restoration_specificity",
            "Control-adjusted $\\Delta^{\\mathrm{CA}}$",
            HELD_OUT,
        ),
        (
            "median_restored_state_probability",
            "Restored-state destination fraction",
            RESTORED,
        ),
    )
    positions = {
        ENDPOINTS[0]: np.array([7.0, 6.1, 5.2, 4.3]),
        ENDPOINTS[1]: np.array([2.7, 1.8, 0.9, 0.0]),
    }
    for endpoint in ENDPOINTS:
        endpoint_data = data.loc[data["held_out_label"].eq(endpoint)].sort_values("seed")
        for position, (column, _, color) in zip(positions[endpoint], rows, strict=True):
            values = endpoint_data[column].to_numpy(dtype=float)
            bars = axis.barh(
                position,
                float(values.mean()),
                height=0.55,
                xerr=float(values.std(ddof=1)),
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
        (ENDPOINTS[0], 0.925, "panel-b-endpoint-label-hla-drhi-cdc2"),
        (ENDPOINTS[1], 0.85, "panel-b-endpoint-label-isg-cdc2"),
    )
    for endpoint, y_position, gid in endpoint_labels:
        fig.text(
            0.975,
            y_position,
            endpoint,
            fontsize=PANEL_B_ENDPOINT_FONT_SIZE,
            fontweight="normal",
            rotation=270,
            ha="center",
            va="center",
            color=INK,
            gid=gid,
        )
    row_positions = np.concatenate([positions[endpoint] for endpoint in ENDPOINTS])
    row_labels = [label for _ in ENDPOINTS for _, label, _ in rows]
    axis.axhline(3.5, color="#B8B8B8", linewidth=0.7)
    axis.set(
        xlim=(0.0, 1.02),
        ylim=(-0.55, 7.55),
        xticks=(0.0, 0.5, 1.0),
        yticks=row_positions,
        yticklabels=row_labels,
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


def _draw_panel_c_metric_bars(fig: plt.Figure, panel_c: pd.DataFrame) -> None:
    fig.text(0.025, 0.755, "C", fontsize=10, fontweight="bold", ha="left", va="top")
    method_order = tuple(method for method, _, _ in PANEL_B_METHODS)
    method_labels = {
        method: "scmap-\ncluster" if display == "scmap-cluster" else display
        for method, _, display in PANEL_B_METHODS
    }
    axes = (
        fig.add_axes((0.13, 0.515, 0.155, 0.185)),
        fig.add_axes((0.33, 0.515, 0.155, 0.185)),
    )
    for index, (axis, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        draw_endpoint_metric_facet(
            axis,
            panel_c,
            endpoint_column="held_out_label",
            endpoint=endpoint,
            endpoint_label=endpoint,
            method_order=method_order,
            method_labels=method_labels,
            method_colors=METHOD_COLORS,
            metrics=PANEL_C_METRICS,
            ink=INK,
            grid_color="#E8E8E8",
            show_method_labels=index == 0,
            prevalence_column="positive_prevalence",
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
            )
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.305, 0.755),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handlelength=1.3,
        handletextpad=0.3,
        columnspacing=0.65,
    )
    fig.text(
        0.31,
        0.49,
        "Metric value",
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        ha="center",
        va="top",
        color=INK,
    )


def _draw_panel_d_metric_bars(fig: plt.Figure, panel_d: pd.DataFrame) -> None:
    fig.text(0.515, 0.755, "D", fontsize=10, fontweight="bold", ha="left", va="top")
    method_order = tuple(method for method, _, _ in PANEL_D_METHODS)
    method_labels = {
        method: "scmap-\ncluster" if display == "scmap-cluster" else display
        for method, _, display in PANEL_D_METHODS
    }
    axes = (
        fig.add_axes((0.625, 0.515, 0.16, 0.185)),
        fig.add_axes((0.815, 0.515, 0.16, 0.185)),
    )
    for index, (axis, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        draw_endpoint_metric_facet(
            axis,
            panel_d,
            endpoint_column="held_out_label",
            endpoint=endpoint,
            endpoint_label=endpoint,
            method_order=method_order,
            method_labels=method_labels,
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
        bbox_to_anchor=(0.80, 0.755),
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
    x_span = max(float(frame["umap_1"].max() - frame["umap_1"].min()), 1.0e-6)
    y_span = max(float(frame["umap_2"].max() - frame["umap_2"].min()), 1.0e-6)
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


def _draw_panel_e_full(fig: plt.Figure, panel_e: pd.DataFrame) -> None:
    fig.text(0.012, 0.46, "E", fontsize=10, fontweight="bold", ha="left", va="top")
    x_positions = np.linspace(0.070, 0.860, len(PANEL_E_MAPS))
    y_positions = (0.360, 0.275)
    width = 0.128
    height = 0.075
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_data = panel_e.loc[panel_e["held_out_label"].eq(endpoint)]
        truth_rows = endpoint_data.loc[endpoint_data["map_id"].eq("truth")]
        xlim, ylim = _umap_limits(truth_rows)
        for column, (map_id, _, display) in enumerate(PANEL_E_MAPS):
            axis = fig.add_axes((float(x_positions[column]), y_positions[row], width, height))
            frame = endpoint_data.loc[endpoint_data["map_id"].eq(map_id)]
            outside = frame.loc[~frame["is_within_cdc2"]]
            eligible = frame.loc[frame["is_within_cdc2"] & ~frame["is_selected"]]
            selected = frame.loc[frame["is_selected"]]
            axis.scatter(
                outside["umap_1"],
                outside["umap_2"],
                s=0.8,
                color=PANEL_E_CONTEXT_GRAY,
                alpha=0.65,
                linewidth=0,
                rasterized=True,
            )
            axis.scatter(
                eligible["umap_1"],
                eligible["umap_2"],
                s=1.1,
                color=PANEL_E_ELIGIBLE_GRAY,
                alpha=0.78,
                linewidth=0,
                rasterized=True,
            )
            highlight = PANEL_E_TRUTH_COLOR if map_id == "truth" else METHOD_COLORS[map_id]
            axis.scatter(
                selected["umap_1"],
                selected["umap_2"],
                s=1.8,
                color=highlight,
                alpha=0.9,
                linewidth=0,
                rasterized=True,
            )
            axis.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            axis.set_aspect("equal", adjustable="box")
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(display, fontsize=MAIN_FIGURE_MIN_FONT_SIZE, pad=2)
        endpoint_label = "HLA-DRhi cDC2" if row == 0 else "ISG+ cDC2"
        fig.text(
            0.032,
            y_positions[row] + height / 2,
            endpoint_label,
            fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
            fontweight="normal",
            ha="center",
            va="center",
            rotation=270,
            color=INK,
            gid=f"panel-e-endpoint-{row}",
        )
    method_color_handle = tuple(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=3.5,
        )
        for method, _, _ in PANEL_B_METHODS
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_TRUTH_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=3.5,
                label="Reference-omitted cells",
            ),
            method_color_handle,
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_ELIGIBLE_GRAY,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=3.5,
                label="Other ranking-cohort cells",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PANEL_E_CONTEXT_GRAY,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=3.5,
                label="Outside ranking cohort",
            ),
        ],
        labels=[
            PANEL_E_LEGEND_LABELS[0],
            PANEL_E_LEGEND_LABELS[1],
            PANEL_E_LEGEND_LABELS[2],
            PANEL_E_LEGEND_LABELS[3],
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.50, 0.247),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handlelength=3.6,
        handletextpad=0.35,
        labelspacing=0.35,
        columnspacing=0.5,
    )


def _draw_panel_f_full(fig: plt.Figure, panel_f: pd.DataFrame) -> None:
    fig.text(0.012, 0.245, "F", fontsize=10, fontweight="bold", ha="left", va="top")
    x_positions = np.linspace(0.070, 0.860, len(PANEL_F_MAPS))
    y_positions = (0.145, 0.060)
    width = 0.128
    height = 0.075
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_data = panel_f.loc[panel_f["held_out_label"].eq(endpoint)]
        truth_rows = endpoint_data.loc[endpoint_data["map_id"].eq("truth")]
        xlim, ylim = _umap_limits(truth_rows)
        for column, (map_id, display) in enumerate(PANEL_F_MAPS):
            axis = fig.add_axes((float(x_positions[column]), y_positions[row], width, height))
            frame = endpoint_data.loc[endpoint_data["map_id"].eq(map_id)]
            represented = frame.loc[frame["is_represented_state"]].sort_values("cell_id")
            held_out = frame.loc[frame["is_held_out_state"]].sort_values("cell_id")
            colors = represented["displayed_assignment"].map(PANEL_F_LABEL_COLORS)
            if colors.isna().any():
                raise HIHAFigure2Error("Composite Panel F contains unmapped biological labels")
            axis.scatter(
                represented["umap_1"],
                represented["umap_2"],
                c=colors,
                s=PANEL_F_REPRESENTED_POINT_SIZE,
                alpha=0.80,
                linewidth=0,
                rasterized=True,
                zorder=1,
            )
            axis.scatter(
                held_out["umap_1"],
                held_out["umap_2"],
                color=PANEL_F_HELD_OUT_COLOR,
                s=PANEL_F_HELD_OUT_POINT_SIZE,
                alpha=1.0,
                linewidth=0,
                rasterized=True,
                zorder=2,
            )
            axis.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            axis.set_aspect("equal", adjustable="box")
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(display, fontsize=MAIN_FIGURE_MIN_FONT_SIZE, pad=2)
        endpoint_label = "HLA-DRhi cDC2" if row == 0 else "ISG+ cDC2"
        fig.text(
            0.032,
            y_positions[row] + height / 2,
            endpoint_label,
            fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
            fontweight="normal",
            ha="center",
            va="center",
            rotation=270,
            color=INK,
            gid=f"panel-f-endpoint-{row}",
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
        for label, color in PANEL_F_LABEL_COLORS.items()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_F_HELD_OUT_COLOR,
            markeredgecolor="none",
            markersize=3.5,
            label="Reference-omitted cells (not evaluated)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.535, 0.010),
        fontsize=MAIN_FIGURE_MIN_FONT_SIZE,
        handletextpad=0.25,
        columnspacing=0.65,
    )


def _draw_main_figure(
    design: dict[str, str],
    panel_b: pd.DataFrame,
    panel_c_summary: pd.DataFrame,
    panel_c_restoration: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
    _draw_panel_a_narrow(fig, design)
    _draw_panel_b_restoration(fig, panel_c_summary, panel_c_restoration)
    _draw_panel_c_metric_bars(fig, panel_b)
    _draw_panel_d_metric_bars(fig, panel_d)
    _draw_panel_e_full(fig, panel_e)
    _draw_panel_f_full(fig, panel_f)
    return fig


def _draw_panel_figure(sources: tuple[object, ...], letter: str) -> plt.Figure:
    (
        design,
        panel_b,
        panel_c_summary,
        panel_c_restoration,
        panel_d,
        panel_e,
        panel_f,
    ) = sources
    drawers = {
        "A": lambda fig: _draw_panel_a_narrow(fig, design),
        "B": lambda fig: _draw_panel_b_restoration(
            fig, panel_c_summary, panel_c_restoration
        ),
        "C": lambda fig: _draw_panel_c_metric_bars(fig, panel_b),
        "D": lambda fig: _draw_panel_d_metric_bars(fig, panel_d),
        "E": lambda fig: _draw_panel_e_full(fig, panel_e),
        "F": lambda fig: _draw_panel_f_full(fig, panel_f),
    }
    if letter not in drawers:
        raise HIHAFigure2Error(f"Unknown Figure 2 panel: {letter}")
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
    drawers[letter](fig)
    return fig


def _panel_crop_bbox_inches(fig: plt.Figure, *, padding_inches: float = 0.04):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        axis.get_tightbbox(renderer)
        for axis in fig.axes
        if axis.get_visible() and axis.get_tightbbox(renderer) is not None
    ]
    boxes.extend(
        text.get_window_extent(renderer)
        for text in fig.findobj(match=Text)
        if text.get_visible() and text.get_text()
    )
    boxes.extend(legend.get_window_extent(renderer) for legend in fig.legends)
    if not boxes:
        raise HIHAFigure2Error("Cannot crop a Figure 2 panel without visible artists")
    return Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(padding_inches)


def generate_panel_images(
    *,
    letters: Iterable[str] = MAIN_FIGURE_PANEL_SET,
    panel_root: Path = PANEL_ROOT,
    source_root: Path = SOURCE_DATA_ROOT,
    artifact_prefix: str = "panel",
) -> dict[str, Path]:
    sources = _read_composite_sources(source_root)
    outputs: dict[str, Path] = {}
    for raw_letter in letters:
        letter = raw_letter.upper()
        fig = _draw_panel_figure(sources, letter)
        _validate_main_figure_fonts(fig)
        crop = _panel_crop_bbox_inches(fig)
        panel_outputs = {
            suffix: panel_root / f"{artifact_prefix}_{letter.lower()}.{suffix}"
            for suffix in ("png", "pdf", "tiff")
        }
        legacy_svg = panel_root / f"{artifact_prefix}_{letter.lower()}.svg"
        if legacy_svg.is_file():
            legacy_svg.unlink()
        for suffix, path in panel_outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {
                "dpi": PANEL_RASTER_DPI if suffix in {"png", "tiff"} else None,
                "facecolor": "white",
                "bbox_inches": crop,
            }
            if suffix == "tiff":
                save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(path, **save_kwargs)
        outputs[f"panel_{letter.lower()}"] = panel_outputs["png"]
        plt.close(fig)
    return outputs


def _validate_main_figure_fonts(fig: plt.Figure) -> None:
    font_sizes = [
        float(text.get_fontsize())
        for text in fig.findobj(match=Text)
        if text.get_text()
        and text.get_gid() not in PANEL_A_SMALL_TEXT_GIDS
        and text.get_gid() not in PANEL_B_ENDPOINT_GIDS
    ]
    if not font_sizes or min(font_sizes) < MAIN_FIGURE_MIN_FONT_SIZE:
        minimum = min(font_sizes) if font_sizes else None
        raise HIHAFigure2Error(
            "Main Figure 2 contains text below the final-size font floor; "
            f"minimum={minimum}, required={MAIN_FIGURE_MIN_FONT_SIZE}"
        )


def _validate_main_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(MAIN_FIGURE_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(MAIN_FIGURE_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        if image.width < expected_width - 4 or image.height < expected_height - 4:
            raise HIHAFigure2Error(
                f"Main Figure 2 raster is below its size target: {path} has {image.size}"
            )
        aspect = image.width / image.height
        if not 0.75 <= aspect <= 0.80:
            raise HIHAFigure2Error(
                f"Unexpected main Figure 2 aspect ratio: {path} has {image.size}"
            )


def generate_main_figure(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    source_root: Path = SOURCE_DATA_ROOT,
    project_root: Path | None = None,
) -> dict[str, Path]:
    (
        design,
        panel_b,
        panel_c_summary,
        panel_c_restoration,
        panel_d,
        panel_e,
        panel_f,
    ) = _read_composite_sources(source_root)
    docs_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    docs_figure_root = docs_root / "figs"
    docs_figure_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        suffix: docs_figure_root / f"manuscript_fig_hiha_main.{suffix}"
        for suffix in ("png", "pdf", "tiff")
    }
    result_outputs = {
        suffix: result_root / f"main_figure.{suffix}" for suffix in ("png", "pdf", "tiff")
    }
    for legacy_svg in (
        docs_figure_root / "manuscript_fig_hiha_main.svg",
        result_root / "main_figure.svg",
    ):
        if legacy_svg.is_file():
            legacy_svg.unlink()
    manifest_path = result_root / "main_figure_manifest.yaml"

    fig = _draw_main_figure(
        design,
        panel_b,
        panel_c_summary,
        panel_c_restoration,
        panel_d,
        panel_e,
        panel_f,
    )
    _validate_main_figure_fonts(fig)
    for suffix, path in outputs.items():
        save_kwargs = {
            "dpi": PANEL_RASTER_DPI if suffix in {"png", "tiff"} else None,
            "facecolor": "white",
        }
        if suffix == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **save_kwargs)
        fig.savefig(result_outputs[suffix], **save_kwargs)
    plt.close(fig)

    source_paths = [
        source_root / "panel_a_design.csv",
        source_root / "panel_b_detection.csv",
        source_root / "panel_c_deficit_by_seed.csv",
        source_root / "panel_c_restoration_by_seed.csv",
        _resolve_source_path(
            source_root,
            "panel_d_label_transfer.csv",
            "panel_e_label_transfer.csv",
        ),
        _resolve_source_path(
            source_root,
            "panel_e_umap_cells.csv",
            "panel_d_umap_cells.csv",
        ),
        source_root / "panel_f_label_assignment_cells.csv",
    ]
    manifest_root = project_root.resolve() if project_root is not None else None

    def manifest_path_value(path: Path) -> str:
        if manifest_root is None:
            return str(path)
        try:
            return str(path.resolve().relative_to(manifest_root))
        except ValueError as error:
            raise HIHAFigure2Error(
                f"Canonical manifest path is outside the repository: {path}"
            ) from error

    deterministic_outputs = [
        path
        for output_group in (outputs, result_outputs)
        for suffix, path in output_group.items()
        if suffix in {"png", "tiff"}
    ]
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel_set": list(MAIN_FIGURE_PANEL_SET),
                "panel_contract": {
                    "B": "paired_reference_restoration_response",
                    "C": "within_cdc2_omitted_state_ranking",
                    "D": "represented_state_label_transfer",
                },
                "panel_d_disposition": "main_figure_panel",
                "panel_e_disposition": "main_figure_panel",
                "panel_f_disposition": "main_figure_panel",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py --panel main"
                ),
                "artifacts": {
                    suffix: manifest_path_value(path) for suffix, path in outputs.items()
                },
                "result_artifacts": {
                    suffix: manifest_path_value(path) for suffix, path in result_outputs.items()
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
                    "panel_a_condition_font_size_points": (PANEL_A_CONDITION_FONT_SIZE),
                    "panel_b_endpoint_label_alignment": "right_edge_rotated_270",
                    "panel_b_endpoint_font_size_points": PANEL_B_ENDPOINT_FONT_SIZE,
                    "panel_b_axis_bounds_fraction": [0.54, 0.82, 0.37, 0.135],
                    "panel_b_encoding": {
                        "rows": [
                            "reference_omitted_cell_median_deficit_decrease",
                            "represented_control_median_deficit_decrease",
                            "control_adjusted_median_deficit_decrease",
                            "median_restored_state_destination_fraction",
                        ],
                        "bar": "arithmetic_mean_across_five_donor_splits",
                        "whisker": "sample_sd_ddof1",
                        "points": "not_rendered",
                        "axis": [0.0, 1.0],
                        "axis_label": "Metric value",
                    },
                    "panel_e_representative_seed": REPRESENTATIVE_SEED,
                    "panel_e_legend_columns": 4,
                    "panel_f_representative_seed": REPRESENTATIVE_SEED,
                    "spatial_panel_layout": {
                        "panels": ["E", "F"],
                        "method_columns": "aligned",
                        "axis_width_fraction": 0.128,
                        "axis_height_fraction": 0.075,
                        "endpoint_labels": "left_border_rotated_270",
                        "endpoint_label_weight": "normal",
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
                    "dense_umap_points_rasterized_in_vector_outputs": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [*outputs.values(), *result_outputs.values(), manifest_path]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(f"Missing or empty main Figure 2 artifact: {path}")
    _validate_main_raster(outputs["png"])
    return {**outputs, "manifest": manifest_path}
