from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import yaml
from PIL import Image


PROJECT_ROOT = Path(".")
DOCS_ROOT = PROJECT_ROOT / "docs"
RESULT_ROOT = PROJECT_ROOT / "results/HIHA_DC/manuscript/figure2"
PANEL_ROOT = PROJECT_ROOT / "results/HIHA_DC/manuscript/figure2_panels"
SOURCE_DATA_ROOT = RESULT_ROOT / "source_data"
OVERVIEW_PATH = (
    PROJECT_ROOT / "results/HIHA_DC/main/experiment_overview/table_1_experiment_overview.csv"
)
COMPARISON_DETECTION_PATH = (
    PROJECT_ROOT
    / "results/HIHA_DC/compare_baselines/tables/compare_detection_by_run.csv"
)
PRIMARY_DETECTION_PATH = (
    PROJECT_ROOT / "results/HIHA_DC/main/tables/main_detection_by_run.csv"
)
COMPARISON_LABEL_TRANSFER_PATH = (
    PROJECT_ROOT
    / "results/HIHA_DC/compare_baselines/tables/"
    "compare_shared_label_transfer_by_run.csv"
)
RUNS_ROOT = PROJECT_ROOT / "runs"
CANDIDATE_SET = "hiha_harmony30_k100"
TRANSPORT_ETA = 1.0e-12

PANEL_A_SIZE_INCHES = (178 / 25.4, 42 / 25.4)
PANEL_B_SIZE_INCHES = (178 / 25.4, 72 / 25.4)
PANEL_C_SIZE_INCHES = (178 / 25.4, 82 / 25.4)
PANEL_D_SIZE_INCHES = (178 / 25.4, 72 / 25.4)
PANEL_E_SIZE_INCHES = (178 / 25.4, 108 / 25.4)
PANEL_F_SIZE_INCHES = (178 / 25.4, 82 / 25.4)
PANEL_RASTER_DPI = 350
PANEL_B_XLABEL = "Average precision (AP) for held-out-state ranking"
PANEL_C_XLABEL = "Median-deficit decrease"
PANEL_D_XLABEL = "Metric value"
PANEL_D_YLABEL = "Metric value"
PANEL_D_LIM = (0.80, 1.0)
PANEL_D_METRICS = (
    ("forced_macro_f1", "Forced macro-F1", "#4C78A8"),
    ("forced_accuracy", "Forced accuracy", "#F2A65A"),
)

ENDPOINTS = ("HLA-DRhi cDC2", "ISG+ cDC2")
PANEL_B_METHODS = (
    ("coreot_full", "u", "CoRe-OT"),
    ("uniform_uot", "u", "Uniform UOT"),
    ("prior_only", "prior_risk", "Prior only"),
    ("seurat_anchor", "u", "Seurat"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
PANEL_E_METHODS = PANEL_B_METHODS
PANEL_E_MAPS = (("truth", "evaluation_truth", "Held-out truth"), *PANEL_E_METHODS)
PANEL_B_POSITIONS = np.array([0.0, 1.0, 2.0, 3.4, 4.4, 5.4])
PANEL_C_REFERENCE = ("coreot_full", "u", "CoRe-OT")
PANEL_C_COMPARATORS = (
    ("coreot_match_only", "u", "CoRe-OT\n(−C)"),
    ("uniform_uot", "u", "Uniform UOT"),
    ("prior_only", "prior_risk", "Prior only"),
)
PANEL_D_METHODS = (
    ("coreot_full", "u", "CoRe-OT"),
    ("uniform_uot", "u", "Uniform UOT"),
    ("seurat_anchor", "u", "Seurat"),
    ("scmap_cluster", "u", "scmap-cluster"),
    ("chetah", "u", "CHETAH"),
)
PANEL_D_CANDIDATE_SET_BY_METHOD = {
    "coreot_full": CANDIDATE_SET,
    "uniform_uot": CANDIDATE_SET,
    "seurat_anchor": "external_reference_mapping",
    "scmap_cluster": "external_reference_mapping",
    "chetah": "external_reference_mapping",
}
PANEL_F_MAPS = (("truth", "Ground truth"),) + tuple(
    (
        method,
        {"coreot_full": "CoRe-OT", "seurat_anchor": "Seurat"}.get(method, display),
    )
    for method, _, display in PANEL_D_METHODS
)
PANEL_F_LABEL_COLORS = {
    "ASDC": "#B279A2",
    "CD14+ cDC2": "#F58518",
    "HLA-DRhi cDC2": "#4C78A8",
    "ISG+ cDC2": "#54A24B",
    "cDC1": "#E45756",
    "pDC": "#72B7B2",
}
PANEL_F_HELD_OUT_COLOR = "#D9D9D9"
PANEL_F_REPRESENTED_POINT_SIZE = 1.5
PANEL_F_HELD_OUT_POINT_SIZE = 2.5
PANEL_F_ENDPOINT_LABEL_Y = (0.73, 0.33)
PANEL_F_LEGEND_NCOLS = 7
PANEL_C_GROUPS = (
    ("held_out", "Held-out cells"),
    ("represented_cdc2", "Represented cDC2"),
)
PANEL_C_CONDITIONS = ("incomplete_reference", "full_reference_control")
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.3
UMAP_RANDOM_STATE = 1
REPRESENTATIVE_SEED = 1
BROAD_DESTINATION_ORDER = ("ASDC", "cDC1", "cDC2", "pDC")
BROAD_DESTINATION_COLORS = {
    "ASDC": "#CC79A7",
    "cDC1": "#E69F00",
    "cDC2": "#0072B2",
    "pDC": "#009E73",
}
METHOD_COLORS = {
    "coreot_full": "#0072B2",
    "uniform_uot": "#D55E00",
    "coreot_match_only": "#009E73",
    "prior_only": "#009E73",
    "seurat_anchor": "#CC79A7",
    "scmap_cluster": "#E69F00",
    "chetah": "#6A3D9A",
}

INK = "#242424"
MUTED_INK = "#5E5E5E"
QUERY_FILL = "#DCECF5"
REFERENCE_FILL = "#F2F2F2"
HELD_OUT = "#D55E00"
RESTORED = "#009E73"
LIGHT_EDGE = "#A8A8A8"
PANEL_E_ELIGIBLE_GRAY = "#C8C8C8"
PANEL_E_CONTEXT_GRAY = "#E8E8E8"
PANEL_E_TRUTH_COLOR = "#D73027"


class HIHAFigure2Error(ValueError):
    """Raised when retained artifacts cannot support HIHA Figure 2."""


def _read_design_source(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["characteristic", "value"]:
        raise HIHAFigure2Error(f"Unexpected experiment-overview schema: {path}")
    if frame["characteristic"].duplicated().any():
        raise HIHAFigure2Error(f"Duplicate experiment-overview characteristics: {path}")
    values = frame.set_index("characteristic")["value"].astype(str).to_dict()
    required = ("Number of subjects", "Donor split", "Main held-out labels")
    missing = [key for key in required if key not in values]
    if missing:
        raise HIHAFigure2Error(f"Missing experiment-overview fields {missing}: {path}")
    if values["Number of subjects"] != "108":
        raise HIHAFigure2Error(
            f"Panel A is specified for 108 subjects; found {values['Number of subjects']}"
        )
    split = values["Donor split"]
    if "80% reference" not in split or "20% query" not in split or "subject-level" not in split:
        raise HIHAFigure2Error(f"Panel A requires the recorded subject-level 80/20 split: {split}")
    for endpoint in ("HLA-DRhi cDC2", "ISG+ cDC2"):
        if endpoint not in values["Main held-out labels"]:
            raise HIHAFigure2Error(f"Panel A endpoint is absent from experiment overview: {endpoint}")
    return values


def _rounded_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = LIGHT_EDGE,
    linewidth: float = 0.8,
    fontsize: float = 7.0,
    text_color: str = INK,
    fontweight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
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
        color=text_color,
        fontweight=fontweight,
        linespacing=1.15,
        zorder=3,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED_INK,
    connectionstyle: str = "arc3",
    linewidth: float = 0.8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=2,
            zorder=2,
        )
    )


def _draw_panel_a(ax: plt.Axes, *, n_subjects: str) -> None:
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.005, 0.965, "A", ha="left", va="top", fontsize=11, fontweight="bold")

    _rounded_box(
        ax,
        x=0.025,
        y=0.34,
        width=0.13,
        height=0.29,
        text=f"HIHA DCs\n{n_subjects} subjects",
        facecolor=REFERENCE_FILL,
        edgecolor=INK,
        linewidth=0.9,
        fontweight="bold",
    )
    _rounded_box(
        ax,
        x=0.175,
        y=0.39,
        width=0.105,
        height=0.20,
        text="Subject-level\nsplit",
        facecolor="white",
        fontsize=6.8,
    )
    _arrow(ax, (0.158, 0.49), (0.172, 0.49))
    _arrow(ax, (0.283, 0.53), (0.312, 0.68), connectionstyle="arc3,rad=-0.08")
    _arrow(ax, (0.283, 0.45), (0.312, 0.29), connectionstyle="arc3,rad=0.08")

    _rounded_box(
        ax,
        x=0.315,
        y=0.58,
        width=0.14,
        height=0.22,
        text="Query 20%\nall states retained",
        facecolor=QUERY_FILL,
        edgecolor="#6B9FBC",
    )
    _rounded_box(
        ax,
        x=0.315,
        y=0.18,
        width=0.14,
        height=0.22,
        text="Reference 80%",
        facecolor=REFERENCE_FILL,
    )

    _arrow(ax, (0.458, 0.29), (0.512, 0.58), connectionstyle="arc3,rad=-0.13")
    _arrow(ax, (0.458, 0.29), (0.512, 0.27), connectionstyle="arc3,rad=0.02")
    ax.plot(
        (0.458, 0.485, 0.485, 0.82),
        (0.69, 0.69, 0.84, 0.84),
        color="#6B9FBC",
        lw=0.8,
        zorder=2,
    )
    _arrow(ax, (0.82, 0.84), (0.82, 0.738), color="#6B9FBC")

    _rounded_box(
        ax,
        x=0.515,
        y=0.51,
        width=0.18,
        height=0.22,
        text="Ablated reference\ntarget absent",
        facecolor="#FFF1EA",
        edgecolor=HELD_OUT,
        linewidth=1.0,
    )
    _rounded_box(
        ax,
        x=0.515,
        y=0.20,
        width=0.18,
        height=0.22,
        text="Matched full reference\ntarget present",
        facecolor="#E9F5EF",
        edgecolor=RESTORED,
        linewidth=1.0,
    )

    _arrow(ax, (0.698, 0.62), (0.732, 0.62), color=HELD_OUT)
    _arrow(ax, (0.698, 0.31), (0.732, 0.31), color=RESTORED)
    _rounded_box(
        ax,
        x=0.735,
        y=0.25,
        width=0.24,
        height=0.48,
        text=(
            "Within-cDC2 evaluation\n"
            "Ablated: held-out-state ranking\n"
            "Ablated vs full: deficit rescue"
        ),
        facecolor="white",
        edgecolor=INK,
        linewidth=0.9,
        fontsize=6.6,
    )


def _validate_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(PANEL_A_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(PANEL_A_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        rounding_tolerance = 4
        if (
            image.width < expected_width - rounding_tolerance
            or image.height < expected_height - rounding_tolerance
        ):
            raise HIHAFigure2Error(
                f"Panel A raster is below its size target: {path} has {image.size}"
            )
        if image.width / image.height < 4.0:
            raise HIHAFigure2Error(f"Panel A is not a horizontal strip: {path} has {image.size}")


def _select_panel_b_detection(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "run_id",
        "evaluation_run_id",
        "score_run_id",
        "held_out_label",
        "seed",
        "method",
        "score",
        "auprc",
        "auprc_baseline",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(f"Panel B source is missing columns {missing}: {path}")

    if "evaluation_scope" in frame.columns:
        frame = frame.loc[
            frame["evaluation_scope"].eq("local_within_broad_state")
        ].copy()

    method_keys = {(method, score) for method, score, _ in PANEL_B_METHODS}
    selected = frame.loc[
        frame["held_out_label"].isin(ENDPOINTS)
        & frame.apply(
            lambda row: (str(row["method"]), str(row["score"])) in method_keys,
            axis=1,
        )
    ].copy()
    expected = {
        (endpoint, seed, method, score)
        for endpoint in ENDPOINTS
        for seed in range(1, 6)
        for method, score, _ in PANEL_B_METHODS
    }
    observed = set(
        zip(
            selected["held_out_label"],
            selected["seed"],
            selected["method"],
            selected["score"],
            strict=False,
        )
    )
    if observed != expected:
        missing_keys = sorted(expected - observed)
        extra_keys = sorted(observed - expected)
        raise HIHAFigure2Error(
            f"Panel B keys differ from the two-endpoint, five-seed contract; "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    keys = ["held_out_label", "seed", "method", "score"]
    if selected.duplicated(keys).any():
        raise HIHAFigure2Error(f"Panel B source has duplicate keys: {keys}")
    for column in ("run_id", "evaluation_run_id", "score_run_id"):
        selected[column] = selected[column].fillna("").astype(str)
        if selected[column].eq("").any():
            raise HIHAFigure2Error(f"Panel B {column} must identify every source run")
    if not selected["run_id"].eq(selected["evaluation_run_id"]).all():
        raise HIHAFigure2Error(
            "Panel B run_id must equal the explicit evaluation_run_id"
        )
    uniform = selected["method"].eq("uniform_uot")
    if selected.loc[uniform, "score_run_id"].eq(
        selected.loc[uniform, "evaluation_run_id"]
    ).any():
        raise HIHAFigure2Error(
            "Panel B Uniform UOT rows must retain their replacement score_run_id"
        )
    for column in ("auprc", "auprc_baseline"):
        values = pd.to_numeric(selected[column], errors="coerce")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise HIHAFigure2Error(f"Panel B {column} must contain finite values in [0, 1]")
        selected[column] = values
    prevalence_counts = selected.groupby(["held_out_label", "seed"])[
        "auprc_baseline"
    ].nunique()
    if not prevalence_counts.eq(1).all():
        raise HIHAFigure2Error("Panel B prevalence differs across methods within a split")

    display = {method: name for method, _, name in PANEL_B_METHODS}
    selected["method_display"] = selected["method"].map(display)
    selected["average_precision"] = selected["auprc"]
    selected["positive_prevalence"] = selected["auprc_baseline"]
    selected["source_path"] = str(path)
    return selected[
        [
            "run_id",
            "evaluation_run_id",
            "score_run_id",
            "held_out_label",
            "seed",
            "method",
            "method_display",
            "score",
            "average_precision",
            "positive_prevalence",
            "source_path",
        ]
    ].sort_values(["held_out_label", "method_display", "seed"], ignore_index=True)


def _draw_panel_b(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=PANEL_B_SIZE_INCHES,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    order = [method for method, _, _ in PANEL_B_METHODS]
    display = {method: name for method, _, name in PANEL_B_METHODS}
    positions = PANEL_B_POSITIONS
    seed_offsets = np.linspace(-0.12, 0.12, 5)

    for column, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        endpoint_data = data.loc[data["held_out_label"].eq(endpoint)]
        for y, method in zip(positions, order, strict=True):
            values = (
                endpoint_data.loc[endpoint_data["method"].eq(method)]
                .sort_values("seed")["average_precision"]
                .to_numpy(dtype=float)
            )
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            color = METHOD_COLORS[method]
            ax.scatter(
                values,
                y + seed_offsets,
                s=20,
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
                markersize=6,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.7,
                ecolor=color,
                elinewidth=1.1,
                capsize=3,
                capthick=1.0,
                zorder=3,
            )

        prevalence_by_seed = endpoint_data.groupby("seed")["positive_prevalence"].first()
        mean_prevalence = float(prevalence_by_seed.mean())
        ax.axvline(
            mean_prevalence,
            color="#777777",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        ax.annotate(
            f"Mean prevalence\n{mean_prevalence:.3f}",
            xy=(mean_prevalence, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=6.5,
            color=MUTED_INK,
        )
        ax.set(
            xlim=(0, 1),
            ylim=(positions[-1] + 0.5, -0.55),
        )
        ax.set_title(endpoint, fontsize=8.5, pad=7)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="both", labelsize=7)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.55)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if column == 0:
            ax.set_yticks(positions, [display[method] for method in order])
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.text(0.012, 0.965, "B", ha="left", va="top", fontsize=11, fontweight="bold")
    fig.text(
        0.58,
        0.15,
        PANEL_B_XLABEL,
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="none",
            alpha=0.45,
            markersize=4.5,
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
            markeredgewidth=0.6,
            markersize=5.5,
            linewidth=1.0,
            label="Mean ± sample SD",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.58, 0.025),
        fontsize=6.8,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.88, bottom=0.25, wspace=0.13)
    return fig


def _validate_panel_b_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(PANEL_B_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(PANEL_B_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        if image.width < expected_width - 4 or image.height < expected_height - 4:
            raise HIHAFigure2Error(
                f"Panel B raster is below its size target: {path} has {image.size}"
            )
        aspect = image.width / image.height
        if not 2.35 <= aspect <= 2.60:
            raise HIHAFigure2Error(f"Unexpected Panel B aspect ratio: {path} has {image.size}")


def _select_panel_c_controls(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "run_id",
        "held_out_label",
        "seed",
        "method",
        "score",
        "auprc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(f"Panel C source is missing columns {missing}: {path}")

    selected_methods = (PANEL_C_REFERENCE, *PANEL_C_COMPARATORS)
    method_keys = {(method, score) for method, score, _ in selected_methods}
    selected = frame.loc[
        frame["held_out_label"].isin(ENDPOINTS)
        & frame.apply(
            lambda row: (str(row["method"]), str(row["score"])) in method_keys,
            axis=1,
        )
    ].copy()
    expected = {
        (endpoint, seed, method, score)
        for endpoint in ENDPOINTS
        for seed in range(1, 6)
        for method, score, _ in selected_methods
    }
    observed = set(
        zip(
            selected["held_out_label"],
            selected["seed"],
            selected["method"],
            selected["score"],
            strict=False,
        )
    )
    if observed != expected:
        raise HIHAFigure2Error(
            "Panel C keys differ from the paired two-endpoint, five-seed contract; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    keys = ["held_out_label", "seed", "method", "score"]
    if selected.duplicated(keys).any():
        raise HIHAFigure2Error(f"Panel C source has duplicate keys: {keys}")
    values = pd.to_numeric(selected["auprc"], errors="coerce")
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise HIHAFigure2Error("Panel C auprc must contain finite values in [0, 1]")
    selected["auprc"] = values

    method_order = [method for method, _, _ in selected_methods]
    wide = selected.pivot(
        index=["held_out_label", "seed"],
        columns="method",
        values="auprc",
    )
    if list(wide.columns.intersection(method_order)) != method_order:
        wide = wide.reindex(columns=method_order)
    if wide.isna().any().any():
        raise HIHAFigure2Error("Panel C method pairing produced missing AP values")

    reference_method = PANEL_C_REFERENCE[0]
    display = {method: name for method, _, name in PANEL_C_COMPARATORS}
    score_by_method = {method: score for method, score, _ in PANEL_C_COMPARATORS}
    rows: list[dict[str, object]] = []
    for (endpoint, seed), values_by_method in wide.iterrows():
        full_ap = float(values_by_method[reference_method])
        run_id = str(
            selected.loc[
                selected["held_out_label"].eq(endpoint)
                & selected["seed"].eq(seed)
                & selected["method"].eq(reference_method),
                "run_id",
            ].iloc[0]
        )
        for method, _, _ in PANEL_C_COMPARATORS:
            comparator_ap = float(values_by_method[method])
            rows.append(
                {
                    "run_id": run_id,
                    "held_out_label": endpoint,
                    "seed": int(seed),
                    "reference_method": reference_method,
                    "reference_score": PANEL_C_REFERENCE[1],
                    "reference_average_precision": full_ap,
                    "comparator_method": method,
                    "comparator_display": display[method].replace("\n", " "),
                    "comparator_score": score_by_method[method],
                    "comparator_average_precision": comparator_ap,
                    "delta_average_precision": full_ap - comparator_ap,
                    "source_path": str(path),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != len(ENDPOINTS) * 5 * len(PANEL_C_COMPARATORS):
        raise HIHAFigure2Error(f"Unexpected Panel C paired-row count: {len(result)}")
    return result.sort_values(
        ["held_out_label", "comparator_method", "seed"], ignore_index=True
    )


def _panel_c_limits(data: pd.DataFrame) -> tuple[float, float]:
    values = [*data["delta_average_precision"].astype(float).tolist()]
    for _, group in data.groupby(["held_out_label", "comparator_method"]):
        mean = float(group["delta_average_precision"].mean())
        sample_sd = float(group["delta_average_precision"].std(ddof=1))
        values.extend([mean - sample_sd, mean + sample_sd])
    lower = min(values)
    upper = max(values)
    step = 0.05
    plot_lower = min(-step, np.floor((lower - 0.02) / step) * step)
    plot_upper = max(step, np.ceil((upper + 0.02) / step) * step)
    return float(max(-1, plot_lower)), float(min(1, plot_upper))


def _draw_panel_c(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=PANEL_C_SIZE_INCHES,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    comparator_order = [method for method, _, _ in PANEL_C_COMPARATORS]
    display = {method: name for method, _, name in PANEL_C_COMPARATORS}
    positions = np.arange(len(comparator_order), dtype=float)
    seed_offsets = np.linspace(-0.13, 0.13, 5)
    lower, upper = _panel_c_limits(data)

    for column, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        endpoint_data = data.loc[data["held_out_label"].eq(endpoint)]
        for x, method in zip(positions, comparator_order, strict=True):
            values = (
                endpoint_data.loc[endpoint_data["comparator_method"].eq(method)]
                .sort_values("seed")["delta_average_precision"]
                .to_numpy(dtype=float)
            )
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            color = METHOD_COLORS[method]
            ax.scatter(
                x + seed_offsets,
                values,
                s=22,
                color=color,
                alpha=0.45,
                linewidth=0,
                zorder=3,
            )
            ax.errorbar(
                x,
                mean,
                yerr=sample_sd,
                fmt="o",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.7,
                ecolor=color,
                elinewidth=1.1,
                capsize=3,
                capthick=1.0,
                zorder=4,
            )
        ax.axhline(0, color=INK, linewidth=0.9, zorder=2)
        ax.set(
            xlim=(-0.5, len(comparator_order) - 0.5),
            ylim=(lower, upper),
            xticks=positions,
            xticklabels=[display[method] for method in comparator_order],
        )
        ax.set_title(endpoint, fontsize=8.5, pad=7)
        ax.tick_params(axis="x", labelsize=7, length=0, pad=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.02,
            0.97,
            "CoRe-OT higher ↑",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            color=MUTED_INK,
        )
        if column == 0:
            ax.set_ylabel("Paired difference in AP", fontsize=7.5)
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.text(0.012, 0.965, "C", ha="left", va="top", fontsize=11, fontweight="bold")
    fig.text(
        0.045,
        0.965,
        "Transport and prior controls",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.56,
        0.91,
        "Paired ΔAP = CoRe-OT − comparator • five fixed donor-split seeds",
        ha="center",
        va="top",
        fontsize=7,
        color=MUTED_INK,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="none",
            alpha=0.45,
            markersize=4.5,
            label="Paired donor split",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color="#777777",
            markerfacecolor="#777777",
            markeredgecolor=INK,
            markeredgewidth=0.6,
            markersize=5.5,
            linewidth=1.0,
            label="Mean ± sample SD",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.56, 0.012),
        fontsize=6.8,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.25, wspace=0.12)
    return fig


def _validate_panel_c_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(PANEL_C_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(PANEL_C_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        if image.width < expected_width - 4 or image.height < expected_height - 4:
            raise HIHAFigure2Error(
                f"Panel C raster is below its size target: {path} has {image.size}"
            )
        aspect = image.width / image.height
        if not 2.40 <= aspect <= 2.70:
            raise HIHAFigure2Error(f"Unexpected Panel C aspect ratio: {path} has {image.size}")


def _panel_c_run_index(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"run_id", "held_out_label", "seed", "method", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(f"Panel C run index is missing columns {missing}: {path}")
    selected = frame.loc[
        frame["held_out_label"].isin(ENDPOINTS)
        & frame["method"].eq(PANEL_C_REFERENCE[0])
        & frame["score"].eq(PANEL_C_REFERENCE[1]),
        ["run_id", "held_out_label", "seed"],
    ].drop_duplicates()
    expected = {(endpoint, seed) for endpoint in ENDPOINTS for seed in range(1, 6)}
    observed = set(zip(selected["held_out_label"], selected["seed"], strict=False))
    if observed != expected:
        raise HIHAFigure2Error(
            "Panel C run index differs from the two-endpoint, five-seed contract; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if selected.duplicated(["held_out_label", "seed"]).any():
        raise HIHAFigure2Error("Panel C run index has duplicate endpoint-seed keys")
    return selected.sort_values(["held_out_label", "seed"], ignore_index=True)


def _read_query_truth(run_root: Path, condition: str) -> pd.DataFrame:
    path = run_root / "benchmark" / condition / "evaluation_truth/query_truth.csv"
    frame = pd.read_csv(path)
    required = {
        "cell_id",
        "true_label",
        "removed_state",
        "is_absent_state",
        "is_shared_state",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(f"Figure 2 truth is missing columns {missing}: {path}")
    frame["cell_id"] = frame["cell_id"].astype(str)
    if frame["cell_id"].duplicated().any():
        raise HIHAFigure2Error(f"Figure 2 truth has duplicate query cell IDs: {path}")
    if frame[list(required - {"cell_id"})].isna().any().any():
        raise HIHAFigure2Error(f"Figure 2 truth contains missing values: {path}")
    return frame.sort_values("cell_id", ignore_index=True)


def _read_target_labels(run_root: Path, condition: str) -> pd.DataFrame:
    path = run_root / "benchmark" / condition / "model_visible/target_labels.csv"
    frame = pd.read_csv(path)
    required = {"cell_id", "target_label", "broad_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(f"Figure 2 target labels are missing columns {missing}: {path}")
    frame["cell_id"] = frame["cell_id"].astype(str)
    if frame["cell_id"].duplicated().any():
        raise HIHAFigure2Error(f"Figure 2 target labels have duplicate cell IDs: {path}")
    if frame[list(required)].isna().any().any():
        raise HIHAFigure2Error(f"Figure 2 target labels contain missing values: {path}")
    mapping_counts = frame.groupby("target_label")["broad_label"].nunique()
    if not mapping_counts.eq(1).all():
        raise HIHAFigure2Error(f"Fine labels do not map uniquely to broad labels: {path}")
    return frame


def _read_panel_c_scores(run_root: Path, condition: str) -> pd.DataFrame:
    path = (
        run_root
        / "transport"
        / condition
        / CANDIDATE_SET
        / "coreot_full/cell_transport_scores.parquet"
    )
    frame = pd.read_parquet(path)
    required = {"cell_id", "a", "a_hat", "u"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(f"Panel C transport scores are missing columns {missing}: {path}")
    frame["cell_id"] = frame["cell_id"].astype(str)
    if frame["cell_id"].duplicated().any():
        raise HIHAFigure2Error(f"Panel C transport scores have duplicate cell IDs: {path}")
    numeric = frame[["a", "a_hat", "u"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise HIHAFigure2Error(f"Panel C transport scores contain nonfinite values: {path}")
    if (numeric["a"] <= 0).any() or (numeric["a_hat"] < 0).any():
        raise HIHAFigure2Error(f"Panel C transport scores contain invalid masses: {path}")
    reconstructed_u = np.maximum(numeric["a"] - numeric["a_hat"], 0.0)
    reconstructed_u /= numeric["a"] + TRANSPORT_ETA
    if not np.allclose(reconstructed_u, numeric["u"], rtol=1.0e-7, atol=1.0e-10):
        raise HIHAFigure2Error(
            f"Panel C deficit does not match [a-a_hat]_+/(a+eta): {path}"
        )
    if not numeric["u"].between(0, 1).all():
        raise HIHAFigure2Error(f"Panel C deficit must lie in [0, 1]: {path}")
    frame[["a", "a_hat", "u"]] = numeric
    return frame[["cell_id", "a", "a_hat", "u"]].sort_values(
        "cell_id", ignore_index=True
    )


def _read_label_probability_matrix(
    run_root: Path,
    condition: str,
    expected_cell_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = (
        run_root
        / "transport"
        / condition
        / CANDIDATE_SET
        / "coreot_full/label_probabilities.npz"
    )
    with np.load(path, allow_pickle=True) as archive:
        required = {"cell_ids", "labels", "probabilities"}
        if set(archive.files) != required:
            raise HIHAFigure2Error(
                f"Panel C label-probability archive has unexpected keys: {path}"
            )
        cell_ids = np.asarray(archive["cell_ids"]).astype(str)
        labels = np.asarray(archive["labels"]).astype(str)
        probabilities = np.asarray(archive["probabilities"], dtype=float)
    if probabilities.shape != (len(cell_ids), len(labels)):
        raise HIHAFigure2Error(f"Panel C label-probability shape mismatch: {path}")
    if len(set(cell_ids)) != len(cell_ids) or len(set(labels)) != len(labels):
        raise HIHAFigure2Error(f"Panel C label-probability IDs or labels are duplicated: {path}")
    if set(cell_ids) != expected_cell_ids:
        raise HIHAFigure2Error(f"Label-probability query IDs differ from truth: {path}")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < -1.0e-12).any()
        or (probabilities > 1 + 1.0e-12).any()
    ):
        raise HIHAFigure2Error(f"Label probabilities are invalid: {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-7):
        raise HIHAFigure2Error(f"Conditional probabilities do not sum to one: {path}")
    return cell_ids, labels, probabilities, path


def _read_restored_probabilities(
    run_root: Path,
    held_out_label: str,
    expected_cell_ids: set[str],
) -> pd.Series:
    cell_ids, labels, probabilities, path = _read_label_probability_matrix(
        run_root,
        "full_reference_control",
        expected_cell_ids,
    )
    if held_out_label not in set(labels):
        raise HIHAFigure2Error(f"Restored label {held_out_label} is absent from {path}")
    restored_column = int(np.flatnonzero(labels == held_out_label)[0])
    return pd.Series(
        probabilities[:, restored_column],
        index=pd.Index(cell_ids, name="cell_id"),
        name="restored_state_probability",
    )


def _collect_panel_c_data(
    *,
    detection_path: Path,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    run_index = _panel_c_run_index(detection_path)
    cell_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    specificity_rows: list[dict[str, object]] = []
    source_paths: list[str] = [str(detection_path)]

    for run in run_index.itertuples(index=False):
        run_root = runs_root / str(run.run_id)
        if not run_root.is_dir():
            raise HIHAFigure2Error(f"Panel C run root is missing: {run_root}")
        truths = {
            condition: _read_query_truth(run_root, condition)
            for condition in PANEL_C_CONDITIONS
        }
        truth_columns = ["cell_id", "true_label", "removed_state"]
        if not truths[PANEL_C_CONDITIONS[0]][truth_columns].equals(
            truths[PANEL_C_CONDITIONS[1]][truth_columns]
        ):
            raise HIHAFigure2Error(f"Panel C paired query truth differs across conditions: {run_root}")
        truth = truths["incomplete_reference"].copy()
        removed = truth["removed_state"].astype(str).unique().tolist()
        if removed != [str(run.held_out_label)]:
            raise HIHAFigure2Error(
                f"Panel C removed-state truth disagrees with run index: {run_root}"
            )

        target_labels = {
            condition: _read_target_labels(run_root, condition)
            for condition in PANEL_C_CONDITIONS
        }
        incomplete_states = set(target_labels["incomplete_reference"]["target_label"])
        full_states = set(target_labels["full_reference_control"]["target_label"])
        if str(run.held_out_label) in incomplete_states:
            raise HIHAFigure2Error(f"Held-out state is present in ablated reference: {run_root}")
        if str(run.held_out_label) not in full_states:
            raise HIHAFigure2Error(f"Held-out state is absent from matched full reference: {run_root}")

        full_mapping = (
            target_labels["full_reference_control"][["target_label", "broad_label"]]
            .drop_duplicates()
            .set_index("target_label")["broad_label"]
        )
        truth["broad_label"] = truth["true_label"].map(full_mapping)
        if truth["broad_label"].isna().any():
            missing_labels = sorted(
                truth.loc[truth["broad_label"].isna(), "true_label"].astype(str).unique()
            )
            raise HIHAFigure2Error(
                f"Panel C query labels lack a full-reference broad mapping "
                f"{missing_labels}: {run_root}"
            )
        truth["truth_group"] = np.select(
            [
                truth["true_label"].eq(str(run.held_out_label)),
                truth["broad_label"].eq("cDC2"),
            ],
            ["held_out", "represented_cdc2"],
            default="other",
        )
        expected_absent = truth["truth_group"].eq("held_out")
        if not expected_absent.equals(truth["is_absent_state"].astype(bool)):
            raise HIHAFigure2Error(f"Panel C held-out grouping disagrees with truth: {run_root}")
        full_truth = truths["full_reference_control"]
        if full_truth["is_absent_state"].astype(bool).any():
            raise HIHAFigure2Error(
                f"Panel C full-reference truth still marks absent query states: {run_root}"
            )
        if not full_truth["is_shared_state"].astype(bool).all():
            raise HIHAFigure2Error(
                f"Panel C full-reference truth does not mark every query state shared: {run_root}"
            )

        scores = {
            condition: _read_panel_c_scores(run_root, condition)
            for condition in PANEL_C_CONDITIONS
        }
        truth_ids = set(truth["cell_id"])
        for condition, frame in scores.items():
            if set(frame["cell_id"]) != truth_ids:
                raise HIHAFigure2Error(
                    f"Panel C score IDs differ from truth for {condition}: {run_root}"
                )
        paired = truth.merge(
            scores["incomplete_reference"][["cell_id", "u"]],
            on="cell_id",
            validate="one_to_one",
        ).merge(
            scores["full_reference_control"][["cell_id", "u"]],
            on="cell_id",
            validate="one_to_one",
            suffixes=("_ablated", "_full"),
        )
        restored_probability = _read_restored_probabilities(
            run_root,
            str(run.held_out_label),
            truth_ids,
        )
        paired["restored_state_probability"] = paired["cell_id"].map(
            restored_probability
        )
        paired["delta_u_cellwise"] = paired["u_ablated"] - paired["u_full"]
        paired = paired.loc[paired["truth_group"].isin(["held_out", "represented_cdc2"])]
        paired.insert(0, "run_id", str(run.run_id))
        paired.insert(1, "held_out_label", str(run.held_out_label))
        paired.insert(2, "seed", int(run.seed))
        paired.loc[
            paired["truth_group"].ne("held_out"), "restored_state_probability"
        ] = np.nan
        cell_frames.append(
            paired[
                [
                    "run_id",
                    "held_out_label",
                    "seed",
                    "cell_id",
                    "true_label",
                    "broad_label",
                    "truth_group",
                    "u_ablated",
                    "u_full",
                    "delta_u_cellwise",
                    "restored_state_probability",
                ]
            ]
        )

        group_summaries: dict[str, dict[str, float | int]] = {}
        for group, _ in PANEL_C_GROUPS:
            subset = paired.loc[paired["truth_group"].eq(group)]
            if subset.empty:
                raise HIHAFigure2Error(
                    f"Panel C group {group} is empty for {run.held_out_label}, seed={run.seed}"
                )
            median_ablated = float(subset["u_ablated"].median())
            median_full = float(subset["u_full"].median())
            group_summaries[group] = {
                "n_cells": int(len(subset)),
                "median_u_ablated": median_ablated,
                "median_u_full": median_full,
                "delta_median_u": median_ablated - median_full,
            }
            summary_rows.append(
                {
                    "run_id": str(run.run_id),
                    "held_out_label": str(run.held_out_label),
                    "seed": int(run.seed),
                    "truth_group": group,
                    **group_summaries[group],
                }
            )
        held = group_summaries["held_out"]
        control = group_summaries["represented_cdc2"]
        held_probabilities = paired.loc[
            paired["truth_group"].eq("held_out"), "restored_state_probability"
        ]
        specificity_rows.append(
            {
                "run_id": str(run.run_id),
                "held_out_label": str(run.held_out_label),
                "seed": int(run.seed),
                "held_out_delta_median_u": held["delta_median_u"],
                "control_delta_median_u": control["delta_median_u"],
                "restoration_specificity": (
                    float(held["delta_median_u"]) - float(control["delta_median_u"])
                ),
                "n_held_out_cells": held["n_cells"],
                "median_restored_state_probability": float(
                    held_probabilities.median()
                ),
                "mean_restored_state_probability": float(held_probabilities.mean()),
            }
        )
        source_paths.extend(
            [
                str(
                    run_root
                    / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
                ),
                str(
                    run_root
                    / "benchmark/full_reference_control/model_visible/target_labels.csv"
                ),
                str(
                    run_root
                    / "transport/incomplete_reference"
                    / CANDIDATE_SET
                    / "coreot_full/cell_transport_scores.parquet"
                ),
                str(
                    run_root
                    / "transport/full_reference_control"
                    / CANDIDATE_SET
                    / "coreot_full/cell_transport_scores.parquet"
                ),
                str(
                    run_root
                    / "transport/full_reference_control"
                    / CANDIDATE_SET
                    / "coreot_full/label_probabilities.npz"
                ),
            ]
        )

    cells = pd.concat(cell_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["held_out_label", "seed", "truth_group"], ignore_index=True
    )
    specificity = pd.DataFrame(specificity_rows).sort_values(
        ["held_out_label", "seed"], ignore_index=True
    )
    if len(summary) != len(ENDPOINTS) * 5 * len(PANEL_C_GROUPS):
        raise HIHAFigure2Error(f"Unexpected Panel C summary-row count: {len(summary)}")
    if len(specificity) != len(ENDPOINTS) * 5:
        raise HIHAFigure2Error(
            f"Unexpected Panel C specificity-row count: {len(specificity)}"
        )
    numeric = specificity[
        [
            "held_out_delta_median_u",
            "control_delta_median_u",
            "restoration_specificity",
            "median_restored_state_probability",
            "mean_restored_state_probability",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise HIHAFigure2Error("Panel C specificity summary contains nonfinite values")
    return cells, summary, specificity, list(dict.fromkeys(source_paths))


def _draw_panel_c_rescue(
    summary: pd.DataFrame,
    specificity: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=PANEL_C_SIZE_INCHES)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, width_ratios=(3.35, 1.10), wspace=0.34)
    rescue_axis = fig.add_subplot(grid[0, 0])
    probability_axis = fig.add_subplot(grid[0, 1], sharey=rescue_axis)

    rescue_rows = (
        ("held_out_delta_median_u", "Held-out cells", "held_out"),
        ("control_delta_median_u", "Represented cDC2", "represented_cdc2"),
        (
            "restoration_specificity",
            "Control-adjusted, $\\Delta^{\\mathrm{CA}}$",
            "specificity",
        ),
    )
    colors = {
        "held_out": METHOD_COLORS["coreot_full"],
        "represented_cdc2": "#777777",
        "specificity": HELD_OUT,
        "destination": RESTORED,
    }
    seed_offsets = np.linspace(-0.14, 0.14, 5)
    endpoint_positions = {
        ENDPOINTS[0]: np.array([5.4, 4.4, 3.4]),
        ENDPOINTS[1]: np.array([1.8, 0.8, -0.2]),
    }
    endpoint_centers = {
        endpoint: float(positions.mean())
        for endpoint, positions in endpoint_positions.items()
    }

    rescue_values = specificity[
        [
            "held_out_delta_median_u",
            "control_delta_median_u",
            "restoration_specificity",
        ]
    ].to_numpy(dtype=float)
    rescue_lower = min(
        -0.01,
        float(np.floor((rescue_values.min() - 0.01) / 0.02) * 0.02),
    )
    rescue_upper = max(
        0.02,
        float(np.ceil((rescue_values.max() + 0.01) / 0.02) * 0.02),
    )
    for endpoint in ENDPOINTS:
        endpoint_specificity = specificity.loc[
            specificity["held_out_label"].eq(endpoint)
        ].sort_values("seed")
        if len(endpoint_specificity) != 5:
            raise HIHAFigure2Error(
                f"Panel C rescue plot lacks five donor splits for {endpoint}"
            )

        for position, (quantity, _, color_key) in zip(
            endpoint_positions[endpoint],
            rescue_rows,
            strict=True,
        ):
            values = endpoint_specificity[quantity].to_numpy(dtype=float)
            rescue_axis.scatter(
                values,
                position + seed_offsets,
                s=22,
                color=colors[color_key],
                alpha=0.45,
                linewidth=0,
                zorder=3,
            )
            rescue_axis.errorbar(
                float(values.mean()),
                position,
                xerr=float(values.std(ddof=1)),
                fmt="o",
                markersize=6,
                markerfacecolor=colors[color_key],
                markeredgecolor=INK,
                markeredgewidth=0.7,
                ecolor=colors[color_key],
                elinewidth=1.1,
                capsize=3,
                capthick=1.0,
                zorder=4,
            )
        rescue_axis.text(
            -0.32,
            endpoint_centers[endpoint],
            endpoint,
            transform=rescue_axis.get_yaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=INK,
        )

        probabilities = endpoint_specificity[
            "median_restored_state_probability"
        ].to_numpy(dtype=float)
        probability_axis.scatter(
            probabilities,
            endpoint_centers[endpoint] + seed_offsets,
            s=22,
            color=colors["destination"],
            alpha=0.45,
            linewidth=0,
            zorder=3,
        )
        probability_axis.errorbar(
            float(probabilities.mean()),
            endpoint_centers[endpoint],
            xerr=float(probabilities.std(ddof=1)),
            fmt="o",
            markersize=6,
            markerfacecolor=colors["destination"],
            markeredgecolor=INK,
            markeredgewidth=0.7,
            ecolor=colors["destination"],
            elinewidth=1.1,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )

    row_positions = np.concatenate(
        [endpoint_positions[endpoint] for endpoint in ENDPOINTS]
    )
    row_labels = [label for _ in ENDPOINTS for _, label, _ in rescue_rows]
    rescue_axis.axvline(0, color=INK, linewidth=0.8, zorder=2)
    rescue_axis.axhline(2.6, color="#B8B8B8", linewidth=0.7, zorder=1)
    for divider in (3.9, 0.3):
        rescue_axis.axhline(
            divider,
            color="#D6D6D6",
            linewidth=0.55,
            linestyle="--",
            zorder=1,
        )
    rescue_axis.set(
        xlim=(rescue_lower, rescue_upper),
        ylim=(-0.7, 5.9),
        yticks=row_positions,
        yticklabels=row_labels,
    )
    rescue_axis.tick_params(axis="both", labelsize=6.5)
    rescue_axis.grid(axis="x", color="#E5E5E5", linewidth=0.5)
    rescue_axis.spines[["top", "right"]].set_visible(False)
    rescue_axis.set_xlabel(
        PANEL_C_XLABEL,
        fontsize=7,
        labelpad=7,
    )

    probability_axis.axhline(2.6, color="#B8B8B8", linewidth=0.7, zorder=1)
    probability_axis.set(
        xlim=(-0.03, 1.05),
        ylim=(-0.7, 5.9),
        xticks=(0, 0.5, 1),
    )
    probability_axis.tick_params(axis="x", labelsize=6.5)
    probability_axis.tick_params(axis="y", left=False, labelleft=False)
    probability_axis.grid(axis="x", color="#E5E5E5", linewidth=0.5)
    probability_axis.spines[["top", "right", "left"]].set_visible(False)
    probability_axis.set_title(
        "Restored-state\nconditional probability",
        fontsize=7,
        pad=5,
    )

    fig.text(0.012, 0.975, "C", ha="left", va="top", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.205, right=0.985, top=0.88, bottom=0.18)
    return fig


def _validate_panel_c_rescue_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(PANEL_C_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(PANEL_C_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        if image.width < expected_width - 4 or image.height < expected_height - 4:
            raise HIHAFigure2Error(
                f"Panel C raster is below its size target: {path} has {image.size}"
            )
        aspect = image.width / image.height
        if not 2.10 <= aspect <= 2.25:
            raise HIHAFigure2Error(f"Unexpected Panel C aspect ratio: {path} has {image.size}")


def _panel_e_spatial_run_index(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"run_id", "held_out_label", "seed", "method", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(
            f"Panel E detection table is missing columns {missing}: {path}"
        )
    method_keys = {(method, score) for method, score, _ in PANEL_E_METHODS}
    observed_method_keys = pd.MultiIndex.from_frame(
        frame[["method", "score"]].astype(str)
    )
    selected = frame.loc[
        frame["held_out_label"].isin(ENDPOINTS)
        & frame["seed"].eq(REPRESENTATIVE_SEED)
        & observed_method_keys.isin(method_keys)
    ].copy()
    expected = {
        (endpoint, method, score)
        for endpoint in ENDPOINTS
        for method, score, _ in PANEL_E_METHODS
    }
    observed = set(
        zip(
            selected["held_out_label"],
            selected["method"],
            selected["score"],
            strict=False,
        )
    )
    if observed != expected:
        raise HIHAFigure2Error(
            "Panel E method keys differ from the two-endpoint seed-1 contract; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    keys = ["held_out_label", "seed", "method", "score"]
    if selected.duplicated(keys).any():
        raise HIHAFigure2Error(f"Panel E detection table has duplicate keys: {keys}")

    rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        endpoint_rows = selected.loc[selected["held_out_label"].eq(endpoint)]
        selected_ids = set(
            endpoint_rows.loc[
                endpoint_rows["method"].isin(
                    {
                        "coreot_full",
                        "prior_only",
                        "seurat_anchor",
                        "scmap_cluster",
                        "chetah",
                    }
                ),
                "run_id",
            ].astype(str)
        )
        if len(selected_ids) != 1:
            raise HIHAFigure2Error(
                f"Panel E selected methods do not share one run for {endpoint}: "
                f"{sorted(selected_ids)}"
            )
        uniform_ids = set(
            endpoint_rows.loc[
                endpoint_rows["method"].eq("uniform_uot"), "run_id"
            ].astype(str)
        )
        if len(uniform_ids) != 1:
            raise HIHAFigure2Error(
                f"Panel E requires one uniform-UOT run for {endpoint}: "
                f"{sorted(uniform_ids)}"
            )
        rows.append(
            {
                "held_out_label": endpoint,
                "seed": REPRESENTATIVE_SEED,
                "selected_run_id": next(iter(selected_ids)),
                "uniform_run_id": next(iter(uniform_ids)),
            }
        )
    return pd.DataFrame(rows)


def _read_query_embedding(run_root: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    embedding_root = run_root / "embeddings/incomplete_reference/hiha_harmony30"
    cells_path = embedding_root / "embedding_cells.csv"
    matrix_path = embedding_root / "embedding.npy"
    cells = pd.read_csv(cells_path)
    required = {"cell_id", "domain"}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise HIHAFigure2Error(
            f"Figure 2 embedding cells are missing columns {missing}: {cells_path}"
        )
    cells["cell_id"] = cells["cell_id"].astype(str)
    if cells["cell_id"].duplicated().any():
        raise HIHAFigure2Error(
            f"Figure 2 embedding cells contain duplicate IDs: {cells_path}"
        )
    matrix = np.load(matrix_path)
    if matrix.shape != (len(cells), 30):
        raise HIHAFigure2Error(
            f"Figure 2 requires a ({len(cells)}, 30) provider embedding; got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise HIHAFigure2Error(
            f"Figure 2 provider embedding contains nonfinite values: {matrix_path}"
        )
    query_mask = cells["domain"].eq("query").to_numpy()
    if int(query_mask.sum()) <= UMAP_N_NEIGHBORS:
        raise HIHAFigure2Error(
            f"Figure 2 query-only UMAP needs more than {UMAP_N_NEIGHBORS} query cells"
        )
    query_ids = cells.loc[query_mask, "cell_id"].to_numpy(dtype=str)
    return query_ids, matrix[query_mask].copy(), [cells_path, matrix_path]


def _compute_query_umap(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != 30:
        raise HIHAFigure2Error(
            f"Figure 2 UMAP input must have 30 columns; got {matrix.shape}"
        )
    if len(matrix) <= UMAP_N_NEIGHBORS or not np.isfinite(matrix).all():
        raise HIHAFigure2Error(
            "Figure 2 UMAP input is too small or contains nonfinite values"
        )
    import anndata as ad
    import scanpy as sc

    query = ad.AnnData(X=matrix.copy())
    sc.pp.neighbors(
        query,
        n_neighbors=UMAP_N_NEIGHBORS,
        use_rep="X",
        metric="euclidean",
        random_state=UMAP_RANDOM_STATE,
    )
    sc.tl.umap(
        query,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    coordinates = np.asarray(query.obsm["X_umap"], dtype=float)
    if coordinates.shape != (len(matrix), 2) or not np.isfinite(coordinates).all():
        raise HIHAFigure2Error(
            f"Figure 2 UMAP returned invalid coordinates: {coordinates.shape}"
        )
    return coordinates


def _read_panel_e_method_scores(
    *,
    path: Path,
    method: str,
    score: str,
    expected_ids: set[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"cell_id", "method", score}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(
            f"Panel E score table is missing columns {missing}: {path}"
        )
    selected = frame.loc[frame["method"].astype(str).eq(method), ["cell_id", score]].copy()
    selected["cell_id"] = selected["cell_id"].astype(str)
    if selected["cell_id"].duplicated().any():
        raise HIHAFigure2Error(
            f"Panel E score rows contain duplicate cell IDs for {method}: {path}"
        )
    observed_ids = set(selected["cell_id"])
    if observed_ids != expected_ids:
        raise HIHAFigure2Error(
            f"Panel E {method} score IDs differ from query truth: {path}"
        )
    selected[score] = pd.to_numeric(selected[score], errors="coerce")
    if not np.isfinite(selected[score].to_numpy(dtype=float)).all():
        raise HIHAFigure2Error(
            f"Panel E {method} scores contain nonfinite values: {path}"
        )
    return selected.rename(columns={score: "oriented_score"})


def _top_n_cell_ids(
    frame: pd.DataFrame,
    *,
    n: int,
) -> set[str]:
    eligible = frame.loc[frame["is_within_cdc2"]].sort_values(
        ["oriented_score", "cell_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    if n <= 0 or n > len(eligible):
        raise HIHAFigure2Error(
            f"Panel E top-N selection is invalid: n={n}, eligible={len(eligible)}"
        )
    return set(eligible.head(n)["cell_id"].astype(str))


def _collect_panel_e_umap_data(
    *,
    detection_path: Path,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    run_index = _panel_e_spatial_run_index(detection_path)
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    source_paths: list[str] = [str(detection_path)]

    for run in run_index.itertuples(index=False):
        selected_root = runs_root / str(run.selected_run_id)
        uniform_root = runs_root / str(run.uniform_run_id)
        truth = _read_query_truth(selected_root, "incomplete_reference")
        truth_ids = set(truth["cell_id"])
        query_ids, query_matrix, embedding_paths = _read_query_embedding(selected_root)
        if set(query_ids) != truth_ids:
            raise HIHAFigure2Error(
                f"Panel E query embedding IDs differ from evaluation truth: {selected_root}"
            )
        coordinates = _compute_query_umap(query_matrix)
        coordinate_frame = pd.DataFrame(
            {
                "cell_id": query_ids,
                "umap_1": coordinates[:, 0],
                "umap_2": coordinates[:, 1],
            }
        )
        full_targets = _read_target_labels(
            selected_root, "full_reference_control"
        )
        true_mapping = (
            full_targets[["target_label", "broad_label"]]
            .drop_duplicates()
            .set_index("target_label")["broad_label"]
        )
        truth["true_broad_label"] = truth["true_label"].map(true_mapping)
        if truth["true_broad_label"].isna().any():
            raise HIHAFigure2Error(
                f"Panel E query truth lacks full-reference broad mappings: {selected_root}"
            )
        base = truth.merge(coordinate_frame, on="cell_id", validate="one_to_one")
        base.insert(0, "held_out_label", str(run.held_out_label))
        base["is_held_out_truth"] = base["is_absent_state"].astype(bool)
        base["is_within_cdc2"] = base["true_broad_label"].eq("cDC2")
        if not base.loc[base["is_held_out_truth"], "is_within_cdc2"].all():
            raise HIHAFigure2Error(
                f"Panel E held-out truth is not contained in cDC2: {selected_root}"
            )
        n_held_out = int(base["is_held_out_truth"].sum())
        n_eligible = int(base["is_within_cdc2"].sum())

        truth_frame = base.copy()
        truth_frame["map_id"] = "truth"
        truth_frame["map_display"] = "Held-out truth"
        truth_frame["score_run_id"] = str(run.selected_run_id)
        truth_frame["score_name"] = "evaluation_truth"
        truth_frame["oriented_score"] = np.nan
        truth_frame["is_selected"] = truth_frame["is_held_out_truth"]
        frames.append(truth_frame)
        summary_rows.append(
            {
                "held_out_label": str(run.held_out_label),
                "seed": int(run.seed),
                "map_id": "truth",
                "map_display": "Held-out truth",
                "score_run_id": str(run.selected_run_id),
                "score_name": "evaluation_truth",
                "n_query": int(len(base)),
                "n_eligible_cdc2": n_eligible,
                "n_selected": n_held_out,
                "n_held_out_overlap": n_held_out,
                "held_out_overlap_fraction": 1.0,
            }
        )

        internal_path = (
            selected_root
            / "scoring/incomplete_reference"
            / CANDIDATE_SET
            / "cell_scores.parquet"
        )
        uniform_path = (
            uniform_root
            / "scoring/incomplete_reference"
            / CANDIDATE_SET
            / "cell_scores.parquet"
        )
        external_path = (
            selected_root
            / "scoring/incomplete_reference/external_reference_mapping/"
            "cell_scores.parquet"
        )
        path_by_method = {
            "coreot_full": internal_path,
            "prior_only": internal_path,
            "uniform_uot": uniform_path,
            "seurat_anchor": external_path,
            "scmap_cluster": external_path,
            "chetah": external_path,
        }
        for method, score, display in PANEL_E_METHODS:
            score_frame = _read_panel_e_method_scores(
                path=path_by_method[method],
                method=method,
                score=score,
                expected_ids=truth_ids,
            )
            method_frame = base.merge(
                score_frame,
                on="cell_id",
                validate="one_to_one",
            )
            selected_ids = _top_n_cell_ids(method_frame, n=n_held_out)
            method_frame["map_id"] = method
            method_frame["map_display"] = display
            method_frame["score_run_id"] = (
                str(run.uniform_run_id)
                if method == "uniform_uot"
                else str(run.selected_run_id)
            )
            method_frame["score_name"] = score
            method_frame["is_selected"] = method_frame["cell_id"].isin(selected_ids)
            if int(method_frame["is_selected"].sum()) != n_held_out:
                raise HIHAFigure2Error(
                    f"Panel E {method} did not select exactly {n_held_out} cells"
                )
            if not method_frame.loc[
                method_frame["is_selected"], "is_within_cdc2"
            ].all():
                raise HIHAFigure2Error(
                    f"Panel E {method} selected a cell outside the cDC2 cohort"
                )
            frames.append(method_frame)
            overlap = int(
                (
                    method_frame["is_selected"]
                    & method_frame["is_held_out_truth"]
                ).sum()
            )
            summary_rows.append(
                {
                    "held_out_label": str(run.held_out_label),
                    "seed": int(run.seed),
                    "map_id": method,
                    "map_display": display,
                    "score_run_id": (
                        str(run.uniform_run_id)
                        if method == "uniform_uot"
                        else str(run.selected_run_id)
                    ),
                    "score_name": score,
                    "n_query": int(len(base)),
                    "n_eligible_cdc2": n_eligible,
                    "n_selected": n_held_out,
                    "n_held_out_overlap": overlap,
                    "held_out_overlap_fraction": overlap / n_held_out,
                }
            )
        source_paths.extend(
            [
                *(str(path) for path in embedding_paths),
                str(
                    selected_root
                    / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
                ),
                str(
                    selected_root
                    / "benchmark/full_reference_control/model_visible/target_labels.csv"
                ),
                str(internal_path),
                str(uniform_path),
                str(external_path),
            ]
        )

    cells = pd.concat(frames, ignore_index=True)
    cells.insert(
        0,
        "run_id",
        cells["held_out_label"].map(
            run_index.set_index("held_out_label")["selected_run_id"]
        ),
    )
    cells.insert(2, "seed", REPRESENTATIVE_SEED)
    cells = cells[
        [
            "run_id",
            "held_out_label",
            "seed",
            "map_id",
            "map_display",
            "score_run_id",
            "score_name",
            "cell_id",
            "true_label",
            "true_broad_label",
            "is_held_out_truth",
            "is_within_cdc2",
            "is_selected",
            "umap_1",
            "umap_2",
            "oriented_score",
        ]
    ]
    summary = pd.DataFrame(summary_rows).sort_values(
        ["held_out_label", "map_id"], ignore_index=True
    )
    expected_summary = {
        (endpoint, map_id)
        for endpoint in ENDPOINTS
        for map_id, _, _ in PANEL_E_MAPS
    }
    observed_summary = set(
        summary[["held_out_label", "map_id"]].itertuples(index=False, name=None)
    )
    if observed_summary != expected_summary:
        raise HIHAFigure2Error(
            "Panel E summary keys differ from the endpoint-map contract"
        )
    if cells.duplicated(["held_out_label", "map_id", "cell_id"]).any():
        raise HIHAFigure2Error("Panel E contains duplicate endpoint-map-cell rows")
    coordinates = cells[["umap_1", "umap_2"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise HIHAFigure2Error("Panel E UMAP coordinates contain nonfinite values")
    method_scores = cells.loc[cells["map_id"].ne("truth"), "oriented_score"].to_numpy(
        dtype=float
    )
    if not np.isfinite(method_scores).all():
        raise HIHAFigure2Error("Panel E method scores contain nonfinite values")
    return cells, summary, list(dict.fromkeys(source_paths))


def _draw_panel_e_umap(cells: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        7,
        figsize=PANEL_E_SIZE_INCHES,
        sharex=False,
        sharey=False,
    )
    fig.patch.set_facecolor("white")

    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_cells = cells.loc[cells["held_out_label"].eq(endpoint)].copy()
        coordinate_rows = endpoint_cells.loc[endpoint_cells["map_id"].eq("truth")]
        if coordinate_rows.empty:
            raise HIHAFigure2Error(f"Panel E has no truth rows for {endpoint}")
        x_span = max(
            float(coordinate_rows["umap_1"].max() - coordinate_rows["umap_1"].min()),
            1.0e-6,
        )
        y_span = max(
            float(coordinate_rows["umap_2"].max() - coordinate_rows["umap_2"].min()),
            1.0e-6,
        )
        xlim = (
            float(coordinate_rows["umap_1"].min() - 0.025 * x_span),
            float(coordinate_rows["umap_1"].max() + 0.025 * x_span),
        )
        ylim = (
            float(coordinate_rows["umap_2"].min() - 0.025 * y_span),
            float(coordinate_rows["umap_2"].max() + 0.025 * y_span),
        )
        for column, (map_id, _, display) in enumerate(PANEL_E_MAPS):
            axis = axes[row, column]
            frame = endpoint_cells.loc[endpoint_cells["map_id"].eq(map_id)]
            if len(frame) != len(coordinate_rows):
                raise HIHAFigure2Error(
                    f"Panel E map {map_id} has incomplete cell coverage for {endpoint}"
                )
            outside = frame.loc[~frame["is_within_cdc2"]]
            eligible = frame.loc[frame["is_within_cdc2"] & ~frame["is_selected"]]
            selected = frame.loc[frame["is_selected"]]
            axis.scatter(
                outside["umap_1"],
                outside["umap_2"],
                s=2.2,
                color=PANEL_E_CONTEXT_GRAY,
                alpha=0.72,
                linewidth=0,
                rasterized=True,
                zorder=1,
            )
            axis.scatter(
                eligible["umap_1"],
                eligible["umap_2"],
                s=3.0,
                color=PANEL_E_ELIGIBLE_GRAY,
                alpha=0.82,
                linewidth=0,
                rasterized=True,
                zorder=2,
            )
            highlight_color = (
                PANEL_E_TRUTH_COLOR if map_id == "truth" else METHOD_COLORS[map_id]
            )
            axis.scatter(
                selected["umap_1"],
                selected["umap_2"],
                s=5.0,
                color=highlight_color,
                alpha=0.95,
                linewidth=0,
                rasterized=True,
                zorder=3,
            )
            axis.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(display, fontsize=7.0, pad=5)
        fig.text(
            0.028,
            PANEL_F_ENDPOINT_LABEL_Y[row],
            endpoint,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=INK,
        )

    fig.text(0.012, 0.975, "D", ha="left", va="top", fontsize=11, fontweight="bold")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_E_TRUTH_COLOR,
            markeredgecolor="none",
            markersize=4.8,
            label="Held-out truth",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=INK,
            markeredgecolor="none",
            markersize=4.8,
            label="Method color: top-$N_\\ell$ weak-support rank within cDC2",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_E_ELIGIBLE_GRAY,
            markeredgecolor="none",
            markersize=4.5,
            label="Eligible cDC2, not selected",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_E_CONTEXT_GRAY,
            markeredgecolor="none",
            markersize=4.2,
            label="Outside cDC2 evaluation cohort",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 0.025),
        fontsize=6.2,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.88,
        bottom=0.13,
        wspace=0.04,
        hspace=0.08,
    )
    return fig


def _validate_panel_e_umap_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(PANEL_E_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(PANEL_E_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        if image.width < expected_width - 4 or image.height < expected_height - 4:
            raise HIHAFigure2Error(
                f"Panel E raster is below its size target: {path} has {image.size}"
            )
        aspect = image.width / image.height
        if not 1.55 <= aspect <= 1.75:
            raise HIHAFigure2Error(
                f"Unexpected Panel E aspect ratio: {path} has {image.size}"
            )


def _select_panel_d_label_transfer(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "run_id",
        "held_out_label",
        "seed",
        "condition_id",
        "candidate_set",
        "method",
        "score",
        "n_shared",
        "n_forced_labeled",
        "forced_macro_f1",
        "forced_accuracy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(f"Panel D source is missing columns {missing}: {path}")

    method_keys = {(method, score) for method, score, _ in PANEL_D_METHODS}
    observed_method_keys = pd.MultiIndex.from_frame(
        frame[["method", "score"]].astype(str)
    )
    selected = frame.loc[
        frame["held_out_label"].isin(ENDPOINTS)
        & frame["condition_id"].eq("incomplete_reference")
        & observed_method_keys.isin(method_keys)
    ].copy()
    expected = {
        (endpoint, seed, method, score)
        for endpoint in ENDPOINTS
        for seed in range(1, 6)
        for method, score, _ in PANEL_D_METHODS
    }
    observed = set(
        zip(
            selected["held_out_label"],
            selected["seed"],
            selected["method"],
            selected["score"],
            strict=False,
        )
    )
    if observed != expected:
        missing_keys = sorted(expected - observed)
        extra_keys = sorted(observed - expected)
        raise HIHAFigure2Error(
            "Panel D keys differ from the two-endpoint, five-seed contract; "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    keys = ["held_out_label", "seed", "method", "score"]
    if selected.duplicated(keys).any():
        raise HIHAFigure2Error(f"Panel D source has duplicate keys: {keys}")
    expected_candidate_sets = selected["method"].map(PANEL_D_CANDIDATE_SET_BY_METHOD)
    if expected_candidate_sets.isna().any() or not selected["candidate_set"].eq(
        expected_candidate_sets
    ).all():
        observed_candidate_sets = (
            selected.groupby("method")["candidate_set"].unique().apply(list).to_dict()
        )
        raise HIHAFigure2Error(
            "Panel D candidate-set identifiers disagree with the method-specific "
            f"contract: {observed_candidate_sets}"
        )

    for metric in ("forced_macro_f1", "forced_accuracy"):
        values = pd.to_numeric(selected[metric], errors="coerce")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise HIHAFigure2Error(
                f"Panel D {metric} must contain finite values in [0, 1]"
            )
        selected[metric] = values
    for column in ("n_shared", "n_forced_labeled"):
        values = pd.to_numeric(selected[column], errors="coerce")
        if not np.isfinite(values).all() or not values.gt(0).all():
            raise HIHAFigure2Error(f"Panel D {column} must contain positive counts")
        selected[column] = values.astype(int)
    if not selected["n_shared"].eq(selected["n_forced_labeled"]).all():
        raise HIHAFigure2Error(
            "Panel D forced-label evaluation does not label every represented-state cell"
        )

    lower, upper = PANEL_D_LIM
    for metric in ("forced_macro_f1", "forced_accuracy"):
        if not selected[metric].between(lower, upper).all():
            raise HIHAFigure2Error(
                f"Panel D {metric} observations fall outside "
                f"the prespecified axis {PANEL_D_LIM}"
            )
    display = {method: name for method, _, name in PANEL_D_METHODS}
    selected["method_display"] = selected["method"].map(display)
    selected["source_path"] = str(path)
    return selected[
        [
            "run_id",
            "held_out_label",
            "seed",
            "condition_id",
            "candidate_set",
            "method",
            "method_display",
            "score",
            "n_shared",
            "n_forced_labeled",
            "forced_macro_f1",
            "forced_accuracy",
            "source_path",
        ]
    ].sort_values(["held_out_label", "method_display", "seed"], ignore_index=True)


def _draw_panel_d_label_transfer(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=PANEL_D_SIZE_INCHES,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    method_displays = {
        method: display for method, _, display in PANEL_D_METHODS
    }
    positions = np.arange(len(PANEL_D_METHODS), dtype=float)
    bar_height = 0.22
    metric_offsets = (-0.16, 0.16)

    for facet_index, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS, strict=True)):
        endpoint_data = data.loc[data["held_out_label"].eq(endpoint)]
        for method_index, (method, _, _) in enumerate(PANEL_D_METHODS):
            method_data = endpoint_data.loc[
                endpoint_data["method"].eq(method)
            ].sort_values("seed")
            for metric_index, (metric, _, color) in enumerate(PANEL_D_METRICS):
                values = method_data[metric].to_numpy(dtype=float)
                mean = float(values.mean())
                sample_sd = float(values.std(ddof=1))
                if not (
                    PANEL_D_LIM[0]
                    <= mean - sample_sd
                    <= mean + sample_sd
                    <= PANEL_D_LIM[1]
                ):
                    raise HIHAFigure2Error(
                        f"Panel D mean ± sample SD falls outside {PANEL_D_LIM} "
                        f"for {endpoint}, {method}, {metric}"
                    )
                center = positions[method_index] + metric_offsets[metric_index]
                ax.barh(
                    center,
                    mean - PANEL_D_LIM[0],
                    height=bar_height * 0.88,
                    left=PANEL_D_LIM[0],
                    color=color,
                    alpha=0.48,
                    edgecolor=color,
                    linewidth=0.65,
                    zorder=2,
                )
                ax.errorbar(
                    mean,
                    center,
                    xerr=sample_sd,
                    fmt="none",
                    ecolor=INK,
                    elinewidth=0.75,
                    capsize=2.0,
                    capthick=0.75,
                    zorder=4,
                )

        ticks = np.arange(0.80, 1.001, 0.05)
        ax.set_xlim(PANEL_D_LIM)
        ax.set_ylim(len(PANEL_D_METHODS) - 0.5, -0.5)
        ax.set_title(endpoint, fontsize=8.5, pad=6)
        ax.set_xticks(ticks)
        ax.set_yticks(positions)
        if facet_index == 0:
            ax.set_yticklabels(
                [method_displays[method] for method, _, _ in PANEL_D_METHODS],
                fontsize=6.8,
            )
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.text(0.012, 0.965, "E", ha="left", va="top", fontsize=11, fontweight="bold")
    metric_handles = [
        Patch(
            facecolor=color,
            edgecolor=color,
            linewidth=0.65,
            alpha=0.55,
            label=display,
        )
        for _, display, color in PANEL_D_METRICS
    ]
    fig.legend(
        handles=metric_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
        fontsize=6.8,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig.text(
        0.56,
        0.075,
        PANEL_D_XLABEL,
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.82, bottom=0.18, wspace=0.12)
    return fig


def _validate_panel_d_label_transfer_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(PANEL_D_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(PANEL_D_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        if image.width < expected_width - 4 or image.height < expected_height - 4:
            raise HIHAFigure2Error(
                f"Panel D raster is below its size target: {path} has {image.size}"
            )
        aspect = image.width / image.height
        if not 2.35 <= aspect <= 2.60:
            raise HIHAFigure2Error(f"Unexpected Panel D aspect ratio: {path} has {image.size}")


def _read_panel_f_assignments(
    *,
    path: Path,
    method: str,
    expected_ids: set[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"cell_id", "method", "forced_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HIHAFigure2Error(
            f"Panel F assignment table is missing columns {missing}: {path}"
        )
    selected = frame.loc[
        frame["method"].astype(str).eq(method),
        ["cell_id", "forced_label"],
    ].copy()
    selected["cell_id"] = selected["cell_id"].astype(str)
    if selected["cell_id"].duplicated().any():
        raise HIHAFigure2Error(
            f"Panel F contains duplicate assignment IDs for {method}: {path}"
        )
    if set(selected["cell_id"]) != expected_ids:
        raise HIHAFigure2Error(
            f"Panel F {method} assignment IDs differ from query truth: {path}"
        )
    selected["forced_label"] = selected["forced_label"].fillna("").astype(str)
    return selected


def _collect_panel_f_label_assignments(
    *,
    label_transfer_path: Path,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    panel_d = _select_panel_d_label_transfer(label_transfer_path)
    seed_rows = panel_d.loc[panel_d["seed"].eq(REPRESENTATIVE_SEED)].copy()
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    source_paths: list[str] = [str(label_transfer_path)]

    for endpoint in ENDPOINTS:
        endpoint_rows = seed_rows.loc[seed_rows["held_out_label"].eq(endpoint)]
        selected_ids = set(
            endpoint_rows.loc[
                endpoint_rows["method"].ne("uniform_uot"), "run_id"
            ].astype(str)
        )
        if len(selected_ids) != 1:
            raise HIHAFigure2Error(
                f"Panel F selected methods do not share one run for {endpoint}: "
                f"{sorted(selected_ids)}"
            )
        uniform_ids = set(
            endpoint_rows.loc[
                endpoint_rows["method"].eq("uniform_uot"), "run_id"
            ].astype(str)
        )
        if len(uniform_ids) != 1:
            raise HIHAFigure2Error(
                f"Panel F requires one uniform-UOT run for {endpoint}: "
                f"{sorted(uniform_ids)}"
            )
        selected_run_id = next(iter(selected_ids))
        uniform_run_id = next(iter(uniform_ids))
        selected_root = runs_root / selected_run_id
        uniform_root = runs_root / uniform_run_id

        truth = _read_query_truth(selected_root, "incomplete_reference")
        truth_ids = set(truth["cell_id"])
        query_ids, query_matrix, embedding_paths = _read_query_embedding(selected_root)
        if set(query_ids) != truth_ids:
            raise HIHAFigure2Error(
                f"Panel F query embedding IDs differ from evaluation truth: {selected_root}"
            )
        coordinates = _compute_query_umap(query_matrix)
        coordinate_frame = pd.DataFrame(
            {
                "cell_id": query_ids,
                "umap_1": coordinates[:, 0],
                "umap_2": coordinates[:, 1],
            }
        )
        reference_path = (
            selected_root
            / "benchmark/incomplete_reference/model_visible/target_labels.csv"
        )
        reference_labels = set(
            _read_target_labels(
                selected_root, "incomplete_reference"
            )["target_label"].astype(str)
        )
        base = truth.merge(coordinate_frame, on="cell_id", validate="one_to_one")
        base.insert(0, "held_out_label", endpoint)
        base["is_represented_state"] = base["is_shared_state"].astype(bool)
        base["is_held_out_state"] = base["is_absent_state"].astype(bool)
        if not (
            base["is_represented_state"] ^ base["is_held_out_state"]
        ).all():
            raise HIHAFigure2Error(
                f"Panel F truth does not partition represented and held-out cells: "
                f"{selected_root}"
            )
        represented_truth = set(
            base.loc[base["is_represented_state"], "true_label"].astype(str)
        )
        if represented_truth != reference_labels:
            raise HIHAFigure2Error(
                f"Panel F represented truth labels differ from reference labels for "
                f"{endpoint}: truth={sorted(represented_truth)}, "
                f"reference={sorted(reference_labels)}"
            )

        truth_frame = base.copy()
        truth_frame["map_id"] = "truth"
        truth_frame["map_display"] = "Ground truth"
        truth_frame["method"] = "truth"
        truth_frame["method_run_id"] = selected_run_id
        truth_frame["forced_label"] = ""
        truth_frame["displayed_assignment"] = np.where(
            truth_frame["is_represented_state"],
            truth_frame["true_label"],
            "Held-out state (not evaluated)",
        )
        frames.append(truth_frame)

        internal_path = (
            selected_root
            / "scoring/incomplete_reference"
            / CANDIDATE_SET
            / "cell_scores.parquet"
        )
        uniform_path = (
            uniform_root
            / "scoring/incomplete_reference"
            / CANDIDATE_SET
            / "cell_scores.parquet"
        )
        external_path = (
            selected_root
            / "scoring/incomplete_reference/external_reference_mapping/"
            "cell_scores.parquet"
        )
        path_by_method = {
            "coreot_full": internal_path,
            "uniform_uot": uniform_path,
            "seurat_anchor": external_path,
            "scmap_cluster": external_path,
            "chetah": external_path,
        }
        display_by_method = {
            method: display for method, _, display in PANEL_D_METHODS
        }
        for method, _, _ in PANEL_D_METHODS:
            assignments = _read_panel_f_assignments(
                path=path_by_method[method],
                method=method,
                expected_ids=truth_ids,
            )
            method_frame = base.merge(
                assignments,
                on="cell_id",
                validate="one_to_one",
            )
            represented = method_frame["is_represented_state"]
            invalid_labels = sorted(
                set(method_frame.loc[represented, "forced_label"]) - reference_labels
            )
            if invalid_labels:
                raise HIHAFigure2Error(
                    f"Panel F {method} assigns labels outside the represented "
                    f"reference for {endpoint}: {invalid_labels}"
                )
            method_frame["map_id"] = method
            method_frame["map_display"] = display_by_method[method]
            method_frame["method"] = method
            method_frame["method_run_id"] = (
                uniform_run_id if method == "uniform_uot" else selected_run_id
            )
            method_frame["displayed_assignment"] = np.where(
                represented,
                method_frame["forced_label"],
                "Held-out state (not evaluated)",
            )
            frames.append(method_frame)

            true_labels = method_frame.loc[represented, "true_label"].astype(str)
            predicted_labels = method_frame.loc[
                represented, "forced_label"
            ].astype(str)
            forced_accuracy = float(accuracy_score(true_labels, predicted_labels))
            forced_macro_f1 = float(
                f1_score(
                    true_labels,
                    predicted_labels,
                    labels=sorted(reference_labels),
                    average="macro",
                    zero_division=0,
                )
            )
            panel_d_row = endpoint_rows.loc[endpoint_rows["method"].eq(method)]
            if len(panel_d_row) != 1:
                raise HIHAFigure2Error(
                    f"Panel F lacks one Panel D seed-1 row for {endpoint}, {method}"
                )
            expected_accuracy = float(panel_d_row.iloc[0]["forced_accuracy"])
            expected_macro_f1 = float(panel_d_row.iloc[0]["forced_macro_f1"])
            if not np.isclose(forced_accuracy, expected_accuracy, atol=1.0e-12):
                raise HIHAFigure2Error(
                    f"Panel F {method} forced accuracy does not reproduce Panel D "
                    f"for {endpoint}: {forced_accuracy} != {expected_accuracy}"
                )
            if not np.isclose(forced_macro_f1, expected_macro_f1, atol=1.0e-12):
                raise HIHAFigure2Error(
                    f"Panel F {method} forced macro-F1 does not reproduce Panel D "
                    f"for {endpoint}: {forced_macro_f1} != {expected_macro_f1}"
                )
            summary_rows.append(
                {
                    "held_out_label": endpoint,
                    "seed": REPRESENTATIVE_SEED,
                    "method": method,
                    "method_display": display_by_method[method],
                    "run_id": (
                        uniform_run_id if method == "uniform_uot" else selected_run_id
                    ),
                    "n_represented": int(represented.sum()),
                    "forced_accuracy": forced_accuracy,
                    "forced_macro_f1": forced_macro_f1,
                    "panel_d_forced_accuracy": expected_accuracy,
                    "panel_d_forced_macro_f1": expected_macro_f1,
                }
            )

        source_paths.extend(
            [
                *(str(path) for path in embedding_paths),
                str(
                    selected_root
                    / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
                ),
                str(reference_path),
                str(internal_path),
                str(uniform_path),
                str(external_path),
            ]
        )

    cells = pd.concat(frames, ignore_index=True)
    cells.insert(2, "seed", REPRESENTATIVE_SEED)
    cells = cells[
        [
            "held_out_label",
            "seed",
            "cell_id",
            "umap_1",
            "umap_2",
            "true_label",
            "is_represented_state",
            "is_held_out_state",
            "map_id",
            "map_display",
            "method",
            "method_run_id",
            "forced_label",
            "displayed_assignment",
        ]
    ].sort_values(["held_out_label", "map_id", "cell_id"], ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["held_out_label", "method"], ignore_index=True
    )
    expected_maps = {
        (endpoint, map_id)
        for endpoint in ENDPOINTS
        for map_id, _ in PANEL_F_MAPS
    }
    observed_maps = set(
        cells[["held_out_label", "map_id"]].drop_duplicates().itertuples(
            index=False, name=None
        )
    )
    if observed_maps != expected_maps:
        raise HIHAFigure2Error("Panel F maps differ from the endpoint-map contract")
    if cells.duplicated(["held_out_label", "map_id", "cell_id"]).any():
        raise HIHAFigure2Error("Panel F contains duplicate endpoint-map-cell rows")
    if not np.isfinite(cells[["umap_1", "umap_2"]].to_numpy(dtype=float)).all():
        raise HIHAFigure2Error("Panel F UMAP coordinates contain nonfinite values")
    return cells, summary, list(dict.fromkeys(source_paths))


def _draw_panel_f_label_assignments(cells: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 6, figsize=PANEL_F_SIZE_INCHES)
    fig.patch.set_facecolor("white")
    for row, endpoint in enumerate(ENDPOINTS):
        endpoint_cells = cells.loc[cells["held_out_label"].eq(endpoint)]
        coordinate_rows = endpoint_cells.loc[endpoint_cells["map_id"].eq("truth")]
        x_span = max(
            float(coordinate_rows["umap_1"].max() - coordinate_rows["umap_1"].min()),
            1.0e-6,
        )
        y_span = max(
            float(coordinate_rows["umap_2"].max() - coordinate_rows["umap_2"].min()),
            1.0e-6,
        )
        xlim = (
            float(coordinate_rows["umap_1"].min() - 0.025 * x_span),
            float(coordinate_rows["umap_1"].max() + 0.025 * x_span),
        )
        ylim = (
            float(coordinate_rows["umap_2"].min() - 0.025 * y_span),
            float(coordinate_rows["umap_2"].max() + 0.025 * y_span),
        )
        for column, (map_id, display) in enumerate(PANEL_F_MAPS):
            axis = axes[row, column]
            frame = endpoint_cells.loc[endpoint_cells["map_id"].eq(map_id)]
            represented = frame.loc[frame["is_represented_state"]].sort_values(
                "cell_id"
            )
            held_out = frame.loc[frame["is_held_out_state"]].sort_values("cell_id")
            colors = represented["displayed_assignment"].map(PANEL_F_LABEL_COLORS)
            if colors.isna().any():
                unexpected = sorted(
                    represented.loc[colors.isna(), "displayed_assignment"].unique()
                )
                raise HIHAFigure2Error(
                    f"Panel F has unmapped biological labels: {unexpected}"
                )
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
                axis.set_title(display, fontsize=7.0, pad=5)
        fig.text(
            0.028,
            0.61 if row == 0 else 0.33,
            endpoint,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=INK,
        )

    fig.text(0.012, 0.975, "F", ha="left", va="top", fontsize=11, fontweight="bold")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=4.8,
            label=label,
        )
        for label, color in PANEL_F_LABEL_COLORS.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_F_HELD_OUT_COLOR,
            markeredgecolor="none",
            markersize=4.8,
            label="Held-out state (not evaluated)",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=PANEL_F_LEGEND_NCOLS,
        frameon=False,
        bbox_to_anchor=(0.51, 0.025),
        fontsize=6.2,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.88,
        bottom=0.19,
        wspace=0.04,
        hspace=0.08,
    )
    return fig


def _validate_panel_f_label_assignment_raster(path: Path) -> None:
    with Image.open(path) as image:
        expected_width = round(PANEL_F_SIZE_INCHES[0] * PANEL_RASTER_DPI)
        expected_height = round(PANEL_F_SIZE_INCHES[1] * PANEL_RASTER_DPI)
        if image.width < expected_width - 4 or image.height < expected_height - 4:
            raise HIHAFigure2Error(
                f"Panel F raster is below its size target: {path} has {image.size}"
            )
        aspect = image.width / image.height
        if not 2.10 <= aspect <= 2.30:
            raise HIHAFigure2Error(
                f"Unexpected Panel F aspect ratio: {path} has {image.size}"
            )


def generate_panel_a(
    *,
    panel_root: Path = PANEL_ROOT,
    result_root: Path = RESULT_ROOT,
    overview_path: Path = OVERVIEW_PATH,
) -> dict[str, Path]:
    design = _read_design_source(overview_path)
    panel_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    outputs = {
        suffix: panel_root / f"panel_a.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    source_path = source_root / "panel_a_design.csv"
    manifest_path = result_root / "panel_a_manifest.yaml"

    fig, ax = plt.subplots(figsize=PANEL_A_SIZE_INCHES)
    fig.patch.set_facecolor("white")
    _draw_panel_a(ax, n_subjects=design["Number of subjects"])
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.98)
    for suffix, path in outputs.items():
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "field": "number_of_subjects",
                "value": design["Number of subjects"],
                "source_path": str(overview_path),
            },
            {
                "field": "donor_split",
                "value": design["Donor split"],
                "source_path": str(overview_path),
            },
            {
                "field": "displayed_endpoints",
                "value": "HLA-DRhi cDC2 | ISG+ cDC2",
                "source_path": str(overview_path),
            },
            {
                "field": "primary_evaluation",
                "value": "within-cDC2 held-out-state ranking",
                "source_path": (
                    "docs/adr/0019-use-within-cdc2-detection-as-the-main-hiha-ranking.md"
                ),
            },
        ]
    ).to_csv(source_path, index=False)

    artifacts = {suffix: str(path) for suffix, path in outputs.items()}
    artifacts["source_data"] = str(source_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel": "A",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py"
                ),
                "artifacts": artifacts,
                "sources": [
                    str(overview_path),
                    "docs/adr/0019-use-within-cdc2-detection-as-the-main-hiha-ranking.md",
                    "docs/manuscript_figure2_hiha_controlled_missing_state.md",
                ],
                "parameters": {
                    "canvas_inches": list(PANEL_A_SIZE_INCHES),
                    "raster_dpi": PANEL_RASTER_DPI,
                    "layout": "horizontal_strip",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [*outputs.values(), source_path, manifest_path]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(f"Missing or empty Panel A artifact: {path}")
    _validate_raster(outputs["png"])
    return {**outputs, "source_data": source_path, "manifest": manifest_path}


def generate_panel_b(
    *,
    panel_root: Path = PANEL_ROOT,
    result_root: Path = RESULT_ROOT,
    detection_path: Path = PRIMARY_DETECTION_PATH,
) -> dict[str, Path]:
    data = _select_panel_b_detection(detection_path)
    panel_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    outputs = {
        suffix: panel_root / f"panel_b.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    source_path = source_root / "panel_b_detection.csv"
    manifest_path = result_root / "panel_b_manifest.yaml"

    data.to_csv(source_path, index=False)
    fig = _draw_panel_b(data)
    for suffix, path in outputs.items():
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    artifacts = {suffix: str(path) for suffix, path in outputs.items()}
    artifacts["source_data"] = str(source_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel": "B",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py"
                ),
                "artifacts": artifacts,
                "sources": [str(detection_path)],
                "regeneration_commands": [
                    (
                        "uv run python experiments/missing_celltype/"
                        "generate_hiha_dc_compare_baselines.py"
                    ),
                    (
                        "uv run python experiments/missing_celltype/"
                        "generate_hiha_dc_reformulation.py"
                    ),
                ],
                "parameters": {
                    "endpoints": list(ENDPOINTS),
                    "seeds": list(range(1, 6)),
                    "methods": [
                        {"method": method, "score": score, "display": display}
                        for method, score, display in PANEL_B_METHODS
                    ],
                    "method_colors": {
                        method: METHOD_COLORS[method]
                        for method, _, _ in PANEL_B_METHODS
                    },
                    "method_color_source": (
                        "experiments/mouse_spleen/generate_figure4_panels.py"
                    ),
                    "metric_artifact_field": "auprc",
                    "metric_display": "average precision (AP)",
                    "interval": "mean_plus_or_minus_sample_sd_ddof1",
                    "prevalence_reference": "mean_across_five_splits",
                    "canvas_inches": list(PANEL_B_SIZE_INCHES),
                    "raster_dpi": PANEL_RASTER_DPI,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [*outputs.values(), source_path, manifest_path]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(f"Missing or empty Panel B artifact: {path}")
    _validate_panel_b_raster(outputs["png"])
    return {**outputs, "source_data": source_path, "manifest": manifest_path}


def _generate_archival_paired_ap_panel(
    *,
    docs_root: Path = DOCS_ROOT,
    result_root: Path = RESULT_ROOT,
    detection_path: Path = COMPARISON_DETECTION_PATH,
) -> dict[str, Path]:
    data = _select_panel_c_controls(detection_path)
    docs_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    archive_root = result_root / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        suffix: archive_root / f"removed_paired_ap_panel.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    source_path = archive_root / "removed_paired_ap_controls.csv"
    manifest_path = archive_root / "removed_paired_ap_manifest.yaml"

    data.to_csv(source_path, index=False)
    fig = _draw_panel_c(data)
    for suffix, path in outputs.items():
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    artifacts = {suffix: str(path) for suffix, path in outputs.items()}
    artifacts["source_data"] = str(source_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel": "removed-paired-ap",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py"
                ),
                "artifacts": artifacts,
                "sources": [str(detection_path)],
                "regeneration_command": (
                    "uv run python "
                    "experiments/missing_celltype/generate_hiha_dc_compare_baselines.py"
                ),
                "parameters": {
                    "endpoints": list(ENDPOINTS),
                    "seeds": list(range(1, 6)),
                    "reference": {
                        "method": PANEL_C_REFERENCE[0],
                        "score": PANEL_C_REFERENCE[1],
                    },
                    "comparators": [
                        {"method": method, "score": score, "display": display}
                        for method, score, display in PANEL_C_COMPARATORS
                    ],
                    "contrast": "reference_average_precision_minus_comparator",
                    "metric_artifact_field": "auprc",
                    "metric_display": "average precision (AP)",
                    "interval": "mean_plus_or_minus_sample_sd_ddof1",
                    "canvas_inches": list(PANEL_C_SIZE_INCHES),
                    "raster_dpi": PANEL_RASTER_DPI,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [*outputs.values(), source_path, manifest_path]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(
                f"Missing or empty removed paired-AP artifact: {path}"
            )
    _validate_panel_c_raster(outputs["png"])
    return {**outputs, "source_data": source_path, "manifest": manifest_path}


def generate_panel_c(
    *,
    panel_root: Path = PANEL_ROOT,
    result_root: Path = RESULT_ROOT,
    detection_path: Path = COMPARISON_DETECTION_PATH,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Path]:
    cells, summary, specificity, source_paths = _collect_panel_c_data(
        detection_path=detection_path,
        runs_root=runs_root,
    )
    panel_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    outputs = {
        suffix: panel_root / f"panel_c.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    cell_path = source_root / "panel_c_paired_cells.parquet"
    summary_path = source_root / "panel_c_deficit_by_seed.csv"
    specificity_path = source_root / "panel_c_restoration_by_seed.csv"
    manifest_path = result_root / "panel_c_manifest.yaml"

    cells.to_parquet(cell_path, index=False)
    summary.to_csv(summary_path, index=False)
    specificity.to_csv(specificity_path, index=False)
    fig = _draw_panel_c_rescue(summary, specificity)
    for suffix, path in outputs.items():
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    artifacts = {suffix: str(path) for suffix, path in outputs.items()}
    artifacts.update(
        {
            "paired_cells": str(cell_path),
            "deficit_by_seed": str(summary_path),
            "restoration_by_seed": str(specificity_path),
        }
    )
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel": "C",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py"
                ),
                "artifacts": artifacts,
                "sources": source_paths,
                "parameters": {
                    "endpoints": list(ENDPOINTS),
                    "seeds": list(range(1, 6)),
                    "method": "coreot_full",
                    "score": "raw_query_marginal_deficit_u",
                    "groups": [group for group, _ in PANEL_C_GROUPS],
                    "group_summary": "within_split_median",
                    "display_rows": [
                        "held_out_delta_median_u",
                        "control_delta_median_u",
                        "restoration_specificity",
                    ],
                    "restoration_specificity": (
                        "held_out_delta_median_u_minus_control_delta_median_u"
                    ),
                    "destination_summary": (
                        "median_conditional_reference_label_probability"
                    ),
                    "destination_axis_limits": [0.0, 1.0],
                    "interval": "mean_plus_or_minus_sample_sd_ddof1",
                    "connect_split_points_across_rows": False,
                    "canvas_inches": list(PANEL_C_SIZE_INCHES),
                    "raster_dpi": PANEL_RASTER_DPI,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [
        *outputs.values(),
        cell_path,
        summary_path,
        specificity_path,
        manifest_path,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(f"Missing or empty Panel C artifact: {path}")
    _validate_panel_c_rescue_raster(outputs["png"])
    return {
        **outputs,
        "paired_cells": cell_path,
        "deficit_by_seed": summary_path,
        "restoration_by_seed": specificity_path,
        "manifest": manifest_path,
    }


def generate_panel_e(
    *,
    panel_root: Path = PANEL_ROOT,
    result_root: Path = RESULT_ROOT,
    detection_path: Path = COMPARISON_DETECTION_PATH,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Path]:
    cells, summary, source_paths = _collect_panel_e_umap_data(
        detection_path=detection_path,
        runs_root=runs_root,
    )
    panel_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    outputs = {
        suffix: panel_root / f"panel_e.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    cell_path = source_root / "panel_e_umap_cells.csv"
    summary_path = source_root / "panel_e_summary.csv"
    manifest_path = result_root / "panel_e_manifest.yaml"

    cells.to_csv(cell_path, index=False)
    summary.to_csv(summary_path, index=False)
    fig = _draw_panel_e_umap(cells)
    for suffix, path in outputs.items():
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    artifacts = {suffix: str(path) for suffix, path in outputs.items()}
    artifacts.update({"umap_cells": str(cell_path), "summary": str(summary_path)})
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel": "E",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py"
                ),
                "artifacts": artifacts,
                "sources": source_paths,
                "parameters": {
                    "endpoints": list(ENDPOINTS),
                    "representative_seed": REPRESENTATIVE_SEED,
                    "condition": "incomplete_reference",
                    "evaluation_subset": "AIFI_L2_equals_cDC2",
                    "maps": [
                        {
                            "map_id": map_id,
                            "score": score,
                            "display": display,
                        }
                        for map_id, score, display in PANEL_E_MAPS
                    ],
                    "method_colors": {
                        method: METHOD_COLORS[method]
                        for method, _, _ in PANEL_E_METHODS
                    },
                    "truth_color": PANEL_E_TRUTH_COLOR,
                    "eligible_unselected_color": PANEL_E_ELIGIBLE_GRAY,
                    "outside_cohort_color": PANEL_E_CONTEXT_GRAY,
                    "selection_rule": (
                        "largest_oriented_scores_within_cDC2_with_count_equal_to_"
                        "held_out_state_count"
                    ),
                    "selection_tie_breaker": "cell_id_ascending",
                    "selection_interpretation": (
                        "descriptive_fixed_size_set_not_calibrated_rejection"
                    ),
                    "uniform_uot_lineage": "tau05_uniform_replacement_grid",
                    "umap_input": "query rows of frozen X_pca_harmony first 30 dimensions",
                    "umap_n_neighbors": UMAP_N_NEIGHBORS,
                    "umap_metric": "euclidean",
                    "umap_min_dist": UMAP_MIN_DIST,
                    "umap_random_state": UMAP_RANDOM_STATE,
                    "layout": "two_endpoint_rows_by_truth_plus_six_method_columns",
                    "hit_counts_displayed": False,
                    "canvas_inches": list(PANEL_E_SIZE_INCHES),
                    "raster_dpi": PANEL_RASTER_DPI,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [*outputs.values(), cell_path, summary_path, manifest_path]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(f"Missing or empty Panel E artifact: {path}")
    _validate_panel_e_umap_raster(outputs["png"])
    return {
        **outputs,
        "umap_cells": cell_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def generate_panel_d(
    *,
    panel_root: Path = PANEL_ROOT,
    result_root: Path = RESULT_ROOT,
    label_transfer_path: Path = COMPARISON_LABEL_TRANSFER_PATH,
) -> dict[str, Path]:
    data = _select_panel_d_label_transfer(label_transfer_path)
    panel_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    outputs = {
        suffix: panel_root / f"panel_d.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    source_path = source_root / "panel_d_label_transfer.csv"
    manifest_path = result_root / "panel_d_manifest.yaml"

    data.to_csv(source_path, index=False)
    fig = _draw_panel_d_label_transfer(data)
    for suffix, path in outputs.items():
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    artifacts = {suffix: str(path) for suffix, path in outputs.items()}
    artifacts["source_data"] = str(source_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel": "D",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py"
                ),
                "artifacts": artifacts,
                "sources": [str(label_transfer_path)],
                "regeneration_command": (
                    "uv run python "
                    "experiments/missing_celltype/generate_hiha_dc_compare_baselines.py"
                ),
                "parameters": {
                    "endpoints": list(ENDPOINTS),
                    "seeds": list(range(1, 6)),
                    "evaluation_subset": "represented_AIFI_L3_states",
                    "methods": [
                        {
                            "method": method,
                            "score": score,
                            "display": display,
                            "candidate_set": PANEL_D_CANDIDATE_SET_BY_METHOD[method],
                        }
                        for method, score, display in PANEL_D_METHODS
                    ],
                    "metric_artifact_fields": [
                        "forced_macro_f1",
                        "forced_accuracy",
                    ],
                    "encoding": (
                        "horizontal_two_grouped_bars_per_method_and_endpoint"
                    ),
                    "metric_colors": {
                        metric: color
                        for metric, _, color in PANEL_D_METRICS
                    },
                    "split_points": "not_displayed",
                    "interval": "mean_plus_or_minus_sample_sd_ddof1",
                    "mean_marker": False,
                    "metric_legend": "above_endpoint_facets",
                    "axis_limits": list(PANEL_D_LIM),
                    "bar_baseline": PANEL_D_LIM[0],
                    "bar_baseline_is_truncated": True,
                    "disposition": "main_figure_panel",
                    "canvas_inches": list(PANEL_D_SIZE_INCHES),
                    "raster_dpi": PANEL_RASTER_DPI,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [*outputs.values(), source_path, manifest_path]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(f"Missing or empty Panel D artifact: {path}")
    _validate_panel_d_label_transfer_raster(outputs["png"])
    return {**outputs, "source_data": source_path, "manifest": manifest_path}


def generate_panel_f(
    *,
    panel_root: Path = PANEL_ROOT,
    result_root: Path = RESULT_ROOT,
    label_transfer_path: Path = COMPARISON_LABEL_TRANSFER_PATH,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Path]:
    cells, summary, source_paths = _collect_panel_f_label_assignments(
        label_transfer_path=label_transfer_path,
        runs_root=runs_root,
    )
    panel_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    outputs = {
        suffix: panel_root / f"panel_f.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    cell_path = source_root / "panel_f_label_assignment_cells.csv"
    summary_path = source_root / "panel_f_label_assignment_summary.csv"
    manifest_path = result_root / "panel_f_manifest.yaml"

    cells.to_csv(cell_path, index=False)
    summary.to_csv(summary_path, index=False)
    fig = _draw_panel_f_label_assignments(cells)
    for suffix, path in outputs.items():
        fig.savefig(
            path,
            dpi=PANEL_RASTER_DPI if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    artifacts = {suffix: str(path) for suffix, path in outputs.items()}
    artifacts.update(
        {
            "label_assignment_cells": str(cell_path),
            "label_assignment_summary": str(summary_path),
        }
    )
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "manuscript-figure",
                "figure": "HIHA controlled missing-state Figure 2",
                "panel": "F",
                "generator": (
                    "experiments/missing_celltype/generate_hiha_dc_figure2_panels.py"
                ),
                "artifacts": artifacts,
                "sources": source_paths,
                "parameters": {
                    "endpoints": list(ENDPOINTS),
                    "representative_seed": REPRESENTATIVE_SEED,
                    "condition": "incomplete_reference",
                    "evaluation_subset": "represented_AIFI_L3_states",
                    "maps": [
                        {"map_id": map_id, "display": display}
                        for map_id, display in PANEL_F_MAPS
                    ],
                    "biological_label_colors": PANEL_F_LABEL_COLORS,
                    "held_out_color": PANEL_F_HELD_OUT_COLOR,
                    "represented_point_area": PANEL_F_REPRESENTED_POINT_SIZE,
                    "represented_point_alpha": 0.80,
                    "held_out_point_area": PANEL_F_HELD_OUT_POINT_SIZE,
                    "held_out_point_alpha": 1.0,
                    "held_out_interpretation": "not_evaluated",
                    "uniform_uot_lineage": "tau05_uniform_replacement_grid",
                    "umap_input": "query rows of frozen X_pca_harmony first 30 dimensions",
                    "umap_n_neighbors": UMAP_N_NEIGHBORS,
                    "umap_metric": "euclidean",
                    "umap_min_dist": UMAP_MIN_DIST,
                    "umap_random_state": UMAP_RANDOM_STATE,
                    "layout": "two_endpoint_rows_by_truth_plus_five_method_columns",
                    "legend_rows": 1,
                    "endpoint_label_y": list(PANEL_F_ENDPOINT_LABEL_Y),
                    "metric_cross_check": (
                        "seed1_forced_accuracy_and_macro_f1_reproduce_panel_d"
                    ),
                    "interpretation": (
                        "descriptive_spatial_view_not_independent_performance_evidence"
                    ),
                    "dense_points_rasterized_in_vector_outputs": True,
                    "canvas_inches": list(PANEL_F_SIZE_INCHES),
                    "raster_dpi": PANEL_RASTER_DPI,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    required = [*outputs.values(), cell_path, summary_path, manifest_path]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise HIHAFigure2Error(f"Missing or empty Panel F artifact: {path}")
    _validate_panel_f_label_assignment_raster(outputs["png"])
    return {
        **outputs,
        "label_assignment_cells": cell_path,
        "label_assignment_summary": summary_path,
        "manifest": manifest_path,
    }
