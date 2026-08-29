from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from coreot.artifacts.run_artifacts import RunArtifacts

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


STAGE = "manuscript-results"
FIGURE_1_METHOD = "coreot_full"
FIGURE_1_HELD_OUT_LABEL = "ISG+ cDC2"
FIGURE_1_SEED = 1
FIGURE_2_METHODS = (
    "prior_only",
    "uniform_uot",
    "coreot_constant_tau",
    "coreot_full",
)
FIGURE_2_HELD_OUT_LABELS = ("CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2")
FIGURE_PALETTE = {
    "blue_face": "#EAF3FF",
    "blue_edge": "#6D94C9",
    "orange_face": "#FFF3E8",
    "orange_edge": "#F47C00",
    "purple_face": "#F7F0FB",
    "purple_edge": "#A57BC1",
    "gray_face": "#F5F5F5",
    "gray_edge": "#8D8D8D",
    "green_face": "#EFF9EF",
    "green_edge": "#61A967",
    "note_face": "#F8F8F8",
    "note_edge": "#9A9A9A",
    "arrow": "#4B6278",
    "text": "#111111",
    "white": "#FFFFFF",
    "chip_blue": "#F7FBFF",
}


@dataclass(frozen=True)
class Figure1Paths:
    png: Path
    pdf: Path


@dataclass(frozen=True)
class Figure2Paths:
    png: Path
    pdf: Path


def write_figure_1(
    *,
    runs_root: str | Path,
    output_root: str | Path,
    detection_summary_path: str | Path,
    forced_label_summary_path: str | Path,
    candidate_set: str,
    condition: str = "incomplete_reference",
    embedding_name: str = "hiha_harmony30",
    run_id: str = "hiha_dc_isg_cdc2_seed1",
    held_out_label: str = FIGURE_1_HELD_OUT_LABEL,
    seed: int = FIGURE_1_SEED,
    method: str = FIGURE_1_METHOD,
) -> Figure1Paths:
    """Write Figure 1 from completed HIHA DC manuscript-results artifacts."""
    output = Path(output_root) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = Figure1Paths(png=output / "figure_1.png", pdf=output / "figure_1.pdf")

    run_root = Path(runs_root) / run_id
    truth = _read_query_truth(run_root, condition)
    scores = _read_cell_scores(run_root, condition, candidate_set)
    detection_summary = pd.read_csv(detection_summary_path)
    forced_summary = pd.read_csv(forced_label_summary_path)

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    flat_axes = axes.ravel()
    _plot_embedding_panel(
        flat_axes[0],
        run_root=run_root,
        condition=condition,
        embedding_name=embedding_name,
        truth=truth,
        held_out_label=held_out_label,
    )
    _plot_score_distribution_panel(flat_axes[1], scores=scores, truth=truth, method=method)
    _plot_detection_bar_panel(
        flat_axes[2], detection_summary=detection_summary, held_out_label=held_out_label
    )
    _plot_abstention_tradeoff_panel(
        flat_axes[3], detection_summary=detection_summary, held_out_label=held_out_label
    )
    _plot_forced_label_panel(
        flat_axes[4],
        forced_summary=forced_summary,
        run_id=run_id,
        held_out_label=held_out_label,
        seed=seed,
        method=method,
    )
    flat_axes[5].axis("off")
    flat_axes[5].text(
        0.0,
        1.0,
        "\n".join(
            [
                "Figure 1 inputs",
                f"Held-out label: {held_out_label}",
                f"Example run: {run_id}",
                f"Panel B/E method: {method}",
                f"Condition: {condition}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    figure.savefig(paths.png, dpi=300)
    figure.savefig(paths.pdf)
    plt.close(figure)
    return paths


def write_figure_2(
    *,
    output_root: str | Path,
    detection_summary_path: str | Path,
    full_reference_summary_path: str | Path,
    methods: tuple[str, ...] = FIGURE_2_METHODS,
    held_out_labels: tuple[str, ...] = FIGURE_2_HELD_OUT_LABELS,
) -> Figure2Paths:
    """Write Figure 2, the main HIHA DC ablation summary."""
    output = Path(output_root) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = Figure2Paths(png=output / "figure_2.png", pdf=output / "figure_2.pdf")

    detection_summary = pd.read_csv(detection_summary_path)
    full_reference_summary = pd.read_csv(full_reference_summary_path)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    _plot_metric_by_label_panel(
        axes[0, 0],
        summary=detection_summary,
        methods=methods,
        held_out_labels=held_out_labels,
        quantity="auroc",
        title="A. AUROC",
        ylabel="AUROC",
    )
    _plot_metric_by_label_panel(
        axes[0, 1],
        summary=detection_summary,
        methods=methods,
        held_out_labels=held_out_labels,
        quantity="auprc",
        title="B. AUPRC",
        ylabel="AUPRC",
    )
    _plot_auprc_delta_panel(
        axes[1, 0],
        summary=detection_summary,
        methods=methods,
        held_out_labels=held_out_labels,
    )
    _plot_full_reference_false_abstention_panel(
        axes[1, 1],
        summary=full_reference_summary,
        methods=methods,
        held_out_labels=held_out_labels,
    )

    figure.savefig(paths.png, dpi=300)
    figure.savefig(paths.pdf)
    plt.close(figure)
    return paths


def _plot_metric_by_label_panel(
    ax: plt.Axes,
    *,
    summary: pd.DataFrame,
    methods: tuple[str, ...],
    held_out_labels: tuple[str, ...],
    quantity: str,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(held_out_labels), dtype=float)
    width = min(0.18, 0.75 / max(len(methods), 1))
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods, strict=True):
        values = [
            _label_summary_value(summary, method, label, quantity, "mean")
            for label in held_out_labels
        ]
        errors = [
            _label_summary_value(summary, method, label, quantity, "std")
            for label in held_out_labels
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            color=_method_face(method),
            edgecolor=_method_edge(method),
            linewidth=1.2,
            label=_display_method(method),
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(label) for label in held_out_labels], rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)


def _plot_auprc_delta_panel(
    ax: plt.Axes,
    *,
    summary: pd.DataFrame,
    methods: tuple[str, ...],
    held_out_labels: tuple[str, ...],
) -> None:
    compared_methods = tuple(method for method in methods if method != "prior_only")
    x = np.arange(len(held_out_labels), dtype=float)
    width = min(0.22, 0.75 / max(len(compared_methods), 1))
    offsets = (
        np.arange(len(compared_methods), dtype=float) - (len(compared_methods) - 1) / 2.0
    ) * width
    for offset, method in zip(offsets, compared_methods, strict=True):
        deltas = []
        for label in held_out_labels:
            value = _label_summary_value(summary, method, label, "auprc", "mean")
            prior = _label_summary_value(summary, "prior_only", label, "auprc", "mean")
            deltas.append(value - prior if not (np.isnan(value) or np.isnan(prior)) else np.nan)
        ax.bar(
            x + offset,
            deltas,
            width=width,
            color=_method_face(method),
            edgecolor=_method_edge(method),
            linewidth=1.2,
            label=_display_method(method),
        )
    ax.axhline(0.0, color=FIGURE_PALETTE["text"], linewidth=0.8)
    ax.set_title("C. AUPRC difference from prior-only")
    ax.set_ylabel("AUPRC difference")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(label) for label in held_out_labels], rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)


def _plot_full_reference_false_abstention_panel(
    ax: plt.Axes,
    *,
    summary: pd.DataFrame,
    methods: tuple[str, ...],
    held_out_labels: tuple[str, ...],
) -> None:
    x = np.arange(len(held_out_labels), dtype=float)
    width = min(0.18, 0.75 / max(len(methods), 1))
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods, strict=True):
        values = [
            _label_summary_value(
                summary, method, label, "full_reference_false_abstention_rate", "mean"
            )
            for label in held_out_labels
        ]
        errors = [
            _label_summary_value(
                summary, method, label, "full_reference_false_abstention_rate", "std"
            )
            for label in held_out_labels
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            color=_method_face(method),
            edgecolor=_method_edge(method),
            linewidth=1.2,
            label=_display_method(method),
        )
    ax.set_title("D. Full-reference false abstention")
    ax.set_ylabel("False-abstention rate")
    max_value = pd.to_numeric(summary["mean"], errors="coerce").max()
    upper = max(0.08, float(max_value) * 1.45) if not np.isnan(max_value) else 0.08
    ax.set_ylim(0.0, upper)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(label) for label in held_out_labels], rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)


def _plot_embedding_panel(
    ax: plt.Axes,
    *,
    run_root: Path,
    condition: str,
    embedding_name: str,
    truth: pd.DataFrame,
    held_out_label: str,
) -> None:
    embedding_root = run_root / "embeddings" / condition / embedding_name
    coordinates_path = embedding_root / "embedding_2d.npy"
    cells_path = embedding_root / "embedding_cells.csv"
    if not coordinates_path.is_file():
        raise FileNotFoundError(f"2D embedding does not exist: {coordinates_path}")
    if not cells_path.is_file():
        raise FileNotFoundError(f"Embedding cells do not exist: {cells_path}")

    coordinates = np.load(coordinates_path)
    cells = pd.read_csv(cells_path)
    if coordinates.shape[0] != len(cells) or coordinates.shape[1] != 2:
        raise ValueError(
            f"Expected embedding_2d with shape ({len(cells)}, 2); got {coordinates.shape}"
        )
    cells = cells.copy()
    cells["x"] = coordinates[:, 0]
    cells["y"] = coordinates[:, 1]
    joined = cells.merge(truth, on="cell_id", how="left", validate="one_to_one")

    reference = joined.loc[joined["domain"] == "reference"]
    query = joined.loc[joined["domain"] == "query"]
    shared = query.loc[query["is_shared_state"].eq(True)]
    absent = query.loc[query["is_absent_state"].eq(True)]

    ax.scatter(
        reference["x"],
        reference["y"],
        s=4,
        c=FIGURE_PALETTE["gray_edge"],
        alpha=0.32,
        linewidths=0,
    )
    ax.scatter(
        shared["x"],
        shared["y"],
        s=7,
        c=FIGURE_PALETTE["blue_edge"],
        alpha=0.58,
        linewidths=0,
    )
    ax.scatter(
        absent["x"],
        absent["y"],
        s=13,
        c=FIGURE_PALETTE["orange_edge"],
        alpha=0.82,
        linewidths=0,
    )
    ax.set_title("A. 2D embedding")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.legend(
        [
            "reference",
            "query shared",
            f"query {held_out_label}",
        ],
        loc="best",
        fontsize=8,
        frameon=False,
        markerscale=2.0,
    )


def _plot_score_distribution_panel(
    ax: plt.Axes, *, scores: pd.DataFrame, truth: pd.DataFrame, method: str
) -> None:
    joined = _joined_method_scores(scores, truth, method)
    absent = pd.to_numeric(
        joined.loc[joined["is_absent_state"].astype(bool), "u"], errors="coerce"
    ).dropna()
    shared = pd.to_numeric(
        joined.loc[joined["is_shared_state"].astype(bool), "u"], errors="coerce"
    ).dropna()
    bins = np.linspace(0.0, float(max(joined["u"].max(), 1e-12)), 36)
    ax.hist(
        shared,
        bins=bins,
        density=True,
        color=FIGURE_PALETTE["blue_face"],
        edgecolor=FIGURE_PALETTE["blue_edge"],
        linewidth=1.0,
        label="shared",
    )
    ax.hist(
        absent,
        bins=bins,
        density=True,
        color=FIGURE_PALETTE["orange_face"],
        edgecolor=FIGURE_PALETTE["orange_edge"],
        linewidth=1.0,
        label="absent",
    )
    ax.set_title("B. Full CoRe-OT source deficit")
    ax.set_xlabel(r"$u_s$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)


def _plot_detection_bar_panel(
    ax: plt.Axes, *, detection_summary: pd.DataFrame, held_out_label: str
) -> None:
    rows = detection_summary.loc[
        (detection_summary["held_out_label"] == held_out_label)
        & (detection_summary["quantity"].isin(["auroc", "auprc"]))
    ].copy()
    rows["method"] = pd.Categorical(rows["method"], categories=_method_order(), ordered=True)
    rows = rows.sort_values(["method", "quantity"])
    methods = [method for method in _method_order() if method in set(rows["method"].astype(str))]
    x = np.arange(len(methods), dtype=float)
    width = 0.36
    for offset, quantity, face, edge, label in [
        (
            -width / 2,
            "auroc",
            FIGURE_PALETTE["blue_face"],
            FIGURE_PALETTE["blue_edge"],
            "AUROC",
        ),
        (
            width / 2,
            "auprc",
            FIGURE_PALETTE["orange_face"],
            FIGURE_PALETTE["orange_edge"],
            "AUPRC",
        ),
    ]:
        values = [_summary_value(rows, method, quantity, "mean") for method in methods]
        errors = [_summary_value(rows, method, quantity, "std") for method in methods]
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            color=face,
            edgecolor=edge,
            linewidth=1.2,
            label=label,
        )
    ax.set_title("C. Detection performance")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_method(method) for method in methods], rotation=35, ha="right")
    ax.set_ylabel("Mean across seeds")
    ax.legend(frameon=False, fontsize=8)


def _plot_abstention_tradeoff_panel(
    ax: plt.Axes, *, detection_summary: pd.DataFrame, held_out_label: str
) -> None:
    rows = detection_summary.loc[
        (detection_summary["held_out_label"] == held_out_label)
        & (
            detection_summary["quantity"].isin(
                ["absent_abstention_rate", "shared_false_abstention_rate"]
            )
        )
    ]
    for method in _method_order():
        x = _summary_value(rows, method, "shared_false_abstention_rate", "mean")
        y = _summary_value(rows, method, "absent_abstention_rate", "mean")
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(
            x,
            y,
            s=44,
            facecolor=_method_face(method),
            edgecolor=_method_edge(method),
            linewidth=1.2,
        )
        ax.annotate(
            _short_method(method),
            (x, y),
            textcoords="offset points",
            xytext=_tradeoff_label_offset(method),
            fontsize=7,
        )
    ax.set_title("D. Abstention tradeoff")
    ax.set_xlabel("Shared-cell false abstention")
    ax.set_ylabel("Absent-cell abstention")
    ax.set_xlim(0.0, max(0.06, float(rows.loc[rows["quantity"] == "shared_false_abstention_rate", "mean"].max()) * 1.25))
    ax.set_ylim(0.0, 1.05)


def _plot_forced_label_panel(
    ax: plt.Axes,
    *,
    forced_summary: pd.DataFrame,
    run_id: str,
    held_out_label: str,
    seed: int,
    method: str,
) -> None:
    rows = forced_summary.loc[
        (forced_summary["run_id"] == run_id)
        & (forced_summary["held_out_label"] == held_out_label)
        & (forced_summary["seed"] == seed)
        & (forced_summary["method"] == method)
        & (forced_summary["subset"] == "absent_state")
    ].copy()
    if rows.empty:
        raise ValueError(
            "No forced-label rows for "
            f"run_id={run_id}, held_out_label={held_out_label}, seed={seed}, method={method}"
        )
    rows = rows.sort_values("n_cells", ascending=False)
    total = float(rows["n_cells"].sum())
    fractions = rows["n_cells"].astype(float) / total if total > 0 else np.nan
    ax.bar(
        np.arange(len(rows)),
        fractions,
        color=FIGURE_PALETTE["green_face"],
        edgecolor=FIGURE_PALETTE["green_edge"],
        linewidth=1.2,
    )
    ax.set_title("E. Forced labels for absent cells")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(rows["forced_label"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(fractions)) * 1.15))


def _read_query_truth(run_root: Path, condition: str) -> pd.DataFrame:
    return RunArtifacts(run_root, STAGE).evaluation_truth(condition).query_truth().read()


def _read_cell_scores(run_root: Path, condition: str, candidate_set: str) -> pd.DataFrame:
    return RunArtifacts(run_root, STAGE).scoring(condition, candidate_set).cell_scores().read()


def _joined_method_scores(scores: pd.DataFrame, truth: pd.DataFrame, method: str) -> pd.DataFrame:
    method_scores = scores.loc[scores["method"] == method].copy()
    if method_scores.empty:
        raise ValueError(f"No cell scores found for method={method}")
    joined = method_scores.merge(truth, on="cell_id", how="left", validate="many_to_one")
    if joined["true_label"].isna().any():
        raise ValueError(f"Missing query truth for method={method}")
    return joined


def _summary_value(rows: pd.DataFrame, method: str, quantity: str, column: str) -> float:
    values = rows.loc[(rows["method"].astype(str) == method) & (rows["quantity"] == quantity), column]
    if len(values) != 1:
        return np.nan
    return float(values.iloc[0])


def _label_summary_value(
    rows: pd.DataFrame, method: str, held_out_label: str, quantity: str, column: str
) -> float:
    values = rows.loc[
        (rows["method"].astype(str) == method)
        & (rows["held_out_label"].astype(str) == held_out_label)
        & (rows["quantity"] == quantity),
        column,
    ]
    if len(values) != 1:
        return np.nan
    return float(values.iloc[0])


def _method_order() -> tuple[str, ...]:
    return (
        "prior_only",
        "nn",
        "uniform_uot",
        "coreot_constant_tau",
        "coreot_full",
    )


def _display_method(method: str) -> str:
    return {
        "prior_only": "Prior-only",
        "nn": "Nearest neighbor",
        "uniform_uot": "Uniform UOT",
        "coreot_constant_tau": "CoRe-OT constant rho",
        "coreot_full": "Full CoRe-OT",
    }.get(method, method)


def _short_method(method: str) -> str:
    return {
        "prior_only": "Prior",
        "nn": "NN",
        "uniform_uot": "Uniform UOT",
        "coreot_constant_tau": "Const rho",
        "coreot_full": "Full",
    }.get(method, method)


def _tradeoff_label_offset(method: str) -> tuple[int, int]:
    return {
        "prior_only": (4, 12),
        "nn": (18, 2),
        "uniform_uot": (4, 6),
        "coreot_constant_tau": (4, 0),
        "coreot_full": (4, -10),
    }.get(method, (4, 4))


def _method_face(method: str) -> str:
    return {
        "prior_only": FIGURE_PALETTE["note_face"],
        "nn": FIGURE_PALETTE["gray_face"],
        "uniform_uot": FIGURE_PALETTE["orange_face"],
        "coreot_constant_tau": FIGURE_PALETTE["green_face"],
        "coreot_full": FIGURE_PALETTE["blue_face"],
    }.get(method, FIGURE_PALETTE["note_face"])


def _method_edge(method: str) -> str:
    return {
        "prior_only": FIGURE_PALETTE["note_edge"],
        "nn": FIGURE_PALETTE["gray_edge"],
        "uniform_uot": FIGURE_PALETTE["orange_edge"],
        "coreot_constant_tau": FIGURE_PALETTE["green_edge"],
        "coreot_full": FIGURE_PALETTE["blue_edge"],
    }.get(method, FIGURE_PALETTE["text"])


def _short_label(label: str) -> str:
    return {
        "CD14+ cDC2": "CD14+",
        "HLA-DRhi cDC2": "HLA-DRhi",
        "ISG+ cDC2": "ISG+",
    }.get(label, label)


# ---------------------------------------------------------------------------
# Tau sensitivity figures
# ---------------------------------------------------------------------------

SENSITIVITY_SWEPT_METHODS = ("uniform_uot", "coreot_constant_tau")
SENSITIVITY_HELD_OUT_LABELS = ("CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2")


def write_tau_sensitivity_figures(
    *,
    detection_summary_path: str | Path,
    shared_summary_path: str | Path,
    fixed_detection: pd.DataFrame,
    fixed_shared: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write tau-sensitivity diagnostic figures.

    Returns a dict mapping figure name to file path.
    """
    det = pd.read_csv(detection_summary_path)
    shared = pd.read_csv(shared_summary_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    held_out_labels = list(SENSITIVITY_HELD_OUT_LABELS)

    # --- Figure: AUROC vs tau ---
    fig, axes = plt.subplots(1, len(held_out_labels), figsize=(5 * len(held_out_labels), 4), squeeze=False)
    for ax_idx, lbl in enumerate(held_out_labels):
        ax = axes[0, ax_idx]
        _line_plot_vs_tau(det, lbl, "auroc", "AUROC", ax)
        ax.set_title(_short_label(lbl))
    fig.tight_layout()
    png_path = out / "auroc_vs_tau.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["auroc_vs_tau"] = png_path

    # --- Figure: AUPRC vs tau ---
    fig, axes = plt.subplots(1, len(held_out_labels), figsize=(5 * len(held_out_labels), 4), squeeze=False)
    for ax_idx, lbl in enumerate(held_out_labels):
        ax = axes[0, ax_idx]
        _line_plot_vs_tau(det, lbl, "auprc", "AUPRC", ax)
        ax.set_title(_short_label(lbl))
    fig.tight_layout()
    png_path = out / "auprc_vs_tau.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["auprc_vs_tau"] = png_path

    # --- Figure: Abstention tradeoff vs tau ---
    fig, axes = plt.subplots(1, len(held_out_labels), figsize=(5 * len(held_out_labels), 4), squeeze=False)
    for ax_idx, lbl in enumerate(held_out_labels):
        ax = axes[0, ax_idx]
        _abstention_tradeoff_plot(det, lbl, ax)
        ax.set_title(_short_label(lbl))
    fig.tight_layout()
    png_path = out / "abstention_tradeoff_vs_tau.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["abstention_tradeoff_vs_tau"] = png_path

    # --- Figure: Shared label-transfer stability ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_idx, qty in enumerate(("post_abstention_macro_f1", "coverage")):
        ax = axes[ax_idx]
        _line_plot_vs_tau_overall(shared, qty, ax)
    fig.tight_layout()
    png_path = out / "shared_transfer_vs_tau.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["shared_transfer_vs_tau"] = png_path

    return paths


def _line_plot_vs_tau(
    det: pd.DataFrame,
    held_out_label: str,
    quantity: str,
    ylabel: str,
    ax: plt.Axes,
) -> None:
    """Line plot of a metric vs tau for a single held-out label."""
    frame = det.loc[
        (det["held_out_label"] == held_out_label)
        & (det["quantity"] == quantity)
        & (det["method"].isin(SENSITIVITY_SWEPT_METHODS))
    ].copy()
    if frame.empty:
        return
    tau_values = sorted(frame["tau"].dropna().unique())
    for method in SENSITIVITY_SWEPT_METHODS:
        method_frame = frame.loc[frame["method"] == method]
        means, stds = [], []
        taus_used = []
        for t in tau_values:
            row = method_frame.loc[method_frame["tau"] == t]
            if row.empty:
                continue
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].iloc[0]) if not (isinstance(row["std"].iloc[0], float) and np.isnan(float(row["std"].iloc[0]))) else 0)
            taus_used.append(t)
        if not taus_used:
            continue
        color = _method_edge(method)
        short = {"uniform_uot": "Uniform UOT", "coreot_constant_tau": "CoRe-OT const τ"}.get(method, method)
        ax.errorbar(taus_used, means, yerr=stds, marker="o", color=color, label=short, capsize=3)
    ax.set_xlabel("τ")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(frame["tau"].dropna().unique()))
    ax.set_xticklabels([str(int(t)) for t in sorted(frame["tau"].dropna().unique())])


def _line_plot_vs_tau_overall(
    shared: pd.DataFrame,
    quantity: str,
    ax: plt.Axes,
) -> None:
    """Line plot of a shared-cell metric vs tau (overall mean)."""
    label_map = {
        "post_abstention_macro_f1": "Post-abstention macro-F1",
        "coverage": "Coverage",
    }
    frame = shared.loc[
        (shared["held_out_label"] == "overall")
        & (shared["quantity"] == quantity)
        & (shared["method"].isin(SENSITIVITY_SWEPT_METHODS))
    ].copy()
    if frame.empty:
        return
    tau_values = sorted(frame["tau"].dropna().unique())
    for method in SENSITIVITY_SWEPT_METHODS:
        method_frame = frame.loc[frame["method"] == method]
        means, stds, taus_used = [], [], []
        for t in tau_values:
            row = method_frame.loc[method_frame["tau"] == t]
            if row.empty:
                continue
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].iloc[0]) if not (isinstance(row["std"].iloc[0], float) and np.isnan(float(row["std"].iloc[0]))) else 0)
            taus_used.append(t)
        if not taus_used:
            continue
        color = _method_edge(method)
        short = {"uniform_uot": "Uniform UOT", "coreot_constant_tau": "CoRe-OT const τ"}.get(method, method)
        ax.errorbar(taus_used, means, yerr=stds, marker="o", color=color, label=short, capsize=3)
    ax.set_xlabel("τ")
    ax.set_ylabel(label_map.get(quantity, quantity))
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(frame["tau"].dropna().unique()))
    ax.set_xticklabels([str(int(t)) for t in sorted(frame["tau"].dropna().unique())])


def _abstention_tradeoff_plot(
    det: pd.DataFrame,
    held_out_label: str,
    ax: plt.Axes,
) -> None:
    """Scatter plot: shared false-abstention vs absent abstention, colored by tau."""
    absent = det.loc[
        (det["held_out_label"] == held_out_label)
        & (det["quantity"] == "absent_abstention_rate")
        & (det["method"].isin(SENSITIVITY_SWEPT_METHODS))
    ]
    shared = det.loc[
        (det["held_out_label"] == held_out_label)
        & (det["quantity"] == "shared_false_abstention_rate")
        & (det["method"].isin(SENSITIVITY_SWEPT_METHODS))
    ]
    if absent.empty or shared.empty:
        return
    absent = absent.set_index(["method", "tau"])["mean"]
    shared = shared.set_index(["method", "tau"])["mean"]
    merged = pd.DataFrame({"absent_abstention": absent, "shared_false_abstention": shared}).reset_index()
    for method in SENSITIVITY_SWEPT_METHODS:
        mf = merged.loc[merged["method"] == method]
        color = _method_edge(method)
        short = {"uniform_uot": "Uniform UOT", "coreot_constant_tau": "CoRe-OT const τ"}.get(method, method)
        ax.scatter(
            mf["shared_false_abstention"],
            mf["absent_abstention"],
            c=color, label=short, s=40,
        )
        for _, pt in mf.iterrows():
            ax.annotate(
                f"τ={int(pt['tau'])}",
                (pt["shared_false_abstention"], pt["absent_abstention"]),
                fontsize=7, alpha=0.8,
                textcoords="offset points", xytext=(4, 4),
            )
    ax.set_xlabel("Shared-cell false-abstention rate")
    ax.set_ylabel("Absent-cell abstention rate")
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Labelwise alpha-tau sensitivity figures
# ---------------------------------------------------------------------------

LABELWISE_METHOD = "coreot_constant_tau"


def write_labelwise_figures(
    *,
    detection_summary_path: str | Path,
    shared_summary_path: str | Path,
    fixed_detection: pd.DataFrame,
    fixed_shared: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write labelwise sensitivity diagnostic figures."""
    det = pd.read_csv(detection_summary_path)
    shared = pd.read_csv(shared_summary_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # --- HLA-DRhi cDC2: AUROC/AUPRC vs alpha at tau=2 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _alpha_sensitivity_plot(det, "HLA-DRhi cDC2", "auroc", "AUROC", ax1)
    _alpha_sensitivity_plot(det, "HLA-DRhi cDC2", "auprc", "AUPRC", ax2)
    fig.suptitle("HLA-DRhi cDC2 — α-sensitivity at τ=2", fontsize=11)
    fig.tight_layout()
    png_path = out / "hla_drhi_alpha_sensitivity.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["hla_drhi_alpha_sensitivity"] = png_path

    # --- CD14+ cDC2: AUROC/AUPRC vs tau at alpha=5 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _tau_at_alpha5_plot(det, "CD14+ cDC2", "auroc", "AUROC", ax1)
    _tau_at_alpha5_plot(det, "CD14+ cDC2", "auprc", "AUPRC", ax2)
    fig.suptitle("CD14+ cDC2 — τ-sensitivity at α=5", fontsize=11)
    fig.tight_layout()
    png_path = out / "cd14_tau_sensitivity_alpha5.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["cd14_tau_sensitivity_alpha5"] = png_path

    # --- ISG+ cDC2: AUROC/AUPRC vs tau at alpha=5 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _tau_at_alpha5_plot(det, "ISG+ cDC2", "auroc", "AUROC", ax1)
    _tau_at_alpha5_plot(det, "ISG+ cDC2", "auprc", "AUPRC", ax2)
    fig.suptitle("ISG+ cDC2 — τ-sensitivity at α=5", fontsize=11)
    fig.tight_layout()
    png_path = out / "isg_tau_sensitivity_alpha5.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["isg_tau_sensitivity_alpha5"] = png_path

    # --- Abstention tradeoff ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_idx, lbl in enumerate(["CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2"]):
        _labelwise_abstention_plot(det, lbl, axes[ax_idx])
    fig.tight_layout()
    png_path = out / "abstention_tradeoff.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["abstention_tradeoff"] = png_path

    # --- Shared transfer stability ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _labelwise_transfer_plot(shared, "post_abstention_macro_f1", "Post-abstention macro-F1", ax1)
    _labelwise_transfer_plot(shared, "coverage", "Coverage", ax2)
    fig.tight_layout()
    png_path = out / "shared_transfer_stability.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["shared_transfer_stability"] = png_path

    return paths


def _alpha_sensitivity_plot(
    det: pd.DataFrame, held_out_label: str, quantity: str, ylabel: str, ax: plt.Axes
) -> None:
    """Plot metric vs alpha for a specific held-out label."""
    frame = det.loc[
        (det["held_out_label"] == held_out_label)
        & (det["quantity"] == quantity)
        & (det["method"] == LABELWISE_METHOD)
    ].copy()
    if frame.empty:
        return
    alpha_values = sorted(frame["alpha"].dropna().unique())
    means, stds = [], []
    for a in alpha_values:
        row = frame.loc[frame["alpha"] == a]
        if row.empty:
            continue
        means.append(float(row["mean"].iloc[0]))
        stds.append(float(row["std"].iloc[0]) if not pd.isna(float(row["std"].iloc[0])) else 0)
    color = _method_edge(LABELWISE_METHOD)
    ax.errorbar(alpha_values, means, yerr=stds, marker="o", color=color, capsize=3)
    ax.set_xlabel("α")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(alpha_values)
    ax.set_xticklabels([str(int(a)) for a in alpha_values])


def _tau_at_alpha5_plot(
    det: pd.DataFrame, held_out_label: str, quantity: str, ylabel: str, ax: plt.Axes
) -> None:
    """Plot metric vs tau for a specific held-out label at fixed alpha=5."""
    frame = det.loc[
        (det["held_out_label"] == held_out_label)
        & (det["quantity"] == quantity)
        & (det["method"] == LABELWISE_METHOD)
    ].copy()
    if frame.empty:
        return
    tau_values = sorted(frame["tau"].dropna().unique())
    means, stds = [], []
    for t in tau_values:
        row = frame.loc[frame["tau"] == t]
        if row.empty:
            continue
        means.append(float(row["mean"].iloc[0]))
        stds.append(float(row["std"].iloc[0]) if not pd.isna(float(row["std"].iloc[0])) else 0)
    color = _method_edge(LABELWISE_METHOD)
    ax.errorbar(tau_values, means, yerr=stds, marker="o", color=color, capsize=3, label="CoRe-OT const τ, α=5")
    ax.set_xlabel("τ")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(tau_values)
    ax.set_xticklabels([str(int(t)) for t in tau_values])
    ax.legend(fontsize=8)


def _labelwise_abstention_plot(
    det: pd.DataFrame, held_out_label: str, ax: plt.Axes
) -> None:
    """Abstention tradeoff scatter for labelwise sweep."""
    absent = det.loc[
        (det["held_out_label"] == held_out_label)
        & (det["quantity"] == "absent_abstention_rate")
        & (det["method"] == LABELWISE_METHOD)
    ]
    shared = det.loc[
        (det["held_out_label"] == held_out_label)
        & (det["quantity"] == "shared_false_abstention_rate")
        & (det["method"] == LABELWISE_METHOD)
    ]
    if absent.empty or shared.empty:
        return
    absent = absent.set_index(["method", "tau", "alpha"])["mean"]
    shared = shared.set_index(["method", "tau", "alpha"])["mean"]
    merged = pd.DataFrame({"absent_abstention": absent, "shared_false_abstention": shared}).reset_index()
    color = _method_edge(LABELWISE_METHOD)
    ax.scatter(
        merged["shared_false_abstention"], merged["absent_abstention"],
        c=color, s=30,
    )
    for _, pt in merged.iterrows():
        if held_out_label == "HLA-DRhi cDC2":
            label = f"α={int(pt['alpha'])}"
        else:
            label = f"τ={int(pt['tau'])}"
        ax.annotate(
            label, (pt["shared_false_abstention"], pt["absent_abstention"]),
            fontsize=6, alpha=0.8, textcoords="offset points", xytext=(4, 4),
        )
    ax.set_title(_short_label(held_out_label), fontsize=9)
    ax.set_xlabel("Shared false-abstention")
    ax.set_ylabel("Absent abstention")


def _labelwise_transfer_plot(
    shared: pd.DataFrame, quantity: str, ylabel: str, ax: plt.Axes
) -> None:
    """Shared transfer metric across swept values for all three labels."""
    frame = shared.loc[
        (shared["held_out_label"] != "overall")
        & (shared["quantity"] == quantity)
        & (shared["method"] == LABELWISE_METHOD)
    ].copy()
    if frame.empty:
        return
    color_map = {
        "CD14+ cDC2": _method_edge("uniform_uot"),
        "HLA-DRhi cDC2": _method_edge("coreot_constant_tau"),
        "ISG+ cDC2": FIGURE_PALETTE["purple_edge"],
    }
    for lbl in frame["held_out_label"].unique():
        lbl_frame = frame.loc[frame["held_out_label"] == lbl]
        swept_values = []
        means = []
        if "alpha" in lbl_frame.columns and lbl == "HLA-DRhi cDC2":
            for a in sorted(lbl_frame["alpha"].dropna().unique()):
                row = lbl_frame.loc[lbl_frame["alpha"] == a]
                if row.empty:
                    continue
                swept_values.append(float(a))
                means.append(float(row["mean"].iloc[0]))
            xlabel = "α"
        else:
            for t in sorted(lbl_frame["tau"].dropna().unique()):
                row = lbl_frame.loc[lbl_frame["tau"] == t]
                if row.empty:
                    continue
                swept_values.append(float(t))
                means.append(float(row["mean"].iloc[0]))
            xlabel = "τ"
        if swept_values:
            ax.plot(swept_values, means, marker="o", color=color_map.get(lbl, "gray"),
                    label=_short_label(lbl))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Broad grid heatmaps
# ---------------------------------------------------------------------------

HEATMAP_QUANTITIES = [
    "auroc",
    "auprc",
    "absent_abstention_rate",
    "shared_false_abstention_rate",
    "coverage",
    "post_abstention_macro_f1",
]


def write_broad_grid_heatmaps(
    *,
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    output_dir: str | Path,
    held_out_labels: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Write one multi-page PDF of tau x alpha heatmaps for coreot metrics.

    Parameters
    ----------
    held_out_labels : optional tuple of label strings to include in a specific order.
        When None, all held_out_label values in the merged data are used, sorted.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    det_wide = detection_summary.loc[
        detection_summary["held_out_label"] != "overall"
    ].pivot_table(
        index=["held_out_label", "method", "tau", "alpha"],
        columns="quantity", values="mean",
    ).reset_index()

    shared_wide = shared_summary.loc[
        shared_summary["held_out_label"] != "overall"
    ].pivot_table(
        index=["held_out_label", "method", "tau", "alpha"],
        columns="quantity", values="mean",
    ).reset_index()

    cols = ["held_out_label", "method", "tau", "alpha"]
    available = [c for c in cols if c in det_wide.columns and c in shared_wide.columns]
    merged = det_wide.merge(shared_wide, on=available, how="outer")

    held_out_labels_list: list[str]
    if held_out_labels is not None:
        held_out_labels_list = [
            label
            for label in held_out_labels
            if label in set(merged["held_out_label"].dropna().unique())
        ]
    else:
        held_out_labels_list = sorted(
            label
            for label in merged["held_out_label"].dropna().unique()
            if label != "overall"
        )
    metrics = [q for q in HEATMAP_QUANTITIES if q in merged.columns]
    panel_pdf_path = out / "heatmaps_panel.pdf"
    panel_png_path = out / "heatmaps_panel.png"
    if held_out_labels_list and metrics:
        figure, axes = plt.subplots(
            len(metrics),
            len(held_out_labels_list),
            figsize=(4.9 * len(held_out_labels_list), 2.45 * len(metrics)),
            squeeze=False,
        )
        for row_index, qty in enumerate(metrics):
            vmin = float(merged[qty].min()) if not merged[qty].isna().all() else 0.0
            vmax = float(merged[qty].max()) if not merged[qty].isna().all() else 1.0
            cmap = (
                "RdYlBu_r"
                if qty in ("shared_false_abstention_rate", "absent_abstention_rate")
                else "RdYlGn"
            )
            for col_index, label in enumerate(held_out_labels_list):
                axis = axes[row_index, col_index]
                label_data = merged.loc[merged["held_out_label"] == label]
                heatmap_data = label_data.pivot_table(
                    index="tau",
                    columns="alpha",
                    values=qty,
                    aggfunc="mean",
                )
                heatmap_data = heatmap_data.reindex(
                    index=sorted(heatmap_data.index),
                    columns=sorted(heatmap_data.columns),
                )
                image = axis.imshow(
                    heatmap_data.values,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                )
                axis.set_xticks(range(len(heatmap_data.columns)))
                axis.set_xticklabels(
                    [_format_heatmap_tick(value) for value in heatmap_data.columns],
                    fontsize=7,
                    rotation=45,
                )
                axis.set_yticks(range(len(heatmap_data.index)))
                axis.set_yticklabels(
                    [_format_heatmap_tick(value) for value in heatmap_data.index],
                    fontsize=7,
                )
                if row_index == len(metrics) - 1:
                    axis.set_xlabel("α", fontsize=8)
                else:
                    axis.set_xlabel("")
                if col_index == 0:
                    axis.set_ylabel(f"{qty}\nτ", fontsize=8)
                else:
                    axis.set_ylabel("")
                if row_index == 0:
                    axis.set_title(_short_label(label), fontsize=10)
                figure.colorbar(image, ax=axis, shrink=0.70)
        figure.suptitle("Fixed-τ α-sensitivity heatmaps", fontsize=12)
        figure.tight_layout()
        figure.savefig(panel_png_path, dpi=300, bbox_inches="tight")
        figure.savefig(panel_pdf_path, bbox_inches="tight")
        plt.close(figure)
        paths["heatmaps_panel_png"] = panel_png_path
        paths["heatmaps_panel_pdf"] = panel_pdf_path

    pdf_path = out / "heatmaps.pdf"
    with PdfPages(pdf_path) as pdf:
        n_labels = len(held_out_labels_list)
        for qty in metrics:
            vmin = float(merged[qty].min()) if not merged[qty].isna().all() else 0.0
            vmax = float(merged[qty].max()) if not merged[qty].isna().all() else 1.0
            n_rows = (n_labels + 1) // 2
            figure, axes = plt.subplots(
                n_rows, 2,
                figsize=(11.0, 4.25 * n_rows),
                squeeze=False,
            )
            cmap = (
                "RdYlBu_r"
                if qty in ("shared_false_abstention_rate", "absent_abstention_rate")
                else "RdYlGn"
            )
            for axis, label in zip(axes.ravel(), held_out_labels_list, strict=False):
                label_data = merged.loc[merged["held_out_label"] == label]
                if label_data.empty:
                    axis.axis("off")
                    continue
                heatmap_data = label_data.pivot_table(
                    index="tau",
                    columns="alpha",
                    values=qty,
                    aggfunc="mean",
                )
                heatmap_data = heatmap_data.reindex(
                    index=sorted(heatmap_data.index),
                    columns=sorted(heatmap_data.columns),
                )
                image = axis.imshow(
                    heatmap_data.values,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                )
                axis.set_xticks(range(len(heatmap_data.columns)))
                axis.set_xticklabels(
                    [_format_heatmap_tick(value) for value in heatmap_data.columns],
                    fontsize=8,
                    rotation=45,
                )
                axis.set_yticks(range(len(heatmap_data.index)))
                axis.set_yticklabels(
                    [_format_heatmap_tick(value) for value in heatmap_data.index],
                    fontsize=8,
                )
                axis.set_xlabel("α", fontsize=9)
                axis.set_ylabel("τ", fontsize=9)
                axis.set_title(
                    "Mean across three held-out labels"
                    if label == "overall"
                    else _short_label(label),
                    fontsize=10,
                )
                figure.colorbar(image, ax=axis, shrink=0.82)
            for axis in axes.ravel()[len(held_out_labels_list):]:
                axis.axis("off")
            figure.suptitle(f"Broad τ×α heatmaps — {qty}", fontsize=12)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    paths["heatmaps_pdf"] = pdf_path
    return paths


def write_broad_grid_alpha_sensitivity_figure(
    *,
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    output_dir: str | Path,
    held_out_label: str = "HLA-DRhi cDC2",
    tau_values: tuple[float, ...] = (1.0, 5.0, 9.0),
    highlight_points: tuple[tuple[float, float], ...] = (
        (1.0, 0.0),
        (1.0, 2.0),
        (5.0, 5.0),
        (9.0, 0.0),
        (9.0, 10.0),
    ),
) -> dict[str, Path]:
    """Write the manuscript-facing HLA-DRhi alpha-sensitivity Figure 2."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / "figure_2_alpha_sensitivity.png"
    pdf_path = out / "figure_2_alpha_sensitivity.pdf"

    det_wide = _summary_to_wide(
        detection_summary,
        held_out_label=held_out_label,
        quantities=(
            "auroc",
            "auprc",
            "absent_abstention_rate",
            "shared_false_abstention_rate",
        ),
    )
    shared_wide = _summary_to_wide(
        shared_summary,
        held_out_label=held_out_label,
        quantities=("coverage", "post_abstention_macro_f1"),
    )
    merged = det_wide.merge(
        shared_wide,
        on=["held_out_label", "method", "tau", "alpha"],
        how="outer",
    )
    merged = merged.loc[merged["tau"].isin(tau_values)].copy()
    if merged.empty:
        raise ValueError(f"No alpha-sensitivity rows found for {held_out_label!r}.")

    colors = {1.0: "#1B6CA8", 5.0: "#7A3E9D", 9.0: "#C45A18"}
    figure = plt.figure(figsize=(13.4, 4.6), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.30, 1.0, 1.0))
    panel_a = grid[0, 0].subgridspec(2, 1, hspace=0.08)
    ax_auroc = figure.add_subplot(panel_a[0, 0])
    ax_auprc = figure.add_subplot(panel_a[1, 0], sharex=ax_auroc)
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[0, 2])

    _plot_alpha_detection_panel(
        ax_auroc=ax_auroc,
        ax_auprc=ax_auprc,
        data=merged,
        tau_values=tau_values,
        colors=colors,
    )
    _plot_alpha_abstention_tradeoff_panel(
        ax_b,
        data=merged,
        tau_values=tau_values,
        colors=colors,
        highlight_points=highlight_points,
    )
    _plot_alpha_selective_transfer_panel(
        ax_c,
        data=merged,
        tau_values=tau_values,
        colors=colors,
        highlight_points=highlight_points,
    )

    _add_panel_label(ax_auroc, "A")
    _add_panel_label(ax_b, "B")
    _add_panel_label(ax_c, "C")

    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {
        "figure_2_alpha_sensitivity_png": png_path,
        "figure_2_alpha_sensitivity_pdf": pdf_path,
    }


SUPPLEMENTARY_OPERATING_POINTS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (1.0, 2.0),
    (5.0, 0.0),
    (5.0, 5.0),
    (9.0, 0.0),
    (9.0, 8.0),
)


def write_hiha_supplementary_heatmaps(
    *,
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    output_dir: str | Path,
    held_out_label: str = "HLA-DRhi cDC2",
) -> dict[str, Path]:
    """Write a supplementary heatmap panel for one manuscript held-out label."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure_spec = {
        "HLA-DRhi cDC2": ("supplementary_figure_s1", "hla"),
        "ISG+ cDC2": ("supplementary_figure_s3", "isg"),
    }
    if held_out_label not in figure_spec:
        raise ValueError(f"Unsupported supplementary heatmap label: {held_out_label!r}.")
    figure_key, label_slug = figure_spec[held_out_label]
    figure_stem = f"{figure_key}_{label_slug}_heatmaps"
    png_path = out / f"{figure_stem}.png"
    pdf_path = out / f"{figure_stem}.pdf"

    detection = _summary_to_wide(
        detection_summary,
        held_out_label=held_out_label,
        quantities=(
            "auroc",
            "auprc",
            "absent_abstention_rate",
            "shared_false_abstention_rate",
        ),
    )
    shared = _summary_to_wide(
        shared_summary,
        held_out_label=held_out_label,
        quantities=("coverage", "post_abstention_macro_f1"),
    )
    data = detection.merge(
        shared,
        on=["held_out_label", "method", "tau", "alpha"],
        how="outer",
    )
    tau_values = tuple(float(value) for value in range(1, 10))
    alpha_values = tuple(float(value) for value in range(0, 11))
    _validate_sensitivity_grid(
        data,
        held_out_label=held_out_label,
        tau_values=tau_values,
        alpha_values=alpha_values,
    )

    panels = (
        ("auroc", "AUROC", "viridis"),
        ("auprc", "AUPRC", "viridis"),
        ("absent_abstention_rate", "Absent-cell abstention", "magma"),
        ("shared_false_abstention_rate", "Shared-cell false abstention", "magma_r"),
        ("coverage", "Shared-cell coverage", "viridis"),
        ("post_abstention_macro_f1", "Post-abstention macro-F1", "viridis"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), constrained_layout=True)
    for panel_index, (axis, (metric, title, cmap)) in enumerate(
        zip(axes.ravel(), panels, strict=True)
    ):
        matrix = (
            data.pivot_table(index="tau", columns="alpha", values=metric, aggfunc="mean")
            .reindex(index=tau_values, columns=alpha_values)
        )
        if matrix.isna().any().any():
            raise ValueError(f"Incomplete {metric} surface for {held_out_label!r}.")
        image = axis.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap=cmap)
        axis.axvline(0.5, color="white", linewidth=1.4)
        axis.set_xticks(range(len(alpha_values)))
        axis.set_xticklabels(["0\nUniform", *[str(value) for value in range(1, 11)]], fontsize=7)
        axis.set_yticks(range(len(tau_values)))
        axis.set_yticklabels([str(value) for value in range(1, 10)], fontsize=8)
        axis.set_xlabel(r"Broad-anchor weight $\alpha$")
        axis.set_ylabel(r"Marginal penalty $\tau$")
        axis.set_title(title)
        _add_panel_label(axis, chr(ord("A") + panel_index))
        colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
        colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(
        f"{held_out_label}: fixed-τ broad-anchor sensitivity\n"
        "α=0: uniform UOT; α>0: constant-τ CoRe-OT",
        fontsize=12,
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {f"{figure_key}_png": png_path, f"{figure_key}_pdf": pdf_path}


def write_hiha_supplementary_seed_tradeoffs(
    *,
    detection_by_run: pd.DataFrame,
    shared_by_run: pd.DataFrame,
    output_dir: str | Path,
    held_out_label: str = "HLA-DRhi cDC2",
    operating_points: tuple[tuple[float, float], ...] = SUPPLEMENTARY_OPERATING_POINTS,
) -> dict[str, Path]:
    """Write Supplementary Figure S2 and its selected seed-level source table."""
    out = Path(output_dir)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    png_path = out / "supplementary_figure_s2_seed_tradeoffs.png"
    pdf_path = out / "supplementary_figure_s2_seed_tradeoffs.pdf"
    data_path = data_dir / "supplementary_figure_s2_operating_points_by_seed.csv"

    key_columns = ["run_id", "held_out_label", "seed", "method", "tau", "alpha"]
    detection_columns = key_columns + [
        "auprc",
        "absent_abstention_rate",
        "shared_false_abstention_rate",
    ]
    shared_columns = key_columns + ["coverage", "post_abstention_macro_f1"]
    missing_detection = set(detection_columns) - set(detection_by_run.columns)
    missing_shared = set(shared_columns) - set(shared_by_run.columns)
    if missing_detection or missing_shared:
        raise ValueError(
            "Missing per-seed columns: "
            f"detection={sorted(missing_detection)}, shared={sorted(missing_shared)}."
        )
    data = detection_by_run.loc[
        detection_by_run["held_out_label"] == held_out_label, detection_columns
    ].merge(
        shared_by_run.loc[shared_by_run["held_out_label"] == held_out_label, shared_columns],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    selected_parts = [
        data.loc[np.isclose(data["tau"], tau) & np.isclose(data["alpha"], alpha)]
        for tau, alpha in operating_points
    ]
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["operating_point"] = selected.apply(
        lambda row: f"({_format_heatmap_tick(row['tau'])},{_format_heatmap_tick(row['alpha'])})",
        axis=1,
    )
    metric_columns = [
        "auprc",
        "absent_abstention_rate",
        "shared_false_abstention_rate",
        "coverage",
        "post_abstention_macro_f1",
    ]
    expected_seeds = {1, 2, 3, 4, 5}
    for tau, alpha in operating_points:
        point = selected.loc[np.isclose(selected["tau"], tau) & np.isclose(selected["alpha"], alpha)]
        if set(point["seed"].astype(int)) != expected_seeds or len(point) != len(expected_seeds):
            raise ValueError(f"Incomplete seed rows for operating point {(tau, alpha)}.")
    if not np.isfinite(selected[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Supplementary seed trade-off metrics must all be finite.")
    selected.to_csv(data_path, index=False)

    labels = [f"({_format_heatmap_tick(tau)},{_format_heatmap_tick(alpha)})" for tau, alpha in operating_points]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(operating_points)))
    color_by_label = dict(zip(labels, colors, strict=True))
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    offsets = np.linspace(-0.16, 0.16, len(expected_seeds))

    for x_position, label in enumerate(labels):
        point = selected.loc[selected["operating_point"] == label].sort_values("seed")
        axes[0].scatter(
            x_position + offsets,
            point["auprc"],
            color=color_by_label[label],
            s=28,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[0].scatter(
            x_position,
            point["auprc"].mean(),
            color=color_by_label[label],
            marker="D",
            s=70,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].set_xlabel(r"Operating point $(\tau,\alpha)$")
    axes[0].set_ylabel("AUPRC")

    for label in labels:
        point = selected.loc[selected["operating_point"] == label]
        color = color_by_label[label]
        axes[1].scatter(
            point["shared_false_abstention_rate"],
            point["absent_abstention_rate"],
            color=color,
            s=30,
            alpha=0.55,
        )
        axes[1].scatter(
            point["shared_false_abstention_rate"].mean(),
            point["absent_abstention_rate"].mean(),
            color=color,
            marker="D",
            s=76,
            edgecolor="black",
            linewidth=0.8,
            label=label,
        )
        axes[2].scatter(
            point["coverage"],
            point["post_abstention_macro_f1"],
            color=color,
            s=30,
            alpha=0.55,
        )
        axes[2].scatter(
            point["coverage"].mean(),
            point["post_abstention_macro_f1"].mean(),
            color=color,
            marker="D",
            s=76,
            edgecolor="black",
            linewidth=0.8,
        )
    axes[1].set_xlabel("Shared-cell false abstention")
    axes[1].set_ylabel("Absent-cell abstention")
    axes[2].set_xlabel("Shared-cell coverage")
    axes[2].set_ylabel("Post-abstention macro-F1")
    for panel_index, axis in enumerate(axes):
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        _add_panel_label(axis, chr(ord("A") + panel_index))
    figure.legend(
        *axes[1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        title=r"Operating point $(\tau,\alpha)$; circles: seeds, diamonds: means",
        fontsize=8,
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {
        "supplementary_figure_s2_png": png_path,
        "supplementary_figure_s2_pdf": pdf_path,
        "supplementary_figure_s2_data": data_path,
    }


def _validate_sensitivity_grid(
    data: pd.DataFrame,
    *,
    held_out_label: str,
    tau_values: tuple[float, ...],
    alpha_values: tuple[float, ...],
) -> None:
    actual = {
        (float(tau), float(alpha))
        for tau, alpha in data[["tau", "alpha"]].drop_duplicates().itertuples(index=False)
    }
    expected = {(tau, alpha) for tau in tau_values for alpha in alpha_values}
    if actual != expected:
        raise ValueError(
            f"Incomplete sensitivity grid for {held_out_label!r}: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )


def _summary_to_wide(
    summary: pd.DataFrame,
    *,
    held_out_label: str,
    quantities: tuple[str, ...],
) -> pd.DataFrame:
    frame = summary.loc[
        (summary["held_out_label"] == held_out_label)
        & (summary["quantity"].isin(quantities))
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["held_out_label", "method", "tau", "alpha", *quantities]
        )
    return (
        frame.pivot_table(
            index=["held_out_label", "method", "tau", "alpha"],
            columns="quantity",
            values="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def _plot_alpha_detection_panel(
    *,
    ax_auroc: plt.Axes,
    ax_auprc: plt.Axes,
    data: pd.DataFrame,
    tau_values: tuple[float, ...],
    colors: dict[float, str],
) -> None:
    for tau in tau_values:
        tau_data = data.loc[data["tau"] == tau].sort_values("alpha")
        if tau_data.empty:
            continue
        color = colors.get(tau, FIGURE_PALETTE["text"])
        label = f"τ={_format_heatmap_tick(tau)}"
        ax_auroc.plot(tau_data["alpha"], tau_data["auroc"], color=color, linewidth=1.8, label=label)
        ax_auprc.plot(tau_data["alpha"], tau_data["auprc"], color=color, linewidth=1.8, label=label)

    ax_auroc.set_ylabel("AUROC")
    ax_auprc.set_ylabel("AUPRC")
    ax_auroc.set_xlabel("α")
    ax_auprc.set_xlabel("α")
    ax_auroc.set_ylim(0.84, 0.96)
    ax_auprc.set_ylim(0.60, 0.84)
    ax_auroc.set_xticks(range(0, 11, 2))
    ax_auprc.set_xticks(range(0, 11, 2))
    ax_auroc.legend(frameon=False, fontsize=8, loc="lower right")
    ax_auprc.legend(frameon=False, fontsize=8, loc="lower right")
    for axis in (ax_auroc, ax_auprc):
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)


def _plot_alpha_abstention_tradeoff_panel(
    ax: plt.Axes,
    *,
    data: pd.DataFrame,
    tau_values: tuple[float, ...],
    colors: dict[float, str],
    highlight_points: tuple[tuple[float, float], ...],
) -> None:
    for tau in tau_values:
        tau_data = data.loc[data["tau"] == tau].sort_values("alpha")
        if tau == 9.0:
            tau_data = tau_data.loc[~tau_data["alpha"].isin([1.0, 2.0])]
        ax.plot(
            tau_data["shared_false_abstention_rate"],
            tau_data["absent_abstention_rate"],
            color=colors.get(tau, FIGURE_PALETTE["text"]),
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )
        sizes = 22 + 4.0 * tau_data["alpha"].to_numpy(dtype=float)
        ax.scatter(
            tau_data["shared_false_abstention_rate"],
            tau_data["absent_abstention_rate"],
            s=sizes,
            color=colors.get(tau, FIGURE_PALETTE["text"]),
            alpha=0.78,
            linewidth=0.5,
            edgecolor=FIGURE_PALETTE["white"],
            label=f"τ={_format_heatmap_tick(tau)}",
            zorder=2,
        )
    _highlight_alpha_points(
        ax,
        data=data,
        x_col="shared_false_abstention_rate",
        y_col="absent_abstention_rate",
        colors=colors,
        highlight_points=highlight_points,
    )
    ax.set_xlabel("Shared-cell false abstention")
    ax.set_ylabel("Absent-cell abstention")
    ax.set_xlim(left=0.035)
    ax.set_ylim(0.30, 0.90)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _plot_alpha_selective_transfer_panel(
    ax: plt.Axes,
    *,
    data: pd.DataFrame,
    tau_values: tuple[float, ...],
    colors: dict[float, str],
    highlight_points: tuple[tuple[float, float], ...],
) -> None:
    for tau in tau_values:
        tau_data = data.loc[data["tau"] == tau].sort_values("alpha")
        ax.plot(
            tau_data["coverage"],
            tau_data["post_abstention_macro_f1"],
            color=colors.get(tau, FIGURE_PALETTE["text"]),
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )
        sizes = 22 + 4.0 * tau_data["alpha"].to_numpy(dtype=float)
        ax.scatter(
            tau_data["coverage"],
            tau_data["post_abstention_macro_f1"],
            s=sizes,
            color=colors.get(tau, FIGURE_PALETTE["text"]),
            alpha=0.78,
            linewidth=0.5,
            edgecolor=FIGURE_PALETTE["white"],
            label=f"τ={_format_heatmap_tick(tau)}",
            zorder=2,
        )
    _highlight_alpha_points(
        ax,
        data=data,
        x_col="coverage",
        y_col="post_abstention_macro_f1",
        colors=colors,
        highlight_points=highlight_points,
    )
    ax.set_xlabel("Shared-cell coverage")
    ax.set_ylabel("Post-abstention macro-F1")
    ax.set_xlim(0.84, 0.98)
    ax.set_ylim(0.88, 0.98)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _highlight_alpha_points(
    ax: plt.Axes,
    *,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    colors: dict[float, str],
    highlight_points: tuple[tuple[float, float], ...],
) -> None:
    offsets = {
        (1.0, 0.0): (5, -12),
        (1.0, 2.0): (5, 5),
        (5.0, 5.0): (5, 5),
        (9.0, 0.0): (5, -13),
        (9.0, 10.0): (-18, -12),
    }
    for tau, alpha in highlight_points:
        row = _point_row(data, tau=tau, alpha=alpha)
        if row is None:
            continue
        x_val = float(row[x_col])
        y_val = float(row[y_col])
        color = colors.get(tau, FIGURE_PALETTE["text"])
        ax.scatter(
            [x_val],
            [y_val],
            s=78,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=5,
        )
        ax.annotate(
            f"({_format_heatmap_tick(tau)},{_format_heatmap_tick(alpha)})",
            (x_val, y_val),
            textcoords="offset points",
            xytext=offsets.get((tau, alpha), (5, 5)),
            fontsize=7,
            color=color,
        )


def _point_row(data: pd.DataFrame, *, tau: float, alpha: float) -> pd.Series | None:
    row = data.loc[
        (np.isclose(data["tau"].astype(float), tau))
        & (np.isclose(data["alpha"].astype(float), alpha))
    ]
    if row.empty:
        return None
    return row.iloc[0]


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _format_heatmap_tick(value: object) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric == int(numeric) else str(numeric)


def write_coreot_full_tau_range_heatmaps(
    *,
    detection_summary: pd.DataFrame,
    shared_summary: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "heatmaps.pdf",
    title_suffix: str = "",
) -> dict[str, Path]:
    """Write one multi-page PDF of tau_min x tau_max heatmaps for coreot_full."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    det_wide = detection_summary.pivot_table(
        index=["held_out_label", "method", "tau_min", "tau_max", "alpha"],
        columns="quantity",
        values="mean",
    ).reset_index()

    shared_wide = shared_summary.pivot_table(
        index=["held_out_label", "method", "tau_min", "tau_max", "alpha"],
        columns="quantity",
        values="mean",
    ).reset_index()

    cols = ["held_out_label", "method", "tau_min", "tau_max", "alpha"]
    merged = det_wide.merge(shared_wide, on=cols, how="outer")

    held_out_labels = [
        label
        for label in ("CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2", "overall")
        if label in set(merged["held_out_label"].dropna().unique())
    ]
    metrics = [q for q in HEATMAP_QUANTITIES if q in merged.columns]
    pdf_path = out / filename
    with PdfPages(pdf_path) as pdf:
        for qty in metrics:
            vmin = float(merged[qty].min()) if not merged[qty].isna().all() else 0.0
            vmax = float(merged[qty].max()) if not merged[qty].isna().all() else 1.0
            figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), squeeze=False)
            cmap = (
                "RdYlBu_r"
                if qty in ("shared_false_abstention_rate", "absent_abstention_rate")
                else "RdYlGn"
            )
            for axis, label in zip(axes.ravel(), held_out_labels, strict=False):
                label_data = merged.loc[merged["held_out_label"] == label]
                if label_data.empty:
                    axis.axis("off")
                    continue
                heatmap_data = label_data.pivot_table(
                    index="tau_min",
                    columns="tau_max",
                    values=qty,
                    aggfunc="mean",
                )
                heatmap_data = heatmap_data.reindex(
                    index=sorted(heatmap_data.index),
                    columns=sorted(heatmap_data.columns),
                )
                image = axis.imshow(
                    heatmap_data.values,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                )
                axis.set_xticks(range(len(heatmap_data.columns)))
                axis.set_xticklabels(
                    [str(int(value)) if value == int(value) else str(value) for value in heatmap_data.columns],
                    fontsize=8,
                    rotation=45,
                )
                axis.set_yticks(range(len(heatmap_data.index)))
                axis.set_yticklabels(
                    [str(int(value)) if value == int(value) else str(value) for value in heatmap_data.index],
                    fontsize=8,
                )
                axis.set_xlabel("τ_max", fontsize=9)
                axis.set_ylabel("τ_min", fontsize=9)
                axis.set_title(
                    "Mean across three held-out labels"
                    if label == "overall"
                    else _short_label(label),
                    fontsize=10,
                )
                figure.colorbar(image, ax=axis, shrink=0.82)
            for axis in axes.ravel()[len(held_out_labels):]:
                axis.axis("off")
            title = f"coreot_full τ_min×τ_max heatmaps — {qty}"
            if title_suffix:
                title = f"{title} ({title_suffix})"
            figure.suptitle(title, fontsize=12)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    key = "heatmaps_pdf" if filename == "heatmaps.pdf" else f"{Path(filename).stem}_pdf"
    paths[key] = pdf_path
    return paths
