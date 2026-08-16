"""Generate the mouse-spleen query-penalty sensitivity figure."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


TAU_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
DISPLAY_TAU_MIN = (2.0, 3.0, 4.0)
DISPLAY_TAU_MAX = (4.0, 5.0, 6.0)
SELECTED_TAU_MIN = 3.0
SELECTED_TAU_MAX = 5.0
FIXED_TAU_REFERENCE = 8.0
FIXED_ALPHA = 40.0
RASTER_DPI = 350


class MouseSpleenParameterSensitivityError(RuntimeError):
    """Raised when the mouse-spleen parameter grid violates its contract."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise MouseSpleenParameterSensitivityError(
            f"{path} is missing required columns: {sorted(missing)}"
        )


def _as_bool(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise MouseSpleenParameterSensitivityError(
            f"{label} contains values other than true and false."
        )
    return normalized.eq("true")


def collect_mouse_spleen_parameter_sensitivity(source_path: Path) -> pd.DataFrame:
    """Collect the complete selected-alpha triangular parameter grid."""
    frame = pd.read_csv(source_path)
    required = (
        "run_id",
        "natural_endpoint",
        "condition",
        "method",
        "tau_min",
        "tau_max",
        "tau_target",
        "alpha",
        "converged",
        "evaluation_scope",
        "n_query",
        "n_positive",
        "auprc",
        "shared_forced_macro_f1",
        "is_canonical",
        "source_run_id",
    )
    _require_columns(frame, required, source_path)
    selected = frame.loc[
        frame["natural_endpoint"].eq("Proliferating")
        & frame["condition"].eq("natural_mismatch")
        & frame["method"].eq("coreot_full")
        & np.isclose(frame["tau_target"].astype(float), FIXED_TAU_REFERENCE)
        & np.isclose(frame["alpha"].astype(float), FIXED_ALPHA),
        list(required),
    ].copy()
    expected = {
        (tau_min, tau_max)
        for tau_min in TAU_VALUES
        for tau_max in TAU_VALUES
        if tau_min <= tau_max
    }
    observed = set(
        selected[["tau_min", "tau_max"]].astype(float).itertuples(index=False, name=None)
    )
    if observed != expected or selected.duplicated(["tau_min", "tau_max"]).any():
        raise MouseSpleenParameterSensitivityError(
            "The mouse-spleen sensitivity source does not contain the exact "
            f"triangular grid; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}."
        )
    selected["converged"] = _as_bool(selected["converged"], label="converged")
    if not selected["converged"].all():
        failed = selected.loc[~selected["converged"], "run_id"].astype(str).tolist()
        raise MouseSpleenParameterSensitivityError(
            f"The manuscript figure does not admit nonconverged cells: {failed}."
        )
    selected["is_canonical"] = _as_bool(
        selected["is_canonical"], label="is_canonical"
    )
    canonical = selected.loc[selected["is_canonical"]]
    if len(canonical) != 1 or not (
        np.isclose(float(canonical.iloc[0]["tau_min"]), SELECTED_TAU_MIN)
        and np.isclose(float(canonical.iloc[0]["tau_max"]), SELECTED_TAU_MAX)
    ):
        raise MouseSpleenParameterSensitivityError(
            "The grid must identify (tau_min, tau_max) = (3, 5) as its unique "
            "reported operating point."
        )
    metric_columns = ("auprc", "shared_forced_macro_f1")
    if not np.isfinite(selected[list(metric_columns)].to_numpy(dtype=float)).all():
        raise MouseSpleenParameterSensitivityError(
            "The mouse-spleen sensitivity metrics must be finite."
        )
    selected = selected.rename(
        columns={
            "auprc": "ap",
            "shared_forced_macro_f1": "represented_state_forced_macro_f1",
        }
    )
    return selected.sort_values(["tau_min", "tau_max"]).reset_index(drop=True)


def _matrix(frame: pd.DataFrame, column: str) -> np.ma.MaskedArray:
    displayed = frame.loc[
        frame["tau_min"].astype(float).isin(DISPLAY_TAU_MIN)
        & frame["tau_max"].astype(float).isin(DISPLAY_TAU_MAX)
    ]
    values = np.full((len(DISPLAY_TAU_MIN), len(DISPLAY_TAU_MAX)), np.nan)
    for row in displayed.itertuples(index=False):
        row_index = DISPLAY_TAU_MIN.index(float(row.tau_min))
        column_index = DISPLAY_TAU_MAX.index(float(row.tau_max))
        values[row_index, column_index] = float(getattr(row, column))
    if np.isnan(values).any():
        raise MouseSpleenParameterSensitivityError(
            f"The displayed local {column} grid is incomplete."
        )
    return np.ma.masked_invalid(values)


def _save_figure(figure: plt.Figure, outputs: dict[str, Path]) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = RASTER_DPI
        figure.savefig(path, **kwargs)
        if suffix == "svg":
            content = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                encoding="utf-8",
            )


def render_mouse_spleen_parameter_sensitivity(
    frame: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    """Render AP and represented-state forced macro-F1 heatmaps."""
    figure = plt.figure(figsize=(8.4, 3.8))
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.91)
    grid = figure.add_gridspec(
        1,
        5,
        width_ratios=(1, 0.045, 0.22, 1, 0.045),
        wspace=0.10,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 3])]
    colorbar_axes = [figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[0, 4])]
    metrics = (
        ("ap", "A. AP", "Proliferating-cell AP"),
        (
            "represented_state_forced_macro_f1",
            "B. Forced macro-F1",
            "Represented-state forced macro-F1",
        ),
    )
    x_labels = [f"{value:g}" for value in DISPLAY_TAU_MAX]
    y_labels = [f"{value:g}" for value in DISPLAY_TAU_MIN]
    selected_row = DISPLAY_TAU_MIN.index(SELECTED_TAU_MIN)
    selected_column = DISPLAY_TAU_MAX.index(SELECTED_TAU_MAX)
    for axis, colorbar_axis, (column, title, colorbar_label) in zip(
        axes, colorbar_axes, metrics, strict=True
    ):
        matrix = _matrix(frame, column)
        finite = matrix.compressed()
        vmin, vmax = float(finite.min()), float(finite.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0e-6
        cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#E3E3E3")
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        for row_index in range(len(DISPLAY_TAU_MIN)):
            for column_index in range(len(DISPLAY_TAU_MAX)):
                value = matrix[row_index, column_index]
                if np.ma.is_masked(value):
                    continue
                numeric = float(value)
                scaled = (numeric - vmin) / (vmax - vmin)
                axis.text(
                    column_index,
                    row_index,
                    f"{numeric:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.1,
                    color=("white" if scaled < 0.22 or scaled > 0.82 else "#1F1F1F"),
                )
        axis.add_patch(
            Rectangle(
                (selected_column - 0.48, selected_row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="white",
                linewidth=1.8,
            )
        )
        axis.scatter(
            selected_column - 0.32,
            selected_row + 0.32,
            marker="*",
            s=38,
            color="white",
            edgecolor="#202020",
            linewidth=0.4,
        )
        axis.set_xticks(range(len(DISPLAY_TAU_MAX)), x_labels)
        axis.set_yticks(range(len(DISPLAY_TAU_MIN)), y_labels)
        axis.set_xlabel(r"$\tau_{\max}$")
        axis.set_ylabel(r"$\tau_{\min}$")
        axis.set_title(title, loc="left", fontsize=9.0, fontweight="bold")
        axis.tick_params(labelsize=6.8)
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(colorbar_label, fontsize=6.8)
        colorbar.ax.tick_params(labelsize=6.2)
    _save_figure(figure, outputs)
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_mouse_spleen_parameter_sensitivity(
    *, project_root: Path = Path(".")
) -> dict[str, Path]:
    """Write source data, figure formats, alt text, and provenance."""
    project_root = project_root.resolve()
    grid_root = (
        project_root
        / "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "figure4_full_tau_alpha_40_tau_target_8"
    )
    source_paths = {
        "grid_table": grid_root / "tables/metrics_by_grid.csv",
        "grid_manifest": grid_root / "manifest.yaml",
        "run_manifest": grid_root / "run_manifest.yaml",
    }
    frame = collect_mouse_spleen_parameter_sensitivity(source_paths["grid_table"])
    output_root = project_root / "results/mouse_spleen_core_ot/manuscript/parameter_sensitivity"
    output_root.mkdir(parents=True, exist_ok=True)
    source_data_path = output_root / "mouse_spleen_parameter_sensitivity.csv"
    frame.to_csv(source_data_path, index=False)

    figure_root = project_root / "docs/figs"
    outputs = {
        suffix: figure_root / f"manuscript_fig_mouse_spleen_supp_parameter_sensitivity.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    render_mouse_spleen_parameter_sensitivity(frame, outputs)
    alt_path = figure_root / "manuscript_fig_mouse_spleen_supp_parameter_sensitivity_alt.txt"
    alt_path.write_text(
        "Two-panel mouse-spleen query-penalty-bound sensitivity figure. Panel A "
        "shows Proliferating-cell average precision, and Panel B shows "
        "represented-state forced macro-F1 over a local three-by-three grid "
        "centered on the reported operating point.\n",
        encoding="utf-8",
    )
    artifacts = {
        "source_data": source_data_path,
        "figure_png": outputs["png"],
        "figure_pdf": outputs["pdf"],
        "figure_svg": outputs["svg"],
        "alt_text": alt_path,
    }
    manifest = {
        "analysis_id": "mouse_spleen_query_penalty_sensitivity_figure",
        "endpoint": "Proliferating",
        "aggregation": "single-dataset point estimates",
        "metrics": ["AP", "represented-state forced macro-F1"],
        "parameter_grid": {
            "source_tau_values": list(TAU_VALUES),
            "source_constraint": "tau_min <= tau_max",
            "display_tau_min": list(DISPLAY_TAU_MIN),
            "display_tau_max": list(DISPLAY_TAU_MAX),
            "tau_reference": FIXED_TAU_REFERENCE,
            "alpha": FIXED_ALPHA,
            "reported_operating_point": [SELECTED_TAU_MIN, SELECTED_TAU_MAX],
        },
        "sources": {
            name: {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256_file(path),
            }
            for name, path in source_paths.items()
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256_file(path),
            }
            for name, path in artifacts.items()
        },
        "interpretation": {
            "descriptive_parameter_sensitivity": True,
            "independent_robustness_validation": False,
            "single_biological_specimen": True,
            "reported_operating_point_in_grid": True,
        },
    }
    manifest_path = output_root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {**artifacts, "manifest": manifest_path}
