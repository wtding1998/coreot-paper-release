"""Generate the manuscript-facing HIHA query-penalty sensitivity figure."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

import anndata
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from coreot.config.load import load_yaml


ENDPOINTS = ("HLA-DRhi cDC2", "ISG+ cDC2")
SEEDS = (1, 2, 3, 4, 5)
HLA_TAU_MIN = (1.0, 1.5, 2.0)
HLA_TAU_MAX = (2.0, 2.5, 3.0)
ISG_TAU_MIN = (0.375, 0.5, 0.625)
ISG_TAU_MAX = (0.625, 0.75, 1.0)
PARAMETERS = {
    "HLA-DRhi cDC2": {
        "tau_min": HLA_TAU_MIN,
        "tau_max": HLA_TAU_MAX,
        "tau_reference": 2.0,
        "alpha": 2.0,
    },
    "ISG+ cDC2": {
        "tau_min": ISG_TAU_MIN,
        "tau_max": ISG_TAU_MAX,
        "tau_reference": 1.0,
        "alpha": 0.25,
    },
}
REPORTED_OPERATING_POINTS = {
    "HLA-DRhi cDC2": (2.5, 3.0),
    "ISG+ cDC2": (0.5, 0.625),
}
RASTER_DPI = 350


class HIHAParameterSensitivityError(RuntimeError):
    """Raised when source artifacts do not satisfy the figure contract."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise HIHAParameterSensitivityError(
            f"{path} is missing required columns: {sorted(missing)}"
        )


def _in_grid(values: pd.Series, expected: tuple[float, ...]) -> pd.Series:
    numeric = values.to_numpy(dtype=float)
    return pd.Series(
        np.logical_or.reduce([np.isclose(numeric, value) for value in expected]),
        index=values.index,
    )


def _validate_keys(frame: pd.DataFrame, *, label: str) -> None:
    expected = {
        (endpoint, seed, tau_min, tau_max)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for tau_min in PARAMETERS[endpoint]["tau_min"]
        for tau_max in PARAMETERS[endpoint]["tau_max"]
    }
    observed = set(
        frame[["held_out_label", "seed", "tau_min", "tau_max"]].itertuples(
            index=False,
            name=None,
        )
    )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise HIHAParameterSensitivityError(
            f"{label} has incorrect parameter-grid keys; missing={missing[:5]}, extra={extra[:5]}."
        )


def build_hla_within_cdc2_detection(
    *,
    project_root: Path,
    runs_root: Path,
    grid_dir: Path,
    transfer_path: Path,
) -> pd.DataFrame:
    """Recompute HLA split-level AP on the prespecified within-cDC2 cohort."""
    transfer = pd.read_csv(transfer_path)
    required = (
        "run_id",
        "held_out_label",
        "seed",
        "method",
        "tau_min",
        "tau_max",
        "alpha",
    )
    _require_columns(transfer, required, transfer_path)
    transfer = transfer.loc[
        transfer["held_out_label"].eq("HLA-DRhi cDC2")
        & transfer["method"].eq("coreot_full")
        & _in_grid(transfer["tau_min"], HLA_TAU_MIN)
        & _in_grid(transfer["tau_max"], HLA_TAU_MAX),
        list(required),
    ].copy()
    if transfer.duplicated(["run_id"]).any():
        raise HIHAParameterSensitivityError(
            f"Duplicate HLA run IDs in {transfer_path}."
        )

    configured_inputs: set[Path] = set()
    for run_id in transfer["run_id"].astype(str):
        config_path = grid_dir / run_id / "raw_import.yaml"
        config = load_yaml(config_path)
        input_path = Path(config["input"]["path"])
        configured_inputs.add(
            input_path if input_path.is_absolute() else project_root / input_path
        )
    if len(configured_inputs) != 1:
        raise HIHAParameterSensitivityError(
            "The HLA sensitivity grid must use one common input dataset; "
            f"found {sorted(str(path) for path in configured_inputs)}."
        )
    input_path = configured_inputs.pop()
    adata = anndata.read_h5ad(input_path, backed="r")
    try:
        metadata = adata.obs[["AIFI_L2"]].copy()
    finally:
        adata.file.close()
    metadata["cell_id"] = metadata.index.astype(str)
    metadata = metadata.reset_index(drop=True)
    input_relative = input_path.relative_to(project_root)
    input_hash = _sha256_file(input_path)

    rows: list[dict[str, object]] = []
    for parameter_row in transfer.itertuples(index=False):
        run_root = runs_root / str(parameter_row.run_id)
        score_path = (
            run_root
            / "scoring/incomplete_reference/hiha_harmony30_k100/cell_scores.parquet"
        )
        truth_path = (
            run_root
            / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
        )
        scores = pd.read_parquet(score_path)
        _require_columns(scores, ("cell_id", "method", "u"), score_path)
        scores = scores.loc[
            scores["method"].eq("coreot_full"), ["cell_id", "u"]
        ].copy()
        if scores.empty or scores["cell_id"].duplicated().any():
            raise HIHAParameterSensitivityError(
                f"Invalid CoRe-OT cell scores in {score_path}."
            )
        truth = pd.read_csv(truth_path)
        _require_columns(truth, ("cell_id", "is_absent_state"), truth_path)
        if truth["cell_id"].duplicated().any():
            raise HIHAParameterSensitivityError(
                f"Duplicate query cell IDs in {truth_path}."
            )
        joined = scores.merge(truth, on="cell_id", validate="one_to_one").merge(
            metadata, on="cell_id", validate="one_to_one"
        )
        incomplete_join = joined[["is_absent_state", "AIFI_L2"]].isna().any().any()
        if len(joined) != len(scores) or incomplete_join:
            raise HIHAParameterSensitivityError(
                f"Incomplete score, truth, or AIFI_L2 join for {parameter_row.run_id}."
            )
        within_cdc2 = joined.loc[joined["AIFI_L2"].astype(str).eq("cDC2")].copy()
        finite = np.isfinite(pd.to_numeric(within_cdc2["u"], errors="coerce"))
        within_cdc2 = within_cdc2.loc[finite]
        positive = within_cdc2["is_absent_state"].astype(bool)
        n_positive = int(positive.sum())
        n_negative = int((~positive).sum())
        if not n_positive or not n_negative:
            raise HIHAParameterSensitivityError(
                f"Degenerate within-cDC2 cohort for {parameter_row.run_id}."
            )
        rows.append(
            {
                **{column: getattr(parameter_row, column) for column in required},
                "evaluation_scope": "local_within_broad_state",
                "n_query": len(within_cdc2),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "auprc": average_precision_score(positive, within_cdc2["u"]),
                "score_path": str(score_path.relative_to(project_root)),
                "score_sha256": _sha256_file(score_path),
                "truth_path": str(truth_path.relative_to(project_root)),
                "truth_sha256": _sha256_file(truth_path),
                "input_path": str(input_relative),
                "input_sha256": input_hash,
            }
        )
    return pd.DataFrame(rows).sort_values(["tau_min", "tau_max", "seed"]).reset_index(
        drop=True
    )


def collect_hiha_parameter_sensitivity(
    *,
    hla_within_cdc2_detection_path: Path,
    hla_transfer_path: Path,
    isg_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect split-level values and donor-equal summaries for the figure."""
    hla_detection = pd.read_csv(hla_within_cdc2_detection_path)
    _require_columns(
        hla_detection,
        (
            "run_id",
            "held_out_label",
            "seed",
            "method",
            "auprc",
            "tau_min",
            "tau_max",
            "alpha",
            "evaluation_scope",
        ),
        hla_within_cdc2_detection_path,
    )
    hla_transfer = pd.read_csv(hla_transfer_path)
    _require_columns(
        hla_transfer,
        (
            "run_id",
            "held_out_label",
            "seed",
            "method",
            "forced_macro_f1",
            "tau_min",
            "tau_max",
            "alpha",
        ),
        hla_transfer_path,
    )
    merge_keys = (
        "run_id",
        "held_out_label",
        "seed",
        "method",
        "tau_min",
        "tau_max",
        "alpha",
    )
    hla = hla_detection.loc[
        hla_detection["held_out_label"].eq("HLA-DRhi cDC2")
        & hla_detection["method"].eq("coreot_full")
        & hla_detection["evaluation_scope"].eq("local_within_broad_state"),
        [*merge_keys, "auprc"],
    ].merge(
        hla_transfer.loc[
            hla_transfer["held_out_label"].eq("HLA-DRhi cDC2")
            & hla_transfer["method"].eq("coreot_full"),
            [*merge_keys, "forced_macro_f1"],
        ],
        on=list(merge_keys),
        how="inner",
        validate="one_to_one",
    )
    hla = hla.loc[
        _in_grid(hla["tau_min"], HLA_TAU_MIN) & _in_grid(hla["tau_max"], HLA_TAU_MAX)
    ].copy()
    hla["tau_reference"] = 2.0
    hla["source_analysis"] = "full_tau_target2_alpha2_hla"

    isg = pd.read_csv(isg_path)
    _require_columns(
        isg,
        (
            "run_id",
            "seed",
            "tau_min",
            "tau_max",
            "tau_reference",
            "alpha",
            "coreot_auprc",
            "coreot_forced_macro_f1",
        ),
        isg_path,
    )
    isg = isg.loc[
        _in_grid(isg["tau_min"], ISG_TAU_MIN) & _in_grid(isg["tau_max"], ISG_TAU_MAX)
    ].copy()
    isg["held_out_label"] = "ISG+ cDC2"
    isg["method"] = "coreot_full"
    isg["auprc"] = isg["coreot_auprc"]
    isg["forced_macro_f1"] = isg["coreot_forced_macro_f1"]
    isg["source_analysis"] = "isg_tau_range_target1_alpha025_discovery"

    columns = [
        "run_id",
        "held_out_label",
        "seed",
        "method",
        "tau_min",
        "tau_max",
        "tau_reference",
        "alpha",
        "auprc",
        "forced_macro_f1",
        "source_analysis",
    ]
    by_split = pd.concat([hla[columns], isg[columns]], ignore_index=True)
    numeric_columns = (
        "tau_min",
        "tau_max",
        "tau_reference",
        "alpha",
        "auprc",
        "forced_macro_f1",
    )
    by_split[list(numeric_columns)] = by_split[list(numeric_columns)].astype(float)
    by_split["seed"] = by_split["seed"].astype(int)
    _validate_keys(by_split, label="HIHA parameter sensitivity")
    if not by_split.groupby("held_out_label")["alpha"].nunique().eq(1).all():
        raise HIHAParameterSensitivityError("Alpha must be fixed within each endpoint.")
    if not by_split.groupby("held_out_label")["tau_reference"].nunique().eq(1).all():
        raise HIHAParameterSensitivityError(
            "The reference penalty must be fixed within each endpoint."
        )

    summary = (
        by_split.groupby(
            [
                "held_out_label",
                "tau_min",
                "tau_max",
                "tau_reference",
                "alpha",
                "source_analysis",
            ],
            sort=True,
        )
        .agg(
            within_cdc2_ap_mean=("auprc", "mean"),
            within_cdc2_ap_std=("auprc", lambda values: values.std(ddof=1)),
            represented_state_forced_macro_f1_mean=("forced_macro_f1", "mean"),
            represented_state_forced_macro_f1_std=(
                "forced_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "nunique"),
        )
        .reset_index()
    )
    if not summary["n_splits"].eq(len(SEEDS)).all():
        raise HIHAParameterSensitivityError(
            "Every sensitivity cell must contain five donor splits."
        )
    return by_split.sort_values(["held_out_label", "tau_min", "tau_max", "seed"]).reset_index(
        drop=True
    ), summary


def _matrix(
    summary: pd.DataFrame,
    endpoint: str,
    column: str,
) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    tau_min_values = PARAMETERS[endpoint]["tau_min"]
    tau_max_values = PARAMETERS[endpoint]["tau_max"]
    subset = summary.loc[summary["held_out_label"].eq(endpoint)]
    matrix = np.full((len(tau_min_values), len(tau_max_values)), np.nan)
    for row in subset.itertuples(index=False):
        row_index = tau_min_values.index(float(row.tau_min))
        column_index = tau_max_values.index(float(row.tau_max))
        matrix[row_index, column_index] = float(getattr(row, column))
    if np.isnan(matrix).any():
        raise HIHAParameterSensitivityError(f"The {column} heatmap is incomplete for {endpoint}.")
    return matrix, tau_min_values, tau_max_values


def _save_figure(fig: plt.Figure, outputs: dict[str, Path]) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = RASTER_DPI
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            content = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                encoding="utf-8",
            )


def render_hiha_parameter_sensitivity(
    summary: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    """Render the two-endpoint, two-metric sensitivity heatmaps."""
    fig = plt.figure(figsize=(7.2, 5.9))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.95)
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1, 0.18, 1, 0.065),
        wspace=0.08,
        hspace=0.55,
    )
    axes = [
        [fig.add_subplot(grid[row, 2 * endpoint_index]) for endpoint_index in range(2)]
        for row in range(2)
    ]
    colorbar_axes = [fig.add_subplot(grid[row, 3]) for row in range(2)]
    metric_rows = (
        ("within_cdc2_ap_mean", "AP", "Mean within-cDC2 AP"),
        (
            "represented_state_forced_macro_f1_mean",
            "forced macro-F1",
            "Mean represented-state forced macro-F1",
        ),
    )
    for metric_index, (column, metric_label, colorbar_label) in enumerate(metric_rows):
        metric_values = summary[column].to_numpy(dtype=float)
        vmin = float(metric_values.min())
        vmax = float(metric_values.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0e-6
        image = None
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            axis = axes[metric_index][endpoint_index]
            matrix, tau_min_values, tau_max_values = _matrix(summary, endpoint, column)
            image = axis.imshow(
                matrix,
                origin="lower",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    scaled = (value - vmin) / (vmax - vmin)
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=("white" if scaled < 0.22 or scaled > 0.82 else "#1F1F1F"),
                    )
            reported_tau_min, reported_tau_max = REPORTED_OPERATING_POINTS[endpoint]
            if (
                reported_tau_min in tau_min_values
                and reported_tau_max in tau_max_values
            ):
                selected_row = tau_min_values.index(reported_tau_min)
                selected_column = tau_max_values.index(reported_tau_max)
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
            axis.set_xticks(
                range(len(tau_max_values)),
                [f"{value:g}" for value in tau_max_values],
            )
            axis.set_yticks(
                range(len(tau_min_values)),
                [f"{value:g}" for value in tau_min_values],
            )
            axis.set_xlabel(r"$\tau_{\max}$")
            if endpoint_index == 0:
                axis.set_ylabel(r"$\tau_{\min}$")
            panel_index = metric_index * len(ENDPOINTS) + endpoint_index
            axis.set_title(
                f"{chr(ord('A') + panel_index)}. {endpoint} {metric_label}",
                loc="left",
                fontsize=8.5,
                fontweight="bold",
            )
            axis.tick_params(labelsize=6.5)
        if image is not None:
            colorbar = fig.colorbar(image, cax=colorbar_axes[metric_index])
            colorbar.set_label(colorbar_label, fontsize=6.5)
            colorbar.ax.tick_params(labelsize=6)
    _save_figure(fig, outputs)
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hiha_parameter_sensitivity(
    *,
    project_root: Path = Path("."),
) -> dict[str, Path]:
    """Write source tables, figure formats, alt text, and a provenance manifest."""
    project_root = project_root.resolve()
    hla_root = project_root / "results/HIHA_DC/sensitivity/full_tau_target2_alpha2_hla/tables"
    isg_root = (
        project_root / "results/HIHA_DC/sensitivity/isg_tau_range_target1_alpha025_discovery/tables"
    )
    hla_transfer_path = hla_root / "shared_label_transfer_by_run.csv"
    hla_within_cdc2_detection_path = hla_root / "detection_within_cdc2_by_run.csv"
    hla_within_cdc2_detection = build_hla_within_cdc2_detection(
        project_root=project_root,
        runs_root=project_root / "runs",
        grid_dir=(
            project_root
            / "experiments/missing_celltype/generated_configs/"
            "hiha_dc_coreot_full_tau_target2_alpha2_hla_grid"
        ),
        transfer_path=hla_transfer_path,
    )
    hla_within_cdc2_detection.to_csv(hla_within_cdc2_detection_path, index=False)
    source_paths = {
        "hla_within_cdc2_detection_by_run": hla_within_cdc2_detection_path,
        "hla_transfer_by_run": hla_transfer_path,
        "isg_discovery_by_split": isg_root / "discovery_by_split.csv",
    }
    by_split, summary = collect_hiha_parameter_sensitivity(
        hla_within_cdc2_detection_path=source_paths[
            "hla_within_cdc2_detection_by_run"
        ],
        hla_transfer_path=source_paths["hla_transfer_by_run"],
        isg_path=source_paths["isg_discovery_by_split"],
    )

    data_root = project_root / "results/HIHA_DC/figures/data"
    data_root.mkdir(parents=True, exist_ok=True)
    by_split_path = data_root / "hiha_parameter_sensitivity_by_split.csv"
    summary_path = data_root / "hiha_parameter_sensitivity_summary.csv"
    by_split.to_csv(by_split_path, index=False)
    summary.to_csv(summary_path, index=False)

    figures_root = project_root / "docs/figs"
    outputs = {
        suffix: figures_root / f"manuscript_fig_hiha_supp_parameter_sensitivity.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    render_hiha_parameter_sensitivity(summary, outputs)
    alt_path = figures_root / "manuscript_fig_hiha_supp_parameter_sensitivity_alt.txt"
    alt_path.write_text(
        "Four-panel HIHA query-penalty-bound sensitivity figure. Panels A and B "
        "show within-cDC2 average precision, and Panels C and D show "
        "represented-state forced macro-F1 for the HLA-DRhi cDC2 and ISG+ "
        "cDC2 endpoints, respectively. Stars mark the reported operating "
        "point in the ISG+ cDC2 panels; the reported HLA-DRhi cDC2 point is "
        "outside its displayed grid.\n",
        encoding="utf-8",
    )

    artifacts = {
        "by_split": by_split_path,
        "summary": summary_path,
        "figure_png": outputs["png"],
        "figure_pdf": outputs["pdf"],
        "figure_svg": outputs["svg"],
        "alt_text": alt_path,
    }
    manifest = {
        "analysis_id": "hiha_query_penalty_sensitivity_figure",
        "aggregation": "arithmetic mean across five fixed donor splits",
        "metrics": ["within-cDC2 AP", "represented-state forced macro-F1"],
        "parameter_grids": PARAMETERS,
        "reported_operating_points": REPORTED_OPERATING_POINTS,
        "sources": {
            name: {"path": str(path.relative_to(project_root)), "sha256": _sha256_file(path)}
            for name, path in source_paths.items()
        },
        "artifacts": {
            name: {"path": str(path.relative_to(project_root)), "sha256": _sha256_file(path)}
            for name, path in artifacts.items()
        },
        "interpretation": {
            "descriptive_parameter_sensitivity": True,
            "independent_robustness_validation": False,
            "hla_reported_operating_point_in_displayed_grid": False,
            "isg_grid_informed_reported_operating_point": True,
        },
    }
    manifest_path = (
        project_root / "results/HIHA_DC/figures/hiha_parameter_sensitivity_manifest.yaml"
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {**artifacts, "manifest": manifest_path}
