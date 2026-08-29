"""Generate the manuscript-facing HIHA query-penalty sensitivity figure."""

from __future__ import annotations

from hashlib import sha256
import json
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

from coreot.config.load import load_yaml


ENDPOINTS = ("HLA-DRhi cDC2", "ISG+ cDC2")
SEEDS = (1, 2, 3, 4, 5)
HLA_TAU_MIN = (2.0, 2.5, 3.0)
HLA_TAU_MAX = (2.5, 3.0, 3.5)
ISG_TAU_MIN = (0.375, 0.5, 0.625)
ISG_TAU_MAX = (0.625, 0.75, 1.0)
LEGACY_HLA_TAU_MIN = (1.0, 1.5, 2.0)
LEGACY_HLA_TAU_MAX = (2.0, 2.5, 3.0)
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
LEGACY_PARAMETERS = {
    **PARAMETERS,
    "HLA-DRhi cDC2": {
        "tau_min": LEGACY_HLA_TAU_MIN,
        "tau_max": LEGACY_HLA_TAU_MAX,
        "tau_reference": 2.0,
        "alpha": 2.0,
    },
}
SELECTED_SETTINGS = {
    "HLA-DRhi cDC2": (2.5, 3.0),
    "ISG+ cDC2": (0.5, 0.625),
}
H1_SPLIT_METRICS_RELATIVE = Path(
    "results/phase2_review/H1/2026-08-21-execution-v4/tables/h1_split_metrics.csv"
)
H1_VERIFICATION_RELATIVE = Path(
    "results/phase2_review/H1/2026-08-21-execution-v4/audit/h1_verification_report.json"
)
H1_CHECKSUMS_RELATIVE = Path(
    "results/phase2_review/H1/2026-08-21-execution-v4/manifests/checksums.sha256"
)
RASTER_DPI = 350
SVG_HASH_SALT = "coreot-hiha-parameter-sensitivity"


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


def _validate_h1_checksums(
    checksums_path: Path,
    *,
    split_metrics_path: Path,
    verification_path: Path,
) -> None:
    declared: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        declared[relative.strip()] = digest
    expected = {
        "tables/h1_split_metrics.csv": split_metrics_path,
        "audit/h1_verification_report.json": verification_path,
    }
    mismatches = {
        relative: (declared.get(relative), _sha256_file(path))
        for relative, path in expected.items()
        if declared.get(relative) != _sha256_file(path)
    }
    if mismatches:
        raise HIHAParameterSensitivityError(
            f"The H1 sources disagree with the immutable-root checksums: {mismatches}."
        )


def _validate_keys(
    frame: pd.DataFrame,
    *,
    label: str,
    parameters: dict[str, dict[str, object]],
    ordered_bound_endpoints: tuple[str, ...] = (),
) -> None:
    expected = {
        (endpoint, seed, tau_min, tau_max)
        for endpoint in ENDPOINTS
        for seed in SEEDS
        for tau_min in parameters[endpoint]["tau_min"]
        for tau_max in parameters[endpoint]["tau_max"]
        if endpoint not in ordered_bound_endpoints or tau_min <= tau_max
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
        & _in_grid(transfer["tau_min"], LEGACY_HLA_TAU_MIN)
        & _in_grid(transfer["tau_max"], LEGACY_HLA_TAU_MAX),
        list(required),
    ].copy()
    if transfer.duplicated(["run_id"]).any():
        raise HIHAParameterSensitivityError(f"Duplicate HLA run IDs in {transfer_path}.")

    configured_inputs: set[Path] = set()
    for run_id in transfer["run_id"].astype(str):
        config_path = grid_dir / run_id / "raw_import.yaml"
        config = load_yaml(config_path)
        input_path = Path(config["input"]["path"])
        configured_inputs.add(input_path if input_path.is_absolute() else project_root / input_path)
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
            run_root / "scoring/incomplete_reference/hiha_harmony30_k100/cell_scores.parquet"
        )
        truth_path = run_root / "benchmark/incomplete_reference/evaluation_truth/query_truth.csv"
        scores = pd.read_parquet(score_path)
        _require_columns(scores, ("cell_id", "method", "u"), score_path)
        scores = scores.loc[scores["method"].eq("coreot_full"), ["cell_id", "u"]].copy()
        if scores.empty or scores["cell_id"].duplicated().any():
            raise HIHAParameterSensitivityError(f"Invalid CoRe-OT cell scores in {score_path}.")
        truth = pd.read_csv(truth_path)
        _require_columns(truth, ("cell_id", "is_absent_state"), truth_path)
        if truth["cell_id"].duplicated().any():
            raise HIHAParameterSensitivityError(f"Duplicate query cell IDs in {truth_path}.")
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
    return pd.DataFrame(rows).sort_values(["tau_min", "tau_max", "seed"]).reset_index(drop=True)


def _load_isg_sensitivity(isg_path: Path) -> pd.DataFrame:
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
    return isg


def _summarize_sensitivity(
    hla: pd.DataFrame,
    isg: pd.DataFrame,
    *,
    parameters: dict[str, dict[str, object]],
    ordered_bound_endpoints: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    key_columns = ["held_out_label", "seed", "tau_min", "tau_max"]
    if by_split.duplicated(key_columns).any():
        raise HIHAParameterSensitivityError(
            "HIHA parameter sensitivity contains duplicate endpoint/split/grid keys."
        )
    _validate_keys(
        by_split,
        label="HIHA parameter sensitivity",
        parameters=parameters,
        ordered_bound_endpoints=ordered_bound_endpoints,
    )
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
    by_split = by_split.sort_values(["held_out_label", "tau_min", "tau_max", "seed"]).reset_index(
        drop=True
    )
    return by_split, summary


def collect_hiha_parameter_sensitivity(
    *,
    hla_within_cdc2_detection_path: Path,
    hla_transfer_path: Path,
    isg_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect the historical Figure S4 sources retained in release packages."""
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
        _in_grid(hla["tau_min"], LEGACY_HLA_TAU_MIN) & _in_grid(hla["tau_max"], LEGACY_HLA_TAU_MAX)
    ].copy()
    hla["tau_reference"] = 2.0
    hla["source_analysis"] = "full_tau_target2_alpha2_hla"
    return _summarize_sensitivity(
        hla,
        _load_isg_sensitivity(isg_path),
        parameters=LEGACY_PARAMETERS,
    )


def collect_hiha_parameter_sensitivity_from_h1(
    *,
    hla_split_metrics_path: Path,
    hla_verification_path: Path,
    hla_checksums_path: Path,
    isg_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect the verified H1 grid and the retained ISG grid for Figure S4."""
    _validate_h1_checksums(
        hla_checksums_path,
        split_metrics_path=hla_split_metrics_path,
        verification_path=hla_verification_path,
    )
    verification = json.loads(hla_verification_path.read_text(encoding="utf-8"))
    required_verification = {
        "unit_id": "phase2_h1_hiha_hladrhi",
        "execution_version": "2026-08-21-execution-v4",
        "status": "verified",
        "passed": True,
        "expected_case_count": 40,
        "observed_case_count": 40,
        "valid_case_count": 40,
        "selection_performed": False,
        "selected_cell_unchanged": True,
    }
    mismatches = {
        key: (verification.get(key), expected)
        for key, expected in required_verification.items()
        if verification.get(key) != expected
    }
    selected_cell = verification.get("selected_cell", {})
    if mismatches or selected_cell != {"tau_max": 3.0, "tau_min": 2.5}:
        raise HIHAParameterSensitivityError(
            f"The H1 verification report does not authorize Figure S4 integration; "
            f"mismatches={mismatches}, selected_cell={selected_cell}."
        )

    source = pd.read_csv(hla_split_metrics_path)
    _require_columns(
        source,
        (
            "endpoint",
            "condition",
            "seed",
            "tau_min",
            "tau_max",
            "run_id",
            "is_selected_cell",
            "within_cdc2_ap",
            "represented_state_forced_macro_f1",
        ),
        hla_split_metrics_path,
    )
    hla = source.loc[
        source["endpoint"].eq("HLA-DRhi cDC2")
        & source["condition"].eq("incomplete_reference")
        & _in_grid(source["tau_min"], HLA_TAU_MIN)
        & _in_grid(source["tau_max"], HLA_TAU_MAX)
        & source["tau_min"].le(source["tau_max"])
    ].copy()
    selected_flag = hla["is_selected_cell"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    expected_selected = hla["tau_min"].eq(2.5) & hla["tau_max"].eq(3.0)
    if selected_flag.isna().any() or not selected_flag.eq(expected_selected).all():
        raise HIHAParameterSensitivityError(
            "The H1 selected-cell flags disagree with the unchanged selected setting."
        )
    hla["held_out_label"] = hla["endpoint"]
    hla["method"] = "coreot_full"
    hla["tau_reference"] = 2.0
    hla["alpha"] = 2.0
    hla["auprc"] = hla["within_cdc2_ap"]
    hla["forced_macro_f1"] = hla["represented_state_forced_macro_f1"]
    hla["source_analysis"] = "phase2_h1_hiha_hladrhi_2026-08-21-execution-v4"
    return _summarize_sensitivity(
        hla,
        _load_isg_sensitivity(isg_path),
        parameters=PARAMETERS,
        ordered_bound_endpoints=("HLA-DRhi cDC2",),
    )


def _matrix(
    summary: pd.DataFrame,
    endpoint: str,
    column: str,
) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    subset = summary.loc[summary["held_out_label"].eq(endpoint)]
    tau_min_values = tuple(sorted(subset["tau_min"].astype(float).unique()))
    tau_max_values = tuple(sorted(subset["tau_max"].astype(float).unique()))
    matrix = np.full((len(tau_min_values), len(tau_max_values)), np.nan)
    for row in subset.itertuples(index=False):
        row_index = tau_min_values.index(float(row.tau_min))
        column_index = tau_max_values.index(float(row.tau_max))
        matrix[row_index, column_index] = float(getattr(row, column))
    missing_valid_cell = np.isnan(matrix) & np.asarray(
        [[tau_min <= tau_max for tau_max in tau_max_values] for tau_min in tau_min_values]
    )
    if missing_valid_cell.any():
        raise HIHAParameterSensitivityError(f"The {column} heatmap is incomplete for {endpoint}.")
    return matrix, tau_min_values, tau_max_values


def _save_figure(fig: plt.Figure, outputs: dict[str, Path]) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = RASTER_DPI
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        elif suffix == "svg":
            kwargs["metadata"] = {"Date": None}
        with plt.rc_context({"svg.hashsalt": SVG_HASH_SALT}):
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
            color_map = (
                matplotlib.colormaps["RdBu_r"].with_extremes(bad="#E6E6E6")
                if np.isnan(matrix).any()
                else "RdBu_r"
            )
            image = axis.imshow(
                matrix,
                origin="lower",
                cmap=color_map,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    if np.isnan(value):
                        continue
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
    """Write source tables, figure formats, and a provenance manifest."""
    project_root = project_root.resolve()
    isg_root = (
        project_root / "results/HIHA_DC/sensitivity/isg_tau_range_target1_alpha025_discovery/tables"
    )
    source_paths = {
        "hla_h1_split_metrics": project_root / H1_SPLIT_METRICS_RELATIVE,
        "hla_h1_verification_report": project_root / H1_VERIFICATION_RELATIVE,
        "hla_h1_checksums": project_root / H1_CHECKSUMS_RELATIVE,
        "isg_discovery_by_split": isg_root / "discovery_by_split.csv",
    }
    by_split, summary = collect_hiha_parameter_sensitivity_from_h1(
        hla_split_metrics_path=source_paths["hla_h1_split_metrics"],
        hla_verification_path=source_paths["hla_h1_verification_report"],
        hla_checksums_path=source_paths["hla_h1_checksums"],
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
    artifacts = {
        "by_split": by_split_path,
        "summary": summary_path,
        "figure_png": outputs["png"],
        "figure_pdf": outputs["pdf"],
        "figure_svg": outputs["svg"],
    }
    manifest = {
        "analysis_id": "hiha_query_penalty_sensitivity_figure",
        "aggregation": "arithmetic mean across five fixed donor splits",
        "metrics": ["within-cDC2 AP", "represented-state forced macro-F1"],
        "parameter_grids": PARAMETERS,
        "selected_settings": SELECTED_SETTINGS,
        "rendering_metadata_policy": {
            "svg_hash_salt": SVG_HASH_SALT,
            "svg_date_removed": True,
            "pdf_creation_and_modification_dates_removed": True,
        },
        "sources": {
            name: {"path": str(path.relative_to(project_root)), "sha256": _sha256_file(path)}
            for name, path in source_paths.items()
        },
        "artifacts": {
            name: {"path": str(path.relative_to(project_root)), "sha256": _sha256_file(path)}
            for name, path in artifacts.items()
        },
        "interpretation": {
            "same_data_parameter_sensitivity": True,
            "independent_validation": False,
            "hla_selection_performed": False,
            "hla_selected_setting_unchanged": True,
            "hla_selected_setting_in_displayed_grid": True,
            "isg_grid_informed_selected_setting": True,
            "cross_endpoint_sensitivity_magnitude_comparison_supported": False,
        },
    }
    manifest_path = (
        project_root / "results/HIHA_DC/figures/hiha_parameter_sensitivity_manifest.yaml"
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {**artifacts, "manifest": manifest_path}
