from __future__ import annotations

from pathlib import Path

import pandas as pd

from coreot.artifacts.run_artifacts import ArtifactContractError, ArtifactFailure, RunArtifacts
from coreot.results.grid_common import RunDescriptor, ResultsGridError, STAGE, _read_yaml

FIXED_REFERENCE_METHODS: tuple[str, ...] = ("prior_only", "nn")
DEFAULT_FIXED_REFERENCE_ROOT = Path("results/HIHA_DC/main")

def _filter_completed_runs(
    runs_root: Path, descriptors: tuple[RunDescriptor, ...]
) -> tuple[tuple[RunDescriptor, ...], list[str]]:
    """Split descriptors into completed (metrics.csv exists) and failed."""
    completed: list[RunDescriptor] = []
    failed: list[str] = []
    for desc in descriptors:
        metrics_path = runs_root / desc.run_id / "evaluation" / "metrics.csv"
        if metrics_path.is_file():
            completed.append(desc)
        else:
            failed.append(desc.run_id)
    return tuple(completed), failed


def _partition_completed_sweep_runs(
    *,
    runs_root: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
) -> tuple[tuple[RunDescriptor, ...], list[ArtifactFailure]]:
    completed: list[RunDescriptor] = []
    failures: list[ArtifactFailure] = []
    for descriptor in descriptors:
        run_root = runs_root / descriptor.run_id
        artifacts = RunArtifacts(run_root, STAGE)
        checks = [
            artifacts.evaluation().metrics(),
            artifacts.evaluation_truth(condition).query_truth(),
            artifacts.scoring(condition, candidate_set).cell_scores(),
        ]
        run_failures: list[ArtifactFailure] = []
        for handle in checks:
            try:
                handle.read()
            except ArtifactContractError as exc:
                run_failures.append(exc.to_failure())
        if run_failures:
            failures.extend(run_failures)
            continue
        completed.append(descriptor)
    return tuple(completed), failures


def _failed_run_ids(failures: list[ArtifactFailure]) -> list[str]:
    failed_run_ids: list[str] = []
    seen: set[str] = set()
    for failure in failures:
        if failure.run_id in seen:
            continue
        seen.add(failure.run_id)
        failed_run_ids.append(failure.run_id)
    return failed_run_ids

def read_fixed_reference_rows(
    detection_summary_path: Path,
    shared_summary_path: Path,
    held_out_labels: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read main-grid summaries and filter to fixed reference methods and labels.

    Returns (detection_summary, shared_summary) with added columns:
    tau=NaN, sensitivity_role='reference'.
    """
    if not detection_summary_path.is_file():
        raise FileNotFoundError(
            f"Main grid detection summary not found: {detection_summary_path}. "
            "Run generate_hiha_dc_report_leave_one_results.py first."
        )
    if not shared_summary_path.is_file():
        raise FileNotFoundError(
            f"Main grid shared label-transfer summary not found: {shared_summary_path}. "
            "Run generate_hiha_dc_report_leave_one_results.py first."
        )
    allowed_labels = set(held_out_labels) | {"overall"}

    det = pd.read_csv(detection_summary_path)
    det = det.loc[
        det["method"].isin(FIXED_REFERENCE_METHODS)
        & det["held_out_label"].isin(allowed_labels)
    ].copy()
    det["tau"] = float("nan")
    det["sensitivity_role"] = "reference"

    shared = pd.read_csv(shared_summary_path)
    shared = shared.loc[
        shared["method"].isin(FIXED_REFERENCE_METHODS)
        & shared["held_out_label"].isin(allowed_labels)
    ].copy()
    shared["tau"] = float("nan")
    shared["sensitivity_role"] = "reference"

    return det, shared


def read_fixed_reference_bundle(
    bundle_root: str | Path,
    held_out_labels: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(bundle_root)
    return read_fixed_reference_rows(
        detection_summary_path=root / "tables" / "main_detection_summary.csv",
        shared_summary_path=root / "tables" / "shared_label_transfer_summary.csv",
        held_out_labels=held_out_labels,
    )

def _is_missing_tau(value: object) -> bool:
    """Return True for fixed-reference rows with no tau value."""
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _tau_or_none(value: object) -> float | None:
    """Normalize a row tau value for lookup and display."""
    if _is_missing_tau(value):
        return None
    return float(value)

LABELWISE_DISPLAY_NAME: dict[str, str] = {
    "coreot_constant_tau": "CoRe-OT constant τ",
}

LABELWISE_LABEL_GRID: dict[str, dict[str, object]] = {
    "HLA-DRhi cDC2": {"fixed_param": "tau", "fixed_value": 2.0, "swept_param": "alpha", "swept_values": (1.0, 2.0, 4.0, 8.0, 16.0)},
    "CD14+ cDC2": {"fixed_param": "alpha", "fixed_value": 5.0, "swept_param": "tau", "swept_values": (1.0, 2.0, 4.0, 8.0, 16.0)},
    "ISG+ cDC2": {"fixed_param": "alpha", "fixed_value": 5.0, "swept_param": "tau", "swept_values": (1.0, 2.0, 4.0, 8.0, 16.0)},
}
LABELWISE_HELD_OUT_LABELS: tuple[str, ...] = tuple(LABELWISE_LABEL_GRID)

def parse_labelwise_params(run_config_dir: Path) -> tuple[float, float]:
    """Read tau_source and alpha from a generated transport.yaml.

    Returns (tau, alpha). Validates that coreot_constant_tau has both params.
    """
    transport_path = run_config_dir / "transport.yaml"
    config = _read_yaml(transport_path)
    methods = config.get("methods", ())
    for method in methods:
        if not isinstance(method, dict):
            continue
        name = method.get("name")
        if name != "coreot_constant_tau":
            continue
        tau = float(method["tau_source"])
        tau_target = float(method.get("tau_target", tau))
        if tau_target != tau:
            raise ResultsGridError(
                f"tau_target ({tau_target}) != tau_source ({tau}) in {transport_path}"
            )
        alpha = float(method.get("alpha", float("nan")))
        if not (0 < alpha < 100):
            raise ResultsGridError(
                f"Invalid alpha={alpha} in {transport_path}"
            )
        return tau, alpha
    raise ResultsGridError(f"No coreot_constant_tau method found in {transport_path}")


def _build_labelwise_param_maps(
    grid_dir: Path, descriptors: tuple[RunDescriptor, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    """Build run_id -> tau and run_id -> alpha mappings."""
    tau_map: dict[str, float] = {}
    alpha_map: dict[str, float] = {}
    for desc in descriptors:
        config_dir = grid_dir / desc.run_id
        t, a = parse_labelwise_params(config_dir)
        tau_map[desc.run_id] = t
        alpha_map[desc.run_id] = a
    return tau_map, alpha_map


def _validate_labelwise_grid(
    descriptors: tuple[RunDescriptor, ...],
    tau_map: dict[str, float],
    alpha_map: dict[str, float],
) -> None:
    """Validate that each run matches the label-specific sweep design."""
    for desc in descriptors:
        lbl = desc.held_out_label
        grid = LABELWISE_LABEL_GRID.get(lbl)
        if grid is None:
            raise ResultsGridError(
                f"Held-out label '{lbl}' is not part of the labelwise grid. "
                f"Expected: {list(LABELWISE_LABEL_GRID)}"
            )
        tau = tau_map[desc.run_id]
        alpha = alpha_map[desc.run_id]
        fixed_param = str(grid["fixed_param"])
        fixed_value = float(grid["fixed_value"])
        swept_values = tuple(grid["swept_values"])

        if fixed_param == "tau":
            if tau != fixed_value:
                raise ResultsGridError(
                    f"Run {desc.run_id}: expected tau={fixed_value} for {lbl}, got tau={tau}"
                )
            if alpha not in swept_values:
                raise ResultsGridError(
                    f"Run {desc.run_id}: alpha={alpha} not in {swept_values} for {lbl}"
                )
        elif fixed_param == "alpha":
            if alpha != fixed_value:
                raise ResultsGridError(
                    f"Run {desc.run_id}: expected alpha={fixed_value} for {lbl}, got alpha={alpha}"
                )
            if tau not in swept_values:
                raise ResultsGridError(
                    f"Run {desc.run_id}: tau={tau} not in {swept_values} for {lbl}"
                )


def _labelwise_sensitivity_axis(held_out_label: str) -> str:
    """Return the swept parameter name for a label."""
    grid = LABELWISE_LABEL_GRID.get(held_out_label)
    if grid is None:
        return "unknown"
    return str(grid["swept_param"])


def _labelwise_swept_display_name(method: str, tau: float, alpha: float) -> str:
    """Build dynamic display name with both tau and alpha."""
    tau_str = str(int(tau)) if tau == int(tau) else str(tau)
    alpha_str = str(int(alpha)) if alpha == int(alpha) else str(alpha)
    base = LABELWISE_DISPLAY_NAME.get(method, method)
    return f"{base}, τ={tau_str}, α={alpha_str}"


def _labelwise_sort_key(row: pd.Series) -> tuple[int, float, float]:
    """Sort: fixed rows first, then swept by (method_order, tau, alpha)."""
    method = str(row["method"])
    if _is_missing_tau(row.get("tau", float("nan"))):
        ref_order = {"prior_only": 0, "nn": 1}
        return (-1, ref_order.get(method, 99), 0)
    tau_val = float(row.get("tau", 0))
    alpha_val = float(row.get("alpha", 0))
    swept_order = {"coreot_constant_tau": 0}
    return (0, swept_order.get(method, 99) * 1000 + alpha_val * 10 + tau_val, tau_val + alpha_val)


def _labelwise_unique_rows(ordered: pd.DataFrame, held_out_label: str) -> pd.DataFrame:
    """Return one render row per (method, tau, alpha) for a held-out label."""
    label_rows = ordered.loc[ordered["held_out_label"] == held_out_label]
    key_cols = ["method"]
    if "tau" in label_rows.columns:
        key_cols.append("tau")
    if "alpha" in label_rows.columns:
        key_cols.append("alpha")
    return label_rows.loc[:, key_cols].drop_duplicates().reset_index(drop=True)


def _annotate_and_filter_labelwise_overall_rows(
    summary: pd.DataFrame,
    by_run: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    required_label_count: int,
) -> pd.DataFrame:
    """Keep only complete labelwise overall aggregates and record label coverage."""
    result = summary.copy()
    result["n_held_out_labels"] = 1
    overall_mask = result["held_out_label"] == "overall"
    if not overall_mask.any():
        return result

    aggregate_columns = tuple(col for col in group_columns if col != "held_out_label")
    contributor_counts = (
        by_run.groupby(list(aggregate_columns), dropna=False)["held_out_label"]
        .nunique()
        .rename("n_held_out_labels")
        .reset_index()
    )
    overall_rows = (
        result.loc[overall_mask]
        .drop(columns=["n_held_out_labels"])
        .merge(contributor_counts, on=list(aggregate_columns), how="left")
    )
    overall_rows = overall_rows.loc[
        overall_rows["n_held_out_labels"] == required_label_count
    ]
    return pd.concat([result.loc[~overall_mask], overall_rows], ignore_index=True)


def _lookup_labelwise_cell(
    summary: pd.DataFrame,
    method: str,
    held_out_label: str,
    quantity: str,
    tau: float | None,
    alpha: float | None,
) -> object:
    """Look up a sensitivity summary cell with tau and alpha awareness."""
    mask = (
        (summary["method"] == method)
        & (summary["held_out_label"] == held_out_label)
        & (summary["quantity"] == quantity)
    )
    if tau is not None:
        mask = mask & (summary["tau"] == tau)
    if alpha is not None:
        mask = mask & (summary["alpha"] == alpha)
    matches = summary.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]
