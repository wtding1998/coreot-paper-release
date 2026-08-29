"""Manuscript-results generation for the HIHA supplementary figures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_HELD_OUT = ("HLA-DRhi cDC2", "ISG+ cDC2")
CONDITIONS = ("incomplete_reference", "full_reference_control")
CONDITION_LABELS = {
    "incomplete_reference": "Reference-omitted condition",
    "full_reference_control": "Restored-reference condition",
}
CONDITION_COLORS = {
    "incomplete_reference": "#E69F00",
    "full_reference_control": "#0072B2",
}
GROUPS = ("held_out", "other_cdc2", "other")
GROUP_LABELS = {
    "held_out": "Reference-omitted cells",
    "other_cdc2": "Represented cDC2\ncontrols",
    "other": "Other shared states",
}
CDC2_STATE_ORDER = ("CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2")
NON_CDC2 = "Non-cDC2"
DEFAULT_CANDIDATE_SET = "hiha_harmony30_k100"
DEFAULT_ETA = 1.0e-12


class HIHASupplementError(RuntimeError):
    """Raised when retained artifacts cannot support a supplementary figure."""


@dataclass(frozen=True)
class RunInfo:
    root: Path
    held_out_label: str
    seed: int


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise HIHASupplementError(
            f"{path} is missing required columns: {sorted(missing)}"
        )


def _as_cell_ids(values: pd.Series) -> pd.Series:
    return values.astype("string")


def _read_input_metadata(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    adata = ad.read_h5ad(input_path)
    metadata = adata.obs.copy()
    if "cell_id" in metadata.columns:
        metadata["cell_id"] = _as_cell_ids(metadata["cell_id"])
    else:
        metadata = metadata.reset_index(names="cell_id")
        metadata["cell_id"] = _as_cell_ids(metadata["cell_id"])
    _require_columns(metadata, ("cell_id", "AIFI_L2"), input_path)
    if metadata["cell_id"].duplicated().any():
        raise HIHASupplementError(f"Duplicate cell IDs in {input_path}.")
    metadata = metadata.loc[:, ["cell_id", "AIFI_L2"]].copy()
    if metadata["AIFI_L2"].isna().any():
        raise HIHASupplementError(f"Missing AIFI_L2 values in {input_path}.")
    return metadata


def _read_query_truth(run_root: Path, condition: str) -> pd.DataFrame:
    path = (
        run_root
        / "benchmark"
        / condition
        / "evaluation_truth"
        / "query_truth.csv"
    )
    truth = pd.read_csv(path)
    _require_columns(truth, ("cell_id", "true_label", "removed_state"), path)
    truth["cell_id"] = _as_cell_ids(truth["cell_id"])
    if truth["cell_id"].duplicated().any():
        raise HIHASupplementError(f"Duplicate query IDs in {path}.")
    return truth.loc[:, ["cell_id", "true_label", "removed_state"]].copy()


def _discover_runs(runs_root: Path) -> list[RunInfo]:
    roots = sorted(runs_root.glob("hiha_dc_*_report_leave_one_HIHA_DC"))
    if not roots:
        raise FileNotFoundError(
            f"No selected HIHA run roots found under {runs_root}."
        )

    discovered: list[RunInfo] = []
    for root in roots:
        truth = _read_query_truth(root, "incomplete_reference")
        held_out = truth["removed_state"].dropna().unique().tolist()
        if len(held_out) != 1:
            raise HIHASupplementError(
                f"Expected one held-out state in {root}, found {held_out}."
            )
        if held_out[0] not in EXPECTED_HELD_OUT:
            continue
        match = re.search(r"_seed([0-9]+)_report_leave_one_HIHA_DC$", root.name)
        if match is None:
            raise HIHASupplementError(f"Cannot parse donor split seed from {root}.")
        discovered.append(
            RunInfo(root=root, held_out_label=str(held_out[0]), seed=int(match.group(1)))
        )

    expected_keys = {
        (held_out, seed)
        for held_out in EXPECTED_HELD_OUT
        for seed in range(1, 6)
    }
    observed_keys = {(run.held_out_label, run.seed) for run in discovered}
    if observed_keys != expected_keys:
        raise HIHASupplementError(
            "Selected HIHA runs do not contain exactly two held-out states and "
            "seeds 1 through 5: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"unexpected={sorted(observed_keys - expected_keys)}"
        )
    return sorted(
        discovered,
        key=lambda run: (EXPECTED_HELD_OUT.index(run.held_out_label), run.seed),
    )


def _transport_root(
    run: RunInfo, condition: str, candidate_set: str, method: str = "coreot_full"
) -> Path:
    return run.root / "transport" / condition / candidate_set / method


def _read_source_frame(
    run: RunInfo,
    condition: str,
    candidate_set: str,
    input_metadata: pd.DataFrame,
) -> pd.DataFrame:
    method_root = _transport_root(run, condition, candidate_set)
    score_path = method_root / "cell_transport_scores.parquet"
    scores = pd.read_parquet(score_path)
    _require_columns(scores, ("cell_id", "a", "a_hat", "u"), score_path)
    scores["cell_id"] = _as_cell_ids(scores["cell_id"])
    if scores["cell_id"].duplicated().any():
        raise HIHASupplementError(f"Duplicate source IDs in {score_path}.")
    if not np.isfinite(scores[["a", "a_hat", "u"]].to_numpy(dtype=float)).all():
        raise HIHASupplementError(f"Non-finite source scores in {score_path}.")
    if (scores[["a", "a_hat"]] < 0).any().any():
        raise HIHASupplementError(f"Negative source masses in {score_path}.")

    truth = _read_query_truth(run.root, condition)
    source = scores.merge(truth, on="cell_id", how="outer", validate="one_to_one")
    if source[["true_label", "removed_state"]].isna().any().any():
        raise HIHASupplementError(
            f"Source score IDs and query truth IDs differ for {run.root}/{condition}."
        )
    source = source.merge(
        input_metadata, on="cell_id", how="left", validate="one_to_one"
    )
    if source["AIFI_L2"].isna().any():
        missing = int(source["AIFI_L2"].isna().sum())
        raise HIHASupplementError(
            f"{missing} source cells from {score_path} are absent from AIFI_L2 metadata."
        )

    held_out = run.held_out_label
    source["source_group"] = np.select(
        [
            source["true_label"].eq(held_out),
            source["AIFI_L2"].eq("cDC2"),
        ],
        ["held_out", "other_cdc2"],
        default="other",
    )
    return source


def _read_target_labels(run: RunInfo, condition: str) -> pd.DataFrame:
    path = (
        run.root
        / "benchmark"
        / condition
        / "model_visible"
        / "target_labels.csv"
    )
    labels = pd.read_csv(path)
    _require_columns(labels, ("cell_id", "target_label", "broad_label"), path)
    labels["cell_id"] = _as_cell_ids(labels["cell_id"])
    if labels["cell_id"].duplicated().any():
        raise HIHASupplementError(f"Duplicate target IDs in {path}.")
    if labels["target_label"].isna().any() or labels["broad_label"].isna().any():
        raise HIHASupplementError(f"Missing target labels in {path}.")
    return labels.loc[:, ["cell_id", "target_label", "broad_label"]].copy()


def _target_state_order(full_labels: pd.DataFrame) -> list[str]:
    cdc2_states = set(
        full_labels.loc[full_labels["broad_label"].eq("cDC2"), "target_label"]
    )
    ordered = [label for label in CDC2_STATE_ORDER if label in cdc2_states]
    ordered.extend(sorted(cdc2_states - set(ordered)))
    if not ordered:
        raise HIHASupplementError(
            "The restored-reference condition has no represented cDC2 target states."
        )
    return ordered + [NON_CDC2]


def _source_summary(
    source: pd.DataFrame, held_out: str, seed: int, condition: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        subset = source.loc[source["source_group"].eq(group)]
        if subset.empty:
            raise HIHASupplementError(
                f"Empty source group {group} for {held_out}, seed={seed}, {condition}."
            )
        empirical_mass = subset["a"].to_numpy(dtype=float)
        if (empirical_mass <= 0).any():
            raise HIHASupplementError(
                f"Non-positive empirical source mass in {held_out}, seed={seed}, {condition}."
            )
        rows.append(
            {
                "held_out_label": held_out,
                "seed": seed,
                "reference_condition": condition,
                "source_group": group,
                "n_cells": int(len(subset)),
                "mean_u": float(subset["u"].mean()),
                "median_u": float(subset["u"].median()),
                "mean_transported_source_mass": float(subset["a_hat"].mean()),
            }
        )
    return rows


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or labels.sum() == 0:
        raise HIHASupplementError("AUPRC is undefined because no positives are present.")
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order].astype(float)
    cumulative_positives = np.cumsum(ranked_labels)
    positions = np.arange(1, len(ranked_labels) + 1, dtype=float)
    precision = cumulative_positives / positions
    return float((precision * ranked_labels).sum() / labels.sum())


def _auprc_summary(
    source: pd.DataFrame, held_out: str, seed: int, condition: str
) -> dict[str, object]:
    local = source.loc[source["AIFI_L2"].eq("cDC2")].copy()
    if local.empty:
        raise HIHASupplementError(
            f"Within-cDC2 source cohort is empty for {held_out}, seed={seed}, {condition}."
        )
    labels = local["true_label"].eq(held_out).to_numpy(dtype=bool)
    if labels.all():
        raise HIHASupplementError(
            f"Within-cDC2 cohort has no negatives for {held_out}, seed={seed}, {condition}."
        )
    return {
        "held_out_label": held_out,
        "seed": seed,
        "reference_condition": condition,
        "n_cells": int(len(local)),
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auprc": _average_precision(labels, local["u"].to_numpy(dtype=float)),
    }


def _destination_summary(
    source: pd.DataFrame,
    coupling: pd.DataFrame,
    target_labels: pd.DataFrame,
    target_state_order: list[str],
    held_out: str,
    seed: int,
    condition: str,
    eta: float,
) -> list[dict[str, object]]:
    required = ("source_cell_id", "target_cell_id", "coupling")
    _require_columns(coupling, required, Path("sparse_coupling.parquet"))
    coupling = coupling.loc[:, list(required)].copy()
    coupling["source_cell_id"] = _as_cell_ids(coupling["source_cell_id"])
    coupling["target_cell_id"] = _as_cell_ids(coupling["target_cell_id"])
    coupling["coupling"] = coupling["coupling"].astype(float)
    if not np.isfinite(coupling["coupling"]).all() or (coupling["coupling"] < 0).any():
        raise HIHASupplementError(
            f"Invalid coupling values for {held_out}, seed={seed}, {condition}."
        )

    target = coupling.merge(
        target_labels,
        left_on="target_cell_id",
        right_on="cell_id",
        how="left",
        validate="many_to_one",
    )
    if target["target_label"].isna().any():
        raise HIHASupplementError(
            f"Coupling contains target IDs absent from target_labels.csv for "
            f"{held_out}, seed={seed}, {condition}."
        )
    score_ids = set(source["cell_id"])
    coupling_ids = set(target["source_cell_id"])
    if not coupling_ids.issubset(score_ids):
        raise HIHASupplementError(
            f"Coupling contains source IDs absent from source scores for "
            f"{held_out}, seed={seed}, {condition}."
        )

    valid = source.loc[source["a_hat"].gt(eta)].copy()
    valid_ids = set(valid["cell_id"])
    row_mass = coupling.groupby("source_cell_id", sort=False)["coupling"].sum()
    missing_rows = valid_ids - set(row_mass.index)
    if missing_rows:
        raise HIHASupplementError(
            f"{len(missing_rows)} source cells with a_hat > eta have no coupling row "
            f"for {held_out}, seed={seed}, {condition}."
        )
    expected_mass = valid.set_index("cell_id")["a_hat"].astype(float)
    observed_mass = row_mass.reindex(expected_mass.index)
    if not np.allclose(
        observed_mass.to_numpy(dtype=float),
        expected_mass.to_numpy(dtype=float),
        rtol=1.0e-7,
        atol=1.0e-12,
    ):
        raise HIHASupplementError(
            f"Coupling row sums do not match a_hat for {held_out}, seed={seed}, {condition}."
        )

    target = target.loc[target["source_cell_id"].isin(valid_ids)].copy()
    target["conditional_destination_mass"] = target["coupling"] / target[
        "source_cell_id"
    ].map(expected_mass)
    target["destination"] = np.where(
        target["target_label"].isin(target_state_order[:-1]),
        target["target_label"],
        NON_CDC2,
    )
    row_probabilities = target.groupby("source_cell_id", sort=False)[
        "conditional_destination_mass"
    ].sum()
    if not np.allclose(row_probabilities.to_numpy(dtype=float), 1.0, atol=1.0e-7):
        raise HIHASupplementError(
            f"Conditional destination probabilities do not sum to one for "
            f"{held_out}, seed={seed}, {condition}."
        )
    cell_destinations = (
        target.groupby(["source_cell_id", "destination"], sort=False)[
            "conditional_destination_mass"
        ]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=valid["cell_id"], columns=target_state_order, fill_value=0.0)
    )
    if not np.allclose(
        cell_destinations.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1.0e-7
    ):
        raise HIHASupplementError(
            f"Destination matrix rows do not sum to one for {held_out}, seed={seed}, "
            f"{condition}."
        )

    valid = valid.set_index("cell_id")
    valid["relative_transported_source_mass"] = valid["a_hat"] / valid["a"]
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        group_ids = valid.index[valid["source_group"].eq(group)]
        if len(group_ids) == 0:
            raise HIHASupplementError(
                f"Empty valid source group {group} for {held_out}, seed={seed}, {condition}."
            )
        group_matrix = cell_destinations.loc[group_ids]
        group_mass = valid.loc[group_ids, "relative_transported_source_mass"]
        for destination in target_state_order:
            rows.append(
                {
                    "held_out_label": held_out,
                    "seed": seed,
                    "reference_condition": condition,
                    "source_group": group,
                    "target_state": destination,
                    "n_cells_valid": int(len(group_ids)),
                    "mean_destination_fraction": float(group_matrix[destination].mean()),
                    "relative_transported_source_mass": float(group_mass.mean()),
                }
            )
    return rows


def _collect_s1_data(
    runs: list[RunInfo],
    input_metadata: pd.DataFrame,
    candidate_set: str,
    eta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_rows: list[dict[str, object]] = []
    auprc_rows: list[dict[str, object]] = []
    destination_rows: list[dict[str, object]] = []

    for run in runs:
        paired_sources: dict[str, pd.DataFrame] = {}
        paired_targets: dict[str, pd.DataFrame] = {}
        for condition in CONDITIONS:
            paired_sources[condition] = _read_source_frame(
                run, condition, candidate_set, input_metadata
            )
            paired_targets[condition] = _read_target_labels(run, condition)

        source_id_sets = {
            condition: set(frame["cell_id"])
            for condition, frame in paired_sources.items()
        }
        if source_id_sets[CONDITIONS[0]] != source_id_sets[CONDITIONS[1]]:
            raise HIHASupplementError(
                f"Paired source-cell IDs differ for {run.held_out_label}, seed={run.seed}."
            )
        source_groups = {
            condition: dict(
                zip(
                    frame["cell_id"],
                    frame["source_group"],
                    strict=True,
                )
            )
            for condition, frame in paired_sources.items()
        }
        if source_groups[CONDITIONS[0]] != source_groups[CONDITIONS[1]]:
            raise HIHASupplementError(
                f"Paired source groups differ for {run.held_out_label}, seed={run.seed}."
            )

        full_labels = paired_targets["full_reference_control"]
        incomplete_labels = paired_targets["incomplete_reference"]
        full_target_states = set(full_labels["target_label"])
        incomplete_target_states = set(incomplete_labels["target_label"])
        if run.held_out_label not in full_target_states:
            raise HIHASupplementError(
                f"Held-out state {run.held_out_label} is absent from the full target "
                f"for seed {run.seed}."
            )
        if run.held_out_label in incomplete_target_states:
            raise HIHASupplementError(
                f"Held-out state {run.held_out_label} is present in the ablated target "
                f"for seed {run.seed}."
            )
        target_state_order = _target_state_order(full_labels)

        for condition in CONDITIONS:
            source = paired_sources[condition]
            source_rows.extend(
                _source_summary(source, run.held_out_label, run.seed, condition)
            )
            auprc_rows.append(
                _auprc_summary(source, run.held_out_label, run.seed, condition)
            )
            coupling_path = (
                _transport_root(run, condition, candidate_set) / "sparse_coupling.parquet"
            )
            coupling = pd.read_parquet(coupling_path)
            destination_rows.extend(
                _destination_summary(
                    source=source,
                    coupling=coupling,
                    target_labels=paired_targets[condition],
                    target_state_order=target_state_order,
                    held_out=run.held_out_label,
                    seed=run.seed,
                    condition=condition,
                    eta=eta,
                )
            )
    return (
        pd.DataFrame(source_rows),
        pd.DataFrame(auprc_rows),
        pd.DataFrame(destination_rows),
    )


def _plot_s1(
    destination_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure = plt.figure(figsize=(11.8, 6.4))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(5.6, 1.15),
        wspace=0.12,
        hspace=0.55,
    )

    for row, held_out in enumerate(EXPECTED_HELD_OUT):
        heat_axis = figure.add_subplot(outer[row, 0])
        mass_axis = figure.add_subplot(outer[row, 1])

        destinations = destination_summary.loc[
            destination_summary["held_out_label"].eq(held_out)
        ]
        columns = [
            state
            for state in CDC2_STATE_ORDER
            if state in set(destinations["target_state"])
        ]
        columns.extend(
            sorted(
                set(destinations["target_state"]) - set(columns) - {NON_CDC2}
            )
        )
        columns.append(NON_CDC2)
        heat_rows = []
        mass_rows = []
        mass_sd_rows = []
        mass_colors = []
        row_labels = []
        row_keys = []
        for group in ("held_out", "other_cdc2"):
            for condition in CONDITIONS:
                raw_subset = destinations.loc[
                    destinations["source_group"].eq(group)
                    & destinations["reference_condition"].eq(condition)
                ]
                subset = (
                    raw_subset
                    .groupby("target_state", as_index=True)
                    .agg(
                        mean_destination_fraction=(
                            "mean_destination_fraction",
                            "mean",
                        ),
                        relative_transported_source_mass=(
                            "relative_transported_source_mass",
                            "mean",
                        ),
                    )
                )
                heat_rows.append(
                    subset.reindex(columns)["mean_destination_fraction"].to_numpy(
                        dtype=float
                    )
                )
                mass_rows.append(
                    float(subset["relative_transported_source_mass"].mean())
                )
                split_masses = (
                    raw_subset.groupby("seed", sort=True)[
                        "relative_transported_source_mass"
                    ]
                    .first()
                    .to_numpy(dtype=float)
                )
                mass_sd_rows.append(float(split_masses.std(ddof=1)))
                mass_colors.append(CONDITION_COLORS[condition])
                row_labels.append(
                    f"{GROUP_LABELS[group]}\n{CONDITION_LABELS[condition]}"
                )
                row_keys.append((group, condition))
        heat = np.asarray(heat_rows, dtype=float)
        heat_axis.imshow(
            heat,
            vmin=0,
            vmax=1,
            cmap=plt.cm.Blues,
            aspect="auto",
            interpolation="nearest",
        )
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                value = heat[i, j]
                structural_unavailability = (
                    row_keys[i][1] == "incomplete_reference"
                    and columns[j] == held_out
                )
                if structural_unavailability:
                    display = "—"
                else:
                    display = f"{value:.2f}"
                heat_axis.text(
                    j,
                    i,
                    display,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#111111" if value < 0.63 else "white",
                )
        heat_axis.set_yticks(range(len(row_labels)), row_labels)
        heat_axis.set_xticks(
            range(len(columns)),
            [label.replace(" cDC2", "\ncDC2") for label in columns],
        )
        heat_axis.tick_params(axis="y", length=0)
        heat_axis.tick_params(axis="x", length=0, pad=3)
        heat_axis.set_title(
            f"{chr(65 + row)}  {held_out}",
            loc="left",
            weight="bold",
            fontsize=10,
        )
        heat_axis.set_xlabel("Reference state")
        for spine in heat_axis.spines.values():
            spine.set_visible(False)

        mass_axis.barh(
            np.arange(len(row_labels)),
            mass_rows,
            xerr=mass_sd_rows,
            color=mass_colors,
            edgecolor="#333333",
            linewidth=0.45,
            alpha=0.82,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
            },
        )
        mass_axis.set_yticks([])
        mass_axis.set_xlim(0.0, 1.0)
        mass_axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        mass_axis.set_xlabel(
            "Mean relative transported\nquery mass, $\\widehat a_i/a_i$",
            fontsize=7,
        )
        mass_axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        mass_axis.spines[["top", "right", "left"]].set_visible(False)
        mass_axis.invert_yaxis()

    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_s1_description(
    *,
    description_path: Path,
    source_path: Path,
    auprc_path: Path,
    destination_path: Path,
    eta: float,
) -> None:
    description_path.write_text(
        "# Supplementary Figure S1\n\n"
        "Reference destinations and relative transported query mass under "
        "endpoint-state omission and restoration. Panels A and B correspond to "
        "HLA-DRhi cDC2 and ISG+ cDC2, respectively, and use CoRe-OT. Heatmap "
        "entries are query-cell-normalized destination fractions averaged across "
        "five donor splits. An em dash marks the endpoint's structural "
        "unavailability under the reference-omitted condition. Orange and blue "
        "bars denote the reference-omitted and restored-reference conditions, "
        "respectively, and show mean relative transported query mass with "
        "sample-standard-deviation error bars. Query groups use raw AIFI_L2 "
        "metadata and evaluation-side true_label only for the "
        "reference-omitted-cell grouping.\n\n"
        f"Source summary: {source_path}\n\n"
        f"AUPRC summary: {auprc_path}\n\n"
        f"Destination summary: {destination_path}\n\n"
        f"Transported-mass inclusion floor: eta = {eta:g}.\n",
        encoding="utf-8",
    )


def write_hiha_s1(
    *,
    runs_root: Path = Path("runs"),
    input_path: Path = Path(
        "data/derived/hiha_dc/"
        "human_immune_health_atlas_dc.with_recomputed_AIFI_L2_score.h5ad"
    ),
    output_root: Path = Path("results/HIHA_DC/figures"),
    candidate_set: str = DEFAULT_CANDIDATE_SET,
    eta: float = DEFAULT_ETA,
) -> dict[str, Path]:
    if eta <= 0:
        raise ValueError("eta must be positive.")
    runs = _discover_runs(runs_root)
    input_metadata = _read_input_metadata(input_path)
    source_summary, auprc_summary, destination_summary = _collect_s1_data(
        runs, input_metadata, candidate_set, eta
    )

    output_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    source_path = data_root / "supplementary_figure_s1_source_summary.csv"
    auprc_path = data_root / "supplementary_figure_s1_auprc_by_seed.csv"
    destination_path = data_root / "supplementary_figure_s1_destinations_by_seed.csv"
    source_summary.to_csv(source_path, index=False)
    auprc_summary.to_csv(auprc_path, index=False)
    destination_summary.to_csv(destination_path, index=False)

    figure_path = output_root / "supplementary_figure_s1_destinations.png"
    _plot_s1(destination_summary, figure_path)

    description_path = output_root / "supplementary_figure_s1_destinations.md"
    _write_s1_description(
        description_path=description_path,
        source_path=source_path,
        auprc_path=auprc_path,
        destination_path=destination_path,
        eta=eta,
    )
    manifest_path = output_root / "supplementary_figure_s1_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "figure": "S1",
                "method": "coreot_full",
                "score": "raw_source_marginal_deficit_u",
                "held_out_labels": list(EXPECTED_HELD_OUT),
                "seeds": list(range(1, 6)),
                "conditions": list(CONDITIONS),
                "candidate_set": candidate_set,
                "eta": float(eta),
                "input_path": str(input_path),
                "outputs": {
                    "figure_png": str(figure_path),
                    "figure_pdf": str(figure_path.with_suffix(".pdf")),
                    "source_summary": str(source_path),
                    "auprc_by_seed": str(auprc_path),
                    "destinations_by_seed": str(destination_path),
                    "description": str(description_path),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "figure_png": figure_path,
        "figure_pdf": figure_path.with_suffix(".pdf"),
        "source_summary": source_path,
        "auprc_by_seed": auprc_path,
        "destinations_by_seed": destination_path,
        "description": description_path,
        "manifest": manifest_path,
    }


def _read_scoring_frame(
    run: RunInfo, condition: str, candidate_set: str
) -> pd.DataFrame:
    path = (
        run.root
        / "scoring"
        / condition
        / candidate_set
        / "cell_scores.parquet"
    )
    scores = pd.read_parquet(path)
    _require_columns(
        scores,
        ("cell_id", "method", "u", "prior_risk"),
        path,
    )
    scores["cell_id"] = _as_cell_ids(scores["cell_id"])
    if scores.duplicated(["cell_id", "method"]).any():
        raise HIHASupplementError(f"Duplicate method-cell rows in {path}.")
    numeric = ["u", "prior_risk"]
    for column in numeric:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    return scores


def _read_s3_frame(
    run: RunInfo,
    input_metadata: pd.DataFrame,
    candidate_set: str,
) -> pd.DataFrame:
    scores = _read_scoring_frame(
        run,
        "incomplete_reference",
        candidate_set,
    )
    scores = scores.merge(
        input_metadata,
        on="cell_id",
        how="left",
        validate="many_to_one",
    )
    if scores["AIFI_L2"].isna().any():
        raise HIHASupplementError(
            f"Scoring rows cannot be joined to AIFI_L2 metadata for {run.root}."
        )
    return scores


def _method_rows(scores: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = scores.loc[scores["method"].eq(method)].copy()
    if rows.empty:
        raise HIHASupplementError(f"Missing {method} rows in HIHA scoring artifacts.")
    return rows


def _read_s4_grid(
    detection_path: Path, shared_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detection = pd.read_csv(detection_path)
    shared = pd.read_csv(shared_path)
    detection_required = (
        "held_out_label",
        "seed",
        "condition_id",
        "method",
        "auprc",
        "tau",
        "alpha",
    )
    shared_required = (
        "held_out_label",
        "seed",
        "condition_id",
        "method",
        "coverage",
        "tau",
        "alpha",
    )
    _require_columns(detection, detection_required, detection_path)
    _require_columns(shared, shared_required, shared_path)
    detection = detection.loc[
        detection["condition_id"].eq("incomplete_reference")
    ].copy()
    shared = shared.loc[shared["condition_id"].eq("incomplete_reference")].copy()
    keys = ["held_out_label", "seed", "method", "tau", "alpha"]
    if detection.duplicated(keys).any():
        raise HIHASupplementError("Duplicate constant-tau detection grid keys.")
    if shared.duplicated(keys).any():
        raise HIHASupplementError("Duplicate constant-tau coverage grid keys.")
    detection["tau"] = pd.to_numeric(detection["tau"], errors="coerce")
    detection["alpha"] = pd.to_numeric(detection["alpha"], errors="coerce")
    shared["tau"] = pd.to_numeric(shared["tau"], errors="coerce")
    shared["alpha"] = pd.to_numeric(shared["alpha"], errors="coerce")
    grid = detection.merge(
        shared.loc[:, keys + ["coverage"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    expected = {
        (held_out, seed, tau, alpha)
        for held_out in EXPECTED_HELD_OUT
        for seed in range(1, 6)
        for tau in range(1, 10)
        for alpha in range(0, 11)
    }
    observed = {
        (row.held_out_label, int(row.seed), int(row.tau), int(row.alpha))
        for row in grid.itertuples()
        if row.held_out_label in EXPECTED_HELD_OUT
    }
    if observed != expected:
        raise HIHASupplementError(
            "Constant-tau grid does not contain all two-state, five-seed, "
            "tau=1..9, alpha=0..10 cells."
        )
    if not np.isfinite(grid[["auprc", "coverage"]].to_numpy(dtype=float)).all():
        raise HIHASupplementError("Non-finite values in the constant-tau grid.")
    return grid, grid.groupby(
        ["held_out_label", "tau", "alpha"], as_index=False
    )[["auprc", "coverage"]].mean()


def _matrix(
    summary: pd.DataFrame, held_out: str, quantity: str
) -> tuple[np.ndarray, list[float], list[float]]:
    subset = summary.loc[summary["held_out_label"].eq(held_out)]
    taus = list(range(1, 10))
    alphas = list(range(0, 11))
    matrix = (
        subset.pivot(index="tau", columns="alpha", values=quantity)
        .reindex(index=taus, columns=alphas)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(matrix).all():
        raise HIHASupplementError(
            f"Missing {quantity} values in constant-tau summary for {held_out}."
        )
    return matrix, taus, alphas


def _padded_limits(values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise HIHASupplementError("Cannot derive axis limits from empty values.")
    if symmetric:
        limit = max(1.0e-6, float(np.abs(finite).max()) * 1.08)
        return -limit, limit
    low = float(finite.min())
    high = float(finite.max())
    pad = max(1.0e-6, (high - low) * 0.05)
    return low - pad, high + pad


def _plot_s4(
    grid_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    absolute_values = grid_summary["auprc"].to_numpy(dtype=float)
    delta_frames = []
    coverage_values = grid_summary["coverage"].to_numpy(dtype=float)
    for held_out in EXPECTED_HELD_OUT:
        absolute, _, _ = _matrix(grid_summary, held_out, "auprc")
        coverage, _, _ = _matrix(grid_summary, held_out, "coverage")
        delta_frames.append(absolute - absolute[:, [0]])
        coverage_values = np.concatenate([coverage_values, coverage.ravel()])
    absolute_limits = _padded_limits(absolute_values)
    delta_limits = _padded_limits(np.concatenate([frame.ravel() for frame in delta_frames]), symmetric=True)
    coverage_limits = _padded_limits(coverage_values)
    figure = plt.figure(figsize=(13.5, 7.2))
    outer = figure.add_gridspec(2, 3, wspace=0.42, hspace=0.62)
    panel_titles = (
        "Global all-query AUPRC",
        "AUPRC difference from alpha=0",
        "Shared-cell coverage",
    )

    for row, held_out in enumerate(EXPECTED_HELD_OUT):
        absolute, taus, alphas = _matrix(grid_summary, held_out, "auprc")
        coverage, _, _ = _matrix(grid_summary, held_out, "coverage")
        delta = absolute - absolute[:, [0]]
        heat_specs = (
            (absolute, "viridis", absolute_limits, False),
            (delta, "RdBu_r", delta_limits, True),
            (coverage, "YlGnBu", coverage_limits, False),
        )
        for col, (values, cmap, limits, centered) in enumerate(heat_specs):
            axis = figure.add_subplot(outer[row, col])
            image = axis.imshow(
                values,
                vmin=limits[0],
                vmax=limits[1],
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
            )
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    axis.text(
                        j,
                        i,
                        f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color="#111111" if values[i, j] < np.mean(limits) else "white",
                    )
            axis.set_xticks(range(len(alphas)), [str(alpha) for alpha in alphas])
            axis.set_yticks(range(len(taus)), [str(tau) for tau in taus])
            axis.set_xlabel("alpha" + (" (0 = uniform UOT)" if col == 0 else ""))
            axis.set_ylabel("tau" if col == 0 else "")
            axis.set_title(
                f"S4{chr(65 + row * 3 + col)}  {held_out}\n{panel_titles[col]}",
                loc="left",
                weight="bold",
            )
            axis.tick_params(length=0, labelsize=6)
            axis.spines[:].set_visible(False)
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            colorbar.ax.tick_params(labelsize=6)
            if col == 0:
                colorbar.set_label("Mean AUPRC", fontsize=7)
            elif col == 1:
                colorbar.set_label("Delta AUPRC", fontsize=7)
            else:
                colorbar.set_label("Coverage", fontsize=7)
            selected = (2, 2) if held_out == "HLA-DRhi cDC2" else (3, 1)
            selected_tau, selected_alpha = selected
            axis.add_patch(
                plt.Rectangle(
                    (selected_alpha - 0.5, selected_tau - 1 - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.4,
                )
            )

    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_hiha_s4(
    *,
    grid_root: Path = Path("results/HIHA_DC/sensitivity/constant_tau_alpha"),
    output_root: Path = Path("results/HIHA_DC/figures"),
) -> dict[str, Path]:
    grid, grid_summary = _read_s4_grid(
        grid_root / "tables" / "detection_by_run.csv",
        grid_root / "tables" / "shared_label_transfer_by_run.csv",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    grid_path = data_root / "supplementary_figure_s4_grid_by_seed.csv"
    grid_summary_path = data_root / "supplementary_figure_s4_grid_summary.csv"
    grid.to_csv(grid_path, index=False)
    grid_summary.to_csv(grid_summary_path, index=False)

    figure_path = output_root / "supplementary_figure_s4_anchor_weight_sensitivity.png"
    _plot_s4(grid_summary, figure_path)
    description_path = output_root / "supplementary_figure_s4_anchor_weight_sensitivity.md"
    description_path.write_text(
        "# Supplementary Figure S4\n\n"
        "State-dependent anchor-weight sensitivity within the compatibility-only "
        "constant-tau family. The heatmaps show descriptive five-split means for "
        "global all-query u-based AP, the paired difference from geometry-only "
        "Uniform UOT at the same tau (alpha=0), and shared-cell coverage. This "
        "grid does not represent the heterogeneous full CoRe-OT model. Black "
        "outlines mark the endpoint-specific compatibility-only reference "
        "settings used in the component table.\n\n"
        f"Grid by seed: {grid_path}\n\n"
        f"Grid summary: {grid_summary_path}\n",
        encoding="utf-8",
    )
    manifest_path = output_root / "supplementary_figure_s4_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "figure": "S4",
                "grid": "constant_tau_alpha",
                "detection_scope": "global_all_query",
                "held_out_labels": list(EXPECTED_HELD_OUT),
                "seeds": list(range(1, 6)),
                "outputs": {
                    "figure_png": str(figure_path),
                    "figure_pdf": str(figure_path.with_suffix(".pdf")),
                    "grid_by_seed": str(grid_path),
                    "grid_summary": str(grid_summary_path),
                    "description": str(description_path),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "figure_png": figure_path,
        "figure_pdf": figure_path.with_suffix(".pdf"),
        "grid_by_seed": grid_path,
        "grid_summary": grid_summary_path,
        "description": description_path,
        "manifest": manifest_path,
    }



S2_CONDITIONS = ("C00", "C01", "C10", "C11")
S2_REFERENCE_CONDITIONS = ("R00", "R10")
S2_CONDITION_LABELS = {
    "C00": "Mean-matched penalties\nalpha=0",
    "C01": "Heterogeneous penalties\nalpha=0",
    "C10": "Mean-matched penalties\nalpha=alpha*",
    "C11": "Heterogeneous penalties\nalpha=alpha*",
    "R00": "Geometry-only Uniform UOT",
    "R10": "Compatibility-only constant-tau CoRe-OT",
}
S2_CONDITION_COLORS = {
    "C00": "#555555",
    "C10": "#9E9E9E",
    "C01": "#5B9BD5",
    "C11": "#0072B2",
}
S2_EFFECTS = (
    (
        "delta_heterogeneity_alpha0",
        "Heterogeneity (alpha=0)",
        "C01",
        "C00",
    ),
    (
        "delta_heterogeneity_alpha_star",
        "Heterogeneity (alpha=alpha*)",
        "C11",
        "C10",
    ),
    (
        "delta_anchor_mean_matched",
        "Anchor (mean-matched)",
        "C10",
        "C00",
    ),
    (
        "delta_anchor_heterogeneous",
        "Anchor (heterogeneous)",
        "C11",
        "C01",
    ),
)
S2_INTERACTION = ("interaction", "Interaction")


def _s2_slug(held_out: str) -> str:
    if held_out == "HLA-DRhi cDC2":
        return "hladrhi_cdc2"
    if held_out == "ISG+ cDC2":
        return "isg_cdc2"
    raise HIHASupplementError(f"Unsupported S2 held-out state: {held_out}.")


def _s2_artifact_spec(
    held_out: str, seed: int, condition: str
) -> tuple[Path, str]:
    slug = _s2_slug(held_out)
    if condition == "C00":
        return (
            Path(f"runs/hiha_dc_{slug}_seed{seed}_mean_matched_alpha0"),
            "coreot_constant_tau",
        )
    if condition == "C10":
        return (
            Path(
                f"runs/hiha_dc_{slug}_seed{seed}_mean_matched_alpha_selected"
            ),
            "coreot_constant_tau",
        )
    if condition == "R00":
        return (
            Path(f"runs/hiha_dc_{slug}_seed{seed}_tau05_uniform"),
            "uniform_uot",
        )
    if condition == "R10":
        tau, alpha = (
            (2, 2) if held_out == "HLA-DRhi cDC2" else (3, 1)
        )
        return (
            Path(
                f"runs/hiha_dc_{slug}_seed{seed}_tau{tau}_alpha{alpha}"
                "_coreot_constant_tau"
            ),
            "coreot_constant_tau",
        )
    report_root = Path(
        f"runs/hiha_dc_{slug}_seed{seed}_report_leave_one_HIHA_DC"
    )
    if condition == "C01":
        return report_root, "coreot_match_only"
    if condition == "C11":
        return report_root, "coreot_full"
    raise HIHASupplementError(f"Unknown S2 condition: {condition}.")


def _read_s2_method_frame(
    root: Path,
    condition: str,
    candidate_set: str,
    method: str,
    input_metadata: pd.DataFrame,
    fallback_tau_source: float | None = None,
) -> pd.DataFrame:
    path = root / "scoring" / condition / candidate_set / "cell_scores.parquet"
    scores = pd.read_parquet(path)
    _require_columns(
        scores,
        (
            "cell_id",
            "method",
            "u",
            "label_entropy",
        ),
        path,
    )
    scores = scores.loc[scores["method"].eq(method)].copy()
    if scores.empty:
        raise HIHASupplementError(f"Method {method} is absent from {path}.")
    scores["cell_id"] = _as_cell_ids(scores["cell_id"])
    if scores["cell_id"].duplicated().any():
        raise HIHASupplementError(
            f"Duplicate {method} source IDs in {path}."
        )
    for column in ("u", "label_entropy"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if not np.isfinite(scores[["u", "label_entropy"]].to_numpy(dtype=float)).all():
        raise HIHASupplementError(f"Non-finite S2 scores in {path}.")

    transport_path = (
        root
        / "transport"
        / condition
        / candidate_set
        / method
        / "cell_transport_scores.parquet"
    )
    if transport_path.is_file():
        transport = pd.read_parquet(transport_path)
        _require_columns(
            transport,
            ("cell_id", "method", "a", "tau_source"),
            transport_path,
        )
        transport = transport.loc[
            transport["method"].eq(method),
            ["cell_id", "a", "tau_source"],
        ].copy()
        transport["cell_id"] = _as_cell_ids(transport["cell_id"])
        for column in ("a", "tau_source"):
            transport[column] = pd.to_numeric(transport[column], errors="coerce")
        if (
            transport["cell_id"].duplicated().any()
            or not np.isfinite(
                transport[["a", "tau_source"]].to_numpy(dtype=float)
            ).all()
            or transport["a"].le(0).any()
        ):
            raise HIHASupplementError(
                f"Invalid S2 source penalties or masses in {transport_path}."
            )
        scores = scores.merge(
            transport,
            on="cell_id",
            how="outer",
            validate="one_to_one",
        )
    elif fallback_tau_source is not None:
        scores["a"] = 1.0 / len(scores)
        scores["tau_source"] = float(fallback_tau_source)
    else:
        raise HIHASupplementError(
            f"Required S2 transport scores are absent: {transport_path}."
        )

    truth = _read_query_truth(root, condition)
    scores = scores.merge(
        truth,
        on="cell_id",
        how="outer",
        validate="one_to_one",
    )
    scores = scores.merge(
        input_metadata,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    if scores[["true_label", "removed_state", "AIFI_L2"]].isna().any().any():
        raise HIHASupplementError(
            f"S2 scores cannot be joined to truth and AIFI_L2 metadata for {path}."
        )
    return scores


def _weighted_tau_mean(frame: pd.DataFrame) -> float:
    return float(np.average(frame["tau_source"], weights=frame["a"]))


def _validate_s2_mean_matching(
    condition_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    held_out: str,
    seed: int,
) -> None:
    for reference_index, reference_label in enumerate(
        ("incomplete_reference", "full_reference_control")
    ):
        for mean_matched, heterogeneous in (("C00", "C01"), ("C10", "C11")):
            mean_value = _weighted_tau_mean(
                condition_frames[mean_matched][reference_index]
            )
            heterogeneous_value = _weighted_tau_mean(
                condition_frames[heterogeneous][reference_index]
            )
            if not np.isclose(mean_value, heterogeneous_value, atol=1.0e-10):
                raise HIHASupplementError(
                    "Mean-matched S2 source penalty differs from the "
                    f"heterogeneous-penalty mean for {held_out}, seed={seed}, "
                    f"{reference_label}, {mean_matched} versus {heterogeneous}: "
                    f"{mean_value} versus {heterogeneous_value}."
                )


def _s2_condition_frame(
    held_out: str,
    seed: int,
    condition_label: str,
    candidate_set: str,
    input_metadata: pd.DataFrame,
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root, method = _s2_artifact_spec(held_out, seed, condition_label)
    root = runs_root / root.relative_to("runs")
    fallback_tau_source = None
    if condition_label == "R00":
        fallback_tau_source = 0.5
    elif condition_label == "R10":
        fallback_tau_source = 2.0 if held_out == "HLA-DRhi cDC2" else 3.0
    incomplete = _read_s2_method_frame(
        root,
        "incomplete_reference",
        candidate_set,
        method,
        input_metadata,
        fallback_tau_source,
    )
    full = _read_s2_method_frame(
        root,
        "full_reference_control",
        candidate_set,
        method,
        input_metadata,
        fallback_tau_source,
    )
    if set(incomplete["cell_id"]) != set(full["cell_id"]):
        raise HIHASupplementError(
            f"Paired S2 source IDs differ for {held_out}, seed={seed}, "
            f"condition={condition_label}."
        )
    return incomplete, full


def _s2_metrics(
    incomplete: pd.DataFrame,
    full: pd.DataFrame,
    held_out: str,
    seed: int,
    condition: str,
    theta_h: float,
) -> dict[str, object]:
    if incomplete["cell_id"].duplicated().any() or full["cell_id"].duplicated().any():
        raise HIHASupplementError("Duplicate S2 source IDs.")
    threshold = float(full["u"].quantile(0.95))
    incomplete = incomplete.copy()
    incomplete["abstain"] = (
        incomplete["u"].gt(threshold)
        | incomplete["label_entropy"].gt(theta_h)
    )
    held_mask = incomplete["true_label"].eq(held_out)
    shared_mask = ~held_mask
    local = incomplete.loc[incomplete["AIFI_L2"].eq("cDC2")]
    labels = local["true_label"].eq(held_out).to_numpy(dtype=bool)
    if labels.sum() == 0 or labels.all():
        raise HIHASupplementError(
            f"Invalid S2 within-cDC2 cohort for {held_out}, seed={seed}, {condition}."
        )
    return {
        "held_out_label": held_out,
        "seed": seed,
        "condition": condition,
        "auprc": _average_precision(
            labels,
            local["u"].to_numpy(dtype=float),
        ),
        "prevalence": float(labels.mean()),
        "threshold_u_95": threshold,
        "theta_h": float(theta_h),
        "mean_tau_source_incomplete": _weighted_tau_mean(incomplete),
        "mean_tau_source_full": _weighted_tau_mean(full),
        "held_out_abstention": float(incomplete.loc[held_mask, "abstain"].mean()),
        "shared_coverage": float((~incomplete.loc[shared_mask, "abstain"]).mean()),
        "n_held_out": int(held_mask.sum()),
        "n_shared": int(shared_mask.sum()),
    }


def _s2_effect_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for held_out in EXPECTED_HELD_OUT:
        for seed in range(1, 6):
            subset = metrics.loc[
                metrics["held_out_label"].eq(held_out)
                & metrics["seed"].eq(seed)
            ].set_index("condition")
            for effect, label, positive, negative in S2_EFFECTS:
                rows.append(
                    {
                        "held_out_label": held_out,
                        "seed": seed,
                        "effect": effect,
                        "effect_label": label,
                        "delta_auprc": float(
                            subset.loc[positive, "auprc"]
                            - subset.loc[negative, "auprc"]
                        ),
                    }
                )
            heterogeneity_alpha0 = float(
                subset.loc["C01", "auprc"] - subset.loc["C00", "auprc"]
            )
            heterogeneity_alpha_star = float(
                subset.loc["C11", "auprc"] - subset.loc["C10", "auprc"]
            )
            rows.append(
                {
                    "held_out_label": held_out,
                    "seed": seed,
                    "effect": S2_INTERACTION[0],
                    "effect_label": S2_INTERACTION[1],
                    "delta_auprc": (
                        heterogeneity_alpha_star - heterogeneity_alpha0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_s2(
    metrics: pd.DataFrame,
    effects: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure = plt.figure(figsize=(14.5, 7.4))
    outer = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.35, 1.35, 1.0, 1.0),
        wspace=0.72,
        hspace=0.58,
    )

    schematic_axis = figure.add_subplot(outer[0, :2])
    schematic_axis.axis("off")
    table = schematic_axis.table(
        cellText=[
            [
                "C00\nMean-matched\nuniform penalties",
                "C01\nHeterogeneous\npenalties",
            ],
            [
                "C10\nMean-matched\nuniform penalties",
                "C11\nHeterogeneous\npenalties",
            ],
        ],
        rowLabels=["alpha = 0", "alpha = alpha*"],
        colLabels=["Mean-matched query penalties", "Heterogeneous query penalties"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        colWidths=[0.40, 0.46],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#BDBDBD")
        cell.set_linewidth(0.7)
        cell.set_height(0.23)
    schematic_axis.set_title(
        "S2A  Component design", loc="left", weight="bold", pad=4
    )

    def plot_condition_auprc(axis: plt.Axes, held_out: str, panel: str) -> None:
        subset = metrics.loc[metrics["held_out_label"].eq(held_out)]
        for position, condition in enumerate(S2_CONDITIONS):
            values = subset.loc[
                subset["condition"].eq(condition), "auprc"
            ].to_numpy(dtype=float)
            axis.scatter(
                np.full(len(values), position),
                values,
                color=S2_CONDITION_COLORS[condition],
                alpha=0.5,
                s=18,
                edgecolor="none",
            )
            axis.scatter(
                position,
                values.mean(),
                color=S2_CONDITION_COLORS[condition],
                s=52,
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
        axis.axhline(
            float(subset["prevalence"].mean()),
            color="#555555",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
        )
        axis.set_xticks(
            range(len(S2_CONDITIONS)),
            list(S2_CONDITIONS),
        )
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Within-cDC2 AUPRC")
        axis.set_title(f"{panel}  {held_out}", loc="left", weight="bold")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    hla_axis = figure.add_subplot(outer[0, 2])
    plot_condition_auprc(hla_axis, "HLA-DRhi cDC2", "S2B")
    isg_axis = figure.add_subplot(outer[0, 3])
    plot_condition_auprc(isg_axis, "ISG+ cDC2", "S2C")

    effect_limits = _padded_limits(
        effects["delta_auprc"].to_numpy(dtype=float), symmetric=True
    )
    for row, held_out in enumerate(EXPECTED_HELD_OUT):
        axis = figure.add_subplot(outer[1, row])
        subset = effects.loc[effects["held_out_label"].eq(held_out)]
        effect_labels = [
            (effect, label) for effect, label, _, _ in S2_EFFECTS
        ] + [S2_INTERACTION]
        for position, (effect, label) in enumerate(effect_labels):
            values = subset.loc[
                subset["effect"].eq(effect), "delta_auprc"
            ].to_numpy(dtype=float)
            axis.scatter(
                values,
                np.full(len(values), position),
                color="#777777",
                alpha=0.55,
                s=14,
                edgecolor="none",
            )
            axis.scatter(
                values.mean(),
                position,
                color="#0072B2",
                s=42,
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_xlim(*effect_limits)
        axis.set_yticks(
            range(len(effect_labels)),
            [label for _, label in effect_labels],
        )
        axis.set_xlabel("Delta AUPRC")
        axis.set_title(
            (
                f"S2D  {held_out}"
                if row == 0
                else f"S2E  {held_out}"
            ),
            loc="left",
            weight="bold",
            fontsize=8,
        )
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    factorial_metrics = metrics.loc[metrics["condition"].isin(S2_CONDITIONS)]
    trade_values = factorial_metrics[
        ["shared_coverage", "held_out_abstention"]
    ].to_numpy(dtype=float)
    x_limits = _padded_limits(trade_values[:, 0])
    y_limits = _padded_limits(trade_values[:, 1])
    for row, held_out in enumerate(EXPECTED_HELD_OUT):
        axis = figure.add_subplot(outer[1, row + 2])
        subset = metrics.loc[metrics["held_out_label"].eq(held_out)]
        for condition in S2_CONDITIONS:
            values = subset.loc[subset["condition"].eq(condition)]
            axis.scatter(
                values["shared_coverage"],
                values["held_out_abstention"],
                color=S2_CONDITION_COLORS[condition],
                alpha=0.48,
                s=18,
                edgecolor="none",
                label=condition,
            )
            axis.scatter(
                values["shared_coverage"].mean(),
                values["held_out_abstention"].mean(),
                color=S2_CONDITION_COLORS[condition],
                s=48,
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("Shared-cell coverage")
        axis.set_ylabel("Held-out-state abstention")
        axis.set_title(
            (
                f"S2F  {held_out}"
                if row == 0
                else held_out
            ),
            loc="left",
            weight="bold",
            fontsize=8,
        )
        axis.grid(color="#E5E5E5", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        if row == 0:
            axis.legend(frameon=False, fontsize=6.5, loc="best")

    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_hiha_s2(
    *,
    runs_root: Path = Path("runs"),
    input_path: Path = Path(
        "data/derived/hiha_dc/"
        "human_immune_health_atlas_dc.with_recomputed_AIFI_L2_score.h5ad"
    ),
    output_root: Path = Path("results/HIHA_DC/figures"),
    candidate_set: str = DEFAULT_CANDIDATE_SET,
    theta_h: float = 0.8,
) -> dict[str, Path]:
    if theta_h < 0:
        raise ValueError("theta_h must be non-negative.")
    input_metadata = _read_input_metadata(input_path)
    metrics_rows: list[dict[str, object]] = []
    for held_out in EXPECTED_HELD_OUT:
        for seed in range(1, 6):
            condition_frames = {}
            source_ids: set[str] | None = None
            source_groups: dict[str, str] | None = None
            for condition in (*S2_CONDITIONS, *S2_REFERENCE_CONDITIONS):
                incomplete, full = _s2_condition_frame(
                    held_out,
                    seed,
                    condition,
                    candidate_set,
                    input_metadata,
                    runs_root,
                )
                current_ids = set(incomplete["cell_id"])
                if source_ids is None:
                    source_ids = current_ids
                    source_groups = dict(
                        zip(
                            incomplete["cell_id"],
                            np.where(
                                incomplete["true_label"].eq(held_out),
                                "held_out",
                                np.where(
                                    incomplete["AIFI_L2"].eq("cDC2"),
                                    "other_cdc2",
                                    "other",
                                ),
                            ),
                            strict=True,
                        )
                    )
                elif current_ids != source_ids:
                    raise HIHASupplementError(
                        f"S2 source IDs differ across conditions for {held_out}, seed={seed}."
                    )
                current_groups = dict(
                    zip(
                        incomplete["cell_id"],
                        np.where(
                            incomplete["true_label"].eq(held_out),
                            "held_out",
                            np.where(
                                incomplete["AIFI_L2"].eq("cDC2"),
                                "other_cdc2",
                                "other",
                            ),
                        ),
                        strict=True,
                    )
                )
                if current_groups != source_groups:
                    raise HIHASupplementError(
                        f"S2 source groups differ across conditions for {held_out}, seed={seed}."
                    )
                condition_frames[condition] = (incomplete, full)
            _validate_s2_mean_matching(condition_frames, held_out, seed)
            for condition, (incomplete, full) in condition_frames.items():
                metrics_rows.append(
                    _s2_metrics(
                        incomplete,
                        full,
                        held_out,
                        seed,
                        condition,
                        theta_h,
                    )
                )

    metrics = pd.DataFrame(metrics_rows)
    metrics["condition_label"] = metrics["condition"].map(S2_CONDITION_LABELS)
    metrics["condition_role"] = np.where(
        metrics["condition"].isin(S2_CONDITIONS),
        "factorial",
        "reference",
    )
    effects = _s2_effect_rows(metrics)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    metrics_path = data_root / "supplementary_figure_s2_metrics_by_seed.csv"
    effects_path = data_root / "supplementary_figure_s2_component_effects_by_seed.csv"
    metrics.to_csv(metrics_path, index=False)
    effects.to_csv(effects_path, index=False)

    figure_path = output_root / "supplementary_figure_s2_component_attribution.png"
    _plot_s2(metrics, effects, figure_path)
    description_path = output_root / "supplementary_figure_s2_component_attribution.md"
    description_path.write_text(
        "# Supplementary Figure S2\n\n"
        "Mean-matched component attribution for query-penalty heterogeneity "
        "and broad-anchor compatibility. C00 and C10 replace each condition's "
        "heterogeneous query penalties by their empirical-mass-weighted mean; "
        "C01 and C11 retain the heterogeneous penalties. C00 and C01 set "
        "alpha=0, whereas C10 and C11 use the endpoint-specific selected "
        "anchor weight. Horizontal contrasts quantify penalty heterogeneity, "
        "vertical contrasts quantify anchor compatibility, and the difference "
        "of horizontal contrasts is reported as a descriptive interaction. "
        "The source table additionally reports geometry-only Uniform UOT and "
        "compatibility-only constant-tau CoRe-OT as nonfactorial reference "
        "conditions. "
        "All detection and operational metrics use query-marginal deficit u. "
        "Each condition's "
        "abstention threshold is the 95th percentile of its own paired "
        "restored-reference-condition u distribution, with the normalized label-entropy "
        "cutoff fixed at theta_H=0.8.\n\n"
        f"Metrics by seed: {metrics_path}\n\n"
        f"Component effects by seed: {effects_path}\n",
        encoding="utf-8",
    )
    manifest_path = output_root / "supplementary_figure_s2_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "figure": "S2",
                "score": "raw_source_marginal_deficit_u",
                "conditions": list(S2_CONDITIONS),
                "reference_conditions": list(S2_REFERENCE_CONDITIONS),
                "held_out_labels": list(EXPECTED_HELD_OUT),
                "seeds": list(range(1, 6)),
                "theta_u_quantile": 0.95,
                "theta_h": float(theta_h),
                "factorial": {
                    "columns": [
                        "mean_matched_uniform_query_penalties",
                        "heterogeneous_query_penalties",
                    ],
                    "rows": ["alpha_0", "alpha_selected"],
                    "interaction": "(C11-C10)-(C01-C00)",
                },
                "outputs": {
                    "figure_png": str(figure_path),
                    "figure_pdf": str(figure_path.with_suffix(".pdf")),
                    "metrics_by_seed": str(metrics_path),
                    "component_effects_by_seed": str(effects_path),
                    "description": str(description_path),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "figure_png": figure_path,
        "figure_pdf": figure_path.with_suffix(".pdf"),
        "metrics_by_seed": metrics_path,
        "component_effects_by_seed": effects_path,
        "description": description_path,
        "manifest": manifest_path,
    }
