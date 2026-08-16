"""Generate the legacy four-endpoint PBMC matched-reference diagnostic.

The durable ``figure_s6_*`` filenames are retained for reproducibility. This
module does not generate the current manuscript Supplementary Figure S6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from coreot.results.matched_reference import (
    CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE,
    POOLED_TRANSPORTED_MASS_DESTINATION_COMPOSITION,
    control_adjusted_median_deficit_decrease,
)


EXPECTED_LABELS = ("B cells", "NK cells", "Dendritic cells", "CD8 T cells")
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
REFERENCE_ORDER = ("ablated", "full")
GROUP_ORDER = ("held_out_stimulated", "same_type_control")
DESTINATION_ORDER = (
    "same_type_stimulated",
    "same_type_control",
    "all_other_targets",
)
GROUP_LABELS = {
    "held_out_stimulated": "Held-out stimulated",
    "same_type_control": "Same-type control",
}
GROUP_COLORS = {
    "held_out_stimulated": "#E69F00",
    "same_type_control": "#7F7F7F",
}
STATE_COLORS = {
    "B cells": "#0072B2",
    "NK cells": "#009E73",
    "Dendritic cells": "#D55E00",
    "CD8 T cells": "#CC79A7",
}
STATE_SHORT_NAMES = {
    "B cells": "B",
    "NK cells": "NK",
    "Dendritic cells": "DC",
    "CD8 T cells": "CD8",
}
DESTINATION_LABELS = {
    "same_type_stimulated": "Same-type\nstimulated",
    "same_type_control": "Same-type\ncontrol",
    "all_other_targets": "All other\ntargets",
}


@dataclass(frozen=True)
class PBMCS6Paths:
    output_root: Path
    data_root: Path
    run_manifest: Path
    score_by_seed: Path
    score_summary: Path
    specificity_by_seed: Path
    mass_by_seed: Path
    mass_summary: Path
    destination_by_seed: Path
    destination_summary: Path
    png: Path
    pdf: Path
    svg: Path
    description: Path


class PBMCS6Error(ValueError):
    pass


def _paths(output_root: Path) -> PBMCS6Paths:
    data_root = output_root / "data"
    return PBMCS6Paths(
        output_root=output_root,
        data_root=data_root,
        run_manifest=data_root / "figure_s6_run_manifest.csv",
        score_by_seed=data_root / "figure_s6_score_rescue_by_seed.csv",
        score_summary=data_root / "figure_s6_score_rescue_summary.csv",
        specificity_by_seed=data_root
        / "figure_s6_restoration_specificity_by_seed.csv",
        mass_by_seed=data_root / "figure_s6_mass_rescue_by_seed.csv",
        mass_summary=data_root / "figure_s6_mass_rescue_summary.csv",
        destination_by_seed=data_root / "figure_s6_destination_by_seed.csv",
        destination_summary=data_root / "figure_s6_destination_summary.csv",
        png=output_root / "figure_s6_pbmc_matched_reference_rescue.png",
        pdf=output_root / "figure_s6_pbmc_matched_reference_rescue.pdf",
        svg=output_root / "figure_s6_pbmc_matched_reference_rescue.svg",
        description=output_root
        / "figure_s6_pbmc_matched_reference_rescue.md",
    )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PBMCS6Error(f"{source} is missing columns {missing}.")


def _normalize_condition(values: pd.Series, source: Path) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower().replace(
        {
            "control": "ctrl",
            "unstim": "ctrl",
            "unstimulated": "ctrl",
            "stimulated": "stim",
            "ifnb": "stim",
        }
    )
    unexpected = sorted(set(normalized) - {"ctrl", "stim"})
    if unexpected:
        raise PBMCS6Error(
            f"{source} contains unsupported condition labels {unexpected}."
        )
    return normalized


def _load_raw_metadata(raw_path: Path) -> pd.DataFrame:
    atlas = ad.read_h5ad(raw_path, backed="r")
    condition_column = "condition" if "condition" in atlas.obs else "label"
    if "cell_type" not in atlas.obs or condition_column not in atlas.obs:
        raise PBMCS6Error(
            f"{raw_path} must contain cell_type and condition/label metadata."
        )
    metadata = atlas.obs[["cell_type", condition_column]].copy()
    metadata.index = metadata.index.astype(str)
    metadata = metadata.rename(columns={condition_column: "condition"})
    metadata["cell_type"] = metadata["cell_type"].astype(str)
    metadata["condition"] = _normalize_condition(metadata["condition"], raw_path)
    if not metadata.index.is_unique:
        raise PBMCS6Error(f"{raw_path} contains duplicate cell identifiers.")
    if getattr(atlas, "file", None) is not None:
        atlas.file.close()
    return metadata


def _load_run_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, {"held_out_label", "seed", "run_id"}, manifest_path)
    selected = manifest[["held_out_label", "seed", "run_id"]].drop_duplicates()
    selected["seed"] = selected["seed"].astype(int)
    if selected[["held_out_label", "seed"]].duplicated().any():
        raise PBMCS6Error(
            f"{manifest_path} has duplicate held-out-state/seed run entries."
        )
    if set(selected["held_out_label"]) != set(EXPECTED_LABELS):
        raise PBMCS6Error(f"{manifest_path} does not cover the four S6 states.")
    if set(selected["seed"]) != set(EXPECTED_SEEDS):
        raise PBMCS6Error(f"{manifest_path} does not cover all five S6 seeds.")
    if len(selected) != len(EXPECTED_LABELS) * len(EXPECTED_SEEDS):
        raise PBMCS6Error(f"{manifest_path} does not contain exactly 20 selected runs.")
    if selected["run_id"].duplicated().any():
        raise PBMCS6Error(f"{manifest_path} reuses a run_id for multiple state/seed pairs.")
    return selected.sort_values(["held_out_label", "seed"]).reset_index(drop=True)


def _read_truth(
    truth_path: Path,
    raw_metadata: pd.DataFrame,
    held_out_label: str,
) -> pd.DataFrame:
    truth = pd.read_csv(truth_path)
    _require_columns(
        truth,
        {"cell_id", "true_label", "is_absent_state"},
        truth_path,
    )
    truth["cell_id"] = truth["cell_id"].astype(str)
    if not truth["cell_id"].is_unique:
        raise PBMCS6Error(f"{truth_path} contains duplicate cell identifiers.")
    joined = raw_metadata.reindex(truth["cell_id"])
    if joined.isna().any().any():
        raise PBMCS6Error(f"{truth_path} contains cells absent from raw metadata.")
    truth["cell_type"] = joined["cell_type"].to_numpy()
    truth["condition"] = joined["condition"].to_numpy()
    if not (
        truth["true_label"].astype(str) == truth["cell_type"].astype(str)
    ).all():
        raise PBMCS6Error(f"{truth_path} disagrees with raw cell-type metadata.")
    expected_positive = (
        (truth["cell_type"] == held_out_label)
        & (truth["condition"] == "stim")
    )
    if not np.array_equal(
        expected_positive.to_numpy(),
        truth["is_absent_state"].astype(bool).to_numpy(),
    ):
        raise PBMCS6Error(
            f"{truth_path} does not define the expected condition-specific positive state."
        )
    return truth


def _read_scores(
    score_path: Path,
    source_ids: set[str],
) -> pd.DataFrame:
    scores = pd.read_parquet(score_path)
    _require_columns(scores, {"cell_id", "method", "a", "a_hat", "u"}, score_path)
    scores = scores.loc[scores["method"] == "coreot_full"].copy()
    scores["cell_id"] = scores["cell_id"].astype(str)
    if not scores["cell_id"].is_unique:
        raise PBMCS6Error(f"{score_path} has duplicate coreot_full cell identifiers.")
    if set(scores["cell_id"]) != source_ids:
        raise PBMCS6Error(
            f"{score_path} cell identifiers do not match evaluation truth."
        )
    numeric = scores[["a", "a_hat", "u"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise PBMCS6Error(f"{score_path} contains non-finite transport scores.")
    if (scores["a"] <= 0.0).any() or (scores["a_hat"] < 0.0).any():
        raise PBMCS6Error(f"{score_path} contains invalid source masses.")
    return scores[["cell_id", "a", "a_hat", "u"]].copy()


def _read_full_truth_ids(
    truth_path: Path,
    incomplete_truth: pd.DataFrame,
) -> None:
    full_truth = pd.read_csv(truth_path)
    _require_columns(full_truth, {"cell_id", "true_label"}, truth_path)
    full_truth["cell_id"] = full_truth["cell_id"].astype(str)
    if set(full_truth["cell_id"]) != set(incomplete_truth["cell_id"]):
        raise PBMCS6Error(
            f"{truth_path} does not match the paired incomplete-reference source cells."
        )
    comparison = incomplete_truth[["cell_id", "true_label"]].merge(
        full_truth[["cell_id", "true_label"]],
        on="cell_id",
        how="outer",
        suffixes=("_incomplete", "_full"),
        validate="one_to_one",
    )
    if not (
        comparison["true_label_incomplete"].astype(str)
        == comparison["true_label_full"].astype(str)
    ).all():
        raise PBMCS6Error(f"{truth_path} disagrees with the paired source truth.")


def _read_run_pair(
    run_root: Path,
    held_out_label: str,
    raw_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    incomplete_truth_path = (
        run_root
        / "benchmark"
        / "incomplete_reference"
        / "evaluation_truth"
        / "query_truth.csv"
    )
    full_truth_path = (
        run_root
        / "benchmark"
        / "full_reference_control"
        / "evaluation_truth"
        / "query_truth.csv"
    )
    truth = _read_truth(incomplete_truth_path, raw_metadata, held_out_label)
    _read_full_truth_ids(full_truth_path, truth)
    source_ids = set(truth["cell_id"])
    score_frames: dict[str, pd.DataFrame] = {}
    for reference, condition in (
        ("ablated", "incomplete_reference"),
        ("full", "full_reference_control"),
    ):
        score_path = (
            run_root
            / "transport"
            / condition
            / "pca30_k100"
            / "coreot_full"
            / "cell_transport_scores.parquet"
        )
        scores = _read_scores(score_path, source_ids)
        score_frames[reference] = scores.rename(
            columns={
                "a": f"a_{reference}",
                "a_hat": f"a_hat_{reference}",
                "u": f"u_{reference}",
            }
        )
    paired = truth.merge(
        score_frames["ablated"],
        on="cell_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        score_frames["full"],
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(truth):
        raise PBMCS6Error(f"{run_root} lost source cells while pairing references.")
    paired["delta_u"] = paired["u_ablated"] - paired["u_full"]
    paired["m_ablated"] = paired["a_hat_ablated"] / paired["a_ablated"]
    paired["m_full"] = paired["a_hat_full"] / paired["a_full"]
    if not np.isfinite(
        paired[["delta_u", "m_ablated", "m_full"]].to_numpy(dtype=float)
    ).all():
        raise PBMCS6Error(f"{run_root} produced non-finite paired rescue values.")
    return paired, truth


def _destination_rows(
    *,
    run_root: Path,
    held_out_label: str,
    seed: int,
    paired: pd.DataFrame,
    raw_metadata: pd.DataFrame,
) -> pd.DataFrame:
    positive_ids = set(
        paired.loc[
            (paired["cell_type"] == held_out_label)
            & (paired["condition"] == "stim"),
            "cell_id",
        ]
    )
    if not positive_ids:
        raise PBMCS6Error(f"{run_root} has no held-out stimulated source cells.")
    rows: list[dict[str, object]] = []
    for reference, condition in (
        ("ablated", "incomplete_reference"),
        ("full", "full_reference_control"),
    ):
        coupling_path = (
            run_root
            / "transport"
            / condition
            / "pca30_k100"
            / "coreot_full"
            / "sparse_coupling.parquet"
        )
        coupling = pd.read_parquet(coupling_path)
        _require_columns(
            coupling,
            {"source_cell_id", "target_cell_id", "coupling"},
            coupling_path,
        )
        coupling["source_cell_id"] = coupling["source_cell_id"].astype(str)
        coupling["target_cell_id"] = coupling["target_cell_id"].astype(str)
        selected = coupling.loc[
            coupling["source_cell_id"].isin(positive_ids)
        ].copy()
        present_sources = set(selected["source_cell_id"])
        if present_sources != positive_ids:
            raise PBMCS6Error(
                f"{coupling_path} lacks positive-source coupling rows for "
                f"{sorted(positive_ids - present_sources)[:5]}."
            )
        if not np.isfinite(selected["coupling"].to_numpy(dtype=float)).all():
            raise PBMCS6Error(f"{coupling_path} contains non-finite coupling mass.")
        if (selected["coupling"] < 0.0).any():
            raise PBMCS6Error(f"{coupling_path} contains negative coupling mass.")
        target_metadata = raw_metadata.reindex(selected["target_cell_id"])
        if target_metadata.isna().any().any():
            raise PBMCS6Error(
                f"{coupling_path} contains target identifiers absent from raw metadata."
            )
        selected["target_cell_type"] = target_metadata["cell_type"].to_numpy()
        selected["target_condition"] = target_metadata["condition"].to_numpy()
        selected["destination_category"] = "all_other_targets"
        same_type = selected["target_cell_type"] == held_out_label
        selected.loc[
            same_type & (selected["target_condition"] == "stim"),
            "destination_category",
        ] = "same_type_stimulated"
        selected.loc[
            same_type & (selected["target_condition"] == "ctrl"),
            "destination_category",
        ] = "same_type_control"
        denominator = float(selected["coupling"].sum())
        expected_denominator = float(
            paired.loc[
                paired["cell_id"].isin(positive_ids),
                f"a_hat_{reference}",
            ].sum()
        )
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise PBMCS6Error(f"{coupling_path} has no positive transported mass.")
        if not np.isclose(
            denominator,
            expected_denominator,
            rtol=2e-5,
            atol=1e-10,
        ):
            raise PBMCS6Error(
                f"{coupling_path} mass does not match recorded a_hat for positive sources."
            )
        category_mass = selected.groupby("destination_category")["coupling"].sum()
        target_counts = selected.groupby("destination_category")[
            "target_cell_id"
        ].nunique()
        has_same_type_stim_target = bool(
            (
                (selected["target_cell_type"] == held_out_label)
                & (selected["target_condition"] == "stim")
            ).any()
        )
        for category in DESTINATION_ORDER:
            is_na = (
                reference == "ablated"
                and category == "same_type_stimulated"
                and not has_same_type_stim_target
            )
            rows.append(
                {
                    "held_out_label": held_out_label,
                    "seed": seed,
                    "run_id": run_root.name,
                    "reference": reference,
                    "destination_category": category,
                    "destination_estimand": (
                        POOLED_TRANSPORTED_MASS_DESTINATION_COMPOSITION
                    ),
                    "fraction": np.nan
                    if is_na
                    else float(category_mass.get(category, 0.0) / denominator),
                    "is_na": is_na,
                    "n_positive_sources": len(positive_ids),
                    "n_target_cells_in_category": int(target_counts.get(category, 0)),
                    "positive_transport_mass": denominator,
                }
            )
    return pd.DataFrame(rows)


def _summarize_seed_values(
    frame: pd.DataFrame,
    keys: list[str],
    value_column: str,
    output_value_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False):
        key_values = (key,) if not isinstance(key, tuple) else key
        row = dict(zip(keys, key_values))
        values = group[value_column].dropna().to_numpy(dtype=float)
        row[f"mean_{output_value_name}"] = (
            float(np.mean(values)) if len(values) else np.nan
        )
        row[f"std_{output_value_name}"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if len(values) == 1
            else np.nan
        )
        row[f"sem_{output_value_name}"] = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
            if len(values) == 1
            else np.nan
        )
        row["n_contributing_seeds"] = int(
            group.loc[group[value_column].notna(), "seed"].nunique()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _sort_state_frame(frame: pd.DataFrame, state_column: str = "held_out_label") -> pd.DataFrame:
    result = frame.copy()
    result["_state_order"] = result[state_column].map(
        {label: index for index, label in enumerate(EXPECTED_LABELS)}
    )
    if "reference" in result:
        result["_reference_order"] = result["reference"].map(
            {name: index for index, name in enumerate(REFERENCE_ORDER)}
        )
    else:
        result["_reference_order"] = 0
    result = result.sort_values(
        ["_state_order", "seed", "_reference_order"]
    ).drop(columns=["_state_order", "_reference_order"])
    return result


def _build_data(
    *,
    manifest: pd.DataFrame,
    runs_root: Path,
    raw_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    specificity_rows: list[dict[str, object]] = []
    mass_rows: list[dict[str, object]] = []
    destination_parts: list[pd.DataFrame] = []
    for entry in manifest.itertuples(index=False):
        run_root = runs_root / str(entry.run_id)
        paired, _ = _read_run_pair(run_root, entry.held_out_label, raw_metadata)
        heldout = (
            (paired["cell_type"] == entry.held_out_label)
            & (paired["condition"] == "stim")
        )
        control = (
            (paired["cell_type"] == entry.held_out_label)
            & (paired["condition"] == "ctrl")
        )
        if not heldout.any() or not control.any():
            raise PBMCS6Error(
                f"{run_root} lacks held-out stimulated or same-type control sources."
            )
        for group_name, mask in (
            ("held_out_stimulated", heldout),
            ("same_type_control", control),
        ):
            for reference in REFERENCE_ORDER:
                score_rows.append(
                    {
                        "held_out_label": entry.held_out_label,
                        "seed": int(entry.seed),
                        "run_id": entry.run_id,
                        "cell_group": group_name,
                        "reference": reference,
                        "n_cells": int(mask.sum()),
                        "median_u": float(paired.loc[mask, f"u_{reference}"].median()),
                    }
                )
        deficit_response = control_adjusted_median_deficit_decrease(
            incomplete=paired["u_ablated"],
            full=paired["u_full"],
            heldout=heldout,
            control=control,
        )
        specificity_rows.append(
            {
                "held_out_label": entry.held_out_label,
                "seed": int(entry.seed),
                "run_id": entry.run_id,
                "heldout_median_deficit_decrease": (
                    deficit_response.heldout_decrease
                ),
                "control_median_deficit_decrease": (
                    deficit_response.control_decrease
                ),
                "restoration_specificity": (
                    deficit_response.control_adjusted_decrease
                ),
                "restoration_specificity_estimand": (
                    CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE
                ),
            }
        )
        for reference in REFERENCE_ORDER:
            mass_rows.append(
                {
                    "held_out_label": entry.held_out_label,
                    "seed": int(entry.seed),
                    "run_id": entry.run_id,
                    "reference": reference,
                    "n_cells": int(heldout.sum()),
                    "median_m": float(paired.loc[heldout, f"m_{reference}"].median()),
                }
            )
        destination_parts.append(
            _destination_rows(
                run_root=run_root,
                held_out_label=entry.held_out_label,
                seed=int(entry.seed),
                paired=paired,
                raw_metadata=raw_metadata,
            )
        )
    score_by_seed = _sort_state_frame(pd.DataFrame(score_rows))
    specificity_by_seed = _sort_state_frame(pd.DataFrame(specificity_rows))
    mass_by_seed = _sort_state_frame(pd.DataFrame(mass_rows))
    destination_by_seed = _sort_state_frame(
        pd.concat(destination_parts, ignore_index=True)
    )
    score_summary = _summarize_seed_values(
        score_by_seed,
        ["held_out_label", "cell_group", "reference"],
        "median_u",
        "median_u",
    )
    mass_summary = _summarize_seed_values(
        mass_by_seed,
        ["held_out_label", "reference"],
        "median_m",
        "median_m",
    )
    destination_summary = _summarize_seed_values(
        destination_by_seed,
        [
            "held_out_label",
            "reference",
            "destination_category",
            "destination_estimand",
        ],
        "fraction",
        "fraction",
    )
    return (
        score_by_seed,
        score_summary,
        specificity_by_seed,
        mass_by_seed,
        mass_summary,
        destination_by_seed,
        destination_summary,
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.set_axisbelow(True)


def _plot_score_facets(
    axes: list[plt.Axes],
    score_by_seed: pd.DataFrame,
) -> None:
    x_values = np.arange(len(REFERENCE_ORDER))
    for axis, state in zip(axes, EXPECTED_LABELS):
        state_frame = score_by_seed.loc[
            score_by_seed["held_out_label"] == state
        ]
        for group_name in GROUP_ORDER:
            group_frame = state_frame.loc[
                state_frame["cell_group"] == group_name
            ]
            color = GROUP_COLORS[group_name]
            for seed, seed_frame in group_frame.groupby("seed", sort=True):
                ordered = seed_frame.set_index("reference").reindex(REFERENCE_ORDER)
                axis.plot(
                    x_values,
                    ordered["median_u"].to_numpy(dtype=float),
                    color=color,
                    linewidth=0.8,
                    alpha=0.55,
                )
                axis.scatter(
                    x_values,
                    ordered["median_u"].to_numpy(dtype=float),
                    s=18,
                    color=[color, "white"],
                    edgecolor=color,
                    linewidth=0.8,
                )
            means = (
                group_frame.groupby("reference")["median_u"]
                .mean()
                .reindex(REFERENCE_ORDER)
            )
            axis.scatter(
                x_values,
                means.to_numpy(dtype=float),
                s=58,
                color=[color, "white"],
                edgecolor="black",
                linewidth=0.8,
            )
        title = f"A. {state}" if state == EXPECTED_LABELS[0] else state
        axis.set_title(title, fontsize=9)
        axis.set_xticks(x_values)
        axis.set_xticklabels(["Ablated", "Full"], fontsize=8)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Median u" if state in (EXPECTED_LABELS[0], EXPECTED_LABELS[2]) else "")
        _style_axis(axis)
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                markerfacecolor=GROUP_COLORS[group],
                markeredgecolor=GROUP_COLORS[group],
                markersize=7,
                linestyle="None",
                label=GROUP_LABELS[group],
            )
            for group in GROUP_ORDER
        ],
        loc="upper left",
        fontsize=7,
        frameon=False,
    )


def _plot_specificity(axis: plt.Axes, specificity_by_seed: pd.DataFrame) -> None:
    x_values = np.arange(len(EXPECTED_LABELS))
    for index, state in enumerate(EXPECTED_LABELS):
        points = specificity_by_seed.loc[
            specificity_by_seed["held_out_label"] == state
        ]
        axis.scatter(
            np.full(len(points), index),
            points["restoration_specificity"],
            color=STATE_COLORS[state],
            s=28,
            alpha=0.65,
        )
        axis.scatter(
            [index],
            [points["restoration_specificity"].mean()],
            color=STATE_COLORS[state],
            edgecolor="black",
            linewidth=0.8,
            s=75,
            zorder=3,
        )
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_title("B. Control-adjusted deficit decrease", loc="left", fontsize=10)
    axis.set_xticks(x_values)
    axis.set_xticklabels(
        [STATE_SHORT_NAMES[state] for state in EXPECTED_LABELS]
    )
    axis.set_ylabel("Control-adjusted\nmedian-deficit decrease")
    _style_axis(axis)


def _plot_mass_facets(
    axes: list[plt.Axes],
    mass_by_seed: pd.DataFrame,
) -> None:
    x_values = np.arange(len(REFERENCE_ORDER))
    for axis, state in zip(axes, EXPECTED_LABELS):
        state_frame = mass_by_seed.loc[
            mass_by_seed["held_out_label"] == state
        ]
        for seed, seed_frame in state_frame.groupby("seed", sort=True):
            ordered = seed_frame.set_index("reference").reindex(REFERENCE_ORDER)
            axis.plot(
                x_values,
                ordered["median_m"].to_numpy(dtype=float),
                color=STATE_COLORS[state],
                linewidth=0.8,
                alpha=0.55,
            )
            axis.scatter(
                x_values,
                ordered["median_m"].to_numpy(dtype=float),
                s=20,
                color=[STATE_COLORS[state], "white"],
                edgecolor=STATE_COLORS[state],
                linewidth=0.8,
            )
        means = (
            state_frame.groupby("reference")["median_m"]
            .mean()
            .reindex(REFERENCE_ORDER)
        )
        axis.scatter(
            x_values,
            means.to_numpy(dtype=float),
            s=62,
            color=[STATE_COLORS[state], "white"],
            edgecolor="black",
            linewidth=0.8,
        )
        title = f"C. {state}" if state == EXPECTED_LABELS[0] else state
        axis.set_title(title, fontsize=9)
        axis.set_xticks(x_values)
        axis.set_xticklabels(["Ablated", "Full"], fontsize=8)
        axis.set_ylabel(
            "Median transported mass ratio m"
            if state in (EXPECTED_LABELS[0], EXPECTED_LABELS[2])
            else ""
        )
        values = state_frame["median_m"].to_numpy(dtype=float)
        upper = max(1.0, float(np.nanmax(values)) * 1.08)
        axis.set_ylim(0.0, upper)
        _style_axis(axis)


def _plot_destination(
    axis: plt.Axes,
    destination_summary: pd.DataFrame,
) -> None:
    row_keys = [
        (state, reference)
        for state in EXPECTED_LABELS
        for reference in REFERENCE_ORDER
    ]
    matrix = np.full((len(row_keys), len(DESTINATION_ORDER)), np.nan)
    for row_index, (state, reference) in enumerate(row_keys):
        for column_index, category in enumerate(DESTINATION_ORDER):
            match = destination_summary.loc[
                (destination_summary["held_out_label"] == state)
                & (destination_summary["reference"] == reference)
                & (destination_summary["destination_category"] == category)
            ]
            if len(match) != 1:
                raise PBMCS6Error(
                    f"Destination summary lacks one row for {state}, {reference}, {category}."
                )
            matrix[row_index, column_index] = float(match.iloc[0]["mean_fraction"])
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#E6E6E6")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_title(
        "D. Pooled transported-mass composition",
        loc="left",
        fontsize=10,
    )
    axis.set_xticks(np.arange(len(DESTINATION_ORDER)))
    axis.set_xticklabels(
        [DESTINATION_LABELS[category] for category in DESTINATION_ORDER],
        fontsize=8,
    )
    axis.set_yticks(np.arange(len(row_keys)))
    axis.set_yticklabels(
        [f"{STATE_SHORT_NAMES[state]} - {reference}" for state, reference in row_keys],
        fontsize=8,
    )
    for row_index, (state, reference) in enumerate(row_keys):
        for column_index, category in enumerate(DESTINATION_ORDER):
            value = matrix[row_index, column_index]
            text = "N/A" if np.isnan(value) else f"{value:.2f}"
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="black" if np.isnan(value) or value < 0.55 else "white",
            )
    axis.set_xlabel("Destination category")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Pooled transported-mass fraction")


def _build_figure(
    score_by_seed: pd.DataFrame,
    specificity_by_seed: pd.DataFrame,
    mass_by_seed: pd.DataFrame,
    destination_summary: pd.DataFrame,
) -> plt.Figure:
    figure = plt.figure(figsize=(17, 12), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, wspace=0.25, hspace=0.28)
    score_grid = outer[0, 0].subgridspec(2, 2, wspace=0.22, hspace=0.35)
    mass_grid = outer[1, 0].subgridspec(2, 2, wspace=0.22, hspace=0.35)
    score_axes = [
        figure.add_subplot(score_grid[row, column])
        for row in range(2)
        for column in range(2)
    ]
    mass_axes = [
        figure.add_subplot(mass_grid[row, column])
        for row in range(2)
        for column in range(2)
    ]
    specificity_axis = figure.add_subplot(outer[0, 1])
    destination_axis = figure.add_subplot(outer[1, 1])
    _plot_score_facets(score_axes, score_by_seed)
    _plot_specificity(specificity_axis, specificity_by_seed)
    _plot_mass_facets(mass_axes, mass_by_seed)
    _plot_destination(destination_axis, destination_summary)
    return figure


def _description(
    *,
    manifest_path: Path,
    runs_root: Path,
    raw_path: Path,
    paths: PBMCS6Paths,
) -> str:
    return "\n".join(
        [
            "# Legacy four-endpoint PBMC matched-reference diagnostic",
            "",
            "This pooled transported-mass analysis is retained as a legacy "
            "reproducibility artifact. It is not the current manuscript "
            "Supplementary Figure S6, which uses split-specific equal-cell "
            "mean conditional destinations for three retained endpoints.",
            "",
            "Matched incomplete- and full-reference runs were aggregated across "
            "the four selected held-out states and five donor splits.",
            "",
            "Panel A summarizes condition-specific group medians of "
            "query-marginal deficit u. Panel B reports the control-adjusted "
            "median-deficit decrease: the held-out-state decrease in group "
            "medians minus the same-type-control decrease in group medians. "
            "The restoration_specificity artifact column is retained only "
            "for schema compatibility. Panel C uses recorded source mass a "
            "and transported mass a_hat to calculate m=a_hat/a for held-out "
            "stimulated cells. Panel D pools sparse coupling mass before "
            "calculating transported-mass destination composition in three "
            "categories.",
            "",
            "The Figure 3 restoration table is used only as the selected-run "
            "manifest. Model-facing transport is not refit. Per-cell deficit "
            "and mass values come from "
            "transport/<condition>/pca30_k100/coreot_full/"
            "cell_transport_scores.parquet; destination composition comes "
            "from the corresponding sparse_coupling.parquet. The ablated "
            "same-type stimulated destination is N/A when that target state "
            "is absent, not a numerical zero.",
            "",
            f"Run manifest: {manifest_path}",
            f"Run artifacts: {runs_root}",
            f"Raw target metadata: {raw_path}",
            f"Score source data: {paths.score_by_seed}",
            f"Mass source data: {paths.mass_by_seed}",
            f"Destination source data: {paths.destination_by_seed}",
            "",
            "Donor split is the quantitative display unit. Cell-level values "
            "are summarized within split before averaging across splits.",
            "",
        ]
    )


def write_pbmc_s6_matched_reference_rescue_figure(
    *,
    manifest_path: Path = Path(
        "results/PBMC/figures/data/figure_3_restoration_by_seed.csv"
    ),
    runs_root: Path = Path("runs"),
    raw_path: Path = Path("data/raw/kang_2018.h5ad"),
    output_root: Path = Path("results/PBMC/figures"),
) -> PBMCS6Paths:
    paths = _paths(Path(output_root))
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.data_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_run_manifest(Path(manifest_path))
    raw_metadata = _load_raw_metadata(Path(raw_path))
    (
        score_by_seed,
        score_summary,
        specificity_by_seed,
        mass_by_seed,
        mass_summary,
        destination_by_seed,
        destination_summary,
    ) = _build_data(
        manifest=manifest,
        runs_root=Path(runs_root),
        raw_metadata=raw_metadata,
    )
    manifest.to_csv(paths.run_manifest, index=False)
    score_by_seed.to_csv(paths.score_by_seed, index=False)
    score_summary.to_csv(paths.score_summary, index=False)
    specificity_by_seed.to_csv(paths.specificity_by_seed, index=False)
    mass_by_seed.to_csv(paths.mass_by_seed, index=False)
    mass_summary.to_csv(paths.mass_summary, index=False)
    destination_by_seed.to_csv(paths.destination_by_seed, index=False)
    destination_summary.to_csv(paths.destination_summary, index=False)
    figure = _build_figure(
        score_by_seed,
        specificity_by_seed,
        mass_by_seed,
        destination_summary,
    )
    figure.savefig(paths.png, dpi=300, bbox_inches="tight")
    figure.savefig(paths.pdf, bbox_inches="tight")
    figure.savefig(paths.svg, bbox_inches="tight")
    plt.close(figure)
    paths.description.write_text(
        _description(
            manifest_path=Path(manifest_path),
            runs_root=Path(runs_root),
            raw_path=Path(raw_path),
            paths=paths,
        ),
        encoding="utf-8",
    )
    return paths
