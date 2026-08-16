from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


EXPECTED_LABELS = ("B cells", "NK cells", "Dendritic cells", "CD8 T cells")
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
EXPECTED_ALPHA = {
    "B cells": 4.0,
    "NK cells": 2.0,
    "Dendritic cells": 3.0,
    "CD8 T cells": 4.0,
}
CURRENT_OPERATING_POINTS = {
    "B cells": (0.5, 1.0),
    "NK cells": (0.5, 1.0),
    "Dendritic cells": (0.75, 1.0),
    "CD8 T cells": (0.5, 1.5),
}

STATE_SHORT_NAMES = {
    "B cells": "B",
    "NK cells": "NK",
    "Dendritic cells": "DC",
    "CD8 T cells": "CD8",
}
STATE_COLORS = {
    "B cells": "#0072B2",
    "NK cells": "#009E73",
    "Dendritic cells": "#D55E00",
    "CD8 T cells": "#CC79A7",
}
STRATEGY_ORDER = (
    "state_specific",
    "common_tau_range",
    "leave_one_state_out",
)
STRATEGY_LABELS = {
    "state_specific": "State-specific selected",
    "common_tau_range": "Common tau range,\nstate-specific alpha",
    "leave_one_state_out": "Leave-one-state-out\ncommon tau range",
}
STRATEGY_COLORS = {
    "state_specific": "#006D77",
    "common_tau_range": "#E69F00",
    "leave_one_state_out": "#4D4D4D",
}
STRATEGY_MARKERS = {
    "state_specific": "o",
    "common_tau_range": "s",
    "leave_one_state_out": "D",
}

CONFIG_KEYS = ["held_out_label", "seed", "tau_min", "tau_max", "alpha"]


@dataclass(frozen=True)
class PBMCS5Paths:
    output_root: Path
    data_root: Path
    by_seed: Path
    summary: Path
    candidates: Path
    selected_configurations: Path
    png: Path
    pdf: Path
    svg: Path
    description: Path


class PBMCS5Error(ValueError):
    pass


def _paths(output_root: Path) -> PBMCS5Paths:
    data_root = output_root / "data"
    return PBMCS5Paths(
        output_root=output_root,
        data_root=data_root,
        by_seed=data_root / "figure_s5_parameter_selection_by_seed.csv",
        summary=data_root / "figure_s5_parameter_selection_summary.csv",
        candidates=data_root / "figure_s5_parameter_selection_candidates.csv",
        selected_configurations=data_root
        / "figure_s5_parameter_selection_configurations.csv",
        png=output_root / "figure_s5_pbmc_parameter_selection.png",
        pdf=output_root / "figure_s5_pbmc_parameter_selection.pdf",
        svg=output_root / "figure_s5_pbmc_parameter_selection.svg",
        description=output_root / "figure_s5_pbmc_parameter_selection.md",
    )


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PBMCS5Error(f"{source} is missing columns {missing}.")


def _sem(values: pd.Series) -> float:
    numeric = values.to_numpy(dtype=float)
    if numeric.size <= 1:
        return 0.0
    return float(np.std(numeric, ddof=1) / np.sqrt(numeric.size))


def _filter_primary_rows(
    frame: pd.DataFrame,
    *,
    source: Path,
    require_detection_columns: bool,
) -> pd.DataFrame:
    required = {
        "held_out_label",
        "seed",
        "condition_id",
        "candidate_set",
        "method",
        "score",
        "tau_min",
        "tau_max",
        "alpha",
    }
    if require_detection_columns:
        required |= {"auprc", "auprc_baseline", "absent_abstention_rate"}
    else:
        required |= {
            "coverage",
            "shared_false_abstention_rate",
            "forced_macro_f1",
        }
    _require_columns(frame, required, source)

    selected = frame.loc[
        (frame["condition_id"] == "incomplete_reference")
        & (frame["candidate_set"] == "pca30_k100")
        & (frame["method"] == "coreot_full")
        & (frame["score"] == "u")
    ].copy()
    if selected.empty:
        raise PBMCS5Error(f"{source} contains no incomplete-reference coreot_full/u rows.")
    return selected


def _validate_grid_coverage(
    frame: pd.DataFrame,
    *,
    source: Path,
    common_pairs: set[tuple[float, float]] | None = None,
) -> set[tuple[float, float]]:
    frame["seed"] = frame["seed"].astype(int)
    for column in ("tau_min", "tau_max", "alpha"):
        frame[column] = frame[column].astype(float)

    if set(frame["held_out_label"]) != set(EXPECTED_LABELS):
        raise PBMCS5Error(
            f"{source} must contain exactly {EXPECTED_LABELS}; "
            f"found {sorted(frame['held_out_label'].unique())}."
        )
    if set(frame["seed"]) != set(EXPECTED_SEEDS):
        raise PBMCS5Error(
            f"{source} must contain exactly seeds {EXPECTED_SEEDS}; "
            f"found {sorted(frame['seed'].unique())}."
        )
    if frame[CONFIG_KEYS].duplicated().any():
        raise PBMCS5Error(f"{source} contains duplicate state/seed/configuration rows.")
    if not np.isfinite(frame[["tau_min", "tau_max", "alpha"]].to_numpy()).all():
        raise PBMCS5Error(f"{source} contains non-finite parameter values.")
    if (frame["tau_min"] > frame["tau_max"]).any():
        raise PBMCS5Error(f"{source} contains tau_min values greater than tau_max.")

    for label in EXPECTED_LABELS:
        label_rows = frame.loc[frame["held_out_label"] == label]
        observed_alpha = label_rows["alpha"].unique()
        if observed_alpha.size != 1 or not np.isclose(
            observed_alpha[0], EXPECTED_ALPHA[label]
        ):
            raise PBMCS5Error(
                f"{source} has unexpected alpha for {label}: {observed_alpha.tolist()}."
            )
        for seed in EXPECTED_SEEDS:
            if int((label_rows["seed"] == seed).sum()) == 0:
                raise PBMCS5Error(f"{source} is missing {label}, seed {seed}.")

    pairs_by_label = {
        label: set(
            zip(
                frame.loc[frame["held_out_label"] == label, "tau_min"],
                frame.loc[frame["held_out_label"] == label, "tau_max"],
            )
        )
        for label in EXPECTED_LABELS
    }
    intersection = set.intersection(*pairs_by_label.values())
    if common_pairs is not None and intersection != common_pairs:
        raise PBMCS5Error(
            f"{source} has a different common tau-range intersection than detection data."
        )
    for label, pairs in pairs_by_label.items():
        for pair in pairs:
            n_rows = len(
                frame.loc[
                    (frame["held_out_label"] == label)
                    & np.isclose(frame["tau_min"], pair[0])
                    & np.isclose(frame["tau_max"], pair[1])
                ]
            )
            if n_rows != len(EXPECTED_SEEDS):
                raise PBMCS5Error(
                    f"{source} has {n_rows} rows for {label}, tau pair {pair}; "
                    f"expected {len(EXPECTED_SEEDS)}."
                )
    return intersection


def _read_metric_tables(
    detection_path: Path,
    shared_path: Path,
) -> tuple[pd.DataFrame, set[tuple[float, float]]]:
    detection = _filter_primary_rows(
        pd.read_csv(detection_path),
        source=detection_path,
        require_detection_columns=True,
    )
    common_pairs = _validate_grid_coverage(detection, source=detection_path)

    shared = _filter_primary_rows(
        pd.read_csv(shared_path),
        source=shared_path,
        require_detection_columns=False,
    )
    _validate_grid_coverage(shared, source=shared_path, common_pairs=common_pairs)

    shared_columns = CONFIG_KEYS + [
        "coverage",
        "shared_false_abstention_rate",
        "forced_macro_f1",
    ]
    merged = detection.merge(
        shared[shared_columns],
        on=CONFIG_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_shared"),
    )
    if len(merged) != len(detection):
        raise PBMCS5Error(
            "Detection and shared-label-transfer tables do not have identical "
            "state/seed/configuration coverage."
        )
    numeric_columns = [
        "auprc",
        "auprc_baseline",
        "absent_abstention_rate",
        "coverage",
        "shared_false_abstention_rate",
        "forced_macro_f1",
    ]
    if not np.isfinite(merged[numeric_columns].to_numpy(dtype=float)).all():
        raise PBMCS5Error("S5 input tables contain non-finite metric values.")
    denominator = 1.0 - merged["auprc_baseline"].to_numpy(dtype=float)
    if (denominator <= 0.0).any():
        raise PBMCS5Error("Prevalence-adjusted AP lift is undefined for prevalence >= 1.")
    merged["prevalence"] = merged["auprc_baseline"].astype(float)
    merged["ap_lift"] = (
        merged["auprc"].astype(float) - merged["prevalence"]
    ) / (1.0 - merged["prevalence"])
    merged["heldout_rejection"] = merged["absent_abstention_rate"].astype(float)
    merged["strategy"] = ""
    return merged, common_pairs


def _candidate_summary(
    frame: pd.DataFrame,
    selection_states: tuple[str, ...],
) -> pd.DataFrame:
    subset = frame.loc[frame["held_out_label"].isin(selection_states)].copy()
    state_group_keys = ["held_out_label", "tau_min", "tau_max"]
    state_counts = subset.groupby(state_group_keys, sort=True).size()
    if not (state_counts == len(EXPECTED_SEEDS)).all():
        raise PBMCS5Error(
            "Candidate selection requires all five donor splits for every state and tau pair."
        )
    state_means = (
        subset.groupby(state_group_keys, sort=True)
        .agg(
            state_mean_ap_lift=("ap_lift", "mean"),
            state_mean_auprc=("auprc", "mean"),
            state_mean_prevalence=("prevalence", "mean"),
            state_mean_coverage=("coverage", "mean"),
            state_mean_shared_false_abstention=(
                "shared_false_abstention_rate",
                "mean",
            ),
            state_mean_heldout_rejection=("heldout_rejection", "mean"),
            state_mean_forced_macro_f1=("forced_macro_f1", "mean"),
        )
        .reset_index()
    )
    candidates = (
        state_means.groupby(["tau_min", "tau_max"], sort=True)
        .agg(
            mean_ap_lift=("state_mean_ap_lift", "mean"),
            sem_ap_lift=("state_mean_ap_lift", _sem),
            mean_auprc=("state_mean_auprc", "mean"),
            mean_prevalence=("state_mean_prevalence", "mean"),
            mean_coverage=("state_mean_coverage", "mean"),
            mean_shared_false_abstention=(
                "state_mean_shared_false_abstention",
                "mean",
            ),
            mean_heldout_rejection=("state_mean_heldout_rejection", "mean"),
            mean_forced_macro_f1=("state_mean_forced_macro_f1", "mean"),
            n_states=("held_out_label", "nunique"),
        )
        .reset_index()
    )
    if set(candidates["n_states"]) != {len(selection_states)}:
        raise PBMCS5Error("Candidate state aggregation is incomplete.")
    return candidates


def _select_candidate(
    candidates: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    if candidates.empty:
        raise PBMCS5Error("No candidate configurations are available for selection.")
    ranked = candidates.copy()
    best_index = ranked["mean_ap_lift"].idxmax()
    best_mean = float(ranked.loc[best_index, "mean_ap_lift"])
    best_sem = float(ranked.loc[best_index, "sem_ap_lift"])
    threshold = best_mean - best_sem
    ranked["one_se_threshold"] = threshold
    ranked["within_one_se"] = ranked["mean_ap_lift"] >= threshold - 1e-12
    eligible = ranked.loc[ranked["within_one_se"]].copy()
    if eligible.empty:
        raise PBMCS5Error("One-standard-error selection produced no eligible candidates.")
    eligible = eligible.sort_values(
        ["mean_coverage", "mean_shared_false_abstention", "tau_min", "tau_max"],
        ascending=[False, True, True, True],
    )
    selected_pair = tuple(eligible.iloc[0][["tau_min", "tau_max"]].astype(float))
    ranked["selected"] = (
        np.isclose(ranked["tau_min"], selected_pair[0])
        & np.isclose(ranked["tau_max"], selected_pair[1])
    )
    return ranked.loc[ranked["selected"]].iloc[0], ranked


def _config_stats(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "selection_mean_ap_lift": float(frame["ap_lift"].mean()),
        "selection_sem_ap_lift": _sem(frame["ap_lift"]),
        "selection_mean_coverage": float(frame["coverage"].mean()),
        "selection_mean_shared_false_abstention": float(
            frame["shared_false_abstention_rate"].mean()
        ),
    }


def _config_rows_for_pair(
    frame: pd.DataFrame,
    *,
    strategy: str,
    evaluation_state: str,
    tau_min: float,
    tau_max: float,
    selection_states: tuple[str, ...],
    excluded_state: str,
    selection_metadata: dict[str, float],
    selection_rule: str,
) -> dict[str, object]:
    label_rows = frame.loc[
        (frame["held_out_label"] == evaluation_state)
        & np.isclose(frame["tau_min"], tau_min)
        & np.isclose(frame["tau_max"], tau_max)
    ]
    if len(label_rows) != len(EXPECTED_SEEDS):
        raise PBMCS5Error(
            f"Selected configuration does not have five seed rows for {evaluation_state}."
        )
    return {
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy].replace("\n", " "),
        "evaluation_state": evaluation_state,
        "excluded_state": excluded_state,
        "selection_states": "; ".join(selection_states),
        "tau_min": float(tau_min),
        "tau_max": float(tau_max),
        "alpha": float(EXPECTED_ALPHA[evaluation_state]),
        "selection_rule": selection_rule,
        **selection_metadata,
    }


def _build_selection_outputs(
    frame: pd.DataFrame,
    common_pairs: set[tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []

    for label in EXPECTED_LABELS:
        tau_min, tau_max = CURRENT_OPERATING_POINTS[label]
        selected = frame.loc[
            (frame["held_out_label"] == label)
            & np.isclose(frame["tau_min"], tau_min)
            & np.isclose(frame["tau_max"], tau_max)
        ]
        if len(selected) != len(EXPECTED_SEEDS):
            raise PBMCS5Error(
                f"Current operating point is absent for {label}: {(tau_min, tau_max)}."
            )
        configs.append(
            _config_rows_for_pair(
                frame,
                strategy="state_specific",
                evaluation_state=label,
                tau_min=tau_min,
                tau_max=tau_max,
                selection_states=(label,),
                excluded_state="",
                selection_metadata=_config_stats(selected),
                selection_rule="current manuscript operating point",
            )
        )

    all_states = tuple(EXPECTED_LABELS)
    common_candidates = _candidate_summary(frame, all_states)
    common_selected, common_candidates = _select_candidate(common_candidates)
    common_candidates = common_candidates.assign(
        strategy="common_tau_range",
        evaluation_state="",
        excluded_state="",
        selection_states="; ".join(all_states),
    )
    candidate_frames.append(common_candidates)
    common_metadata = {
        "selection_mean_ap_lift": float(common_selected["mean_ap_lift"]),
        "selection_sem_ap_lift": float(common_selected["sem_ap_lift"]),
        "selection_mean_coverage": float(common_selected["mean_coverage"]),
        "selection_mean_shared_false_abstention": float(
            common_selected["mean_shared_false_abstention"]
        ),
        "one_se_threshold": float(common_selected["one_se_threshold"]),
    }
    for label in EXPECTED_LABELS:
        configs.append(
            _config_rows_for_pair(
                frame,
                strategy="common_tau_range",
                evaluation_state=label,
                tau_min=float(common_selected["tau_min"]),
                tau_max=float(common_selected["tau_max"]),
                selection_states=all_states,
                excluded_state="",
                selection_metadata=common_metadata,
                selection_rule="state-balanced common tau-range selection",
            )
        )

    for excluded_state in EXPECTED_LABELS:
        selection_states = tuple(
            label for label in EXPECTED_LABELS if label != excluded_state
        )
        loo_candidates = _candidate_summary(frame, selection_states)
        loo_selected, loo_candidates = _select_candidate(loo_candidates)
        loo_candidates = loo_candidates.assign(
            strategy="leave_one_state_out",
            evaluation_state=excluded_state,
            excluded_state=excluded_state,
            selection_states="; ".join(selection_states),
        )
        candidate_frames.append(loo_candidates)
        loo_metadata = {
            "selection_mean_ap_lift": float(loo_selected["mean_ap_lift"]),
            "selection_sem_ap_lift": float(loo_selected["sem_ap_lift"]),
            "selection_mean_coverage": float(loo_selected["mean_coverage"]),
            "selection_mean_shared_false_abstention": float(
                loo_selected["mean_shared_false_abstention"]
            ),
            "one_se_threshold": float(loo_selected["one_se_threshold"]),
        }
        configs.append(
            _config_rows_for_pair(
                frame,
                strategy="leave_one_state_out",
                evaluation_state=excluded_state,
                tau_min=float(loo_selected["tau_min"]),
                tau_max=float(loo_selected["tau_max"]),
                selection_states=selection_states,
                excluded_state=excluded_state,
                selection_metadata=loo_metadata,
                selection_rule="state-balanced leave-one-state-out tau-range selection",
            )
        )

    selected_configurations = pd.DataFrame(configs)
    candidate_table = pd.concat(candidate_frames, ignore_index=True)
    candidate_table["common_grid"] = True
    candidate_table["n_common_tau_pairs"] = len(common_pairs)
    candidate_table = candidate_table.sort_values(
        ["strategy", "excluded_state", "tau_min", "tau_max"]
    ).reset_index(drop=True)

    by_seed_parts: list[pd.DataFrame] = []
    for config in configs:
        selected = frame.loc[
            (frame["held_out_label"] == config["evaluation_state"])
            & np.isclose(frame["tau_min"], float(config["tau_min"]))
            & np.isclose(frame["tau_max"], float(config["tau_max"]))
        ].copy()
        selected["strategy"] = config["strategy"]
        selected["strategy_label"] = config["strategy_label"]
        selected["evaluation_state"] = config["evaluation_state"]
        selected["selected_tau_min"] = float(config["tau_min"])
        selected["selected_tau_max"] = float(config["tau_max"])
        selected["selected_alpha"] = float(config["alpha"])
        selected["selection_states"] = config["selection_states"]
        selected["excluded_state"] = config["excluded_state"]
        selected["selection_mean_ap_lift"] = config["selection_mean_ap_lift"]
        selected["selection_sem_ap_lift"] = config["selection_sem_ap_lift"]
        selected["selection_mean_coverage"] = config["selection_mean_coverage"]
        selected["selection_mean_shared_false_abstention"] = config[
            "selection_mean_shared_false_abstention"
        ]
        selected["one_se_threshold"] = config.get("one_se_threshold", np.nan)
        by_seed_parts.append(selected)
    by_seed = pd.concat(by_seed_parts, ignore_index=True)

    current_auprc = (
        by_seed.loc[by_seed["strategy"] == "state_specific", [
            "evaluation_state",
            "seed",
            "auprc",
        ]]
        .rename(columns={"auprc": "current_state_specific_auprc"})
    )
    by_seed = by_seed.merge(
        current_auprc,
        on=["evaluation_state", "seed"],
        how="left",
        validate="many_to_one",
    )
    by_seed["delta_auprc"] = (
        by_seed["auprc"] - by_seed["current_state_specific_auprc"]
    )

    summary_rows: list[dict[str, object]] = []
    summary_metrics = (
        "auprc",
        "ap_lift",
        "prevalence",
        "coverage",
        "heldout_rejection",
        "shared_false_abstention_rate",
        "forced_macro_f1",
        "delta_auprc",
    )
    for (strategy, evaluation_state), group in by_seed.groupby(
        ["strategy", "evaluation_state"], sort=False
    ):
        row: dict[str, object] = {
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy].replace("\n", " "),
            "evaluation_state": evaluation_state,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in summary_metrics:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=1))
            row[f"sem_{metric}"] = _sem(group[metric])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["_strategy_order"] = summary["strategy"].map(
        {name: index for index, name in enumerate(STRATEGY_ORDER)}
    )
    summary["_state_order"] = summary["evaluation_state"].map(
        {name: index for index, name in enumerate(EXPECTED_LABELS)}
    )
    summary = summary.sort_values(["_strategy_order", "_state_order"]).drop(
        columns=["_strategy_order", "_state_order"]
    )
    by_seed["_strategy_order"] = by_seed["strategy"].map(
        {name: index for index, name in enumerate(STRATEGY_ORDER)}
    )
    by_seed["_state_order"] = by_seed["evaluation_state"].map(
        {name: index for index, name in enumerate(EXPECTED_LABELS)}
    )
    by_seed = by_seed.sort_values(
        ["_strategy_order", "_state_order", "seed"]
    ).drop(columns=["_strategy_order", "_state_order"])
    return by_seed, summary.reset_index(drop=True), candidate_table, selected_configurations


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.set_axisbelow(True)


def _plot_panel_a(
    axes: list[plt.Axes],
    by_seed: pd.DataFrame,
    y_limits: tuple[float, float],
) -> None:
    x_values = np.arange(len(STRATEGY_ORDER))
    for axis, state in zip(axes, EXPECTED_LABELS):
        state_frame = by_seed.loc[by_seed["evaluation_state"] == state]
        for seed, seed_frame in state_frame.groupby("seed", sort=True):
            ordered = seed_frame.set_index("strategy").reindex(STRATEGY_ORDER)
            axis.plot(
                x_values,
                ordered["auprc"].to_numpy(dtype=float),
                color="#BDBDBD",
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
            axis.scatter(
                x_values,
                ordered["auprc"].to_numpy(dtype=float),
                color=[STRATEGY_COLORS[name] for name in STRATEGY_ORDER],
                s=18,
                alpha=0.7,
                zorder=2,
            )
        means = (
            state_frame.groupby("strategy")["auprc"]
            .mean()
            .reindex(STRATEGY_ORDER)
        )
        axis.scatter(
            x_values,
            means.to_numpy(dtype=float),
            color=[STRATEGY_COLORS[name] for name in STRATEGY_ORDER],
            edgecolor="black",
            linewidth=0.7,
            s=58,
            zorder=3,
        )
        prevalence = (
            state_frame.drop_duplicates(["seed"])["prevalence"].mean()
        )
        axis.axhline(
            prevalence,
            color="#777777",
            linestyle="--",
            linewidth=0.9,
            label="Prevalence" if state == EXPECTED_LABELS[0] else None,
        )
        title = f"A. {state}" if state == EXPECTED_LABELS[0] else state
        axis.set_title(title, fontsize=9)
        axis.set_xticks(x_values)
        axis.set_xticklabels(
            ["State-specific", "Common tau\nrange", "Leave-one-\nstate-out"],
            fontsize=7,
        )
        axis.set_ylim(*y_limits)
        axis.set_ylabel("AUPRC" if state in (EXPECTED_LABELS[0], EXPECTED_LABELS[2]) else "")
        _style_axis(axis)


def _plot_panel_b(
    axes: list[plt.Axes],
    by_seed: pd.DataFrame,
    y_limits: tuple[float, float],
) -> None:
    alternatives = ("common_tau_range", "leave_one_state_out")
    x_values = np.arange(len(alternatives))
    for axis, state in zip(axes, EXPECTED_LABELS):
        state_frame = by_seed.loc[
            (by_seed["evaluation_state"] == state)
            & by_seed["strategy"].isin(alternatives)
        ]
        for seed, seed_frame in state_frame.groupby("seed", sort=True):
            ordered = seed_frame.set_index("strategy").reindex(alternatives)
            axis.plot(
                x_values,
                ordered["delta_auprc"].to_numpy(dtype=float),
                color="#BDBDBD",
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
            axis.scatter(
                x_values,
                ordered["delta_auprc"].to_numpy(dtype=float),
                color=[STRATEGY_COLORS[name] for name in alternatives],
                s=20,
                alpha=0.75,
                zorder=2,
            )
        means = (
            state_frame.groupby("strategy")["delta_auprc"]
            .mean()
            .reindex(alternatives)
        )
        axis.scatter(
            x_values,
            means.to_numpy(dtype=float),
            color=[STRATEGY_COLORS[name] for name in alternatives],
            edgecolor="black",
            linewidth=0.7,
            s=60,
            zorder=3,
        )
        axis.axhline(0.0, color="#333333", linewidth=0.9)
        title = f"B. {state}" if state == EXPECTED_LABELS[0] else state
        axis.set_title(title, fontsize=9)
        axis.set_xticks(x_values)
        axis.set_xticklabels(["Common tau\nrange", "Leave-one-\nstate-out"], fontsize=7)
        axis.set_ylim(*y_limits)
        axis.set_ylabel(
            "AUPRC difference\nfrom state-specific"
            if state in (EXPECTED_LABELS[0], EXPECTED_LABELS[2])
            else ""
        )
        _style_axis(axis)


def _plot_panel_c(axis: plt.Axes, by_seed: pd.DataFrame) -> None:
    axis.set_title("C. Operational behavior", loc="left", fontsize=10)
    for state in EXPECTED_LABELS:
        state_frame = by_seed.loc[by_seed["evaluation_state"] == state]
        for strategy in STRATEGY_ORDER:
            points = state_frame.loc[state_frame["strategy"] == strategy]
            axis.scatter(
                points["coverage"],
                points["heldout_rejection"],
                color=STATE_COLORS[state],
                marker=STRATEGY_MARKERS[strategy],
                s=25,
                alpha=0.4,
                linewidth=0.0,
            )
            means = points[["coverage", "heldout_rejection"]].mean()
            axis.scatter(
                [means["coverage"]],
                [means["heldout_rejection"]],
                color=STATE_COLORS[state],
                marker=STRATEGY_MARKERS[strategy],
                edgecolor="black",
                linewidth=0.8,
                s=75,
                zorder=4,
            )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Shared-state coverage")
    axis.set_ylabel("Held-out-state rejection")
    _style_axis(axis)
    strategy_handles = [
        Line2D(
            [0],
            [0],
            marker=STRATEGY_MARKERS[strategy],
            color="white",
            markerfacecolor="#666666",
            markeredgecolor="#333333",
            markersize=7,
            linestyle="None",
            label=STRATEGY_LABELS[strategy].replace("\n", " "),
        )
        for strategy in STRATEGY_ORDER
    ]
    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=STATE_COLORS[state],
            markeredgecolor=STATE_COLORS[state],
            markersize=7,
            linestyle="None",
            label=STATE_SHORT_NAMES[state],
        )
        for state in EXPECTED_LABELS
    ]
    legend_1 = axis.legend(
        handles=strategy_handles,
        loc="lower left",
        fontsize=7,
        frameon=False,
        title="Strategy",
        title_fontsize=8,
    )
    axis.add_artist(legend_1)
    axis.legend(
        handles=state_handles,
        loc="lower right",
        fontsize=7,
        frameon=False,
        title="Held-out state",
        title_fontsize=8,
    )


def _format_value(value: float) -> str:
    return f"{value:g}"


def _plot_panel_d(axis: plt.Axes, configurations: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    common = configurations.loc[
        configurations["strategy"] == "common_tau_range"
    ].iloc[0]
    rows.append(
        [
            "Common tau range",
            _format_value(float(common["tau_min"])),
            _format_value(float(common["tau_max"])),
            "; ".join(
                f"{STATE_SHORT_NAMES[label]}={_format_value(EXPECTED_ALPHA[label])}"
                for label in EXPECTED_LABELS
            ),
            "B, NK, DC, CD8",
        ]
    )
    for excluded_state in EXPECTED_LABELS:
        row = configurations.loc[
            (configurations["strategy"] == "leave_one_state_out")
            & (configurations["excluded_state"] == excluded_state)
        ].iloc[0]
        rows.append(
            [
                f"Leave-{STATE_SHORT_NAMES[excluded_state]}-out",
                _format_value(float(row["tau_min"])),
                _format_value(float(row["tau_max"])),
                "; ".join(
                    f"{STATE_SHORT_NAMES[label]}={_format_value(EXPECTED_ALPHA[label])}"
                    for label in EXPECTED_LABELS
                ),
                "; ".join(STATE_SHORT_NAMES[label] for label in EXPECTED_LABELS if label != excluded_state),
            ]
        )
    axis.axis("off")
    axis.set_title("D. Selected tau ranges and fixed alpha scope", loc="left", fontsize=10)
    table = axis.table(
        cellText=rows,
        colLabels=[
            "Selection",
            "tau_min",
            "tau_max",
            "alpha by state",
            "Selection states",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.22, 0.11, 0.11, 0.28, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.8)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if row_index == 0:
            cell.set_facecolor("#EDEDED")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#FAFAFA")


def _build_figure(
    by_seed: pd.DataFrame,
    configurations: pd.DataFrame,
) -> plt.Figure:
    all_panel_a_values = pd.concat(
        [
            by_seed["auprc"],
            by_seed["prevalence"],
        ],
        ignore_index=True,
    )
    panel_a_min = max(0.0, min(0.4, float(all_panel_a_values.min()) - 0.03))
    panel_a_max = min(1.0, float(all_panel_a_values.max()) + 0.03)
    if panel_a_max - panel_a_min < 0.1:
        panel_a_max = min(1.0, panel_a_min + 0.1)

    delta_limit = float(np.abs(by_seed["delta_auprc"]).max())
    delta_limit = max(0.02, delta_limit * 1.15)
    panel_b_limits = (-delta_limit, delta_limit)

    figure = plt.figure(figsize=(17, 11), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, wspace=0.25, hspace=0.28)
    panel_a_grid = outer[0, 0].subgridspec(2, 2, wspace=0.22, hspace=0.35)
    panel_b_grid = outer[0, 1].subgridspec(2, 2, wspace=0.22, hspace=0.35)
    panel_a_axes = [
        figure.add_subplot(panel_a_grid[row, column])
        for row in range(2)
        for column in range(2)
    ]
    panel_b_axes = [
        figure.add_subplot(panel_b_grid[row, column])
        for row in range(2)
        for column in range(2)
    ]
    panel_c_axis = figure.add_subplot(outer[1, 0])
    panel_d_axis = figure.add_subplot(outer[1, 1])

    _plot_panel_a(panel_a_axes, by_seed, (panel_a_min, panel_a_max))
    _plot_panel_b(panel_b_axes, by_seed, panel_b_limits)
    _plot_panel_c(panel_c_axis, by_seed)
    _plot_panel_d(panel_d_axis, configurations)

    return figure


def _description(
    *,
    detection_path: Path,
    shared_path: Path,
    paths: PBMCS5Paths,
) -> str:
    return "\n".join(
        [
            "# Supplementary Figure S5",
            "",
            "## Scope",
            "",
            "This figure compares the current state-specific operating points "
            "with a state-balanced common tau range and leave-one-state-out "
            "tau ranges. The completed grid fixes alpha by held-out state, so "
            "the common analyses retain state-specific alpha values and do "
            "not constitute fully state-agnostic parameter tuning.",
            "",
            "Query-marginal deficit u is the only score used. The "
            "secondary counterfactual score is excluded.",
            "",
            "## Selection rule",
            "",
            "For each candidate tau pair, AP lift is computed for every state "
            "and seed as (AUPRC - prevalence) / (1 - prevalence). Seed values "
            "are averaged within state, and state means receive equal weight. "
            "The one-standard-error eligibility threshold uses the four "
            "state means for the common analysis and the three training-state "
            "means for each leave-one-state-out analysis. Eligible candidates "
            "are ranked by descending mean shared coverage, ascending mean "
            "shared false abstention, tau_min, and tau_max.",
            "",
            "## Sources",
            "",
            f"Detection source: {detection_path}",
            f"Shared-label-transfer source: {shared_path}",
            f"Seed-level source data: {paths.by_seed}",
            f"Selection candidates: {paths.candidates}",
            f"Selected configurations: {paths.selected_configurations}",
            "",
            "The five donor splits are the quantitative display units. "
            "Variation across splits is descriptive and no inferential "
            "intervals or significance tests are shown.",
            "",
        ]
    )


def write_pbmc_s5_parameter_selection_figure(
    *,
    detection_path: Path = Path(
        "results/PBMC/sensitivity/full_tau_labelwise/tables/detection_by_run.csv"
    ),
    shared_path: Path = Path(
        "results/PBMC/sensitivity/full_tau_labelwise/tables/"
        "shared_label_transfer_by_run.csv"
    ),
    output_root: Path = Path("results/PBMC/figures"),
) -> PBMCS5Paths:
    paths = _paths(Path(output_root))
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.data_root.mkdir(parents=True, exist_ok=True)

    frame, common_pairs = _read_metric_tables(Path(detection_path), Path(shared_path))
    by_seed, summary, candidates, configurations = _build_selection_outputs(
        frame, common_pairs
    )
    by_seed.to_csv(paths.by_seed, index=False)
    summary.to_csv(paths.summary, index=False)
    candidates.to_csv(paths.candidates, index=False)
    configurations.to_csv(paths.selected_configurations, index=False)

    figure = _build_figure(by_seed, configurations)
    figure.savefig(paths.png, dpi=300, bbox_inches="tight")
    figure.savefig(paths.pdf, bbox_inches="tight")
    figure.savefig(paths.svg, bbox_inches="tight")
    plt.close(figure)
    paths.description.write_text(
        _description(
            detection_path=Path(detection_path),
            shared_path=Path(shared_path),
            paths=paths,
        ),
        encoding="utf-8",
    )
    return paths
