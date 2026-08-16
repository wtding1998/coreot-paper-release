from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


EXPECTED_LABELS = ("B cells", "NK cells", "Dendritic cells", "CD8 T cells")
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
CONDITIONS = ("ctrl", "stim")
Q_VALUES = tuple(
    sorted([value / 100 for value in range(85, 100)] + [0.975, 0.995])
)
PRIMARY_QUANTILE = 0.95
SECONDARY_QUANTILES = (0.90, 0.975)
THETA_H = 0.8
MIN_CONTRIBUTING_SPLITS = 3
REASON_ORDER = ("retained", "score_only", "entropy_only", "both")
REASON_LABELS = {
    "retained": "Retained",
    "score_only": "Score only",
    "entropy_only": "Label entropy only",
    "both": "Both",
}
REASON_COLORS = {
    "retained": "#D9D9D9",
    "score_only": "#E69F00",
    "entropy_only": "#56B4E9",
    "both": "#CC79A7",
}
GROUP_LABELS = {
    "held_out_stimulated": "Held-out stimulated",
    "same_type_control": "Same-type control",
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
CELL_TYPE_ORDER = (
    "B cells",
    "NK cells",
    "Dendritic cells",
    "CD4 T cells",
    "CD8 T cells",
    "CD14+ Monocytes",
    "FCGR3A+ Monocytes",
    "Megakaryocytes",
)


@dataclass(frozen=True)
class PBMCS7Paths:
    output_root: Path
    data_root: Path
    run_manifest: Path
    thresholds_by_seed: Path
    calibration_by_seed: Path
    calibration_summary: Path
    reasons_by_seed: Path
    reasons_summary: Path
    cd8_offsets: Path
    shared_class_by_seed: Path
    shared_class_summary: Path
    png: Path
    pdf: Path
    svg: Path
    description: Path


class PBMCS7Error(ValueError):
    pass


def _paths(output_root: Path) -> PBMCS7Paths:
    data_root = output_root / "data"
    return PBMCS7Paths(
        output_root=output_root,
        data_root=data_root,
        run_manifest=data_root / "figure_s7_run_manifest.csv",
        thresholds_by_seed=data_root / "figure_s7_thresholds_by_seed.csv",
        calibration_by_seed=data_root / "figure_s7_calibration_by_seed.csv",
        calibration_summary=data_root / "figure_s7_calibration_summary.csv",
        reasons_by_seed=data_root / "figure_s7_abstention_reasons_by_seed.csv",
        reasons_summary=data_root / "figure_s7_abstention_reasons_summary.csv",
        cd8_offsets=data_root / "figure_s7_cd8_threshold_offsets.csv",
        shared_class_by_seed=data_root / "figure_s7_shared_class_by_seed.csv",
        shared_class_summary=data_root / "figure_s7_shared_class_summary.csv",
        png=output_root / "figure_s7_pbmc_calibration.png",
        pdf=output_root / "figure_s7_pbmc_calibration.pdf",
        svg=output_root / "figure_s7_pbmc_calibration.svg",
        description=output_root / "figure_s7_pbmc_calibration.md",
    )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PBMCS7Error(f"{source} is missing columns {missing}.")


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
    unexpected = sorted(set(normalized) - set(CONDITIONS))
    if unexpected:
        raise PBMCS7Error(
            f"{source} contains unsupported condition labels {unexpected}."
        )
    return normalized


def _load_raw_metadata(raw_path: Path) -> pd.DataFrame:
    atlas = ad.read_h5ad(raw_path, backed="r")
    condition_column = "condition" if "condition" in atlas.obs else "label"
    if "cell_type" not in atlas.obs or condition_column not in atlas.obs:
        raise PBMCS7Error(
            f"{raw_path} must contain cell_type and condition/label metadata."
        )
    metadata = atlas.obs[["cell_type", condition_column]].copy()
    metadata.index = metadata.index.astype(str)
    metadata = metadata.rename(columns={condition_column: "condition"})
    metadata["cell_type"] = metadata["cell_type"].astype(str)
    metadata["condition"] = _normalize_condition(metadata["condition"], raw_path)
    if not metadata.index.is_unique:
        raise PBMCS7Error(f"{raw_path} contains duplicate cell identifiers.")
    if getattr(atlas, "file", None) is not None:
        atlas.file.close()
    return metadata


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    _require_columns(
        manifest,
        {"held_out_label", "seed", "run_id", "method", "score"},
        manifest_path,
    )
    manifest = manifest.loc[
        (manifest["method"] == "coreot_full")
        & (manifest["score"] == "u")
    ].copy()
    if manifest.empty:
        raise PBMCS7Error(
            f"{manifest_path} has no coreot_full/u manifest rows."
        )
    selected = manifest[["held_out_label", "seed", "run_id"]].drop_duplicates()
    selected["seed"] = selected["seed"].astype(int)
    if selected[["held_out_label", "seed"]].duplicated().any():
        raise PBMCS7Error(f"{manifest_path} has duplicate state/seed entries.")
    if set(selected["held_out_label"]) != set(EXPECTED_LABELS):
        raise PBMCS7Error(f"{manifest_path} does not cover all S7 states.")
    if set(selected["seed"]) != set(EXPECTED_SEEDS):
        raise PBMCS7Error(f"{manifest_path} does not cover all S7 seeds.")
    if len(selected) != len(EXPECTED_LABELS) * len(EXPECTED_SEEDS):
        raise PBMCS7Error(f"{manifest_path} does not contain exactly 20 runs.")
    if selected["run_id"].duplicated().any():
        raise PBMCS7Error(f"{manifest_path} reuses a run_id.")
    return selected.sort_values(["held_out_label", "seed"]).reset_index(drop=True)


def _read_truth(
    truth_path: Path,
    raw_metadata: pd.DataFrame,
    held_out_label: str,
) -> pd.DataFrame:
    truth = pd.read_csv(truth_path)
    _require_columns(truth, {"cell_id", "true_label", "is_absent_state"}, truth_path)
    truth["cell_id"] = truth["cell_id"].astype(str)
    if not truth["cell_id"].is_unique:
        raise PBMCS7Error(f"{truth_path} contains duplicate cell identifiers.")
    joined = raw_metadata.reindex(truth["cell_id"])
    if joined.isna().any().any():
        raise PBMCS7Error(f"{truth_path} contains cells absent from raw metadata.")
    truth["cell_type"] = joined["cell_type"].to_numpy()
    truth["condition"] = joined["condition"].to_numpy()
    if not (
        truth["true_label"].astype(str) == truth["cell_type"].astype(str)
    ).all():
        raise PBMCS7Error(f"{truth_path} disagrees with raw cell-type metadata.")
    expected_positive = (
        (truth["cell_type"] == held_out_label)
        & (truth["condition"] == "stim")
    )
    if not np.array_equal(
        expected_positive.to_numpy(),
        truth["is_absent_state"].astype(bool).to_numpy(),
    ):
        raise PBMCS7Error(f"{truth_path} has an unexpected absent-state definition.")
    return truth


def _read_scores(
    score_path: Path,
    source_ids: set[str],
) -> pd.DataFrame:
    scores = pd.read_parquet(score_path)
    _require_columns(
        scores,
        {"cell_id", "method", "u", "label_entropy", "forced_label"},
        score_path,
    )
    scores = scores.loc[scores["method"] == "coreot_full"].copy()
    scores["cell_id"] = scores["cell_id"].astype(str)
    if not scores["cell_id"].is_unique:
        raise PBMCS7Error(f"{score_path} has duplicate coreot_full cell identifiers.")
    if set(scores["cell_id"]) != source_ids:
        raise PBMCS7Error(f"{score_path} does not match evaluation truth.")
    numeric = scores[["u", "label_entropy"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise PBMCS7Error(f"{score_path} contains non-finite u or entropy values.")
    scores["forced_label"] = scores["forced_label"].fillna("").astype(str)
    return scores[["cell_id", "u", "label_entropy", "forced_label"]]


def _read_theta_h(config_path: Path) -> float:
    if not config_path.is_file():
        raise PBMCS7Error(f"Missing selected scoring configuration {config_path}.")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        value = float(payload["thresholds"]["theta_H"]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PBMCS7Error(f"Invalid theta_H configuration in {config_path}.") from exc
    if not np.isclose(value, THETA_H):
        raise PBMCS7Error(
            f"{config_path} sets theta_H={value}, expected the fixed S7 value {THETA_H}."
        )
    return value


def _reason(score_flag: bool, entropy_flag: bool) -> str:
    if score_flag and entropy_flag:
        return "both"
    if score_flag:
        return "score_only"
    if entropy_flag:
        return "entropy_only"
    return "retained"


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = int(np.sum(y_true & y_pred))
    false_positive = int(np.sum(~y_true & y_pred))
    false_negative = int(np.sum(y_true & ~y_pred))
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return float("nan")
    return float(2 * true_positive / denominator)


def _summary(
    frame: pd.DataFrame,
    keys: list[str],
    value_column: str,
    output_name: str,
    *,
    minimum_splits: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False):
        key_values = (key,) if not isinstance(key, tuple) else key
        values = group[value_column].dropna().to_numpy(dtype=float)
        row = dict(zip(keys, key_values))
        n_contributing = int(group.loc[group[value_column].notna(), "seed"].nunique())
        row[f"raw_mean_{output_name}"] = (
            float(np.mean(values)) if len(values) else np.nan
        )
        row[f"raw_std_{output_name}"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if len(values) == 1
            else np.nan
        )
        row[f"raw_sem_{output_name}"] = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
            if len(values) == 1
            else np.nan
        )
        row["n_contributing_seeds"] = n_contributing
        row["is_na"] = n_contributing < minimum_splits
        row[f"mean_{output_name}"] = (
            np.nan
            if row["is_na"]
            else row[f"raw_mean_{output_name}"]
        )
        row[f"std_{output_name}"] = (
            np.nan
            if row["is_na"]
            else row[f"raw_std_{output_name}"]
        )
        row[f"sem_{output_name}"] = (
            np.nan
            if row["is_na"]
            else row[f"raw_sem_{output_name}"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_data(
    *,
    run_root: Path,
    config_root: Path,
    held_out_label: str,
    seed: int,
    raw_metadata: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
]:
    truth_path = (
        run_root
        / "benchmark"
        / "incomplete_reference"
        / "evaluation_truth"
        / "query_truth.csv"
    )
    truth = _read_truth(truth_path, raw_metadata, held_out_label)
    source_ids = set(truth["cell_id"])
    incomplete_path = (
        run_root
        / "scoring"
        / "incomplete_reference"
        / "pca30_k100"
        / "cell_scores.parquet"
    )
    full_path = (
        run_root
        / "scoring"
        / "full_reference_control"
        / "pca30_k100"
        / "cell_scores.parquet"
    )
    incomplete = _read_scores(incomplete_path, source_ids).rename(
        columns={
            "u": "u_incomplete",
            "label_entropy": "label_entropy_incomplete",
            "forced_label": "forced_label_incomplete",
        }
    )
    full = _read_scores(full_path, source_ids).rename(
        columns={
            "u": "u_full",
            "label_entropy": "label_entropy_full",
        }
    )
    if not np.isfinite(full["u_full"].to_numpy(dtype=float)).all():
        raise PBMCS7Error(f"{full_path} has non-finite full-reference u values.")
    paired = truth.merge(
        incomplete,
        on="cell_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        full,
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(truth):
        raise PBMCS7Error(f"{run_root} lost source cells while joining scores.")
    theta_h = _read_theta_h(config_root / run_root.name / "scoring.yaml")
    thresholds: list[dict[str, object]] = []
    reason_rows: list[dict[str, object]] = []
    for quantile in Q_VALUES:
        theta_u = float(np.quantile(paired["u_full"], quantile))
        score_flag = paired["u_incomplete"] > theta_u
        entropy_flag = paired["label_entropy_incomplete"] > theta_h
        abstain = score_flag | entropy_flag
        heldout = paired["is_absent_state"].astype(bool).to_numpy()
        shared = ~heldout
        thresholds.append(
            {
                "held_out_label": held_out_label,
                "seed": seed,
                "run_id": run_root.name,
                "quantile": quantile,
                "theta_u": theta_u,
                "theta_H": theta_h,
                "full_reference_score_rejection": float(
                    (paired["u_full"] > theta_u).mean()
                ),
                "heldout_rejection": float(abstain[heldout].mean()),
                "shared_coverage": float(1.0 - abstain[shared].mean()),
                "shared_false_abstention": float(abstain[shared].mean()),
            }
        )
        if np.isclose(quantile, PRIMARY_QUANTILE):
            for population, population_mask in (
                ("held_out_stimulated", heldout),
                ("shared_cells", shared),
            ):
                population_size = int(population_mask.sum())
                for category in REASON_ORDER:
                    categories = np.array(
                        [
                            _reason(bool(score), bool(entropy))
                            for score, entropy in zip(
                                score_flag[population_mask],
                                entropy_flag[population_mask],
                            )
                        ]
                    )
                    count = int(np.sum(categories == category))
                    reason_rows.append(
                        {
                            "held_out_label": held_out_label,
                            "seed": seed,
                            "run_id": run_root.name,
                            "population": population,
                            "reason": category,
                            "n_cells": population_size,
                            "n_cells_in_reason": count,
                            "fraction": float(count / population_size)
                            if population_size
                            else np.nan,
                        }
                    )
    cd8_rows: list[dict[str, object]] = []
    if held_out_label == "CD8 T cells":
        primary_theta = float(np.quantile(paired["u_full"], PRIMARY_QUANTILE))
        paired["offset"] = paired["u_incomplete"] - primary_theta
        for group, mask in (
            (
                "held_out_stimulated",
                (paired["cell_type"] == "CD8 T cells")
                & (paired["condition"] == "stim"),
            ),
            (
                "same_type_control",
                (paired["cell_type"] == "CD8 T cells")
                & (paired["condition"] == "ctrl"),
            ),
        ):
            for row in paired.loc[mask, ["cell_id", "offset"]].itertuples(
                index=False
            ):
                cd8_rows.append(
                    {
                        "held_out_label": held_out_label,
                        "seed": seed,
                        "run_id": run_root.name,
                        "cell_id": row.cell_id,
                        "group": group,
                        "offset": float(row.offset),
                        "theta_u": primary_theta,
                    }
                )
    primary_theta = float(np.quantile(paired["u_full"], PRIMARY_QUANTILE))
    primary_score_flag = paired["u_incomplete"] > primary_theta
    primary_entropy_flag = paired["label_entropy_incomplete"] > theta_h
    primary_abstain = primary_score_flag | primary_entropy_flag
    shared = ~paired["is_absent_state"].astype(bool)
    class_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        condition_shared = shared & (paired["condition"] == condition).to_numpy()
        condition_frame = paired.loc[condition_shared]
        for cell_type in CELL_TYPE_ORDER:
            class_mask = condition_shared & (
                paired["cell_type"] == cell_type
            ).to_numpy()
            class_frame = paired.loc[class_mask]
            coverage = (
                float((~primary_abstain[class_mask]).mean())
                if len(class_frame)
                else np.nan
            )
            y_true = (condition_frame["true_label"].astype(str) == cell_type).to_numpy()
            y_pred = (
                condition_frame["forced_label_incomplete"].astype(str)
                == cell_type
            ).to_numpy()
            forced_f1 = (
                _f1_binary(y_true, y_pred)
                if bool(y_true.any())
                else np.nan
            )
            for metric, value in (
                ("coverage", coverage),
                ("forced_f1", forced_f1),
            ):
                class_rows.append(
                    {
                        "held_out_label": held_out_label,
                        "seed": seed,
                        "run_id": run_root.name,
                        "cell_type": cell_type,
                        "condition": condition,
                        "metric": metric,
                        "value": value,
                        "n_cells": int(len(class_frame)),
                    }
                )
    return (
        pd.DataFrame(thresholds),
        pd.DataFrame(reason_rows),
        pd.DataFrame(cd8_rows),
        pd.DataFrame(class_rows),
        theta_h,
    )


def _sort_states(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_state_order"] = result["held_out_label"].map(
        {label: index for index, label in enumerate(EXPECTED_LABELS)}
    )
    result = result.sort_values(["_state_order", "seed"]).drop(
        columns=["_state_order"]
    )
    return result.reset_index(drop=True)


def _build_data(
    *,
    manifest: pd.DataFrame,
    runs_root: Path,
    config_root: Path,
    raw_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    threshold_parts: list[pd.DataFrame] = []
    reason_parts: list[pd.DataFrame] = []
    cd8_parts: list[pd.DataFrame] = []
    class_parts: list[pd.DataFrame] = []
    theta_values: list[float] = []
    for entry in manifest.itertuples(index=False):
        (
            thresholds,
            reasons,
            cd8,
            classes,
            theta_h,
        ) = _run_data(
            run_root=runs_root / str(entry.run_id),
            config_root=config_root,
            held_out_label=str(entry.held_out_label),
            seed=int(entry.seed),
            raw_metadata=raw_metadata,
        )
        threshold_parts.append(thresholds)
        reason_parts.append(reasons)
        cd8_parts.append(cd8)
        class_parts.append(classes)
        theta_values.append(theta_h)
    if not all(np.isclose(value, THETA_H) for value in theta_values):
        raise PBMCS7Error("Selected runs do not share the fixed theta_H value.")
    thresholds_by_seed = _sort_states(pd.concat(threshold_parts, ignore_index=True))
    reasons_by_seed = _sort_states(pd.concat(reason_parts, ignore_index=True))
    cd8_offsets = _sort_states(pd.concat(cd8_parts, ignore_index=True))
    shared_class_by_seed = _sort_states(pd.concat(class_parts, ignore_index=True))
    calibration_summary = _summary(
        thresholds_by_seed,
        ["held_out_label", "quantile"],
        "heldout_rejection",
        "heldout_rejection",
    ).merge(
        _summary(
            thresholds_by_seed,
            ["held_out_label", "quantile"],
            "shared_coverage",
            "shared_coverage",
        ),
        on=["held_out_label", "quantile", "n_contributing_seeds", "is_na"],
        how="outer",
        validate="one_to_one",
    )
    reason_summary = _summary(
        reasons_by_seed,
        ["held_out_label", "population", "reason"],
        "fraction",
        "fraction",
    )
    shared_class_summary = _summary(
        shared_class_by_seed,
        ["held_out_label", "cell_type", "condition", "metric"],
        "value",
        "value",
        minimum_splits=MIN_CONTRIBUTING_SPLITS,
    )
    return (
        thresholds_by_seed,
        calibration_summary,
        reasons_by_seed,
        reason_summary,
        cd8_offsets,
        shared_class_by_seed,
        shared_class_summary,
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.set_axisbelow(True)


def _plot_calibration(
    axes: list[plt.Axes],
    thresholds_by_seed: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    for axis, state in zip(axes, EXPECTED_LABELS):
        state_frame = thresholds_by_seed.loc[
            thresholds_by_seed["held_out_label"] == state
        ]
        for seed, seed_frame in state_frame.groupby("seed", sort=True):
            ordered = seed_frame.sort_values("quantile")
            axis.plot(
                ordered["shared_coverage"],
                ordered["heldout_rejection"],
                color=STATE_COLORS[state],
                alpha=0.3,
                linewidth=0.8,
            )
        mean_frame = summary.loc[
            summary["held_out_label"] == state
        ].sort_values("quantile")
        axis.plot(
            mean_frame["mean_shared_coverage"],
            mean_frame["mean_heldout_rejection"],
            color=STATE_COLORS[state],
            linewidth=2.0,
        )
        for quantile, size, label in (
            (PRIMARY_QUANTILE, 65, "95%"),
            (SECONDARY_QUANTILES[0], 38, "90%"),
            (SECONDARY_QUANTILES[1], 38, "97.5%"),
        ):
            point = mean_frame.loc[np.isclose(mean_frame["quantile"], quantile)]
            if len(point) != 1:
                raise PBMCS7Error(
                    f"Calibration summary lacks q={quantile} for {state}."
                )
            row = point.iloc[0]
            axis.scatter(
                [row["mean_shared_coverage"]],
                [row["mean_heldout_rejection"]],
                color=STATE_COLORS[state],
                edgecolor="black" if quantile == PRIMARY_QUANTILE else "white",
                linewidth=0.8,
                s=size,
                zorder=4,
            )
            axis.annotate(
                label,
                (row["mean_shared_coverage"], row["mean_heldout_rejection"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_title(state, fontsize=9)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("Shared-state coverage")
        axis.set_ylabel("Held-out rejection" if state in EXPECTED_LABELS[:2] else "")
        _style_axis(axis)


def _plot_reasons(axis: plt.Axes, reasons_summary: pd.DataFrame) -> None:
    populations = ("held_out_stimulated", "shared_cells")
    x_values = np.arange(len(EXPECTED_LABELS) * len(populations))
    bottoms = np.zeros(len(x_values))
    for reason in REASON_ORDER:
        heights: list[float] = []
        for state in EXPECTED_LABELS:
            for population in populations:
                row = reasons_summary.loc[
                    (reasons_summary["held_out_label"] == state)
                    & (reasons_summary["population"] == population)
                    & (reasons_summary["reason"] == reason)
                ]
                if len(row) != 1:
                    raise PBMCS7Error(
                        f"Reason summary lacks {state}, {population}, {reason}."
                    )
                heights.append(float(row.iloc[0]["mean_fraction"]))
        axis.bar(
            x_values,
            heights,
            bottom=bottoms,
            color=REASON_COLORS[reason],
            edgecolor="white",
            linewidth=0.4,
            label=REASON_LABELS[reason],
        )
        bottoms += np.asarray(heights)
    labels = [
        f"{STATE_SHORT_NAMES[state]}\n{population.replace('_', ' ')}"
        for state in EXPECTED_LABELS
        for population in populations
    ]
    axis.set_title("B. Why cells are abstained", loc="left", fontsize=10)
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Fraction of cells")
    axis.legend(loc="upper left", fontsize=7, frameon=False)
    _style_axis(axis)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values.astype(float))
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def _plot_cd8(axis_grid, cd8_offsets: pd.DataFrame) -> None:
    values = cd8_offsets["offset"].to_numpy(dtype=float)
    if not len(values):
        raise PBMCS7Error("No CD8 threshold-offset values are available.")
    lower = float(values.min())
    upper = float(values.max())
    margin = max(0.02, (upper - lower) * 0.08)
    for axis, seed in zip(axis_grid, EXPECTED_SEEDS):
        seed_frame = cd8_offsets.loc[cd8_offsets["seed"] == seed]
        for group, color in (
            ("held_out_stimulated", "#E69F00"),
            ("same_type_control", "#7F7F7F"),
        ):
            group_values = seed_frame.loc[
                seed_frame["group"] == group, "offset"
            ].to_numpy(dtype=float)
            if not len(group_values):
                raise PBMCS7Error(f"CD8 seed {seed} lacks {group} offsets.")
            x, y = _ecdf(group_values)
            axis.plot(
                x,
                y,
                color=color,
                linewidth=1.5,
                label=GROUP_LABELS[group] if seed == EXPECTED_SEEDS[0] else None,
            )
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_title(f"Seed {seed}", fontsize=8)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("u - theta_0.95", fontsize=8)
        if seed == EXPECTED_SEEDS[0]:
            axis.set_ylabel("Cumulative fraction", fontsize=8)
            axis.legend(fontsize=6, frameon=False, loc="lower right")
        else:
            axis.set_ylabel("")
        _style_axis(axis)


def _heatmap(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    rows = [
        (cell_type, condition)
        for cell_type in CELL_TYPE_ORDER
        for condition in CONDITIONS
    ]
    matrix = np.full((len(rows), len(EXPECTED_LABELS)), np.nan)
    for row_index, (cell_type, condition) in enumerate(rows):
        for column_index, state in enumerate(EXPECTED_LABELS):
            match = data.loc[
                (data["held_out_label"] == state)
                & (data["cell_type"] == cell_type)
                & (data["condition"] == condition)
                & (data["metric"] == metric)
            ]
            if len(match) != 1:
                raise PBMCS7Error(
                    f"Shared-class summary lacks {state}, {cell_type}, {condition}, {metric}."
                )
            matrix[row_index, column_index] = float(match.iloc[0]["mean_value"])
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#E6E6E6")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_title(title, loc="left", fontsize=9)
    axis.set_xticks(np.arange(len(EXPECTED_LABELS)))
    axis.set_xticklabels(
        [STATE_SHORT_NAMES[state] for state in EXPECTED_LABELS],
        fontsize=8,
    )
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(
        [f"{cell_type} - {condition}" for cell_type, condition in rows],
        fontsize=6,
    )
    for row_index in range(len(rows)):
        for column_index in range(len(EXPECTED_LABELS)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                "N/A" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if np.isnan(value) or value < 0.55 else "white",
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(metric.replace("_", " "))


def _build_figure(
    thresholds_by_seed: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    reasons_summary: pd.DataFrame,
    cd8_offsets: pd.DataFrame,
    shared_class_summary: pd.DataFrame,
) -> plt.Figure:
    figure = plt.figure(figsize=(19, 14), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, wspace=0.26, hspace=0.3)
    panel_a_grid = outer[0, 0].subgridspec(2, 2, wspace=0.22, hspace=0.35)
    panel_c_grid = outer[1, 0].subgridspec(1, 5, wspace=0.25)
    panel_d_grid = outer[1, 1].subgridspec(2, 1, hspace=0.35)
    panel_a_axes = [
        figure.add_subplot(panel_a_grid[row, column])
        for row in range(2)
        for column in range(2)
    ]
    panel_c_axes = [
        figure.add_subplot(panel_c_grid[0, index]) for index in range(5)
    ]
    panel_b_axis = figure.add_subplot(outer[0, 1])
    panel_d_coverage = figure.add_subplot(panel_d_grid[0, 0])
    panel_d_f1 = figure.add_subplot(panel_d_grid[1, 0])
    _plot_calibration(panel_a_axes, thresholds_by_seed, calibration_summary)
    _plot_reasons(panel_b_axis, reasons_summary)
    _plot_cd8(panel_c_axes, cd8_offsets)
    _heatmap(
        panel_d_coverage,
        shared_class_summary,
        "coverage",
        "D1. Shared-state coverage",
    )
    _heatmap(
        panel_d_f1,
        shared_class_summary,
        "forced_f1",
        "D2. Forced per-class F1",
    )
    return figure


def _description(
    *,
    manifest_path: Path,
    runs_root: Path,
    config_root: Path,
    raw_path: Path,
    paths: PBMCS7Paths,
) -> str:
    return "\n".join(
        [
            "# Supplementary Figure S7",
            "",
            "S7 recalibrates the full-reference u threshold over the "
            "requested quantiles while preserving the fixed label-entropy "
            "cutoff theta_H=0.8. The 95th-percentile threshold is the primary "
            "operating point.",
            "",
            "Panel A shows rejection-coverage curves. Panel B decomposes "
            "abstention into retained, score only, label entropy only, and "
            "both. Panel C shows CD8 incomplete-reference score offsets "
            "relative to the matched full-reference 95th-percentile cutoff. "
            "Panel D summarizes shared-cell coverage and forced per-class F1.",
            "",
            "Shared-state panels exclude the held-out stimulated state. "
            "Rows with fewer than three contributing donor splits are N/A; "
            "the source tables retain the contribution counts.",
            "",
            f"Run manifest: {manifest_path}",
            f"Run artifacts: {runs_root}",
            f"Selected scoring configurations: {config_root}",
            f"Raw metadata: {raw_path}",
            f"Thresholds: {paths.thresholds_by_seed}",
            f"Calibration source data: {paths.calibration_by_seed}",
            f"Abstention reasons: {paths.reasons_by_seed}",
            f"CD8 offsets: {paths.cd8_offsets}",
            f"Shared-class source data: {paths.shared_class_by_seed}",
            "",
            "Donor split is the quantitative unit. Cell-level distributions "
            "are used only to construct within-split operating summaries.",
            "",
        ]
    )


def write_pbmc_s7_calibration_figure(
    *,
    manifest_path: Path = Path(
        "results/PBMC/figures/data/figure_3_abstention_tradeoff_by_seed.csv"
    ),
    runs_root: Path = Path("runs"),
    config_root: Path = Path(
        "experiments/pbmc_state/generated_configs/"
        "pbmc_coreot_full_compare_baseline"
    ),
    raw_path: Path = Path("data/raw/kang_2018.h5ad"),
    output_root: Path = Path("results/PBMC/figures"),
) -> PBMCS7Paths:
    paths = _paths(Path(output_root))
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.data_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(Path(manifest_path))
    raw_metadata = _load_raw_metadata(Path(raw_path))
    (
        thresholds_by_seed,
        calibration_summary,
        reasons_by_seed,
        reasons_summary,
        cd8_offsets,
        shared_class_by_seed,
        shared_class_summary,
    ) = _build_data(
        manifest=manifest,
        runs_root=Path(runs_root),
        config_root=Path(config_root),
        raw_metadata=raw_metadata,
    )
    manifest.to_csv(paths.run_manifest, index=False)
    thresholds_by_seed.to_csv(paths.thresholds_by_seed, index=False)
    thresholds_by_seed.to_csv(paths.calibration_by_seed, index=False)
    calibration_summary.to_csv(paths.calibration_summary, index=False)
    reasons_by_seed.to_csv(paths.reasons_by_seed, index=False)
    reasons_summary.to_csv(paths.reasons_summary, index=False)
    cd8_offsets.to_csv(paths.cd8_offsets, index=False)
    shared_class_by_seed.to_csv(paths.shared_class_by_seed, index=False)
    shared_class_summary.to_csv(paths.shared_class_summary, index=False)
    figure = _build_figure(
        thresholds_by_seed,
        calibration_summary,
        reasons_summary,
        cd8_offsets,
        shared_class_summary,
    )
    figure.savefig(paths.png, dpi=300, bbox_inches="tight")
    figure.savefig(paths.pdf, bbox_inches="tight")
    figure.savefig(paths.svg, bbox_inches="tight")
    plt.close(figure)
    paths.description.write_text(
        _description(
            manifest_path=Path(manifest_path),
            runs_root=Path(runs_root),
            config_root=Path(config_root),
            raw_path=Path(raw_path),
            paths=paths,
        ),
        encoding="utf-8",
    )
    return paths
