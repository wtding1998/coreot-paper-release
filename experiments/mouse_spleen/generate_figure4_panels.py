from __future__ import annotations

# ruff: noqa: E402 -- direct CLI execution requires the repository bootstrap below.

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
import umap
import yaml

from coreot.artifacts.hashes import sha256_file
from coreot.results.mouse_spleen_parameter_sensitivity import (
    render_mouse_spleen_parameter_sensitivity,
)
from experiments.mouse_spleen.generate_proliferating_report import (
    BOOTSTRAP_SEED,
    ENDPOINT,
    N_BOOTSTRAP,
    SUPPLEMENT_METHOD_ORDER,
    _load_detection_scores,
)
from experiments.mouse_spleen.pipeline import fixed_true_label_macro_f1


N_QUERY = 4_333
N_PROLIFERATING = 62
N_SHARED = 4_271
TOP_RANKED_CELL_COUNT = N_PROLIFERATING
UMAP_SEED = 20260713
METHOD_ORDER = [
    "coreot_full",
    "pamona",
    "scotv2",
    "seurat_anchor",
    "scmap_cluster",
    "chetah",
]
TRANSFER_METHOD_ORDER = METHOD_ORDER.copy()
RETAINED_FIGURE4_METHOD_ORDER = [
    "coreot_full",
    "seurat_anchor",
    "scmap_cluster",
    "chetah",
]
SUPPLEMENT_TRANSFER_METHOD_ORDER = [
    method for method in SUPPLEMENT_METHOD_ORDER if method != "prior_only"
]
METHOD_LABELS = {
    "coreot_full": "CoRe-OT",
    "uniform_uot": "Uniform UOT",
    "prior_only": "Prior-only",
    "pamona": "Pamona",
    "scotv2": "SCOTv2",
    "seurat_anchor": "Seurat",
    "scmap_cluster": "scmap-cluster",
    "chetah": "CHETAH",
}
METHOD_COLORS = {
    "coreot_full": "#0072B2",
    "uniform_uot": "#D55E00",
    "prior_only": "#009E73",
    "pamona": "#D55E00",
    "scotv2": "#009E73",
    "seurat_anchor": "#CC79A7",
    "scmap_cluster": "#E69F00",
    "chetah": "#6A3D9A",
}
LABEL_COLORS = {
    "CD4 T": "#4C78A8",
    "CD8 T": "#72B7B2",
    "Treg": "#54A24B",
    "Follicular B": "#E45756",
    "Marginal zone B": "#F58518",
    "Transitional B": "#EECA3B",
    "DC": "#B279A2",
    "Macrophage": "#9D755D",
    "Granulocyte": "#FF9DA6",
    "NK": "#79706E",
}
PROLIFERATING_COLOR = "#D9D9D9"
MM_PER_INCH = 25.4
MAIN_FIGURE_WIDTH_MM = 180.0
MAIN_FIGURE_HEIGHT_MM = 235.0
PANEL_D_SHARED_POINT_SIZE = 2.5
PANEL_D_ENDPOINT_POINT_SIZE = 5.0
PANEL_D_SHARED_ALPHA = 0.80
RANK_LOCALIZATION_ENDPOINT_TITLE = "Proliferating cells"
PANEL_D_TITLE = "Evaluation labels"
PANEL_E_RETENTION_LABEL = "Query-marginal\ndeficit"
PANEL_E_DESTINATION_LABEL = "Reference destination"
PANEL_E_WEIGHT_LABEL = "Destination fraction"
PANEL_E_ASSIGNMENT_LABEL = "Forced assignment among represented labels"
PANEL_E_DEFICIT_CMAP = "viridis"
PANEL_E_COMPACT_METHOD_FONTSIZE = 4.0
PREVALENCE_TOLERANCE = 1.0e-12
PREVALENCE_LINE_COLOR = "#555555"
PREVALENCE_LABEL = "Mean prevalence (AP)"


def _weak_support_percentiles(scores: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=scores.index, dtype=float)
    valid = scores.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return result
    if n_valid == 1:
        result.loc[valid] = 0.5
        return result
    ranks = scores.loc[valid].rank(method="average")
    result.loc[valid] = (ranks - 1.0) / (n_valid - 1.0)
    return result


def _top_n_weak_support_mask(scores: pd.Series, cell_ids: pd.Series, *, n: int) -> pd.Series:
    if n < 0 or n > len(scores):
        raise ValueError(f"Top-n selection requires 0 <= n <= {len(scores)}, found {n}")
    if scores.isna().any():
        raise ValueError("Top-n weak-support selection does not accept missing scores")
    ranking = pd.DataFrame(
        {
            "position": np.arange(len(scores)),
            "score": scores.to_numpy(dtype=float),
            "cell_id": cell_ids.astype(str).to_numpy(),
        }
    ).sort_values(
        ["score", "cell_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected = ranking.head(n)["position"].to_numpy(dtype=int)
    result = pd.Series(False, index=scores.index, dtype=bool)
    result.iloc[selected] = True
    return result


def _order_destination_cells(
    destinations: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    required = {
        "cell_id",
        "dominant_coreot_label",
        "coreot_deficit",
        "dominant_probability",
    }
    missing = sorted(required - set(destinations.columns))
    if missing:
        raise ValueError(f"Destination ordering input lacks columns: {missing}")
    groups = (
        destinations.groupby("dominant_coreot_label", as_index=False)
        .agg(
            group_cell_count=("cell_id", "size"),
            group_total_conditional_mass=("dominant_probability", "sum"),
        )
        .sort_values(
            [
                "group_cell_count",
                "group_total_conditional_mass",
                "dominant_coreot_label",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    group_rank = dict(zip(groups["dominant_coreot_label"], groups.index, strict=True))
    ordered = destinations.copy()
    ordered["_group_rank"] = ordered["dominant_coreot_label"].map(group_rank)
    ordered = ordered.sort_values(
        ["_group_rank", "coreot_deficit", "cell_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop(columns="_group_rank")
    group_sizes = ordered.groupby("dominant_coreot_label", sort=False, observed=True).size()
    separators = np.cumsum(group_sizes.to_numpy())[:-1].astype(int).tolist()
    return ordered.reset_index(drop=True), separators


def _assert_query_metadata(frame: pd.DataFrame) -> None:
    if len(frame) != N_QUERY or frame["cell_id"].nunique() != N_QUERY:
        raise ValueError(f"Expected {N_QUERY} unique query cells, found {len(frame)}")
    n_positive = int(frame["is_proliferating"].sum())
    n_shared = int(frame["is_shared"].sum())
    if (n_positive, n_shared) != (N_PROLIFERATING, N_SHARED):
        raise ValueError(
            f"Unexpected endpoint composition: Proliferating={n_positive}, shared={n_shared}"
        )
    if not (frame["is_proliferating"] ^ frame["is_shared"]).all():
        raise ValueError("Every query cell must be exactly Proliferating or shared")


def _paired_label_stratified_transfer_bootstrap(
    predictions: pd.DataFrame,
    recorded_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    method_order: Sequence[str] = TRANSFER_METHOD_ORDER,
) -> pd.DataFrame:
    method_order = list(method_order)
    shared = predictions.loc[predictions["is_shared"]].copy()
    truth = (
        shared[["cell_id", "true_label"]]
        .drop_duplicates()
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    if truth.empty:
        raise ValueError("Shared-state transfer bootstrap requires at least one cell")
    prediction_wide = shared.pivot(
        index="cell_id", columns="method", values="forced_predicted_label"
    ).reindex(index=truth["cell_id"], columns=method_order)
    if prediction_wide.isna().any().any():
        raise ValueError("Shared-state forced predictions are incomplete")

    true_labels = truth["true_label"].astype(str).to_numpy()
    predicted = {method: prediction_wide[method].astype(str).to_numpy() for method in method_order}
    fixed_labels = sorted(np.unique(true_labels))
    label_indices = [np.flatnonzero(true_labels == label) for label in fixed_labels]
    rng = np.random.default_rng(seed)
    bootstrap_metrics = {method: np.empty((n_bootstrap, 2), dtype=float) for method in method_order}
    for replicate in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in label_indices]
        )
        sampled_truth = true_labels[sampled]
        for method in method_order:
            sampled_prediction = predicted[method][sampled]
            bootstrap_metrics[method][replicate] = (
                accuracy_score(sampled_truth, sampled_prediction),
                fixed_true_label_macro_f1(
                    sampled_truth,
                    sampled_prediction,
                    labels=fixed_labels,
                ),
            )

    recorded = recorded_metrics.set_index("method")
    rows: list[dict[str, object]] = []
    for method in method_order:
        accuracy = float(accuracy_score(true_labels, predicted[method]))
        macro_f1 = fixed_true_label_macro_f1(
            true_labels, predicted[method], labels=fixed_labels
        )
        if not np.isclose(
            accuracy, float(recorded.loc[method, "forced_accuracy"]), atol=1e-12, rtol=0
        ):
            raise ValueError(f"Forced-accuracy mismatch for method={method}")
        if not np.isclose(
            macro_f1,
            float(recorded.loc[method, "forced_macro_f1"]),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"Forced macro-F1 mismatch for method={method}")
        values = bootstrap_metrics[method]
        rows.append(
            {
                "method": method,
                "forced_accuracy": accuracy,
                "forced_accuracy_lower": float(np.quantile(values[:, 0], 0.025)),
                "forced_accuracy_upper": float(np.quantile(values[:, 0], 0.975)),
                "forced_macro_f1": macro_f1,
                "forced_macro_f1_lower": float(np.quantile(values[:, 1], 0.025)),
                "forced_macro_f1_upper": float(np.quantile(values[:, 1], 0.975)),
                "n_shared": len(truth),
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": seed,
                "bootstrap_strata": "true_shared_state_label",
            }
        )
    return pd.DataFrame(rows)


def _load_or_compute_umap(*, run_root: Path, source_path: Path) -> pd.DataFrame:
    if source_path.is_file():
        coordinates = pd.read_csv(source_path)
        required = {"cell_id", "umap_1", "umap_2"}
        if required <= set(coordinates.columns):
            return coordinates.loc[:, ["cell_id", "umap_1", "umap_2"]]
    embedding_root = run_root / "embeddings/natural_mismatch/mouse_spleen_provider"
    cells = pd.read_csv(embedding_root / "embedding_cells.csv")
    embedding = np.load(embedding_root / "embedding.npy")
    if len(cells) != len(embedding):
        raise ValueError("Embedding metadata and matrix have different row counts")
    query = cells.loc[cells["domain"].astype(str).eq("query")].copy()
    query_embedding = embedding[query["row_index"].to_numpy(dtype=int)]
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        n_components=2,
        metric="euclidean",
        random_state=UMAP_SEED,
        transform_seed=UMAP_SEED,
        n_jobs=1,
    )
    values = reducer.fit_transform(query_embedding)
    return pd.DataFrame(
        {
            "cell_id": query["cell_id"].astype(str).to_numpy(),
            "umap_1": values[:, 0],
            "umap_2": values[:, 1],
        }
    )


def _load_retained_source_objects(
    project_root: Path,
    source_root: Path,
) -> dict[str, pd.DataFrame]:
    output_root = project_root / "results/mouse_spleen_core_ot"
    run_root = output_root / "runs/mouse_spleen_natural_proliferating"
    baseline_root = output_root / "natural_mismatch/compare_baselines/tables"
    detection_path = baseline_root / "baseline_detection_by_run.csv"
    truth, all_scores, detection = _load_detection_scores(
        run_root=run_root, detection_path=detection_path
    )
    truth = truth.rename(columns={"is_shared_state": "is_shared"})
    truth["is_proliferating"] = truth["true_label"].astype(str).eq(ENDPOINT)
    truth["is_shared"] = truth["is_shared"].astype(bool)
    coordinates = _load_or_compute_umap(
        run_root=run_root, source_path=source_root / "query_umap_coordinates.csv"
    )
    query_metadata = truth[["cell_id", "true_label", "is_proliferating", "is_shared"]].merge(
        coordinates, on="cell_id", validate="one_to_one"
    )
    _assert_query_metadata(query_metadata)

    score_rows: list[pd.DataFrame] = []
    positive = query_metadata["is_proliferating"].to_numpy(dtype=bool)
    for method in RETAINED_FIGURE4_METHOD_ORDER:
        raw = pd.Series(all_scores[method], index=query_metadata.index, dtype=float)
        oriented = raw.copy()
        observed_ap = float(average_precision_score(positive, oriented))
        observed_auroc = float(roc_auc_score(positive, oriented))
        recorded = detection.loc[detection["method"].eq(method)].iloc[0]
        if not np.isclose(observed_ap, float(recorded["auprc"]), atol=1e-12, rtol=0):
            raise ValueError(f"AP mismatch for method={method}")
        if not np.isclose(observed_auroc, float(recorded["auroc"]), atol=1e-12, rtol=0):
            raise ValueError(f"AUROC mismatch for method={method}")
        percentiles = _weak_support_percentiles(oriented)
        top_n = _top_n_weak_support_mask(
            oriented,
            query_metadata["cell_id"],
            n=TOP_RANKED_CELL_COUNT,
        )
        score_rows.append(
            pd.DataFrame(
                {
                    "cell_id": query_metadata["cell_id"],
                    "method": method,
                    "raw_detection_score": raw,
                    "score_direction": "larger_is_weaker_reference_support",
                    "oriented_detection_score": oriented,
                    "weak_support_percentile": percentiles,
                    "is_top_62_weak_support": top_n,
                    "is_proliferating": positive,
                }
            )
        )
    detection_scores = pd.concat(score_rows, ignore_index=True)

    bootstrap = pd.read_csv(output_root / "manuscript/tables/proliferating_detection_bootstrap.csv")
    summary_rows = []
    prevalence = N_PROLIFERATING / N_QUERY
    for method in RETAINED_FIGURE4_METHOD_ORDER:
        method_bootstrap = bootstrap.loc[bootstrap["method"].eq(method)]
        ap = method_bootstrap.loc[method_bootstrap["metric"].eq("auprc")].iloc[0]
        auroc = method_bootstrap.loc[method_bootstrap["metric"].eq("auroc")].iloc[0]
        summary_rows.append(
            {
                "method": method,
                "ap": float(ap["estimate"]),
                "ap_lower": float(ap["ci_lower"]),
                "ap_upper": float(ap["ci_upper"]),
                "auroc": float(auroc["estimate"]),
                "auroc_lower": float(auroc["ci_lower"]),
                "auroc_upper": float(auroc["ci_upper"]),
                "prevalence": prevalence,
                "n_positive": N_PROLIFERATING,
                "n_negative": N_SHARED,
                "bootstrap_replicates": int(ap["n_bootstrap"]),
                "bootstrap_seed": int(ap["bootstrap_seed"]),
            }
        )
    detection_summary = pd.DataFrame(summary_rows)

    transport_root = run_root / "transport/natural_mismatch/mouse_spleen_provider_k100"
    coreot_transport = pd.read_parquet(transport_root / "coreot_full/cell_transport_scores.parquet")
    label_arrays = np.load(
        transport_root / "coreot_full/label_probabilities.npz", allow_pickle=True
    )
    probabilities = pd.DataFrame(
        label_arrays["probabilities"],
        index=label_arrays["cell_ids"].astype(str),
        columns=label_arrays["labels"].astype(str),
    )
    positive_ids = query_metadata.loc[query_metadata["is_proliferating"], "cell_id"].astype(str)
    positive_probabilities = probabilities.loc[positive_ids]
    row_sums = positive_probabilities.sum(axis=1)
    coreot_by_id = coreot_transport.set_index(coreot_transport["cell_id"].astype(str))
    positive_transport = coreot_by_id.loc[positive_ids]
    eta = 1.0e-12
    eligible = positive_transport["a_hat"].to_numpy(dtype=float) > eta
    if not np.allclose(row_sums.to_numpy()[eligible], 1.0, atol=1e-6, rtol=0):
        raise ValueError("Conditional destination probabilities do not sum to one")
    if (~eligible).any():
        raise ValueError("A Proliferating cell has transported mass at or below eta")
    dominant_labels = positive_probabilities.idxmax(axis=1)
    dominant_probabilities = positive_probabilities.max(axis=1)
    destination_wide = pd.DataFrame(
        {
            "cell_id": positive_ids.to_numpy(),
            "coreot_deficit": positive_transport["u"].to_numpy(dtype=float),
            "transported_query_mass": positive_transport["a_hat"].to_numpy(dtype=float),
            "empirical_query_mass": positive_transport["a"].to_numpy(dtype=float),
            "dominant_coreot_label": dominant_labels.to_numpy(),
            "dominant_probability": dominant_probabilities.to_numpy(dtype=float),
        }
    )
    destination_long = (
        positive_probabilities.reset_index(names="cell_id")
        .melt(
            id_vars="cell_id",
            var_name="reference_label",
            value_name="conditional_transport_probability",
        )
        .merge(destination_wide, on="cell_id", validate="many_to_one")
    )

    internal_scores = pd.read_parquet(
        run_root / "scoring/natural_mismatch/mouse_spleen_provider_k100/cell_scores.parquet"
    )
    external_scores = pd.read_parquet(
        run_root / "scoring/natural_mismatch/external_reference_mapping/cell_scores.parquet"
    )
    prediction_rows = []
    truth_labels = query_metadata[["cell_id", "true_label", "is_shared", "is_proliferating"]]
    for method in SUPPLEMENT_TRANSFER_METHOD_ORDER:
        source = (
            internal_scores if method in {"coreot_full", "uniform_uot", "nn"} else external_scores
        )
        method_predictions = source.loc[
            source["method"].astype(str).eq(method), ["cell_id", "forced_label"]
        ].rename(columns={"forced_label": "forced_predicted_label"})
        joined = truth_labels.merge(method_predictions, on="cell_id", validate="one_to_one")
        joined.insert(1, "method", method)
        prediction_rows.append(joined)
    supplementary_forced_predictions = pd.concat(prediction_rows, ignore_index=True)
    forced_predictions = supplementary_forced_predictions.loc[
        supplementary_forced_predictions["method"].isin(RETAINED_FIGURE4_METHOD_ORDER)
    ].copy()

    transfer_path = baseline_root / "baseline_shared_label_transfer_by_run.csv"
    transfer = pd.read_csv(transfer_path)
    transfer = transfer.loc[
        transfer["holdout_label"].eq(ENDPOINT)
        & transfer["method"].isin(SUPPLEMENT_TRANSFER_METHOD_ORDER)
    ].copy()
    if set(transfer["method"]) != set(SUPPLEMENT_TRANSFER_METHOD_ORDER):
        raise ValueError("Shared-label transfer table omits a supplementary method")
    supplementary_transfer_summary = _paired_label_stratified_transfer_bootstrap(
        supplementary_forced_predictions,
        transfer,
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
        method_order=SUPPLEMENT_TRANSFER_METHOD_ORDER,
    )
    transfer_summary = supplementary_transfer_summary.set_index("method").loc[
        RETAINED_FIGURE4_METHOD_ORDER
    ].reset_index()

    return {
        "query_metadata": query_metadata,
        "detection_scores": detection_scores,
        "detection_summary": detection_summary,
        "destination_long": destination_long,
        "destination_wide": destination_wide,
        "forced_predictions": forced_predictions,
        "shared_transfer_summary": transfer_summary,
        "supplementary_forced_predictions": supplementary_forced_predictions,
        "supplementary_shared_transfer_summary": supplementary_transfer_summary,
    }


def _panel_letter(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.08,
        1.08,
        letter,
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )


def _subfigure_letter(
    figure: SubFigure,
    letter: str,
    *,
    x: float = 0.0,
    y: float = 0.98,
) -> None:
    figure.text(x, y, letter, fontsize=14, fontweight="bold", va="top")


def _validated_prevalence(summary: pd.DataFrame) -> float:
    if "prevalence" not in summary.columns:
        raise ValueError("Displayed ranking rows must contain a prevalence field")
    values = pd.to_numeric(summary["prevalence"], errors="coerce").to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Displayed ranking rows must contain a prevalence value")
    if not np.isfinite(values).all():
        raise ValueError("Prevalence must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Prevalence must lie within [0, 1]")
    if not np.allclose(values, values[0], atol=PREVALENCE_TOLERANCE, rtol=0.0):
        raise ValueError("Prevalence must be unique across displayed rows")
    return float(values[0])


def _draw_paired_metric_bars(
    axis: plt.Axes,
    summary: pd.DataFrame,
    *,
    first_metric: tuple[str, str, str, str],
    second_metric: tuple[str, str, str, str],
    show_prevalence_reference: bool = False,
    compact: bool = False,
) -> None:
    ordered = summary.set_index("method").loc[METHOD_ORDER]
    prevalence = _validated_prevalence(ordered) if show_prevalence_reference else None
    first, first_lower, first_upper, first_label = first_metric
    second, second_lower, second_upper, second_label = second_metric
    first_values = ordered[first].to_numpy(dtype=float)
    second_values = ordered[second].to_numpy(dtype=float)
    group_positions = np.arange(len(METHOD_ORDER), dtype=float)
    first_positions = group_positions - 0.19
    second_positions = group_positions + 0.19
    colors = [METHOD_COLORS[method] for method in METHOD_ORDER]

    axis.barh(
        first_positions,
        first_values,
        height=0.32,
        color=colors,
        edgecolor=colors,
        linewidth=0.7,
        alpha=0.9,
        zorder=2,
    )
    axis.errorbar(
        first_values,
        first_positions,
        xerr=np.vstack(
            [
                first_values - ordered[first_lower].to_numpy(dtype=float),
                ordered[first_upper].to_numpy(dtype=float) - first_values,
            ]
        ),
        fmt="none",
        ecolor="#202020",
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.8,
        zorder=3,
    )
    axis.barh(
        second_positions,
        second_values,
        height=0.32,
        facecolor="white",
        edgecolor=colors,
        linewidth=1.3,
        zorder=2,
    )
    axis.errorbar(
        second_values,
        second_positions,
        xerr=np.vstack(
            [
                second_values - ordered[second_lower].to_numpy(dtype=float),
                ordered[second_upper].to_numpy(dtype=float) - second_values,
            ]
        ),
        fmt="none",
        ecolor="#202020",
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.8,
        zorder=3,
    )
    if prevalence is not None:
        axis.axvline(
            prevalence,
            color=PREVALENCE_LINE_COLOR,
            linestyle="--",
            linewidth=0.9,
            alpha=0.9,
            zorder=1,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.7, len(METHOD_ORDER) - 0.3)
    axis.invert_yaxis()
    axis.set_yticks(group_positions)
    axis.set_yticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
    axis.set_xlabel("Metric value", fontsize=8 if compact else 9)
    axis.grid(axis="x", color="#E3E3E3", linewidth=0.65, zorder=0)
    axis.tick_params(axis="both", labelsize=6.5 if compact else 8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    legend_handles: list[Patch | Line2D] = [
        Patch(facecolor="#777777", edgecolor="#777777", label=first_label),
        Patch(
            facecolor="white",
            edgecolor="#777777",
            linewidth=1.3,
            label=second_label,
        ),
    ]
    if prevalence is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PREVALENCE_LINE_COLOR,
                linestyle="--",
                linewidth=0.9,
                label=PREVALENCE_LABEL,
            )
        )
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=5.8 if compact else 8,
        columnspacing=0.8 if compact else 1.5,
        handlelength=1.6 if compact else 1.8,
    )


def _draw_endpoint_ranking_panel(
    objects: dict[str, pd.DataFrame],
    axis: plt.Axes,
    *,
    compact: bool = False,
    show_letter: bool = True,
) -> None:
    _draw_paired_metric_bars(
        axis,
        objects["detection_summary"],
        first_metric=("ap", "ap_lower", "ap_upper", "AP"),
        second_metric=("auroc", "auroc_lower", "auroc_upper", "AUROC"),
        show_prevalence_reference=True,
        compact=compact,
    )
    if show_letter:
        _panel_letter(axis, "A")


def _plot_endpoint_ranking_panel(objects: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.7), constrained_layout=True)
    _draw_endpoint_ranking_panel(objects, axis)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _draw_represented_state_transfer_panel(
    objects: dict[str, pd.DataFrame],
    axis: plt.Axes,
    *,
    compact: bool = False,
    show_letter: bool = True,
) -> None:
    _draw_paired_metric_bars(
        axis,
        objects["shared_transfer_summary"],
        first_metric=(
            "forced_accuracy",
            "forced_accuracy_lower",
            "forced_accuracy_upper",
            "Forced accuracy",
        ),
        second_metric=(
            "forced_macro_f1",
            "forced_macro_f1_lower",
            "forced_macro_f1_upper",
            "Forced macro-F1",
        ),
        compact=compact,
    )
    if show_letter:
        _panel_letter(axis, "B")


def _plot_represented_state_transfer_panel(
    objects: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.7), constrained_layout=True)
    _draw_represented_state_transfer_panel(objects, axis)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _draw_rank_localization_panel(
    objects: dict[str, pd.DataFrame],
    axes: np.ndarray,
    *,
    compact: bool = False,
    show_letter: bool = True,
) -> None:
    metadata = objects["query_metadata"]
    scores = objects["detection_scores"]
    flat = axes.ravel()
    shared = metadata.loc[metadata["is_shared"]]
    positive = metadata.loc[metadata["is_proliferating"]]
    flat[0].scatter(shared["umap_1"], shared["umap_2"], s=4, c="#D9D9D9", linewidths=0)
    flat[0].scatter(positive["umap_1"], positive["umap_2"], s=14, c="#D73027", linewidths=0)
    flat[0].set_title(RANK_LOCALIZATION_ENDPOINT_TITLE, fontsize=10, x=0.62)
    for axis, method in zip(flat[1:7], METHOD_ORDER, strict=True):
        frame = metadata.merge(
            scores.loc[
                scores["method"].eq(method),
                [
                    "cell_id",
                    "weak_support_percentile",
                    "is_top_62_weak_support",
                ],
            ],
            on="cell_id",
            validate="one_to_one",
        )
        missing = frame.loc[frame["weak_support_percentile"].isna()]
        selected = frame.loc[frame["is_top_62_weak_support"]]
        remaining = frame.loc[~frame["is_top_62_weak_support"]]
        axis.scatter(
            remaining["umap_1"],
            remaining["umap_2"],
            s=4,
            c="#D9D9D9",
            linewidths=0,
            rasterized=True,
        )
        if not missing.empty:
            axis.scatter(
                missing["umap_1"],
                missing["umap_2"],
                s=4,
                c="#A6A6A6",
                linewidths=0,
            )
        axis.scatter(
            selected["umap_1"],
            selected["umap_2"],
            c=METHOD_COLORS[method],
            s=7,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(METHOD_LABELS[method], fontsize=10)
    info = flat[7]
    info.axis("off")
    entry_fontsize = 5.2 if compact else 7.5
    for y, color, label in (
        (0.72, "#D73027", "Proliferating cells"),
        (0.28, "#D9D9D9", "Other ranking-cohort cells"),
    ):
        info.scatter(0.02, y, s=18, color=color, transform=info.transAxes)
        info.text(
            0.09,
            y,
            label,
            transform=info.transAxes,
            va="center",
            fontsize=entry_fontsize,
        )
    method_key_x = np.linspace(0.02, 0.20, len(METHOD_ORDER))
    info.scatter(
        method_key_x,
        np.full(len(METHOD_ORDER), 0.50),
        s=18,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        transform=info.transAxes,
    )
    info.text(
        0.25,
        0.50,
        "Method-specific top-ranked $N_+$ cells",
        transform=info.transAxes,
        va="center",
        fontsize=entry_fontsize,
    )
    limits = metadata[["umap_1", "umap_2"]]
    for axis in flat[:7]:
        axis.set_xlim(limits["umap_1"].min(), limits["umap_1"].max())
        axis.set_ylim(limits["umap_2"].min(), limits["umap_2"].max())
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    if show_letter:
        _panel_letter(flat[0], "C")


def _plot_rank_localization_panel(objects: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.7), constrained_layout=True)
    _draw_rank_localization_panel(objects, axes)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_destination_fitted_mass_panel(
    objects: dict[str, pd.DataFrame],
    figure: Figure | SubFigure,
    *,
    letter: str,
    show_legend: bool,
    compact: bool = False,
    show_letter: bool = True,
) -> None:
    destination = objects["destination_wide"]
    ordered, separators = _order_destination_cells(destination)
    long = objects["destination_long"]
    labels = sorted(long["reference_label"].astype(str).unique())
    probabilities = (
        long.pivot(
            index="reference_label", columns="cell_id", values="conditional_transport_probability"
        )
        .reindex(index=labels, columns=ordered["cell_id"])
        .to_numpy(dtype=float)
    )
    unexpected_labels = sorted(set(labels) - set(LABEL_COLORS))
    if unexpected_labels:
        raise ValueError(f"Destination panel has unmapped labels: {unexpected_labels}")
    label_colors = {label: LABEL_COLORS[label] for label in labels}
    predictions = objects["forced_predictions"]
    strip_values = np.empty((len(TRANSFER_METHOD_ORDER), len(ordered)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for row, method in enumerate(TRANSFER_METHOD_ORDER):
        method_predictions = predictions.loc[predictions["method"].eq(method)].set_index("cell_id")[
            "forced_predicted_label"
        ]
        ordered_labels = method_predictions.reindex(ordered["cell_id"]).fillna("Unassigned")
        unexpected = sorted(set(ordered_labels.astype(str)) - set(labels))
        if unexpected:
            labels.extend(label for label in unexpected if label not in labels)
            for label in unexpected:
                label_to_index[label] = len(label_to_index)
                label_colors[label] = "#BDBDBD"
        strip_values[row] = [label_to_index[str(value)] for value in ordered_labels]
    grid = figure.add_gridspec(3, 1, height_ratios=[0.8, 3.95, 1.4])
    deficit_axis = figure.add_subplot(grid[0])
    heatmap_axis = figure.add_subplot(grid[1], sharex=deficit_axis)
    strip_axis = figure.add_subplot(grid[2], sharex=deficit_axis)
    deficit_axis.imshow(
        ordered["coreot_deficit"].to_numpy()[None, :],
        aspect="auto",
        cmap=PANEL_E_DEFICIT_CMAP,
        vmin=0,
        vmax=1,
    )
    deficit_axis.set_yticks([0], labels=[PANEL_E_RETENTION_LABEL])
    deficit_axis.set_xticks([])
    deficit_colorbar = figure.colorbar(
        ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=PANEL_E_DEFICIT_CMAP),
        ax=deficit_axis,
        fraction=0.055,
        pad=0.012,
        aspect=5,
    )
    deficit_colorbar.set_ticks([0.0, 0.5, 1.0])
    image = heatmap_axis.imshow(probabilities, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    heatmap_axis.set_yticks(
        range(len(labels)),
        labels=labels,
        fontsize=6.5 if compact else 8,
    )
    heatmap_axis.set_xticks([])
    heatmap_axis.set_ylabel(PANEL_E_DESTINATION_LABEL, fontsize=7.5 if compact else None)
    strip_cmap = mcolors.ListedColormap([label_colors[label] for label in label_to_index])
    strip_axis.imshow(
        strip_values,
        aspect="auto",
        cmap=strip_cmap,
        vmin=-0.5,
        vmax=len(label_to_index) - 0.5,
        interpolation="nearest",
    )
    strip_axis.set_yticks(
        range(len(TRANSFER_METHOD_ORDER)),
        labels=[METHOD_LABELS[m] for m in TRANSFER_METHOD_ORDER],
        fontsize=PANEL_E_COMPACT_METHOD_FONTSIZE if compact else 8,
    )
    strip_axis.set_xticks([])
    strip_axis.set_xlabel("Proliferating query cells", fontsize=7.5 if compact else None)
    for separator in separators:
        for axis in (deficit_axis, heatmap_axis, strip_axis):
            axis.axvline(separator - 0.5, color="#222222", linewidth=0.45)
    probability_colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        label=PANEL_E_WEIGHT_LABEL,
        fraction=0.022,
        pad=0.012,
    )
    if compact:
        deficit_axis.tick_params(labelsize=5.5)
        deficit_colorbar.ax.tick_params(labelsize=5.5)
        probability_colorbar.ax.tick_params(labelsize=6.5)
        probability_colorbar.set_label(PANEL_E_WEIGHT_LABEL, fontsize=7.5)
    if show_legend:
        legend = [
            Line2D([0], [0], marker="s", linestyle="", color=label_colors[label], label=label)
            for label in label_to_index
        ]
        if compact:
            figure.legend(
                handles=legend,
                loc="outside lower center",
                frameon=False,
                fontsize=4.5,
                ncol=len(legend),
                columnspacing=0.45,
                handletextpad=0.2,
                markerscale=0.75,
            )
        else:
            strip_axis.legend(
                handles=legend,
                title=PANEL_E_ASSIGNMENT_LABEL,
                bbox_to_anchor=(1.005, 0.5),
                loc="center left",
                frameon=False,
                fontsize=7,
                title_fontsize=7,
                ncol=1,
            )
    if show_letter:
        _panel_letter(deficit_axis, letter)


def _plot_destination_fitted_mass_panel(
    objects: dict[str, pd.DataFrame],
    path: Path,
    *,
    letter: str,
) -> None:
    figure = plt.figure(figsize=(13.2, 7.0), constrained_layout=True)
    _draw_destination_fitted_mass_panel(
        objects,
        figure,
        letter=letter,
        show_legend=True,
    )
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _label_assignment_panels(
    objects: dict[str, pd.DataFrame],
) -> list[tuple[str, pd.DataFrame, str]]:
    metadata = objects["query_metadata"].copy()
    predictions = objects["forced_predictions"]
    shared_labels = sorted(metadata.loc[metadata["is_shared"], "true_label"].astype(str).unique())
    if shared_labels != sorted(LABEL_COLORS):
        raise ValueError(
            "Label-assignment UMAP labels disagree with the fixed palette: "
            f"observed={shared_labels}, expected={sorted(LABEL_COLORS)}"
        )

    panels: list[tuple[str, pd.DataFrame, str]] = [(PANEL_D_TITLE, metadata, "true_label")]
    for method in TRANSFER_METHOD_ORDER:
        method_predictions = predictions.loc[
            predictions["method"].eq(method), ["cell_id", "forced_predicted_label"]
        ]
        frame = metadata.merge(method_predictions, on="cell_id", validate="one_to_one")
        panels.append((METHOD_LABELS[method], frame, "forced_predicted_label"))
    return panels


def _draw_label_assignment_axis(
    axis: plt.Axes,
    panel: tuple[str, pd.DataFrame, str],
    limits: pd.DataFrame,
) -> None:
    title, frame, label_column = panel
    shared = frame.loc[frame["is_shared"]].sort_values("cell_id")
    endpoint = frame.loc[frame["is_proliferating"]]
    colors = shared[label_column].astype(str).map(LABEL_COLORS)
    if colors.isna().any():
        unexpected = sorted(shared.loc[colors.isna(), label_column].astype(str).unique())
        raise ValueError(f"Label-assignment UMAP has unmapped labels: {unexpected}")
    axis.scatter(
        shared["umap_1"],
        shared["umap_2"],
        c=colors,
        s=PANEL_D_SHARED_POINT_SIZE,
        linewidths=0,
        alpha=PANEL_D_SHARED_ALPHA,
        rasterized=True,
    )
    axis.scatter(
        endpoint["umap_1"],
        endpoint["umap_2"],
        c=PROLIFERATING_COLOR,
        s=PANEL_D_ENDPOINT_POINT_SIZE,
        linewidths=0,
        rasterized=True,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlim(limits["umap_1"].min(), limits["umap_1"].max())
    axis.set_ylim(limits["umap_2"].min(), limits["umap_2"].max())
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _label_assignment_legend() -> list[Line2D]:
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=label,
            markersize=5,
        )
        for label, color in LABEL_COLORS.items()
    ]
    legend.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=PROLIFERATING_COLOR,
            label="Proliferating (not evaluated)",
            markersize=5,
        )
    )
    return legend


def _plot_label_assignment_umaps(
    objects: dict[str, pd.DataFrame],
    path: Path,
    *,
    letter: str | None = None,
) -> None:
    metadata = objects["query_metadata"]
    panels = _label_assignment_panels(objects)

    figure, axes = plt.subplots(2, 4, figsize=(13.2, 8.0), constrained_layout=True)
    limits = metadata[["umap_1", "umap_2"]]
    for axis, panel in zip(axes.ravel(), panels, strict=False):
        _draw_label_assignment_axis(axis, panel, limits)
    for axis in axes.ravel()[len(panels) :]:
        axis.axis("off")
    if letter is not None:
        _panel_letter(axes.ravel()[0], letter)
    figure.legend(
        handles=_label_assignment_legend(),
        loc="outside lower center",
        ncol=len(_label_assignment_legend()),
        frameon=False,
        fontsize=6.5,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_label_assignment_block(
    objects: dict[str, pd.DataFrame],
    figure: Figure | SubFigure,
) -> None:
    panels = _label_assignment_panels(objects)
    metadata = objects["query_metadata"]
    limits = metadata[["umap_1", "umap_2"]]
    axes = figure.subplots(2, 4)
    for axis, panel in zip(axes.ravel(), panels, strict=False):
        _draw_label_assignment_axis(axis, panel, limits)
    for axis in axes.ravel()[len(panels) :]:
        axis.axis("off")
    figure.legend(
        handles=_label_assignment_legend(),
        loc="outside lower center",
        ncol=len(_label_assignment_legend()),
        frameon=False,
        fontsize=5.2,
        columnspacing=0.25,
        handletextpad=0.15,
        markerscale=0.75,
    )
    _subfigure_letter(figure, "D")


def _plot_parameter_sensitivity_supplement(
    f1: pd.DataFrame,
    f2: pd.DataFrame,
    canonical_ap: float,
    path: Path,
    *,
    letter: str | None = None,
) -> None:
    del f1, canonical_ap
    if letter is not None:
        raise ValueError("The standalone parameter-sensitivity figure has fixed A/B labels")
    figure_frame = f2.rename(
        columns={
            "auprc": "ap",
            "shared_forced_macro_f1": "represented_state_forced_macro_f1",
        }
    )
    render_mouse_spleen_parameter_sensitivity(figure_frame, {"png": path})


def _build_main_figure(objects: dict[str, pd.DataFrame]) -> Figure:
    width = MAIN_FIGURE_WIDTH_MM / MM_PER_INCH
    height = MAIN_FIGURE_HEIGHT_MM / MM_PER_INCH
    figure = plt.figure(figsize=(width, height), constrained_layout=True)
    outer = figure.add_gridspec(
        4,
        1,
        height_ratios=[1.55, 1.95, 2.30, 1.75],
    )

    metrics_grid = outer[0].subgridspec(1, 2, wspace=0.16)
    panel_a_figure = figure.add_subfigure(metrics_grid[0])
    _draw_endpoint_ranking_panel(
        objects,
        panel_a_figure.subplots(),
        compact=True,
        show_letter=False,
    )
    _subfigure_letter(panel_a_figure, "A")
    panel_b_figure = figure.add_subfigure(metrics_grid[1])
    _draw_represented_state_transfer_panel(
        objects,
        panel_b_figure.subplots(),
        compact=True,
        show_letter=False,
    )
    _subfigure_letter(panel_b_figure, "B")

    panel_c_figure = figure.add_subfigure(outer[1])
    panel_c_axes = panel_c_figure.subplots(2, 4)
    _draw_rank_localization_panel(
        objects, panel_c_axes, compact=True, show_letter=False
    )
    _subfigure_letter(panel_c_figure, "C")

    label_assignment_figure = figure.add_subfigure(outer[2])
    _draw_label_assignment_block(objects, label_assignment_figure)

    destination_figure = figure.add_subfigure(outer[3])
    _draw_destination_fitted_mass_panel(
        objects,
        destination_figure,
        letter="E",
        show_legend=True,
        compact=True,
        show_letter=False,
    )
    _subfigure_letter(destination_figure, "E", y=1.04)

    return figure


def _plot_main_figure(
    objects: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> None:
    figure = _build_main_figure(objects)

    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(paths["pdf"], dpi=300, facecolor="white")
    figure.savefig(paths["png"], dpi=300, facecolor="white")
    figure.savefig(
        paths["tiff"],
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def _read_figure4_source_objects(source_root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "query_metadata": source_root / "query_cell_metadata.csv",
        "detection_scores": source_root / "detection_scores.csv",
        "detection_summary": source_root / "detection_summary.csv",
        "destination_long": source_root / "coreot_destinations.csv",
        "forced_predictions": source_root / "forced_predictions.csv",
        "shared_transfer_summary": source_root / "shared_label_transfer_summary.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Figure 4 source is missing: {path}")
    objects = {key: pd.read_csv(path) for key, path in paths.items()}
    destination_columns = [
        "cell_id",
        "coreot_deficit",
        "transported_query_mass",
        "empirical_query_mass",
        "dominant_coreot_label",
        "dominant_probability",
    ]
    objects["destination_wide"] = objects["destination_long"][
        destination_columns
    ].drop_duplicates()
    _assert_query_metadata(objects["query_metadata"])
    for key, methods in (
        ("detection_scores", METHOD_ORDER),
        ("detection_summary", METHOD_ORDER),
        ("forced_predictions", TRANSFER_METHOD_ORDER),
        ("shared_transfer_summary", TRANSFER_METHOD_ORDER),
    ):
        observed = set(objects[key]["method"].astype(str))
        if observed != set(methods):
            raise ValueError(
                f"Figure 4 {key} methods differ from contract: {observed}"
            )
    return objects


def _read_candidate_source_objects(source_root: Path) -> dict[str, pd.DataFrame]:
    return _read_figure4_source_objects(source_root)


def render_figure4_bundle(
    *,
    source_root: Path,
    panel_paths: Mapping[str, Path],
    composite_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """Render the complete A--E Figure 4 bundle from validated source tables."""

    expected_panels = set("ABCDE")
    if set(panel_paths) != expected_panels:
        raise ValueError(
            "Figure 4 panel paths must contain exactly A--E; found "
            f"{sorted(panel_paths)}"
        )
    expected_composites = {"png", "pdf", "tiff"}
    if set(composite_paths) != expected_composites:
        raise ValueError(
            "Figure 4 composite paths must contain exactly png, pdf, and tiff; found "
            f"{sorted(composite_paths)}"
        )

    resolved_panels = {key: Path(value) for key, value in panel_paths.items()}
    resolved_composites = {key: Path(value) for key, value in composite_paths.items()}
    for path in [*resolved_panels.values(), *resolved_composites.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)

    objects = _read_figure4_source_objects(source_root)
    _plot_endpoint_ranking_panel(objects, resolved_panels["A"])
    _plot_represented_state_transfer_panel(objects, resolved_panels["B"])
    _plot_rank_localization_panel(objects, resolved_panels["C"])
    _plot_label_assignment_umaps(objects, resolved_panels["D"], letter="D")
    _plot_destination_fitted_mass_panel(objects, resolved_panels["E"], letter="E")
    _plot_main_figure(objects, resolved_composites)
    return {
        **{f"panel_{key.lower()}": value for key, value in resolved_panels.items()},
        **{
            f"composite_{key}": value
            for key, value in resolved_composites.items()
        },
    }


def generate_figure4_candidate(
    *,
    source_root: Path,
    candidate_root: Path,
) -> dict[str, Path]:
    """Render a review candidate from normalized, sealed-evidence sources."""

    panel_root = candidate_root / "standalone"
    composite_root = candidate_root / "composite"
    panel_root.mkdir(parents=True, exist_ok=True)
    composite_root.mkdir(parents=True, exist_ok=True)
    panel_paths = {
        letter: panel_root / f"figure_4_mouse_spleen_panel_{letter.lower()}.png"
        for letter in "ABCDE"
    }
    composite_paths = {
        suffix: composite_root / f"figure_4_mouse_spleen_main.{suffix}"
        for suffix in ("png", "pdf", "tiff")
    }
    rendered = render_figure4_bundle(
        source_root=source_root,
        panel_paths=panel_paths,
        composite_paths=composite_paths,
    )
    manifest_path = candidate_root / "render_manifest.yaml"
    all_sources = sorted(path for path in source_root.iterdir() if path.is_file())
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "render_mouse_spleen_figure4_candidate",
                "method_order": METHOD_ORDER,
                "transfer_method_order": TRANSFER_METHOD_ORDER,
                "artifacts": {
                    **{
                        f"panel_{letter.lower()}": str(path)
                        for letter, path in panel_paths.items()
                    },
                    **{
                        f"composite_{suffix}": str(path)
                        for suffix, path in composite_paths.items()
                    },
                },
                "source_data_sha256": {
                    str(path): sha256_file(path) for path in all_sources
                },
                "output_sha256": {
                    str(path): sha256_file(path)
                    for path in [*panel_paths.values(), *composite_paths.values()]
                },
                "model_fitting_executed": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        **rendered,
        "manifest": manifest_path,
    }


def generate_figure4_canonical(
    *,
    source_root: Path,
    project_root: Path,
) -> dict[str, Path]:
    """Render canonical Figure 4 artifacts from normalized source data."""

    panel_root = (
        project_root / "results/mouse_spleen_core_ot/manuscript/figure4_panels"
    )
    docs_root = project_root / "docs/figs"
    panel_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    panel_paths = {
        letter: panel_root / f"panel_{letter.lower()}.png" for letter in "ABCDE"
    }
    composite_paths = {
        suffix: docs_root / f"manuscript_fig_mouse_spleen_main.{suffix}"
        for suffix in ("png", "pdf", "tiff")
    }
    rendered = render_figure4_bundle(
        source_root=source_root,
        panel_paths=panel_paths,
        composite_paths=composite_paths,
    )

    manifest_path = panel_root / "figure4_panels_manifest.yaml"
    source_paths = sorted(path for path in source_root.iterdir() if path.is_file())
    artifacts = [*panel_paths.values(), *composite_paths.values()]

    def manifest_path_value(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root.resolve()))
        except ValueError as error:
            raise ValueError(
                f"Canonical manifest path is outside the repository: {path}"
            ) from error

    def source_manifest_path_value(path: Path) -> str:
        try:
            return manifest_path_value(path)
        except ValueError:
            return f"<declared-source-root>/{path.relative_to(source_root).as_posix()}"

    deterministic_artifacts = [
        path for path in artifacts if path.suffix in {".png", ".tiff"}
    ]
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "render_mouse_spleen_figure4_canonical",
                "method_order": METHOD_ORDER,
                "transfer_method_order": TRANSFER_METHOD_ORDER,
                "artifacts": {
                    **{
                        f"panel_{letter.lower()}": manifest_path_value(path)
                        for letter, path in panel_paths.items()
                    },
                    **{
                        f"composite_{suffix}": manifest_path_value(path)
                        for suffix, path in composite_paths.items()
                    },
                },
                "source_data_sha256": {
                    source_manifest_path_value(path): sha256_file(path)
                    for path in source_paths
                },
                "output_sha256": {
                    manifest_path_value(path): sha256_file(path)
                    for path in deterministic_artifacts
                },
                "container_metadata_policy": {
                    "authoritative_hashed_formats": ["png", "tiff"],
                    "pdf_hashes_recorded": False,
                    "pdf_reason": (
                        "PDF container timestamps may vary; PDF is an untracked "
                        "delivery format and is excluded from deterministic identity."
                    ),
                },
                "model_fitting_executed": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        **rendered,
        "manifest": manifest_path,
    }


def prepare_retained_figure4_sources(
    *,
    project_root: Path,
    source_root: Path,
) -> dict[str, Path]:
    """Reconstruct retained Figure 4 sources from current per-run artifacts."""

    source_root.mkdir(parents=True, exist_ok=True)
    objects = _load_retained_source_objects(project_root, source_root)
    source_paths = {
        "query_metadata": source_root / "query_cell_metadata.csv",
        "query_umap": source_root / "query_umap_coordinates.csv",
        "detection_scores": source_root / "detection_scores.csv",
        "detection_summary": source_root / "detection_summary.csv",
        "destinations": source_root / "coreot_destinations.csv",
        "forced_predictions": source_root / "forced_predictions.csv",
        "shared_transfer_summary": source_root / "shared_label_transfer_summary.csv",
        "supplementary_forced_predictions": (source_root / "supplementary_forced_predictions.csv"),
        "supplementary_shared_transfer_summary": (
            source_root / "supplementary_shared_label_transfer_summary.csv"
        ),
    }
    objects["query_metadata"].to_csv(source_paths["query_metadata"], index=False)
    objects["detection_scores"].to_csv(source_paths["detection_scores"], index=False)
    objects["detection_summary"].to_csv(source_paths["detection_summary"], index=False)
    objects["destination_long"].to_csv(source_paths["destinations"], index=False)
    objects["forced_predictions"].to_csv(source_paths["forced_predictions"], index=False)
    objects["shared_transfer_summary"].to_csv(source_paths["shared_transfer_summary"], index=False)
    objects["supplementary_forced_predictions"].to_csv(
        source_paths["supplementary_forced_predictions"], index=False
    )
    objects["supplementary_shared_transfer_summary"].to_csv(
        source_paths["supplementary_shared_transfer_summary"], index=False
    )
    objects["query_metadata"][["cell_id", "umap_1", "umap_2"]].to_csv(
        source_paths["query_umap"], index=False
    )
    return source_paths

def generate_figure4_panels(project_root: Path) -> list[Path]:
    """Prepare fresh retained sources without overwriting canonical sources."""

    source_root = (
        project_root
        / "results/mouse_spleen_core_ot/manuscript/figure4_panels/retained_source_data"
    )
    return list(
        prepare_retained_figure4_sources(
            project_root=project_root,
            source_root=source_root,
        ).values()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct retained mouse-spleen Figure 4 source tables."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Explicit retained Figure 4 source-data root. When supplied, render "
            "the canonical five-panel bundle under --project-root instead of "
            "reconstructing retained source tables."
        ),
    )
    args = parser.parse_args(argv)
    if args.source_root is not None:
        rendered = generate_figure4_canonical(
            source_root=args.source_root.resolve(),
            project_root=args.project_root.resolve(),
        )
        paths = list(rendered.values())
    else:
        paths = generate_figure4_panels(args.project_root.resolve())
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
