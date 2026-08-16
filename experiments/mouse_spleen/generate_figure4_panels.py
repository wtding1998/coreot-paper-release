from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
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


N_QUERY = 4_333
N_PROLIFERATING = 62
N_SHARED = 4_271
PANEL_A_TOP_N = N_PROLIFERATING
UMAP_SEED = 20260713
METHOD_ORDER = [
    "coreot_full",
    "uniform_uot",
    "prior_only",
    "seurat_anchor",
    "scmap_cluster",
    "chetah",
]
TRANSFER_METHOD_ORDER = [
    "coreot_full",
    "uniform_uot",
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
    "seurat_anchor": "Seurat",
    "scmap_cluster": "scmap-cluster",
    "chetah": "CHETAH",
}
METHOD_COLORS = {
    "coreot_full": "#0072B2",
    "uniform_uot": "#D55E00",
    "prior_only": "#009E73",
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
    label_indices = [np.flatnonzero(true_labels == label) for label in np.unique(true_labels)]
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
                f1_score(
                    sampled_truth,
                    sampled_prediction,
                    average="macro",
                    zero_division=0,
                ),
            )

    recorded = recorded_metrics.set_index("method")
    rows: list[dict[str, object]] = []
    for method in method_order:
        accuracy = float(accuracy_score(true_labels, predicted[method]))
        macro_f1 = float(f1_score(true_labels, predicted[method], average="macro", zero_division=0))
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


def _load_source_objects(project_root: Path, source_root: Path) -> dict[str, pd.DataFrame]:
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
    for method in METHOD_ORDER:
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
        top_n = _top_n_weak_support_mask(oriented, query_metadata["cell_id"], n=PANEL_A_TOP_N)
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
    for method in METHOD_ORDER:
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
        supplementary_forced_predictions["method"].isin(TRANSFER_METHOD_ORDER)
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
    transfer_summary = (
        supplementary_transfer_summary.set_index("method").loc[TRANSFER_METHOD_ORDER].reset_index()
    )

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


def _draw_panel_a(
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
    flat[0].set_title("Evaluation endpoint", fontsize=10, x=0.62)
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
    heading_fontsize = 6.2 if compact else 8.5
    entry_fontsize = 5.7 if compact else 7.5
    info.text(
        0.02,
        0.94,
        "Method color (Panels A, B, C, E)",
        transform=info.transAxes,
        va="top",
        fontsize=heading_fontsize,
        fontweight="bold",
    )
    method_positions = [
        (0.02, 0.77),
        (0.02, 0.64),
        (0.02, 0.51),
        (0.52, 0.77),
        (0.52, 0.64),
        (0.52, 0.51),
    ]
    for method, (x, y) in zip(METHOD_ORDER, method_positions, strict=True):
        info.scatter(x, y, s=16, color=METHOD_COLORS[method], transform=info.transAxes)
        info.text(
            x + 0.07,
            y,
            METHOD_LABELS[method],
            transform=info.transAxes,
            va="center",
            fontsize=entry_fontsize,
        )
    encoding_rows = [
        (0.33, "#D73027", "Proliferating endpoint"),
        (0.19, "#D9D9D9", "Remaining query cells"),
    ]
    for y, color, label in encoding_rows:
        info.scatter(0.02, y, s=18, color=color, transform=info.transAxes)
        info.text(
            0.09,
            y,
            label,
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
        _panel_letter(flat[0], "A")


def _plot_panel_a(objects: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.7), constrained_layout=True)
    _draw_panel_a(objects, axes)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_panel_b(
    objects: dict[str, pd.DataFrame],
    axis: plt.Axes,
    *,
    compact: bool = False,
    show_letter: bool = True,
    show_legend: bool = True,
) -> None:
    scores = objects["detection_scores"]
    for method in METHOD_ORDER:
        frame = scores.loc[scores["method"].eq(method)]
        precision, recall, _ = precision_recall_curve(
            frame["is_proliferating"], frame["oriented_detection_score"]
        )
        axis.plot(
            recall,
            precision,
            color=METHOD_COLORS[method],
            linewidth=2.0 if method == "coreot_full" else 1.0,
            label=METHOD_LABELS[method],
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    if show_legend:
        axis.legend(
            frameon=False,
            fontsize=6 if compact else 8,
            ncol=2 if compact else 1,
            loc="upper right",
            columnspacing=0.8,
            handlelength=2.0,
        )
    if compact:
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    if show_letter:
        _panel_letter(axis, "B")


def _plot_panel_b(objects: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.1), constrained_layout=True)
    _draw_panel_b(objects, axis)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_panel_c(
    objects: dict[str, pd.DataFrame],
    axis: plt.Axes,
    *,
    compact: bool = False,
    show_letter: bool = True,
) -> None:
    summary = objects["detection_summary"].set_index("method").loc[METHOD_ORDER]
    for method in METHOD_ORDER:
        row = summary.loc[method]
        ap = float(row["ap"])
        auroc = float(row["auroc"])
        axis.errorbar(
            ap,
            auroc,
            xerr=[[ap - float(row["ap_lower"])], [float(row["ap_upper"]) - ap]],
            yerr=[
                [auroc - float(row["auroc_lower"])],
                [float(row["auroc_upper"]) - auroc],
            ],
            fmt="o",
            color=METHOD_COLORS[method],
            capsize=1.0,
            markersize=3,
            elinewidth=0.6,
            capthick=0.6,
        )
    axis.set(
        xlim=(0, 0.6),
        ylim=(0, 1),
        xlabel="Average precision (AP)",
        ylabel="AUROC",
    )
    if compact:
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    if show_letter:
        _panel_letter(axis, "C")


def _plot_panel_c(objects: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.1), constrained_layout=True)
    _draw_panel_c(objects, axis)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_destination_panel(
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
    grid = figure.add_gridspec(3, 1, height_ratios=[0.55, 4.4, 1.2])
    deficit_axis = figure.add_subplot(grid[0])
    heatmap_axis = figure.add_subplot(grid[1], sharex=deficit_axis)
    strip_axis = figure.add_subplot(grid[2], sharex=deficit_axis)
    deficit_axis.imshow(
        ordered["coreot_deficit"].to_numpy()[None, :],
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    deficit_axis.set_yticks([0], labels=["CoRe-OT\ndeficit $u$"])
    deficit_axis.set_xticks([])
    deficit_colorbar = figure.colorbar(
        ScalarMappable(norm=mcolors.Normalize(0, 1), cmap="magma"),
        ax=deficit_axis,
        fraction=0.022,
        pad=0.012,
    )
    image = heatmap_axis.imshow(probabilities, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    heatmap_axis.set_yticks(
        range(len(labels)),
        labels=labels,
        fontsize=6.5 if compact else 8,
    )
    heatmap_axis.set_xticks([])
    heatmap_axis.set_ylabel("Represented reference label", fontsize=7.5 if compact else None)
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
        fontsize=5.0 if compact else 8,
    )
    strip_axis.set_xticks([])
    strip_axis.set_xlabel("Proliferating query cells", fontsize=7.5 if compact else None)
    for separator in separators:
        for axis in (deficit_axis, heatmap_axis, strip_axis):
            axis.axvline(separator - 0.5, color="#222222", linewidth=0.45)
    probability_colorbar = figure.colorbar(
        image, ax=heatmap_axis, label="Conditional transported mass", fraction=0.022, pad=0.012
    )
    if compact:
        deficit_axis.tick_params(labelsize=6.5)
        deficit_colorbar.ax.tick_params(labelsize=6.5)
        probability_colorbar.ax.tick_params(labelsize=6.5)
        probability_colorbar.set_label("Conditional transported mass", fontsize=7.5)
    if show_legend:
        legend = [
            Line2D([0], [0], marker="s", linestyle="", color=label_colors[label], label=label)
            for label in label_to_index
        ]
        strip_axis.legend(
            handles=legend,
            title="Forced represented destination",
            bbox_to_anchor=(1.005, 0.5),
            loc="center left",
            frameon=False,
            fontsize=7,
            title_fontsize=7,
            ncol=1,
        )
    if show_letter:
        _panel_letter(deficit_axis, letter)


def _plot_destination_panel(
    objects: dict[str, pd.DataFrame],
    path: Path,
    *,
    letter: str,
) -> None:
    figure = plt.figure(figsize=(13.2, 7.0), constrained_layout=True)
    _draw_destination_panel(objects, figure, letter=letter, show_legend=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_panel_e(
    objects: dict[str, pd.DataFrame],
    axis: plt.Axes,
    *,
    compact: bool = False,
    show_letter: bool = True,
) -> None:
    transfer = objects["shared_transfer_summary"].set_index("method").loc[TRANSFER_METHOD_ORDER]
    for method in TRANSFER_METHOD_ORDER:
        row = transfer.loc[method]
        macro_f1 = float(row["forced_macro_f1"])
        accuracy = float(row["forced_accuracy"])
        axis.errorbar(
            macro_f1,
            accuracy,
            xerr=[
                [macro_f1 - float(row["forced_macro_f1_lower"])],
                [float(row["forced_macro_f1_upper"]) - macro_f1],
            ],
            yerr=[
                [accuracy - float(row["forced_accuracy_lower"])],
                [float(row["forced_accuracy_upper"]) - accuracy],
            ],
            fmt="o",
            color=METHOD_COLORS[method],
            capsize=1.0,
            markersize=3,
            elinewidth=0.6,
            capthick=0.6,
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Forced macro-F1",
        ylabel="Forced accuracy",
    )
    if compact:
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(7.5)
        axis.yaxis.label.set_size(7.5)
    if show_letter:
        _panel_letter(axis, "E")


def _plot_panel_e(objects: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.1), constrained_layout=True)
    _draw_panel_e(objects, axis)
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

    panels: list[tuple[str, pd.DataFrame, str]] = [("Ground truth", metadata, "true_label")]
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

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    limits = metadata[["umap_1", "umap_2"]]
    for axis, panel in zip(axes.ravel(), panels, strict=True):
        _draw_label_assignment_axis(axis, panel, limits)
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
    axes = figure.subplots(2, 3)
    for axis, panel in zip(axes.ravel(), panels, strict=True):
        _draw_label_assignment_axis(axis, panel, limits)
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


def _plot_panel_f(
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


def _plot_main_figure(
    objects: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> None:
    width = MAIN_FIGURE_WIDTH_MM / MM_PER_INCH
    height = MAIN_FIGURE_HEIGHT_MM / MM_PER_INCH
    figure = plt.figure(figsize=(width, height), constrained_layout=True)
    outer = figure.add_gridspec(
        4,
        1,
        height_ratios=[2.0, 1.05, 2.35, 1.8],
    )

    panel_a_figure = figure.add_subfigure(outer[0])
    panel_a_axes = panel_a_figure.subplots(2, 4)
    _draw_panel_a(objects, panel_a_axes, compact=True, show_letter=False)
    _subfigure_letter(panel_a_figure, "A")

    detection_grid = outer[1].subgridspec(1, 3, wspace=0.14)
    panel_b_figure = figure.add_subfigure(detection_grid[0])
    _draw_panel_b(
        objects,
        panel_b_figure.subplots(),
        compact=True,
        show_letter=False,
        show_legend=False,
    )
    _subfigure_letter(panel_b_figure, "B")
    panel_c_figure = figure.add_subfigure(detection_grid[1])
    _draw_panel_c(
        objects,
        panel_c_figure.subplots(),
        compact=True,
        show_letter=False,
    )
    _subfigure_letter(panel_c_figure, "C")
    panel_e_figure = figure.add_subfigure(detection_grid[2])
    panel_e_axis = panel_e_figure.subplots()
    _draw_panel_e(
        objects,
        panel_e_axis,
        compact=True,
        show_letter=False,
    )
    _subfigure_letter(panel_e_figure, "E", x=-0.06)

    label_assignment_figure = figure.add_subfigure(outer[2])
    _draw_label_assignment_block(objects, label_assignment_figure)

    destination_figure = figure.add_subfigure(outer[3])
    _draw_destination_panel(
        objects,
        destination_figure,
        letter="F",
        show_legend=False,
        compact=True,
        show_letter=False,
    )
    _subfigure_letter(destination_figure, "F")

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


def generate_figure4_panels(project_root: Path) -> list[Path]:
    output_root = project_root / "results/mouse_spleen_core_ot"
    panel_root = output_root / "manuscript/figure4_panels"
    source_root = panel_root / "source_data"
    panel_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    objects = _load_source_objects(project_root, source_root)
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

    f1_path = (
        output_root
        / "natural_mismatch/sensitivity/figure4_constant_tau_alpha_tau_target_8/tables/metrics_by_grid.csv"
    )
    f2_path = (
        output_root
        / "natural_mismatch/sensitivity/figure4_full_tau_alpha_40_tau_target_8/tables/metrics_by_grid.csv"
    )
    if not f1_path.is_file() or not f2_path.is_file():
        raise FileNotFoundError(
            "Figure 4 sensitivity tables are missing; run "
            "experiments.mouse_spleen.run_figure4_sensitivity first"
        )
    f1 = pd.read_csv(f1_path)
    f2 = pd.read_csv(f2_path)
    f1.to_csv(source_root / "parameter_sensitivity_f1.csv", index=False)
    f2.to_csv(source_root / "parameter_sensitivity_f2.csv", index=False)
    parameter_sensitivity = pd.concat(
        [
            f1.assign(panel_family="F1_constant_tau"),
            f2.assign(panel_family="F2_adaptive_tau"),
        ],
        ignore_index=True,
        sort=False,
    )
    parameter_sensitivity.to_csv(source_root / "parameter_sensitivity.csv", index=False)

    panel_paths = {letter: panel_root / f"panel_{letter.lower()}.png" for letter in "ABCDEF"}
    docs_figure_root = project_root / "docs/figs"
    docs_figure_root.mkdir(parents=True, exist_ok=True)
    supplementary_sensitivity_path = (
        docs_figure_root / "manuscript_fig_mouse_spleen_supp_parameter_sensitivity.png"
    )
    main_figure_paths = {
        extension: docs_figure_root / f"manuscript_fig_mouse_spleen_main.{extension}"
        for extension in ("pdf", "tiff", "png")
    }
    _plot_panel_a(objects, panel_paths["A"])
    _plot_panel_b(objects, panel_paths["B"])
    _plot_panel_c(objects, panel_paths["C"])
    _plot_label_assignment_umaps(objects, panel_paths["D"], letter="D")
    _plot_panel_e(objects, panel_paths["E"])
    _plot_destination_panel(objects, panel_paths["F"], letter="F")
    canonical_ap = float(objects["detection_summary"].set_index("method").loc["coreot_full", "ap"])
    _plot_panel_f(f1, f2, canonical_ap, supplementary_sensitivity_path)
    _plot_main_figure(objects, main_figure_paths)

    uniform_transport_manifest_path = (
        output_root
        / "runs/mouse_spleen_natural_proliferating/transport/natural_mismatch/"
        "mouse_spleen_provider_k100/uniform_uot/transport_manifest.yaml"
    )
    uniform_transport_manifest = yaml.safe_load(
        uniform_transport_manifest_path.read_text(encoding="utf-8")
    )
    uniform_metadata = uniform_transport_manifest["metadata"]
    if float(uniform_metadata["tau_source"]) != 4.0:
        raise ValueError(
            "The manuscript-facing mouse-spleen Uniform UOT fit must use tau_q=4"
        )

    manifest_path = panel_root / "figure4_panels_manifest.yaml"
    manifest = {
        "stage": "generate_mouse_spleen_figure4_panels",
        "artifacts": {
            **{f"panel_{key.lower()}": str(value) for key, value in panel_paths.items()},
            "supplementary_parameter_sensitivity": str(supplementary_sensitivity_path),
            **{f"main_figure_{key}": str(value) for key, value in main_figure_paths.items()},
            **{f"source_{key}": str(value) for key, value in source_paths.items()},
            "source_parameter_sensitivity_f1": str(source_root / "parameter_sensitivity_f1.csv"),
            "source_parameter_sensitivity_f2": str(source_root / "parameter_sensitivity_f2.csv"),
            "source_parameter_sensitivity": str(source_root / "parameter_sensitivity.csv"),
        },
        "metadata": {
            "endpoint": ENDPOINT,
            "n_query": N_QUERY,
            "n_proliferating": N_PROLIFERATING,
            "n_shared": N_SHARED,
            "metric_label": "AP",
            "metric_implementation": "sklearn.metrics.average_precision_score",
            "bootstrap_replicates": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "uniform_uot_operating_point": {
                "tau_q": float(uniform_metadata["tau_source"]),
                "tau_r": float(uniform_metadata["tau_target"]),
                "alpha": float(uniform_metadata["alpha"]),
                "epsilon": float(uniform_metadata["epsilon"]),
                "converged": bool(uniform_metadata["converged"]),
                "n_iter": int(uniform_metadata["n_iter"]),
            },
            "umap": {
                "input": "query rows of frozen 50-dimensional MultiMAP embedding",
                "n_neighbors": 30,
                "min_dist": 0.3,
                "metric": "euclidean",
                "random_state": UMAP_SEED,
            },
            "conditional_probability_eta": 1.0e-12,
            "panel_a_highlight": {
                "quantity": "oriented_weak_support_score",
                "top_n": PANEL_A_TOP_N,
                "selection_basis": "fixed_count_equal_to_endpoint_size",
                "tie_breaker": "cell_id_ascending",
            },
            "panel_e_bootstrap": {
                "metrics": ["forced_macro_f1", "forced_accuracy"],
                "resampling_unit": "query_cell",
                "strata": "true_shared_state_label",
                "paired_across_methods": True,
                "interval": "percentile_95",
            },
            "supplementary_table_s17_bootstrap": {
                "methods": SUPPLEMENT_TRANSFER_METHOD_ORDER,
                "metrics": ["forced_macro_f1", "forced_accuracy"],
                "resampling_unit": "query_cell",
                "strata": "true_shared_state_label",
                "paired_across_methods": True,
                "interval": "percentile_95",
            },
            "main_figure_layout": {
                "width_mm": MAIN_FIGURE_WIDTH_MM,
                "height_mm": MAIN_FIGURE_HEIGHT_MM,
                "orientation": "portrait",
                "rows": [
                    "A_weak_support_umaps",
                    "B_precision_recall_C_ap_auroc_E_shared_transfer",
                    "D_label_assignment_umaps",
                    "F_conditional_represented_destinations",
                ],
                "dense_scatter_layers_rasterized": True,
                "pdf_text_and_axes_vector": True,
                "tiff_dpi": 600,
                "png_dpi": 300,
                "parameter_sensitivity_location": "supplement",
                "shared_method_legend": {
                    "owner": "panel_a_lower_right",
                    "applies_to": ["A", "B", "C", "E"],
                },
            },
            "panel_format": "png_300_dpi",
            "input_hashes": {
                "baseline_detection": sha256_file(
                    output_root
                    / "natural_mismatch/compare_baselines/tables/baseline_detection_by_run.csv"
                ),
                "bootstrap": sha256_file(
                    output_root / "manuscript/tables/proliferating_detection_bootstrap.csv"
                ),
                "baseline_shared_label_transfer": sha256_file(
                    output_root / "natural_mismatch/compare_baselines/tables/"
                    "baseline_shared_label_transfer_by_run.csv"
                ),
                "uniform_uot_transport_manifest": sha256_file(
                    uniform_transport_manifest_path
                ),
                "f1": sha256_file(f1_path),
                "f2": sha256_file(f2_path),
            },
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return [
        *panel_paths.values(),
        supplementary_sensitivity_path,
        *main_figure_paths.values(),
        *source_paths.values(),
        manifest_path,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate standalone mouse-spleen Figure 4 panels."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    paths = generate_figure4_panels(args.project_root.resolve())
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
