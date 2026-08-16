from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold


class FullPriorError(ValueError):
    """Raised when leakage-free full-CoRe-OT priors cannot be constructed."""


HUMAN_UC_MARKER_MODULES = {
    "B_cells": ("CD79A", "MS4A1", "CD37", "MZB1", "JCHAIN"),
    "Endothelial": ("PECAM1", "EMCN", "VWF", "KDR"),
    "Epithelial": ("EPCAM", "KRT8", "KRT18", "KRT19"),
    "Fibroblasts": ("COL1A1", "COL1A2", "COL3A1", "DCN", "COL6A1"),
    "Glia": ("S100B", "SLC1A3", "PLP1", "SOX10"),
    "Myeloid": ("LYZ", "FCER1G", "TYROBP", "CTSS", "TPSAB1", "TPSB2", "KIT"),
    "T_cells": ("CD3D", "CD3E", "TRBC1", "NKG7"),
}


def derive_full_priors(
    run_root: str | Path,
    *,
    provider: str = "pca30",
    profile: str = "full_coreot",
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    anchor_method: str = "calibrated_logistic",
    base_profile: str = "full_coreot",
) -> None:
    root = Path(run_root)
    condition = "incomplete_reference"
    model_visible = root / "benchmark" / condition / "model_visible"
    cells = pd.read_csv(model_visible / "cells.csv")
    labels = pd.read_csv(model_visible / "target_labels.csv")
    embedding = np.load(root / "embeddings" / condition / provider / "embedding.npy")
    embedding_cells = pd.read_csv(
        root / "embeddings" / condition / provider / "embedding_cells.csv"
    )
    if list(cells["cell_id"].astype(str)) != list(embedding_cells["cell_id"].astype(str)):
        raise FullPriorError("embedding and model-visible cell order differ")
    reference = cells["domain"].astype(str).eq("reference").to_numpy()
    label_by_id = labels.set_index(labels["cell_id"].astype(str))["broad_label"].astype(str)
    y = cells.loc[reference, "cell_id"].astype(str).map(label_by_id)
    if y.isna().any():
        raise FullPriorError("reference broad labels are incomplete")
    groups = cells.loc[reference, "donor_id"].astype(str).to_numpy()
    classifier, temperature = _fit_calibrated_classifier(
        embedding[reference], y.to_numpy(), groups, c_grid
    )
    source = cells["domain"].astype(str).eq("query").to_numpy()
    source_ids = cells.loc[source, "cell_id"].astype(str).reset_index(drop=True)
    source_logits = classifier.decision_function(embedding[source])
    if source_logits.ndim == 1:
        source_logits = np.column_stack([-source_logits, source_logits])
    closed_set_probabilities = _softmax(source_logits / temperature)
    classes = classifier.classes_.astype(str)
    predicted_classes = classes[np.argmax(closed_set_probabilities, axis=1)]
    if anchor_method == "calibrated_logistic":
        support = np.ones(len(source_ids), dtype=float)
        support_distance = np.full(len(source_ids), np.nan)
        support_q80 = np.full(len(source_ids), np.nan)
        support_q99 = np.full(len(source_ids), np.nan)
        support_metadata = None
    elif anchor_method == "coarse_logistic_ood_gated":
        (
            support,
            support_distance,
            support_q80,
            support_q99,
            support_metadata,
        ) = _fit_class_support_gate(
            embedding[reference],
            y.to_numpy(),
            groups,
            embedding[source],
            predicted_classes,
        )
    else:
        raise FullPriorError(f"unsupported anchor method: {anchor_method}")
    probabilities = (
        support[:, None] * closed_set_probabilities
        + (1.0 - support[:, None]) / len(classes)
    )
    entropy = -(probabilities * np.log(np.clip(probabilities, 1.0e-300, 1.0))).sum(axis=1)
    confidence = 1.0 - entropy / np.log(len(classes))
    base_source = None
    if anchor_method == "coarse_logistic_ood_gated":
        base_source = _verified_base_source_priors(
            root,
            base_profile,
            source_ids,
            classes,
            predicted_classes,
            closed_set_probabilities,
        )
        rho = base_source["rho"].to_numpy(dtype=float)
    else:
        rho = _qc_matchability(
            root, source_ids, closed_set_probabilities, classes, cells.loc[source]
        )
    shared_source = pd.DataFrame(
        {
            "cell_id": source_ids,
            "rho": rho,
            "rho_source": "one_sided_qc_reliability",
            "rho_recipe": "donor_by_predicted_broad_qc_tail_ranks",
            "prior_risk": 1.0 - rho,
            "pmax_reference_classifier": closed_set_probabilities.max(axis=1),
            "anchor_class_pred": predicted_classes,
            "anchor_closed_set_confidence": closed_set_probabilities.max(axis=1),
            "anchor_support_q": support,
            "anchor_support_distance": support_distance,
            "anchor_support_q80": support_q80,
            "anchor_support_q99": support_q99,
            "anchor_confidence": confidence,
            "anchor_method": anchor_method,
        }
    )
    if base_source is None:
        shared_source["empirical_mass"] = _donor_mass(cells.loc[source])
    else:
        shared_source["rho_source"] = base_source["rho_source"].astype(str)
        shared_source["rho_recipe"] = base_source["rho_recipe"].astype(str)
        shared_source["prior_risk"] = base_source["prior_risk"].to_numpy(dtype=float)
        shared_source["empirical_mass"] = base_source["empirical_mass"].to_numpy(dtype=float)
    for index, anchor_class in enumerate(classes):
        shared_source[f"anchor_probability::{anchor_class}"] = probabilities[:, index]

    for condition in ("incomplete_reference", "full_reference_control"):
        condition_cells = pd.read_csv(
            root / "benchmark" / condition / "model_visible" / "cells.csv"
        )
        condition_labels = pd.read_csv(
            root / "benchmark" / condition / "model_visible" / "target_labels.csv"
        )
        source_ids_condition = condition_cells.loc[
            condition_cells["domain"].astype(str).eq("query"), "cell_id"
        ].astype(str)
        source_priors = shared_source.set_index("cell_id").loc[source_ids_condition].reset_index()
        target_cells = condition_cells.loc[
            condition_cells["domain"].astype(str).eq("reference"), ["cell_id", "donor_id"]
        ]
        target = target_cells.merge(condition_labels, on="cell_id", validate="one_to_one")
        target_priors = pd.DataFrame(
            {
                "cell_id": target["cell_id"].astype(str),
                "target_label_visible": target["target_label"].astype(str),
                "broad_anchor_class": target["broad_label"].astype(str),
                "rho_target": 1.0,
                "empirical_mass": _donor_mass(target),
            }
        )
        output = root / "derived" / condition / "prior_profiles" / profile
        output.mkdir(parents=True, exist_ok=True)
        source_priors.to_csv(output / "source_priors.csv", index=False)
        target_priors.to_csv(output / "target_priors.csv", index=False)
        manifest = {
            "profile": profile,
            "anchor_method": anchor_method,
            "anchor_vocabulary": classes.tolist(),
            "classifier": {
                "model": "multinomial_logistic_regression",
                "class_weight": "balanced",
                "c_grid": list(c_grid),
                "calibration": "temperature_scaling_on_donor_grouped_oof_logits",
                "temperature": float(temperature),
            },
        }
        if support_metadata is not None:
            manifest["ood_gate"] = support_metadata
            manifest["fixed_matchability_source_profile"] = base_profile
        (output / "prior_manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )


def derive_marker_module_priors(
    run_root: str | Path,
    *,
    profile: str = "broad_marker_module",
    base_profile: str = "full_coreot",
    counts_path: str | Path = (
        "data/derived/human_uc_colon_smillie_scp259.model_visible_counts.h5ad"
    ),
    temperature: float = 1.0,
) -> None:
    root = Path(run_root)
    model_visible = root / "benchmark" / "incomplete_reference" / "model_visible"
    cells = pd.read_csv(model_visible / "cells.csv")
    labels = pd.read_csv(model_visible / "target_labels.csv")
    source = cells["domain"].astype(str).eq("query")
    reference = cells["domain"].astype(str).eq("reference")
    source_ids = cells.loc[source, "cell_id"].astype(str).reset_index(drop=True)
    reference_ids = cells.loc[reference, "cell_id"].astype(str).reset_index(drop=True)
    label_by_id = labels.set_index(labels["cell_id"].astype(str))["broad_label"].astype(str)
    reference_y = reference_ids.map(label_by_id)
    if reference_y.isna().any():
        raise FullPriorError("reference broad labels are incomplete")
    classes = np.sort(reference_y.unique().astype(str))
    if set(classes) != set(HUMAN_UC_MARKER_MODULES):
        raise FullPriorError(
            "marker modules do not match the Human UC anchor vocabulary: "
            f"classes={classes.tolist()}"
        )

    selected_ids = pd.concat([source_ids, reference_ids], ignore_index=True)
    expression, used_markers, missing_markers = _read_log_normalized_marker_expression(
        Path(counts_path), selected_ids, HUMAN_UC_MARKER_MODULES
    )
    n_source = len(source_ids)
    probabilities, standardization = _marker_module_probabilities(
        expression[:n_source],
        expression[n_source:],
        classes,
        used_markers,
        reference_labels=reference_y.to_numpy(),
        temperature=temperature,
    )
    predicted_classes = classes[np.argmax(probabilities, axis=1)]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1.0e-300, 1.0))).sum(axis=1)
    confidence = 1.0 - entropy / np.log(len(classes))
    base = _load_base_source_priors(root, base_profile, source_ids)
    shared_source = pd.DataFrame(
        {
            "cell_id": source_ids,
            "rho": base["rho"].to_numpy(dtype=float),
            "rho_source": base["rho_source"].astype(str),
            "rho_recipe": base["rho_recipe"].astype(str),
            "prior_risk": base["prior_risk"].to_numpy(dtype=float),
            "pmax_reference_classifier": probabilities.max(axis=1),
            "anchor_class_pred": predicted_classes,
            "anchor_probability_max": probabilities.max(axis=1),
            "anchor_support_q": 1.0,
            "anchor_confidence": confidence,
            "anchor_method": "broad_marker_module",
            "empirical_mass": base["empirical_mass"].to_numpy(dtype=float),
        }
    )
    for index, anchor_class in enumerate(classes):
        shared_source[f"anchor_probability::{anchor_class}"] = probabilities[:, index]

    manifest = {
        "profile": profile,
        "anchor_method": "broad_marker_module",
        "anchor_vocabulary": classes.tolist(),
        "fixed_matchability_source_profile": base_profile,
        "expression": {
            "source": str(counts_path),
            "normalization": "log1p_library_size_10000",
        },
        "marker_modules": {
            str(anchor_class): {
                "used": list(used_markers[anchor_class]),
                "missing_from_dataset": list(missing_markers[anchor_class]),
            }
            for anchor_class in classes
        },
        "reference_standardization": standardization,
        "temperature": float(temperature),
    }
    for condition in ("incomplete_reference", "full_reference_control"):
        base_root = root / "derived" / condition / "prior_profiles" / base_profile
        target = pd.read_csv(base_root / "target_priors.csv")
        output = root / "derived" / condition / "prior_profiles" / profile
        output.mkdir(parents=True, exist_ok=True)
        shared_source.to_csv(output / "source_priors.csv", index=False)
        target.to_csv(output / "target_priors.csv", index=False)
        (output / "prior_manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )


def derive_prototype_priors(
    run_root: str | Path,
    *,
    provider: str = "pca30",
    profile: str = "coarse_prototype",
    base_profile: str = "full_coreot",
) -> None:
    root = Path(run_root)
    condition = "incomplete_reference"
    model_visible = root / "benchmark" / condition / "model_visible"
    cells = pd.read_csv(model_visible / "cells.csv")
    labels = pd.read_csv(model_visible / "target_labels.csv")
    embedding = np.load(root / "embeddings" / condition / provider / "embedding.npy")
    embedding_cells = pd.read_csv(
        root / "embeddings" / condition / provider / "embedding_cells.csv"
    )
    if list(cells["cell_id"].astype(str)) != list(embedding_cells["cell_id"].astype(str)):
        raise FullPriorError("embedding and model-visible cell order differ")
    source = cells["domain"].astype(str).eq("query").to_numpy()
    reference = cells["domain"].astype(str).eq("reference").to_numpy()
    source_ids = cells.loc[source, "cell_id"].astype(str).reset_index(drop=True)
    label_by_id = labels.set_index(labels["cell_id"].astype(str))["broad_label"].astype(str)
    reference_y = cells.loc[reference, "cell_id"].astype(str).map(label_by_id)
    if reference_y.isna().any():
        raise FullPriorError("reference broad labels are incomplete")
    groups = cells.loc[reference, "donor_id"].astype(str).to_numpy()
    probabilities, distances, temperature, prototype_metadata = _fit_prototype_prior(
        embedding[reference],
        reference_y.to_numpy(),
        groups,
        embedding[source],
    )
    classes = np.sort(reference_y.unique().astype(str))
    predicted_classes = classes[np.argmax(probabilities, axis=1)]
    predicted_distance = distances[np.arange(len(distances)), np.argmax(probabilities, axis=1)]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1.0e-300, 1.0))).sum(axis=1)
    confidence = 1.0 - entropy / np.log(len(classes))
    base = _load_base_source_priors(root, base_profile, source_ids)
    shared_source = pd.DataFrame(
        {
            "cell_id": source_ids,
            "rho": base["rho"].to_numpy(dtype=float),
            "rho_source": base["rho_source"].astype(str),
            "rho_recipe": base["rho_recipe"].astype(str),
            "prior_risk": base["prior_risk"].to_numpy(dtype=float),
            "pmax_reference_classifier": probabilities.max(axis=1),
            "anchor_class_pred": predicted_classes,
            "anchor_probability_max": probabilities.max(axis=1),
            "anchor_support_q": 1.0,
            "anchor_prototype_distance_pred": predicted_distance,
            "anchor_confidence": confidence,
            "anchor_method": "coarse_prototype",
            "empirical_mass": base["empirical_mass"].to_numpy(dtype=float),
        }
    )
    for index, anchor_class in enumerate(classes):
        shared_source[f"anchor_probability::{anchor_class}"] = probabilities[:, index]

    manifest = {
        "profile": profile,
        "anchor_method": "coarse_prototype",
        "anchor_vocabulary": classes.tolist(),
        "provider": provider,
        "distance": "squared_euclidean",
        "prototype_estimator": "equal_weight_mean_of_donor_class_centroids",
        "temperature": float(temperature),
        "temperature_calibration": "donor_grouped_out_of_fold_multiclass_log_loss",
        "fixed_matchability_source_profile": base_profile,
        **prototype_metadata,
    }
    for paired_condition in ("incomplete_reference", "full_reference_control"):
        base_root = root / "derived" / paired_condition / "prior_profiles" / base_profile
        target = pd.read_csv(base_root / "target_priors.csv")
        output = root / "derived" / paired_condition / "prior_profiles" / profile
        output.mkdir(parents=True, exist_ok=True)
        shared_source.to_csv(output / "source_priors.csv", index=False)
        target.to_csv(output / "target_priors.csv", index=False)
        (output / "prior_manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )


def _fit_calibrated_classifier(x, y, groups, c_grid):
    classes = np.unique(y)
    valid_splits = _class_covering_group_splits(y, groups)
    best_c = None
    best_loss = np.inf
    for c_value in c_grid:
        losses = []
        for train, test in valid_splits:
            model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000)
            model.fit(x[train], y[train])
            losses.append(log_loss(y[test], model.predict_proba(x[test]), labels=classes))
        if np.mean(losses) < best_loss:
            best_loss, best_c = float(np.mean(losses)), c_value
    logits = np.zeros((len(y), len(classes)))
    for train, test in valid_splits:
        model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(x[train], y[train])
        fold_logits = model.decision_function(x[test])
        if fold_logits.ndim == 1:
            fold_logits = np.column_stack([-fold_logits, fold_logits])
        logits[test] = fold_logits
    encoded = pd.Categorical(y, categories=classes).codes
    result = minimize_scalar(
        lambda log_t: log_loss(encoded, _softmax(logits / np.exp(log_t)), labels=np.arange(len(classes))),
        bounds=(-5.0, 5.0), method="bounded",
    )
    final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(x, y)
    return final, float(np.exp(result.x))


def _class_covering_group_splits(y, groups):
    classes = np.unique(y)
    for n_splits in range(min(5, len(np.unique(groups))), 1, -1):
        splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
        if all(
            set(y[train]) == set(classes) and set(y[test]) == set(classes)
            for train, test in splits
        ):
            return splits
    raise FullPriorError("fewer than two class-covering donor folds are available")


def _fit_class_support_gate(
    reference_x,
    reference_y,
    groups,
    source_x,
    predicted_classes,
    *,
    lower_quantile: float = 0.80,
    upper_quantile: float = 0.99,
):
    reference_x = np.asarray(reference_x, dtype=float)
    reference_y = np.asarray(reference_y).astype(str)
    source_x = np.asarray(source_x, dtype=float)
    predicted_classes = np.asarray(predicted_classes).astype(str)
    splits = _class_covering_group_splits(reference_y, groups)
    classes = np.unique(reference_y)
    oof_distances = np.full(len(reference_y), np.nan)
    for train, test in splits:
        for anchor_class in classes:
            train_class = train[reference_y[train] == anchor_class]
            test_class = test[reference_y[test] == anchor_class]
            model = LedoitWolf().fit(reference_x[train_class])
            oof_distances[test_class] = _mahalanobis_squared(
                reference_x[test_class], model.location_, model.precision_
            )
    if not np.isfinite(oof_distances).all():
        raise FullPriorError("non-finite donor-out-of-fold anchor support distances")

    thresholds = {
        anchor_class: (
            float(np.quantile(oof_distances[reference_y == anchor_class], lower_quantile)),
            float(np.quantile(oof_distances[reference_y == anchor_class], upper_quantile)),
        )
        for anchor_class in classes
    }
    distances = np.empty(len(source_x), dtype=float)
    q80 = np.empty(len(source_x), dtype=float)
    q99 = np.empty(len(source_x), dtype=float)
    for anchor_class in classes:
        source_class = np.flatnonzero(predicted_classes == anchor_class)
        if not len(source_class):
            continue
        model = LedoitWolf().fit(reference_x[reference_y == anchor_class])
        distances[source_class] = _mahalanobis_squared(
            source_x[source_class], model.location_, model.precision_
        )
        q80[source_class], q99[source_class] = thresholds[anchor_class]
    if not np.isfinite(distances).all():
        raise FullPriorError("non-finite source anchor support distances")
    width = q99 - q80
    support = np.where(
        distances <= q80,
        1.0,
        np.where(
            distances >= q99,
            0.0,
            np.divide(q99 - distances, width, out=np.zeros_like(distances), where=width > 0),
        ),
    )
    support = np.clip(support, 0.0, 1.0)
    metadata = {
        "support_model": "predicted_class_conditional_mahalanobis_squared",
        "covariance_estimator": "ledoit_wolf_shrinkage",
        "reference_distance_estimation": "donor_grouped_out_of_fold",
        "n_folds": len(splits),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "class_thresholds": {
            str(anchor_class): {"q80": values[0], "q99": values[1]}
            for anchor_class, values in thresholds.items()
        },
    }
    return support, distances, q80, q99, metadata


def _mahalanobis_squared(x, location, precision):
    centered = np.asarray(x, dtype=float) - np.asarray(location, dtype=float)
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


def _read_log_normalized_marker_expression(
    counts_path: Path,
    cell_ids: pd.Series,
    marker_modules: dict[str, tuple[str, ...]],
    *,
    chunk_size: int = 2048,
):
    shared = ad.read_h5ad(counts_path, backed="r")
    try:
        gene_to_index = pd.Series(np.arange(shared.n_vars), index=shared.var_names.astype(str))
        used = {
            anchor_class: tuple(gene for gene in genes if gene in gene_to_index)
            for anchor_class, genes in marker_modules.items()
        }
        missing = {
            anchor_class: tuple(gene for gene in genes if gene not in gene_to_index)
            for anchor_class, genes in marker_modules.items()
        }
        if any(not genes for genes in used.values()):
            empty = sorted(anchor_class for anchor_class, genes in used.items() if not genes)
            raise FullPriorError(f"marker modules have no observed genes for classes: {empty}")
        marker_genes = sorted({gene for genes in used.values() for gene in genes})
        marker_indices = gene_to_index.loc[marker_genes].to_numpy(dtype=int)
        cell_lookup = pd.Series(
            np.arange(shared.n_obs), index=shared.obs["cell_id"].astype(str)
        )
        rows = cell_ids.map(cell_lookup)
        if rows.isna().any():
            raise FullPriorError("model-visible marker count matrix is missing source/reference cells")
        rows = rows.to_numpy(dtype=int)
        output = np.empty((len(rows), len(marker_genes)), dtype=float)
        for start in range(0, len(rows), chunk_size):
            stop = min(start + chunk_size, len(rows))
            counts = shared.X[rows[start:stop]]
            totals = np.asarray(counts.sum(axis=1)).ravel()
            selected = counts[:, marker_indices].toarray()
            output[start:stop] = np.log1p(
                10000.0 * selected / np.maximum(totals[:, None], 1.0)
            )
    finally:
        shared.file.close()
    return pd.DataFrame(output, columns=marker_genes), used, missing


def _marker_module_probabilities(
    source_expression: pd.DataFrame,
    reference_expression: pd.DataFrame,
    classes,
    used_markers,
    *,
    reference_labels,
    temperature: float,
):
    if temperature <= 0.0:
        raise FullPriorError("marker-module temperature must be positive")
    source_scores = np.column_stack(
        [source_expression.loc[:, used_markers[anchor_class]].mean(axis=1) for anchor_class in classes]
    )
    reference_scores = np.column_stack(
        [
            reference_expression.loc[:, used_markers[anchor_class]].mean(axis=1)
            for anchor_class in classes
        ]
    )
    reference_labels = np.asarray(reference_labels).astype(str)
    medians = np.empty(len(classes), dtype=float)
    mads = np.empty(len(classes), dtype=float)
    for index, anchor_class in enumerate(classes):
        class_scores = reference_scores[reference_labels == anchor_class, index]
        if not len(class_scores):
            raise FullPriorError(f"no reference marker scores for class {anchor_class}")
        medians[index] = np.median(class_scores)
        mads[index] = np.median(np.abs(class_scores - medians[index]))
    scales = mads + 1.0e-6
    standardized = (source_scores - medians) / scales
    probabilities = _softmax(standardized / temperature)
    metadata = {
        str(anchor_class): {
            "reference_median": float(medians[index]),
            "reference_mad": float(mads[index]),
            "epsilon": 1.0e-6,
            "reference_population": "same_broad_anchor_class",
        }
        for index, anchor_class in enumerate(classes)
    }
    return probabilities, metadata


def _fit_prototype_prior(reference_x, reference_y, groups, source_x):
    reference_x = np.asarray(reference_x, dtype=float)
    reference_y = np.asarray(reference_y).astype(str)
    groups = np.asarray(groups).astype(str)
    source_x = np.asarray(source_x, dtype=float)
    classes = np.unique(reference_y)
    splits = _class_covering_group_splits(reference_y, groups)
    oof_logits = np.empty((len(reference_y), len(classes)), dtype=float)
    for train, test in splits:
        prototypes, _ = _donor_balanced_prototypes(
            reference_x[train], reference_y[train], groups[train], classes
        )
        oof_logits[test] = -_squared_euclidean(reference_x[test], prototypes)
    encoded = pd.Categorical(reference_y, categories=classes).codes
    result = minimize_scalar(
        lambda log_t: log_loss(
            encoded,
            _softmax(oof_logits / np.exp(log_t)),
            labels=np.arange(len(classes)),
        ),
        bounds=(-5.0, 10.0),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.x):
        raise FullPriorError("prototype temperature calibration failed")
    temperature = float(np.exp(result.x))
    prototypes, donor_counts = _donor_balanced_prototypes(
        reference_x, reference_y, groups, classes
    )
    distances = _squared_euclidean(source_x, prototypes)
    probabilities = _softmax(-distances / temperature)
    metadata = {
        "calibration_n_folds": len(splits),
        "class_reference_donor_counts": {
            str(anchor_class): int(donor_counts[index])
            for index, anchor_class in enumerate(classes)
        },
        "class_prototypes": {
            str(anchor_class): [float(value) for value in prototypes[index]]
            for index, anchor_class in enumerate(classes)
        },
    }
    return probabilities, distances, temperature, metadata


def _donor_balanced_prototypes(x, y, groups, classes):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(str)
    groups = np.asarray(groups).astype(str)
    prototypes = np.empty((len(classes), x.shape[1]), dtype=float)
    donor_counts = np.empty(len(classes), dtype=int)
    for index, anchor_class in enumerate(classes):
        class_groups = np.unique(groups[y == anchor_class])
        if not len(class_groups):
            raise FullPriorError(f"prototype class has no reference donors: {anchor_class}")
        donor_centroids = np.vstack(
            [x[(y == anchor_class) & (groups == donor)].mean(axis=0) for donor in class_groups]
        )
        prototypes[index] = donor_centroids.mean(axis=0)
        donor_counts[index] = len(class_groups)
    return prototypes, donor_counts


def _squared_euclidean(x, prototypes):
    x = np.asarray(x, dtype=float)
    prototypes = np.asarray(prototypes, dtype=float)
    return np.maximum(
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(prototypes * prototypes, axis=1)[None, :]
        - 2.0 * x @ prototypes.T,
        0.0,
    )


def _load_base_source_priors(root, base_profile, source_ids):
    path = (
        root
        / "derived"
        / "incomplete_reference"
        / "prior_profiles"
        / base_profile
        / "source_priors.csv"
    )
    if not path.is_file():
        raise FullPriorError(f"base source prior profile is missing: {path}")
    base = pd.read_csv(path).set_index("cell_id").reindex(source_ids).reset_index()
    required = {
        "rho",
        "rho_source",
        "rho_recipe",
        "prior_risk",
        "empirical_mass",
    }
    missing = required - set(base.columns)
    if missing or base.loc[:, list(required)].isna().any().any():
        raise FullPriorError(
            f"base source prior profile is incomplete; missing or null fields: {sorted(missing)}"
        )
    return base


def _verified_base_source_priors(
    root,
    base_profile,
    source_ids,
    classes,
    predicted_classes,
    closed_set_probabilities,
):
    base = _load_base_source_priors(root, base_profile, source_ids)
    required = {
        "anchor_class_pred",
        *(f"anchor_probability::{anchor_class}" for anchor_class in classes),
    }
    missing = required - set(base.columns)
    if missing or base.loc[:, list(required)].isna().any().any():
        raise FullPriorError(
            f"base source prior profile is incomplete; missing or null fields: {sorted(missing)}"
        )
    if not np.array_equal(base["anchor_class_pred"].astype(str), predicted_classes):
        raise FullPriorError("refitted closed-set anchor predictions differ from the base profile")
    stored_probabilities = base[
        [f"anchor_probability::{anchor_class}" for anchor_class in classes]
    ].to_numpy(dtype=float)
    if not np.allclose(stored_probabilities, closed_set_probabilities, rtol=1.0e-10, atol=1.0e-12):
        raise FullPriorError("refitted calibrated probabilities differ from the base profile")
    return base


def _qc_matchability(root, source_ids, probabilities, classes, source_cells):
    shared = ad.read_h5ad(
        root / "benchmark" / "shared" / "model_visible" / "counts.h5ad",
        backed="r",
    )
    try:
        lookup = pd.Series(np.arange(shared.n_obs), index=shared.obs["cell_id"].astype(str))
        rows = source_ids.map(lookup).to_numpy(dtype=int)
        counts = shared.X[rows]
        gene_names = np.asarray(shared.var_names.astype(str), dtype=str)
    finally:
        shared.file.close()
    n_gene = np.asarray((counts > 0).sum(axis=1)).ravel()
    n_umi = np.asarray(counts.sum(axis=1)).ravel()
    mito = np.zeros(len(rows))
    mt = np.char.startswith(np.char.upper(gene_names), "MT-")
    if mt.any():
        mito = 100.0 * np.asarray(counts[:, mt].sum(axis=1)).ravel() / np.maximum(n_umi, 1.0)
    frame = pd.DataFrame({"donor": source_cells["donor_id"].astype(str).to_numpy(), "pred": classes[np.argmax(probabilities, axis=1)], "gene": np.log1p(n_gene), "umi": np.log1p(n_umi), "mito": mito})
    ranks = pd.DataFrame(index=frame.index)
    for column in ("gene", "umi", "mito"):
        ranks[column] = frame.groupby(["donor", "pred"])[column].rank(pct=True)
        small = frame.groupby(["donor", "pred"])[column].transform("size") < 50
        ranks.loc[small, column] = frame.loc[small].groupby("pred")[column].rank(pct=True)
        small_class = frame.groupby("pred")[column].transform("size") < 50
        ranks.loc[small_class, column] = frame.loc[small_class, column].rank(pct=True)
    q_gene = np.minimum(1.0, ranks["gene"] / 0.20)
    q_umi = np.minimum(1.0, ranks["umi"] / 0.20)
    q_mito = np.minimum(1.0, (1.0 - ranks["mito"]) / 0.20)
    return np.clip(0.4 * q_gene + 0.4 * q_umi + 0.2 * q_mito, 0.05, 0.95).to_numpy()


def _donor_mass(cells):
    donor = cells["donor_id"].astype(str)
    counts = donor.value_counts()
    return donor.map(lambda value: 1.0 / (len(counts) * counts[value])).to_numpy()


def _softmax(logits):
    values = np.asarray(logits, dtype=float)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)
