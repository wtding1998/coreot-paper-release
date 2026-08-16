from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from coreot.artifacts.hashes import sha256_file
from coreot.artifacts.manifests import Manifest, write_manifest
from coreot.evaluation.metrics import safe_auprc, safe_auroc
from coreot.transport.runner import _run_coreot_constant_tau, _run_coreot_match_only


MANUSCRIPT_METRICS = (
    ("auroc", "$u$-based AUROC"),
    ("auprc", "$u$-based AP"),
    ("forced_accuracy", "Forced accuracy"),
    ("forced_macro_f1", "Forced macro-F1"),
)

ABSOLUTE_METRIC_DISPLAY_ROWS = (
    ("auprc", "AP"),
    ("auroc", "AUROC"),
    ("forced_accuracy", "Forced accuracy"),
    ("forced_macro_f1", "Forced macro-F1"),
)

COMPUTED_METRICS = (
    *MANUSCRIPT_METRICS,
    ("post_abstention_accuracy", "Post-abstention accuracy"),
    ("post_abstention_macro_f1", "Post-abstention macro-F1"),
    ("coverage", "Coverage"),
    ("shared_false_abstention_rate", "Represented-state abstention rate"),
)

RETAINED_FIT_FILES = (
    "cell_transport_scores.parquet",
    "label_probabilities.npz",
    "method_params.yaml",
    "sparse_coupling.parquet",
    "transport_manifest.yaml",
)

RETAINABLE_VARIANTS = frozenset({"compatibility_only", "match_only"})


@dataclass(frozen=True)
class ComponentAblationSettings:
    endpoint: str
    match_tau_min_values: tuple[float, ...]
    match_tau_max_values: tuple[float, ...]
    compatibility_tau_values: tuple[float, ...]
    alpha_values: tuple[float, ...]
    tau_target: float
    max_iterations: int
    retained_fit_variants: tuple[str, ...]


def _settings(config: dict[str, Any]) -> ComponentAblationSettings:
    from experiments.mouse_spleen.pipeline import MouseSpleenConfigError, _mapping

    natural = _mapping(_mapping(config, "experiments"), "natural_mismatch")
    ablation = _mapping(natural, "component_ablation")
    match_tau_min_values = tuple(
        sorted({float(value) for value in ablation.get("match_tau_min_values", ())})
    )
    match_tau_max_values = tuple(
        sorted({float(value) for value in ablation.get("match_tau_max_values", ())})
    )
    compatibility_tau_values = tuple(
        sorted({float(value) for value in ablation.get("compatibility_tau_values", ())})
    )
    alpha_values = tuple(sorted({float(value) for value in ablation.get("alpha_values", ())}))
    endpoint = str(ablation.get("endpoint", "Proliferating"))
    tau_target = float(ablation.get("tau_target", 8.0))
    max_iterations = int(ablation.get("max_iterations", 5000))
    retained_fit_variants_raw = ablation.get("retain_fit_artifacts", ())
    if not isinstance(retained_fit_variants_raw, (list, tuple)):
        raise MouseSpleenConfigError(
            "Component-ablation retain_fit_artifacts must be a list of variant names"
        )
    retained_fit_variants = tuple(
        sorted({str(value) for value in retained_fit_variants_raw})
    )
    unknown_retained_variants = sorted(
        set(retained_fit_variants) - RETAINABLE_VARIANTS
    )
    if unknown_retained_variants:
        raise MouseSpleenConfigError(
            "Component-ablation retain_fit_artifacts contains unknown variants: "
            f"{unknown_retained_variants}"
        )
    if endpoint != "Proliferating":
        raise MouseSpleenConfigError(
            "Natural component ablation is restricted to the Proliferating endpoint"
        )
    if not match_tau_min_values or any(value <= 0.0 for value in match_tau_min_values):
        raise MouseSpleenConfigError(
            "Component-ablation match_tau_min_values must be nonempty and strictly positive"
        )
    if not match_tau_max_values or any(value <= 0.0 for value in match_tau_max_values):
        raise MouseSpleenConfigError(
            "Component-ablation match_tau_max_values must be nonempty and strictly positive"
        )
    if not compatibility_tau_values or any(value <= 0.0 for value in compatibility_tau_values):
        raise MouseSpleenConfigError(
            "Component-ablation compatibility_tau_values must be nonempty and strictly positive"
        )
    if not alpha_values or any(value < 0.0 for value in alpha_values):
        raise MouseSpleenConfigError(
            "Component-ablation alpha_values must be nonempty and nonnegative"
        )
    if tau_target <= 0.0 or max_iterations <= 0:
        raise MouseSpleenConfigError(
            "Component-ablation tau_target and max_iterations must be positive"
        )
    return ComponentAblationSettings(
        endpoint=endpoint,
        match_tau_min_values=match_tau_min_values,
        match_tau_max_values=match_tau_max_values,
        compatibility_tau_values=compatibility_tau_values,
        alpha_values=alpha_values,
        tau_target=tau_target,
        max_iterations=max_iterations,
        retained_fit_variants=retained_fit_variants,
    )


def _project_root(output_root: Path) -> Path:
    return output_root.parent.parent if output_root.parent.name == "results" else output_root.parent


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("Convergence column contains values other than true/false")
    return normalized.eq("true")


def _checkpoint(
    path: Path,
    *,
    columns: list[str],
    identity: dict[str, object],
) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Component-ablation checkpoint lacks columns {missing}: {path}")
    for column, expected in identity.items():
        observed = frame[column]
        if isinstance(expected, (int, float)):
            matches = np.isclose(observed.astype(float), float(expected)).all()
        else:
            matches = observed.astype(str).eq(str(expected)).all()
        if not matches:
            raise ValueError(
                f"Component-ablation checkpoint identity mismatch for {column}: {path}"
            )
    frame["converged"] = _as_bool(frame["converged"])
    return frame.loc[:, columns]


def _evaluate(
    *,
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    endpoint: str,
) -> dict[str, float | int | str]:
    from experiments.mouse_spleen.pipeline import _label_transfer_metrics

    joined = truth.merge(scores, on="cell_id", validate="one_to_one")
    positive = joined["true_label"].astype(str).eq(endpoint)
    shared = joined.loc[joined["is_shared_state"].astype(bool)].copy()
    shared["abstain_u_or_entropy"] = False
    transfer = _label_transfer_metrics(shared)
    return {
        "evaluation_scope": "global_all_query",
        "n_query": int(len(joined)),
        "n_positive": int(positive.sum()),
        "auroc": safe_auroc(positive, joined["u"]),
        "auprc": safe_auprc(positive, joined["u"]),
        "forced_accuracy": transfer["forced_accuracy"],
        "forced_macro_f1": transfer["forced_macro_f1"],
        "post_abstention_accuracy": transfer["post_abstention_accuracy"],
        "post_abstention_macro_f1": transfer["post_abstention_macro_f1"],
        "coverage": transfer["coverage"],
        "shared_false_abstention_rate": transfer["shared_false_abstention_rate"],
        "threshold_applicability": "undefined_without_matched_full_reference",
    }


def _run_grid(
    *,
    variant: str,
    combinations: list[tuple[float, float]],
    parameter_names: tuple[str, str],
    method_name: str,
    runner: Callable[..., None],
    root: Path,
    condition: str,
    candidate: str,
    run_root: Path,
    candidates: pd.DataFrame,
    source_priors: pd.DataFrame,
    target_priors: pd.DataFrame,
    truth: pd.DataFrame,
    settings: ComponentAblationSettings,
    transport: dict[str, Any],
    source_priors_hash: str,
    candidate_edges_hash: str,
    retain_fit_artifacts: bool,
) -> Path:
    first, second = parameter_names
    columns = [
        "run_id",
        "natural_endpoint",
        "variant",
        "method",
        first,
        second,
        "tau_target",
        "epsilon",
        "max_iterations",
        "tolerance",
        "converged",
        "n_iterations",
        "runtime_seconds",
        "evaluation_scope",
        "n_query",
        "n_positive",
        *[metric for metric, _ in COMPUTED_METRICS],
        "threshold_applicability",
        "source_priors_sha256",
        "candidate_edges_sha256",
    ]
    checkpoint_path = root / variant / "checkpoint.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "natural_endpoint": settings.endpoint,
        "variant": variant,
        "method": method_name,
        "tau_target": settings.tau_target,
        "epsilon": float(transport.get("epsilon", 0.05)),
        "max_iterations": settings.max_iterations,
        "tolerance": float(transport.get("tolerance", 1.0e-6)),
        "source_priors_sha256": source_priors_hash,
        "candidate_edges_sha256": candidate_edges_hash,
    }
    frame = _checkpoint(checkpoint_path, columns=columns, identity=identity)
    configured = set(combinations)
    def method_root(first_value: float, second_value: float) -> Path:
        storage = "fits" if retain_fit_artifacts else "tmp"
        return (
            root
            / storage
            / variant
            / f"{first}_{first_value:g}_{second}_{second_value:g}"
        )

    def complete_fit_artifacts(path: Path) -> bool:
        return all((path / filename).is_file() for filename in RETAINED_FIT_FILES)

    keep_rows = []
    for row in frame.itertuples(index=False):
        first_value = float(getattr(row, first))
        second_value = float(getattr(row, second))
        keep = (first_value, second_value) in configured
        if keep and retain_fit_artifacts:
            keep = complete_fit_artifacts(method_root(first_value, second_value))
        keep_rows.append(keep)
    frame = frame.loc[keep_rows].copy()
    frame.to_csv(checkpoint_path, index=False)
    completed = {
        (float(getattr(row, first)), float(getattr(row, second)))
        for row in frame.itertuples(index=False)
    }
    for first_value, second_value in combinations:
        if (first_value, second_value) in completed:
            continue
        fit_root = method_root(first_value, second_value)
        if fit_root.exists() and not complete_fit_artifacts(fit_root):
            shutil.rmtree(fit_root)
        fit_root.mkdir(parents=True, exist_ok=True)
        if variant == "match_only":
            method_config = {
                "name": method_name,
                "tau_min": first_value,
                "tau_max": second_value,
                "tau_target": settings.tau_target,
                "alpha": 0.0,
            }
        else:
            method_config = {
                "name": method_name,
                "tau_source": first_value,
                "tau_target": settings.tau_target,
                "alpha": second_value,
            }
        method_config.update(
            {
                "epsilon": float(transport.get("epsilon", 0.05)),
                "max_iter": settings.max_iterations,
                "tol": float(transport.get("tolerance", 1.0e-6)),
                "eta": float(transport.get("eta", 1.0e-12)),
            }
        )
        (fit_root / "method_params.yaml").write_text(
            yaml.safe_dump(method_config, sort_keys=True),
            encoding="utf-8",
        )
        started = time.perf_counter()
        runner(
            fit_root,
            condition,
            candidates,
            source_priors,
            target_priors,
            method_config,
        )
        runtime = time.perf_counter() - started
        scores = pd.read_parquet(fit_root / "cell_transport_scores.parquet")
        method_manifest = yaml.safe_load(
            (fit_root / "transport_manifest.yaml").read_text(encoding="utf-8")
        )
        metadata = method_manifest.get("metadata", {})
        row = {
            "run_id": run_root.name,
            "natural_endpoint": settings.endpoint,
            "variant": variant,
            "method": method_name,
            first: first_value,
            second: second_value,
            **identity,
            "converged": bool(metadata.get("converged", False)),
            "n_iterations": int(metadata.get("n_iter", 0)),
            "runtime_seconds": runtime,
            **_evaluate(scores=scores, truth=truth, endpoint=settings.endpoint),
        }
        new_row = pd.DataFrame([row], columns=columns)
        frame = (
            new_row if frame.empty else pd.concat([frame, new_row], ignore_index=True)
        ).sort_values([first, second])
        frame.to_csv(checkpoint_path, index=False)
        if retain_fit_artifacts:
            if not complete_fit_artifacts(fit_root):
                missing = [
                    filename
                    for filename in RETAINED_FIT_FILES
                    if not (fit_root / filename).is_file()
                ]
                raise ValueError(
                    f"Retained component-ablation fit lacks artifacts {missing}: "
                    f"{fit_root}"
                )
        else:
            shutil.rmtree(fit_root)
    return checkpoint_path


def run_component_ablation(config: dict[str, Any]):
    from experiments.mouse_spleen.pipeline import (
        MouseSpleenConfigError,
        StageResult,
        _mapping,
        _natural_endpoint_manifests,
        _string,
    )

    settings = _settings(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    root = output_root / "natural_mismatch/sensitivity/component_ablation_proliferating"
    endpoint_manifests = [
        manifest
        for manifest in _natural_endpoint_manifests(config)
        if str(manifest["metadata"]["natural_endpoint"]) == settings.endpoint
    ]
    if len(endpoint_manifests) != 1:
        raise MouseSpleenConfigError(
            "Expected exactly one Proliferating natural-mismatch endpoint manifest"
        )
    endpoint_manifest = endpoint_manifests[0]
    run_root = Path(str(endpoint_manifest["artifacts"]["run_root"]))
    candidate = str(endpoint_manifest["metadata"]["candidate_set"])
    condition = "natural_mismatch"
    candidate_path = run_root / f"candidates/{condition}/{candidate}/candidate_edges.parquet"
    profile = run_root / f"derived/{condition}/prior_profiles/default"
    source_path = profile / "source_priors.csv"
    target_path = profile / "target_priors.csv"
    truth_path = run_root / f"benchmark/{condition}/evaluation_truth/query_truth.csv"
    candidates = pd.read_parquet(candidate_path)
    source_priors = pd.read_csv(source_path)
    target_priors = pd.read_csv(target_path)
    truth = pd.read_csv(truth_path)
    transport = _mapping(config, "transport")
    source_hash = sha256_file(source_path)
    candidate_hash = sha256_file(candidate_path)
    match_pairs = [
        (low, high)
        for low in settings.match_tau_min_values
        for high in settings.match_tau_max_values
        if low <= high
    ]
    compatibility_pairs = [
        (tau, alpha) for tau in settings.compatibility_tau_values for alpha in settings.alpha_values
    ]
    match_checkpoint = _run_grid(
        variant="match_only",
        combinations=match_pairs,
        parameter_names=("tau_min", "tau_max"),
        method_name="coreot_match_only",
        runner=_run_coreot_match_only,
        root=root,
        condition=condition,
        candidate=candidate,
        run_root=run_root,
        candidates=candidates,
        source_priors=source_priors,
        target_priors=target_priors,
        truth=truth,
        settings=settings,
        transport=transport,
        source_priors_hash=source_hash,
        candidate_edges_hash=candidate_hash,
        retain_fit_artifacts="match_only" in settings.retained_fit_variants,
    )
    compatibility_checkpoint = _run_grid(
        variant="compatibility_only",
        combinations=compatibility_pairs,
        parameter_names=("tau_source", "alpha"),
        method_name="coreot_constant_tau",
        runner=_run_coreot_constant_tau,
        root=root,
        condition=condition,
        candidate=candidate,
        run_root=run_root,
        candidates=candidates,
        source_priors=source_priors,
        target_priors=target_priors,
        truth=truth,
        settings=settings,
        transport=transport,
        source_priors_hash=source_hash,
        candidate_edges_hash=candidate_hash,
        retain_fit_artifacts="compatibility_only" in settings.retained_fit_variants,
    )
    tmp_root = root / "tmp"
    if tmp_root.is_dir() and not any(tmp_root.rglob("*")):
        shutil.rmtree(tmp_root)
    manifest_path = root / "run_manifest.yaml"
    artifacts = {
        "match_only_checkpoint": match_checkpoint,
        "compatibility_only_checkpoint": compatibility_checkpoint,
    }
    for variant in settings.retained_fit_variants:
        artifacts[f"{variant}_fit_root"] = root / "fits" / variant
    write_manifest(
        manifest_path,
        Manifest(
            stage="run_natural_component_ablation",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "endpoint": settings.endpoint,
                "match_tau_min_values": list(settings.match_tau_min_values),
                "match_tau_max_values": list(settings.match_tau_max_values),
                "compatibility_tau_values": list(settings.compatibility_tau_values),
                "alpha_values": list(settings.alpha_values),
                "tau_target": settings.tau_target,
                "max_iterations": settings.max_iterations,
                "n_match_only_expected": len(match_pairs),
                "n_compatibility_only_expected": len(compatibility_pairs),
                "checkpointed": True,
                "retained_fit_variants": list(settings.retained_fit_variants),
                "source_priors_sha256": source_hash,
                "candidate_edges_sha256": candidate_hash,
            },
        ),
    )
    return StageResult("run_natural_component_ablation", root, artifacts, manifest_path)


def _metric_limits(frames: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for metric, _ in MANUSCRIPT_METRICS:
        values = pd.concat([frame[metric] for frame in frames]).astype(float)
        lower = float(values.min())
        upper = float(values.max())
        if np.isclose(lower, upper):
            limits[metric] = (0.0, 1.0)
        else:
            limits[metric] = (lower, upper)
    return limits


def _plot_variant(
    *,
    frame: pd.DataFrame,
    variant: str,
    settings: ComponentAblationSettings,
    limits: dict[str, tuple[float, float]],
    output_path: Path,
) -> None:
    if variant == "match_only":
        row_name, column_name = "tau_min", "tau_max"
        row_values = list(settings.match_tau_min_values)
        column_values = list(settings.match_tau_max_values)
    else:
        row_name, column_name = "tau_source", "alpha"
        row_values = list(settings.compatibility_tau_values)
        column_values = list(settings.alpha_values)
    display_metrics = ABSOLUTE_METRIC_DISPLAY_ROWS
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), squeeze=False)
    convergence = frame.pivot(index=row_name, columns=column_name, values="converged").reindex(
        index=row_values, columns=column_values
    )
    for axis, (metric, title) in zip(axes.flat, display_metrics, strict=True):
        pivot = frame.pivot(index=row_name, columns=column_name, values=metric).reindex(
            index=row_values, columns=column_values
        )
        image = axis.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            vmin=limits[metric][0],
            vmax=limits[metric][1],
            cmap="RdBu_r",
        )
        axis.set_xticks(range(len(column_values)), labels=[f"{value:g}" for value in column_values])
        axis.set_yticks(range(len(row_values)), labels=[f"{value:g}" for value in row_values])
        axis.set_xlabel(r"$\alpha$" if variant == "compatibility_only" else r"$\tau_{\max}$")
        axis.set_ylabel(r"$\tau_q$" if variant == "compatibility_only" else r"$\tau_{\min}$")
        axis.set_title(title)
        for row_index, row_value in enumerate(row_values):
            for column_index, column_value in enumerate(column_values):
                value = pivot.loc[row_value, column_value]
                if pd.isna(value):
                    continue
                red, green, blue, _ = image.cmap(image.norm(float(value)))
                text_color = (
                    "white"
                    if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                    else "black"
                )
                axis.text(
                    column_index,
                    row_index,
                    f"{float(value):.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )
                converged = convergence.loc[row_value, column_value]
                if not pd.isna(converged) and not bool(converged):
                    axis.text(
                        column_index,
                        row_index + 0.28,
                        "×",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="red",
                        fontweight="bold",
                    )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, parameters: tuple[str, str]) -> str:
    columns = [
        *parameters,
        "converged",
        "n_iterations",
        *[metric for metric, _ in MANUSCRIPT_METRICS],
    ]
    labels = {
        "tau_min": "Tau min",
        "tau_max": "Tau max",
        "tau_source": "Tau source",
        "alpha": "Alpha",
        "converged": "Converged",
        "n_iterations": "Iterations",
        **dict(MANUSCRIPT_METRICS),
    }
    lines = [
        "| " + " | ".join(labels[column] for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        rendered: list[str] = []
        for column, value in zip(columns, row, strict=True):
            if column == "converged":
                rendered.append("yes" if bool(value) else "no")
            elif column == "n_iterations":
                rendered.append(str(int(value)))
            elif column in parameters:
                rendered.append(f"{float(value):g}")
            else:
                rendered.append(f"{float(value):.3f}")
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    frame: pd.DataFrame,
    variant: str,
    table_path: Path,
    figure_path: Path,
    report_path: Path,
    settings: ComponentAblationSettings,
) -> None:
    if variant == "match_only":
        title = "Heterogeneous-query-penalty sensitivity at $\\alpha=0$"
        parameters = ("tau_min", "tau_max")
        definition = (
            "Compatibility is disabled with $\\alpha=0$; matchability-informed "
            "source penalties vary over the rectangular "
            "$(\\tau_{\\min},\\tau_{\\max})$ grid."
        )
    else:
        title = "Constant-$\\tau_q$ compatibility sensitivity"
        parameters = ("tau_source", "alpha")
        definition = (
            "Cell-specific matchability heterogeneity is removed by a constant source "
            "penalty; $(\\tau_{\\mathrm{source}},\\alpha)$ varies over the rectangular grid."
        )
    content = f"""# {title}: Proliferating component-ablation results

{definition} The target penalty is fixed at $\\tau_{{\\mathrm{{target}}}}={settings.tau_target:g}$. All cells report descriptive point estimates from the terminal solver output. A red `×` in the figure marks iteration-cap termination; such values are not converged solutions.

The figure and manuscript-facing table report $u$-based AUROC and
AP, forced represented-state accuracy, and forced represented-state macro-F1.
Abstention-dependent outcomes are omitted because this natural-mismatch
experiment has no matched full-reference control for threshold calibration.

![{title} heatmaps]({os.path.relpath(figure_path, start=report_path.parent)})

Full-precision source: [{table_path.name}]({os.path.relpath(table_path, start=report_path.parent)}).

{_markdown_table(frame, parameters)}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def aggregate_component_ablation(config: dict[str, Any]):
    from experiments.mouse_spleen.pipeline import StageResult, _mapping, _string

    run_result = run_component_ablation(config)
    settings = _settings(config)
    output_root = Path(_string(_mapping(config, "experiment"), "output_dir"))
    project_root = _project_root(output_root)
    docs_root = project_root / "docs"
    tables_root = run_result.root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    match = pd.read_csv(run_result.artifacts["match_only_checkpoint"])
    compatibility = pd.read_csv(run_result.artifacts["compatibility_only_checkpoint"])
    match["converged"] = _as_bool(match["converged"])
    compatibility["converged"] = _as_bool(compatibility["converged"])
    match_path = tables_root / "match_only_metrics.csv"
    compatibility_path = tables_root / "compatibility_only_metrics.csv"
    combined_path = tables_root / "component_ablation_metrics.csv"
    match.to_csv(match_path, index=False)
    compatibility.to_csv(compatibility_path, index=False)
    pd.concat([match, compatibility], ignore_index=True).to_csv(combined_path, index=False)
    limits = _metric_limits((match, compatibility))
    docs_figure_root = docs_root / "figs"
    docs_figure_root.mkdir(parents=True, exist_ok=True)
    match_figure = docs_figure_root / "manuscript_fig_mouse_spleen_proliferating_match_only.png"
    compatibility_figure = docs_figure_root / "manuscript_fig_mouse_spleen_component_minus_m.png"
    _plot_variant(
        frame=match,
        variant="match_only",
        settings=settings,
        limits=limits,
        output_path=match_figure,
    )
    _plot_variant(
        frame=compatibility,
        variant="compatibility_only",
        settings=settings,
        limits=limits,
        output_path=compatibility_figure,
    )
    match_report = docs_root / "manuscript_supp_mouse_spleen_proliferating_match_only.md"
    compatibility_report = (
        docs_root / "manuscript_supp_mouse_spleen_proliferating_compatibility_only.md"
    )
    _write_report(
        frame=match,
        variant="match_only",
        table_path=match_path,
        figure_path=match_figure,
        report_path=match_report,
        settings=settings,
    )
    _write_report(
        frame=compatibility,
        variant="compatibility_only",
        table_path=compatibility_path,
        figure_path=compatibility_figure,
        report_path=compatibility_report,
        settings=settings,
    )
    artifacts = {
        "match_only_metrics": match_path,
        "compatibility_only_metrics": compatibility_path,
        "combined_metrics": combined_path,
        "match_only_figure": match_figure,
        "compatibility_only_figure": compatibility_figure,
        "match_only_report": match_report,
        "compatibility_only_report": compatibility_report,
    }
    manifest_path = run_result.root / "manifest.yaml"
    write_manifest(
        manifest_path,
        Manifest(
            stage="aggregate_natural_component_ablation",
            artifacts={key: str(value) for key, value in artifacts.items()},
            metadata={
                "endpoint": settings.endpoint,
                "match_tau_min_values": list(settings.match_tau_min_values),
                "match_tau_max_values": list(settings.match_tau_max_values),
                "compatibility_tau_values": list(settings.compatibility_tau_values),
                "alpha_values": list(settings.alpha_values),
                "tau_target": settings.tau_target,
                "max_iterations": settings.max_iterations,
                "n_match_only_completed": int(len(match)),
                "n_compatibility_only_completed": int(len(compatibility)),
                "n_match_only_converged": int(match["converged"].sum()),
                "n_compatibility_only_converged": int(compatibility["converged"].sum()),
                "nonconverged_policy": "terminal_metrics_displayed_with_x_marker",
                "heatmap_metrics": [
                    metric for metric, _ in ABSOLUTE_METRIC_DISPLAY_ROWS
                ],
                "heatmap_layout": "2_by_2_row_major",
                "computed_source_metrics": [metric for metric, _ in COMPUTED_METRICS],
                "shared_color_limits": {metric: list(bounds) for metric, bounds in limits.items()},
                "point_estimates_only": True,
            },
        ),
    )
    return StageResult(
        "aggregate_natural_component_ablation", run_result.root, artifacts, manifest_path
    )
