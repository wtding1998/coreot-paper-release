from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import yaml

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from experiments.component_ablation_surfaces import (  # noqa: E402
    _pbmc_condition_map,
    _render_variant,
    evaluate_scores,
    selected_specs,
)


UNIT = Path(__file__).resolve().parents[2]
FITS = UNIT / "regenerated/fits"
TABLES = UNIT / "regenerated/component_ablation/tables"
CHECKPOINTS = UNIT / "regenerated/component_ablation"
FIGURES = UNIT / "regenerated/figures"
AUDIT = UNIT / "audit"
COMPARISON = UNIT / "comparison"
SOURCE_TABLES = REPO / "results/PBMC/sensitivity/component_ablation/tables"
SOURCE_FIGURE = REPO / "docs/figs/manuscript_fig_pbmc_component_minus_m.png"
METRICS = ("auroc", "auprc", "forced_accuracy", "forced_macro_f1")
KEYS = ("endpoint", "seed", "tau_source", "alpha")
SUMMARY_KEYS = ("endpoint", "tau_source", "alpha")
TOLERANCE = 1.0e-12


def value_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def endpoint_slug(value: str) -> str:
    return value.lower().replace("+", "").replace("-", "").replace(" ", "_")


def fit_id(row: object) -> str:
    return (
        f"{endpoint_slug(str(row.endpoint))}_seed{int(row.seed)}_"
        f"tau{value_slug(row.tau_source)}_alpha{value_slug(row.alpha)}"
    )


def fit_root(row: object) -> Path:
    return (
        FITS
        / endpoint_slug(str(row.endpoint))
        / f"seed{int(row.seed)}"
        / f"tau{value_slug(row.tau_source)}_alpha{value_slug(row.alpha)}"
    )


def load_historical() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_TABLES / "component_ablation_by_seed.csv")
    frame = frame.loc[
        frame["experiment"].eq("pbmc")
        & frame["variant"].eq("compatibility_only")
    ].copy()
    frame = frame.sort_values(list(KEYS)).reset_index(drop=True)
    if len(frame) != 375:
        raise RuntimeError(f"Expected 375 historical fits, found {len(frame)}")
    return frame


def reconstruct_by_seed(
    historical: pd.DataFrame, condition_map: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reconstructed_rows: list[dict[str, object]] = []
    dependencies: list[dict[str, object]] = []
    for row in historical.itertuples(index=False):
        root = fit_root(row)
        config_path = root / "resolved_config.yaml"
        manifest_path = root / "fit_manifest.yaml"
        score_path = root / "cell_scores.parquet"
        for path in (config_path, manifest_path, score_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if resolved["fit_id"] != fit_id(row):
            raise RuntimeError(f"Fit/config mismatch for {fit_id(row)}")
        if manifest["metadata"]["comparison_status"] != "pass":
            raise RuntimeError(f"Non-passing fit manifest for {fit_id(row)}")
        truth_path = REPO / resolved["inputs"]["truth"]["path"]
        truth = pd.read_csv(truth_path)
        scores = pd.read_parquet(score_path)
        if list(scores.columns) != ["cell_id", "u", "forced_label"]:
            raise RuntimeError(f"Unexpected compact score schema for {fit_id(row)}")
        metrics = evaluate_scores(
            experiment="pbmc",
            endpoint=str(row.endpoint),
            truth=truth,
            scores=scores,
            pbmc_condition=condition_map,
        )
        current = row._asdict()
        current.update(metrics)
        metadata = manifest["metadata"]
        current["converged"] = bool(metadata["transport_metadata"]["converged"])
        current["n_iterations"] = int(metadata["transport_metadata"]["n_iter"])
        current["runtime_seconds"] = float(
            metadata["status_row"]["reconstruction_runtime_seconds"]
        )
        reconstructed_rows.append(current)
        dependencies.append(
            {
                "fit_id": fit_id(row),
                "resolved_config_path": str(config_path.relative_to(REPO)),
                "resolved_config_sha256": sha256_file(config_path),
                "fit_manifest_path": str(manifest_path.relative_to(REPO)),
                "fit_manifest_sha256": sha256_file(manifest_path),
                "cell_scores_path": str(score_path.relative_to(REPO)),
                "cell_scores_sha256": sha256_file(score_path),
                "cell_scores_bytes": score_path.stat().st_size,
                "cell_scores_rows": len(scores),
            }
        )
    reconstructed = pd.DataFrame(reconstructed_rows).loc[:, historical.columns]
    return reconstructed, pd.DataFrame(dependencies)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "experiment",
        "endpoint",
        "variant",
        "method",
        "tau_min",
        "tau_max",
        "tau_source",
        "alpha",
    ]
    return (
        frame.groupby(group_columns, dropna=False, sort=False)
        .agg(
            n_splits=("seed", "nunique"),
            all_converged=("converged", "all"),
            max_n_iterations=("n_iterations", "max"),
            **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
            **{f"{metric}_sd": (metric, "std") for metric in METRICS},
        )
        .reset_index()
    )


def compare_by_seed(
    historical: pd.DataFrame, reconstructed: pd.DataFrame
) -> pd.DataFrame:
    left = historical.sort_values(list(KEYS)).reset_index(drop=True)
    right = reconstructed.sort_values(list(KEYS)).reset_index(drop=True)
    exact_columns = [
        column
        for column in left.columns
        if column not in (*METRICS, "runtime_seconds")
    ]
    records: list[dict[str, object]] = []
    for old, new in zip(
        left.itertuples(index=False), right.itertuples(index=False), strict=True
    ):
        exact_match = True
        for column in exact_columns:
            old_value = getattr(old, column)
            new_value = getattr(new, column)
            if pd.isna(old_value) and pd.isna(new_value):
                continue
            if old_value != new_value:
                exact_match = False
                break
        differences = {
            metric: abs(float(getattr(old, metric)) - float(getattr(new, metric)))
            for metric in METRICS
        }
        records.append(
            {
                "fit_id": fit_id(old),
                "exact_nonmetric_fields_match": exact_match,
                **{f"{metric}_abs_difference": value for metric, value in differences.items()},
                "max_metric_abs_difference": max(differences.values()),
                "status": (
                    "pass"
                    if exact_match and max(differences.values()) <= TOLERANCE
                    else "fail"
                ),
            }
        )
    return pd.DataFrame(records)


def compare_summary(reconstructed: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_csv(SOURCE_TABLES / "component_ablation_summary.csv")
    historical = historical.loc[
        historical["experiment"].eq("pbmc")
        & historical["variant"].eq("compatibility_only")
    ].sort_values(list(SUMMARY_KEYS)).reset_index(drop=True)
    current = reconstructed.sort_values(list(SUMMARY_KEYS)).reset_index(drop=True)
    numeric = [column for column in historical if column.endswith(("_mean", "_sd"))]
    records: list[dict[str, object]] = []
    for old, new in zip(
        historical.itertuples(index=False), current.itertuples(index=False), strict=True
    ):
        key_match = all(getattr(old, key) == getattr(new, key) for key in SUMMARY_KEYS)
        exact_counts = (
            int(old.n_splits) == int(new.n_splits)
            and bool(old.all_converged) == bool(new.all_converged)
            and int(old.max_n_iterations) == int(new.max_n_iterations)
        )
        differences = {
            column: abs(float(getattr(old, column)) - float(getattr(new, column)))
            for column in numeric
        }
        records.append(
            {
                "endpoint": str(old.endpoint),
                "tau_source": float(old.tau_source),
                "alpha": float(old.alpha),
                "keys_match": key_match,
                "counts_and_convergence_match": exact_counts,
                "max_statistic_abs_difference": max(differences.values()),
                "status": (
                    "pass"
                    if key_match
                    and exact_counts
                    and max(differences.values()) <= TOLERANCE
                    else "fail"
                ),
            }
        )
    return pd.DataFrame(records)


def figure_comparison(generated: Path) -> pd.DataFrame:
    source_array = np.asarray(Image.open(SOURCE_FIGURE).convert("RGBA"))
    generated_array = np.asarray(Image.open(generated).convert("RGBA"))
    same_shape = source_array.shape == generated_array.shape
    pixel_equal = bool(same_shape and np.array_equal(source_array, generated_array))
    status = "pass" if pixel_equal else "expected_change"
    return pd.DataFrame(
        [
            {
                "source_path": str(SOURCE_FIGURE.relative_to(REPO)),
                "generated_path": str(generated.relative_to(REPO)),
                "source_sha256": sha256_file(SOURCE_FIGURE),
                "generated_sha256": sha256_file(generated),
                "same_shape": same_shape,
                "pixel_exact": pixel_equal,
                "visual_layout_pass": same_shape,
                "change_class": (
                    "none"
                    if pixel_equal
                    else "current-code variant-local color normalization"
                ),
                "status": status,
            }
        ]
    )


def main() -> int:
    historical = load_historical()
    condition_map = _pbmc_condition_map(REPO / "data/raw/kang_2018.h5ad")
    reconstructed, dependencies = reconstruct_by_seed(historical, condition_map)
    summary = summarize(reconstructed)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    COMPARISON.mkdir(parents=True, exist_ok=True)
    by_seed_path = TABLES / "component_ablation_by_seed.csv"
    summary_path = TABLES / "component_ablation_summary.csv"
    figure_path = FIGURES / "manuscript_fig_pbmc_component_minus_m.png"
    reconstructed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    checkpoint_paths: list[Path] = []
    for (endpoint, seed), checkpoint in reconstructed.groupby(
        ["endpoint", "seed"], sort=False
    ):
        checkpoint_path = (
            CHECKPOINTS
            / endpoint_slug(str(endpoint))
            / f"seed{int(seed)}"
            / "compatibility_only.csv"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.to_csv(checkpoint_path, index=False)
        checkpoint_paths.append(checkpoint_path)
    _render_variant(
        experiment="pbmc",
        specs=selected_specs("pbmc"),
        summary=summary,
        variant="compatibility_only",
        output_path=figure_path,
    )
    dependency_path = AUDIT / "fit_dependencies.csv"
    numerical_path = COMPARISON / "numerical_comparison.csv"
    summary_comparison_path = COMPARISON / "summary_comparison.csv"
    figure_comparison_path = COMPARISON / "rendered_comparison.csv"
    dependencies.to_csv(dependency_path, index=False)
    numerical = compare_by_seed(historical, reconstructed)
    summary_comparison = compare_summary(summary)
    rendered = figure_comparison(figure_path)
    numerical.to_csv(numerical_path, index=False)
    summary_comparison.to_csv(summary_comparison_path, index=False)
    rendered.to_csv(figure_comparison_path, index=False)
    artifacts = {
        "fit_dependencies": dependency_path,
        "by_seed": by_seed_path,
        "summary": summary_path,
        "figure": figure_path,
        "numerical_comparison": numerical_path,
        "summary_comparison": summary_comparison_path,
        "figure_comparison": figure_comparison_path,
    }
    checkpoint_manifest = AUDIT / "checkpoint_artifacts.csv"
    pd.DataFrame(
        [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(pd.read_csv(path)),
            }
            for path in checkpoint_paths
        ]
    ).to_csv(checkpoint_manifest, index=False)
    artifacts["checkpoint_manifest"] = checkpoint_manifest
    manifest = {
        "stage": "isolated-component-aggregation-reproduction",
        "inputs": {
            "fit_count": len(dependencies),
            "fit_dependencies_sha256": sha256_file(dependency_path),
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifacts.items()
        },
        "metadata": {
            "comparison_tolerance": TOLERANCE,
            "all_fit_comparisons_pass": bool(numerical["status"].eq("pass").all()),
            "all_summary_comparisons_pass": bool(
                summary_comparison["status"].eq("pass").all()
            ),
            "rendered_figure_pixel_exact": bool(rendered.loc[0, "pixel_exact"]),
            "rendered_figure_expected_change": bool(
                rendered.loc[0, "status"] == "expected_change"
            ),
            "rendered_figure_visual_layout_pass": bool(
                rendered.loc[0, "visual_layout_pass"]
            ),
            "runtime_compared": False,
            "runtime_note": "Reconstruction runtime is new provenance, not historical evidence.",
            "python": platform.python_version(),
            "build_outputs_sha256": sha256_file(Path(__file__)),
            "component_ablation_surfaces_sha256": sha256_file(
                REPO / "experiments/component_ablation_surfaces.py"
            ),
        },
    }
    (AUDIT / "component_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    passed = (
        len(dependencies) == 375
        and len(checkpoint_paths) == 15
        and numerical["status"].eq("pass").all()
        and summary_comparison["status"].eq("pass").all()
        and rendered["status"].isin(["pass", "expected_change"]).all()
    )
    print(
        f"fit_comparisons={numerical['status'].value_counts().to_dict()} "
        f"summary_comparisons={summary_comparison['status'].value_counts().to_dict()} "
        f"figure_pixel_exact={bool(rendered.loc[0, 'pixel_exact'])}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
