from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, __version__ as pillow_version
import yaml


REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coreot.artifacts.hashes import sha256_file  # noqa: E402
from experiments.component_ablation_surfaces import (  # noqa: E402
    METRICS,
    _pbmc_condition_map,
    evaluate_scores,
    selected_specs,
)
from experiments.pbmc_rho_tau_heatmap import (  # noqa: E402
    FOCUSED_TAU_VALUES,
    _paired_columns,
    relative_percent,
    render_s2_aligned_heatmap,
    summarize,
)
from experiments.rho_attribution_search import (  # noqa: E402
    _checkpoint_columns,
    _render,
    summarize_results,
)


UNIT = REPO / (
    "results/submission_verification/PBMC/2026-08-05/"
    "pbmc_matchability_attribution_lineage_reconstruction"
)
FITS = UNIT / "regenerated/fits"
REGENERATED = UNIT / "regenerated"
COMPARISON = UNIT / "comparison"
RAW = REPO / "data/raw/kang_2018.h5ad"
METRIC_NAMES = tuple(metric for metric, _ in METRICS)
ENDPOINT_ORDER = [spec.endpoint for spec in selected_specs("pbmc")]
ENDPOINT_RANK = {endpoint: index for index, endpoint in enumerate(ENDPOINT_ORDER)}
TOLERANCE = 1.0e-12
DERIVATION_CODE_PATHS = {
    "pbmc_rho_tau_heatmap": REPO / "experiments/pbmc_rho_tau_heatmap.py",
    "rho_attribution_search": REPO / "experiments/rho_attribution_search.py",
    "component_ablation_surfaces": REPO / "experiments/component_ablation_surfaces.py",
    "mouse_component_ablation": REPO
    / "experiments/mouse_spleen/component_ablation.py",
    "artifact_hashes": REPO / "src/coreot/artifacts/hashes.py",
    "evaluation_metrics": REPO / "src/coreot/evaluation/metrics.py",
    "derivation_worker": Path(__file__).resolve(),
}

FOCUSED_CANONICAL = REPO / (
    "results/PBMC/sensitivity/"
    "rho_attribution_tau_surface_alpha0_range075_175"
)
ALPHA_CANONICAL = REPO / (
    "results/PBMC/sensitivity/rho_attribution_alpha_search"
)
FOCUSED_REGENERATED = REGENERATED / (
    "rho_attribution_tau_surface_alpha0_range075_175"
)
ALPHA_REGENERATED = REGENERATED / "rho_attribution_alpha_search"


def _load_yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _variant_paths(config_path: Path) -> tuple[Path, Path]:
    case_root = config_path.parent.parent
    return (
        case_root / "heterogeneous",
        case_root / "mean_matched_uniform",
    )


def _evaluate(
    *,
    config: dict[str, object],
    score_path: Path,
    condition: pd.Series,
    truth_cache: dict[str, pd.DataFrame],
) -> dict[str, object]:
    inputs = config["inputs"]
    truth_relative = str(inputs["truth"]["path"])
    if truth_relative not in truth_cache:
        truth_cache[truth_relative] = pd.read_csv(REPO / truth_relative)
    scores = pd.read_parquet(score_path)
    return evaluate_scores(
        experiment="pbmc",
        endpoint=str(config["endpoint"]),
        truth=truth_cache[truth_relative],
        scores=scores,
        pbmc_condition=condition,
    )


def _transport_metadata(root: Path) -> dict[str, object]:
    return _load_yaml(root / "fit_manifest.yaml")["metadata"][
        "transport_metadata"
    ]


def _source_hash(config: dict[str, object]) -> str:
    return str(config["combined_source_priors_sha256"])


def _pair_records() -> Iterable[tuple[dict[str, object], dict[str, object]]]:
    configs = sorted(FITS.rglob("heterogeneous/resolved_config.yaml"))
    if len(configs) != 300:
        raise RuntimeError(f"Expected 300 fit pairs; found {len(configs)}")
    for path in configs:
        heterogeneous_root, uniform_root = _variant_paths(path)
        yield _load_yaml(heterogeneous_root / "resolved_config.yaml"), {
            "heterogeneous_root": heterogeneous_root,
            "uniform_root": uniform_root,
            "uniform_config": _load_yaml(uniform_root / "resolved_config.yaml"),
        }


def _sort(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["__endpoint_rank"] = result["endpoint"].map(ENDPOINT_RANK)
    return (
        result.sort_values(["__endpoint_rank", *columns], kind="stable")
        .drop(columns="__endpoint_rank")
        .reset_index(drop=True)
    )


def build_tables() -> tuple[dict[str, Path], pd.DataFrame]:
    condition = _pbmc_condition_map(RAW)
    truth_cache: dict[str, pd.DataFrame] = {}
    focused_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    dependency_rows: list[dict[str, object]] = []
    status = pd.read_csv(UNIT / "audit/reconstruction_status.csv").set_index(
        "fit_id", verify_integrity=True
    )

    for heterogeneous, paired in _pair_records():
        heterogeneous_root = paired["heterogeneous_root"]
        uniform_root = paired["uniform_root"]
        uniform = paired["uniform_config"]
        h_metrics = _evaluate(
            config=heterogeneous,
            score_path=heterogeneous_root / "cell_scores.parquet",
            condition=condition,
            truth_cache=truth_cache,
        )
        u_metrics = _evaluate(
            config=uniform,
            score_path=uniform_root / "cell_scores.parquet",
            condition=condition,
            truth_cache=truth_cache,
        )
        h_transport = _transport_metadata(heterogeneous_root)
        u_transport = _transport_metadata(uniform_root)
        h_status = status.loc[str(heterogeneous["fit_id"])]
        u_status = status.loc[str(uniform["fit_id"])]
        method = heterogeneous["method"]
        common = {
            "endpoint": str(heterogeneous["endpoint"]),
            "seed": int(heterogeneous["seed"]),
            "run_id": str(heterogeneous["base_run_id"]),
            "tau_min": float(method["tau_min"]),
            "tau_max": float(method["tau_max"]),
            "mean_matched_tau": float(heterogeneous["mean_matched_tau"]),
            "alpha": float(method["alpha"]),
            "tau_target": float(method["tau_target"]),
            "candidate_edges_sha256": str(
                heterogeneous["inputs"]["candidates"]["sha256"]
            ),
            "source_priors_sha256": _source_hash(heterogeneous),
            "heterogeneous_converged": bool(h_transport["converged"]),
            "heterogeneous_n_iterations": int(h_transport["n_iter"]),
            "uniform_converged": bool(u_transport["converged"]),
            "uniform_n_iterations": int(u_transport["n_iter"]),
        }
        for metric in METRIC_NAMES:
            h_value = float(h_metrics[metric])
            u_value = float(u_metrics[metric])
            delta = h_value - u_value
            common[f"heterogeneous_{metric}"] = h_value
            common[f"uniform_{metric}"] = u_value
            common[f"delta_{metric}_heterogeneous_minus_uniform"] = delta

        family = str(heterogeneous["analysis_family"])
        if family == "focused_tau":
            row = common.copy()
            for metric in METRIC_NAMES:
                row[
                    f"relative_{metric}_heterogeneous_minus_uniform_percent"
                ] = relative_percent(
                    float(row[f"delta_{metric}_heterogeneous_minus_uniform"]),
                    float(row[f"uniform_{metric}"]),
                )
            row["diagonal_identity_verified"] = bool(
                np.isclose(row["tau_min"], row["tau_max"])
            )
            focused_rows.append(row)
        elif family == "alpha_search":
            selected_alpha = {
                "B cells": 4.0,
                "NK cells": 2.0,
                "Dendritic cells": 3.0,
            }[str(heterogeneous["endpoint"])]
            row = {
                "experiment": "pbmc",
                **common,
                "replicate": f"seed{int(heterogeneous['seed'])}",
                "alpha_ratio": float(method["alpha"]) / selected_alpha,
                "epsilon": float(method["epsilon"]),
                "max_iterations": int(method["max_iter"]),
                "tolerance": float(method["tol"]),
                "numerical_floor": float(method["numerical_floor"]),
                "heterogeneous_runtime_seconds": float(
                    h_status["reconstruction_runtime_seconds"]
                ),
                "uniform_runtime_seconds": float(
                    u_status["reconstruction_runtime_seconds"]
                ),
                "evaluation_scope": str(h_metrics["evaluation_scope"]),
                "n_detection": int(h_metrics["n_detection"]),
                "n_positive": int(h_metrics["n_positive"]),
            }
            alpha_rows.append(row)
        else:
            raise RuntimeError(f"Unknown family: {family}")

        for variant, root, config in (
            ("heterogeneous", heterogeneous_root, heterogeneous),
            ("mean_matched_uniform", uniform_root, uniform),
        ):
            dependency_rows.append(
                {
                    "analysis_family": family,
                    "endpoint": str(config["endpoint"]),
                    "seed": int(config["seed"]),
                    "fit_id": str(config["fit_id"]),
                    "variant": variant,
                    "resolved_config": str(
                        (root / "resolved_config.yaml").relative_to(REPO)
                    ),
                    "fit_manifest": str(
                        (root / "fit_manifest.yaml").relative_to(REPO)
                    ),
                    "cell_scores": str(
                        (root / "cell_scores.parquet").relative_to(REPO)
                    ),
                }
            )

    focused = _sort(
        pd.DataFrame(focused_rows), ["seed", "tau_min", "tau_max"]
    )
    alpha = _sort(pd.DataFrame(alpha_rows), ["seed", "alpha"])
    if len(focused) != 225 or len(alpha) != 75:
        raise RuntimeError(
            f"Unexpected reconstructed rows: focused={len(focused)}, alpha={len(alpha)}"
        )
    diagonal = np.isclose(focused["tau_min"], focused["tau_max"])
    delta_columns = [
        f"delta_{metric}_heterogeneous_minus_uniform" for metric in METRIC_NAMES
    ]
    diagonal_max = focused.loc[diagonal, delta_columns].abs().to_numpy().max()
    if diagonal_max > TOLERANCE:
        raise RuntimeError(f"Diagonal identity failed: {diagonal_max}")

    focused_checkpoint_columns = _paired_columns()
    focused_table_columns = [*focused_checkpoint_columns, "diagonal_identity_verified"]
    alpha_columns = _checkpoint_columns()
    focused = focused.loc[:, focused_table_columns]
    alpha = alpha.loc[:, alpha_columns]
    focused_summary = summarize(focused)
    alpha_summary = summarize_results(alpha)

    focused_tables = FOCUSED_REGENERATED / "tables"
    alpha_tables = ALPHA_REGENERATED / "tables"
    focused_tables.mkdir(parents=True, exist_ok=True)
    alpha_tables.mkdir(parents=True, exist_ok=True)
    focused_by_seed = focused_tables / "rho_tau_surface_range075_175_by_seed.csv"
    focused_summary_path = focused_tables / "rho_tau_surface_range075_175_summary.csv"
    alpha_by_replicate = alpha_tables / "rho_attribution_by_replicate.csv"
    alpha_summary_path = alpha_tables / "rho_attribution_summary.csv"
    focused.to_csv(focused_by_seed, index=False)
    focused_summary.to_csv(focused_summary_path, index=False)
    alpha.to_csv(alpha_by_replicate, index=False)
    alpha_summary.to_csv(alpha_summary_path, index=False)

    for endpoint in ENDPOINT_ORDER:
        slug = next(
            spec.slug for spec in selected_specs("pbmc") if spec.endpoint == endpoint
        )
        for seed in range(1, 6):
            focused_checkpoint = (
                FOCUSED_REGENERATED / "checkpoints" / slug / f"seed{seed}.csv"
            )
            alpha_checkpoint = (
                ALPHA_REGENERATED / "checkpoints" / slug / f"seed{seed}.csv"
            )
            focused_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            alpha_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            focused.loc[
                focused["endpoint"].eq(endpoint) & focused["seed"].eq(seed),
                focused_checkpoint_columns,
            ].to_csv(focused_checkpoint, index=False)
            alpha.loc[
                alpha["endpoint"].eq(endpoint) & alpha["seed"].eq(seed),
                alpha_columns,
            ].to_csv(alpha_checkpoint, index=False)

    figures = REGENERATED / "figures"
    focused_figure = figures / (
        "manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png"
    )
    alpha_figure = figures / "manuscript_fig_pbmc_rho_attribution_alpha_search.png"
    render_s2_aligned_heatmap(
        focused_summary,
        focused_figure,
        tau_values=FOCUSED_TAU_VALUES,
    )
    _render(alpha_summary, alpha, alpha_figure)

    artifacts = {
        "focused_by_seed": focused_by_seed,
        "focused_summary": focused_summary_path,
        "focused_figure": focused_figure,
        "alpha_by_replicate": alpha_by_replicate,
        "alpha_summary": alpha_summary_path,
        "alpha_figure": alpha_figure,
    }
    dependencies = pd.DataFrame(dependency_rows).sort_values("fit_id")
    dependencies.to_csv(UNIT / "audit/fit_dependencies.csv", index=False)
    return artifacts, dependencies


def _compare_frames(
    *,
    artifact: str,
    expected_path: Path,
    observed_path: Path,
    keys: list[str],
    excluded: set[str] | None = None,
) -> list[dict[str, object]]:
    excluded = excluded or set()
    expected = pd.read_csv(expected_path).sort_values(keys).reset_index(drop=True)
    observed = pd.read_csv(observed_path).sort_values(keys).reset_index(drop=True)
    if list(expected.columns) != list(observed.columns):
        raise RuntimeError(f"Column mismatch for {artifact}")
    if len(expected) != len(observed):
        raise RuntimeError(f"Row mismatch for {artifact}")
    for key in keys:
        if not expected[key].astype(str).equals(observed[key].astype(str)):
            raise RuntimeError(f"Key mismatch for {artifact}: {key}")
    rows: list[dict[str, object]] = []
    for column in expected.columns:
        if column in excluded:
            rows.append(
                {
                    "artifact": artifact,
                    "column": column,
                    "comparison": "excluded_reconstruction_runtime",
                    "max_absolute_difference": np.nan,
                    "status": "not_compared",
                }
            )
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
            same_nan = np.array_equal(np.isnan(left), np.isnan(right))
            difference = np.abs(left - right)
            maximum = float(np.nanmax(difference)) if difference.size else 0.0
            passed = same_nan and maximum <= TOLERANCE
            comparison = "numeric"
        else:
            passed = expected[column].fillna("<NA>").astype(str).equals(
                observed[column].fillna("<NA>").astype(str)
            )
            maximum = 0.0 if passed else np.nan
            comparison = "exact"
        rows.append(
            {
                "artifact": artifact,
                "column": column,
                "comparison": comparison,
                "max_absolute_difference": maximum,
                "status": "pass" if passed else "fail",
            }
        )
    return rows


def compare_outputs(artifacts: dict[str, Path]) -> None:
    COMPARISON.mkdir(parents=True, exist_ok=True)
    numerical: list[dict[str, object]] = []
    numerical.extend(
        _compare_frames(
            artifact="focused_by_seed",
            expected_path=FOCUSED_CANONICAL
            / "tables/rho_tau_surface_range075_175_by_seed.csv",
            observed_path=artifacts["focused_by_seed"],
            keys=["endpoint", "seed", "tau_min", "tau_max"],
        )
    )
    numerical.extend(
        _compare_frames(
            artifact="alpha_by_replicate",
            expected_path=ALPHA_CANONICAL / "tables/rho_attribution_by_replicate.csv",
            observed_path=artifacts["alpha_by_replicate"],
            keys=["endpoint", "seed", "alpha"],
            excluded={
                "heterogeneous_runtime_seconds",
                "uniform_runtime_seconds",
            },
        )
    )
    summaries: list[dict[str, object]] = []
    summaries.extend(
        _compare_frames(
            artifact="focused_summary",
            expected_path=FOCUSED_CANONICAL
            / "tables/rho_tau_surface_range075_175_summary.csv",
            observed_path=artifacts["focused_summary"],
            keys=["endpoint", "tau_min", "tau_max"],
        )
    )
    summaries.extend(
        _compare_frames(
            artifact="alpha_summary",
            expected_path=ALPHA_CANONICAL / "tables/rho_attribution_summary.csv",
            observed_path=artifacts["alpha_summary"],
            keys=["endpoint", "alpha"],
        )
    )
    numerical_frame = pd.DataFrame(numerical)
    summary_frame = pd.DataFrame(summaries)
    numerical_frame.to_csv(COMPARISON / "numerical_comparison.csv", index=False)
    summary_frame.to_csv(COMPARISON / "summary_comparison.csv", index=False)

    rendered_rows = []
    for artifact, canonical, regenerated in (
        (
            "focused_figure",
            REPO
            / "docs/figs/manuscript_fig_pbmc_rho_tau_surface_alpha0_range075_175.png",
            artifacts["focused_figure"],
        ),
        (
            "alpha_figure",
            REPO / "docs/figs/manuscript_fig_pbmc_rho_attribution_alpha_search.png",
            artifacts["alpha_figure"],
        ),
    ):
        with Image.open(canonical) as expected_image, Image.open(regenerated) as observed_image:
            expected_pixels = np.asarray(expected_image)
            observed_pixels = np.asarray(observed_image)
            same_dimensions = expected_pixels.shape == observed_pixels.shape
            pixel_equal = same_dimensions and np.array_equal(
                expected_pixels, observed_pixels
            )
            max_pixel_difference = (
                int(
                    np.abs(
                        expected_pixels.astype(np.int16)
                        - observed_pixels.astype(np.int16)
                    ).max()
                )
                if same_dimensions
                else np.nan
            )
        rendered_rows.append(
            {
                "artifact": artifact,
                "canonical_path": str(canonical.relative_to(REPO)),
                "regenerated_path": str(regenerated.relative_to(REPO)),
                "canonical_sha256": sha256_file(canonical),
                "regenerated_sha256": sha256_file(regenerated),
                "byte_exact": sha256_file(canonical) == sha256_file(regenerated),
                "same_dimensions": same_dimensions,
                "pixel_exact": pixel_equal,
                "max_pixel_difference": max_pixel_difference,
                "status": "pass" if pixel_equal else "fail",
            }
        )
    rendered = pd.DataFrame(rendered_rows)
    rendered.to_csv(COMPARISON / "rendered_comparison.csv", index=False)

    failed = int(numerical_frame["status"].eq("fail").sum())
    failed += int(summary_frame["status"].eq("fail").sum())
    failed += int(rendered["status"].eq("fail").sum())
    if failed:
        raise RuntimeError(f"Output comparison has {failed} failed checks")


def write_manifests(artifacts: dict[str, Path], dependencies: pd.DataFrame) -> None:
    dependency_index = UNIT / "audit/fit_dependencies.csv"
    code_sha256 = {
        name: sha256_file(path) for name, path in DERIVATION_CODE_PATHS.items()
    }
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": pillow_version,
        "platform": platform.platform(),
        "pyproject_sha256": sha256_file(REPO / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(REPO / "uv.lock"),
    }
    for family, root, names in (
        (
            "focused_tau",
            FOCUSED_REGENERATED,
            ("focused_by_seed", "focused_summary", "focused_figure"),
        ),
        (
            "alpha_search",
            ALPHA_REGENERATED,
            ("alpha_by_replicate", "alpha_summary", "alpha_figure"),
        ),
    ):
        local_dependencies = dependencies.loc[
            dependencies["analysis_family"].eq(family)
        ]
        payload = {
            "stage": "isolated-rho-attribution-derived-regeneration",
            "analysis_family": family,
            "fit_count": int(len(local_dependencies)),
            "paired_case_count": int(len(local_dependencies) // 2),
            "fit_dependency_index": str(
                dependency_index.relative_to(REPO)
            ),
            "fit_dependency_index_sha256": sha256_file(dependency_index),
            "fit_dependency_index_bytes": dependency_index.stat().st_size,
            "artifacts": {
                name: {
                    "path": str(artifacts[name].relative_to(REPO)),
                    "sha256": sha256_file(artifacts[name]),
                    "bytes": artifacts[name].stat().st_size,
                }
                for name in names
            },
            "comparison_tolerance": TOLERANCE,
            "code_sha256": code_sha256,
            "environment": environment,
            "command": (
                ".venv/bin/python "
                "results/submission_verification/PBMC/2026-08-05/"
                "pbmc_matchability_attribution_lineage_reconstruction/"
                "audit/workers/build_outputs.py"
            ),
            "status": "pass",
        }
        (root / "manifest.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


def main() -> None:
    artifacts, dependencies = build_tables()
    compare_outputs(artifacts)
    write_manifests(artifacts, dependencies)
    numeric = pd.read_csv(COMPARISON / "numerical_comparison.csv")
    summary = pd.read_csv(COMPARISON / "summary_comparison.csv")
    rendered = pd.read_csv(COMPARISON / "rendered_comparison.csv")
    maximum = pd.concat(
        [numeric["max_absolute_difference"], summary["max_absolute_difference"]]
    ).max()
    print(
        "derived outputs pass: "
        f"max_numeric_difference={maximum:.17g}, "
        f"pixel_exact={int(rendered['pixel_exact'].sum())}/{len(rendered)}"
    )


if __name__ == "__main__":
    main()
