from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import yaml

from coreot.results.hiha_figure2 import (
    ENDPOINTS as HIHA_ENDPOINTS,
    PANEL_B_METHODS as HIHA_DETECTION_METHODS,
    PANEL_D_METHODS as HIHA_TRANSFER_METHODS,
    PANEL_F_LABEL_COLORS as HIHA_LABEL_COLORS,
    REPRESENTATIVE_SEED as HIHA_REPRESENTATIVE_SEED,
)
from coreot.results.pbmc_figure3 import (
    CELL_TYPE_COLORS as PBMC_LABEL_COLORS,
    ENDPOINTS as PBMC_ENDPOINTS,
    PANEL_B_METHODS as PBMC_DETECTION_METHODS,
    PANEL_D_SEED as PBMC_REPRESENTATIVE_SEED,
    PANEL_F_METHODS as PBMC_TRANSFER_METHODS,
    SEEDS as PBMC_SEEDS,
)
from coreot.results.uot_baseline_figure_inputs import (
    FigureInputBinding,
    SealedFigureInputError,
    UOTBaselineFigureInputs,
    exact_cell_join,
)


CONTROLLED_METHODS = ("scdot", "tacco_ot")
CONTROLLED_DISPLAY = {"scdot": "scDOT", "tacco_ot": "TACCO-OT"}
RETAINED_METHODS = ("coreot_full", "seurat_anchor", "scmap_cluster", "chetah")
MOUSE_METHODS = ("coreot_full", "pamona", "scotv2", *RETAINED_METHODS[1:])
MOUSE_EXTERNAL_METHODS = ("pamona", "scotv2")
MOUSE_DISPLAY = {"pamona": "Pamona", "scotv2": "SCOTv2"}
HIHA_CONDITION_ID = "incomplete_reference"
HIHA_CANDIDATE_SET = "hiha_harmony30_k100"
HIHA_DETECTION_SCOPE = "local_within_broad_state"
HIHA_TRANSFER_SCOPE = "represented_state"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_evidence_path(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        results_index = path.parts.index("results")
    except ValueError as error:
        raise SealedFigureInputError(
            f"Evidence path is not under the repository results root: {path}"
        ) from error
    return str(Path(*path.parts[results_index:]))


def _copy_source_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Canonical figure source root is missing: {source}")
    if destination.exists():
        raise FileExistsError(f"Candidate source root already exists: {destination}")
    shutil.copytree(source, destination)


def _metric_value(frame: pd.DataFrame, metric: str) -> float:
    selected = frame.loc[frame["metric"].eq(metric)]
    if len(selected) != 1:
        raise SealedFigureInputError(f"Expected one sealed metric row for {metric}")
    value = float(selected.iloc[0]["value"])
    if not np.isfinite(value):
        raise SealedFigureInputError(f"Sealed metric is nonfinite: {metric}")
    return value


def _metric_row_id(frame: pd.DataFrame, metric: str) -> str:
    selected = frame.loc[frame["metric"].eq(metric), "row_id"]
    if len(selected) != 1:
        raise SealedFigureInputError(f"Expected one sealed row id for {metric}")
    return str(selected.iloc[0])


def _prediction_lineage(
    *,
    figure: str,
    panel: str,
    binding: FigureInputBinding,
    predictions: pd.DataFrame,
    transformation: str,
) -> dict[str, object]:
    return {
        "figure": figure,
        "panel": panel,
        "family": binding.family,
        "endpoint": binding.endpoint,
        "seed": binding.seed,
        "method": binding.method,
        "condition_id": binding.condition_id,
        "evaluation_scope": "per_cell",
        "metric": "z_absent_score_or_forced_label",
        "sealed_summary_row_id": "",
        "per_run_artifact": predictions["prediction_artifact"].iloc[0],
        "per_run_sha256": predictions["prediction_sha256"].iloc[0],
        "aggregation_code": (
            "src/coreot/results/main_figure_external_baselines.py::"
            f"{transformation}"
        ),
        "figure_source": transformation,
    }


def _controlled_metric_lineage(
    *,
    figure: str,
    panel: str,
    binding: FigureInputBinding,
    metrics: pd.DataFrame,
    metric: str,
    source_name: str,
) -> dict[str, object]:
    row = metrics.loc[metrics["metric"].eq(metric)]
    if len(row) != 1:
        raise SealedFigureInputError(f"Missing lineage row for {metric}")
    row = row.iloc[0]
    return {
        "figure": figure,
        "panel": panel,
        "family": binding.family,
        "endpoint": binding.endpoint,
        "seed": binding.seed,
        "method": binding.method,
        "condition_id": binding.condition_id,
        "evaluation_scope": str(row["evaluation_scope"]),
        "metric": metric,
        "sealed_summary_row_id": str(row["row_id"]),
        "per_run_artifact": str(row["prediction_artifact"]),
        "per_run_sha256": str(row["prediction_sha256"]),
        "aggregation_code": (
            "experiments/uot_baseline_pilot/portable_package.py -> "
            "src/coreot/results/main_figure_external_baselines.py"
        ),
        "figure_source": source_name,
    }


def _authoritative_hiha_rows(
    *,
    table_path: Path,
    evaluation_scope: str,
) -> pd.DataFrame:
    frame = pd.read_csv(table_path)
    required = {
        "run_id",
        "held_out_label",
        "seed",
        "condition_id",
        "candidate_set",
        "method",
        "score",
    }
    if evaluation_scope == HIHA_DETECTION_SCOPE:
        required.update({"evaluation_scope", "auprc", "auroc", "prevalence"})
    else:
        required.update(
            {"n_shared", "n_forced_labeled", "forced_macro_f1", "forced_accuracy"}
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SealedFigureInputError(
            f"Authoritative HIHA table is missing columns {missing}: {table_path}"
        )
    selected = frame.loc[
        frame["held_out_label"].isin(HIHA_ENDPOINTS)
        & frame["seed"].isin(range(1, 6))
        & frame["condition_id"].eq(HIHA_CONDITION_ID)
        & frame["candidate_set"].eq(HIHA_CANDIDATE_SET)
        & frame["method"].eq("coreot_full")
        & frame["score"].eq("u")
    ].copy()
    if "evaluation_scope" in selected:
        selected = selected.loc[
            selected["evaluation_scope"].eq(evaluation_scope)
        ].copy()
    keys = [
        "held_out_label",
        "seed",
        "condition_id",
        "candidate_set",
        "method",
        "score",
    ]
    expected = {(endpoint, seed) for endpoint in HIHA_ENDPOINTS for seed in range(1, 6)}
    observed = set(selected[["held_out_label", "seed"]].itertuples(index=False, name=None))
    if observed != expected or selected.duplicated(keys).any():
        raise SealedFigureInputError(
            "Authoritative HIHA CoRe-OT rows do not provide one row per endpoint "
            f"and seed: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return selected


def _authoritative_coreot_predictions(
    *,
    runs_root: Path,
    run_id: str,
) -> tuple[pd.DataFrame, Path]:
    path = (
        runs_root
        / run_id
        / "scoring"
        / HIHA_CONDITION_ID
        / HIHA_CANDIDATE_SET
        / "cell_scores.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Authoritative HIHA cell scores are missing: {path}")
    frame = pd.read_parquet(path)
    required = {"cell_id", "method", "u", "forced_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SealedFigureInputError(
            f"Authoritative HIHA cell scores are missing columns {missing}: {path}"
        )
    selected = frame.loc[
        frame["method"].astype(str).eq("coreot_full"),
        ["cell_id", "u", "forced_label"],
    ].copy()
    selected["cell_id"] = selected["cell_id"].astype(str)
    if selected.empty or selected["cell_id"].duplicated().any():
        raise SealedFigureInputError(
            f"Authoritative HIHA CoRe-OT cell rows are empty or duplicated: {path}"
        )
    selected["u"] = pd.to_numeric(selected["u"], errors="coerce")
    if not np.isfinite(selected["u"].to_numpy(dtype=float)).all():
        raise SealedFigureInputError(
            f"Authoritative HIHA CoRe-OT scores are nonfinite: {path}"
        )
    selected["forced_label"] = selected["forced_label"].fillna("").astype(str)
    return selected, path


def _write_candidate_manifest(
    *,
    figure_root: Path,
    figure: str,
    package_root: Path,
    canonical_source_root: Path,
    lineage_path: Path,
    preserved_panels: Iterable[str],
    additional_input_hashes: dict[str, str] | None = None,
) -> Path:
    source_files = sorted(
        path for path in figure_root.rglob("*") if path.is_file()
    )
    manifest_path = figure_root / "candidate_manifest.yaml"
    manifest = {
        "schema_version": "main_figure_external_baseline_candidate_v1",
        "figure": figure,
        "sealed_package": _portable_evidence_path(package_root),
        "sealed_verification_report": _portable_evidence_path(
            package_root / "verification/report.json"
        ),
        "canonical_source_root": str(canonical_source_root),
        "preserved_panels": list(preserved_panels),
        "lineage": str(lineage_path),
        "model_fitting_executed": False,
        "source_data_sha256": {
            str(path.relative_to(figure_root)): sha256_file(path)
            for path in source_files
            if path != manifest_path
        },
        "input_hashes": {
            "sealed_manifest": sha256_file(package_root / "manifest.json"),
            "sealed_verification_report": sha256_file(
                package_root / "verification/report.json"
            ),
            "sealed_all_by_run_metrics": sha256_file(
                package_root / "metrics/all_by_run_metrics.csv"
            ),
            "sealed_controlled_summary_metrics": sha256_file(
                package_root / "metrics/controlled_summary_metrics.csv"
            ),
            "sealed_mouse_summary_metrics": sha256_file(
                package_root / "metrics/mouse_summary_metrics.csv"
            ),
            "sealed_mouse_bootstrap_metrics": sha256_file(
                package_root / "metrics/mouse_bootstrap_metrics.csv"
            ),
            **(additional_input_hashes or {}),
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def prepare_hiha_candidate_sources(
    *,
    package_root: Path,
    canonical_source_root: Path,
    candidate_figure_root: Path,
    authoritative_table_root: Path = Path("results/HIHA_DC/main/tables"),
    authoritative_runs_root: Path,
) -> dict[str, Path]:
    adapter = UOTBaselineFigureInputs(package_root)
    source_root = candidate_figure_root / "source_data"
    _copy_source_tree(canonical_source_root, source_root)
    lineage: list[dict[str, object]] = []

    panel_b_path = source_root / "panel_b_detection.csv"
    panel_b = pd.read_csv(panel_b_path)
    prevalence = panel_b.groupby(["held_out_label", "seed"])[
        "positive_prevalence"
    ].nunique()
    if not prevalence.eq(1).all():
        raise SealedFigureInputError("HIHA retained prevalence is not unique by split")
    prevalence_map = (
        panel_b.groupby(["held_out_label", "seed"])["positive_prevalence"]
        .first()
        .to_dict()
    )
    retained = panel_b.loc[
        panel_b["method"].isin(RETAINED_METHODS)
        & panel_b["method"].ne("coreot_full")
    ].copy()
    authoritative_detection_path = authoritative_table_root / "main_detection_by_run.csv"
    all_authoritative_detection = pd.read_csv(authoritative_detection_path)
    retained_authority = all_authoritative_detection.loc[
        all_authoritative_detection["held_out_label"].isin(HIHA_ENDPOINTS)
        & all_authoritative_detection["seed"].isin(range(1, 6))
        & all_authoritative_detection["condition_id"].eq(HIHA_CONDITION_ID)
        & all_authoritative_detection["evaluation_scope"].eq(HIHA_DETECTION_SCOPE)
        & all_authoritative_detection["method"].isin(
            set(RETAINED_METHODS) - {"coreot_full"}
        ),
        ["held_out_label", "seed", "method", "score", "auprc", "auroc"],
    ].rename(columns={"auprc": "authority_ap", "auroc": "authority_auroc"})
    retained = retained.merge(
        retained_authority,
        on=["held_out_label", "seed", "method", "score"],
        how="left",
        validate="one_to_one",
    )
    if retained[["authority_ap", "authority_auroc"]].isna().any().any():
        raise SealedFigureInputError(
            "Authoritative HIHA detection rows do not cover every retained method"
        )
    if not np.allclose(
        retained["average_precision"],
        retained["authority_ap"],
        rtol=0,
        atol=1.0e-12,
    ):
        raise SealedFigureInputError(
            "Retained HIHA Panel C AP differs from the authoritative table"
        )
    if "auroc" in retained and retained["auroc"].notna().any():
        if not np.allclose(
            retained["auroc"],
            retained["authority_auroc"],
            rtol=0,
            atol=1.0e-12,
        ):
            raise SealedFigureInputError(
                "Retained HIHA Panel C AUROC differs from the authoritative table"
            )
    retained["auroc"] = retained["authority_auroc"].astype(float)
    retained["auroc_source_path"] = _portable_evidence_path(
        authoritative_detection_path
    )
    retained["sealed_auroc_row_id"] = ""
    retained = retained.drop(columns=["authority_ap", "authority_auroc"])
    authoritative_detection = _authoritative_hiha_rows(
        table_path=authoritative_detection_path,
        evaluation_scope=HIHA_DETECTION_SCOPE,
    )
    authoritative_input_hashes = {
        "authoritative_hiha_detection_table": sha256_file(
            authoritative_detection_path
        )
    }
    coreot_rows = []
    for row in authoritative_detection.itertuples(index=False):
        _, prediction_path = _authoritative_coreot_predictions(
            runs_root=authoritative_runs_root,
            run_id=str(row.run_id),
        )
        prediction_hash = sha256_file(prediction_path)
        artifact_key = f"authoritative_coreot_{row.held_out_label}_seed{row.seed}"
        authoritative_input_hashes[artifact_key] = prediction_hash
        coreot_rows.append(
            {
                "run_id": row.run_id,
                "held_out_label": row.held_out_label,
                "seed": row.seed,
                "method": "coreot_full",
                "method_display": "CoRe-OT",
                "score": "u",
                "average_precision": float(row.auprc),
                "auroc": float(row.auroc),
                "positive_prevalence": float(row.prevalence),
                "source_path": (
                    f"{_portable_evidence_path(authoritative_detection_path)}#"
                    f"endpoint={row.held_out_label};seed={row.seed};"
                    f"condition_id={row.condition_id};candidate_set={row.candidate_set};"
                    "method=coreot_full;score=u;"
                    f"evaluation_scope={HIHA_DETECTION_SCOPE}"
                ),
                "sealed_row_id": "",
                "auroc_source_path": _portable_evidence_path(
                    authoritative_detection_path
                ),
                "sealed_auroc_row_id": "",
                "prediction_artifact": (
                    f"runs/{row.run_id}/scoring/{HIHA_CONDITION_ID}/"
                    f"{HIHA_CANDIDATE_SET}/cell_scores.parquet"
                ),
                "prediction_sha256": prediction_hash,
            }
        )
        lineage.append(
            {
                "figure": "Figure 2",
                "panel": "C",
                "family": "hiha",
                "endpoint": row.held_out_label,
                "seed": row.seed,
                "condition_id": row.condition_id,
                "candidate_set": row.candidate_set,
                "method": row.method,
                "score": row.score,
                "evaluation_scope": row.evaluation_scope,
                "metric": "average_precision_and_auroc",
                "sealed_summary_row_id": "",
                "summary_table": _portable_evidence_path(
                    authoritative_detection_path
                ),
                "summary_table_sha256": sha256_file(
                    authoritative_detection_path
                ),
                "summary_row_key": (
                    f"endpoint={row.held_out_label};seed={row.seed};"
                    f"condition_id={row.condition_id};candidate_set={row.candidate_set};"
                    f"method={row.method};score={row.score};"
                    f"evaluation_scope={row.evaluation_scope}"
                ),
                "per_run_artifact": (
                    f"runs/{row.run_id}/scoring/{HIHA_CONDITION_ID}/"
                    f"{HIHA_CANDIDATE_SET}/cell_scores.parquet"
                ),
                "per_run_sha256": prediction_hash,
                "aggregation_code": (
                    "src/coreot/results/hiha.py::write_hiha_reformulation -> "
                    "src/coreot/results/main_figure_external_baselines.py::"
                    "prepare_hiha_candidate_sources"
                ),
                "figure_source": "source_data/panel_b_detection.csv",
            }
        )
    new_rows: list[dict[str, object]] = []
    for endpoint in HIHA_ENDPOINTS:
        for seed in range(1, 6):
            for method in CONTROLLED_METHODS:
                binding = FigureInputBinding(
                    "hiha", endpoint, seed, method, "incomplete_reference"
                )
                metrics = adapter.controlled_metrics(
                    binding, evaluation_scope="within_cdc2"
                )
                ap_row = metrics.loc[
                    metrics["metric"].eq("average_precision")
                ].iloc[0]
                auroc_row = metrics.loc[metrics["metric"].eq("auroc")].iloc[0]
                new_rows.append(
                    {
                        "run_id": ap_row["run_id"],
                        "held_out_label": endpoint,
                        "seed": seed,
                        "method": method,
                        "method_display": CONTROLLED_DISPLAY[method],
                        "score": "z_absent_score",
                        "average_precision": float(ap_row["value"]),
                        "auroc": float(auroc_row["value"]),
                        "positive_prevalence": float(prevalence_map[(endpoint, seed)]),
                        "source_path": (
                            f"{_portable_evidence_path(package_root)}/"
                            "metrics/all_by_run_metrics.csv#"
                            f"row_id={ap_row['row_id']}"
                        ),
                        "sealed_row_id": ap_row["row_id"],
                        "auroc_source_path": (
                            f"{_portable_evidence_path(package_root)}/"
                            "metrics/all_by_run_metrics.csv#"
                            f"row_id={auroc_row['row_id']}"
                        ),
                        "sealed_auroc_row_id": auroc_row["row_id"],
                        "prediction_artifact": ap_row["prediction_artifact"],
                        "prediction_sha256": ap_row["prediction_sha256"],
                    }
                )
                for metric in ("average_precision", "auroc"):
                    lineage.append(
                        _controlled_metric_lineage(
                            figure="Figure 2",
                            panel="C",
                            binding=binding,
                            metrics=metrics,
                            metric=metric,
                            source_name="source_data/panel_b_detection.csv",
                        )
                    )
    panel_b = pd.concat(
        [retained, pd.DataFrame(coreot_rows), pd.DataFrame(new_rows)],
        ignore_index=True,
        sort=False,
    )
    order = [method for method, _, _ in HIHA_DETECTION_METHODS]
    panel_b["method"] = pd.Categorical(panel_b["method"], order, ordered=True)
    panel_b.sort_values(["held_out_label", "method", "seed"]).to_csv(
        panel_b_path, index=False
    )

    panel_d_path = source_root / "panel_e_label_transfer.csv"
    panel_d = pd.read_csv(panel_d_path)
    retained = panel_d.loc[
        panel_d["method"].isin(RETAINED_METHODS)
        & panel_d["method"].ne("coreot_full")
    ].copy()
    authoritative_transfer_path = (
        authoritative_table_root / "shared_label_transfer_by_run.csv"
    )
    authoritative_transfer = _authoritative_hiha_rows(
        table_path=authoritative_transfer_path,
        evaluation_scope=HIHA_TRANSFER_SCOPE,
    )
    authoritative_input_hashes["authoritative_hiha_transfer_table"] = sha256_file(
        authoritative_transfer_path
    )
    coreot_rows = []
    for row in authoritative_transfer.itertuples(index=False):
        _, prediction_path = _authoritative_coreot_predictions(
            runs_root=authoritative_runs_root,
            run_id=str(row.run_id),
        )
        prediction_hash = sha256_file(prediction_path)
        coreot_rows.append(
            {
                "run_id": row.run_id,
                "held_out_label": row.held_out_label,
                "seed": row.seed,
                "condition_id": row.condition_id,
                "candidate_set": row.candidate_set,
                "method": "coreot_full",
                "method_display": "CoRe-OT",
                "score": "u",
                "n_shared": int(row.n_shared),
                "n_forced_labeled": int(row.n_forced_labeled),
                "forced_macro_f1": float(row.forced_macro_f1),
                "forced_accuracy": float(row.forced_accuracy),
                "source_path": _portable_evidence_path(authoritative_transfer_path),
                "sealed_macro_f1_row_id": "",
                "sealed_accuracy_row_id": "",
            }
        )
        lineage.append(
            {
                "figure": "Figure 2",
                "panel": "D",
                "family": "hiha",
                "endpoint": row.held_out_label,
                "seed": row.seed,
                "condition_id": row.condition_id,
                "candidate_set": row.candidate_set,
                "method": row.method,
                "score": row.score,
                "evaluation_scope": HIHA_TRANSFER_SCOPE,
                "metric": "forced_accuracy_and_macro_f1",
                "sealed_summary_row_id": "",
                "summary_table": _portable_evidence_path(
                    authoritative_transfer_path
                ),
                "summary_table_sha256": sha256_file(
                    authoritative_transfer_path
                ),
                "summary_row_key": (
                    f"endpoint={row.held_out_label};seed={row.seed};"
                    f"condition_id={row.condition_id};candidate_set={row.candidate_set};"
                    f"method={row.method};score={row.score};"
                    f"evaluation_scope={HIHA_TRANSFER_SCOPE}"
                ),
                "per_run_artifact": (
                    f"runs/{row.run_id}/scoring/{HIHA_CONDITION_ID}/"
                    f"{HIHA_CANDIDATE_SET}/cell_scores.parquet"
                ),
                "per_run_sha256": prediction_hash,
                "aggregation_code": (
                    "src/coreot/results/compare_baselines.py -> "
                    "src/coreot/results/main_figure_external_baselines.py::"
                    "prepare_hiha_candidate_sources"
                ),
                "figure_source": "source_data/panel_e_label_transfer.csv",
            }
        )
    new_rows = []
    for endpoint in HIHA_ENDPOINTS:
        for seed in range(1, 6):
            for method in CONTROLLED_METHODS:
                binding = FigureInputBinding(
                    "hiha", endpoint, seed, method, "incomplete_reference"
                )
                metrics = adapter.controlled_metrics(
                    binding, evaluation_scope="represented_state"
                )
                run_id = str(metrics.iloc[0]["run_id"])
                n_cells = int(metrics.iloc[0]["n_cells"])
                new_rows.append(
                    {
                        "run_id": run_id,
                        "held_out_label": endpoint,
                        "seed": seed,
                        "condition_id": "incomplete_reference",
                        "candidate_set": "external_reference_mapping",
                        "method": method,
                        "method_display": CONTROLLED_DISPLAY[method],
                        "score": "z_absent_score",
                        "n_shared": n_cells,
                        "n_forced_labeled": n_cells,
                        "forced_macro_f1": _metric_value(metrics, "forced_macro_f1"),
                        "forced_accuracy": _metric_value(metrics, "forced_accuracy"),
                        "source_path": _portable_evidence_path(
                            package_root / "metrics/all_by_run_metrics.csv"
                        ),
                        "sealed_macro_f1_row_id": _metric_row_id(
                            metrics, "forced_macro_f1"
                        ),
                        "sealed_accuracy_row_id": _metric_row_id(
                            metrics, "forced_accuracy"
                        ),
                    }
                )
                for metric in ("forced_macro_f1", "forced_accuracy"):
                    lineage.append(
                        _controlled_metric_lineage(
                            figure="Figure 2",
                            panel="D",
                            binding=binding,
                            metrics=metrics,
                            metric=metric,
                            source_name="source_data/panel_e_label_transfer.csv",
                        )
                    )
    panel_d = pd.concat(
        [retained, pd.DataFrame(coreot_rows), pd.DataFrame(new_rows)],
        ignore_index=True,
        sort=False,
    )
    panel_d["method"] = pd.Categorical(
        panel_d["method"],
        [method for method, _, _ in HIHA_TRANSFER_METHODS],
        ordered=True,
    )
    panel_d.sort_values(["held_out_label", "method", "seed"]).to_csv(
        panel_d_path, index=False
    )

    panel_e_path = source_root / "panel_d_umap_cells.csv"
    panel_e = pd.read_csv(panel_e_path, float_precision="round_trip")
    truth = panel_e.loc[panel_e["map_id"].eq("truth")].copy()
    retained_methods = tuple(
        method for method in RETAINED_METHODS if method != "coreot_full"
    )
    retained = panel_e.loc[panel_e["map_id"].isin(retained_methods)].copy()
    new_frames = []
    for endpoint in HIHA_ENDPOINTS:
        base = truth.loc[
            truth["held_out_label"].eq(endpoint)
            & truth["seed"].eq(HIHA_REPRESENTATIVE_SEED)
        ].copy()
        run_id = str(
            authoritative_detection.loc[
                authoritative_detection["held_out_label"].eq(endpoint)
                & authoritative_detection["seed"].eq(HIHA_REPRESENTATIVE_SEED),
                "run_id",
            ].iloc[0]
        )
        predictions, prediction_path = _authoritative_coreot_predictions(
            runs_root=authoritative_runs_root,
            run_id=run_id,
        )
        joined = exact_cell_join(base, predictions[["cell_id", "u"]])
        eligible = joined["is_within_cdc2"].astype(bool)
        n_select = int(joined["is_held_out_truth"].sum())
        ranked_ids = (
            joined.loc[eligible, ["cell_id", "u"]]
            .sort_values(
                ["u", "cell_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(n_select)["cell_id"]
        )
        joined["map_id"] = "coreot_full"
        joined["map_display"] = "CoRe-OT"
        joined["score_run_id"] = run_id
        joined["score_name"] = "u"
        joined["is_selected"] = joined["cell_id"].isin(ranked_ids)
        joined["oriented_score"] = joined["u"]
        new_frames.append(joined[panel_e.columns])
        lineage.append(
            {
                "figure": "Figure 2",
                "panel": "E",
                "family": "hiha",
                "endpoint": endpoint,
                "seed": HIHA_REPRESENTATIVE_SEED,
                "method": "coreot_full",
                "condition_id": HIHA_CONDITION_ID,
                "candidate_set": HIHA_CANDIDATE_SET,
                "evaluation_scope": HIHA_DETECTION_SCOPE,
                "score": "u",
                "metric": "u_and_oracle_sized_rank",
                "sealed_summary_row_id": "",
                "per_run_artifact": (
                    f"runs/{run_id}/scoring/{HIHA_CONDITION_ID}/"
                    f"{HIHA_CANDIDATE_SET}/cell_scores.parquet"
                ),
                "per_run_sha256": sha256_file(prediction_path),
                "aggregation_code": (
                    "src/coreot/results/main_figure_external_baselines.py::"
                    "prepare_hiha_candidate_sources"
                ),
                "figure_source": "source_data/panel_d_umap_cells.csv",
            }
        )
        for method in CONTROLLED_METHODS:
            binding = FigureInputBinding(
                "hiha",
                endpoint,
                HIHA_REPRESENTATIVE_SEED,
                method,
                "incomplete_reference",
            )
            predictions = adapter.predictions(binding)
            joined = exact_cell_join(
                base,
                predictions[["cell_id", "weak_support_score"]],
            )
            eligible = joined["is_within_cdc2"].astype(bool)
            n_select = int(joined["is_held_out_truth"].sum())
            ranked_ids = (
                joined.loc[eligible, ["cell_id", "weak_support_score"]]
                .sort_values(
                    ["weak_support_score", "cell_id"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                .head(n_select)["cell_id"]
            )
            joined["map_id"] = method
            joined["map_display"] = CONTROLLED_DISPLAY[method]
            joined["score_run_id"] = joined["run_id"]
            joined["score_name"] = "z_absent_score"
            joined["is_selected"] = joined["cell_id"].isin(ranked_ids)
            joined["oriented_score"] = joined["weak_support_score"]
            new_frames.append(joined[panel_e.columns])
            lineage.append(
                _prediction_lineage(
                    figure="Figure 2",
                    panel="E",
                    binding=binding,
                    predictions=predictions,
                    transformation="source_data/panel_d_umap_cells.csv",
                )
            )
    panel_e = pd.concat([truth, retained, *new_frames], ignore_index=True)
    panel_e["map_id"] = pd.Categorical(
        panel_e["map_id"],
        ["truth", *[method for method, _, _ in HIHA_DETECTION_METHODS]],
        ordered=True,
    )
    panel_e.sort_values(
        ["held_out_label", "map_id"], kind="mergesort"
    ).to_csv(panel_e_path, index=False)

    panel_f_path = source_root / "panel_f_label_assignment_cells.csv"
    panel_f = pd.read_csv(panel_f_path)
    truth = panel_f.loc[panel_f["map_id"].eq("truth")].copy()
    retained = panel_f.loc[panel_f["map_id"].isin(retained_methods)].copy()
    new_frames = []
    for endpoint in HIHA_ENDPOINTS:
        base = truth.loc[
            truth["held_out_label"].eq(endpoint)
            & truth["seed"].eq(HIHA_REPRESENTATIVE_SEED)
        ].copy()
        panel_d_row = authoritative_transfer.loc[
            authoritative_transfer["held_out_label"].eq(endpoint)
            & authoritative_transfer["seed"].eq(HIHA_REPRESENTATIVE_SEED)
        ]
        run_id = str(panel_d_row.iloc[0]["run_id"])
        predictions, prediction_path = _authoritative_coreot_predictions(
            runs_root=authoritative_runs_root,
            run_id=run_id,
        )
        joined = exact_cell_join(
            base,
            predictions[["cell_id", "forced_label"]].rename(
                columns={"forced_label": "authoritative_forced_label"}
            ),
        )
        represented = joined["is_represented_state"].astype(bool)
        represented_labels = joined.loc[
            represented, "authoritative_forced_label"
        ]
        unexpected = sorted(set(represented_labels) - set(HIHA_LABEL_COLORS))
        if unexpected:
            raise SealedFigureInputError(
                f"HIHA CoRe-OT predictions contain invalid labels: {unexpected}"
            )
        forced_accuracy = float(
            accuracy_score(
                joined.loc[represented, "true_label"], represented_labels
            )
        )
        labels = sorted(joined.loc[represented, "true_label"].astype(str).unique())
        forced_macro_f1 = float(
            f1_score(
                joined.loc[represented, "true_label"],
                represented_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        expected_accuracy = float(panel_d_row.iloc[0]["forced_accuracy"])
        expected_macro_f1 = float(panel_d_row.iloc[0]["forced_macro_f1"])
        if not np.isclose(forced_accuracy, expected_accuracy, atol=1.0e-12):
            raise SealedFigureInputError(
                f"HIHA CoRe-OT seed-1 forced accuracy differs from Panel D for "
                f"{endpoint}: {forced_accuracy} != {expected_accuracy}"
            )
        if not np.isclose(forced_macro_f1, expected_macro_f1, atol=1.0e-12):
            raise SealedFigureInputError(
                f"HIHA CoRe-OT seed-1 forced macro-F1 differs from Panel D for "
                f"{endpoint}: {forced_macro_f1} != {expected_macro_f1}"
            )
        joined["map_id"] = "coreot_full"
        joined["map_display"] = "CoRe-OT"
        joined["method"] = "coreot_full"
        joined["method_run_id"] = run_id
        joined["forced_label"] = joined["authoritative_forced_label"]
        joined["displayed_assignment"] = np.where(
            joined["is_held_out_state"],
            "Held-out state (not evaluated)",
            joined["authoritative_forced_label"],
        )
        new_frames.append(joined[panel_f.columns])
        lineage.append(
            {
                "figure": "Figure 2",
                "panel": "F",
                "family": "hiha",
                "endpoint": endpoint,
                "seed": HIHA_REPRESENTATIVE_SEED,
                "method": "coreot_full",
                "condition_id": HIHA_CONDITION_ID,
                "candidate_set": HIHA_CANDIDATE_SET,
                "evaluation_scope": HIHA_TRANSFER_SCOPE,
                "score": "u",
                "metric": "forced_label",
                "sealed_summary_row_id": "",
                "per_run_artifact": (
                    f"runs/{run_id}/scoring/{HIHA_CONDITION_ID}/"
                    f"{HIHA_CANDIDATE_SET}/cell_scores.parquet"
                ),
                "per_run_sha256": sha256_file(prediction_path),
                "aggregation_code": (
                    "src/coreot/results/main_figure_external_baselines.py::"
                    "prepare_hiha_candidate_sources"
                ),
                "figure_source": "source_data/panel_f_label_assignment_cells.csv",
            }
        )
        for method in CONTROLLED_METHODS:
            binding = FigureInputBinding(
                "hiha",
                endpoint,
                HIHA_REPRESENTATIVE_SEED,
                method,
                "incomplete_reference",
            )
            predictions = adapter.predictions(binding)
            joined = exact_cell_join(
                base,
                predictions[["cell_id", "forced_label"]].rename(
                    columns={"forced_label": "sealed_forced_label"}
                ),
            )
            represented_labels = joined.loc[
                joined["is_represented_state"], "sealed_forced_label"
            ]
            unexpected = sorted(set(represented_labels) - set(HIHA_LABEL_COLORS))
            if unexpected:
                raise SealedFigureInputError(
                    f"HIHA external predictions contain invalid labels: {unexpected}"
                )
            joined["map_id"] = method
            joined["map_display"] = CONTROLLED_DISPLAY[method]
            joined["method"] = method
            joined["forced_label"] = joined["sealed_forced_label"]
            joined["displayed_assignment"] = np.where(
                joined["is_held_out_state"],
                "Held-out state (not evaluated)",
                joined["sealed_forced_label"],
            )
            new_frames.append(joined[panel_f.columns])
            lineage.append(
                _prediction_lineage(
                    figure="Figure 2",
                    panel="F",
                    binding=binding,
                    predictions=predictions,
                    transformation="source_data/panel_f_label_assignment_cells.csv",
                )
            )
    panel_f = pd.concat([truth, retained, *new_frames], ignore_index=True)
    panel_f["map_id"] = pd.Categorical(
        panel_f["map_id"],
        ["truth", *[method for method, _, _ in HIHA_TRANSFER_METHODS]],
        ordered=True,
    )
    panel_f.sort_values(
        ["held_out_label", "map_id"], kind="mergesort"
    ).to_csv(panel_f_path, index=False)
    panel_f_summary_path = source_root / "panel_f_label_assignment_summary.csv"
    panel_f_summary = pd.read_csv(panel_f_summary_path)
    retained_summary = panel_f_summary.loc[
        panel_f_summary["method"].isin(retained_methods)
    ].copy()
    new_summary = []
    for endpoint in HIHA_ENDPOINTS:
        row = authoritative_transfer.loc[
            authoritative_transfer["held_out_label"].eq(endpoint)
            & authoritative_transfer["seed"].eq(HIHA_REPRESENTATIVE_SEED)
        ].iloc[0]
        new_summary.append(
            {
                "held_out_label": endpoint,
                "seed": HIHA_REPRESENTATIVE_SEED,
                "method": "coreot_full",
                "method_display": "CoRe-OT",
                "run_id": row["run_id"],
                "n_represented": int(row["n_shared"]),
                "forced_accuracy": float(row["forced_accuracy"]),
                "forced_macro_f1": float(row["forced_macro_f1"]),
                "panel_e_forced_accuracy": float(row["forced_accuracy"]),
                "panel_e_forced_macro_f1": float(row["forced_macro_f1"]),
            }
        )
    for endpoint in HIHA_ENDPOINTS:
        for method in CONTROLLED_METHODS:
            binding = FigureInputBinding(
                "hiha",
                endpoint,
                HIHA_REPRESENTATIVE_SEED,
                method,
                "incomplete_reference",
            )
            metrics = adapter.controlled_metrics(
                binding, evaluation_scope="represented_state"
            )
            new_summary.append(
                {
                    "held_out_label": endpoint,
                    "seed": HIHA_REPRESENTATIVE_SEED,
                    "method": method,
                    "method_display": CONTROLLED_DISPLAY[method],
                    "run_id": metrics.iloc[0]["run_id"],
                    "n_represented": int(metrics.iloc[0]["n_cells"]),
                    "forced_accuracy": _metric_value(metrics, "forced_accuracy"),
                    "forced_macro_f1": _metric_value(metrics, "forced_macro_f1"),
                    "panel_e_forced_accuracy": _metric_value(
                        metrics, "forced_accuracy"
                    ),
                    "panel_e_forced_macro_f1": _metric_value(
                        metrics, "forced_macro_f1"
                    ),
                }
            )
    pd.concat(
        [retained_summary, pd.DataFrame(new_summary)],
        ignore_index=True,
    ).to_csv(panel_f_summary_path, index=False)

    lineage_path = candidate_figure_root / "numerical_lineage.csv"
    pd.DataFrame(lineage).to_csv(lineage_path, index=False)
    manifest_path = _write_candidate_manifest(
        figure_root=candidate_figure_root,
        figure="Figure 2",
        package_root=package_root,
        canonical_source_root=canonical_source_root,
        lineage_path=lineage_path,
        preserved_panels=("A", "C"),
        additional_input_hashes=authoritative_input_hashes,
    )
    return {"source_root": source_root, "lineage": lineage_path, "manifest": manifest_path}


def prepare_mouse_candidate_sources(
    *,
    package_root: Path,
    canonical_source_root: Path,
    candidate_figure_root: Path,
) -> dict[str, Path]:
    adapter = UOTBaselineFigureInputs(package_root)
    source_root = candidate_figure_root / "source_data"
    _copy_source_tree(canonical_source_root, source_root)
    lineage: list[dict[str, object]] = []
    metadata = pd.read_csv(source_root / "query_cell_metadata.csv")
    if metadata["cell_id"].duplicated().any():
        raise SealedFigureInputError("Mouse query metadata contains duplicate cell IDs")

    score_path = source_root / "detection_scores.csv"
    scores = pd.read_csv(score_path)
    retained_scores = scores.loc[scores["method"].isin(RETAINED_METHODS)].copy()
    external_scores = []
    prediction_cache: dict[str, pd.DataFrame] = {}
    for method in MOUSE_EXTERNAL_METHODS:
        binding = FigureInputBinding(
            "mouse_spleen",
            "Proliferating",
            20260713,
            method,
            "natural_mismatch",
        )
        predictions = adapter.predictions(binding)
        prediction_cache[method] = predictions
        joined = exact_cell_join(
            metadata[["cell_id", "is_proliferating"]],
            predictions[["cell_id", "weak_support_score"]],
        )
        ranked = joined.sort_values(
            ["weak_support_score", "cell_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected_ids = set(ranked.head(62)["cell_id"])
        percentiles = joined["weak_support_score"].rank(method="average")
        percentiles = (percentiles - 1.0) / (len(joined) - 1.0)
        external_scores.append(
            pd.DataFrame(
                {
                    "cell_id": joined["cell_id"],
                    "method": method,
                    "raw_detection_score": joined["weak_support_score"],
                    "score_direction": "larger_is_weaker_reference_support",
                    "oriented_detection_score": joined["weak_support_score"],
                    "weak_support_percentile": percentiles,
                    "is_top_62_weak_support": joined["cell_id"].isin(selected_ids),
                    "is_proliferating": joined["is_proliferating"].astype(bool),
                }
            )
        )
        lineage.append(
            _prediction_lineage(
                figure="Figure 4",
                panel="C",
                binding=binding,
                predictions=predictions,
                transformation="source_data/detection_scores.csv",
            )
        )
    display_scores = pd.concat(
        [retained_scores, *external_scores], ignore_index=True
    )
    display_scores["method"] = pd.Categorical(
        display_scores["method"], MOUSE_METHODS, ordered=True
    )
    display_scores.sort_values(["method", "cell_id"]).to_csv(score_path, index=False)

    summary_path = source_root / "detection_summary.csv"
    summary = pd.read_csv(summary_path)
    retained_summary = summary.loc[summary["method"].isin(RETAINED_METHODS)].copy()
    new_summary = []
    for method in MOUSE_EXTERNAL_METHODS:
        sealed = adapter.mouse_summary(method, evaluation_scope="endpoint_detection")
        ap = sealed.loc[sealed["metric"].eq("average_precision")].iloc[0]
        auroc = sealed.loc[sealed["metric"].eq("auroc")].iloc[0]
        new_summary.append(
            {
                "method": method,
                "ap": float(ap["value"]),
                "ap_lower": float(ap["ci_low"]),
                "ap_upper": float(ap["ci_high"]),
                "auroc": float(auroc["value"]),
                "auroc_lower": float(auroc["ci_low"]),
                "auroc_upper": float(auroc["ci_high"]),
                "prevalence": 62 / 4333,
                "n_positive": 62,
                "n_negative": 4271,
                "bootstrap_replicates": int(ap["bootstrap_replicates"]),
                "bootstrap_seed": int(ap["bootstrap_seed"]),
                "sealed_ap_row_id": str(ap["row_id"]),
                "sealed_auroc_row_id": str(auroc["row_id"]),
            }
        )
        for metric_row in (ap, auroc):
            lineage.append(
                {
                    "figure": "Figure 4",
                    "panel": "A",
                    "family": "mouse_spleen",
                    "endpoint": "Proliferating",
                    "seed": 20260713,
                    "method": method,
                    "condition_id": "natural_mismatch",
                    "evaluation_scope": "endpoint_detection",
                    "metric": str(metric_row["metric"]),
                    "sealed_summary_row_id": str(metric_row["row_id"]),
                    "per_run_artifact": str(metric_row["prediction_artifact"]),
                    "per_run_sha256": str(metric_row["prediction_sha256"]),
                    "aggregation_code": (
                        "experiments/uot_baseline_pilot/portable_package.py -> "
                        "src/coreot/results/main_figure_external_baselines.py"
                    ),
                    "figure_source": "source_data/detection_summary.csv",
                }
            )
    display_summary = pd.concat(
        [retained_summary, pd.DataFrame(new_summary)], ignore_index=True
    )
    display_summary["method"] = pd.Categorical(
        display_summary["method"], MOUSE_METHODS, ordered=True
    )
    display_summary.sort_values("method").to_csv(summary_path, index=False)

    predictions_path = source_root / "forced_predictions.csv"
    forced = pd.read_csv(predictions_path)
    retained_forced = forced.loc[forced["method"].isin(RETAINED_METHODS)].copy()
    truth = metadata[
        ["cell_id", "true_label", "is_shared", "is_proliferating"]
    ].copy()
    new_forced = []
    allowed_labels = set(metadata.loc[metadata["is_shared"], "true_label"].astype(str))
    for method in MOUSE_EXTERNAL_METHODS:
        binding = FigureInputBinding(
            "mouse_spleen",
            "Proliferating",
            20260713,
            method,
            "natural_mismatch",
        )
        predictions = prediction_cache[method]
        joined = exact_cell_join(truth, predictions[["cell_id", "forced_label"]])
        unexpected = sorted(set(joined["forced_label"]) - allowed_labels)
        if unexpected:
            raise SealedFigureInputError(
                f"Mouse external predictions contain invalid labels: {unexpected}"
            )
        joined.insert(1, "method", method)
        joined = joined.rename(columns={"forced_label": "forced_predicted_label"})
        new_forced.append(joined[forced.columns])
        lineage.append(
            _prediction_lineage(
                figure="Figure 4",
                panel="D/E",
                binding=binding,
                predictions=predictions,
                transformation="source_data/forced_predictions.csv",
            )
        )
    display_forced = pd.concat([retained_forced, *new_forced], ignore_index=True)
    display_forced["method"] = pd.Categorical(
        display_forced["method"], MOUSE_METHODS, ordered=True
    )
    display_forced.sort_values(["method", "cell_id"]).to_csv(
        predictions_path, index=False
    )

    transfer_path = source_root / "shared_label_transfer_summary.csv"
    transfer = pd.read_csv(transfer_path)
    retained_transfer = transfer.loc[transfer["method"].isin(RETAINED_METHODS)].copy()
    new_transfer = []
    for method in MOUSE_EXTERNAL_METHODS:
        sealed = adapter.mouse_summary(method, evaluation_scope="represented_state")
        accuracy = sealed.loc[sealed["metric"].eq("forced_accuracy")].iloc[0]
        macro_f1 = sealed.loc[sealed["metric"].eq("forced_macro_f1")].iloc[0]
        new_transfer.append(
            {
                "method": method,
                "forced_accuracy": float(accuracy["value"]),
                "forced_accuracy_lower": float(accuracy["ci_low"]),
                "forced_accuracy_upper": float(accuracy["ci_high"]),
                "forced_macro_f1": float(macro_f1["value"]),
                "forced_macro_f1_lower": float(macro_f1["ci_low"]),
                "forced_macro_f1_upper": float(macro_f1["ci_high"]),
                "n_shared": 4271,
                "bootstrap_replicates": int(accuracy["bootstrap_replicates"]),
                "bootstrap_seed": int(accuracy["bootstrap_seed"]),
                "bootstrap_strata": "true_shared_state_label",
                "sealed_accuracy_row_id": str(accuracy["row_id"]),
                "sealed_macro_f1_row_id": str(macro_f1["row_id"]),
            }
        )
        for metric_row in (accuracy, macro_f1):
            lineage.append(
                {
                    "figure": "Figure 4",
                    "panel": "B",
                    "family": "mouse_spleen",
                    "endpoint": "Proliferating",
                    "seed": 20260713,
                    "method": method,
                    "condition_id": "natural_mismatch",
                    "evaluation_scope": "represented_state",
                    "metric": str(metric_row["metric"]),
                    "sealed_summary_row_id": str(metric_row["row_id"]),
                    "per_run_artifact": str(metric_row["prediction_artifact"]),
                    "per_run_sha256": str(metric_row["prediction_sha256"]),
                    "aggregation_code": (
                        "experiments/uot_baseline_pilot/portable_package.py -> "
                        "src/coreot/results/main_figure_external_baselines.py"
                    ),
                    "figure_source": "source_data/shared_label_transfer_summary.csv",
                }
            )
    display_transfer = pd.concat(
        [retained_transfer, pd.DataFrame(new_transfer)], ignore_index=True
    )
    display_transfer["method"] = pd.Categorical(
        display_transfer["method"], MOUSE_METHODS, ordered=True
    )
    display_transfer.sort_values("method").to_csv(transfer_path, index=False)

    lineage_path = candidate_figure_root / "numerical_lineage.csv"
    pd.DataFrame(lineage).to_csv(lineage_path, index=False)
    manifest_path = _write_candidate_manifest(
        figure_root=candidate_figure_root,
        figure="Figure 4",
        package_root=package_root,
        canonical_source_root=canonical_source_root,
        lineage_path=lineage_path,
        preserved_panels=(),
    )
    return {"source_root": source_root, "lineage": lineage_path, "manifest": manifest_path}


def _pbmc_summary(by_seed: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for endpoint in PBMC_ENDPOINTS:
        for method in methods:
            selected = by_seed.loc[
                by_seed["endpoint"].eq(endpoint) & by_seed["method"].eq(method)
            ]
            for metric in ("auprc", "auroc"):
                values = selected[metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "endpoint": endpoint,
                        "method": method,
                        "score": selected["score"].iloc[0],
                        "display_name": selected["display_name"].iloc[0],
                        "metric": metric,
                        "mean": float(values.mean()),
                        "sample_sd": float(values.std(ddof=1)),
                        "n_seeds": len(values),
                    }
                )
    return pd.DataFrame(rows)


def prepare_pbmc_candidate_sources(
    *,
    package_root: Path,
    canonical_source_root: Path,
    candidate_figure_root: Path,
) -> dict[str, Path]:
    adapter = UOTBaselineFigureInputs(package_root)
    source_root = candidate_figure_root / "figure_3_pbmc_source_data"
    _copy_source_tree(canonical_source_root, source_root)
    lineage: list[dict[str, object]] = []

    panel_b_path = source_root / "figure_3_within_celltype_detection_by_seed.csv"
    panel_b = pd.read_csv(panel_b_path)
    cohort = panel_b.groupby(["endpoint", "seed"])[
        ["n_positive", "n_negative", "prevalence"]
    ].first()
    retained = panel_b.loc[panel_b["method"].isin(RETAINED_METHODS)].copy()
    new_rows = []
    for endpoint in PBMC_ENDPOINTS:
        for seed in PBMC_SEEDS:
            for method in CONTROLLED_METHODS:
                binding = FigureInputBinding(
                    "pbmc", endpoint, seed, method, "incomplete_reference"
                )
                metrics = adapter.controlled_metrics(
                    binding, evaluation_scope="within_cell_type"
                )
                counts = cohort.loc[(endpoint, seed)]
                new_rows.append(
                    {
                        "endpoint": endpoint,
                        "seed": seed,
                        "run_id": metrics.iloc[0]["run_id"],
                        "method": method,
                        "score": "z_absent_score",
                        "display_name": CONTROLLED_DISPLAY[method],
                        "score_orientation": "larger_is_weaker_reference_support",
                        "n_positive": int(counts["n_positive"]),
                        "n_negative": int(counts["n_negative"]),
                        "prevalence": float(counts["prevalence"]),
                        "auprc": _metric_value(metrics, "average_precision"),
                        "auroc": _metric_value(metrics, "auroc"),
                        "sealed_ap_row_id": _metric_row_id(
                            metrics, "average_precision"
                        ),
                        "sealed_auroc_row_id": _metric_row_id(metrics, "auroc"),
                    }
                )
                for metric in ("average_precision", "auroc"):
                    lineage.append(
                        _controlled_metric_lineage(
                            figure="Figure 3",
                            panel="B",
                            binding=binding,
                            metrics=metrics,
                            metric=metric,
                            source_name=(
                                "figure_3_pbmc_source_data/"
                                "figure_3_within_celltype_detection_by_seed.csv"
                            ),
                        )
                    )
    panel_b = pd.concat([retained, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    methods = [method for method, _, _ in PBMC_DETECTION_METHODS]
    panel_b["method"] = pd.Categorical(panel_b["method"], methods, ordered=True)
    panel_b = panel_b.sort_values(["endpoint", "method", "seed"]).reset_index(drop=True)
    panel_b.to_csv(panel_b_path, index=False)
    _pbmc_summary(panel_b, methods).to_csv(
        source_root / "figure_3_within_celltype_detection_summary.csv", index=False
    )

    panel_d_path = source_root / "figure_3_represented_celltype_transfer_by_seed.csv"
    panel_d = pd.read_csv(panel_d_path)
    retained = panel_d.loc[panel_d["method"].isin(RETAINED_METHODS)].copy()
    new_rows = []
    for endpoint in PBMC_ENDPOINTS:
        for seed in PBMC_SEEDS:
            for method in CONTROLLED_METHODS:
                binding = FigureInputBinding(
                    "pbmc", endpoint, seed, method, "incomplete_reference"
                )
                metrics = adapter.controlled_metrics(
                    binding, evaluation_scope="represented_state"
                )
                new_rows.append(
                    {
                        "endpoint": endpoint,
                        "seed": seed,
                        "run_id": metrics.iloc[0]["run_id"],
                        "method": method,
                        "score": "z_absent_score",
                        "display_name": CONTROLLED_DISPLAY[method],
                        "forced_accuracy": _metric_value(metrics, "forced_accuracy"),
                        "forced_macro_f1": _metric_value(metrics, "forced_macro_f1"),
                        "coverage": 1.0,
                        "post_abstention_macro_f1": np.nan,
                        "sealed_accuracy_row_id": _metric_row_id(
                            metrics, "forced_accuracy"
                        ),
                        "sealed_macro_f1_row_id": _metric_row_id(
                            metrics, "forced_macro_f1"
                        ),
                    }
                )
                for metric in ("forced_accuracy", "forced_macro_f1"):
                    lineage.append(
                        _controlled_metric_lineage(
                            figure="Figure 3",
                            panel="D",
                            binding=binding,
                            metrics=metrics,
                            metric=metric,
                            source_name=(
                                "figure_3_pbmc_source_data/"
                                "figure_3_represented_celltype_transfer_by_seed.csv"
                            ),
                        )
                    )
    panel_d = pd.concat([retained, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    transfer_methods = [method for method, _, _ in PBMC_TRANSFER_METHODS]
    panel_d["method"] = pd.Categorical(panel_d["method"], transfer_methods, ordered=True)
    panel_d = panel_d.sort_values(["endpoint", "method", "seed"]).reset_index(drop=True)
    panel_d.to_csv(panel_d_path, index=False)
    panel_d.groupby(["endpoint", "method", "score", "display_name"], observed=True)[
        ["forced_accuracy", "forced_macro_f1"]
    ].agg(["mean", "std"]).reset_index().to_csv(
        source_root / "figure_3_represented_celltype_transfer_summary.csv",
        index=False,
    )

    panel_e_path = source_root / "figure_3_weak_support_umap_cells.parquet"
    panel_e = pd.read_parquet(panel_e_path)
    truth = panel_e.loc[panel_e["map_id"].eq("truth")].copy()
    retained = panel_e.loc[panel_e["map_id"].isin(RETAINED_METHODS)].copy()
    new_frames = []
    for endpoint in PBMC_ENDPOINTS:
        base = truth.loc[
            truth["endpoint"].eq(endpoint)
            & truth["seed"].eq(PBMC_REPRESENTATIVE_SEED)
        ].copy()
        for method in CONTROLLED_METHODS:
            binding = FigureInputBinding(
                "pbmc",
                endpoint,
                PBMC_REPRESENTATIVE_SEED,
                method,
                "incomplete_reference",
            )
            predictions = adapter.predictions(binding)
            joined = exact_cell_join(
                base, predictions[["cell_id", "weak_support_score"]]
            )
            eligible = joined["is_evaluation_cohort"].astype(bool)
            n_select = int(joined["is_held_out"].sum())
            selected_ids = (
                joined.loc[eligible, ["cell_id", "weak_support_score"]]
                .sort_values(
                    ["weak_support_score", "cell_id"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                .head(n_select)["cell_id"]
            )
            joined["map_id"] = method
            joined["map_display"] = CONTROLLED_DISPLAY[method]
            joined["method"] = method
            joined["score"] = joined["weak_support_score"]
            joined["is_selected"] = joined["cell_id"].isin(selected_ids)
            new_frames.append(joined[panel_e.columns])
            lineage.append(
                _prediction_lineage(
                    figure="Figure 3",
                    panel="E",
                    binding=binding,
                    predictions=predictions,
                    transformation=(
                        "figure_3_pbmc_source_data/"
                        "figure_3_weak_support_umap_cells.parquet"
                    ),
                )
            )
    pd.concat([truth, retained, *new_frames], ignore_index=True).to_parquet(
        panel_e_path, index=False
    )

    panel_f_path = source_root / "figure_3_label_assignment_umap_cells.parquet"
    panel_f = pd.read_parquet(panel_f_path)
    truth = panel_f.loc[panel_f["map_id"].eq("truth")].copy()
    retained = panel_f.loc[panel_f["map_id"].isin(RETAINED_METHODS)].copy()
    new_frames = []
    valid_labels = set(PBMC_LABEL_COLORS) - {"Held-out state (not evaluated)"}
    for endpoint in PBMC_ENDPOINTS:
        base = truth.loc[
            truth["endpoint"].eq(endpoint)
            & truth["seed"].eq(PBMC_REPRESENTATIVE_SEED)
        ].copy()
        for method in CONTROLLED_METHODS:
            binding = FigureInputBinding(
                "pbmc",
                endpoint,
                PBMC_REPRESENTATIVE_SEED,
                method,
                "incomplete_reference",
            )
            predictions = adapter.predictions(binding)
            joined = exact_cell_join(
                base,
                predictions[["cell_id", "forced_label"]].rename(
                    columns={"forced_label": "sealed_forced_label"}
                ),
            )
            represented = joined.loc[
                joined["is_represented"], "sealed_forced_label"
            ]
            unexpected = sorted(set(represented) - valid_labels)
            if unexpected:
                raise SealedFigureInputError(
                    f"PBMC external predictions contain invalid labels: {unexpected}"
                )
            joined["map_id"] = method
            joined["map_display"] = CONTROLLED_DISPLAY[method]
            joined["method"] = method
            joined["forced_label"] = joined["sealed_forced_label"]
            joined["displayed_assignment"] = np.where(
                joined["is_held_out"],
                "Held-out state (not evaluated)",
                joined["sealed_forced_label"],
            )
            new_frames.append(joined[panel_f.columns])
            lineage.append(
                _prediction_lineage(
                    figure="Figure 3",
                    panel="F",
                    binding=binding,
                    predictions=predictions,
                    transformation=(
                        "figure_3_pbmc_source_data/"
                        "figure_3_label_assignment_umap_cells.parquet"
                    ),
                )
            )
    pd.concat([truth, retained, *new_frames], ignore_index=True).to_parquet(
        panel_f_path, index=False
    )
    panel_f_summary_path = (
        source_root / "figure_3_label_assignment_umap_summary.csv"
    )
    panel_f_summary = pd.read_csv(panel_f_summary_path)
    retained_summary = panel_f_summary.loc[
        panel_f_summary["method"].isin(RETAINED_METHODS)
    ].copy()
    new_summary = []
    for endpoint in PBMC_ENDPOINTS:
        for method in CONTROLLED_METHODS:
            binding = FigureInputBinding(
                "pbmc",
                endpoint,
                PBMC_REPRESENTATIVE_SEED,
                method,
                "incomplete_reference",
            )
            metrics = adapter.controlled_metrics(
                binding, evaluation_scope="represented_state"
            )
            new_summary.append(
                {
                    "endpoint": endpoint,
                    "seed": PBMC_REPRESENTATIVE_SEED,
                    "method": method,
                    "display_name": CONTROLLED_DISPLAY[method],
                    "run_id": metrics.iloc[0]["run_id"],
                    "n_represented": int(metrics.iloc[0]["n_cells"]),
                    "forced_accuracy": _metric_value(metrics, "forced_accuracy"),
                    "forced_macro_f1": _metric_value(metrics, "forced_macro_f1"),
                    "panel_d_forced_accuracy": _metric_value(
                        metrics, "forced_accuracy"
                    ),
                    "panel_d_forced_macro_f1": _metric_value(
                        metrics, "forced_macro_f1"
                    ),
                }
            )
    pd.concat(
        [retained_summary, pd.DataFrame(new_summary)],
        ignore_index=True,
    ).to_csv(panel_f_summary_path, index=False)

    lineage_path = candidate_figure_root / "numerical_lineage.csv"
    pd.DataFrame(lineage).to_csv(lineage_path, index=False)
    manifest_path = _write_candidate_manifest(
        figure_root=candidate_figure_root,
        figure="Figure 3",
        package_root=package_root,
        canonical_source_root=canonical_source_root,
        lineage_path=lineage_path,
        preserved_panels=("A", "C"),
    )
    return {"source_root": source_root, "lineage": lineage_path, "manifest": manifest_path}


def write_review_index(
    *,
    review_root: Path,
    figure_manifests: dict[str, Path],
    review_version: str,
) -> Path:
    path = review_root / "review.md"
    lines = [
        f"# Figures 2--4 external-baseline review candidate: {review_version}",
        "",
        "Review each composite independently for scientific quantity, cohort, method order,",
        "intervals, score orientation, spatial identity, endpoint exclusions, clipping,",
        "readability, and standalone/composite agreement.",
        "",
    ]
    for figure, manifest in figure_manifests.items():
        lines.extend(
            [
                f"## {figure}",
                "",
                f"- Candidate manifest: `{manifest.relative_to(review_root)}`",
                "- Decision: pending explicit author review",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
