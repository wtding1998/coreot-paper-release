from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import platform
import shlex
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coreot.external_baselines.runner import (  # noqa: E402
    _celltypist_config,
    _write_predictions,
    run_celltypist_l3,
    write_external_baseline_cell_scores,
)
from coreot.results.external_baselines import write_external_baseline_results  # noqa: E402
from coreot.results.pbmc_figure3 import compute_within_celltype_metrics  # noqa: E402
from coreot.results.pbmc_supplement import (  # noqa: E402
    _read_method_scores,
    _read_raw_metadata,
    _read_truth,
)
from coreot.submission.celltypist_promotion import (  # noqa: E402
    CellTypistCandidateRootError,
)


ENDPOINTS = ("B cells", "NK cells", "Dendritic cells")
SEEDS = (1, 2, 3, 4, 5)
CONDITIONS = ("incomplete_reference", "full_reference_control")
METHOD = "celltypist_l3"
CANDIDATE_SET = "external_reference_mapping"
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "results/phase1_evidence_freeze/retained_job_inventory/2026-08-24-v3/"
    "execution_roots/PBMC"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results/phase1_evidence_freeze/pbmc_celltypist_centered_full_sgd/"
    "2026-08-25-v1"
)
RETAINED_ROOT = PROJECT_ROOT / "results/PBMC/manuscript/supplement"
DEFAULT_INPUT_REPAIR_MANIFEST = (
    PROJECT_ROOT
    / "results/submission_verification/PBMC/"
    "2026-08-28-celltypist-input-repair-r1/derivation.yaml"
)
DEFAULT_PROMOTION_ROOT = (
    PROJECT_ROOT
    / "results/phase1_evidence_freeze/celltypist_supplement_promotion/"
    "2026-08-25-v2/evidence/PBMC"
)
DEFAULT_CURRENT_DATA3_ROOT = (
    PROJECT_ROOT
    / "results/submission_verification/manuscript_reproducibility/"
    "2026-08-28-current-closure-r5/provisional_data/Supplementary_Data_3/"
    "units/pbmc_primary/supplement"
)
CURRENT_AUTHORITY_ATOL = 1e-15


@dataclass(frozen=True)
class InputRepair:
    manifest_path: Path
    run_id: str
    condition: str
    counts_path: Path
    sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return self.run_id, self.condition


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _ensure_fresh_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise CellTypistCandidateRootError(f"Candidate root already exists: {output_root}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_input_repair(manifest_path: Path) -> InputRepair:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Input-repair manifest does not exist: {manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input-repair manifest must be a mapping")
    if payload.get("stage") != "pbmc-celltypist-model-visible-input-repair":
        raise ValueError("Input-repair manifest has an unsupported stage")
    if payload.get("status") != "verified":
        raise ValueError("Input-repair manifest must have status=verified")
    run_id = str(payload.get("run_id", "")).strip()
    condition = str(payload.get("condition", "")).strip()
    if not run_id:
        raise ValueError("Input-repair manifest requires a nonempty run_id")
    if condition not in CONDITIONS:
        raise ValueError(
            f"Input-repair manifest condition must be one of {list(CONDITIONS)}"
        )
    derivation = payload.get("derivation")
    hashes = payload.get("sha256")
    if not isinstance(derivation, dict) or not isinstance(hashes, dict):
        raise ValueError("Input-repair manifest requires derivation and sha256 mappings")
    output = str(derivation.get("output", "")).strip()
    expected_sha256 = str(hashes.get("repaired_counts", "")).strip().lower()
    if not output or len(expected_sha256) != 64:
        raise ValueError(
            "Input-repair manifest requires derivation.output and repaired_counts SHA-256"
        )
    counts_path = Path(output)
    if not counts_path.is_absolute():
        counts_path = PROJECT_ROOT / counts_path
    counts_path = counts_path.resolve()
    if not counts_path.is_file():
        raise FileNotFoundError(f"Repaired counts do not exist: {counts_path}")
    observed_sha256 = _sha256(counts_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Input-repair counts checksum does not match the verified manifest: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    return InputRepair(
        manifest_path=manifest_path,
        run_id=run_id,
        condition=condition,
        counts_path=counts_path,
        sha256=observed_sha256,
    )


def _load_input_repairs(paths: tuple[Path, ...]) -> dict[tuple[str, str], InputRepair]:
    repairs: dict[tuple[str, str], InputRepair] = {}
    for path in paths:
        repair = _load_input_repair(path)
        if repair.key in repairs:
            raise ValueError(f"Duplicate input repair for {repair.key}")
        repairs[repair.key] = repair
    return repairs


def _source_run(config: dict[str, object]) -> Path:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict) or not str(outputs.get("root", "")).strip():
        raise ValueError("External-baseline config requires outputs.root")
    run_id = str(config.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("External-baseline config requires run_id")
    return Path(str(outputs["root"])) / run_id


def _validate_count_input_coverage(
    configs: list[tuple[Path, dict[str, object]]],
    repairs: dict[tuple[str, str], InputRepair],
) -> None:
    expected_keys: set[tuple[str, str]] = set()
    missing_keys: set[tuple[str, str]] = set()
    for _, config in configs:
        run_id = str(config["run_id"])
        run_root = _source_run(config)
        for condition in CONDITIONS:
            key = (run_id, condition)
            expected_keys.add(key)
            counts = run_root / "benchmark" / condition / "model_visible/counts.h5ad"
            if not counts.is_file():
                missing_keys.add(key)

    extra_repairs = set(repairs) - expected_keys
    if extra_repairs:
        raise ValueError(f"Input repairs do not match selected runs: {sorted(extra_repairs)}")
    redundant_repairs = set(repairs) - missing_keys
    if redundant_repairs:
        raise ValueError(
            "Input repair targets a canonical matrix that already exists: "
            f"{sorted(redundant_repairs)}"
        )
    uncovered = missing_keys - set(repairs)
    if uncovered:
        raise FileNotFoundError(
            "Model-visible counts are missing with no declared repair: "
            f"{sorted(uncovered)}"
        )


@contextmanager
def _repaired_run_view(
    *,
    source_run: Path,
    condition: str,
    repair: InputRepair | None,
) -> Iterator[Path]:
    if repair is None:
        yield source_run
        return
    canonical = source_run / "benchmark" / condition / "model_visible"
    if (canonical / "counts.h5ad").is_file():
        raise ValueError(
            f"Refusing to repair an existing canonical counts matrix: {canonical}"
        )
    with TemporaryDirectory(prefix="coreot_pbmc_celltypist_repair_") as temporary:
        staged_run = Path(temporary) / source_run.name
        staged_visible = staged_run / "benchmark" / condition / "model_visible"
        staged_visible.mkdir(parents=True)
        for source in sorted(canonical.iterdir()):
            if source.name == "counts.h5ad":
                continue
            (staged_visible / source.name).symlink_to(
                source.resolve(),
                target_is_directory=source.is_dir(),
            )
        (staged_visible / "counts.h5ad").symlink_to(repair.counts_path)
        yield staged_run


def _source_configs(source_root: Path) -> list[tuple[Path, dict[str, object]]]:
    config_root = source_root.parent.parent / "resolved_configs"
    selected: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(config_root.glob("pbmc_*_external/external_baselines.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and str(payload.get("heldout_label")) in ENDPOINTS
            and int(payload.get("repeat", -1)) in SEEDS
        ):
            selected.append((path, payload))
    observed = {
        (str(payload["heldout_label"]), int(payload["repeat"]))
        for _, payload in selected
    }
    expected = {(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS}
    if observed != expected or len(selected) != len(expected):
        raise RuntimeError(
            "PBMC CellTypist rerun requires exactly the 15 endpoint-seed configs; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return selected


def _prepare_output_run(
    *,
    source_run: Path,
    output_run: Path,
    endpoint: str,
    seed: int,
    config_path: Path,
    parameters: dict[str, object],
    repairs: dict[tuple[str, str], InputRepair],
) -> None:
    for condition in CONDITIONS:
        source_truth = (
            source_run
            / "benchmark"
            / condition
            / "evaluation_truth/query_truth.csv"
        )
        target_truth = (
            output_run
            / "benchmark"
            / condition
            / "evaluation_truth/query_truth.csv"
        )
        target_truth.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_truth, target_truth)

    _write_yaml(
        output_run.parent.parent / "grid" / output_run.name / "benchmark.yaml",
        {
            "run_id": output_run.name,
            "removed_state": endpoint,
            "repeat": seed,
            "split": {"seed": seed},
        },
    )
    output_config: dict[str, object] = {
        "run_id": output_run.name,
        "heldout_label": endpoint,
        "repeat": seed,
        "conditions": list(CONDITIONS),
        "candidate_set": CANDIDATE_SET,
        "methods": [METHOD],
        "outputs": {"root": str(output_run.parent)},
        "threshold_quantile": 0.95,
        "celltypist": {
            "profile": parameters["profile"],
            "random_state": parameters["random_state"],
        },
        "source_v3_config": _display_path(config_path),
        "source_v3_run": _display_path(source_run),
    }
    applied_repairs = [
        {
            "condition": condition,
            "manifest": _display_path(
                repairs[(output_run.name, condition)].manifest_path
            ),
            "counts": _display_path(repairs[(output_run.name, condition)].counts_path),
            "sha256": repairs[(output_run.name, condition)].sha256,
        }
        for condition in CONDITIONS
        if (output_run.name, condition) in repairs
    ]
    if applied_repairs:
        output_config["input_repairs"] = applied_repairs
    _write_yaml(
        output_run.parent.parent
        / "resolved_configs"
        / output_run.name
        / "external_baselines.yaml",
        output_config,
    )


def _fit_one_run(
    *,
    source_run: Path,
    output_run: Path,
    endpoint: str,
    seed: int,
    parameters: dict[str, object],
    repairs: dict[tuple[str, str], InputRepair],
) -> None:
    for condition in CONDITIONS:
        repair = repairs.get((source_run.name, condition))
        with _repaired_run_view(
            source_run=source_run,
            condition=condition,
            repair=repair,
        ) as fit_run:
            frame = run_celltypist_l3(
                run_root=fit_run,
                condition=condition,
                heldout_label=endpoint,
                repeat=seed,
                is_full_reference_control=condition == "full_reference_control",
                training_parameters=parameters,
            )
        _write_predictions(
            output_run,
            condition,
            METHOD,
            frame,
            parameters=parameters,
        )
    write_external_baseline_cell_scores(
        run_root=output_run,
        conditions=CONDITIONS,
        candidate_set=CANDIDATE_SET,
        methods=(METHOD,),
        threshold_quantile=0.95,
        method_parameters={METHOD: parameters},
    )


def _within_celltype_tables(
    *,
    detection_path: Path,
    runs_root: Path,
    raw_data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detection = pd.read_csv(detection_path)
    metadata = _read_raw_metadata(raw_data_path)
    rows: list[dict[str, object]] = []
    for record in detection.itertuples(index=False):
        run_root = runs_root / str(record.run_id)
        truth = _read_truth(run_root)
        scores = _read_method_scores(
            run_root,
            candidate_set=CANDIDATE_SET,
            method=METHOD,
            score="u",
            expected_ids=set(truth["cell_id"]),
        )
        local = compute_within_celltype_metrics(
            truth,
            scores,
            metadata,
            endpoint=str(record.held_out_label),
            score_column="u",
        )
        rows.append(
            {
                "held_out_label": str(record.held_out_label),
                "seed": int(record.seed),
                "run_id": str(record.run_id),
                "candidate_set": CANDIDATE_SET,
                "method": METHOD,
                "score": "u",
                "display_name": "CellTypist",
                "n_positive": int(local["n_positive"]),
                "n_negative": int(local["n_negative"]),
                "prevalence": float(local["prevalence"]),
                "auprc": float(local["auprc"]),
                "auroc": float(local["auroc"]),
            }
        )
    by_seed = pd.DataFrame(rows).sort_values(["held_out_label", "seed"])
    summary = (
        by_seed.groupby(
            ["held_out_label", "method", "score", "display_name"],
            sort=False,
        )
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", lambda values: values.std(ddof=1)),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", lambda values: values.std(ddof=1)),
            prevalence_mean=("prevalence", "mean"),
            prevalence_std=("prevalence", lambda values: values.std(ddof=1)),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )
    return by_seed, summary


def _summarize_global(detection: pd.DataFrame) -> pd.DataFrame:
    return (
        detection.groupby(["held_out_label", "method"], sort=False)
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", lambda values: values.std(ddof=1)),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", lambda values: values.std(ddof=1)),
            n_splits=("seed", "size"),
        )
        .reset_index()
        .assign(score="u", display_name="CellTypist")
    )


def _summarize_transfer(transfer: pd.DataFrame, detection: pd.DataFrame) -> pd.DataFrame:
    joined = transfer.merge(
        detection.loc[
            : ,
            ["held_out_label", "seed", "method", "absent_abstention_rate"],
        ],
        on=["held_out_label", "seed", "method"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("PBMC transfer and detection endpoint-seed keys do not match")
    joined = joined.drop(columns="_merge")
    return (
        joined.groupby(["held_out_label", "method"], sort=False)
        .agg(
            forced_accuracy_mean=("forced_accuracy", "mean"),
            forced_accuracy_std=("forced_accuracy", lambda values: values.std(ddof=1)),
            forced_macro_f1_mean=("forced_macro_f1", "mean"),
            forced_macro_f1_std=("forced_macro_f1", lambda values: values.std(ddof=1)),
            post_abstention_macro_f1_mean=("post_abstention_macro_f1", "mean"),
            post_abstention_macro_f1_std=("post_abstention_macro_f1", lambda values: values.std(ddof=1)),
            held_out_abstention_mean=("absent_abstention_rate", "mean"),
            held_out_abstention_std=("absent_abstention_rate", lambda values: values.std(ddof=1)),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", lambda values: values.std(ddof=1)),
            shared_false_abstention_mean=("shared_false_abstention_rate", "mean"),
            shared_false_abstention_std=("shared_false_abstention_rate", lambda values: values.std(ddof=1)),
            n_splits=("seed", "size"),
        )
        .reset_index()
        .assign(
            score="u",
            display_name="CellTypist",
            selective_metrics_applicable=True,
        )
    )


def _operational_abstention_by_seed(
    transfer: pd.DataFrame,
    detection: pd.DataFrame,
) -> pd.DataFrame:
    """Return the Table S14 quantities under the primary abstention rule."""

    joined = transfer.merge(
        detection.loc[
            :, ["held_out_label", "seed", "method", "absent_abstention_rate"]
        ],
        on=["held_out_label", "seed", "method"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("PBMC transfer and detection endpoint-seed keys do not match")
    joined = joined.drop(columns="_merge").assign(
        score="u",
        display_name="CellTypist",
    )
    return joined.loc[
        :,
        [
            "held_out_label",
            "seed",
            "run_id",
            "candidate_set",
            "method",
            "score",
            "display_name",
            "absent_abstention_rate",
            "coverage",
            "post_abstention_macro_f1",
        ],
    ].rename(
        columns={
            "absent_abstention_rate": "held_out_abstention",
            "coverage": "represented_state_coverage",
        }
    )


def _summarize_operational_abstention(by_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        by_seed.groupby(
            ["held_out_label", "method", "score", "display_name"],
            sort=False,
        )
        .agg(
            held_out_abstention_mean=("held_out_abstention", "mean"),
            held_out_abstention_std=(
                "held_out_abstention",
                lambda values: values.std(ddof=1),
            ),
            coverage_mean=("represented_state_coverage", "mean"),
            coverage_std=(
                "represented_state_coverage",
                lambda values: values.std(ddof=1),
            ),
            post_abstention_macro_f1_mean=("post_abstention_macro_f1", "mean"),
            post_abstention_macro_f1_std=(
                "post_abstention_macro_f1",
                lambda values: values.std(ddof=1),
            ),
            n_splits=("seed", "size"),
        )
        .reset_index()
    )


def _summarize_calibration(full_reference: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": [METHOD],
            "score": ["u"],
            "display_name": ["CellTypist"],
            "false_abstention_mean": [
                full_reference["full_reference_false_abstention_rate"].mean()
            ],
            "false_abstention_std": [
                full_reference["full_reference_false_abstention_rate"].std(ddof=1)
            ],
            "coverage_mean": [full_reference["coverage"].mean()],
            "coverage_std": [full_reference["coverage"].std(ddof=1)],
            "post_abstention_macro_f1_mean": [
                full_reference["post_abstention_macro_f1"].mean()
            ],
            "post_abstention_macro_f1_std": [
                full_reference["post_abstention_macro_f1"].std(ddof=1)
            ],
            "n_endpoint_splits": [len(full_reference)],
        }
    )


def _comparison_rows(
    *,
    table: str,
    rerun: pd.DataFrame,
    retained_path: Path,
    keys: list[str],
) -> list[dict[str, object]]:
    retained = pd.read_csv(retained_path)
    retained = retained.loc[retained["method"].astype(str).eq(METHOD)].copy()
    if retained.duplicated(keys).any() or rerun.duplicated(keys).any():
        raise ValueError(f"{table} comparison keys must be unique: {keys}")
    retained_keys = set(retained[keys].itertuples(index=False, name=None))
    rerun_keys = set(rerun[keys].itertuples(index=False, name=None))
    if retained_keys != rerun_keys:
        raise ValueError(
            f"{table} comparison keys differ: "
            f"missing={sorted(retained_keys - rerun_keys)}, "
            f"extra={sorted(rerun_keys - retained_keys)}"
        )
    merged = retained.merge(
        rerun,
        on=keys,
        suffixes=("_retained", "_rerun"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for column in retained.select_dtypes(include="number").columns:
        retained_column = f"{column}_retained" if column in rerun.columns else column
        rerun_column = f"{column}_rerun" if column in rerun.columns else None
        if rerun_column is None or retained_column not in merged or rerun_column not in merged:
            continue
        for record in merged.itertuples(index=False):
            retained_value = float(getattr(record, retained_column))
            rerun_value = float(getattr(record, rerun_column))
            rows.append(
                {
                    "table": table,
                    **{key: getattr(record, key) for key in keys},
                    "metric": column,
                    "retained": retained_value,
                    "rerun": rerun_value,
                    "difference": rerun_value - retained_value,
                }
            )
    return rows


def _compare_authority_frame(
    *,
    table: str,
    authority_name: str,
    rerun: pd.DataFrame,
    authority: pd.DataFrame,
    keys: list[str],
    columns: dict[str, str],
) -> pd.DataFrame:
    rerun_required = [*keys, *columns]
    authority_required = [*keys, *columns.values()]
    missing_rerun = sorted(set(rerun_required) - set(rerun.columns))
    missing_authority = sorted(set(authority_required) - set(authority.columns))
    if missing_rerun or missing_authority:
        raise ValueError(
            f"{table} current authority schema mismatch: "
            f"rerun_missing={missing_rerun}, authority_missing={missing_authority}"
        )
    if rerun.duplicated(keys).any() or authority.duplicated(keys).any():
        raise ValueError(f"{table} current authority keys must be unique: {keys}")
    rerun_keys = set(rerun[keys].itertuples(index=False, name=None))
    authority_keys = set(authority[keys].itertuples(index=False, name=None))
    if rerun_keys != authority_keys:
        raise ValueError(
            f"{table} current authority keys differ: "
            f"missing={sorted(rerun_keys - authority_keys)}, "
            f"extra={sorted(authority_keys - rerun_keys)}"
        )

    rerun_sorted = rerun.sort_values(keys).reset_index(drop=True)
    authority_sorted = authority.sort_values(keys).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for rerun_column, authority_column in columns.items():
        left = rerun_sorted[rerun_column]
        right = authority_sorted[authority_column]
        numeric = (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
            and not pd.api.types.is_bool_dtype(left)
            and not pd.api.types.is_bool_dtype(right)
        )
        if numeric:
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            nan_equal = np.array_equal(np.isnan(left_values), np.isnan(right_values))
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            max_difference = (
                float(np.max(np.abs(left_values[finite] - right_values[finite])))
                if finite.any()
                else 0.0
            )
            passed = nan_equal and max_difference <= CURRENT_AUTHORITY_ATOL
        else:
            max_difference = None
            passed = left.astype("string").equals(right.astype("string"))
        status = "pass" if passed else "fail"
        rows.append(
            {
                "table": table,
                "authority": authority_name,
                "metric": rerun_column,
                "authority_metric": authority_column,
                "n_rows": len(rerun_sorted),
                "max_abs_difference": max_difference,
                "tolerance": CURRENT_AUTHORITY_ATOL if numeric else 0.0,
                "status": status,
            }
        )
        if not passed:
            failures.append(
                f"{rerun_column}->{authority_column} "
                f"(max_abs_difference={max_difference})"
            )
    report = pd.DataFrame(rows)
    if failures:
        raise ValueError(
            f"{table} current authority mismatch against {authority_name}: "
            + "; ".join(failures)
        )
    return report


def _write_current_authority_comparison(
    *,
    outputs: dict[str, pd.DataFrame],
    promotion_root: Path,
    current_data3_root: Path,
    output_path: Path,
) -> Path:
    specs = (
        (
            "S11-by-seed",
            "s11_within_celltype_by_seed.csv",
            "s11_by_seed.csv",
            "pbmc_within_celltype_detection_by_seed.csv",
            ["held_out_label", "seed", "method"],
            {
                "run_id": "run_id",
                "candidate_set": "candidate_set",
                "score": "score",
                "display_name": "display_name",
                "n_positive": "n_positive",
                "n_negative": "n_negative",
                "prevalence": "prevalence",
                "auprc": "auprc",
                "auroc": "auroc",
            },
        ),
        (
            "S11-summary",
            "s11_within_celltype_summary.csv",
            "s11_summary.csv",
            "pbmc_within_celltype_detection_summary.csv",
            ["held_out_label", "method"],
            {
                "score": "score",
                "display_name": "display_name",
                "auprc_mean": "auprc_mean",
                "auprc_std": "auprc_std",
                "auroc_mean": "auroc_mean",
                "auroc_std": "auroc_std",
                "prevalence_mean": "prevalence_mean",
                "prevalence_std": "prevalence_std",
                "n_splits": "n_splits",
            },
        ),
        (
            "S12-by-seed",
            "s12_global_detection_by_seed.csv",
            "s12_by_seed.csv",
            "pbmc_global_detection_by_seed.csv",
            ["held_out_label", "seed", "method"],
            {
                "run_id": "run_id",
                "condition_id": "condition_id",
                "candidate_set": "candidate_set",
                "method_group": "method_group",
                "auroc": "auroc",
                "auprc": "auprc",
                "auprc_baseline": "auprc_baseline",
                "absent_abstention_rate": "absent_abstention_rate",
                "shared_false_abstention_rate": "shared_false_abstention_rate",
            },
        ),
        (
            "S12-summary",
            "s12_global_detection_summary.csv",
            "s12_summary.csv",
            "pbmc_global_detection_summary.csv",
            ["held_out_label", "method"],
            {
                "score": "score",
                "display_name": "display_name",
                "auprc_mean": "auprc_mean",
                "auprc_std": "auprc_std",
                "auroc_mean": "auroc_mean",
                "auroc_std": "auroc_std",
                "n_splits": "n_splits",
            },
        ),
        (
            "S13-by-seed",
            "s13_represented_transfer_by_seed.csv",
            "s13_by_seed.csv",
            "pbmc_represented_transfer_by_seed.csv",
            ["held_out_label", "seed", "method"],
            {
                "run_id": "run_id",
                "candidate_set": "candidate_set",
                "forced_accuracy": "forced_accuracy",
                "forced_macro_f1": "forced_macro_f1",
                "post_abstention_macro_f1": "post_abstention_macro_f1",
                "coverage": "coverage",
                "shared_false_abstention_rate": "shared_false_abstention_rate",
            },
        ),
        (
            "S13-summary",
            "s13_represented_transfer_summary.csv",
            "s13_summary.csv",
            "pbmc_represented_transfer_summary.csv",
            ["held_out_label", "method"],
            {
                "score": "score",
                "display_name": "display_name",
                "selective_metrics_applicable": "selective_metrics_applicable",
                "forced_accuracy_mean": "forced_accuracy_mean",
                "forced_accuracy_std": "forced_accuracy_std",
                "forced_macro_f1_mean": "forced_macro_f1_mean",
                "forced_macro_f1_std": "forced_macro_f1_std",
                "post_abstention_macro_f1_mean": "post_abstention_macro_f1_mean",
                "post_abstention_macro_f1_std": "post_abstention_macro_f1_std",
                "held_out_abstention_mean": "held_out_abstention_mean",
                "held_out_abstention_std": "held_out_abstention_std",
                "coverage_mean": "coverage_mean",
                "coverage_std": "coverage_std",
                "shared_false_abstention_mean": "shared_false_abstention_mean",
                "shared_false_abstention_std": "shared_false_abstention_std",
                "n_splits": "n_splits",
            },
        ),
        (
            "S14-by-seed",
            "s14_operational_abstention_by_seed.csv",
            "s13_by_seed.csv",
            "pbmc_represented_transfer_by_seed.csv",
            ["held_out_label", "seed", "method"],
            {
                "run_id": "run_id",
                "candidate_set": "candidate_set",
                "score": "score",
                "display_name": "display_name",
                "held_out_abstention": "absent_abstention_rate",
                "represented_state_coverage": "coverage",
                "post_abstention_macro_f1": "post_abstention_macro_f1",
            },
        ),
        (
            "S14-summary",
            "s14_operational_abstention_summary.csv",
            "s13_summary.csv",
            "pbmc_represented_transfer_summary.csv",
            ["held_out_label", "method"],
            {
                "score": "score",
                "display_name": "display_name",
                "held_out_abstention_mean": "held_out_abstention_mean",
                "held_out_abstention_std": "held_out_abstention_std",
                "coverage_mean": "coverage_mean",
                "coverage_std": "coverage_std",
                "post_abstention_macro_f1_mean": "post_abstention_macro_f1_mean",
                "post_abstention_macro_f1_std": "post_abstention_macro_f1_std",
                "n_splits": "n_splits",
            },
        ),
    )
    reports: list[pd.DataFrame] = []
    for table, output_name, promotion_name, data3_name, keys, columns in specs:
        rerun = outputs[output_name]
        for authority_name, root, filename in (
            ("certified_promotion", promotion_root, promotion_name),
            ("current_data3", current_data3_root, data3_name),
        ):
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"{table} current authority table does not exist: {path}"
                )
            authority = pd.read_csv(path)
            authority = authority.loc[
                authority["method"].astype(str).eq(METHOD)
            ].copy()
            reports.append(
                _compare_authority_frame(
                    table=table,
                    authority_name=authority_name,
                    rerun=rerun,
                    authority=authority,
                    keys=keys,
                    columns=columns,
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(reports, ignore_index=True).to_csv(output_path, index=False)
    return output_path


def _write_checksums(root: Path) -> Path:
    path = root / "checksums.sha256"
    files = sorted(
        item for item in root.rglob("*") if item.is_file() and item != path
    )
    path.write_text(
        "".join(f"{_sha256(item)}  {item.relative_to(root)}\n" for item in files),
        encoding="utf-8",
    )
    return path


def _command_string(
    *,
    source_root: Path,
    output_root: Path,
    raw_data_path: Path,
    input_repair_manifests: tuple[Path, ...],
    promotion_root: Path,
    current_data3_root: Path,
) -> str:
    command = [
        "uv",
        "run",
        "python",
        "experiments/pbmc_state/rerun_pbmc_celltypist_centered.py",
        "--source-root",
        _display_path(source_root),
        "--output-root",
        _display_path(output_root),
        "--raw-data-path",
        _display_path(raw_data_path),
        "--promotion-root",
        _display_path(promotion_root),
        "--current-data3-root",
        _display_path(current_data3_root),
    ]
    for manifest in input_repair_manifests:
        command.extend(["--input-repair-manifest", _display_path(manifest)])
    return shlex.join(command)


def run(
    *,
    source_root: Path,
    output_root: Path,
    raw_data_path: Path,
    input_repair_manifests: tuple[Path, ...] = (),
    promotion_root: Path = DEFAULT_PROMOTION_ROOT,
    current_data3_root: Path = DEFAULT_CURRENT_DATA3_ROOT,
) -> None:
    configs = _source_configs(source_root)
    repairs = _load_input_repairs(input_repair_manifests)
    _validate_count_input_coverage(configs, repairs)
    _ensure_fresh_output_root(output_root)
    runs_root = output_root / "runs"
    parameters_by_seed: dict[int, dict[str, object]] = {}
    for index, (config_path, config) in enumerate(configs, start=1):
        endpoint = str(config["heldout_label"])
        seed = int(config["repeat"])
        run_id = str(config["run_id"])
        source_run = _source_run(config)
        output_run = runs_root / run_id
        parameters = _celltypist_config(
            {
                "celltypist": {
                    "profile": "centered_full_sgd",
                    "random_state": seed,
                }
            },
            repeat=seed,
        )
        parameters_by_seed[seed] = parameters
        _prepare_output_run(
            source_run=source_run,
            output_run=output_run,
            endpoint=endpoint,
            seed=seed,
            config_path=config_path,
            parameters=parameters,
            repairs=repairs,
        )
        print(f"[{index:02d}/15] fitting {endpoint}, seed {seed}", flush=True)
        _fit_one_run(
            source_run=source_run,
            output_run=output_run,
            endpoint=endpoint,
            seed=seed,
            parameters=parameters,
            repairs=repairs,
        )

    external = write_external_baseline_results(
        runs_root=runs_root,
        grid_dir=output_root / "grid",
        output_root=output_root / "external_results",
        expected_held_out_labels=ENDPOINTS,
        expected_seeds=SEEDS,
        report_title="PBMC seeded centered full-SGD CellTypist rerun",
        report_description=(
            "CellTypist-only isolated rerun for Supplementary Tables S11-S14."
        ),
        source_note=str(output_root),
        methods=(METHOD,),
    )

    tables_root = output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    within_by_seed, within_summary = _within_celltype_tables(
        detection_path=external.detection_by_run,
        runs_root=runs_root,
        raw_data_path=raw_data_path,
    )
    detection = pd.read_csv(external.detection_by_run)
    transfer = pd.read_csv(external.shared_label_transfer_by_run)
    full_reference = pd.read_csv(external.full_reference_by_run)
    global_summary = _summarize_global(detection)
    transfer_summary = _summarize_transfer(transfer, detection)
    operational_abstention = _operational_abstention_by_seed(transfer, detection)
    operational_abstention_summary = _summarize_operational_abstention(
        operational_abstention
    )
    calibration_summary = _summarize_calibration(full_reference)

    outputs = {
        "s11_within_celltype_by_seed.csv": within_by_seed,
        "s11_within_celltype_summary.csv": within_summary,
        "s12_global_detection_by_seed.csv": detection,
        "s12_global_detection_summary.csv": global_summary,
        "s13_represented_transfer_by_seed.csv": transfer,
        "s13_represented_transfer_summary.csv": transfer_summary,
        "s14_operational_abstention_by_seed.csv": operational_abstention,
        "s14_operational_abstention_summary.csv": operational_abstention_summary,
        "diagnostic_full_reference_calibration_by_run.csv": full_reference,
        "diagnostic_full_reference_calibration_summary.csv": calibration_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables_root / name, index=False)

    current_comparison = _write_current_authority_comparison(
        outputs=outputs,
        promotion_root=promotion_root,
        current_data3_root=current_data3_root,
        output_path=tables_root / "current_authority_comparison.csv",
    )

    comparisons = [
        *_comparison_rows(
            table="S11",
            rerun=within_summary,
            retained_path=RETAINED_ROOT / "pbmc_within_celltype_detection_summary.csv",
            keys=["held_out_label", "method", "score", "display_name"],
        ),
        *_comparison_rows(
            table="S12",
            rerun=global_summary,
            retained_path=RETAINED_ROOT / "pbmc_global_detection_summary.csv",
            keys=["held_out_label", "method", "score", "display_name"],
        ),
        *_comparison_rows(
            table="S13",
            rerun=transfer_summary,
            retained_path=RETAINED_ROOT / "pbmc_represented_transfer_summary.csv",
            keys=["held_out_label", "method", "score", "display_name"],
        ),
        *_comparison_rows(
            table="S14",
            rerun=operational_abstention_summary,
            retained_path=RETAINED_ROOT / "pbmc_represented_transfer_summary.csv",
            keys=["held_out_label", "method", "score", "display_name"],
        ),
    ]
    historical_comparison = tables_root / "historical_retained_comparison.csv"
    pd.DataFrame(comparisons).to_csv(historical_comparison, index=False)

    _write_yaml(
        output_root / "manifest.yaml",
        {
            "stage": "pbmc-celltypist-centered-full-sgd-rerun",
            "artifacts": {
                "runs": str(runs_root),
                "tables": str(tables_root),
                "current_authority_comparison": str(current_comparison),
                "historical_retained_comparison": str(historical_comparison),
            },
            "metadata": {
                "dataset": "PBMC",
                "source_v3_root": _display_path(source_root),
                "certified_promotion_root": _display_path(promotion_root),
                "current_data3_root": _display_path(current_data3_root),
                "historical_retained_supplement_root": _display_path(RETAINED_ROOT),
                "raw_data_path": _display_path(raw_data_path),
                "conditions": list(CONDITIONS),
                "endpoints": list(ENDPOINTS),
                "seeds": list(SEEDS),
                "execution_policy": "global_serial",
                "celltypist_workers": 1,
                "celltypist_parameters_by_seed": parameters_by_seed,
                "current_authority_atol": CURRENT_AUTHORITY_ATOL,
                "input_repairs": [
                    {
                        "run_id": repair.run_id,
                        "condition": repair.condition,
                        "manifest": _display_path(repair.manifest_path),
                        "counts": _display_path(repair.counts_path),
                        "sha256": repair.sha256,
                    }
                    for repair in repairs.values()
                ],
                "command": _command_string(
                    source_root=source_root,
                    output_root=output_root,
                    raw_data_path=raw_data_path,
                    input_repair_manifests=input_repair_manifests,
                    promotion_root=promotion_root,
                    current_data3_root=current_data3_root,
                ),
                "runner_sha256": _sha256(
                    PROJECT_ROOT / "src/coreot/external_baselines/runner.py"
                ),
                "candidate_script": str(Path(__file__).resolve()),
                "candidate_script_sha256": _sha256(Path(__file__).resolve()),
                "python": platform.python_version(),
                "celltypist": importlib.metadata.version("celltypist"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
            },
        },
    )
    checksum_path = _write_checksums(output_root)
    print(f"completed: {output_root}", flush=True)
    print(f"checksums: {checksum_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun PBMC CellTypist with seeded centered full-SGD training."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=PROJECT_ROOT / "data/raw/kang_2018.h5ad",
    )
    parser.add_argument(
        "--input-repair-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Declare a verified repair for a missing model-visible counts matrix. "
            "May be repeated; repairs for existing matrices are rejected."
        ),
    )
    parser.add_argument(
        "--promotion-root",
        type=Path,
        default=DEFAULT_PROMOTION_ROOT,
    )
    parser.add_argument(
        "--current-data3-root",
        type=Path,
        default=DEFAULT_CURRENT_DATA3_ROOT,
    )
    args = parser.parse_args()
    run(
        source_root=args.source_root.resolve(),
        output_root=args.output_root.resolve(),
        raw_data_path=args.raw_data_path.resolve(),
        input_repair_manifests=tuple(
            path.resolve() for path in args.input_repair_manifest
        ),
        promotion_root=args.promotion_root.resolve(),
        current_data3_root=args.current_data3_root.resolve(),
    )


if __name__ == "__main__":
    main()
