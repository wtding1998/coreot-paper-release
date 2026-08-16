from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.missing_celltype import (
    generate_hiha_dc_compare_baselines,
    generate_hiha_dc_coreot_constant_tau_alpha_broad_grid,
    generate_hiha_dc_coreot_constant_tau_alpha_broad_grid_configs,
    generate_hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid,
    generate_hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid_configs,
    generate_hiha_dc_report_leave_one_configs,
)
import coreot.results.compare_baselines as compare_baselines
from coreot.artifacts.manifests import read_manifest
from coreot.artifacts.run_artifacts import ArtifactBatchError, ArtifactInvalid
from coreot.results.figures import write_figure_1
from coreot.results.grid import (
    FIXED_REFERENCE_METHODS,
    ResultsGridError,
    RunDescriptor,
    _filter_completed_runs,
    parse_coreot_full_tau_range_params,
    read_fixed_reference_rows,
    write_broad_grid_results,
    write_compare_baselines_results,
    write_coreot_full_tau_range_results,
    write_grid_results,
)


def test_write_grid_results_creates_detection_and_shared_tables(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    grid_dir = tmp_path / "generated_configs"
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id="hiha_dc_isg_cdc2_seed1",
        held_out_label="ISG+ cDC2",
        seed=1,
        nn_auroc=0.8,
        prior_auroc=0.6,
    )
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id="hiha_dc_isg_cdc2_seed2",
        held_out_label="ISG+ cDC2",
        seed=2,
        nn_auroc=0.9,
        prior_auroc=0.7,
    )
    output_root = tmp_path / "results" / "hiha_dc_main_grid"
    (output_root / "tables").mkdir(parents=True)
    (output_root / "tables" / "main_table_1.md").write_text("stale", encoding="utf-8")
    (output_root / "tables" / "main_table_2.md").write_text("stale", encoding="utf-8")

    paths = write_grid_results(
        runs_root=runs_root,
        grid_dir=grid_dir,
        output_root=output_root,
        candidate_set="toy_k1",
        write_figures=False,
        methods=("nn", "prior_only"),
    )

    detection_by_run = pd.read_csv(paths.detection_by_run)
    expected_detection_columns = [
        "run_id",
        "held_out_label",
        "seed",
        "condition_id",
        "candidate_set",
        "method",
        "method_group",
        "primary_score",
        "score",
        "auroc",
        "auprc",
        "auprc_baseline",
        "absent_abstention_rate",
        "shared_false_abstention_rate",
        "median_absent",
        "median_shared",
        "absent_minus_shared_median",
    ]
    assert detection_by_run.columns.tolist() == expected_detection_columns
    assert (
        detection_by_run.duplicated(["run_id", "held_out_label", "seed", "method"]).sum()
        == 0
    )
    assert len(detection_by_run) == 4

    nn_rows = detection_by_run.loc[detection_by_run["method"] == "nn"]
    assert set(nn_rows["primary_score"]) == {"nn_distance"}
    assert nn_rows["auroc"].tolist() == [0.8, 0.9]
    assert nn_rows["auprc_baseline"].tolist() == [pytest.approx(1 / 3), pytest.approx(1 / 3)]
    assert nn_rows["shared_false_abstention_rate"].tolist() == [0.5, 0.5]

    detection_summary = pd.read_csv(paths.detection_summary)
    summary_row = detection_summary.loc[
        (detection_summary["held_out_label"] == "ISG+ cDC2")
        & (detection_summary["method"] == "nn")
        & (detection_summary["quantity"] == "auroc")
    ].iloc[0]
    assert summary_row["mean"] == pytest.approx(0.85)
    assert summary_row["n_runs"] == 2
    assert summary_row["sem"] == pytest.approx(0.05)

    shared_by_run = pd.read_csv(paths.shared_label_transfer_by_run)
    assert set(shared_by_run["method"]) == {"nn"}
    shared = shared_by_run.iloc[0]
    assert shared["forced_accuracy"] == pytest.approx(0.5)
    assert shared["post_abstention_accuracy"] == pytest.approx(1.0)
    assert shared["coverage"] == pytest.approx(0.5)
    assert shared["shared_false_abstention_rate"] == pytest.approx(0.5)

    assert paths.manifest.is_file()
    assert paths.main_tables.is_file()
    assert not (output_root / "tables" / "main_table_1.md").exists()
    assert not (output_root / "tables" / "main_table_2.md").exists()
    main_tables = paths.main_tables.read_text(encoding="utf-8")
    assert "Main Table 1: absent-state detection" in main_tables
    assert "Main Table 2: shared-cell label transfer" in main_tables
    assert "Main Table S1: full-reference negative control" in main_tables
    assert paths.markdown_summary.is_file()
    assert "Main Detection Summary" in paths.markdown_summary.read_text(encoding="utf-8")
    full_reference = pd.read_csv(paths.full_reference_false_abstention_by_run)
    assert set(full_reference["condition_id"]) == {"full_reference_control"}
    assert full_reference.loc[
        full_reference["method"] == "nn", "full_reference_false_abstention_rate"
    ].tolist() == [pytest.approx(1 / 3), pytest.approx(1 / 3)]
    assert full_reference.loc[
        full_reference["method"] == "nn", "score_median"
    ].tolist() == [pytest.approx(0.2), pytest.approx(0.2)]
    assert full_reference.loc[
        full_reference["method"] == "nn", "coverage"
    ].tolist() == [pytest.approx(2 / 3), pytest.approx(2 / 3)]
    full_reference_summary = pd.read_csv(paths.full_reference_false_abstention_summary)
    assert {
        "score_median",
        "score_p95",
        "forced_accuracy",
        "forced_macro_f1",
        "post_abstention_accuracy",
        "post_abstention_macro_f1",
        "coverage",
    }.issubset(set(full_reference_summary["quantity"]))


def test_u_tilde_override_uses_one_policy_across_result_tables(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    grid_dir = tmp_path / "generated_configs"
    run_id = "hiha_dc_hladrhi_cdc2_seed1"
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id=run_id,
        held_out_label="HLA-DRhi cDC2",
        seed=1,
        nn_auroc=0.8,
        prior_auroc=0.6,
    )
    scoring_path = grid_dir / run_id / "scoring.yaml"
    scoring_path.write_text(
        yaml.safe_dump(
            {
                "thresholds": {
                    "theta_u": {
                        "source_condition": "full_reference_control",
                        "quantile": 0.95,
                    },
                    "theta_H": {"value": 0.8},
                }
            }
        ),
        encoding="utf-8",
    )
    scores_path = (
        runs_root
        / run_id
        / "scoring"
        / "incomplete_reference"
        / "toy_k1"
        / "cell_scores.parquet"
    )
    scores = pd.read_parquet(scores_path)
    scores.loc[
        (scores["method"] == "coreot_full") & (scores["cell_id"] == "q_shared_b"),
        "u_tilde",
    ] = 0.9
    scores.to_parquet(scores_path, index=False)

    paths = write_grid_results(
        runs_root=runs_root,
        grid_dir=grid_dir,
        output_root=tmp_path / "results",
        candidate_set="toy_k1",
        write_figures=False,
        methods=("coreot_full",),
        score_overrides={"coreot_full": "u_tilde"},
    )

    detection = pd.read_csv(paths.detection_by_run)
    shared = pd.read_csv(paths.shared_label_transfer_by_run)
    full_reference = pd.read_csv(paths.full_reference_false_abstention_by_run)
    detection_u_tilde = detection.loc[detection["score"] == "u_tilde"].iloc[0]
    shared_u_tilde = shared.loc[shared["score"] == "u_tilde"].iloc[0]
    full_reference_u_tilde = full_reference.loc[
        full_reference["score"] == "u_tilde"
    ].iloc[0]

    assert detection_u_tilde["absent_abstention_rate"] == pytest.approx(1.0)
    assert detection_u_tilde["shared_false_abstention_rate"] == pytest.approx(0.5)
    assert shared_u_tilde["shared_false_abstention_rate"] == pytest.approx(0.5)
    assert shared_u_tilde["coverage"] == pytest.approx(0.5)
    assert shared_u_tilde["post_abstention_accuracy"] == pytest.approx(1.0)
    assert full_reference_u_tilde["full_reference_false_abstention_rate"] == pytest.approx(
        1 / 3
    )


def test_write_compare_baselines_results_merges_internal_and_external_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_write_grid_results(**kwargs: object) -> SimpleNamespace:
        output_root = Path(kwargs["output_root"])
        tables = output_root / "tables"
        tables.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "run_id": "run1",
                    "held_out_label": "ISG+ cDC2",
                    "seed": 1,
                    "condition_id": "incomplete_reference",
                    "candidate_set": "hiha_harmony30_k100",
                    "method": "coreot_full",
                    "method_group": "coreot",
                    "primary_score": "u",
                    "score": "u",
                    "auroc": 0.9,
                    "auprc": 0.8,
                    "auprc_baseline": 0.2,
                    "absent_abstention_rate": 0.7,
                    "shared_false_abstention_rate": 0.1,
                }
            ]
        ).to_csv(tables / "main_detection_by_run.csv", index=False)
        _write_summary(
            tables / "main_detection_summary.csv",
            method="coreot_full",
            method_group="coreot",
            score="u",
            quantities=("auroc", "auprc", "auprc_baseline", "absent_abstention_rate", "shared_false_abstention_rate"),
        )
        pd.DataFrame(
            [
                {
                    "run_id": "run1",
                    "held_out_label": "ISG+ cDC2",
                    "seed": 1,
                    "condition_id": "incomplete_reference",
                    "candidate_set": "hiha_harmony30_k100",
                    "method": "coreot_full",
                    "method_group": "coreot",
                    "score": "u",
                    "forced_accuracy": 0.9,
                    "forced_macro_f1": 0.8,
                    "post_abstention_accuracy": 0.95,
                    "post_abstention_macro_f1": 0.85,
                    "coverage": 0.9,
                    "shared_false_abstention_rate": 0.1,
                }
            ]
        ).to_csv(tables / "shared_label_transfer_by_run.csv", index=False)
        _write_summary(
            tables / "shared_label_transfer_summary.csv",
            method="coreot_full",
            method_group="coreot",
            score="u",
            quantities=(
                "forced_accuracy",
                "forced_macro_f1",
                "post_abstention_accuracy",
                "post_abstention_macro_f1",
                "coverage",
                "shared_false_abstention_rate",
            ),
        )
        pd.DataFrame(
            [
                {
                    "run_id": "run1",
                    "held_out_label": "ISG+ cDC2",
                    "seed": 1,
                    "condition_id": "full_reference_control",
                    "candidate_set": "hiha_harmony30_k100",
                    "method": "coreot_full",
                    "method_group": "coreot",
                    "score": "u",
                    "full_reference_false_abstention_rate": 0.05,
                    "score_median": 0.1,
                    "score_p95": 0.2,
                    "forced_accuracy": 0.9,
                    "forced_macro_f1": 0.8,
                    "post_abstention_accuracy": 0.95,
                    "post_abstention_macro_f1": 0.85,
                    "coverage": 0.95,
                }
            ]
        ).to_csv(tables / "full_reference_false_abstention_by_run.csv", index=False)
        _write_summary(
            tables / "full_reference_false_abstention_summary.csv",
            method="coreot_full",
            method_group="coreot",
            score="u",
            quantities=(
                "full_reference_false_abstention_rate",
                "score_median",
                "score_p95",
                "forced_accuracy",
                "forced_macro_f1",
                "post_abstention_accuracy",
                "post_abstention_macro_f1",
                "coverage",
            ),
        )
        return SimpleNamespace(
            output_root=output_root,
            detection_by_run=tables / "main_detection_by_run.csv",
            detection_summary=tables / "main_detection_summary.csv",
            shared_label_transfer_by_run=tables / "shared_label_transfer_by_run.csv",
            shared_label_transfer_summary=tables / "shared_label_transfer_summary.csv",
            full_reference_false_abstention_by_run=tables / "full_reference_false_abstention_by_run.csv",
            full_reference_false_abstention_summary=tables / "full_reference_false_abstention_summary.csv",
        )

    def fake_write_external_baseline_results(**kwargs: object) -> SimpleNamespace:
        output_root = Path(kwargs["output_root"])
        tables = output_root / "tables"
        tables.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "run_id": "run1",
                    "held_out_label": "ISG+ cDC2",
                    "seed": 1,
                    "condition_id": "incomplete_reference",
                    "candidate_set": "external_reference_mapping",
                    "method": "seurat_anchor",
                    "method_group": "external_baseline",
                    "auroc": 0.7,
                    "auprc": 0.6,
                    "auprc_baseline": 0.2,
                    "absent_abstention_rate": 0.1,
                    "shared_false_abstention_rate": 0.02,
                }
            ]
        ).to_csv(tables / "external_detection_by_run.csv", index=False)
        _write_summary(
            tables / "external_detection_summary.csv",
            method="seurat_anchor",
            method_group="external_baseline",
            score=None,
            quantities=("auroc", "auprc", "auprc_baseline", "absent_abstention_rate", "shared_false_abstention_rate"),
        )
        pd.DataFrame(
            [
                {
                    "run_id": "run1",
                    "held_out_label": "ISG+ cDC2",
                    "seed": 1,
                    "condition_id": "incomplete_reference",
                    "candidate_set": "external_reference_mapping",
                    "method": "seurat_anchor",
                    "method_group": "external_baseline",
                    "forced_accuracy": 0.85,
                    "forced_macro_f1": 0.75,
                    "post_abstention_accuracy": 0.86,
                    "post_abstention_macro_f1": 0.76,
                    "coverage": 0.98,
                    "shared_false_abstention_rate": 0.02,
                }
            ]
        ).to_csv(tables / "external_shared_label_transfer_by_run.csv", index=False)
        _write_summary(
            tables / "external_shared_label_transfer_summary.csv",
            method="seurat_anchor",
            method_group="external_baseline",
            score=None,
            quantities=(
                "forced_accuracy",
                "forced_macro_f1",
                "post_abstention_accuracy",
                "post_abstention_macro_f1",
                "coverage",
                "shared_false_abstention_rate",
            ),
        )
        pd.DataFrame(
            [
                {
                    "run_id": "run1",
                    "held_out_label": "ISG+ cDC2",
                    "seed": 1,
                    "condition_id": "full_reference_control",
                    "candidate_set": "external_reference_mapping",
                    "method": "seurat_anchor",
                    "method_group": "external_baseline",
                    "full_reference_false_abstention_rate": 0.05,
                    "score_median": 0.1,
                    "score_p95": 0.2,
                    "forced_accuracy": 0.85,
                    "forced_macro_f1": 0.75,
                    "post_abstention_accuracy": 0.86,
                    "post_abstention_macro_f1": 0.76,
                    "coverage": 0.95,
                }
            ]
        ).to_csv(tables / "external_full_reference_by_run.csv", index=False)
        _write_summary(
            tables / "external_full_reference_summary.csv",
            method="seurat_anchor",
            method_group="external_baseline",
            score=None,
            quantities=(
                "full_reference_false_abstention_rate",
                "score_median",
                "score_p95",
                "forced_accuracy",
                "forced_macro_f1",
                "post_abstention_accuracy",
                "post_abstention_macro_f1",
                "coverage",
            ),
        )
        return SimpleNamespace(
            output_root=output_root,
            detection_by_run=tables / "external_detection_by_run.csv",
            detection_summary=tables / "external_detection_summary.csv",
            shared_label_transfer_by_run=tables / "external_shared_label_transfer_by_run.csv",
            shared_label_transfer_summary=tables / "external_shared_label_transfer_summary.csv",
            full_reference_by_run=tables / "external_full_reference_by_run.csv",
            full_reference_summary=tables / "external_full_reference_summary.csv",
        )

    monkeypatch.setattr(compare_baselines, "write_grid_results", fake_write_grid_results)
    monkeypatch.setattr(
        compare_baselines,
        "write_external_baseline_results",
        fake_write_external_baseline_results,
    )

    paths = write_compare_baselines_results(
        runs_root=tmp_path / "runs",
        grid_dir=tmp_path / "grid",
        output_root=tmp_path / "results" / "HIHA_DC" / "compare_baselines",
    )

    detection = pd.read_csv(paths.detection_summary)
    assert set(detection["result_family"]) == {"internal", "external_baseline"}
    assert set(detection["method"]) == {"coreot_full", "seurat_anchor"}
    assert set(detection["score"]) == {"u"}
    assert paths.main_tables.is_file()
    assert "Table 1: absent-state detection" in paths.main_tables.read_text(encoding="utf-8")
    manifest = read_manifest(paths.manifest)
    assert manifest.metadata["internal_candidate_set"] == "hiha_harmony30_k100"
    assert manifest.metadata["external_methods"] == ["seurat_anchor"]
    assert manifest.artifacts["compare_detection_summary"] == str(paths.detection_summary)


def test_select_primary_comparison_rows_excludes_secondary_scores() -> None:
    comparison = pd.DataFrame(
        {
            "method": [
                "coreot_full",
                "coreot_full",
                "coreot_match_only",
                "coreot_match_only",
                "prior_only",
                "seurat_anchor",
            ],
            "score": ["u", "u_tilde", "u", "u_tilde", "prior_risk", "u"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )

    selected = compare_baselines.select_primary_comparison_rows(comparison)

    assert set(selected[["method", "score"]].itertuples(index=False, name=None)) == {
        ("coreot_full", "u"),
        ("coreot_match_only", "u"),
        ("prior_only", "prior_risk"),
        ("seurat_anchor", "u"),
    }
    assert selected["value"].tolist() == [1, 3, 5, 6]

    with pytest.raises(ValueError, match="missing primary-score rows"):
        compare_baselines.select_primary_comparison_rows(
            comparison.loc[
                comparison["method"].eq("coreot_full")
                & comparison["score"].eq("u_tilde")
            ]
        )


def test_generate_hiha_dc_compare_baselines_cli_passes_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    def fake_write_compare_baselines_results(**kwargs: object) -> SimpleNamespace:
        calls.update(kwargs)
        return SimpleNamespace(
            output_root=tmp_path / "results" / "HIHA_DC" / "compare_baselines"
        )

    monkeypatch.setattr(
        generate_hiha_dc_compare_baselines,
        "write_compare_baselines_results",
        fake_write_compare_baselines_results,
    )

    exit_code = generate_hiha_dc_compare_baselines.main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--grid-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(tmp_path / "results" / "HIHA_DC" / "compare_baselines"),
        ]
    )

    assert exit_code == 0
    assert calls["runs_root"] == tmp_path / "runs"
    assert calls["grid_dir"] == tmp_path / "generated"
    assert calls["output_root"] == tmp_path / "results" / "HIHA_DC" / "compare_baselines"
    assert calls["internal_candidate_set"] == "hiha_harmony30_k100"
    assert calls["internal_method_grid_dirs"] == {
        "uniform_uot": Path(
            "experiments/missing_celltype/generated_configs/"
            "hiha_dc_uniform_uot_tau05"
        )
    }
    assert capsys.readouterr().out.strip() == str(
        tmp_path / "results" / "HIHA_DC" / "compare_baselines"
    )


def test_write_grid_results_rejects_missing_primary_metric(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    grid_dir = tmp_path / "generated_configs"
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id="hiha_dc_isg_cdc2_seed1",
        held_out_label="ISG+ cDC2",
        seed=1,
        nn_auroc=0.8,
        prior_auroc=0.6,
    )
    metrics_path = runs_root / "hiha_dc_isg_cdc2_seed1" / "evaluation" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[~((metrics["method"] == "nn") & (metrics["score"] == "nn_distance"))]
    metrics.to_csv(metrics_path, index=False)

    with pytest.raises(ResultsGridError, match="Expected exactly one metric value"):
        write_grid_results(
            runs_root=runs_root,
            grid_dir=grid_dir,
            output_root=tmp_path / "results",
            candidate_set="toy_k1",
            write_figures=False,
            methods=("nn",),
        )


def test_write_grid_results_reports_missing_artifacts_in_batch(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    grid_dir = tmp_path / "generated_configs"
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id="hiha_dc_isg_cdc2_seed1",
        held_out_label="ISG+ cDC2",
        seed=1,
        nn_auroc=0.8,
        prior_auroc=0.6,
    )
    missing_path = (
        runs_root
        / "hiha_dc_isg_cdc2_seed1"
        / "benchmark"
        / "full_reference_control"
        / "evaluation_truth"
        / "query_truth.csv"
    )
    missing_path.unlink()

    with pytest.raises(ArtifactBatchError) as exc_info:
        write_grid_results(
            runs_root=runs_root,
            grid_dir=grid_dir,
            output_root=tmp_path / "results",
            candidate_set="toy_k1",
            write_figures=False,
            methods=("nn",),
        )

    failures = exc_info.value.failures
    assert len(failures) == 1
    assert failures[0].run_id == "hiha_dc_isg_cdc2_seed1"
    assert failures[0].artifact_kind == "query_truth"
    assert failures[0].condition == "full_reference_control"


def test_write_grid_results_skip_incomplete_mode_uses_completed_runs_and_records_skips(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    grid_dir = tmp_path / "generated_configs"
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id="hiha_dc_isg_cdc2_seed1",
        held_out_label="ISG+ cDC2",
        seed=1,
        nn_auroc=0.8,
        prior_auroc=0.6,
    )
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id="hiha_dc_isg_cdc2_seed2",
        held_out_label="ISG+ cDC2",
        seed=2,
        nn_auroc=0.9,
        prior_auroc=0.7,
    )
    missing_path = (
        runs_root
        / "hiha_dc_isg_cdc2_seed2"
        / "benchmark"
        / "full_reference_control"
        / "evaluation_truth"
        / "query_truth.csv"
    )
    missing_path.unlink()

    paths = write_grid_results(
        runs_root=runs_root,
        grid_dir=grid_dir,
        output_root=tmp_path / "results" / "hiha_dc_main_grid",
        candidate_set="toy_k1",
        write_figures=False,
        methods=("nn",),
        skip_incomplete=True,
    )

    detection_by_run = pd.read_csv(paths.detection_by_run)
    assert set(detection_by_run["run_id"]) == {"hiha_dc_isg_cdc2_seed1"}

    manifest = read_manifest(paths.manifest)
    assert manifest.metadata["skip_incomplete"] is True
    assert manifest.metadata["n_runs_expected"] == 2
    assert manifest.metadata["n_runs_completed"] == 1
    skipped_runs = manifest.metadata["skipped_runs"]
    assert len(skipped_runs) == 1
    skipped = skipped_runs[0]
    assert skipped["run_id"] == "hiha_dc_isg_cdc2_seed2"
    assert skipped["artifact_kind"] == "query_truth"
    assert skipped["state"] == "missing"
    assert skipped["condition"] == "full_reference_control"
    assert skipped["path"] == str(missing_path)

    summary = paths.markdown_summary.read_text(encoding="utf-8")
    assert "## Skipped Runs" in summary
    assert "hiha_dc_isg_cdc2_seed2" in summary
    assert "missing" in summary


def test_write_grid_results_can_write_figure_1(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    grid_dir = tmp_path / "generated_configs"
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id="hiha_dc_isg_cdc2_seed1",
        held_out_label="ISG+ cDC2",
        seed=1,
        nn_auroc=0.8,
        prior_auroc=0.6,
    )

    paths = write_grid_results(
        runs_root=runs_root,
        grid_dir=grid_dir,
        output_root=tmp_path / "results" / "hiha_dc_main_grid",
        candidate_set="toy_k1",
        embedding_name="toy_embedding",
        methods=("coreot_full",),
    )

    assert paths.figure_1 is not None
    assert paths.figure_1.png.is_file()
    assert paths.figure_1.pdf.is_file()
    assert paths.figure_1.png.stat().st_size > 0
    assert paths.figure_1.pdf.stat().st_size > 0
    assert paths.figure_2 is not None
    assert paths.figure_2.png.is_file()
    assert paths.figure_2.pdf.is_file()
    assert paths.figure_2.png.stat().st_size > 0
    assert paths.figure_2.pdf.stat().st_size > 0

    manifest = read_manifest(paths.manifest)
    assert manifest.artifacts["figure_1_png"] == str(paths.figure_1.png)
    assert manifest.artifacts["figure_1_pdf"] == str(paths.figure_1.pdf)
    assert manifest.artifacts["figure_2_png"] == str(paths.figure_2.png)
    assert manifest.artifacts["figure_2_pdf"] == str(paths.figure_2.pdf)


def test_write_figure_1_uses_artifact_contract_for_invalid_query_truth(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    grid_dir = tmp_path / "generated_configs"
    run_id = "hiha_dc_isg_cdc2_seed1"
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id=run_id,
        held_out_label="ISG+ cDC2",
        seed=1,
        nn_auroc=0.8,
        prior_auroc=0.6,
    )
    paths = write_grid_results(
        runs_root=runs_root,
        grid_dir=grid_dir,
        output_root=tmp_path / "results" / "hiha_dc_main_grid",
        candidate_set="toy_k1",
        write_figures=False,
        methods=("coreot_full",),
    )
    truth_path = (
        runs_root
        / run_id
        / "benchmark"
        / "incomplete_reference"
        / "evaluation_truth"
        / "query_truth.csv"
    )
    invalid_truth = pd.read_csv(truth_path).drop(columns=["true_label"])
    invalid_truth.to_csv(truth_path, index=False)

    with pytest.raises(ArtifactInvalid, match="query_truth.csv"):
        write_figure_1(
            runs_root=runs_root,
            output_root=tmp_path / "standalone_results",
            detection_summary_path=paths.detection_summary,
            forced_label_summary_path=paths.forced_label_summary_by_run,
            candidate_set="toy_k1",
            embedding_name="toy_embedding",
            run_id=run_id,
        )


def test_generate_broad_grid_configs_creates_split_coreot_and_uniform_dirs(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir(parents=True)
    for name in (
        "raw_import",
        "benchmark",
        "derivation",
        "embedding",
        "candidates",
        "transport",
        "scoring",
        "evaluation",
        "report",
    ):
        payload: dict[str, object] = {"run_id": "base"}
        if name == "benchmark":
            payload["removed_state"] = "ISG+ cDC2"
            payload["split"] = {"seed": 1}
        elif name == "transport":
            payload = {
                "run_id": "base",
                "methods": [
                    {"name": "uniform_uot", "tau_source": 1.0, "tau_target": 1.0},
                    {
                        "name": "coreot_constant_tau",
                        "tau_source": 1.0,
                        "tau_target": 1.0,
                        "alpha": 1.0,
                    },
                    "prior_only",
                ],
            }
        elif name in ("scoring", "evaluation"):
            payload = {
                "run_id": "base",
                "methods": ["uniform_uot", "coreot_constant_tau", "prior_only"],
            }
        (base_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    coreot_dir = tmp_path / "generated" / "coreot"
    uniform_dir = tmp_path / "generated" / "uniform"
    exit_code = generate_hiha_dc_coreot_constant_tau_alpha_broad_grid_configs.main(
        [
            "--base-dir",
            str(base_dir),
            "--coreot-output-dir",
            str(coreot_dir),
            "--uniform-output-dir",
            str(uniform_dir),
            "--held-out-label",
            "ISG+ cDC2",
            "--seed",
            "1",
            "--tau-min",
            "1",
            "--tau-max",
            "2",
            "--alpha-min",
            "1",
            "--alpha-max",
            "2",
        ]
    )

    assert exit_code == 0
    coreot_runs = sorted(path.name for path in coreot_dir.iterdir() if path.is_dir())
    uniform_runs = sorted(path.name for path in uniform_dir.iterdir() if path.is_dir())
    assert coreot_runs == [
        "hiha_dc_isg_cdc2_seed1_tau1_alpha1_coreot_constant_tau",
        "hiha_dc_isg_cdc2_seed1_tau1_alpha2_coreot_constant_tau",
        "hiha_dc_isg_cdc2_seed1_tau2_alpha1_coreot_constant_tau",
        "hiha_dc_isg_cdc2_seed1_tau2_alpha2_coreot_constant_tau",
    ]
    assert uniform_runs == [
        "hiha_dc_isg_cdc2_seed1_tau1_alpha0_uniform",
        "hiha_dc_isg_cdc2_seed1_tau2_alpha0_uniform",
    ]

    coreot_transport = yaml.safe_load(
        (coreot_dir / coreot_runs[0] / "transport.yaml").read_text(encoding="utf-8")
    )
    uniform_transport = yaml.safe_load(
        (uniform_dir / uniform_runs[0] / "transport.yaml").read_text(encoding="utf-8")
    )
    assert coreot_transport["methods"] == [
        {
            "name": "coreot_constant_tau",
            "tau_source": 1.0,
            "tau_target": 1.0,
            "alpha": 1.0,
        },
        "prior_only",
    ]
    assert uniform_transport["methods"] == [
        {"name": "uniform_uot", "tau_source": 1.0, "tau_target": 1.0},
        "prior_only",
    ]
    uniform_scoring = yaml.safe_load(
        (uniform_dir / uniform_runs[0] / "scoring.yaml").read_text(encoding="utf-8")
    )
    assert uniform_scoring["methods"] == ["uniform_uot", "prior_only"]


def test_generate_broad_grid_results_cli_passes_split_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    def fake_write_broad_grid_results(**kwargs: object) -> SimpleNamespace:
        calls.update(kwargs)
        return SimpleNamespace(output_root=tmp_path / "results" / "broad")

    monkeypatch.setattr(
        generate_hiha_dc_coreot_constant_tau_alpha_broad_grid,
        "write_broad_grid_results",
        fake_write_broad_grid_results,
    )

    exit_code = generate_hiha_dc_coreot_constant_tau_alpha_broad_grid.main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--coreot-grid-dir",
            str(tmp_path / "generated" / "coreot"),
            "--uniform-grid-dir",
            str(tmp_path / "generated" / "uniform"),
            "--output-root",
            str(tmp_path / "results" / "broad"),
            "--main-grid-root",
            str(tmp_path / "results" / "results_leave_one_HIHA_DC" / "tuned_baseline"),
            "--figures",
        ]
    )

    assert exit_code == 0
    assert calls["coreot_grid_dir"] == tmp_path / "generated" / "coreot"
    assert calls["uniform_grid_dir"] == tmp_path / "generated" / "uniform"
    assert calls["main_grid_root"] == (
        tmp_path / "results" / "results_leave_one_HIHA_DC" / "tuned_baseline"
    )


def test_generate_coreot_full_tau_range_configs_creates_coreot_only_grid(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir(parents=True)
    for name in (
        "raw_import",
        "benchmark",
        "derivation",
        "embedding",
        "candidates",
        "transport",
        "scoring",
        "evaluation",
        "report",
    ):
        payload: dict[str, object] = {"run_id": "base"}
        if name == "benchmark":
            payload["removed_state"] = "ISG+ cDC2"
            payload["split"] = {"seed": 1}
        elif name == "transport":
            payload = {
                "run_id": "base",
                "methods": [
                    {"name": "uniform_uot", "tau_source": 1.0, "tau_target": 1.0},
                    {
                        "name": "coreot_full",
                        "tau_min": 0.05,
                        "tau_max": 1.0,
                        "tau_target": 1.0,
                        "alpha": 0.25,
                    },
                    "prior_only",
                ],
            }
        elif name in ("scoring", "evaluation"):
            payload = {
                "run_id": "base",
                "methods": ["uniform_uot", "coreot_full", "prior_only"],
            }
        (base_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    output_dir = tmp_path / "generated" / "coreot_full"
    exit_code = generate_hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid_configs.main(
        [
            "--base-dir",
            str(base_dir),
            "--output-dir",
            str(output_dir),
            "--held-out-label",
            "ISG+ cDC2",
            "--seed",
            "1",
            "--tau-min-values",
            "6",
            "8",
            "--tau-max-values",
            "7",
            "9",
            "--alpha",
            "5",
        ]
    )

    assert exit_code == 0
    runs = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
    assert runs == [
        "hiha_dc_isg_cdc2_seed1_taumin6_taumax7_alpha5_coreot_full",
        "hiha_dc_isg_cdc2_seed1_taumin6_taumax9_alpha5_coreot_full",
        "hiha_dc_isg_cdc2_seed1_taumin8_taumax9_alpha5_coreot_full",
    ]

    transport = yaml.safe_load(
        (output_dir / runs[0] / "transport.yaml").read_text(encoding="utf-8")
    )
    assert transport["methods"] == [
        {
            "name": "coreot_full",
            "tau_min": 6.0,
            "tau_max": 7.0,
            "tau_target": 1.0,
            "alpha": 5.0,
        },
        "prior_only",
    ]
    scoring = yaml.safe_load(
        (output_dir / runs[0] / "scoring.yaml").read_text(encoding="utf-8")
    )
    assert scoring["methods"] == ["coreot_full", "prior_only"]


def test_generate_report_leave_one_configs_keeps_baselines_and_selected_params(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir(parents=True)
    for name in (
        "raw_import",
        "benchmark",
        "derivation",
        "embedding",
        "candidates",
        "transport",
        "scoring",
        "evaluation",
        "report",
    ):
        payload: dict[str, object] = {"run_id": "base"}
        if name == "benchmark":
            payload["removed_state"] = "ISG+ cDC2"
            payload["split"] = {"seed": 1}
        elif name == "transport":
            payload = {
                "run_id": "base",
                "methods": [
                    {"name": "nn"},
                    {"name": "uniform_uot", "tau_source": 1.0, "tau_target": 1.0},
                    {"name": "balanced_ot", "alpha": 0.0},
                    {
                        "name": "coreot_full",
                        "tau_min": 0.05,
                        "tau_max": 1.0,
                        "tau_target": 1.0,
                        "alpha": 0.25,
                    },
                    {"name": "coreot_constant_tau", "tau_source": 1.0},
                    {"name": "prior_only"},
                ],
            }
        elif name in ("scoring", "evaluation"):
            payload = {
                "run_id": "base",
                "methods": [
                    "nn",
                    "uniform_uot",
                    "balanced_ot",
                    "coreot_full",
                    "coreot_constant_tau",
                    "prior_only",
                ],
            }
        (base_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    output_dir = tmp_path / "generated" / "report_leave_one_HIHA_DC"
    exit_code = generate_hiha_dc_report_leave_one_configs.main(
        [
            "--base-dir",
            str(base_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "1",
        ]
    )

    assert exit_code == 0
    runs = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
    assert runs == [
        "hiha_dc_hladrhi_cdc2_seed1_report_leave_one_HIHA_DC",
        "hiha_dc_isg_cdc2_seed1_report_leave_one_HIHA_DC",
    ]

    hla_transport = yaml.safe_load(
        (output_dir / runs[0] / "transport.yaml").read_text(encoding="utf-8")
    )
    isg_transport = yaml.safe_load(
        (output_dir / runs[1] / "transport.yaml").read_text(encoding="utf-8")
    )
    assert [method["name"] for method in hla_transport["methods"]] == [
        "nn",
        "uniform_uot",
        "balanced_ot",
        "coreot_full",
        "coreot_match_only",
        "prior_only",
    ]
    hla_uniform = hla_transport["methods"][1]
    hla_full = hla_transport["methods"][3]
    hla_match_only = hla_transport["methods"][4]
    isg_uniform = isg_transport["methods"][1]
    isg_full = isg_transport["methods"][3]
    assert hla_uniform["tau_source"] == pytest.approx(0.5)
    assert hla_uniform["tau_target"] == pytest.approx(0.5)
    assert hla_full["tau_min"] == pytest.approx(2.5)
    assert hla_full["tau_max"] == pytest.approx(3.0)
    assert hla_full["tau_target"] == pytest.approx(2.0)
    assert hla_full["alpha"] == pytest.approx(2.0)
    assert hla_match_only == hla_full | {
        "name": "coreot_match_only",
        "alpha": 0.0,
    }
    assert isg_uniform["tau_source"] == pytest.approx(0.5)
    assert isg_uniform["tau_target"] == pytest.approx(0.5)
    assert isg_full["tau_min"] == pytest.approx(0.5)
    assert isg_full["tau_max"] == pytest.approx(0.625)
    assert isg_full["tau_target"] == pytest.approx(1.0)
    assert isg_full["alpha"] == pytest.approx(0.25)

    hla_scoring = yaml.safe_load(
        (output_dir / runs[0] / "scoring.yaml").read_text(encoding="utf-8")
    )
    assert hla_scoring["methods"] == [
        "nn",
        "uniform_uot",
        "balanced_ot",
        "coreot_full",
        "coreot_match_only",
        "prior_only",
    ]

    uniform_output_dir = tmp_path / "generated" / "hiha_dc_uniform_uot_tau05"
    exit_code = generate_hiha_dc_report_leave_one_configs.main(
        [
            "--base-dir",
            str(base_dir),
            "--output-dir",
            str(uniform_output_dir),
            "--seed",
            "1",
            "--run-id-suffix",
            "_tau05_uniform",
            "--uniform-uot-only",
        ]
    )

    assert exit_code == 0
    uniform_run = uniform_output_dir / "hiha_dc_hladrhi_cdc2_seed1_tau05_uniform"
    uniform_transport = yaml.safe_load(
        (uniform_run / "transport.yaml").read_text(encoding="utf-8")
    )
    assert [method["name"] for method in uniform_transport["methods"]] == [
        "uniform_uot",
        "prior_only",
    ]
    assert uniform_transport["methods"][0]["tau_source"] == pytest.approx(0.5)
    assert uniform_transport["methods"][0]["tau_target"] == pytest.approx(0.5)


def test_generate_coreot_full_tau_range_results_cli_passes_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    def fake_write_coreot_full_tau_range_results(**kwargs: object) -> SimpleNamespace:
        calls.update(kwargs)
        return SimpleNamespace(output_root=tmp_path / "results" / "coreot_full")

    monkeypatch.setattr(
        generate_hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid,
        "write_coreot_full_tau_range_results",
        fake_write_coreot_full_tau_range_results,
    )

    exit_code = generate_hiha_dc_coreot_full_tau_min_tau_max_alpha5_grid.main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--grid-dir",
            str(tmp_path / "generated" / "coreot_full"),
            "--output-root",
            str(tmp_path / "results" / "coreot_full"),
            "--main-grid-root",
            str(tmp_path / "results" / "results_leave_one_HIHA_DC" / "tuned_baseline"),
            "--figures",
        ]
    )

    assert exit_code == 0
    assert calls["grid_dir"] == tmp_path / "generated" / "coreot_full"
    assert calls["main_grid_root"] == (
        tmp_path / "results" / "results_leave_one_HIHA_DC" / "tuned_baseline"
    )
    assert calls["write_figures"] is True
    assert capsys.readouterr().out.strip() == str(tmp_path / "results" / "coreot_full")


def _write_summary(
    path: Path,
    *,
    method: str,
    method_group: str,
    score: str | None,
    quantities: tuple[str, ...],
) -> None:
    rows = []
    for held_out_label in ("ISG+ cDC2", "overall"):
        for quantity in quantities:
            row = {
                "held_out_label": held_out_label,
                "method": method,
                "method_group": method_group,
                "quantity": quantity,
                "mean": 0.8,
                "std": 0.1,
                "sem": 0.1,
                "n_runs": 1,
            }
            if score is not None:
                row["score"] = score
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    run_id: str,
    held_out_label: str,
    seed: int,
    nn_auroc: float,
    prior_auroc: float,
) -> None:
    run_config_dir = grid_dir / run_id
    run_config_dir.mkdir(parents=True)
    (run_config_dir / "benchmark.yaml").write_text(
        yaml.safe_dump({"removed_state": held_out_label, "split": {"seed": seed}}),
        encoding="utf-8",
    )

    run_root = runs_root / run_id
    truth_root = run_root / "benchmark" / "incomplete_reference" / "evaluation_truth"
    truth_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_id": ["q_absent", "q_shared_b", "q_shared_c"],
            "true_label": [held_out_label, "B", "C"],
            "removed_state": [held_out_label, held_out_label, held_out_label],
            "is_absent_state": [True, False, False],
            "is_shared_state": [False, True, True],
        }
    ).to_csv(truth_root / "query_truth.csv", index=False)
    full_reference_truth_root = (
        run_root / "benchmark" / "full_reference_control" / "evaluation_truth"
    )
    full_reference_truth_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_id": ["q_absent", "q_shared_b", "q_shared_c"],
            "true_label": [held_out_label, "B", "C"],
            "removed_state": [held_out_label, held_out_label, held_out_label],
            "is_absent_state": [False, False, False],
            "is_shared_state": [True, True, True],
        }
    ).to_csv(full_reference_truth_root / "query_truth.csv", index=False)

    embedding_root = run_root / "embeddings" / "incomplete_reference" / "toy_embedding"
    embedding_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "cell_id": ["r_shared", "q_absent", "q_shared_b", "q_shared_c"],
            "domain": ["reference", "query", "query", "query"],
            "condition_id": ["incomplete_reference"] * 4,
        }
    ).to_csv(embedding_root / "embedding_cells.csv", index=False)
    np.save(
        embedding_root / "embedding_2d.npy",
        np.asarray([[0.0, 0.0], [1.0, 1.0], [0.8, 0.1], [0.2, 0.9]], dtype=float),
    )

    scoring_root = run_root / "scoring" / "incomplete_reference" / "toy_k1"
    scoring_root.mkdir(parents=True)
    base_scores = pd.DataFrame(
        {
            "cell_id": ["q_absent", "q_shared_b", "q_shared_c"] * 2,
            "condition_id": ["incomplete_reference"] * 6,
            "method": [
                "nn",
                "nn",
                "nn",
                "prior_only",
                "prior_only",
                "prior_only",
            ],
            "u": [0.0, 0.0, 0.0, None, None, None],
            "u_tilde": [0.0, 0.0, 0.0, None, None, None],
            "prior_risk": [0.2, 0.1, 0.1, 0.2, 0.1, 0.1],
            "e": [0.0, 0.0, 0.0, None, None, None],
            "hub_exposure": [0.0, 0.0, 0.0, None, None, None],
            "max_label_probability": [1.0, 1.0, 1.0, None, None, None],
            "label_uncertainty": [0.0, 0.0, 0.0, None, None, None],
            "label_entropy": [0.0, 0.0, 0.0, None, None, None],
            "forced_label": [held_out_label, "B", "B", "", "", ""],
            "nn_distance": [2.0, 0.1, 0.2, None, None, None],
            "abstain_u": [False, False, True, False, False, False],
            "abstain_u_or_entropy": [False, False, True, False, False, False],
            "final_label_abstention_aware": [held_out_label, "B", "", "", "", ""],
        }
    )
    coreot_scores = pd.DataFrame(
        {
            "cell_id": ["q_absent", "q_shared_b", "q_shared_c"],
            "condition_id": ["incomplete_reference"] * 3,
            "method": ["coreot_full"] * 3,
            "u": [0.7, 0.1, 0.2],
            "u_tilde": [0.6, 0.3, 0.1],
            "prior_risk": [0.2, 0.1, 0.1],
            "e": [0.0, 0.0, 0.0],
            "hub_exposure": [0.0, 0.0, 0.0],
            "max_label_probability": [0.8, 0.9, 0.9],
            "label_uncertainty": [0.2, 0.1, 0.1],
            "label_entropy": [0.4, 0.2, 0.2],
            "forced_label": ["B", "B", "C"],
            "nn_distance": [0.0, 0.0, 0.0],
            "abstain_u": [True, False, False],
            "abstain_u_or_entropy": [True, False, False],
            "final_label_abstention_aware": ["", "B", "C"],
        }
    )
    pd.concat([base_scores, coreot_scores], ignore_index=True).to_parquet(
        scoring_root / "cell_scores.parquet", index=False
    )
    full_reference_scoring_root = (
        run_root / "scoring" / "full_reference_control" / "toy_k1"
    )
    full_reference_scoring_root.mkdir(parents=True)
    pd.concat([base_scores, coreot_scores], ignore_index=True).to_parquet(
        full_reference_scoring_root / "cell_scores.parquet", index=False
    )

    evaluation_root = run_root / "evaluation"
    evaluation_root.mkdir(parents=True)
    pd.DataFrame(
        [
            *_metric_rows("nn", "nn_distance", nn_auroc),
            _abstention_row("nn", 0.0),
            *_metric_rows("prior_only", "prior_risk", prior_auroc),
            _abstention_row("prior_only", 0.0),
            *_metric_rows("coreot_full", "u", 0.75),
            *_metric_rows("coreot_full", "u_tilde", 0.80),
            _abstention_row("coreot_full", 1.0),
        ]
    ).to_csv(evaluation_root / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "condition_id": ["incomplete_reference", "incomplete_reference"],
            "candidate_set": ["toy_k1", "toy_k1"],
            "method": ["nn", "coreot_full"],
            "subset": ["absent_state", "absent_state"],
            "forced_label": ["B", "B"],
            "n_cells": [1, 1],
            "mean_max_label_probability": [1.0, 0.8],
            "abstention_rate": [0.0, 1.0],
        }
    ).to_csv(evaluation_root / "forced_label_summary.csv", index=False)


def _metric_rows(method: str, score: str, auroc: float) -> list[dict[str, object]]:
    values = {
        "auroc": auroc,
        "auprc": auroc - 0.1,
        "median_absent": 2.0,
        "median_shared": 0.5,
        "absent_minus_shared_median": 1.5,
    }
    return [
        {
            "condition_id": "incomplete_reference",
            "candidate_set": "toy_k1",
            "method": method,
            "score": score,
            "metric": metric,
            "value": value,
        }
        for metric, value in values.items()
    ]


def _abstention_row(method: str, value: float) -> dict[str, object]:
    return {
        "condition_id": "incomplete_reference",
        "candidate_set": "toy_k1",
        "method": method,
        "score": "abstain_u_or_entropy",
        "metric": "absent_abstention_rate",
        "value": value,
    }


def _write_coreot_full_tau_range_transport_config(
    config_dir: Path,
    run_id: str,
    tau_min: float,
    tau_max: float,
    alpha: float,
) -> None:
    payload = {
        "run_id": run_id,
        "conditions": ["incomplete_reference", "full_reference_control"],
        "candidate_sets": [
            {"name": "toy_k1", "provider": "toy_embedding", "prior_profile": "default"}
        ],
        "outputs": {"root": "runs/"},
        "methods": [
            {
                "name": "coreot_full",
                "epsilon": 0.05,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "tau_target": 1.0,
                "alpha": alpha,
                "max_iter": 2000,
                "tol": 1e-6,
                "numerical_floor": 1e-300,
            },
            "prior_only",
        ],
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "transport.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    (config_dir / "scoring.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "conditions": ["incomplete_reference", "full_reference_control"],
                "candidate_sets": ["toy_k1"],
                "methods": ["coreot_full", "prior_only"],
                "thresholds": {
                    "theta_u": {
                        "source_condition": "full_reference_control",
                        "source_method": "method",
                        "quantile": 0.95,
                    },
                    "theta_H": {"value": 0.8},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class TestParseCoreotFullTauRangeParams:
    def test_parses_tau_min_tau_max_and_alpha(self, tmp_path: Path) -> None:
        _write_coreot_full_tau_range_transport_config(
            tmp_path,
            "test",
            5.0,
            7.0,
            5.0,
        )
        assert parse_coreot_full_tau_range_params(tmp_path) == (5.0, 7.0, 5.0)

    def test_rejects_tau_max_below_tau_min(self, tmp_path: Path) -> None:
        _write_coreot_full_tau_range_transport_config(
            tmp_path,
            "test",
            8.0,
            7.0,
            5.0,
        )
        with pytest.raises(ResultsGridError, match="tau_max"):
            parse_coreot_full_tau_range_params(tmp_path)


# ---------------------------------------------------------------------------
# Tau sensitivity tests
# ---------------------------------------------------------------------------


def _write_tau_transport_config(config_dir: Path, run_id: str, tau: float) -> None:
    """Write a minimal transport.yaml with coreot_constant_tau at given tau."""
    payload = {
        "run_id": run_id,
        "conditions": ["incomplete_reference", "full_reference_control"],
        "candidate_sets": [
            {"name": "toy_k1", "provider": "toy_embedding", "prior_profile": "default"}
        ],
        "outputs": {"root": "runs/"},
        "methods": [
            {
                "name": "uniform_uot",
                "epsilon": 0.05,
                "tau_source": tau,
                "tau_target": tau,
                "max_iter": 2000,
                "tol": 1e-6,
                "numerical_floor": 1e-300,
            },
            {
                "name": "coreot_constant_tau",
                "epsilon": 0.05,
                "tau_source": tau,
                "tau_target": tau,
                "alpha": 1.0,
                "max_iter": 2000,
                "tol": 1e-6,
                "numerical_floor": 1e-300,
            },
            "prior_only",
        ],
        "sinkhorn": {"max_iter": 5000, "tol": 1e-7, "numerical_floor": 1e-12, "log_domain": True},
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "transport.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _write_tau_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    run_id: str,
    held_out_label: str,
    seed: int,
    tau: float,
    uot_auroc: float,
    coreot_auroc: float,
) -> None:
    """Write a minimal tau-swept run with transport config and metrics."""
    _write_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id=run_id,
        held_out_label=held_out_label,
        seed=seed,
        nn_auroc=uot_auroc,  # reuse fixture; will be filtered by methods param
        prior_auroc=uot_auroc,
    )
    # Overwrite with tau transport config
    _write_tau_transport_config(grid_dir / run_id, run_id, tau)

    # Add uniform_uot and coreot_constant_tau metric rows
    metrics_path = runs_root / run_id / "evaluation" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    new_rows = []
    for method, auroc in [("uniform_uot", uot_auroc), ("coreot_constant_tau", coreot_auroc)]:
        new_rows.extend(_metric_rows(method, "u", auroc))
        new_rows.append(_abstention_row(method, 0.0))
    extended = pd.concat([metrics, pd.DataFrame(new_rows)], ignore_index=True)
    extended.to_csv(metrics_path, index=False)

    # Add scoring data for swept methods
    for condition in ("incomplete_reference", "full_reference_control"):
        scoring_root = runs_root / run_id / "scoring" / condition / "toy_k1"
        scoring_root.mkdir(parents=True, exist_ok=True)
        score_rows = []
        for method in ("uniform_uot", "coreot_constant_tau", "prior_only"):
            score_rows.append(
                {
                    "cell_id": "q_absent",
                    "condition_id": condition,
                    "method": method,
                    "u": 0.5 if method != "prior_only" else None,
                    "u_tilde": 0.5 if method != "prior_only" else None,
                    "prior_risk": 0.2,
                    "e": 0.0,
                    "hub_exposure": 0.0,
                    "max_label_probability": 0.9,
                    "label_uncertainty": 0.1,
                    "label_entropy": 0.2,
                    "forced_label": "B" if method != "prior_only" else "",
                    "nn_distance": 0.0,
                    "abstain_u": False,
                    "abstain_u_or_entropy": False,
                    "final_label_abstention_aware": "B" if method != "prior_only" else "",
                }
            )
            score_rows.append(
                {
                    "cell_id": "q_shared_b",
                    "condition_id": condition,
                    "method": method,
                    "u": 0.1 if method != "prior_only" else None,
                    "u_tilde": 0.1 if method != "prior_only" else None,
                    "prior_risk": 0.1,
                    "e": 0.0,
                    "hub_exposure": 0.0,
                    "max_label_probability": 1.0,
                    "label_uncertainty": 0.0,
                    "label_entropy": 0.0,
                    "forced_label": "B",
                    "nn_distance": 0.0,
                    "abstain_u": False,
                    "abstain_u_or_entropy": False,
                    "final_label_abstention_aware": "B",
                }
            )
        pd.DataFrame(score_rows).to_parquet(scoring_root / "cell_scores.parquet", index=False)


def _write_labelwise_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    run_id: str,
    held_out_label: str,
    seed: int,
    tau: float,
    alpha: float,
    coreot_auroc: float,
) -> None:
    """Write a minimal CoRe-OT tau/alpha run for broad-grid tests."""
    _write_tau_run(
        runs_root=runs_root,
        grid_dir=grid_dir,
        run_id=run_id,
        held_out_label=held_out_label,
        seed=seed,
        tau=tau,
        uot_auroc=coreot_auroc,
        coreot_auroc=coreot_auroc,
    )
    transport_path = grid_dir / run_id / "transport.yaml"
    transport = yaml.safe_load(transport_path.read_text(encoding="utf-8"))
    coreot_method = next(
        method
        for method in transport["methods"]
        if isinstance(method, dict) and method.get("name") == "coreot_constant_tau"
    )
    coreot_method["alpha"] = alpha
    transport["methods"] = [coreot_method, "prior_only"]
    transport_path.write_text(
        yaml.safe_dump(transport, sort_keys=False), encoding="utf-8"
    )


class TestFilterCompletedRuns:
    def test_splits_completed_and_failed(self, tmp_path: Path) -> None:
        (tmp_path / "complete" / "evaluation").mkdir(parents=True)
        (tmp_path / "complete" / "evaluation" / "metrics.csv").write_text("x")
        desc_complete = RunDescriptor("complete", "ISG+ cDC2", 1)
        desc_failed = RunDescriptor("failed", "CD14+ cDC2", 2)
        completed, failed = _filter_completed_runs(
            tmp_path, (desc_complete, desc_failed)
        )
        assert len(completed) == 1
        assert completed[0].run_id == "complete"
        assert failed == ["failed"]


class TestReadFixedReferenceRows:
    def test_filters_methods_and_labels(self, tmp_path: Path) -> None:
        tables = tmp_path / "tables"
        tables.mkdir(parents=True)
        det = pd.DataFrame(
            {
                "held_out_label": ["CD14+ cDC2", "ISG+ cDC2", "ASDC", "overall"],
                "method": ["prior_only", "nn", "coreot_full", "balanced_ot"],
                "method_group": ["prior", "baseline", "coreot", "baseline"],
                "primary_score": ["prior_risk", "nn_distance", "u", "label_uncertainty"],
                "quantity": ["auroc"] * 4,
                "mean": [0.6, 0.8, 0.9, 0.75],
                "std": [0.1, 0.1, 0.1, 0.1],
                "sem": [0.05, 0.05, 0.05, 0.05],
                "n_runs": [5, 5, 5, 5],
            }
        )
        shared = pd.DataFrame(
            {
                "held_out_label": ["CD14+ cDC2", "ISG+ cDC2"],
                "method": ["prior_only", "nn"],
                "method_group": ["prior", "baseline"],
                "quantity": ["forced_accuracy"] * 2,
                "mean": [0.5, 0.6],
                "std": [0.1, 0.1],
                "sem": [0.05, 0.05],
                "n_runs": [5, 5],
            }
        )
        det.to_csv(tables / "main_detection_summary.csv", index=False)
        shared.to_csv(tables / "shared_label_transfer_summary.csv", index=False)

        det_out, shared_out = read_fixed_reference_rows(
            detection_summary_path=tables / "main_detection_summary.csv",
            shared_summary_path=tables / "shared_label_transfer_summary.csv",
            held_out_labels=("CD14+ cDC2", "ISG+ cDC2"),
        )
        # Only fixed reference methods, only specified labels
        assert set(det_out["method"]) <= set(FIXED_REFERENCE_METHODS)
        assert set(det_out["held_out_label"]) <= {"CD14+ cDC2", "ISG+ cDC2", "overall"}
        assert set(shared_out["method"]) <= set(FIXED_REFERENCE_METHODS)
        assert all(pd.isna(det_out["tau"]))
        assert set(det_out["sensitivity_role"]) == {"reference"}
        # ASDC and coreot_full should be excluded
        assert "ASDC" not in set(det_out["held_out_label"])

    def test_raises_on_missing_files(self, tmp_path: Path) -> None:
        with pytest.raises(
            FileNotFoundError, match="generate_hiha_dc_report_leave_one_results"
        ):
            read_fixed_reference_rows(
                detection_summary_path=tmp_path / "nonexistent.csv",
                shared_summary_path=tmp_path / "also_nonexistent.csv",
                held_out_labels=("ISG+ cDC2",),
            )


class TestWriteBroadGridResults:
    def test_integration_writes_combined_report_and_heatmap_pdf(
        self, tmp_path: Path
    ) -> None:
        runs_root = tmp_path / "runs"
        coreot_grid_dir = tmp_path / "generated_configs" / "coreot"
        uniform_grid_dir = tmp_path / "generated_configs" / "uniform"
        output_root = tmp_path / "results" / "broad"

        _write_labelwise_run(
            runs_root=runs_root,
            grid_dir=coreot_grid_dir,
            run_id="coreot_tau1_alpha2_seed1",
            held_out_label="HLA-DRhi cDC2",
            seed=1,
            tau=1.0,
            alpha=2.0,
            coreot_auroc=0.86,
        )
        _write_labelwise_run(
            runs_root=runs_root,
            grid_dir=coreot_grid_dir,
            run_id="coreot_tau2_alpha3_seed1",
            held_out_label="HLA-DRhi cDC2",
            seed=1,
            tau=2.0,
            alpha=3.0,
            coreot_auroc=0.88,
        )
        _write_tau_run(
            runs_root=runs_root,
            grid_dir=uniform_grid_dir,
            run_id="uniform_tau1_seed1",
            held_out_label="ISG+ cDC2",
            seed=1,
            tau=1.0,
            uot_auroc=0.80,
            coreot_auroc=0.80,
        )
        _write_tau_run(
            runs_root=runs_root,
            grid_dir=uniform_grid_dir,
            run_id="uniform_tau2_seed1",
            held_out_label="ISG+ cDC2",
            seed=1,
            tau=2.0,
            uot_auroc=0.82,
            coreot_auroc=0.82,
        )

        tuned_baseline = tmp_path / "results" / "results_leave_one_HIHA_DC" / "tuned_baseline"
        tuned_tables = tuned_baseline / "tables"
        tuned_tables.mkdir(parents=True)
        pd.DataFrame(
            {
                "held_out_label": ["ISG+ cDC2", "ISG+ cDC2", "overall"],
                "method": ["prior_only", "nn", "balanced_ot"],
                "method_group": ["prior", "baseline", "baseline"],
                "primary_score": ["prior_risk", "nn_distance", "label_uncertainty"],
                "quantity": ["auroc", "auroc", "auroc"],
                "mean": [0.60, 0.70, 0.75],
                "std": [0.03, 0.02, 0.02],
                "sem": [0.01, 0.01, 0.01],
                "n_runs": [5, 5, 5],
            }
        ).to_csv(tuned_tables / "main_detection_summary.csv", index=False)
        pd.DataFrame(
            {
                "held_out_label": ["ISG+ cDC2", "overall"],
                "method": ["nn", "balanced_ot"],
                "method_group": ["baseline", "baseline"],
                "quantity": ["forced_accuracy", "forced_accuracy"],
                "mean": [0.65, 0.72],
                "std": [0.04, 0.03],
                "sem": [0.02, 0.01],
                "n_runs": [5, 5],
            }
        ).to_csv(tuned_tables / "shared_label_transfer_summary.csv", index=False)

        paths = write_broad_grid_results(
            runs_root=runs_root,
            coreot_grid_dir=coreot_grid_dir,
            uniform_grid_dir=uniform_grid_dir,
            output_root=output_root,
            main_grid_root=tuned_baseline,
            candidate_set="toy_k1",
            write_figures=True,
        )

        assert paths.detection_by_run.is_file()
        assert paths.detection_summary.is_file()
        assert paths.shared_label_transfer_by_run.is_file()
        assert paths.shared_label_transfer_summary.is_file()
        assert paths.pareto_table.is_file()
        assert paths.table_1.is_file()
        assert paths.table_2.is_file()
        assert paths.report.is_file()
        assert paths.manifest.is_file()
        assert paths.figures["heatmaps_pdf"].is_file()

        detection_summary = pd.read_csv(paths.detection_summary)
        assert set(detection_summary["method"]) == {"uniform_uot", "coreot_constant_tau"}
        assert set(detection_summary.loc[detection_summary["method"] == "uniform_uot", "alpha"]) == {0.0}
        assert set(
            detection_summary.loc[detection_summary["method"] == "coreot_constant_tau", "alpha"]
        ) == {2.0, 3.0}

        report = paths.report.read_text(encoding="utf-8")
        assert "Broad τ/α Tuning Report" in report
        assert "Uniform UOT α=0 Rows" in report
        assert "Fixed reference source" in report
        assert str(tuned_baseline) in report
        assert "heatmaps.pdf" in report
        assert "## Main Table 1" in report
        assert "## Main Table 2" in report

        manifest = read_manifest(paths.manifest)
        assert manifest.metadata["fixed_reference_root"] == str(tuned_baseline)
        assert manifest.metadata["n_coreot_runs_expected"] == 2
        assert manifest.metadata["n_uniform_runs_expected"] == 2


class TestWriteCoreotFullTauRangeResults:
    def test_integration_writes_report_tables_manifest_and_heatmaps(
        self,
        tmp_path: Path,
    ) -> None:
        runs_root = tmp_path / "runs"
        grid_dir = tmp_path / "generated_configs" / "coreot_full"
        output_root = tmp_path / "results" / "coreot_full"

        for run_id, tau_min, tau_max in (
            ("coreot_full_taumin5_taumax7_seed1", 5.0, 7.0),
            ("coreot_full_taumin6_taumax8_seed1", 6.0, 8.0),
        ):
            _write_run(
                runs_root=runs_root,
                grid_dir=grid_dir,
                run_id=run_id,
                held_out_label="ISG+ cDC2",
                seed=1,
                nn_auroc=0.70,
                prior_auroc=0.60,
            )
            _write_coreot_full_tau_range_transport_config(
                grid_dir / run_id,
                run_id,
                tau_min,
                tau_max,
                5.0,
            )

        tuned_baseline = tmp_path / "results" / "results_leave_one_HIHA_DC" / "tuned_baseline"
        tuned_tables = tuned_baseline / "tables"
        tuned_tables.mkdir(parents=True)
        pd.DataFrame(
            {
                "held_out_label": ["ISG+ cDC2", "ISG+ cDC2", "overall"],
                "method": ["prior_only", "nn", "balanced_ot"],
                "method_group": ["prior", "baseline", "baseline"],
                "primary_score": ["prior_risk", "nn_distance", "label_uncertainty"],
                "quantity": ["auroc", "auroc", "auroc"],
                "mean": [0.60, 0.70, 0.75],
                "std": [0.03, 0.02, 0.02],
                "sem": [0.01, 0.01, 0.01],
                "n_runs": [5, 5, 5],
            }
        ).to_csv(tuned_tables / "main_detection_summary.csv", index=False)
        pd.DataFrame(
            {
                "held_out_label": ["ISG+ cDC2", "overall"],
                "method": ["nn", "balanced_ot"],
                "method_group": ["baseline", "baseline"],
                "quantity": ["forced_accuracy", "forced_accuracy"],
                "mean": [0.65, 0.72],
                "std": [0.04, 0.03],
                "sem": [0.02, 0.01],
                "n_runs": [5, 5],
            }
        ).to_csv(tuned_tables / "shared_label_transfer_summary.csv", index=False)

        paths = write_coreot_full_tau_range_results(
            runs_root=runs_root,
            grid_dir=grid_dir,
            output_root=output_root,
            main_grid_root=tuned_baseline,
            candidate_set="toy_k1",
            write_figures=True,
        )

        assert paths.detection_by_run.is_file()
        assert paths.detection_summary.is_file()
        assert paths.u_vs_u_tilde_detection_by_run.is_file()
        assert paths.u_vs_u_tilde_detection_summary.is_file()
        assert paths.shared_label_transfer_by_run.is_file()
        assert paths.shared_label_transfer_summary.is_file()
        assert paths.u_tilde_counterfactual_shared_label_transfer_by_run.is_file()
        assert paths.u_tilde_counterfactual_shared_label_transfer_summary.is_file()
        assert paths.table_1.is_file()
        assert paths.table_2.is_file()
        assert paths.table_3.is_file()
        assert paths.report.is_file()
        assert paths.manifest.is_file()
        assert paths.figures["heatmaps_pdf"].is_file()
        assert paths.figures["u_tilde_heatmaps_pdf"].is_file()

        detection_summary = pd.read_csv(paths.detection_summary)
        assert set(detection_summary["method"]) == {"coreot_full"}
        assert set(detection_summary["tau_min"]) == {5.0, 6.0}
        assert set(detection_summary["tau_max"]) == {7.0, 8.0}
        assert set(detection_summary["alpha"]) == {5.0}
        u_vs_u_tilde_summary = pd.read_csv(paths.u_vs_u_tilde_detection_summary)
        assert set(u_vs_u_tilde_summary["score"]) == {"u", "u_tilde"}
        assert set(u_vs_u_tilde_summary["score_role"]) == {
            "primary",
            "secondary_counterfactual",
        }
        u_tilde_shared = pd.read_csv(
            paths.u_tilde_counterfactual_shared_label_transfer_summary
        )
        assert set(u_tilde_shared["score"]) == {"u_tilde"}

        report = paths.report.read_text(encoding="utf-8")
        assert "coreot_full τ_min/τ_max" in report
        assert "Fixed reference source" in report
        assert str(tuned_baseline) in report
        assert "heatmaps.pdf" in report
        assert "u_tilde_heatmaps.pdf" in report
        assert "## Main Table 1" in report
        assert "## Main Table 2" in report
        assert "## Table 3" in report

        manifest = read_manifest(paths.manifest)
        assert manifest.metadata["method"] == "coreot_full"
        assert manifest.metadata["fixed_reference_root"] == str(tuned_baseline)
        assert manifest.metadata["n_runs_expected"] == 2
        assert manifest.metadata["n_runs_completed"] == 2
        assert (
            manifest.metadata["u_tilde_policy"]
            == "secondary_counterfactual_full_reference_quantile_with_entropy"
        )
        assert (
            manifest.artifacts["u_vs_u_tilde_detection_summary"]
            == str(paths.u_vs_u_tilde_detection_summary)
        )


# ---------------------------------------------------------------------------
# Labelwise tuned baseline tests
# ---------------------------------------------------------------------------

SELECTED_POINTS = (
    ("HLA-DRhi cDC2", "uniform_uot", 2.0, 0.0),
    ("ISG+ cDC2", "uniform_uot", 3.0, 0.0),
    ("HLA-DRhi cDC2", "coreot_constant_tau", 2.0, 2.0),
    ("ISG+ cDC2", "coreot_constant_tau", 3.0, 1.0),
)


def _write_broad_grid_source_tables(source_root: Path) -> None:
    """Write synthetic broad-grid CSVs with the 4 selected points plus distractors."""
    tables = source_root / "tables"
    tables.mkdir(parents=True)

    # --- detection_by_run.csv ---
    by_run_rows: list[dict[str, object]] = []
    for held_out_label, method, tau, alpha in SELECTED_POINTS:
        for seed in range(1, 6):
            run_id = f"run_{held_out_label.replace(' ', '_')}_{method}_tau{tau}_alpha{alpha}_seed{seed}"
            by_run_rows.append({
                "run_id": run_id,
                "held_out_label": held_out_label,
                "seed": seed,
                "condition_id": "incomplete_reference",
                "candidate_set": "hiha_harmony30_k100",
                "method": method,
                "method_group": "baseline" if method == "uniform_uot" else "ablation",
                "primary_score": "u",
                "auroc": 0.85 + seed * 0.01,
                "auprc": 0.55 + seed * 0.01,
                "auprc_baseline": 0.25,
                "absent_abstention_rate": 0.3 + seed * 0.02,
                "shared_false_abstention_rate": 0.05 + seed * 0.005,
                "median_absent": 0.3,
                "median_shared": 0.1,
                "absent_minus_shared_median": 0.2,
                "tau": tau,
                "sensitivity_role": "swept",
                "alpha": alpha,
            })

    # Add distractors: CD14+ cDC2 row (should be excluded)
    for seed in range(1, 6):
        by_run_rows.append({
            "run_id": f"run_CD14_cDC2_uniform_uot_tau2_alpha0_seed{seed}",
            "held_out_label": "CD14+ cDC2",
            "seed": seed,
            "condition_id": "incomplete_reference",
            "candidate_set": "hiha_harmony30_k100",
            "method": "uniform_uot",
            "method_group": "baseline",
            "primary_score": "u",
            "auroc": 0.75,
            "auprc": 0.45,
            "auprc_baseline": 0.25,
            "absent_abstention_rate": 0.35,
            "shared_false_abstention_rate": 0.06,
            "median_absent": 0.28,
            "median_shared": 0.12,
            "absent_minus_shared_median": 0.16,
            "tau": 2.0,
            "sensitivity_role": "swept",
            "alpha": 0.0,
        })
    # Add non-selected tau row for ISG+ cDC2 (tau=1.0, should be excluded)
    by_run_rows.append({
        "run_id": "run_ISG_cDC2_uniform_uot_tau1_alpha0_seed1",
        "held_out_label": "ISG+ cDC2",
        "seed": 1,
        "condition_id": "incomplete_reference",
        "candidate_set": "hiha_harmony30_k100",
        "method": "uniform_uot",
        "method_group": "baseline",
        "primary_score": "u",
        "auroc": 0.80,
        "auprc": 0.50,
        "auprc_baseline": 0.25,
        "absent_abstention_rate": 0.40,
        "shared_false_abstention_rate": 0.07,
        "median_absent": 0.25,
        "median_shared": 0.10,
        "absent_minus_shared_median": 0.15,
        "tau": 1.0,
        "sensitivity_role": "swept",
        "alpha": 0.0,
    })

    pd.DataFrame(by_run_rows).to_csv(tables / "detection_by_run.csv", index=False)

    # --- detection_summary.csv ---
    summary_rows: list[dict[str, object]] = []
    quantities = ["auroc", "auprc", "auprc_baseline", "absent_abstention_rate", "shared_false_abstention_rate"]
    for held_out_label, method, tau, alpha in SELECTED_POINTS:
        for qty in quantities:
            summary_rows.append({
                "held_out_label": held_out_label,
                "method": method,
                "method_group": "baseline" if method == "uniform_uot" else "ablation",
                "primary_score": "u",
                "tau": tau,
                "alpha": alpha,
                "quantity": qty,
                "mean": 0.85,
                "std": 0.03,
                "sem": 0.013,
                "n_runs": 5,
                "sensitivity_role": "swept",
            })
    # CD14+ cDC2 distractor in summary
    for qty in quantities:
        summary_rows.append({
            "held_out_label": "CD14+ cDC2",
            "method": "uniform_uot",
            "method_group": "baseline",
            "primary_score": "u",
            "tau": 2.0,
            "alpha": 0.0,
            "quantity": qty,
            "mean": 0.75,
            "std": 0.02,
            "sem": 0.009,
            "n_runs": 5,
            "sensitivity_role": "swept",
        })

    pd.DataFrame(summary_rows).to_csv(tables / "detection_summary.csv", index=False)

    # --- shared_label_transfer_by_run.csv ---
    shared_by_run_rows: list[dict[str, object]] = []
    for held_out_label, method, tau, alpha in SELECTED_POINTS:
        for seed in range(1, 6):
            run_id = f"run_{held_out_label.replace(' ', '_')}_{method}_tau{tau}_alpha{alpha}_seed{seed}"
            shared_by_run_rows.append({
                "run_id": run_id,
                "held_out_label": held_out_label,
                "seed": seed,
                "condition_id": "incomplete_reference",
                "candidate_set": "hiha_harmony30_k100",
                "method": method,
                "method_group": "baseline" if method == "uniform_uot" else "ablation",
                "n_shared": 100,
                "n_forced_labeled": 100,
                "n_non_abstained": 85 + seed,
                "forced_accuracy": 0.90 + seed * 0.005,
                "forced_macro_f1": 0.85 + seed * 0.005,
                "post_abstention_accuracy": 0.92 + seed * 0.005,
                "post_abstention_macro_f1": 0.88 + seed * 0.005,
                "coverage": 0.85 + seed * 0.01,
                "shared_false_abstention_rate": 0.05 + seed * 0.005,
                "tau": tau,
                "sensitivity_role": "swept",
                "alpha": alpha,
            })
    # CD14+ cDC2 distractor in shared by-run
    for seed in range(1, 6):
        shared_by_run_rows.append({
            "run_id": f"run_CD14_cDC2_uniform_uot_tau2_alpha0_seed{seed}",
            "held_out_label": "CD14+ cDC2",
            "seed": seed,
            "condition_id": "incomplete_reference",
            "candidate_set": "hiha_harmony30_k100",
            "method": "uniform_uot",
            "method_group": "baseline",
            "n_shared": 100,
            "n_forced_labeled": 100,
            "n_non_abstained": 80,
            "forced_accuracy": 0.88,
            "forced_macro_f1": 0.83,
            "post_abstention_accuracy": 0.90,
            "post_abstention_macro_f1": 0.85,
            "coverage": 0.80,
            "shared_false_abstention_rate": 0.06,
            "tau": 2.0,
            "sensitivity_role": "swept",
            "alpha": 0.0,
        })

    pd.DataFrame(shared_by_run_rows).to_csv(tables / "shared_label_transfer_by_run.csv", index=False)

    # --- shared_label_transfer_summary.csv ---
    shared_quantities = [
        "forced_accuracy", "forced_macro_f1", "post_abstention_accuracy",
        "post_abstention_macro_f1", "coverage", "shared_false_abstention_rate",
    ]
    shared_summary_rows: list[dict[str, object]] = []
    for held_out_label, method, tau, alpha in SELECTED_POINTS:
        for qty in shared_quantities:
            shared_summary_rows.append({
                "held_out_label": held_out_label,
                "method": method,
                "method_group": "baseline" if method == "uniform_uot" else "ablation",
                "tau": tau,
                "alpha": alpha,
                "quantity": qty,
                "mean": 0.88,
                "std": 0.02,
                "sem": 0.009,
                "n_runs": 5,
                "sensitivity_role": "swept",
            })
    # CD14+ cDC2 distractor in shared summary
    for qty in shared_quantities:
        shared_summary_rows.append({
            "held_out_label": "CD14+ cDC2",
            "method": "uniform_uot",
            "method_group": "baseline",
            "tau": 2.0,
            "alpha": 0.0,
            "quantity": qty,
            "mean": 0.85,
            "std": 0.015,
            "sem": 0.007,
            "n_runs": 5,
            "sensitivity_role": "swept",
        })

    pd.DataFrame(shared_summary_rows).to_csv(tables / "shared_label_transfer_summary.csv", index=False)


def _write_valid_source_manifest(source_root: Path) -> None:
    """Write a source manifest that passes validation checks."""
    manifest = {
        "stage": "broad-grid-full",
        "artifacts": {},
        "metadata": {
            "uniform_tau_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "coreot_tau_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "coreot_alpha_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "n_uniform_runs_completed": 90,
            "n_uniform_runs_expected": 90,
            "n_coreot_runs_completed": 900,
            "n_coreot_runs_expected": 900,
            "failed_runs": [],
        },
    }
    (source_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
