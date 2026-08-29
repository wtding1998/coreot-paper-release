from __future__ import annotations

import hashlib

import pandas as pd
import pytest
import yaml

from coreot.submission.celltypist_promotion import CellTypistCandidateRootError
from experiments.pbmc_state import rerun_pbmc_celltypist_centered as candidate


def _write_repair_manifest(
    tmp_path,
    *,
    run_id: str = "run1",
    condition: str = "incomplete_reference",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    counts = tmp_path / "counts.h5ad"
    counts.write_bytes(b"repaired-counts")
    manifest = tmp_path / "derivation.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "stage": "pbmc-celltypist-model-visible-input-repair",
                "status": "verified",
                "run_id": run_id,
                "condition": condition,
                "derivation": {"output": str(counts)},
                "sha256": {
                    "repaired_counts": hashlib.sha256(counts.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, counts


def test_verified_input_repair_is_explicit_and_checksummed(tmp_path) -> None:
    manifest, counts = _write_repair_manifest(tmp_path)

    repair = candidate._load_input_repair(manifest)

    assert repair.run_id == "run1"
    assert repair.condition == "incomplete_reference"
    assert repair.counts_path == counts.resolve()
    assert repair.sha256 == hashlib.sha256(counts.read_bytes()).hexdigest()

    counts.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        candidate._load_input_repair(manifest)


def test_input_repair_covers_only_a_missing_canonical_matrix(tmp_path) -> None:
    source_runs = tmp_path / "runs"
    run_root = source_runs / "run1"
    missing = run_root / "benchmark/incomplete_reference/model_visible/counts.h5ad"
    present = run_root / "benchmark/full_reference_control/model_visible/counts.h5ad"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"present")
    manifest, _ = _write_repair_manifest(tmp_path / "repair")
    repair = candidate._load_input_repair(manifest)
    configs = [
        (
            tmp_path / "external_baselines.yaml",
            {"run_id": "run1", "outputs": {"root": str(source_runs)}},
        )
    ]

    candidate._validate_count_input_coverage(configs, {repair.key: repair})

    with pytest.raises(FileNotFoundError, match="no declared repair"):
        candidate._validate_count_input_coverage(configs, {})

    missing.parent.mkdir(parents=True, exist_ok=True)
    missing.write_bytes(b"canonical")
    with pytest.raises(ValueError, match="already exists"):
        candidate._validate_count_input_coverage(configs, {repair.key: repair})


def test_repaired_run_view_is_ephemeral_and_preserves_canonical_source(tmp_path) -> None:
    source_run = tmp_path / "runs/run1"
    visible = source_run / "benchmark/incomplete_reference/model_visible"
    visible.mkdir(parents=True)
    (visible / "cells.csv").write_text("cell_id\ncell1\n", encoding="utf-8")
    (visible / "target_labels.csv").write_text(
        "cell_id,target_label,broad_label\ncell1,B cells,Lymphoid\n",
        encoding="utf-8",
    )
    manifest, counts = _write_repair_manifest(tmp_path / "repair")
    repair = candidate._load_input_repair(manifest)

    with candidate._repaired_run_view(
        source_run=source_run,
        condition="incomplete_reference",
        repair=repair,
    ) as staged_run:
        staged_counts = (
            staged_run
            / "benchmark/incomplete_reference/model_visible/counts.h5ad"
        )
        assert staged_run != source_run
        assert staged_counts.read_bytes() == counts.read_bytes()
        assert staged_counts.is_symlink()
        staged_root = staged_run.parent

    assert not staged_root.exists()
    assert not (visible / "counts.h5ad").exists()


def test_current_authority_comparison_is_fail_closed() -> None:
    rerun = pd.DataFrame(
        {
            "held_out_label": ["B cells", "NK cells"],
            "seed": [1, 1],
            "method": [candidate.METHOD, candidate.METHOD],
            "score": ["u", "u"],
            "metric": [0.1, 0.2],
        }
    )
    authority = rerun.copy()
    authority.loc[1, "metric"] += 2e-16

    report = candidate._compare_authority_frame(
        table="S-test",
        authority_name="promotion",
        rerun=rerun,
        authority=authority,
        keys=["held_out_label", "seed", "method"],
        columns={
            "score": "score",
            "metric": "metric",
        },
    )

    assert report["status"].eq("pass").all()
    assert report.loc[report["metric"].eq("metric"), "max_abs_difference"].iloc[0] < 1e-15

    authority.loc[1, "metric"] = 0.21
    with pytest.raises(ValueError, match="current authority mismatch"):
        candidate._compare_authority_frame(
            table="S-test",
            authority_name="promotion",
            rerun=rerun,
            authority=authority,
            keys=["held_out_label", "seed", "method"],
            columns={"score": "score", "metric": "metric"},
        )


def test_s14_operational_abstention_uses_held_out_and_represented_cohorts() -> None:
    transfer = pd.DataFrame(
        {
            "held_out_label": ["B cells", "B cells"],
            "seed": [1, 2],
            "run_id": ["run1", "run2"],
            "candidate_set": [candidate.CANDIDATE_SET] * 2,
            "method": [candidate.METHOD] * 2,
            "score": ["u", "u"],
            "display_name": ["CellTypist", "CellTypist"],
            "coverage": [0.8, 1.0],
            "post_abstention_macro_f1": [0.6, 0.8],
        }
    )
    detection = pd.DataFrame(
        {
            "held_out_label": ["B cells", "B cells"],
            "seed": [1, 2],
            "method": [candidate.METHOD] * 2,
            "absent_abstention_rate": [0.25, 0.75],
        }
    )

    by_seed = candidate._operational_abstention_by_seed(transfer, detection)
    summary = candidate._summarize_operational_abstention(by_seed)

    assert by_seed["held_out_abstention"].tolist() == [0.25, 0.75]
    assert by_seed["represented_state_coverage"].tolist() == [0.8, 1.0]
    row = summary.iloc[0]
    assert row["held_out_abstention_mean"] == 0.5
    assert row["coverage_mean"] == 0.9
    assert row["post_abstention_macro_f1_mean"] == 0.7
    assert row["n_splits"] == 2


def test_s14_operational_abstention_adds_constant_presentation_columns() -> None:
    transfer = pd.DataFrame(
        {
            "held_out_label": ["B cells"],
            "seed": [1],
            "run_id": ["run1"],
            "candidate_set": [candidate.CANDIDATE_SET],
            "method": [candidate.METHOD],
            "coverage": [0.8],
            "post_abstention_macro_f1": [0.6],
        }
    )
    detection = pd.DataFrame(
        {
            "held_out_label": ["B cells"],
            "seed": [1],
            "method": [candidate.METHOD],
            "absent_abstention_rate": [0.25],
        }
    )

    by_seed = candidate._operational_abstention_by_seed(transfer, detection)

    assert by_seed["score"].tolist() == ["u"]
    assert by_seed["display_name"].tolist() == ["CellTypist"]


def test_s14_operational_abstention_rejects_incomplete_seed_pairing() -> None:
    transfer = pd.DataFrame(
        {
            "held_out_label": ["B cells"],
            "seed": [1],
            "run_id": ["run1"],
            "candidate_set": [candidate.CANDIDATE_SET],
            "method": [candidate.METHOD],
            "score": ["u"],
            "display_name": ["CellTypist"],
            "coverage": [1.0],
            "post_abstention_macro_f1": [0.8],
        }
    )
    detection = pd.DataFrame(
        {
            "held_out_label": ["B cells", "B cells"],
            "seed": [1, 2],
            "method": [candidate.METHOD, candidate.METHOD],
            "absent_abstention_rate": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="endpoint-seed keys do not match"):
        candidate._operational_abstention_by_seed(transfer, detection)


def test_comparison_rejects_incomplete_or_duplicate_stable_keys(tmp_path) -> None:
    retained = pd.DataFrame(
        {
            "held_out_label": ["B cells", "NK cells"],
            "method": [candidate.METHOD, candidate.METHOD],
            "score": ["u", "u"],
            "display_name": ["CellTypist", "CellTypist"],
            "metric_mean": [0.1, 0.2],
        }
    )
    retained_path = tmp_path / "retained.csv"
    retained.to_csv(retained_path, index=False)
    rerun = retained.iloc[:1].copy()

    with pytest.raises(ValueError, match="comparison keys differ"):
        candidate._comparison_rows(
            table="S14",
            rerun=rerun,
            retained_path=retained_path,
            keys=["held_out_label", "method", "score", "display_name"],
        )

    duplicated = pd.concat([retained, retained.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="comparison keys must be unique"):
        candidate._comparison_rows(
            table="S14",
            rerun=duplicated,
            retained_path=retained_path,
            keys=["held_out_label", "method", "score", "display_name"],
        )


def test_fresh_root_refusal_preserves_existing_content(tmp_path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(CellTypistCandidateRootError):
        candidate._ensure_fresh_output_root(root)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
