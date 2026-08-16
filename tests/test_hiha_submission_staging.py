from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.missing_celltype.build_hiha_submission_staging import PackageBuilder


def test_method_filter_records_report_uniform_exclusion(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "package"
    source = repository / "runs/example/scoring/cell_scores.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_id": ["a", "a", "b", "b"],
            "method": ["coreot_full", "uniform_uot"] * 2,
            "u": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_parquet(source, index=False)

    builder = PackageBuilder(repository_root=repository, output_root=output)
    builder.filter_methods(
        Path("runs/example/scoring/cell_scores.parquet"),
        Path("data/runs/example/scoring/cell_scores.parquet"),
        selector=lambda methods: methods.ne("uniform_uot"),
        selector_name="method != uniform_uot",
        note="test",
    )

    staged = pd.read_parquet(output / "data/runs/example/scoring/cell_scores.parquet")
    assert set(staged["method"]) == {"coreot_full"}
    assert builder.transformations[0].source_rows == 4
    assert builder.transformations[0].package_rows == 2
    assert builder.transformations[0].removed_rows == 2
    assert builder.included[0].source_sha256 != builder.included[0].package_sha256


def test_method_filter_retains_only_replacement_uniform(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "package"
    source = repository / "runs/example/evaluation/metrics.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "method": ["uniform_uot", "prior_only"],
            "metric": ["auprc", "auprc"],
            "value": [0.8, 0.4],
        }
    ).to_csv(source, index=False)

    builder = PackageBuilder(repository_root=repository, output_root=output)
    builder.filter_methods(
        Path("runs/example/evaluation/metrics.csv"),
        Path("data/runs/example/evaluation/metrics.csv"),
        selector=lambda methods: methods.eq("uniform_uot"),
        selector_name="method == uniform_uot",
        note="test",
    )

    staged = pd.read_csv(output / "data/runs/example/evaluation/metrics.csv")
    assert staged["method"].tolist() == ["uniform_uot"]
    assert builder.transformations[0].package_methods == "uniform_uot"


def test_builder_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "package"
    output.mkdir()
    (output / "existing.txt").write_text("preserve", encoding="utf-8")
    builder = PackageBuilder(repository_root=tmp_path, output_root=output)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.build()

    assert (output / "existing.txt").read_text(encoding="utf-8") == "preserve"
