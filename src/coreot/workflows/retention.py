from __future__ import annotations

import shutil
from pathlib import Path


def prune_intermediate_artifacts(run_root: str | Path, completed_stage: str) -> tuple[Path, ...]:
    """Delete large stage artifacts after their last downstream consumer."""

    root = Path(run_root)
    removed: list[Path] = []

    if completed_stage == "benchmark-build":
        removed.extend(_remove_dirs(root / "raw"))
    elif completed_stage == "transport":
        removed.extend(
            _remove_dirs(
                root / "derived",
                root / "embeddings",
                root / "candidates",
            )
        )
        benchmark_root = root / "benchmark"
        if benchmark_root.is_dir():
            for condition_root in benchmark_root.iterdir():
                if condition_root.is_dir():
                    removed.extend(_remove_dirs(condition_root / "model_visible"))

    return tuple(removed)


def prune_results_only_artifacts(run_root: str | Path) -> tuple[Path, ...]:
    """Delete run artifacts that are not needed for broad-grid result generation.

    Preserves only the files read by the HIHA-DC and PBMC broad-grid result writers,
    together with calibrated provider-reliability provenance:
    - `benchmark/split_manifest.csv`
    - `benchmark/<condition>/evaluation_truth/query_truth.csv`
    - `evaluation/metrics.csv`
    - `scoring/<condition>/<candidate_set>/cell_scores.parquet`
    - `transport/provider_reliability/`
    """

    root = Path(run_root)
    removed: list[Path] = []

    for path in (
        root / "raw",
        root / "derived",
        root / "embeddings",
        root / "candidates",
        root / "reports",
    ):
        removed.extend(_remove_dirs(path))

    removed.extend(
        _remove_children_except(
            root / "transport",
            keep_names={"provider_reliability"},
        )
    )

    benchmark_root = root / "benchmark"
    if benchmark_root.is_dir():
        for condition_root in benchmark_root.iterdir():
            if not condition_root.is_dir():
                continue
            removed.extend(_remove_dirs(condition_root / "model_visible"))
            removed.extend(
                _remove_children_except(
                    condition_root / "evaluation_truth",
                    keep_names={"query_truth.csv"},
                )
            )

    removed.extend(
        _remove_children_except(
            root / "evaluation",
            keep_names={"metrics.csv"},
        )
    )

    scoring_root = root / "scoring"
    if scoring_root.is_dir():
        for condition_root in scoring_root.iterdir():
            if not condition_root.is_dir():
                continue
            for candidate_set_root in condition_root.iterdir():
                if not candidate_set_root.is_dir():
                    continue
                removed.extend(
                    _remove_children_except(
                        candidate_set_root,
                        keep_names={"cell_scores.parquet"},
                    )
                )

    return tuple(removed)


def _remove_dirs(*paths: Path) -> list[Path]:
    removed: list[Path] = []
    for path in paths:
        if path.is_symlink():
            path.unlink()
            removed.append(path)
            continue
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def _remove_children_except(root: Path, *, keep_names: set[str]) -> list[Path]:
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for child in root.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(child)
    return removed
