from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd


REPORT_GRID = "report_leave_one_HIHA_DC"
EXTERNAL_GRID = "report_leave_one_HIHA_DC_external_baselines"
UNIFORM_GRID = "hiha_dc_uniform_uot_tau05"
CONDITIONS = ("incomplete_reference", "full_reference_control")
INTERNAL_CANDIDATE_SET = "hiha_harmony30_k100"
EXTERNAL_CANDIDATE_SET = "external_reference_mapping"
VERIFICATION_DATE = "2026-08-02"


@dataclass(frozen=True)
class IncludedFile:
    package_path: str
    source_path: str
    action: str
    source_bytes: int
    package_bytes: int
    source_sha256: str
    package_sha256: str
    note: str


@dataclass(frozen=True)
class Transformation:
    package_path: str
    source_path: str
    selector: str
    source_rows: int
    package_rows: int
    removed_rows: int
    package_methods: str


@dataclass(frozen=True)
class Exclusion:
    source_path: str
    exclusion_type: str
    reason: str
    replacement: str


class PackageBuilder:
    def __init__(self, *, repository_root: Path, output_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.output_root = output_root.resolve()
        self.included: list[IncludedFile] = []
        self.transformations: list[Transformation] = []
        self.exclusions: list[Exclusion] = []

    def build(self) -> None:
        self._require_empty_output()
        self.output_root.mkdir(parents=True)
        self._copy_code()
        self._copy_configs()
        self._copy_verified_results()
        self._copy_run_artifacts()
        self._write_external_dependencies()
        self._write_exclusions()
        self._write_transformations()
        self._write_included_files()
        self._write_readme()
        self._write_checksums()

    def _require_empty_output(self) -> None:
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite nonempty staging root: {self.output_root}"
            )

    def _copy_code(self) -> None:
        for relative in (
            Path("pyproject.toml"),
            Path("uv.lock"),
        ):
            self.copy_file(relative, Path("code") / relative)
        self.copy_tree(Path("src/coreot"), Path("code/src/coreot"))
        for name in (
            "build_hiha_submission_staging.py",
            "generate_grid_configs.py",
            "generate_hiha_dc_compare_baselines.py",
            "generate_hiha_dc_experiment_overview.py",
            "generate_hiha_dc_figure2_panels.py",
            "generate_hiha_dc_reformulation.py",
            "generate_hiha_dc_report_leave_one_configs.py",
        ):
            source = Path("experiments/missing_celltype") / name
            self.copy_file(source, Path("code") / source)
        for name in (
            "test_hiha_figure2.py",
            "test_hiha_selected_operating_point.py",
            "test_hiha_submission_staging.py",
        ):
            source = Path("tests") / name
            if (self.repository_root / source).is_file():
                self.copy_file(source, Path("code") / source)

    def _copy_configs(self) -> None:
        config_root = Path("experiments/missing_celltype/generated_configs")
        for grid in (REPORT_GRID, EXTERNAL_GRID, UNIFORM_GRID):
            self.copy_tree(config_root / grid, Path("configs") / grid)

    def _copy_verified_results(self) -> None:
        verification_root = (
            Path("results/submission_verification/HIHA_DC") / VERIFICATION_DATE
        )
        for name in ("main", "compare_baselines", "figure2", "figure2_panels"):
            self.copy_tree(
                verification_root / name,
                Path("verified_results") / name,
                include=lambda path: "archive" not in path.parts,
            )
        self.copy_tree(
            Path("results/HIHA_DC/main/experiment_overview"),
            Path("verified_results/main/experiment_overview"),
        )
        self.copy_tree(
            verification_root / "docs/figs",
            Path("figures"),
        )
        for name in (
            "comparison.md",
            "figure2_data_code_audit.md",
            "hla_uniform_lineage_audit.md",
            "rank_recovery_scope_check.md",
        ):
            self.copy_file(verification_root / name, Path("audits") / name)
        self.copy_tree(
            verification_root / "reproducibility",
            Path("audits/reproducibility"),
            include=lambda path: "archive" not in path.parts,
        )

    def _copy_run_artifacts(self) -> None:
        config_root = self.repository_root / "experiments/missing_celltype/generated_configs"
        report_run_ids = sorted(path.name for path in (config_root / REPORT_GRID).iterdir())
        uniform_run_ids = sorted(path.name for path in (config_root / UNIFORM_GRID).iterdir())
        self._validate_run_ids(report_run_ids, suffix="_report_leave_one_HIHA_DC")
        self._validate_run_ids(uniform_run_ids, suffix="_tau05_uniform")
        for run_id in report_run_ids:
            self._copy_report_run(run_id)
        for run_id in uniform_run_ids:
            self._copy_uniform_run(run_id)

    @staticmethod
    def _validate_run_ids(run_ids: list[str], *, suffix: str) -> None:
        if len(run_ids) != 10 or not all(run_id.endswith(suffix) for run_id in run_ids):
            raise ValueError(
                f"Expected 10 HIHA run directories ending in {suffix!r}; got {run_ids}"
            )

    def _copy_report_run(self, run_id: str) -> None:
        run = Path("runs") / run_id
        destination = Path("data/runs") / run_id
        for relative in (
            Path("benchmark/condition_manifest.csv"),
            Path("benchmark/split_manifest.csv"),
        ):
            self.copy_file(run / relative, destination / relative)
        for condition in CONDITIONS:
            for relative in (
                Path(f"benchmark/{condition}/evaluation_truth/query_truth.csv"),
                Path(f"benchmark/{condition}/evaluation_truth/state_presence.csv"),
                Path(
                    f"benchmark/{condition}/evaluation_truth/benchmark_truth_manifest.yaml"
                ),
                Path(f"benchmark/{condition}/model_visible/target_labels.csv"),
            ):
                self.copy_file(run / relative, destination / relative)

            internal = Path(
                f"scoring/{condition}/{INTERNAL_CANDIDATE_SET}/cell_scores.parquet"
            )
            self.filter_methods(
                run / internal,
                destination / internal,
                selector=lambda methods: methods.ne("uniform_uot"),
                selector_name="method != uniform_uot",
                note="Removed stale report-run Uniform UOT rows.",
            )
            external = Path(
                f"scoring/{condition}/{EXTERNAL_CANDIDATE_SET}/cell_scores.parquet"
            )
            self.copy_file(run / external, destination / external)

            transport = Path(
                f"transport/{condition}/{INTERNAL_CANDIDATE_SET}/coreot_full"
            )
            self.copy_tree(run / transport, destination / transport)

        for name in ("metrics.csv", "forced_label_summary.csv"):
            relative = Path("evaluation") / name
            self.filter_methods(
                run / relative,
                destination / relative,
                selector=lambda methods: methods.ne("uniform_uot"),
                selector_name="method != uniform_uot",
                note="Removed stale report-run Uniform UOT rows.",
            )

        if "_seed1_" in run_id:
            embedding = Path("embeddings/incomplete_reference/hiha_harmony30")
            for name in ("embedding.npy", "embedding_cells.csv", "manifest.json"):
                relative = embedding / name
                self.copy_file(run / relative, destination / relative)

        stale_root = run / Path(
            f"transport/*/{INTERNAL_CANDIDATE_SET}/uniform_uot"
        )
        self.exclusions.append(
            Exclusion(
                source_path=str(stale_root),
                exclusion_type="method_directory",
                reason=(
                    "Report-run Uniform UOT transport parameters do not match the "
                    "submission lineage."
                ),
                replacement=f"data/runs/{run_id.replace('_report_leave_one_HIHA_DC', '_tau05_uniform')}",
            )
        )

    def _copy_uniform_run(self, run_id: str) -> None:
        run = Path("runs") / run_id
        destination = Path("data/runs") / run_id
        for relative in (
            Path("benchmark/condition_manifest.csv"),
            Path("benchmark/split_manifest.csv"),
        ):
            self.copy_file(run / relative, destination / relative)
        for condition in CONDITIONS:
            for relative in (
                Path(f"benchmark/{condition}/evaluation_truth/query_truth.csv"),
                Path(f"benchmark/{condition}/evaluation_truth/state_presence.csv"),
                Path(
                    f"benchmark/{condition}/evaluation_truth/benchmark_truth_manifest.yaml"
                ),
            ):
                self.copy_file(run / relative, destination / relative)
            scores = Path(
                f"scoring/{condition}/{INTERNAL_CANDIDATE_SET}/cell_scores.parquet"
            )
            self.filter_methods(
                run / scores,
                destination / scores,
                selector=lambda methods: methods.eq("uniform_uot"),
                selector_name="method == uniform_uot",
                note="Retained only the dedicated tau=0.5 Uniform UOT rows.",
            )
            transport = Path(
                f"transport/{condition}/{INTERNAL_CANDIDATE_SET}/uniform_uot"
            )
            self.copy_tree(run / transport, destination / transport)

        for name in ("metrics.csv", "forced_label_summary.csv"):
            relative = Path("evaluation") / name
            self.filter_methods(
                run / relative,
                destination / relative,
                selector=lambda methods: methods.eq("uniform_uot"),
                selector_name="method == uniform_uot",
                note="Retained only the dedicated tau=0.5 Uniform UOT rows.",
            )

        self.exclusions.append(
            Exclusion(
                source_path=str(run / "transport/*/*/prior_only"),
                exclusion_type="method_directory",
                reason="The replacement grid is included only for Uniform UOT.",
                replacement="none",
            )
        )

    def copy_tree(
        self,
        source: Path,
        destination: Path,
        *,
        include: Callable[[Path], bool] | None = None,
    ) -> None:
        source_root = self.repository_root / source
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.name == ".DS_Store":
                continue
            relative = path.relative_to(source_root)
            if include is not None and not include(relative):
                continue
            self.copy_file(source / relative, destination / relative)

    def copy_file(self, source: Path, destination: Path, *, note: str = "") -> None:
        source_path = self.repository_root / source
        destination_path = self.output_root / destination
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        self._record_file(
            source=source,
            destination=destination,
            action="copied",
            note=note,
        )

    def filter_methods(
        self,
        source: Path,
        destination: Path,
        *,
        selector: Callable[[pd.Series], pd.Series],
        selector_name: str,
        note: str,
    ) -> None:
        source_path = self.repository_root / source
        destination_path = self.output_root / destination
        if source_path.suffix == ".parquet":
            frame = pd.read_parquet(source_path)
        elif source_path.suffix == ".csv":
            frame = pd.read_csv(source_path)
        else:
            raise ValueError(f"Unsupported filtered table format: {source_path}")
        if "method" not in frame.columns:
            raise ValueError(f"Filtered artifact lacks method column: {source_path}")
        keep = selector(frame["method"].astype(str))
        filtered = frame.loc[keep].copy()
        if filtered.empty:
            raise ValueError(f"Method filter produced an empty artifact: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.suffix == ".parquet":
            filtered.to_parquet(destination_path, index=False)
        else:
            filtered.to_csv(destination_path, index=False)
        self._record_file(
            source=source,
            destination=destination,
            action="method_filtered",
            note=note,
        )
        methods = ";".join(sorted(filtered["method"].astype(str).unique()))
        self.transformations.append(
            Transformation(
                package_path=destination.as_posix(),
                source_path=source.as_posix(),
                selector=selector_name,
                source_rows=len(frame),
                package_rows=len(filtered),
                removed_rows=len(frame) - len(filtered),
                package_methods=methods,
            )
        )

    def _record_file(
        self,
        *,
        source: Path,
        destination: Path,
        action: str,
        note: str,
    ) -> None:
        source_path = self.repository_root / source
        destination_path = self.output_root / destination
        self.included.append(
            IncludedFile(
                package_path=destination.as_posix(),
                source_path=source.as_posix(),
                action=action,
                source_bytes=source_path.stat().st_size,
                package_bytes=destination_path.stat().st_size,
                source_sha256=sha256(source_path),
                package_sha256=sha256(destination_path),
                note=note,
            )
        )

    def _write_external_dependencies(self) -> None:
        dependencies = (
            (
                Path(
                    "data/derived/hiha_dc/"
                    "human_immune_health_atlas_dc.with_recomputed_AIFI_L2_score.h5ad"
                ),
                "Required to rerun the scope-aware main-table aggregation and all model-facing stages.",
            ),
            (
                Path(
                    "data/models/celltypist/"
                    "ref_pbmc_clean_celltypist_model_AIFI_L2_2024-04-19.pkl"
                ),
                "Required only to recreate the recomputed AIFI Level-2 confidence input.",
            ),
        )
        rows = []
        for relative, role in dependencies:
            path = self.repository_root / relative
            rows.append(
                {
                    "repository_path": relative.as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "included_in_package": False,
                    "role": role,
                }
            )
        self._write_csv(Path("manifests/external_dependencies.csv"), rows)

    def _write_exclusions(self) -> None:
        self.exclusions.extend(
            [
                Exclusion(
                    source_path="results/HIHA_DC/figures/hiha_main_results.*",
                    exclusion_type="historical_figure_family",
                    reason="Superseded and outside the current manuscript submission scope.",
                    replacement="verified_results/figure2 and figures/manuscript_fig_hiha_main.*",
                ),
                Exclusion(
                    source_path="results/HIHA_DC/figures/*rank_recovery*",
                    exclusion_type="historical_figure_family",
                    reason="Superseded, retains the excluded u_tilde contract, and is outside submission scope.",
                    replacement="none",
                ),
                Exclusion(
                    source_path="runs/*/scoring/*/*/label_probabilities.npz",
                    exclusion_type="unused_mixed_artifact",
                    reason="The retained aggregation and Figure 2 generators do not read combined scoring archives.",
                    replacement="method-specific transport archives where consumed",
                ),
                Exclusion(
                    source_path="data/derived/hiha_dc/*.h5ad",
                    exclusion_type="external_dependency",
                    reason="Large refitting input is checksummed but not duplicated in this result-reproduction package.",
                    replacement="manifests/external_dependencies.csv",
                ),
            ]
        )
        self._write_csv(Path("manifests/excluded_files.csv"), self.exclusions)

    def _write_transformations(self) -> None:
        self._write_csv(Path("manifests/transformations.csv"), self.transformations)

    def _write_included_files(self) -> None:
        self._write_csv(Path("manifests/included_files.csv"), self.included)

    def _write_csv(self, relative: Path, rows: Iterable[object]) -> None:
        path = self.output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        materialized = list(rows)
        if not materialized:
            raise ValueError(f"Refusing to write an empty manifest: {path}")
        first = materialized[0]
        if hasattr(first, "__dataclass_fields__"):
            fieldnames = list(first.__dataclass_fields__)
            records = [
                {field: getattr(row, field) for field in fieldnames}
                for row in materialized
            ]
        else:
            records = materialized
            fieldnames = list(records[0])
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def _write_readme(self) -> None:
        readme = self.output_root / "README.md"
        readme.write_text(
            "# HIHA DC primary-result submission package\n\n"
            "This isolated package contains the verified primary tables, Figure 2, "
            "the exact run-level result artifacts consumed by their generators, "
            "the relevant code and configurations, and machine-readable provenance.\n\n"
            "## Scope\n\n"
            "The package is a result-reproduction bundle, not a complete copy of every "
            "intermediate generated during model fitting. Report-run Uniform UOT rows and "
            "transport directories are excluded because their fitted parameters do not "
            "belong to the retained lineage. Uniform UOT is supplied only by the dedicated "
            "`*_tau05_uniform` runs. Original repository artifacts were not modified.\n\n"
            "The processed 983 MB `.h5ad` and the CellTypist model are not duplicated. "
            "Their paths, sizes, roles, and SHA-256 checksums are recorded in "
            "`manifests/external_dependencies.csv`. The `.h5ad` is required for the "
            "scope-aware main-table aggregation because it supplies AIFI Level-2 metadata; "
            "it is not required to render the preserved composite from source data.\n\n"
            "## Structure\n\n"
            "- `code/`: pinned project metadata, `coreot` source, focused drivers, and tests.\n"
            "- `configs/`: selected report, external-baseline, and tau=0.5 Uniform grids.\n"
            "- `data/runs/`: filtered report results and Uniform-only replacement results.\n"
            "- `verified_results/`: audited main, comparison, and Figure 2 artifacts.\n"
            "- `figures/`: submission-facing Figure 2 files.\n"
            "- `audits/`: human-readable and machine-readable verification evidence.\n"
            "- `manifests/`: inclusion, exclusion, transformation, dependency, and checksum records.\n\n"
            "## Regeneration\n\n"
            "Run commands from this package root. Use Python 3.11 and the locked environment "
            "in `code/uv.lock`. First regenerate the comparator bundle:\n\n"
            "```bash\n"
            "uv run --project code python code/experiments/missing_celltype/generate_hiha_dc_compare_baselines.py \\\n"
            "  --runs-root data/runs \\\n"
            "  --grid-dir configs/report_leave_one_HIHA_DC \\\n"
            "  --uniform-uot-grid-dir configs/hiha_dc_uniform_uot_tau05 \\\n"
            "  --output-root reproduced/HIHA_DC/compare_baselines\n"
            "```\n\n"
            "After placing the checksummed `.h5ad` at the path recorded in its copied "
            "`raw_import.yaml`, regenerate the scope-aware main tables:\n\n"
            "```bash\n"
            "uv run --project code python code/experiments/missing_celltype/generate_hiha_dc_reformulation.py \\\n"
            "  --runs-root data/runs \\\n"
            "  --grid-dir configs/report_leave_one_HIHA_DC \\\n"
            "  --uniform-grid-dir configs/hiha_dc_uniform_uot_tau05 \\\n"
            "  --output-root reproduced/HIHA_DC\n"
            "```\n\n"
            "Generate Panels A--F with the copied overview, regenerated tables, and staged "
            "run results; then generate `--panel main` from the resulting source data. "
            "The focused driver exposes all required paths through `--overview`, "
            "`--detection-table`, `--panel-b-detection-table`, "
            "`--label-transfer-table`, `--runs-root`, `--panel-root`, and `--output-root`.\n\n"
            "## Integrity\n\n"
            "Verify all staged files with `shasum -a 256 -c manifests/checksums.sha256`. "
            "Filtered files have new package checksums and explicit row counts in "
            "`manifests/transformations.csv`; copied files retain identical source and "
            "package checksums in `manifests/included_files.csv`.\n",
            encoding="utf-8",
        )

    def _write_checksums(self) -> None:
        path = self.output_root / "manifests/checksums.sha256"
        files = sorted(
            candidate
            for candidate in self.output_root.rglob("*")
            if candidate.is_file() and candidate != path
        )
        lines = [
            f"{sha256(candidate)}  {candidate.relative_to(self.output_root).as_posix()}"
            for candidate in files
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    default_output = (
        repository_root
        / "results/submission_verification/HIHA_DC"
        / VERIFICATION_DATE
        / "package_staging/HIHA_DC"
    )
    parser = argparse.ArgumentParser(
        description="Build the isolated, method-filtered HIHA primary-result package."
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--output-root", type=Path, default=default_output)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    builder = PackageBuilder(
        repository_root=args.repository_root,
        output_root=args.output_root,
    )
    builder.build()
    print(builder.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
