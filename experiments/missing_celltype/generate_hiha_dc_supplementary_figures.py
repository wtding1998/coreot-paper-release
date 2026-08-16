"""Generate the canonical HIHA supplementary figures one at a time."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping
from pathlib import Path

import yaml

import coreot.results.hiha_supplement as hiha_supplement_module
from coreot.artifacts.hashes import sha256_file
from coreot.results.hiha_supplement import write_hiha_s1, write_hiha_s2, write_hiha_s3, write_hiha_s4


S3_BASENAME = "supplementary_figure_s3_prior_dependence"
S3_SUBMISSION_BASENAME = "manuscript_fig_hiha_supp_s3_prior_dependence"


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _hash_record(path: Path, project_root: Path) -> dict[str, str]:
    return {
        "path": _display_path(path, project_root),
        "sha256": sha256_file(path),
    }


def _existing_hiha_s3_outputs(output_root: Path) -> dict[str, Path]:
    data_root = output_root / "data"
    outputs = {
        "figure_png": output_root / f"{S3_BASENAME}.png",
        "figure_pdf": output_root / f"{S3_BASENAME}.pdf",
        "figure_svg": output_root / f"{S3_BASENAME}.svg",
        "prior_bins_by_seed": data_root
        / "supplementary_figure_s3_prior_bins_by_seed.csv",
        "prior_bins_donor_equal": data_root
        / "supplementary_figure_s3_prior_bins_donor_equal.csv",
        "correlations_by_seed": data_root
        / "supplementary_figure_s3_prior_correlations_by_seed.csv",
        "correlations_summary": data_root
        / "supplementary_figure_s3_prior_correlations_summary.csv",
        "description": output_root / f"{S3_BASENAME}.md",
        "manifest": output_root / f"{S3_BASENAME}_manifest.yaml",
    }
    missing = [path for path in outputs.values() if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing S3 artifacts: {missing_text}")
    return outputs


def promote_hiha_s3_submission_copies(
    *,
    outputs: Mapping[str, Path],
    docs_fig_root: Path,
    project_root: Path = Path.cwd(),
) -> dict[str, Path]:
    """Promote the complete S3 render bundle and record its exact lineage."""
    table_keys = (
        "prior_bins_by_seed",
        "prior_bins_donor_equal",
        "correlations_by_seed",
        "correlations_summary",
    )
    required_keys = (
        "figure_png",
        "figure_pdf",
        "figure_svg",
        *table_keys,
        "description",
        "manifest",
    )
    missing = [key for key in required_keys if key not in outputs]
    if missing:
        raise KeyError(f"Missing S3 output keys: {', '.join(missing)}")
    missing_files = [Path(outputs[key]) for key in required_keys if not Path(outputs[key]).is_file()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing S3 artifacts: {missing_text}")

    docs_fig_root.mkdir(parents=True, exist_ok=True)
    promoted: dict[str, Path] = {}
    result_renderings: dict[str, dict[str, str]] = {}
    submission_copies: dict[str, dict[str, str | bool]] = {}
    for suffix in ("png", "pdf", "svg"):
        source = Path(outputs[f"figure_{suffix}"])
        destination = docs_fig_root / f"{S3_SUBMISSION_BASENAME}.{suffix}"
        shutil.copyfile(source, destination)
        source_record = _hash_record(source, project_root)
        destination_record = _hash_record(destination, project_root)
        if source_record["sha256"] != destination_record["sha256"]:
            raise RuntimeError(
                f"S3 submission copy differs from its source: {destination}"
            )
        promoted[f"submission_{suffix}"] = destination
        result_renderings[suffix] = source_record
        submission_copies[suffix] = {
            "source": source_record["path"],
            "destination": destination_record["path"],
            "sha256": destination_record["sha256"],
            "byte_identical": True,
        }

    driver_path = Path(__file__)
    implementation_path = Path(hiha_supplement_module.__file__)
    release = {
        "schema_version": 1,
        "stage": "hiha_s3_submission_render_promotion",
        "configuration_manifest": _hash_record(
            Path(outputs["manifest"]), project_root
        ),
        "source_tables": {
            key: _hash_record(Path(outputs[key]), project_root)
            for key in table_keys
        },
        "description": _hash_record(Path(outputs["description"]), project_root),
        "code": {
            "driver": _hash_record(driver_path, project_root),
            "implementation": _hash_record(implementation_path, project_root),
        },
        "result_renderings": result_renderings,
        "submission_copies": submission_copies,
    }
    release_path = Path(outputs["manifest"]).with_name(
        f"{S3_BASENAME}_release_manifest.yaml"
    )
    release_path.write_text(
        yaml.safe_dump(release, sort_keys=False), encoding="utf-8"
    )
    promoted["release_manifest"] = release_path
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("s1", "s2", "s3", "s4"),
        default="s1",
        help="Supplementary figure to generate.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(
            "data/derived/hiha_dc/"
            "human_immune_health_atlas_dc.with_recomputed_AIFI_L2_score.h5ad"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/HIHA_DC/figures")
    )
    parser.add_argument(
        "--docs-fig-root",
        type=Path,
        help="Promote an S3 render bundle to this submission figure directory.",
    )
    parser.add_argument(
        "--promote-existing-s3",
        action="store_true",
        help="Promote existing S3 outputs without reading run data or HDF5 input.",
    )
    parser.add_argument("--candidate-set", default="hiha_harmony30_k100")
    parser.add_argument("--eta", type=float, default=1.0e-12)
    args = parser.parse_args()

    if args.promote_existing_s3:
        if args.figure != "s3":
            parser.error("--promote-existing-s3 requires --figure s3")
        if args.docs_fig_root is None:
            parser.error("--promote-existing-s3 requires --docs-fig-root")
        outputs = _existing_hiha_s3_outputs(args.output_root)
    elif args.figure == "s1":
        outputs = write_hiha_s1(
            runs_root=args.runs_root,
            input_path=args.input_path,
            output_root=args.output_root,
            candidate_set=args.candidate_set,
            eta=args.eta,
        )
    elif args.figure == "s2":
        outputs = write_hiha_s2(
            runs_root=args.runs_root,
            input_path=args.input_path,
            output_root=args.output_root,
            candidate_set=args.candidate_set,
        )
    elif args.figure == "s3":
        outputs = write_hiha_s3(
            runs_root=args.runs_root,
            input_path=args.input_path,
            output_root=args.output_root,
            candidate_set=args.candidate_set,
        )
    elif args.figure == "s4":
        outputs = write_hiha_s4(output_root=args.output_root)
    else:
        raise SystemExit(
            f"Supplementary Figure {args.figure.upper()} is not implemented yet; "
            "run with --figure s1."
        )
    if args.figure == "s3" and args.docs_fig_root is not None:
        outputs.update(
            promote_hiha_s3_submission_copies(
                outputs=outputs,
                docs_fig_root=args.docs_fig_root,
            )
        )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
