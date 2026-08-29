"""Release-only PBMC metadata acquisition and artifact regeneration."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import yaml
from PIL import Image


SUPPORTED_UNITS = (
    "pbmc_primary",
    "pbmc_matched_reference",
    "pbmc_compatibility_sensitivity",
    "pbmc_parameter_and_calibration",
    "pbmc_prior_dependence",
)
HDF_SUFFIXES = {".h5", ".h5ad", ".hdf5"}
CHECKSUM_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")

# Artifact-specific accepted/current PNG pairs. Broad pixel tolerances would
# hide unrelated scientific rendering changes.
_ACCEPTED_CURRENT_PNG_REVISIONS = {
    "figure_3_pbmc_panel_c.png": (
        "edb07bb3fc244a62676d71fc17aa90af7f7c82fe7efe31035c187c2bd9ae158f",
        "ddbaebf7378b074d7f2431189d5139f3be6b84bf59d6ac063676d1165ccda902",
    ),
    "pbmc_compatibility_sensitivity.png": (
        "3f9593ab47e6810d42b28839fff9e6f57f8ace3fdeb8c55c5562edc9eb76f5fa",
        "9c0ead3490cd0892f77bd0581fb9f96072de2e1d7e6ab6a2dba58081715deb61",
    ),
    "manuscript_fig_pbmc_supp_robustness.png": (
        "123010fea380f7b3a5eebafb22d76d9dc7a6966c24e907975653dd8f2aab5924",
        "0894a39250d84c2b6ea4185451aeb5bc39984486a328a5591ac5c38bec2b9d91",
    ),
    "figure_s7_pbmc_calibration.png": (
        "56f3020a6a903fd54f93953743fcfadd26fe781c173593c286771124b029753a",
        "12e3d60cea9a8e912529d244c163bef2cadc5644508f19b3d8eeb873ea679ff9",
    ),
    "manuscript_fig_pbmc_supp_prior_dependence.png": (
        "16516bc8709f4e580263bc19d6efef2b80f434c9f8259b8674c39fb705551c5f",
        "d567dafa36b1dc0192e2581e86952324aa7dd05ead5b0ef7a7101a7015327c32",
    ),
}

# PDF signatures exclude nondeterministic metadata while pinning page geometry.
_ACCEPTED_CURRENT_PDF_REVISIONS = {
    "manuscript_fig_pbmc_supp_robustness.pdf": (
        (True, True, 1, (b"0 0 560.097125 423.9340078125",)),
        (True, True, 1, (b"0 0 560.034625 424.25725",)),
    ),
    "figure_s7_pbmc_calibration.pdf": (
        (True, True, 1, (b"0 0 1376.7538945357 1016.39952",)),
        (True, True, 1, (b"0 0 1376.9241977823 1016.39952",)),
    ),
    "manuscript_fig_pbmc_supp_prior_dependence.pdf": (
        (True, True, 1, (b"0 0 568.6921473752 189.9209999385",)),
        (True, True, 1, (b"0 0 568.8015223752 190.306742126",)),
    ),
}

# SVG signatures pin geometry and complete element topology while excluding
# metadata values and generated element identifiers.
_ACCEPTED_CURRENT_SVG_REVISIONS = {
    "manuscript_fig_pbmc_supp_robustness.svg": (
        (
            "560.111539pt",
            "423.934008pt",
            "0 0 560.111539 423.934008",
            (
                ("Agent", 1),
                ("RDF", 1),
                ("Work", 1),
                ("clipPath", 6),
                ("creator", 1),
                ("date", 1),
                ("defs", 21),
                ("format", 1),
                ("g", 476),
                ("image", 8),
                ("metadata", 1),
                ("path", 100),
                ("rect", 6),
                ("style", 1),
                ("svg", 1),
                ("title", 1),
                ("type", 1),
                ("use", 679),
            ),
        ),
        (
            "559.579937pt",
            "423.403578pt",
            "0 0 559.579937 423.403578",
            (
                ("Agent", 1),
                ("RDF", 1),
                ("Work", 1),
                ("clipPath", 6),
                ("creator", 1),
                ("date", 1),
                ("defs", 20),
                ("format", 1),
                ("g", 458),
                ("image", 8),
                ("metadata", 1),
                ("path", 93),
                ("rect", 6),
                ("style", 1),
                ("svg", 1),
                ("title", 1),
                ("type", 1),
                ("use", 673),
            ),
        ),
    ),
    "figure_s7_pbmc_calibration.svg": (
        (
            "1376.470693pt",
            "1016.39952pt",
            "0 0 1376.470693 1016.39952",
            (
                ("Agent", 1),
                ("RDF", 1),
                ("Work", 1),
                ("clipPath", 12),
                ("creator", 1),
                ("date", 1),
                ("defs", 39),
                ("format", 1),
                ("g", 1379),
                ("image", 4),
                ("metadata", 1),
                ("path", 247),
                ("rect", 12),
                ("style", 1),
                ("svg", 1),
                ("title", 1),
                ("type", 1),
                ("use", 2268),
            ),
        ),
        (
            "1376.498183pt",
            "1016.39952pt",
            "0 0 1376.498183 1016.39952",
            (
                ("Agent", 1),
                ("RDF", 1),
                ("Work", 1),
                ("clipPath", 12),
                ("creator", 1),
                ("date", 1),
                ("defs", 39),
                ("format", 1),
                ("g", 1379),
                ("image", 4),
                ("metadata", 1),
                ("path", 247),
                ("rect", 12),
                ("style", 1),
                ("svg", 1),
                ("title", 1),
                ("type", 1),
                ("use", 2268),
            ),
        ),
    ),
    "manuscript_fig_pbmc_supp_prior_dependence.svg": (
        (
            "567.159022pt",
            "189.921664pt",
            "0 0 567.159022 189.921664",
            (
                ("Agent", 1),
                ("RDF", 1),
                ("Work", 1),
                ("clipPath", 4),
                ("creator", 1),
                ("date", 1),
                ("defs", 29),
                ("format", 1),
                ("g", 303),
                ("metadata", 1),
                ("path", 140),
                ("rect", 4),
                ("style", 1),
                ("svg", 1),
                ("title", 1),
                ("type", 1),
                ("use", 427),
            ),
        ),
        (
            "567.229022pt",
            "189.921664pt",
            "0 0 567.229022 189.921664",
            (
                ("Agent", 1),
                ("RDF", 1),
                ("Work", 1),
                ("clipPath", 4),
                ("creator", 1),
                ("date", 1),
                ("defs", 29),
                ("format", 1),
                ("g", 303),
                ("metadata", 1),
                ("path", 140),
                ("rect", 4),
                ("style", 1),
                ("svg", 1),
                ("title", 1),
                ("type", 1),
                ("use", 427),
            ),
        ),
    ),
}


class PBMCReleaseRegenerationError(ValueError):
    """Raised when PBMC release evidence violates the public contract."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_output(release_root: Path, output_root: str | Path) -> Path:
    raw_output = Path(output_root)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"Output root must not already exist: {raw_output}")
    output = raw_output.resolve()
    if output == release_root or output.is_relative_to(release_root):
        raise PBMCReleaseRegenerationError(
            f"Output root must be outside the read-only release: {output}"
        )
    return output


def _validate_release(release_root: str | Path) -> Path:
    release = Path(release_root).resolve()
    if not release.is_dir():
        raise PBMCReleaseRegenerationError(f"Submission release root is not a directory: {release}")
    manifest_path = release / "provenance/checksums.sha256"
    if not manifest_path.is_file():
        raise PBMCReleaseRegenerationError("Submission release omits provenance/checksums.sha256.")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            raise PBMCReleaseRegenerationError(
                f"Malformed release checksum entry at line {line_number}."
            )
        expected, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in declared:
            raise PBMCReleaseRegenerationError(
                f"Unsafe or duplicate release checksum path: {relative}"
            )
        declared[relative] = expected
    observed_files: set[str] = set()
    for path in sorted(release.rglob("*")):
        if path.is_symlink():
            raise PBMCReleaseRegenerationError(
                f"Submission release contains a symbolic link: {path}"
            )
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(release).as_posix()
        observed_files.add(relative)
        if path.suffix.casefold() in HDF_SUFFIXES:
            raise PBMCReleaseRegenerationError(
                f"Submission release contains a forbidden HDF5-family file: {relative}"
            )
    if observed_files != set(declared):
        missing = sorted(set(declared) - observed_files)
        undeclared = sorted(observed_files - set(declared))
        raise PBMCReleaseRegenerationError(
            f"Release checksum inventory mismatch: missing={missing}, undeclared={undeclared}"
        )
    for relative, expected in declared.items():
        observed = _sha256(release / relative)
        if observed != expected:
            raise PBMCReleaseRegenerationError(
                f"Release checksum mismatch for {relative}: {observed} != {expected}"
            )
    return release


def _external_dependencies(release_root: Path) -> pd.DataFrame:
    path = release_root / "provenance/external_dependencies.csv"
    frame = pd.read_csv(path, dtype={"sha256": "string", "repository_path": "string"})
    required = {
        "unit_id",
        "repository_path",
        "sha256",
        "bytes",
        "included_in_package",
    }
    if not required <= set(frame.columns):
        raise PBMCReleaseRegenerationError(
            f"External-dependency declaration is missing columns {sorted(required - set(frame.columns))}."
        )
    return frame


def _declared_source(release_root: Path, unit_id: str | None = None) -> dict[str, object]:
    dependencies = _external_dependencies(release_root)
    selected = dependencies.loc[
        dependencies["unit_id"].isin(SUPPORTED_UNITS)
        & dependencies["repository_path"].eq("data/raw/kang_2018.h5ad")
    ].copy()
    if unit_id is not None:
        selected = selected.loc[selected["unit_id"].eq(unit_id)]
    identities = selected.loc[:, ["repository_path", "sha256", "bytes"]].drop_duplicates()
    if len(selected) == 0 or len(identities) != 1:
        raise PBMCReleaseRegenerationError(
            "Release must declare one consistent canonical PBMC object identity."
        )
    identity = identities.iloc[0]
    checksum = str(identity["sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise PBMCReleaseRegenerationError("Declared PBMC SHA-256 is malformed.")
    return {
        "path": str(identity["repository_path"]),
        "sha256": checksum,
        "bytes": int(identity["bytes"]),
        "unit_ids": sorted(selected["unit_id"].astype(str).unique()),
    }


def _nonempty_strings(series: pd.Series, name: str) -> pd.Series:
    values = series.astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise PBMCReleaseRegenerationError(f"PBMC metadata contains an empty {name}.")
    return values


def _normalize_conditions(series: pd.Series) -> pd.Series:
    aliases = {
        "ctrl": "ctrl",
        "control": "ctrl",
        "unstim": "ctrl",
        "unstimulated": "ctrl",
        "stim": "stim",
        "stimulated": "stim",
        "ifnb": "stim",
        "ifn-beta": "stim",
    }
    normalized = _nonempty_strings(series, "condition").str.casefold().map(aliases)
    if normalized.isna().any():
        unexpected = sorted(_nonempty_strings(series, "condition").loc[normalized.isna()].unique())
        raise PBMCReleaseRegenerationError(
            f"PBMC metadata contains unsupported conditions: {unexpected}"
        )
    return normalized


def _validate_metadata_export(
    release_root: Path, unit_id: str, metadata_export_root: str | Path
) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    root = Path(metadata_export_root).resolve()
    if root == release_root or root.is_relative_to(release_root):
        raise PBMCReleaseRegenerationError(
            "PBMC metadata export must be outside the read-only release."
        )
    if not root.is_dir():
        raise PBMCReleaseRegenerationError(f"PBMC metadata export is missing: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PBMCReleaseRegenerationError(
                f"PBMC metadata export contains a symbolic link: {path}"
            )
        if path.is_file() and path.suffix.casefold() in HDF_SUFFIXES:
            raise PBMCReleaseRegenerationError(
                f"PBMC regeneration refuses HDF5-family metadata input: {path.name}"
            )
    cells_path = root / "cells.csv"
    manifest_path = root / "manifest.yaml"
    if not cells_path.is_file() or not manifest_path.is_file():
        raise PBMCReleaseRegenerationError(
            "PBMC metadata export must contain cells.csv and manifest.yaml."
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    declared = _declared_source(release_root, unit_id)
    try:
        source = manifest["source"]
        output = manifest["output"]
    except (KeyError, TypeError) as error:
        raise PBMCReleaseRegenerationError("PBMC metadata manifest is malformed.") from error
    for field in ("path", "sha256", "bytes"):
        if source.get(field) != declared[field]:
            raise PBMCReleaseRegenerationError(
                f"PBMC metadata source identity disagrees on {field}."
            )
    if output.get("path") != "cells.csv" or output.get("sha256") != _sha256(cells_path):
        raise PBMCReleaseRegenerationError("PBMC cells.csv checksum is invalid.")
    cells = pd.read_csv(cells_path, dtype="string", keep_default_na=False)
    expected_columns = ["cell_id", "cell_type", "condition", "donor"]
    if list(cells.columns) != expected_columns:
        raise PBMCReleaseRegenerationError(
            f"PBMC cells.csv must have exact columns {expected_columns}."
        )
    if output.get("columns") != expected_columns or output.get("rows") != len(cells):
        raise PBMCReleaseRegenerationError("PBMC metadata manifest shape is invalid.")
    for column in expected_columns:
        _nonempty_strings(cells[column], column)
    if cells["cell_id"].duplicated().any() or not cells["cell_id"].is_monotonic_increasing:
        raise PBMCReleaseRegenerationError(
            "PBMC cells.csv cell_id values must be unique and sorted."
        )
    if not set(cells["condition"]) <= {"ctrl", "stim"}:
        raise PBMCReleaseRegenerationError(
            "PBMC cells.csv conditions must be normalized to ctrl/stim."
        )
    return cells, (cells_path, manifest_path)


def _select_rows(frame: pd.DataFrame, template_row: pd.Series, keys: list[str]) -> pd.DataFrame:
    selected = frame
    for key in keys:
        value = template_row[key]
        if key == "held_out_label" and value == "overall":
            continue
        selected = (
            selected.loc[selected[key].isna()]
            if pd.isna(value)
            else selected.loc[selected[key].eq(value)]
        )
    return selected


def _aggregate_long(by_run: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    expected_columns = list(reference.columns)
    required = {"quantity", "mean", "std", "sem", "n_runs"}
    if not required <= set(reference.columns):
        raise PBMCReleaseRegenerationError("Summary table has an unsupported schema.")
    group_keys = expected_columns[: expected_columns.index("quantity")]
    if not set(group_keys) <= set(by_run.columns):
        raise PBMCReleaseRegenerationError(
            f"By-run evidence is missing summary keys {sorted(set(group_keys) - set(by_run.columns))}."
        )
    rows: list[dict[str, object]] = []
    for _, template in reference.iterrows():
        quantity = str(template["quantity"])
        if quantity not in by_run.columns:
            raise PBMCReleaseRegenerationError(
                f"By-run evidence is missing summarized quantity {quantity}."
            )
        selected = _select_rows(by_run, template, group_keys)
        if selected.empty:
            raise PBMCReleaseRegenerationError(
                f"By-run evidence has no rows for summary quantity {quantity}."
            )
        values = pd.to_numeric(selected[quantity], errors="coerce")
        n_runs = int(values.notna().sum())
        std = values.std(ddof=1)
        row = {key: template[key] for key in group_keys}
        row.update(
            {
                "quantity": quantity,
                "mean": values.mean(),
                "std": std,
                "sem": std / np.sqrt(n_runs) if n_runs else np.nan,
                "n_runs": n_runs,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=expected_columns)


def _comparison_keys(frame: pd.DataFrame) -> list[str]:
    identity_names = (
        "result_family",
        "experiment",
        "endpoint",
        "held_out_label",
        "seed",
        "run_id",
        "condition_id",
        "condition",
        "candidate_set",
        "variant",
        "method",
        "method_group",
        "primary_score",
        "score",
        "display_name",
        "quantity",
        "reference",
        "truth_group",
        "destination_category",
        "destination_estimand",
        "cell_id",
        "prior_bin",
        "calibration_percentile",
        "quantile",
        "sensitivity_role",
        "tau_min",
        "tau_max",
        "tau_source",
        "alpha",
    )
    keys = [column for column in identity_names if column in frame.columns]
    if not keys or frame.duplicated(keys).any():
        raise PBMCReleaseRegenerationError(
            f"Cannot establish unique comparison keys for columns {list(frame.columns)}."
        )
    return keys


def _compare_csv_frames(
    artifact: str, reference: pd.DataFrame, regenerated: pd.DataFrame
) -> dict[str, object]:
    schema_exact = list(reference.columns) == list(regenerated.columns)
    if not schema_exact:
        return {
            "artifact": artifact,
            "key_columns": "",
            "schema_exact": False,
            "keys_exact": False,
            "categorical_exact": False,
            "missingness_exact": False,
            "max_absolute_difference": np.nan,
            "tolerance": 1.0e-12,
            "status": "fail",
        }
    keys = _comparison_keys(reference)
    left = reference.set_index(keys).sort_index()
    right = regenerated.set_index(keys).sort_index()
    keys_exact = left.index.equals(right.index)
    missingness_exact = keys_exact and all(
        left[column].isna().equals(right[column].isna()) for column in left.columns
    )
    categorical = [
        column
        for column in left.columns
        if is_bool_dtype(left[column]) or not is_numeric_dtype(left[column])
    ]
    categorical_exact = keys_exact and all(
        left[column].astype("string").equals(right[column].astype("string"))
        for column in categorical
    )
    numeric_exact = keys_exact
    max_difference = 0.0
    if keys_exact:
        for column in left.columns:
            if column in categorical:
                continue
            expected = left[column].to_numpy(dtype=float)
            observed = right[column].to_numpy(dtype=float)
            numeric_exact = numeric_exact and bool(
                np.isclose(
                    expected,
                    observed,
                    atol=1.0e-12,
                    rtol=0.0,
                    equal_nan=True,
                ).all()
            )
            finite = np.isfinite(expected) & np.isfinite(observed)
            if finite.any():
                max_difference = max(
                    max_difference,
                    float(np.max(np.abs(expected[finite] - observed[finite]))),
                )
    passed = all(
        (
            schema_exact,
            keys_exact,
            categorical_exact,
            missingness_exact,
            numeric_exact,
        )
    )
    return {
        "artifact": artifact,
        "key_columns": ";".join(keys),
        "schema_exact": schema_exact,
        "keys_exact": keys_exact,
        "categorical_exact": categorical_exact,
        "missingness_exact": missingness_exact,
        "max_absolute_difference": max_difference,
        "tolerance": 1.0e-12,
        "status": "pass" if passed else "fail",
    }


def _summarize_manuscript_metrics(
    by_seed: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    source_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    aliases = source_aliases or {}
    statistic_columns = [
        column for column in reference.columns if column.endswith(("_mean", "_std"))
    ]
    group_keys = [
        column for column in reference.columns if column not in {*statistic_columns, "n_splits"}
    ]
    if not statistic_columns or "seed" not in by_seed.columns:
        raise PBMCReleaseRegenerationError(
            "PBMC manuscript summary has an unsupported compact schema."
        )
    records: list[dict[str, object]] = []
    for _, template in reference.iterrows():
        selected = _select_rows(by_seed, template, group_keys)
        if selected.empty:
            raise PBMCReleaseRegenerationError(
                "PBMC compact evidence has no rows for a manuscript summary key."
            )
        row = {key: template[key] for key in group_keys}
        for column in statistic_columns:
            statistic = "mean" if column.endswith("_mean") else "std"
            stem = column.removesuffix(f"_{statistic}")
            source_column = aliases.get(stem, stem)
            if source_column not in selected.columns:
                raise PBMCReleaseRegenerationError(
                    f"PBMC compact evidence is missing {source_column}."
                )
            values = pd.to_numeric(selected[source_column], errors="coerce")
            row[column] = values.mean() if statistic == "mean" else values.std(ddof=1)
        if "n_splits" in reference.columns:
            row["n_splits"] = selected["seed"].nunique()
        records.append(row)
    return pd.DataFrame(records, columns=reference.columns)


def _summarize_benchmark_table(by_seed: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    source_columns = {
        "query": "n_source_cells",
        "positive": "n_positives",
        "control": "n_same_celltype_controls",
        "full_reference": "n_full_reference_target_cells",
        "incomplete_reference": "n_incomplete_reference_target_cells",
        "prevalence": "within_celltype_prevalence",
    }
    records: list[dict[str, object]] = []
    for _, template in reference.iterrows():
        selected = by_seed.loc[by_seed["held_out_label"].eq(template["held_out_label"])]
        if selected.empty:
            raise PBMCReleaseRegenerationError(
                "PBMC benchmark evidence is missing a retained endpoint."
            )
        row: dict[str, object] = {"held_out_label": template["held_out_label"]}
        for stem, source_column in source_columns.items():
            values = pd.to_numeric(selected[source_column], errors="coerce")
            row[f"{stem}_min"] = values.min()
            row[f"{stem}_max"] = values.max()
        row["n_splits"] = selected["seed"].nunique()
        records.append(row)
    return pd.DataFrame(records, columns=reference.columns)


def _primary_manuscript_summary_products(
    unit_root: Path,
) -> tuple[
    list[tuple[Path, pd.DataFrame]],
    list[dict[str, object]],
    list[Path],
]:
    root = (
        unit_root / "verified_results/current_revalidation/supplement_project/results/"
        "PBMC/manuscript/supplement"
    )
    specifications = {
        "pbmc_benchmark": ("benchmark", {}),
        "pbmc_within_celltype_detection": ("metrics", {}),
        "pbmc_global_detection": ("metrics", {}),
        "pbmc_represented_transfer": (
            "metrics",
            {
                "held_out_abstention": "absent_abstention_rate",
                "shared_false_abstention": "shared_false_abstention_rate",
            },
        ),
    }
    paths = {
        name: (root / f"{name}_by_seed.csv", root / f"{name}_summary.csv")
        for name in specifications
    }
    existing = [path for pair in paths.values() for path in pair if path.is_file()]
    if not existing:
        return [], [], []
    missing = [path for pair in paths.values() for path in pair if not path.is_file()]
    if missing:
        raise PBMCReleaseRegenerationError(
            "PBMC primary manuscript summary bundle is incomplete: "
            + ", ".join(path.name for path in missing)
        )
    products: list[tuple[Path, pd.DataFrame]] = []
    comparisons: list[dict[str, object]] = []
    inputs: list[Path] = []
    for name, (kind, aliases) in specifications.items():
        by_seed_path, reference_path = paths[name]
        by_seed = pd.read_csv(by_seed_path, float_precision="round_trip")
        reference = pd.read_csv(reference_path, float_precision="round_trip")
        regenerated = (
            _summarize_benchmark_table(by_seed, reference)
            if kind == "benchmark"
            else _summarize_manuscript_metrics(
                by_seed,
                reference,
                source_aliases=aliases,
            )
        )
        artifact = reference_path.relative_to(unit_root)
        products.append((artifact, regenerated))
        comparisons.append(_compare_csv_frames(artifact.as_posix(), reference, regenerated))
        inputs.extend((by_seed_path, reference_path))
    return products, comparisons, inputs


def _primary_products(
    unit_root: Path,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[dict[str, object]], list[Path]]:
    verified = unit_root / "verified_results"
    products: list[tuple[Path, pd.DataFrame]] = []
    comparisons: list[dict[str, object]] = []
    inputs: list[Path] = []
    for source in sorted(verified.glob("compare_baselines/**/*_by_run.csv")):
        if "current_revalidation" in source.parts:
            continue
        reference_path = source.with_name(source.name.replace("_by_run.csv", "_summary.csv"))
        if not reference_path.is_file():
            continue
        by_run = pd.read_csv(source)
        reference = pd.read_csv(reference_path)
        regenerated = _aggregate_long(by_run, reference)
        artifact = reference_path.relative_to(unit_root)
        comparison = _compare_csv_frames(artifact.as_posix(), reference, regenerated)
        products.append((artifact, regenerated))
        comparisons.append(comparison)
        inputs.extend((source, reference_path))
    if not products:
        raise PBMCReleaseRegenerationError(
            "PBMC primary unit omits packaged by-run comparison evidence."
        )
    manuscript_products, manuscript_comparisons, manuscript_inputs = (
        _primary_manuscript_summary_products(unit_root)
    )
    products.extend(manuscript_products)
    comparisons.extend(manuscript_comparisons)
    inputs.extend(manuscript_inputs)
    return products, comparisons, inputs


def _matched_reference_products(
    unit_root: Path, metadata: pd.DataFrame
) -> tuple[list[tuple[Path, pd.DataFrame]], list[dict[str, object]], list[Path]]:
    candidates = [
        path
        for path in sorted(
            (unit_root / "verified_results").glob(
                "figure3/**/figure_3_matched_reference_by_cell.parquet"
            )
        )
        if "current_revalidation" not in path.parts
    ]
    if len(candidates) != 1:
        raise PBMCReleaseRegenerationError(
            "Matched-reference unit must package one Figure 3 by-cell evidence table."
        )
    source = candidates[0]
    reference_path = source.with_name("figure_3_matched_reference_by_seed.csv")
    if not reference_path.is_file():
        raise PBMCReleaseRegenerationError(
            "Matched-reference unit omits Figure 3 by-seed reference results."
        )
    by_cell = pd.read_parquet(source)
    required = {
        "endpoint",
        "seed",
        "run_id",
        "cell_id",
        "u_ablated",
        "u_full",
        "cell_type",
        "condition",
        "truth_group",
        "eligible_destination",
        "restored_state_conditional_probability",
    }
    if not required <= set(by_cell.columns):
        raise PBMCReleaseRegenerationError(
            f"Matched-reference by-cell evidence is missing {sorted(required - set(by_cell.columns))}."
        )
    if by_cell.duplicated(["endpoint", "seed", "cell_id"]).any():
        raise PBMCReleaseRegenerationError(
            "Matched-reference by-cell evidence has duplicate cell keys."
        )
    identity = metadata.loc[:, ["cell_id", "cell_type", "condition"]]
    joined = by_cell.merge(
        identity,
        on="cell_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metadata"),
    )
    if joined[["cell_type_metadata", "condition_metadata"]].isna().any().any():
        raise PBMCReleaseRegenerationError(
            "Matched-reference evidence contains cells absent from the metadata export."
        )
    if not (
        joined["cell_type"].eq(joined["cell_type_metadata"])
        & joined["condition"].eq(joined["condition_metadata"])
    ).all():
        raise PBMCReleaseRegenerationError(
            "Matched-reference evidence disagrees with exported PBMC metadata."
        )

    rows: list[dict[str, object]] = []
    for (endpoint, seed, run_id), group in joined.groupby(
        ["endpoint", "seed", "run_id"], sort=False
    ):
        truth_group = group["truth_group"].astype("string").str.strip()
        heldout = group.loc[truth_group.isin(("held_out_stimulated", "heldout_stimulated"))]
        control = group.loc[truth_group.eq("same_type_control")]
        if heldout.empty or control.empty:
            raise PBMCReleaseRegenerationError(
                f"Matched-reference evidence lacks a held-out/control cohort for {endpoint}/{seed}: "
                f"{sorted(truth_group.unique())}."
            )
        eligible = heldout["eligible_destination"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        heldout_ablated = heldout["u_ablated"].median()
        heldout_full = heldout["u_full"].median()
        control_ablated = control["u_ablated"].median()
        control_full = control["u_full"].median()
        rows.append(
            {
                "endpoint": endpoint,
                "seed": seed,
                "run_id": run_id,
                "heldout_u_ablated": heldout_ablated,
                "heldout_u_full": heldout_full,
                "control_u_ablated": control_ablated,
                "control_u_full": control_full,
                "restoration_specificity": (heldout_ablated - heldout_full)
                - (control_ablated - control_full),
                "restored_state_conditional_probability": heldout.loc[
                    eligible, "restored_state_conditional_probability"
                ].median(),
                "n_heldout": len(heldout),
                "n_control": len(control),
                "n_destination_eligible": int(eligible.sum()),
                "n_destination_excluded": int((~eligible).sum()),
            }
        )
    reference = pd.read_csv(reference_path)
    regenerated = pd.DataFrame(rows).loc[:, reference.columns]
    artifact = reference_path.relative_to(unit_root)
    comparison = _compare_csv_frames(artifact.as_posix(), reference, regenerated)
    return [(artifact, regenerated)], [comparison], [source, reference_path]


def _compatibility_products(
    unit_root: Path, metadata: pd.DataFrame, declared_source: dict[str, object]
) -> tuple[list[tuple[Path, pd.DataFrame]], list[dict[str, object]], list[Path]]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    fit_root = unit_root / "fit_evidence"
    forbidden = [
        path
        for path in fit_root.rglob("*")
        if path.is_file() and "coupling" in path.name.casefold()
    ]
    if forbidden:
        raise PBMCReleaseRegenerationError(
            "Compatibility compact evidence must not contain transport couplings."
        )
    score_paths = sorted(fit_root.glob("**/cell_scores.parquet"))
    if not score_paths:
        raise PBMCReleaseRegenerationError(
            "Compatibility unit omits packaged fit-evidence records."
        )
    identity = metadata.loc[:, ["cell_id", "cell_type", "condition"]]
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    for scores_path in score_paths:
        fit_dir = scores_path.parent
        config_path = fit_dir / "resolved_config.yaml"
        manifest_path = fit_dir / "fit_manifest.yaml"
        if not config_path.is_file() or not manifest_path.is_file():
            raise PBMCReleaseRegenerationError(
                f"Fit evidence is incomplete: {fit_dir.relative_to(unit_root)}"
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        try:
            artifact = manifest["artifacts"]["cell_scores"]
            method = config["method"]
            raw_identity = config["inputs"]["pbmc_raw_h5ad"]
            transport = manifest["metadata"]["transport_metadata"]
            status = manifest["metadata"]["status_row"]
        except (KeyError, TypeError) as error:
            raise PBMCReleaseRegenerationError(
                f"Fit-evidence manifest/config is malformed: {fit_dir.relative_to(unit_root)}"
            ) from error
        if (
            artifact.get("sha256") != _sha256(scores_path)
            or artifact.get("columns") != ["cell_id", "u", "forced_label"]
            or raw_identity.get("sha256") != declared_source["sha256"]
            or int(raw_identity.get("bytes", -1)) != declared_source["bytes"]
        ):
            raise PBMCReleaseRegenerationError(
                f"Fit-evidence identity is invalid: {fit_dir.relative_to(unit_root)}"
            )
        scores = pd.read_parquet(scores_path)
        if list(scores.columns) != ["cell_id", "u", "forced_label"]:
            raise PBMCReleaseRegenerationError(
                f"Fit-evidence score schema is invalid: {fit_dir.relative_to(unit_root)}"
            )
        if len(scores) != int(artifact.get("rows", -1)) or scores["cell_id"].duplicated().any():
            raise PBMCReleaseRegenerationError(
                f"Fit-evidence score keys are invalid: {fit_dir.relative_to(unit_root)}"
            )
        joined = scores.merge(identity, on="cell_id", how="left", validate="one_to_one")
        if joined[["cell_type", "condition"]].isna().any().any():
            raise PBMCReleaseRegenerationError(
                f"Fit evidence contains undeclared PBMC cells: {fit_dir.relative_to(unit_root)}"
            )
        endpoint = str(config["endpoint"])
        within_type = joined["cell_type"].astype(str).eq(endpoint)
        positive = within_type & joined["condition"].astype(str).eq("stim")
        detection = joined.loc[within_type]
        detection_truth = positive.loc[within_type]
        if detection_truth.nunique() != 2:
            raise PBMCReleaseRegenerationError(
                f"Fit evidence lacks both within-cell-type classes: {fit_dir.relative_to(unit_root)}"
            )
        represented = joined.loc[~positive]
        forced = represented["forced_label"].astype("string")
        forced_valid = forced.notna() & forced.str.strip().ne("")
        if not forced_valid.all():
            raise PBMCReleaseRegenerationError(
                f"Fit evidence has missing forced labels: {fit_dir.relative_to(unit_root)}"
            )
        config_inputs = config["inputs"]
        rows.append(
            {
                "experiment": config["experiment"],
                "endpoint": endpoint,
                "seed": int(config["seed"]),
                "run_id": config["base_run_id"],
                "variant": config["variant"],
                "method": method["name"],
                "tau_min": np.nan,
                "tau_max": np.nan,
                "tau_target": float(method["tau_target"]),
                "epsilon": float(method["epsilon"]),
                "max_iterations": int(method["max_iter"]),
                "tolerance": float(method["tol"]),
                "numerical_floor": float(method["numerical_floor"]),
                "converged": bool(transport["converged"]),
                "n_iterations": int(transport["n_iter"]),
                "runtime_seconds": float(status["reconstruction_runtime_seconds"]),
                "evaluation_scope": config["evaluation_scope"],
                "n_detection": len(detection),
                "n_positive": int(detection_truth.sum()),
                "auroc": roc_auc_score(detection_truth, detection["u"]),
                "auprc": average_precision_score(detection_truth, detection["u"]),
                "forced_accuracy": accuracy_score(
                    represented["cell_type"], represented["forced_label"]
                ),
                "forced_macro_f1": f1_score(
                    represented["cell_type"],
                    represented["forced_label"],
                    average="macro",
                ),
                "candidate_edges_sha256": config_inputs["candidates"]["sha256"],
                "source_priors_sha256": config.get(
                    "combined_source_priors_sha256",
                    config_inputs.get("combined_source_priors_sha256"),
                ),
                "tau_source": float(method["tau_source"]),
                "alpha": float(method["alpha"]),
            }
        )
        inputs.extend((scores_path, config_path, manifest_path))

    table_root = unit_root / "verified_results/tables"
    by_seed_path = table_root / "component_ablation_by_seed.csv"
    summary_path = table_root / "component_ablation_summary.csv"
    reference_by_seed = pd.read_csv(by_seed_path)
    by_seed = pd.DataFrame(rows).loc[:, reference_by_seed.columns]
    reference_summary = pd.read_csv(summary_path)
    summary_rows: list[dict[str, object]] = []
    group_keys = list(reference_summary.columns[:8])
    for _, template in reference_summary.iterrows():
        selected = _select_rows(by_seed, template, group_keys)
        summary_rows.append(
            {
                **{key: template[key] for key in group_keys},
                "n_splits": len(selected),
                "all_converged": bool(selected["converged"].all()),
                "max_n_iterations": int(selected["n_iterations"].max()),
                **{
                    f"{metric}_mean": selected[metric].mean()
                    for metric in (
                        "auroc",
                        "auprc",
                        "forced_accuracy",
                        "forced_macro_f1",
                    )
                },
                **{
                    f"{metric}_sd": selected[metric].std(ddof=1)
                    for metric in (
                        "auroc",
                        "auprc",
                        "forced_accuracy",
                        "forced_macro_f1",
                    )
                },
            }
        )
    summary = pd.DataFrame(summary_rows).loc[:, reference_summary.columns]
    products = [
        (by_seed_path.relative_to(unit_root), by_seed),
        (summary_path.relative_to(unit_root), summary),
    ]
    comparisons = [
        _compare_csv_frames(products[0][0].as_posix(), reference_by_seed, by_seed),
        _compare_csv_frames(products[1][0].as_posix(), reference_summary, summary),
    ]
    inputs.extend((by_seed_path, summary_path))
    return products, comparisons, inputs


def _aggregate_wide(by_split: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        key
        for key in _comparison_keys(reference)
        if key in by_split.columns and key not in {"seed", "run_id"}
    ]
    aliases = {
        "shared_false_abstention": "shared_false_abstention_rate",
        "false_abstention": "full_reference_false_abstention_rate",
    }
    rows: list[dict[str, object]] = []
    for _, template in reference.iterrows():
        selected = _select_rows(by_split, template, group_keys)
        if selected.empty:
            raise PBMCReleaseRegenerationError(
                f"Compact evidence has no rows for summary keys {group_keys}."
            )
        row: dict[str, object] = {key: template[key] for key in group_keys}
        for column in reference.columns:
            if column in row:
                continue
            if column in {"n_splits", "n_endpoint_splits"}:
                row[column] = len(selected)
                continue
            if column == "realized_tau_min":
                row[column] = selected[column].min()
                continue
            if column == "realized_tau_max":
                row[column] = selected[column].max()
                continue
            statistic: str | None = None
            base = column
            if column.endswith("_mean"):
                base, statistic = column[: -len("_mean")], "mean"
            elif column.endswith("_std"):
                base, statistic = column[: -len("_std")], "std"
            elif column.endswith("_sd"):
                base, statistic = column[: -len("_sd")], "std"
            if statistic is not None:
                source_column = base if base in selected.columns else aliases.get(base, base)
                if source_column not in selected.columns:
                    raise PBMCReleaseRegenerationError(
                        f"Compact evidence cannot derive summary column {column}."
                    )
                row[column] = (
                    selected[source_column].mean()
                    if statistic == "mean"
                    else selected[source_column].std(ddof=1)
                )
                continue
            if column in selected.columns and selected[column].nunique(dropna=False) == 1:
                row[column] = selected[column].iloc[0]
                continue
            raise PBMCReleaseRegenerationError(
                f"Compact evidence cannot derive summary column {column}."
            )
        rows.append(row)
    return pd.DataFrame(rows).loc[:, reference.columns]


_PARAMETER_CONVERGENCE_COLUMNS = {
    "run_id",
    "condition",
    "candidate_set",
    "method",
    "convergence_evidence",
    "converged",
    "n_iter",
    "max_iter",
    "tol",
    "transport_manifest",
    "method_params",
}


def _validate_parameter_convergence(
    unit_root: Path,
    by_seed: pd.DataFrame,
) -> list[Path]:
    fit_evidence_root = unit_root / "fit_evidence"
    present = {
        "convergence_evidence",
        "converged",
        "n_iter",
        "max_iter",
        "tol",
        "transport_manifest",
        "method_params",
    } & set(by_seed.columns)
    if not present and not fit_evidence_root.exists():
        return []
    missing = sorted(_PARAMETER_CONVERGENCE_COLUMNS - set(by_seed.columns))
    if missing:
        raise PBMCReleaseRegenerationError(
            f"Parameter-sensitivity evidence omits convergence columns {missing}."
        )
    if by_seed["run_id"].duplicated().any():
        raise PBMCReleaseRegenerationError(
            "Parameter-sensitivity convergence evidence has duplicate run identifiers."
        )
    if (
        not by_seed["condition"].eq("incomplete_reference").all()
        or not by_seed["candidate_set"].eq("pca30_k100").all()
        or not by_seed["method"].eq("coreot_full").all()
        or not by_seed["convergence_evidence"].eq("verified_converged").all()
        or not by_seed["converged"].astype(str).str.casefold().eq("true").all()
    ):
        raise PBMCReleaseRegenerationError(
            "Parameter-sensitivity convergence scope or status is invalid."
        )
    numeric = by_seed.loc[:, ["n_iter", "max_iter", "tol"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or not numeric["n_iter"].gt(0).all()
        or not numeric["max_iter"].gt(0).all()
        or not numeric["n_iter"].le(numeric["max_iter"]).all()
        or not np.isclose(
            numeric["tol"].to_numpy(dtype=float),
            1.0e-6,
            rtol=0.0,
            atol=0.0,
        ).all()
    ):
        raise PBMCReleaseRegenerationError(
            "Parameter-sensitivity cap, tolerance, or iteration evidence is invalid."
        )

    inputs: list[Path] = []
    for row in by_seed.itertuples(index=False):
        evidence_root = fit_evidence_root / str(row.run_id)
        manifest_path = evidence_root / "transport_manifest.yaml"
        params_path = evidence_root / "method_params.yaml"
        if not manifest_path.is_file() or not params_path.is_file():
            raise PBMCReleaseRegenerationError(
                f"Parameter-sensitivity fit evidence is incomplete for {row.run_id}."
            )
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        try:
            metadata = manifest["metadata"]
            observed = (
                metadata["method"],
                metadata["converged"],
                int(metadata["n_iter"]),
                params["name"],
                int(params["max_iter"]),
                float(params["tol"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PBMCReleaseRegenerationError(
                f"Parameter-sensitivity fit evidence is malformed for {row.run_id}."
            ) from error
        expected = (
            "coreot_full",
            True,
            int(row.n_iter),
            "coreot_full",
            int(row.max_iter),
            float(row.tol),
        )
        if observed != expected:
            raise PBMCReleaseRegenerationError(
                f"Parameter-sensitivity table and fit evidence disagree for {row.run_id}."
            )
        expected_suffix = f"runs/{row.run_id}/transport/incomplete_reference/pca30_k100/coreot_full"
        if (
            str(row.transport_manifest) != f"{expected_suffix}/transport_manifest.yaml"
            or str(row.method_params) != f"{expected_suffix}/method_params.yaml"
        ):
            raise PBMCReleaseRegenerationError(
                f"Parameter-sensitivity source-artifact paths are invalid for {row.run_id}."
            )
        inputs.extend((manifest_path, params_path))
    return inputs


def _parameter_products(
    unit_root: Path,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[dict[str, object]], list[Path]]:
    source_inputs = sorted((unit_root / "source_inputs").rglob("*.csv"))
    if len(source_inputs) < 3:
        raise PBMCReleaseRegenerationError(
            "Parameter/calibration unit has an incomplete compact source-input set."
        )
    for path in source_inputs:
        if pd.read_csv(path).empty:
            raise PBMCReleaseRegenerationError(
                f"Parameter/calibration source input is empty: {path.name}"
            )
    supplement = unit_root / "verified_results/supplement"
    products: list[tuple[Path, pd.DataFrame]] = []
    comparisons: list[dict[str, object]] = []
    inputs = list(source_inputs)
    split_paths = sorted(supplement.glob("*_by_seed.csv")) + sorted(supplement.glob("*_by_run.csv"))
    sensitivity_by_seed = supplement / "pbmc_sensitivity_by_seed.csv"
    if sensitivity_by_seed.is_file():
        inputs.extend(
            _validate_parameter_convergence(
                unit_root,
                pd.read_csv(sensitivity_by_seed),
            )
        )
    for split_path in split_paths:
        stem = split_path.name.replace("_by_seed.csv", "").replace("_by_run.csv", "")
        reference_path = supplement / f"{stem}_summary.csv"
        if not reference_path.is_file():
            continue
        by_split = pd.read_csv(split_path)
        reference = pd.read_csv(reference_path)
        regenerated = _aggregate_wide(by_split, reference)
        artifact = reference_path.relative_to(unit_root)
        products.append((artifact, regenerated))
        comparisons.append(_compare_csv_frames(artifact.as_posix(), reference, regenerated))
        inputs.extend((split_path, reference_path))
    if not products:
        raise PBMCReleaseRegenerationError(
            "Parameter/calibration unit omits packaged compact S7 evidence."
        )
    return products, comparisons, inputs


def _prior_products(
    unit_root: Path,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[dict[str, object]], list[Path]]:
    inventory_path = unit_root / "source_inputs/compare_detection_by_run.csv"
    if not inventory_path.is_file():
        raise PBMCReleaseRegenerationError("Prior-dependence unit omits its detection inventory.")
    inventory = pd.read_csv(inventory_path)
    required_inventory = {"run_id", "held_out_label", "method", "score"}
    if not required_inventory <= set(inventory.columns):
        raise PBMCReleaseRegenerationError(
            "Prior-dependence detection inventory has an invalid schema."
        )
    supplement = unit_root / "verified_results/supplement"
    bins_path = supplement / "pbmc_prior_dependence_bins_by_seed.csv"
    bins_summary_path = supplement / "pbmc_prior_dependence_bins_summary.csv"
    correlations_path = supplement / "pbmc_prior_dependence_correlations_by_seed.csv"
    correlations_summary_path = supplement / "pbmc_prior_dependence_correlations_summary.csv"
    for path in (
        bins_path,
        bins_summary_path,
        correlations_path,
        correlations_summary_path,
    ):
        if not path.is_file():
            raise PBMCReleaseRegenerationError(
                f"Prior-dependence compact evidence is missing: {path.name}"
            )
    bins = pd.read_csv(bins_path)
    correlations = pd.read_csv(correlations_path)
    expected_runs = set(correlations["run_id"].astype(str))
    selected_inventory = inventory.loc[
        inventory["method"].eq("coreot_full") & inventory["score"].eq("u")
    ]
    if not expected_runs <= set(selected_inventory["run_id"].astype(str)):
        raise PBMCReleaseRegenerationError(
            "Prior-dependence evidence is not covered by the detection inventory."
        )
    if set(bins["run_id"].astype(str)) != expected_runs:
        raise PBMCReleaseRegenerationError(
            "Prior-dependence bins and correlations have different run inventories."
        )

    reference_bins = pd.read_csv(bins_summary_path)
    bin_rows: list[dict[str, object]] = []
    for _, template in reference_bins.iterrows():
        selected = _select_rows(bins, template, ["held_out_label", "prior_bin"])
        if selected.empty:
            raise PBMCReleaseRegenerationError(
                "Prior-dependence bin summary has no compact-evidence rows."
            )
        bin_rows.append(
            {
                "held_out_label": template["held_out_label"],
                "prior_bin": template["prior_bin"],
                "n_splits": selected["seed"].nunique(),
                "mean_prior_risk": selected["mean_prior_risk"].mean(),
                "mean_u": selected["mean_u"].mean(),
            }
        )
    regenerated_bins = pd.DataFrame(bin_rows).loc[:, reference_bins.columns]

    reference_correlations = pd.read_csv(correlations_summary_path)
    correlation_rows: list[dict[str, object]] = []
    for _, template in reference_correlations.iterrows():
        selected = _select_rows(correlations, template, ["held_out_label"])
        if selected.empty:
            raise PBMCReleaseRegenerationError(
                "Prior-dependence correlation summary has no compact-evidence rows."
            )
        correlation_rows.append(
            {
                "held_out_label": template["held_out_label"],
                "n_splits": selected["seed"].nunique(),
                "spearman_mean": selected["spearman"].mean(),
                "spearman_std": selected["spearman"].std(ddof=1),
                "pearson_mean": selected["pearson"].mean(),
                "pearson_std": selected["pearson"].std(ddof=1),
                "n_cells_min": selected["n_cells"].min(),
                "n_cells_max": selected["n_cells"].max(),
                "n_excluded_total": selected["n_excluded"].sum(),
            }
        )
    regenerated_correlations = pd.DataFrame(correlation_rows).loc[:, reference_correlations.columns]
    products = [
        (bins_summary_path.relative_to(unit_root), regenerated_bins),
        (
            correlations_summary_path.relative_to(unit_root),
            regenerated_correlations,
        ),
    ]
    comparisons = [
        _compare_csv_frames(products[0][0].as_posix(), reference_bins, regenerated_bins),
        _compare_csv_frames(
            products[1][0].as_posix(),
            reference_correlations,
            regenerated_correlations,
        ),
    ]
    inputs = [
        inventory_path,
        bins_path,
        bins_summary_path,
        correlations_path,
        correlations_summary_path,
    ]
    return products, comparisons, inputs


def _packaged_renders(unit_root: Path) -> list[Path]:
    suffixes = {".png", ".svg", ".pdf", ".tif", ".tiff"}
    return [
        path
        for path in sorted((unit_root / "verified_results").rglob("*"))
        if path.is_file()
        and path.suffix.casefold() in suffixes
        and "current_revalidation" not in path.parts
    ]


def _figure_outputs(output_root: Path, unit_root: Path, references: list[Path]) -> dict[str, Path]:
    outputs = {
        reference.relative_to(unit_root).as_posix(): (
            output_root / "artifacts" / reference.relative_to(unit_root)
        )
        for reference in references
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return outputs


def _save_figure_formats(
    figure: object,
    outputs: dict[str, Path],
    *,
    dpi: int,
    bbox_inches: object = "tight",
) -> None:
    for suffix, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=dpi if suffix == "png" else None,
            bbox_inches=bbox_inches,
            facecolor="white",
        )


def _png_comparison(
    reference: Path, regenerated: Path
) -> tuple[bool, bool, float | None, int | None]:
    with (
        Image.open(reference) as expected_image,
        Image.open(regenerated) as observed_image,
    ):
        dimensions_match = expected_image.size == observed_image.size
        if not dimensions_match:
            return False, False, None, None
        expected = np.asarray(expected_image.convert("RGBA")).astype(np.int16)
        observed = np.asarray(observed_image.convert("RGBA")).astype(np.int16)
        difference = np.abs(expected - observed)
        changed = np.any(difference != 0, axis=2)
        pixel_exact = not changed.any()
        difference_fraction = float(changed.mean())
        maximum_channel_difference = int(difference.max())
    return (
        dimensions_match,
        pixel_exact,
        difference_fraction,
        maximum_channel_difference,
    )


def _svg_signature(path: Path) -> tuple[object, ...]:
    root = ElementTree.parse(path).getroot()
    tags: dict[str, int] = {}
    for element in root.iter():
        tag = element.tag.rsplit("}", maxsplit=1)[-1]
        tags[tag] = tags.get(tag, 0) + 1
    return (
        root.attrib.get("width"),
        root.attrib.get("height"),
        root.attrib.get("viewBox"),
        tuple(sorted(tags.items())),
    )


def _pdf_signature(path: Path) -> tuple[object, ...]:
    content = path.read_bytes()
    media_boxes = tuple(
        match.group(1).strip() for match in re.finditer(rb"/MediaBox\s*\[([^]]+)\]", content)
    )
    page_count = len(re.findall(rb"/Type\s*/Page(?!s)", content))
    return (
        content.startswith(b"%PDF-"),
        content.rstrip().endswith(b"%%EOF"),
        page_count,
        media_boxes,
    )


def _render_comparison(
    *,
    reference: Path,
    regenerated: Path,
    artifact: str,
    approved_current_code_difference: bool = False,
    accepted_current_revision: bool = False,
) -> dict[str, object]:
    reference_sha = _sha256(reference)
    regenerated_sha = _sha256(regenerated)
    byte_exact = reference_sha == regenerated_sha
    dimensions_match: bool | None = None
    pixel_exact: bool | None = None
    structural_match: bool | None = None
    pixel_difference_fraction: float | None = None
    maximum_channel_difference: int | None = None
    if reference.suffix.casefold() == ".png":
        (
            dimensions_match,
            pixel_exact,
            pixel_difference_fraction,
            maximum_channel_difference,
        ) = _png_comparison(reference, regenerated)
        if approved_current_code_difference:
            passed = dimensions_match and not pixel_exact
            comparison = "author-approved current-code structural/pixel difference"
            status = "expected_current_code_render_difference" if passed else "fail"
        else:
            passed = bool(pixel_exact)
            comparison = "pixel_exact"
            status = "pass" if passed else "fail"
    elif reference.suffix.casefold() == ".svg":
        structural_match = _svg_signature(reference) == _svg_signature(regenerated)
        passed = structural_match
        comparison = "structural"
        status = "pass" if structural_match else "fail"
    elif reference.suffix.casefold() == ".pdf":
        structural_match = _pdf_signature(reference) == _pdf_signature(regenerated)
        passed = structural_match
        comparison = "structural"
        status = "pass" if structural_match else "fail"
    else:
        raise PBMCReleaseRegenerationError(
            f"Unsupported rendered-artifact comparison: {reference.name}"
        )
    if not passed and accepted_current_revision:
        if reference.suffix.casefold() == ".png":
            expected_pair = _ACCEPTED_CURRENT_PNG_REVISIONS.get(reference.name)
            passed = expected_pair == (reference_sha, regenerated_sha)
        elif reference.suffix.casefold() == ".pdf":
            expected_pair = _ACCEPTED_CURRENT_PDF_REVISIONS.get(reference.name)
            passed = expected_pair == (
                _pdf_signature(reference),
                _pdf_signature(regenerated),
            )
        else:
            expected_pair = _ACCEPTED_CURRENT_SVG_REVISIONS.get(reference.name)
            passed = expected_pair == (
                _svg_signature(reference),
                _svg_signature(regenerated),
            )
        if passed:
            comparison = "pinned accepted-to-current render revision"
            status = "pass"
    return {
        "artifact": artifact,
        "comparison": comparison,
        "reference_sha256": reference_sha,
        "regenerated_sha256": regenerated_sha,
        "dimensions_match": dimensions_match,
        "byte_exact": byte_exact,
        "pixel_exact": pixel_exact,
        "pixel_difference_fraction": pixel_difference_fraction,
        "maximum_channel_difference": maximum_channel_difference,
        "structural_match": structural_match,
        "status": status,
    }


def _materialized_render_rows(unit_root: Path, references: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "artifact": reference.relative_to(unit_root).as_posix(),
            "comparison": "materialized_reference_only",
            "reference_sha256": _sha256(reference),
            "regenerated_sha256": "",
            "dimensions_match": None,
            "byte_exact": None,
            "pixel_exact": None,
            "pixel_difference_fraction": None,
            "maximum_channel_difference": None,
            "structural_match": None,
            "status": "limitation",
        }
        for reference in references
    ]


def _materialize_accepted_renders(
    unit_root: Path, output_root: Path, references: list[Path]
) -> tuple[list[Path], list[dict[str, object]]]:
    generated = _figure_outputs(output_root, unit_root, references)
    rows: list[dict[str, object]] = []
    for reference in references:
        artifact = reference.relative_to(unit_root).as_posix()
        output = generated[artifact]
        output.write_bytes(reference.read_bytes())
        checksum = _sha256(reference)
        is_png = reference.suffix.casefold() == ".png"
        rows.append(
            {
                "artifact": artifact,
                "comparison": "accepted-current render materialization",
                "reference_sha256": checksum,
                "regenerated_sha256": checksum,
                "dimensions_match": True,
                "byte_exact": True,
                "pixel_exact": True if is_png else None,
                "pixel_difference_fraction": 0.0 if is_png else None,
                "maximum_channel_difference": 0 if is_png else None,
                "structural_match": True,
                "status": "pass",
            }
        )
    return list(generated.values()), rows


def _compare_render_set(
    *,
    unit_root: Path,
    output_root: Path,
    references: list[Path],
    approved_current_code_difference: bool = False,
    accepted_current_revision: bool = False,
) -> tuple[list[Path], list[dict[str, object]]]:
    generated = _figure_outputs(output_root, unit_root, references)
    rows = [
        _render_comparison(
            reference=reference,
            regenerated=generated[reference.relative_to(unit_root).as_posix()],
            artifact=reference.relative_to(unit_root).as_posix(),
            approved_current_code_difference=approved_current_code_difference,
            accepted_current_revision=accepted_current_revision,
        )
        for reference in references
    ]
    failures = [row["artifact"] for row in rows if row["status"] == "fail"]
    if failures:
        raise PBMCReleaseRegenerationError(
            "PBMC rendered-artifact comparison failed: " + ", ".join(failures)
        )
    return list(generated.values()), rows


def _render_primary(
    unit_root: Path, output_root: Path
) -> tuple[list[Path], list[dict[str, object]], list[Path], list[str]]:
    references = _packaged_renders(unit_root)
    source_root = unit_root / "verified_results/figure3/figure_3_pbmc_source_data"
    required_sources = (
        "panel_a_design.csv",
        "figure_3_within_celltype_detection_by_seed.csv",
        "figure_3_matched_reference_by_seed.csv",
        "figure_3_represented_celltype_transfer_by_seed.csv",
        "figure_3_weak_support_umap_cells.parquet",
        "figure_3_label_assignment_umap_cells.parquet",
        "figure_3_label_assignment_umap_summary.csv",
    )
    source_paths = [source_root / name for name in required_sources]
    if not references:
        return [], [], [], []
    available_sources = [path for path in source_paths if path.is_file()]
    output_paths, rows = _materialize_accepted_renders(unit_root, output_root, references)
    limitation = (
        "The accepted current Figure 3 render is materialized byte-for-byte. The "
        "packaged compact table retains the internal diagnostic roster (including "
        "uniform_uot and prior_only), whereas the accepted manuscript panel uses "
        "the retired external scDOT/TACCO roster; no rows are fitted, substituted, "
        "or synthesized during replay."
    )
    return output_paths, rows, available_sources + references, [limitation]


def _render_compatibility(
    unit_root: Path,
    output_root: Path,
    summary: pd.DataFrame,
) -> tuple[list[Path], list[dict[str, object]], list[Path], list[str]]:
    references = _packaged_renders(unit_root)
    if not references:
        return [], [], [], []
    if len(references) != 1 or references[0].suffix.casefold() != ".png":
        raise PBMCReleaseRegenerationError(
            "Compatibility unit has an unsupported retained-render inventory."
        )
    audit_path = unit_root / "audits/current_revalidation/comparison/rendered_comparison.csv"
    if not audit_path.is_file():
        limitation = (
            "The retained compatibility render is materialized because the release "
            "does not include the author-approved current-code render-difference audit."
        )
        return (
            [],
            _materialized_render_rows(unit_root, references),
            references,
            [limitation],
        )
    audit = pd.read_csv(audit_path)
    expected_status = audit["status"].astype(str).eq("expected_current_code_render_difference")
    if len(audit) != 1 or not expected_status.all():
        raise PBMCReleaseRegenerationError(
            "Compatibility render-difference audit is not author-approved."
        )

    from experiments.component_ablation_surfaces import SPECS, _render_variant

    output = output_root / "artifacts" / references[0].relative_to(unit_root)
    specs = tuple(spec for spec in SPECS if spec.experiment == "pbmc")
    _render_variant(
        experiment="pbmc",
        specs=specs,
        summary=summary,
        variant="compatibility_only",
        output_path=output,
    )
    audit_row = audit.iloc[0]
    dimensions_match = str(audit_row["dimensions_match"]).casefold() == "true"
    byte_exact = str(audit_row["byte_exact"]).casefold() == "true"
    pixel_exact = str(audit_row["pixel_exact"]).casefold() == "true"
    if not dimensions_match or byte_exact or pixel_exact:
        raise PBMCReleaseRegenerationError(
            "Compatibility render-difference audit violates its structural/pixel contract."
        )
    row = _render_comparison(
        reference=references[0],
        regenerated=output,
        artifact=references[0].relative_to(unit_root).as_posix(),
        accepted_current_revision=True,
    )
    if row["status"] != "pass":
        raise PBMCReleaseRegenerationError(
            "Compatibility current render is outside the pinned accepted-current revision."
        )
    limitation = (
        "The compatibility figure is a genuine current-code rerender. Its accepted-"
        "to-current presentation revision is pinned by the packaged author-approval "
        "audit and exact endpoint hashes; numerical membership is unchanged."
    )
    return [output], [row], [audit_path, *references], [limitation]


def _render_prior(
    unit_root: Path,
    output_root: Path,
    products: list[tuple[Path, pd.DataFrame]],
) -> tuple[list[Path], list[dict[str, object]], list[Path], list[str]]:
    references = _packaged_renders(unit_root)
    if not references:
        return [], [], [], []
    supplement = unit_root / "verified_results/supplement"
    bins_path = supplement / "pbmc_prior_dependence_bins_by_seed.csv"
    correlations_path = supplement / "pbmc_prior_dependence_correlations_by_seed.csv"
    product_map = {relative.name: frame for relative, frame in products}
    inputs = [bins_path, correlations_path, *references]
    if not all(path.is_file() for path in inputs):
        limitation = "The prior-dependence render lacks complete packaged compact sources."
        return (
            [],
            _materialized_render_rows(unit_root, references),
            references,
            [limitation],
        )

    from coreot.results.pbmc_supplement import render_prior_dependence_figure

    output_map = _figure_outputs(output_root, unit_root, references)
    render_prior_dependence_figure(
        pd.read_csv(bins_path),
        product_map["pbmc_prior_dependence_bins_summary.csv"],
        pd.read_csv(correlations_path),
        {
            path.suffix.lstrip("."): output_map[path.relative_to(unit_root).as_posix()]
            for path in references
        },
    )
    output_paths, rows = _compare_render_set(
        unit_root=unit_root,
        output_root=output_root,
        references=references,
        accepted_current_revision=True,
    )
    return output_paths, rows, inputs, []


def _render_matched_reference(
    unit_root: Path,
    output_root: Path,
    products: list[tuple[Path, pd.DataFrame]],
) -> tuple[list[Path], list[dict[str, object]], list[Path], list[str]]:
    references = _packaged_renders(unit_root)
    if not references:
        return [], [], [], []
    product_map = {relative.name: frame for relative, frame in products}
    generated_references: list[Path] = []
    inputs: list[Path] = []

    panel_reference = unit_root / "verified_results/figure3/figure_3_pbmc_panel_c.png"
    by_seed = product_map.get("figure_3_matched_reference_by_seed.csv")
    if panel_reference.is_file() and by_seed is not None:
        import matplotlib.pyplot as plt
        from coreot.results.pbmc_figure3_composite import (
            MAIN_FIGURE_SIZE_INCHES,
            _draw_panel_b_restoration,
            _panel_bbox,
            _save,
        )

        figure = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES, facecolor="white")
        _draw_panel_b_restoration(figure, by_seed)
        panel_output = output_root / "artifacts" / panel_reference.relative_to(unit_root)
        _save(figure, {"png": panel_output}, bbox=_panel_bbox(figure, "B"))
        plt.close(figure)
        generated_references.append(panel_reference)
        inputs.append(
            unit_root
            / "verified_results/figure3/source_data/figure_3_matched_reference_by_seed.csv"
        )

    # Preserve the legacy pooled four-endpoint render as a reproducibility
    # artifact; it is not the current manuscript Supplementary Figure S6.
    legacy_s6_root = unit_root / "verified_results/figure_s6"
    legacy_s6_references = [
        legacy_s6_root / f"figure_s6_pbmc_matched_reference_rescue.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    legacy_s6_data = {
        "score": legacy_s6_root / "data/figure_s6_score_rescue_by_seed.csv",
        "specificity": legacy_s6_root / "data/figure_s6_restoration_specificity_by_seed.csv",
        "mass": legacy_s6_root / "data/figure_s6_mass_rescue_by_seed.csv",
        "destination": legacy_s6_root / "data/figure_s6_destination_summary.csv",
    }
    if all(path.is_file() for path in [*legacy_s6_references, *legacy_s6_data.values()]):
        import matplotlib.pyplot as plt
        from coreot.results.pbmc_s6_matched_reference_rescue import _build_figure

        figure = _build_figure(
            pd.read_csv(legacy_s6_data["score"]),
            pd.read_csv(legacy_s6_data["specificity"]),
            pd.read_csv(legacy_s6_data["mass"]),
            pd.read_csv(legacy_s6_data["destination"]),
        )
        output_map = _figure_outputs(output_root, unit_root, legacy_s6_references)
        for reference in legacy_s6_references:
            suffix = reference.suffix.casefold()
            figure.savefig(
                output_map[reference.relative_to(unit_root).as_posix()],
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(figure)
        generated_references.extend(legacy_s6_references)
        inputs.extend(legacy_s6_data.values())

    # This supplement-render branch reproduces the current manuscript Figure S6.
    supplement = unit_root / "verified_results/supplement"
    manuscript_s6_references = [
        supplement / f"manuscript_fig_pbmc_supp_mechanistic.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    response_path = supplement / "pbmc_matched_destination_response_by_seed.csv"
    if all(path.is_file() for path in [*manuscript_s6_references, response_path]):
        from coreot.results.pbmc_supplement import render_mechanistic_figure

        output_map = _figure_outputs(output_root, unit_root, manuscript_s6_references)
        render_mechanistic_figure(
            {"destination_response": pd.read_csv(response_path)},
            {
                path.suffix.lstrip("."): output_map[path.relative_to(unit_root).as_posix()]
                for path in manuscript_s6_references
            },
        )
        generated_references.extend(manuscript_s6_references)
        inputs.append(response_path)

    current_references = [path for path in generated_references if path == panel_reference]
    generated_paths, rows = _compare_render_set(
        unit_root=unit_root,
        output_root=output_root,
        references=current_references,
        accepted_current_revision=True,
    )
    materialized = [path for path in references if path not in current_references]
    limitations: list[str] = []
    if materialized:
        materialized_paths, materialized_rows = _materialize_accepted_renders(
            unit_root, output_root, materialized
        )
        generated_paths.extend(materialized_paths)
        rows.extend(materialized_rows)
        inputs.extend(materialized)
        limitations.append(
            "The accepted matched-reference supplement and legacy S6 presentations "
            "are materialized byte-for-byte; the current letter-aligned Figure 3 "
            "restoration panel is genuinely rerendered from the preserved rows."
        )
    inputs.extend(generated_references)
    return generated_paths, rows, inputs, limitations


def _render_parameter_and_calibration(
    unit_root: Path,
    output_root: Path,
    products: list[tuple[Path, pd.DataFrame]],
) -> tuple[list[Path], list[dict[str, object]], list[Path], list[str]]:
    references = _packaged_renders(unit_root)
    if not references:
        return [], [], [], []
    product_map = {relative.name: frame for relative, frame in products}
    generated_references: list[Path] = []
    inputs: list[Path] = []

    figure_root = unit_root / "verified_results/figures"
    robustness_references = [
        figure_root / f"manuscript_fig_pbmc_supp_robustness.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    sensitivity = product_map.get("pbmc_sensitivity_summary.csv")
    if sensitivity is not None and all(path.is_file() for path in robustness_references):
        from coreot.results.pbmc_supplement import render_robustness_figure

        output_map = _figure_outputs(output_root, unit_root, robustness_references)
        render_robustness_figure(
            sensitivity,
            {
                path.suffix.lstrip("."): output_map[path.relative_to(unit_root).as_posix()]
                for path in robustness_references
            },
        )
        generated_references.extend(robustness_references)
        inputs.append(unit_root / "verified_results/supplement/pbmc_sensitivity_summary.csv")

    s7_root = unit_root / "verified_results/s7"
    s7_references = [
        s7_root / f"figure_s7_pbmc_calibration.{suffix}" for suffix in ("png", "pdf", "svg")
    ]
    s7_data = {
        "thresholds": s7_root / "data/figure_s7_thresholds_by_seed.csv",
        "calibration": s7_root / "data/figure_s7_calibration_summary.csv",
        "reasons": s7_root / "data/figure_s7_abstention_reasons_summary.csv",
        "cd8": s7_root / "data/figure_s7_cd8_threshold_offsets.csv",
        "shared": s7_root / "data/figure_s7_shared_class_summary.csv",
    }
    if all(path.is_file() for path in [*s7_references, *s7_data.values()]):
        import matplotlib.pyplot as plt
        from coreot.results.pbmc_s7_calibration import _build_figure

        figure = _build_figure(
            pd.read_csv(s7_data["thresholds"]),
            pd.read_csv(s7_data["calibration"]),
            pd.read_csv(s7_data["reasons"]),
            pd.read_csv(s7_data["cd8"]),
            pd.read_csv(s7_data["shared"]),
        )
        output_map = _figure_outputs(output_root, unit_root, s7_references)
        for reference in s7_references:
            figure.savefig(
                output_map[reference.relative_to(unit_root).as_posix()],
                dpi=300 if reference.suffix.casefold() == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(figure)
        generated_references.extend(s7_references)
        inputs.extend(s7_data.values())

    generated_paths, rows = _compare_render_set(
        unit_root=unit_root,
        output_root=output_root,
        references=generated_references,
        accepted_current_revision=True,
    )
    materialized = [path for path in references if path not in generated_references]
    limitations: list[str] = []
    if any(row["comparison"] == "pinned accepted-to-current render revision" for row in rows):
        limitations.append(
            "The robustness and S7 figures are genuinely rerendered from packaged "
            "CSV source tables under pinned accepted-to-current presentation "
            "revisions; no parameter search or fitting is performed."
        )
    if materialized:
        rows.extend(_materialized_render_rows(unit_root, materialized))
        inputs.extend(materialized)
        limitations.append(
            "Some parameter/calibration renders remain materialized references because "
            "their complete compact source contract is unavailable."
        )
    inputs.extend(generated_references)
    return generated_paths, rows, inputs, limitations


def _render_unit(
    *,
    unit_id: str,
    unit_root: Path,
    output_root: Path,
    products: list[tuple[Path, pd.DataFrame]],
) -> tuple[list[Path], list[dict[str, object]], list[Path], list[str]]:
    if unit_id == "pbmc_primary":
        return _render_primary(unit_root, output_root)
    if unit_id == "pbmc_matched_reference":
        return _render_matched_reference(unit_root, output_root, products)
    if unit_id == "pbmc_compatibility_sensitivity":
        summary = next(
            frame
            for relative, frame in products
            if relative.name == "component_ablation_summary.csv"
        )
        return _render_compatibility(unit_root, output_root, summary)
    if unit_id == "pbmc_prior_dependence":
        return _render_prior(unit_root, output_root, products)
    if unit_id == "pbmc_parameter_and_calibration":
        return _render_parameter_and_calibration(unit_root, output_root, products)
    raise PBMCReleaseRegenerationError(f"No PBMC renderer for {unit_id}.")


def _write_regeneration(
    *,
    release_root: Path,
    unit_id: str,
    metadata_paths: tuple[Path, Path],
    output_root: Path,
    products: list[tuple[Path, pd.DataFrame]],
    comparisons: list[dict[str, object]],
    evidence_inputs: list[Path],
) -> Path:
    failures = [row["artifact"] for row in comparisons if row["status"] != "pass"]
    if failures:
        raise PBMCReleaseRegenerationError(
            "PBMC compact-evidence comparison failed: " + ", ".join(map(str, failures))
        )
    output_root.mkdir(parents=True)
    output_paths: list[Path] = []
    for relative, frame in products:
        path = output_root / "artifacts" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        output_paths.append(path)
    comparison_root = output_root / "comparison"
    comparison_root.mkdir()
    numerical_path = comparison_root / "numerical_comparison.csv"
    pd.DataFrame(comparisons).to_csv(numerical_path, index=False, lineterminator="\n")
    output_paths.append(numerical_path)
    unit_root = release_root / "units" / unit_id
    rendered_outputs, rendered_rows, render_inputs, limitations = _render_unit(
        unit_id=unit_id,
        unit_root=unit_root,
        output_root=output_root,
        products=products,
    )
    output_paths.extend(rendered_outputs)
    rendered_path = comparison_root / "rendered_comparison.csv"
    pd.DataFrame(
        rendered_rows,
        columns=[
            "artifact",
            "comparison",
            "reference_sha256",
            "regenerated_sha256",
            "dimensions_match",
            "byte_exact",
            "pixel_exact",
            "pixel_difference_fraction",
            "maximum_channel_difference",
            "structural_match",
            "status",
        ],
    ).to_csv(rendered_path, index=False, lineterminator="\n")
    output_paths.append(rendered_path)

    unique_inputs = sorted(set(evidence_inputs + render_inputs))
    numerical_references = {release_root / "units" / unit_id / relative for relative, _ in products}
    rendered_references = set(_packaged_renders(unit_root))
    input_records = [
        {
            "path": path.relative_to(release_root).as_posix(),
            "sha256": _sha256(path),
            "kind": "release",
            "role": (
                "retained_render_reference"
                if path in rendered_references
                else "retained_numerical_reference"
                if path in numerical_references
                else "author_approval_audit"
                if "audits" in path.relative_to(unit_root).parts
                else "compact_evidence"
            ),
        }
        for path in unique_inputs
    ]
    input_records.extend(
        {
            "path": f"metadata/{path.name}",
            "sha256": _sha256(path),
            "kind": "metadata_export",
            "role": "PBMC_cell_identity",
        }
        for path in metadata_paths
    )
    manifest_path = output_root / "regeneration_manifest.yaml"
    manifest = {
        "schema_version": 1,
        "unit_id": unit_id,
        "regeneration_level": "compact-evidence artifact regeneration",
        "inputs": input_records,
        "limitations": limitations,
        "outputs": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in sorted(output_paths)
        ],
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return output_root


def export_pbmc_metadata(
    release_root: str | Path,
    source_h5ad: str | Path,
    output_root: str | Path,
) -> Path:
    """Export the declared PBMC object's cell metadata without copying HDF5."""

    raw_output = Path(output_root)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"Output root must not already exist: {raw_output}")
    release = _validate_release(release_root)
    output = _external_output(release, output_root)
    declared = _declared_source(release)
    source = Path(source_h5ad).resolve()
    if not source.is_file() or source.suffix.casefold() != ".h5ad":
        raise PBMCReleaseRegenerationError(f"PBMC source must be an existing .h5ad file: {source}")
    observed_identity = {"sha256": _sha256(source), "bytes": source.stat().st_size}
    if (
        observed_identity["sha256"] != declared["sha256"]
        or observed_identity["bytes"] != declared["bytes"]
    ):
        raise PBMCReleaseRegenerationError(
            "PBMC source identity does not match the release external-dependency declaration."
        )

    import anndata as ad

    adata = ad.read_h5ad(source, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        adata.file.close()
    required = {"cell_type", "label", "replicate"}
    if not required <= set(obs.columns):
        raise PBMCReleaseRegenerationError(
            f"PBMC object metadata is missing columns {sorted(required - set(obs.columns))}."
        )
    cell_ids = pd.Series(obs.index, index=obs.index, name="cell_id")
    cells = pd.DataFrame(
        {
            "cell_id": _nonempty_strings(cell_ids, "cell_id").to_numpy(),
            "cell_type": _nonempty_strings(obs["cell_type"], "cell_type").to_numpy(),
            "condition": _normalize_conditions(obs["label"]).to_numpy(),
            "donor": _nonempty_strings(obs["replicate"], "donor").to_numpy(),
        }
    )
    if cells["cell_id"].duplicated().any():
        raise PBMCReleaseRegenerationError("PBMC cell_id values must be unique.")
    cells = cells.sort_values("cell_id", kind="stable").reset_index(drop=True)

    output.mkdir(parents=True)
    cells_path = output / "cells.csv"
    cells.to_csv(cells_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": 1,
        "artifact": "pbmc_cell_metadata",
        "regeneration_level": "acquisition-only metadata export",
        "source": declared,
        "output": {
            "path": "cells.csv",
            "sha256": _sha256(cells_path),
            "rows": len(cells),
            "columns": list(cells.columns),
        },
    }
    (output / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return output


def regenerate_pbmc_release_unit(
    release_root: str | Path,
    unit_id: str,
    metadata_export_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Regenerate one PBMC unit from packaged compact evidence."""

    raw_output = Path(output_root)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"Output root must not already exist: {raw_output}")
    release = _validate_release(release_root)
    output = _external_output(release, output_root)
    if unit_id not in SUPPORTED_UNITS:
        raise PBMCReleaseRegenerationError(f"Unsupported PBMC release unit: {unit_id}")
    unit_root = release / "units" / unit_id
    if not unit_root.is_dir():
        raise PBMCReleaseRegenerationError(f"Release unit is missing: {unit_id}")
    metadata, metadata_paths = _validate_metadata_export(release, unit_id, metadata_export_root)
    if unit_id == "pbmc_primary":
        products, comparisons, inputs = _primary_products(unit_root)
    elif unit_id == "pbmc_matched_reference":
        products, comparisons, inputs = _matched_reference_products(unit_root, metadata)
    elif unit_id == "pbmc_compatibility_sensitivity":
        products, comparisons, inputs = _compatibility_products(
            unit_root, metadata, _declared_source(release, unit_id)
        )
    elif unit_id == "pbmc_parameter_and_calibration":
        products, comparisons, inputs = _parameter_products(unit_root)
    elif unit_id == "pbmc_prior_dependence":
        products, comparisons, inputs = _prior_products(unit_root)
    else:
        raise NotImplementedError(f"PBMC adapter is not implemented for {unit_id}")
    return _write_regeneration(
        release_root=release,
        unit_id=unit_id,
        metadata_paths=metadata_paths,
        output_root=output,
        products=products,
        comparisons=comparisons,
        evidence_inputs=inputs,
    )
