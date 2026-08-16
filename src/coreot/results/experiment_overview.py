from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import anndata as ad

STAGE = "experiment-overview"


@dataclass(frozen=True)
class ExperimentOverviewPaths:
    output_root: Path
    csv_path: Path
    markdown_path: Path


def write_experiment_overview_table(
    *,
    raw_data_path: str | Path,
    output_root: str | Path,
    embedding_name: str = "hiha_harmony30",
    embedding_obsm_key: str = "X_pca_harmony",
    embedding_n_components: int = 30,
) -> ExperimentOverviewPaths:
    """Generate the experiment overview Table 1 (dataset summary and benchmark design).

    This table describes the experimental setup — dataset characteristics,
    annotation resolution, held-out labels, donor split, embedding, and prior
    variables — rather than result metrics.  It reads the raw ``.h5ad`` file to
    extract actual subject/cell/label counts.
    """
    raw = Path(raw_data_path)
    if not raw.is_file():
        raise FileNotFoundError(f"Raw data file not found: {raw}")

    adata = ad.read_h5ad(raw)

    n_cells = adata.n_obs
    n_subjects = int(adata.obs["subject.subjectGuid"].nunique())
    n_samples = int(adata.obs["sample.sampleKitGuid"].nunique())

    aifi_l2_classes = sorted(adata.obs["AIFI_L2"].dropna().unique().tolist())
    aifi_l2_counts = adata.obs["AIFI_L2"].value_counts()
    aifi_l2_label = ", ".join(
        f"{c} ({aifi_l2_counts[c]})" for c in aifi_l2_classes
    )

    aifi_l3_labels = sorted(
        adata.obs["AIFI_L3"].dropna().unique().tolist(),
        key=lambda x: adata.obs["AIFI_L3"].value_counts().get(x, 0),
        reverse=True,
    )
    aifi_l3_counts = adata.obs["AIFI_L3"].value_counts()
    aifi_l3_label = ", ".join(
        f"{c} ({aifi_l3_counts[c]})" for c in aifi_l3_labels
    )

    has_recomputed_score = "AIFI_L2_score_recomputed" in adata.obs.columns

    rows: list[dict[str, str]] = [
        {
            "characteristic": "Dataset",
            "value": "Human Immune Health Atlas (HIHA) dendritic cells",
        },
        {
            "characteristic": "Number of subjects",
            "value": str(n_subjects),
        },
        {
            "characteristic": "Number of samples",
            "value": f"{n_samples} (one sample per subject)",
        },
        {
            "characteristic": "Number of cells",
            "value": f"{n_cells:,}",
        },
        {
            "characteristic": "AIFI_L2 classes (broad anchor)",
            "value": f"{aifi_l2_label} — {len(aifi_l2_classes)} classes",
        },
        {
            "characteristic": "AIFI_L3 labels (high-resolution)",
            "value": f"{aifi_l3_label} — {len(aifi_l3_labels)} labels",
        },
        {
            "characteristic": "Main held-out labels",
            "value": "CD14+ cDC2, HLA-DRhi cDC2, ISG+ cDC2 "
            "(three cDC2 subtypes sharing the AIFI_L2 cDC2 parent)",
        },
        {
            "characteristic": "Donor split",
            "value": "80% reference / 20% query, donor-aware (subject-level); "
            "≥30 query cells of held-out label required, ≤100 resample attempts",
        },
        {
            "characteristic": "Query subsampling",
            "value": "≤500 cells per AIFI_L3 label (stratified)",
        },
        {
            "characteristic": "Reference subsampling",
            "value": "≤1,000 cells per AIFI_L3 label (stratified)",
        },
        {
            "characteristic": "Embedding",
            "value": f"Harmony-corrected PCA, {embedding_n_components} dimensions "
            f"(`{embedding_obsm_key}[:, :{embedding_n_components}]`); "
            f"provider name: `{embedding_name}`",
        },
        {
            "characteristic": "Broad anchor variable",
            "value": "AIFI_L2 (4 broad dendritic cell classes: "
            "ASDC, cDC1, cDC2, pDC)",
        },
        {
            "characteristic": "Matchability variable",
            "value": (
                "ρ = clip(AIFI_L2_score_recomputed, 0.05, 0.95); "
                "recomputed CellTypist Level-2 prediction score "
                "(provisional — official AIFI_L2_score unavailable "
                "in local HIHA object)"
                if has_recomputed_score
                else "AIFI_L2_score_recomputed not found in .obs"
            ),
        },
        {
            "characteristic": "Candidate set",
            "value": "Exact Euclidean k-NN (k=100) in the embedding space; "
            f"candidate set name: `{embedding_name}_k100`",
        },
        {
            "characteristic": "Transport methods",
            "value": "prior_only, nn, balanced_ot (dense), uniform_uot (sparse), "
            "coreot_constant_tau (sparse), coreot_full (sparse)",
        },
        {
            "characteristic": "Donor-split seeds",
            "value": "1, 2, 3, 4, 5 (5 repeats per held-out label × method)",
        },
        {
            "characteristic": "Conditions per run",
            "value": "incomplete_reference, full_reference_control (negative control "
            "for abstention threshold calibration)",
        },
    ]

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "table_1_experiment_overview.csv"
    _write_csv(csv_path, rows)

    markdown_path = output / "table_1_experiment_overview.md"
    markdown_path.write_text(_render_markdown(rows, csv_path), encoding="utf-8")

    return ExperimentOverviewPaths(
        output_root=output,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["characteristic", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, str]], csv_path: Path) -> str:
    lines = [
        "# Table 1: HIHA DC dataset summary and leave-one-cell-type-out benchmark design",
        "",
        "| Characteristic | Value |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['characteristic']} | {row['value']} |")

    lines.extend([
        "",
        f"_Source data: `{csv_path}`. Generated from raw `.h5ad` file._",
        "",
    ])
    return "\n".join(lines)
