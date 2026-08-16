from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from coreot.results.hiha_supplement import (
    EXPECTED_HELD_OUT,
    _complete_bin_means,
    _correlation_summary,
    _equal_frequency_bins,
    _plot_s3,
    _prior_correlation_row,
    _within_cdc2_rows,
)
from experiments.missing_celltype.generate_hiha_dc_supplementary_figures import (
    promote_hiha_s3_submission_copies,
)


def test_within_cdc2_rows_excludes_other_level2_cells() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["cdc2_1", "cdc2_2", "cdc1"],
            "AIFI_L2": ["cDC2", "cDC2", "cDC1"],
        }
    )

    result = _within_cdc2_rows(frame, "HLA-DRhi cDC2", 1)

    assert result["cell_id"].tolist() == ["cdc2_1", "cdc2_2"]


def test_equal_frequency_bins_preserve_prior_risk_ties() -> None:
    prior_risk = np.concatenate(
        [np.full(4, 0.1), np.linspace(0.2, 0.9, 36)]
    )
    frame = pd.DataFrame(
        {
            "cell_id": [f"cell_{index:02d}" for index in range(len(prior_risk))],
            "u": np.linspace(0.05, 0.4, len(prior_risk)),
            "prior_risk": prior_risk,
        }
    )

    result = pd.DataFrame(
        _equal_frequency_bins(frame, "HLA-DRhi cDC2", 1)
    )

    first = result.loc[result["prior_bin"].eq(1)].iloc[0]
    assert first["n_cells"] == 4
    assert first["mean_prior_risk"] == pytest.approx(0.1)
    assert 2 not in set(result["prior_bin"])
    assert result["n_cells"].sum() == len(frame)
    assert result["n_occupied_bins"].nunique() == 1
    assert result["n_occupied_bins"].iloc[0] < 20


def test_prior_correlation_uses_all_finite_pairs() -> None:
    frame = pd.DataFrame(
        {
            "u": [0.1, 0.2, 0.3, 0.4, np.nan],
            "prior_risk": [0.05, 0.15, 0.25, np.nan, 0.45],
        }
    )

    result = _prior_correlation_row(frame, "HLA-DRhi cDC2", 1)

    assert result["metric"] == "u_vs_prior_risk"
    assert result["n_cells"] == 3
    assert result["n_excluded"] == 2
    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson"] == pytest.approx(1.0)


def test_complete_bin_means_require_all_five_splits() -> None:
    rows = [
        {
            "held_out_label": "HLA-DRhi cDC2",
            "seed": seed,
            "prior_bin": prior_bin,
            "mean_prior_risk": 0.1 * prior_bin + 0.001 * seed,
            "mean_u": 0.2 * prior_bin + 0.001 * seed,
        }
        for prior_bin, seeds in ((1, range(1, 6)), (2, range(1, 5)))
        for seed in seeds
    ]

    result = _complete_bin_means(pd.DataFrame(rows))

    assert result["prior_bin"].tolist() == [1]
    assert result["n_splits"].tolist() == [5]


def test_plot_s3_writes_raster_and_vector_artwork(tmp_path: Path) -> None:
    bin_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for endpoint_index, endpoint in enumerate(EXPECTED_HELD_OUT):
        for seed in range(1, 6):
            for prior_bin in range(1, 21):
                bin_rows.append(
                    {
                        "held_out_label": endpoint,
                        "seed": seed,
                        "prior_bin": prior_bin,
                        "mean_prior_risk": 0.02 * prior_bin + 0.001 * seed,
                        "mean_u": (
                            0.04
                            + 0.01 * endpoint_index
                            + 0.004 * prior_bin
                            + 0.001 * seed
                        ),
                    }
                )
            correlation_rows.append(
                {
                    "held_out_label": endpoint,
                    "seed": seed,
                    "metric": "u_vs_prior_risk",
                    "spearman": 0.1 + 0.1 * endpoint_index + 0.01 * seed,
                    "pearson": 0.1 + 0.1 * endpoint_index,
                    "n_cells": 100,
                    "excluded_fraction": 0.0,
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    summary = _correlation_summary(correlations)
    assert summary["n_splits"].eq(5).all()

    output_path = tmp_path / "prior_dependence.png"
    _plot_s3(pd.DataFrame(bin_rows), correlations, output_path)

    for path in (
        output_path,
        output_path.with_suffix(".pdf"),
        output_path.with_suffix(".svg"),
    ):
        assert path.is_file()
        assert path.stat().st_size > 0


def test_promote_s3_submission_copies_hashes_complete_render_bundle(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results/HIHA_DC/figures"
    data_root = output_root / "data"
    docs_root = tmp_path / "docs/figs"
    data_root.mkdir(parents=True)
    outputs = {
        "figure_png": output_root / "supplementary_figure_s3_prior_dependence.png",
        "figure_pdf": output_root / "supplementary_figure_s3_prior_dependence.pdf",
        "figure_svg": output_root / "supplementary_figure_s3_prior_dependence.svg",
        "prior_bins_by_seed": data_root
        / "supplementary_figure_s3_prior_bins_by_seed.csv",
        "prior_bins_donor_equal": data_root
        / "supplementary_figure_s3_prior_bins_donor_equal.csv",
        "correlations_by_seed": data_root
        / "supplementary_figure_s3_prior_correlations_by_seed.csv",
        "correlations_summary": data_root
        / "supplementary_figure_s3_prior_correlations_summary.csv",
        "description": output_root / "supplementary_figure_s3_prior_dependence.md",
        "manifest": output_root
        / "supplementary_figure_s3_prior_dependence_manifest.yaml",
    }
    for index, path in enumerate(outputs.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}\n".encode())

    promoted = promote_hiha_s3_submission_copies(
        outputs=outputs,
        docs_fig_root=docs_root,
        project_root=tmp_path,
    )

    release = yaml.safe_load(promoted["release_manifest"].read_text())
    assert set(release["submission_copies"]) == {"png", "pdf", "svg"}
    assert set(release["source_tables"]) == {
        "prior_bins_by_seed",
        "prior_bins_donor_equal",
        "correlations_by_seed",
        "correlations_summary",
    }
    for suffix in ("png", "pdf", "svg"):
        source = outputs[f"figure_{suffix}"]
        destination = promoted[f"submission_{suffix}"]
        assert destination.read_bytes() == source.read_bytes()
        assert release["submission_copies"][suffix]["byte_identical"] is True
        assert (
            release["submission_copies"][suffix]["sha256"]
            == release["result_renderings"][suffix]["sha256"]
        )


def test_canonical_s3_release_manifest_matches_complete_render_bundle() -> None:
    project_root = Path(__file__).resolve().parents[1]
    release_path = (
        project_root
        / "results/HIHA_DC/figures/"
        "supplementary_figure_s3_prior_dependence_release_manifest.yaml"
    )
    release = yaml.safe_load(release_path.read_text(encoding="utf-8"))

    records = [
        release["configuration_manifest"],
        release["description"],
        *release["source_tables"].values(),
        *release["code"].values(),
        *release["result_renderings"].values(),
    ]
    for record in records:
        path = project_root / record["path"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]

    for suffix, record in release["submission_copies"].items():
        source = project_root / record["source"]
        destination = project_root / record["destination"]
        assert record["byte_identical"] is True
        assert source.read_bytes() == destination.read_bytes()
        assert record["sha256"] == release["result_renderings"][suffix]["sha256"]


def test_manuscript_prior_section_matches_regenerated_summary() -> None:
    manuscript = Path("docs/manuscript_supp.md").read_text(encoding="utf-8")
    summary = pd.read_csv(
        "results/HIHA_DC/figures/data/"
        "supplementary_figure_s3_prior_correlations_summary.csv"
    )

    assert (
        "#### S2.5.4. Association of query-marginal deficit with prior risk"
        in manuscript
    )
    assert "the reference-hub exposure is" not in manuscript
    assert "reference-hub exposure" not in manuscript
    assert (
        "figs/manuscript_fig_hiha_supp_s3_prior_dependence.png"
        in manuscript
    )
    assert summary["metric"].eq("u_vs_prior_risk").all()

    for row in summary.itertuples(index=False):
        displayed = f"{row.spearman_mean:.3f}\\pm{row.spearman_sd:.3f}"
        assert displayed in manuscript

    caption_match = re.search(
        r"\*\*Supplementary Figure S5\.(.*?)\*\*(.*?)\n\n",
        manuscript,
        flags=re.DOTALL,
    )
    assert caption_match is not None
    caption_words = re.findall(
        r"\b[\w-]+\b", "".join(caption_match.groups())
    )
    assert 95 <= len(caption_words) <= 145
