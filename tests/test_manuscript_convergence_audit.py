import re
from pathlib import Path

import pandas as pd
import yaml

from experiments.generate_manuscript_convergence_audit import (
    AUXILIARY_INDEX_INPUTS,
    EVIDENCE_RECORD_UNAVAILABLE,
    EVIDENCE_VERIFIED_CONVERGED,
    RETAINED_FIT_FAMILIES,
    generate_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_MANUSCRIPT_SCOPES = {
    ("HIHA DC", "selected comparison"): (
        "S2.3 selected ranking and represented-state forced-transfer "
        "comparisons; S2.4 matched full-reference response; S2.5.3 "
        "selected-fit calibration; S2.5.4 selected-fit prior-dependence "
        "diagnostic"
    ),
    ("HIHA DC", "component-ablation surface"): (
        "S2.5.1 manuscript-facing compatibility-sensitivity surface; "
        "supporting component-sensitivity fits retained in Supplementary "
        "Data 3"
    ),
    (
        "HIHA DC",
        "compatibility-weight mean-matched attribution",
    ): (
        "Supporting across-alpha mean-matched attribution associated with "
        "S2.5.2 and retained in Supplementary Data 3; not the "
        "manuscript-facing fixed-alpha-zero surface"
    ),
    ("HIHA DC", "fixed-alpha-zero matchability surface"): (
        "S2.5.2 fixed-alpha-zero mean-matched matchability-penalty "
        "attribution"
    ),
    ("HIHA DC", "constant-tau compatibility sensitivity"): (
        "Supporting constant-tau compatibility-sensitivity analysis for "
        "S2.5.1 retained in Supplementary Data 3; not a separate "
        "manuscript-facing display"
    ),
    ("HIHA DC", "query-penalty sensitivity"): (
        "S2.5.3 query-penalty parameter sensitivity"
    ),
    ("PBMC", "parameter sensitivity"): (
        "S3.2 selected CoRe-OT operating points; S3.3 selected CoRe-OT "
        "results; S3.4 matched full-reference response; S3.5.3 CoRe-OT "
        "parameter and calibration sensitivity; S3.5.4 selected-fit "
        "prior-dependence diagnostic"
    ),
    ("PBMC", "selected comparison"): (
        "S3.3 selected match-only and Uniform UOT comparisons; S3.5.3 "
        "Uniform UOT calibration sensitivity"
    ),
    ("PBMC", "component-ablation surface"): (
        "S3.5.1 manuscript-facing constant-tau compatibility-sensitivity "
        "surface; supporting alpha-zero penalty-sensitivity fits retained "
        "in Supplementary Data 4"
    ),
    ("PBMC", "fixed-alpha-zero matchability surface"): (
        "S3.5.2 fixed-alpha-zero mean-matched matchability-penalty "
        "attribution"
    ),
    (
        "PBMC",
        "compatibility-weight mean-matched attribution",
    ): (
        "Supporting across-alpha mean-matched attribution associated with "
        "S3.5.2 and retained in Supplementary Data 4; not the "
        "manuscript-facing fixed-alpha-zero surface"
    ),
    ("Mouse spleen", "selected comparison"): (
        "S4.2 selected operating point; S4.3 selected transport "
        "comparisons; S4.4 selected-fit destination and "
        "transported-support summaries; S4.5.4 selected-fit "
        "prior-dependence diagnostic"
    ),
    (
        "Mouse spleen",
        "constant-tau compatibility-sensitivity surface",
    ): "S4.5.1 constant-tau compatibility-sensitivity surface",
    ("Mouse spleen", "fixed-alpha-five matchability surface"): (
        "S4.5.2 fixed-alpha-five mean-matched matchability-penalty "
        "attribution"
    ),
    ("Mouse spleen", "query-penalty sensitivity"): (
        "S4.5.3 query-penalty parameter sensitivity"
    ),
}


def test_retained_family_section_map_matches_current_supplement() -> None:
    actual = {
        (item["experiment"], item["analysis_family"]): item[
            "manuscript_scope"
        ]
        for item in RETAINED_FIT_FAMILIES
    }
    assert actual == EXPECTED_MANUSCRIPT_SCOPES

    supplement = (
        PROJECT_ROOT / "docs/manuscript_supp.md"
    ).read_text(encoding="utf-8")
    headings = {
        match.group(1)
        for match in re.finditer(
            r"^#{2,4} (S\d+(?:\.\d+)+)\.",
            supplement,
            flags=re.MULTILINE,
        )
    }
    referenced_sections = {
        section
        for scope in actual.values()
        for section in re.findall(r"S\d+(?:\.\d+)+", scope)
    }
    assert referenced_sections <= headings


def test_retained_index_does_not_invent_reference_conditions(
    tmp_path: Path,
) -> None:
    by_fit_path, _, coverage_path, manifest_path = generate_audit(
        PROJECT_ROOT,
        output_root=tmp_path,
    )
    by_fit = pd.read_csv(by_fit_path)
    coverage = pd.read_csv(coverage_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["auxiliary_index_inputs"] == [
        str(path) for path in AUXILIARY_INDEX_INPUTS
    ]

    hiha_constant_tau = by_fit.loc[
        by_fit["experiment"].eq("HIHA DC")
        & by_fit["analysis_family"].eq(
            "constant-tau compatibility sensitivity"
        )
    ]
    assert len(hiha_constant_tau) == 990
    assert set(hiha_constant_tau["condition"]) == {"incomplete_reference"}

    pbmc_sensitivity = by_fit.loc[
        by_fit["experiment"].eq("PBMC")
        & by_fit["analysis_family"].eq("parameter sensitivity")
    ]
    assert pbmc_sensitivity["condition"].value_counts().to_dict() == {
        "incomplete_reference": 135,
        "full_reference_control": 15,
    }

    for experiment, family, expected_count in (
        ("HIHA DC", "query-penalty sensitivity", 90),
        ("Mouse spleen", "query-penalty sensitivity", 28),
    ):
        family_rows = by_fit.loc[
            by_fit["experiment"].eq(experiment)
            & by_fit["analysis_family"].eq(family)
        ]
        assert len(family_rows) == expected_count
        assert family_rows["convergence_evidence"].eq(
            EVIDENCE_VERIFIED_CONVERGED
        ).all()

    gaps = coverage.loc[
        coverage["coverage_status"].eq("indexed_with_record_gaps"),
        ["experiment", "analysis_family", "n_without_per_fit_record"],
    ]
    assert set(gaps.itertuples(index=False, name=None)) == {
        ("HIHA DC", "constant-tau compatibility sensitivity", 990),
        ("PBMC", "parameter sensitivity", 120),
    }


def test_retained_transport_families_are_fully_indexed(
    tmp_path: Path,
) -> None:
    by_fit_path, summary_path, coverage_path, manifest_path = generate_audit(
        PROJECT_ROOT,
        output_root=tmp_path,
    )
    by_fit = pd.read_csv(by_fit_path)
    summary = pd.read_csv(summary_path)
    coverage = pd.read_csv(coverage_path)

    family_columns = ["experiment", "analysis_family"]
    expected_families = {
        (item["experiment"], item["analysis_family"])
        for item in RETAINED_FIT_FAMILIES
    }
    assert set(
        coverage.loc[:, family_columns].itertuples(index=False, name=None)
    ) == expected_families
    assert set(
        by_fit.loc[:, family_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == expected_families
    assert set(
        summary.loc[:, family_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == expected_families

    recoverable = by_fit.loc[
        by_fit["convergence_evidence"].ne(EVIDENCE_RECORD_UNAVAILABLE)
    ]
    assert recoverable["convergence_evidence"].eq(
        EVIDENCE_VERIFIED_CONVERGED
    ).all()

    gaps = coverage.loc[
        coverage["coverage_status"].eq("indexed_with_record_gaps"),
        ["experiment", "analysis_family", "n_without_per_fit_record"],
    ]
    assert {
        tuple(item)
        for item in gaps.itertuples(index=False, name=None)
    } == {
        ("HIHA DC", "constant-tau compatibility sensitivity", 990),
        ("PBMC", "parameter sensitivity", 120),
    }
    supplement = (
        PROJECT_ROOT / "docs/manuscript_supp.md"
    ).read_text(encoding="utf-8")
    assert "family-level audit coverage is complete" in supplement
    assert "1,980 HIHA" in supplement
    assert "240 PBMC" in supplement
    assert manifest_path.is_file()
