import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.generate_manuscript_convergence_audit import (
    AUXILIARY_INDEX_INPUTS,
    EVIDENCE_RECORD_UNAVAILABLE,
    EVIDENCE_VERIFIED_CONVERGED,
    RETAINED_FIT_FAMILIES,
    RETIRED_FIT_FAMILIES,
    SETTINGS_CONFIGURED_ONLY,
    SETTINGS_FIT_METHOD_PARAMS,
    SETTINGS_RETAINED_PER_FIT_TABLE,
    _manifest_row,
    generate_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_MANUSCRIPT_SCOPES = {
    ("HIHA DC", "selected comparison"): (
        "S2.3 selected ranking and represented-state forced-transfer "
        "comparisons; S2.4 paired reference-restoration response; S2.5 "
        "operational abstention and calibration sensitivity"
    ),
    ("HIHA DC", "component-ablation surface"): (
        "S2.6.1 manuscript-facing compatibility-sensitivity surface; "
        "supporting component-sensitivity fits retained in Supplementary "
        "Data 2"
    ),
    (
        "HIHA DC",
        "compatibility-weight mean-matched attribution",
    ): (
        "Supporting across-alpha mean-matched attribution associated with "
        "S2.6.2 and retained in Supplementary Data 2; not the "
        "manuscript-facing fixed-alpha-zero surface"
    ),
    ("HIHA DC", "fixed-alpha-zero matchability surface"): (
        "S2.6.2 fixed-alpha-zero mean-matched matchability-penalty "
        "attribution"
    ),
    ("HIHA DC", "query-penalty sensitivity"): (
        "S2.6.3 query-penalty parameter sensitivity"
    ),
    ("PBMC", "parameter sensitivity"): (
        "S3.2 selected CoRe-OT operating points; S3.3 selected CoRe-OT "
        "results; S3.4 paired reference-restoration response; S3.5 "
        "operational abstention and calibration sensitivity; S3.6.3 "
        "query-penalty sensitivity; S3.6.4 selected-fit "
        "prior-dependence diagnostic"
    ),
    ("PBMC", "selected comparison"): (
        "S3.3 selected match-only and Uniform UOT comparisons; S3.5 "
        "Uniform UOT calibration sensitivity"
    ),
    ("PBMC", "component-ablation surface"): (
        "S3.6.1 manuscript-facing constant-tau compatibility-sensitivity "
        "surface; supporting alpha-zero penalty-sensitivity fits retained "
        "in Supplementary Data 3"
    ),
    ("PBMC", "fixed-alpha-zero matchability surface"): (
        "S3.6.2 fixed-alpha-zero mean-matched matchability-penalty "
        "attribution"
    ),
    (
        "PBMC",
        "compatibility-weight mean-matched attribution",
    ): (
        "Supporting across-alpha mean-matched attribution associated with "
        "S3.6.2 and retained in Supplementary Data 3; not the "
        "manuscript-facing fixed-alpha-zero surface"
    ),
    ("Mouse spleen", "selected comparison"): (
        "S4.2 selected operating point; S4.3 selected transport "
        "comparisons; S4.4 selected-fit destination and "
        "transported-support summaries"
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
    assert manifest["retired_analysis_families"] == list(
        RETIRED_FIT_FAMILIES
    )
    assert ("Mouse spleen", "prior-risk diagnostic") in {
        (item["experiment"], item["analysis_family"])
        for item in RETIRED_FIT_FAMILIES
    }

    assert "constant-tau compatibility sensitivity" not in set(
        by_fit.loc[by_fit["experiment"].eq("HIHA DC"), "analysis_family"]
    )
    assert "prior-risk diagnostic" not in set(
        by_fit.loc[
            by_fit["experiment"].eq("Mouse spleen"), "analysis_family"
        ]
    )

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
        ["experiment", "analysis_family"],
    ]
    assert set(gaps.itertuples(index=False, name=None)) <= {
        ("PBMC", "parameter sensitivity"),
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
    assert recoverable[["n_iter", "max_iter", "tol"]].notna().all().all()
    assert recoverable["n_iter"].gt(0).all()
    assert recoverable["n_iter"].le(recoverable["max_iter"]).all()

    gaps = coverage.loc[
        coverage["coverage_status"].eq("indexed_with_record_gaps"),
        ["experiment", "analysis_family"],
    ]
    assert set(gaps.itertuples(index=False, name=None)) <= {
        ("PBMC", "parameter sensitivity"),
    }
    supplement = (
        PROJECT_ROOT / "docs/manuscript_supp.md"
    ).read_text(encoding="utf-8")
    assert (
        "Every transport fit contributing to the reported primary and sensitivity"
        in " ".join(supplement.split())
    )
    assert manifest_path.is_file()


def test_manifest_backed_rows_read_solver_settings_from_method_params(
    tmp_path: Path,
) -> None:
    by_fit_path, _, _, _ = generate_audit(
        PROJECT_ROOT,
        output_root=tmp_path,
    )
    by_fit = pd.read_csv(by_fit_path)
    manifest_backed = by_fit.loc[
        by_fit["solver_settings_evidence"].eq(SETTINGS_FIT_METHOD_PARAMS)
    ]
    assert not manifest_backed.empty
    for item in manifest_backed.itertuples(index=False):
        manifest_path = PROJECT_ROOT / item.source_artifact
        params_path = PROJECT_ROOT / item.method_params_artifact
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        assert manifest_path.parent == params_path.parent
        assert manifest["metadata"]["method"] == item.method
        assert params["name"] == item.method
        assert int(params["max_iter"]) == item.max_iter
        assert float(params["tol"]) == item.tol

    unavailable = by_fit.loc[
        by_fit["convergence_evidence"].eq(EVIDENCE_RECORD_UNAVAILABLE)
    ]
    assert unavailable["solver_settings_evidence"].eq(
        SETTINGS_CONFIGURED_ONLY
    ).all()
    assert unavailable["method_params_artifact"].isna().all()

    retained_tables = by_fit.loc[
        by_fit["solver_settings_evidence"].eq(
            SETTINGS_RETAINED_PER_FIT_TABLE
        )
    ]
    assert not retained_tables.empty


def _write_manifest_fixture(
    project_root: Path,
    *,
    manifest_method: str = "coreot_full",
    params_method: str = "coreot_full",
    params_artifact: str | None = None,
    write_params: bool = True,
) -> Path:
    fit_directory = (
        project_root
        / "runs/example/transport/incomplete_reference/candidates/coreot_full"
    )
    fit_directory.mkdir(parents=True)
    method_params_path = fit_directory / "method_params.yaml"
    recorded_artifact = params_artifact or str(
        method_params_path.relative_to(project_root)
    )
    (fit_directory / "transport_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "artifacts": {"method_params": recorded_artifact},
                "metadata": {
                    "converged": True,
                    "method": manifest_method,
                    "n_iter": 123,
                },
            }
        ),
        encoding="utf-8",
    )
    if write_params:
        method_params_path.write_text(
            yaml.safe_dump(
                {
                    "name": params_method,
                    "max_iter": 5000,
                    "tol": 2.0e-7,
                }
            ),
            encoding="utf-8",
        )
    return fit_directory


def _fixture_manifest_row(project_root: Path) -> dict[str, object]:
    return _manifest_row(
        project_root=project_root,
        experiment="fixture",
        analysis_family="fixture",
        endpoint="fixture",
        seed=1,
        condition="incomplete_reference",
        run_id="example",
        candidate_set="candidates",
        method="coreot_full",
        max_iter=2000,
        tol=1.0e-6,
    )


def test_manifest_row_uses_observed_fit_settings_not_index_fallback(
    tmp_path: Path,
) -> None:
    _write_manifest_fixture(tmp_path)
    row = _fixture_manifest_row(tmp_path)
    assert row["max_iter"] == 5000
    assert row["tol"] == 2.0e-7
    assert row["solver_settings_evidence"] == SETTINGS_FIT_METHOD_PARAMS


def test_manifest_row_labels_fallback_only_when_fit_files_are_absent(
    tmp_path: Path,
) -> None:
    row = _fixture_manifest_row(tmp_path)
    assert row["convergence_evidence"] == EVIDENCE_RECORD_UNAVAILABLE
    assert row["max_iter"] == 2000
    assert row["tol"] == 1.0e-6
    assert row["solver_settings_evidence"] == SETTINGS_CONFIGURED_ONLY
    assert row["method_params_artifact"] == ""


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"manifest_method": "uniform_uot"}, "manifest method mismatch"),
        ({"params_method": "uniform_uot"}, "identity mismatch"),
        (
            {"params_artifact": "runs/other/method_params.yaml"},
            "same fit directory",
        ),
        ({"write_params": False}, "incomplete manifest/method-parameter pair"),
    ],
)
def test_manifest_row_rejects_inconsistent_fit_lineage(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    message: str,
) -> None:
    _write_manifest_fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(ValueError, match=message):
        _fixture_manifest_row(tmp_path)
