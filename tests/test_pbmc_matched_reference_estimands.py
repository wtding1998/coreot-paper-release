from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.pbmc_state.generate_pbmc_s6_matched_reference_rescue import (
    build_parser,
)
from coreot.results.matched_reference import (
    CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE,
    MEAN_CELL_CONDITIONAL_DESTINATION,
    POOLED_TRANSPORTED_MASS_DESTINATION_COMPOSITION,
    control_adjusted_median_deficit_decrease,
)
from coreot.results.pbmc_s6_matched_reference_rescue import (
    PBMCS6Paths,
    write_pbmc_s6_matched_reference_rescue_figure,
)
from coreot.results.pbmc_supplement import collect_matched_destination_response


@pytest.fixture(scope="module")
def generated_s6(tmp_path_factory: pytest.TempPathFactory) -> PBMCS6Paths:
    return write_pbmc_s6_matched_reference_rescue_figure(
        output_root=tmp_path_factory.mktemp("pbmc_s6")
    )


@pytest.fixture(scope="module")
def matched_destination_response() -> pd.DataFrame:
    selected_runs = pd.read_csv(
        "results/PBMC/compare_baselines/tables/compare_detection_by_run.csv"
    )
    response, _ = collect_matched_destination_response(
        selected_runs=selected_runs,
        runs_root=Path("runs"),
        raw_data_path=Path("data/raw/kang_2018.h5ad"),
    )
    return response


def test_control_adjusted_decrease_uses_differences_of_group_medians() -> None:
    response = control_adjusted_median_deficit_decrease(
        incomplete=np.array([0.0, 100.0, 101.0, 4.0, 6.0]),
        full=np.array([0.0, 1.0, 100.0, 3.0, 5.0]),
        heldout=np.array([True, True, True, False, False]),
        control=np.array([False, False, False, True, True]),
    )

    assert response.heldout_decrease == 99.0
    assert response.control_decrease == 1.0
    assert response.control_adjusted_decrease == 98.0


def test_control_adjusted_decrease_rejects_overlapping_cohorts() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        control_adjusted_median_deficit_decrease(
            incomplete=np.array([1.0, 2.0]),
            full=np.array([0.5, 1.5]),
            heldout=np.array([True, False]),
            control=np.array([True, True]),
        )


def test_s6_specificity_matches_emitted_group_medians(
    generated_s6: PBMCS6Paths,
) -> None:
    score = pd.read_csv(generated_s6.score_by_seed)
    specificity = pd.read_csv(generated_s6.specificity_by_seed)
    decreases = score.pivot(
        index=["held_out_label", "seed", "run_id", "cell_group"],
        columns="reference",
        values="median_u",
    )
    decreases["decrease"] = decreases["ablated"] - decreases["full"]
    expected = (
        decreases["decrease"]
        .unstack("cell_group")
        .assign(
            expected=lambda frame: frame["held_out_stimulated"]
            - frame["same_type_control"]
        )["expected"]
        .rename("expected")
        .reset_index()
    )
    compared = specificity.merge(
        expected,
        on=["held_out_label", "seed", "run_id"],
        how="inner",
        validate="one_to_one",
    )

    np.testing.assert_allclose(
        compared["restoration_specificity"],
        compared["expected"],
        rtol=0.0,
        atol=1e-15,
    )
    assert set(specificity["restoration_specificity_estimand"]) == {
        CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE
    }


def test_legacy_four_endpoint_s6_destinations_declare_pooled_mass_and_conserve_it(
    generated_s6: PBMCS6Paths,
) -> None:
    destination = pd.read_csv(generated_s6.destination_by_seed)
    assert set(destination["destination_estimand"]) == {
        POOLED_TRANSPORTED_MASS_DESTINATION_COMPOSITION
    }

    unavailable = destination.loc[
        destination["reference"].eq("ablated")
        & destination["destination_category"].eq("same_type_stimulated")
    ]
    assert unavailable["is_na"].all()
    assert unavailable["fraction"].isna().all()

    totals = destination.groupby(
        ["held_out_label", "seed", "run_id", "reference"],
        sort=False,
    )["fraction"].sum()
    np.testing.assert_allclose(totals, 1.0, rtol=0.0, atol=1e-12)


def test_supplement_destinations_declare_mean_cell_conditional_composition(
    matched_destination_response: pd.DataFrame,
) -> None:
    assert set(matched_destination_response["destination_estimand"]) == {
        MEAN_CELL_CONDITIONAL_DESTINATION
    }
    totals = matched_destination_response.groupby(
        [
            "held_out_label",
            "seed",
            "run_id",
            "cell_group",
            "reference_condition",
        ],
        sort=False,
    )["mean_destination_fraction"].sum()
    np.testing.assert_allclose(totals, 1.0, rtol=0.0, atol=1e-12)


def test_legacy_four_endpoint_s6_render_preserves_pooled_estimand_terms(
    generated_s6: PBMCS6Paths,
) -> None:
    description = generated_s6.description.read_text(encoding="utf-8")
    svg = generated_s6.svg.read_text(encoding="utf-8")

    assert "control-adjusted median-deficit decrease" in description
    assert "cell_transport_scores.parquet" in description
    assert "sparse_coupling.parquet" in description
    assert "Control-adjusted deficit decrease" in svg
    assert "Pooled transported-mass composition" in svg
    assert "Restoration specificity" not in svg
    assert "# Supplementary Figure S6" not in description
    assert "# Legacy four-endpoint PBMC matched-reference diagnostic" in description
    assert "It is not the current manuscript Supplementary Figure S6" in description


def test_legacy_four_endpoint_s6_cli_and_design_spec_disclaim_manuscript_role() -> None:
    description = build_parser().description
    assert description is not None
    assert "legacy four-endpoint" in description.casefold()
    assert "not the current manuscript Supplementary Figure S6" in description

    design = Path("results/PBMC/figures.md").read_text(encoding="utf-8")
    compact_design = " ".join(design.split())
    assert "# Supplementary Figure S6" not in design
    assert "# Legacy four-endpoint PBMC matched-reference diagnostic" in design
    assert "It is not the current" in compact_design
    assert "manuscript Supplementary Figure S6" in compact_design


def test_s6_manifest_covers_each_endpoint_seed_once(
    generated_s6: PBMCS6Paths,
) -> None:
    manifest = pd.read_csv(generated_s6.run_manifest)
    expected = {
        (endpoint, seed)
        for endpoint in ("B cells", "NK cells", "Dendritic cells", "CD8 T cells")
        for seed in range(1, 6)
    }
    assert set(
        manifest[["held_out_label", "seed"]].itertuples(index=False, name=None)
    ) == expected
    assert manifest["run_id"].is_unique
