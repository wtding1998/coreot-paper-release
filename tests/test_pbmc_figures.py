from __future__ import annotations

import numpy as np
import pandas as pd

from coreot.results.pbmc_figures import (
    EXPECTED_LABELS,
    EXPECTED_SEEDS,
    METHOD_SCORES,
    _validate_comparison_coverage,
    compute_destination_composition,
    compute_restoration_summaries,
    compute_within_celltype_metrics,
)


def test_method_scores_retire_u_tilde() -> None:
    assert all(item.score != "u_tilde" for item in METHOD_SCORES)


def test_restoration_summaries_use_within_group_medians() -> None:
    truth = pd.DataFrame({
        "cell_id": ["s1", "s2", "c1", "c2"],
        "is_absent_state": [True, True, False, False],
    })
    ablated = pd.DataFrame({
        "cell_id": ["s1", "s2", "c1", "c2"], "method": ["coreot_full"] * 4,
        "u": [0.8, 0.6, 0.2, 0.4],
    })
    full = pd.DataFrame({
        "cell_id": ["c2", "s2", "c1", "s1"], "method": ["coreot_full"] * 4,
        "u": [0.3, 0.2, 0.1, 0.4],
    })
    metadata = pd.DataFrame({
        "cell_type": ["B cells"] * 4,
        "condition": ["stim", "stim", "ctrl", "ctrl"],
    }, index=["s1", "s2", "c1", "c2"])

    actual = compute_restoration_summaries(truth, ablated, full, metadata, "B cells")
    values = actual.set_index(["cell_group", "reference"])["median_u"]

    np.testing.assert_allclose(
        [
            values.loc[("held_out_stimulated", "ablated")],
            values.loc[("held_out_stimulated", "full")],
            values.loc[("same_type_control", "ablated")],
            values.loc[("same_type_control", "full")],
        ],
        [0.7, 0.3, 0.3, 0.2],
    )


def test_comparison_coverage_allows_methods_not_displayed_in_figure() -> None:
    rows = [
        {
            "result_family": item.family,
            "run_id": f"{label}-{seed}",
            "held_out_label": label,
            "seed": seed,
            "method": item.method,
            "score": item.score,
            "auroc": 0.8,
            "auprc": 0.4,
            "auprc_baseline": 0.1,
            "absent_abstention_rate": 0.5,
            "shared_false_abstention_rate": 0.05,
        }
        for label in EXPECTED_LABELS
        for seed in EXPECTED_SEEDS
        for item in METHOD_SCORES
    ]
    rows.append(
        {
            **rows[0],
            "method": "nearest_neighbor",
            "score": "u",
        }
    )

    _validate_comparison_coverage(pd.DataFrame(rows))


def test_within_celltype_metrics_exclude_other_cell_types() -> None:
    truth = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "true_label": ["B cells", "B cells", "NK cells", "NK cells"],
            "is_absent_state": [True, False, False, False],
        }
    )
    scores = pd.DataFrame({"cell_id": ["d", "b", "a", "c"], "u": [0.99, 0.1, 0.9, 0.98]})
    metadata = pd.DataFrame(
        {"cell_type": ["B cells", "B cells", "NK cells", "NK cells"],
         "condition": ["stim", "ctrl", "stim", "ctrl"]},
        index=["a", "b", "c", "d"],
    )

    actual = compute_within_celltype_metrics(truth, scores, metadata, "B cells", "u")

    assert actual["n_positives"] == 1
    assert actual["n_negatives"] == 1
    assert actual["auroc"] == 1.0
    assert actual["auprc"] == 1.0
    assert actual["auprc_baseline"] == 0.5


def test_destination_composition_normalizes_each_source_before_averaging() -> None:
    coupling = pd.DataFrame(
        {
            "source_cell_id": ["s1", "s1", "s2", "s2", "ignored"],
            "target_cell_id": ["t1", "t2", "t1", "t2", "t1"],
            "coupling": [9.0, 1.0, 0.1, 0.9, 100.0],
        }
    )
    metadata = pd.DataFrame(
        {"cell_type": ["B cells", "B cells"], "condition": ["ctrl", "stim"]},
        index=["t1", "t2"],
    )

    actual = compute_destination_composition(coupling, {"s1", "s2"}, metadata)
    values = actual.set_index("target_condition")["proportion"]

    np.testing.assert_allclose(values.loc["ctrl"], 0.5)
    np.testing.assert_allclose(values.loc["stim"], 0.5)
    np.testing.assert_allclose(values.sum(), 1.0)

    permuted = compute_destination_composition(
        coupling.sample(frac=1.0, random_state=7), {"s1", "s2"}, metadata
    )
    pd.testing.assert_frame_equal(actual, permuted)


def test_destination_composition_records_zero_transport_sources() -> None:
    coupling = pd.DataFrame(
        {
            "source_cell_id": ["s1"],
            "target_cell_id": ["t1"],
            "coupling": [2.0],
        }
    )
    metadata = pd.DataFrame(
        {"cell_type": ["B cells"], "condition": ["ctrl"]}, index=["t1"]
    )

    actual = compute_destination_composition(coupling, {"s1", "s2"}, metadata)

    assert actual["n_positive_transport_sources"].unique().tolist() == [1]
    assert actual["n_zero_transport_sources"].unique().tolist() == [1]
    np.testing.assert_allclose(actual["proportion"].sum(), 1.0)
