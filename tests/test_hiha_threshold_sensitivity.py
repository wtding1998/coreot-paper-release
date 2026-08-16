from __future__ import annotations

import pandas as pd
import pytest

from experiments.missing_celltype.generate_hiha_dc_threshold_sensitivity import (
    POLICIES,
    compute_run_threshold_sensitivity,
    summarize_threshold_sensitivity,
)


def test_threshold_sensitivity_uses_reference_quantiles_and_entropy_clause() -> None:
    reference = _score_frame("full_reference_control", [0.0, 1.0, 2.0, 3.0])
    incomplete = _score_frame("incomplete_reference", [0.5, 1.5, 2.5, 3.5])
    incomplete.loc[incomplete["cell_id"].eq("c2"), "label_entropy"] = 0.9
    truth = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2", "c3"],
            "true_label": ["held", "held", "A", "A"],
            "removed_state": ["held"] * 4,
            "is_absent_state": [True, True, False, False],
            "is_shared_state": [False, False, True, True],
        }
    )

    result = compute_run_threshold_sensitivity(
        full_reference_scores=reference,
        incomplete_scores=incomplete,
        truth=truth,
        run_id="run1",
        held_out_label="held",
        seed=1,
        entropy_threshold=0.8,
        percentiles=(0.5,),
    )

    assert len(result) == 2
    assert result["threshold"].eq(1.5).all()
    assert result["absent_abstention_rate"].eq(0.0).all()
    assert result["shared_false_abstention_rate"].eq(1.0).all()
    assert result["coverage"].eq(0.0).all()


def test_threshold_sensitivity_can_evaluate_one_policy_run_family() -> None:
    reference = _score_frame("full_reference_control", [0.0, 1.0])
    incomplete = _score_frame("incomplete_reference", [0.5, 1.5])
    truth = pd.DataFrame(
        {
            "cell_id": ["c0", "c1"],
            "true_label": ["held", "A"],
            "removed_state": ["held", "held"],
            "is_absent_state": [True, False],
            "is_shared_state": [False, True],
        }
    )

    result = compute_run_threshold_sensitivity(
        full_reference_scores=reference,
        incomplete_scores=incomplete,
        truth=truth,
        run_id="uniform_run",
        held_out_label="held",
        seed=1,
        entropy_threshold=0.8,
        percentiles=(0.5,),
        policies=(POLICIES[1],),
    )

    assert result["method"].tolist() == ["uniform_uot"]
    assert result["run_id"].tolist() == ["uniform_run"]


def test_threshold_sensitivity_summary_uses_seed_sample_standard_deviation() -> None:
    frame = pd.DataFrame(
        {
            "held_out_label": ["HLA-DRhi cDC2"] * 2,
            "method": ["coreot_full"] * 2,
            "score": ["u"] * 2,
            "display_name": [r"CoRe-OT ($u$)"] * 2,
            "threshold_percentile": [0.95] * 2,
            "seed": [1, 2],
            "full_reference_false_abstention_rate": [0.04, 0.06],
            "absent_abstention_rate": [0.3, 0.5],
            "shared_false_abstention_rate": [0.02, 0.04],
            "coverage": [0.98, 0.96],
            "post_abstention_macro_f1": [0.9, 0.94],
        }
    )

    summary = summarize_threshold_sensitivity(frame)

    assert summary.loc[0, "absent_abstention_rate_mean"] == pytest.approx(0.4)
    assert summary.loc[0, "absent_abstention_rate_std"] == pytest.approx(2**0.5 / 10)
    assert summary.loc[0, "n_runs"] == 2


def _score_frame(condition: str, scores: list[float]) -> pd.DataFrame:
    rows = []
    for method in ("coreot_full", "uniform_uot"):
        for index, value in enumerate(scores):
            rows.append(
                {
                    "cell_id": f"c{index}",
                    "condition_id": condition,
                    "method": method,
                    "u": value,
                    "u_tilde": value,
                    "label_entropy": 0.0,
                    "forced_label": "A",
                }
            )
    return pd.DataFrame(rows)
