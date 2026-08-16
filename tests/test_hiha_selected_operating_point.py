from __future__ import annotations

import pytest

from experiments.missing_celltype.generate_hiha_dc_report_leave_one_configs import (
    SELECTED_OPERATING_POINTS,
)


def test_isg_cdc2_selected_operating_point() -> None:
    selected = SELECTED_OPERATING_POINTS["ISG+ cDC2"]

    assert (
        selected.coreot_tau_min,
        selected.coreot_tau_max,
        selected.coreot_tau_target,
        selected.coreot_alpha,
    ) == pytest.approx((0.5, 0.625, 1.0, 0.25))
