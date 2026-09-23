
import numpy as np
import pytest

from coreot.results.matched_reference import (
    control_adjusted_median_deficit_decrease,
)


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
