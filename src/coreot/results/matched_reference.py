from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


CONTROL_ADJUSTED_MEDIAN_DEFICIT_DECREASE = (
    "control_adjusted_median_deficit_decrease"
)
TYPICAL_CELL_CONDITIONAL_DESTINATION = "typical_cell_conditional_destination"
MEAN_CELL_CONDITIONAL_DESTINATION = "mean_cell_conditional_destination"
POOLED_TRANSPORTED_MASS_DESTINATION_COMPOSITION = (
    "pooled_transported_mass_destination_composition"
)


@dataclass(frozen=True)
class ControlAdjustedMedianDeficitDecrease:
    heldout_decrease: float
    control_decrease: float
    control_adjusted_decrease: float


def control_adjusted_median_deficit_decrease(
    *,
    incomplete: ArrayLike,
    full: ArrayLike,
    heldout: ArrayLike,
    control: ArrayLike,
) -> ControlAdjustedMedianDeficitDecrease:
    incomplete_values = np.asarray(incomplete, dtype=float)
    full_values = np.asarray(full, dtype=float)
    heldout_mask = np.asarray(heldout, dtype=bool)
    control_mask = np.asarray(control, dtype=bool)
    arrays = (incomplete_values, full_values, heldout_mask, control_mask)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Matched-reference estimand inputs must be one-dimensional.")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Matched-reference estimand inputs must have equal length.")
    if not heldout_mask.any() or not control_mask.any():
        raise ValueError("Held-out and control cohorts must both be nonempty.")
    if np.any(heldout_mask & control_mask):
        raise ValueError("Held-out and control cohorts must be disjoint.")

    heldout_decrease = float(
        np.median(incomplete_values[heldout_mask])
        - np.median(full_values[heldout_mask])
    )
    control_decrease = float(
        np.median(incomplete_values[control_mask])
        - np.median(full_values[control_mask])
    )
    return ControlAdjustedMedianDeficitDecrease(
        heldout_decrease=heldout_decrease,
        control_decrease=control_decrease,
        control_adjusted_decrease=heldout_decrease - control_decrease,
    )
