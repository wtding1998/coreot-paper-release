from __future__ import annotations

import numpy as np


class CostScalingError(ValueError):
    """Raised when candidate-cost scaling configuration is invalid."""


def scale_distances(
    distances: np.ndarray,
    *,
    scale: str,
    clip_quantile: float,
    delta: float,
) -> np.ndarray:
    if distances.size == 0:
        return distances.astype(float)
    if delta <= 0:
        raise CostScalingError("cost_scaling.delta must be positive")
    if not 0 < clip_quantile <= 1:
        raise CostScalingError("cost_scaling.clip_quantile must be in (0, 1]")
    if scale != "median":
        raise CostScalingError(f"Only median cost scaling is implemented; got {scale!r}")

    denominator = float(np.median(distances)) + delta
    scaled = distances.astype(float) / denominator
    clip_value = float(np.quantile(scaled, clip_quantile))
    return np.minimum(scaled, clip_value)
