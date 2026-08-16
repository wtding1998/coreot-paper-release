"""External reference-mapping baselines for controlled missing-state benchmarks."""

from coreot.external_baselines.runner import (
    DEFAULT_EXTERNAL_BASELINE_METHODS,
    EXTERNAL_BASELINE_METHODS,
    ExternalBaselineError,
    run_external_baselines,
    write_external_baseline_cell_scores,
)

__all__ = [
    "DEFAULT_EXTERNAL_BASELINE_METHODS",
    "EXTERNAL_BASELINE_METHODS",
    "ExternalBaselineError",
    "run_external_baselines",
    "write_external_baseline_cell_scores",
]
