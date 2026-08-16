# ADR-0043: Preserve the calibrated PBMC abstention policy in S7

## Status

Accepted

## Context

Supplementary Figure S7 separates threshold-free ranking from the calibrated
abstention policy. The selected PBMC runs already contain matched incomplete-
and full-reference score artifacts and a fixed label-entropy cutoff. The
figure should vary the score threshold without changing the method or
re-fitting transport.

The operational analysis also needs to distinguish held-out-state rejection
from shared-state coverage and identify the contribution of score and label
entropy clauses separately.

## Decision

S7 will preserve the configured label-entropy cutoff \(\theta_H=0.8\) and
recompute only full-reference \(u\) quantiles at
\[
\{0.85,0.86,\ldots,0.99,0.975,0.995\}.
\]
The 95th-percentile threshold is the primary operating point. Abstention
reasons are retained, score only, label entropy only, and both.

CD8 threshold-offset ECDFs use incomplete-reference scores relative to the
matched full-reference 95th-percentile cutoff. Shared-state heatmaps exclude
the held-out stimulated state. Forced per-class F1 is computed within each
donor split on shared cells and displayed only when at least three donor
splits contribute.

## Consequences

- S7 evaluates operating-policy sensitivity without new transport fits.
- The full-reference control remains the calibration source and is not used
  as an absent-state evaluation set.
- Shared coverage and forced F1 cannot be interpreted as performance on the
  held-out stimulated class.
- Rows with fewer than three contributing splits are displayed as N/A while
  their raw contribution counts remain in the source tables.
