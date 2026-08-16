# CoRe-OT paper release

This is the private preparation repository for the source-code and
reproducibility release accompanying the manuscript **“CoRe-OT: cell-adaptive
unbalanced optimal transport for single-cell mapping with incomplete reference
representation.”** Public release has not yet been authorized.

CoRe-OT (correspondence-aware reference mapping by optimal transport) is an
asymmetric unbalanced optimal-transport framework that separates reference
destination from model-conditioned transported support in single-cell
reference mapping.

## Repository contents

- `src/`, `experiments/`, `tests/`, and `verification/` contain the validated
  paper-release source snapshot and its focused verification code.
- `paper-release/workflows/` contains the artifact-regeneration dispatcher and
  manuscript-artifact map.
- `paper-release/provenance/` contains the release registry, file inventory,
  external-dependency records, and checksum ledger for the complete evidence
  archive.

The complete evidence units are intentionally not stored in ordinary Git
history. After final authorization, a checksum-addressed evidence archive will
be attached to a tagged GitHub Release. The provenance checksum ledger applies
to that complete archive, not to this source-only checkout.

## Environment

The locked project supports Python 3.11 and 3.12. With `uv` installed, create
the source-development environment with:

```bash
uv sync --locked
```

The complete evidence archive includes a separate release README with the
clean-room artifact-regeneration procedure and its external-input identities.

## External-data boundary

No HDF5-family biological-data object is included in this repository. The
processed HIHA and PBMC objects and the mouse-spleen inputs remain
acquisition-only external dependencies. Their identities and acquisition
records are maintained in
`paper-release/provenance/external_dependencies.csv`.

## Licensing

Software is licensed under the MIT License in `LICENSE`. Author-created
documentation and paper-facing evidence are licensed under CC BY 4.0. See
`LICENSE_SCOPE.md` for scope and exclusions. Copyright-holder and institution
fields remain private-draft TODOs and must be completed before publication.

## Citation and contact

Final citation metadata and corresponding-author contact details will be added
before public release.
