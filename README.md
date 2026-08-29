# CoRe-OT paper release

This is the private staging repository for the source-code and reproducibility
release accompanying the manuscript **“CoRe-OT: cell-adaptive unbalanced
optimal transport for single-cell mapping with incomplete reference
representation.”** The 2026-08-29 release candidate has been technically
prepared, but public release remains subject to corresponding-author
authorization and completion of the metadata listed below.

CoRe-OT (correspondence-aware reference mapping by optimal transport) is an
asymmetric unbalanced optimal-transport framework that separates reference
destination from model-conditioned transported support in single-cell
reference mapping.

## Repository contents

- `src/`, `experiments/`, `tests/`, and `verification/` contain the validated
  paper-release source snapshot and its focused verification code.
- `experiments/pbmc_state/rerun_pbmc_celltypist_centered.py` contains the
  fail-closed current-environment CellTypist reproduction entry point, with its
  contract tests in `tests/test_pbmc_celltypist_candidate.py`.
- `paper-release/workflows/` contains the artifact-regeneration dispatcher and
  manuscript-artifact map.
- `paper-release/provenance/` contains the release registry, file inventory,
  external-dependency records, and checksum ledger for the complete evidence
  archive.
- `SOURCE_SNAPSHOT.yaml` records how this source-only checkout was derived
  from the validated evidence candidate and the tagged 2026-08-29 source.

## Release identity

- Source checkpoint: `wtding1998/CoReOT` tag
  `release-candidate-2026-08-29`, commit
  `15431b1b506103abb5d4375b03d739205bf930fb`.
- Validated evidence authority:
  `2026-08-28-current-executable-r4` (16 verified units).
- Public-source candidate tag: `release-candidate-2026-08-29`.

The 2026-08-29 CellTypist work adds a reproducible full-refit verification
path and fail-closed comparisons. It does not replace or silently rewrite the
checksum-covered scientific members of the validated evidence authority.

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

The self-contained source checks can be run without the evidence archive:

```bash
uv run pytest --collect-only -q
uv run pytest tests/test_pbmc_celltypist_candidate.py -q
uv run ruff check --no-cache src experiments tests
```

Tests that compare canonical manuscript figures, tables, and package members
require the checksum-addressed evidence archive. They are not expected to pass
against the source-only checkout without those declared inputs.

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

Final citation metadata, corresponding-author contact details, copyright
holder, and institutional attribution must be completed before public release.
