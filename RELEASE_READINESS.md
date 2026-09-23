# Release readiness

This is a private source staging repository. Documentation alignment is not
scientific revalidation or authorization to publish data.

## Current metadata

- [x] Current manuscript title, author order, affiliations and corresponding
  email are recorded in README.md and CITATION.cff.
- [x] README terminology uses reference destination and fitted mass retention.
- [x] Historical source and evidence identities remain in SOURCE_SNAPSHOT.yaml
  and paper-release/provenance/ without rewriting their checksum ledger.
- [ ] Confirm software copyright ownership and institutional attribution in
  LICENSE and LICENSE_SCOPE.md. Manuscript authorship is not proof of ownership.

## Data and reproducibility

- [x] Original-data acquisition records are indexed in
  paper-release/provenance/external_dependencies.csv.
- [ ] Verify acquisition instructions and availability of the exact external
  inputs in an independent environment.
- [ ] Reconcile the final manuscript displays and generated Supplementary Data
  1–4 with the frozen evidence archive; documentation changes do not close this check.
- [ ] Attach the complete evidence archive and Supplementary Data files using
  the approved delivery mechanism, with checksums and actual access instructions.
- [ ] Obtain an archival DOI and update citation/access metadata.
- [ ] Complete the full evidence-dependent reproduction checks with the
  declared inputs. Passing source tests alone does not satisfy this item.

## Publication

- [ ] Obtain final author approval for the exact public release contents.
- [ ] Make the repository public at submission and verify anonymous access.
- [ ] Replace manuscript planned-access wording with verified links and the
  release identifier only after access is established.

## Validation of this update

Checks were run on 2026-09-23 against this checkout using the existing
development environment (Python 3.12), not a newly installed clean environment:

- [x] `pytest --collect-only -q`: completed successfully.
- [x] `pytest tests/test_pbmc_celltypist_candidate.py -q`: all nine tests passed.
- [x] `ruff check --no-cache src experiments tests`: passed.
- [x] `python -m paper-release.workflows.reproduce --help`: passed with this
  checkout's `src` on the Python import path. Invoke as a module because the
  dispatcher uses relative imports.
- [x] Tried the full `pytest` suite: it did not pass. The reported failures
  and setup errors require absent `results/` evidence, `docs/` manuscript
  assets, or `experiments/mouse_spleen/configs/mouse_spleen_core_ot.yaml`.
  Examples include sealed figure inputs, PBMC figure tables, the current
  supplement and mouse-spleen configuration. These are not supplied by a
  source-only clone; this run is not a complete reproduction certificate.
- [ ] A fresh `uv sync --locked` environment and evidence-backed full run
  remain unverified in this documentation update.
