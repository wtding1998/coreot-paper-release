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
- [ ] Complete evidence-backed reproduction using the declared inputs and
  development-repository validation checks. External-asset tests are excluded
  from this source release; passing release tests does not satisfy this item.

## Publication

- [ ] Obtain final author approval for the exact public release contents.
- [ ] Make the repository public at submission and verify anonymous access.
- [ ] Replace manuscript planned-access wording with verified links and the
  release identifier only after access is established.

## Validation of this update

Checks were run on 2026-09-23 against this checkout using the existing
development environment (Python 3.12), not a newly installed clean environment:

- [x] Removed tests requiring external result tables, manuscript assets,
  biological data and development-only configs, plus their unused fixtures
  and imports. The removed checks remain recoverable from Git history.
- [x] `ruff check --no-cache src experiments tests`: passed.
- [x] `python -m paper-release.workflows.reproduce --help`: passed with this
  checkout's `src` on the Python import path. Invoke as a module because the
  dispatcher uses relative imports.
- [x] Full remaining release suite: `pytest -o addopts='' -q` reports
  **168 passed**, with one non-failing joblib CPU-count detection warning.
  No tests were skipped. This validates the retained code tests, not the
  removed manuscript/evidence checks.
- [ ] A fresh `uv sync --locked` environment and evidence-backed full run
  remain unverified in this documentation update.
