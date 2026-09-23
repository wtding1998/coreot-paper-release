# CoRe-OT paper release

This is the private staging repository for the source-code and reproducibility
release accompanying the manuscript **“CoRe-OT: unbalanced optimal transport
for single-cell mapping under incomplete reference coverage.”** The manuscript
is being prepared for submission to *Briefings in Bioinformatics* as a
Problem solving protocol. Documentation was updated on 2026-09-23; the
underlying source checkpoint remains the 2026-08-29 candidate.

The authors plan to make this repository public at submission. It is currently
private, and the complete evidence archive and archival DOI are not yet
available. See [release readiness](RELEASE_READINESS.md) for remaining checks.

CoRe-OT (correspondence-aware reference mapping by optimal transport) is an
asymmetric unbalanced optimal-transport framework that separates reference
destination from fitted mass retention in single-cell reference mapping.
Query-marginal deficit is conditional on the fitted model and inputs; it is
not a calibrated probability that a biological state is absent from the reference.

## Repository contents

- `src/`, `experiments/`, `tests/`, and `verification/` contain the validated
  paper-release source snapshot and its focused verification code. Historical
  validation records do not establish that every analysis has been rerun for
  the current manuscript.
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

Do not run the evidence checksum ledger against this checkout: its paths
refer to the complete evidence archive. The archived manifests retain their
original terminology and paths to preserve provenance. The current manuscript
also retains the qualification that scDOT and TACCO-OT comparisons use saved
predictions without a current independent full refit.

## Environment

The locked project supports Python 3.11 and 3.12. With `uv` installed, create
the source-development environment with:

```bash
uv sync --locked
```

The complete evidence archive includes a separate release README with the
clean-room artifact-regeneration procedure and its external-input identities.

The release test suite uses temporary or in-memory fixtures and does not
require the external evidence archive:

```bash
uv run pytest
uv run ruff check --no-cache src experiments tests
```

Tests that require saved result tables, development-only configurations,
biological datasets or manuscript files have been removed from this release
checkout. They remain recoverable from Git history and in the development
repository. Passing the release tests checks code behavior with fixtures; it
does not certify reproduction of the manuscript's numerical results. See
RELEASE_READINESS.md for the validation scope.

For manuscript-artifact regeneration, first obtain and extract the complete
evidence archive, then follow its README. The dispatcher is
`paper-release/workflows/reproduce.py`; inspect its options with:

```bash
uv run python -m paper-release.workflows.reproduce --help
```

It requires `--release-root` pointing to the extracted evidence release and
`--output-root` for generated files. Cloning this repository alone does not
provide the fitted outputs needed for that workflow. Regenerating artifacts
from saved fits is distinct from independently rerunning model fitting.

## External-data boundary

No HDF5-family biological-data object is included in this repository. The
processed HIHA and PBMC objects and the mouse-spleen inputs remain
acquisition-only external dependencies. Their identities and acquisition
records are maintained in
`paper-release/provenance/external_dependencies.csv`.

The manuscript uses the Allen Institute Human Immune Health Atlas, the
Kang PBMC dataset (GEO GSE96583; the analyzed processed object is distributed
through Pertpy), and the mouse-spleen RNA/ATAC inputs distributed with
MultiMAP (E-MTAB-9769 and E-MTAB-6714). The provenance CSV identifies the
specific input paths, acquisition records, preparation steps and checksums.
Rows describing locally derived inputs are not direct download links.

Public access to the original studies does not supply this project's generated
Supplementary Data 1–4. Those files comprise the analysis/environment index,
HIHA results, PBMC results and mouse-spleen results, respectively; their
delivery location remains pending. No raw biological data are redistributed
by this documentation update.

## Licensing

Software is licensed under the MIT License in `LICENSE`. Author-created
documentation and paper-facing evidence are licensed under CC BY 4.0. See
`LICENSE_SCOPE.md` for scope and exclusions. Copyright-holder and institution
fields remain private-draft TODOs and must be completed before publication.

## Citation and contact

Manuscript authors, in order:

1. **Zhili Lin** — Department of Hematology, The First Affiliated Hospital of
   Wenzhou Medical University, Wenzhou, Zhejiang, China.
2. **Zhouxiang Jin (corresponding author)** — Department of Hepatobiliary
   Surgery, The Second Affiliated Hospital and Yuying Children's Hospital of
   Wenzhou Medical University, Wenzhou, Zhejiang, China.

Correspondence: **wzjinzx@163.com**. Citation metadata are provided in
[CITATION.cff](CITATION.cff). No publication DOI or accepted-publication status
is claimed. The manuscript author list does not establish software copyright
ownership; copyright-holder and institution fields still require confirmation.
