from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[7]
UNIT = Path(__file__).resolve().parents[2]
FITS = UNIT / "regenerated/fits"
FOCUSED = (
    REPO
    / "results/PBMC/sensitivity/rho_attribution_tau_surface_alpha0_range075_175"
    / "tables/rho_tau_surface_range075_175_by_seed.csv"
)
ALPHA = (
    REPO
    / "results/PBMC/sensitivity/rho_attribution_alpha_search/tables"
    / "rho_attribution_by_replicate.csv"
)
VARIANTS = ("heterogeneous", "mean_matched_uniform")


def value_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def endpoint_slug(value: str) -> str:
    return value.lower().replace("+", "").replace("-", "").replace(" ", "_")


def fit_roots() -> list[Path]:
    focused = pd.read_csv(FOCUSED)
    alpha = pd.read_csv(ALPHA)
    if len(focused) != 225:
        raise RuntimeError(f"Expected 225 focused pairs, found {len(focused)}")
    if len(alpha) != 75:
        raise RuntimeError(f"Expected 75 alpha-search pairs, found {len(alpha)}")
    roots: list[Path] = []
    for row in focused.itertuples(index=False):
        case = (
            f"tau_min_{value_slug(row.tau_min)}_"
            f"tau_max_{value_slug(row.tau_max)}"
        )
        base = (
            FITS
            / "focused_tau"
            / endpoint_slug(str(row.endpoint))
            / f"seed{int(row.seed)}"
            / case
        )
        roots.extend(base / variant for variant in VARIANTS)
    for row in alpha.itertuples(index=False):
        case = f"alpha_{value_slug(row.alpha)}"
        base = (
            FITS
            / "alpha_search"
            / endpoint_slug(str(row.endpoint))
            / f"seed{int(row.seed)}"
            / case
        )
        roots.extend(base / variant for variant in VARIANTS)
    if len(roots) != 600 or len(set(roots)) != 600:
        raise RuntimeError("Expected 600 unique analysis-scoped fit roots")
    return roots


def main() -> None:
    missing: list[str] = []
    roots = fit_roots()
    for root in roots:
        for name in ("resolved_config.yaml", "fit_manifest.yaml", "cell_scores.parquet"):
            path = root / name
            if not path.is_file():
                missing.append(str(path.relative_to(REPO)))
    if missing:
        raise RuntimeError(
            f"Gate 2 lineage incomplete: {len(missing)} required fit artifacts "
            f"missing; first={missing[0]}"
        )
    print("Gate 2 lineage complete: 600 fits, 1800 required artifacts")


if __name__ == "__main__":
    main()
