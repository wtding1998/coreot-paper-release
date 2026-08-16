from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[7]
UNIT = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results/PBMC/sensitivity/component_ablation/tables/component_ablation_by_seed.csv"
FITS = UNIT / "regenerated/fits"


def value_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def endpoint_slug(value: str) -> str:
    return value.lower().replace("+", "").replace("-", "").replace(" ", "_")


def fit_root(row: object) -> Path:
    return (
        FITS
        / endpoint_slug(str(row.endpoint))
        / f"seed{int(row.seed)}"
        / f"tau{value_slug(row.tau_source)}_alpha{value_slug(row.alpha)}"
    )


def main() -> None:
    frame = pd.read_csv(SOURCE)
    frame = frame.loc[
        frame["experiment"].eq("pbmc")
        & frame["variant"].eq("compatibility_only")
    ].reset_index(drop=True)
    if len(frame) != 375:
        raise RuntimeError(f"Expected 375 compatibility fits, found {len(frame)}")

    missing: list[str] = []
    for row in frame.itertuples(index=False):
        root = fit_root(row)
        for name in ("resolved_config.yaml", "fit_manifest.yaml", "cell_scores.parquet"):
            path = root / name
            if not path.is_file():
                missing.append(str(path.relative_to(REPO)))
    if missing:
        raise RuntimeError(
            f"Gate 2 lineage incomplete: {len(missing)} required fit artifacts missing; "
            f"first={missing[0]}"
        )
    print("Gate 2 lineage complete: 375 fits, 1125 required artifacts")


if __name__ == "__main__":
    main()

