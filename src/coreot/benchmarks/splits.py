from __future__ import annotations

import math

import numpy as np
import pandas as pd


class SplitError(ValueError):
    """Raised when benchmark splitting cannot produce query/reference domains."""


def make_query_mask(obs: pd.DataFrame, strategy: str, query_fraction: float, seed: int) -> pd.Series:
    if not 0 < query_fraction < 1:
        raise SplitError("split.query_fraction must be strictly between 0 and 1")

    if strategy == "donor_aware":
        return _donor_aware_query_mask(obs, query_fraction, seed)
    if strategy == "stratified_donor_aware":
        return _stratified_donor_aware_query_mask(obs, query_fraction, seed)
    if strategy == "stratified_random":
        return _stratified_random_query_mask(obs, query_fraction, seed)
    raise SplitError(f"Unsupported split strategy: {strategy}")


def _donor_aware_query_mask(obs: pd.DataFrame, query_fraction: float, seed: int) -> pd.Series:
    donors = sorted(str(donor) for donor in obs["donor_id"].unique())
    if len(donors) < 2:
        raise SplitError("donor_aware split requires at least two donor_id values")

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(donors))
    n_query_donors = min(max(1, round(len(donors) * query_fraction)), len(donors) - 1)
    query_donors = set(shuffled[:n_query_donors])
    return obs["donor_id"].astype(str).isin(query_donors)


def _stratified_donor_aware_query_mask(
    obs: pd.DataFrame, query_fraction: float, seed: int
) -> pd.Series:
    donors = _donor_strata(obs)
    if len(donors) < 2:
        raise SplitError("stratified_donor_aware split requires at least two donor_id values")

    rng = np.random.default_rng(seed)
    query_donors: set[str] = set()
    for _, group in donors.groupby("stratum", sort=True):
        stratum_donors = group["donor_id"].astype(str).to_numpy()
        rng.shuffle(stratum_donors)
        if len(stratum_donors) == 1:
            continue
        n_query = min(max(1, round(len(stratum_donors) * query_fraction)), len(stratum_donors) - 1)
        query_donors.update(stratum_donors[:n_query])

    if not query_donors:
        return _donor_aware_query_mask(obs, query_fraction, seed)
    mask = obs["donor_id"].astype(str).isin(query_donors)
    if mask.all() or not mask.any():
        raise SplitError("stratified_donor_aware split must leave query and reference donors")
    return mask


def _donor_strata(obs: pd.DataFrame) -> pd.DataFrame:
    if "state_condition" in obs.columns:
        stratum_column = "state_condition"
    elif "disease_status" in obs.columns:
        stratum_column = "disease_status"
    else:
        return pd.DataFrame(
            {
                "donor_id": sorted(str(donor) for donor in obs["donor_id"].unique()),
                "stratum": "all",
            }
        )

    rows = []
    for donor, group in obs.groupby("donor_id", sort=True, observed=False):
        values = sorted(
            {
                str(value)
                for value in group[stratum_column]
                if pd.notna(value) and str(value) and str(value).lower() != "<na>"
            }
        )
        if len(values) == 1:
            stratum = values[0]
        elif not values:
            stratum = "unknown"
        else:
            stratum = "mixed"
        rows.append({"donor_id": str(donor), "stratum": stratum})
    return pd.DataFrame(rows)


def _stratified_random_query_mask(obs: pd.DataFrame, query_fraction: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    query_indices: list[object] = []

    for _, group in obs.groupby("cell_type", sort=True):
        n_query = min(max(1, math.floor(len(group) * query_fraction)), len(group))
        sampled_positions = rng.choice(group.index.to_numpy(), size=n_query, replace=False)
        query_indices.extend(sampled_positions.tolist())

    mask = obs.index.isin(query_indices)
    if mask.all() or not mask.any():
        raise SplitError("stratified_random split must leave nonempty query and reference domains")
    return pd.Series(mask, index=obs.index)
