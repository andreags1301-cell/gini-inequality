"""Data import and reshaping.

Mirrors sections 1-4 of the legacy R script: load the long-format Gini data,
pivot it to wide format (one column per income concept), attach regional labels
and build the analysis frames used by every estimator.
"""
from __future__ import annotations

import pandas as pd

from . import config


def load_long(path=config.DATA_FILE) -> pd.DataFrame:
    """Load the raw long-format data and coerce column types."""
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["gini"] = pd.to_numeric(df["gini"], errors="coerce")
    df["cnt"] = df["cnt"].str.strip()
    df["inc_cat"] = df["inc_cat"].str.strip()
    return df.dropna(subset=["gini"])


def _assign_region(code: str) -> str:
    for region, members in config.REGIONS.items():
        if code in members:
            return region
    return "Other"


def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long -> wide (one column per income concept) and add region."""
    wide = (
        df_long.pivot_table(index=["cnt", "year"], columns="inc_cat", values="gini")
        .reset_index()
        .sort_values(["cnt", "year"])
    )
    wide.columns.name = None
    wide["region"] = wide["cnt"].map(_assign_region)
    return wide


def analysis_frames(wide: pd.DataFrame) -> dict:
    """Build the filtered frames each estimator needs.

    Returns a dict with:
      - ``AB``:  rows with earng, ehearng and ehdinc_ present (Models A and B)
      - ``C``:   rows with ehearng and ehginc present (Model C)
      - ``between``: 32 country time-averages plus redistribution measures
    """
    df_ab = wide.dropna(subset=["earng", "ehearng", "ehdinc_"]).copy()
    df_c = wide.dropna(subset=["ehearng", "ehginc"]).copy()

    between = (
        df_ab.groupby(["cnt", "region"], as_index=False)[config.INCOME_CONCEPTS]
        .mean()
    )
    # Redistribution measures used by the reaction model (all in Gini points).
    between["reduc_total"] = between["earng"] - between["ehdinc_"]
    between["reduc_public"] = between["ehearng"] - between["ehdinc_"]
    between["reduc_transf"] = between["ehearng"] - between["ehginc"]
    between["gain_transf"] = between["ehginc"] - between["ehearng"]

    return {"AB": df_ab, "C": df_c, "between": between}


def build() -> dict:
    """Convenience entry point: load -> reshape -> frames."""
    wide = to_wide(load_long())
    frames = analysis_frames(wide)
    frames["wide"] = wide
    return frames
