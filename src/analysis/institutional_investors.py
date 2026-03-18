"""
Institutional investor ownership analysis around climate litigation filings.

Implements Section 4.2 of the paper:
    - Process 13f holdings data (Thomson/Refinitiv S34)
    - Classify investors: UNPRI signatories, E-tilt, Active Share
    - Track ownership changes for defendant vs. matched control firms
    - Panel regressions with firm, sector×quarter, country×quarter FEs

Key investor categories (following Pastor, Stambaugh, Taylor 2023):
    - UNPRI Signatories
    - UNPRI × Top Half(Negative E-tilt × Active Share)
    - Non-UNPRI × Top Half(Negative E-tilt × Active Share)
    - Top Half(Positive E-tilt × Active Share)
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import get_paths
from src.utils.matching import fuzzy_match_entities


def load_13f_holdings(holdings_dir: str, cusips: list[str]) -> pd.DataFrame:
    """
    Load quarterly 13f institutional holdings for relevant firms.

    Parameters
    ----------
    holdings_dir : str
        Directory containing S34 holding files.
    cusips : list of str
        8-digit CUSIPs to filter for.

    Returns
    -------
    pd.DataFrame
        Quarterly holdings: cusip, manager_id, manager_name, shares, date.
    """
    all_holdings = []

    for filename in sorted(os.listdir(holdings_dir)):
        if not filename.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(holdings_dir, filename))

        # Filter to relevant CUSIPs
        if "cusip" in df.columns:
            df = df[df["cusip"].str[:8].isin(cusips)]

        all_holdings.append(df)

    if not all_holdings:
        return pd.DataFrame()

    return pd.concat(all_holdings, ignore_index=True)


def classify_unpri_signatories(
    managers: pd.DataFrame,
    unpri_list_path: str,
    match_threshold: int = 90,
) -> pd.DataFrame:
    """
    Identify UNPRI signatory status for institutional managers.

    Uses fuzzy matching between S34 manager names and the UNPRI
    signatory database.

    Parameters
    ----------
    managers : pd.DataFrame
        Unique managers from 13f data with 'mgrname' column.
    unpri_list_path : str
        Path to UNPRI signatory list.
    match_threshold : int
        Fuzzy match acceptance score.

    Returns
    -------
    pd.DataFrame
        Managers with added 'is_unpri_signatory' boolean column.
    """
    unpri = pd.read_csv(unpri_list_path)
    unpri_names = unpri["signatory_name"].dropna().unique().tolist()
    mgr_names = managers["mgrname"].dropna().unique().tolist()

    matches = fuzzy_match_entities(
        source_names=mgr_names,
        target_names=unpri_names,
        score_threshold=match_threshold,
    )

    signatory_names = set(matches[matches["above_threshold"]]["source_name"])
    managers["is_unpri_signatory"] = managers["mgrname"].isin(signatory_names)

    return managers


def compute_e_tilts(
    holdings: pd.DataFrame,
    emissions: pd.DataFrame,
    market_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Environmental Tilts following Pastor, Stambaugh, Taylor (2023).

    E-tilt measures how much an investor's portfolio over- or under-weights
    stocks based on their environmental characteristics relative to the market.

    Parameters
    ----------
    holdings : pd.DataFrame
        Quarterly portfolio holdings with portfolio weights.
    emissions : pd.DataFrame
        Firm-level emissions data for environmental scoring.
    market_weights : pd.DataFrame
        Market capitalization weights.

    Returns
    -------
    pd.DataFrame
        Manager-quarter level E-tilts and Active Share.
    """
    # Merge emissions scores onto holdings
    holdings = holdings.merge(emissions, on=["cusip", "quarter"], how="left")
    holdings = holdings.merge(market_weights, on=["cusip", "quarter"], how="left")

    # Portfolio weight
    holdings["port_weight"] = holdings["shares"] / holdings.groupby(
        ["manager_id", "quarter"]
    )["shares"].transform("sum")

    # E-tilt = sum of (portfolio_weight - market_weight) × environmental_score
    holdings["weight_diff"] = holdings["port_weight"] - holdings["market_weight"]
    holdings["e_tilt_contrib"] = holdings["weight_diff"] * holdings["env_score"]

    e_tilts = holdings.groupby(["manager_id", "quarter"]).agg(
        e_tilt=("e_tilt_contrib", "sum"),
        active_share=("weight_diff", lambda x: np.abs(x).sum() / 2),
    ).reset_index()

    return e_tilts


def compute_ownership_percentages(
    holdings: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    manager_classifications: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute percentage ownership by investor category for each firm-quarter.

    Categories:
        - Total institutional ownership
        - UNPRI signatory ownership
        - Top-quartile E-tilt × Active Share
        - Various cross-classifications

    Parameters
    ----------
    holdings : pd.DataFrame
        Quarterly holdings data.
    shares_outstanding : pd.DataFrame
        Firm-quarter shares outstanding.
    manager_classifications : pd.DataFrame
        Manager-level classifications (UNPRI, E-tilt, Active Share).

    Returns
    -------
    pd.DataFrame
        Firm-quarter ownership percentages by investor category.
    """
    # Merge manager classifications
    merged = holdings.merge(manager_classifications, on="manager_id", how="left")

    # Aggregate by firm-quarter and investor type
    ownership = merged.groupby(["gvkey", "fdate"]).apply(
        lambda g: pd.Series({
            "pct_io_total": g["shares"].sum(),
            "pct_io_unpri": g.loc[g["is_unpri_signatory"], "shares"].sum(),
            "pct_io_high_etilt_as": g.loc[g["high_etilt_active_share"], "shares"].sum()
            if "high_etilt_active_share" in g.columns else 0,
        })
    ).reset_index()

    # Convert to percentage of shares outstanding
    ownership = ownership.merge(shares_outstanding, on=["gvkey", "fdate"], how="left")
    for col in ["pct_io_total", "pct_io_unpri", "pct_io_high_etilt_as"]:
        ownership[col] = ownership[col] / ownership["total_shares"] * 100

    return ownership


if __name__ == "__main__":
    paths = get_paths()
    print("Institutional ownership module ready.")
    print("Run build_institutional_ownership.py for the full pipeline.")
