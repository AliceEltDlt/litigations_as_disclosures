"""
Peer effects analysis: impact of climate litigations on non-defendant firms.

Implements Section 6 of the paper:
    - Identify N=1, 3, 5 closest industry peers for each defendant
    - Peers defined by 4-digit GICS industry, matched on emissions + size
    - DiD comparing peer firms to more distant firms in the same industry
    - Outcomes: Scope 1 emissions, EPA emissions, climate risk disclosures

Key finding: limited peer effects on emissions, but significant increase
in voluntary climate risk disclosures (especially legal risk) among peers.
"""

import pandas as pd
import numpy as np

from src.utils.config import get_paths, MATCHING_VARIABLES
from src.utils.matching import find_industry_peers


def identify_peer_firms(
    defendants: pd.DataFrame,
    all_firms: pd.DataFrame,
    max_peers: int = 5,
    sector_col: str = "gind_4d",
) -> pd.DataFrame:
    """
    Identify the closest industry peers for each defendant firm.

    Parameters
    ----------
    defendants : pd.DataFrame
        Defendant firm-year observations.
    all_firms : pd.DataFrame
        Universe of firms in the Compustat panel.
    max_peers : int
        Maximum number of peers per defendant (default: 5 for N=1,3,5 analysis).
    sector_col : str
        4-digit GICS industry code column.

    Returns
    -------
    pd.DataFrame
        Peer assignments: defendant_gvkey, peer_gvkey, year, peer_rank, distance.
    """
    peers = find_industry_peers(
        defendants=defendants,
        all_firms=all_firms,
        match_vars=MATCHING_VARIABLES,
        sector_col=sector_col,
        n_peers=max_peers,
    )

    return peers


def build_peer_did_panel(
    peers: pd.DataFrame,
    compustat: pd.DataFrame,
    cases: pd.DataFrame,
    cars: pd.DataFrame,
    n_peers: int = 1,
) -> pd.DataFrame:
    """
    Build DiD panel for peer effects analysis at a specific N.

    Treatment group: N closest peers of defendant firms.
    Control group: Remaining firms in the same 4-digit GICS industry
    that are neither defendants nor close peers.

    Parameters
    ----------
    peers : pd.DataFrame
        Peer assignments from identify_peer_firms.
    compustat : pd.DataFrame
        Full firm-year panel.
    cases : pd.DataFrame
        Litigation events.
    cars : pd.DataFrame
        CARs for identifying negative-CAR cases.
    n_peers : int
        Number of peers to include (1, 3, or 5).

    Returns
    -------
    pd.DataFrame
        Peer DiD panel with treatment indicators:
        - is_peer: firm is among the N closest peers
        - post_filing: observation is after the associated litigation filing
        - associated_negative_car: associated defendant had CAR < 0
    """
    # Filter to top N peers
    peer_subset = peers[peers["neighbor_rank"] <= n_peers].copy()

    # Get peer and defendant gvkeys
    peer_gvkeys = set(peer_subset["peer_gvkey"].unique())
    defendant_gvkeys = set(peers["defendant_gvkey"].unique())

    # Merge CARs to identify negative-CAR cases
    case_cars = cases.merge(
        cars[["gvkey", "event_date", "CAR"]],
        left_on=["gvkey_side_B", "First Litigation Event"],
        right_on=["gvkey", "event_date"],
        how="left",
    )
    case_cars["negative_car"] = case_cars["CAR"] < 0

    # Link peers to their defendants' filing info
    peer_filing = peer_subset.merge(
        case_cars[["gvkey_side_B", "Year First Event", "negative_car"]].drop_duplicates(),
        left_on="defendant_gvkey",
        right_on="gvkey_side_B",
        how="left",
    )

    # Build the panel
    # Include: peers + all non-defendant, non-peer firms in same industries
    relevant_industries = compustat[
        compustat["gvkey"].isin(defendant_gvkeys)
    ]["gind_4d"].unique()

    panel = compustat[compustat["gind_4d"].isin(relevant_industries)].copy()
    panel = panel[~panel["gvkey"].isin(defendant_gvkeys)]  # Exclude defendants

    # Mark peers
    panel["is_peer"] = panel["gvkey"].isin(peer_gvkeys)

    # Merge filing year and CAR status from associated defendant
    peer_info = peer_filing.groupby("peer_gvkey").agg({
        "Year First Event": "min",
        "negative_car": "any",
    }).reset_index().rename(columns={
        "peer_gvkey": "gvkey",
        "Year First Event": "associated_filing_year",
        "negative_car": "associated_negative_car",
    })

    panel = panel.merge(peer_info, on="gvkey", how="left")

    # Timing indicators
    panel["years_since_filing"] = panel["fyear"] - panel["associated_filing_year"]
    panel["post_filing"] = panel["years_since_filing"] >= 0
    panel["post_negative_car"] = panel["post_filing"] & panel["associated_negative_car"].fillna(False)

    return panel


def run_peer_analysis(paths: dict | None = None) -> dict[int, pd.DataFrame]:
    """
    Run the full peer effects analysis for N=1, 3, 5.

    Returns
    -------
    dict
        {n_peers: panel_dataframe} for each specification.
    """
    if paths is None:
        paths = get_paths()

    cases = pd.read_csv(paths["output"] + "clean_all_cases.csv", sep=";")
    cases["First Litigation Event"] = pd.to_datetime(cases["First Litigation Event"])
    cases["Year First Event"] = cases["First Litigation Event"].dt.year

    cars = pd.read_csv(paths["cars"] + "CARs.csv")
    compustat = pd.read_csv(paths["output"] + "compustat_with_emissions.csv")

    # Identify defendant observations
    defendant_gvkeys = cases["gvkey_side_B"].dropna().unique()
    defendants = compustat[compustat["gvkey"].isin(defendant_gvkeys)].copy()

    # Find peers
    all_peers = identify_peer_firms(defendants, compustat, max_peers=5)

    # Build panels for each N
    panels = {}
    for n in [1, 3, 5]:
        panel = build_peer_did_panel(all_peers, compustat, cases, cars, n_peers=n)
        panels[n] = panel
        print(f"N={n}: {panel.shape[0]} obs, {panel['is_peer'].sum()} peer firm-years")

    return panels


if __name__ == "__main__":
    panels = run_peer_analysis()
    paths = get_paths()
    for n, panel in panels.items():
        panel.to_stata(paths["output"] + f"peer_effects_N{n}.dta", write_index=False)
    print("Exported peer effects panels")
