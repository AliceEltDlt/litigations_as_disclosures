"""
Link EPA FLIGHT facility-level emissions data to Compustat firms.

The EPA Greenhouse Gas Reporting Program (GHGRP / FLIGHT) requires U.S. facilities
emitting >25,000 metric tons CO2e to report direct emissions. This module:

    1. Loads multi-year FLIGHT data from EPA Excel exports
    2. Matches facility parent companies to Compustat firms (fuzzy + manual)
    3. Aggregates facility emissions to the firm-year level
    4. Enables decomposition of Scope 1 reductions into EPA-regulated vs. residual

Data sources:
    - EPA FLIGHT database (facility-level, annual)
    - Compustat for firm identifiers
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import get_paths
from src.utils.matching import fuzzy_match_entities


def similar(a: str, b: str) -> float:
    """Compute string similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def load_flight_data(epa_dir: str) -> pd.DataFrame:
    """
    Load EPA FLIGHT emissions data from Excel exports.

    Handles multi-sheet Excel files with data across reporting years.

    Parameters
    ----------
    epa_dir : str
        Directory containing FLIGHT Excel files.

    Returns
    -------
    pd.DataFrame
        Facility-year level emissions with columns:
        facility_id, facility_name, parent_company, year,
        total_emissions_mt_co2e, latitude, longitude, state.
    """
    sheets = pd.read_excel(epa_dir + "flight.xls", skiprows=6, sheet_name=None)

    all_years = []
    for sheet_name, df in sheets.items():
        df["source_sheet"] = sheet_name
        all_years.append(df)

    flight = pd.concat(all_years, ignore_index=True)
    return flight


def match_flight_to_compustat(
    flight_companies: List[str],
    compustat_companies: List[str],
    manual_matches: Optional[Dict[str, str]] = None,
    threshold: int = 85,
) -> pd.DataFrame:
    """
    Match EPA FLIGHT parent company names to Compustat firms.

    Combines fuzzy matching with optional manual overrides for
    difficult cases (e.g., subsidiary names, abbreviations).

    Parameters
    ----------
    flight_companies : list of str
        Unique parent company names from FLIGHT.
    compustat_companies : list of str
        Unique company names from Compustat.
    manual_matches : dict or None
        Manual overrides: {flight_name: compustat_name}.
    threshold : int
        Fuzzy match acceptance threshold.

    Returns
    -------
    pd.DataFrame
        Crosswalk: flight_company, compustat_company, match_type, score.
    """
    # Start with fuzzy matching
    fuzzy = fuzzy_match_entities(
        source_names=flight_companies,
        target_names=compustat_companies,
        score_threshold=threshold,
    )

    crosswalk = fuzzy.rename(columns={
        "source_name": "flight_company",
        "matched_name": "compustat_company",
    })
    crosswalk["match_type"] = np.where(
        crosswalk["above_threshold"], "fuzzy_auto", "unmatched"
    )

    # Apply manual overrides
    if manual_matches:
        for flight_name, compustat_name in manual_matches.items():
            mask = crosswalk["flight_company"] == flight_name
            crosswalk.loc[mask, "compustat_company"] = compustat_name
            crosswalk.loc[mask, "match_type"] = "manual"
            crosswalk.loc[mask, "above_threshold"] = True

    return crosswalk


def aggregate_facility_emissions(
    flight: pd.DataFrame,
    crosswalk: pd.DataFrame,
    emissions_col: str = "total_emissions_mt_co2e",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Aggregate facility-level emissions to the firm-year level.

    Parameters
    ----------
    flight : pd.DataFrame
        Facility-year emissions data.
    crosswalk : pd.DataFrame
        Mapping from FLIGHT parent companies to Compustat firms.
    emissions_col : str
        Column name for emissions values.
    year_col : str
        Column name for reporting year.

    Returns
    -------
    pd.DataFrame
        Firm-year level: compustat_company, year, epa_total_emissions,
        n_facilities, facility_list.
    """
    # Merge crosswalk
    matched = flight.merge(
        crosswalk[crosswalk["above_threshold"]][["flight_company", "compustat_company"]],
        left_on="parent_company",
        right_on="flight_company",
        how="inner",
    )

    # Aggregate
    agg = matched.groupby(["compustat_company", year_col]).agg(
        epa_total_emissions=(emissions_col, "sum"),
        n_facilities=("facility_id", "nunique"),
    ).reset_index()

    return agg


def track_facility_ownership_changes(
    flight: pd.DataFrame,
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Track facility ownership transfers (sales to private entities, etc.).

    Used to decompose emission reductions into genuine abatement vs.
    divestiture of polluting assets (cf. Duchin et al., 2022).

    Parameters
    ----------
    flight : pd.DataFrame
        Multi-year facility data with parent company information.

    Returns
    -------
    pd.DataFrame
        Facility ownership changes with year, old_owner, new_owner, emissions.
    """
    # Sort by facility and year
    flight = flight.sort_values(["facility_id", year_col])

    # Detect ownership changes
    flight["prev_owner"] = flight.groupby("facility_id")["parent_company"].shift(1)
    changes = flight[
        (flight["prev_owner"].notna())
        & (flight["parent_company"] != flight["prev_owner"])
    ].copy()

    changes = changes.rename(columns={
        "prev_owner": "old_owner",
        "parent_company": "new_owner",
    })

    return changes


if __name__ == "__main__":
    paths = get_paths()
    flight = load_flight_data(paths["epa"])
    print(f"Loaded FLIGHT data: {flight.shape[0]} facility-year observations")
