"""
Difference-in-differences analysis of climate litigation effects on defendant firms.

Implements the empirical strategy from Section 5 of the paper:
    - Match each defendant to a control firm (same sector, disclosure status,
      nearest on emissions and size)
    - Staggered DiD with firm, sector×year, and country×year fixed effects
    - Separate effects for cases with CAR[-2,2] < 0 vs. others
    - Short-term (1 year) vs. long-term (2+ years) decomposition

Outcomes:
    - Panel A: Scope 1 Emissions, Scope 1 Intensity, EPA Emissions, Residual
    - Panel B: Climate risk disclosures (Physical, Regulatory, Legal, Other Transition)
"""

import pandas as pd
import numpy as np

from src.utils.config import get_paths, MATCHING_VARIABLES
from src.utils.matching import match_treated_to_controls


def prepare_defendant_panel(
    cases: pd.DataFrame,
    cars: pd.DataFrame,
    compustat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the defendant firm-year panel for DiD analysis.

    Merges litigation events with CARs and accounting data,
    then constructs treatment timing indicators.

    Parameters
    ----------
    cases : pd.DataFrame
        Litigation events with defendant gvkeys.
    cars : pd.DataFrame
        Case-level CARs from the event study.
    compustat : pd.DataFrame
        Firm-year accounting and emissions panel.

    Returns
    -------
    pd.DataFrame
        Defendant panel with treatment indicators:
        - is_defendant: firm is a defendant
        - post_filing: observation is after litigation filing
        - negative_car: case had CAR[-2,2] < 0
        - post_negative_car: post-filing AND negative CAR
    """
    # Merge CARs with case data
    cases = cases.merge(
        cars[["gvkey", "event_date", "CAR"]],
        left_on=["gvkey_side_B", "First Litigation Event"],
        right_on=["gvkey", "event_date"],
        how="left",
    )

    # Flag negative-CAR cases
    cases["negative_car"] = cases["CAR"] < 0

    # Merge with Compustat panel
    defendants = cases[["gvkey_side_B", "First Litigation Event", "Year First Event",
                        "CAR", "negative_car", "Side A.0 nature"]].copy()
    defendants = defendants.rename(columns={"gvkey_side_B": "gvkey"})

    panel = compustat.merge(
        defendants,
        on="gvkey",
        how="inner",
    )

    # Construct treatment timing
    panel["years_since_filing"] = panel["fyear"] - panel["Year First Event"]
    panel["post_filing"] = panel["years_since_filing"] >= 0
    panel["post_1yr"] = panel["years_since_filing"] == 1
    panel["post_2yr_plus"] = panel["years_since_filing"] >= 2
    panel["post_negative_car"] = panel["post_filing"] & panel["negative_car"]
    panel["post_2yr_negative_car"] = panel["post_2yr_plus"] & panel["negative_car"]

    return panel


def construct_matched_sample(
    defendants: pd.DataFrame,
    all_firms: pd.DataFrame,
    match_vars: list[str] = MATCHING_VARIABLES,
    sector_col: str = "gind_2d",
    disclosure_col: str = "disclosed_physical_risk",
) -> pd.DataFrame:
    """
    Construct matched treatment-control pairs for the DiD.

    For each defendant firm-year, finds the nearest non-defendant firm
    in the same sector with the same physical risk disclosure status.

    Parameters
    ----------
    defendants : pd.DataFrame
        Defendant firm observations (year before filing).
    all_firms : pd.DataFrame
        Universe of potential control firms.
    match_vars : list of str
        Variables for nearest-neighbor distance.
    sector_col : str
        Industry classification for exact matching.
    disclosure_col : str
        Binary disclosure variable for exact matching.

    Returns
    -------
    pd.DataFrame
        Matched pairs with treatment and control firm IDs.
    """
    # Exclude defendants from the control pool
    defendant_ids = set(defendants["gvkey"].unique())
    control_pool = all_firms[~all_firms["gvkey"].isin(defendant_ids)].copy()

    exact_vars = [sector_col]
    if disclosure_col in defendants.columns:
        exact_vars.append(disclosure_col)

    matched = match_treated_to_controls(
        treated=defendants,
        pool=control_pool,
        match_vars=match_vars,
        exact_match_vars=exact_vars,
        n_neighbors=1,
    )

    return matched


def build_did_panel(
    matched_pairs: pd.DataFrame,
    compustat: pd.DataFrame,
    cases: pd.DataFrame,
    cars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full DiD panel for defendant and matched control firms.

    Parameters
    ----------
    matched_pairs : pd.DataFrame
        Treatment-control pairings.
    compustat : pd.DataFrame
        Full firm-year panel.
    cases, cars : pd.DataFrame
        Litigation events and CARs for treatment indicator construction.

    Returns
    -------
    pd.DataFrame
        Stacked panel with both treated and control firms,
        ready for regression analysis in Stata or statsmodels.
    """
    # Get treated firm panel
    treated_gvkeys = matched_pairs["treated_gvkey"].unique()
    control_gvkeys = matched_pairs["control_gvkey"].unique()

    all_gvkeys = list(set(treated_gvkeys) | set(control_gvkeys))
    panel = compustat[compustat["gvkey"].isin(all_gvkeys)].copy()

    # Add treatment indicators
    panel["is_defendant"] = panel["gvkey"].isin(treated_gvkeys)

    # Merge filing year for treated firms
    filing_years = cases.groupby("gvkey_side_B")["Year First Event"].min().reset_index()
    filing_years.columns = ["gvkey", "filing_year"]

    panel = panel.merge(filing_years, on="gvkey", how="left")

    # For control firms, assign the matched defendant's filing year
    control_filing = matched_pairs[["control_gvkey", "year"]].rename(
        columns={"control_gvkey": "gvkey", "year": "filing_year_control"}
    )
    panel = panel.merge(control_filing, on="gvkey", how="left")
    panel["filing_year"] = panel["filing_year"].fillna(panel["filing_year_control"])

    # Timing indicators
    panel["years_since_filing"] = panel["fyear"] - panel["filing_year"]
    panel["post_filing"] = panel["years_since_filing"] >= 0

    return panel


def export_for_stata(panel: pd.DataFrame, output_path: str) -> None:
    """Export the DiD panel as a Stata .dta file for regression analysis."""
    # Select relevant columns and clean for Stata compatibility
    cols_to_export = [
        "gvkey", "fyear", "datadate", "conm",
        "at", "sale", "scope1_emissions", "scope2_emissions",
        "scope1_intensity", "is_defendant", "post_filing",
        "years_since_filing", "filing_year",
    ]
    cols_available = [c for c in cols_to_export if c in panel.columns]
    panel[cols_available].to_stata(output_path, write_index=False)


if __name__ == "__main__":
    paths = get_paths()

    # Load inputs
    cases = pd.read_csv(paths["output"] + "clean_all_cases.csv")
    cases["First Litigation Event"] = pd.to_datetime(cases["First Litigation Event"])
    cases["Year First Event"] = cases["First Litigation Event"].dt.year

    cars = pd.read_csv(paths["cars"] + "CARs.csv")
    compustat = pd.read_csv(paths["output"] + "compustat_with_emissions.csv")

    # Build defendant panel
    defendant_panel = prepare_defendant_panel(cases, cars, compustat)

    # Export
    export_for_stata(defendant_panel, paths["output"] + "defendant_firm_lvl.dta")
    print(f"Exported firm-level panel: {defendant_panel.shape[0]} obs")
