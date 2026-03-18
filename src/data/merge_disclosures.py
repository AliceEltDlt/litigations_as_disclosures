"""
Merge CDP climate risk disclosure data with Compustat firm panel.

The Carbon Disclosure Project (CDP) collects voluntary survey responses
from firms on their climate risk exposure. This module:

    1. Loads multi-year CDP public data (emissions + risk disclosures)
    2. Handles duplicates across reporting years and emission scopes
    3. Fuzzy-matches CDP respondent names to Compustat firms
    4. Creates binary disclosure indicators:
       - Physical risk, Regulatory risk, Legal risk, Other Transition risk

Data sources:
    - CDP Public Climate Change Data (annual Excel files, 2013-2020)
    - Compustat for firm identifiers
"""

from typing import Optional

import pandas as pd

from src.utils.config import get_paths
from src.utils.matching import fuzzy_match_entities


def load_cdp_disclosures(cdp_dir: str, years: list[int] | None = None) -> pd.DataFrame:
    """
    Load CDP climate risk disclosure data across multiple years.

    Handles the evolving CDP questionnaire format (sheet names change
    across vintages).

    Parameters
    ----------
    cdp_dir : str
        Directory containing CDP Public Climate Change Data Excel files.
    years : list of int or None
        Years to load. If None, loads all available.

    Returns
    -------
    pd.DataFrame
        Firm-year disclosure records with risk type classifications.
    """
    if years is None:
        years = list(range(2013, 2021))

    all_disclosures = []

    # Sheet name mapping (CDP changed format over time)
    sheet_mappings = {
        2013: "8. Emissions Data",
        2014: "8. Emissions Data",
        2015: "CC8. Emissions Data",
        2016: "CC8. Emissions Data",
        2017: "CC8. Emissions Data",
    }

    for year in years:
        filepath = f"{cdp_dir}Public Climate Change Data/{year} Public Climate Change Data.xlsx"
        sheet = sheet_mappings.get(year, "CC8. Emissions Data")

        try:
            df = pd.read_excel(filepath, sheet_name=sheet)
            df["cdp_year"] = year
            all_disclosures.append(df)
        except (FileNotFoundError, ValueError):
            continue

    if not all_disclosures:
        return pd.DataFrame()

    return pd.concat(all_disclosures, ignore_index=True)


def deduplicate_cdp(disclosures: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate CDP records across reporting scopes and years.

    CDP data can contain multiple rows per firm-year when firms report
    emissions from different organizational boundaries or scopes.

    Parameters
    ----------
    disclosures : pd.DataFrame
        Raw CDP disclosure data.

    Returns
    -------
    pd.DataFrame
        Deduplicated firm-year records.
    """
    id_cols = ["account_id", "accounting_year"]

    if "row" in disclosures.columns:
        # Keep the first row per firm-year (aggregate scope)
        disclosures = disclosures.sort_values(id_cols + ["row"])
        disclosures = disclosures.drop_duplicates(subset=id_cols, keep="first")
    else:
        disclosures = disclosures.drop_duplicates(subset=id_cols, keep="first")

    return disclosures


def create_disclosure_indicators(disclosures: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary indicators for climate risk disclosure types.

    Categories:
        - disclosed_physical_risk: physical climate risk exposure
        - disclosed_regulatory_risk: regulatory/policy transition risk
        - disclosed_legal_risk: litigation/legal risk
        - disclosed_other_transition: market and reputation risk

    Parameters
    ----------
    disclosures : pd.DataFrame
        CDP disclosure data with risk type columns.

    Returns
    -------
    pd.DataFrame
        Firm-year level binary disclosure indicators.
    """
    indicators = disclosures[["account_id", "organization", "accounting_year"]].copy()

    risk_col_mappings = {
        "disclosed_physical_risk": ["physical", "acute", "chronic"],
        "disclosed_regulatory_risk": ["regulatory", "policy", "carbon tax"],
        "disclosed_legal_risk": ["legal", "litigation"],
        "disclosed_other_transition": ["market", "reputation", "technology"],
    }

    for indicator_name, keywords in risk_col_mappings.items():
        # Search for keywords in risk-related columns
        risk_cols = [c for c in disclosures.columns if "risk" in c.lower()]
        if risk_cols:
            pattern = "|".join(keywords)
            indicators[indicator_name] = disclosures[risk_cols].apply(
                lambda row: row.str.contains(pattern, case=False, na=False).any(),
                axis=1,
            ).astype(int)
        else:
            indicators[indicator_name] = 0

    return indicators


def merge_disclosures_with_compustat(
    compustat: pd.DataFrame,
    disclosures: pd.DataFrame,
    match_threshold: int = 85,
) -> pd.DataFrame:
    """
    Merge CDP disclosure indicators onto the Compustat panel.

    Parameters
    ----------
    compustat : pd.DataFrame
        Compustat firm-year panel.
    disclosures : pd.DataFrame
        CDP disclosure indicators with organization names.
    match_threshold : int
        Fuzzy match threshold for name matching.

    Returns
    -------
    pd.DataFrame
        Compustat panel with disclosure indicator columns added.
    """
    # Build name crosswalk
    compustat_names = compustat["conm"].dropna().unique().tolist()
    cdp_names = disclosures["organization"].dropna().unique().tolist()

    matches = fuzzy_match_entities(
        source_names=compustat_names,
        target_names=cdp_names,
        score_threshold=match_threshold,
    )

    crosswalk = matches[matches["above_threshold"]].rename(columns={
        "source_name": "conm",
        "matched_name": "cdp_organization",
    })

    # Merge crosswalk → disclosures → compustat
    compustat = compustat.merge(crosswalk[["conm", "cdp_organization"]], on="conm", how="left")

    disclosure_cols = [c for c in disclosures.columns if c.startswith("disclosed_")]
    merge_cols = ["cdp_organization", "accounting_year"] + disclosure_cols

    compustat = compustat.merge(
        disclosures[["organization", "accounting_year"] + disclosure_cols].rename(
            columns={"organization": "cdp_organization"}
        ),
        left_on=["cdp_organization", "fyear"],
        right_on=["cdp_organization", "accounting_year"],
        how="left",
    )

    # Fill missing disclosures with 0 (non-respondents treated as non-disclosers)
    for col in disclosure_cols:
        compustat[col] = compustat[col].fillna(0).astype(int)

    return compustat


if __name__ == "__main__":
    paths = get_paths()

    disclosures = load_cdp_disclosures(paths["cdp"])
    disclosures = deduplicate_cdp(disclosures)
    indicators = create_disclosure_indicators(disclosures)

    print(f"CDP disclosures: {indicators.shape[0]} firm-year records")
    for col in [c for c in indicators.columns if c.startswith("disclosed_")]:
        print(f"  {col}: {indicators[col].sum()} firms disclosing")
