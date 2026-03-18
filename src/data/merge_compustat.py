"""
Merge Compustat firm-level accounting data with carbon emissions from Trucost.

Pipeline:
    1. Load Compustat NA and Global annual files
    2. Filter to relevant countries (defendant + peer firm jurisdictions)
    3. Fuzzy-match Trucost company names to Compustat company names
    4. Merge Scope 1 and Scope 2 emissions at the firm-year level
    5. Compute emissions intensity (tonnes CO2e per dollar of revenue)

Data sources:
    - Compustat North America and Global (annual fundamentals)
    - Trucost / S&P Global: Scope 1 & Scope 2 GHG emissions
"""

import pandas as pd
from tqdm import tqdm

from src.utils.config import EU_COUNTRIES, get_paths
from src.utils.matching import fuzzy_match_entities


def load_compustat(
    na_path: str,
    global_path: str,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate Compustat NA and Global annual data.

    Parameters
    ----------
    na_path, global_path : str
        Paths to the Compustat CSV files.
    countries : list of str or None
        ISO country codes to retain. If None, keep all.

    Returns
    -------
    pd.DataFrame
        Combined firm-year panel with accounting variables.
    """
    na = pd.read_csv(na_path)
    glob = pd.read_csv(global_path)
    combined = pd.concat([na, glob], ignore_index=True)

    if countries is not None:
        combined = combined[combined["loc"].isin(countries)]

    return combined


def load_trucost(dir_trucost: str) -> pd.DataFrame:
    """
    Load and merge Trucost emissions datasets (GHG direct + purchased).

    Parameters
    ----------
    dir_trucost : str
        Directory containing Trucost CSV exports.

    Returns
    -------
    pd.DataFrame
        Firm-year emissions data with Scope 1 and Scope 2 columns.
    """
    ghg_direct = pd.read_csv(dir_trucost + "EDX_025_GHG_CPUCPCU_AllYr_20230219.csv")
    ghg_purchased = pd.read_csv(dir_trucost + "EDX_025_GHG_PU_AllYr_20230219.csv")

    # Merge direct and purchased emissions
    trucost = ghg_direct.merge(
        ghg_purchased,
        on=["company_name", "fiscal_year"],
        how="outer",
        suffixes=("_direct", "_purchased"),
    )

    return trucost


def match_compustat_to_trucost(
    compustat_names: list[str],
    trucost_names: list[str],
    threshold: int = 85,
) -> pd.DataFrame:
    """
    Create a crosswalk between Compustat and Trucost company names.

    Parameters
    ----------
    compustat_names : list of str
        Unique company names from Compustat.
    trucost_names : list of str
        Unique company names from Trucost.
    threshold : int
        Minimum fuzzy match score to accept.

    Returns
    -------
    pd.DataFrame
        Crosswalk with columns: compustat_name, trucost_name, match_score.
    """
    matches = fuzzy_match_entities(
        source_names=compustat_names,
        target_names=trucost_names,
        score_threshold=threshold,
    )

    crosswalk = matches[matches["above_threshold"]].rename(columns={
        "source_name": "compustat_name",
        "matched_name": "trucost_name",
        "score": "match_score",
    })

    return crosswalk[["compustat_name", "trucost_name", "match_score"]]


def merge_emissions(
    compustat: pd.DataFrame,
    trucost: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge emissions data onto the Compustat panel via the name crosswalk.

    Also computes:
        - scope1_intensity: Scope 1 emissions / revenue
        - scope2_intensity: Scope 2 emissions / revenue

    Parameters
    ----------
    compustat : pd.DataFrame
        Compustat firm-year panel.
    trucost : pd.DataFrame
        Trucost emissions data.
    crosswalk : pd.DataFrame
        Name matching crosswalk.

    Returns
    -------
    pd.DataFrame
        Compustat panel enriched with emissions data.
    """
    # Link Compustat names to Trucost names
    compustat = compustat.merge(
        crosswalk,
        left_on="conm",
        right_on="compustat_name",
        how="left",
    )

    # Merge emissions
    merged = compustat.merge(
        trucost,
        left_on=["trucost_name", "fyear"],
        right_on=["company_name", "fiscal_year"],
        how="left",
    )

    # Compute intensities
    if "sale" in merged.columns:
        merged["scope1_intensity"] = merged["scope1_emissions"] / merged["sale"].replace(0, pd.NA)
        merged["scope2_intensity"] = merged["scope2_emissions"] / merged["sale"].replace(0, pd.NA)

    return merged


def build_compustat_emissions_panel(
    paths: dict | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: load, match, and merge Compustat with Trucost emissions.

    Parameters
    ----------
    paths : dict or None
        Data directory paths. If None, loads from config.

    Returns
    -------
    pd.DataFrame
        Analysis-ready firm-year panel.
    """
    if paths is None:
        paths = get_paths()

    # Determine relevant countries from litigation data
    cases = pd.read_csv(paths["output"] + "clean_all_cases.csv")
    case_countries = list(cases["Geography ISO"].unique()) if "Geography ISO" in cases.columns else []
    countries = list(set(case_countries + EU_COUNTRIES + ["KOR"]))

    # Load data
    compustat = load_compustat(
        na_path=paths["compustat"] + "na_industry_level.csv",
        global_path=paths["compustat"] + "global_industry_level.csv",
        countries=countries,
    )
    trucost = load_trucost(paths["trucost"])

    # Build crosswalk
    compustat_names = compustat["conm"].dropna().unique().tolist()
    trucost_names = trucost["company_name"].dropna().unique().tolist()
    crosswalk = match_compustat_to_trucost(compustat_names, trucost_names)

    # Merge
    panel = merge_emissions(compustat, trucost, crosswalk)

    return panel


if __name__ == "__main__":
    panel = build_compustat_emissions_panel()
    paths = get_paths()
    panel.to_csv(paths["output"] + "compustat_with_emissions.csv", index=False)
    print(f"Saved panel: {panel.shape[0]} firm-years, {panel['gvkey'].nunique()} firms")
