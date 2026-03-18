"""
Build the master analysis-ready dataset by chaining all data modules.

Pipeline order:
    1. Prepare litigation data (Sabin + LSE → unified cases)
    2. Merge Compustat NA + Global with Trucost emissions
    3. Merge EPA FLIGHT facility-level data
    4. Merge CDP climate risk disclosures
    5. Aggregate patent data (KPSS)
    6. Build institutional ownership panel (13f + UNPRI + E-tilts)
    7. Link litigation events to firm panel → master dataset

Each step can be run independently or as part of the full pipeline.
Intermediate outputs are cached to avoid reprocessing.
"""

import os
from pathlib import Path

import pandas as pd

from src.utils.config import get_paths


def step_1_litigation_data(paths: dict, force: bool = False) -> pd.DataFrame:
    """Prepare unified litigation dataset."""
    output = paths["output"] + "clean_all_cases.csv"
    if os.path.exists(output) and not force:
        print("  Step 1: Loading cached litigation data")
        return pd.read_csv(output)

    from src.data.prepare_litigation_data import create_unified_dataset

    print("  Step 1: Preparing litigation data...")
    dataset = create_unified_dataset(
        sabin_path=paths["litigation_data"] + "US-Case-Bundles-2022-04-07.csv",
        lse_path=paths["litigation_data"] + "nonUS_cases_16122022.csv",
        google_path=paths["hand_collected"] + "clean_US_cases_with_Google.csv",
        complaint_path=paths["complaints"] + "Complaint_data.csv",
    )
    dataset.to_csv(output, index=False)
    return dataset


def step_2_compustat_emissions(paths: dict, force: bool = False) -> pd.DataFrame:
    """Merge Compustat with Trucost emissions."""
    output = paths["output"] + "compustat_with_emissions.csv"
    if os.path.exists(output) and not force:
        print("  Step 2: Loading cached Compustat+emissions panel")
        return pd.read_csv(output)

    from src.data.merge_compustat import build_compustat_emissions_panel

    print("  Step 2: Merging Compustat with Trucost emissions...")
    panel = build_compustat_emissions_panel(paths)
    panel.to_csv(output, index=False)
    return panel


def step_3_epa_data(paths: dict, compustat: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Merge EPA FLIGHT facility data."""
    output = paths["output"] + "compustat_with_epa.csv"
    if os.path.exists(output) and not force:
        print("  Step 3: Loading cached EPA merge")
        return pd.read_csv(output)

    from src.data.merge_epa import (
        load_flight_data,
        match_flight_to_compustat,
        aggregate_facility_emissions,
    )

    print("  Step 3: Merging EPA FLIGHT data...")
    flight = load_flight_data(paths["epa"])
    compustat_names = compustat["conm"].dropna().unique().tolist()
    flight_names = flight["parent_company"].dropna().unique().tolist()

    crosswalk = match_flight_to_compustat(flight_names, compustat_names)
    epa_agg = aggregate_facility_emissions(flight, crosswalk)

    panel = compustat.merge(epa_agg, left_on=["conm", "fyear"],
                            right_on=["compustat_company", "year"], how="left")
    panel.to_csv(output, index=False)
    return panel


def step_4_disclosures(paths: dict, compustat: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Merge CDP disclosure indicators."""
    output = paths["output"] + "compustat_with_disclosures.csv"
    if os.path.exists(output) and not force:
        print("  Step 4: Loading cached disclosure merge")
        return pd.read_csv(output)

    from src.data.merge_disclosures import (
        load_cdp_disclosures,
        deduplicate_cdp,
        create_disclosure_indicators,
        merge_disclosures_with_compustat,
    )

    print("  Step 4: Merging CDP disclosures...")
    cdp = load_cdp_disclosures(paths["cdp"])
    cdp = deduplicate_cdp(cdp)
    indicators = create_disclosure_indicators(cdp)

    panel = merge_disclosures_with_compustat(compustat, indicators)
    panel.to_csv(output, index=False)
    return panel


def step_5_patents(paths: dict, force: bool = False) -> pd.DataFrame:
    """Aggregate patent data."""
    output = paths["patents"] + "firm_year_patent_counts.csv"
    if os.path.exists(output) and not force:
        print("  Step 5: Loading cached patent aggregates")
        return pd.read_csv(output)

    from src.data.aggregate_patents import aggregate_monthly_patents

    print("  Step 5: Aggregating patent data...")
    patents = aggregate_monthly_patents(paths["patents"] + "By_Month/")
    patents.to_csv(output, index=False)
    return patents


def build_master_dataset(force: bool = False) -> pd.DataFrame:
    """
    Run the full data pipeline and produce the master analysis dataset.

    Parameters
    ----------
    force : bool
        If True, recompute all steps even if cached outputs exist.

    Returns
    -------
    pd.DataFrame
        Master firm-year panel ready for analysis.
    """
    paths = get_paths()
    print("Building master dataset...")

    # Step 1: Litigation data
    cases = step_1_litigation_data(paths, force)
    print(f"  → {cases.shape[0]} litigation events")

    # Step 2: Compustat + emissions
    compustat = step_2_compustat_emissions(paths, force)
    print(f"  → {compustat.shape[0]} firm-year observations")

    # Step 3: EPA facility data
    compustat = step_3_epa_data(paths, compustat, force)

    # Step 4: CDP disclosures
    compustat = step_4_disclosures(paths, compustat, force)

    # Step 5: Patent data
    patents = step_5_patents(paths, force)

    # Merge patents into main panel
    compustat = compustat.merge(
        patents.rename(columns={"assignee_organization": "conm", "year": "fyear"}),
        on=["conm", "fyear"],
        how="left",
    )
    compustat["n_patents"] = compustat["n_patents"].fillna(0)

    # Save master dataset
    master_output = paths["output"] + "master_panel.csv"
    compustat.to_csv(master_output, index=False)
    print(f"\nMaster dataset saved: {compustat.shape[0]} rows, {compustat.shape[1]} columns")
    print(f"  Unique firms: {compustat['gvkey'].nunique()}")
    print(f"  Year range: {compustat['fyear'].min()}-{compustat['fyear'].max()}")

    return compustat


if __name__ == "__main__":
    build_master_dataset(force=False)
