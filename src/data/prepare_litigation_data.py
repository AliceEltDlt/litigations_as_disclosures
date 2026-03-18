"""
Prepare and unify litigation data from the Sabin Center for Climate Change Law
and the LSE Grantham Research Institute.

Pipeline:
    1. Parse Sabin Center U.S. case bundles → extract parties, dates, legal grounds
    2. Parse LSE/Grantham non-U.S. corporate litigation records
    3. Merge hand-collected data (Google search volume, complaint metadata)
    4. Create unified case-level dataset with standardized fields

Data sources:
    - Sabin Center: US-Case-Bundles CSV (Columbia Law School)
    - LSE Grantham: Non-US corporate litigation records
    - Hand-collected: Google Trends data, complaint factual allegation lengths
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import get_paths


def parse_sabin_cases(filepath: str) -> pd.DataFrame:
    """
    Parse Sabin Center U.S. climate litigation case bundles.

    Extracts claimant/defendant names, filing dates, legal grounds,
    and case descriptions from the raw CSV.

    Parameters
    ----------
    filepath : str
        Path to the Sabin Center CSV export.

    Returns
    -------
    pd.DataFrame
        Cleaned case-level data with columns:
        Title, Side_A (claimant), Side_B (defendant), filing_date,
        legal_grounds, description, geography.
    """
    cases = pd.read_csv(filepath)
    cases = cases.drop_duplicates()

    # Parse party names from case title
    sides = cases["Case Name"].str.split(" v. ", expand=True, n=1)
    if sides.shape[1] >= 2:
        sides.columns = ["Side_A", "Side_B"]
    else:
        sides.columns = ["Side_A"]
        sides["Side_B"] = np.nan

    cases = pd.concat([cases, sides], axis=1)

    # Clean text fields
    for col in ["Case Name", "Description"]:
        if col in cases.columns:
            cases[col] = cases[col].str.replace("\n", " ", regex=False).str.strip()

    return cases


def parse_event_timeline(cases: pd.DataFrame) -> pd.DataFrame:
    """
    Parse semicolon-delimited event strings into structured timeline.

    Events in the raw data are formatted as:
        "date1|event_type1;date2|event_type2;..."

    Parameters
    ----------
    cases : pd.DataFrame
        Must contain an 'Events' column with the delimited event string.

    Returns
    -------
    pd.DataFrame
        Original data with additional columns for each event
        (Event_0_date, Event_0_nature, Event_1_date, ...).
    """
    if "Events" not in cases.columns:
        return cases

    all_events = cases["Events"].str.split(pat=";", expand=True)

    for i in range(all_events.shape[1]):
        event_parts = all_events.iloc[:, i].str.split(pat="|", expand=True)
        if event_parts.shape[1] >= 2:
            event_parts = event_parts.iloc[:, :2]
            event_parts.columns = [f"Event_{i}_date", f"Event_{i}_nature"]
            for col in event_parts.columns:
                event_parts[col] = event_parts[col].str.strip()
            cases = pd.concat([cases, event_parts], axis=1)

    return cases


def parse_lse_cases(filepath: str) -> pd.DataFrame:
    """
    Parse LSE/Grantham non-U.S. corporate litigation records.

    Parameters
    ----------
    filepath : str
        Path to the non-US cases CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned case data with parsed event timelines.
    """
    cases = pd.read_csv(filepath, sep=";")
    cases = cases.drop_duplicates()
    cases = parse_event_timeline(cases)
    return cases


def merge_hand_collected_data(
    cases: pd.DataFrame,
    google_data_path: str,
    complaint_data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge hand-collected Google search volume and complaint metadata.

    Parameters
    ----------
    cases : pd.DataFrame
        Base case-level dataset.
    google_data_path : str
        Path to CSV with Google Trends search volume data per case.
    complaint_data_path : str or None
        Path to CSV with complaint factual allegation word counts.

    Returns
    -------
    pd.DataFrame
        Cases enriched with Google search volume and complaint length.
    """
    # Merge Google Trends data
    google = pd.read_csv(google_data_path, sep=";", skiprows=1)
    for col in ["Case Name", "Description"]:
        if col in google.columns:
            google[col] = google[col].str.replace("\n", " ").str.strip()

    merge_cols = [c for c in ["Case Name", "Description"] if c in google.columns]
    cases = cases.merge(
        google[merge_cols + ["Number of Google Pages"]],
        on=merge_cols,
        how="left",
    )

    # Merge complaint metadata
    if complaint_data_path is not None:
        complaints = pd.read_csv(complaint_data_path)
        merge_cols_c = [c for c in ["Title", "Event 0 date"] if c in complaints.columns]
        if merge_cols_c:
            cases = cases.merge(complaints, on=merge_cols_c, how="left")

    return cases


def create_unified_dataset(
    sabin_path: str,
    lse_path: Optional[str] = None,
    google_path: Optional[str] = None,
    complaint_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the unified litigation dataset from all sources.

    Parameters
    ----------
    sabin_path : str
        Path to Sabin Center case bundles.
    lse_path : str or None
        Path to LSE/Grantham non-U.S. cases.
    google_path : str or None
        Path to hand-collected Google Trends data.
    complaint_path : str or None
        Path to complaint metadata.

    Returns
    -------
    pd.DataFrame
        Unified case dataset ready for merging with financial data.
    """
    # Parse U.S. cases
    us_cases = parse_sabin_cases(sabin_path)
    us_cases["geography"] = "USA"

    # Parse non-U.S. cases
    if lse_path is not None:
        non_us = parse_lse_cases(lse_path)
        all_cases = pd.concat([us_cases, non_us], ignore_index=True)
    else:
        all_cases = us_cases

    # Identify first litigation event date
    date_cols = [c for c in all_cases.columns if c.endswith("_date") and "Event" in c]
    if date_cols:
        for col in date_cols:
            all_cases[col] = pd.to_datetime(all_cases[col], errors="coerce")
        all_cases["First Litigation Event"] = all_cases[date_cols].min(axis=1)

    # Merge hand-collected data
    if google_path is not None:
        all_cases = merge_hand_collected_data(all_cases, google_path, complaint_path)

    return all_cases


if __name__ == "__main__":
    paths = get_paths()
    dataset = create_unified_dataset(
        sabin_path=paths["litigation_data"] + "US-Case-Bundles-2022-04-07.csv",
        lse_path=paths["litigation_data"] + "nonUS_cases_16122022.csv",
        google_path=paths["hand_collected"] + "clean_US_cases_with_Google.csv",
        complaint_path=paths["complaints"] + "Complaint_data.csv",
    )
    dataset.to_csv(paths["output"] + "clean_all_cases.csv", index=False)
    print(f"Saved unified dataset: {dataset.shape[0]} cases")
