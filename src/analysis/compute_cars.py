"""
Compute Cumulative Abnormal Returns and Volumes around climate litigation filings.

Implements the event study methodology from Section 4.1 of the paper:
    - Fama-French 3-factor model (Developed Markets factors)
    - Event window: [-2, +2] business days around filing date
    - Estimation window: [-302, -52] business days (250 days, 50-day gap)
    - Standard errors clustered at the lawsuit level

Outputs:
    - Case-level CARs and CAVs with cross-sectional determinants
    - Inputs for Tables 3 and 4 (determinants regressions)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import get_paths, EVENT_STUDY_PARAMS
from src.utils.event_study import compute_car, compute_cav


def load_fama_french_factors(filepath: str) -> pd.DataFrame:
    """
    Load daily Fama-French 3-factor data (Developed Markets).

    Parameters
    ----------
    filepath : str
        Path to the CSV from Kenneth French's data library.

    Returns
    -------
    pd.DataFrame
        Daily factor returns indexed by date.
    """
    ff = pd.read_csv(filepath, skiprows=1, header=1)

    # Parse date column (format varies)
    date_col = ff.columns[0]
    ff[date_col] = pd.to_datetime(ff[date_col].astype(str), format="%Y%m%d", errors="coerce")
    ff = ff.dropna(subset=[date_col])
    ff = ff.set_index(date_col)
    ff.index.name = "datadate"

    # Scale from percentage to decimal if needed
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        if col in ff.columns and ff[col].abs().max() > 1:
            ff[col] = ff[col] / 100

    return ff


def load_daily_returns(filepath: str) -> pd.DataFrame:
    """
    Load daily stock returns from Compustat Securities Daily.

    Parameters
    ----------
    filepath : str
        Path to the daily returns CSV.

    Returns
    -------
    pd.DataFrame
        Returns pivoted to (date × gvkey) format.
    """
    ret = pd.read_csv(filepath)
    ret["datadate"] = pd.to_datetime(ret["datadate"])

    # Pivot to wide format: dates as index, gvkeys as columns
    ret_wide = ret.pivot_table(
        index="datadate",
        columns="gvkey",
        values="ret",
    )

    return ret_wide


def load_litigation_events(filepath: str, sample_start: int = 2012, sample_end: int = 2019) -> pd.DataFrame:
    """
    Load litigation events and filter to the analysis sample.

    Parameters
    ----------
    filepath : str
        Path to the unified litigation dataset.
    sample_start, sample_end : int
        Filing year range for the analysis sample.

    Returns
    -------
    pd.DataFrame
        Filtered litigation events with parsed dates.
    """
    cases = pd.read_csv(filepath, sep=";")
    cases["First Litigation Event"] = pd.to_datetime(cases["First Litigation Event"])
    cases["Year First Event"] = cases["First Litigation Event"].dt.year

    # Filter to U.S. cases in sample period
    cases = cases[cases["Geography ISO"] == "USA"]
    cases = cases[
        (cases["Year First Event"] >= sample_start)
        & (cases["Year First Event"] <= sample_end)
    ]

    # Keep only cases with identified defendant gvkeys
    cases = cases[cases["gvkey_side_B"].notna()]

    return cases


def run_event_study(paths: dict | None = None) -> pd.DataFrame:
    """
    Execute the full event study pipeline.

    Returns
    -------
    pd.DataFrame
        Case-firm level CARs with metadata for cross-sectional analysis.
    """
    if paths is None:
        paths = get_paths()

    # Load data
    ff_factors = load_fama_french_factors(paths["other"] + "Developed_3_Factors_Daily.csv")
    returns = load_daily_returns(paths["compustat"] + "daily_returns.csv")
    events = load_litigation_events(paths["output"] + "clean_all_cases.csv")

    print(f"Computing CARs for {len(events)} litigation events...")

    # Compute CARs
    cars = compute_car(
        litigation_events=events,
        daily_returns=returns,
        ff_factors=ff_factors,
        **EVENT_STUDY_PARAMS,
    )

    # Merge back case metadata for cross-sectional analysis
    cars = cars.merge(
        events[[
            "gvkey_side_B", "First Litigation Event", "Title",
            "Side A.0 nature", "Number of Google Pages",
        ]].rename(columns={"gvkey_side_B": "gvkey", "First Litigation Event": "event_date"}),
        on=["gvkey", "event_date"],
        how="left",
    )

    print(f"Computed CARs for {len(cars)} firm-event pairs")
    print(f"  Mean CAR[-2,+2]: {cars['CAR'].mean():.4f}")
    print(f"  Median CAR[-2,+2]: {cars['CAR'].median():.4f}")

    return cars


if __name__ == "__main__":
    paths = get_paths()
    cars = run_event_study(paths)
    cars.to_csv(paths["cars"] + "CARs.csv", index=False)
    print(f"Saved to {paths['cars']}CARs.csv")
