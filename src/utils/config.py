"""
Centralized configuration for data paths and analysis parameters.

Update PATHS to match your local directory structure, or load from configs/paths.yaml.
"""

import os
from pathlib import Path
from typing import Dict

import yaml


# ---------------------------------------------------------------------------
# Default directory layout (override via configs/paths.yaml)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATHS: Dict[str, str] = {
    "litigation_data": "~/Google Drive/Mon Drive/Litigations/Litigation data/",
    "hand_collected": "~/Google Drive/Mon Drive/Litigations/Hand collected data/",
    "compustat": "~/Google Drive/Mon Drive/Litigations/Compustat data/",
    "trucost": "~/Google Drive/Mon Drive/Litigations/Trucost data/",
    "epa": "~/Google Drive/Mon Drive/Litigations/EPA data/",
    "cdp": "~/Dropbox/CDP Data/",
    "patents": "~/Google Drive/Mon Drive/Litigations/Patent data/",
    "institutional_investors": "~/Google Drive/Mon Drive/Litigations/Institutional Investors/",
    "complaints": "~/Google Drive/Mon Drive/Litigations/Complaints/",
    "other": "~/Google Drive/Mon Drive/Litigations/Other data/",
    "cars": "~/Google Drive/Mon Drive/Litigations/CARs/",
    "output": "~/Google Drive/Mon Drive/Litigations/",
    "figures": str(PROJECT_ROOT / "output" / "figures"),
}


def load_paths(config_path: str | None = None) -> Dict[str, str]:
    """Load paths from a YAML config file, falling back to defaults."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "paths.yaml")

    if os.path.exists(config_path):
        with open(config_path) as f:
            user_paths = yaml.safe_load(f)
        if user_paths:
            PATHS.update(user_paths)

    # Expand ~ in all paths
    return {k: os.path.expanduser(v) for k, v in PATHS.items()}


def get_paths() -> Dict[str, str]:
    """Return resolved data paths."""
    return load_paths()


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

# Event study parameters
EVENT_STUDY_PARAMS = {
    "event_window_pre": 2,       # Days before event
    "event_window_post": 2,      # Days after event
    "estimation_gap": 50,        # Gap between estimation and event window (business days)
    "estimation_length": 250,    # Length of estimation window (business days)
}

# Fama-French factors
FF_FACTORS = ["Mkt-RF", "SMB", "HML"]

# Sample period
SAMPLE_START_YEAR = 2012
SAMPLE_END_YEAR = 2019

# Matching parameters
MATCHING_VARIABLES = ["scope1_emissions", "firm_size"]
FUZZY_MATCH_THRESHOLD = 85  # Minimum score for fuzzy entity matching

# Sectors
EU_COUNTRIES = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
]
