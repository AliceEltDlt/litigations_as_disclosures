"""
Matching utilities for cross-dataset entity resolution and control group construction.

Includes:
    - Fuzzy string matching for linking firm names across databases
      (Compustat ↔ Trucost, Compustat ↔ EPA, Compustat ↔ UNPRI)
    - Nearest-neighbor matching for constructing DiD control groups
      based on emissions, firm size, and disclosure status
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance
from thefuzz import fuzz, process
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fuzzy string matching
# ---------------------------------------------------------------------------

def fuzzy_match_entities(
    source_names: List[str],
    target_names: List[str],
    score_threshold: int = 85,
    scorer=fuzz.token_sort_ratio,
) -> pd.DataFrame:
    """
    Match entity names between two datasets using fuzzy string matching.

    Used to link firm names across Compustat, Trucost, EPA FLIGHT, and
    UNPRI signatory lists where no common identifier exists.

    Parameters
    ----------
    source_names : list of str
        Names to match (e.g., Compustat company names).
    target_names : list of str
        Candidate names to match against (e.g., Trucost company names).
    score_threshold : int
        Minimum fuzzy match score (0-100) to include in results.
    scorer : callable
        Scoring function from thefuzz (default: token_sort_ratio).

    Returns
    -------
    pd.DataFrame
        Columns: source_name, matched_name, score, above_threshold.
    """
    matches = []

    for name in tqdm(source_names, desc="Fuzzy matching"):
        result = process.extractOne(name, target_names, scorer=scorer)
        if result is not None:
            matched_name, score, _ = result
            matches.append({
                "source_name": name,
                "matched_name": matched_name,
                "score": score,
                "above_threshold": score >= score_threshold,
            })
        else:
            matches.append({
                "source_name": name,
                "matched_name": None,
                "score": 0,
                "above_threshold": False,
            })

    return pd.DataFrame(matches)


def validate_fuzzy_matches(
    matches: pd.DataFrame,
    score_threshold: int = 85,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split fuzzy matches into confident and manual-review sets.

    Parameters
    ----------
    matches : pd.DataFrame
        Output of fuzzy_match_entities.
    score_threshold : int
        Minimum score for automatic acceptance.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (accepted_matches, needs_review)
    """
    accepted = matches[matches["score"] >= score_threshold].copy()
    needs_review = matches[matches["score"] < score_threshold].copy()
    return accepted, needs_review


# ---------------------------------------------------------------------------
# Nearest-neighbor matching for DiD control groups
# ---------------------------------------------------------------------------

def compute_matching_distance(
    treated_row: pd.Series,
    candidates: pd.DataFrame,
    match_vars: List[str],
) -> pd.Series:
    """
    Compute squared Euclidean distance between a treated firm and candidates.

    Parameters
    ----------
    treated_row : pd.Series
        Characteristics of the treated (defendant) firm.
    candidates : pd.DataFrame
        Potential control firms with the same columns.
    match_vars : list of str
        Variable names to use for distance computation.

    Returns
    -------
    pd.Series
        Distance for each candidate, indexed by candidate index.
    """
    treated_vec = treated_row[match_vars].values.astype(float)
    candidate_mat = candidates[match_vars].values.astype(float)

    distances = np.sum((candidate_mat - treated_vec) ** 2, axis=1)
    return pd.Series(distances, index=candidates.index)


def match_treated_to_controls(
    treated: pd.DataFrame,
    pool: pd.DataFrame,
    match_vars: List[str],
    exact_match_vars: Optional[List[str]] = None,
    n_neighbors: int = 1,
    firm_id_col: str = "gvkey",
    year_col: str = "fyear",
) -> pd.DataFrame:
    """
    For each treated firm-year, find the nearest control firm(s).

    Implements the matching procedure from the paper:
    1. Restrict to same 2-digit GICS sector (via exact_match_vars)
    2. Require same physical risk disclosure status
    3. Select nearest neighbor on standardized (emissions, firm_size)

    Parameters
    ----------
    treated : pd.DataFrame
        Treated (defendant) firm-year observations.
    pool : pd.DataFrame
        Pool of potential control firms (never defendants).
    match_vars : list of str
        Continuous variables for distance computation (standardized internally).
    exact_match_vars : list of str or None
        Variables requiring exact matches (e.g., sector, disclosure status).
    n_neighbors : int
        Number of nearest neighbors to select.
    firm_id_col, year_col : str
        Column names for firm identifier and year.

    Returns
    -------
    pd.DataFrame
        Matched pairs with columns: treated_gvkey, control_gvkey, year, distance.
    """
    if exact_match_vars is None:
        exact_match_vars = []

    # Standardize match variables
    combined = pd.concat([treated[match_vars], pool[match_vars]], axis=0)
    means = combined.mean()
    stds = combined.std().replace(0, 1)

    treated_std = treated.copy()
    pool_std = pool.copy()
    for var in match_vars:
        treated_std[var] = (treated[var] - means[var]) / stds[var]
        pool_std[var] = (pool[var] - means[var]) / stds[var]

    matched_pairs = []

    for idx, row in treated_std.iterrows():
        # Filter candidates by exact match criteria
        candidates = pool_std.copy()
        for evar in exact_match_vars:
            candidates = candidates[candidates[evar] == row[evar]]

        if candidates.empty:
            continue

        # Compute distances and select nearest neighbors
        dists = compute_matching_distance(row, candidates, match_vars)
        nearest_idx = dists.nsmallest(n_neighbors).index

        for nn_idx in nearest_idx:
            matched_pairs.append({
                "treated_gvkey": row[firm_id_col],
                "control_gvkey": pool.loc[nn_idx, firm_id_col],
                "year": row[year_col],
                "distance": dists[nn_idx],
                "neighbor_rank": list(nearest_idx).index(nn_idx) + 1,
            })

    return pd.DataFrame(matched_pairs)


def find_industry_peers(
    defendants: pd.DataFrame,
    all_firms: pd.DataFrame,
    match_vars: List[str],
    sector_col: str = "gind_4d",
    n_peers: int = 5,
    firm_id_col: str = "gvkey",
    year_col: str = "fyear",
) -> pd.DataFrame:
    """
    Identify the N closest industry peers to each defendant firm.

    Used for the peer-effects analysis (Section 6 of the paper).
    Peers are firms in the same 4-digit GICS industry that have never
    been defendants, ranked by Euclidean distance on emissions and size.

    Parameters
    ----------
    defendants : pd.DataFrame
        Defendant firm-year observations.
    all_firms : pd.DataFrame
        Universe of firms (defendants will be excluded from peer candidates).
    match_vars : list of str
        Variables for distance computation.
    sector_col : str
        Column for industry classification.
    n_peers : int
        Maximum number of peers to select per defendant.

    Returns
    -------
    pd.DataFrame
        Columns: defendant_gvkey, peer_gvkey, year, peer_rank, distance.
    """
    defendant_ids = set(defendants[firm_id_col].unique())
    non_defendants = all_firms[~all_firms[firm_id_col].isin(defendant_ids)]

    return match_treated_to_controls(
        treated=defendants,
        pool=non_defendants,
        match_vars=match_vars,
        exact_match_vars=[sector_col],
        n_neighbors=n_peers,
        firm_id_col=firm_id_col,
        year_col=year_col,
    )
