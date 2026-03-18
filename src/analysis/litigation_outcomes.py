"""
Analyze litigation outcomes: rulings, settlements, and case progression.

Tracks the judicial timeline of each climate litigation case:
    - Filing → motions → rulings → appeals → final disposition
    - Categorizes outcomes: dismissal, settlement, ruling for/against defendant
    - Identifies cases reaching appellate courts or Supreme Court
    - Links case outcomes to the CAR analysis (non-frivolous case identification)

Data: Hand-collected from court records and news sources,
supplemented with Sato et al. (2023) outcome classifications.
"""

import re

import pandas as pd

from src.utils.config import get_paths


def load_litigation_timeline(filepath: str) -> pd.DataFrame:
    """
    Load the hand-collected litigation timeline data.

    Parameters
    ----------
    filepath : str
        Path to the CSV with case timelines (long format:
        one row per case × decision event).

    Returns
    -------
    pd.DataFrame
        Parsed timeline with date columns converted.
    """
    timeline = pd.read_csv(filepath, sep=";")
    timeline["Decision Date"] = pd.to_datetime(timeline["Decision Date"], errors="coerce")
    return timeline


def identify_final_decisions(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the most recent (final) decision for each case.

    Parameters
    ----------
    timeline : pd.DataFrame
        Case timeline in long format.

    Returns
    -------
    pd.DataFrame
        One row per case with the final decision date and nature.
    """
    final = (
        timeline.groupby(["Case Name", "Filing Year"])["Decision Date"]
        .max()
        .reset_index()
        .rename(columns={"Decision Date": "Final Decision Date"})
    )

    timeline = timeline.merge(final, on=["Case Name", "Filing Year"])
    final_decisions = timeline[
        timeline["Decision Date"] == timeline["Final Decision Date"]
    ].copy()

    return final_decisions


def classify_case_outcomes(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each case's current status and outcome.

    Categories:
        - Ongoing: case still in progress
        - Dismissed: case dismissed by court
        - Settlement: parties reached a settlement
        - Ruling for defendant: court ruled in defendant's favor
        - Ruling for claimant: court ruled in claimant's favor

    Parameters
    ----------
    timeline : pd.DataFrame
        Case timeline with final decisions identified.

    Returns
    -------
    pd.DataFrame
        Case-level outcomes with classification columns.
    """
    cases = identify_final_decisions(timeline)

    # Detect ongoing cases
    if "Other Elements Decision" in cases.columns:
        cases["is_ongoing"] = cases["Other Elements Decision"].str.contains(
            "Ongoing", case=False, na=False
        )
    else:
        cases["is_ongoing"] = False

    # Classify decision nature
    if "Decision Nature" in cases.columns:
        cases["outcome_category"] = cases["Decision Nature"].map({
            "Dismissal": "dismissed",
            "Settlement": "settlement",
            "Ruling in favor of defendant": "defendant_wins",
            "Ruling in favor of claimant": "claimant_wins",
        }).fillna("other")

        # Override with ongoing status
        cases.loc[cases["is_ongoing"], "outcome_category"] = "ongoing"

    return cases


def identify_court_progression(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Track case progression through the court system.

    Identifies whether a case reached:
        - Federal court (initial filing)
        - State court (initial filing)
        - Appellate court
        - Supreme Court (or certiorari granted)

    Parameters
    ----------
    timeline : pd.DataFrame
        Full case timeline.

    Returns
    -------
    pd.DataFrame
        Case-level court progression indicators.
    """
    court_cols = []
    for col_name, keywords in {
        "filed_federal": ["federal", "district court"],
        "filed_state": ["state court", "superior court"],
        "reached_appellate": ["appellate", "circuit court", "court of appeals"],
        "reached_supreme": ["supreme court", "certiorari"],
    }.items():
        if "Court" in timeline.columns:
            pattern = "|".join(keywords)
            court_flags = (
                timeline.groupby(["Case Name", "Filing Year"])
                .apply(lambda g: g["Court"].str.contains(pattern, case=False, na=False).any())
                .reset_index(name=col_name)
            )
            court_cols.append(court_flags)

    if court_cols:
        result = court_cols[0]
        for df in court_cols[1:]:
            result = result.merge(df, on=["Case Name", "Filing Year"], how="outer")
        return result

    return timeline[["Case Name", "Filing Year"]].drop_duplicates()


if __name__ == "__main__":
    paths = get_paths()

    timeline = load_litigation_timeline(
        paths["hand_collected"] + "US_cases_with_timeline_long_March24.csv"
    )
    outcomes = classify_case_outcomes(timeline)

    print(f"Case outcomes:")
    if "outcome_category" in outcomes.columns:
        print(outcomes["outcome_category"].value_counts().to_string())
