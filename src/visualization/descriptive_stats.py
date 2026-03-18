"""
Descriptive statistics and figures for the climate litigation paper.

Generates:
    - Table 1: Sample distribution (filing years, action types, parties)
    - Table B.1: Summary statistics for the full Compustat sample
    - Figure B.1: Number of cases by filing year
    - Figure B.2: Cases by GICS sector
    - Figure B.3: Cases by claimant type
    - Figure B.4: Distribution of CAR[-2,+2]
    - Figure B.6: Legal grounds used by claimants
    - Figure B.7: Distribution of complaint factual allegation lengths
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import get_paths


# Plotting defaults
sns.set(rc={"figure.figsize": (12, 6)}, style="white")
COLORS = {"primary": "#2C3E50", "accent": "#E74C3C", "muted": "#95A5A6"}


def summary_statistics(
    df: pd.DataFrame,
    variables: list[str],
    percentiles: list[float] = [0.05, 0.50, 0.95],
) -> pd.DataFrame:
    """
    Compute summary statistics for a set of variables.

    Parameters
    ----------
    df : pd.DataFrame
        Data to summarize.
    variables : list of str
        Column names to include.
    percentiles : list of float
        Quantiles to report.

    Returns
    -------
    pd.DataFrame
        Summary table with N, Mean, Sd, Min, percentiles, Max.
    """
    stats = []
    for var in variables:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        row = {
            "Variable": var,
            "N": len(s),
            "Mean": s.mean(),
            "Sd": s.std(),
            "Min": s.min(),
        }
        for p in percentiles:
            row[f"p{int(p*100)}"] = s.quantile(p)
        row["Max"] = s.max()
        stats.append(row)

    return pd.DataFrame(stats).set_index("Variable")


def plot_cases_by_year(
    cases: pd.DataFrame,
    year_col: str = "Year First Event",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of climate litigation filings by year (Figure B.1).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = cases[year_col].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color=COLORS["primary"], edgecolor="white")
    ax.set_xlabel("Filing Year")
    ax.set_ylabel("Number of Cases")
    ax.set_title("")
    sns.despine()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cases_by_sector(
    cases: pd.DataFrame,
    sector_col: str = "gind_6d",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of cases by GICS sector (Figure B.2).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = cases[sector_col].value_counts().head(10)
    counts.plot(kind="barh", ax=ax, color=COLORS["primary"], edgecolor="white")
    ax.set_xlabel("Number of Cases")
    ax.set_ylabel("")
    ax.invert_yaxis()
    sns.despine()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_car_distribution(
    cars: pd.DataFrame,
    car_col: str = "CAR",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of CAR[-2,+2] distribution with normal overlay (Figure B.4).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    data = cars[car_col].dropna() * 100  # Convert to percentage

    ax.hist(data, bins=25, density=True, color=COLORS["muted"],
            edgecolor="white", alpha=0.8, label="Observed")

    # Normal overlay
    x = np.linspace(data.min(), data.max(), 100)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, data.mean(), data.std()),
            color=COLORS["accent"], linewidth=2, label="Normal")

    ax.set_xlabel("CAR[-2,+2] (%)")
    ax.set_ylabel("Density")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend()
    sns.despine()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_complaint_length_distribution(
    cases: pd.DataFrame,
    length_col: str = "n_words_factual_allegations",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of factual allegation word counts (Figure B.7).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    data = cases[length_col].dropna()

    ax.hist(data, bins=20, color=COLORS["primary"], edgecolor="white")
    ax.set_xlabel("Number of Words in Allegations Part of Complaint")
    ax.set_ylabel("Number of Cases")

    # Add summary stats annotation
    stats_text = f"Minimum: {data.min():.0f}\nMedian: {data.median():.0f}\nMaximum: {data.max():.0f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    sns.despine()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def generate_sample_table(cases: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Generate Table 1 panels: filing years, action types, party types.

    Returns
    -------
    dict
        Keys: 'panel_a' (years), 'panel_b' (action types),
        'panel_c' (lawsuit types), 'panel_d' (parties).
    """
    panels = {}

    # Panel A: Filing years
    if "Year First Event" in cases.columns:
        sample = cases[(cases["Year First Event"] >= 2012) & (cases["Year First Event"] < 2020)]
        newer = cases[cases["Year First Event"] >= 2020]
        older = cases[cases["Year First Event"] < 2012]

        panels["panel_a"] = pd.DataFrame({
            "Period": ["2012-2019", "2020+", "Pre-2012"],
            "N Cases": [sample["Title"].nunique() if "Title" in sample.columns else len(sample),
                        newer["Title"].nunique() if "Title" in newer.columns else len(newer),
                        older["Title"].nunique() if "Title" in older.columns else len(older)],
        })

    # Panel B: Type of action
    if "Type of Action" in cases.columns:
        panels["panel_b"] = cases["Type of Action"].value_counts().reset_index()

    # Panel D: Party type
    if "Side A.0 nature" in cases.columns:
        panels["panel_d"] = cases["Side A.0 nature"].value_counts().reset_index()

    return panels


if __name__ == "__main__":
    paths = get_paths()

    cases = pd.read_csv(paths["output"] + "clean_all_cases.csv")
    cases["First Litigation Event"] = pd.to_datetime(cases["First Litigation Event"])
    cases["Year First Event"] = cases["First Litigation Event"].dt.year

    # Generate tables
    tables = generate_sample_table(cases)
    for name, table in tables.items():
        print(f"\n{name}:")
        print(table.to_string(index=False))
