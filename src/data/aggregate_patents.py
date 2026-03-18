"""
Aggregate patent data from the KPSS database (Kogan, Papanikolaou, Seru, Stoffman 2017).

Processes monthly patent grant files into firm-year aggregates,
with classification of "green" patents based on CPC Y02 codes
(following Dalla Fontana and Nanda, 2023).

Data: KPSS patent-CRSP matching (released May 2023, through end of 2022).
"""

from pathlib import Path

import pandas as pd

from src.utils.config import get_paths


MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def aggregate_monthly_patents(
    monthly_dir: str,
    years: list[int] = list(range(2012, 2023)),
) -> pd.DataFrame:
    """
    Aggregate monthly patent data into firm-year counts.

    Parameters
    ----------
    monthly_dir : str
        Directory containing monthly CSV files (e.g., January_2017.csv).
    years : list of int
        Years to process.

    Returns
    -------
    pd.DataFrame
        Columns: assignee_organization, year, n_patents.
    """
    yearly_aggregates = []

    for year in years:
        year_data = []
        for month in MONTHS:
            filepath = Path(monthly_dir) / f"{month}_{year}.csv"
            if not filepath.exists():
                continue

            monthly = pd.read_csv(filepath)
            monthly_agg = (
                monthly.groupby("assignee_organization")["patent_number"]
                .nunique()
                .reset_index()
            )
            year_data.append(monthly_agg)

        if not year_data:
            continue

        year_combined = pd.concat(year_data, ignore_index=True)
        year_agg = (
            year_combined.groupby("assignee_organization")["patent_number"]
            .sum()
            .reset_index()
            .rename(columns={"patent_number": "n_patents"})
        )
        year_agg["year"] = year
        yearly_aggregates.append(year_agg)

    result = pd.concat(yearly_aggregates, ignore_index=True)
    result = result[result["assignee_organization"] != "None"]

    return result


def classify_green_patents(patents: pd.DataFrame, cpc_col: str = "cpc_code") -> pd.DataFrame:
    """
    Classify patents as green based on CPC Y02 classification.

    Y02 codes cover technologies for mitigation or adaptation against
    climate change (energy, transport, buildings, ICT, etc.).

    Parameters
    ----------
    patents : pd.DataFrame
        Patent-level data with CPC classification codes.
    cpc_col : str
        Column containing CPC codes.

    Returns
    -------
    pd.DataFrame
        Input data with added 'is_green' boolean column.
    """
    patents["is_green"] = patents[cpc_col].str.startswith("Y02", na=False)
    return patents


if __name__ == "__main__":
    paths = get_paths()

    patents = aggregate_monthly_patents(
        monthly_dir=paths["patents"] + "By_Month/",
        years=list(range(2012, 2023)),
    )

    patents.to_csv(paths["patents"] + "firm_year_patent_counts.csv", index=False)
    print(f"Aggregated patents: {patents.shape[0]} firm-year observations")
    print(f"  Firms with >100 patents/year: {(patents['n_patents'] > 100).sum()}")
