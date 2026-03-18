# Notebook → Module Mapping

This document tracks how the original Jupyter notebooks were refactored
into the clean module structure.

## Data Pipeline

| Original Notebook | Module | Description |
|---|---|---|
| `Prepare_Sabin_for_Unique.ipynb` | `src/data/prepare_litigation_data.py` | Parse Sabin Center U.S. case bundles |
| `Prepare_LSE_for_Unique.ipynb` | `src/data/prepare_litigation_data.py` | Parse LSE non-U.S. litigation records |
| `Create_Unique_Database.ipynb` | `src/data/prepare_litigation_data.py` | Merge sources into unified case dataset |
| `Merge_Compustat_and_Carbon_Emissions_data.ipynb` | `src/data/merge_compustat.py` | Compustat + Trucost emissions |
| `Code_to_do_matching_by_hand.ipynb` | `src/utils/matching.py` | Manual entity matching helpers |
| `Merge_EPA_and_Compustat.ipynb` | `src/data/merge_epa.py` | EPA FLIGHT facility data |
| `Merge_Disclosures_and_Compustat.ipynb` + `Disclosures.ipynb` | `src/data/merge_disclosures.py` | CDP disclosure data |
| `Aggregate_Monthly_Patent_Data.ipynb` | `src/data/aggregate_patents.py` | KPSS patent aggregation |
| `Get_IO_data.ipynb` | `src/analysis/institutional_investors.py` | 13f holdings + UNPRI + E-tilts |

## Analysis

| Original Notebook | Module | Description |
|---|---|---|
| `Compute_CARs.ipynb` | `src/analysis/compute_cars.py` + `src/utils/event_study.py` | Event study (CARs & CAVs) |
| `Firm_Lvl_Outcomes.ipynb` | `src/analysis/firm_level_outcomes.py` | Defendant firm DiD |
| `Industry_Lvl_Outcomes.ipynb` | `src/analysis/industry_level_outcomes.py` | Peer effects DiD |
| `IO_turnover_analysis.ipynb` | `src/analysis/institutional_investors.py` | IO ownership changes |
| `Outcomes_Litigations.ipynb` | `src/analysis/litigation_outcomes.py` | Case outcome classification |

## Visualization

| Original Notebook | Module | Description |
|---|---|---|
| `Descriptive_Statistics.ipynb` | `src/visualization/descriptive_stats.py` | Summary tables + figures |
| `Number_of_cases_--_Populate_Desc__Stats_Table_--_Graphs_on_Legal_Grounds.ipynb` | `src/visualization/descriptive_stats.py` | Sample composition tables |
| `LSE_Data_Exp.ipynb` | `src/visualization/descriptive_stats.py` | Non-U.S. case exploration |
| `Climate_Laws.ipynb` | *(not refactored — exploratory only)* | U.S. climate legislation reference |

## Shared Utilities

| Functionality | Module | Used by |
|---|---|---|
| Fama-French CAR/CAV engine | `src/utils/event_study.py` | `compute_cars.py` |
| Fuzzy string matching | `src/utils/matching.py` | All merge modules |
| Nearest-neighbor matching | `src/utils/matching.py` | `firm_level_outcomes.py`, `industry_level_outcomes.py` |
| Path configuration | `src/utils/config.py` | All modules |
