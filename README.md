# Climate Litigations and Involuntary Disclosures

**Empirical analysis of how climate litigations serve as involuntary disclosures, affecting investor behavior, firm value, and corporate environmental policies.**

This repository contains the data pipeline and analysis code for the paper *"Involuntary Disclosures through Climate Litigations: Impact on Investors and Corporate Policies"*.

---

## Overview

Climate litigations filed against publicly listed companies can reveal new information about firms' climate risk exposure. This project builds a comprehensive dataset linking litigation events to financial market reactions, institutional ownership changes, and corporate emissions outcomes.

**Key methods:**
- Event study analysis (Cumulative Abnormal Returns & Volumes) using Fama-French 3-factor model
- Staggered difference-in-differences with matched control firms
- Cross-sectional regressions on determinants of market reactions
- Panel regressions with firm, sector×year, and country×year fixed effects
- Nearest-neighbor matching on emissions, size, and disclosure status

**Data sources:** Sabin Center for Climate Change Law, Compustat (NA + Global), Trucost, CDP, EPA FLIGHT, KPSS Patents, Thomson/Refinitiv 13f Holdings, Google Trends (ASVI).

---

## Repository Structure

```
├── src/
│   ├── data/                    # Data ingestion and merging
│   │   ├── prepare_litigation_data.py    # Parse Sabin Center & LSE litigation records
│   │   ├── merge_compustat.py            # Merge Compustat NA + Global with emissions
│   │   ├── merge_epa.py                  # Link EPA FLIGHT facility data to firms
│   │   ├── merge_disclosures.py          # Merge CDP disclosure data with Compustat
│   │   ├── build_institutional_ownership.py  # Process 13f holdings, UNPRI, E-tilts
│   │   ├── aggregate_patents.py          # Aggregate KPSS patent data (green vs. non-green)
│   │   └── create_master_dataset.py      # Combine all sources into analysis-ready panels
│   │
│   ├── analysis/                # Core empirical analysis
│   │   ├── compute_cars.py               # Event study: CARs and CAVs around filings
│   │   ├── firm_level_outcomes.py        # DiD on defendant firm emissions & disclosures
│   │   ├── industry_level_outcomes.py    # Peer effects on neighbor firms
│   │   ├── institutional_investors.py    # IO ownership changes post-litigation
│   │   └── litigation_outcomes.py        # Case outcome analysis (rulings, settlements)
│   │
│   ├── utils/                   # Shared utilities
│   │   ├── event_study.py                # Fama-French CAR/CAV computation engine
│   │   ├── matching.py                   # Nearest-neighbor & fuzzy matching utilities
│   │   └── config.py                     # Centralized paths and parameters
│   │
│   └── visualization/           # Figures and descriptive statistics
│       └── descriptive_stats.py          # Summary tables, distribution plots, maps
│
├── configs/
│   └── paths.yaml               # Data directory configuration
│
├── notebooks/                   # Original exploratory notebooks (reference only)
│
├── data/
│   ├── raw/                     # Raw input data (not tracked in git)
│   ├── processed/               # Intermediate cleaned datasets
│   └── output/                  # Final analysis-ready datasets
│
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/[your-username]/climate-litigation-disclosures.git
cd climate-litigation-disclosures
pip install -r requirements.txt

# Configure data paths
cp configs/paths.yaml.example configs/paths.yaml
# Edit paths.yaml to point to your local data directories

# Run the full pipeline
python -m src.data.create_master_dataset      # Build merged dataset
python -m src.analysis.compute_cars            # Compute CARs around filing dates
python -m src.analysis.firm_level_outcomes     # Defendant firm-level DiD
python -m src.analysis.industry_level_outcomes # Peer effects analysis
```

## Requirements

- Python 3.9+
- pandas, numpy, statsmodels, scipy
- thefuzz (fuzzy string matching for cross-dataset entity resolution)
- matplotlib, seaborn

## Citation

If you use this code or data pipeline, please cite:

```
Eliet-Doillet, A. (2024). "Involuntary Disclosures through Climate Litigations:
Impact on Investors and Corporate Policies." Working Paper.
```

## License

This code is provided for academic and research purposes.
