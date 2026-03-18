# Climate Litigations and Involuntary Disclosures

**Empirical analysis of how litigations serve as involuntary disclosures, affecting investor behavior, firm value, and corporate policies.**


**Sample:** Climate litigations filed in the U.S. between 2012 and 2019, targeting 118 distinct public companies. Data sourced from the Sabin Center for Climate Change Law at Columbia University.

---

## Methodology

### Event Study: Abnormal Returns and Volumes

Market reactions are measured using **Cumulative Abnormal Returns (CARs)** and **Cumulative Abnormal Volumes (CAVs)** in a [-2, +2] day window around each litigation filing date.

Expected returns are estimated via the **Fama-French 3-factor model** (Developed Markets factors) over a 250-day estimation window with a 50-day gap before the event:

```
R_ft = α_f + β₁·(Mkt-RF) + β₂·SMB + β₃·HML + ε_ft
AR_ft = R_ft - R̂_ft
CAR = Σ AR_ft  for t ∈ [-2, +2]
```

Abnormal volume follows **Chae (2005)**: log turnover minus mean log turnover over the [-40, -11] pre-event baseline.

Cross-sectional determinants of CARs are then analyzed, including:
- **Information content**: length of factual allegations in the complaint (proxy for disclosure substance)
- **Investor attention**: Abnormal Search Volume Index on Google for the defendant's ticker (Da et al., 2011)
- **Pre-existing information environment**: prior voluntary climate risk disclosures from CDP
- **Political context**: Republican vs. Democratic administration at the time of filing

→ Code: [`src/utils/event_study.py`](src/utils/event_study.py), [`src/analysis/compute_cars.py`](src/analysis/compute_cars.py)

### Matching and Difference-in-Differences

Firm-level and peer-level outcomes are estimated using a **staggered difference-in-differences** design with matched control firms.

**Matching procedure** (for each defendant firm-year):
1. Restrict candidates to the same 2-digit GICS sector
2. Require identical physical climate risk disclosure status
3. Select the nearest neighbor on **standardized Euclidean distance** over Scope 1 Emissions and Firm Size (measured in the year before filing)

The DiD specification includes:
- Firm fixed effects
- Sector × year fixed effects
- Country × year fixed effects
- Interaction terms isolating cases where CAR[-2,+2] < 0 (non-frivolous cases)
- Short-term (1 year post-filing) vs. long-term (2+ years) decomposition

→ Code: [`src/utils/matching.py`](src/utils/matching.py), [`src/analysis/firm_level_outcomes.py`](src/analysis/firm_level_outcomes.py)

### Peer Effects

To test for spillover effects on non-defendant firms, the **N = 1, 3, 5 closest industry peers** of each defendant are identified within the same 4-digit GICS industry using the same distance metric. A separate DiD is run for each N, comparing peer firms to more distant firms in the same industry.

→ Code: [`src/analysis/industry_level_outcomes.py`](src/analysis/industry_level_outcomes.py)

### Institutional Ownership

Changes in ownership by "green" institutional investors are tracked quarterly using **13f holdings** (Thomson/Refinitiv S34). Investors are classified following **Pastor, Stambaugh, and Taylor (2023)**:
- UNPRI signatory status (fuzzy-matched to S34 manager names)
- Environmental portfolio tilt (E-tilt)
- Active Share (portfolio rebalancing intensity)

→ Code: [`src/analysis/institutional_investors.py`](src/analysis/institutional_investors.py)

---

## Data Pipeline

The analysis merges 7 data sources into a single firm-year panel through a multi-step pipeline with fuzzy entity resolution at each join:

```
Sabin Center ──┐
LSE Grantham ──┤ prepare_litigation_data.py ──► Unified litigation events
Hand-collected ┘                                          │
                                                          ▼
Compustat NA ──┐                                  create_master_dataset.py
Compustat Gl ──┤ merge_compustat.py ─► Firm-year panel ◄──────────┐
Trucost ───────┘       │                                          │
                       ▼                                          │
EPA FLIGHT ──── merge_epa.py ────► Facility-level emissions       │
                       │                                          │
CDP ──────── merge_disclosures.py ► Disclosure indicators         │
                       │                                          │
KPSS Patents ─ aggregate_patents.py ► Green/non-green counts      │
                       │                                          │
13f Holdings ─┐                                                   │
UNPRI list ───┤ institutional_investors.py ► IO percentages ──────┘
PST E-tilts ──┘
```

**Entity resolution** across datasets is performed via fuzzy string matching (token sort ratio ≥ 85) between Compustat company names and the naming conventions in Trucost, EPA FLIGHT, CDP, and UNPRI signatory lists. Matches below the threshold are flagged for manual review.

→ Code: [`src/data/create_master_dataset.py`](src/data/create_master_dataset.py) orchestrates the full pipeline with intermediate caching.

---

## Repository Structure

```
├── src/
│   ├── data/                    # Data ingestion and merging
│   │   ├── prepare_litigation_data.py    # Parse Sabin Center & LSE litigation records
│   │   ├── merge_compustat.py            # Merge Compustat NA + Global with Trucost emissions
│   │   ├── merge_epa.py                  # Link EPA FLIGHT facility data to firms
│   │   ├── merge_disclosures.py          # Merge CDP disclosure data with Compustat
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
│       └── descriptive_stats.py          # Summary tables, distribution plots
│
├── configs/
│   └── paths.yaml               # Data directory configuration (gitignored)
│
├── notebooks/                   # Original exploratory notebooks (reference only)
├── requirements.txt
├── NOTEBOOK_MAPPING.md          # Maps original notebooks → refactored modules
└── README.md
```

## Quick Start

```bash
git clone https://github.com/[your-username]/climate-litigation-disclosures.git
cd climate-litigation-disclosures
pip install -r requirements.txt

# Configure data paths
cp configs/paths.yaml.example configs/paths.yaml
# Edit paths.yaml to point to your local data directories

# Run the full pipeline
python -m src.data.create_master_dataset       # Build merged dataset
python -m src.analysis.compute_cars            # Compute CARs around filing dates
python -m src.analysis.firm_level_outcomes     # Defendant firm-level DiD
python -m src.analysis.industry_level_outcomes # Peer effects analysis
```

## Requirements

Python 3.9+ with: pandas, numpy, statsmodels, scipy, thefuzz (fuzzy string matching), matplotlib, seaborn, pyyaml, tqdm, openpyxl.

## Citation

```
Eliet-Doillet, A. (2024). "Involuntary Disclosures through Climate Litigations:
Impact on Investors and Corporate Policies." Working Paper.
```

## License

This code is provided for academic and research purposes.
