# UFC Fight Prediction

Predicts UFC fight winners using historical fight data from ufcstats.com.

## Pipeline

1. **Scraping** — `scrape_ufc_stats-main/` scrapes ufcstats.com into 5 CSV datasets
2. **Feature Engineering** — `MODEL.ipynb` loads the CSVs, joins them chronologically, and engineers 82 features (ELO, striking/grappling scores, physicals, opponent-adjusted metrics, rolling form, etc.)
3. **Modeling** — Trains XGBoost, Random Forest, and Logistic Regression to predict the winner
4. **Meta Analysis** — Rolling window analysis tracking how each advantage type's importance evolved over time

## Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| XGBoost | ~62% | ~0.66 |
| Random Forest | ~62% | ~0.66 |
| Logistic Regression | ~61% | ~0.65 |
| Baseline (ELO only) | ~54% | — |

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Open `UFC Prediction Model v2fix2.ipynb` in Jupyter and run all cells
3. The notebook loads data from `data/`, regenerates `ufc_master_features.csv`, and trains all models
4. `MODEL.ipynb` is an earlier version of the same notebook. Will be dropped in future versions

## Data

Raw fight data is in `data/`:
- `ufc_fight_results.csv` — one row per fight, with outcome and method
- `ufc_fight_stats.csv` — per-round stats for each fighter
- `ufc_fighter_details.csv` — fighter nickname lookup
- `ufc_fighter_tott.csv` — physical attributes (height, weight, reach, stance, DOB)
- `ufc_event_details.csv` — event dates and locations

