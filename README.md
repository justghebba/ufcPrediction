# UFCPREDv3 — Modular UFC Fight Prediction Pipeline

## 1. Project Overview

UFCPREDv3 is a machine learning system that predicts the winner of UFC fights. Given two fighters and a fight date, it produces a win probability by combining:

- **Career statistics** (striking volume, accuracy, grappling, knockdowns, submissions)
- **Opponent-adjusted metrics** (how good a fighter's stats look after accounting for strength of schedule)
- **Physical attributes** (height, reach, weight, stance, age)
- **ELO rating** (a dynamic skill rating updated after every fight) – based on the Arpad Elo model, adjusted by a k-factor decided by number of fights (less fight = higher k, more fight = lower k)
- **Rolling form** (performance in the most recent 3 fights) 

The project spans data from **March 1994 to May 2026** (8,701 fights, 2,678 fighters, 4+ million scraped stat rows) and trains three classifiers: XGBoost, Random Forest, and Logistic Regression.

The primary model (XGBoost) achieves:

| Metric | Value |
|---|---|
| Validation Accuracy | ~65.5% |
| AUC-ROC | ~0.703 |
| Lift over ELO baseline | +8.2 pp |
| Best threshold | tuned per validation fold |

---

## 2. How It Works

The pipeline runs in five sequential phases:

### Phase 1 — Data Loading (`data_loader.py`)

Five raw CSV files, scraped from ufcstats.com are loaded:

| File | Contents |
|---|---|
| `ufc_fight_results.csv` | 8,701 fight records (outcome, method, date) |
| `ufc_fight_stats.csv` | 40,998 per-round fighter stats (strikes, takedowns, control time) |
| `ufc_event_details.csv` | 774 event records (name, date, location) |
| `ufc_fighter_tott.csv` | 4,496 fighter physical profiles (height, weight, reach, stance, DOB) |
| `ufc_fighter_details.csv` | Fighter name/id mapping |

Cleaning steps: sentinel strings like `"---"` replaced with NaN, `"X of Y"` and `"M:SS"` strings parsed into numeric values, height strings like `5' 11"` converted to inches.

### Phase 2 — Master DataFrame (`data_loader.py`)

All five tables are joined (left-joined from fights to events, stats, etc.) to produce a single wide DataFrame with one row per fight. The winner is encoded as target = 1 (red corner) / 0 (blue corner). Draws and No Contests are excluded (~1.8% of fights).

### Phase 3 — Feature Engineering (`features.py`, `pairwise.py`)

The core of the system is a **chronological feature engine** that processes fights in date order. For each fight, it computes features using **only data available before that fight** — no look-ahead, making sure there is no leak in predictions.

**Fighter-level features** (computed per-fighter, per-fight from `fighter_histories` dict):

| Category | Features |
|---|---|
| Career totals | num_fights, wins, losses, win_rate, finish_rate, ko_rate, sub_rate |
| Striking | sig_landed_per_fight, sig_acc, kd_per_fight, sig_absorbed_per_fight, striking_defense |
| Grappling | td_landed_per_fight, td_acc, ctrl_sec_per_fight, sub_att_per_fight, td_defense |
| ELO | Rating starting at 1500, K-factor decaying with experience (64/32/16) |
| Opponent quality | avg_opp_elo, avg_opp_win_rate |
| Recency | days_since_last_fight |
| Rolling form (last 3) | recent_sig_landed_3, recent_ctrl_sec_3, recent_finish_rate_3 |
| Physical | height_inches, weight_lbs, reach_inches, stance, age, weight_cut_pct |
| Composite | striking_score, grappling_score (normalized against rolling global averages) |
| Career context | weight_class_changed, num_weight_class_changes, champ_round_experience |
| Loss recency | last_loss_date |

**Matchup/pairwise features** (computed between red and blue fighters):

- `reach_diff`, `height_diff`, `weight_diff`, `age_diff`
- `elo_diff`, `striking_score_diff`, `grappling_score_diff`, `num_fights_diff`, `champ_exp_diff`
- `same_stance`, indicator flags for southpaw
- **Opponent-adjusted scores**: striking_offense * (opponent_striking_defense / global_avg_striking_def), computed separately for striking and grappling — this prices in schedule strength.

The full feature matrix after symmetrization (see Phase 4) contains **105 feature columns**.

### Phase 4 — Modeling (`models.py`, `inference.py`)

#### Symmetrization

Each fight produces **two training rows** (one from red's perspective, one from blue's) with swapped labels (target and 1-target). This doubles the dataset to ~17,094 rows and eliminates any red-corner bias baked into the raw data. Per-corner physical features (e.g., `fighter_r_reach_inches`) are excluded from training because after symmetrization they carry no stable signal — only the `_diff` versions are used.

#### Temporal Split

Dataset is split at a cutoff date (default: June 2024) into:
- **Train**: ~15,078 rows (1994–2024)
- **Validation**: ~2,016 rows (2024–2026)

This respects the chronological nature of fight data — no future data leaks into training.

#### Models

| Model | NaN Handling | Scaling | Hyperparameters |
|---|---|---|---|
| XGBoost | Native (learns splits around NaN) | Not needed | n_estimators=500, max_depth=6, lr=0.03, subsample=0.8, colsample_bytree=0.8, early_stopping=20 |
| Random Forest | Median imputation + missing indicators | Not needed | n_estimators=500, max_depth=10, min_samples_leaf=5 |
| Logistic Regression | Median imputation + missing indicators | StandardScaler | C=1.0, max_iter=1000, solver=lbfgs |

#### Time-Series Cross-Validation (`--no-cv` skips this)

An **expanding-window** CV with 5 splits. Each fold uses an early cutoff that expands toward the present. This tests stability across different eras while still respecting temporal ordering (unlike random k-fold).

#### Evaluation

- Accuracy, AUC-ROC, precision/recall/F1 per class
- Confusion matrix + calibration curve
- **Baseline**: predict the fighter with the higher ELO rating (~54.5% accuracy)
- **Lift**: XGBoost gain over ELO baseline (~+9.2 pp)
- Threshold tuning (scan 0.30–0.70 for max accuracy)
- SHAP analysis for feature importance

#### Inference

`predict_fight(fighter_a, fighter_b, date, ...)` and `predict_advantages(...)` provide the public API. They reconstruct fighter histories up to the given date and run the trained XGBoost model.

### Phase 5 — Diagnostics (`diagnostics.py`)

A **rolling-window meta analysis** trains 32 separate Logistic Regression models on sliding 3-year windows (stepped every 6 months) using only 10 "advantage difference" features. The goal is to track how each feature's predictive weight changes over UFC history. This is **not** integrated into the main pipeline — the overlapping windows produce correlated coefficients and the feature sets differ per window, so the results are diagnostic only.

---

## 3. How to Run

### Prerequisites

- Python 3.10+ (I suggest using python 3.11.15 since it's what I tested everything on and showed no compatibility issue. First versions had problems with SHAP compatibility in Python 3.14+, this should not, but no testing has been done)
- 5 raw CSV files in a `data/` directory (see "Data Setup" below)

### Installation

```bash
# 1. Clone the repo (or copy the UFCPREDv3/ directory)
cd UFCPREDv3

# 2. Install dependencies
pip install -r requirements.txt
```

### Data Setup

The pipeline expects five CSV files at the path defined in `config.py`:

```
data/
  ufc_fight_results.csv
  ufc_fight_stats.csv
  ufc_event_details.csv
  ufc_fighter_tott.csv
  ufc_fighter_details.csv
```

By default, `config.py` resolves the `data/` directory as a sibling of `UFCPREDv3/`. If your data lives elsewhere, update `DATA_DIR` in `config.py`.

### Running the Pipeline

```bash
# Full pipeline (feature engineering + training + evaluation)
python run_pipeline.py

# Skip feature engineering (reuse previously exported master_df.csv)
python run_pipeline.py --skip-features

# Skip time-series cross-validation (faster)
python run_pipeline.py --no-cv

# Skip rolling-window diagnostics
python run_pipeline.py --no-diagnostics

# Combine flags
python run_pipeline.py --no-cv --no-diagnostics
```

**Expected runtime** (mid-range laptop):
- Full pipeline (with features): 5–15 minutes (the chronological engine is the bottleneck)
- After `--skip-features`: ~2–3 minutes

### Notebook Interface

The thin orchestration notebook provides an interactive wrapper:

```bash
jupyter notebook notebooks/01_orchestration.ipynb
```

### Outputs

All artifacts are written to `outputs/`:

```
outputs/
  data/
    ufc_master_features.csv   — exported feature matrix (after Phase 3)
  models/
    (model pickles — future)
  plots/
    xgb_evaluation_dashboard.png   — confusion matrix, ROC, PR, calibration, grouped importance
    model_comparison.png           — ROC/PR for all 3 models + ELO baseline (full + 2005+)
    rolling_window_meta.png        — only if diagnostics enabled
```

### Config

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CUTOFF_DATE` | `2024-06-01` | Temporal train/val split date |
| `CUTOFF_2005` | `2005-01-01` | Post-2005 subset lower bound |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `XGB_PARAMS` | `n_estimators=500, ...` | XGBoost hyperparameters |
| `RF_PARAMS` | `n_estimators=500, ...` | Random Forest hyperparameters |
| `LR_PARAMS` | `C=1.0, ...` | Logistic Regression hyperparameters |

---

## 4. Changelog

### MODEL.ipynb (Original — ~3,744 lines)

The first working version. A monolithic Jupyter notebook containing the entire pipeline in sequential cells.

**Architecture**: Single notebook, no modular structure.

**Features**:
- Basic chronological feature engine with ELO ratings
- Cumulative career stats (num_fights, win_rate, finish_rate, ko_rate, sub_rate)
- Striking/grappling stats per fight (sig_landed, td_landed, ctrl_sec, KD)
- Simple striking defense (1 - sig_absorbed/sig_attempted)
- Physical attributes (height, weight, reach, stance)
- Age at fight time
- Matchup diffs (height, reach, weight, age, ELO)
- Same-stance binary indicator

**Models**: XGBoost, Random Forest, Logistic Regression on raw features.

**Evaluation**: 80/20 temporal split. Accuracy ~60–63%, AUC-ROC ~0.65.
Baseline (ELO): ~53.9%.

**Known issues**:
- Red-corner bias not addressed (red wins ~64% of dataset)
- No opponent-adjusted metrics
- No rolling form features
- No missing-indicator columns (RF/LR got silently dropped rows)
- No threshold tuning
- No SHAP analysis
- No time-series cross-validation
- Global averages computed from full dataset leak future information into feature engineering

---

### UFC Prediction Model v2fix2.ipynb (~5,642 lines)

Major evolution. Still a monolithic notebook but substantially expanded.

**Changes from MODEL.ipynb**:

| Area | Original | v2fix2 |
|---|---|---|
| Data range | Up to early 2024 | Up to May 2026 (+~700 fights) |
| Striking defense | Simple absorbed/attempted ratio | Full defensive rates from opponent stats |
| Grappling defense | Not present | TD defense from opponent stats |
| Rolling form | Not present | Last-3-fight averages (sig_landed, ctrl_sec, finish_rate) |
| Opponent quality | Not present | avg_opp_elo, avg_opp_win_rate, opponent finish rate tracking |
| Opponent-adjusted scores | Not present | adjusted_striking_r/b, adjusted_grappling_r/b |
| Composite scores | Not present | striking_score, grappling_score (weighted components, normalized globally) |
| Champion experience | Not present | champ_round_experience, weight_class_changed tracking |
| Loss recency | Not present | last_loss_date feature |
| Missing indicators | Not present | Added for RF/LR (79 indicator columns) |
| Threshold tuning | Not present | Scans 0.30–0.70 threshold grid |
| SHAP analysis | Not present | Summary + dependence plots |
| Rolling meta analysis | Not present | 32 sliding-window LogReg models |
| Feature count | ~60 features | 82 features + 79 indicators = 161 |
| Validation accuracy | ~62% | ~63–64% |
| AUC-ROC | ~0.65 | ~0.66–0.69 |

**New features introduced**:
- `striking_score` = weighted composite: 30% volume + 20% accuracy + 15% KD + 35% defense
- `grappling_score` = weighted composite: 35% TD volume + 20% TD accuracy + 25% control + 20% sub attempts
- Opponent-adjusted striking/grappling: `offense * (opponent_defense / global_avg_defense)`
- ELO K-factor decay (64 for <10 fights, 32 for 10–30, 16 for 30+)

**Limitations carried over**:
- Still monolithic notebook (no importable modules)
- Global defense averages still computed from full dataset (subtle leakage persists)
- No time-series cross-validation
- No proper expanding-window evaluation
- No command-line interface
- No reproducibility seed in some random operations
- Rolling meta analysis evaluated on training data (highly optimistic)

---

### UFCPREDv3 (Current — 11 modules, ~2,077 lines)

Complete rewrite from monolithic notebook to modular, importable Python package.

**Changes from v2fix2**:

| Area | v2fix2 | v3 |
|---|---|---|
| Structure | Single notebook (5,642 lines) | 11 files across 3 directories (2,077 lines) |
| Imports | Cell-based, inline | `from models import ...` — proper Python package |
| Config | Hardcoded magic values throughout | `config.py` — all paths, seeds, hyperparams in one place |
| Theme | Default matplotlib theme | UFC-branded seaborn dark theme (`theme.py`) |
| CLI | None | `argparse` — `--skip-features`, `--no-cv`, `--no-diagnostics` |
| CV | Single temporal split only | Expanding-window time-series CV (5 folds) |
| ELO baseline | Not shown in comparison table | Printed alongside model results with lift metric |
| Reproducibility | Inconsistent | `RANDOM_SEED=42` passed to every random operation |
| Inference | Embedded in notebook | `inference.py` — `predict_fight()`, `predict_advantages()` API |
| Data export | Not available | `outputs/data/ufc_master_features.csv` exported after feature engineering |
| Plot export | Notebook-dependent | Saved to `outputs/plots/` automatically |
| Bug fixes | `sum_KD` key error in history update | Fixed (history keys normalized to lowercase) |
| Code reuse | Copy-paste between cells | Shared utility functions (e.g., `_init_fighter_history`) |
| Comments | Inline cell comments | Block comments explaining *what* and *why* |

**New or improved features**:
- `run_pipeline.py` — single entry point, all phases orchestrated sequentially
- Expanding-window CV: `time_series_cv()` with `n_splits=5` at evenly spaced percentiles past 20% of data
- `plot_evaluation_dashboard()` — comprehensive single-figure evaluation (confusion matrix, calibration, feature importance, ROC, PR curve, threshold vs accuracy)
- Post-2005 subset comparison as a standard step
- Proper `__init__.py`, `requirements.txt`, thin orchestration notebook
- All paths relative to `os.path.dirname(os.path.abspath(__file__))` — no hardcoded absolute paths
- Diagnostics module separated and flagged as non-production

**Pipeline phases (formalized)**:

```
Phase 1 → data_loader.py   — load + clean raw CSVs
Phase 2 → data_loader.py   — join → master_df with outcomes
Phase 3 → features.py       — chronological engine (slowest: 5–15 min)
       → pairwise.py        — matchup diffs + adjusted scores
Phase 4 → models.py         — symmetrization, temporal split, 3 models, CV, evaluation
       → inference.py       — predict_fight / predict_advantages
Phase 5 → diagnostics.py    — rolling window meta analysis (non-production)
```

**Performance** (full validation, same temporal split as v2fix2, v3.1 numbers):

| Model | Accuracy | AUC-ROC |
|---|---|---|
| XGBoost | **0.655** | **0.703** |
| Random Forest | 0.654 | 0.699 |
| Logistic Regression | 0.644 | 0.693 |
| ELO baseline | 0.545 | — |

XGBoost and Random Forest are essentially tied on this validation period, with XGBoost having a slight edge in AUC-ROC.

**Known limitations** (carried forward):
- Global defense averages for opponent-adjusted features computed over full dataset (pre-symmetrization). This is a minor leak — the globals incorporate future fights. A proper fix would compute rolling global averages within the chronological loop.
- ~64% red-corner win rate in raw data. Symmetrization eliminates this for training, but inference still sees the original corner assignment.
- No hyperparameter search (grid/Random search).
- Rolling meta analysis not integrated (diagnostic only).

---

### UFCPREDv3.1 (Normed + Grouped + Comparison Plots)

Feature normalization, grouped importance, and multi-model comparison plots.

**Changes from v3:**

| Area | v3 | v3.1 |
|---|---|---|
| Corner physicals | 14 per-corner features (`r_age`, `b_height`, etc.) compete with `_diff` versions | Dropped from training (only `_diff` survives). Raw values preserved in exported CSV. Feature count: **105 → ~85**. |
| Feature normalization | Only Logistic Regression got StandardScaler | All 3 models receive StandardScaled inputs — feature importances are scale-independent. |
| Feature importance | Per-feature bars — correlated corner+diff pairs double-count | `_aggregate_importance()` groups conceptually related features (e.g., "Striking Volume" sums 6 individual features). Groups shown in red, singletons in blue. |
| Post-2005 comparison | Separate printed tables | Combined side-by-side table (`Full vs 2005+`) + `plot_model_comparison()` with 6-panel figure (ROC + PR + Accuracy/AUC bars for both subsets). |
| Model comparison ROC | Only XGBoost | Overlaid ROC and PR curves for XGBoost, Random Forest, Logistic Regression, and ELO baseline. |

**New features added**:
- `plot_model_comparison(y_val, probs_dict, elo_prob, y_val_2005, ...)` — 2×3 figure with ROC/PR/bar for full and 2005+ subsets
- `_aggregate_importance(features, importances)` — sums importances by `FEATURE_GROUPS` from `config.py`
- `FEATURE_GROUPS` in `config.py` — 24 group definitions mapping ~85 features into ~30 display units
- Combined comparison table showing Accuracy + AUC-ROC for full and 2005+ subsets side by side

**Performance** (v3.1, with grouped importance displayed):

| Model | Full Acc | Full AUC | 2005+ Acc | 2005+ AUC |
|---|---|---|---|---|
| XGBoost | 0.655 | 0.703 | 0.661 | 0.704 |
| Random Forest | 0.654 | 0.699 | 0.658 | 0.702 |
| Logistic Regression | 0.644 | 0.693 | 0.644 | 0.696 |
| ELO baseline | 0.545 | — | 0.545 | — |

**Top grouped features** (from XGBoost, after aggregation):
1. Composite Striking (r/b scores + diff)
2. ELO (r/b elo + diffs)
3. Striking Volume (landed, absorbed, diff)
4. Opponent Quality (avg opp elo, win rate, finish rate)
5. reach_diff
6. Experience (num_fights, champ_exp, debut)
7. ...

**Why drop corner physicals?** After symmetrization the same fighter appears as both red and blue —
the model can't learn corner-position associations. The only stable signal is the **diff**:
`reach_diff = +2"` always means the same thing regardless of corner. The raw values are still
available in `ufc_master_features.csv` for exploratory analysis.

**Why normalize all models?** Without scaling, features with larger numeric ranges
(e.g., reach in 60-80 inches) appear more important than bounded features (e.g., win_rate 0-1)
even when the latter is more predictive. Normalizing makes every feature's importance
comparable on the same scale. Tree models are theoretically scale-invariant, but
regularization and early stopping can still be affected — the empirical results show
a slight improvement (XGBoost 0.644 → 0.655).
