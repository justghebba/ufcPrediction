import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "data"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RANDOM_SEED = 42

CUTOFF_DATE = "2024-06-06"
CUTOFF_2005 = "2005-01-01"

CLASS_LIMITS = {
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Women's Strawweight": 115,
    "Women's Flyweight": 125,
    "Women's Bantamweight": 135,
    "Women's Featherweight": 145,
}

ADVANTAGE_FEATURE_COLS = [
    "elo_diff",
    "striking_score_diff",
    "grappling_score_diff",
    "num_fights_diff",
    "champ_exp_diff",
    "reach_diff",
    "height_diff",
    "age_diff",
    "fighter_r_southpaw",
    "weight_diff",
]

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "early_stopping_rounds": 20,
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}
