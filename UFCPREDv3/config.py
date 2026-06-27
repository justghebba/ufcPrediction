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

FEATURE_GROUPS = {
    "Striking Volume": [
        "fighter_r_sig_landed_per_fight", "fighter_b_sig_landed_per_fight",
        "fighter_r_sig_absorbed_per_fight", "fighter_b_sig_absorbed_per_fight",
        "fighter_r_sig_diff_per_fight", "fighter_b_sig_diff_per_fight",
    ],
    "Striking Accuracy": ["fighter_r_sig_acc", "fighter_b_sig_acc"],
    "Striking Defense": ["fighter_r_striking_defense", "fighter_b_striking_defense"],
    "Knockdowns": ["fighter_r_kd_per_fight", "fighter_b_kd_per_fight"],
    "Takedowns": [
        "fighter_r_td_landed_per_fight", "fighter_b_td_landed_per_fight",
        "fighter_r_td_diff_per_fight", "fighter_b_td_diff_per_fight",
    ],
    "TD Accuracy": ["fighter_r_td_acc", "fighter_b_td_acc"],
    "TD Defense": ["fighter_r_td_defense", "fighter_b_td_defense"],
    "Control": ["fighter_r_ctrl_sec_per_fight", "fighter_b_ctrl_sec_per_fight"],
    "Submissions": ["fighter_r_sub_att_per_fight", "fighter_b_sub_att_per_fight"],
    "Composite Striking": [
        "fighter_r_striking_score", "fighter_b_striking_score",
        "striking_score_diff",
    ],
    "Composite Grappling": [
        "fighter_r_grappling_score", "fighter_b_grappling_score",
        "grappling_score_diff",
    ],
    "Adjusted Striking": ["adjusted_striking_r", "adjusted_striking_b"],
    "Adjusted Grappling": ["adjusted_grappling_r", "adjusted_grappling_b"],
    "ELO": [
        "fighter_r_elo", "fighter_b_elo", "elo_diff",
        "fighter_r_elo_finish", "fighter_b_elo_finish", "elo_finish_diff",
    ],
    "Experience": [
        "fighter_r_num_fights", "fighter_b_num_fights", "num_fights_diff",
        "fighter_r_champ_round_experience", "fighter_b_champ_round_experience",
        "champ_exp_diff", "fighter_r_is_debut", "fighter_b_is_debut",
    ],
    "Win Rates": [
        "fighter_r_win_rate", "fighter_b_win_rate",
        "fighter_r_finish_rate", "fighter_b_finish_rate",
        "fighter_r_ko_rate", "fighter_b_ko_rate",
        "fighter_r_sub_rate", "fighter_b_sub_rate",
    ],
    "Opponent Quality": [
        "fighter_r_avg_opp_elo", "fighter_b_avg_opp_elo",
        "fighter_r_avg_opp_win_rate", "fighter_b_avg_opp_win_rate",
        "fighter_r_avg_opp_finish_rate", "fighter_b_avg_opp_finish_rate",
        "fighter_r_opponent_elo", "fighter_b_opponent_elo",
    ],
    "Streaks": [
        "fighter_r_win_streak", "fighter_b_win_streak",
        "fighter_r_loss_streak", "fighter_b_loss_streak",
        "fighter_r_streak_diff", "fighter_b_streak_diff",
    ],
    "Recent Form": [
        "fighter_r_recent_sig_landed_3", "fighter_b_recent_sig_landed_3",
        "fighter_r_recent_ctrl_sec_3", "fighter_b_recent_ctrl_sec_3",
        "fighter_r_recent_finish_rate_3", "fighter_b_recent_finish_rate_3",
    ],
    "Recency": [
        "fighter_r_days_since_last_fight", "fighter_b_days_since_last_fight",
        "fighter_r_days_since_last_loss", "fighter_b_days_since_last_loss",
    ],
    "Weight Cut": [
        "fighter_r_weight_cut_pct", "fighter_b_weight_cut_pct",
        "fighter_r_weight_class_changed", "fighter_b_weight_class_changed",
    ],
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
