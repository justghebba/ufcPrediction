import numpy as np
import pandas as pd


def _clean_stance(s):
    """Normalize stance string to one of: southpaw, orthodox, switch, or NaN."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if "southpaw" in s:
        return "southpaw"
    if "orthodox" in s:
        return "orthodox"
    if "switch" in s:
        return "switch"
    return np.nan


def _adjust_score(offense, defense, global_avg):
    """Inflation factor: boost offensive score when opponent's defense is above average."""
    result = offense.copy()
    mask = offense.notna() & defense.notna() & (global_avg > 0)
    result[mask] = offense[mask] * (defense[mask] / global_avg)
    return result


def add_physical_diffs(df):
    """Add reach, height, and weight differences between red and blue corners.

    These simple physical-advantage features are consistently among the top
    predictors in UFC fight outcome models.
    """
    df["reach_diff"] = df["fighter_r_reach_inches"] - df["fighter_b_reach_inches"]
    df["height_diff"] = df["fighter_r_height_inches"] - df["fighter_b_height_inches"]
    df["weight_diff"] = df["fighter_r_weight_lbs"] - df["fighter_b_weight_lbs"]
    return df


def add_stance_features(df):
    """Add same-stance flag and southpaw advantage indicator.

    Southpaw vs orthodox has a well-documented statistical edge in MMA,
    so we capture both the binary matchup (same_stance) and the directional
    advantage (fighter_r_southpaw).
    """
    df["fighter_r_stance_clean"] = df["fighter_r_stance"].apply(_clean_stance)
    df["fighter_b_stance_clean"] = df["fighter_b_stance"].apply(_clean_stance)

    df["same_stance"] = (
        (df["fighter_r_stance_clean"] == df["fighter_b_stance_clean"])
        & df["fighter_r_stance_clean"].notna()
    ).astype(int)

    sp_r = (df["fighter_r_stance_clean"] == "southpaw").astype(int)
    sp_b = (df["fighter_b_stance_clean"] == "southpaw").astype(int)
    df["fighter_r_southpaw"] = sp_r - sp_b
    df["fighter_b_southpaw"] = sp_b

    df.drop(columns=["fighter_r_stance_clean", "fighter_b_stance_clean"], inplace=True)
    return df


def add_age_features(df, tott_clean):
    """Compute age at fight time for both fighters plus polynomial terms.

    Age is a well-known UFC performance factor (peak ~30-33). Polynomial
    terms help linear models capture the non-linear decline at older ages.
    """
    fighter_dob = tott_clean.set_index("FIGHTER")["DOB"].to_dict()

    df["fighter_r_age"] = df.apply(
        lambda r: (
            (r["DATE"] - fighter_dob.get(r["fighter_r"], pd.NaT)).days / 365.25
            if pd.notna(fighter_dob.get(r["fighter_r"], pd.NaT))
            else np.nan
        ),
        axis=1,
    )
    df["fighter_b_age"] = df.apply(
        lambda r: (
            (r["DATE"] - fighter_dob.get(r["fighter_b"], pd.NaT)).days / 365.25
            if pd.notna(fighter_dob.get(r["fighter_b"], pd.NaT))
            else np.nan
        ),
        axis=1,
    )
    df["age_diff"] = df["fighter_r_age"] - df["fighter_b_age"]

    for prefix in ["fighter_r_", "fighter_b_"]:
        age = df[f"{prefix}age"]
        df[f"{prefix}age_sq"] = np.where(age.notna(), age**2, np.nan)
        df[f"{prefix}age_cubed"] = np.where(age.notna(), age**3, np.nan)

    return df


def add_adjusted_scores(df, global_stats):
    """Adjust striking/grappling scores by opponent's defensive ability.

    A fighter's striking score means more if they achieved it against a
    good defensive striker. This inflates scores against strong opponents
    and deflates them against weak ones.
    """
    global_avg_striking_def = (
        1
        - global_stats["sum_sig_landed"] / global_stats["sum_sig_attempted"]
    )
    global_avg_td_def = (
        1 - global_stats["sum_td_landed"] / global_stats["sum_td_attempted"]
    )

    df["adjusted_striking_r"] = _adjust_score(
        df["fighter_r_striking_score"],
        df["fighter_b_striking_defense"],
        global_avg_striking_def,
    )
    df["adjusted_striking_b"] = _adjust_score(
        df["fighter_b_striking_score"],
        df["fighter_r_striking_defense"],
        global_avg_striking_def,
    )
    df["adjusted_grappling_r"] = _adjust_score(
        df["fighter_r_grappling_score"],
        df["fighter_b_td_defense"],
        global_avg_td_def,
    )
    df["adjusted_grappling_b"] = _adjust_score(
        df["fighter_b_grappling_score"],
        df["fighter_r_td_defense"],
        global_avg_td_def,
    )

    return df


def add_diff_features(df):
    """Add difference features capturing the net advantage for red over blue.

    These capture the directional margin in a single interpretable column.
    """
    df["elo_diff"] = df["fighter_r_elo"] - df["fighter_b_elo"]
    df["elo_finish_diff"] = df["fighter_r_elo_finish"] - df["fighter_b_elo_finish"]
    df["striking_score_diff"] = (
        df["fighter_r_striking_score"] - df["fighter_b_striking_score"]
    )
    df["grappling_score_diff"] = (
        df["fighter_r_grappling_score"] - df["fighter_b_grappling_score"]
    )
    df["num_fights_diff"] = (
        df["fighter_r_num_fights"] - df["fighter_b_num_fights"]
    )
    df["champ_exp_diff"] = (
        df["fighter_r_champ_round_experience"] - df["fighter_b_champ_round_experience"]
    )
    return df


def add_all_pairwise_features(df, tott_clean, global_stats):
    """Apply all pairwise feature builders in sequence.

    This is the standard entry point used by run_pipeline.py.
    """
    df = add_physical_diffs(df)
    df = add_stance_features(df)
    df = add_age_features(df, tott_clean)
    df = add_adjusted_scores(df, global_stats)
    df = add_diff_features(df)

    pairwise_cols = [
        "reach_diff", "height_diff", "weight_diff",
        "same_stance", "fighter_r_southpaw", "fighter_b_southpaw",
        "fighter_r_age", "fighter_b_age", "age_diff",
        "fighter_r_age_sq", "fighter_b_age_sq",
        "fighter_r_age_cubed", "fighter_b_age_cubed",
        "adjusted_striking_r", "adjusted_striking_b",
        "adjusted_grappling_r", "adjusted_grappling_b",
        "elo_diff", "elo_finish_diff",
        "striking_score_diff", "grappling_score_diff",
        "num_fights_diff", "champ_exp_diff",
    ]
    n_added = len([c for c in pairwise_cols if c in df.columns])
    print(f"Pairwise features added: {n_added}")
    return df
