import numpy as np
import pandas as pd
from features import compute_fighter_features


def _cs(s):
    """Normalize stance string for comparison."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    return "southpaw" if "southpaw" in s else ("orthodox" if "orthodox" in s else ("switch" if "switch" in s else np.nan))


def predict_fight(fighter_a, fighter_b, fight_date,
                  histories, tott_df, xgb_model,
                  def_striking, def_td,
                  feature_cols, weight_class=None):
    """Return probability that fighter_a defeats fighter_b.

    Builds a complete feature vector for a hypothetical matchup using the
    same feature engineering pipeline that produced the training data.
    """
    elo_a = histories.get(fighter_a, {}).get("elo", 1500)
    elo_b = histories.get(fighter_b, {}).get("elo", 1500)

    feats_a = compute_fighter_features(fighter_a, fight_date, histories, tott_df,
                                       opponent_elo=elo_b, weight_class=weight_class)
    feats_b = compute_fighter_features(fighter_b, fight_date, histories, tott_df,
                                       opponent_elo=elo_a, weight_class=weight_class)

    row = {}
    for k, v in feats_a.items():
        row[f"fighter_r_{k}"] = v
    for k, v in feats_b.items():
        row[f"fighter_b_{k}"] = v

    for attr, dname in [("reach_inches", "reach_diff"),
                        ("height_inches", "height_diff"),
                        ("weight_lbs", "weight_diff")]:
        rv = row.get(f"fighter_r_{attr}")
        bv = row.get(f"fighter_b_{attr}")
        row[dname] = rv - bv if pd.notna(rv) and pd.notna(bv) else np.nan

    sr, sb = _cs(row.get("fighter_r_stance")), _cs(row.get("fighter_b_stance"))
    row["same_stance"] = 1 if (pd.notna(sr) and pd.notna(sb) and sr == sb) else 0
    row["fighter_r_southpaw"] = (1 if sr == "southpaw" else 0) - (1 if sb == "southpaw" else 0)
    row["fighter_b_southpaw"] = 1 if sb == "southpaw" else 0

    dob_map = tott_df.set_index("FIGHTER")["DOB"].to_dict()
    for prefix, fname in [("fighter_r_", fighter_a), ("fighter_b_", fighter_b)]:
        dob = dob_map.get(fname)
        row[f"{prefix}age"] = (fight_date - dob).days / 365.25 if pd.notna(dob) else np.nan
    row["age_diff"] = row["fighter_r_age"] - row["fighter_b_age"]

    for prefix in ["fighter_r_", "fighter_b_"]:
        age = row.get(f"{prefix}age")
        row[f"{prefix}age_sq"] = age ** 2 if pd.notna(age) else np.nan
        row[f"{prefix}age_cubed"] = age ** 3 if pd.notna(age) else np.nan

    def _adj(s_col, d_col, gavg):
        sv, dv = row.get(s_col), row.get(d_col)
        if pd.notna(sv) and pd.notna(dv) and gavg > 0:
            return sv * (dv / gavg)
        return sv

    row["adjusted_striking_r"] = _adj("fighter_r_striking_score", "fighter_b_striking_defense", def_striking)
    row["adjusted_striking_b"] = _adj("fighter_b_striking_score", "fighter_r_striking_defense", def_striking)
    row["adjusted_grappling_r"] = _adj("fighter_r_grappling_score", "fighter_b_td_defense", def_td)
    row["adjusted_grappling_b"] = _adj("fighter_b_grappling_score", "fighter_r_td_defense", def_td)

    row["elo_diff"] = row["fighter_r_elo"] - row["fighter_b_elo"]
    row["elo_finish_diff"] = row["fighter_r_elo_finish"] - row["fighter_b_elo_finish"]
    row["striking_score_diff"] = row["fighter_r_striking_score"] - row["fighter_b_striking_score"]
    row["grappling_score_diff"] = row["fighter_r_grappling_score"] - row["fighter_b_grappling_score"]
    row["num_fights_diff"] = row["fighter_r_num_fights"] - row["fighter_b_num_fights"]
    row["champ_exp_diff"] = row["fighter_r_champ_round_experience"] - row["fighter_b_champ_round_experience"]

    X_pred = pd.DataFrame([row])[feature_cols]
    prob = xgb_model.predict_proba(X_pred)[0, 1]
    return prob


def predict_advantages(fighter_a, fighter_b, fight_date,
                       histories, tott_df,
                       window_models, window_scalers, window_imputers,
                       window_midpoints, window_features,
                       advantage_features,
                       weight_class=None):
    """Decompose a win probability into per-advantage contributions.

    Uses the closest rolling-window Logistic Regression model to attribute
    the prediction to ELO, striking, grappling, experience, reach, height,
    weight, stance, and age.
    """
    fight_date_ts = pd.Timestamp(fight_date)
    idx = np.argmin([abs(m - fight_date_ts) for m in window_midpoints])
    model = window_models[idx]
    scaler = window_scalers[idx]
    imp = window_imputers[idx]
    this_feats = window_features[idx]

    elo_a = histories.get(fighter_a, {}).get("elo", 1500)
    elo_b = histories.get(fighter_b, {}).get("elo", 1500)

    fa = compute_fighter_features(fighter_a, fight_date, histories, tott_df,
                                  opponent_elo=elo_b, weight_class=weight_class)
    fb = compute_fighter_features(fighter_b, fight_date, histories, tott_df,
                                  opponent_elo=elo_a, weight_class=weight_class)

    row = {
        "elo_diff": fa["elo"] - fb["elo"],
        "striking_score_diff": (fa.get("striking_score") or 0) - (fb.get("striking_score") or 0),
        "grappling_score_diff": (fa.get("grappling_score") or 0) - (fb.get("grappling_score") or 0),
        "num_fights_diff": fa["num_fights"] - fb["num_fights"],
        "champ_exp_diff": (fa.get("champ_round_experience") or 0) - (fb.get("champ_round_experience") or 0),
        "reach_diff": (fa.get("reach_inches") or np.nan) - (fb.get("reach_inches") or np.nan),
        "height_diff": (fa.get("height_inches") or np.nan) - (fb.get("height_inches") or np.nan),
        "weight_diff": (fa.get("weight_lbs") or np.nan) - (fb.get("weight_lbs") or np.nan),
    }

    def _sp(s):
        return 1 if pd.notna(s) and "southpaw" in str(s).lower() else 0
    row["fighter_r_southpaw"] = _sp(fa.get("stance")) - _sp(fb.get("stance"))

    dob_map = tott_df.set_index("FIGHTER")["DOB"].to_dict()
    dob_a, dob_b = dob_map.get(fighter_a), dob_map.get(fighter_b)
    age_a = (fight_date_ts - dob_a).days / 365.25 if pd.notna(dob_a) else np.nan
    age_b = (fight_date_ts - dob_b).days / 365.25 if pd.notna(dob_b) else np.nan
    row["age_diff"] = age_a - age_b

    X_row = pd.DataFrame([row]).reindex(columns=this_feats, fill_value=np.nan)
    X_imp = imp.transform(X_row)
    X_scl = scaler.transform(X_imp)

    contribs = {}
    for i, col in enumerate(this_feats):
        contribs[col] = model.coef_[0][i] * X_scl[0][i]

    prob = model.predict_proba(X_scl)[0, 1]
    return contribs, prob, window_midpoints[idx], fa, fb
