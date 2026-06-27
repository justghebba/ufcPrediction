import pandas as pd
import numpy as np
from config import CLASS_LIMITS


# ---------------------------------------------------------------------------
# 1.  Parse raw per-round stats into per-fight aggregates
# ---------------------------------------------------------------------------

def _parse_x_of_y(series):
    """Split 'X of Y' strings into (landed, attempted) numeric columns."""
    split = series.str.split(" of ", expand=True)
    return pd.to_numeric(split[0], errors="coerce"), pd.to_numeric(
        split[1], errors="coerce"
    )


def _parse_ctrl(series):
    """Convert 'M:SS' control-time strings into total seconds."""
    split = series.str.split(":", expand=True)
    minutes = pd.to_numeric(split[0], errors="coerce").fillna(0)
    seconds = pd.to_numeric(split[1], errors="coerce").fillna(0)
    return minutes * 60 + seconds


def parse_stats_table(stats):
    """Unpack 'X of Y' stat columns and sum per (event, bout, fighter).

    Returns fighter_fights with one row per fighter per bout, containing
    landed/attempted counts, control time, KD, sub attempts, reversals,
    and derived accuracy columns.
    """
    stats = stats.copy()

    col_map = {
        "SIG.STR.": "sig",
        "TOTAL STR.": "total_str",
        "HEAD": "head",
        "BODY": "body",
        "LEG": "leg",
        "DISTANCE": "distance",
        "CLINCH": "clinch",
        "GROUND": "ground",
        "TD": "td",
    }

    for raw_col, prefix in col_map.items():
        landed, attempted = _parse_x_of_y(stats[raw_col])
        stats[f"{prefix}_landed"] = landed
        stats[f"{prefix}_attempted"] = attempted

    stats["ctrl_sec"] = _parse_ctrl(stats["CTRL"])
    for col in ["KD", "SUB.ATT", "REV."]:
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)

    sum_cols = [f"{p}_{s}" for p in col_map.values() for s in ("landed", "attempted")]
    sum_cols += ["ctrl_sec", "KD", "SUB.ATT", "REV."]

    fighter_fights = (
        stats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)[sum_cols]
        .sum()
    )

    fighter_fights["sig_acc"] = np.where(
        fighter_fights["sig_attempted"] > 0,
        fighter_fights["sig_landed"] / fighter_fights["sig_attempted"],
        np.nan,
    )
    fighter_fights["td_acc"] = np.where(
        fighter_fights["td_attempted"] > 0,
        fighter_fights["td_landed"] / fighter_fights["td_attempted"],
        np.nan,
    )
    return fighter_fights


# ---------------------------------------------------------------------------
# 2.  Parse fighter physicals (height, weight, reach, DOB)
# ---------------------------------------------------------------------------

def _parse_height(h):
    """Convert height string like "5' 11\\"" or "5'11\\"" to total inches."""
    if pd.isna(h):
        return np.nan
    h = str(h).replace('"', "").strip()
    if h == "--":
        return np.nan
    for sep in ("' ", "'"):
        parts = h.split(sep)
        if len(parts) == 2:
            try:
                return int(parts[0]) * 12 + int(parts[1].strip())
            except (ValueError, TypeError):
                continue
    return np.nan


def parse_physicals(tott):
    """Parse height, weight, reach, DOB from the fighter-details table.

    Returns a cleaned DataFrame where physical measurements are numeric and
    ready to merge into the chronological fight-fighter table.
    """
    tott = tott.copy()
    tott["height_inches"] = tott["HEIGHT"].apply(_parse_height)
    tott["weight_lbs"] = pd.to_numeric(
        tott["WEIGHT"].str.replace(" lbs.", "", regex=False), errors="coerce"
    )
    tott["reach_inches"] = pd.to_numeric(
        tott["REACH"].astype(str).str.strip('"').str.strip(), errors="coerce"
    )
    tott["DOB"] = pd.to_datetime(tott["DOB"], format="mixed", errors="coerce")
    return tott


# ---------------------------------------------------------------------------
# 3.  Parse fight metadata (method categories, weight classes)
# ---------------------------------------------------------------------------

def _classify_method(m):
    if pd.isna(m):
        return np.nan
    ms = str(m).lower()
    if "ko" in ms or "tko" in ms:
        return "KO/TKO"
    if "submission" in ms:
        return "SUB"
    if "decision" in ms:
        return "DEC"
    if "dq" in ms:
        return "DQ"
    return "OTHER"


def parse_fight_metadata(fights):
    """Derive method category, clean weight class, and max-rounds info."""
    fights = fights.copy()
    fights["method_cat"] = fights["METHOD"].apply(_classify_method)
    fights["weight_class"] = fights["WEIGHTCLASS"].str.replace(" Bout", "", regex=False)
    fights["max_rounds"] = (
        fights["TIME FORMAT"].str.extract(r"(\d+) Rnd", expand=False)
        .astype(float)
        .fillna(3)
    )
    fights["champ_round_possible"] = fights["max_rounds"] >= 5
    return fights


# ---------------------------------------------------------------------------
# 4.  Build chronological fight-fighter table
# ---------------------------------------------------------------------------

def build_chrono_table(master_df, fighter_fights, tott_clean, fights_meta):
    """Create one row per (fight, fighter) with stats, physicals, and metadata.

    Rows are sorted chronologically and indexed by fight_idx so the feature
    engine can iterate fight by fight, updating histories past-only.
    """
    chrono_rows = []
    for _, row in master_df.iterrows():
        for role, fighter in [("r", row["fighter_r"]), ("b", row["fighter_b"])]:
            chrono_rows.append(
                {
                    "fight_id": row["fight_id"],
                    "date": row["DATE"],
                    "event": row["EVENT"],
                    "bout": row["BOUT"],
                    "fighter": fighter,
                    "role": role,
                    "opponent": (
                        row["fighter_b"] if role == "r" else row["fighter_r"]
                    ),
                    "winner_label": row["winner"],
                }
            )

    chrono = pd.DataFrame(chrono_rows)

    ff = fighter_fights.rename(
        columns={"EVENT": "event", "BOUT": "bout", "FIGHTER": "fighter"}
    )
    tc = tott_clean.rename(columns={"FIGHTER": "fighter"})
    fc = fights_meta.rename(columns={"EVENT": "event", "BOUT": "bout"})

    chrono = chrono.merge(ff, on=["event", "bout", "fighter"], how="left")
    chrono = chrono.merge(
        tc[["fighter", "height_inches", "weight_lbs", "reach_inches", "STANCE", "DOB"]],
        on="fighter",
        how="left",
    )
    chrono = chrono.merge(
        fc[["event", "bout", "method_cat", "weight_class", "champ_round_possible"]],
        on=["event", "bout"],
        how="left",
    )

    def get_outcome(row):
        w = row["winner_label"]
        if w == "Draw":
            return "draw"
        if w == "No Contest":
            return "no_contest"
        return "win" if w == row["fighter"] else "loss"

    chrono["outcome"] = chrono.apply(get_outcome, axis=1)
    chrono = chrono.sort_values("date").reset_index(drop=True)

    fight_order = (
        chrono[["fight_id", "date"]]
        .drop_duplicates("fight_id")
        .sort_values("date")
        .reset_index(drop=True)
    )
    fight_order["fight_idx"] = range(len(fight_order))
    chrono = chrono.merge(fight_order, on=["fight_id", "date"], how="left")
    chrono = chrono.sort_values("fight_idx").reset_index(drop=True)

    return chrono


# ---------------------------------------------------------------------------
# 5.  Feature Engine — chronological loop
# ---------------------------------------------------------------------------

def _init_fighter_history(name):
    """Create a fresh history accumulator for a fighter.

    Tracks cumulative stats, opponent quality, streaks, ELO, and recent form.
    """
    return {
        "fighter": name,
        "num_fights": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "ko_wins": 0,
        "sub_wins": 0,
        "dec_wins": 0,
        "sum_sig_landed": 0,
        "sum_sig_attempted": 0,
        "sum_td_landed": 0,
        "sum_td_attempted": 0,
        "sum_ctrl_sec": 0,
        "sum_kd": 0,
        "sum_sub_att": 0,
        "sum_rev": 0,
        "sum_opp_sig_landed": 0,
        "sum_opp_sig_attempted": 0,
        "sum_opp_td_landed": 0,
        "sum_opp_td_attempted": 0,
        "opponent_elos": [],
        "opponent_win_rates": [],
        "recent_fights": [],
        "last_fight_date": None,
        "last_weight_class": None,
        "num_weight_class_changes": 0,
        "champ_round_fights": 0,
        "sum_sig_diff": 0,
        "sum_td_diff": 0,
        "current_win_streak": 0,
        "current_loss_streak": 0,
        "last_loss_date": None,
        "sum_opp_finish_rate": 0,
        "opponent_finish_rates": [],
        "elo_finish": 1500,
        "elo": 1500,
    }


def _k_factor(n):
    """ELO K-factor: higher for inexperienced fighters (more volatile), lower for veterans."""
    return 64 if n < 5 else 32 if n < 20 else 16


class FeatureEngine:
    """Computes pre-fight features via a chronological loop.

    Processes fights in date order, maintaining per-fighter histories.
    Global statistics are updated *after* feature extraction so normalization
    uses only past data — no future leakage.
    """

    def __init__(self):
        self.fighter_histories = {}
        self.global_stats = {
            "sum_sig_landed": 0,
            "sum_sig_attempted": 0,
            "sum_td_landed": 0,
            "sum_td_attempted": 0,
            "sum_kd": 0,
            "sum_ctrl_sec": 0,
            "sum_sub_att": 0,
            "total_fights": 0,
        }

    def _global_avg(self, key):
        if self.global_stats["total_fights"] == 0:
            return 0.01
        return self.global_stats[key] / self.global_stats["total_fights"]

    def _compute_striking_score(self, hist):
        n = hist["num_fights"]
        if n == 0:
            return np.nan
        vol = (hist["sum_sig_landed"] / n) / max(self._global_avg("sum_sig_landed"), 0.01)
        acc = (hist["sum_sig_landed"] / max(hist["sum_sig_attempted"], 1)) / max(
            self._global_avg("sum_sig_landed") / max(self._global_avg("sum_sig_attempted"), 0.01),
            0.01,
        )
        kd = (hist["sum_kd"] / n) / max(self._global_avg("sum_kd"), 0.001)
        opp = hist["sum_opp_sig_attempted"]
        if opp > 0:
            my_d = 1 - hist["sum_opp_sig_landed"] / opp
            g_d = 1 - self._global_avg("sum_sig_landed") / max(
                self._global_avg("sum_sig_attempted"), 0.01
            )
            d_norm = my_d / max(g_d, 0.01)
        else:
            d_norm = 1.0
        return 0.30 * vol + 0.20 * acc + 0.15 * kd + 0.35 * d_norm

    def _compute_grappling_score(self, hist):
        n = hist["num_fights"]
        if n == 0:
            return np.nan
        td_v = (hist["sum_td_landed"] / n) / max(self._global_avg("sum_td_landed"), 0.01)
        g_td_acc = self._global_avg("sum_td_landed") / max(
            self._global_avg("sum_td_attempted"), 0.01
        )
        td_a = (hist["sum_td_landed"] / max(hist["sum_td_attempted"], 1)) / max(g_td_acc, 0.01)
        ctrl = (hist["sum_ctrl_sec"] / n) / max(self._global_avg("sum_ctrl_sec"), 0.1)
        sub = (hist["sum_sub_att"] / n) / max(self._global_avg("sum_sub_att"), 0.001)
        return 0.35 * td_v + 0.20 * td_a + 0.25 * ctrl + 0.20 * sub

    def _extract_features(self, hist, fighter_row):
        n = hist["num_fights"]
        is_debut = n == 0
        feats = {"is_debut": int(is_debut), "num_fights": n}

        if is_debut:
            feats.update(
                {
                    "win_rate": np.nan,
                    "finish_rate": np.nan,
                    "ko_rate": np.nan,
                    "sub_rate": np.nan,
                    "sig_landed_per_fight": np.nan,
                    "sig_acc": np.nan,
                    "sig_absorbed_per_fight": np.nan,
                    "striking_defense": np.nan,
                    "td_landed_per_fight": np.nan,
                    "td_acc": np.nan,
                    "td_defense": np.nan,
                    "ctrl_sec_per_fight": np.nan,
                    "kd_per_fight": np.nan,
                    "sub_att_per_fight": np.nan,
                    "avg_opp_elo": np.nan,
                    "avg_opp_win_rate": np.nan,
                    "days_since_last_fight": np.nan,
                    "champ_round_experience": 0,
                    "weight_class_changed": 0,
                    "num_weight_class_changes": 0,
                    "striking_score": np.nan,
                    "grappling_score": np.nan,
                    "recent_sig_landed_3": np.nan,
                    "recent_ctrl_sec_3": np.nan,
                    "recent_finish_rate_3": np.nan,
                    "elo": 1500,
                    "sig_diff_per_fight": np.nan,
                    "td_diff_per_fight": np.nan,
                    "win_streak": 0,
                    "loss_streak": 0,
                    "streak_diff": 0,
                    "days_since_last_loss": np.nan,
                    "avg_opp_finish_rate": np.nan,
                    "elo_finish": 1500,
                }
            )
        else:
            w = hist["wins"]
            r = hist["recent_fights"]
            feats.update(
                {
                    "win_rate": w / n,
                    "finish_rate": (hist["ko_wins"] + hist["sub_wins"]) / w if w > 0 else 0,
                    "ko_rate": hist["ko_wins"] / w if w > 0 else 0,
                    "sub_rate": hist["sub_wins"] / w if w > 0 else 0,
                    "sig_landed_per_fight": hist["sum_sig_landed"] / n,
                    "sig_acc": (
                        hist["sum_sig_landed"] / hist["sum_sig_attempted"]
                        if hist["sum_sig_attempted"] > 0
                        else np.nan
                    ),
                    "sig_absorbed_per_fight": hist["sum_opp_sig_landed"] / n,
                    "striking_defense": (
                        1 - hist["sum_opp_sig_landed"] / hist["sum_opp_sig_attempted"]
                        if hist["sum_opp_sig_attempted"] > 0
                        else np.nan
                    ),
                    "td_landed_per_fight": hist["sum_td_landed"] / n,
                    "td_acc": (
                        hist["sum_td_landed"] / hist["sum_td_attempted"]
                        if hist["sum_td_attempted"] > 0
                        else np.nan
                    ),
                    "td_defense": (
                        1 - hist["sum_opp_td_landed"] / hist["sum_opp_td_attempted"]
                        if hist["sum_opp_td_attempted"] > 0
                        else np.nan
                    ),
                    "ctrl_sec_per_fight": hist["sum_ctrl_sec"] / n,
                    "kd_per_fight": hist["sum_kd"] / n,
                    "sub_att_per_fight": hist["sum_sub_att"] / n,
                    "avg_opp_elo": (
                        np.mean(hist["opponent_elos"]) if hist["opponent_elos"] else np.nan
                    ),
                    "avg_opp_win_rate": (
                        np.mean(hist["opponent_win_rates"]) if hist["opponent_win_rates"] else np.nan
                    ),
                    "days_since_last_fight": (
                        (fighter_row["date"] - hist["last_fight_date"]).days
                        if hist["last_fight_date"] is not None
                        else np.nan
                    ),
                    "champ_round_experience": hist["champ_round_fights"],
                    "weight_class_changed": int(
                        hist["last_weight_class"] is not None
                        and hist["last_weight_class"] != fighter_row.get("weight_class")
                    ),
                    "num_weight_class_changes": hist["num_weight_class_changes"],
                    "elo": hist["elo"],
                    "striking_score": self._compute_striking_score(hist),
                    "grappling_score": self._compute_grappling_score(hist),
                    "sig_diff_per_fight": hist["sum_sig_diff"] / n,
                    "td_diff_per_fight": hist["sum_td_diff"] / n,
                    "win_streak": hist["current_win_streak"],
                    "loss_streak": hist["current_loss_streak"],
                    "streak_diff": hist["current_win_streak"] - hist["current_loss_streak"],
                    "days_since_last_loss": (
                        (fighter_row["date"] - hist["last_loss_date"]).days
                        if hist["last_loss_date"] is not None
                        else np.nan
                    ),
                    "avg_opp_finish_rate": (
                        np.mean(hist["opponent_finish_rates"])
                        if hist["opponent_finish_rates"]
                        else np.nan
                    ),
                    "elo_finish": hist["elo_finish"],
                }
            )
            if len(r) >= 3:
                r3 = r[-3:]
                feats["recent_sig_landed_3"] = np.mean([f["sig_landed"] for f in r3])
                feats["recent_ctrl_sec_3"] = np.mean([f["ctrl_sec"] for f in r3])
                feats["recent_finish_rate_3"] = sum(f["finished"] for f in r3) / len(r3)
            else:
                feats["recent_sig_landed_3"] = np.nan
                feats["recent_ctrl_sec_3"] = np.nan
                feats["recent_finish_rate_3"] = np.nan

        feats["height_inches"] = fighter_row.get("height_inches", np.nan)
        feats["weight_lbs"] = fighter_row.get("weight_lbs", np.nan)
        feats["reach_inches"] = fighter_row.get("reach_inches", np.nan)
        feats["stance"] = fighter_row.get("STANCE", np.nan)

        class_limit = CLASS_LIMITS.get(fighter_row.get("weight_class"))
        if class_limit and pd.notna(fighter_row.get("weight_lbs")):
            feats["weight_cut_pct"] = (class_limit - fighter_row["weight_lbs"]) / class_limit
        else:
            feats["weight_cut_pct"] = np.nan
        return feats

    def _update_history(self, hist, my_row, opp_row, opp_hist):
        """Update a fighter's history after a fight using opponent stats for defense."""
        outcome = my_row["outcome"]
        if outcome == "no_contest":
            return
        hist["num_fights"] += 1
        hist["last_fight_date"] = my_row["date"]

        if outcome == "win":
            hist["wins"] += 1
            mc = my_row.get("method_cat")
            if mc == "KO/TKO":
                hist["ko_wins"] += 1
            elif mc == "SUB":
                hist["sub_wins"] += 1
            elif mc == "DEC":
                hist["dec_wins"] += 1
        elif outcome == "loss":
            hist["losses"] += 1
        else:
            hist["draws"] += 1

        hist["sum_sig_landed"] += np.nan_to_num(my_row.get("sig_landed", 0))
        hist["sum_sig_attempted"] += np.nan_to_num(my_row.get("sig_attempted", 0))
        hist["sum_td_landed"] += np.nan_to_num(my_row.get("td_landed", 0))
        hist["sum_td_attempted"] += np.nan_to_num(my_row.get("td_attempted", 0))
        hist["sum_ctrl_sec"] += np.nan_to_num(my_row.get("ctrl_sec", 0))
        hist["sum_kd"] += np.nan_to_num(my_row.get("KD", 0))
        hist["sum_sub_att"] += np.nan_to_num(my_row.get("SUB.ATT", 0))
        hist["sum_rev"] += np.nan_to_num(my_row.get("REV.", 0))

        hist["sum_opp_sig_landed"] += np.nan_to_num(opp_row.get("sig_landed", 0))
        hist["sum_opp_sig_attempted"] += np.nan_to_num(opp_row.get("sig_attempted", 0))
        hist["sum_opp_td_landed"] += np.nan_to_num(opp_row.get("td_landed", 0))
        hist["sum_opp_td_attempted"] += np.nan_to_num(opp_row.get("td_attempted", 0))

        my_sig = np.nan_to_num(my_row.get("sig_landed", 0))
        opp_sig = np.nan_to_num(opp_row.get("sig_landed", 0))
        hist["sum_sig_diff"] += my_sig - opp_sig
        my_td = np.nan_to_num(my_row.get("td_landed", 0))
        opp_td = np.nan_to_num(opp_row.get("td_landed", 0))
        hist["sum_td_diff"] += my_td - opp_td

        if outcome == "win":
            hist["current_win_streak"] += 1
            hist["current_loss_streak"] = 0
        elif outcome == "loss":
            hist["current_loss_streak"] += 1
            hist["current_win_streak"] = 0

        if outcome == "loss":
            hist["last_loss_date"] = my_row["date"]

        opp_w = opp_hist["wins"]
        opp_finish_rate = (
            (opp_hist["ko_wins"] + opp_hist["sub_wins"]) / opp_w if opp_w > 0 else 0
        )
        hist["sum_opp_finish_rate"] += opp_finish_rate
        hist["opponent_finish_rates"].append(opp_finish_rate)
        hist["opponent_elos"].append(opp_hist["elo"])
        nr = opp_hist["num_fights"]
        hist["opponent_win_rates"].append(opp_hist["wins"] / nr if nr > 0 else 0)

        hist["recent_fights"].append(
            {
                "sig_landed": my_row.get("sig_landed", 0) or 0,
                "ctrl_sec": my_row.get("ctrl_sec", 0) or 0,
                "finished": outcome == "win"
                and my_row.get("method_cat") in ("KO/TKO", "SUB"),
            }
        )
        if len(hist["recent_fights"]) > 5:
            hist["recent_fights"].pop(0)

        if (
            hist["last_weight_class"] is not None
            and hist["last_weight_class"] != my_row.get("weight_class")
        ):
            hist["num_weight_class_changes"] += 1
        hist["last_weight_class"] = my_row.get("weight_class")

        if my_row.get("champ_round_possible", False):
            hist["champ_round_fights"] += 1

    def run(self, chrono):
        """Iterate fights chronologically, extract pre-fight features, update histories.

        Returns
        -------
        feature_df : pd.DataFrame  — one row per fight_id with fighter_r_* and fighter_b_* columns
        fighter_histories : dict   — final state of every fighter's history (for inference)
        global_stats : dict        — final global averages (for pairwise features)
        """
        fight_ids_in_order = (
            chrono[["fight_id", "date"]]
            .drop_duplicates("fight_id")
            .sort_values("date")["fight_id"]
            .tolist()
        )
        print(f"Processing {len(fight_ids_in_order)} fights chronologically...")

        features_cache = {}

        for i, fight_id in enumerate(fight_ids_in_order):
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(fight_ids_in_order)}")

            frows = chrono[chrono["fight_id"] == fight_id]
            if len(frows) != 2:
                continue
            r_row = frows[frows["role"] == "r"].iloc[0]
            b_row = frows[frows["role"] == "b"].iloc[0]
            f_r, f_b = r_row["fighter"], b_row["fighter"]

            if f_r not in self.fighter_histories:
                self.fighter_histories[f_r] = _init_fighter_history(f_r)
            if f_b not in self.fighter_histories:
                self.fighter_histories[f_b] = _init_fighter_history(f_b)
            h_r, h_b = self.fighter_histories[f_r], self.fighter_histories[f_b]

            feat_r = self._extract_features(h_r, r_row)
            feat_b = self._extract_features(h_b, b_row)
            feat_r["opponent_elo"] = h_b["elo"]
            feat_b["opponent_elo"] = h_r["elo"]
            features_cache[fight_id] = {"r": feat_r, "b": feat_b}

            o_r = r_row["outcome"]

            if o_r != "no_contest":
                self.global_stats["total_fights"] += 1
                col_map2 = {
                    "sum_sig_landed": "sig_landed",
                    "sum_sig_attempted": "sig_attempted",
                    "sum_td_landed": "td_landed",
                    "sum_td_attempted": "td_attempted",
                    "sum_kd": "KD",
                    "sum_ctrl_sec": "ctrl_sec",
                    "sum_sub_att": "SUB.ATT",
                }
                for k, col in col_map2.items():
                    self.global_stats[k] += np.nan_to_num(r_row.get(col, 0)) + np.nan_to_num(
                        b_row.get(col, 0)
                    )

            elo_r, elo_b = h_r["elo"], h_b["elo"]
            expected_r = 1 / (1 + 10 ** ((elo_b - elo_r) / 400))
            actual_r = 1 if o_r == "win" else (0 if o_r == "loss" else 0.5)
            if o_r != "no_contest":
                h_r["elo"] += _k_factor(h_r["num_fights"]) * (actual_r - expected_r)
                h_b["elo"] += _k_factor(h_b["num_fights"]) * ((1 - actual_r) - (1 - expected_r))

                if o_r == "win":
                    actual_finish_r = (
                        1 if r_row.get("method_cat") in ("KO/TKO", "SUB") else 0.5
                    )
                elif o_r == "loss":
                    actual_finish_r = 0
                else:
                    actual_finish_r = 0.5
                h_r["elo_finish"] += _k_factor(h_r["num_fights"]) * (actual_finish_r - expected_r)
                h_b["elo_finish"] += _k_factor(h_b["num_fights"]) * (
                    (1 - actual_finish_r) - (1 - expected_r)
                )

            self._update_history(h_r, r_row, b_row, h_b)
            self._update_history(h_b, b_row, r_row, h_r)

        print("Done processing fights.")
        feature_rows = []
        for fid, feats in features_cache.items():
            row = {"fight_id": fid}
            for k, v in feats["r"].items():
                row[f"fighter_r_{k}"] = v
            for k, v in feats["b"].items():
                row[f"fighter_b_{k}"] = v
            feature_rows.append(row)

        feature_df = pd.DataFrame(feature_rows)
        return feature_df, self.fighter_histories, self.global_stats


# ---------------------------------------------------------------------------
# 6.  Inference helper — compute features for any fighter at any date
# ---------------------------------------------------------------------------

def compute_fighter_features(name, fight_date, histories, tott_df,
                             opponent_elo=1500, weight_class=None):
    """Build pre-fight feature dict for *name* as if fighting at *fight_date*.

    Used by inference.py to generate feature vectors for hypothetical matchups.
    """
    import pandas as pd
    if name not in histories:
        hist = _init_fighter_history(name)
    else:
        hist = histories[name]

    phys = tott_df[tott_df["FIGHTER"] == name]
    if len(phys) > 0:
        p = phys.iloc[0]
    else:
        p = pd.Series(
            {
                "height_inches": np.nan,
                "weight_lbs": np.nan,
                "reach_inches": np.nan,
                "STANCE": np.nan,
            }
        )

    from config import CLASS_LIMITS
    fighter_row = {
        "date": fight_date,
        "height_inches": p["height_inches"],
        "weight_lbs": p["weight_lbs"],
        "reach_inches": p["reach_inches"],
        "STANCE": p["STANCE"],
        "weight_class": weight_class,
        "champ_round_possible": False,
    }

    engine = FeatureEngine()
    feats = engine._extract_features(hist, fighter_row)
    feats["opponent_elo"] = opponent_elo
    return feats
