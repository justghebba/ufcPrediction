import os
import pandas as pd
import numpy as np


def load_raw_data(data_dir: str):
    """Read the five raw CSVs scraped from ufcstats.com.

    Returns
    -------
    fights, stats, events, details, tott : pd.DataFrame
    """
    fights = pd.read_csv(os.path.join(data_dir, "ufc_fight_results.csv"))
    stats = pd.read_csv(os.path.join(data_dir, "ufc_fight_stats.csv"))
    events = pd.read_csv(os.path.join(data_dir, "ufc_event_details.csv"))
    details = pd.read_csv(os.path.join(data_dir, "ufc_fighter_details.csv"))
    tott = pd.read_csv(os.path.join(data_dir, "ufc_fighter_tott.csv"))
    return fights, stats, events, details, tott


def clean_raw_data(events, fights, stats, details, tott):
    """Replace '---' placeholders with proper NaN so numeric parsing works.

    ufcstats.com uses '---' to indicate missing data; converting to NaN early
    prevents misclassification as valid strings during feature engineering.
    """
    events = events.replace("---", np.nan)
    fights = fights.replace("---", np.nan)
    stats = stats.replace("---", np.nan)
    details = details.replace("---", np.nan)
    tott = tott.replace("---", np.nan)
    return events, fights, stats, details, tott


def build_master_df(fights, events):
    """Join fights → events, split BOUT into red/blue corners, derive winner.

    This is the central table: one row per bout, with fighter names, date,
    outcome, and a fight_id URL that serves as a join key for stat tables.
    """
    fights = fights.copy()
    events = events.copy()

    fights["EVENT"] = fights["EVENT"].str.strip()
    events["EVENT"] = events["EVENT"].str.strip()
    events["DATE"] = pd.to_datetime(events["DATE"], format="mixed", errors="coerce")

    master_df = fights.merge(events[["EVENT", "DATE"]], on="EVENT", how="left")
    master_df[["fighter_r", "fighter_b"]] = master_df["BOUT"].str.split(
        " vs. ", expand=True
    )

    def get_winner(row):
        if row["OUTCOME"] == "W/L":
            return row["fighter_r"]
        if row["OUTCOME"] == "L/W":
            return row["fighter_b"]
        if row["OUTCOME"] == "D/D":
            return "Draw"
        if row["OUTCOME"] == "NC/NC":
            return "No Contest"
        return None

    master_df["winner"] = master_df.apply(get_winner, axis=1)
    master_df = master_df[
        ["URL", "DATE", "EVENT", "BOUT", "fighter_r", "fighter_b", "winner"]
    ].rename(columns={"URL": "fight_id"})

    return master_df


def export_master_df(master_df, path: str):
    """Persist master_df to CSV so downstream consumers can skip feature re-build."""
    master_df.to_csv(path, index=False)
    print(f"Exported master_df ({master_df.shape}) to {path}")
