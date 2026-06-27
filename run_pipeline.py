#!/usr/bin/env python3
"""End-to-end UFC fight prediction pipeline.

Usage:
    python run_pipeline.py                        # Full pipeline
    python run_pipeline.py --skip-features        # Load pre-computed master_df.csv
    python run_pipeline.py --no-cv                # Skip time-series CV (faster)
    python run_pipeline.py --no-diagnostics       # Skip rolling-window meta analysis
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "UFCPREDv3"))

from theme import set_ufc_theme
from config import (
    DATA_DIR, OUTPUT_DIR, CUTOFF_DATE, CUTOFF_2005,
    ADVANTAGE_FEATURE_COLS, RANDOM_SEED
)
from data_loader import load_raw_data, clean_raw_data, build_master_df
from features import (
    parse_stats_table, parse_physicals, parse_fight_metadata,
    build_chrono_table, FeatureEngine
)
from pairwise import add_all_pairwise_features
from models import (
    prepare_symmetrized_data, temporal_split, run_model_comparison,
    time_series_cv, plot_evaluation_dashboard, plot_model_comparison,
    impute_and_scale, print_comparison_table
)


def main(argv=None):
    """argv — pass a list to override sys.argv (used from Jupyter notebooks)."""
    set_ufc_theme()
    parser = argparse.ArgumentParser(description="UFC Prediction Pipeline v3")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering; load pre-computed master_df")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip time-series cross-validation")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="Skip rolling-window meta analysis")
    args = parser.parse_args(argv)

    # Ensure output directories exist
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

    # =========================================================================
    # PHASE 1: DATA LOADING & CLEANING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Loading and cleaning raw data")
    print("=" * 70)

    fights, stats, events, details, tott = load_raw_data(DATA_DIR)
    events, fights, stats, details, tott = clean_raw_data(
        events, fights, stats, details, tott
    )
    print(f"  Raw files loaded: fights={len(fights)}, stats={len(stats)}, "
          f"events={len(events)}, tott={len(tott)}")

    # =========================================================================
    # PHASE 2: BUILD MASTER DATAFRAME
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Building master DataFrame (join + outcome)")
    print("=" * 70)

    if args.skip_features:
        csv_path = os.path.join(OUTPUT_DIR, "data", "ufc_master_features.csv")
        print(f"  Loading pre-computed master_df from {csv_path}...")
        master_df = pd.read_csv(csv_path)
        master_df["DATE"] = pd.to_datetime(master_df["DATE"], errors="coerce")
        global_stats = None
        fighter_histories = None
    else:
        master_df = build_master_df(fights, events)

        # =====================================================================
        # PHASE 3: FEATURE ENGINEERING
        # =====================================================================
        print("\n" + "=" * 70)
        print("PHASE 3: Feature engineering")
        print("=" * 70)

        # -- 3a: Parse stats, physicals, metadata --
        print("\n  3a: Parsing stats table...")
        fighter_fights = parse_stats_table(stats)
        print(f"      fighter_fights shape: {fighter_fights.shape}")

        print("\n  3b: Parsing physicals & metadata...")
        tott_clean = parse_physicals(tott)
        fights_meta = parse_fight_metadata(fights)

        print("\n  3c: Building chronological table...")
        chrono = build_chrono_table(master_df, fighter_fights, tott_clean, fights_meta)
        print(f"      chrono shape: {chrono.shape}, date range: {chrono['date'].min()} → {chrono['date'].max()}")

        # -- 3d: Chronological feature engine --
        print("\n  3d: Running chronological feature engine...")
        engine = FeatureEngine()
        feature_df, fighter_histories, global_stats = engine.run(chrono)

        # -- 3e: Merge features into master_df --
        master_df = master_df.merge(feature_df, on="fight_id", how="left")
        n_feat = len([c for c in master_df.columns if c.startswith("fighter_")])
        print(f"      master_df shape with features: {master_df.shape}, feature cols: {n_feat}")

        # -- 3f: Pairwise / matchup features --
        print("\n  3e: Adding pairwise matchup features...")
        master_df = add_all_pairwise_features(master_df, tott_clean, global_stats)
        print(f"      master_df shape after pairwise: {master_df.shape}")

        # Export to CSV for future use
        export_path = os.path.join(OUTPUT_DIR, "data", "ufc_master_features.csv")
        master_df.to_csv(export_path, index=False)
        print(f"\n      Exported master_df ({master_df.shape}) to {export_path}")

    # =========================================================================
    # PHASE 4: MODELING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Modeling")
    print("=" * 70)

    # -- 4a: Symmetrize and prepare target --
    print("\n  4a: Preparing symmetrized dataset...")
    sym_df, feature_cols = prepare_symmetrized_data(master_df)

    # -- 4b: Temporal split (full dataset) --
    print("\n  4b: Temporal split (full dataset)...")
    X_train, X_val, y_train, y_val, train_df, val_df = temporal_split(
        sym_df, feature_cols, cutoff=CUTOFF_DATE
    )

    # -- 4c: Full-dataset model comparison --
    print("\n  4c: Training models (full dataset)...")
    results_full, xgb_model, rf_model, lr_model, probs_full = run_model_comparison(
        X_train, X_val, y_train, y_val, label_suffix="", return_probs=True
    )
    print_comparison_table(results_full)

    # -- 4d: ELO baseline on unique fights --
    unique_val = val_df.drop_duplicates(subset="fight_id", keep="first")
    elo_pred = (unique_val["elo_diff"] > 0).astype(int)
    elo_baseline = accuracy_score(unique_val["target"], elo_pred)

    elo_diff_val = np.nan_to_num(val_df["elo_diff"].values, nan=0.0)
    elo_range = elo_diff_val.max() - elo_diff_val.min()
    elo_prob_full = np.where(elo_range > 1e-8,
                             (elo_diff_val - elo_diff_val.min()) / elo_range,
                             np.full_like(elo_diff_val, 0.5))
    elo_full = accuracy_score(val_df["target"], (elo_diff_val > 0).astype(int))
    print(f"  ELO baseline (unique val fights): {elo_baseline:.4f}")

    xgb_pred_uniq = xgb_model.predict(unique_val[feature_cols])
    xgb_lift = accuracy_score(unique_val["target"], xgb_pred_uniq) - elo_baseline
    print(f"  XGBoost lift over ELO: {xgb_lift:+.4f}")

    # -- 4e: Evaluation dashboard --
    y_prob_xgb = probs_full["XGBoost"]
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
    plot_evaluation_dashboard(
        y_val, y_prob_xgb, y_pred_xgb, feature_cols, xgb_model,
        save_path=os.path.join(OUTPUT_DIR, "plots", "xgb_evaluation_dashboard.png"),
    )

    # -- 4f: Post-2005 subset comparison --
    print("\n  4f: Post-2005 subset...")
    sym_df_2005 = sym_df[sym_df["DATE"] >= pd.Timestamp(CUTOFF_2005)].copy()
    train_2005 = sym_df_2005[sym_df_2005["DATE"] <= pd.Timestamp(CUTOFF_DATE)]
    val_2005 = sym_df_2005[sym_df_2005["DATE"] > pd.Timestamp(CUTOFF_DATE)]
    X_tr_2005, X_va_2005 = train_2005[feature_cols], val_2005[feature_cols]
    y_tr_2005, y_va_2005 = train_2005["target"], val_2005["target"]
    print(f"  2005+ Train: {len(train_2005)}, Val: {len(val_2005)}")

    results_2005, model_2005, _, _, probs_2005 = run_model_comparison(
        X_tr_2005, X_va_2005, y_tr_2005, y_va_2005,
        label_suffix=" (2005+)", return_probs=True
    )
    print_comparison_table(results_2005)

    # ELO baseline for 2005+ subset
    elo_diff_2005 = np.nan_to_num(val_2005["elo_diff"].values, nan=0.0)
    elo_range_2005 = elo_diff_2005.max() - elo_diff_2005.min()
    elo_prob_2005 = np.where(elo_range_2005 > 1e-8,
                             (elo_diff_2005 - elo_diff_2005.min()) / elo_range_2005,
                             np.full_like(elo_diff_2005, 0.5))
    elo_2005 = accuracy_score(val_2005["target"], (elo_diff_2005 > 0).astype(int))
    print(f"  ELO baseline (2005+ val): {elo_2005:.4f}")

    # Combined table
    print(f"\n{'='*70}")
    print(f"{'Full Dataset vs 2005+ Comparison':^70}")
    print(f"{'='*70}")
    h = f"{'Model':<20} {'Acc(F)':<8} {'AUC(F)':<8} {'Acc(05+)':<8} {'AUC(05+)':<8}"
    print(h)
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label in ["XGBoost", "Random Forest", "Logistic Reg"]:
        r_full = results_full.get(label, {})
        r_2005 = results_2005.get(f"{label} (2005+)", {})
        print(f"{label:<20} {r_full.get('accuracy', 0):<8.4f} "
              f"{r_full.get('auc_roc', 0):<8.4f} "
              f"{r_2005.get('accuracy', 0):<8.4f} "
              f"{r_2005.get('auc_roc', 0):<8.4f}")
    print(f"{'ELO':<20} {elo_full:<8.4f} {'':<8} {elo_2005:<8.4f}")
    print(f"{'='*70}\n")

    # Model comparison plot
    plot_model_comparison(
        y_val, probs_full, elo_prob_full,
        y_va_2005, probs_2005, elo_prob_2005,
        save_path=os.path.join(OUTPUT_DIR, "plots", "model_comparison.png"),
    )

    # -- 4g: Time-series cross-validation --
    if not args.no_cv:
        print("\n  4g: Time-series cross-validation (expanding window, XGBoost)...")
        cv_results, cv_models = time_series_cv(sym_df, feature_cols, n_splits=5)
    else:
        print("\n  4g: Time-series CV skipped (--no-cv).")

    # =========================================================================
    # PHASE 5: INFERENCE DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Inference demo")
    print("=" * 70)

    if not args.skip_features and fighter_histories is not None and global_stats is not None:
        g_striking_def = 1 - (global_stats["sum_sig_landed"] / global_stats["sum_sig_attempted"])
        g_td_def = 1 - (global_stats["sum_td_landed"] / global_stats["sum_td_attempted"])

        from inference import predict_fight
        demo_a, demo_b = "Islam Makhachev", "Charles Oliveira"
        demo_date = pd.Timestamp("2026-12-31")

        prob = predict_fight(
            demo_a, demo_b, demo_date,
            fighter_histories, tott_clean if not args.skip_features else None,
            xgb_model, g_striking_def, g_td_def, feature_cols,
            weight_class="Lightweight"
        )
        print(f"\n  PREDICTION: {demo_a} vs {demo_b}")
        print(f"  {demo_a} win probability: {prob:.1%}")
        print(f"  {demo_b} win probability: {1 - prob:.1%}")

    # =========================================================================
    # PHASE 6: DIAGNOSTICS (rolling window meta analysis)
    # =========================================================================
    if not args.no_diagnostics and not args.skip_features:
        print("\n" + "=" * 70)
        print("PHASE 6: Rolling-window meta analysis (diagnostics)")
        print("=" * 70)
        from diagnostics import run_rolling_meta, plot_rolling_results
        result = run_rolling_meta(sym_df_2005)
        if result is not None:
            coeff_df, wm, wa, ws, *_ = result
            plot_rolling_results(
                coeff_df, wm, wa, ws,
                save_path=os.path.join(OUTPUT_DIR, "plots", "rolling_window_meta.png"),
            )

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
