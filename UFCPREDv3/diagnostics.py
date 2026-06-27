"""Rolling-window meta analysis: tracks how each advantage feature's predictive
weight changes over UFC history.  Diagnostic only — not integrated into the
main prediction pipeline pending leakage fixes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import ADVANTAGE_FEATURE_COLS


def run_rolling_meta(sym_df_2005, window_years=3, step_days=183):
    """Train Logistic Regression on sliding windows and return per-window coefficients.

    Parameters
    ----------
    sym_df_2005 : pd.DataFrame — symmetrized data filtered to >= 2005
    window_years : int — length of each rolling window
    step_days : int — step size between windows

    Returns
    -------
    coeff_df : pd.DataFrame — coefficient trajectories indexed by window midpoint
    window_midpoints, window_accs, window_sizes : lists for plotting
    window_models, window_scalers, window_imputers, window_features : for inference
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    available = [f for f in ADVANTAGE_FEATURE_COLS if f in sym_df_2005.columns]
    print(f"Using {len(available)} advantage features: {available}")

    meta_df = sym_df_2005.sort_values("DATE").reset_index(drop=True)
    dates = meta_df["DATE"]

    window_size = pd.Timedelta(days=365 * window_years)
    step_size = pd.Timedelta(days=step_days)

    window_models = []
    window_scalers = []
    window_imputers = []
    window_midpoints = []
    window_sizes = []
    window_accs = []
    window_features = []

    cur = pd.Timestamp("2005-01-01")
    end = dates.max()

    while cur + window_size <= end:
        mask = (dates >= cur) & (dates < cur + window_size)
        n = mask.sum()
        if n < 200:
            cur += step_size
            continue

        X_sub = meta_df.loc[mask, available].copy()
        y_sub = meta_df.loc[mask, "target"]

        valid = X_sub.dropna(how="all").index
        X_sub = X_sub.loc[valid]
        y_sub = y_sub.loc[valid]

        all_nan = [c for c in X_sub.columns if X_sub[c].isna().all()]
        if all_nan:
            X_sub = X_sub.drop(columns=all_nan)

        this_feats = list(X_sub.columns)

        if y_sub.nunique() < 2:
            cur += step_size
            continue

        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_sub)

        scaler = StandardScaler()
        X_scl = scaler.fit_transform(X_imp)

        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_scl, y_sub)

        window_models.append(lr)
        window_scalers.append(scaler)
        window_imputers.append(imp)
        window_midpoints.append(cur + window_size / 2)
        window_sizes.append(n)
        window_features.append(this_feats)
        window_accs.append(accuracy_score(y_sub, lr.predict(X_scl)))

        cur += step_size

    print(f"Trained {len(window_models)} window models")
    print(f"Date range: {window_midpoints[0].date()} to {window_midpoints[-1].date()}")
    print(f"Window sizes: {min(window_sizes)} - {max(window_sizes)} fights")
    print(f"Window accuracy range: {min(window_accs):.3f} to {max(window_accs):.3f}")

    # Build coefficient DataFrame — windows may have different feature sets
    all_rows = []
    for m, feats in zip(window_models, window_features):
        row = dict(zip(feats, m.coef_[0]))
        all_rows.append(row)
    coeff_df = pd.DataFrame(all_rows, index=window_midpoints).fillna(0)
    coeff_cols = [c for c in available if c in coeff_df.columns]
    coeff_df = coeff_df[coeff_cols]

    print(f"\nAverage |coefficient| by feature (higher = more important):")
    print(coeff_df.abs().mean().sort_values(ascending=False).to_string())

    return (coeff_df, window_midpoints, window_accs, window_sizes,
            window_models, window_scalers, window_imputers, window_features)


def plot_rolling_results(coeff_df, window_midpoints, window_accs, window_sizes, save_path=None):
    """Two-panel figure: coefficient trajectories + accuracy/size."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    ax = axes[0]
    for col in coeff_df.columns:
        ax.plot(coeff_df.index, coeff_df[col], label=col, linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Window Midpoint")
    ax.set_ylabel("Coefficient (standardized features)")
    ax.set_title("Advantage Importance Over Time (Rolling Windows, LogReg)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(window_midpoints, window_accs, "b-o", label="Accuracy", linewidth=2)
    ax2_twin.bar(window_midpoints, window_sizes, alpha=0.3, color="gray", label="Fights")
    ax2.set_xlabel("Window Midpoint")
    ax2.set_ylabel("Accuracy", color="b")
    ax2_twin.set_ylabel("Window Size", color="gray")
    ax2.set_title("Window Accuracy & Size")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
