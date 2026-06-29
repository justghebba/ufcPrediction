import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from config import CUTOFF_DATE, CUTOFF_2005, XGB_PARAMS, RF_PARAMS, LR_PARAMS, OUTPUT_DIR, FEATURE_GROUPS, FEATURE_LABELS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

# ---------------------------------------------------------------------------
# 1.  Symmetrization & target preparation
# ---------------------------------------------------------------------------

def prepare_symmetrized_data(master_df):
    """Drop draws/NCs, create binary target, and symmetrize to eliminate red-corner bias.

    Symmetrization duplicates every fight with r/b roles swapped so the model
    cannot learn corner position as a predictive signal.  Returns the symmetrized
    DataFrame plus the list of feature column names.
    """
    df = master_df.copy()
    df["target"] = np.where(
        df["winner"] == df["fighter_r"], 1,
        np.where(df["winner"] == df["fighter_b"], 0, np.nan),
    )
    model_df = df.dropna(subset=["target"]).copy()
    model_df["target"] = model_df["target"].astype(int)

    r_feat_cols = sorted([c for c in model_df.columns if c.startswith("fighter_r_")])
    b_feat_cols = sorted([c for c in model_df.columns if c.startswith("fighter_b_")])

    swapped = model_df.copy()
    for r_col in r_feat_cols:
        b_col = r_col.replace("fighter_r_", "fighter_b_", 1)
        if b_col in model_df.columns:
            swapped[r_col] = model_df[b_col].values
            swapped[b_col] = model_df[r_col].values

    swapped["fighter_r"] = model_df["fighter_b"].values
    swapped["fighter_b"] = model_df["fighter_r"].values

    diff_cols = [c for c in swapped.columns if c.endswith("_diff")]
    for col in diff_cols:
        swapped[col] = -model_df[col].values

    swapped["target"] = 1 - model_df["target"].values

    sym_df = pd.concat([model_df, swapped], ignore_index=True)
    sym_df = sym_df.sort_values("DATE").reset_index(drop=True)

    corner_physicals = {
        # Dropped per-corner physicals — the _diff versions carry the matchup signal
        "fighter_r_reach_inches", "fighter_b_reach_inches",
        "fighter_r_height_inches", "fighter_b_height_inches",
        "fighter_r_weight_lbs", "fighter_b_weight_lbs",
        "fighter_r_age", "fighter_b_age",
        "fighter_r_age_sq", "fighter_b_age_sq",
        "fighter_r_age_cubed", "fighter_b_age_cubed",
        "fighter_r_southpaw", "fighter_b_southpaw",
    }
    exclude_cols = {
        "fight_id", "DATE", "EVENT", "BOUT",
        "fighter_r", "fighter_b", "winner", "target",
        "fighter_r_stance", "fighter_b_stance",
    } | corner_physicals
    feature_cols = [c for c in sym_df.columns if c not in exclude_cols]
    print(f"  Original fights (excl. draws/NC): {len(model_df)}")
    print(f"  After symmetrization: {len(sym_df)} rows, target balance={sym_df['target'].mean():.3f}")
    print(f"  Feature columns: {len(feature_cols)}")
    return sym_df, feature_cols


# ---------------------------------------------------------------------------
# 2.  Temporal train/val split
# ---------------------------------------------------------------------------

def temporal_split(sym_df, feature_cols, cutoff=CUTOFF_DATE, date_col="DATE"):
    """Strictly chronological split — no future leakage into training.

    Uses a date cutoff (default 2024-06-06) rather than random sampling
    to ensure temporal validity.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train = sym_df[sym_df[date_col] <= cutoff_ts].copy()
    val = sym_df[sym_df[date_col] > cutoff_ts].copy()

    X_train, y_train = train[feature_cols], train["target"]
    X_val, y_val = val[feature_cols], val["target"]

    print(f"  Train: {len(train)} rows ({train[date_col].min().date()} → {train[date_col].max().date()})")
    print(f"  Val:   {len(val)} rows ({val[date_col].min().date()} → {val[date_col].max().date()})")
    print(f"  X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val, train, val


# ---------------------------------------------------------------------------
# 3.  Imputation & scaling (for RF and LR — XGBoost handles NaN natively)
# ---------------------------------------------------------------------------

def impute_and_scale(X_train, X_val):
    """Impute medians, add missing-indicator columns, and standardize.

    Two-step pipeline:
    1. Drop all-NaN columns, add binary missing-indicators, median-impute.
    2. StandardScaler (fit on train, transform both).
    Returns the imputed DataFrames and the scaled arrays.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_tr = X_train.copy()
    X_va = X_val.copy()

    all_nan = [c for c in X_tr.columns if X_tr[c].isna().all()]
    if all_nan:
        X_tr.drop(columns=all_nan, inplace=True)
        X_va.drop(columns=all_nan, errors="ignore", inplace=True)

    nan_cols = [c for c in X_tr.columns if X_tr[c].isna().any()]
    missing_train = pd.DataFrame(
        {f"{col}_missing": X_tr[col].isna().astype(int) for col in nan_cols}
    )
    missing_val = pd.DataFrame(
        {f"{col}_missing": X_va[col].isna().astype(int) for col in nan_cols}
    )
    X_tr = pd.concat([X_tr, missing_train], axis=1)
    X_va = pd.concat([X_va, missing_val], axis=1)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

    imputer = SimpleImputer(strategy="median")
    X_tr_arr = imputer.fit_transform(X_tr)
    X_va_arr = imputer.transform(X_va)
    feat_cols_imp = list(imputer.feature_names_in_)
    X_tr_imp = pd.DataFrame(X_tr_arr, columns=feat_cols_imp, index=X_train.index)
    X_va_imp = pd.DataFrame(X_va_arr, columns=feat_cols_imp, index=X_val.index)

    scaler = StandardScaler()
    X_tr_scl = scaler.fit_transform(X_tr_imp)
    X_va_scl = scaler.transform(X_va_imp)

    return X_tr_imp, X_va_imp, X_tr_scl, X_va_scl, imputer, scaler


# ---------------------------------------------------------------------------
# 4.  Model training functions
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, X_val, y_val):
    """Train XGBoost with early stopping on validation logloss.

    XGBoost is chosen as the primary model because it handles NaN natively,
    captures non-linear feature interactions via trees, and is robust to
    irrelevant features via column subsampling.
    """
    import xgboost as xgb
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=0,
    )
    print(f"  XGBoost best iteration: {model.best_iteration}, best logloss: {model.best_score:.4f}")
    return model


def train_rf(X_train, y_train):
    """Train Random Forest on imputed features (no scaling needed).

    RF provides a tree-based ensemble benchmark that is less prone to
    overfitting than a single XGBoost with deep trees.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_lr(X_train_scaled, y_train):
    """Train Logistic Regression on scaled imputed features.

    LR serves as a simple linear baseline and provides interpretable
    coefficients showing the directional impact of each feature.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train_scaled, y_train)
    return model


# ---------------------------------------------------------------------------
# 5.  Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, y_prob=None):
    """Return a dict of classification metrics.

    Includes accuracy, AUC-ROC, precision/recall/F1 for red-corner wins.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = np.nan
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1])
    metrics["precision_r"] = p[0]
    metrics["recall_r"] = r[0]
    metrics["f1_r"] = f1[0]
    return metrics


def print_comparison_table(results):
    """Print side-by-side comparison of multiple models."""
    print(f"\n{'='*70}")
    print(f"{'Model Comparison':^70}")
    print(f"{'='*70}")
    header = f"{'Model':<20} {'Accuracy':<10} {'AUC-ROC':<10} {'Prec(R)':<10} {'Recall(R)':<10} {'F1(R)':<10}"
    print(header)
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for label, m in results.items():
        print(f"{label:<20} {m['accuracy']:<10.4f} {m.get('auc_roc', 0):<10.4f} "
              f"{m['precision_r']:<10.4f} {m['recall_r']:<10.4f} {m['f1_r']:<10.4f}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# 6.  Full model comparison pipeline (train + evaluate all three models)
# ---------------------------------------------------------------------------

def run_model_comparison(X_train, X_val, y_train, y_val, label_suffix="",
                         return_probs=False):
    """Train XGBoost, RF, and LR on the same split and return their metrics.

    All three models now receive StandardScaled inputs so feature importances
    are scale-independent. XGBoost preserves NaN via scipy sparse; RF and LR
    additionally get imputation + missing-indicator columns.

    If return_probs=True, also returns a dict of probability arrays for
    multi-model comparison plotting.
    """
    from sklearn.preprocessing import StandardScaler

    # 1. Scale all features (preserves NaN — XGBoost handles it natively)
    scaler = StandardScaler()
    X_tr_scl_all = scaler.fit_transform(X_train)
    X_va_scl_all = scaler.transform(X_val)
    X_tr_scl_all = pd.DataFrame(X_tr_scl_all, columns=X_train.columns, index=X_train.index)
    X_va_scl_all = pd.DataFrame(X_va_scl_all, columns=X_val.columns, index=X_val.index)

    # XGBoost on scaled data (NaNs still present)
    xgb_model = train_xgb(X_tr_scl_all, y_train, X_va_scl_all, y_val)
    y_prob_xgb = xgb_model.predict_proba(X_va_scl_all)[:, 1]
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
    xgb_metrics = evaluate_model(y_val, y_pred_xgb, y_prob_xgb)

    # 2. Impute & scale for RF/LR (uses original unscaled data internally)
    X_tr_imp, X_va_imp, X_tr_scl, X_va_scl, _, _ = impute_and_scale(X_train, X_val)

    # Random Forest
    rf_model = train_rf(X_tr_imp, y_train)
    y_prob_rf = rf_model.predict_proba(X_va_imp)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    rf_metrics = evaluate_model(y_val, y_pred_rf, y_prob_rf)

    # Logistic Regression
    lr_model = train_lr(X_tr_scl, y_train)
    y_prob_lr = lr_model.predict_proba(X_va_scl)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    lr_metrics = evaluate_model(y_val, y_pred_lr, y_prob_lr)

    results = {
        f"XGBoost{label_suffix}": xgb_metrics,
        f"Random Forest{label_suffix}": rf_metrics,
        f"Logistic Reg{label_suffix}": lr_metrics,
    }
    if return_probs:
        return results, xgb_model, rf_model, lr_model, {
            "XGBoost": y_prob_xgb, "Random Forest": y_prob_rf, "Logistic Reg": y_prob_lr,
        }
    return results, xgb_model, rf_model, lr_model


# ---------------------------------------------------------------------------
# 7.  Time-series cross-validation (expanding window)
# ---------------------------------------------------------------------------

def time_series_cv(sym_df, feature_cols, n_splits=5, date_col="DATE"):
    """Walk-forward validation over chronological folds.

    Each fold uses an expanding training window so earlier data is always
    included. Returns per-fold metrics for mean ± std reporting — more
    reliable than a single temporal split.
    """
    all_dates = sorted(sym_df[date_col].unique())
    # Create n_splits cutoffs at evenly spaced percentiles past 20% of data
    start_idx = int(len(all_dates) * 0.2)
    split_indices = np.linspace(start_idx, len(all_dates) - 1, n_splits, dtype=int)

    fold_results = []
    xgb_models = []

    for fold, idx in enumerate(split_indices):
        cutoff = all_dates[idx]
        train = sym_df[sym_df[date_col] <= cutoff]
        val = sym_df[sym_df[date_col] > cutoff]

        if len(val) < 100:
            continue

        X_tr, y_tr = train[feature_cols], train["target"]
        X_va, y_va = val[feature_cols], val["target"]

        print(f"  Fold {fold+1}: cutoff={cutoff.date()}, train={len(train)}, val={len(val)}")

        model = train_xgb(X_tr, y_tr, X_va, y_va)
        y_prob = model.predict_proba(X_va)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_model(y_va, y_pred, y_prob)
        fold_results.append(metrics)
        xgb_models.append(model)

    if not fold_results:
        print("  No valid folds produced.")
        return None, []

    df = pd.DataFrame(fold_results)
    print(f"\n  CV Summary ({n_splits} folds, expanding window):")
    for col in df.columns:
        print(f"    {col:<15s}  mean={df[col].mean():.4f}  std={df[col].std():.4f}")
    return df, xgb_models


# ---------------------------------------------------------------------------
# 8.  Threshold tuning
# ---------------------------------------------------------------------------

def find_best_threshold(y_true, y_prob, model_label="", thresholds=None):
    """Scan decision thresholds to maximize accuracy.

    Default threshold of 0.50 is often suboptimal for imbalanced data;
    this function finds the optimal cutoff for a given model's probability
    distribution.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if thresholds is None:
        thresholds = [x / 100 for x in range(30, 71)]

    best_t, best_acc = 0.5, 0
    for t in thresholds:
        y_t = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_t)
        if acc > best_acc:
            best_t, best_acc = t, acc

    return best_t, best_acc


def print_threshold_summary(y_val_dict, y_prob_dict):
    """Print summary of optimal vs default threshold for each model/dataset."""
    print(f"\n{'='*70}")
    print(f"{'Threshold Tuning Summary':^70}")
    print(f"{'='*70}")
    h = f"{'Model':<25} {'Data':<10} {'Opt T':<8} {'Opt Acc':<10} {'0.55 Acc':<10} {'Gain':<10}"
    print(h)
    print(f"{'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for (label, y_true), (_, y_prob) in zip(y_val_dict.items(), y_prob_dict.items()):
        opt_t, opt_acc = find_best_threshold(y_true, y_prob, label)
        from sklearn.metrics import accuracy_score
        acc55 = accuracy_score(y_true, (y_prob >= 0.55).astype(int))
        print(f"{label:<25} {'':<10} {opt_t:<8.2f} {opt_acc:<10.4f} {acc55:<10.4f} {opt_acc-acc55:<+10.4f}")


# ---------------------------------------------------------------------------
# 9.  Plotting helpers (evaluation dashboard — two 3-panel figures)
# ---------------------------------------------------------------------------

def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return "ipykernel" in shell.__class__.__module__
    except (ImportError, AttributeError):
        return False


def _save_and_show(save_path, fig, label):
    """Save a figure if a path is given, then show it (inline in Jupyter)."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {label}")
    if _in_notebook():
        plt.show()
    else:
        fig.show()


def _qtext(ax, question):
    """Add an italic grey question subtitle just below the axes' title."""
    ax.text(0.5, 1.0, question, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, fontstyle="italic",
            color="#999999")


def _make_roc_axes(ax, y_val, y_prob, lime):
    """Draw ROC curve + random baseline on a single axes."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=lime, lw=1.5, alpha=0.7, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=11, fontweight="bold")
    _qtext(ax, "Can the model rank a true winner above a true loser?")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return roc_auc


def _make_pr_axes(ax, y_val, y_prob, lime):
    """Draw PR curve + baseline on a single axes."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(y=y_val.mean(), linestyle="--", color=lime, lw=1.5, alpha=0.7,
               label=f"Baseline ({y_val.mean():.2f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    _qtext(ax, "When the model predicts red, how reliable is that prediction?")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return ap


def _make_calibration_axes(ax, y_val, y_prob, lime):
    """Draw calibration curve + perfect line on a single axes."""
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", lw=2, label="XGBoost")
    ax.plot([0, 1], [0, 1], "--", color=lime, lw=1.5, alpha=0.7,
            label="Perfect calibration")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve", fontsize=11, fontweight="bold")
    _qtext(ax, "When the model says 70%, is it right 70% of the time?")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)


def _make_importance_axes(ax, feature_cols, model):
    """Draw grouped feature importance bar chart on a single axes."""
    imp_agg = _aggregate_importance(
        feature_cols[:len(model.feature_importances_)],
        model.feature_importances_
    ).head(15).sort_values("importance")
    colors_imp = ["#E63946" if r["n"] > 1 else "#457B9D"
                  for _, r in imp_agg.iterrows()]
    ax.barh(imp_agg["feature"], imp_agg["importance"], color=colors_imp)
    ax.set_xlabel("Grouped Importance")
    ax.set_title("Top 15 Features (grouped, red=multi-feat)", fontsize=11, fontweight="bold")
    _qtext(ax, "What actually matters for prediction?")
    ax.grid(alpha=0.3, axis="x")


def _make_shap_axes(ax, X_val, feature_cols, model):
    """Draw SHAP beeswarm onto a given axes object."""
    try:
        import shap
        import xgboost as xgb
        X_sample = X_val.iloc[:200]
        booster = model.get_booster()
        contribs = booster.predict(xgb.DMatrix(X_sample), pred_contribs=True)
        shap_values = contribs[:, :-1]
        readable_names = [FEATURE_LABELS.get(c, c) for c in feature_cols]
        explanation = shap.Explanation(shap_values, data=X_sample.values,
                                       feature_names=readable_names)
        shap.plots.beeswarm(explanation, max_display=15, ax=ax, show=False,
                            plot_size=None)
        ax.tick_params(colors="white")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("white")
        if hasattr(ax, "xaxis") and ax.xaxis.label is not None:
            ax.xaxis.label.set_color("white")
        if hasattr(ax, "yaxis") and ax.yaxis.label is not None:
            ax.yaxis.label.set_color("white")
        ax.set_title("SHAP Feature Impact (200-sample validation)", fontsize=10, fontweight="bold")
        _qtext(ax, "How does each feature push a prediction toward red or blue?")
    except Exception as e:
        print(f"  [SHAP] failed ({e}).")
        from xgboost import plot_importance
        plot_importance(model.get_booster(), max_num_features=15, ax=ax,
                        importance_type="weight")
        ax.set_title("XGBoost Importance (SHAP unavailable)")
    return ax


def plot_evaluation_dashboard(y_val, y_prob, y_pred, X_val, feature_cols, model,
                              save_path=None, title_suffix=""):
    """Produce a single 6-panel diagnostic figure for the XGBoost model.

    2x3 grid:
      Row 0: confusion matrix, ROC curve, PR curve.
      Row 1: calibration curve, grouped feature importance, SHAP beeswarm.

    Each subplot has a question subtitle for presentation clarity.
    """
    from sklearn.metrics import confusion_matrix

    lime = "limegreen"
    suffix = f" {title_suffix}" if title_suffix else ""
    base = f"XGBoost — Evaluation Diagnostics{suffix}"

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(base, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(pad=5.0)

    # ---- Row 0, Col 0: Confusion Matrix ----
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred B", "Pred R"],
                yticklabels=["True B", "True R"], ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    _qtext(axes[0, 0], "How many fights did the model get right?")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    # ---- Row 0, Col 1: ROC ----
    _make_roc_axes(axes[0, 1], y_val, y_prob, lime)

    # ---- Row 0, Col 2: PR ----
    _make_pr_axes(axes[0, 2], y_val, y_prob, lime)

    # ---- Row 1, Col 0: Calibration ----
    _make_calibration_axes(axes[1, 0], y_val, y_prob, lime)

    # ---- Row 1, Col 1: Feature Importance ----
    _make_importance_axes(axes[1, 1], feature_cols, model)

    # ---- Row 1, Col 2: SHAP ----
    _make_shap_axes(axes[1, 2], X_val, feature_cols, model)

    _save_and_show(save_path, fig, base)
    plt.close(fig)
    print(f"  Evaluation dashboard rendered ({suffix}).")


def _aggregate_importance(feature_names, raw_importances):
    """Group feature importances by FEATURE_GROUPS, sum within each group.

    Returns a DataFrame with 'feature' (group name or individual name)
    and 'importance' columns, sorted by importance descending.
    """
    remaining = set(feature_names)
    groups = {}
    for group_name, members in FEATURE_GROUPS.items():
        matched = [f for f in members if f in remaining]
        if matched:
            groups[group_name] = matched
            remaining -= set(matched)

    imp_map = dict(zip(feature_names, raw_importances))
    rows = []
    for group_name, members in groups.items():
        total = sum(imp_map.get(m, 0) for m in members)
        rows.append({"feature": group_name, "importance": total, "n": len(members)})
    for feat in sorted(remaining):
        rows.append({"feature": FEATURE_LABELS.get(feat, feat), "importance": imp_map.get(feat, 0), "n": 1})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def _make_model_roc_axes(ax, y_val, probs, subset_label):
    """Plot overlaid ROC curves for all models on a single axes."""
    from sklearn.metrics import roc_curve, auc
    colors = {"XGBoost": "#E63946", "Random Forest": "#457B9D",
              "Logistic Reg": "#2A9D8F", "ELO": "#E9C46A"}
    styles = {"XGBoost": "-", "Random Forest": "--",
              "Logistic Reg": ":", "ELO": "-."}
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_val, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, "gray"),
                linestyle=styles.get(name, "-"),
                lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC \u2014 {subset_label}", fontsize=11, fontweight="bold")
    _qtext(ax, "Which model best separates winners from losers?")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)


def _make_model_pr_axes(ax, y_val, probs, subset_label):
    """Plot overlaid PR curves for all models on a single axes."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    colors = {"XGBoost": "#E63946", "Random Forest": "#457B9D",
              "Logistic Reg": "#2A9D8F", "ELO": "#E9C46A"}
    styles = {"XGBoost": "-", "Random Forest": "--",
              "Logistic Reg": ":", "ELO": "-."}
    for name, prob in probs.items():
        precision, recall, _ = precision_recall_curve(y_val, prob)
        ap = average_precision_score(y_val, prob)
        ax.plot(recall, precision, color=colors.get(name, "gray"),
                linestyle=styles.get(name, "-"),
                lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=y_val.mean(), color="gray", linestyle="--", lw=1, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR \u2014 {subset_label}", fontsize=11, fontweight="bold")
    _qtext(ax, "Do confident predictions stay reliable across models?")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)


def _make_model_bar_axes(ax, y_val, probs, subset_label):
    """Plot side-by-side accuracy/AUC bars for all models on a single axes."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    labels = list(probs.keys())
    accs = []
    aucs = []
    for name in labels:
        pred = (probs[name] >= 0.5).astype(int)
        accs.append(accuracy_score(y_val, pred))
        aucs.append(roc_auc_score(y_val, probs[name]))
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#457B9D")
    bars2 = ax.bar(x + w/2, aucs, w, label="AUC-ROC", color="#2A9D8F")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("Score")
    ax.set_title(f"Accuracy & AUC \u2014 {subset_label}", fontsize=11, fontweight="bold")
    _qtext(ax, "Which model achieves the highest scores?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)


def plot_model_comparison(y_val_full, probs_full, elo_prob_full,
                          y_val_2005, probs_2005, elo_prob_2005,
                          save_path=None):
    """Single 2x3 figure comparing 3 models + ELO on full and 2005+ subsets.

    Row 0: Full dataset — ROC, PR, Accuracy/AUC bars.
    Row 1: 2005+ subset — ROC, PR, Accuracy/AUC bars.
    """
    subset_a = "Full Dataset"
    subset_b = "2005+ Subset"

    all_probs_a = dict(probs_full)
    if elo_prob_full is not None:
        all_probs_a["ELO"] = elo_prob_full

    all_probs_b = dict(probs_2005)
    if elo_prob_2005 is not None:
        all_probs_b["ELO"] = elo_prob_2005

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(pad=5.0)

    # Row 0: Full dataset
    _make_model_roc_axes(axes[0, 0], y_val_full, all_probs_a, subset_a)
    _make_model_pr_axes(axes[0, 1], y_val_full, all_probs_a, subset_a)
    _make_model_bar_axes(axes[0, 2], y_val_full, probs_full, subset_a)

    # Row 1: 2005+ subset
    _make_model_roc_axes(axes[1, 0], y_val_2005, all_probs_b, subset_b)
    _make_model_pr_axes(axes[1, 1], y_val_2005, all_probs_b, subset_b)
    _make_model_bar_axes(axes[1, 2], y_val_2005, probs_2005, subset_b)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.show()
    plt.close(fig)
    print("  Model comparison plot saved.")
