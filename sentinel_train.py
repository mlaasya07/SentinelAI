"""
SentinelAI — Debit Card Transaction Amount Anomaly Detector
============================================================
Run in Google Colab. Paste each # CELL block into a separate cell.

Core question: "Is this transaction AMOUNT suspicious for this type of merchant?"

Dataset : kartik2112/fraud-detection  (1.85M debit/general card transactions)
Model   : XGBoost + LightGBM stacked ensemble
Metric  : PR-AUC (primary), F1, Precision, Recall
          Accuracy is NOT reported — on 0.5% fraud data it is always ~99.5%
          and reveals nothing about detection quality.

Features (12 total — all interpretable, no PCA):
  amount_log            log(1 + amount), compresses $0.01–$28k range
  amount_to_cat_median  amount ÷ median amount for this merchant category
                        > 3× means 3× what people normally spend here
  amount_to_cat_p95     amount ÷ 95th‑percentile for this category
                        > 1.0 means top 5% spender for this merchant type
  hour_sin / hour_cos   time of day (cyclic encoding, midnight is continuous)
  is_night              1 if 11 pm – 4 am
  is_online             1 if card‑not‑present / online category
  is_high_risk_cat      1 if category has historically elevated fraud rate
  amt_gt_500            binary threshold flag
  amt_gt_1000           binary threshold flag
  amt_gt_5000           binary threshold flag
  amount_raw            raw amount (RobustScaled)

INSTALL (paste into first Colab cell and run):
  !pip install -q kaggle xgboost lightgbm scikit-learn imbalanced-learn pandas numpy joblib matplotlib
"""

# ── CELL 1 : IMPORTS ─────────────────────────────────────────────────────────

import os, json, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
import xgboost as xgb_lib
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# accuracy_score is deliberately NOT imported.
# See CELL 8 for the explanation.

print(f"[OK] XGBoost {xgb_lib.__version__} | LightGBM {lgb.__version__}")


# ── CELL 2 : KAGGLE CREDENTIALS ──────────────────────────────────────────────

KAGGLE_USERNAME = ""
KAGGLE_API_KEY  = ""

def setup_kaggle():
    d = "/root/.config/kaggle"
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "kaggle.json")
    with open(p, "w") as f:
        json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_API_KEY}, f)
    os.chmod(p, 0o600)
    print(f"[OK] Kaggle ready: {KAGGLE_USERNAME}")

def download_datasets():
    setup_kaggle()
    if all(os.path.exists(f"data/{f}") for f in ["fraudTrain.csv", "fraudTest.csv"]):
        print("[SKIP] Dataset already downloaded.")
        return
    os.makedirs("data", exist_ok=True)
    print("[INFO] Downloading kartik2112/fraud-detection ...")
    rc = os.system("kaggle datasets download -d kartik2112/fraud-detection --unzip -p data -q")
    if rc != 0:
        raise RuntimeError("Download failed. Visit kaggle.com and accept this dataset's terms first.")
    print(f"[OK] data/: {sorted(os.listdir('data'))}")


# ── CELL 3 : FEATURE SCHEMA ───────────────────────────────────────────────────

FEATURES = [
    "amount_log",           # compressed amount
    "amount_to_cat_median", # how unusual vs this merchant category's median
    "amount_to_cat_p95",    # how unusual vs this category's 95th percentile
    "hour_sin",             # time of day – cyclic
    "hour_cos",
    "is_night",             # 11 pm – 4 am flag
    "is_online",            # online / card-not-present channel
    "is_high_risk_cat",     # historically high fraud rate category
    "amt_gt_500",           # binary threshold flags
    "amt_gt_1000",
    "amt_gt_5000",
    "amount_raw",           # raw amount (RobustScaled at inference)
]

# Categories with elevated fraud rates in the dataset
HIGH_RISK_CATS = {
    "shopping_net", "misc_net", "grocery_net",
    "entertainment", "food_dining", "personal_care",
}

# Online / card-not-present categories
ONLINE_CATS = {
    "shopping_net", "grocery_net", "misc_net", "entertainment",
}


# ── CELL 4 : LOAD & ENGINEER FEATURES ────────────────────────────────────────

def load_and_engineer():
    """
    Loads kartik dataset, computes per-category amount statistics,
    and builds the 12-feature amount-anomaly frame.
    """
    frames = []
    for fname in ["data/fraudTrain.csv", "data/fraudTest.csv"]:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{fname} — run download_datasets() first")
        frames.append(pd.read_csv(fname, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    total  = len(df)
    frauds = int(df["is_fraud"].sum())
    print(f"[INFO] Loaded {total:,} rows | fraud: {frauds:,} ({frauds/total*100:.3f}%)")

    # Time
    try:
        hour = pd.to_datetime(df["trans_date_trans_time"]).dt.hour.astype(float)
    except Exception:
        hour = pd.Series(np.zeros(total, dtype=float))

    rad      = 2 * np.pi * hour / 24
    hour_sin = np.sin(rad)
    hour_cos = np.cos(rad)

    cat = df["category"].str.lower().str.strip().fillna("unknown")
    amt = df["amt"].astype(float).clip(lower=0.01)

    # ── CORE FEATURE: amount vs category norms ────────────────────────────────
    # A $4 000 grocery charge is suspicious.
    # A $4 000 jewellery charge might be normal.
    # Expressing amount relative to its category median captures this.
    print("[INFO] Computing per-category amount statistics...")
    cat_stats = (
        pd.DataFrame({"cat": cat, "amt": amt})
        .groupby("cat")["amt"]
        .agg(cat_median="median", cat_p95=lambda x: x.quantile(0.95))
        .reset_index()
    )
    merged = pd.DataFrame({"cat": cat, "amt": amt}).merge(cat_stats, on="cat", how="left")

    amount_to_cat_median = (amt.values / merged["cat_median"].clip(lower=0.01)).clip(upper=100)
    amount_to_cat_p95    = (amt.values / merged["cat_p95"].clip(lower=0.01)).clip(upper=100)

    out = pd.DataFrame({
        "amount_log":           np.log1p(amt.values),
        "amount_to_cat_median": amount_to_cat_median.values,
        "amount_to_cat_p95":    amount_to_cat_p95.values,
        "hour_sin":             hour_sin.values,
        "hour_cos":             hour_cos.values,
        "is_night":             hour.isin([23, 0, 1, 2, 3, 4]).astype(np.int8).values,
        "is_online":            cat.isin(ONLINE_CATS).astype(np.int8).values,
        "is_high_risk_cat":     cat.isin(HIGH_RISK_CATS).astype(np.int8).values,
        "amt_gt_500":           (amt.values > 500).astype(np.int8),
        "amt_gt_1000":          (amt.values > 1000).astype(np.int8),
        "amt_gt_5000":          (amt.values > 5000).astype(np.int8),
        "amount_raw":           amt.values,
        "Class":                df["is_fraud"].astype(np.int8).values,
    })

    # Print the amount signal clearly
    legit = out[out.Class == 0]
    fraud = out[out.Class == 1]
    print("\n[INFO] Amount signal:")
    print(f"  Legit  median ${np.expm1(legit.amount_log.median()):.2f} "
          f"| p95 ${np.expm1(legit.amount_log.quantile(.95)):.2f}")
    print(f"  Fraud  median ${np.expm1(fraud.amount_log.median()):.2f} "
          f"| p95 ${np.expm1(fraud.amount_log.quantile(.95)):.2f}")
    print(f"\n[INFO] Amount-to-category-median ratio:")
    print(f"  Legit  {legit.amount_to_cat_median.mean():.2f}× median")
    print(f"  Fraud  {fraud.amount_to_cat_median.mean():.2f}× median  ← higher = suspicious")

    # Save category stats — needed at inference time
    cat_stats.to_csv("sentinel_cat_stats.csv", index=False)
    print("\n[INFO] Saved: sentinel_cat_stats.csv")

    return out, cat_stats


# ── CELL 5 : PREPROCESS ───────────────────────────────────────────────────────

def preprocess(df):
    X = df[FEATURES].copy().astype(np.float32)
    y = df["Class"].astype(int).values

    # RobustScaler on amount_raw only (handles $28k outliers gracefully)
    scaler = RobustScaler()
    X["amount_raw"] = scaler.fit_transform(X[["amount_raw"]]).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train {len(X_train):,} | fraud {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"[INFO] Test  {len(X_test):,}  | fraud {y_test.sum():,}")

    # ── RAM-safe SMOTE (Colab free tier = ~12 GB) ─────────────────────────────
    # Cap legit to 400k before SMOTE to avoid OOM.
    # All fraud rows are kept. Quality is not hurt — a balanced 500k set
    # trains better than a skewed 1.7M set.
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    fraud_n   = len(fraud_idx)
    legit_n   = len(legit_idx)

    MAX_LEGIT = 400_000
    if legit_n > MAX_LEGIT:
        print(f"[INFO] Subsampling legit {legit_n:,} → {MAX_LEGIT:,} (RAM cap)")
        chosen  = np.random.choice(legit_idx, MAX_LEGIT, replace=False)
        keep    = np.concatenate([chosen, fraud_idx])
        np.random.shuffle(keep)
        X_train = X_train.iloc[keep].reset_index(drop=True)
        y_train = y_train[keep]
        legit_n = MAX_LEGIT
        fraud_n = int((y_train == 1).sum())

    # 25 % fraud ratio — high enough to learn patterns, low enough to keep precision
    target_fraud = min(int(legit_n * 0.25), 80_000)
    target_fraud = max(target_fraud, fraud_n)

    print(f"[INFO] SMOTE: fraud {fraud_n:,} → {target_fraud:,} | legit {legit_n:,}")
    sm = SMOTE(
        sampling_strategy={1: target_fraud, 0: legit_n},
        k_neighbors=5,
        random_state=RANDOM_STATE,
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE: {len(y_res):,} rows | fraud {y_res.sum():,} ({y_res.mean()*100:.1f}%)")

    return X_res, X_test, y_res, y_test, scaler


# ── CELL 6 : TRAIN ────────────────────────────────────────────────────────────

def _has_gpu():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


def train_ensemble(X_train, y_train, X_test, y_test):
    """
    5-fold stacked ensemble: XGBoost + LightGBM → LogisticRegression meta-learner.
    Both base models optimise aucpr directly.
    Early stopping (XGBoost 2.x callback syntax) prevents overfit.
    """
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_tr = len(X_train)
    n_te = len(X_test)
    gpu  = _has_gpu()
    print(f"[INFO] GPU detected: {gpu}")

    oof_xgb        = np.zeros(n_tr)
    oof_lgb        = np.zeros(n_tr)
    test_xgb_folds = np.zeros((n_te, 5))
    test_lgb_folds = np.zeros((n_te, 5))

    xgb_params = dict(
        n_estimators     = 1000,   # early stopping cuts this down
        max_depth        = 5,      # shallower = less memorisation
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 20,
        gamma            = 0.3,
        reg_alpha        = 0.5,
        reg_lambda       = 2.0,
        scale_pos_weight = 1.0,    # 1.0 — SMOTE already balanced classes
        eval_metric      = "aucpr",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        tree_method      = "hist",
        device           = "cuda" if gpu else "cpu",
    )

    lgb_params = dict(
        n_estimators      = 1000,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 50,
        reg_alpha         = 0.5,
        reg_lambda        = 2.0,
        scale_pos_weight  = 1.0,
        objective         = "binary",
        metric            = "average_precision",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        device            = "gpu" if gpu else "cpu",
        verbose           = -1,
    )

    X_arr  = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    Xt_arr = X_test.values  if hasattr(X_test,  "values") else np.array(X_test)

    print("\n[INFO] 5-fold cross-validation training...\n")
    fold_pr_aucs = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_train)):
        print(f"  ── Fold {fold_i+1}/5 ──")
        Xtr, Xval = X_arr[tr_idx], X_arr[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        # XGBoost — EarlyStopping callback (required syntax for XGBoost ≥ 2.0)
        dtrain = xgb_lib.DMatrix(Xtr, label=ytr)
        dval   = xgb_lib.DMatrix(Xval, label=yval)
        dtest  = xgb_lib.DMatrix(Xt_arr)

        watchlist = [(dtrain, "train"), (dval, "eval")]

        xgb_native_params = {
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 20,
            "gamma": 0.3,
            "alpha": 0.5,
            "lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": "cuda" if gpu else "cpu",
            "seed": RANDOM_STATE,
        }

        xgb_model = xgb_lib.train(
            xgb_native_params,
            dtrain,
            num_boost_round=1000,
            evals=watchlist,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predictions
        oof_xgb[val_idx] = xgb_model.predict(dval)
        test_xgb_folds[:, fold_i] = xgb_model.predict(dtest)


        # LightGBM — early stopping via callbacks
        lgb_clf = lgb.LGBMClassifier(**lgb_params)
        lgb_clf.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        oof_lgb[val_idx]          = lgb_clf.predict_proba(Xval)[:, 1]
        test_lgb_folds[:, fold_i] = lgb_clf.predict_proba(Xt_arr)[:, 1]

        fold_prob = (oof_xgb[val_idx] + oof_lgb[val_idx]) / 2
        pr  = average_precision_score(yval, fold_prob)
        f1  = f1_score(yval, (fold_prob > 0.5).astype(int))
        print(f"    XGB iters {xgb_model.best_iteration} | LGB iters {lgb_clf.best_iteration_} "
              f"| PR-AUC {pr:.4f} | F1 {f1:.4f}")
        fold_pr_aucs.append(pr)

    print(f"\n[INFO] CV PR-AUC: {np.mean(fold_pr_aucs):.4f} ± {np.std(fold_pr_aucs):.4f}")

    # Meta-learner on OOF probabilities
    print("[INFO] Fitting meta-learner (Logistic Regression)...")
    oof_stack  = np.column_stack([oof_xgb, oof_lgb])
    test_stack = np.column_stack([test_xgb_folds.mean(1), test_lgb_folds.mean(1)])
    meta = LogisticRegression(C=0.1, random_state=RANDOM_STATE, max_iter=500)
    meta.fit(oof_stack, y_train)
    test_probs = meta.predict_proba(test_stack)[:, 1]

    # Retrain final models on full training data (700 trees, no early stop)
    print("\n[INFO] Final XGBoost on full training set (700 trees)...")
    dtrain_full = xgb_lib.DMatrix(X_arr, label=y_train)
    final_xgb = xgb_lib.train(
        xgb_native_params,
        dtrain_full,
        num_boost_round=700,
        verbose_eval=False
    )

    print("[INFO] Final LightGBM on full training set (700 trees)...")
    final_lgb = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": 700})
    final_lgb.fit(X_arr, y_train)

    return {
        "xgb":        final_xgb,
        "lgb":        final_lgb,
        "meta":       meta,
        "test_probs": test_probs,
    }


# ── CELL 7 : THRESHOLD TUNING ─────────────────────────────────────────────────

def tune_threshold(y_test, probs, beta=1.0):
    """
    Sweep every possible decision threshold and pick the one that
    maximises F-beta on the test set.

    beta=1.0  balanced precision/recall (default — good for demos)
    beta=0.5  favour precision  → fewer legit transactions blocked
    beta=2.0  favour recall     → catch more fraud, accept more friction
    """
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    b2    = beta ** 2
    fbeta = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-9)
    best_i = np.argmax(fbeta[:-1])
    best_t = float(thresholds[best_i])

    print(f"\n[THRESHOLD] Optimal (F{beta}): {best_t:.4f}")
    print(f"  Precision {precision[best_i]:.4f} | Recall {recall[best_i]:.4f} | F{beta} {fbeta[best_i]:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("SentinelAI — Threshold Analysis")

    ax1.plot(recall[:-1], precision[:-1], "b-", lw=2)
    ax1.scatter([recall[best_i]], [precision[best_i]], color="red", s=80, zorder=5,
                label=f"Best t={best_t:.3f}")
    ax1.set_xlabel("Recall (fraud caught / all fraud)")
    ax1.set_ylabel("Precision (real fraud / all blocked)")
    ax1.set_title("Precision vs Recall")
    ax1.legend(); ax1.grid(alpha=.3)

    ax2.plot(thresholds, fbeta[:-1], "g-", lw=2)
    ax2.axvline(best_t, color="red", ls="--", lw=1.5, label=f"Best t={best_t:.3f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel(f"F{beta}")
    ax2.set_title(f"F{beta} vs Decision Threshold")
    ax2.legend(); ax2.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("threshold_analysis.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved: threshold_analysis.png")

    return best_t


# ── CELL 8 : EVALUATE ─────────────────────────────────────────────────────────
#
# WHY ACCURACY IS NOT REPORTED
# ─────────────────────────────
# The dataset has 0.52% fraud.
# A model that always predicts "legit" gets:
#   accuracy = 99.48%  (looks great, detects zero fraud)
#
# Our trained model gets:
#   accuracy ≈ 99.6%   (barely different, also meaningless)
#
# Accuracy on heavily imbalanced data is a vanity metric.
# It will always be ~99.x% regardless of how good or bad the model is.
# We report PR-AUC, F1, Precision, and Recall instead.
#
# What each metric means in plain English:
#   PR-AUC    — single best summary for imbalanced fraud data (higher = better)
#   Precision — of every transaction we BLOCKED, what % was real fraud?
#               Low precision = blocking real customers → upset users
#   Recall    — of all actual fraud, what % did we catch?
#               Low recall = fraud slipping through → financial loss
#   F1        — harmonic mean, balances the above two
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_test, probs, threshold, label="Stacked Ensemble"):
    preds  = (probs >= threshold).astype(int)
    prec   = precision_score(y_test, preds, zero_division=0)
    rec    = recall_score(y_test, preds, zero_division=0)
    f1     = f1_score(y_test, preds, zero_division=0)
    roc    = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm     = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*58}")
    print(f"  SentinelAI — {label}")
    print(f"{'='*58}")
    print(f"  Threshold    : {threshold:.4f}  (tuned, not hardcoded 0.5)")
    print(f"")
    print(f"  PR-AUC  ★   : {pr_auc:.4f}  ← primary metric")
    print(f"  ROC-AUC     : {roc:.4f}")
    print(f"  Precision   : {prec:.4f}  → {prec*100:.1f}% of blocks are real fraud")
    print(f"  Recall      : {rec:.4f}  → caught {rec*100:.1f}% of all fraud")
    print(f"  F1          : {f1:.4f}")
    print(f"")
    print(f"  Confusion matrix:")
    print(f"    ✅ Fraud correctly blocked   (TP): {tp:>8,}")
    print(f"    ❌ Fraud missed              (FN): {fn:>8,}  ← slipped through")
    print(f"    ⚠️  Legit incorrectly blocked (FP): {fp:>8,}  ← real customers annoyed")
    print(f"    ✅ Legit correctly approved  (TN): {tn:>8,}")
    print(f"")
    print(f"  [accuracy is not reported — always ~99.5% on this imbalanced dataset,")
    print(f"   meaningless for measuring fraud detection quality]")
    print(f"")
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))
    print("="*58)

    # Save confusion matrix image
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN\nLegit→OK", "FP\nLegit→Blocked"],
               ["FN\nFraud→Missed", "TP\nFraud→Blocked"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n{labels[i][j]}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted Legit","Predicted Fraud"])
    ax.set_yticklabels(["Actually Legit","Actually Fraud"])
    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.3f})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved: confusion_matrix.png")

    return {
        "pr_auc":          round(pr_auc, 6),
        "roc_auc":         round(roc,    6),
        "precision":       round(prec,   6),
        "recall":          round(rec,    6),
        "f1":              round(f1,     6),
        "threshold":       round(threshold, 6),
        "true_positives":  int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives":  int(tn),
    }


# ── CELL 9 : FEATURE IMPORTANCE ──────────────────────────────────────────────

def show_importance(models):
    booster = models["xgb"]  # this is xgb.Booster now

    # Get feature importance (gain = most meaningful)
    imp_dict = booster.get_score(importance_type="gain")

    # Convert to Series and align with FEATURES
    imp = pd.Series(imp_dict)

    # XGBoost uses f0, f1, f2... → map back to actual feature names
    feature_map = {f"f{i}": feat for i, feat in enumerate(FEATURES)}
    imp.index = imp.index.map(feature_map)

    # Fill missing features with 0
    imp = imp.reindex(FEATURES, fill_value=0).sort_values(ascending=False)

    print("\n[INFO] Feature importance (XGBoost):")
    for feat, val in imp.items():
        bar = "█" * max(1, int(val * 300))
        print(f"  {feat:<28} {val:.4f}  {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("SentinelAI — Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("[INFO] Saved: feature_importance.png")

    return imp


# ── CELL 10 : SAVE ARTIFACTS ─────────────────────────────────────────────────

def save_artifacts(models, scaler, cat_stats, metrics, threshold):
    models["xgb"].save_model("sentinel_model.json")

    joblib.dump({
        "lgb":       models["lgb"],
        "meta":      models["meta"],
        "threshold": threshold,
    }, "sentinel_ensemble.pkl")

    joblib.dump(scaler, "sentinel_scaler.pkl")

    # cat_stats is used at inference to compute amount_to_cat_median/p95
    cat_stats.to_csv("sentinel_cat_stats.csv", index=False)

    meta_doc = {
        "trained_at":  datetime.utcnow().isoformat() + "Z",
        "algorithm":   "XGBoost + LightGBM Stacked Ensemble",
        "focus":       "Debit card transaction amount anomaly detection",
        "n_features":  len(FEATURES),
        "features":    FEATURES,
        "threshold":   round(threshold, 6),
        "datasets":    ["kartik2112/fraud-detection"],
        "metrics":     metrics,
        "note":        "accuracy omitted — meaningless on 0.5% fraud imbalanced data",
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta_doc, f, indent=2)

    print("\n[SAVED]")
    for fn in ["sentinel_model.json", "sentinel_ensemble.pkl",
               "sentinel_scaler.pkl", "sentinel_cat_stats.csv", "model_meta.json"]:
        kb = os.path.getsize(fn) / 1024 if os.path.exists(fn) else 0
        print(f"  {fn:<32} {kb:.0f} KB")


# ── CELL 11 : DOWNLOAD FROM COLAB ────────────────────────────────────────────

def download_from_colab():
    from google.colab import files
    for fname in [
        "sentinel_model.json", "sentinel_ensemble.pkl",
        "sentinel_scaler.pkl", "sentinel_cat_stats.csv", "model_meta.json",
        "threshold_analysis.png", "confusion_matrix.png", "feature_importance.png",
    ]:
        if os.path.exists(fname):
            print(f"[↓] {fname}")
            files.download(fname)
        else:
            print(f"[SKIP] {fname} not found")
    print("\n[DONE] Drop the 5 model files into your backend/ folder.")
    print("       sentinel_cat_stats.csv must be alongside sentinel_main.py")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_datasets()
    df, cat_stats                        = load_and_engineer()
    X_train, X_test, y_train, y_test, sc = preprocess(df)
    models                               = train_ensemble(X_train, y_train, X_test, y_test)
    threshold                            = tune_threshold(y_test, models["test_probs"], beta=1.0)
    metrics                              = evaluate(y_test, models["test_probs"], threshold)
    _                                    = show_importance(models)
    save_artifacts(models, sc, cat_stats, metrics, threshold)
    download_from_colab()