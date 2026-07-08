"""
Fast baseline: extract classical features, train RandomForest + XGBoost.
Uses a chronological split (NOT random shuffle) to avoid leakage between
adjacent, highly-correlated EEG windows.

Usage:
    python train_baseline.py --data_dir processed/chb01 --sfreq 256
"""
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from features import extract_features


def chronological_split(X, y, test_frac=0.2):
    n = len(y)
    split = int(n * (1 - test_frac))
    return X[:split], X[split:], y[:split], y[split:]


def main(args):
    X = np.load(f"{args.data_dir}/X.npy")
    y = np.load(f"{args.data_dir}/y.npy")
    print(f"Loaded {len(y)} windows. Extracting features...")

    feats = extract_features(X, args.sfreq)
    X_train, X_test, y_train, y_test = chronological_split(feats, y)

    print(f"Train: {len(y_train)} (preictal={y_train.sum()}) | Test: {len(y_test)} (preictal={y_test.sum()})")

    # class weighting since interictal >> preictal
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, rf.predict(X_test)))
    print("AUC:", roc_auc_score(y_test, rf_probs))

    print("\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, xgb.predict(X_test)))
    print("AUC:", roc_auc_score(y_test, xgb_probs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sfreq", type=float, default=256.0)
    args = parser.parse_args()
    main(args)
