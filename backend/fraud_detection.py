"""Model training and evaluation.

Imbalance handling: class weights (class_weight='balanced'), NOT oversampling.
Class weights are applied only while fitting, so no synthetic rows can leak
from the training set into validation/test data.

Data split
----------
80% train_full / 20% test (stratified)
train_full -> 80% fit / 20% validation (stratified)
Validation set picks the decision threshold and the better model.
The test set is touched once, for the final honest metrics.
"""
from __future__ import annotations

import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .config import MODEL_PATH
from .preprocessing import (align_columns, build_pipeline, coerce_features, default_feature_cols,
                            detect_target, pipeline_feature_order)

BUNDLE_VERSION = 1
REQUIRED_KEYS = {"model", "feature_cols", "threshold", "metrics", "selected", "ref_mean", "ref_std", "importance"}


def best_f1_threshold(y_true, scores) -> float:
    p, r, t = precision_recall_curve(y_true, scores)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(np.clip(t[int(np.nanargmax(f1))], 0.01, 0.99))


def evaluate(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }
    if 0 < y_true.sum() < len(y_true):
        out["pr_auc"] = float(average_precision_score(y_true, scores))
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        out["pr_auc"] = out["roc_auc"] = None
    return out


def _feature_importance(pipe, feature_cols: list[str], kind: str) -> dict[str, float]:
    clf = pipe.named_steps["clf"]
    raw = clf.feature_importances_ if kind == "rf" else np.abs(clf.coef_[0])
    order = pipeline_feature_order(feature_cols)  # column order seen by the classifier
    imp = dict(zip(order, np.asarray(raw, float)))
    total = sum(imp.values()) or 1.0
    return {c: imp[c] / total for c in feature_cols}


def prepare_training_frame(df: pd.DataFrame, target: str | None = None, feature_cols: list[str] | None = None):
    """Validate a dataset for training. Returns (df, target, feature_cols, info)."""
    target = detect_target(df, target)
    if target is None:
        raise ValueError("Could not find a 0/1 target column. Rename it to 'Class' or pick it manually.")
    feature_cols = feature_cols or default_feature_cols(df, target)
    df = align_columns(df, feature_cols)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    y = pd.to_numeric(df[target], errors="coerce")
    keep = y.notna()
    info = {"rows_dropped_missing_target": int((~keep).sum()),
            "missing_values": int(df[feature_cols].isna().sum().sum())}
    df = df.loc[keep].copy()
    df[target] = y[keep].astype(int)
    n_fraud = int(df[target].sum())
    if n_fraud < 20 or (len(df) - n_fraud) < 20:
        raise ValueError(f"Need at least 20 rows of each class to train (found {n_fraud} fraud rows).")
    return df, target, feature_cols, info


def train_models(df: pd.DataFrame, target: str | None = None, feature_cols: list[str] | None = None,
                 n_estimators: int = 150, max_rows: int | None = None, random_state: int = 42,
                 dataset_name: str = "uploaded dataset", progress=None, save: bool = True) -> dict:
    """Train Random Forest and Logistic Regression, pick the better one, save a joblib bundle."""
    say = progress or (lambda msg: None)
    df, target, feature_cols, info = prepare_training_frame(df, target, feature_cols)

    if max_rows and len(df) > max_rows:  # keep class balance of the original data
        frac = max_rows / len(df)
        df = pd.concat([g.sample(frac=frac, random_state=random_state) for _, g in df.groupby(target)])
        say(f"Using a stratified sample of {len(df):,} rows.")

    X = coerce_features(df, feature_cols)
    y = df[target].astype(int)

    X_full, X_test, y_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full,
                                                random_state=random_state)
    say(f"Split: {len(X_tr):,} train / {len(X_val):,} validation / {len(X_test):,} test rows.")

    candidates = {
        "Random Forest": ("rf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=14, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=random_state)),
        "Logistic Regression": ("lr", LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=random_state)),
    }
    fitted, val_scores, results = {}, {}, {}
    for name, (kind, est) in candidates.items():
        say(f"Training {name} ...")
        t0 = time.time()
        pipe = build_pipeline(feature_cols, est)
        pipe.fit(X_tr, y_tr)
        fit_seconds = time.time() - t0
        s_val = pipe.predict_proba(X_val)[:, 1]
        thr = best_f1_threshold(y_val, s_val)
        s_test = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, s_test, thr)
        metrics["val_pr_auc"] = float(average_precision_score(y_val, s_val))
        metrics["fit_seconds"] = round(fit_seconds, 1)
        fitted[name], results[name] = (kind, pipe), metrics
        val_scores[name] = s_val
        say(f"{name}: test PR-AUC={metrics['pr_auc']:.3f}, recall={metrics['recall']:.3f}, "
            f"precision={metrics['precision']:.3f}")

    # Model choice uses VALIDATION PR-AUC only, so the test set stays untouched.
    selected = max(results, key=lambda n: results[n]["val_pr_auc"])
    kind, pipe = fitted[selected]
    thr = results[selected]["threshold"]

    s_test = pipe.predict_proba(X_test)[:, 1]
    p, r, _ = precision_recall_curve(y_test, s_test)
    idx = np.linspace(0, len(p) - 1, num=min(200, len(p))).astype(int)

    legit = X_tr[y_tr == 0]
    ref_mean = legit.mean().fillna(0.0)
    ref_std = legit.std().replace(0, 1.0).fillna(1.0)
    imp = _feature_importance(pipe, feature_cols, kind)
    imp_max = max(imp.values()) or 1.0

    amount_q = None
    if "Amount" in X_tr:
        a = legit["Amount"].dropna()
        amount_q = {"p50": float(a.quantile(.5)), "p95": float(a.quantile(.95)), "p99": float(a.quantile(.99))}

    samples = {}
    for label, name in ((1, "fraud"), (0, "legitimate")):
        rows = X_test[y_test == label].head(5)
        samples[name] = [{c: float(v) for c, v in row.fillna(0).items()} for _, row in rows.iterrows()]

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "model": pipe,
        "model_name": selected,
        "selected": selected,
        "feature_cols": feature_cols,
        "target": target,
        "threshold": float(thr),
        "metrics": results,
        "importance": {c: float(imp[c]) for c in feature_cols},
        "importance_scaled": [float(imp[c] / imp_max) for c in feature_cols],
        "ref_mean": [float(ref_mean[c]) for c in feature_cols],
        "ref_std": [float(ref_std[c]) for c in feature_cols],
        "defaults": {c: float(legit[c].median()) if legit[c].notna().any() else 0.0 for c in feature_cols},
        "amount_q": amount_q,
        "pr_curve": {"precision": p[idx].tolist(), "recall": r[idx].tolist()},
        "samples": samples,
        "dataset_name": dataset_name,
        "n_rows": int(len(df)),
        "n_train": int(len(X_tr)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "fraud_rate": float(y.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "data_info": info,
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)
        say(f"Saved model to {MODEL_PATH.name}.")
    return bundle
