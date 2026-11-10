"""Loading the trained model, scoring transactions and explaining the result."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import database as db
from .config import FLAG_RISK, MAX_STORED_ROWS_PER_BATCH, MODEL_PATH, STORE_MIN_RISK
from .fraud_detection import REQUIRED_KEYS, evaluate
from .preprocessing import coerce_features

LEVELS = ["Low", "Medium", "High", "Critical"]

RECOMMENDATIONS = {
    "Low": ["Approve the transaction; no action needed.",
            "Keep normal background monitoring on the account."],
    "Medium": ["Approve, but add the account to a short-term watch list.",
               "Consider step-up verification (OTP or app confirmation) if the customer's behaviour looks unusual.",
               "Re-check the account if more medium-risk transactions follow."],
    "High": ["Hold the transaction and send it to manual review.",
             "Contact the customer through a verified channel (not details from the transaction).",
             "Check recent device, location and merchant history for the card."],
    "Critical": ["Block or hold the transaction immediately.",
                 "Contact the customer on a verified number and consider freezing the card.",
                 "Escalate to the fraud investigation team and preserve all related records."],
}


# ----------------------------------------------------------------- bundle
def load_bundle() -> dict | None:
    """Load the saved model. Returns None if missing or unreadable (never fake a model)."""
    if not MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception:
        return None
    if not isinstance(bundle, dict) or not REQUIRED_KEYS <= set(bundle):
        return None
    return bundle


# ---------------------------------------------------------------- scoring
def score_frame(bundle: dict, X: pd.DataFrame, chunk: int = 20000, progress=None) -> np.ndarray:
    out = []
    for start in range(0, len(X), chunk):
        out.append(bundle["model"].predict_proba(X.iloc[start:start + chunk])[:, 1])
        if progress:
            progress(min(1.0, (start + chunk) / len(X)))
    return np.concatenate(out) if out else np.array([])


def risk_scores(scores, threshold: float) -> np.ndarray:
    """Map raw model scores to 0-100 so the decision threshold sits exactly at 50.
    This is a monotonic rescaling for readability, not a calibrated probability."""
    s = np.asarray(scores, float)
    t = float(np.clip(threshold, 1e-3, 1 - 1e-3))
    r = np.where(s < t, 50 * s / t, 50 + 50 * (s - t) / (1 - t))
    return np.clip(r, 0, 100)


def risk_levels(risk) -> np.ndarray:
    r = np.asarray(risk, float)
    return np.select([r < 25, r < 50, r < 75], ["Low", "Medium", "High"], default="Critical")


def confidence(scores, threshold: float) -> np.ndarray:
    """Decision confidence: 0.5 right at the threshold, 1.0 at the extremes.
    This measures distance from the decision boundary. It is NOT a calibrated probability."""
    s = np.asarray(scores, float)
    t = float(np.clip(threshold, 1e-3, 1 - 1e-3))
    return np.where(s >= t, 0.5 + 0.5 * (s - t) / (1 - t), 0.5 + 0.5 * (t - s) / t)


# ------------------------------------------------------------ explanation
def explain_rows(bundle: dict, X: pd.DataFrame, k: int = 3) -> tuple[list[str], list[list[dict]]]:
    """Template-based explanations. A feature is 'suspicious' when it sits far from the
    typical legitimate value (z-score) AND the model relies on it (importance)."""
    cols = bundle["feature_cols"]
    mean, std = np.array(bundle["ref_mean"]), np.array(bundle["ref_std"])
    imp = np.array(bundle["importance_scaled"])
    vals = X[cols].to_numpy(float)
    vals = np.where(np.isnan(vals), mean, vals)
    z = np.clip((vals - mean) / std, -15, 15)
    contrib = np.abs(z) * imp
    top = np.argsort(-contrib, axis=1)[:, :k]
    aq = bundle.get("amount_q")
    texts, all_drivers = [], []
    for i in range(len(X)):
        drivers, lines = [], []
        for j in top[i]:
            if abs(z[i, j]) < 2:
                continue
            direction = "above" if z[i, j] > 0 else "below"
            drivers.append({"feature": cols[j], "value": float(vals[i, j]), "z": float(z[i, j]),
                            "direction": direction})
            lines.append(f"{cols[j]} = {vals[i, j]:.3g} is {abs(z[i, j]):.1f} standard deviations "
                         f"{direction} the typical legitimate value (important feature for this model).")
        if aq and "Amount" in cols:
            amt = float(vals[i, cols.index("Amount")])
            if amt > aq["p99"]:
                lines.append(f"Amount {amt:,.2f} is above the 99th percentile of legitimate training "
                             f"transactions ({aq['p99']:,.2f}).")
                drivers.append({"feature": "Amount", "value": amt, "z": 0.0, "direction": "above p99"})
        if not lines:
            lines.append("No single feature is far outside the normal range; the score comes from a "
                         "combination of smaller deviations.")
        texts.append(" ".join(lines))
        all_drivers.append(drivers)
    return texts, all_drivers


# ------------------------------------------------------------ main entry
def run_analysis(bundle: dict, df: pd.DataFrame, source: str, name: str,
                 target_col: str | None = None, progress=None) -> dict:
    """Score a dataframe, store the batch and the riskiest rows, return results."""
    cols = bundle["feature_cols"]
    thr = bundle["threshold"]
    X = coerce_features(df, cols)
    scores = score_frame(bundle, X, progress=progress)
    risk = risk_scores(scores, thr)
    level = risk_levels(risk)
    flagged = risk >= FLAG_RISK
    conf = confidence(scores, thr)

    out = df.copy()
    out["fraud_score"] = np.round(scores, 6)
    out["risk_score"] = np.round(risk, 2)
    out["risk_level"] = level
    out["prediction"] = np.where(flagged, "Fraudulent", "Legitimate")
    out["confidence"] = np.round(conf, 4)

    amount = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).to_numpy() if "Amount" in df else np.zeros(len(df))
    time_sec = pd.to_numeric(df["Time"], errors="coerce").to_numpy() if "Time" in df else None

    hourly = None
    if time_sec is not None:
        hour = ((np.nan_to_num(time_sec) % 86400) // 3600).astype(int)
        hourly = {"total": np.bincount(hour, minlength=24).tolist(),
                  "flagged": np.bincount(hour[flagged], minlength=24).tolist()}
    risk_hist = np.histogram(risk, bins=10, range=(0, 100))[0].tolist()

    label_metrics, actual = None, None
    if target_col and target_col in df:
        y = pd.to_numeric(df[target_col], errors="coerce")
        if y.notna().all() and set(y.unique()) <= {0, 1}:
            actual = y.astype(int).to_numpy()
            label_metrics = evaluate(actual, scores, thr)

    # choose which rows to store: everything for manual entry, else the riskiest rows
    if source == "manual":
        idx = np.arange(len(df))
    else:
        eligible = np.where(risk >= STORE_MIN_RISK)[0]
        idx = eligible[np.argsort(-risk[eligible])][:MAX_STORED_ROWS_PER_BATCH]
    texts, drivers = explain_rows(bundle, X.iloc[idx]) if len(idx) else ([], [])

    batch_id = db.create_batch(
        source, name, len(df), int(flagged.sum()), float(amount.sum()), float(amount[flagged].sum()),
        label_metrics is not None, label_metrics, hourly, risk_hist, bundle["model_name"])
    rows = []
    Xv = X.to_numpy(float)
    for n, i in enumerate(idx):
        rows.append({
            "row_index": int(i) + 1, "amount": float(amount[i]),
            "time_sec": None if time_sec is None or np.isnan(time_sec[i]) else float(time_sec[i]),
            "features": {c: (None if np.isnan(Xv[i, j]) else float(Xv[i, j])) for j, c in enumerate(cols)},
            "prediction": str(out["prediction"].iloc[i]), "risk_score": float(risk[i]),
            "risk_level": str(level[i]), "confidence": float(conf[i]),
            "explanation": texts[n], "drivers": drivers[n],
            "actual_label": None if actual is None else int(actual[i]),
        })
    db.insert_transactions(batch_id, rows)

    first_id = None
    if source == "manual":
        with db.connect() as conn:
            first_id = conn.execute("SELECT MAX(id) FROM transactions WHERE batch_id = ?", (batch_id,)).fetchone()[0]

    return {
        "batch_id": batch_id, "transaction_id": first_id, "result": out,
        "n_total": len(df), "n_flagged": int(flagged.sum()),
        "amount_flagged": float(amount[flagged].sum()), "label_metrics": label_metrics,
        "stored_rows": len(rows), "explanations": texts, "drivers": drivers,
        "levels": {lv: int((level == lv).sum()) for lv in LEVELS},
    }
