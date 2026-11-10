"""Aggregations for the dashboard and analytics pages. Everything comes from the database."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from . import database as db


def overall_risk_label(rate: float) -> str:
    """Portfolio-level label based on the share of flagged transactions."""
    if rate < 0.01:
        return "Low"
    if rate < 0.05:
        return "Medium"
    if rate < 0.15:
        return "High"
    return "Critical"


def get_kpis() -> dict:
    k = db.kpi_totals()
    total, flagged = k["n_total"], k["n_flagged"]
    rate = flagged / total if total else 0.0
    return {**k, "n_legit": total - flagged, "flag_rate": rate, "risk_label": overall_risk_label(rate) if total else "—"}


def trend(freq: str = "D") -> pd.DataFrame:
    """freq: 'D' daily, 'W' weekly, 'M' monthly (by detection date)."""
    b = db.list_batches(limit=10000)
    if b.empty:
        return pd.DataFrame(columns=["period", "analyzed", "flagged", "legitimate", "flag_rate", "amount_flagged"])
    b["created_at"] = pd.to_datetime(b["created_at"])
    alias = {"D": "D", "W": "W-MON", "M": "MS"}[freq]
    g = (b.set_index("created_at")
          .resample(alias)[["n_total", "n_flagged", "amount_flagged"]].sum().reset_index())
    g.columns = ["period", "analyzed", "flagged", "amount_flagged"]
    g["legitimate"] = g["analyzed"] - g["flagged"]
    g["flag_rate"] = np.where(g["analyzed"] > 0, g["flagged"] / g["analyzed"].replace(0, np.nan) * 100, 0)
    return g[g["analyzed"] > 0].reset_index(drop=True)


def hourly_activity() -> pd.DataFrame:
    b = db.list_batches(limit=10000)
    total, flagged = np.zeros(24), np.zeros(24)
    for h in b["hourly"].dropna():
        d = json.loads(h)
        total += np.array(d["total"])
        flagged += np.array(d["flagged"])
    return pd.DataFrame({"hour": range(24), "transactions": total.astype(int), "flagged": flagged.astype(int)})


def risk_distribution() -> pd.DataFrame:
    b = db.list_batches(limit=10000)
    hist = np.zeros(10)
    for h in b["risk_hist"].dropna():
        hist += np.array(json.loads(h))
    labels = [f"{i * 10}-{i * 10 + 10}" for i in range(10)]
    return pd.DataFrame({"risk_range": labels, "transactions": hist.astype(int)})


def risk_heatmap(max_batches: int = 12) -> pd.DataFrame:
    """Rows = recent batches, columns = relative hour of day, values = flagged rate (%)."""
    b = db.list_batches(limit=max_batches).iloc[::-1]
    rows = {}
    for _, r in b.iterrows():
        if not r["hourly"]:
            continue
        d = json.loads(r["hourly"])
        t, f = np.array(d["total"], float), np.array(d["flagged"], float)
        rate = np.where(t > 0, f / np.where(t > 0, t, 1) * 100, np.nan)
        rows[f"#{r['id']} {str(r['name'])[:18]}"] = rate
    return pd.DataFrame(rows, index=range(24)).T if rows else pd.DataFrame()


def flagged_transactions(limit: int = 5000) -> pd.DataFrame:
    return db.list_transactions(min_risk=50, limit=limit)


def amount_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["amount_range", "flagged", "amount"])
    bins = [-0.01, 10, 100, 500, 1000, 5000, np.inf]
    labels = ["0-10", "10-100", "100-500", "500-1k", "1k-5k", "5k+"]
    c = pd.cut(df["amount"].fillna(0), bins=bins, labels=labels)
    g = df.groupby(c, observed=False)["amount"].agg(["count", "sum"]).reset_index()
    g.columns = ["amount_range", "flagged", "amount"]
    return g


def pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Which features most often drive a fraud flag (from stored explanations)."""
    counter: Counter = Counter()
    for d in df.get("drivers", []):
        try:
            for item in json.loads(d):
                counter[(item["feature"], item["direction"])] += 1
        except (TypeError, ValueError, KeyError):
            continue
    rows = [{"feature": f, "direction": d, "flagged_transactions": n} for (f, d), n in counter.most_common(12)]
    return pd.DataFrame(rows, columns=["feature", "direction", "flagged_transactions"])


def recent_alerts(limit: int = 6) -> pd.DataFrame:
    """Newest High/Critical transactions still waiting for review."""
    df = db.list_transactions(min_risk=50, statuses=["Open"], limit=500)
    if df.empty:
        return df
    return df.sort_values(["created_at", "risk_score"], ascending=False).head(limit)


def label_metrics_history() -> pd.DataFrame:
    """Metrics for batches that included true labels (e.g. the Class column)."""
    b = db.list_batches(limit=1000)
    rows = []
    for _, r in b.iterrows():
        if r["has_labels"] and r["label_metrics"]:
            m = json.loads(r["label_metrics"])
            rows.append({"batch": r["id"], "created_at": r["created_at"], "name": r["name"], "rows": r["n_total"],
                         "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "pr_auc": m.get("pr_auc")})
    return pd.DataFrame(rows)


def monitoring(bundle: dict | None) -> dict:
    """Compare the live flag rate with the fraud rate seen in the held-out test data."""
    k = get_kpis()
    if not bundle or not k["n_total"]:
        return {"available": False}
    expected = bundle.get("test_fraud_rate", 0)
    m = bundle["metrics"][bundle["selected"]]
    cm = m["confusion_matrix"]
    expected_flag = (cm[0][1] + cm[1][1]) / max(m["n"], 1)
    ratio = k["flag_rate"] / expected_flag if expected_flag else None
    status = "Stable"
    if ratio is not None and (ratio > 3 or ratio < 0.33):
        status = "Review: flag rate differs a lot from test data"
    return {"available": True, "live_rate": k["flag_rate"], "expected_rate": expected_flag,
            "true_rate_test": expected, "ratio": ratio, "status": status}
