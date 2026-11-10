"""Generate a SYNTHETIC transaction file for testing the app's upload and training flow.

IMPORTANT: this is artificial data in the same column layout as the Kaggle credit card
dataset (Time, V1-V28, Amount, Class). It is only for checking that the app works.
Any metrics you get from it describe this fake data, NOT real-world fraud detection.

Usage:  python scripts/make_sample_data.py [rows] [fraud_rate]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(n_rows: int = 20000, fraud_rate: float = 0.02, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 30)
    n_legit = n_rows - n_fraud
    legit = rng.normal(0, 1, size=(n_legit, 28))
    fraud = rng.normal(0, 1.4, size=(n_fraud, 28))
    for col, shift in {13: -4.0, 11: -3.0, 9: -2.5, 3: 2.5, 16: -2.0, 2: -2.0}.items():  # V14, V12, V10, V4, V17, V3
        fraud[:, col] += shift
    X = np.vstack([legit, fraud])
    y = np.r_[np.zeros(n_legit, int), np.ones(n_fraud, int)]
    amount = np.r_[rng.lognormal(3.3, 1.2, n_legit), rng.lognormal(4.2, 1.6, n_fraud)].round(2)
    time = np.r_[rng.uniform(0, 172800, n_legit), rng.uniform(0, 172800, n_fraud)].round(0)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = y
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    out = Path(__file__).resolve().parent.parent / "data" / "sample_synthetic_transactions.csv"
    make_synthetic(rows, rate).to_csv(out, index=False)
    print(f"Wrote SYNTHETIC data ({rows:,} rows) to {out}")
