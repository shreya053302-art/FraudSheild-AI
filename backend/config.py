"""Central configuration: paths, limits and environment variables."""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional; the app still runs without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("FRAUDSHIELD_DB", str(DATA_DIR / "fraudshield.db")))
MODEL_PATH = Path(os.getenv("FRAUDSHIELD_MODEL", str(MODELS_DIR / "fraud_model.joblib")))
DEFAULT_DATASET = DATA_DIR / "creditcard.csv"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "300"))
MAX_ROWS = 2_000_000
MAX_COLUMNS = 500

# Bulk runs keep only the riskiest rows in the database (all rows are still
# scored and counted in the batch totals). This keeps the SQLite file small.
MAX_STORED_ROWS_PER_BATCH = int(os.getenv("MAX_STORED_ROWS_PER_BATCH", "5000"))
STORE_MIN_RISK = 25.0  # rows at or above this risk score are eligible for storage

FLAG_RISK = 50.0  # risk score at/above which a transaction is called fraudulent
RISK_BANDS = [(25.0, "Low"), (50.0, "Medium"), (75.0, "High"), (101.0, "Critical")]

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def gemini_key() -> str | None:
    """Read the API key at call time so it is never stored in code or the DB."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None
