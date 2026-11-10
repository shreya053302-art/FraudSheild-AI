"""SQLite persistence layer.

Tables
------
batches         one row per analysis run (manual entry or CSV upload)
transactions    individual scored transactions that are worth keeping
investigations  review status per transaction
notes           investigator notes (many per transaction)
settings        key/value user preferences and app settings
chat            ShieldBot conversation history
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .config import DB_PATH

STATUSES = ["Open", "Under Review", "Reviewed", "Confirmed Fraud", "False Positive"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    name TEXT,
    n_total INTEGER NOT NULL DEFAULT 0,
    n_flagged INTEGER NOT NULL DEFAULT 0,
    amount_total REAL NOT NULL DEFAULT 0,
    amount_flagged REAL NOT NULL DEFAULT 0,
    has_labels INTEGER NOT NULL DEFAULT 0,
    label_metrics TEXT,
    hourly TEXT,
    risk_hist TEXT,
    model_name TEXT
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    row_index INTEGER,
    created_at TEXT NOT NULL,
    amount REAL,
    time_sec REAL,
    features TEXT,
    prediction TEXT NOT NULL,
    risk_score REAL NOT NULL,
    risk_level TEXT NOT NULL,
    confidence REAL,
    explanation TEXT,
    drivers TEXT,
    actual_label INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tx_risk ON transactions(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_tx_batch ON transactions(batch_id);
CREATE TABLE IF NOT EXISTS investigations (
    transaction_id INTEGER PRIMARY KEY REFERENCES transactions(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'Open',
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    author TEXT,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def _df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ----------------------------------------------------------------- batches
def create_batch(source: str, name: str, n_total: int, n_flagged: int, amount_total: float,
                 amount_flagged: float, has_labels: bool, label_metrics: dict | None,
                 hourly: dict | None, risk_hist: list | None, model_name: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO batches (created_at, source, name, n_total, n_flagged, amount_total,
               amount_flagged, has_labels, label_metrics, hourly, risk_hist, model_name)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now(), source, name, int(n_total), int(n_flagged), float(amount_total),
             float(amount_flagged), int(has_labels),
             json.dumps(label_metrics) if label_metrics else None,
             json.dumps(hourly) if hourly else None,
             json.dumps(risk_hist) if risk_hist is not None else None, model_name),
        )
        return int(cur.lastrowid)


def insert_transactions(batch_id: int, rows: list[dict]) -> None:
    if not rows:
        return
    ts = now()
    with connect() as conn:
        conn.executemany(
            """INSERT INTO transactions (batch_id, row_index, created_at, amount, time_sec, features,
               prediction, risk_score, risk_level, confidence, explanation, drivers, actual_label)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(batch_id, r.get("row_index"), ts, r.get("amount"), r.get("time_sec"),
              json.dumps(r.get("features", {})), r["prediction"], float(r["risk_score"]),
              r["risk_level"], r.get("confidence"), r.get("explanation", ""),
              json.dumps(r.get("drivers", [])), r.get("actual_label")) for r in rows],
        )


def list_batches(limit: int = 200) -> pd.DataFrame:
    return _df("SELECT * FROM batches ORDER BY id DESC LIMIT ?", (limit,))


def kpi_totals() -> dict:
    with connect() as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM(n_total),0) AS n_total, COALESCE(SUM(n_flagged),0) AS n_flagged,
               COALESCE(SUM(amount_total),0) AS amount_total, COALESCE(SUM(amount_flagged),0) AS amount_flagged,
               COUNT(*) AS n_batches FROM batches"""
        ).fetchone()
    return dict(row)


# ------------------------------------------------------------ transactions
def list_transactions(min_risk: float = 0, statuses: list[str] | None = None,
                      levels: list[str] | None = None, search: str = "",
                      date_from: str | None = None, date_to: str | None = None,
                      limit: int = 2000) -> pd.DataFrame:
    sql = """SELECT t.id, t.batch_id, t.row_index, t.created_at, t.amount, t.time_sec, t.prediction,
                    t.risk_score, t.risk_level, t.confidence, t.explanation, t.drivers, t.actual_label,
                    COALESCE(i.status,'Open') AS status
             FROM transactions t LEFT JOIN investigations i ON i.transaction_id = t.id
             WHERE t.risk_score >= ?"""
    params: list = [min_risk]
    if statuses:
        sql += f" AND COALESCE(i.status,'Open') IN ({','.join('?' * len(statuses))})"
        params += statuses
    if levels:
        sql += f" AND t.risk_level IN ({','.join('?' * len(levels))})"
        params += levels
    if search:
        s = search.strip().lstrip("#")
        if s.isdigit():
            sql += " AND (t.id = ? OR t.row_index = ?)"
            params += [int(s), int(s)]
        else:
            sql += " AND (t.explanation LIKE ? OR t.drivers LIKE ? OR t.prediction LIKE ?)"
            params += [f"%{s}%"] * 3
    if date_from:
        sql += " AND date(t.created_at) >= date(?)"
        params.append(date_from)
    if date_to:
        sql += " AND date(t.created_at) <= date(?)"
        params.append(date_to)
    sql += " ORDER BY t.risk_score DESC, t.id DESC LIMIT ?"
    params.append(limit)
    return _df(sql, tuple(params))


def get_transaction(tx_id: int) -> dict | None:
    df = _df(
        """SELECT t.*, COALESCE(i.status,'Open') AS status, i.updated_at AS status_updated
           FROM transactions t LEFT JOIN investigations i ON i.transaction_id = t.id WHERE t.id = ?""",
        (int(tx_id),),
    )
    return None if df.empty else df.iloc[0].to_dict()


def set_status(tx_id: int, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"Unknown status: {status}")
    with connect() as conn:
        conn.execute(
            """INSERT INTO investigations (transaction_id, status, updated_at) VALUES (?,?,?)
               ON CONFLICT(transaction_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at""",
            (int(tx_id), status, now()),
        )


def add_note(tx_id: int, note: str, author: str = "Analyst") -> None:
    note = (note or "").strip()
    if not note:
        raise ValueError("Note cannot be empty.")
    with connect() as conn:
        conn.execute("INSERT INTO notes (transaction_id, author, note, created_at) VALUES (?,?,?,?)",
                     (int(tx_id), author, note[:2000], now()))


def get_notes(tx_id: int) -> pd.DataFrame:
    return _df("SELECT author, note, created_at FROM notes WHERE transaction_id = ? ORDER BY id DESC",
               (int(tx_id),))


def all_notes() -> pd.DataFrame:
    return _df("SELECT transaction_id, author, note, created_at FROM notes ORDER BY id")


# ---------------------------------------------------------------- settings
def get_setting(key: str, default: str | None = None) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return default if row is None else row["value"]


def set_setting(key: str, value) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


# -------------------------------------------------------------------- chat
def add_chat(role: str, content: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO chat (role, content, created_at) VALUES (?,?,?)", (role, content, now()))


def get_chat(limit: int = 100) -> list[dict]:
    df = _df("SELECT role, content FROM (SELECT * FROM chat ORDER BY id DESC LIMIT ?) ORDER BY id", (limit,))
    return df.to_dict("records")


def clear_chat() -> None:
    with connect() as conn:
        conn.execute("DELETE FROM chat")


# ------------------------------------------------------------------ admin
def clear_analysis_data() -> None:
    """Delete batches, transactions, investigations and notes (settings are kept)."""
    with connect() as conn:
        for table in ("notes", "investigations", "transactions", "batches"):
            conn.execute(f"DELETE FROM {table}")


def db_size_bytes() -> int:
    return DB_PATH.stat().st_size if DB_PATH.exists() else 0
