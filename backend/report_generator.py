"""CSV and PDF report generation (ReportLab)."""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import REPORTS_DIR

DISCLAIMER = ("FraudShield AI is a decision-support tool. Its predictions are statistical estimates and are not "
              "proof of fraud. Every flagged transaction should be reviewed by a qualified person.")
NAVY = colors.HexColor("#0b1230")
BLUE = colors.HexColor("#3b82f6")


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:80] or "report"


def save_copy(data: bytes, filename: str) -> str:
    """Keep a copy of each generated report in the reports/ folder."""
    path = REPORTS_DIR / safe_filename(filename)
    path.write_bytes(data)
    return str(path)


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=22, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=BLUE, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13),
        "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=8, textColor=colors.grey, leading=10),
    }


def _table(rows, col_widths=None, header=True):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fb")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def summary_pdf(kpis: dict, bundle: dict | None, flagged: pd.DataFrame, trend_df: pd.DataFrame) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="FraudShield AI - Fraud Analysis Report")
    el = [_p("FraudShield AI", st["title"]), _p("Fraud Analysis Report", st["h2"]),
          _p(f"Generated {datetime.now():%Y-%m-%d %H:%M}", st["small"]), Spacer(1, 6)]

    el.append(_p("1. Summary", st["h2"]))
    rows = [["Metric", "Value"],
            ["Transactions analyzed", f"{kpis['n_total']:,}"],
            ["Flagged as fraudulent", f"{kpis['n_flagged']:,}"],
            ["Legitimate", f"{kpis['n_legit']:,}"],
            ["Flag rate", f"{kpis['flag_rate'] * 100:.2f}%"],
            ["Overall risk label", kpis["risk_label"]],
            ["Total amount flagged", f"{kpis['amount_flagged']:,.2f}"],
            ["Analysis runs", f"{kpis['n_batches']:,}"]]
    el.append(_table(rows, [70 * mm, 60 * mm]))

    if bundle:
        m = bundle["metrics"][bundle["selected"]]
        cm = m["confusion_matrix"]
        el.append(_p("2. Model (held-out test set)", st["h2"]))
        el.append(_p(f"Selected model: {bundle['selected']} (trained {bundle['trained_at']} on "
                     f"'{bundle['dataset_name']}', {bundle['n_rows']:,} rows). Class imbalance handled with class "
                     f"weights; the threshold was chosen on a separate validation set.", st["body"]))
        el.append(Spacer(1, 4))
        el.append(_table([["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "Accuracy"],
                          [f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                           f"{m['pr_auc']:.3f}" if m["pr_auc"] is not None else "n/a",
                           f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "n/a", f"{m['accuracy']:.4f}"]]))
        el.append(Spacer(1, 4))
        el.append(_table([["", "Predicted legit", "Predicted fraud"], ["Actual legit", cm[0][0], cm[0][1]],
                          ["Actual fraud", cm[1][0], cm[1][1]]], [35 * mm, 40 * mm, 40 * mm]))
        top = sorted(bundle["importance"].items(), key=lambda kv: -kv[1])[:8]
        el.append(_p("Top model features", st["h2"]))
        el.append(_table([["Feature", "Importance"]] + [[f, f"{v:.3f}"] for f, v in top], [50 * mm, 40 * mm]))

    el.append(_p("3. Detection trend", st["h2"]))
    if trend_df.empty:
        el.append(_p("No analyses have been run yet.", st["body"]))
    else:
        rows = [["Period", "Analyzed", "Flagged", "Flag rate %"]]
        for _, r in trend_df.tail(15).iterrows():
            rows.append([f"{r['period']:%Y-%m-%d}", f"{int(r['analyzed']):,}", f"{int(r['flagged']):,}", f"{r['flag_rate']:.2f}"])
        el.append(_table(rows))

    el.append(_p("4. Highest-risk flagged transactions", st["h2"]))
    if flagged.empty:
        el.append(_p("No flagged transactions stored.", st["body"]))
    else:
        rows = [["ID", "Amount", "Risk", "Level", "Status"]]
        for _, r in flagged.head(20).iterrows():
            rows.append([int(r["id"]), f"{r['amount']:,.2f}", f"{r['risk_score']:.1f}", r["risk_level"], r["status"]])
        el.append(_table(rows))

    el += [Spacer(1, 10), _p(DISCLAIMER, st["small"])]
    doc.build(el)
    return buf.getvalue()


def case_pdf(tx: dict, notes: pd.DataFrame) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"Investigation report #{tx['id']}")
    el = [_p("FraudShield AI", st["title"]), _p(f"Investigation Report - Transaction #{tx['id']}", st["h2"]),
          _p(f"Generated {datetime.now():%Y-%m-%d %H:%M}", st["small"]), Spacer(1, 6)]
    rows = [["Field", "Value"], ["Prediction", tx["prediction"]], ["Risk score", f"{tx['risk_score']:.1f} / 100"],
            ["Risk level", tx["risk_level"]], ["Decision confidence (not a probability)", f"{(tx['confidence'] or 0) * 100:.0f}%"],
            ["Amount", f"{(tx['amount'] or 0):,.2f}"], ["Time value", tx["time_sec"]],
            ["Detected at", tx["created_at"]], ["Source row in file", tx["row_index"]],
            ["Investigation status", tx["status"]]]
    if tx.get("actual_label") is not None and not pd.isna(tx.get("actual_label")):
        rows.append(["Actual label in file", "Fraud" if int(tx["actual_label"]) == 1 else "Legitimate"])
    el.append(_table(rows, [75 * mm, 90 * mm]))
    el.append(_p("Why it was flagged", st["h2"]))
    el.append(_p(tx.get("explanation") or "-", st["body"]))
    el.append(_p("Investigator notes", st["h2"]))
    if notes.empty:
        el.append(_p("No notes recorded.", st["body"]))
    else:
        for _, n in notes.iterrows():
            el.append(_p(f"[{n['created_at']}] {n['author']}: {n['note']}", st["body"]))
    el += [Spacer(1, 10), _p(DISCLAIMER, st["small"])]
    doc.build(el)
    return buf.getvalue()


def investigation_csv(df: pd.DataFrame, notes: pd.DataFrame) -> bytes:
    """Flat export of cases with their notes joined into one column."""
    out = df.drop(columns=[c for c in ("drivers",) if c in df.columns]).copy()
    if not notes.empty:
        joined = notes.groupby("transaction_id")["note"].apply(lambda s: " | ".join(s))
        out["notes"] = out["id"].map(joined).fillna("")
    else:
        out["notes"] = ""
    return out.to_csv(index=False).encode("utf-8")


def summary_text(kpis: dict, bundle: dict | None) -> str:
    if not kpis["n_total"]:
        return "No transactions have been analyzed yet. Run a bulk or manual analysis to generate a summary."
    text = (f"{kpis['n_total']:,} transactions were analyzed across {kpis['n_batches']} run(s). "
            f"{kpis['n_flagged']:,} ({kpis['flag_rate'] * 100:.2f}%) were flagged as likely fraud, with a combined "
            f"amount of {kpis['amount_flagged']:,.2f}. Overall portfolio risk: {kpis['risk_label']}.")
    if bundle:
        m = bundle["metrics"][bundle["selected"]]
        text += (f" The {bundle['selected']} model reached recall {m['recall']:.2f} and precision {m['precision']:.2f} "
                 f"on its held-out test set.")
    return text
