"""Fraud Investigation Center: search, review, notes, exports."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from backend import database as db
from backend import report_generator as rg

from .common import PAGES, goto
from .styles import disclaimer, empty_state, level_badge, page_header


def render(bundle) -> None:
    page_header("Investigation Center", "Review flagged transactions, add notes and export case reports", "🕵️")
    prefs_author = db.get_setting("display_name", "Analyst")

    with st.expander("🔎 Search and filters", expanded=True):
        c1, c2, c3 = st.columns([2, 2, 2])
        search = c1.text_input("Search (ID, row number or keyword)", key="inv_search",
                               placeholder="e.g. 12 or V14")
        levels = c2.multiselect("Risk level", ["Medium", "High", "Critical"], default=["High", "Critical"], key="inv_levels")
        statuses = c3.multiselect("Status", db.STATUSES, default=[], key="inv_status",
                                  placeholder="All statuses")
        d1, d2 = st.columns(2)
        use_dates = d1.checkbox("Filter by detection date", key="inv_use_dates")
        date_from = date_to = None
        if use_dates:
            rng = d2.date_input("Date range", value=(pd.Timestamp.today().date(), pd.Timestamp.today().date()), key="inv_dates")
            if isinstance(rng, (tuple, list)) and len(rng) == 2:
                date_from, date_to = str(rng[0]), str(rng[1])
    if not levels:
        levels = ["Medium", "High", "Critical"]

    df = db.list_transactions(min_risk=25, statuses=statuses or None, levels=levels, search=search,
                              date_from=date_from, date_to=date_to, limit=2000)
    if df.empty:
        empty_state("No matching transactions",
                    "Nothing has been flagged yet, or your filters exclude everything. Run an analysis or widen the filters.", "🕵️")
        st.button("Go to Bulk Analysis", on_click=goto, args=(PAGES[2],), key="inv_to_bulk")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching cases", f"{len(df):,}")
    m2.metric("Open", int((df["status"] == "Open").sum()))
    m3.metric("Confirmed fraud", int((df["status"] == "Confirmed Fraud").sum()))
    m4.metric("False positives", int((df["status"] == "False Positive").sum()))

    table = df[["id", "created_at", "amount", "risk_score", "risk_level", "prediction", "status"]].copy()
    table.columns = ["ID", "Detected", "Amount", "Risk", "Level", "Prediction", "Status"]
    st.dataframe(table, hide_index=True, height=300, column_config={
        "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
        "Amount": st.column_config.NumberColumn(format="%.2f")})

    e1, e2 = st.columns(2)
    notes_all = db.all_notes()
    csv = rg.investigation_csv(df, notes_all)
    e1.download_button("⬇️ Export filtered cases (CSV)", csv, "investigation_cases.csv", "text/csv", key="inv_csv")

    st.divider()
    ids = df["id"].astype(int).tolist()
    sel = st.selectbox("Open a case", ids, format_func=lambda i: f"Transaction #{i}", key="inv_sel")
    tx = db.get_transaction(sel)
    if tx is None:
        st.error("Transaction not found.")
        return

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader(f"Transaction #{tx['id']}")
        st.markdown(f"{level_badge(tx['risk_level'])} &nbsp; **{tx['prediction']}** · risk "
                    f"**{tx['risk_score']:.1f}** · status **{tx['status']}**", unsafe_allow_html=True)
        st.write(f"**Detected:** {tx['created_at']}  \n**Amount:** {tx['amount']:,.2f}  \n"
                 f"**Time value:** {tx['time_sec']}  \n**Source row:** {tx['row_index']}")
        if tx.get("actual_label") is not None and not pd.isna(tx["actual_label"]):
            st.write(f"**Actual label in uploaded file:** {'Fraud' if int(tx['actual_label']) == 1 else 'Legitimate'}")
        st.markdown("**Why it was flagged**")
        st.write(tx["explanation"] or "-")
        with st.expander("Feature values"):
            try:
                st.dataframe(pd.Series(json.loads(tx["features"])).rename("value").to_frame())
            except (TypeError, ValueError):
                st.write("No feature values stored.")
    with right:
        st.subheader("Review")
        if st.button("✅ Mark as Reviewed", key=f"rev_{sel}"):
            db.set_status(sel, "Reviewed")
            st.rerun()
        new_status = st.selectbox("Set status", db.STATUSES, index=db.STATUSES.index(tx["status"]),
                                  key=f"st_{sel}")
        if st.button("Update status", key=f"upd_{sel}"):
            db.set_status(sel, new_status)
            st.rerun()
        st.markdown("**Investigation notes**")
        note = st.text_area("Add a note", key=f"note_{sel}", max_chars=2000, height=90,
                            placeholder="What did you check? What did the customer say?")
        if st.button("Save note", key=f"savenote_{sel}"):
            try:
                db.add_note(sel, note, prefs_author)
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
        notes = db.get_notes(sel)
        if notes.empty:
            st.caption("No notes yet.")
        for _, n in notes.iterrows():
            st.markdown(f"> {n['note']}  \n> <span class='note'>{n['author']} · {n['created_at']}</span>",
                        unsafe_allow_html=True)
        pdf = rg.case_pdf(tx, notes)
        st.download_button("⬇️ Download case report (PDF)", pdf, f"case_{tx['id']}.pdf", "application/pdf",
                           key=f"pdf_{sel}")
    disclaimer()
