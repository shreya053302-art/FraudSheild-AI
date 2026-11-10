"""Profile, preferences, system health and data management."""
from __future__ import annotations

import platform
import sys

import sklearn
import streamlit as st

from backend import database as db
from backend.config import DB_PATH, MAX_STORED_ROWS_PER_BATCH, MODEL_PATH, gemini_key

from .styles import page_header


def render(bundle) -> None:
    page_header("Profile & Settings", "Preferences, system health and data management", "⚙️")

    st.subheader("Profile and preferences")
    with st.form("prefs"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Display name (used as author of investigation notes)", value=db.get_setting("display_name", "Analyst"),
                             max_chars=40)
        role = c2.selectbox("Role", ["Student", "Fraud analyst", "Data scientist", "Compliance officer", "Other"],
                            index=["Student", "Fraud analyst", "Data scientist", "Compliance officer", "Other"].index(
                                db.get_setting("role", "Student")))
        toasts = st.checkbox("Show alert pop-ups when fraud is detected",
                             value=db.get_setting("alerts", "1") == "1")
        if st.form_submit_button("Save preferences"):
            clean = name.strip() or "Analyst"
            db.set_setting("display_name", clean)
            db.set_setting("role", role)
            db.set_setting("alerts", "1" if toasts else "0")
            st.success("Preferences saved.")
    st.caption("This is a single-user local app, so there is no login and no password is stored anywhere. "
               "API keys live only in environment variables (.env).")

    st.subheader("System health")
    k = db.kpi_totals()
    rows = [
        ("Database", f"OK · {DB_PATH.name} · {db.db_size_bytes() / 1024:,.0f} KB"),
        ("Saved records", f"{k['n_batches']} run(s), {k['n_total']:,} transactions counted"),
        ("Model", f"OK · {bundle['selected']} · trained {bundle['trained_at']}" if bundle else "Not trained"),
        ("Model file", f"{MODEL_PATH.name} · {MODEL_PATH.stat().st_size / 1024 / 1024:,.1f} MB" if MODEL_PATH.exists() else "missing"),
        ("ShieldBot", "AI mode (Gemini key found)" if gemini_key() else "Built-in mode (no GEMINI_API_KEY)"),
        ("Stored rows per bulk run", f"up to {MAX_STORED_ROWS_PER_BATCH:,}"),
        ("Python / scikit-learn", f"{platform.python_version()} / {sklearn.__version__}"),
        ("Platform", f"{platform.system()} {platform.release()} · {sys.version.split()[0]}"),
    ]
    st.dataframe({"Component": [r[0] for r in rows], "Status": [r[1] for r in rows]}, hide_index=True)
    if bundle and bundle.get("sklearn_version") != sklearn.__version__:
        st.warning(f"The model was saved with scikit-learn {bundle.get('sklearn_version')} but you are running "
                   f"{sklearn.__version__}. If predictions fail, retrain in the Model Center.")

    st.subheader("Data management")
    st.markdown("Delete saved analysis runs, flagged transactions, notes and statuses. The trained model and your "
                "preferences are kept.")
    confirm = st.checkbox("I understand this cannot be undone", key="confirm_clear")
    if st.button("Delete all analysis data", disabled=not confirm, key="clear_all"):
        db.clear_analysis_data()
        st.session_state.pop("last_single", None)
        st.session_state.pop("bulk_result", None)
        st.success("All analysis data deleted.")
