"""CSV upload: validate, preview, score, download."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from backend import database as db
from backend import preprocessing as pp
from backend import report_generator as rg
from backend.config import DEFAULT_DATASET, MAX_STORED_ROWS_PER_BATCH, MAX_UPLOAD_MB
from backend.prediction import run_analysis

from .common import PAGES, goto, require_model
from .styles import disclaimer, kpi_row, page_header, style_fig


def _run(bundle, df, name: str, target) -> None:
    bar = st.progress(0.0, text="Scoring transactions ...")
    try:
        res = run_analysis(bundle, df, "bulk", name, target_col=target,
                           progress=lambda f: bar.progress(min(f, 1.0), text=f"Scoring ... {f * 100:.0f}%"))
    except Exception as exc:
        bar.empty()
        st.error(f"Analysis failed: {exc}")
        return
    bar.empty()
    with st.spinner("Preparing the download file ..."):  # built once, not on every page refresh
        csv_bytes = res["result"].to_csv(index=False).encode("utf-8")
    st.session_state["bulk_result"] = {"res": res, "name": name, "csv": csv_bytes}
    if res["n_flagged"] and db.get_setting("alerts", "1") == "1":
        st.toast(f"🚨 {res['n_flagged']:,} transactions flagged", icon="🚨")


def render(bundle) -> None:
    page_header("Bulk Fraud Detection", "Upload a CSV of transactions and score every row", "📂")
    if not require_model(bundle):
        return
    need = ", ".join(bundle["feature_cols"][:4]) + f" ... ({len(bundle['feature_cols'])} columns)"
    st.caption(f"Required columns: {need}. A 0/1 target column such as 'Class' is optional; if present, "
               f"real precision and recall are calculated for your file. Limit {MAX_UPLOAD_MB} MB.")

    up = st.file_uploader("Upload transactions (.csv)", type=["csv"], key="bulk_upload")
    df, name = None, None
    if up is not None:
        try:
            df, name = pp.read_csv_safely(up, up.name), up.name
        except ValueError as exc:
            st.error(str(exc))
    if df is None and DEFAULT_DATASET.exists():
        st.info("Found data/creditcard.csv. You can analyze it directly. Note: the model was trained on part of "
                "this file, so results on it look better than they would on new data. The held-out test metrics "
                "on the Analytics page are the honest ones.")
        if st.button("Use data/creditcard.csv", key="use_local"):
            try:
                st.session_state["local_df"] = pp.read_csv_safely(DEFAULT_DATASET, DEFAULT_DATASET.name)
            except ValueError as exc:
                st.error(str(exc))
        if "local_df" in st.session_state:
            df, name = st.session_state["local_df"], DEFAULT_DATASET.name
    if df is None:
        st.markdown("")
    else:
        check = pp.validate_for_model(df, bundle["feature_cols"])
        df = check["df"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Target column", check["target"] or "none found")
        for w in check["warnings"]:
            st.warning(w)
        for e in check["errors"]:
            st.error(e)
        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(df.head(10))
        if check["ok"]:
            if st.button("🚀 Analyze all transactions", key="bulk_go"):
                _run(bundle, df, name, check["target"])
        else:
            st.error("Fix the problems above, then upload the file again.")

    out = st.session_state.get("bulk_result")
    if not out:
        return
    res, name = out["res"], out["name"]
    st.divider()
    st.subheader(f"Results for {name}")
    rate = res["n_flagged"] / max(res["n_total"], 1) * 100
    kpi_row([("Analyzed", f"{res['n_total']:,}", "rows scored", "blue"),
             ("Flagged fraudulent", f"{res['n_flagged']:,}", f"{rate:.2f}% of rows", "red"),
             ("Amount flagged", f"{res['amount_flagged']:,.2f}", "sum of flagged Amount", "orange")])
    st.write("")
    c1, c2 = st.columns(2)
    lv = res["levels"]
    fig = px.bar(x=list(lv.keys()), y=list(lv.values()), title="Rows by risk level", color=list(lv.keys()),
                 color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"})
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Transactions")
    fig.update_yaxes(type="log")
    c1.plotly_chart(style_fig(fig), key="bulk_levels")
    with c2:
        m = res["label_metrics"]
        if m:
            st.markdown("**Accuracy against the labels in your file**")
            cm = m["confusion_matrix"]
            st.dataframe({"Metric": ["Precision", "Recall", "F1", "PR-AUC"],
                          "Value": [round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4),
                                    None if m["pr_auc"] is None else round(m["pr_auc"], 4)]}, hide_index=True)
            st.caption(f"Caught {cm[1][1]:,} of {cm[1][0] + cm[1][1]:,} real fraud rows; "
                       f"{cm[0][1]:,} legitimate rows were flagged by mistake.")
        else:
            st.info("No 0/1 target column in this file, so accuracy cannot be measured. Only predictions are shown.")

    st.subheader("Most suspicious transactions")
    result = res["result"]
    top = result.sort_values("risk_score", ascending=False).head(25)
    keep = [c for c in ("Time", "Amount", "Class") if c in top.columns] + ["risk_score", "risk_level", "prediction"]
    st.dataframe(top[keep], column_config={
        "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=100, format="%.0f")})
    st.caption(f"The riskiest {res['stored_rows']:,} rows (max {MAX_STORED_ROWS_PER_BATCH:,}, risk ≥ 25) were saved to "
               "the Investigation Center. All rows are included in the totals and the download.")

    csv_bytes = out["csv"]
    d1, d2 = st.columns([1, 1])
    d1.download_button("⬇️ Download analyzed dataset (CSV)", csv_bytes, f"analyzed_{rg.safe_filename(name)}",
                       "text/csv", key="dl_bulk")
    d2.button("Open Investigation Center", on_click=goto, args=(PAGES[3],), key="bulk_to_inv")
    disclaimer()
