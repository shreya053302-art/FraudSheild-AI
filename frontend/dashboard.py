"""Main dashboard. Every number and chart comes from the database or the saved model."""
from __future__ import annotations

import html

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend import analytics as an
from backend import database as db

from .common import PAGES, fmt_money, goto
from .styles import COLORS, LEVEL_COLORS, empty_state, kpi_row, level_badge, page_header, style_fig


def render(bundle) -> None:
    page_header("Security Dashboard", "Live overview of analyzed transactions and detected fraud", "📊")
    k = an.get_kpis()

    acc_value, acc_sub = "—", "Train a model first"
    if bundle:
        m = bundle["metrics"][bundle["selected"]]
        acc_value = f"{m['accuracy'] * 100:.2f}%"
        acc_sub = f"Test set · recall {m['recall'] * 100:.0f}% · precision {m['precision'] * 100:.0f}%"
    kpi_row([
        ("Transactions analyzed", f"{k['n_total']:,}", f"{k['n_batches']} analysis run(s)", "blue"),
        ("Fraudulent detected", f"{k['n_flagged']:,}", f"{k['flag_rate'] * 100:.2f}% of analyzed", "red"),
        ("Legitimate", f"{k['n_legit']:,}", "Passed the model", "green"),
    ])
    st.write("")
    kpi_row([
        ("Detection accuracy", acc_value, acc_sub, "purple"),
        ("Overall fraud risk", k["risk_label"], "Based on share flagged", "orange"),
        ("Amount flagged", fmt_money(k["amount_flagged"]), f"of {fmt_money(k['amount_total'])} analyzed", "red"),
    ])
    st.caption("Accuracy is shown for reference only. Fraud is rare, so accuracy looks high even for weak models. "
               "Precision, recall and PR-AUC (Analytics page) are the honest measures.")

    if k["n_total"] == 0:
        st.write("")
        empty_state("No transactions analyzed yet",
                    "Upload a CSV in Bulk Analysis or score a single transaction. Charts will appear here "
                    "as soon as there is real data.", "📂")
        c1, c2, _ = st.columns([1, 1, 2])
        c1.button("Open Bulk Analysis", on_click=goto, args=(PAGES[2],), key="d_bulk")
        c2.button("Analyze a Transaction", on_click=goto, args=(PAGES[1],), key="d_single")
        return

    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        tr = an.trend("D")
        fig = go.Figure()
        fig.add_bar(x=tr["period"], y=tr["legitimate"], name="Legitimate", marker_color="#3b82f6")
        fig.add_bar(x=tr["period"], y=tr["flagged"], name="Fraudulent", marker_color="#ef4444")
        fig.update_layout(barmode="stack", title="Fraud detection trend (by detection date)")
        st.plotly_chart(style_fig(fig), key="d_trend")
    with c2:
        fig = px.pie(names=["Legitimate", "Fraudulent"], values=[k["n_legit"], k["n_flagged"]], hole=.62,
                     color=["Legitimate", "Fraudulent"],
                     color_discrete_map={"Legitimate": "#3b82f6", "Fraudulent": "#ef4444"},
                     title="Legitimate vs fraudulent")
        st.plotly_chart(style_fig(fig), key="d_pie")

    c3, c4 = st.columns(2)
    with c3:
        h = an.hourly_activity()
        if h["transactions"].sum() > 0:
            fig = go.Figure()
            fig.add_scatter(x=h["hour"], y=h["transactions"], name="Transactions", fill="tozeroy",
                            line=dict(color="#8b5cf6"))
            fig.add_scatter(x=h["hour"], y=h["flagged"], name="Flagged", line=dict(color="#ef4444"))
            fig.update_layout(title="Transaction activity by relative hour", xaxis_title="Hour (Time column mod 24h)")
            st.plotly_chart(style_fig(fig), key="d_hour")
        else:
            st.info("Hourly activity needs a 'Time' column in the analyzed data.")
    with c4:
        rd = an.risk_distribution()
        fig = px.bar(rd, x="risk_range", y="transactions", title="Fraud risk distribution (log scale)")
        fig.update_traces(marker_color="#22d3ee")
        fig.update_yaxes(type="log")
        fig.update_xaxes(title="Risk score range")
        st.plotly_chart(style_fig(fig), key="d_risk")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recent suspicious transactions")
        df = db.list_transactions(min_risk=50, limit=10)
        if df.empty:
            st.info("No transactions have been flagged yet.")
        else:
            show = df[["id", "created_at", "amount", "risk_score", "risk_level", "status"]].copy()
            show.columns = ["ID", "Detected", "Amount", "Risk", "Level", "Status"]
            st.dataframe(show, hide_index=True, column_config={
                "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                "Amount": st.column_config.NumberColumn(format="%.2f")})
    with right:
        st.subheader("🚨 Fraud alerts")
        alerts = an.recent_alerts(5)
        if alerts.empty:
            st.success("No open high-risk alerts.")
        for _, a in alerts.iterrows():
            c = LEVEL_COLORS.get(a["risk_level"], "#ef4444")
            st.markdown(
                f'<div class="alert" style="border-color:{c}66"><b>Transaction #{int(a["id"])}</b> '
                f'{level_badge(a["risk_level"])}<div class="meta">Risk {a["risk_score"]:.0f} · amount '
                f'{a["amount"]:,.2f} · {html.escape(str(a["created_at"]))}</div></div>', unsafe_allow_html=True)
        st.button("Open Investigation Center", on_click=goto, args=(PAGES[3],), key="d_inv")
