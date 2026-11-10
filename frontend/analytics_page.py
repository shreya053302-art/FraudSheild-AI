"""Analytics, model performance, monitoring and report downloads."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from backend import analytics as an
from backend import report_generator as rg

from .common import PAGES, goto
from .styles import empty_state, page_header, style_fig


def _trends() -> None:
    k = an.get_kpis()
    if k["n_total"] == 0:
        empty_state("No analysis data yet", "Run a bulk or manual analysis to see trends.", "📈")
        st.button("Open Bulk Analysis", on_click=goto, args=(PAGES[2],), key="an_to_bulk")
        return
    freq = st.radio("Period", ["Daily", "Weekly", "Monthly"], horizontal=True, key="an_freq")
    tr = an.trend({"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq])
    fig = go.Figure()
    fig.add_bar(x=tr["period"], y=tr["flagged"], name="Flagged", marker_color="#ef4444")
    fig.add_scatter(x=tr["period"], y=tr["flag_rate"], name="Flag rate %", yaxis="y2", mode="lines+markers",
                    line=dict(color="#22d3ee"))
    fig.update_layout(title=f"{freq} fraud detection (by detection date)",
                      yaxis2=dict(overlaying="y", side="right", title="Flag rate %", showgrid=False))
    st.plotly_chart(style_fig(fig, 360), key="an_trend")
    flagged = an.flagged_transactions()
    c1, c2 = st.columns(2)
    if flagged.empty:
        c1.info("No flagged transactions stored yet.")
    else:
        b = an.amount_buckets(flagged)
        fig = px.bar(b, x="amount_range", y="flagged", title="Flagged transactions by amount")
        fig.update_traces(marker_color="#8b5cf6")
        c1.plotly_chart(style_fig(fig), key="an_amt_count")
        fig = px.bar(b, x="amount_range", y="amount", title="Total flagged amount by range")
        fig.update_traces(marker_color="#3b82f6")
        c2.plotly_chart(style_fig(fig), key="an_amt_sum")


def _risk_patterns() -> None:
    k = an.get_kpis()
    if k["n_total"] == 0:
        empty_state("No analysis data yet", "Risk charts appear after your first analysis.", "🎯")
        return
    rd = an.risk_distribution()
    fig = px.bar(rd, x="risk_range", y="transactions", title="Transaction risk distribution (log scale)")
    fig.update_traces(marker_color="#22d3ee")
    fig.update_yaxes(type="log")
    st.plotly_chart(style_fig(fig), key="an_riskdist")

    hm = an.risk_heatmap()
    if hm.empty:
        st.info("The risk heatmap needs a 'Time' column in the analyzed data.")
    else:
        fig = px.imshow(hm, aspect="auto", color_continuous_scale=["#0b1230", "#3b82f6", "#8b5cf6", "#ef4444"],
                        labels=dict(x="Relative hour of day", y="Analysis run", color="Flagged %"),
                        title="Transaction risk heatmap: flagged % by hour")
        st.plotly_chart(style_fig(fig, 340), key="an_heatmap")
        st.caption("In the Kaggle data, 'Time' counts seconds from the first transaction, so hours are relative to "
                   "the start of the data, not wall-clock time.")
    st.subheader("Fraud patterns")
    pat = an.pattern_summary(an.flagged_transactions())
    if pat.empty:
        st.info("No flagged transactions with explanations yet.")
    else:
        pat["label"] = pat["feature"] + " (" + pat["direction"] + " normal)"
        fig = px.bar(pat.iloc[::-1], x="flagged_transactions", y="label", orientation="h",
                     title="Features most often behind a fraud flag")
        fig.update_traces(marker_color="#ef4444")
        st.plotly_chart(style_fig(fig, 380), key="an_patterns")


def _model(bundle) -> None:
    if not bundle:
        empty_state("No trained model", "Train a model in the Model Center to see its performance.", "🧠")
        st.button("Go to Model Center", on_click=goto, args=(PAGES[6],), key="an_to_mc")
        return
    st.caption(f"Trained {bundle['trained_at']} on '{bundle['dataset_name']}' · {bundle['n_rows']:,} rows · "
               f"split {bundle['n_train']:,} train / {bundle['n_val']:,} validation / {bundle['n_test']:,} test. "
               "Class imbalance handled with class weights (no oversampling, so nothing leaks into the test set).")
    rows = []
    for name, m in bundle["metrics"].items():
        rows.append({"Model": name + (" ✅ selected" if name == bundle["selected"] else ""),
                     "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4), "F1": round(m["f1"], 4),
                     "PR-AUC": round(m["pr_auc"], 4) if m["pr_auc"] is not None else None,
                     "ROC-AUC": round(m["roc_auc"], 4) if m["roc_auc"] is not None else None,
                     "Accuracy": round(m["accuracy"], 4), "Threshold": round(m["threshold"], 3),
                     "Fit time (s)": m["fit_seconds"]})
    st.dataframe(rows, hide_index=True)
    st.caption("Models are compared on validation PR-AUC. The numbers above come from the untouched test set.")

    sel = bundle["metrics"][bundle["selected"]]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Confusion matrix · {bundle['selected']}**")
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        fig.patch.set_alpha(0)
        sns.heatmap(np.array(sel["confusion_matrix"]), annot=True, fmt=",d", cmap="mako", cbar=False, ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], linewidths=1)
        ax.set_xlabel("Predicted", color="#c7cdea"); ax.set_ylabel("Actual", color="#c7cdea")
        ax.tick_params(colors="#c7cdea")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=160, bbox_inches="tight")
        plt.close(fig)
        st.image(buf)
    with c2:
        pc = bundle["pr_curve"]
        fig = go.Figure(go.Scatter(x=pc["recall"], y=pc["precision"], fill="tozeroy", line=dict(color="#8b5cf6")))
        fig.update_layout(title=f"Precision-recall curve (PR-AUC {sel['pr_auc']:.3f})",
                          xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(style_fig(fig, 330), key="an_prcurve")

    imp = sorted(bundle["importance"].items(), key=lambda kv: -kv[1])[:15]
    fig = px.bar(x=[v for _, v in imp][::-1], y=[f for f, _ in imp][::-1], orientation="h",
                 title="Explainable AI: top feature importance")
    fig.update_traces(marker_color="#3b82f6")
    st.plotly_chart(style_fig(fig, 420), key="an_importance")

    st.subheader("Model performance monitoring")
    mon = an.monitoring(bundle)
    if not mon["available"]:
        st.info("Monitoring starts once transactions have been analyzed.")
    else:
        a, b, c = st.columns(3)
        a.metric("Live flag rate", f"{mon['live_rate'] * 100:.2f}%")
        b.metric("Flag rate on test data", f"{mon['expected_rate'] * 100:.2f}%")
        c.metric("Status", "Stable" if mon["status"] == "Stable" else "Review")
        if mon["status"] != "Stable":
            st.warning(mon["status"] + ". This can mean new fraud patterns, different data, or a model that needs retraining.")
    hist = an.label_metrics_history()
    if not hist.empty:
        st.markdown("**Batches with true labels**")
        st.dataframe(hist.round(4), hide_index=True)


def _reports(bundle) -> None:
    k = an.get_kpis()
    st.markdown("#### Fraud analysis summary")
    st.write(rg.summary_text(k, bundle))
    flagged = an.flagged_transactions()
    c1, c2 = st.columns(2)
    if flagged.empty:
        st.info("Reports need at least one flagged transaction or analysis run.")
    csv = flagged.drop(columns=["drivers"], errors="ignore").to_csv(index=False).encode("utf-8")
    c1.download_button("⬇️ Download flagged transactions (CSV)", csv, "flagged_transactions.csv", "text/csv",
                       key="rep_csv", disabled=flagged.empty)
    tr = an.trend("D")
    pdf = rg.summary_pdf(k, bundle, flagged, tr)
    if c2.download_button("⬇️ Download fraud analysis report (PDF)", pdf, "fraudshield_report.pdf",
                          "application/pdf", key="rep_pdf"):
        rg.save_copy(pdf, "fraudshield_report.pdf")
    st.caption("Downloaded reports are also saved in the project's reports/ folder.")
    b = an.trend("D")
    if not b.empty:
        st.markdown("#### Detection history by day")
        st.dataframe(b.assign(period=b["period"].dt.strftime("%Y-%m-%d")).round(2), hide_index=True)


def render(bundle) -> None:
    page_header("Analytics & Reports", "Trends, risk patterns, model quality and downloadable reports", "📈")
    t1, t2, t3, t4 = st.tabs(["Trends", "Risk & patterns", "Model performance", "Reports"])
    with t1:
        _trends()
    with t2:
        _risk_patterns()
    with t3:
        _model(bundle)
    with t4:
        _reports(bundle)
