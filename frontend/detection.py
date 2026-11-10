"""Single-transaction analysis with explanation and recommended actions."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import database as db
from backend.prediction import RECOMMENDATIONS, run_analysis

from .common import PAGES, goto, require_model
from .styles import disclaimer, level_badge, page_header, risk_meter, style_fig, verdict_card


def _fkey(c: str) -> str:
    return f"feat_{c}"


def _fill(bundle, values: dict) -> None:
    for c in bundle["feature_cols"]:
        st.session_state[_fkey(c)] = float(values.get(c, bundle["defaults"][c]))


def _load_preset(bundle) -> None:
    choice = st.session_state.get("preset", "Typical legitimate values")
    if choice == "Typical legitimate values":
        _fill(bundle, bundle["defaults"])
    else:
        key = "fraud" if "fraud" in choice.lower() else "legitimate"
        i = int(choice.split("#")[-1]) - 1
        samples = bundle["samples"].get(key, [])
        if i < len(samples):
            _fill(bundle, samples[i])


def _validate(bundle, vals: dict) -> list[str]:
    errors = []
    for c, v in vals.items():
        if v is None or isinstance(v, bool) or not math.isfinite(v):
            errors.append(f"{c} must be a valid number.")
        elif abs(v) > 1e7:
            errors.append(f"{c} is out of the accepted range (±10,000,000).")
    if vals.get("Amount", 0) is not None and vals.get("Amount", 0) < 0:
        errors.append("Amount cannot be negative.")
    if vals.get("Time", 0) is not None and vals.get("Time", 0) < 0:
        errors.append("Time cannot be negative.")
    return errors


def render(bundle) -> None:
    page_header("Analyze a Transaction", "Enter transaction details and get a risk score with an explanation", "🔍")
    if not require_model(bundle):
        return
    cols = bundle["feature_cols"]
    for c in cols:  # initialise widget state once
        st.session_state.setdefault(_fkey(c), float(bundle["defaults"][c]))

    presets = ["Typical legitimate values"]
    presets += [f"Known legitimate sample #{i + 1}" for i in range(len(bundle["samples"].get("legitimate", [])))]
    presets += [f"Known fraud sample #{i + 1}" for i in range(len(bundle["samples"].get("fraud", [])))]
    st.selectbox("Start from", presets, key="preset", on_change=_load_preset, args=(bundle,),
                 help="Samples come from the held-out test rows of your training data, so you can see how the "
                      "model reacts to real examples.")

    main = [c for c in ("Amount", "Time") if c in cols]
    rest = [c for c in cols if c not in main]
    with st.form("tx_form"):
        mc = st.columns(max(len(main), 1))
        for col, c in zip(mc, main):
            col.number_input(c + (" (seconds since first transaction)" if c == "Time" else ""),
                             key=_fkey(c), step=1.0 if c == "Time" else 0.01, format="%.2f")
        if rest:
            with st.expander(f"Model features ({len(rest)})", expanded=len(main) == 0):
                grid = st.columns(4)
                for i, c in enumerate(rest):
                    grid[i % 4].number_input(c, key=_fkey(c), step=0.1, format="%.4f")
        submitted = st.form_submit_button("🔎 ANALYZE TRANSACTION", width="stretch")

    if submitted:
        vals = {c: st.session_state.get(_fkey(c)) for c in cols}
        errors = _validate(bundle, vals)
        if errors:
            for e in errors[:6]:
                st.error(e)
        else:
            try:
                res = run_analysis(bundle, pd.DataFrame([vals]), "manual", "Manual entry")
                st.session_state["last_single"] = res
                if res["n_flagged"] and db.get_setting("alerts", "1") == "1":
                    st.toast("🚨 Fraud alert: transaction flagged for review", icon="🚨")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    res = st.session_state.get("last_single")
    if not res:
        return
    row = res["result"].iloc[0]
    st.divider()
    a, b, c = st.columns([1.1, 1.1, 1])
    with a:
        verdict_card(row["prediction"], row["risk_level"])
    with b:
        risk_meter(float(row["risk_score"]), row["risk_level"])
    with c:
        st.markdown(f'<div class="glass"><div class="note">DECISION CONFIDENCE</div>'
                    f'<div style="font-size:2rem;font-weight:800">{row["confidence"] * 100:.0f}%</div>'
                    f'<div class="note">Distance from the decision line. This is <b>not</b> a calibrated '
                    f'probability of fraud.</div></div>', unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(row["risk_score"]), number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#8b5cf6"},
               "steps": [{"range": [0, 25], "color": "rgba(16,185,129,.35)"},
                         {"range": [25, 50], "color": "rgba(245,158,11,.35)"},
                         {"range": [50, 75], "color": "rgba(249,115,22,.4)"},
                         {"range": [75, 100], "color": "rgba(239,68,68,.45)"}]}))
    left, right = st.columns([1, 1.3])
    left.plotly_chart(style_fig(fig, 260), key="single_gauge")
    with right:
        st.subheader("Why this result?")
        st.write(res["explanations"][0])
        drivers = res["drivers"][0]
        if drivers:
            d = pd.DataFrame(drivers)[["feature", "value", "z", "direction"]]
            d.columns = ["Feature", "Value", "Z-score", "Direction vs normal"]
            st.dataframe(d, hide_index=True)
        st.subheader("Recommended actions")
        for r in RECOMMENDATIONS[row["risk_level"]]:
            st.markdown(f"- {r}")
    st.markdown(f"Saved as transaction **#{res['transaction_id']}** {level_badge(row['risk_level'])}",
                unsafe_allow_html=True)
    if res["n_flagged"]:
        st.button("Open in Investigation Center", on_click=goto, args=(PAGES[3],), key="single_to_inv")
    disclaimer()
