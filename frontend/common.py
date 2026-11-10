"""Shared helpers for the Streamlit pages."""
from __future__ import annotations

import streamlit as st

PAGES = [
    "📊 Dashboard",
    "🔍 Analyze Transaction",
    "📂 Bulk Analysis",
    "🕵️ Investigation Center",
    "📈 Analytics & Reports",
    "🤖 ShieldBot",
    "🧠 Model Center",
    "⚙️ Settings",
]


def goto(page: str) -> None:
    """Button callback that switches the sidebar page."""
    st.session_state["nav"] = page


def require_model(bundle) -> bool:
    """Show a clear message and a shortcut when no trained model exists."""
    if bundle is not None:
        return True
    from .styles import empty_state

    empty_state("No trained model yet",
                "FraudShield never shows predictions without a real trained model. Train one in the Model Center "
                "using creditcard.csv (or any CSV with a 0/1 Class column).", "🧠")
    st.button("Go to Model Center", on_click=goto, args=(PAGES[6],), key=f"mc_{st.session_state.get('nav')}")
    return False


def fmt_money(x: float) -> str:
    return f"{x:,.2f}"
