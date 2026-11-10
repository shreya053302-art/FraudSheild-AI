"""ShieldBot chat page."""
from __future__ import annotations

import streamlit as st

from backend import assistant as bot
from backend import database as db
from backend.config import gemini_key

from .styles import page_header


def _ask(question: str) -> None:
    history = db.get_chat()
    db.add_chat("user", question)
    with st.spinner("ShieldBot is thinking ..."):
        answer, source = bot.ask(question, history)
    db.add_chat("assistant", answer)
    st.session_state["bot_source"] = source


def render(bundle) -> None:
    page_header("ShieldBot", "Your fraud assistant: ask about risk scores, flagged transactions and security", "🤖")
    if gemini_key():
        st.caption("🟢 AI mode: answers come from Gemini (key read from the server environment).")
    else:
        st.caption("🟡 Built-in mode: no GEMINI_API_KEY set, so ShieldBot uses its built-in fraud knowledge base. "
                   "See README to enable AI mode.")

    st.markdown("**Suggested questions**")
    cols = st.columns(3)
    for i, q in enumerate(bot.SUGGESTED):
        if cols[i % 3].button(q, key=f"sug_{i}", type="secondary", width="stretch"):
            _ask(q)

    history = db.get_chat()
    if not history:
        st.chat_message("assistant").write(
            "Hi, I'm ShieldBot 👋 I can explain fraud concepts, risk scores and why a transaction was flagged. "
            "Try 'why was transaction 3 flagged?' once you have analyzed some data.")
    for m in history:
        st.chat_message(m["role"]).write(m["content"])

    prompt = st.chat_input("Ask ShieldBot about fraud ...", max_chars=1000)
    if prompt:
        _ask(prompt)
        st.rerun()
    if history and st.button("Clear chat history", key="clear_chat", type="secondary"):
        db.clear_chat()
        st.rerun()
    st.caption("ShieldBot gives general guidance. It cannot confirm that a transaction is fraud.")
