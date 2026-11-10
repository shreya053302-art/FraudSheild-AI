"""FraudShield AI - entry point.  Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from backend import database as db  # noqa: E402
from backend.config import MODEL_PATH  # noqa: E402
from backend.prediction import load_bundle  # noqa: E402
from frontend import (analytics_page, assistant, bulk_analysis, dashboard, detection,  # noqa: E402
                      investigation, model_center, settings)
from frontend.common import PAGES  # noqa: E402
from frontend.styles import disclaimer, inject_css  # noqa: E402


@st.cache_resource(show_spinner=False)
def _cached_bundle(mtime: float):
    return load_bundle()


def get_bundle():
    return _cached_bundle(MODEL_PATH.stat().st_mtime) if MODEL_PATH.exists() else None


def main() -> None:
    inject_css()
    db.init_db()
    bundle = get_bundle()

    st.session_state.setdefault("nav", PAGES[0])
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">🛡️</div><div><div class="name">FraudShield AI</div>'
                    '<div class="tag">Transaction Intelligence</div></div></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        if bundle:
            st.markdown(f'<span class="pulse" style="background:#10b981"></span>Model online · {bundle["selected"]}',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="pulse" style="background:#f59e0b"></span>No trained model', unsafe_allow_html=True)
        st.caption("Database online" if db.db_size_bytes() else "Database initialising")
        disclaimer()

    routes = {
        PAGES[0]: dashboard.render, PAGES[1]: detection.render, PAGES[2]: bulk_analysis.render,
        PAGES[3]: investigation.render, PAGES[4]: analytics_page.render, PAGES[5]: assistant.render,
        PAGES[6]: model_center.render, PAGES[7]: settings.render,
    }
    try:
        routes[page](bundle)
    except Exception as exc:  # never show a blank page
        st.error(f"Something went wrong on this page: {exc}")
        st.caption("Try reloading the page. If it keeps happening, check the terminal for details.")
        raise


main()
