"""Theme, CSS and small reusable UI components (dark cybersecurity look).

Note: Streamlit does not run <script> tags inside st.markdown, so animations
(count-up bars, glow, hover lift) are done in pure CSS.
"""
from __future__ import annotations

import html

import streamlit as st

COLORS = {"bg": "#050816", "blue": "#3b82f6", "cyan": "#22d3ee", "purple": "#8b5cf6", "red": "#ef4444",
          "orange": "#f59e0b", "green": "#10b981", "text": "#e5e9ff", "muted": "#8b94b8"}
LEVEL_COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}
ACCENTS = {"blue": ("#3b82f6", "#22d3ee"), "purple": ("#8b5cf6", "#3b82f6"), "red": ("#ef4444", "#f97316"),
           "green": ("#10b981", "#22d3ee"), "orange": ("#f59e0b", "#ef4444")}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(59,130,246,.18), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,.16), transparent 60%),
                     linear-gradient(180deg, #050816 0%, #070b1f 100%); color: #e5e9ff; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3, h4 { color: #f1f4ff; letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #080d26 0%, #050816 100%);
                            border-right: 1px solid rgba(139,148,184,.18); }
.brand { display:flex; align-items:center; gap:.7rem; padding:.4rem .2rem 1rem; }
.brand .logo { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
               font-size:1.4rem; background:linear-gradient(135deg,#3b82f6,#8b5cf6); box-shadow:0 0 22px rgba(99,102,241,.55); }
.brand .name { font-weight:800; font-size:1.2rem; color:#fff; line-height:1.1; }
.brand .tag { font-size:.7rem; color:#8b94b8; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:.25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.55rem .8rem; border-radius:12px; border:1px solid transparent;
        transition: all .2s ease; width:100%; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(59,130,246,.12); border-color: rgba(59,130,246,.35); transform: translateX(3px); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: linear-gradient(90deg, rgba(59,130,246,.28), rgba(139,92,246,.22));
        border-color: rgba(99,102,241,.6); box-shadow: 0 0 18px rgba(99,102,241,.25); }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none; }
.page-head { display:flex; align-items:center; gap:1rem; margin-bottom:1.2rem; }
.page-head .ico { font-size:1.7rem; width:54px; height:54px; display:flex; align-items:center; justify-content:center; border-radius:16px;
        background: linear-gradient(135deg, rgba(59,130,246,.35), rgba(139,92,246,.35)); border:1px solid rgba(139,148,184,.25); }
.page-head h1 { margin:0; font-size:1.75rem; font-weight:800; }
.page-head p { margin:.1rem 0 0; color:#8b94b8; font-size:.95rem; }
.glass { background: rgba(16,22,52,.62); border:1px solid rgba(139,148,184,.2); border-radius:18px; padding:1.1rem 1.25rem;
         backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); box-shadow: 0 8px 30px rgba(0,0,0,.35);
         transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; }
.glass:hover { transform: translateY(-3px); border-color: rgba(99,102,241,.55); box-shadow: 0 14px 38px rgba(59,130,246,.18); }
.kpi { position:relative; overflow:hidden; min-height:118px; }
.kpi::before { content:""; position:absolute; inset:0 auto 0 0; width:4px; background: linear-gradient(180deg, var(--a), var(--b)); }
.kpi .label { color:#8b94b8; font-size:.74rem; font-weight:600; letter-spacing:.07em; text-transform:uppercase; }
.kpi .value { font-size:1.9rem; font-weight:800; margin:.25rem 0 .1rem; background:linear-gradient(90deg,#fff,var(--b));
              -webkit-background-clip:text; background-clip:text; color:transparent; animation: rise .6s ease both; }
.kpi .sub { color:#8b94b8; font-size:.78rem; }
@keyframes rise { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform:none; } }
.badge { display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.75rem; font-weight:700; border:1px solid; }
.verdict { text-align:center; padding:1.4rem 1rem; border-radius:20px; border:1px solid; animation: rise .5s ease both; }
.verdict .big { font-size:1.9rem; font-weight:800; letter-spacing:.02em; }
.verdict .small { color:#c7cdea; font-size:.9rem; }
.meter { height:16px; border-radius:999px; background: rgba(139,148,184,.2); overflow:hidden; margin:.6rem 0 .2rem; }
.meter > span { display:block; height:100%; width:var(--w); border-radius:999px; background:var(--c);
                box-shadow:0 0 16px var(--c); animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both; }
@keyframes grow { from { width:0; } to { width:var(--w); } }
.alert { border-radius:14px; padding:.7rem .95rem; margin:.4rem 0; border:1px solid; background: rgba(239,68,68,.08); animation: rise .4s ease both; }
.alert b { color:#fff; } .alert .meta { color:#8b94b8; font-size:.78rem; }
.pulse { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.45rem; animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(16,185,129,.7);} 70% { box-shadow:0 0 0 9px rgba(16,185,129,0);} 100% { box-shadow:0 0 0 0 rgba(16,185,129,0);} }
.empty { text-align:center; padding:2.2rem 1rem; color:#8b94b8; }
.empty h3 { color:#e5e9ff; margin-bottom:.3rem; }
.note { color:#8b94b8; font-size:.8rem; }
.disclaimer { border:1px dashed rgba(245,158,11,.5); background:rgba(245,158,11,.07); color:#f6d58e; border-radius:12px;
              padding:.55rem .8rem; font-size:.78rem; margin-top:.8rem; }
.stButton > button, .stDownloadButton > button { border-radius:12px; border:1px solid rgba(99,102,241,.55); font-weight:600;
        background: linear-gradient(135deg, rgba(59,130,246,.9), rgba(139,92,246,.9)); color:#fff; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow:0 8px 24px rgba(99,102,241,.45); border-color:#a5b4fc; color:#fff; }
.stButton > button[kind="secondary"] { background: rgba(16,22,52,.7); }
div[data-testid="stTabs"] button[role="tab"] { font-weight:600; }
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid rgba(139,148,184,.2); }
div[data-baseweb="input"], div[data-baseweb="select"] > div, textarea { border-radius:12px !important; }
@media (max-width: 768px) {
  .block-container { padding-left:.8rem; padding-right:.8rem; }
  .page-head h1 { font-size:1.35rem; } .kpi .value { font-size:1.5rem; } .verdict .big { font-size:1.4rem; }
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = "🛡️") -> None:
    st.markdown(f'<div class="page-head"><div class="ico">{icon}</div><div><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div></div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", accent: str = "blue") -> str:
    a, b = ACCENTS.get(accent, ACCENTS["blue"])
    return (f'<div class="glass kpi" style="--a:{a};--b:{b}"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(kpi_card(*item), unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return (f'<span class="badge" style="color:{color};border-color:{color};background:{color}22">'
            f'{html.escape(text)}</span>')


def level_badge(level: str) -> str:
    return badge(level, LEVEL_COLORS.get(level, "#8b94b8"))


def risk_meter(score: float, level: str) -> None:
    c = LEVEL_COLORS.get(level, "#3b82f6")
    st.markdown(f'<div class="glass"><div class="kpi"><div class="label" style="color:#8b94b8;font-size:.74rem;'
                f'font-weight:600;letter-spacing:.07em">RISK SCORE</div></div>'
                f'<div style="font-size:2.4rem;font-weight:800;color:{c}">{score:.1f}'
                f'<span style="font-size:1rem;color:#8b94b8"> / 100</span></div>'
                f'<div class="meter" style="--w:{min(max(score, 0), 100)}%;--c:{c}"><span></span></div>'
                f'<div>{level_badge(level)}</div></div>', unsafe_allow_html=True)


def verdict_card(prediction: str, level: str) -> None:
    fraud = prediction == "Fraudulent"
    c = "#ef4444" if fraud else "#10b981"
    icon = "🚨" if fraud else "✅"
    sub = "This transaction looks suspicious." if fraud else "This transaction looks normal."
    st.markdown(f'<div class="verdict" style="border-color:{c};background:{c}1f;box-shadow:0 0 30px {c}33">'
                f'<div style="font-size:2.2rem">{icon}</div><div class="big" style="color:{c}">'
                f'{prediction.upper()}</div><div class="small">{sub}</div></div>', unsafe_allow_html=True)


def empty_state(title: str, text: str, icon: str = "📭") -> None:
    st.markdown(f'<div class="glass empty"><div style="font-size:2.2rem">{icon}</div><h3>{html.escape(title)}</h3>'
                f'<div>{html.escape(text)}</div></div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown('<div class="disclaimer">⚠️ Decision-support tool. Predictions are statistical estimates, '
                'not proof of fraud. Always review flagged transactions.</div>', unsafe_allow_html=True)


def style_fig(fig, height: int = 330):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", color="#c7cdea"),
                      colorway=["#3b82f6", "#8b5cf6", "#22d3ee", "#ef4444", "#f59e0b", "#10b981"],
                      legend=dict(orientation="h", y=1.12, x=0), title=dict(font=dict(size=15)),
                      transition=dict(duration=500))
    fig.update_xaxes(gridcolor="rgba(139,148,184,.12)")
    fig.update_yaxes(gridcolor="rgba(139,148,184,.12)")
    return fig
