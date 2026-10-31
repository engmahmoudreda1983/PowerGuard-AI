"""Look and feel of the PowerGuard AI platform: colours, CSS, small HTML components and the Plotly template."""
import base64
import io

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import pg_paths

NAVY, NAVY2, BLUE, SKY, INK, MUTED, LINE, BG = '#0b2e59', '#123f78', '#2f80ed', '#4da6ff', '#14213d', '#5b6b82', '#e3e9f2', '#f5f8fc'
GREEN, AMBER, RED, VIOLET = '#16a34a', '#e6a100', '#dc2626', '#7c3aed'
STATUS = {'EXCELLENT': ('NORMAL', GREEN, '#e8f7ee'), 'WATCH': ('WATCH', AMBER, '#fff6e0'), 'ACTION': ('ACTION', RED, '#fdecec'),
          'TEST': ('TEST', BLUE, '#e8f1fe')}

pio.templates['powerguard'] = go.layout.Template(layout=dict(
    font=dict(family='Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif', size=12, color=INK),
    colorway=[NAVY, BLUE, '#14b8a6', AMBER, RED, VIOLET, '#64748b'],
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(gridcolor='#edf1f7', zerolinecolor='#dfe6f0'), yaxis=dict(gridcolor='#edf1f7', zerolinecolor='#dfe6f0'),
    title=dict(font=dict(size=14, color=NAVY)), margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor='white', font_size=12)))

CSS = f"""
<style>
:root {{ --pg-navy:{NAVY}; --pg-blue:{BLUE}; --pg-ink:{INK}; --pg-muted:{MUTED}; --pg-line:{LINE}; --pg-bg:{BG}; }}
.stApp {{ background: linear-gradient(180deg, #f7f9fd 0%, #eef3fa 100%); }}
html, body, [class*="css"] {{ font-family: Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {NAVY} 0%, #0a2448 60%, #081b36 100%); }}
section[data-testid="stSidebar"] * {{ color: #dbe7fb !important; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{ border-radius: 10px; margin: 2px 6px; padding: 6px 10px; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {{ background: rgba(77,166,255,.16); }}
section[data-testid="stSidebar"] a[aria-current="page"] {{ background: linear-gradient(90deg, rgba(47,128,237,.55), rgba(47,128,237,.15)); }}
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {{ color: #8fb4e8 !important; letter-spacing: .08em; font-size: .72rem; text-transform: uppercase; }}
section[data-testid="stSidebar"] .stButton button {{ background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); }}
.block-container {{ padding-top: 1.6rem; max-width: 1500px; }}
.pg-hero {{ background: radial-gradient(1200px 300px at 85% -40%, rgba(77,166,255,.45), transparent 60%), linear-gradient(120deg, {NAVY} 0%, {NAVY2} 55%, #1d5fb3 100%);
           color: #fff; border-radius: 18px; padding: 22px 26px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(11,46,89,.18);
           display: flex; align-items: center; justify-content: space-between; gap: 18px; }}
.pg-hero h1 {{ margin: 0; font-size: 1.75rem; font-weight: 750; letter-spacing: -.01em; color: #fff; }}
.pg-hero p {{ margin: 6px 0 0 0; color: #bcd6ff; font-size: .98rem; }}
.pg-hero .meta {{ text-align: right; font-size: .85rem; color: #cfe1ff; }}
.pg-hero img {{ height: 58px; background: #fff; border-radius: 12px; padding: 5px 10px; }}
.pg-kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 14px 16px; box-shadow: 0 4px 14px rgba(20,33,61,.05); height: 100%; }}
.pg-kpi .lab {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.pg-kpi {{ min-height: 118px; }}
.pg-tile {{ min-height: 262px; }}
.pg-kpi .val {{ font-size: 1.9rem; font-weight: 750; margin-top: 4px; line-height: 1.1; }}
.pg-kpi .sub {{ color: {MUTED}; font-size: .82rem; margin-top: 4px; }}
.pg-tile {{ background: #fff; border: 1px solid {LINE}; border-radius: 18px; padding: 16px 18px 12px 18px; box-shadow: 0 6px 18px rgba(20,33,61,.06);
           transition: transform .12s ease, box-shadow .12s ease; height: 100%; }}
.pg-tile:hover {{ transform: translateY(-2px); box-shadow: 0 12px 26px rgba(20,33,61,.10); }}
.pg-tile .hd {{ display: flex; align-items: center; gap: 10px; }}
.pg-tile .ic {{ width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;
               background: linear-gradient(135deg, #e8f1fe, #d6e6fd); }}
.pg-tile .nm {{ font-weight: 750; color: {NAVY}; font-size: 1.05rem; }}
.pg-tile .ds {{ color: {MUTED}; font-size: .82rem; margin: 8px 0 10px 0; min-height: 2.4em; }}
.pg-bar {{ display: flex; height: 9px; border-radius: 6px; overflow: hidden; background: #eef2f7; margin: 6px 0 8px 0; }}
.pg-bar span {{ display: block; height: 100%; }}
.pg-counts {{ display: flex; gap: 12px; font-size: .8rem; color: {MUTED}; }}
.pg-counts b {{ color: {INK}; }}
.pg-worst {{ font-size: .8rem; color: {INK}; margin-top: 8px; background: {BG}; border-radius: 10px; padding: 6px 9px; }}
.pg-chip {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 700; letter-spacing: .03em; }}
.pg-section {{ display: flex; align-items: baseline; justify-content: space-between; margin: 18px 0 8px 0; }}
.pg-section .h {{ margin: 0; color: {NAVY}; font-size: 1.15rem; font-weight: 750; }}
.pg-section span {{ color: {MUTED}; font-size: .82rem; }}
.pg-card {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 14px 16px; box-shadow: 0 4px 14px rgba(20,33,61,.05); }}
.pg-row {{ display: flex; gap: 10px; align-items: flex-start; padding: 9px 2px; border-bottom: 1px solid #eef2f7; }}
.pg-row:last-child {{ border-bottom: none; }}
.pg-row .t {{ font-weight: 650; color: {INK}; font-size: .9rem; }}
.pg-row .s {{ color: {MUTED}; font-size: .8rem; }}
.pg-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-top: 6px; flex: 0 0 10px; }}
.pg-foot {{ text-align: center; color: #8a97ab; font-size: 12px; padding: 22px 10px 6px 10px; }}
.pg-login {{ max-width: 430px; margin: 7vh auto 0 auto; background: #fff; border-radius: 22px; padding: 28px 30px 10px 30px;
            box-shadow: 0 24px 60px rgba(11,46,89,.18); border: 1px solid {LINE}; text-align: center; }}
.pg-login img {{ height: 70px; }}
.pg-login h2 {{ color: {NAVY}; margin: 10px 0 2px 0; font-weight: 800; }}
.pg-login p {{ color: {MUTED}; margin: 0 0 6px 0; font-size: .92rem; }}
[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 10px 14px; }}
</style>
"""

FOOTER = ("<div class='pg-foot'><b>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim.</b> All rights reserved. "
          "Confidential and proprietary.</div>")


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def logo_b64(h=200):
    try:
        from PIL import Image
        im = Image.open(pg_paths.LOGO); im.thumbnail((480, h))
        buf = io.BytesIO(); im.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ''


@st.cache_data(show_spinner=False)
def small_logo_path():
    """A small copy of the logo for the sidebar (the original file is large)."""
    import os
    import tempfile
    from PIL import Image
    p = os.path.join(tempfile.gettempdir(), 'powerguard_logo_small.png')
    if not os.path.exists(p):
        im = Image.open(pg_paths.LOGO); im.thumbnail((420, 160)); im.save(p)
    return p


def hero(title, sub, meta=''):
    lg = logo_b64()
    img = f"<img src='data:image/png;base64,{lg}'>" if lg else ''
    st.markdown(f"<div class='pg-hero'><div><h1>{title}</h1><p>{sub}</p></div><div class='meta'>{img}<div style='margin-top:6px'>{meta}</div></div></div>",
                unsafe_allow_html=True)


def chip(status, text=None):
    lab, col, bg = STATUS.get(status, (status, MUTED, '#eef2f7'))
    return f"<span class='pg-chip' style='color:{col};background:{bg};border:1px solid {col}33'>{text or lab}</span>"


def kpi(col, label, value, sub='', color=NAVY):
    col.markdown(f"<div class='pg-kpi'><div class='lab'>{label}</div><div class='val' style='color:{color}'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def section(title, note=''):
    st.markdown(f"<div class='pg-section'><div class='h'>{title}</div><span>{note}</span></div>", unsafe_allow_html=True)


def band(risk):
    return 'EXCELLENT' if risk <= 35 else ('WATCH' if risk <= 70 else 'ACTION')
