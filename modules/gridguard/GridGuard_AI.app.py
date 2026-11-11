import hmac
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pg_paths
import gridguard_core as gc
import gridguard_report_pdf as pdfrep
import train_gridguard_model as tg

pass  # page config is set once by the PowerGuard AI platform
STRETCH = "stretch"
NAVY, BLUE = "#0b2e59", "#4da6ff"
GREEN, AMBER, RED = "#1a9e5c", "#d99a00", "#d63a3a"
STATUS_COLOR = {'EXCELLENT': GREEN, 'WATCH': AMBER, 'ACTION': RED}
STATUS_LABEL = {'EXCELLENT': 'NORMAL (Low Failure Risk)', 'WATCH': 'WATCH (Elevated Failure Risk)',
                'ACTION': 'ACTION REQUIRED (High Failure Risk)'}
MODE_ORDER = gc.MODE_ORDER
DEMO_ORIGIN = '2026-08-10'
STUDY_START, STUDY_END = '2025-05-01', '2026-08-24'

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:16px 22px;border-radius:10px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.pg-header img {height:64px;background:#fff;border-radius:8px;padding:4px 8px;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-banner {border:2px solid;border-radius:8px;padding:10px 14px;text-align:center;margin:6px 0 12px 0;}
.pg-banner h2 {margin:0;font-size:1.5rem;}
.pg-banner p {margin:2px 0 0 0;font-size:.95rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;margin-bottom:10px;}
.pg-card {border:1px solid #d5dce6;border-radius:10px;padding:12px 14px;text-align:center;}
.pg-card h4 {margin:0;color:#0b2e59;font-size:1.05rem;}
.pg-card .big {font-size:1.7rem;font-weight:700;margin:4px 0;}
.pg-card .sub {font-size:.85rem;color:#5a6575;}
.pg-foot {text-align:center;color:#888;font-size:12px;padding:10px;}
</style>
""", unsafe_allow_html=True)

FOOTER = """
<div class='pg-foot'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
<em>Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.</em></div>
"""


def banner(title, sub, color):
    st.markdown(f"<div class='pg-banner' style='border-color:{color}'><h2 style='color:{color}'>{title}</h2>"
                f"<p>{sub}</p></div>", unsafe_allow_html=True)


def secrets_ok():
    try:
        return "APP_USERNAME" in st.secrets and "APP_PASSWORD" in st.secrets
    except Exception:
        return False


# ==========================================
# --- 1. Login ---
# ==========================================
for key, default in [('logged_in', False), ('register', []), ('hist', None), ('fc_last', None), ('th_last', None), ('model_user', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align:center;color:#4da6ff;'>⚡ GridGuard AI Login</h1>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", width=STRETCH):
                if not secrets_ok():
                    st.error("Login is not configured: add APP_USERNAME and APP_PASSWORD to the app's Secrets.")
                elif hmac.compare_digest(username, st.secrets["APP_USERNAME"]) and \
                        hmac.compare_digest(password, st.secrets["APP_PASSWORD"]):
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()

# ==========================================
# --- 2. Header and data ---
# ==========================================
_, col_r = st.columns([10, 1])
with col_r:
    if False:  # sign out is in the platform header and sidebar
        st.session_state['logged_in'] = False
        st.rerun()


@st.cache_data
def logo_b64():
    try:
        import base64, io
        from PIL import Image
        im = Image.open(pg_paths.LOGO)
        im.thumbnail((360, 200))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


_logo = logo_b64()
_logo_html = f"<img src='data:image/png;base64,{_logo}' alt='PowerGuard AI'>" if _logo else ""
st.markdown(f"""
<div class='pg-header'>
  <div>
    <h1>⚡ GridGuard AI: Transformer Load and Thermal Forecasting</h1>
    <p>AI forecast of every transformer's hourly load up to 7 days ahead, turned into hot-spot temperature, insulation ageing and overload risk</p>
  </div>
  {_logo_html}
</div>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the AI model…")
def load_file_model():
    return gc.load_model(pg_paths.model('gridguard_model.pkl'))


def MODEL():
    return st.session_state['model_user'] or load_file_model()


if MODEL() is None:
    st.error("AI model file 'gridguard_model.pkl' not found.")
    st.stop()


@st.cache_data(show_spinner=False)
def demo_full():
    return gc.demo_data()


@st.cache_data(show_spinner=False)
def demo_dcs_csv(tid='TR-2', days=60):
    L = demo_full()['loads']
    L = L[L['Transformer ID'] == tid]
    L = L[pd.to_datetime(L['Timestamp']) < pd.Timestamp(DEMO_ORIGIN)].tail(24 * days)
    return gc.demo_power(L).to_csv(index=False).encode('utf-8-sig')


def demo_history(tid):
    d = demo_full()
    L = d['loads'][d['loads']['Transformer ID'] == tid]
    return pd.to_datetime(L['Timestamp']), L['Load (MVA)'].to_numpy(float)


@st.cache_data(show_spinner=False)
def demo_weather_fc(origin, days=gc.HORIZON_DAYS):
    return gc.weather_forecast_from(demo_full()['weather'], pd.Timestamp(origin), days)


def run_forecast(bundle, reg, hts, hl, tts, tv, fts, fv, origin, days):
    return gc.forecast(bundle, reg, hts, hl, tts, tv, fts, fv, origin, days)


@st.cache_data(show_spinner="Forecasting the demo substation…")
def demo_snapshot():
    d = demo_full(); w = d['weather']; o = pd.Timestamp(DEMO_ORIGIN)
    wf = demo_weather_fc(DEMO_ORIGIN)
    out = []
    for r in d['register'].to_dict('records'):
        hts, hl = demo_history(r['Transformer ID'])
        m = hts < o
        wp = w[pd.to_datetime(w['Timestamp']) < o]
        fc = run_forecast(MODEL(), r, hts[m], hl[m], pd.to_datetime(wp['Timestamp']), wp['Ambient (C)'], pd.to_datetime(wf['Timestamp']),
                          wf['Ambient (C)'], o, gc.HORIZON_DAYS)
        out.append(to_record(gc.assess(r, fc), 'Demo substation'))
    return out


# ==========================================
# --- 3. Charts ---
# ==========================================
def gauge(value, color, title='Failure Risk (%)', height=290):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), number={'suffix': '%', 'font': {'size': 34}, 'valueformat': '.0f'},
        title={'text': title, 'font': {'size': 15}}, domain={'x': [0, 1], 'y': [0, 0.85]},
        gauge={'axis': {'range': [0, 100], 'tickvals': [0, 35, 70, 100], 'tickfont': {'size': 12}},
               'bar': {'color': color, 'thickness': 0.3},
               'steps': [{'range': [0, 35], 'color': 'rgba(26,158,92,.25)'},
                         {'range': [35, 70], 'color': 'rgba(217,154,0,.25)'},
                         {'range': [70, 100], 'color': 'rgba(214,58,58,.25)'}]}))
    fig.update_layout(height=height, margin=dict(l=35, r=35, t=40, b=10))
    return fig


def band_color(v):
    return GREEN if v <= gc.EXCELLENT_MAX else AMBER if v <= gc.WATCH_MAX else RED


def rpn_color(r):
    return GREEN if r < 40 else AMBER if r < 100 else RED


def mode_plot(risks, title='Risk by failure mode (risk drivers)', height=280):
    names = sorted(risks, key=lambda k: risks[k])
    vals = [risks[n] for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=[band_color(v) for v in vals],
                           text=[f"{v:.0f}%" for v in vals], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x in (35, 70):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title=title, height=height, xaxis=dict(range=[0, 115], title='Risk (%)'), margin=dict(l=10, r=20, t=45, b=40))
    return fig


def rpn_plot(a, height=230):
    rpn = int(a['RPN'])
    fig = go.Figure()
    for x0, x1, c, t in [(0, 40, 'rgba(26,158,92,.18)', 'Low'), (40, 100, 'rgba(217,154,0,.18)', 'Medium'),
                         (100, 200, 'rgba(214,58,58,.18)', 'High')]:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=-0.5, y1=0.5, fillcolor=c, line_width=0, layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=t, showarrow=False, font=dict(size=11, color='#555'))
    fig.add_bar(x=[rpn], y=[0], orientation='h', width=0.45, marker_color=rpn_color(rpn),
                text=[f"RPN {rpn} = S {int(a['S'])} × O {int(a['O'])} × D {int(a['D'])}"],
                textposition='inside' if rpn > 110 else 'outside', insidetextanchor='middle', cliponaxis=False,
                hoverinfo='skip', textfont=dict(color='white' if rpn > 110 else '#222'))
    fig.update_layout(title='RPN of the governing mode (max 200)', height=height, showlegend=False,
                      xaxis=dict(range=[0, 200], tickvals=[0, 40, 100, 200], title='RPN'),
                      yaxis=dict(visible=False, range=[-0.6, 0.8]), margin=dict(l=10, r=10, t=45, b=40))
    return fig


def fmea_rpn_plot(fmea, height=250):
    t = sorted(fmea, key=lambda m: m['RPN'])
    fig = go.Figure(go.Bar(x=[m['RPN'] for m in t], y=[m['Failure Mode'] for m in t], orientation='h',
                           marker_color=[rpn_color(m['RPN']) for m in t], text=[str(m['RPN']) for m in t],
                           textposition='outside', cliponaxis=False, hovertemplate='%{y}: RPN %{x}<extra></extra>'))
    for x in (40, 100):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title='RPN by failure mode', height=height, xaxis=dict(range=[0, 200], title='RPN'), margin=dict(l=10, r=20, t=45, b=40))
    return fig


def forecast_plot(fc, hist_ts=None, hist_load=None, hs_limit=110, height=560, actual=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=['Load (MVA)', 'Temperatures (°C)'])
    ts = pd.DatetimeIndex(fc['ts'])
    if hist_ts is not None and len(hist_ts):
        fig.add_scatter(x=hist_ts, y=hist_load, mode='lines', name='Measured load', line=dict(color='#7d8ca3', width=1.2), row=1, col=1)
    fig.add_scatter(x=ts, y=fc['load_hi'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip', row=1, col=1)
    fig.add_scatter(x=ts, y=fc['load_lo'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(77,166,255,.22)',
                    name='80% band', hoverinfo='skip', row=1, col=1)
    fig.add_scatter(x=ts, y=fc['load'], mode='lines', name='AI forecast', line=dict(color=NAVY, width=2), row=1, col=1)
    if actual is not None:
        fig.add_scatter(x=ts, y=actual, mode='lines', name='Actual (for comparison)', line=dict(color=GREEN, width=1.2, dash='dot'), row=1, col=1)
    fig.add_hline(y=fc['rating'], line_dash='dash', line_color=RED, annotation_text=f"Rating {fc['rating']:g} MVA", annotation_position='top left', row=1, col=1)
    fig.add_scatter(x=ts, y=fc['hot_spot'], mode='lines', name='Hot-spot', line=dict(color=RED, width=2), row=2, col=1)
    fig.add_scatter(x=ts, y=fc['top_oil'], mode='lines', name='Top oil', line=dict(color=AMBER, width=1.6), row=2, col=1)
    fig.add_scatter(x=ts, y=fc['ambient'], mode='lines', name='Ambient (forecast)', line=dict(color='#2a78d6', width=1.2), row=2, col=1)
    fig.add_hline(y=hs_limit, line_dash='dash', line_color=AMBER, annotation_text=f'{hs_limit:.0f} °C', annotation_position='top left', row=2, col=1)
    fig.add_hline(y=130, line_dash='dash', line_color=RED, annotation_text='130 °C planned limit', annotation_position='top left', row=2, col=1)
    fig.add_vline(x=ts[0], line_dash='dot', line_color='#555')
    fig.update_layout(height=height, legend=dict(orientation='h', y=-0.08), margin=dict(l=50, r=20, t=40, b=40), hovermode='x unified')
    return fig


def ranking_plot(R, top=20):
    t = R.sort_values('Overall Risk (%)', ascending=False).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t['Overall Risk (%)'], y=t['Transformer ID'], orientation='h', marker_color=[STATUS_COLOR[s] for s in t['Status']],
                           text=[f"{r:.0f}%  ·  {g}" for r, g in zip(t['Overall Risk (%)'], t['Governing Mode'])],
                           textposition='outside', cliponaxis=False, hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x, tx in [(35, 'Watch'), (70, 'Action')]:
        fig.add_vline(x=x, line_dash='dash', line_color='#888', annotation_text=tx, annotation_position='top')
    fig.update_layout(title=f"Transformer risk ranking (next {gc.HORIZON_DAYS} days)", height=140 + 34 * len(t),
                      xaxis=dict(range=[0, 140], title='Failure Risk (%)'), yaxis=dict(type='category'), margin=dict(l=10, r=30, t=60, b=40))
    return fig


def heat_plot(R):
    t = R.sort_values('Overall Risk (%)', ascending=False)
    z = [[r['Mode Risks'][m] for m in MODE_ORDER] for _, r in t.iterrows()]
    fig = go.Figure(go.Heatmap(z=z, x=MODE_ORDER, y=list(t['Transformer ID']), zmin=0, zmax=100,
                               colorscale=[[0, '#e8f5ee'], [0.35, '#bfe3cd'], [0.351, '#fbe8b3'], [0.7, '#f2c14e'], [0.701, '#f2a6a6'], [1, RED]],
                               text=[[f"{v:.0f}" for v in row] for row in z], texttemplate='%{text}',
                               hovertemplate='%{y} · %{x}: %{z:.0f}%<extra></extra>', colorbar=dict(title='%', thickness=12)))
    fig.update_layout(title='Risk by transformer and failure mode', height=150 + 34 * len(t), yaxis=dict(type='category', autorange='reversed'),
                      margin=dict(l=10, r=10, t=50, b=40))
    return fig


def gov_plot(R, height=320):
    above = R[R['Status'] != 'EXCELLENT']
    fig = go.Figure()
    for stt, name in [('WATCH', 'Watch'), ('ACTION', 'Action')]:
        fig.add_bar(x=MODE_ORDER, y=[int(((above['Governing Mode'] == m) & (above['Status'] == stt)).sum()) for m in MODE_ORDER],
                    name=name, marker_color=STATUS_COLOR[stt])
    fig.update_layout(barmode='stack', title='Transformers above normal by governing failure mode', height=height,
                      yaxis=dict(title='Transformers', dtick=1), legend=dict(orientation='h', y=-0.3), margin=dict(l=40, r=10, t=50, b=40))
    return fig


def peak_plot(R, height=320):
    t = R.sort_values('Peak Load (%)')
    fig = go.Figure(go.Bar(x=t['Peak Load (%)'], y=t['Transformer ID'], orientation='h',
                           marker_color=[RED if v > 100 else AMBER if v > 90 else GREEN for v in t['Peak Load (%)']],
                           text=[f"{p:.0f}% · {h:.0f} °C" for p, h in zip(t['Peak Load (%)'], t['Max Hot-Spot (C)'])], textposition='outside', cliponaxis=False))
    fig.add_vline(x=100, line_dash='dash', line_color=RED, annotation_text='Rating', annotation_position='top')
    fig.update_layout(title='Forecast peak load (% of rating) and hot-spot', height=height, xaxis=dict(range=[0, 140], title='% of rating'),
                      yaxis=dict(type='category'), margin=dict(l=10, r=30, t=50, b=40))
    return fig


def fmea_table(a):
    return pd.DataFrame([{'Failure Mode': m['Failure Mode'], 'Risk (%)': round(m['Risk (%)']), 'S': m['S'], 'O': m['O'], 'D': m['D'],
                          'RPN': m['RPN'], 'RPN Level': m['RPN Level']} for m in a['FMEA']])


def hourly_table(fc):
    return pd.DataFrame({'Timestamp': pd.DatetimeIndex(fc['ts']).strftime('%Y-%m-%d %H:%M'), 'Ambient (C)': np.round(fc['ambient'], 1),
                         'Load (MVA)': np.round(fc['load'], 2), 'Load Low (MVA)': np.round(fc['load_lo'], 2), 'Load High (MVA)': np.round(fc['load_hi'], 2),
                         'Load (% of rating)': np.round(fc['K'] * 100, 1), 'Top Oil (C)': np.round(fc['top_oil'], 1),
                         'Hot-Spot (C)': np.round(fc['hot_spot'], 1), 'Ageing Factor': np.round(fc['faa'], 3)})


REC_KEYS = ['Transformer ID', 'Bay', 'Load Type', 'Cooling', 'Rating (MVA)', 'Peak Load (MVA)', 'Peak Load (%)', 'Max Hot-Spot (C)', 'Max Top-Oil (C)',
            'Max Ambient (C)', 'Hours Above Rating', 'Worst Day Ageing (x)', 'Load Transfer Needed (MVA)', 'Governing Mode', 'Overall Risk (%)',
            'Status', 'S', 'O', 'D', 'RPN', 'RPN Level', 'Action', 'Mode Risks', 'FMEA']


def to_record(a, source):
    rec = {k: a.get(k) for k in REC_KEYS}
    rec['Peak Time'] = pd.Timestamp(a['Peak Time']).strftime('%Y-%m-%d %H:%M')
    rec['Forecast From'] = pd.Timestamp(a['Forecast From']).strftime('%Y-%m-%d %H:%M')
    rec['Source'] = source
    rec['Assessed'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    rec['Findings'] = gc.findings(a)
    return rec


SRC_PAGES, SRC_DEMO = "Results from the analysis pages", "Demo substation (synthetic)"


def current_records():
    if st.session_state.get('ov_src', SRC_PAGES) == SRC_DEMO:
        return demo_snapshot()
    return list(st.session_state['register'])


def read_csv(label, key, need):
    f = st.file_uploader(label, type='csv', key=key)
    if f is None:
        return None, None
    try:
        df = pd.read_csv(f)
        if 'Load (MVA)' in need:
            df, _ = gc.loads_from_power(df)
    except Exception as e:
        return None, f"Cannot read the file: {e}"
    miss = [c for c in need if c not in df.columns]
    return (None, f"Missing columns: {', '.join(miss)}") if miss else (df, None)


NEED = {'register': ['Transformer ID', 'Rating (MVA)', 'Cooling', 'Load Type'], 'loads': ['Timestamp', 'Transformer ID', 'Load (MVA)'],
        'weather': ['Timestamp', 'Ambient (C)']}


# ==========================================
# --- 4. Forecast page ---
# ==========================================
DEF = {'id': 'TR-7', 'bay': 'Feeder group C', 'type': 'Residential', 'rating': 50.0, 'cool': 'ONAF', 'to': 50.0, 'hs': 30.0,
       'hsrc': 'Demo transformer history', 'dtr': 'TR-2', 'wsrc': 'Normal climate of a Gulf city', 'city': 'Dammam', 'shift': 0.0,
       'start': date(2026, 8, 10), 'days': 7, 'lim': 110.0, 'rmv': 0.0, 'rlv': 0.0}


def init_fp():
    for k, v in DEF.items():
        st.session_state.setdefault(f'fp_{k}', v)


def fill_demo_tr():
    tid = st.session_state.get('fp_ex')
    if not tid or tid == '—':
        return
    r = demo_full()['register'].set_index('Transformer ID').loc[tid]
    ss = st.session_state
    ss['fp_id'] = tid; ss['fp_bay'] = r['Bay']; ss['fp_type'] = r['Load Type']; ss['fp_rating'] = float(r['Rating (MVA)'])
    ss['fp_cool'] = r['Cooling']; ss['fp_to'] = float(r['Top-Oil Rise (K)']); ss['fp_hs'] = float(r['Hot-Spot Gradient (K)'])
    ss['fp_hsrc'] = 'Demo transformer history'; ss['fp_dtr'] = tid; ss['fp_wsrc'] = 'Demo weather forecast'
    ss['fp_start'] = pd.Timestamp(DEMO_ORIGIN).date()


def set_cooling_defaults():
    p = gc.COOLING[st.session_state['fp_cool']]
    st.session_state['fp_to'] = p['to_rise']; st.session_state['fp_hs'] = p['hs_grad']


def forecast_page():
    init_fp()
    ss = st.session_state
    left, right = st.columns([1, 1.8])
    with left:
        st.markdown("#### ⚡ Transformer and forecast inputs")
        st.selectbox("Fill with a demo transformer (optional)", ['—'] + list(demo_full()['register']['Transformer ID']), key='fp_ex',
                     on_change=fill_demo_tr, format_func=lambda x: x if x == '—' else f"{x} (forecast from {DEMO_ORIGIN})")
        st.markdown("**Transformer data**")
        c1, c2 = st.columns(2)
        c1.text_input("Transformer ID", key='fp_id'); c2.text_input("Bay / area", key='fp_bay')
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Load type", gc.LOAD_TYPES, key='fp_type', help="Shapes the daily profile the AI expects.")
        c2.number_input("Rating (MVA)", 1.0, 2000.0, key='fp_rating')
        c3.selectbox("Cooling", list(gc.COOLING), key='fp_cool', on_change=set_cooling_defaults)
        c1, c2 = st.columns(2)
        c1.number_input("Top-oil rise at rated load (K)", 20.0, 80.0, key='fp_to', help="From the heat-run test report; IEEE C57.91 typical value by cooling if unknown.")
        c2.number_input("Hot-spot gradient at rated load (K)", 10.0, 50.0, key='fp_hs')
        st.markdown("**Load history (at least the last 7 days, hourly)**")
        st.radio("Source", ['Demo transformer history', 'Upload CSV'], key='fp_hsrc', horizontal=True)
        up_hist = None
        if ss['fp_hsrc'] == 'Upload CSV':
            up_hist = st.file_uploader("Timestamp + either Load (MVA) or MW and MVAR from the DCS (HV, MV, LV sides; e.g. 'HV MW', 'HV MVAR')"
                                       "  (optional: Transformer ID, Ambient (C))", type='csv', key='fp_hfile')
            c1, c2 = st.columns(2)
            c1.number_input("MV winding rating (MVA, 0 = not used)", 0.0, 2000.0, key='fp_rmv')
            c2.number_input("LV winding rating (MVA, 0 = not used)", 0.0, 2000.0, key='fp_rlv')
            st.download_button("📥 Demo DCS file (MW / MVAR, HV-MV-LV) for TR-2", demo_dcs_csv(), "GridGuard_demo_DCS_TR-2.csv", 'text/csv', key='fp_dcs')
        else:
            st.selectbox("Demo transformer", list(demo_full()['register']['Transformer ID']), key='fp_dtr')
        st.markdown("**Weather forecast (hourly ambient)**")
        st.radio("Source", ['Normal climate of a Gulf city', 'Upload forecast CSV', 'Demo weather forecast'], key='fp_wsrc')
        up_w = None
        if ss['fp_wsrc'] == 'Normal climate of a Gulf city':
            c1, c2 = st.columns(2)
            c1.selectbox("City", list(gc.CLIMATE), key='fp_city')
            c2.number_input("Heat offset (°C)", -10.0, 10.0, step=0.5, key='fp_shift', help="Add degrees for a forecast heat wave.")
        elif ss['fp_wsrc'] == 'Upload forecast CSV':
            up_w = st.file_uploader("Timestamp, Ambient (C)", type='csv', key='fp_wfile')
        st.markdown("**Forecast**")
        c1, c2, c3 = st.columns(3)
        c1.date_input("Start (00:00)", key='fp_start', help="Normally tomorrow; the history must end before it.")
        c2.number_input("Days ahead", 1, 7, key='fp_days')
        c3.number_input("Hot-spot limit (°C)", 90.0, 140.0, step=5.0, key='fp_lim', help="Used for the load transfer needed. IEEE C57.91: 110 °C normal.")
        go_btn = st.button("🔍 Analyze Transformer", width=STRETCH, key='fp_go', type='primary')
    with right:
        if go_btn:
            try:
                res = collect_and_forecast(up_hist, up_w)
                ss['fc_last'] = res
                a = res['a']
                ss['register'] = [r for r in ss['register'] if r['Transformer ID'] != a['Transformer ID']] + [to_record(a, 'Forecast page')]
            except Exception as e:
                st.error(f"Cannot run the forecast: {e}")
                ss['fc_last'] = None
        res = ss.get('fc_last')
        if res is None:
            st.markdown("<div class='pg-note'>Enter the transformer data, its recent hourly load and the weather forecast on the left "
                        "(or fill them from a demo transformer), then press <b>Analyze Transformer</b>. The AI forecasts the hourly load; the IEEE "
                        "C57.91 thermal model turns it into top-oil and hot-spot temperatures and insulation ageing. The result is added to the "
                        "Substation Overview and the Combined Report.</div>", unsafe_allow_html=True)
        else:
            show_forecast(res)


def collect_and_forecast(up_hist, up_w):
    ss = st.session_state
    reg = {'Transformer ID': ss['fp_id'].strip() or 'TR', 'Bay': ss['fp_bay'], 'Load Type': ss['fp_type'], 'Rating (MVA)': float(ss['fp_rating']),
           'Cooling': ss['fp_cool'], 'Top-Oil Rise (K)': float(ss['fp_to']), 'Hot-Spot Gradient (K)': float(ss['fp_hs'])}
    origin = pd.Timestamp(ss['fp_start']); days = int(ss['fp_days'])
    tts = tv = None
    dcs_sides = None
    if ss['fp_hsrc'] == 'Upload CSV':
        if up_hist is None:
            raise ValueError("upload the load history CSV.")
        df, sides = gc.loads_from_power(pd.read_csv(up_hist))
        if 'Timestamp' not in df or 'Load (MVA)' not in df:
            raise ValueError("the load history needs Timestamp and either Load (MVA) or MW and MVAR columns (e.g. 'HV MW', 'HV MVAR').")
        if 'Transformer ID' in df and (df['Transformer ID'].astype(str) == reg['Transformer ID']).any():
            df = df[df['Transformer ID'].astype(str) == reg['Transformer ID']]
        hts = pd.to_datetime(df['Timestamp'], errors='coerce'); ok = hts.notna()
        hts, hl = hts[ok], pd.to_numeric(df['Load (MVA)'], errors='coerce')[ok].to_numpy(float)
        if 'Ambient (C)' in df:
            tts, tv = hts, pd.to_numeric(df['Ambient (C)'], errors='coerce')[ok].to_numpy(float)
        if sides:
            dcs_sides = gc.side_summary(df[ok], sides, {'HV': reg['Rating (MVA)'], 'MV': ss.get('fp_rmv') or None, 'LV': ss.get('fp_rlv') or None})
    else:
        hts, hl = demo_history(ss['fp_dtr'])
        w = demo_full()['weather']; tts, tv = pd.to_datetime(w['Timestamp']), w['Ambient (C)'].to_numpy(float)
    m = hts < origin
    if m.sum() < 168:
        raise ValueError(f"at least 7 days of hourly load before {origin:%Y-%m-%d} are needed (found {int(m.sum())} hours).")
    hts_p, hl_p = hts[m], hl[m]
    if tts is None:
        tts = pd.DatetimeIndex(hts_p); tv = gc.normal_temperature(tts, ss.get('fp_city', 'Dammam'))
    fts = pd.date_range(origin, periods=days * 24, freq='h')
    if ss['fp_wsrc'] == 'Normal climate of a Gulf city':
        fv = gc.normal_temperature(fts, ss['fp_city'], float(ss['fp_shift']))
        wsrc = f"normal climate of {ss['fp_city']}" + (f" {ss['fp_shift']:+.1f} C" if ss['fp_shift'] else '')
    elif ss['fp_wsrc'] == 'Upload forecast CSV':
        if up_w is None:
            raise ValueError("upload the weather forecast CSV.")
        wd = pd.read_csv(up_w)
        if 'Timestamp' not in wd or 'Ambient (C)' not in wd:
            raise ValueError("the weather forecast needs the columns Timestamp and Ambient (C).")
        s = pd.Series(pd.to_numeric(wd['Ambient (C)'], errors='coerce').to_numpy(float), index=pd.to_datetime(wd['Timestamp'])).sort_index()
        if s.index.max() < fts[-1] - pd.Timedelta(hours=6) or s.index.min() > fts[0] + pd.Timedelta(hours=6):
            raise ValueError(f"the weather forecast must cover {fts[0]:%Y-%m-%d %H:%M} to {fts[-1]:%Y-%m-%d %H:%M}.")
        fv = s.reindex(s.index.union(fts)).interpolate(method='time', limit_direction='both').reindex(fts).to_numpy(); wsrc = 'uploaded forecast'
    else:
        wf = demo_weather_fc(str(origin.date()), days); fv = wf['Ambient (C)'].to_numpy(float); wsrc = 'demo weather forecast'
    fc = gc.forecast(MODEL(), reg, hts_p, hl_p, tts, tv, fts, fv, origin, days)
    a = gc.assess(reg, fc, float(ss['fp_lim']))
    actual = None
    if ss['fp_hsrc'] != 'Upload CSV' or True:
        s_all = pd.Series(hl, index=pd.DatetimeIndex(hts))
        s_all = s_all[~s_all.index.duplicated()]
        act = s_all.reindex(fc['ts'])
        actual = act.to_numpy() if act.notna().mean() > 0.9 else None
    h0 = pd.DatetimeIndex(hts_p) >= origin - pd.Timedelta(days=3)
    return {'a': a, 'fc': fc, 'reg': reg, 'hist_ts': pd.DatetimeIndex(hts_p)[h0], 'hist_load': hl_p[h0], 'lim': float(ss['fp_lim']), 'wsrc': wsrc, 'actual': actual,
            'dcs_sides': dcs_sides}


def show_forecast(res):
    a, fc = res['a'], res['fc']
    color = STATUS_COLOR[a['Status']]
    banner(STATUS_LABEL[a['Status']], f"{a['Transformer ID']}  ·  {a['Bay']}  ·  next {len(fc['ts']) // 24} days  ·  governing mode: {a['Governing Mode']}", color)
    m = st.columns(4)
    m[0].metric("Peak load", f"{a['Peak Load (MVA)']:.1f} MVA", f"{a['Peak Load (%)']:.0f}% of rating", delta_color='off', delta_arrow='off')
    m[1].metric("Max hot-spot", f"{a['Max Hot-Spot (C)']:.0f} °C", pd.Timestamp(a['Hot-Spot Time']).strftime('%a %d %b %H:00'), delta_color='off', delta_arrow='off')
    m[2].metric("Max top oil", f"{a['Max Top-Oil (C)']:.0f} °C")
    m[3].metric("Load transfer needed", "none" if a['Load Transfer Needed (MVA)'] == 0 else f"{a['Load Transfer Needed (MVA)']:.1f} MVA",
                f"to keep hot-spot ≤ {res['lim']:.0f} °C", delta_color='off', delta_arrow='off')
    c1, c2 = st.columns([1, 1.3])
    c1.plotly_chart(gauge(a['Overall Risk (%)'], color), width=STRETCH, key='fp_g')
    c2.plotly_chart(mode_plot(a['Mode Risks']), width=STRETCH, key='fp_m')
    if res.get('dcs_sides') is not None:
        st.markdown("#### Load from the DCS (MW / MVAR per side)")
        st.caption("MVA = √(MW² + MVAR²) for every side; the transformer load used by the AI and the thermal model is the HV side "
                   "(or the sum of the MV and LV sides when the HV side is not metered). Power factor = MW / MVA.")
        st.dataframe(res['dcs_sides'], hide_index=True, width=STRETCH)
    st.plotly_chart(forecast_plot(fc, res['hist_ts'], res['hist_load'], res['lim'], actual=res.get('actual')), width=STRETCH, key='fp_fc')
    if res.get('actual') is not None:
        ape = np.nanmean(np.abs(fc['load'] - res['actual']) / res['actual']) * 100
        st.caption(f"The actual load of this period is in the file, so it is shown for comparison (not used by the forecast): mean error {ape:.1f}%.")
    st.markdown("#### Why this result")
    for t in gc.findings(a):
        st.markdown(f"- {t}")
    st.warning(f"**Action:** {a['Action']}")
    st.markdown("#### 🧮 FMEA")
    f1, f2 = st.columns([1.2, 1])
    f1.dataframe(fmea_table(a), hide_index=True, width=STRETCH)
    f1.caption("RPN = Severity × Occurrence × Detection. S = severity of the failure mode; O = mode risk ÷ 10; D = 2 (forecast ahead of time). "
               "Low < 40, Medium 40-99, High ≥ 100.")
    with f2:
        st.plotly_chart(rpn_plot(a), width=STRETCH, key='fp_rpn')
        st.plotly_chart(fmea_rpn_plot(a['FMEA']), width=STRETCH, key='fp_fr')
    H = hourly_table(fc)
    with st.expander("📋 Hourly forecast"):
        st.dataframe(H, hide_index=True, width=STRETCH)
    d1, d2 = st.columns(2)
    try:
        pdf = pdfrep.forecast_pdf(a, fc, gc.findings(a), res['hist_ts'], res['hist_load'], res['lim'], res['wsrc'])
        d1.download_button("📄 Download the forecast report (PDF)", pdf, f"GridGuard_Forecast_{a['Transformer ID']}_{pd.Timestamp(fc['origin']):%Y%m%d}.pdf",
                           'application/pdf', key='fp_pdf')
    except Exception as e:
        d1.error(f"PDF report could not be built: {e}")
    d2.download_button("📥 Download the hourly forecast for a database (CSV)", H.assign(**{'Transformer ID': a['Transformer ID']}).to_csv(index=False).encode('utf-8-sig'),
                       f"GridGuard_Forecast_{a['Transformer ID']}_{pd.Timestamp(fc['origin']):%Y%m%d}.csv", 'text/csv', key='fp_csv')
    st.caption("Added to the Substation Overview and the Combined Report.")


# ==========================================
# --- 5. Thermal and ageing check ---
# ==========================================
def thermal_page():
    st.markdown("<div class='pg-note'>IEEE C57.91 thermal check of one daily load cycle: top-oil and hot-spot temperatures, insulation ageing "
                "and the highest load the transformer can carry at this ambient. Quick check only: not added to the overview.</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.7])
    with left:
        c1, c2 = st.columns(2)
        rating = c1.number_input("Rating (MVA)", 1.0, 2000.0, 50.0, key='th_rating')
        cool = c2.selectbox("Cooling", list(gc.COOLING), 1, key='th_cool')
        c1, c2 = st.columns(2)
        to = c1.number_input("Top-oil rise (K)", 20.0, 80.0, gc.COOLING[cool]['to_rise'], key=f'th_to_{cool}')
        hs = c2.number_input("Hot-spot gradient (K)", 10.0, 50.0, gc.COOLING[cool]['hs_grad'], key=f'th_hs_{cool}')
        st.markdown("**Daily load cycle**")
        shape = st.radio("Shape", ['Typical for a load type', 'Enter 24 hourly values (MVA)'], key='th_shape')
        if shape.startswith('Typical'):
            c1, c2 = st.columns(2)
            lt = c1.selectbox("Load type", gc.LOAD_TYPES, key='th_type')
            peak = c2.number_input("Daily peak (MVA)", 0.1, 3000.0, 52.0, key='th_peak')
            hrs = np.arange(24)
            prof = gc._profile(lt, hrs, np.zeros(24), np.zeros(24), np.zeros(24))
            prof = prof * (0.75 + 0.25 * np.clip(np.sin(np.pi * (hrs - 6) / 14), 0, None) ** 1.2) if lt != 'Industrial' else prof
            load = prof / prof.max() * peak
        else:
            txt = st.text_area("24 values, 00:00 to 23:00, comma separated", "30,28,27,26,26,27,29,32,35,38,41,44,47,49,51,52,51,49,47,46,45,43,38,34", key='th_vals')
            try:
                load = np.array([float(x) for x in txt.replace(';', ',').split(',') if x.strip()])
                if len(load) != 24:
                    raise ValueError
            except ValueError:
                st.error("Enter exactly 24 numbers."); return
        st.markdown("**Ambient**")
        c1, c2 = st.columns(2)
        tmax = c1.number_input("Daily maximum (°C)", -10.0, 60.0, 46.0, key='th_tmax')
        tmin = c2.number_input("Daily minimum (°C)", -20.0, 50.0, 32.0, key='th_tmin')
        go_btn = st.button("🔍 Analyze", width=STRETCH, key='th_go', type='primary')
    with right:
        if go_btn:
            h = np.arange(24)
            shapeT = np.where((h >= 5) & (h <= 14.5), 0.5 - 0.5 * np.cos(np.pi * (h - 5) / 9.5), 0.5 + 0.5 * np.cos(np.pi * ((h - 14.5) % 24) / 14.5))
            amb = tmin + (tmax - tmin) * shapeT
            p = gc.cooling_params(cool, to, hs)
            K3 = np.tile(load / rating, 4); A3 = np.tile(amb, 4)
            th = gc.thermal(K3, A3, p)
            HS, TO = th['hot_spot'][0][-24:], th['top_oil'][0][-24:]
            fa = gc.faa(HS)

            def peak_for(lim):
                lo, hi = 0.05, 3.0
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if gc.thermal(K3 * mid, A3, p)['hot_spot'][0][-24:].max() <= lim:
                        lo = mid
                    else:
                        hi = mid
                return lo * load.max()
            st.session_state['th_last'] = dict(load=load, amb=amb, HS=HS, TO=TO, fa=fa, rating=rating, p110=peak_for(110), p130=peak_for(130), p140=peak_for(140))
        r = st.session_state.get('th_last')
        if r is None:
            st.markdown("<div class='pg-note'>Enter the transformer data, a daily load cycle and the ambient, then press <b>Analyze</b>.</div>", unsafe_allow_html=True)
            return
        hsmax = float(r['HS'].max()); feqa = float(r['fa'].mean())
        risk = max(gc.band3(hsmax, 90, 110, 130, 150), gc.band3(feqa, 0, 1, 4, 16))
        stt = gc.status_of(risk)
        banner(STATUS_LABEL[stt], f"Hot-spot {hsmax:.0f} °C · top oil {r['TO'].max():.0f} °C · ageing {feqa:.2f}× normal", STATUS_COLOR[stt])
        m = st.columns(3)
        m[0].metric("Max daily peak for hot-spot ≤ 110 °C", f"{r['p110']:.1f} MVA", f"{r['p110'] / r['rating'] * 100:.0f}% of rating", delta_color='off', delta_arrow='off')
        m[1].metric("… ≤ 130 °C (planned overload)", f"{r['p130']:.1f} MVA", f"{r['p130'] / r['rating'] * 100:.0f}% of rating", delta_color='off', delta_arrow='off')
        m[2].metric("Life used per day", f"{feqa * 24:.1f} h", "24 h = normal", delta_color='off', delta_arrow='off')
        hrs = [f"{h:02d}:00" for h in range(24)]
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_bar(x=hrs, y=r['load'], name='Load (MVA)', marker_color='rgba(11,46,89,.35)', secondary_y=True)
        fig.add_scatter(x=hrs, y=r['HS'], name='Hot-spot', line=dict(color=RED, width=2.5))
        fig.add_scatter(x=hrs, y=r['TO'], name='Top oil', line=dict(color=AMBER, width=2))
        fig.add_scatter(x=hrs, y=r['amb'], name='Ambient', line=dict(color='#2a78d6'))
        for y, c in [(110, AMBER), (130, RED)]:
            fig.add_hline(y=y, line_dash='dash', line_color=c, annotation_text=f'{y} °C', annotation_position='top left')
        fig.update_layout(title='Daily cycle (steady state after 3 days)', height=420, legend=dict(orientation='h', y=-0.18), margin=dict(l=50, r=50, t=50, b=40))
        fig.update_yaxes(title_text='°C', secondary_y=False); fig.update_yaxes(title_text='MVA', secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width=STRETCH, key='th_fig')
        st.markdown(f"- Hot-spot {hsmax:.0f} °C at {int(np.argmax(r['HS'])):02d}:00 (IEEE C57.91: 110 °C normal, 130 °C planned loading beyond nameplate, 140 °C long-time emergency).")
        st.markdown(f"- Paper ageing {feqa:.2f}× the normal rate: one day at this cycle uses {feqa * 24:.1f} hours of normal life.")
        if stt != 'EXCELLENT':
            st.warning(f"**Action:** keep the daily peak below {r['p110']:.1f} MVA at this ambient for normal ageing; above {r['p130']:.1f} MVA the hot-spot passes the planned overload limit.")


# ==========================================
# --- 6. Model and accuracy ---
# ==========================================
@st.cache_data(show_spinner="Back-testing the model on the demo period…")
def demo_backtest(tid, origin):
    d = demo_full(); o = pd.Timestamp(origin); r = d['register'].set_index('Transformer ID').loc[tid].to_dict(); r['Transformer ID'] = tid
    hts, hl = demo_history(tid); m = hts < o
    w = d['weather']; wp = w[pd.to_datetime(w['Timestamp']) < o]; wf = demo_weather_fc(str(o.date()))
    fc = gc.forecast(MODEL(), r, hts[m], hl[m], pd.to_datetime(wp['Timestamp']), wp['Ambient (C)'], pd.to_datetime(wf['Timestamp']), wf['Ambient (C)'], o)
    act = pd.Series(hl, index=pd.DatetimeIndex(hts)).reindex(fc['ts']).to_numpy()
    return fc, act, pd.DatetimeIndex(hts[m])[-72:], hl[m][-72:]


def model_page():
    B = MODEL()
    acc = B.get('accuracy', {})
    st.markdown("<div class='pg-note'><b>AI load model.</b> A gradient-boosting model (scikit-learn HistGradientBoosting) predicts each hour's load as a "
                "multiple of the transformer's last-7-day average, from the weather forecast (hourly ambient, daily maximum and minimum, heat build-up), "
                "the calendar (hour, weekday, Friday-Saturday weekend, Ramadan, Eid and national holidays, season), the load type and the load of the "
                "same hour one week earlier. One model serves every transformer; the recent average carries each unit's own size and growth. "
                "Beyond the first 7 days the week-earlier load is itself taken from the forecast.</div>", unsafe_allow_html=True)
    src = 'your files (this session)' if st.session_state['model_user'] else 'demo substation (synthetic data)'
    m = st.columns(4)
    m[0].metric("Trained on", f"{B.get('n_rows', 0):,} hours", f"{B.get('n_transformers', 0)} transformers · {src}", delta_color='off', delta_arrow='off')
    m[1].metric("Trained until", B.get('trained_until', '—'))
    mb = acc.get('mape_by_day') or [None]
    m[2].metric("Error 1 day ahead", f"{mb[0]:.1f}%" if mb[0] is not None else '—', "mean absolute % error, hourly load", delta_color='off', delta_arrow='off')
    m[3].metric(f"Error {len(mb)} days ahead", f"{mb[-1]:.1f}%" if mb[-1] is not None else '—', f"test {acc.get('test_from', '')} to {acc.get('test_to', '')}", delta_color='off', delta_arrow='off')
    if acc:
        c1, c2 = st.columns(2)
        f = go.Figure(go.Bar(x=[f"Day {i + 1}" for i in range(len(mb))], y=mb, marker_color=NAVY, text=[f"{v:.1f}%" for v in mb], textposition='outside'))
        f.update_layout(title='Hourly load error by forecast day (data the model never saw)', height=320, yaxis=dict(title='MAPE (%)', range=[0, max(mb) * 1.3]),
                        margin=dict(l=50, r=10, t=50, b=40))
        c1.plotly_chart(f, width=STRETCH, key='md_day')
        bt = acc.get('mape_by_tr', {})
        f2 = go.Figure(go.Bar(x=list(bt), y=list(bt.values()), marker_color=BLUE, text=[f"{v:.1f}%" for v in bt.values()], textposition='outside'))
        f2.update_layout(title='Day-ahead error by transformer', height=320, yaxis=dict(title='MAPE (%)', range=[0, max(bt.values() or [1]) * 1.3]),
                         margin=dict(l=50, r=10, t=50, b=40))
        c2.plotly_chart(f2, width=STRETCH, key='md_tr')
        st.caption(f"Error of the daily peak, 1 day ahead: {acc.get('peak_err_day1', 0):.1f}%. {acc.get('n_forecasts', 0)} forecasts of 7 days, each made with a "
                   "realistic weather forecast (its error grows with the lead time), compared with the actual load.")
    if not st.session_state['model_user']:
        st.markdown("#### Forecast against the actual load (demo)")
        c1, c2 = st.columns(2)
        tid = c1.selectbox("Transformer", list(demo_full()['register']['Transformer ID']), 1, key='md_tid')
        o = c2.date_input("Forecast start", date(2026, 6, 18), min_value=date(2025, 5, 1), max_value=date(2026, 8, 24), key='md_o')
        fc, act, hts, hl = demo_backtest(tid, str(o))
        st.plotly_chart(forecast_plot(fc, hts, hl, 110, 520, actual=act), width=STRETCH, key='md_bt')
        st.caption(f"Mean error over these 7 days: {np.nanmean(np.abs(fc['load'] - act) / act) * 100:.1f}%.")
    st.markdown("#### 🧠 Train the AI on your own data")
    st.markdown("Upload the transformer register, the hourly load history and the hourly ambient temperature of the same period (at least one year is "
                "recommended). The model is trained on everything before the split date and tested on everything after it; the new model is used "
                "by every page for this session and can be downloaded.")
    cols = st.columns(3)
    files = {}
    errs = []
    for col, (k, lab) in zip(cols, [('register', 'Transformer register'), ('loads', 'Load history'), ('weather', 'Ambient temperature')]):
        with col:
            files[k], e = read_csv(lab, f'md_{k}', NEED[k])
            if e:
                errs.append(f"{lab}: {e}")
    for e in errs:
        st.error(e)
    with st.expander("📄 File formats and demo files"):
        st.markdown("- **Transformer register:** `Transformer ID, Bay, Load Type (Residential/Commercial/Industrial/Mixed), Rating (MVA), Cooling (ONAN/ONAF/OFAF), "
                    "Top-Oil Rise (K), Hot-Spot Gradient (K)` (the last two optional).\n- **Load history:** `Timestamp, Transformer ID, Load (MVA)` or the DCS MW and MVAR per side (`HV MW, HV MVAR, MV MW, MV MVAR, LV MW, LV MVAR`), hourly.\n"
                    "- **Ambient temperature:** `Timestamp, Ambient (C)`, hourly (station or weather service history).")
        d = demo_full(); cc = st.columns(3)
        cc[0].download_button("Demo register", d['register'].to_csv(index=False).encode('utf-8-sig'), 'demo_gridguard_register.csv', 'text/csv', key='dl_r')
        cc[1].download_button("Demo load history", d['loads'].to_csv(index=False).encode('utf-8-sig'), 'demo_gridguard_loads.csv', 'text/csv', key='dl_l')
        cc[2].download_button("Demo weather", d['weather'].to_csv(index=False).encode('utf-8-sig'), 'demo_gridguard_weather.csv', 'text/csv', key='dl_w')
    if all(files.get(k) is not None for k in NEED) and not errs:
        t = pd.to_datetime(files['loads']['Timestamp'], errors='coerce')
        default_split = (t.max() - pd.Timedelta(days=120)).date()
        split = st.date_input("Split date (train before, test after)", default_split, key='md_split')
        if st.button("🧠 Train the model", type='primary', key='md_train'):
            try:
                with st.spinner("Training and testing the model…"):
                    b = tg.fit(files['register'], files['loads'], files['weather'], str(split))
                    E = tg.evaluate(b, files['register'], files['loads'], files['weather'], str(split), t.max() - pd.Timedelta(days=gc.HORIZON_DAYS), step_days=3)
                    if len(E):
                        sd = E.groupby('Day Ahead')['Ratio Error'].std().reindex(range(1, gc.HORIZON_DAYS + 1)).ffill().fillna(0.05)
                        b['err_sd_by_day'] = [round(float(v), 4) for v in sd]
                        b['accuracy'] = {'test_from': str(split), 'test_to': str((t.max() - pd.Timedelta(days=gc.HORIZON_DAYS)).date()),
                                         'mape_by_day': [round(float(v), 2) for v in E.groupby('Day Ahead')['APE'].mean()],
                                         'mape_by_tr': {k: round(float(v), 2) for k, v in E[E['Day Ahead'] == 1].groupby('Transformer ID')['APE'].mean().items()},
                                         'peak_err_day1': 0.0, 'n_forecasts': int(E.groupby(['Transformer ID', 'Origin']).ngroups)}
                    st.session_state['model_user'] = b
                st.success("Model trained: every page now uses it for this session.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
    if st.session_state['model_user']:
        import pickle
        c1, c2 = st.columns(2)
        c1.download_button("💾 Download the trained model", pickle.dumps(st.session_state['model_user']), 'gridguard_model.pkl', 'application/octet-stream', key='md_dl')
        if c2.button("↩️ Back to the demo model", key='md_reset'):
            st.session_state['model_user'] = None
            st.rerun()


# ==========================================
# --- 7. Tabs ---
# ==========================================
# ---- about this module: shown open at the top of the overview page (it can be closed)
ABOUT = {
 "lead": "Forecasts the load and the winding hot-spot temperature of power transformers for the coming days and the summer peak, from their load history and the weather forecast, and checks overload and insulation ageing before they happen.",
 "in": [
  "Load history from the DCS: timestamp + MVA, or MW and MVAr (HV, MV, LV sides)",
  "Weather forecast (ambient °C), or the normal climate of a Gulf city",
  "Transformer ratings and a daily load shape (optional)"
 ],
 "out": [
  "Load and hot-spot forecast",
  "Hours above rating, peak hot-spot, insulation ageing",
  "Accuracy of the AI forecast model",
  "Combined PDF report"
 ],
 "start": [
  "Load & Thermal Forecast: choose the transformer and upload its load history",
  "Choose the weather source and press Analyze",
  "Check the ageing in Thermal & Ageing Check"
 ],
 "pages": [
  "Substation Overview: every transformer and its risk",
  "Load & Thermal Forecast, Thermal & Ageing Check",
  "AI Model & Accuracy",
  "Combined Report, Historical Study"
 ],
 "std": "IEEE C57.91 (loading and ageing) · ISO 55000 / IEC 31010 (risk)"
}


def about_module():
    with st.expander('ℹ️ About this module: what it does and how to use it', expanded=True):
        st.markdown(ABOUT['lead'])
        for (t1, k1), (t2, k2) in ((('What you upload', 'in'), ('What you get', 'out')), (('How to start', 'start'), ('Pages', 'pages'))):
            for col, title, key in zip(st.columns(2), (t1, t2), (k1, k2)):
                with col.container(border=True):
                    st.markdown(f'**{title}**\n\n' + '\n'.join((f'{i + 1}. ' if key == 'start' else '- ') + x for i, x in enumerate(ABOUT[key])))
        st.caption('Standards: ' + ABOUT['std'])


tabs = st.tabs(["🏭 Substation Overview", "⚡ Load & Thermal Forecast", "🌡️ Thermal & Ageing Check", "🧠 AI Model & Accuracy",
                "📄 Combined Report", "📈 Historical Study"])

with tabs[0]:
    about_module()
    st.radio("Show", [SRC_PAGES, SRC_DEMO], horizontal=True, key='ov_src')
    R0 = current_records()
    if st.session_state['ov_src'] == SRC_PAGES:
        st.markdown("<div class='pg-note'>Every transformer analyzed on the <b>Load & Thermal Forecast</b> page is collected here. "
                    "The latest result per transformer is kept.</div>", unsafe_allow_html=True)
        if R0 and st.button("🗑 Clear results"):
            st.session_state['register'] = []
            st.rerun()
    else:
        st.info(f"Synthetic demo substation: 6 transformers (residential, commercial, industrial and mixed feeders, 30 to 60 MVA) in a Gulf city; "
                f"7-day forecast from {DEMO_ORIGIN}, made with a realistic weather forecast. Not real measurements.")
    if not R0:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:30px'>💡 No results yet: analyze a transformer on the "
                    "Load & Thermal Forecast page, or show the demo substation.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(R0).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transformers", len(R)); k2.metric("🔴 Action required", int((R['Status'] == 'ACTION').sum()))
        k3.metric("🟠 Watch", int((R['Status'] == 'WATCH').sum())); k4.metric("🟢 Normal", int((R['Status'] == 'EXCELLENT').sum()))
        w = R.iloc[0]
        rp = R.sort_values('RPN', ascending=False).iloc[0]
        hot = R.sort_values('Max Hot-Spot (C)', ascending=False).iloc[0]
        tr = R[R['Load Transfer Needed (MVA)'] > 0]
        ov = R[R['Hours Above Rating'] > 0]
        cards = [("Highest risk", f"{w['Overall Risk (%)']:.0f}%", STATUS_COLOR[w['Status']], f"{w['Transformer ID']} · {w['Governing Mode']}"),
                 ("Highest RPN", f"{int(rp['RPN'])}", rpn_color(rp['RPN']), f"{rp['Transformer ID']} · {rp['RPN Level']}"),
                 ("Hottest hot-spot ahead", f"{hot['Max Hot-Spot (C)']:.0f} °C", band_color(gc.band3(hot['Max Hot-Spot (C)'], 90, 110, 130, 150)), f"{hot['Transformer ID']}"),
                 ("Above rating ahead", f"{len(ov)}", RED if len(ov) else GREEN, ', '.join(ov['Transformer ID'].head(4)) or 'none'),
                 ("Load transfer needed", f"{tr['Load Transfer Needed (MVA)'].sum():.1f} MVA", AMBER if len(tr) else GREEN, ', '.join(tr['Transformer ID'].head(4)) or 'none')]
        for col, (t, big, c, sub) in zip(st.columns(5), cards):
            col.markdown(f"<div class='pg-card' style='border-top:5px solid {c}'><h4>{t}</h4><div class='big' style='color:{c}'>{big}</div>"
                         f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1.1, 1])
        left.plotly_chart(ranking_plot(R), width=STRETCH, key='ov_rank')
        right.plotly_chart(peak_plot(R), width=STRETCH, key='ov_peak')
        c1, c2 = st.columns([1.1, 1])
        c1.plotly_chart(heat_plot(R), width=STRETCH, key='ov_heat')
        c2.plotly_chart(gov_plot(R), width=STRETCH, key='ov_gov')
        st.markdown("#### All transformers")
        show_cols = ['Transformer ID', 'Bay', 'Load Type', 'Rating (MVA)', 'Peak Load (%)', 'Peak Time', 'Max Hot-Spot (C)', 'Max Top-Oil (C)',
                     'Hours Above Rating', 'Load Transfer Needed (MVA)', 'Governing Mode', 'Overall Risk (%)', 'Status', 'RPN', 'RPN Level', 'Forecast From', 'Source', 'Action']
        st.dataframe(R[show_cols].round(1), hide_index=True, width=STRETCH)
        st.download_button("📥 Download transformer ranking (CSV)", R[show_cols].round(1).to_csv(index=False).encode('utf-8-sig'),
                           f"GridGuard_overview_{datetime.now():%Y%m%d}.csv", 'text/csv')

with tabs[1]:
    forecast_page()
with tabs[2]:
    thermal_page()
with tabs[3]:
    model_page()

with tabs[4]:
    recs = current_records()
    if not recs:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:40px'>💡 Analyze transformers on the Load & Thermal Forecast page "
                    "(or show the demo substation on the Overview page) first.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(recs).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        n_a, n_w = int((R['Status'] == 'ACTION').sum()), int((R['Status'] == 'WATCH').sum())
        src = st.session_state.get('ov_src', SRC_PAGES).lower()
        st.markdown(f"<div class='pg-note'><b>📄 Substation Loading Forecast Report</b><br>{len(R)} transformers · source: {src} · "
                    f"generated {datetime.now():%Y-%m-%d %H:%M}</div>", unsafe_allow_html=True)
        top = R.iloc[0]
        tr_total = R['Load Transfer Needed (MVA)'].sum()
        if n_a:
            summary_txt = (f"{n_a} transformer(s) need action and {n_w} are on watch in the forecast period. Highest risk: {top['Transformer ID']} "
                           f"({top['Governing Mode']}, {top['Overall Risk (%)']:.0f}%, hot-spot up to {top['Max Hot-Spot (C)']:.0f} C). "
                           + (f"About {tr_total:.1f} MVA of load transfer at the peaks keeps every hot-spot within the limit." if tr_total > 0 else ''))
        elif n_w:
            summary_txt = f"No transformer needs immediate action; {n_w} are on watch. Highest: {top['Transformer ID']} ({top['Governing Mode']})."
        else:
            summary_txt = "All transformers stay within normal loading and temperature limits in the forecast period."
        st.markdown(f"**Executive summary:** {summary_txt}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transformers", len(R)); k2.metric("🔴 Action", n_a); k3.metric("🟠 Watch", n_w); k4.metric("🟢 Normal", len(R) - n_a - n_w)
        c1, c2 = st.columns([1.1, 1])
        c1.plotly_chart(ranking_plot(R), width=STRETCH, key='rep_rank')
        c2.plotly_chart(peak_plot(R), width=STRETCH, key='rep_peak')
        cols = ['Transformer ID', 'Bay', 'Rating (MVA)', 'Peak Load (%)', 'Max Hot-Spot (C)', 'Load Transfer Needed (MVA)', 'Governing Mode',
                'Overall Risk (%)', 'Status', 'RPN', 'RPN Level', 'Forecast From', 'Source', 'Action']
        st.markdown("#### All transformers")
        st.dataframe(R[cols].round(1), hide_index=True, width=STRETCH)
        st.markdown("#### Transformers above the normal band")
        above = R[R['Status'] != 'EXCELLENT']
        if above.empty:
            st.success("No transformer above the normal band.")
        for i, (_, r) in enumerate(above.iterrows()):
            with st.expander(f"{r['Transformer ID']}  ·  {r['Governing Mode']}  ·  {r['Overall Risk (%)']:.0f}% {r['Status']}", expanded=i < 3):
                e1, e2 = st.columns([1, 1.1])
                with e1:
                    for t in r['Findings']:
                        st.markdown(f"- {t}")
                    st.markdown(f"**RPN** = {r['S']} × {r['O']} × {r['D']} = **{r['RPN']}** ({r['RPN Level']})")
                e2.plotly_chart(mode_plot(r['Mode Risks'], height=260), width=STRETCH, key=f"rep_m_{i}")
                st.warning(f"**Action:** {r['Action']}")
        lines = ["=========================================", "GRIDGUARD AI - SUBSTATION LOADING FORECAST REPORT",
                 "=========================================", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                 f"Transformers: {len(R)}  (action {n_a}, watch {n_w})", "", "EXECUTIVE SUMMARY:", summary_txt, ""]
        for _, r in R.iterrows():
            lines += ["-----------------------------------------",
                      f"{r['Transformer ID']} ({r['Bay']}, {r['Rating (MVA)']:g} MVA {r['Cooling']}) - {r['Status']} - failure risk {r['Overall Risk (%)']:.0f}%",
                      f"Peak {r['Peak Load (MVA)']:.1f} MVA ({r['Peak Load (%)']:.0f}%) at {r['Peak Time']}; hot-spot up to {r['Max Hot-Spot (C)']:.0f} C; "
                      f"load transfer needed {r['Load Transfer Needed (MVA)']:.1f} MVA",
                      f"Governing mode: {r['Governing Mode']}  |  RPN = {r['S']} x {r['O']} x {r['D']} = {r['RPN']} ({r['RPN Level']})",
                      "Mode risks: " + ", ".join(f"{m} {v:.0f}%" for m, v in r['Mode Risks'].items()),
                      f"Source: {r['Source']} (forecast from {r['Forecast From']})"] + [f"* {t}" for t in r['Findings']] + [f"Action: {r['Action']}"]
        lines += ["=========================================", "Generated by GridGuard AI Engine"]
        fm = pd.DataFrame([dict(**{'Transformer ID': r['Transformer ID']}, **m) for _, r in R.iterrows() for m in r['FMEA']])
        d0, d1, d2, d3 = st.columns(4)
        d0.download_button("📄 Download Report (PDF)", pdfrep.combined_pdf(R, summary_txt, src, n_a, n_w),
                           f"GridGuard_Report_{datetime.now():%Y%m%d}.pdf", 'application/pdf', key='rep_pdf')
        d1.download_button("📥 Download Full Text Report", "\n".join(lines).encode('utf-8'), f"GridGuard_Report_{datetime.now():%Y%m%d}.txt", 'text/plain')
        d2.download_button("📥 Download Table (CSV)", R[cols].round(1).to_csv(index=False).encode('utf-8-sig'), f"GridGuard_Report_{datetime.now():%Y%m%d}.csv", 'text/csv')
        d3.download_button("📥 Download FMEA, all modes (CSV)", fm.to_csv(index=False).encode('utf-8-sig'), f"GridGuard_FMEA_{datetime.now():%Y%m%d}.csv", 'text/csv')


@st.cache_data(show_spinner="Replaying every day: forecasting each transformer 7 days ahead…")
def run_replay(_bundle, bundle_key, register, loads, weather, start, end):
    return gc.replay(_bundle, register, loads, weather, start, end)


with tabs[5]:
    st.markdown("<div class='pg-note'>Replays the history: every day at 00:00 each transformer is forecast 7 days ahead with only the load known up to "
                "that moment and a weather forecast. The forecast hot-spots are then compared with what really happened. The result shows how many "
                "over-temperature episodes the system would have warned about, how many days ahead, and how many alerts had no reason.</div>", unsafe_allow_html=True)
    hsrc = st.radio("Data", ["Demo substation (synthetic, forecasts from May 2025 to Aug 2026)", "Upload client files"], horizontal=True, key='h_src')
    files, errs = {}, []
    if hsrc.startswith("Upload"):
        cols = st.columns(3)
        for col, (k, lab) in zip(cols, [('register', 'Transformer register'), ('loads', 'Load history'), ('weather', 'Ambient temperature (history)')]):
            with col:
                files[k], e = read_csv(lab, f'h_{k}', NEED[k])
                if e:
                    errs.append(f"{k}: {e}")
        st.caption("Same formats as on the AI Model & Accuracy page. The weather forecast of each day is simulated from the measured ambient "
                   "(its error grows with the lead time); with archived weather forecasts the study becomes exact.")
    h1, h2, h3, h4 = st.columns(4)
    thr = h1.number_input("Event: hot-spot reaches (°C)", 100, 150, 120, 5)
    alert = h2.number_input("Alert when the forecast reaches (°C)", 95, 150, 117, 1, help="A few degrees below the event level, for the forecast error.")
    hs_start = h3.date_input("From", date(2025, 5, 1), key='h_from')
    hs_end = h4.date_input("To", date(2026, 8, 24), key='h_to')
    if st.button("▶️ Run Historical Study", type='primary', key='h_go'):
        data = None
        if hsrc.startswith("Demo"):
            data = demo_full()
        else:
            for e in errs:
                st.error(e)
            if not errs and all(files.get(k) is not None for k in NEED):
                data = files
            elif not errs:
                st.error("Load the transformer register, the load history and the ambient temperature.")
        if data is not None:
            B = MODEL()
            R = run_replay(B, B.get('trained_until', '') + str(B.get('n_rows', '')), data['register'], data['loads'], data['weather'], str(hs_start), str(hs_end))
            if R.empty:
                st.error("No forecast could be made in this period (each forecast needs 7 days of history before it and 7 days of data after it).")
            else:
                E, S = gc.historical_study(R, float(thr), float(alert))
                st.session_state['hist'] = dict(R=R, E=E, S=S, thr=float(thr), alert=float(alert), demo=hsrc.startswith("Demo"),
                                                trained=B.get('trained_until'))
    H = st.session_state['hist']
    if H is not None:
        R, E, S = H['R'], H['E'], H['S']
        if H['demo']:
            st.info(f"Synthetic demo data. The AI model was trained only on data before {H['trained']}: every forecast after that date is on data it never saw.")
        k = st.columns(4)
        k[0].metric("Over-temperature episodes", S['events'])
        k[1].metric("Warned in advance", f"{S['caught']} / {S['events']}" if S['events'] else "no episodes")
        k[2].metric("Median warning", f"{S['median_lead']:.0f} days ahead" if S['median_lead'] else '—', f"forecast horizon {gc.HORIZON_DAYS} days", delta_color='off', delta_arrow='off')
        k[3].metric("Day-ahead alerts with no reason", f"{S['false_alerts']} of {S['alerts']}", f"actual hot-spot below {H['thr'] - 10:.0f} °C", delta_color='off', delta_arrow='off')
        c1, c2 = st.columns(2)
        pm = S['peak_mape_by_day']; hm = S['hs_mae_by_day']
        f1 = go.Figure(go.Bar(x=[f"Day {d}" for d in pm], y=list(pm.values()), marker_color=NAVY, text=[f"{v:.1f}%" for v in pm.values()], textposition='outside'))
        f1.update_layout(title='Daily peak load error by forecast day', height=300, yaxis=dict(title='%', range=[0, max(pm.values()) * 1.3]), margin=dict(l=50, r=10, t=50, b=40))
        c1.plotly_chart(f1, width=STRETCH, key='h_pm')
        f2 = go.Figure(go.Bar(x=[f"Day {d}" for d in hm], y=list(hm.values()), marker_color=RED, text=[f"{v:.1f} °C" for v in hm.values()], textposition='outside'))
        f2.update_layout(title='Daily maximum hot-spot error by forecast day', height=300, yaxis=dict(title='°C', range=[0, max(hm.values()) * 1.3]), margin=dict(l=50, r=10, t=50, b=40))
        c2.plotly_chart(f2, width=STRETCH, key='h_hm')
        if len(E):
            st.markdown("#### Over-temperature episodes vs. warnings")
            st.dataframe(E, hide_index=True, width=STRETCH)
        st.markdown("#### One transformer in detail")
        tids = sorted(R['Transformer ID'].unique())
        default = E['Transformer ID'].iloc[0] if len(E) else tids[0]
        sel = st.selectbox("Transformer", tids, index=tids.index(default), key='h_sel')
        g = R[R['Transformer ID'] == sel]
        a1 = g[g['Day Ahead'] == 1].sort_values('Date'); a3 = g[g['Day Ahead'] == 3].sort_values('Date'); a7 = g[g['Day Ahead'] == 7].sort_values('Date')
        fig = go.Figure()
        fig.add_scatter(x=a1['Date'], y=a1['Actual Max Hot-Spot (C)'], mode='lines', name='Actual', line=dict(color='#222', width=1.6))
        fig.add_scatter(x=a1['Date'], y=a1['Forecast Max Hot-Spot (C)'], mode='lines', name='Forecast 1 day ahead', line=dict(color=RED, width=1.2))
        fig.add_scatter(x=a7['Date'], y=a7['Forecast Max Hot-Spot (C)'], mode='lines', name='Forecast 7 days ahead', line=dict(color=BLUE, width=1, dash='dot'))
        fig.add_hline(y=H['thr'], line_dash='dash', line_color=RED, annotation_text=f"Event {H['thr']:.0f} °C", annotation_position='top left')
        fig.add_hline(y=H['alert'], line_dash='dot', line_color=AMBER, annotation_text=f"Alert {H['alert']:.0f} °C", annotation_position='bottom left')
        fig.update_layout(title=f'{sel}: daily maximum hot-spot, actual and forecast', height=420, yaxis=dict(title='°C'), legend=dict(orientation='h', y=-0.18),
                          margin=dict(l=50, r=20, t=50, b=40))
        st.plotly_chart(fig, width=STRETCH, key='h_fig')
        st.caption("An episode = days whose actual hot-spot reached the event level (days less than 3 apart = one episode). It is warned when a forecast "
                   "issued at the latest at 00:00 of its first day reached the alert level for that day; the warning time is the earliest such forecast "
                   f"(at most {gc.HORIZON_DAYS} days).")
        d1, d2 = st.columns(2)
        d1.download_button("📥 Download all forecasts vs. actual (CSV)", R.round(2).to_csv(index=False).encode('utf-8-sig'), 'gridguard_replay.csv', 'text/csv', key='h_dl')
        if len(E):
            d2.download_button("📥 Download episodes vs. warnings (CSV)", E.to_csv(index=False).encode('utf-8-sig'), 'gridguard_episodes.csv', 'text/csv', key='h_dl2')

    with st.expander("How the forecast and the risk are calculated"):
        st.markdown("""
**AI load forecast.** Gradient boosting (HistGradientBoosting) predicts each hour's load ÷ the transformer's last-7-day average from: hour, weekday,
Friday-Saturday weekend, Ramadan, Eid and national holidays, season, hourly ambient, its 4-hour build-up, the day's maximum and minimum, cooling degrees
above 24 °C, the load type and the load of the same hour one week earlier. The 80% band comes from the measured error at each forecast day.

**Thermal model (IEEE C57.91 clause 7).** Top-oil rise and hot-spot gradient follow the load with their time constants (top oil 90 to 210 min by cooling,
winding 7 min). Ultimate top-oil rise = rated rise × ((K²R + 1)/(R + 1))ⁿ, hot-spot gradient = rated gradient × K²ᵐ. Ageing factor = exp(15000/383 − 15000/(θH + 273)).

| Failure mode | Normal (≤ 35%) | Watch (35-70%) | Action (> 70%) |
|---|---|---|---|
| Overload (peak load ÷ rating) | ≤ 0.9 | 0.9 to 1.1 | 1.1 to 1.3 (+ hours above rating) |
| Hot-spot temperature | ≤ 110 °C | 110 to 130 °C | 130 to 150 °C |
| Top-oil temperature | ≤ 90 °C | 90 to 105 °C | 105 to 115 °C |
| Insulation ageing (worst day) | ≤ 1× | 1 to 4× | 4 to 16× |

**Overall risk** = the highest mode risk + 5 points for every other mode above normal. **FMEA:** RPN = S × O × D with S = 7 overload, 9 hot-spot,
8 top oil, 6 ageing; O = risk ÷ 10; D = 2. **Load transfer needed** = the share of the peak to move so the hot-spot stays at the chosen limit.
""")

st.markdown(FOOTER, unsafe_allow_html=True)
