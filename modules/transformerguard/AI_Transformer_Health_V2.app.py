# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential.
# Unauthorized copying, modification, distribution, or reverse engineering
# of this software, including the physics-informed AI guardrails, is strictly prohibited.
# ==============================================================================

import hmac
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pg_paths
import powerguard_core as pc

pass  # page config is set once by the PowerGuard AI platform
STRETCH = "stretch"
NAVY, BLUE = "#0b2e59", "#4da6ff"
GREEN, AMBER, RED = "#1a9e5c", "#d99a00", "#d63a3a"

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:16px 22px;border-radius:10px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.pg-header img {height:64px;background:#fff;border-radius:8px;padding:4px 8px;}
.pg-card {border:1px solid #d5dce6;border-radius:10px;padding:12px 14px;text-align:center;}
.pg-card h4 {margin:0;color:#0b2e59;font-size:1.05rem;}
.pg-card .big {font-size:1.7rem;font-weight:700;margin:4px 0;}
.pg-card .sub {font-size:.85rem;color:#5a6575;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-banner {border:2px solid;border-radius:8px;padding:10px 14px;text-align:center;margin:6px 0 12px 0;}
.pg-banner h2 {margin:0;font-size:1.5rem;}
.pg-banner p {margin:2px 0 0 0;font-size:.95rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;}
.pg-foot {text-align:center;color:#888;font-size:12px;padding:10px;}
</style>
""", unsafe_allow_html=True)

FOOTER = """
<div class='pg-foot'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
<em>Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.</em></div>
"""

# ==========================================
# --- 1. Login ---
# ==========================================
for key, default in [('logged_in', False), ('latest_eval', None), ('history', None),
                     ('fmea_log', pd.DataFrame(columns=['Timestamp', 'Asset ID', 'Asset Type', 'DGA Index', 'Pred Temp',
                                                       'Measured Temp', 'Temp Alarm',
                                                       'Overall Risk (%)', 'Status', 'Driver', 'Insulation Life (yrs)',
                                                       'Failure Mode', 'S', 'O', 'D', 'RPN', 'RPN Level',
                                                       'Action Required']))]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align:center;color:#4da6ff;'>⚡ TransformerGuard AI Login</h1>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", width=STRETCH):
                if "APP_USERNAME" not in st.secrets or "APP_PASSWORD" not in st.secrets:
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
# --- 2. Header & models ---
# ==========================================
col_l, col_r = st.columns([10, 1])
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
    <h1>⚡ TransformerGuard AI: Transformer and Reactor Health</h1>
    <p>Risk-based maintenance: 9 dissolved gases + thermal load, fused into one asset risk</p>
  </div>
  {_logo_html}
</div>""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        return joblib.load(pg_paths.model('rf_health_model.pkl')), joblib.load(pg_paths.model('rf_thermal_model.pkl'))
    except Exception:
        return None, None


model_health, model_thermal = load_models()
if model_health is None or model_thermal is None:
    st.error("⚠️ Model files not detected! Please ensure 'rf_health_model.pkl' and 'rf_thermal_model.pkl' are in the repository.")
    st.stop()


# ==========================================
# --- 3. Shared UI helpers ---
# ==========================================
def gauge(value, rng, ticks, steps, title=None, height=280):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title} if title else None,
        number={'font': {'size': 40}},
        gauge={'axis': {'range': rng, 'tickmode': 'array', 'tickvals': ticks,
                        'ticktext': [str(t) for t in ticks], 'tickfont': {'size': 14}},
               'bar': {'color': "#666", 'thickness': 0.18},
               'steps': [{'range': r, 'color': c} for r, c in steps],
               'threshold': {'line': {'color': "#666", 'width': 4}, 'thickness': 0.75, 'value': value}}))
    fig.update_layout(height=height, margin=dict(l=25, r=40, t=40 if title else 25, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


DI_STEPS = [((0, pc.DI_SAFE_MAX), "rgba(26,158,92,.45)"), ((pc.DI_SAFE_MAX, pc.DI_WARNING_MAX), "rgba(217,154,0,.45)"),
            ((pc.DI_WARNING_MAX, 100), "rgba(214,58,58,.45)")]
TEMP_STEPS = [((20, 90), "rgba(26,158,92,.45)"), ((90, 100), "rgba(217,154,0,.45)"), ((100, 120), "rgba(214,58,58,.45)")]
RISK_STEPS = [((0, pc.RISK_LOW_MAX), "rgba(26,158,92,.45)"), ((pc.RISK_LOW_MAX, pc.RISK_MEDIUM_MAX), "rgba(217,154,0,.45)"),
              ((pc.RISK_MEDIUM_MAX, 100), "rgba(214,58,58,.45)")]


def status_color(status):
    return {'SAFE': GREEN, 'NORMAL': GREEN, 'EXCELLENT': GREEN, 'WARNING': AMBER, 'WATCH': AMBER}.get(status, RED)


def banner(title, subtitle, color):
    st.markdown(f"<div class='pg-banner' style='border-color:{color};'><h2 style='color:{color};'>{title}</h2>"
                f"<p>{subtitle}</p></div>", unsafe_allow_html=True)


def duval_figure(ch4, c2h4, c2h2, diagnosis, height=340):
    a, b, c = pc.duval_percentages(ch4, c2h4, c2h2)
    fig = go.Figure()
    fig.add_trace(go.Scatterternary(a=[100, 0, 0], b=[0, 100, 0], c=[0, 0, 100], mode='markers',
                                    marker=dict(color='rgba(0,0,0,0)', size=1), hoverinfo='skip'))
    fig.add_trace(go.Scatterternary(a=[a], b=[b], c=[c], mode='markers', name='Sample',
                                    marker=dict(size=14, color=RED, symbol='cross', line=dict(width=2, color='white'))))
    fig.update_layout(title="Duval Triangle 1", height=height, margin=dict(l=30, r=30, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      ternary=dict(sum=100, aaxis=dict(title="CH4 %", min=0), baxis=dict(title="C2H4 %", min=0),
                                   caxis=dict(title="C2H2 %", min=0)))
    return fig


CONDITION_NAMES = {1: "1 · Normal", 2: "2 · Above normal", 3: "3 · High", 4: "4 · Excessive"}


def gas_condition_table(dga_res):
    """Each gas against the IEEE C57.104 limits. 'Normal up to' is the Condition 1 limit;
    the next columns are the upper limits of Conditions 2 and 3. Above the last = Condition 4."""
    rows = []
    for f, cond in dga_res['gas_conditions'].items():
        lim = pc.GAS_CONDITION_LIMITS[f]
        rows.append({'Gas': pc.GAS_LABELS[f], 'Value (ppm)': dga_res['inputs'][f],
                     'Normal up to': lim[0], 'Above normal up to': lim[1], 'High up to': lim[2],
                     'Condition': CONDITION_NAMES[cond]})
    df = pd.DataFrame(rows)

    def colour(v):
        return f"color: {[GREEN, AMBER, RED, RED][int(v[0]) - 1]}; font-weight: bold"
    return df.style.map(colour, subset=['Condition']).format(
        {'Value (ppm)': '{:,.1f}', 'Normal up to': '{:,.0f}', 'Above normal up to': '{:,.0f}', 'High up to': '{:,.0f}'})


def impact_chart(model, features, values, baselines, scale='Blues', top=None, title=None):
    imp = model.feature_importances_ * (np.abs(np.array(values) - baselines) / np.where(baselines == 0, 1, baselines))
    imp = imp / imp.sum() * 100 if imp.sum() > 0 else model.feature_importances_ * 100
    df = pd.DataFrame({'Feature': features, 'Deviation-weighted impact (%)': imp})
    df = df.sort_values('Deviation-weighted impact (%)', ascending=False)
    if top:
        df = df.head(top)
    df = df.sort_values('Deviation-weighted impact (%)')
    fig = px.bar(df, x='Deviation-weighted impact (%)', y='Feature', orientation='h',
                 color='Deviation-weighted impact (%)', color_continuous_scale=scale, title=title)
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, margin=dict(l=10, r=10, t=40 if title else 10, b=10))
    return fig


def fmea_excel(log_df, latest_df):
    """Excel workbook: sheet 1 = fleet ranking with a native Excel bar chart, sheet 2 = full register."""
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.chart.marker import DataPoint
    from openpyxl.chart.label import DataLabelList
    from openpyxl.chart.axis import ChartLines
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    fill_hex = {'EXCELLENT': '1A9E5C', 'WATCH': 'D99A00', 'ACTION': 'D63A3A'}
    life_col = 'Insulation Life (yrs)'
    show_life = life_col in log_df.columns and log_df[life_col].notna().any()
    show_meas = 'Measured Temp' in log_df.columns and log_df['Measured Temp'].notna().any()
    drop = ([] if show_life else [life_col]) + ([] if show_meas else ['Measured Temp', 'Temp Alarm'])
    log_df = log_df.drop(columns=drop, errors='ignore')
    latest_df = latest_df.drop(columns=drop, errors='ignore')
    header_fill = PatternFill('solid', fgColor='0B2E59')
    wb = Workbook()

    ws = wb.active
    ws.title = "Fleet Ranking"
    ws['A1'] = "PowerGuard AI - Fleet Risk Ranking"
    ws['A1'].font = Font(bold=True, size=14, color='0B2E59')
    ws['A2'] = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · latest assessment per transformer"
    rank = latest_df.sort_values('Overall Risk (%)', ascending=False)
    cols = ['Asset ID', 'Overall Risk (%)', 'Status', 'Driver', 'RPN', 'RPN Level', 'Failure Mode',
            'DGA Index', 'Pred Temp'] + (['Measured Temp', 'Temp Alarm'] if show_meas else []) + \
        ([life_col] if show_life else []) + ['Action Required']
    ws.append([])
    ws.append(cols)
    for c in range(1, len(cols) + 1):
        cell = ws.cell(row=4, column=c)
        cell.font, cell.fill = Font(bold=True, color='FFFFFF'), header_fill
    for _, r in rank.iterrows():
        row_vals = [r['Asset ID'], round(float(r['Overall Risk (%)']), 1), r['Status'], r['Driver'],
                    int(r['RPN']), r['RPN Level'], r['Failure Mode'],
                    round(float(r['DGA Index']), 1), None if pd.isna(r['Pred Temp']) else round(float(r['Pred Temp']), 1)]
        if show_meas:
            row_vals += [None if pd.isna(r['Measured Temp']) else round(float(r['Measured Temp']), 1),
                         None if pd.isna(r['Temp Alarm']) else r['Temp Alarm']]
        if show_life:
            row_vals.append(None if pd.isna(r[life_col]) else round(float(r[life_col]), 1))
        ws.append(row_vals + [r['Action Required']])
        row = ws.max_row
        ws.cell(row=row, column=3).font = Font(bold=True, color=fill_hex.get(r['Status'], '000000'))
    widths = [12, 16, 12, 12, 8, 11, 40, 12, 12] + ([15, 18] if show_meas else []) + \
        ([18] if show_life else []) + [70]
    for c, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(c)].width = w

    n = len(rank)
    chart = BarChart()
    chart.type = "bar"
    chart.style = 10
    chart.title = "Overall Risk by Transformer (%)"
    chart.legend = None
    chart.gapWidth = 60
    chart.add_data(Reference(ws, min_col=2, min_row=4, max_row=4 + n), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + n))
    # Category axis: highest risk on top; value axis kept at the bottom
    chart.x_axis.scaling.orientation = "maxMin"
    chart.x_axis.delete = False
    chart.x_axis.title = None
    chart.y_axis.delete = False
    chart.y_axis.crosses = "max"
    chart.y_axis.scaling.min, chart.y_axis.scaling.max = 0, 110
    chart.y_axis.majorUnit = 10
    chart.y_axis.number_format = '0'
    chart.y_axis.title = "Overall Risk (%)"
    chart.y_axis.majorGridlines = ChartLines()
    series = chart.series[0]
    series.dLbls = DataLabelList()
    series.dLbls.showVal = True
    series.dLbls.showSerName = series.dLbls.showCatName = series.dLbls.showLegendKey = False
    series.dLbls.position = 'outEnd'
    series.dLbls.numFmt = '0"%"'
    for i, (_, r) in enumerate(rank.iterrows()):
        pt = DataPoint(idx=i)
        pt.graphicalProperties.solidFill = fill_hex.get(r['Status'], '888888')
        pt.graphicalProperties.line.solidFill = fill_hex.get(r['Status'], '888888')
        series.dPt.append(pt)
    chart.height, chart.width = max(7.0, 3.0 + 1.1 * n), 22
    ws.add_chart(chart, f"A{7 + n}")
    ws.page_setup.orientation = 'landscape'
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0

    ws2 = wb.create_sheet("FMEA Register")
    ws2.append(list(log_df.columns))
    for c in range(1, len(log_df.columns) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font, cell.fill = Font(bold=True, color='FFFFFF'), header_fill
    for _, r in log_df.iterrows():
        ws2.append([None if pd.isna(v) else (int(v) if isinstance(v, (int, np.integer)) else
                                              round(float(v), 1) if isinstance(v, (float, np.floating)) else v)
                    for v in r.values])
        ws2.cell(row=ws2.max_row, column=list(log_df.columns).index('Status') + 1).font = Font(
            bold=True, color=fill_hex.get(r['Status'], '000000'))
    for c, col in enumerate(log_df.columns, start=1):
        ws2.column_dimensions[get_column_letter(c)].width = {'Action Required': 70, 'Failure Mode': 40,
                                                             'S': 5, 'O': 5, 'D': 5, 'RPN': 8}.get(col, 18)
    ws2.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def show_temp_compare(t):
    """Predicted vs measured top-oil temperature. The model always predicts; the measured value is a check."""
    if t.get('measured') is None:
        st.caption("No measured top-oil temperature: the result uses the model's predicted value only.")
        return
    st.markdown("**🌡️ Predicted vs Measured Top-Oil**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted (model)", f"{t['temp']:.1f} °C")
    c2.metric("Measured (DCS)", f"{t['measured']:.1f} °C")
    c3.metric("Difference", f"{t['measured_deviation']:+.1f} °C")
    lim = pc.MEASURED_DEVIATION_ALARM_C
    if t['deviation_flag'] == 'ABOVE':
        st.error(f"🚨 Temperature mismatch alarm: the transformer runs {t['measured_deviation']:.1f} °C hotter than the "
                 f"model expects at this load (limit ±{lim:.0f} °C). Risk is assessed on the measured "
                 f"{t['measured']:.1f} °C. Verify the reading, then check cooling and oil level.")
    elif t['deviation_flag'] == 'BELOW':
        st.info(f"Measured is {-t['measured_deviation']:.1f} °C below the model (limit ±{lim:.0f} °C). Risk stays on the "
                f"higher, predicted value. The model over-predicts here, or the sensor needs checking.")
    else:
        st.success(f"✅ Model and measurement agree within ±{lim:.0f} °C. Risk uses the higher of the two "
                   f"({t['assessed_temp']:.1f} °C).")


def show_fmea(fm):
    st.markdown("**🧮 FMEA: RPN = Severity × Occurrence × Detection**")
    st.dataframe(pd.DataFrame(fm['modes'])[['Failure Mode', 'Source', 'S', 'O', 'D', 'RPN', 'RPN Level']],
                 width=STRETCH, hide_index=True)
    st.caption(f"S = severity of the failure mode (fully automatic, all assets treated as critical). "
               f"O = occurrence from the live risk of that dimension (risk ÷ 10, 1-10). "
               f"D = {pc.DETECTION_ONLINE} for continuous online monitoring. "
               f"RPN levels: Low < {pc.RPN_MEDIUM_MIN}, Medium {pc.RPN_MEDIUM_MIN}-{pc.RPN_HIGH_MIN - 1}, "
               f"High ≥ {pc.RPN_HIGH_MIN} (maximum 200 with online detection).")


DGA_BASELINES = np.array([pc.DGA_DEFAULTS[f] if pc.DGA_DEFAULTS[f] != 0 else 0.1 for f in pc.DGA_FEATURES])
THERMAL_BASELINES = np.array([12.5, 3.2, 8.1, 1.5, 4.5, 0.8, 12.0, 6.0])

DGA_HELP = {
    'Hydrogen': "Partial discharge or low-energy faults.", 'Oxigen': "Sealing integrity / oxidation.",
    'Nitrogen': "Atmospheric leak or nitrogen blanket.", 'Methane': "Low-temperature thermal fault (< 300°C).",
    'CO': "Paper insulation degradation.", 'CO2': "Cellulose aging; abnormal CO2/CO implies overheating.",
    'Ethylene': "High-temperature thermal fault (> 700°C).", 'Ethane': "Moderate thermal fault (300-700°C).",
    'Acethylene': "CRITICAL: arcing / high-energy discharge.",
}
DGA_LABELS = {'Hydrogen': "Hydrogen H2 (ppm)", 'Oxigen': "Oxygen O2 (ppm)", 'Nitrogen': "Nitrogen N2 (ppm)",
              'Methane': "Methane CH4 (ppm)", 'CO': "Carbon Monoxide CO (ppm)", 'CO2': "Carbon Dioxide CO2 (ppm)",
              'Ethylene': "Ethylene C2H4 (ppm)", 'Ethane': "Ethane C2H6 (ppm)", 'Acethylene': "Acetylene C2H2 (ppm)"}


def dga_inputs(prefix, two_columns=False):
    vals = {}
    gases = ['Hydrogen', 'Methane', 'Ethylene', 'Ethane', 'Acethylene', 'CO', 'CO2', 'Oxigen', 'Nitrogen']
    st.markdown("**1. Dissolved Gases (ppm)**")
    cols = st.columns(2) if two_columns else [st.container()]
    for i, f in enumerate(gases):
        vals[f] = cols[i % len(cols)].number_input(DGA_LABELS[f], value=float(pc.DGA_DEFAULTS[f]), help=DGA_HELP[f],
                                                   key=f"{prefix}_{f}", min_value=0.0)
    return vals


def capacity_inputs(prefix):
    c1, c2 = st.columns([1, 1.3])
    unit = c1.selectbox("Units", ["MVA", "kVA"], key=f"{prefix}_unit",
                        help="MVA: capacity in MVA and loads in MW / MVAR. kVA: capacity in kVA and loads in kW / kVAR.")
    cap = c2.number_input(f"Rated Capacity ({unit})", min_value=0.01, value=150.0 if unit == "MVA" else 50.0,
                          key=f"{prefix}_cap_{unit}")
    return unit, cap


def load_inputs(prefix, unit, defaults=(60.0, 12.0, 30.0, 6.0, 18.0, 3.0)):
    p, q = ("MW", "MVAR") if unit == "MVA" else ("kW", "kVAR")
    scale = 1.0 if unit == "MVA" else 0.25
    names = [('HUFL', f"HV Active ({p})"), ('HULL', f"HV Reactive ({q})"), ('MUFL', f"MV Active ({p})"),
             ('MULL', f"MV Reactive ({q})"), ('LUFL', f"LV Active ({p})"), ('LULL', f"LV Reactive ({q})")]
    vals, cols = {}, st.columns(2)
    for i, (k, label) in enumerate(names):
        vals[k] = cols[i % 2].number_input(label, value=defaults[i] * scale, min_value=0.0, key=f"{prefix}_{k}_{unit}")
    return {k: pc.to_kw(v, unit) for k, v in vals.items()}


def insulation_inputs(prefix):
    """Asset data needed for the IEEE C57.91 insulation-aging calculation."""
    c1, c2, c3 = st.columns([1, 1, 1.2])
    age = c1.number_input("Age in service (years)", min_value=0.0, max_value=80.0, value=10.0, step=1.0, key=f"{prefix}_age")
    cooling = c2.selectbox("Cooling", ["ONAN", "ONAF", "OFAF", "ODAF"], key=f"{prefix}_cool")
    upgraded = c3.selectbox("Paper", ["Thermally upgraded", "Normal Kraft"], key=f"{prefix}_paper") == "Thermally upgraded"
    with st.expander("Insulation assumptions"):
        grad = st.number_input("Rated hot-spot gradient over top oil (K)", 10.0, 40.0, 25.0, 1.0, key=f"{prefix}_grad",
                               help="From the heat-run test report. 25 K is a typical value when unknown.")
        past = st.number_input("Average aging factor so far", 0.05, 3.0, 0.5, 0.05, key=f"{prefix}_past",
                               help="1.0 = the transformer ran at its rated hot spot all its life. Most units run below "
                                    "that; 0.5 is a conservative default. Refine it from historical loading data.")
    return dict(age_years=age, cooling=cooling, upgraded_paper=upgraded, rated_hotspot_gradient=grad,
                past_aging_factor=past)


def years_txt(y):
    return "> 60 yrs" if y > 60 else f"{y:.1f} yrs"


def show_insulation(ins):
    st.markdown("#### 🧻 Insulation Life (IEEE C57.91)")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Hot-Spot Temp", f"{ins['hot_spot']:.1f} °C")
    i2.metric("Aging Rate (FAA)", f"{ins['faa']:.2f} x" if ins['faa'] >= 0.1 else f"{ins['faa']:.3f} x",
              help="1.0 = normal aging at the reference hot spot. Below 1 the insulation ages slower than normal.")
    i3.metric("Life Consumed", f"{ins['consumed_pct']:.0f} %")
    i4.metric("Remaining Life", years_txt(ins['remaining_years_now']))
    st.caption(f"Remaining life if the transformer keeps running at this operating point. At rated aging (FAA = 1) it "
               f"would be {years_txt(ins['remaining_years_normal'])}. Normal insulation life = 180,000 h; hot spot = "
               f"top oil + {ins['gradient']:.1f} K. The estimate depends on the top-oil temperature, so a measured value "
               f"from the DCS gives a better result than the model.")
    for x in ins['findings']:
        st.markdown(f"- {x}")


def asset_type_from_id(asset_id):
    """R1, RB1, RL2 ... = shunt reactor; everything else = transformer."""
    return 'Shunt reactor' if str(asset_id).strip().upper().startswith('R') else 'Transformer'


def log_entry(asset_id, asset_type, d, t, f, fm, ins, when=None):
    """One FMEA-register row. Used by the Asset Risk tab and the demo fleet."""
    g = fm['governing']
    return {
        'Timestamp': (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"), 'Asset ID': asset_id, 'Asset Type': asset_type,
        'DGA Index': round(d['index'], 1), 'Pred Temp': round(t['temp'], 1) if t else None,
        'Measured Temp': t['measured'] if t else None,
        'Temp Alarm': ('ALARM' if t['deviation_flag'] == 'ABOVE' else
                       'Check model/sensor' if t['deviation_flag'] == 'BELOW' else 'OK')
        if t and t['measured'] is not None else None,
        'Overall Risk (%)': round(f['overall'], 1), 'Status': f['status'], 'Driver': f['driver'],
        'Insulation Life (yrs)': round(min(ins['remaining_years_now'], 60.0), 1) if ins else None,
        'Failure Mode': g['Failure Mode'], 'S': g['S'], 'O': g['O'], 'D': g['D'], 'RPN': g['RPN'],
        'RPN Level': g['RPN Level'], 'Action Required': f['action']}


# Demo fleet: 400/220 kV substation, each asset assessed with the same engine as the Asset Risk tab.
DEMO_FLEET = [
    # id, type, gas changes from normal, loads (MW / MVAR: HV, MV, LV), hour, month, measured top-oil, age
    ('T1', 'Transformer', {'Hydrogen': 22, 'Methane': 12, 'CO': 210, 'CO2': 1900, 'Ethylene': 6, 'Ethane': 9},
     (62, 12, 30, 6, 18, 3), 11, 4, None, 12),
    ('T2', 'Transformer', {'Hydrogen': 420, 'Methane': 60, 'Acethylene': 14, 'Ethylene': 38, 'Ethane': 10, 'CO': 260, 'CO2': 2100},
     (60, 12, 30, 6, 18, 3), 10, 3, None, 18),
    ('T3', 'Transformer', {'Hydrogen': 90, 'Methane': 310, 'Ethylene': 260, 'Ethane': 120, 'CO': 330, 'CO2': 2600},
     (95, 20, 45, 9, 28, 6), 15, 8, 88.0, 22),
    ('T4', 'Transformer', {'Hydrogen': 30, 'Methane': 20, 'CO': 280, 'CO2': 2300, 'Ethylene': 9, 'Ethane': 15},
     (80, 16, 40, 8, 24, 5), 15, 8, 79.0, 9),
    ('RB1', 'Shunt reactor', {'Hydrogen': 140, 'Methane': 8, 'CO': 120, 'CO2': 900},
     (40, 8, 20, 4, 10, 2), 3, 1, None, 15),
    ('RB2', 'Shunt reactor', {'Hydrogen': 15, 'Methane': 8, 'CO': 120, 'CO2': 900},
     (40, 8, 20, 4, 10, 2), 3, 1, None, 7),
    ('RL1', 'Shunt reactor', {'Hydrogen': 25, 'Methane': 14, 'CO': 180, 'CO2': 1500, 'Ethylene': 4},
     (45, 9, 22, 4, 11, 2), 14, 7, 74.0, 11),
    ('RL2', 'Shunt reactor', {'Hydrogen': 35, 'Methane': 30, 'CO': 240, 'CO2': 2000, 'Ethylene': 12},
     (45, 9, 22, 4, 11, 2), 14, 7, 92.0, 25),
]


def demo_fleet():
    """8 quick assessments, so no cache: edits to DEMO_FLEET show at once."""
    rows = []
    when = datetime(2026, 3, 1, 8, 0)
    for aid, typ, gas, loads, hr, mo, meas, age in DEMO_FLEET:
        g = dict(pc.DGA_DEFAULTS); g.update(gas)
        d = pc.assess_dga(g, model_health)
        L = {k: pc.to_kw(v, 'MVA') for k, v in zip(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'], loads)}
        t = pc.assess_thermal(L, pc.to_kva(150, 'MVA'), hr, mo, model_thermal, meas)
        ins = pc.insulation_life(meas if meas else t['temp'], t['load_factor'], age)
        f = pc.fuse_risk(d, t)
        fm = pc.fmea_assessment(d, t)
        rows.append(log_entry(aid, typ, d, t, f, fm, ins, when))
    return pd.DataFrame(rows)


# ---- about this module: shown open at the top of the overview page (it can be closed)
ABOUT = {
 "lead": "Checks the condition of power transformers and shunt reactors from the gases dissolved in their oil (DGA) and from their loading and temperatures, and ranks them by risk so maintenance goes where it is needed first.",
 "in": [
  "Oil gas results (DGA): CSV or Excel, one row per sample",
  "Loading and temperatures from SCADA (optional)",
  "Failures and trips history, asset list with ratings"
 ],
 "out": [
  "Fault type from the gases (Duval triangle, IEEE gas conditions)",
  "Hot-spot temperature, ageing and remaining life",
  "Health index, risk ranking (P1–P4) and FMEA register",
  "PDF diagnostic report"
 ],
 "start": [
  "Open DGA Analysis and enter one sample, or upload a file in Batch Analysis",
  "Add loading data in Thermal Load",
  "Read the ranking in Fleet Overview and Asset Risk & FMEA"
 ],
 "pages": [
  "Fleet Overview: all units and their risk",
  "DGA Analysis: one oil sample",
  "Thermal Load: hot-spot and ageing",
  "Asset Risk & FMEA, Batch Analysis, AI Diagnostic Report, Historical Study"
 ],
 "std": "IEC 60599 · IEEE C57.104 (gases) · IEEE C57.91 (loading and ageing) · ISO 55000 / IEC 31010 (risk)"
}


def about_module():
    with st.expander('ℹ️ About this module: what it does and how to use it', expanded=True):
        st.markdown(ABOUT['lead'])
        for (t1, k1), (t2, k2) in ((('What you upload', 'in'), ('What you get', 'out')), (('How to start', 'start'), ('Pages', 'pages'))):
            for col, title, key in zip(st.columns(2), (t1, t2), (k1, k2)):
                with col.container(border=True):
                    st.markdown(f'**{title}**\n\n' + '\n'.join((f'{i + 1}. ' if key == 'start' else '- ') + x for i, x in enumerate(ABOUT[key])))
        st.caption('Standards: ' + ABOUT['std'])


tabs = st.tabs(["🏭 Fleet Overview", "🧪 DGA Analysis", "🌡️ Thermal Load", "📑 Asset Risk & FMEA",
                "📁 Batch Analysis", "📄 AI Diagnostic Report", "📈 Historical Study"])

# ==========================================
# --- TAB 1: DGA (9 DISSOLVED GASES) ---
# ==========================================
with tabs[1]:
    col_input, col_output = st.columns([1, 2.2])
    with col_input:
        with st.form("dga_form"):
            st.markdown("### 📊 Dissolved Gas Sample")
            dga_vals = dga_inputs("t1")
            submitted = st.form_submit_button("🔍 Analyze Gases", width=STRETCH)

    with col_output:
        if submitted:
            r = pc.assess_dga(dga_vals, model_health)
            color = status_color(r['status'])
            banner(r['status'], f"Worst gas condition: IEEE Condition {r['worst_condition']}  ·  Duval: {r['duval']}", color)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Degradation Index", f"{r['index']:.1f} / 100")
            m2.metric("DGA Risk", f"{r['risk']:.0f} %")
            m3.metric("Worst Gas Condition", f"{r['worst_condition']} of 4")
            m4.metric("Model Agreement", f"{r['agreement']:.1f} %")

            g1, g2 = st.columns([1, 1.2])
            with g1:
                st.plotly_chart(gauge(r['index'], [0, 100], [0, int(pc.DI_SAFE_MAX), int(pc.DI_WARNING_MAX), 100],
                                      DI_STEPS, "Degradation Index"), width=STRETCH)
            with g2:
                st.plotly_chart(duval_figure(dga_vals['Methane'], dga_vals['Ethylene'], dga_vals['Acethylene'], r['duval']),
                                width=STRETCH)

            st.markdown("#### Findings")
            if r['findings']:
                for f in r['findings']:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- ✅ Dissolved gases are within normal limits.")

            st.markdown("#### IEEE C57.104 Key-Gas Conditions")
            st.caption("Limits in ppm for each gas. Condition 1 = normal; 2 = above normal, re-sample and "
                       "watch the trend; 3 = high decomposition, investigate; 4 = excessive, consider removal "
                       "from service. The worst gas sets the transformer's condition.")
            st.dataframe(gas_condition_table(r), width=STRETCH, hide_index=True)

            st.markdown("#### Variables Driving This Result")
            st.plotly_chart(impact_chart(model_health, pc.DGA_FEATURES, r['input_df'].values[0], DGA_BASELINES),
                            width=STRETCH)
        else:
            st.markdown("<div class='pg-note'>Enter the dissolved-gas sample on the left and press <b>Analyze Gases</b>.</div>",
                        unsafe_allow_html=True)

# ==========================================
# --- TAB 2: THERMAL LOAD ---
# ==========================================
with tabs[2]:
    col_t_in, col_t_out = st.columns([1, 1.5])
    with col_t_in:
        st.markdown("### 🌡️ Loading Snapshot")
        unit, cap = capacity_inputs("t2")
        t2_life_on = st.toggle("Insulation life calculation (IEEE C57.91)", value=True, key="t2_life_on",
                               help="Optional. Calculates paper-insulation aging with IEEE C57.91 from the hot-spot temperature. When off, the section, its inputs and its lines in the register, Excel file and report are hidden.")
        with st.form("thermal_form"):
            c1, c2 = st.columns(2)
            hour = c1.number_input("Hour (0-23)", min_value=0, max_value=23, value=14, key="t2_hour")
            month = c2.number_input("Month (1-12)", min_value=1, max_value=12, value=7, key="t2_month")
            loads = load_inputs("t2", unit)
            measured = st.number_input("Measured Top-Oil Temp (°C, optional)", value=0.0, min_value=0.0,
                                       help="If the DCS gives the measured top-oil temperature, enter it. The model always predicts the expected temperature; the risk uses the higher of the two, and a reading more than 5 °C hotter than expected raises an alarm. Leave 0 if not available.")
            t2_ins = None
            if t2_life_on:
                st.markdown("**Asset Data (for insulation life)**")
                t2_ins = insulation_inputs("t2")
            sub_thermal = st.form_submit_button("🌡️ Predict Oil Temperature", width=STRETCH)

    with col_t_out:
        if sub_thermal:
            t = pc.assess_thermal(loads, pc.to_kva(cap, unit), hour, month, model_thermal,
                                  measured if measured > 0 else None)
            banner(t['status'], f"Load factor {t['load_factor']:.1f}%  ·  Thermal risk {t['risk']:.0f}%", status_color(t['status']))
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Top-Oil Temp ({t['temp_basis']})", f"{t['assessed_temp']:.1f} °C")
            m2.metric("Load Factor", f"{t['load_factor']:.1f} %")
            m3.metric("Model Agreement", f"{t['agreement']:.1f} %")
            st.plotly_chart(gauge(t['assessed_temp'], [20, 120], [20, 90, 100, 120], TEMP_STEPS,
                                  f"Top-Oil Temperature used for risk (°C, {t['temp_basis'].lower()})", 320),
                            width=STRETCH)
            show_temp_compare(t)
            for f in t['findings'] or ["✅ Thermal parameters and loading are within safe limits."]:
                st.markdown(f"- {f}")
            if t2_ins:
                top_oil = measured if measured > 0 else t['temp']
                show_insulation(pc.insulation_life(top_oil, t['load_factor'], **t2_ins))
            st.markdown("#### Variables Driving This Result")
            st.plotly_chart(impact_chart(model_thermal, pc.THERMAL_FEATURES, t['input_df'].values[0],
                                         THERMAL_BASELINES, 'Reds'), width=STRETCH)
        else:
            st.markdown("<div class='pg-note'>Enter a loading snapshot from the DCS / SCADA and press <b>Predict Oil Temperature</b>.</div>",
                        unsafe_allow_html=True)

# ==========================================
# --- TAB 4: BATCH ANALYSIS ---
# ==========================================
with tabs[4]:
    st.markdown("### 📁 Batch Analysis")
    st.caption("Every row is assessed with the same rules as the single assessment, including the IEEE gas conditions and guardrails.")
    analysis_type = st.radio("Analysis Type", ["🧪 DGA Analysis", "🌡️ SCADA & Thermal"], horizontal=True)
    if "SCADA" in analysis_type:
        b_unit, b_cap = capacity_inputs("t3")
    up = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'], key="batch_up")
    if up:
        df_b = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
        if "DGA" in analysis_type:
            missing = [c for c in pc.DGA_FEATURES if c not in df_b.columns]
            if missing:
                st.error(f"⚠️ Missing columns: {', '.join(missing)}")
            else:
                out = []
                gas = df_b[pc.DGA_FEATURES].apply(pd.to_numeric, errors='coerce')
                for i, row in gas.iterrows():
                    if row.isna().any():
                        out.append({'Degradation Index': None, 'Gas Condition': None, 'DGA Risk (%)': None,
                                    'Status': 'INCOMPLETE', 'Duval': 'Missing or non-numeric: ' +
                                    ', '.join(pc.GAS_LABELS.get(f, f) for f in pc.DGA_FEATURES if pd.isna(row[f]))})
                        continue
                    r = pc.assess_dga(row.to_dict(), model_health)
                    out.append({'Degradation Index': round(r['index'], 1), 'Gas Condition': r['worst_condition'],
                                'DGA Risk (%)': round(r['risk'], 1), 'Status': r['status'], 'Duval': r['duval']})
                n_bad = sum(1 for o in out if o['Status'] == 'INCOMPLETE')
                if n_bad:
                    st.warning(f"{n_bad} row(s) have missing or non-numeric gas values and were not assessed (Status = INCOMPLETE).")
                res = pd.concat([df_b.reset_index(drop=True), pd.DataFrame(out)], axis=1)
                fig = px.line(res, y='Degradation Index', markers=True, title="Degradation Index by Sample")
                fig.add_hrect(y0=pc.DI_WARNING_MAX, y1=100, fillcolor=RED, opacity=0.08, line_width=0)
                st.plotly_chart(fig, width=STRETCH)
                st.dataframe(res, width=STRETCH)
                st.download_button("📥 Download Results (CSV)", res.to_csv(index=False), "dga_batch_results.csv", "text/csv")
        else:
            needed = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month']
            missing = [c for c in needed if c not in df_b.columns]
            if missing:
                st.error(f"⚠️ Missing columns: {', '.join(missing)}")
            else:
                cap_kva = pc.to_kva(b_cap, b_unit)
                res_t = [pc.assess_thermal(pc.thermal_inputs_from_row(row, b_unit), cap_kva, row['Hour'], row['Month'],
                                           model_thermal, row.get('Top Oil Temp') if 'Top Oil Temp' in df_b.columns else None)
                         for _, row in df_b.iterrows()]
                df_b['Predicted Top-Oil (°C)'] = np.round([x['temp'] for x in res_t], 1)
                df_b['Top-Oil Used for Risk (°C)'] = np.round([x['assessed_temp'] for x in res_t], 1)
                df_b['Thermal Risk (%)'] = np.round([x['risk'] for x in res_t], 1)
                df_b['Thermal Status'] = [x['status'] for x in res_t]
                fig = px.line(df_b, y='Predicted Top-Oil (°C)', markers=True, title="Predicted Top-Oil Temperature",
                              color_discrete_sequence=[RED])
                st.plotly_chart(fig, width=STRETCH)
                st.dataframe(df_b, width=STRETCH)
                st.download_button("📥 Download Results (CSV)", df_b.to_csv(index=False), "thermal_batch_results.csv", "text/csv")

# ==========================================
# --- TAB 3: ASSET RISK & FMEA ---
# ==========================================
with tabs[3]:
    st.markdown("### 📑 Overall Asset Risk (DGA + Thermal Fusion)")
    col_ex_in, col_ex_out = st.columns([1.15, 1.5])

    with col_ex_in:
        c_id1, c_id2 = st.columns(2)
        asset_id = c_id1.text_input("Asset ID", value="T01",
                                    help="IDs starting with R (e.g. R1, RB1, RL2) are filed as shunt reactors; all others as transformers.")
        asset_type = asset_type_from_id(asset_id)
        use_thermal = c_id2.toggle("Thermal data available", value=True,
                                   help="Turn off when the client has no DCS/SCADA loading data. Gases then carry 100% of the weight.")
        e_life_on = False
        if use_thermal:
            e_unit, e_cap = capacity_inputs("t4")
            e_life_on = st.toggle("Insulation life calculation (IEEE C57.91)", value=True, key="t4_life_on",
                                  help="Optional. Calculates paper-insulation aging with IEEE C57.91 from the hot-spot temperature. When off, the section, its inputs and its lines in the register, Excel file and report are hidden.")
        with st.form("risk_form"):
            e_dga = dga_inputs("t4", two_columns=True)
            e_thermal_inputs = None
            if use_thermal:
                st.markdown("**2. Loading & Ambient**")
                c1, c2 = st.columns(2)
                e_hour = c1.number_input("Hour (0-23)", min_value=0, max_value=23, value=10, key="t4_hour")
                e_month = c2.number_input("Month (1-12)", min_value=1, max_value=12, value=3, key="t4_month")
                e_loads = load_inputs("t4", e_unit)
                e_measured = st.number_input("Measured Top-Oil Temp (°C, optional)", value=0.0, min_value=0.0,
                                             key="t4_measured",
                                             help="From the DCS. The model always predicts the expected temperature; "
                                                  "the risk uses the higher of the two, and a reading more than 5 °C hotter "
                                                  "than expected raises an alarm. Leave 0 if not available.")
                e_ins = None
                if e_life_on:
                    st.markdown("**3. Asset Data (for insulation life)**")
                    e_ins = insulation_inputs("t4")
                e_thermal_inputs = (e_loads, pc.to_kva(e_cap, e_unit), e_hour, e_month, e_unit, e_cap,
                                    e_measured if e_measured > 0 else None)
            gen_report = st.form_submit_button("📊 Evaluate & Log to FMEA Register", width=STRETCH)

    with col_ex_out:
        if gen_report:
            d = pc.assess_dga(e_dga, model_health)
            t, ins = None, None
            if e_thermal_inputs:
                loads, cap_kva, hr, mo, unit, cap, meas = e_thermal_inputs
                t = pc.assess_thermal(loads, cap_kva, hr, mo, model_thermal, meas)
                if e_ins:
                    ins = pc.insulation_life(meas if meas else t['temp'], t['load_factor'], **e_ins)
                if ins and d['worst_condition'] >= 3:
                    ins['findings'].append("Fault gases are at IEEE Condition 3 or above: the real remaining life may be "
                                           "much shorter than this thermal estimate.")
            f = pc.fuse_risk(d, t)
            fm = pc.fmea_assessment(d, t)
            g = fm['governing']

            st.session_state['latest_eval'] = {'asset_id': asset_id, 'time': datetime.now(), 'dga': d, 'thermal': t,
                                               'fusion': f, 'insulation': ins, 'fmea': fm,
                                               'capacity': f"{e_thermal_inputs[5]:g} {e_thermal_inputs[4]}" if t else "n/a"}
            entry = pd.DataFrame([log_entry(asset_id, asset_type, d, t, f, fm, ins)])
            st.session_state['fmea_log'] = pd.concat([st.session_state['fmea_log'], entry], ignore_index=True).tail(100)

            banner(f['label'], f"Driver: {f['driver']}  ·  RPN {g['RPN']} ({g['RPN Level']})", f['color'])
            if f['missing']:
                st.warning(f"⚠️ Not assessed: {', '.join(f['missing'])} data. The overall risk uses the available dimension only "
                           "(re-weighted to 100%).")

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Degradation Index", f"{d['index']:.1f} / 100")
            rc2.metric(f"Top-Oil Temp ({t['temp_basis']})" if t else "Top-Oil Temp",
                       f"{t['assessed_temp']:.1f} °C" if t else "n/a")
            rc3.metric("Overall Risk", f"{f['overall']:.1f} %")
            agree = d['agreement'] if t is None else (d['agreement'] + t['agreement']) / 2
            rc4.metric("Model Agreement", f"{agree:.1f} %")

            st.plotly_chart(gauge(f['overall'], [0, 100], [0, int(pc.RISK_LOW_MAX), int(pc.RISK_MEDIUM_MAX), 100],
                                  RISK_STEPS, "Overall Asset Risk (%)", 300), width=STRETCH)

            dims = pd.DataFrame([
                {'Dimension': 'DGA (60%)' if t else 'DGA (100%)', 'Risk (%)': round(d['risk'], 1)},
                {'Dimension': 'Thermal (40%)', 'Risk (%)': round(t['risk'], 1) if t else None},
            ])
            st.dataframe(dims, width=STRETCH, hide_index=True)
            if f['reasons']:
                st.markdown("**Why this result:**")
                for reason in f['reasons']:
                    st.markdown(f"- {reason}")
            st.warning(f"**Asset Manager Action:** {f['action']}")
            if t:
                show_temp_compare(t)
            show_fmea(fm)
            if ins:
                show_insulation(ins)
            st.caption("Logged to this session's FMEA register below. In a live deployment this entry and its alert are "
                       "pushed to the client's CMMS / e-mail; that link is not connected in this prototype.")
        else:
            st.markdown("<div class='pg-note'>Enter the latest dissolved-gas reading and, if available, the loading snapshot, "
                        "then press <b>Evaluate & Log to FMEA Register</b>.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 FMEA Register (this session)")
    if not st.session_state['fmea_log'].empty:
        # Fleet view: latest assessment of each transformer, highest risk on top
        latest = (st.session_state['fmea_log'].groupby('Asset ID', as_index=False).last()
                  .sort_values('Overall Risk (%)', ascending=True))
        fig_fleet = go.Figure(go.Bar(
            x=latest['Overall Risk (%)'], y=latest['Asset ID'], orientation='h',
            marker_color=[status_color(s) for s in latest['Status']],
            text=[f"{r:.0f}%  ·  {d}" for r, d in zip(latest['Overall Risk (%)'], latest['Driver'])],
            textposition='outside', cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        for x, lbl in [(pc.RISK_LOW_MAX, "Watch"), (pc.RISK_MEDIUM_MAX, "Action")]:
            fig_fleet.add_vline(x=x, line_dash="dash", line_color="#888",
                                annotation_text=lbl, annotation_position="top")
        fig_fleet.update_layout(title="Fleet Risk Ranking (latest assessment per transformer)",
                                xaxis=dict(range=[0, 115], title="Overall Risk (%)"), yaxis_title=None,
                                height=140 + 38 * len(latest), margin=dict(l=10, r=30, t=60, b=40),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_fleet, width=STRETCH)
        st.download_button("📥 Download FMEA Register + Chart (Excel)",
                           fmea_excel(st.session_state['fmea_log'], latest),
                           f"PowerGuard_FMEA_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption("Excel file: sheet 1 has the fleet ranking with its chart, sheet 2 the full register. "
                   "To save the chart alone as an image, use the camera icon above the chart.")

        def color_risk(val):
            return f"color: {status_color(val)}; font-weight: bold"
        reg = st.session_state['fmea_log']
        empty = [c for c in ('Insulation Life (yrs)', 'Measured Temp', 'Temp Alarm') if c in reg.columns and reg[c].isna().all()]
        reg = reg.drop(columns=empty)
        st.dataframe(reg.style.map(color_risk, subset=['Status']).format(precision=1),
                     width=STRETCH, hide_index=True)
    else:
        st.caption("No assessments yet.")

# ==========================================
# --- TAB 5: AI DIAGNOSTIC REPORT ---
# ==========================================
with tabs[5]:
    ev = st.session_state.get('latest_eval')
    if ev is None:
        st.markdown("<br><h3 style='text-align:center;color:#888;'>💡 Run an assessment in the Asset Risk & FMEA tab first.</h3>",
                    unsafe_allow_html=True)
    else:
        d, t, f = ev['dga'], ev['thermal'], ev['fusion']
        st.markdown(f"<div class='pg-note'><h3 style='margin:0;color:{NAVY};'>📄 AI Executive Diagnostic Report</h3>"
                    f"Asset ID: <b>{ev['asset_id']}</b> · Assessed: {ev['time'].strftime('%Y-%m-%d %H:%M')}</div><br>",
                    unsafe_allow_html=True)

        if f['status'] == 'EXCELLENT':
            summary = (f"Transformer {ev['asset_id']} is in good condition with an overall risk of {f['overall']:.1f}%. "
                       f"Recommended action: {f['action']}")
        elif f['status'] == 'WATCH':
            summary = (f"Transformer {ev['asset_id']} needs attention. Overall risk is {f['overall']:.1f}%, driven by "
                       f"{f['driver'].lower()} indicators. Recommended action: {f['action']}")
        else:
            summary = (f"CRITICAL: Transformer {ev['asset_id']} has an overall risk of {f['overall']:.1f}%, driven by "
                       f"{f['driver'].lower()} indicators. {f['action']}")
        if f['missing']:
            summary += f" Note: {', '.join(f['missing'])} data was not available, so that dimension was not assessed."
        st.markdown(f"**Executive Summary:** {summary}")
        st.markdown("---")

        r1, r2 = st.columns([1, 1.6])
        with r1:
            st.plotly_chart(gauge(f['overall'], [0, 100], [0, int(pc.RISK_LOW_MAX), int(pc.RISK_MEDIUM_MAX), 100],
                                  RISK_STEPS, "Overall Risk (%)", 300), width=STRETCH)
        with r2:
            g = ev['fmea']['governing']
            banner(f['label'], f"Driver: {f['driver']} · RPN {g['RPN']} ({g['RPN Level']})", f['color'])
            st.markdown(f"**Mandatory Action:** {f['action']}")
            for reason in f['reasons']:
                st.markdown(f"- {reason}")

        st.markdown("---")
        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(duval_figure(d['inputs']['Methane'], d['inputs']['Ethylene'], d['inputs']['Acethylene'],
                                         d['duval'], 420), width=STRETCH)
            st.markdown(f"<p style='text-align:center;'><b>Duval result:</b> {d['duval']}</p>", unsafe_allow_html=True)
        with v2:
            vals = list(d['input_df'].values[0])
            feats, base = list(pc.DGA_FEATURES), list(DGA_BASELINES)
            if t is not None:
                imp_d = model_health.feature_importances_ * np.abs(np.array(vals) - DGA_BASELINES) / DGA_BASELINES
                imp_t = model_thermal.feature_importances_ * np.abs(t['input_df'].values[0] - THERMAL_BASELINES) / THERMAL_BASELINES
                all_imp = np.concatenate([imp_d, imp_t])
                all_imp = all_imp / all_imp.sum() * 100 if all_imp.sum() > 0 else all_imp
                df_imp = pd.DataFrame({'Feature': feats + pc.THERMAL_FEATURES, 'Deviation-weighted impact (%)': all_imp})
                df_imp = df_imp.sort_values('Deviation-weighted impact (%)', ascending=False).head(10).sort_values('Deviation-weighted impact (%)')
                fig = px.bar(df_imp, x='Deviation-weighted impact (%)', y='Feature', orientation='h',
                             color='Deviation-weighted impact (%)', color_continuous_scale='Blues',
                             title="Top 10 Variables (Combined)")
                fig.update_layout(height=420, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width=STRETCH)
            else:
                st.plotly_chart(impact_chart(model_health, feats, vals, DGA_BASELINES, top=10,
                                             title="Top 10 Variables (DGA)"), width=STRETCH)

        st.markdown("---")
        st.markdown("### 🧠 Expert System Findings")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**🧪 DGA Analysis**")
            for x in d['findings'] or ["✅ Dissolved gases are within normal limits."]:
                st.markdown(f"- {x}")
        with c2:
            st.warning("**🌡️ Thermal & Loading**")
            if t is None:
                st.markdown("- Not assessed: no loading data provided.")
            else:
                for x in t['findings'] or ["✅ Thermal parameters and loading are within safe limits."]:
                    st.markdown(f"- {x}")
        if t is not None:
            show_temp_compare(t)
        show_fmea(ev['fmea'])
        if ev.get('insulation'):
            show_insulation(ev['insulation'])

        report = io.StringIO()
        report.write("=========================================\nPOWERGUARD AI - EXECUTIVE DIAGNOSTIC REPORT\n"
                     "=========================================\n")
        report.write(f"Asset ID: {ev['asset_id']}\nAssessed: {ev['time'].strftime('%Y-%m-%d %H:%M')}\n"
                     f"Rated Capacity: {ev['capacity']}\n\nEXECUTIVE SUMMARY:\n{summary}\n\n")
        report.write("1. DISSOLVED GAS ANALYSIS (9 gases)\n-----------------------------------------\n")
        report.write(f"Degradation Index: {d['index']:.1f} / 100\nWorst IEEE gas condition: {d['worst_condition']}\n"
                     f"Duval: {d['duval']}\n")
        for x in d['findings']:
            report.write(f"* {x}\n")
        report.write("\n2. THERMAL & LOADING\n-----------------------------------------\n")
        if t is None:
            report.write("Not assessed (no loading data).\n")
        else:
            report.write(f"Predicted Top-Oil Temp: {t['temp']:.1f} °C\nLoad Factor: {t['load_factor']:.1f}%\n")
            if t.get('measured') is not None:
                report.write(f"Measured Top-Oil Temp (DCS): {t['measured']:.1f} °C  (difference {t['measured_deviation']:+.1f} °C)\n"
                             f"Temperature used for risk: {t['assessed_temp']:.1f} °C ({t['temp_basis'].lower()}, the higher of the two)\n")
            for x in t['findings']:
                report.write(f"* {x}\n")
        ins = ev.get('insulation')
        if ins:
            report.write("\nINSULATION LIFE (IEEE C57.91)\n-----------------------------------------\n")
            report.write(f"Hot-spot temperature: {ins['hot_spot']:.1f} °C\nAging acceleration factor: {ins['faa']:.2f}\n"
                         f"Life consumed: {ins['consumed_pct']:.0f}%\n"
                         f"Remaining life at this operating point: {years_txt(ins['remaining_years_now'])}\n"
                         f"Remaining life at rated aging: {years_txt(ins['remaining_years_normal'])}\n")
            for x in ins['findings']:
                report.write(f"* {x}\n")
        report.write(f"\n3. FMEA DIRECTIVE\n-----------------------------------------\nOverall Risk: {f['overall']:.1f}%\n"
                     f"Status: {f['status']}\nDriver: {f['driver']}\n"
                     f"Governing failure mode: {g['Failure Mode']}\n"
                     f"RPN = S {g['S']} x O {g['O']} x D {g['D']} = {g['RPN']} ({g['RPN Level']})\n"
                     f"Action: {f['action']}\n")
        report.write("All active failure modes:\n")
        for m in ev['fmea']['modes']:
            report.write(f"  - {m['Failure Mode']}: S{m['S']} x O{m['O']} x D{m['D']} = {m['RPN']} ({m['RPN Level']})\n")
        for reason in f['reasons']:
            report.write(f"* {reason}\n")
        report.write("=========================================\nGenerated by PowerGuard AI Engine\n")
        st.download_button("📥 Download Full Text Report", report.getvalue(),
                           f"Report_{ev['asset_id']}_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")

# ==========================================
# --- TAB 6: HISTORICAL STUDY ---
# ==========================================
with tabs[6]:
    st.markdown("### 📈 Historical Study")
    st.markdown("<div class='pg-note'>Replays each transformer's history in date order. Every assessment uses only the "
                "data that existed on that date, then the alerts are compared with the recorded failures and major "
                "maintenance events. The result shows how early the system would have warned.</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='pg-note'><b>No client data yet?</b> Keep <b>Use demo data</b> on. The app then "
                "generates a synthetic history for 3 transformers (one healthy, one developing arcing, one developing a "
                "thermal fault) with 2 recorded failures, so you can see how the study works. Nothing is uploaded, and "
                "the results are labeled as demo. When a client sends real files, turn it off and upload them.</div>",
                unsafe_allow_html=True)

    with st.expander("📄 Required file formats, and the demo files to open in Excel"):
        st.markdown(
            "- **Gas samples (required):** `Asset ID`, `Date`, and the 9 gas columns "
            f"(`{'`, `'.join(pc.DGA_FEATURES)}`).\n"
            "- **Loading data (optional):** `Asset ID`, `Timestamp`, `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, "
            "and optionally `Top Oil Temp`. Loads in MW/MVAR or kW/kVAR (choose below).\n"
            "- **Events (recommended):** `Asset ID`, `Date`, `Event` — failures, trips, forced outages.\n"
            "- **Assets (needed with loading data):** `Asset ID`, `Rated Capacity`.\n\n"
            "The buttons below download the full demo data set. Use them as templates for client files.")
        demo_dga, demo_scada, demo_events, demo_assets = pc.demo_history()
        t1, t2, t3, t4 = st.columns(4)
        t1.download_button("Demo gas samples", demo_dga.round(1).to_csv(index=False), "demo_gas_samples.csv", "text/csv")
        t2.download_button("Demo loading data", demo_scada.round(2).to_csv(index=False), "demo_loading.csv", "text/csv")
        t3.download_button("Demo events", demo_events.to_csv(index=False), "demo_events.csv", "text/csv")
        t4.download_button("Demo assets", demo_assets.to_csv(index=False), "demo_assets.csv", "text/csv")

    def read_any(f):
        return None if f is None else (pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f))

    s1, s2 = st.columns(2)
    with s1:
        f_dga = st.file_uploader("Gas samples", type=['csv', 'xlsx'], key="hs_dga")
        f_scada = st.file_uploader("Loading data (optional)", type=['csv', 'xlsx'], key="hs_scada")
    with s2:
        f_events = st.file_uploader("Events (failures / trips)", type=['csv', 'xlsx'], key="hs_events")
        f_assets = st.file_uploader("Assets (rated capacity)", type=['csv', 'xlsx'], key="hs_assets")

    o1, o2, o3, o4 = st.columns(4)
    hs_unit = o1.selectbox("Units", ["MVA", "kVA"], key="hs_unit")
    alert_thr = o2.number_input("Alert threshold (%)", 50.0, 95.0, 70.0, 5.0)
    lookback = o3.number_input("Look-back window (days)", 30, 1095, 365, 30)
    use_demo = o4.toggle("Use demo data", value=f_dga is None,
                         help="Synthetic data for demonstration only. Turn off and upload client files for a real study.")

    if st.button("▶️ Run Historical Study", type="primary"):
        if use_demo:
            dga_df, scada_df, events_df, assets_df = pc.demo_history()
            hs_unit_used = "MVA"
        else:
            dga_df, scada_df, events_df, assets_df = read_any(f_dga), read_any(f_scada), read_any(f_events), read_any(f_assets)
            hs_unit_used = hs_unit
        if dga_df is None:
            st.error("Upload the gas samples file, or turn on demo data.")
        else:
            missing = [c for c in ['Asset ID', 'Date'] + pc.DGA_FEATURES if c not in dga_df.columns]
            if missing:
                st.error(f"Gas samples file is missing: {', '.join(missing)}")
            else:
                caps = {}
                if assets_df is not None and len(assets_df):
                    if not {'Asset ID', 'Rated Capacity'} <= set(assets_df.columns):
                        st.error("Assets file needs the columns `Asset ID` and `Rated Capacity`.")
                        st.stop()
                    caps = {a: pc.to_kva(c, hs_unit_used) for a, c in zip(assets_df['Asset ID'], assets_df['Rated Capacity'])}
                if events_df is not None and len(events_df) and not {'Asset ID', 'Date'} <= set(events_df.columns):
                    st.error("Events file needs the columns `Asset ID`, `Date` and `Event`.")
                    st.stop()
                if scada_df is not None and len(scada_df):
                    need = ['Asset ID', 'Timestamp', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
                    miss = [c for c in need if c not in scada_df.columns]
                    if miss:
                        st.error(f"Loading data file is missing: {', '.join(miss)}")
                        st.stop()
                if scada_df is not None and not caps:
                    st.warning("Loading data was provided without an assets file, so the thermal dimension is skipped.")
                with st.spinner("Replaying history..."):
                    tl, es, fa = pc.run_historical_study(dga_df, scada_df, events_df, caps, hs_unit_used,
                                                         model_health, model_thermal, alert_thr, lookback)
                st.session_state['history'] = {'timeline': tl, 'events': es, 'false_alarms': fa,
                                               'events_raw': events_df, 'demo': use_demo, 'threshold': alert_thr}

    h = st.session_state.get('history')
    if h is not None and len(h['timeline']):
        tl, es, fa = h['timeline'], h['events'], h['false_alarms']
        if h['demo']:
            st.info("Showing **synthetic demo data**. These are not real measurements.")

        caught = int((es['Caught'] == 'Yes').sum()) if len(es) else 0
        lead = es['Lead Time (days)'].dropna()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transformers", tl['Asset ID'].nunique())
        k2.metric("Events caught", f"{caught} / {len(es)}" if len(es) else "no events")
        k3.metric("Median warning before event", f"{int(lead.median())} days" if len(lead) else "—")
        k4.metric("Alerts with no event", f"{fa['DGA'] + fa['Combined']} DGA · {fa['Thermal']} thermal")

        fig = px.line(tl, x='Date', y='Overall Risk (%)', color='Asset ID', markers=True,
                      title="Overall Risk Over Time")
        fig.add_hline(y=h['threshold'], line_dash="dash", line_color=RED,
                      annotation_text=f"Alert threshold {h['threshold']:.0f}%", annotation_position="top left")
        ev_raw = h['events_raw']
        if ev_raw is not None and len(ev_raw):
            for _, e in ev_raw.iterrows():
                fig.add_vline(x=pd.to_datetime(e['Date']), line_dash="dot", line_color="#555")
                fig.add_annotation(x=pd.to_datetime(e['Date']), y=102, text=f"⚠ {e['Asset ID']}", showarrow=False,
                                   font=dict(size=11))
        fig.update_layout(height=430, yaxis_range=[0, 108], paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width=STRETCH)

        if len(es):
            st.markdown("#### Events vs. Warnings")
            st.dataframe(es, width=STRETCH, hide_index=True)

        st.markdown("#### Assessment Timeline")
        asset_pick = st.selectbox("Transformer", sorted(tl['Asset ID'].unique()))
        view = tl[tl['Asset ID'] == asset_pick].copy()
        view['Date'] = view['Date'].dt.strftime('%Y-%m-%d')

        def colour(v):
            return f"color: {status_color(v)}; font-weight: bold"
        st.dataframe(view.style.map(colour, subset=['Status']).format(precision=1), width=STRETCH, hide_index=True)
        st.download_button("📥 Download Full Timeline (CSV)", tl.to_csv(index=False), "historical_study_timeline.csv",
                           "text/csv")
        st.caption("'Alerts with no event' counts alert episodes not followed by a recorded event within the look-back "
                   "window. Thermal alerts usually mean overloading or a cooling problem, not an internal fault.")


# ==========================================
# --- TAB 0: FLEET OVERVIEW (drawn last so it includes this run's assessments) ---
# ==========================================
SRC_PAGES, SRC_DEMO = "Results from the Asset Risk & FMEA page", "Demo fleet (synthetic)"
with tabs[0]:
    about_module()
    ov_src = st.radio("Show", [SRC_PAGES, SRC_DEMO], horizontal=True, key='ov_src')
    if ov_src == SRC_PAGES:
        st.markdown("<div class='pg-note'>Every transformer and reactor evaluated on the <b>Asset Risk &amp; FMEA</b> page "
                    "is collected here. The latest assessment per asset is kept.</div>", unsafe_allow_html=True)
        log = st.session_state['fmea_log']
    else:
        st.info("Synthetic demo fleet (4 transformers and 4 shunt reactors of a 400/220 kV substation), assessed with the "
                "same engine as the Asset Risk & FMEA page. Not real measurements.")
        log = demo_fleet()
    if log.empty:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:30px'>💡 No results yet: evaluate an asset on the "
                    "Asset Risk &amp; FMEA page, or show the demo fleet.</h3>", unsafe_allow_html=True)
    else:
        R = log.groupby('Asset ID', as_index=False).last()
        if 'Asset Type' not in R.columns:
            R['Asset Type'] = 'Transformer'
        R['Asset Type'] = R['Asset Type'].fillna('Transformer')
        R = R.sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Assets", len(R))
        k2.metric("🔴 Action required", int((R['Status'] == 'ACTION').sum()))
        k3.metric("🟠 Watch", int((R['Status'] == 'WATCH').sum()))
        k4.metric("🟢 Normal", int((R['Status'] == 'EXCELLENT').sum()))

        def card(col, title, sub_df, extra):
            if sub_df.empty:
                col.markdown(f"<div class='pg-card'><h4>{title}</h4><div class='big' style='color:#999'>—</div>"
                             f"<div class='sub'>no results</div></div>", unsafe_allow_html=True)
                return
            w = sub_df.iloc[0]
            c = status_color(w['Status'])
            col.markdown(f"<div class='pg-card' style='border-top:5px solid {c}'><h4>{title}</h4>"
                         f"<div class='big' style='color:{c}'>{w['Overall Risk (%)']:.0f}%</div>"
                         f"<div class='sub'>{extra(w, sub_df)}</div></div>", unsafe_allow_html=True)

        cc = st.columns(4)
        summ = lambda w, df_: (f"worst: {w['Asset ID']} · {w['Driver']}<br>{len(df_)} assets · "
                               f"{int((df_['Status'] == 'ACTION').sum())} action · {int((df_['Status'] == 'WATCH').sum())} watch")
        card(cc[0], "Transformers", R[R['Asset Type'] == 'Transformer'], summ)
        card(cc[1], "Shunt reactors", R[R['Asset Type'] != 'Transformer'], summ)
        card(cc[2], "Highest RPN", R.sort_values('RPN', ascending=False),
             lambda w, df_: f"{w['Asset ID']} · RPN {int(w['RPN'])} ({w['RPN Level']})<br>{w['Failure Mode']}")
        life = R.dropna(subset=['Insulation Life (yrs)']) if 'Insulation Life (yrs)' in R.columns else R.iloc[0:0]
        if life.empty:
            cc[3].markdown("<div class='pg-card'><h4>Shortest insulation life</h4><div class='big' style='color:#999'>—</div>"
                           "<div class='sub'>not calculated</div></div>", unsafe_allow_html=True)
        else:
            w = life.sort_values('Insulation Life (yrs)').iloc[0]
            v = w['Insulation Life (yrs)']
            c = RED if v < 10 else AMBER if v < 20 else GREEN
            cc[3].markdown(f"<div class='pg-card' style='border-top:5px solid {c}'><h4>Shortest insulation life</h4>"
                           f"<div class='big' style='color:{c}'>{'> 60' if v >= 60 else f'{v:.1f}'} yrs</div>"
                           f"<div class='sub'>{w['Asset ID']} · at its current operating point</div></div>", unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1.25, 1])
        with left:
            top = R.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top['Overall Risk (%)'], y=top['Asset ID'], orientation='h',
                                   marker_color=[status_color(s) for s in top['Status']],
                                   text=[f"{r:.0f}%  ·  {d}" for r, d in zip(top['Overall Risk (%)'], top['Driver'])],
                                   textposition='outside', cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            for x, lbl in [(pc.RISK_LOW_MAX, "Watch"), (pc.RISK_MEDIUM_MAX, "Action")]:
                fig.add_vline(x=x, line_dash="dash", line_color="#888", annotation_text=lbl, annotation_position="top")
            fig.update_layout(title=f"Fleet risk ranking (top {len(top)})", xaxis=dict(range=[0, 118], title="Overall Risk (%)"),
                              yaxis=dict(type='category'), height=140 + 34 * len(top), margin=dict(l=10, r=30, t=60, b=40),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, width=STRETCH, key='ov_rank')
        with right:
            types = [t_ for t_ in ['Transformer', 'Shunt reactor'] if t_ in set(R['Asset Type'])]
            ft = go.Figure()
            for stt, nm in [('EXCELLENT', 'Normal'), ('WATCH', 'Watch'), ('ACTION', 'Action')]:
                ft.add_bar(x=types, y=[int(((R['Asset Type'] == t_) & (R['Status'] == stt)).sum()) for t_ in types],
                           name=nm, marker_color=status_color(stt), texttemplate='%{y}', textposition='inside')
            ft.update_layout(barmode='stack', title='Assets by type and status', height=300,
                             yaxis=dict(title='Assets', dtick=1 if len(R) <= 12 else None),
                             legend=dict(orientation='h', y=-0.2), margin=dict(l=40, r=10, t=50, b=40))
            st.plotly_chart(ft, width=STRETCH, key='ov_type')
            drv = R['Driver'].value_counts()
            fd = go.Figure(go.Bar(x=drv.index.tolist(), y=drv.values.tolist(), marker_color=NAVY, texttemplate='%{y}',
                                  textposition='inside'))
            fd.update_layout(title='Main risk driver', height=260, yaxis=dict(title='Assets', dtick=1 if len(R) <= 12 else None),
                             margin=dict(l=40, r=10, t=50, b=40))
            st.plotly_chart(fd, width=STRETCH, key='ov_drv')
        st.markdown("#### All assets (latest assessment)")
        show = ['Asset ID', 'Asset Type', 'Overall Risk (%)', 'Status', 'Driver', 'Failure Mode', 'RPN', 'RPN Level',
                'DGA Index', 'Pred Temp', 'Insulation Life (yrs)', 'Timestamp', 'Action Required']
        show = [c for c in show if c in R.columns and not R[c].isna().all()]
        st.dataframe(R[show].style.map(lambda v: f"color: {status_color(v)}; font-weight: bold", subset=['Status'])
                     .format(precision=1), width=STRETCH, hide_index=True)
        st.download_button("📥 Download Fleet Ranking + Register (Excel)", fmea_excel(log, R),
                           f"PowerGuard_Fleet_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='ov_xlsx')

st.markdown("---")
st.markdown(FOOTER, unsafe_allow_html=True)
