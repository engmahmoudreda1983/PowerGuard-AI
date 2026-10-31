import hmac
import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pg_paths
import networkguard_core as ng
import networkguard_plots as npl
import networkguard_report_pdf as pdfrep

pass  # page config is set once by the PowerGuard AI platform
STRETCH = 'stretch'
NAVY, BLUE = '#0b2e59', '#4da6ff'
GREEN, AMBER, RED = '#1a9e5c', '#d99a00', '#d63a3a'
STATUS_COLOR = {'EXCELLENT': GREEN, 'WATCH': AMBER, 'ACTION': RED, 'OUT': '#9aa5b5', 'PASS': GREEN, 'CHECK': RED}
STATUS_LABEL = {'EXCELLENT': 'NORMAL', 'WATCH': 'WATCH', 'ACTION': 'ACTION', 'OUT': 'OUT OF SERVICE'}
SRC_PAGES, SRC_DEMO = 'Results from the analysis pages', 'Demo network (simulated, anonymised)'
ANALYSES = [('lf', '⚡', 'Load Flow', 'Voltages, flows, loading and losses'), ('vd', '📉', 'Voltage Drop', 'Voltage along every circuit and bus limits'),
            ('sc', '💥', 'Short Circuit', 'IEC 60909 fault levels and breaker duty'), ('prot', '🛡️', 'Protection Coordination', 'Distance zones and overcurrent grading'),
            ('cont', '🔁', 'N-1 Contingency', 'Outage of every circuit and transformer'), ('val', '✅', 'Validation', 'Engine checked against published results')]

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:16px 22px;border-radius:10px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.pg-header img {height:64px;background:#fff;border-radius:8px;padding:4px 8px;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;margin-bottom:10px;}
.ng-card {border:1px solid #d5dce6;border-radius:14px;padding:14px 16px;background:#fff;height:100%;}
.ng-card .ic {font-size:1.7rem;}
.ng-card h4 {margin:4px 0 2px 0;color:#0b2e59;font-size:1.02rem;}
.ng-card .sub {font-size:.8rem;color:#5a6575;min-height:2.2em;}
.ng-card .res {font-size:.85rem;margin-top:6px;}
.ng-kpi {border:1px solid #d5dce6;border-radius:12px;padding:10px 14px;background:#fff;}
.ng-kpi .l {font-size:.72rem;color:#5a6575;text-transform:uppercase;letter-spacing:.05em;font-weight:600;}
.ng-kpi .v {font-size:1.45rem;font-weight:750;color:#0b2e59;}
.ng-chip {display:inline-block;padding:2px 9px;border-radius:999px;font-size:.72rem;font-weight:700;color:#fff;}
.pg-foot {text-align:center;color:#888;font-size:12px;padding:10px;}
</style>
""", unsafe_allow_html=True)

FOOTER = """
<div class='pg-foot'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
<em>Confidential and Proprietary.</em></div>
"""


@st.cache_data
def logo_b64():
    try:
        import base64
        from PIL import Image
        im = Image.open(pg_paths.LOGO); im.thumbnail((360, 200))
        buf = io.BytesIO(); im.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def secrets_ok():
    try:
        return "APP_USERNAME" in st.secrets and "APP_PASSWORD" in st.secrets
    except Exception:
        return False


# ==========================================
# --- Login ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align:center;color:#4da6ff;'>⚡ NetworkGuard AI Login</h1>", unsafe_allow_html=True)
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

_, col_r = st.columns([10, 1])
with col_r:
    if False:  # sign out is in the platform header and sidebar
        st.session_state['logged_in'] = False
        st.rerun()

_logo = logo_b64()
st.markdown(f"""
<div class='pg-header'>
  <div>
    <h1>⚡ NetworkGuard AI: Network Analysis</h1>
    <p>Single-line diagram from the equipment data, load flow, voltage drop, short circuit (IEC 60909), protection coordination and N-1 contingency</p>
  </div>
  {f"<img src='data:image/png;base64,{_logo}' alt='PowerGuard AI'>" if _logo else ''}
</div>""", unsafe_allow_html=True)

ss = st.session_state
for key, default in [('net_data', None), ('net_name', 'Demo network'), ('res', {}), ('register', []), ('studies', []), ('sel', None)]:
    if key not in ss:
        ss[key] = default
if ss['net_data'] is None:
    ss['net_data'] = ng.demo()


def chip(status, text=None):
    return f"<span class='ng-chip' style='background:{STATUS_COLOR.get(status, '#64748b')}'>{text or STATUS_LABEL.get(status, status)}</span>"


def kpis(items):
    cols = st.columns(len(items))
    for c, (l, v) in zip(cols, items):
        c.markdown(f"<div class='ng-kpi'><div class='l'>{l}</div><div class='v'>{v}</div></div>", unsafe_allow_html=True)


def style_status(df, col='Status'):
    def f(v):
        c = STATUS_COLOR.get(v)
        return f'background-color:{c};color:white;font-weight:600' if c else ''
    try:
        sty = df.style.format(precision=2, na_rep='-', thousands=',')
        return sty.map(f, subset=[col]) if col in df else sty
    except Exception:
        return df


def show(df, col='Status', height=None):
    st.dataframe(style_status(df, col), width=STRETCH, hide_index=True, height=height or 'auto')


def csv_button(df, name, label='⬇️ CSV', key=None):
    st.download_button(label, df.to_csv(index=False).encode(), name, 'text/csv', key=key or f'dl_{name}')


def invalidate():
    ss['res'] = {}


def update_register():
    r = ss['res']
    R = ng.asset_rows(r.get('lf'), r.get('sc'), r.get('prot'), r.get('cont'))
    ss['register'] = R.to_dict('records') if len(R) else []


def status_of(key, r):
    if key == 'lf':
        sm = r['summary']; return 'ACTION' if sm['Violations'] else ('WATCH' if sm['Warnings'] else 'EXCELLENT')
    if key == 'vd':
        return 'WATCH' if (r['line']['Status'] == 'WATCH').any() else 'EXCELLENT'
    if key == 'sc':
        return 'ACTION' if (r['bus']['Status'] == 'ACTION').any() else ('WATCH' if (r['bus']['Status'] == 'WATCH').any() else 'EXCELLENT')
    if key == 'prot':
        s = pd.concat([r['distance'].get('Status', pd.Series(dtype=str)), r['oc_pairs'].get('Status', pd.Series(dtype=str))])
        return 'ACTION' if (s == 'ACTION').any() else ('WATCH' if (s == 'WATCH').any() else 'EXCELLENT')
    if key == 'cont':
        s = r['table']['Status']; return 'ACTION' if (s == 'ACTION').any() else ('WATCH' if (s == 'WATCH').any() else 'EXCELLENT')
    if key == 'val':
        return 'EXCELLENT' if (r['Result'] == 'PASS').all() else 'ACTION'
    return 'EXCELLENT'


def headline(key, r):
    if key == 'lf':
        sm = r['summary']; return f"V {sm['Min V (pu)']:.3f}-{sm['Max V (pu)']:.3f} pu · max loading {sm['Max Line Loading (%)']:.0f}% · losses {sm['Losses (MW)']:.0f} MW"
    if key == 'vd':
        return f"largest drop {r['line']['|Drop| (%)'].max():.1f}% on {r['line'].iloc[0]['Line']}"
    if key == 'sc':
        b = r['bus'].sort_values('Breaker Duty (%)', ascending=False).iloc[0]
        return f"highest duty {b['Breaker Duty (%)']:.0f}% at {b['Bus']} ({max(b['Ik3 (kA)'], b['Ik1 (kA)']):.1f} kA)"
    if key == 'prot':
        n = int((r['distance'].get('Status', pd.Series(dtype=str)) != 'EXCELLENT').sum()) + int((r['oc_pairs'].get('Status', pd.Series(dtype=str)) != 'EXCELLENT').sum())
        return f"{len(r['distance'])} distance relays, {len(r['oc_pairs'])} grading pairs · {n} finding(s)"
    if key == 'cont':
        t = r['table']; return f"{len(t)} outages · {(t['Status'] == 'ACTION').sum()} with violations or separation"
    if key == 'val':
        return f"{(r['Result'] == 'PASS').sum()} of {len(r)} checks pass"
    return ''


@st.cache_data(show_spinner=False)
def demo_results():
    D = ng.demo()
    lf = ng.load_flow(D)
    return {'lf': lf, 'vd': ng.voltage_drop(lf), 'sc': ng.short_circuit(D, 'max', 'CJ-400'), 'prot': ng.protection(D), 'cont': ng.contingency(D), 'val': ng.validation()}


def element_options(data, kind):
    return list(ng.clean(data)[kind][ng.id_col(kind)])


def sld(data, key, **kw):
    fig = npl.sld_figure(data, select=ss.get('sel'), **kw)
    ev = st.plotly_chart(fig, width=STRETCH, key=key, on_select='rerun', selection_mode='points', config={'scrollZoom': True, 'displaylogo': False})
    try:
        pts = ev.selection.points if ev and ev.selection else []
        for p in pts:
            cd = p.get('customdata')
            if cd and len(cd) == 2:
                if ss.get('sel') != (cd[0], cd[1]):
                    ss['sel'] = (cd[0], cd[1])
                    ss['ng_pending'] = (cd[0], cd[1])
                    st.rerun()
                break
    except Exception:
        pass


tabs = st.tabs(['🏠 Network Overview', '🗺️ Network & Equipment Data', '⚡ Load Flow', '📉 Voltage Drop', '💥 Short Circuit',
                '🛡️ Protection Coordination', '🔁 N-1 Contingency', '📄 Combined Report', '✅ Validation'])

# ============================================================== 0. Overview
with tabs[0]:
    src = st.radio('Show', [SRC_PAGES, SRC_DEMO], horizontal=True, key='ov_src')
    if src == SRC_DEMO:
        with st.spinner('Running all analyses on the demo network...'):
            R = demo_results()
        D = ng.demo(); name = 'Demo network'
    else:
        R = ss['res']; D = ss['net_data']; name = ss['net_name']
    d = ng.clean(D)
    kpis([('Network', name), ('Substations', len(d['Substations'])), ('Buses', len(d['Buses'])), ('Circuits', len(d['Lines'])),
          ('Transformers', len(d['Transformers'])), ('Relays', len(d['Relays']))])
    st.markdown('')
    cols = st.columns(3)
    for i, (k, ic, nm, ds) in enumerate(ANALYSES):
        r = R.get(k)
        if r is not None:
            stt = status_of(k, r)
            body = f"<div class='res'>{chip(stt)} {headline(k, r)}</div>"
        else:
            body = "<div class='res' style='color:#8a97ab'>Not run yet: open its tab and press Analyze.</div>"
        cols[i % 3].markdown(f"<div class='ng-card' style='margin-bottom:12px'><div class='ic'>{ic}</div><h4>{nm}</h4><div class='sub'>{ds}</div>{body}</div>",
                             unsafe_allow_html=True)
    lf, scr = R.get('lf'), R.get('sc')
    st.markdown('#### Single-line diagram' + (' with the load flow results' if lf else ''))
    st.plotly_chart(npl.sld_figure(D, lf=lf, scr=scr, mode='lf' if lf else 'network'), width=STRETCH, key='ov_sld', config={'scrollZoom': True, 'displaylogo': False})
    F = ng.asset_rows(lf, scr, R.get('prot'), R.get('cont'))
    if len(F):
        F = F[F['Status'] != 'EXCELLENT']
        st.markdown(f'#### Findings ({len(F)})')
        show(F)
    elif not R:
        st.info('No analysis has been run on this network yet.')
    else:
        st.success('No finding: all equipment within limits.')

# ============================================================== 1. Network & data
with tabs[1]:
    if ss.get('ng_pending'):
        ss['ed_kind'], ss['ed_id'] = ss.pop('ng_pending')
        ss['ed_new'] = False
    st.markdown("<div class='pg-note'>Press an equipment on the diagram (or choose it below) to open its data. Fill the values here or "
                "download the equipment's Excel form, fill it and upload it. Empty values use typical values and are marked.</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.2])
    nm = c1.text_input('Network name', ss['net_name'], key='ng_name')
    if nm != ss['net_name']:
        ss['net_name'] = nm
    if c2.button('🏭 Load the demo network', width=STRETCH):
        ss['net_data'] = ng.demo(); ss['net_name'] = 'Demo network'; ss['sel'] = None; invalidate(); st.rerun()
    if c3.button('🆕 Start an empty network', width=STRETCH):
        ss['net_data'] = ng.clean(ng.empty()); ss['net_name'] = 'New network'; ss['sel'] = None; invalidate(); st.rerun()
    c4.download_button('⬇️ Whole network (Excel)', ng.network_excel(ss['net_data']), f"{ss['net_name'].replace(' ', '_')}_network.xlsx", width=STRETCH)
    with st.expander('⬆️ Upload a whole network (Excel with one sheet per equipment type)'):
        up = st.file_uploader('Network Excel', type=['xlsx', 'xls'], key='ng_up_net')
        if up is not None and st.button('Load this network', key='ng_load_net'):
            try:
                ss['net_data'] = ng.read_network_excel(up); ss['net_name'] = up.name.rsplit('.', 1)[0]; ss['sel'] = None; invalidate(); st.rerun()
            except Exception as e:
                st.error(f'Could not read the file: {e}')
    data = ss['net_data']
    errs, miss = ng.validate_links(data), ng.missing_required(data)
    _, notes = ng.filled(data)
    a, b, c = st.columns(3)
    a.metric('Wrong references', len(errs)); b.metric('Missing required values', len(miss)); c.metric('Typical / computed values used', len(notes))
    if errs:
        st.error('  \n'.join(errs[:12]))
    left, right = st.columns([1.55, 1])
    with left:
        sld(data, 'ed_sld', height=640)
    with right:
        kinds = ng.ORDER
        kind = st.selectbox('Equipment type', kinds, format_func=lambda k: f'{ng.ICON[k]} {k}', key='ed_kind')
        opts = element_options(data, kind)
        new = st.toggle('Add a new equipment', key='ed_new')
        if new:
            eid = st.text_input(f'New {ng.TYPE_OF[kind].lower()} name', key='ed_new_id')
        else:
            if not opts:
                st.info(f'No {kind.lower()} yet. Use "Add a new equipment".')
                eid = None
            else:
                if ss.get('ed_id') not in opts:
                    ss['ed_id'] = opts[0]
                eid = st.selectbox('Equipment', opts, key='ed_id')
                ss['sel'] = (kind, eid)
        if eid:
            T = ng.element_table(data, kind, eid)
            T['Value'] = T['Value'].astype(str).replace({'nan': ''})
            st.markdown(f"**{ng.ICON[kind]} {eid}**")
            E = st.data_editor(T, key=f'ed_tab_{kind}_{eid}', hide_index=True, width=STRETCH, height=min(560, 38 + 35 * len(T)),
                               disabled=['Parameter', 'Unit', 'Description', 'Source'],
                               column_config={'Description': st.column_config.TextColumn(width='large'), 'Value': st.column_config.TextColumn(width='small')})
            b1, b2 = st.columns(2)
            if b1.button('💾 Save', type='primary', width=STRETCH, key='ed_save'):
                ss['net_data'] = ng.set_element(data, kind, eid, E); invalidate(); st.rerun()
            if not new and b2.button('🗑️ Delete', width=STRETCH, key='ed_del'):
                ss['net_data'] = ng.delete_element(data, kind, eid); ss['sel'] = None; invalidate(); st.rerun()
            st.download_button(f'⬇️ Excel form of {eid}', ng.element_excel(data, kind, eid), f'{kind}_{eid}.xlsx'.replace('/', '-'), width=STRETCH, key='ed_xl')
            ef = st.file_uploader(f'⬆️ Upload the filled form of {eid}', type=['xlsx', 'xls'], key=f'ed_up_{kind}_{eid}')
            if ef is not None:
                try:
                    k2, id2, t2 = ng.read_element_excel(ef)
                    st.caption(f"Form for {k2 or kind} {id2 or eid}: {t2['Value'].notna().sum()} values")
                    if st.button('Apply the uploaded values', key='ed_apply'):
                        ss['net_data'] = ng.set_element(data, k2 if k2 in ng.SCHEMA else kind, id2 or eid, t2); invalidate(); st.rerun()
                except Exception as e:
                    st.error(f'Could not read the form: {e}')
    with st.expander(f'All {kind.lower()} ({len(ng.clean(data)[kind])})'):
        st.dataframe(ng.clean(data)[kind], width=STRETCH, hide_index=True)
    with st.expander('📐 Line parameters from conductor and tower (libraries)'):
        cc1, cc2, cc3 = st.columns(3)
        cn = cc1.selectbox('Conductor', list(ng.CONDUCTORS['Conductor']), key='lp_c')
        tw = cc2.selectbox('Tower', list(ng.TOWERS['Tower']), key='lp_t')
        tt = cc3.number_input('Conductor temperature (°C)', 20.0, 120.0, 75.0, key='lp_temp')
        p = ng.line_parameters(cn, tw, tt)
        st.dataframe(pd.DataFrame([{'R1 ohm/km': round(p['r1'], 5), 'X1 ohm/km': round(p['x1'], 4), 'C1 nF/km': round(p['c1'], 2),
                                    'R0 ohm/km': round(p['r0'], 4), 'X0 ohm/km': round(p['x0'], 4), 'C0 nF/km': round(p['c0'], 2),
                                    'GMD m': round(p['gmd'], 2), 'Bundle GMR m': round(p['gmr_b'], 4), 'Earth depth De m': round(p['De'], 0)}]), hide_index=True, width=STRETCH)
        st.caption('X1 = ω·2·10⁻⁴·ln(GMD/GMR) Ω/km; C1 = 55.63/ln(GMD/r_eq) nF/km; zero sequence by Carson (earth resistivity from the tower library), earth wires neglected.')
        st.dataframe(ng.CONDUCTORS, hide_index=True, width=STRETCH); st.dataframe(ng.TOWERS, hide_index=True, width=STRETCH)
    if notes:
        with st.expander(f'Values filled with typical or computed values ({len(notes)})'):
            st.dataframe(pd.DataFrame(notes, columns=['Table', 'Equipment', 'Parameter', 'Value used']), hide_index=True, width=STRETCH)
    if miss:
        with st.expander(f'Missing required values ({len(miss)})'):
            st.dataframe(pd.DataFrame(miss, columns=['Table', 'Equipment', 'Parameter']), hide_index=True, width=STRETCH)

data = ss['net_data']
d = ng.clean(data)
LINES = list(d['Lines']['Line']); TRAFOS = list(d['Transformers']['Transformer']); REACT = list(d['Reactors']['Reactor']); BUSES = list(d['Buses']['Bus'])


def lf_inputs(prefix):
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    scale = c1.number_input('Load scaling (%)', 10.0, 200.0, 100.0, 5.0, key=f'{prefix}_scale') / 100
    outs = c2.multiselect('Equipment out of service', LINES + TRAFOS, key=f'{prefix}_out')
    roff = c3.multiselect('Reactors switched out', REACT, key=f'{prefix}_roff')
    warn = c4.number_input('Loading warning (%)', 50.0, 100.0, 80.0, 5.0, key=f'{prefix}_warn')
    return scale, tuple(outs), tuple(roff), warn


def run_lf(scale, outs, roff, warn):
    lf = ng.load_flow(data, scale, outs, roff, warn)
    ss['res'] = {**ss['res'], 'lf': lf}
    return lf


# ============================================================== 2. Load flow
with tabs[2]:
    scale, outs, roff, warn = lf_inputs('lf')
    if st.button('⚡ Analyze load flow', type='primary', key='lf_go'):
        if ng.validate_links(data):
            st.error('Fix the wrong references on the data page first.')
        else:
            with st.spinner('Newton-Raphson load flow...'):
                try:
                    run_lf(scale, outs, roff, warn); update_register()
                except Exception as e:
                    st.error(f'The load flow did not converge: {e}')
    lf = ss['res'].get('lf')
    if lf:
        sm = lf['summary']
        st.markdown(f"{chip(status_of('lf', lf))} &nbsp; {'Converged' if lf['converged'] else 'Not converged'} · {sm['Violations']} violation(s), {sm['Warnings']} warning(s)",
                    unsafe_allow_html=True)
        kpis([('Losses', f"{sm['Losses (MW)']} MW"), ('Losses %', f"{sm['Losses (%)']}%"), ('Min V', f"{sm['Min V (pu)']:.3f} pu"), ('Max V', f"{sm['Max V (pu)']:.3f} pu"),
              ('Max line loading', f"{sm['Max Line Loading (%)']:.0f}%"), ('Line charging', f"{sm['Line Charging (MVAr)']:.0f} MVAr"), ('Reactors', f"{sm['Reactors (MVAr)']:.0f} MVAr")])
        sld(data, 'lf_sld', lf=lf, mode='lf', height=620)
        st.markdown('##### Bus voltages'); show(lf['bus']); csv_button(lf['bus'], 'load_flow_buses.csv')
        st.markdown('##### Circuits'); show(lf['line'].drop(columns=['In Service'])); csv_button(lf['line'], 'load_flow_circuits.csv')
        if len(lf['trafo']):
            st.markdown('##### Transformers'); show(lf['trafo'].drop(columns=['In Service'])); csv_button(lf['trafo'], 'load_flow_transformers.csv')
        st.markdown('##### Grids, converters and reactors'); st.dataframe(lf['sources'], width=STRETCH, hide_index=True)
        st.download_button('📄 Load flow report (PDF)', pdfrep.study_pdf(data, {'lf': lf}, 'Load Flow Study', ss['net_name']), 'Load_Flow_Report.pdf', key='lf_pdf')
    else:
        st.info('Set the case and press Analyze.')

# ============================================================== 3. Voltage drop
with tabs[3]:
    scale, outs, roff, warn = lf_inputs('vd')
    lim = st.number_input('Voltage drop limit along a circuit (%)', 0.5, 15.0, 3.0, 0.5, key='vd_lim')
    if st.button('📉 Analyze voltage drop', type='primary', key='vd_go'):
        with st.spinner('Load flow and voltage drop...'):
            try:
                lf = run_lf(scale, outs, roff, warn)
                ss['res'] = {**ss['res'], 'vd': ng.voltage_drop(lf, lim)}; update_register()
            except Exception as e:
                st.error(f'The load flow did not converge: {e}')
    vd = ss['res'].get('vd')
    if vd:
        for n in vd['notes']:
            st.markdown(f"<div class='pg-note'>{n}</div>", unsafe_allow_html=True)
        B = vd['bus']
        fig = go.Figure()
        fig.add_bar(x=B['Bus'], y=B['V (pu)'], marker_color=[STATUS_COLOR[s] for s in B['Status']], text=[f'{v:.3f}' for v in B['V (pu)']], textposition='outside')
        fig.add_hline(y=float(B['Vmax'].iloc[0]), line=dict(color=RED, dash='dash'), annotation_text='Vmax')
        fig.add_hline(y=float(B['Vmin'].iloc[0]), line=dict(color=RED, dash='dash'), annotation_text='Vmin')
        fig.update_layout(template='plotly_white', height=360, title='Bus voltages against the limits', yaxis=dict(range=[min(0.9, B['V (pu)'].min() - 0.02), max(1.1, B['V (pu)'].max() + 0.02)], title='pu'))
        st.plotly_chart(fig, width=STRETCH)
        L = vd['line']
        fig2 = go.Figure(go.Bar(x=L['Line'], y=L['Voltage Drop (%)'], marker_color=[AMBER if s == 'WATCH' else (BLUE if v < 0 else GREEN) for s, v in zip(L['Status'], L['Voltage Drop (%)'])]))
        fig2.add_hline(y=lim, line=dict(color=AMBER, dash='dash')); fig2.add_hline(y=-lim, line=dict(color=AMBER, dash='dash'))
        fig2.update_layout(template='plotly_white', height=340, title='Voltage drop along each circuit (negative = voltage rise)', yaxis_title='%')
        st.plotly_chart(fig2, width=STRETCH)
        show(L); csv_button(L, 'voltage_drop.csv')
        st.markdown('##### Bus voltage margins'); show(B[['Bus', 'Substation', 'kV', 'V (pu)', 'V (kV)', 'Vmin', 'Vmax', 'Margin to Vmin (%)', 'Margin to Vmax (%)', 'Status']])
    else:
        st.info('Set the case and press Analyze.')

# ============================================================== 4. Short circuit
with tabs[4]:
    c1, c2 = st.columns(2)
    case = c1.radio('Case (IEC 60909)', ['max', 'min'], horizontal=True, format_func=lambda x: 'Maximum (c = 1.1, breaker duty)' if x == 'max' else 'Minimum (c = 1.0, protection sensitivity)', key='sc_case')
    fb = c2.selectbox('Fault location for the branch currents', BUSES, index=BUSES.index('CJ-400') if 'CJ-400' in BUSES else 0, key='sc_bus')
    if st.button('💥 Analyze short circuit', type='primary', key='sc_go'):
        with st.spinner('IEC 60909 three-phase and single-phase faults at every bus...'):
            try:
                ss['res'] = {**ss['res'], 'sc': ng.short_circuit(data, case, fb)}; update_register()
            except Exception as e:
                st.error(f'Short circuit calculation failed: {e}')
    scr = ss['res'].get('sc')
    if scr:
        T = scr['bus']
        w = T.sort_values('Breaker Duty (%)', ascending=False).iloc[0]
        kpis([('Highest fault level', f"{max(T['Ik3 (kA)'].max(), T['Ik1 (kA)'].max()):.1f} kA"), ('Highest breaker duty', f"{w['Breaker Duty (%)']:.0f}% at {w['Bus']}"),
              ('Buses above rating', int((T['Status'] == 'ACTION').sum())), ('Buses at 90-100 %', int((T['Status'] == 'WATCH').sum()))])
        sld(data, 'sc_sld', scr=scr, fault=scr.get('contrib', {}).get('bus'), mode='sc', height=600)
        fig = go.Figure()
        fig.add_bar(x=T['Bus'], y=T['Ik3 (kA)'], name="Ik3''", marker_color=NAVY)
        fig.add_bar(x=T['Bus'], y=T['Ik1 (kA)'], name="Ik1''", marker_color=BLUE)
        fig.add_scatter(x=T['Bus'], y=T['Breaking (kA)'], mode='markers', name='Breaker rating', marker=dict(symbol='line-ew-open', size=34, color=RED, line=dict(width=3)))
        fig.update_layout(template='plotly_white', barmode='group', height=360, title='Fault currents against the breaker rating', yaxis_title='kA', legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig, width=STRETCH)
        show(T); csv_button(T, 'short_circuit.csv')
        c = scr.get('contrib')
        if c:
            st.markdown(f"##### Fault at {c['bus']}: three-phase {c['Ik (kA)']} kA, current in every branch")
            st.dataframe(c['branches'], width=STRETCH, hide_index=True)
        st.caption("IEC 60909: equivalent voltage source c·Un/√3 at the fault; line capacitances, shunt reactors and loads neglected; HVDC converters give no "
                   "contribution. Breaker duty uses the higher of Ik3'' and Ik1''; making duty ip against 2.5 × breaking current.")
        st.download_button('📄 Short circuit report (PDF)', pdfrep.study_pdf(data, {'sc': scr}, 'Short Circuit Study', ss['net_name']), 'Short_Circuit_Report.pdf', key='sc_pdf')
    else:
        st.info('Choose the case and press Analyze.')

# ============================================================== 5. Protection
with tabs[5]:
    cti = st.number_input('Coordination time interval (s)', 0.1, 0.6, 0.3, 0.05, key='pr_cti')
    if st.button('🛡️ Analyze protection coordination', type='primary', key='pr_go'):
        with st.spinner('Faults at every bus, relay operating times and zone checks...'):
            try:
                ss['res'] = {**ss['res'], 'prot': ng.protection(data, cti)}; update_register()
            except Exception as e:
                st.error(f'Protection study failed: {e}')
    pr = ss['res'].get('prot')
    if pr:
        D_ = pr['distance']; P_ = pr['oc_pairs']
        kpis([('Distance relays', len(D_)), ('Distance findings', int((D_.get('Status', pd.Series(dtype=str)) != 'EXCELLENT').sum())),
              ('Grading pairs', len(P_)), ('Grading findings', int((P_.get('Status', pd.Series(dtype=str)) != 'EXCELLENT').sum()))])
        st.markdown('##### Distance protection')
        if len(D_):
            show(D_); csv_button(D_, 'distance_zones.csv')
            rel = st.selectbox('Relay for the R-X diagram', list(D_['Relay']), key='pr_rx')
            r = D_.set_index('Relay').loc[rel]
            st.plotly_chart(npl.rx_figure(r, r['Line Z (ohm)'], r['Line Angle (deg)'], f"{rel}: zones on the R-X plane (primary ohms)"), width=STRETCH)
        else:
            st.info('No distance relays (function 21) in the relay table.')
        st.markdown('##### Overcurrent and earth-fault grading')
        if len(P_):
            show(P_); csv_button(P_, 'overcurrent_grading.csv')
            P_ = P_.assign(Pair=P_['Fault at'] + ' · ' + P_['Primary'] + ' / ' + P_['Backup'])
            pair = st.selectbox('Pair for the time-current curves', list(P_['Pair']), key='pr_pair')
            pr_ = P_[P_['Pair'] == pair].iloc[0]
            RR = d['Relays'].set_index('Relay')
            rels = ng.filled(data)[0]['Relays']
            rels = rels[rels['Relay'].isin([pr_['Primary'], pr_['Backup']])]
            st.plotly_chart(npl.tcc_figure(rels, [(f"primary {pr_['Primary I (A)']:.0f} A", pr_['Primary I (A)']), (f"backup {pr_['Backup I (A)']:.0f} A", pr_['Backup I (A)'])],
                                           f"Fault at {pr_['Fault at']} ({pr_['Fault']}): margin {pr_['Margin (s)']} s"), width=STRETCH)
            ss['res']['prot']['tcc'] = (rels, [('primary', pr_['Primary I (A)']), ('backup', pr_['Backup I (A)'])])
        else:
            st.info('No overcurrent (51) or earth-fault (51N) grading pairs found.')
        st.download_button('📄 Protection report (PDF)', pdfrep.study_pdf(data, {'prot': pr}, 'Protection Coordination Study', ss['net_name']), 'Protection_Report.pdf', key='pr_pdf')
    else:
        st.info('Press Analyze.')

# ============================================================== 6. Contingency
with tabs[6]:
    c1, c2 = st.columns(2)
    sc_ = c1.number_input('Load scaling (%)', 10.0, 200.0, 100.0, 5.0, key='ct_scale') / 100
    wl = c2.number_input('Loading warning (%)', 50.0, 100.0, 80.0, 5.0, key='ct_warn')
    if st.button('🔁 Analyze N-1 contingency', type='primary', key='ct_go'):
        with st.spinner('One load flow per outage...'):
            try:
                ss['res'] = {**ss['res'], 'cont': ng.contingency(data, sc_, wl)}; update_register()
            except Exception as e:
                st.error(f'Contingency study failed: {e}')
    ct = ss['res'].get('cont')
    if ct:
        T = ct['table']
        kpis([('Outages studied', len(T)), ('With violations / separation', int((T['Status'] == 'ACTION').sum())), ('Close to limits', int((T['Status'] == 'WATCH').sum())),
              ('Base case max loading', f"{ct['base']['Max Line Loading (%)']:.0f}%")])
        fig = go.Figure(go.Bar(x=T['Outage'], y=T['Max Loading (%)'].fillna(0), marker_color=[STATUS_COLOR[s] for s in T['Status']], text=T['Most Loaded'], textposition='outside'))
        fig.add_hline(y=100, line=dict(color=RED, dash='dash')); fig.add_hline(y=wl, line=dict(color=AMBER, dash='dot'))
        fig.update_layout(template='plotly_white', height=380, title='Highest branch loading after each outage (label = most loaded branch)', yaxis_title='%')
        st.plotly_chart(fig, width=STRETCH)
        show(T); csv_button(T, 'n1_contingency.csv')
    else:
        st.info('Press Analyze.')

# ============================================================== 7. Combined report
with tabs[7]:
    st.markdown("<div class='pg-note'>One report with every analysis that has been run on the current network. "
                "Press <b>Run all analyses</b> to run them all with the default settings.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    eng = c1.text_input('Prepared by', key='rp_eng')
    ttl = c2.text_input('Report title', 'Network Study Report', key='rp_title')
    if st.button('▶️ Run all analyses', type='primary', key='rp_all'):
        with st.spinner('Load flow, voltage drop, short circuit, protection, N-1 and validation...'):
            try:
                lf = ng.load_flow(data)
                ss['res'] = {'lf': lf, 'vd': ng.voltage_drop(lf), 'sc': ng.short_circuit(data, 'max', BUSES[0] if BUSES else None), 'prot': ng.protection(data),
                             'cont': ng.contingency(data), 'val': ng.validation()}
                update_register()
            except Exception as e:
                st.error(f'An analysis failed: {e}')
    R = ss['res']
    done = [nm for k, ic, nm, _ in ANALYSES if k in R]
    st.write('Included: ' + (', '.join(done) if done else 'nothing yet'))
    if done:
        pdf = pdfrep.study_pdf(data, R, ttl, ss['net_name'], eng)
        b1, b2 = st.columns(2)
        b1.download_button('📄 Combined report (PDF)', pdf, f"{ttl.replace(' ', '_')}.pdf", 'application/pdf', type='primary', width=STRETCH)
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, 'w') as z:
            if 'lf' in R:
                z.writestr('load_flow_buses.csv', R['lf']['bus'].to_csv(index=False)); z.writestr('load_flow_circuits.csv', R['lf']['line'].to_csv(index=False))
                z.writestr('load_flow_transformers.csv', R['lf']['trafo'].to_csv(index=False))
            if 'vd' in R:
                z.writestr('voltage_drop.csv', R['vd']['line'].to_csv(index=False))
            if 'sc' in R:
                z.writestr('short_circuit.csv', R['sc']['bus'].to_csv(index=False))
            if 'prot' in R:
                z.writestr('distance_zones.csv', R['prot']['distance'].to_csv(index=False)); z.writestr('overcurrent_grading.csv', R['prot']['oc_pairs'].to_csv(index=False))
            if 'cont' in R:
                z.writestr('n1_contingency.csv', R['cont']['table'].to_csv(index=False))
            z.writestr('network.xlsx', ng.network_excel(data))
        b2.download_button('🗂️ All results (CSV + network Excel, ZIP)', zb.getvalue(), 'NetworkGuard_Results.zip', 'application/zip', width=STRETCH)
        if st.button('📌 Save this study to the history', key='rp_save'):
            ss['studies'] = ss['studies'] + [{'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Network': ss['net_name'], 'Title': ttl,
                                               **{nm: (STATUS_LABEL.get(status_of(k, R[k]), '') + ' · ' + headline(k, R[k])) for k, ic, nm, _ in ANALYSES if k in R}}]
    if ss['studies']:
        st.markdown('##### Study history')
        st.dataframe(pd.DataFrame(ss['studies']), width=STRETCH, hide_index=True)

# ============================================================== 8. Validation
with tabs[8]:
    st.markdown("<div class='pg-note'>The engine is checked against results that are published for standard test systems and against hand "
                "calculations with the standard formulas. The same equations give the same results in any program when the data is the same.</div>",
                unsafe_allow_html=True)
    if st.button('✅ Run the validation', type='primary', key='va_go'):
        with st.spinner('IEEE test systems, IEC 60909 and IEC 60255 checks...'):
            ss['res'] = {**ss['res'], 'val': ng.validation()}
    va = ss['res'].get('val')
    if va is not None:
        kpis([('Checks', len(va)), ('Pass', int((va['Result'] == 'PASS').sum())), ('Largest difference', f"{va['Difference'].abs().max():.3f}")])
        show(va, 'Result')
        st.caption('IEEE 9-bus and 14-bus: published load flow solutions (bus voltages to 3 decimals, angles to 0.01-0.02°). IEC 60909: Ik3, Ik1 and ip of a grid '
                   'plus 400 kV line computed by hand with the standard formulas. IEC 60255-151: standard inverse and very inverse curve times.')
    else:
        st.info('Press Run the validation.')

st.markdown(FOOTER, unsafe_allow_html=True)
