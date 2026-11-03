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
import networkguard_import as ni
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
            ('cont', '🔁', 'N-1 Contingency', 'Outage of every circuit and transformer')]

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


# ==========================================
# --- Two networks: the real one (built by the engineer) and the demo ---
# ==========================================
ss = st.session_state
WS = {'real': '🏭 My network', 'demo': '🧪 Demo network'}
for key, default in [('ws', 'real'), ('nets', None), ('names', {'real': 'My network', 'demo': 'Demo network'}), ('results', {'real': {}, 'demo': {}}),
                     ('register', []), ('studies', []), ('sel', None), ('sld_prop', None)]:
    if key not in ss:
        ss[key] = default
if ss['nets'] is None:
    ss['nets'] = {'real': ng.clean(ng.empty()), 'demo': ng.demo()}


def data():
    return ss['nets'][ss['ws']]


def set_data(d):
    ss['nets'] = {**ss['nets'], ss['ws']: ng.clean(d)}
    ss['results'] = {**ss['results'], ss['ws']: {}}


def res():
    return ss['results'][ss['ws']]


def set_res(**kw):
    ss['results'] = {**ss['results'], ss['ws']: {**res(), **kw}}
    if ss['ws'] == 'real':
        update_register()


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


def update_register():
    r = ss['results']['real']
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
        return f"largest drop {r['line']['|Drop| (%)'].max():.1f}% on {r['line'].iloc[0]['Line']}" if len(r['line']) else 'no circuits'
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
def verification_table():
    return ng.validation()


def sld(d, key, **kw):
    fig = npl.sld_figure(d, select=ss.get('sel'), **kw)
    ev = st.plotly_chart(fig, width=STRETCH, key=key, on_select='rerun', selection_mode='points', config={'scrollZoom': True, 'displaylogo': False})
    try:
        for p in (ev.selection.points if ev and ev.selection else []):
            cd = p.get('customdata')
            if cd and len(cd) == 2:
                if ss.get('sel') != (cd[0], cd[1]):
                    ss['sel'] = (cd[0], cd[1]); ss['ng_pending'] = (cd[0], cd[1]); st.rerun()
                break
    except Exception:
        pass


def ready():
    """True when the network can be analysed; otherwise says what is missing."""
    d = data()
    if not len(d['Buses']):
        st.info('The network is empty. Build it on the 🗺️ Single-Line Diagram page first' + (' or switch to the demo network.' if ss['ws'] == 'real' else '.'))
        return False
    errs = ng.validate_links(d)
    if errs:
        st.error('Fix these on the data pages first:  \n' + '  \n'.join(errs[:10]))
        return False
    miss = ng.missing_required(d)
    if miss:
        M = pd.DataFrame(miss, columns=['Table', 'Equipment', 'Parameter'])
        st.warning(f"{len(miss)} required value(s) are still missing, so the analysis cannot run yet. Enter them on the 📋 Equipment Data page "
                   "(form, Excel form or relay settings files):  \n" +
                   '  \n'.join(f"• {r.Table} – {r.Parameter}: {r.n} item(s)" for r in M.groupby(['Table', 'Parameter']).size().reset_index(name='n').itertuples()))
        return False
    return True


# ---------------------------------------------------------------- network selector
c1, c2 = st.columns([2, 3])
with c1:
    ws = st.segmented_control('Network', list(WS), default=ss['ws'], format_func=WS.get, key='ws_pick')
    if ws and ws != ss['ws']:
        ss['ws'] = ws; ss['sel'] = None; st.rerun()
with c2:
    d0 = data()
    st.markdown(f"<div class='pg-note' style='margin:0'>Working on <b>{ss['names'][ss['ws']]}</b>: {len(d0['Substations'])} substations, "
                f"{len(d0['Buses'])} buses, {len(d0['Lines'])} circuits, {len(d0['Transformers'])} transformers, {len(d0['Relays'])} relays"
                f"{' · simulated values, anonymised names' if ss['ws'] == 'demo' else ''}</div>", unsafe_allow_html=True)

tabs = st.tabs(['🏠 Overview', '🗺️ Single-Line Diagram', '📋 Equipment Data', '⚡ Load Flow', '📉 Voltage Drop', '💥 Short Circuit',
                '🛡️ Protection Coordination', '🔁 N-1 Contingency', '⚡ Fault Study', '📄 Combined Report'])

# ============================================================== 0. Overview
with tabs[0]:
    D, R = data(), res()
    if ss['ws'] == 'real' and not len(D['Buses']):
        st.markdown('#### Start with your network in three steps')
        c = st.columns(3)
        for col, (ic, t, x) in zip(c, [('🗺️', '1. Single-line diagram', 'Upload the drawing (PDF), build it step by step (substation, busbars, lines, '
                                                                        'transformers, reactors, breakers) or upload an Excel file.'),
                                        ('📋', '2. Equipment data', 'Fill each equipment form, upload its Excel form, or upload the relay settings '
                                                                    'and equipment documents (PDF, picture, Excel): the values are read and proposed.'),
                                        ('⚡', '3. Analyses', 'Load flow, voltage drop, short circuit, protection coordination and N-1, '
                                                             'then one combined report.')]):
            col.markdown(f"<div class='ng-card'><div class='ic'>{ic}</div><h4>{t}</h4><div class='sub' style='min-height:5em'>{x}</div></div>", unsafe_allow_html=True)
        st.caption('To see every analysis working, switch to 🧪 Demo network at the top.')
    else:
        cols = st.columns(3)
        for i, (k, ic, nm, ds) in enumerate(ANALYSES):
            r = R.get(k)
            body = (f"<div class='res'>{chip(status_of(k, r))} {headline(k, r)}</div>" if r is not None else
                    "<div class='res' style='color:#8a97ab'>Not run yet: open its tab and press Analyze.</div>")
            cols[i % 3].markdown(f"<div class='ng-card' style='margin-bottom:12px'><div class='ic'>{ic}</div><h4>{nm}</h4><div class='sub'>{ds}</div>{body}</div>",
                                 unsafe_allow_html=True)
        if ss['ws'] == 'demo' and not R:
            if st.button('▶️ Run all analyses on the demo network', type='primary', key='ov_demo_all'):
                with st.spinner('Running all analyses...'):
                    lf = ng.load_flow(D)
                    set_res(lf=lf, vd=ng.voltage_drop(lf), sc=ng.short_circuit(D, 'max', 'CJ-400'), prot=ng.protection(D), cont=ng.contingency(D))
                st.rerun()
        lf, scr = R.get('lf'), R.get('sc')
        st.markdown('#### Single-line diagram' + (' with the load flow results' if lf else ''))
        st.plotly_chart(npl.sld_figure(D, lf=lf, scr=scr, mode='lf' if lf else 'network'), width=STRETCH, key='ov_sld', config={'scrollZoom': True, 'displaylogo': False})
        F = ng.asset_rows(lf, scr, R.get('prot'), R.get('cont'))
        if len(F):
            F = F[F['Status'] != 'EXCELLENT']
            st.markdown(f'#### Findings ({len(F)})'); show(F)
        elif R:
            st.success('No finding: all equipment within limits.')

# ============================================================== 1. Single-line diagram
with tabs[1]:
    D = data()
    how = st.radio('How do you want to enter the single-line diagram?', ['📄 From a PDF drawing', '✏️ Step by step', '📊 From an Excel file'], horizontal=True, key='sld_how')
    if how.startswith('📄'):
        st.markdown("<div class='pg-note'>The drawing is read for substation names and voltages, circuit names (for example <b>41_SALWA_SILLA</b> = circuit 1 "
                    "Salwa-Silla), shunt reactors (RB/RL), power transformers (T4xx), neighbouring grids and HVDC converters. Check and correct the "
                    "proposal below, then press <b>Use this network</b>. A scanned drawing is read by OCR.</div>", unsafe_allow_html=True)
        up = st.file_uploader('Single-line diagram (PDF or picture)', type=['pdf', 'png', 'jpg', 'jpeg'], key='sld_pdf')
        if up is not None and st.button('🔎 Read the drawing', type='primary', key='sld_read'):
            raw = up.getvalue()
            if not up.name.lower().endswith('.pdf'):
                from PIL import Image
                im = Image.open(io.BytesIO(raw)).convert('RGB'); b = io.BytesIO(); im.save(b, format='PDF'); raw = b.getvalue()
            with st.spinner('Reading the drawing...'):
                try:
                    tabs_, log, png = ni.sld_from_pdf(raw)
                    ss['sld_prop'] = {'tabs': tabs_, 'log': log, 'png': png, 'name': up.name.rsplit('.', 1)[0]}
                except Exception as e:
                    st.error(f'Could not read the drawing: {e}')
        P = ss.get('sld_prop')
        if P:
            for l_ in P['log']:
                st.markdown(f'• {l_}')
            with st.expander('🖼️ The drawing'):
                st.image(P['png'], width=STRETCH)
            st.markdown('##### Proposed network (edit, add or delete rows)')
            ed = {}
            ptabs = st.tabs([f"{ng.ICON.get(k, '')} {k} ({len(v)})" for k, v in P['tabs'].items()])
            for t_, (k, v) in zip(ptabs, P['tabs'].items()):
                with t_:
                    ed[k] = st.data_editor(v, key=f'prop_{k}', num_rows='dynamic', hide_index=True, width=STRETCH)
            prev = ni.to_network(ed)
            st.markdown('##### Preview')
            st.plotly_chart(npl.sld_figure(prev, height=560), width=STRETCH, key='prop_sld')
            c1, c2 = st.columns(2)
            if c1.button('✅ Use this network', type='primary', width=STRETCH, key='prop_use'):
                set_data(prev); ss['names'][ss['ws']] = P['name']; ss['sld_prop'] = None; st.success('Network created. Now add the equipment data.'); st.rerun()
            if c2.button('✖️ Discard', width=STRETCH, key='prop_drop'):
                ss['sld_prop'] = None; st.rerun()
    elif how.startswith('✏️'):
        left, right = st.columns([1, 1.5])
        B_ = list(D['Buses']['Bus'])
        with left:
            step = st.selectbox('Add', ['🏢 Substation with its busbars', '🗼 Line or cable', '🔁 Transformer', '🌀 Shunt reactor', '🌐 Neighbouring grid',
                                        '🔀 HVDC converter / power injection', '🔲 Circuit breaker'], key='b_step')
            with st.form('b_form', clear_on_submit=False):
                if step.startswith('🏢'):
                    nm = st.text_input('Substation name')
                    kvs = st.text_input('Busbar voltages (kV), separated by commas', '400')
                    c_ = st.columns(3)
                    x = c_[0].number_input('Position X', value=float(ng.next_position(D)[0]), step=0.5)
                    y = c_[1].number_input('Position Y', value=float(ng.next_position(D)[1]), step=0.5)
                    brk = c_[2].number_input('Breaker rating (kA)', value=40.0, step=1.0)
                    if st.form_submit_button('➕ Add substation', type='primary') and nm:
                        try:
                            kv = [float(v) for v in kvs.replace(';', ',').split(',') if v.strip()]
                            set_data(ng.add_substation(D, nm.strip(), kv, x, y, brk)); st.rerun()
                        except ValueError:
                            st.error('Write the voltages as numbers, e.g. 400, 220')
                elif step.startswith('🗼'):
                    if len(B_) < 2:
                        st.info('Add two substations first.')
                    a = st.selectbox('From busbar', B_); b = st.selectbox('To busbar', B_, index=min(1, max(len(B_) - 1, 0)))
                    c_ = st.columns(3)
                    n = c_[0].number_input('Circuits', 1, 6, 2); L_ = c_[1].number_input('Length (km)', 0.0, 2000.0, 0.0); typ = c_[2].selectbox('Type', ['OHL', 'Cable'])
                    cond = st.selectbox('Conductor (overhead line)', [''] + list(ng.CONDUCTORS['Conductor']))
                    tw = st.selectbox('Tower (overhead line)', [''] + list(ng.TOWERS['Tower']))
                    rt = st.number_input('Rating (A)', 100.0, 6000.0, 2000.0, 100.0)
                    bk = st.checkbox('Add circuit breakers at both ends', True)
                    if st.form_submit_button('➕ Add line', type='primary') and a and b and a != b:
                        set_data(ng.add_line(D, a, b, n, L_ or None, typ, cond, tw, rt, breakers=bk)); st.rerun()
                elif step.startswith('🔁'):
                    a = st.selectbox('HV busbar', B_); b = st.selectbox('LV busbar', B_, index=min(1, max(len(B_) - 1, 0)))
                    c_ = st.columns(3)
                    n = c_[0].number_input('Number of units', 1, 6, 1); mva = c_[1].number_input('MVA', 0.0, 3000.0, 0.0); uk = c_[2].number_input('uk %', 0.0, 30.0, 0.0)
                    bk = st.checkbox('Add circuit breakers (HV and LV)', True)
                    if st.form_submit_button('➕ Add transformer', type='primary') and a and b and a != b:
                        set_data(ng.add_transformer(D, a, b, n, mva or None, uk or None, breakers=bk)); st.rerun()
                elif step.startswith('🌀'):
                    a = st.selectbox('Busbar', B_)
                    ln_ = st.selectbox('Line (for a line reactor)', [''] + list(D['Lines']['Line']))
                    c_ = st.columns(2)
                    n = c_[0].number_input('Number of reactors', 1, 6, 1); q = c_[1].number_input('MVAr each', 0.0, 1000.0, 0.0)
                    if st.form_submit_button('➕ Add reactor', type='primary') and a:
                        set_data(ng.add_reactor(D, a, q or None, ln_, n)); st.rerun()
                elif step.startswith('🌐'):
                    nm = st.text_input('Grid name'); a = st.selectbox('Busbar', B_)
                    c_ = st.columns(3)
                    skx = c_[0].number_input('Sk max (MVA)', 0.0, 200000.0, 0.0); skn = c_[1].number_input('Sk min (MVA)', 0.0, 200000.0, 0.0)
                    pim = c_[2].number_input('Import (MW)', -5000.0, 5000.0, 0.0)
                    sl = st.checkbox('Reference (slack) grid of the load flow', not len(D['Grids']))
                    if st.form_submit_button('➕ Add grid', type='primary') and nm and a:
                        set_data(ng.add_grid(D, nm, a, skx or None, skn or None, sl, pim)); st.rerun()
                elif step.startswith('🔀'):
                    nm = st.text_input('Name'); a = st.selectbox('Busbar', B_)
                    c_ = st.columns(2)
                    p_ = c_[0].number_input('Power into the network (MW)', -5000.0, 5000.0, 0.0); rt = c_[1].number_input('Rated MW', 0.0, 5000.0, 600.0)
                    if st.form_submit_button('➕ Add', type='primary') and nm and a:
                        set_data(ng.add_converter(D, nm, a, p_, rt)); st.rerun()
                else:
                    nm = st.text_input('Breaker name'); a = st.selectbox('Busbar', B_)
                    el = st.selectbox('Switches', [''] + list(D['Lines']['Line']) + list(D['Transformers']['Transformer']) + list(D['Reactors']['Reactor']))
                    c_ = st.columns(2)
                    kab = c_[0].number_input('Breaking (kA)', 0.0, 100.0, 40.0); ra = c_[1].number_input('Rated (A)', 0.0, 8000.0, 2000.0)
                    if st.form_submit_button('➕ Add breaker', type='primary') and nm and a:
                        set_data(ng.add_breaker(D, nm, a, el, kab, ra)); st.rerun()
            st.caption('Each item appears on the diagram at once. Edit or delete any item on the 📋 Equipment Data page.')
        with right:
            sld(data(), 'b_sld', height=620)
    else:
        st.markdown("<div class='pg-note'>One Excel file with a sheet per equipment type. Download the current network (or the empty template), "
                    "fill it and upload it.</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.download_button('⬇️ Current network (Excel)', ng.network_excel(D), f"{ss['names'][ss['ws']].replace(' ', '_')}.xlsx", width=STRETCH)
        c2.download_button('⬇️ Empty template (Excel)', ng.network_excel(ng.clean(ng.empty())), 'NetworkGuard_template.xlsx', width=STRETCH)
        up = st.file_uploader('Network Excel', type=['xlsx', 'xls'], key='ng_up_net')
        if up is not None and st.button('📥 Load this network', type='primary', key='ng_load_net'):
            try:
                set_data(ng.read_network_excel(up)); ss['names'][ss['ws']] = up.name.rsplit('.', 1)[0]; st.rerun()
            except Exception as e:
                st.error(f'Could not read the file: {e}')
        st.plotly_chart(npl.sld_figure(D, height=560), width=STRETCH, key='xl_sld')
    with st.expander('⚠️ Clear this network'):
        if st.button('Delete all equipment of this network', key='clear_net'):
            set_data(ng.demo() if ss['ws'] == 'demo' else ng.clean(ng.empty())); st.rerun()

# ============================================================== 2. Equipment data
with tabs[2]:
    if ss.get('ng_pending'):
        ss['ed_kind'], ss['ed_id'] = ss.pop('ng_pending')
    D = data()
    errs, miss = ng.validate_links(D), ng.missing_required(D)
    _, notes = ng.filled(D)
    a, b, c = st.columns(3)
    a.metric('Wrong references', len(errs)); b.metric('Missing required values', len(miss)); c.metric('Typical / computed values used', len(notes))
    if errs:
        st.error('  \n'.join(errs[:12]))
    if miss:
        with st.expander(f'📝 Missing required values ({len(miss)}): what to enter next', expanded=ss['ws'] == 'real'):
            M = pd.DataFrame(miss, columns=['Table', 'Equipment', 'Parameter'])
            st.dataframe(M.groupby(['Table', 'Parameter']).agg(Equipment=('Equipment', lambda s: ', '.join(s[:8]) + (' ...' if len(s) > 8 else '')),
                                                               Count=('Equipment', 'size')).reset_index(), hide_index=True, width=STRETCH)
    src = st.radio('Enter the data', ['✍️ Form per equipment', '📑 From relay settings files', '📄 From transformer / reactor documents'], horizontal=True, key='ed_src')
    if src.startswith('✍️'):
        st.markdown("<div class='pg-note'>Press an equipment on the diagram, or choose it on the right. Write the values, or download its Excel form, "
                    "fill it and upload it.</div>", unsafe_allow_html=True)
        left, right = st.columns([1.4, 1])
        with left:
            sld(D, 'ed_sld', height=620)
        with right:
            kind = st.selectbox('Equipment type', ng.ORDER, format_func=lambda k: f'{ng.ICON[k]} {k}', key='ed_kind')
            opts = list(D[kind][ng.id_col(kind)])
            new = st.toggle('Add a new equipment', key='ed_new')
            eid = None
            if new:
                eid = st.text_input(f'New {ng.TYPE_OF[kind].lower()} name', key='ed_new_id')
            elif opts:
                if ss.get('ed_id') not in opts:
                    ss['ed_id'] = opts[0]
                eid = st.selectbox('Equipment', opts, key='ed_id')
                ss['sel'] = (kind, eid)
            else:
                st.info(f'No {kind.lower()} yet. Use "Add a new equipment" or build the diagram first.')
            if eid:
                T = ng.element_table(D, kind, eid)
                T['Value'] = T['Value'].astype(str).replace({'nan': ''})
                st.markdown(f"**{ng.ICON[kind]} {eid}**")
                E = st.data_editor(T, key=f'ed_tab_{ss["ws"]}_{kind}_{eid}', hide_index=True, width=STRETCH, height=min(560, 38 + 35 * len(T)),
                                   disabled=['Parameter', 'Unit', 'Description', 'Source'],
                                   column_config={'Description': st.column_config.TextColumn(width='large'), 'Value': st.column_config.TextColumn(width='small')})
                b1, b2 = st.columns(2)
                if b1.button('💾 Save', type='primary', width=STRETCH, key='ed_save'):
                    set_data(ng.set_element(D, kind, eid, E)); st.rerun()
                if not new and b2.button('🗑️ Delete', width=STRETCH, key='ed_del'):
                    set_data(ng.delete_element(D, kind, eid)); ss['sel'] = None; st.rerun()
                st.download_button(f'⬇️ Excel form of {eid}', ng.element_excel(D, kind, eid), f'{kind}_{eid}.xlsx'.replace('/', '-'), width=STRETCH, key='ed_xl')
                ef = st.file_uploader(f'⬆️ Upload the filled form of {eid}', type=['xlsx', 'xls'], key=f'ed_up_{kind}_{eid}')
                if ef is not None:
                    try:
                        k2, id2, t2 = ng.read_element_excel(ef)
                        if st.button('Apply the uploaded values', key='ed_apply'):
                            set_data(ng.set_element(D, k2 if k2 in ng.SCHEMA else kind, id2 or eid, t2)); st.rerun()
                    except Exception as e:
                        st.error(f'Could not read the form: {e}')
        with st.expander(f'All {kind.lower()} ({len(D[kind])})'):
            st.dataframe(D[kind], width=STRETCH, hide_index=True)
    elif src.startswith('📑'):
        st.markdown("<div class='pg-note'>Upload the settings files of the line relays (distance, line differential) of any number of relays: "
                    "PDF exports, scanned PDF, pictures, Excel, CSV or TXT. Each file is read for the line length, impedance, angle and kZN, the CT and VT "
                    "ratios and the zone reaches and times. Choose the line and the relay end, check the proposal and apply it.</div>", unsafe_allow_html=True)
        if not len(D['Lines']):
            st.info('Build the single-line diagram first: the settings are applied to its lines.')
        files = st.file_uploader('Relay settings files', type=['pdf', 'png', 'jpg', 'jpeg', 'xlsx', 'xls', 'csv', 'txt'], accept_multiple_files=True, key='set_files')
        for k, f in enumerate(files or []):
            with st.container(border=True):
                raw = f.getvalue()
                key = f'set_{k}_{f.name}'
                if key not in ss:
                    with st.spinner(f'Reading {f.name}...'):
                        txt, how_ = ni.extract_text(raw, f.name)
                        ss[key] = (ni.parse_settings(txt), how_)
                sts, how_ = ss[key]
                st.markdown(f"**{f.name}** · read as {how_} · relay {sts.get('device') or '?'} · {sts.get('description') or ''}")
                found = {ni.SET_LABEL.get(a_, a_): v for a_, v in sts.items() if a_ in ni.SET_LABEL and a_ not in ('device', 'description')}
                if found:
                    st.dataframe(pd.DataFrame([found]), hide_index=True, width=STRETCH)
                if not len(D['Lines']):
                    continue
                lines = list(D['Lines']['Line'])
                g = ni.guess_line(sts, D)
                c1, c2 = st.columns(2)
                ln = c1.selectbox('Line', lines, index=lines.index(g) if g in lines else 0, key=f'{key}_ln')
                row = D['Lines'][D['Lines']['Line'] == ln].iloc[0]
                ends = [row['From Bus'], row['To Bus']]
                gb = ni.guess_bus(sts, ends)
                bus = c2.selectbox('Relay at the busbar', ends, index=ends.index(gb), key=f'{key}_bus')
                P, nts = ni.settings_proposal(sts, D, ln, bus)
                for n_ in nts:
                    st.caption(n_)
                if len(P):
                    E = st.data_editor(P, key=f'{key}_prop', hide_index=True, width=STRETCH, disabled=['Table', 'Equipment', 'Parameter', 'Current', 'From'])
                    if st.button(f'✅ Apply to {ln}', type='primary', key=f'{key}_apply'):
                        set_data(ni.apply_proposal(D, E)); st.success('Applied.'); st.rerun()
                else:
                    st.warning('No line or relay data found in this file.')
    else:
        st.markdown("<div class='pg-note'>Upload a transformer or reactor document (name plate, test report, protection settings): the rating, "
                    "voltages, impedance and vector group are searched in the text. Check the proposal and apply it.</div>", unsafe_allow_html=True)
        kind = st.radio('Equipment', ['Transformers', 'Reactors'], horizontal=True, key='doc_kind')
        opts = list(D[kind][ng.id_col(kind)])
        if not opts:
            st.info(f'No {kind.lower()} in the network yet.')
        else:
            eid = st.selectbox(ng.TYPE_OF[kind], opts, key='doc_id')
            f = st.file_uploader('Document', type=['pdf', 'png', 'jpg', 'jpeg', 'xlsx', 'xls', 'csv', 'txt'], key='doc_file')
            if f is not None:
                txt, how_ = ni.extract_text(f.getvalue(), f.name)
                vals = ni.equipment_values(txt)
                st.caption(f'Read as {how_}. Found: ' + (', '.join(f'{k} {v}' for k, v in vals.items()) or 'nothing recognised'))
                P = ni.equipment_proposal(vals, D, kind, eid)
                if len(P):
                    E = st.data_editor(P, key='doc_prop', hide_index=True, width=STRETCH, disabled=['Table', 'Equipment', 'Parameter', 'Current', 'From'])
                    if st.button(f'✅ Apply to {eid}', type='primary', key='doc_apply'):
                        set_data(ni.apply_proposal(D, E)); st.rerun()
    with st.expander('📐 Line parameters from conductor and tower (libraries)'):
        cc1, cc2, cc3 = st.columns(3)
        cn = cc1.selectbox('Conductor', list(ng.CONDUCTORS['Conductor']), key='lp_c')
        tw = cc2.selectbox('Tower', list(ng.TOWERS['Tower']), key='lp_t')
        tt = cc3.number_input('Conductor temperature (°C)', 20.0, 120.0, 75.0, key='lp_temp')
        p = ng.line_parameters(cn, tw, tt)
        st.dataframe(pd.DataFrame([{'R1 ohm/km': round(p['r1'], 5), 'X1 ohm/km': round(p['x1'], 4), 'C1 nF/km': round(p['c1'], 2),
                                    'R0 ohm/km': round(p['r0'], 4), 'X0 ohm/km': round(p['x0'], 4), 'C0 nF/km': round(p['c0'], 2),
                                    'GMD m': round(p['gmd'], 2), 'Bundle GMR m': round(p['gmr_b'], 4)}]), hide_index=True, width=STRETCH)
        st.caption('X1 = ω·2·10⁻⁴·ln(GMD/GMR) Ω/km; C1 = 55.63/ln(GMD/r_eq) nF/km; zero sequence by Carson, earth wires neglected.')
    if notes:
        with st.expander(f'Values filled with typical or computed values ({len(notes)})'):
            st.dataframe(pd.DataFrame(notes, columns=['Table', 'Equipment', 'Parameter', 'Value used']), hide_index=True, width=STRETCH)

D = data()
LINES = list(D['Lines']['Line']); TRAFOS = list(D['Transformers']['Transformer']); REACT = list(D['Reactors']['Reactor']); BUSES = list(D['Buses']['Bus'])


def lf_inputs(prefix):
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    scale = c1.number_input('Load scaling (%)', 10.0, 200.0, 100.0, 5.0, key=f'{prefix}_scale') / 100
    outs = c2.multiselect('Equipment out of service', LINES + TRAFOS, key=f'{prefix}_out')
    roff = c3.multiselect('Reactors switched out', REACT, key=f'{prefix}_roff')
    warn = c4.number_input('Loading warning (%)', 50.0, 100.0, 80.0, 5.0, key=f'{prefix}_warn')
    return scale, tuple(outs), tuple(roff), warn


def demo_note():
    if ss['ws'] == 'demo':
        st.caption('🧪 Demo network: simulated values and anonymised names.')


# ============================================================== 3. Load flow
with tabs[3]:
    demo_note()
    scale, outs, roff, warn = lf_inputs('lf')
    if st.button('⚡ Analyze load flow', type='primary', key='lf_go') and ready():
        with st.spinner('Newton-Raphson load flow...'):
            try:
                set_res(lf=ng.load_flow(D, scale, outs, roff, warn))
            except Exception as e:
                st.error(f'The load flow did not converge: {e}')
    lf = res().get('lf')
    if lf:
        sm = lf['summary']
        st.markdown(f"{chip(status_of('lf', lf))} &nbsp; {'Converged' if lf['converged'] else 'Not converged'} · {sm['Violations']} violation(s), {sm['Warnings']} warning(s)",
                    unsafe_allow_html=True)
        kpis([('Losses', f"{sm['Losses (MW)']} MW"), ('Losses %', f"{sm['Losses (%)']}%"), ('Min V', f"{sm['Min V (pu)']:.3f} pu"), ('Max V', f"{sm['Max V (pu)']:.3f} pu"),
              ('Max line loading', f"{sm['Max Line Loading (%)']:.0f}%"), ('Line charging', f"{sm['Line Charging (MVAr)']:.0f} MVAr"), ('Reactors', f"{sm['Reactors (MVAr)']:.0f} MVAr")])
        sld(D, 'lf_sld', lf=lf, mode='lf', height=620)
        st.markdown('##### Bus voltages'); show(lf['bus']); csv_button(lf['bus'], 'load_flow_buses.csv')
        st.markdown('##### Circuits'); show(lf['line'].drop(columns=['In Service'])); csv_button(lf['line'], 'load_flow_circuits.csv')
        if len(lf['trafo']):
            st.markdown('##### Transformers'); show(lf['trafo'].drop(columns=['In Service'])); csv_button(lf['trafo'], 'load_flow_transformers.csv')
        st.markdown('##### Grids, converters and reactors'); st.dataframe(lf['sources'], width=STRETCH, hide_index=True)
        st.download_button('📄 Load flow report (PDF)', pdfrep.study_pdf(D, {'lf': lf}, 'Load Flow Study', ss['names'][ss['ws']]), 'Load_Flow_Report.pdf', key='lf_pdf')
    else:
        st.info('Set the case and press Analyze.')

# ============================================================== 4. Voltage drop
with tabs[4]:
    demo_note()
    scale, outs, roff, warn = lf_inputs('vd')
    lim = st.number_input('Voltage drop limit along a circuit (%)', 0.5, 15.0, 3.0, 0.5, key='vd_lim')
    if st.button('📉 Analyze voltage drop', type='primary', key='vd_go') and ready():
        with st.spinner('Load flow and voltage drop...'):
            try:
                lf = ng.load_flow(D, scale, outs, roff, warn); set_res(lf=lf, vd=ng.voltage_drop(lf, lim))
            except Exception as e:
                st.error(f'The load flow did not converge: {e}')
    vd = res().get('vd')
    if vd:
        for n in vd['notes']:
            st.markdown(f"<div class='pg-note'>{n}</div>", unsafe_allow_html=True)
        B = vd['bus']
        fig = go.Figure()
        fig.add_bar(x=B['Bus'], y=B['V (pu)'], marker_color=[STATUS_COLOR.get(s, BLUE) for s in B['Status']], text=[f'{v:.3f}' for v in B['V (pu)']], textposition='outside')
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

# ============================================================== 5. Short circuit
with tabs[5]:
    demo_note()
    c1, c2 = st.columns(2)
    case = c1.radio('Case (IEC 60909)', ['max', 'min'], horizontal=True, format_func=lambda x: 'Maximum (c = 1.1, breaker duty)' if x == 'max' else 'Minimum (c = 1.0, protection sensitivity)', key='sc_case')
    fb = c2.selectbox('Fault location for the branch currents', BUSES, index=BUSES.index('CJ-400') if 'CJ-400' in BUSES else 0, key='sc_bus') if BUSES else None
    if st.button('💥 Analyze short circuit', type='primary', key='sc_go') and ready():
        with st.spinner('IEC 60909 three-phase and single-phase faults at every bus...'):
            try:
                set_res(sc=ng.short_circuit(D, case, fb))
            except Exception as e:
                st.error(f'Short circuit calculation failed: {e}')
    scr = res().get('sc')
    if scr and fb and (scr.get('contrib') or {}).get('bus') != fb:
        # the fault location changed: move the fault and recompute the branch currents (no need to analyze again)
        try:
            scr = {**scr, 'contrib': ng.fault_contributions(D, fb, scr.get('case', 'max'))}
            ss['results'] = {**ss['results'], ss['ws']: {**res(), 'sc': scr}}
        except Exception as e:
            st.warning(f'Could not compute the fault at {fb}: {e}')
    if scr:
        T = scr['bus']
        w = T.sort_values('Breaker Duty (%)', ascending=False).iloc[0]
        kpis([('Highest fault level', f"{max(T['Ik3 (kA)'].max(), T['Ik1 (kA)'].max()):.1f} kA"), ('Highest breaker duty', f"{w['Breaker Duty (%)']:.0f}% at {w['Bus']}"),
              ('Buses above rating', int((T['Status'] == 'ACTION').sum())), ('Buses at 90-100 %', int((T['Status'] == 'WATCH').sum()))])
        sld(D, 'sc_sld', scr=scr, fault=scr.get('contrib', {}).get('bus'), mode='sc', height=600)
        fig = go.Figure()
        fig.add_bar(x=T['Bus'], y=T['Ik3 (kA)'], name="Ik3''", marker_color=NAVY)
        fig.add_bar(x=T['Bus'], y=T['Ik1 (kA)'], name="Ik1''", marker_color=BLUE)
        fig.add_scatter(x=T['Bus'], y=T['Breaking (kA)'], mode='markers', name='Breaker rating', marker=dict(symbol='line-ew-open', size=34, color=RED, line=dict(width=3)))
        fig.update_layout(template='plotly_white', barmode='group', height=360, title='Fault currents against the breaker rating', yaxis_title='kA', legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig, width=STRETCH)
        st.markdown('##### Busbars'); show(T); csv_button(T, 'short_circuit.csv')
        if scr.get('breakers') is not None and len(scr['breakers']):
            st.markdown('##### Circuit breakers'); show(scr['breakers']); csv_button(scr['breakers'], 'breaker_duty.csv')
        c = scr.get('contrib')
        if c:
            st.markdown(f"##### Fault at {c['bus']}: three-phase {c['Ik (kA)']} kA, current in every branch")
            st.dataframe(c['branches'], width=STRETCH, hide_index=True)
        st.caption("IEC 60909: equivalent voltage source c·Un/√3 at the fault; line capacitances, shunt reactors and loads neglected; HVDC converters give no "
                   "contribution. Breaker duty uses the higher of Ik3'' and Ik1''; making duty ip against 2.5 × breaking current.")
        st.download_button('📄 Short circuit report (PDF)', pdfrep.study_pdf(D, {'sc': scr}, 'Short Circuit Study', ss['names'][ss['ws']]), 'Short_Circuit_Report.pdf', key='sc_pdf')
    else:
        st.info('Choose the case and press Analyze.')

# ============================================================== 6. Protection
with tabs[6]:
    demo_note()
    cti = st.number_input('Coordination time interval (s)', 0.1, 0.6, 0.3, 0.05, key='pr_cti')
    if st.button('🛡️ Analyze protection coordination', type='primary', key='pr_go') and ready():
        with st.spinner('Faults at every bus, relay operating times and zone checks...'):
            try:
                set_res(prot=ng.protection(D, cti))
            except Exception as e:
                st.error(f'Protection study failed: {e}')
    pr = res().get('prot')
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
            rels = ng.filled(D)[0]['Relays']
            rels = rels[rels['Relay'].isin([pr_['Primary'], pr_['Backup']])]
            st.plotly_chart(npl.tcc_figure(rels, [(f"primary {pr_['Primary I (A)']:.0f} A", pr_['Primary I (A)']), (f"backup {pr_['Backup I (A)']:.0f} A", pr_['Backup I (A)'])],
                                           f"Fault at {pr_['Fault at']} ({pr_['Fault']}): margin {pr_['Margin (s)']} s"), width=STRETCH)
            pr['tcc'] = (rels, [('primary', pr_['Primary I (A)']), ('backup', pr_['Backup I (A)'])])
        else:
            st.info('No overcurrent (51) or earth-fault (51N) grading pairs found.')
        st.download_button('📄 Protection report (PDF)', pdfrep.study_pdf(D, {'prot': pr}, 'Protection Coordination Study', ss['names'][ss['ws']]), 'Protection_Report.pdf', key='pr_pdf')
    else:
        st.info('Press Analyze.')

# ============================================================== 7. Contingency
with tabs[7]:
    demo_note()
    c1, c2 = st.columns(2)
    sc_ = c1.number_input('Load scaling (%)', 10.0, 200.0, 100.0, 5.0, key='ct_scale') / 100
    wl = c2.number_input('Loading warning (%)', 50.0, 100.0, 80.0, 5.0, key='ct_warn')
    if st.button('🔁 Analyze N-1 contingency', type='primary', key='ct_go') and ready():
        with st.spinner('One load flow per outage...'):
            try:
                set_res(cont=ng.contingency(D, sc_, wl))
            except Exception as e:
                st.error(f'Contingency study failed: {e}')
    ct = res().get('cont')
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

# ============================================================== 8. Fault study
@st.cache_resource(show_spinner=False)
def fg_engine():
    """FaultGuard engine and its fault-type model (reads disturbance records and event lists)."""
    import faultguard_core as fg
    return fg, fg.load_model(pg_paths.model('faultguard_model.pkl'))


def faultguard_incidents():
    """Incidents analysed in FaultGuard on the PowerGuard AI platform (same session)."""
    try:
        reg = ss.get('_pg_store', {}).get('faultguard', {}).get('register') or []
        return pd.DataFrame(reg)
    except Exception:
        return pd.DataFrame()


def _num(v):
    try:
        import re as _re
        m = _re.search(r'-?\d+(?:\.\d+)?', str(v))
        return float(m.group(0)) if m else None
    except Exception:
        return None


def end_of(station, line_row):
    """'from' or 'to': the line end whose substation name matches a recorder's station name."""
    sub = dict(zip(D['Buses']['Bus'], D['Buses']['Substation']))
    st_ = ''.join(ch for ch in str(station).upper() if ch.isalpha())
    for side, b in (('from', line_row['From Bus']), ('to', line_row['To Bus'])):
        nm = ''.join(ch for ch in str(sub.get(b, b)).upper() if ch.isalpha()).replace('AL', '', 1) if str(sub.get(b, b)).upper().startswith('AL ') else \
            ''.join(ch for ch in str(sub.get(b, b)).upper() if ch.isalpha())
        if st_ and nm and (st_[:3] in nm or nm[:3] in st_):
            return side
    return None


def read_record_files(files, fg):
    import zipfile as _z
    by = {}
    for f in files:
        if f.name.lower().endswith('.zip'):
            with _z.ZipFile(io.BytesIO(f.getvalue())) as z:
                for n in z.namelist():
                    if not n.endswith('/'):
                        by[n.rsplit('/', 1)[-1]] = z.read(n)
        else:
            by[f.name] = f.getvalue()
    low = {n.lower(): n for n in by}
    out = []
    for n in by:
        if n.lower().endswith('.cfg'):
            dn = low.get(n[:-4].lower() + '.dat')
            if dn:
                out.append((n, fg.read_comtrade(by[n].decode('latin-1'), by[dn])))
    return out


with tabs[8]:
    demo_note()
    st.markdown("<div class='pg-note'>Study a real fault on a line: take it from <b>FaultGuard</b>, read it from the <b>disturbance records and event "
                "lists</b> of the relays, or <b>enter it by hand</b>. The fault is placed at the same point of the network, and the calculated currents "
                "from both ends, the voltage dips and the expected relay operation are compared with what was measured.</div>", unsafe_allow_html=True)
    if not LINES:
        st.info('Build the network first: the fault is placed on one of its lines.')
    else:
        src = st.radio('Fault from', ['🔍 FaultGuard', '📂 Disturbance records and events', '✍️ Enter by hand'], horizontal=True, key='fs_src')
        pre = ss.get('fs_pre', {})
        if src.startswith('🔍'):
            I = faultguard_incidents()
            if I.empty:
                st.caption('No FaultGuard incidents in this session. On the PowerGuard AI platform the incidents analysed in FaultGuard appear here; '
                           'here you can upload the FaultGuard incidents CSV.')
                up = st.file_uploader('FaultGuard incidents (CSV)', type=['csv'], key='fs_fgcsv')
                if up is not None:
                    I = pd.read_csv(up)
            if not I.empty:
                I = I.reset_index(drop=True)
                lab = [f"{r.get('Incident', i)} · {r.get('Date', '')} · {r.get('Line', '')} · {r.get('Fault Type', '')}" for i, r in I.iterrows()]
                k = st.selectbox('Incident', range(len(I)), format_func=lambda i: lab[i], key='fs_inc')
                r = I.iloc[k]
                ln, sc_ = ng.match_line(r.get('Line', ''), D)
                loc = _num(r.get('Location (km)')) if 'Location (km)' in r else _num(r.get('Location'))
                if st.button('Use this incident', key='fs_use_inc'):
                    ss['fs_pre'] = {'line': ln, 'd_km': loc, 'type': str(r.get('Fault Type', '') or 'AG').upper(),
                                    'I_from': _num(r.get('Fault Current (kA)')), 'I_to': None, 'note': f"From FaultGuard incident {r.get('Incident', '')}: "
                                    f"line '{r.get('Line', '')}' matched to {ln}; the fault current is the recording relay's current, set at the From end - move it if the "
                                    "record was taken at the other end."}
                    st.rerun()
        elif src.startswith('📂'):
            c1, c2 = st.columns(2)
            recs_up = c1.file_uploader('Disturbance records (COMTRADE .cfg + .dat or .zip), any number of relays', type=['cfg', 'dat', 'zip'],
                                       accept_multiple_files=True, key='fs_rec')
            ev_up = c2.file_uploader('Event lists (PDF, picture, CSV, TXT, Excel), optional', type=['pdf', 'png', 'jpg', 'jpeg', 'csv', 'txt', 'xlsx', 'xls'],
                                     accept_multiple_files=True, key='fs_ev')
            if recs_up:
                fg, model = fg_engine()
                recs = read_record_files(recs_up, fg)
                if not recs:
                    st.warning('No complete record: upload each .cfg with its .dat.')
                else:
                    names = ' '.join(f"{rc.get('station', '')} {rc.get('device', '')} {n}" for n, rc in recs)
                    g, _ = ng.match_line(names, D)
                    ln = st.selectbox('Line of the fault', LINES, index=LINES.index(g) if g in LINES else 0, key='fs_rec_line')
                    if st.button('🔎 Read the records', type='primary', key='fs_read'):
                        with st.spinner('Reading the records with the FaultGuard engine...'):
                            Lrow = ng.filled(D)[0]['Lines'].set_index('Line').loc[ln]
                            kv = float(D['Buses'].set_index('Bus').at[Lrow['From Bus'], 'kV'])
                            line_d = {'length_km': float(Lrow['Length km']), 'r1': float(Lrow['R1 ohm/km']), 'x1': float(Lrow['X1 ohm/km']),
                                      'r0': float(Lrow['R0 ohm/km']), 'x0': float(Lrow['X0 ohm/km']), 'kv': kv}
                            events = []
                            for f in ev_up or []:
                                try:
                                    e = fg.parse_events(f.getvalue(), f.name)
                                    if len(e):
                                        events.append((f.name, e))
                                except Exception:
                                    pass
                            try:
                                a, rec = fg.analyze_incident(recs, events or None, None, line_d, None, model=model)
                                ends = {}
                                for n, rc in recs:
                                    try:
                                        ai = fg.analyze(rc, None, line_d, None, model=model)
                                        side = end_of(rc.get('station', '') or n, Lrow.to_dict())
                                        if side and ai.get('I_fault_max_kA'):
                                            ends[side] = (ai['I_fault_max_kA'], ai.get('loc_km'), rc.get('station', '') or n)
                                    except Exception:
                                        continue
                                main_side = end_of(rec.get('station', ''), Lrow.to_dict()) or 'from'
                                if not ends and a.get('I_fault_max_kA'):     # recorder station not recognised: its current goes to the From end
                                    ends[main_side] = (a['I_fault_max_kA'], a.get('loc_km'), rec.get('station', '') or 'record')
                                loc = a.get('loc_km')
                                d_from = (Lrow['Length km'] - loc) if (loc is not None and main_side == 'to') else loc
                                ss['fs_pre'] = {'line': ln, 'd_km': d_from, 'type': str(a.get('type', '') or 'AG').upper(),
                                                'I_from': ends.get('from', (None,))[0], 'I_to': ends.get('to', (None,))[0],
                                                'note': f"Records: {', '.join(n for n, _ in recs)}. FaultGuard: {a.get('type')} fault, "
                                                        f"{a.get('I_fault_max_kA', 0):.2f} kA, {('%.1f km from ' % loc + str(rec.get('station') or 'the recording end')) if loc is not None else 'not located'}."
                                                        + ''.join(f" {s.title()} end ({v[2]}): {v[0]:.2f} kA." for s, v in ends.items())}
                                st.rerun()
                            except Exception as e:
                                st.error(f'Could not analyse the records: {e}')
        pre = ss.get('fs_pre', {})
        if pre.get('note'):
            st.info(pre['note'])
        st.markdown('##### Fault applied to the network')
        FT = list(ng.PHASE_FAULTS)
        ft0 = str(pre.get('type', 'AG')).upper().replace('-', '').replace(' ', '').replace('PHASE', '')
        ft0 = ft0 if ft0 in FT else {'1PH': 'AG', '2PH': 'BC', '3PH': 'ABC'}.get(ft0, 'AG')
        c1, c2, c3, c4 = st.columns([2, 1, 1.4, 1])
        ln = c1.selectbox('Line', LINES, index=LINES.index(pre['line']) if pre.get('line') in LINES else 0, key=f"fs_line_{pre.get('line')}")
        Lr = ng.filled(D)[0]['Lines'].set_index('Line').loc[ln]
        Lkm = float(Lr['Length km']) if pd.notna(Lr['Length km']) else 100.0
        dk = c2.number_input(f"Distance from {Lr['From Bus']} (km)", 0.0, Lkm, float(min(max(pre.get('d_km') or Lkm / 2, 0.0), Lkm)), 0.5, key=f"fs_d_{pre.get('d_km')}_{ln}")
        typ = c3.selectbox('Fault type (faulted phases)', FT, index=FT.index(ft0),
                           format_func=lambda k: f"{k} · {ng.FAULT_NAME[ng.PHASE_FAULTS[k][0]]}", key=f"fs_t_{ft0}")
        rf = c4.number_input('Fault resistance (Ω)', 0.0, 300.0, 0.0, 1.0, key='fs_rf')
        if st.button('⚡ Apply the fault to the network', type='primary', key='fs_go') and ready():
            with st.spinner('Short circuit on the whole network, relay operation and tripping sequence...'):
                try:
                    set_res(fs=ng.fault_analysis(D, ln, dk, typ, rf))
                except Exception as e:
                    st.error(f'Fault analysis failed: {e}')
        fs = res().get('fs')
        if fs and 'sequence' in fs:
            ends_ = list(fs['end_I'].items())
            clr = fs['sequence'][fs['sequence']['Role'] == 'Main']['Time (s)'].max() if len(fs['sequence']) else None
            kpis([('Fault', f"{fs['type']} at {fs['d_km']:.1f} km ({fs['pos'] * 100:.0f}%)"), ('Fault current', f"{fs['Ik (kA)']:.2f} kA")]
                 + [(f'From {b}', f'{v:.2f} kA') for b, v in ends_[:2]]
                 + [('Cleared in', f"{clr * 1000:.0f} ms" if clr == clr and clr is not None else 'not cleared')])
            for n in fs['notes']:
                st.markdown(f"<div class='pg-note'>{n}</div>", unsafe_allow_html=True)
            st.plotly_chart(npl.sld_figure(D, fault_line=(fs['line'], fs['pos']), height=560), width=STRETCH, key='fs_sld')
            st.markdown('##### Sequence of tripping')
            if len(fs['sequence']):
                show(fs['sequence'])
                st.plotly_chart(npl.trip_sequence_figure(fs['sequence']), width=STRETCH, key='fs_seq')
            else:
                st.info('No relay operates: add the relays of this line and the next lines to the relay table (or load their settings files).')
            st.markdown('##### Every relay that sees the fault')
            if len(fs['relays']):
                show(fs['relays'], col=None)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('##### Phase voltages during the fault')
                show(fs['buses'])
            with c2:
                st.markdown('##### Current from each grid')
                st.dataframe(fs['grids'], hide_index=True, width=STRETCH)
            st.markdown('##### Current in every line and transformer')
            st.dataframe(fs['branches'], hide_index=True, width=STRETCH, height=360)
            st.caption('Symmetrical components on the whole network, pre-fault voltage 1.0 pu, loads neglected. Trip time = relay setting + '
                       f'{int(ng.RELAY_OP_S * 1000)} ms relay + {int(ng.BREAKER_S * 1000)} ms breaker.')
            st.download_button('📄 Fault analysis report (PDF)', pdfrep.study_pdf(D, {'fs': fs}, 'Fault Analysis', ss['names'][ss['ws']]), 'Fault_Analysis.pdf', key='fs_pdf')

# ============================================================== 9. Combined report
with tabs[9]:
    demo_note()
    st.markdown("<div class='pg-note'>One report with every analysis that has been run on this network. "
                "Press <b>Run all analyses</b> to run them all with the default settings.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    eng = c1.text_input('Prepared by', key='rp_eng')
    ttl = c2.text_input('Report title', 'Network Study Report', key='rp_title')
    if st.button('▶️ Run all analyses', type='primary', key='rp_all') and ready():
        with st.spinner('Load flow, voltage drop, short circuit, protection and N-1...'):
            try:
                lf = ng.load_flow(D)
                set_res(lf=lf, vd=ng.voltage_drop(lf), sc=ng.short_circuit(D, 'max', BUSES[0] if BUSES else None), prot=ng.protection(D),
                        cont=ng.contingency(D))
            except Exception as e:
                st.error(f'An analysis failed: {e}')
    R = res()
    done = [nm for k, ic, nm, _ in ANALYSES if k in R] + (['Fault Study'] if 'fs' in R else [])
    st.write('Included: ' + (', '.join(done) if done else 'nothing yet'))
    inc_v = st.checkbox('Add the verification document of the calculation engine to the report (appendix)', key='rp_val',
                        help='The engine checked against the IEEE 9 and 14 bus published results, IEC 60909 hand calculations and the IEC 60255 relay curves.')
    if done:
        RR = {k: v for k, v in R.items() if k != 'val'}
        if inc_v:
            RR['val'] = verification_table()
        pdf = pdfrep.study_pdf(D, RR, ttl, ss['names'][ss['ws']], eng)
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
                if R['sc'].get('breakers') is not None and len(R['sc']['breakers']):
                    z.writestr('breaker_duty.csv', R['sc']['breakers'].to_csv(index=False))
            if 'prot' in R:
                z.writestr('distance_zones.csv', R['prot']['distance'].to_csv(index=False)); z.writestr('overcurrent_grading.csv', R['prot']['oc_pairs'].to_csv(index=False))
            if 'cont' in R:
                z.writestr('n1_contingency.csv', R['cont']['table'].to_csv(index=False))
            if 'fs' in R:
                z.writestr('fault_relays.csv', R['fs']['relays'].to_csv(index=False)); z.writestr('fault_branch_currents.csv', R['fs']['branches'].to_csv(index=False))
                if 'sequence' in R['fs']:
                    z.writestr('fault_trip_sequence.csv', R['fs']['sequence'].to_csv(index=False)); z.writestr('fault_bus_voltages.csv', R['fs']['buses'].to_csv(index=False))
            z.writestr('network.xlsx', ng.network_excel(D))
        b2.download_button('🗂️ All results (CSV + network Excel, ZIP)', zb.getvalue(), 'NetworkGuard_Results.zip', 'application/zip', width=STRETCH)
        if st.button('📌 Save this study to the history', key='rp_save'):
            ss['studies'] = ss['studies'] + [{'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Network': ss['names'][ss['ws']], 'Title': ttl,
                                               **{nm: (STATUS_LABEL.get(status_of(k, R[k]), '') + ' · ' + headline(k, R[k])) for k, ic, nm, _ in ANALYSES if k in R}}]
    st.download_button('📘 Verification document of the calculation engine (PDF)', pdfrep.verification_pdf(verification_table()),
                       'NetworkGuard_Verification.pdf', 'application/pdf', key='rp_vdoc')
    if ss['studies']:
        st.markdown('##### Study history')
        st.dataframe(pd.DataFrame(ss['studies']), width=STRETCH, hide_index=True)

st.markdown(FOOTER, unsafe_allow_html=True)
