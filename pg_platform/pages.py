"""Platform pages: Command Center, Asset 360 and About."""
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pg_platform import assets as A
from pg_platform import fleet as FL
from pg_platform import report as RP
from pg_platform import runner
from pg_platform import theme as T

SOURCES = {'demo': '🏭 Demo substation', 'pages': '🧪 My analyses'}
COLORS = {'EXCELLENT': T.GREEN, 'WATCH': T.AMBER, 'ACTION': T.RED}
LABEL = {'EXCELLENT': 'Normal', 'WATCH': 'Watch', 'ACTION': 'Action'}


def _source_picker(key):
    ss = st.session_state
    cur = ss.get('pg_source', 'demo')
    v = st.segmented_control('Data source', list(SOURCES), default=cur, format_func=SOURCES.get, key=f'pg_src_{key}',
                             label_visibility='collapsed')
    if v and v != cur:
        ss['pg_source'] = v
        cur = v
    return cur


def _source_bar(key):
    l, r = st.columns([3, 2])
    with r:
        src = _source_picker(key)
    l.markdown(f"<div style='color:#5b6b82;font-size:.85rem;padding-top:8px'>Showing: <b style='color:#0b2e59'>{SOURCES[src]}</b>"
               f"{' (simulated, anonymised)' if src == 'demo' else ' (results analyzed on the module pages in this session)'}</div>", unsafe_allow_html=True)
    return src


def _data(source):
    if source == 'demo':
        with st.spinner('Loading the demo substation from all modules...'):
            F = FL.fleet('demo')
    else:
        F = FL.fleet('pages', {k: runner.module_state(k) for k in FL.MODULE_INFO})
    return F


def _empty_state(pages):
    st.markdown("<div class='pg-card' style='text-align:center;padding:30px'><div style='font-size:2.2rem'>🧪</div>"
                "<h3 style='color:#0b2e59;margin:6px 0'>No analyses yet</h3><p style='color:#5b6b82'>Analyze assets on the module pages "
                "(press <b>Analyze</b>) and their results appear here, or switch to the demo substation.</p></div>", unsafe_allow_html=True)
    cols = st.columns(6)
    for c, (k, (nm, ic, _)) in zip(cols, FL.MODULE_INFO.items()):
        c.page_link(pages[k], label=nm, icon=ic)


def _bar_html(g):
    n = max(len(g), 1)
    parts = ''.join(f"<span style='width:{100 * (g['Status'] == s).sum() / n:.1f}%;background:{COLORS[s]}'></span>" for s in ('ACTION', 'WATCH', 'EXCELLENT'))
    return f"<div class='pg-bar'>{parts}</div>"


def _tile(col, key, g, page):
    nm, ic, ds = FL.MODULE_INFO[key]
    with col:
        if len(g):
            w = g.sort_values('Risk (%)', ascending=False).iloc[0]
            cnt = (f"<div class='pg-counts'><span>🔴 <b>{(g['Status'] == 'ACTION').sum()}</b> action</span><span>🟠 <b>{(g['Status'] == 'WATCH').sum()}</b> watch</span>"
                   f"<span>🟢 <b>{(g['Status'] == 'EXCELLENT').sum()}</b> normal</span></div>")
            wr = '%.0f%%' % w['Risk (%)']
            worst = f"<div class='pg-worst'>Highest: <b>{w['Asset ID']}</b> {T.chip(w['Status'], wr)} · {w['Main Driver']}</div>"
            body = _bar_html(g) + cnt + worst
            n = f"{len(g)} {'incidents' if key == 'faultguard' else 'assets'}"
        else:
            body, n = "<div class='pg-worst'>No results yet: open the module and press Analyze.</div>", 'no results'
        st.markdown(f"<div class='pg-tile'><div class='hd'><div class='ic'>{ic}</div><div><div class='nm'>{nm} AI</div>"
                    f"<div style='color:#5b6b82;font-size:.78rem'>{n}</div></div></div><div class='ds'>{ds}</div>{body}</div>", unsafe_allow_html=True)
        st.page_link(page, label=f'Open {nm}', icon=':material/arrow_forward:')


def _alerts(R, n=6):
    X = R[R['Status'] != 'EXCELLENT'].head(n)
    if X.empty:
        return "<div class='pg-card'>No asset needs attention. ✅</div>"
    rows = ''.join(f"<div class='pg-row'><div class='pg-dot' style='background:{COLORS[r['Status']]}'></div><div style='flex:1'>"
                   f"<div class='t'>{A.TYPE_ICON.get(r['Type'], '📦')} {r['Name']} &nbsp;{T.chip(r['Status'], '%.0f%%' % r['Combined Risk (%)'])}</div>"
                   f"<div class='s'>{r['Main Driver']}</div><div class='s' style='color:#14213d'>➜ {r['Action']}</div></div></div>" for _, r in X.iterrows())
    return f"<div class='pg-card'>{rows}</div>"


def _incidents_html(I, n=5):
    if I.empty:
        return "<div class='pg-card'>No incidents.</div>"
    I = I.assign(_d=pd.to_datetime(I.get('Date'), errors='coerce')).sort_values('_d', ascending=False).head(n)
    rows = ''.join(f"<div class='pg-row'><div class='pg-dot' style='background:{COLORS.get(r['Status'], T.MUTED)}'></div><div style='flex:1'>"
                   f"<div class='t'>{r.get('Incident', '')} · {r['Asset']} &nbsp;{T.chip(r['Status'])}</div>"
                   f"<div class='s'>{r.get('Date', '')} · {r.get('Fault Type', '') or ''} · {r['Main Driver']}</div>"
                   f"<div class='s' style='color:#14213d'>{r.get('Summary', '') or ''}</div></div></div>" for _, r in I.iterrows())
    return f"<div class='pg-card'>{rows}</div>"


def _status_fig(F):
    g = F.groupby(['Module', 'Status']).size().unstack(fill_value=0).reindex(columns=['ACTION', 'WATCH', 'EXCELLENT'], fill_value=0)
    g = g.loc[[FL.MODULE_INFO[k][0] for k in FL.MODULE_INFO if FL.MODULE_INFO[k][0] in g.index]][::-1]
    fig = go.Figure()
    for s in ['ACTION', 'WATCH', 'EXCELLENT']:
        fig.add_bar(y=g.index, x=g[s], name=LABEL[s], orientation='h', marker_color=COLORS[s], text=g[s].where(g[s] > 0, None), textposition='inside')
    fig.update_layout(template='powerguard', barmode='stack', height=330, title='Results by module and status', legend=dict(orientation='h', y=-0.12),
                      margin=dict(l=10, r=10, t=45, b=10), xaxis_title=None)
    return fig


def _treemap(R):
    if R.empty:
        return None
    D = R.assign(Risk=R['Combined Risk (%)'].clip(lower=1), Label=R['Site Asset'])
    fig = px.treemap(D, path=[px.Constant('Site'), 'Type', 'Label'], values=[1] * len(D), color='Combined Risk (%)', range_color=[0, 100],
                     color_continuous_scale=[[0, '#16a34a'], [0.35, '#86d39b'], [0.5, '#f5c542'], [0.7, '#f59e0b'], [1, '#dc2626']])
    fig.update_traces(texttemplate='<b>%{label}</b>', hovertemplate='<b>%{label}</b><br>Risk %{color:.0f}%<extra></extra>',
                      root_color='#eef3fa', marker=dict(cornerradius=5))
    fig.update_layout(template='powerguard', height=330, title='Site risk map (combined risk)', margin=dict(l=8, r=8, t=45, b=8),
                      coloraxis_colorbar=dict(title='%', thickness=10))
    return fig


# ================================================================ Command Center
def command_center(pages):
    runner.enter('platform')
    ss = st.session_state
    now = datetime.now().strftime('%A %d %B %Y, %H:%M')
    T.hero('PowerGuard AI Command Center', 'Asset Intelligence Platform · one view of transformers, reactors, GIS, cables, motors, relays and lines', now)
    source = _source_bar('cc')
    F = _data(source)
    R = A.register(F)
    if F.empty:
        _empty_state(pages)
        return

    n = len(R); act = int((R['Status'] == 'ACTION').sum()); wat = int((R['Status'] == 'WATCH').sum())
    health = 100 - R['Combined Risk (%)'].mean() if n else 100
    inc = F[F['Module'] == 'FaultGuard']
    c = st.columns(6)
    T.kpi(c[0], 'Fleet health', f'{health:.0f}%', '100 − average risk', T.BLUE)
    T.kpi(c[1], 'Site assets', n, f"{len(F)} module results", T.NAVY)
    T.kpi(c[2], 'Action', act, 'need action now', T.RED)
    T.kpi(c[3], 'Watch', wat, 'trend and re-test', T.AMBER)
    T.kpi(c[4], 'Normal', n - act - wat, 'no issue found', T.GREEN)
    T.kpi(c[5], 'Trips analyzed', len(inc), f"{int((inc['Status'] == 'ACTION').sum())} need action" if len(inc) else 'FaultGuard', T.VIOLET)

    T.section('Modules', 'press a module to open it')
    keys = list(FL.MODULE_INFO)
    for row in (keys[:4], keys[4:]):
        cols = st.columns(4)
        for col, k in zip(cols, row):
            _tile(col, k, F[F['Module'] == FL.MODULE_INFO[k][0]], pages[k])

    l, r = st.columns([1.05, 1])
    with l:
        T.section('Priority alerts', 'site assets with the highest combined risk')
        st.markdown(_alerts(R), unsafe_allow_html=True)
        st.page_link(pages['asset360'], label='Open Asset 360', icon=':material/travel_explore:')
    with r:
        T.section('Fleet picture', '')
        tf = _treemap(R)
        if tf is not None:
            st.plotly_chart(tf, width='stretch', config={'displayModeBar': False})
        st.plotly_chart(_status_fig(F), width='stretch', config={'displayModeBar': False})

    if len(inc):
        T.section('Latest trips and incidents', 'FaultGuard')
        st.markdown(_incidents_html(inc), unsafe_allow_html=True)

    T.section('Unified asset register', 'all module results in one table')
    f1, f2, f3 = st.columns([2, 2, 3])
    mods = f1.multiselect('Module', sorted(F['Module'].unique()), key='pg_f_mod', placeholder='All modules')
    sts = f2.multiselect('Status', ['ACTION', 'WATCH', 'EXCELLENT'], format_func=lambda s: LABEL[s], key='pg_f_st', placeholder='All statuses')
    q = f3.text_input('Search', key='pg_f_q', placeholder='Asset, driver or action...')
    V = F.copy()
    if mods:
        V = V[V['Module'].isin(mods)]
    if sts:
        V = V[V['Status'].isin(sts)]
    if q:
        V = V[V[['Asset ID', 'Asset', 'Main Driver', 'Action']].astype(str).apply(lambda s: s.str.contains(q, case=False)).any(axis=1)]
    V = V.sort_values('Risk (%)', ascending=False)
    V = V.assign(Status=V['Status'].map({'ACTION': '🔴 Action', 'WATCH': '🟠 Watch', 'EXCELLENT': '🟢 Normal'}).fillna(V['Status']))
    st.dataframe(V[FL.COLS], width='stretch', hide_index=True, height=min(520, 38 + 35 * len(V)),
                 column_config={'Risk (%)': st.column_config.ProgressColumn('Risk (%)', min_value=0, max_value=100, format='%.0f%%'),
                                'Status': st.column_config.TextColumn('Status')})

    T.section('Reports', '')
    d1, d2, d3 = st.columns(3)
    d1.download_button('📄 Fleet Status Report (PDF)', RP.fleet_pdf(F, R, SOURCES[source].split(' ', 1)[1]), 'PowerGuard_Fleet_Status_Report.pdf',
                       'application/pdf', width='stretch', type='primary')
    d2.download_button('📊 Module results (CSV)', F.to_csv(index=False).encode(), 'PowerGuard_Module_Results.csv', 'text/csv', width='stretch')
    d3.download_button('🗂️ Site asset register (CSV)', R.to_csv(index=False).encode(), 'PowerGuard_Site_Assets.csv', 'text/csv', width='stretch')
    st.markdown(T.FOOTER, unsafe_allow_html=True)


# ================================================================ Asset 360
def _gauge(v, status):
    fig = go.Figure(go.Indicator(mode='gauge+number', value=v, number=dict(suffix='%', font=dict(size=40, color=COLORS[status])),
                                 gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 35, 70, 100], tickfont=dict(size=10)), bar=dict(color=COLORS[status], thickness=0.3),
                                            steps=[dict(range=[0, 35], color='#e8f7ee'), dict(range=[35, 70], color='#fff6e0'), dict(range=[70, 100], color='#fdecec')],
                                            borderwidth=0)))
    fig.update_layout(template='powerguard', height=220, margin=dict(l=20, r=20, t=30, b=0), title=dict(text='Combined risk', x=0.5))
    return fig


def _module_card(r):
    key = next((k for k, v in FL.MODULE_INFO.items() if v[0] == r['Module']), None)
    nm, ic, ds = FL.MODULE_INFO.get(key, (r['Module'], '📦', ''))
    extra = ''
    for c in ['DGA Index', 'Top-Oil (C)', 'Insulation Life (yrs)', 'Diagnosis', 'Peak Load (%)', 'Max Hot-Spot (C)', 'Load Transfer (MVA)',
              'dB Above Baseline', 'Incident', 'Date', 'Fault Type', 'Location (km)', 'Clearing (ms)']:
        v = r.get(c)
        if isinstance(v, float) and not pd.isna(v):
            v = f'{v:.1f}'
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and v != '-':
            extra += f"<span style='display:inline-block;margin:3px 10px 0 0;font-size:.78rem;color:#5b6b82'>{c}: <b style='color:#14213d'>{v}</b></span>"
    return (f"<div class='pg-tile' style='margin-bottom:12px'><div class='hd'><div class='ic'>{ic}</div><div style='flex:1'><div class='nm'>{nm} AI</div>"
            f"<div style='color:#5b6b82;font-size:.78rem'>{r['Asset']}</div></div><div style='text-align:right'>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{COLORS.get(r['Status'], T.INK)}'>{r['Risk (%)']:.0f}%</div>{T.chip(r['Status'])}</div></div>"
            f"<div style='margin-top:10px;font-size:.86rem'><b>Driver:</b> {r['Main Driver']}</div>"
            f"<div style='font-size:.86rem;margin-top:3px'><b>Action:</b> {r['Action']}</div><div>{extra}</div></div>")


def _forecast_fig(fc, title):
    ts = pd.to_datetime(fc['ts'])
    fig = go.Figure()
    fig.add_scatter(x=ts, y=fc['load'], name='Load forecast (MVA)', line=dict(color=T.BLUE, width=2.2), fill='tozeroy', fillcolor='rgba(47,128,237,.08)')
    fig.add_hline(y=fc['rating'], line=dict(color=T.RED, dash='dash'), annotation_text=f"Rating {fc['rating']:.0f} MVA")
    fig.add_scatter(x=ts, y=fc['hot_spot'], name='Hot-spot (°C)', yaxis='y2', line=dict(color=T.AMBER, width=2))
    fig.update_layout(template='powerguard', height=330, title=title, legend=dict(orientation='h', y=-0.18),
                      yaxis=dict(title='MVA'), yaxis2=dict(title='°C', overlaying='y', side='right', showgrid=False), margin=dict(l=10, r=10, t=45, b=10))
    return fig


def asset360(pages):
    runner.enter('platform')
    ss = st.session_state
    T.hero('Asset 360', 'Every module\'s view of one asset, with one combined risk', datetime.now().strftime('%A %d %B %Y'))
    source = _source_bar('a360')
    F = _data(source)
    R = A.register(F)
    if R.empty:
        _empty_state(pages)
        return
    types = ['All'] + sorted(R['Type'].unique())
    c1, c2 = st.columns([1, 3])
    t = c1.selectbox('Asset type', types, key='pg_a_type')
    RR = R if t == 'All' else R[R['Type'] == t]
    opts = list(RR['Site Asset'])
    prev = ss.get('pg_a_sel_v')
    if prev not in opts and len(RR):
        rich = RR[RR['Module Count'] >= 3]
        prev = (rich if len(rich) else RR).iloc[0]['Site Asset']
    sid = c2.selectbox('Asset', opts, index=opts.index(prev) if prev in opts else 0, key='pg_a_sel',
                       format_func=lambda x: f"{A.TYPE_ICON.get(R.set_index('Site Asset').at[x, 'Type'], '📦')} {x} · {R.set_index('Site Asset').at[x, 'Name']}"
                       f"  ({R.set_index('Site Asset').at[x, 'Combined Risk (%)']:.0f}%)")
    ss['pg_a_sel_v'] = sid
    a = R.set_index('Site Asset').loc[sid]
    rows = A.rows_of(F, sid)

    l, r = st.columns([1, 2.2])
    with l:
        st.markdown(f"<div class='pg-card' style='text-align:center'><div style='font-size:2.4rem'>{A.TYPE_ICON.get(a['Type'], '📦')}</div>"
                    f"<div style='font-weight:800;color:#0b2e59;font-size:1.15rem'>{a['Name']}</div><div style='color:#5b6b82;font-size:.85rem'>{a['Type']} · {sid}</div>"
                    f"<div style='margin-top:8px'>{T.chip(a['Status'])}</div></div>", unsafe_allow_html=True)
        st.plotly_chart(_gauge(a['Combined Risk (%)'], a['Status']), width='stretch', config={'displayModeBar': False})
        st.markdown(f"<div class='pg-card'><div style='font-size:.8rem;color:#5b6b82'>MAIN DRIVER</div><div style='font-weight:650'>{a['Main Driver']}</div>"
                    f"<div style='font-size:.8rem;color:#5b6b82;margin-top:8px'>ACTION</div><div>{a['Action']}</div>"
                    f"<div style='font-size:.75rem;color:#8a97ab;margin-top:10px'>Combined risk = highest module risk + 5% for every other module above 35%.</div></div>",
                    unsafe_allow_html=True)
    with r:
        T.section('What each module sees', f"{rows['Module'].nunique()} module(s)")
        if len(rows) > 1:
            fig = go.Figure(go.Bar(x=rows['Risk (%)'], y=rows['Module'] + ' · ' + rows['Asset ID'].astype(str), orientation='h',
                                   marker_color=[COLORS[s] for s in rows['Status']], text=[f'{v:.0f}%' for v in rows['Risk (%)']], textposition='outside'))
            fig.update_layout(template='powerguard', height=70 + 42 * len(rows), xaxis=dict(range=[0, 110], title='Risk (%)'),
                              yaxis=dict(autorange='reversed'), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        cols = st.columns(2)
        for i, (_, rr) in enumerate(rows.iterrows()):
            cols[i % 2].markdown(_module_card(rr), unsafe_allow_html=True)

    gg = rows[rows['Module'] == 'GridGuard']
    if source == 'demo' and len(gg):
        _, fcs = FL.grid_demo()
        fc = fcs.get(gg.iloc[0]['Asset ID'])
        if fc:
            T.section('Load and hot-spot forecast', 'GridGuard')
            st.plotly_chart(_forecast_fig(fc, f"{gg.iloc[0]['Asset ID']}: next {len(fc['ts']) // 24} days"), width='stretch', config={'displayModeBar': False})
    I = F[(F['Module'] == 'FaultGuard') & (F['Asset ID'] == sid)]
    if len(I):
        T.section('Trips on this asset', 'FaultGuard')
        st.markdown(_incidents_html(I, 10), unsafe_allow_html=True)

    links = st.columns(len(rows['Module'].unique()) + 1)
    for c, m in zip(links, rows['Module'].unique()):
        k = next(k for k, v in FL.MODULE_INFO.items() if v[0] == m)
        c.page_link(pages[k], label=f'Open {m}', icon=FL.MODULE_INFO[k][1])
    extra = [(f"{rr['Module']} ({rr['Asset ID']})", f"Risk {rr['Risk (%)']:.0f}%. Driver: {rr['Main Driver']}. Action: {rr['Action']}")
             for _, rr in rows.iterrows()]
    links[-1].download_button('📄 Asset 360 Report (PDF)', RP.asset_pdf(sid, a['Name'], a['Type'], rows, a['Combined Risk (%)'], a['Status'], extra),
                              f'PowerGuard_Asset360_{sid}.pdf', 'application/pdf', type='primary', width='stretch')
    if source == 'demo':
        st.caption('Demo: the module demos are separate simulated substations; they are joined here by asset ID (T1 = TR-1 in GridGuard, '
                   'L1-M1/M2 relays = Line 1) to show the combined view.')
    st.markdown(T.FOOTER, unsafe_allow_html=True)


# ================================================================ About
def about(pages):
    runner.enter('platform')
    T.hero('About PowerGuard AI', 'Asset Intelligence Platform', 'Version 1.0')
    st.markdown("""
<div class='pg-card'>
<p><b>PowerGuard AI</b> brings seven condition-monitoring, analysis and network-study modules into one platform with one login, one asset register and one
risk scale (0-100 %, <b>NORMAL</b> ≤ 35 %, <b>WATCH</b> 35-70 %, <b>ACTION</b> > 70 %, FMEA per IEC 60812).</p>
</div>""", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (k, (nm, ic, ds)) in enumerate(FL.MODULE_INFO.items()):
        with cols[i % 3]:
            st.markdown(f"<div class='pg-tile' style='margin-bottom:12px'><div class='hd'><div class='ic'>{ic}</div><div class='nm'>{nm} AI</div></div>"
                        f"<div class='ds'>{ds}</div></div>", unsafe_allow_html=True)
            st.page_link(pages[k], label=f'Open {nm}', icon=':material/arrow_forward:')
    T.section('How the platform works', '')
    st.markdown("""
- **Command Center**: the fleet at a glance: health index, alerts, module status, latest trips, unified register and reports.
- **Asset 360**: one asset seen by every module (for example a transformer: DGA and thermal, load forecast, partial discharge, its protection relays),
  with a combined risk = highest module risk + 5 % for every other module above 35 %.
- **Modules**: each module keeps its own pages (overview, analysis, combined report, history) and downloads (PDF, CSV).
- **Data source**: *Demo substation* (simulated, anonymised) or *My analyses* (the results analyzed on the module pages in this session).
""")
    st.markdown(T.FOOTER, unsafe_allow_html=True)
