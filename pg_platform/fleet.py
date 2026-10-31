"""Fleet results of every module in one table (demo substation, or the results analyzed on the module pages).

All modules grade an asset the same way (risk 0-100 %, EXCELLENT / WATCH / ACTION, FMEA), so their results can be put
side by side. The demo substation joins the module demos by asset ID (T1, T2, RB1, Line 1 ...)."""
import ast
import os
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st

import pg_paths

COLS = ['Module', 'Asset ID', 'Asset', 'Asset Type', 'Risk (%)', 'Status', 'Main Driver', 'Action']
MODULE_INFO = {
    'transformerguard': ('TransformerGuard', '🛢️', 'Transformer and reactor health from DGA, oil and thermal loading'),
    'gridguard': ('GridGuard', '📈', 'Load and hot-spot forecast, insulation ageing, load transfer'),
    'pdguard': ('PDGuard', '⚡', 'Partial discharge risk of GIS, transformers, reactors, cables and motors'),
    'motorguard': ('MotorGuard', '⚙️', 'HV motor condition: stator, bearings, vibration, supply quality'),
    'relayguard': ('RelayGuard', '🛡️', 'Protection relay condition, tests and disturbance records'),
    'faultguard': ('FaultGuard', '🔍', 'Trip and incident analysis from disturbance records and events'),
}


def _paths():
    for d in [os.path.join(pg_paths.MODULES, '_common')] + [os.path.join(pg_paths.MODULES, x) for x in sorted(os.listdir(pg_paths.MODULES))]:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


def _const(module, script, name):
    """A constant defined in a module's app script (kept in one place: the module)."""
    with open(os.path.join(pg_paths.MODULES, module, script), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == name for e in t.elts):
                    vals = ast.literal_eval(node.value)
                    return vals[[e.id for e in t.elts].index(name)]
    return None


def _row(module, aid, name, typ, risk, status, driver, action):
    return {'Module': MODULE_INFO[module][0], 'Asset ID': str(aid), 'Asset': name or str(aid), 'Asset Type': typ,
            'Risk (%)': round(float(risk or 0), 1), 'Status': status, 'Main Driver': driver or '-', 'Action': action or '-'}


# ---------------------------------------------------------------- per module (demo)
@st.cache_data(show_spinner=False)
def transformer_demo():
    _paths()
    import joblib
    import powerguard_core as pc
    mh = joblib.load(pg_paths.model('rf_health_model.pkl')); mt = joblib.load(pg_paths.model('rf_thermal_model.pkl'))
    rows = []
    for aid, typ, gas, loads, hr, mo, meas, age in _const('transformerguard', 'AI_Transformer_Health_V2.app.py', 'DEMO_FLEET'):
        g = dict(pc.DGA_DEFAULTS); g.update(gas)
        d = pc.assess_dga(g, mh)
        L = {k: pc.to_kw(v, 'MVA') for k, v in zip(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'], loads)}
        t = pc.assess_thermal(L, pc.to_kva(150, 'MVA'), hr, mo, mt, meas)
        f = pc.fuse_risk(d, t); fm = pc.fmea_assessment(d, t)
        ins = pc.insulation_life(meas if meas else t['temp'], t['load_factor'], age)
        r = _row('transformerguard', aid, ('Transformer ' if typ == 'Transformer' else 'Reactor ') + aid, typ, f['overall'], f['status'],
                 f"{fm['governing']['Failure Mode']}", f['action'])
        r.update({'DGA Index': round(d['index'], 1), 'Top-Oil (C)': round(t['temp'], 1), 'Insulation Life (yrs)': round(min(ins['remaining_years_now'], 60), 1),
                  'Diagnosis': d.get('diagnosis') or d.get('duval') or '-'})
        rows.append(r)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def grid_demo():
    _paths()
    import gridguard_core as gc
    model = gc.load_model(pg_paths.model('gridguard_model.pkl'))
    origin = _const('gridguard', 'GridGuard_AI.app.py', 'DEMO_ORIGIN')
    d = gc.demo_data(); w = d['weather']; o = pd.Timestamp(origin)
    wf = gc.weather_forecast_from(w, o, gc.HORIZON_DAYS)
    wt = pd.to_datetime(w['Timestamp'])
    rows, fcs = [], {}
    for r in d['register'].to_dict('records'):
        L = d['loads'][d['loads']['Transformer ID'] == r['Transformer ID']]
        hts = pd.to_datetime(L['Timestamp']); hl = L['Load (MVA)'].to_numpy(float); m = hts < o
        wp = wt < o
        fc = gc.forecast(model, r, hts[m], hl[m], wt[wp], w['Ambient (C)'][wp], pd.to_datetime(wf['Timestamp']), wf['Ambient (C)'], o, gc.HORIZON_DAYS)
        a = gc.assess(r, fc)
        x = _row('gridguard', r['Transformer ID'], f"{r['Transformer ID']} ({r['Bay']})", 'Transformer', a['Overall Risk (%)'], a['Status'],
                 a['Governing Mode'], a['Action'])
        x.update({'Peak Load (%)': a['Peak Load (%)'], 'Max Hot-Spot (C)': a['Max Hot-Spot (C)'], 'Load Transfer (MVA)': a['Load Transfer Needed (MVA)']})
        rows.append(x)
        fcs[r['Transformer ID']] = {'ts': list(pd.DatetimeIndex(fc['ts'])), 'load': list(fc['load']), 'hot_spot': list(fc['hot_spot']),
                                    'rating': fc['rating'], 'ambient': list(fc['ambient'])}
    return pd.DataFrame(rows), fcs


@st.cache_data(show_spinner=False)
def pd_demo():
    _paths()
    import joblib
    import pdguard_core as pc
    model = joblib.load(pg_paths.model('pd_classifier.pkl'))
    d = pc.demo_data(end='2026-03-01')
    A = pc.assess(d['online'], d['sensors'], model, d['dga'], d['acoustic'], d['sf6'])
    L = pc.latest(A); L = L[~L['Noise Reference']]
    worst = L.loc[L.groupby('Asset ID')['PD Risk (%)'].idxmax()]
    rows = []
    for _, r in worst.iterrows():
        x = _row('pdguard', r['Asset ID'], r['Asset ID'], r.get('Asset Type', '-'), r['PD Risk (%)'], r['Status'],
                 f"PD: {r.get('Defect', '-')}", r.get('Action', '-'))
        x.update({'dB Above Baseline': r.get('dB Above Baseline')})
        rows.append(x)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def motor_demo():
    _paths()
    import motorguard_core as mc
    try:
        with open(pg_paths.model('thermal_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        model.predict(np.array([[80.0, 1.0, 35.0]]))
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        X, y = mc.make_thermal_training()
        model = RandomForestRegressor(n_estimators=60, max_depth=11, min_samples_leaf=8, random_state=0, n_jobs=-1).fit(X[:16000], y[:16000])
    day = _const('motorguard', 'MotorGuard_AI.app.py', 'DEMO_DATE')
    d = mc.demo_data()
    rows = [a for a in mc.assess_series(d['online'], d['register'], model, d['tests']) if a['Timestamp'] <= pd.Timestamp(day) + pd.Timedelta(days=1)]
    out = []
    for a in mc.latest(rows):
        out.append(_row('motorguard', a['Motor ID'], f"{a['Motor ID']} {a.get('Service', '')}".strip(), 'Motor', a['Overall Risk (%)'], a['Status'],
                        a.get('Governing Mode'), a.get('Action')))
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def relay_demo():
    _paths()
    import relayguard_core as rc
    ai = rc.load_record_model(pg_paths.model('record_anomaly_model.pkl'))
    day = _const('relayguard', 'RelayGuard_AI.app.py', 'DEMO_DATE')
    d = rc.demo_data()
    P = rc.prepare(d['register'], d['tests'], d['events'], d['records'], ai)
    out = []
    for rid, r in P['reg'].items():
        a = rc.assess(r, rc.summary(P, rid, pd.Timestamp(day)), day)
        out.append(_row('relayguard', rid, f"{rid} ({a.get('Bay', '')})", 'Protection relay', a['Overall Risk (%)'], a['Status'],
                        a.get('Governing Mode'), a.get('Action')) | {'Bay': a.get('Bay')})
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def fault_demo():
    _paths()
    import faultguard_core as fg
    model = fg.load_model(pg_paths.model('faultguard_model.pkl'))
    out = []
    for i in range(len(fg.DEMO_INCIDENTS)):
        meta, cfg, dat, E, line = fg.demo_incident(i)
        a = fg.analyze(fg.read_comtrade(cfg, dat), E, line, model=model, meta=meta)
        bay = fg.DEMO_INCIDENTS[i][1]
        x = _row('faultguard', bay, f"{bay} to {fg.DEMO_INCIDENTS[i][2]}", 'Line', a['Overall Risk (%)'], a['Status'],
                 f"{a['Governing Mode']} ({fg.main_reason(a)})", a['Action'])
        x.update({'Incident': meta['Incident'], 'Date': meta['Date'], 'Fault Type': a.get('type'), 'Location (km)': round(a['loc_km'], 1) if a.get('loc_km') is not None else None,
                  'Clearing (ms)': round(a['clearing_ms']) if a.get('clearing_ms') else None, 'Summary': fg.summary_text(a)})
        out.append(x)
    return pd.DataFrame(out)


DEMO = {'transformerguard': transformer_demo, 'gridguard': lambda: grid_demo()[0], 'pdguard': pd_demo, 'motorguard': motor_demo,
        'relayguard': relay_demo, 'faultguard': fault_demo}


# ---------------------------------------------------------------- results analyzed on the module pages
def _from_register(module, recs):
    rows = []
    for r in recs or []:
        try:
            if module == 'transformerguard':
                rows.append(_row(module, r.get('Asset ID'), r.get('Asset ID'), r.get('Asset Type'), r.get('Overall Risk (%)'), r.get('Status'),
                                 r.get('Failure Mode') or r.get('Driver'), r.get('Action Required')))
            elif module == 'gridguard':
                rows.append(_row(module, r['Transformer ID'], r['Transformer ID'], 'Transformer', r['Overall Risk (%)'], r['Status'], r.get('Governing Mode'), r.get('Action')))
            elif module == 'pdguard':
                rows.append(_row(module, r['Asset ID'], r['Asset ID'], r.get('Asset Type'), r.get('PD Risk (%)'), r['Status'], f"PD: {r.get('Defect', '-')}", r.get('Action')))
            elif module == 'motorguard':
                rows.append(_row(module, r['Motor ID'], r['Motor ID'], 'Motor', r['Overall Risk (%)'], r['Status'], r.get('Governing Mode'), r.get('Action')))
            elif module == 'relayguard':
                rows.append(_row(module, r['Relay ID'], r['Relay ID'], 'Protection relay', r['Overall Risk (%)'], r['Status'], r.get('Governing Mode'), r.get('Action')))
            elif module == 'faultguard':
                x = _row(module, r.get('Line'), r.get('Line'), 'Line', r['Overall Risk (%)'], r['Status'],
                         f"{r.get('Highest-Risk Mode', '')} ({r.get('Main Reason', '')})", r.get('Action'))
                x.update({'Incident': r.get('Incident'), 'Date': r.get('Date'), 'Fault Type': r.get('Fault Type'), 'Summary': r.get('Summary')})
                rows.append(x)
        except Exception:
            continue
    return pd.DataFrame(rows)


def fleet(source='demo', states=None):
    """All module results in one table. source: 'demo' or 'pages' (results analyzed on the module pages)."""
    parts = []
    for k in MODULE_INFO:
        if source == 'demo':
            try:
                parts.append(DEMO[k]())
            except Exception as e:
                parts.append(pd.DataFrame([_row(k, '-', f'demo not available: {e}', '-', 0, 'EXCELLENT', '-', '-')]))
        else:
            stt = (states or {}).get(k, {})
            if k == 'transformerguard':
                log = stt.get('fmea_log')
                recs = log.groupby('Asset ID', as_index=False).last().to_dict('records') if isinstance(log, pd.DataFrame) and len(log) else []
            else:
                recs = stt.get('register') or []
            parts.append(_from_register(k, recs))
    F = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    return F
