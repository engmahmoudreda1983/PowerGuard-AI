"""Combined asset risk: likelihood of failure x consequence of failure (5 x 5 risk matrix).

Basis: ISO 55000 / IEC 31010 (likelihood x consequence matrix), the Ofgem CNAIM approach (condition indices combined into
a probability of failure, network impact as the consequence), CIGRE asset-management practice, IEC 60599 / IEEE C57.104
(DGA fault types) and IEC 60270 (partial discharge).

Rules
  Likelihood (1-5), from the condition modules of the asset:
    1  start from the highest condition risk: <20 % -> 1, 20-35 -> 2, 35-55 -> 3, 55-75 -> 4, >75 -> 5
    2  confirmation: two different modules that measure in different ways and see the same failure mechanism, each at
       35 % or more, raise the likelihood by one step (the defect is confirmed by two sources)
    3  unrelated findings do not raise it; they are listed
    (an asset seen only by the protection modules takes its likelihood from them)
  Consequence (1-5):
    4  criticality of the asset: 400 kV plant 4, 220 / 132 kV 3, 33 kV and below or auxiliary motors 2; an asset marked
       critical by the engineer (e.g. an interconnector) 5, or any value the engineer sets
    5  network impact (NetworkGuard N-1, outage of this asset): load lost, islanding, no solution or another branch above
       120 % -> 5; another branch 100-120 % -> 4; otherwise the criticality
    6  protection readiness (RelayGuard, FaultGuard, for the relays and breaker of this asset): 55 % or more -> +1,
       75 % or more -> +2 (a slow or failed clearing makes the damage and the outage larger)
  Risk = likelihood x consequence (1-25):  15-25 priority 1 (critical), 10-14 priority 2 (high), 5-9 priority 3 (monitor),
  1-4 priority 4 (normal).
"""
import re

import pandas as pd

MECH = {
    'discharge': 'Insulation discharge',
    'thermal': 'Overheating / overload',
    'paper': 'Paper insulation ageing',
    'motor_ins': 'Motor stator insulation',
    'mechanical': 'Mechanical (bearing, rotor, vibration)',
    'supply': 'Supply quality',
    'protection': 'Protection and breaker',
    'short_circuit': 'Short-circuit duty',
    'voltage': 'Voltage',
    'general': 'General condition',
}
CONDITION, PROTECTION = 'condition', 'protection'
L_BANDS = [(20, 1), (35, 2), (55, 3), (75, 4)]                  # upper bound of the band -> likelihood
CONFIRM_MIN = 35.0
PROT_ADD = [(75.0, 2), (55.0, 1)]
NOT_CLEARING = r'^(Age and obsolescence|Test overdue)'          # protection findings that do not change how a fault is cleared
PRIORITY = [(15, 1, 'Critical', 'ACTION', 'Act within weeks'), (10, 2, 'High', 'WATCH', 'Plan within 3 months'),
            (5, 3, 'Monitor', 'WATCH', 'Monitor; test in the next cycle'), (0, 4, 'Normal', 'EXCELLENT', 'Routine maintenance')]
L_TEXT = {1: 'Rare', 2: 'Unlikely', 3: 'Possible', 4: 'Likely', 5: 'Almost certain'}
C_TEXT = {1: 'Negligible', 2: 'Minor', 3: 'Moderate', 4: 'Major', 5: 'Severe'}
TYPE_CRIT = {'Transformer': 4, 'Reactor': 4, 'Line': 4, 'GIS': 4, 'Busbar': 4, 'Switchgear': 4, 'Bus': 4, 'Cable': 3,
             'Motor': 2, 'Protection relay': 4, 'Other': 3}


def mechanism(module, driver, asset_type=''):
    """(mechanism key, kind) of one module result."""
    d = str(driver or '')
    if module == 'TransformerGuard':
        if re.search(r'\((PD|D1|D2|DT)\)', d):
            return 'discharge', CONDITION
        if re.search(r'\(T[123]\)|Overheating|Cooling', d):
            return 'thermal', CONDITION
        if 'Paper' in d:
            return 'paper', CONDITION
        return 'general', CONDITION
    if module == 'GridGuard':
        return 'thermal', CONDITION
    if module == 'PDGuard':
        if re.search(r'No PD|External noise', d):
            return 'general', CONDITION
        return ('motor_ins' if str(asset_type) == 'Motor' else 'discharge'), CONDITION
    if module == 'MotorGuard':
        if 'Stator insulation' in d:
            return 'motor_ins', CONDITION
        if re.search(r'Bearing|Rotor|vibration', d, re.I):
            return 'mechanical', CONDITION
        if 'Cooling' in d:
            return 'thermal', CONDITION
        if 'Supply' in d:
            return 'supply', CONDITION
        return 'general', CONDITION
    if module in ('RelayGuard', 'FaultGuard'):
        return 'protection', PROTECTION
    if module == 'NetworkGuard':
        if d.startswith('Load flow'):
            return 'thermal', CONDITION
        if d.startswith('Short circuit'):
            return 'short_circuit', CONDITION
        if d.startswith('Voltage'):
            return 'voltage', CONDITION
        if re.match(r'Distance settings|Coordination', d):
            return 'protection', PROTECTION
    return 'general', CONDITION


def likelihood_of(risk):
    for hi, l in L_BANDS:
        if risk < hi:
            return l
    return 5


def criticality_of(typ, name=''):
    """Default criticality from the asset type and its voltage."""
    m = re.search(r'(\d{2,3})\s*kV', str(name))
    if m and typ not in ('Motor',):
        kv = int(m.group(1))
        return 4 if kv >= 380 else (3 if kv >= 100 else 2)
    return TYPE_CRIT.get(typ, 3)


def impact_level(n1):
    """Consequence level from an N-1 result of the outage of this asset (None when it does not raise it)."""
    if not n1:
        return None, ''
    if n1.get('converged') is False or n1.get('islanded') or (n1.get('max_loading') or 0) > 120:
        why = ('separates ' + n1['islanded']) if n1.get('islanded') else (f"loads {n1.get('most_loaded')} to {n1['max_loading']:.0f} %"
                                                                            if n1.get('max_loading') else 'no load-flow solution')
        return 5, why
    if (n1.get('max_loading') or 0) > 100:
        return 4, f"loads {n1.get('most_loaded')} to {n1['max_loading']:.0f} %"
    return None, (f"leaves the network within limits (highest loading {n1['max_loading']:.0f} %)" if n1.get('max_loading') is not None else '')


def priority_of(score):
    for lo, p, name, status, act in PRIORITY:
        if score >= lo:
            return p, name, status, act
    return PRIORITY[-1][1:]


def assess_asset(rows, typ, name, crit=None, n1=None, n1_source=''):
    """Combined risk of one site asset from its module rows (DataFrame with Module, Asset ID, Risk (%), Main Driver, Action, Asset Type)."""
    R = rows.copy()
    mk = [mechanism(m, d, t) for m, d, t in zip(R['Module'], R['Main Driver'], R.get('Asset Type', [''] * len(R)))]
    R['Mechanism'] = [MECH[k] for k, _ in mk]
    R['_mk'] = [k for k, _ in mk]
    R['Kind'] = [k for _, k in mk]
    cond, prot = R[R['Kind'] == CONDITION], R[R['Kind'] == PROTECTION]
    lik_rows = cond if len(cond) else prot
    steps = []
    # --- likelihood
    top = lik_rows.sort_values('Risk (%)', ascending=False).iloc[0]
    L = likelihood_of(float(top['Risk (%)']))
    steps.append(f"Likelihood starts at {L} ({L_TEXT[L]}): highest {'condition' if len(cond) else 'protection'} risk "
                 f"{top['Risk (%)']:.0f} % from {top['Module']} ({top['Main Driver']}).")
    confirmed = []
    for k, g in lik_rows[lik_rows['Risk (%)'] >= CONFIRM_MIN].groupby('_mk'):
        if k == 'general' or g['Module'].nunique() < 2:
            continue
        confirmed.append((MECH[k], sorted(g['Module'].unique())))
    if confirmed:
        L0, L = L, min(5, L + 1)
        txt = '; '.join(f"{m} seen by {' and '.join(ms)}" for m, ms in confirmed)
        steps.append(f"Confirmed by two sources: {txt}. " + (f"Likelihood {L0} -> {L} ({L_TEXT[L]})." if L > L0 else 'Likelihood already at the top (5).'))
    others = lik_rows[(lik_rows['Risk (%)'] >= CONFIRM_MIN) & (lik_rows['_mk'] != top['_mk'])]
    other_txt = [f"{r['Module']}: {r['Main Driver']} ({r['Risk (%)']:.0f} %)" for _, r in others.iterrows()
                 if not any(MECH[r['_mk']] == m for m, _ in confirmed)]
    if other_txt:
        steps.append('Other findings (not the same mechanism, do not raise the likelihood): ' + '; '.join(other_txt) + '.')
    # --- consequence
    c0 = criticality_of(typ, name)
    if crit is not None:
        C, src = int(crit), 'set by the engineer'
    else:
        kv = re.search(r'(\d{2,3})\s*kV', str(name))
        C, src = c0, f"default for a {typ.lower()}" + (f" ({kv.group(1)} kV)" if kv else '')
    steps.append(f"Consequence starts at {C} ({C_TEXT[C]}): criticality {src}.")
    lvl, why = impact_level(n1)
    if lvl is not None and lvl > C:
        steps.append(f"Network impact (N-1{', ' + n1_source if n1_source else ''}): its outage {why}. Consequence {C} -> {lvl}.")
        C = lvl
    elif n1 and why:
        lv = f" (consequence {lvl}, already {C})" if lvl is not None else ''
        steps.append(f"Network impact (N-1{', ' + n1_source if n1_source else ''}): its outage {why}{lv}; no change.")
    elif not n1 and typ in ('Line', 'Transformer', 'Reactor', 'Busbar', 'Cable'):
        steps.append('Network impact: no N-1 result for this asset (NetworkGuard not run or not matched); criticality used.')
    pc = prot[~prot['Main Driver'].astype(str).str.contains(NOT_CLEARING, regex=True)]     # findings that slow or stop the clearing
    if len(cond) and len(prot) and len(pc):
        p = pc.sort_values('Risk (%)', ascending=False).iloc[0]
        add = next((a for lim, a in PROT_ADD if p['Risk (%)'] >= lim), 0)
        if add:
            C0, C = C, min(5, C + add)
            steps.append(f"Protection readiness: {p['Module']} {p['Asset ID']} at {p['Risk (%)']:.0f} % ({p['Main Driver']}): a fault would be "
                         f"cleared slower or not at all. Consequence {C0} -> {C}.")
        else:
            steps.append(f"Protection readiness: {p['Module']} {p['Asset ID']} at {p['Risk (%)']:.0f} %: ready, no change.")
    elif len(cond) and len(prot):
        steps.append('Protection readiness: the protection findings (age, test overdue) do not change how a fault is cleared; no change.')
    # --- matrix
    score = L * C
    pnum, pname, status, pact = priority_of(score)
    steps.append(f"Risk = {L} x {C} = {score} of 25: priority {pnum} ({pname}).")
    # one-line reason
    bits = []
    if confirmed:
        bits.append('; '.join(f"{m} confirmed by {' + '.join(ms)}" for m, ms in confirmed))
    else:
        bits.append(f"{top['Module']}: {top['Main Driver']}")
    if lvl is not None and lvl >= 4:
        bits.append(f"outage {why}")
    if len(cond) and len(pc):
        p = pc.sort_values('Risk (%)', ascending=False).iloc[0]
        if p['Risk (%)'] >= PROT_ADD[-1][0]:
            bits.append(f"protection weak ({p['Module']} {p['Asset ID']}: {p['Main Driver']})")
    acts = [pact]
    for _, r in R.sort_values('Risk (%)', ascending=False).iterrows():
        if r['Risk (%)'] >= CONFIRM_MIN and r['Action'] not in ('-', '', None) and r['Action'] not in acts:
            acts.append(r['Action'])
    return {'Likelihood': L, 'Consequence': C, 'Score': score, 'Priority': pnum, 'Priority Name': pname, 'Status': status,
            'Combined Risk (%)': round(score / 25 * 100, 1), 'Main Driver': ' · '.join(bits), 'Action': ' '.join(f'{i + 1}) {a}' for i, a in enumerate(acts[:4])),
            'Confirmed': ', '.join(m for m, _ in confirmed) or '-', 'Criticality': c0, 'Steps': steps,
            'Mechanisms': R[['Module', 'Asset ID', 'Risk (%)', 'Mechanism', 'Kind']]}


# ---------------------------------------------------------------- N-1 results
def n1_from_table(T):
    """{asset id: N-1 result} from a NetworkGuard contingency table."""
    out = {}
    if T is None or not len(T):
        return out
    for _, r in T.iterrows():
        out[str(r['Outage'])] = {'converged': bool(r.get('Converged', True)), 'max_loading': None if pd.isna(r.get('Max Loading (%)')) else float(r['Max Loading (%)']),
                                 'most_loaded': r.get('Most Loaded'), 'islanded': '' if r.get('Islanded Buses') in (None, '-') else str(r['Islanded Buses']),
                                 'violations': r.get('Violations')}
    return out


# illustrative N-1 results for the demo substation's joined assets (the module demos are separate simulated substations)
DEMO_N1 = {
    'T1': {'converged': True, 'max_loading': 118.0, 'most_loaded': 'T2', 'islanded': ''},
    'T2': {'converged': True, 'max_loading': 96.0, 'most_loaded': 'T1', 'islanded': ''},
    'Line 1': {'converged': True, 'max_loading': 104.0, 'most_loaded': 'Line 2', 'islanded': ''},
    'Line 2': {'converged': True, 'max_loading': 92.0, 'most_loaded': 'Line 1', 'islanded': ''},
}


def matrix_fig(R, height=380, title='Risk matrix (likelihood x consequence)', compact=False):
    """5 x 5 heat map with the site assets placed in their cells."""
    import plotly.graph_objects as go
    z = [[l * c for c in range(1, 6)] for l in range(1, 6)]
    colors = [[0.0, '#16a34a'], [0.16, '#86d39b'], [0.2, '#f5c542'], [0.4, '#f59e0b'], [0.6, '#ef4444'], [1.0, '#b91c1c']]
    fig = go.Figure(go.Heatmap(z=z, x=list(range(1, 6)), y=list(range(1, 6)), colorscale=colors, zmin=1, zmax=25, showscale=False,
                               hovertemplate='Likelihood %{y} x consequence %{x} = %{z}<extra></extra>', xgap=3, ygap=3))
    if R is not None and len(R):
        cells = {}
        for _, r in R.iterrows():
            cells.setdefault((int(r['Likelihood']), int(r['Consequence'])), []).append(r['Site Asset'])
        for (l, c), ids in cells.items():
            show = ids[:3]
            txt = '<br>'.join(x if len(x) <= 14 else x[:13] + '…' for x in show) + (f'<br><i>+{len(ids) - 3} more</i>' if len(ids) > 3 else '')
            fig.add_annotation(x=c, y=l, text=f'<b>{txt}</b>', showarrow=False, font=dict(size=9, color='#0b1220'),
                               bgcolor='rgba(255,255,255,.80)', borderpad=2)
    fig.update_layout(template='powerguard', height=height, title=title, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis=dict(title='Consequence', tickvals=list(range(1, 6)), showgrid=False,
                                 ticktext=[str(i) if compact else f'{i} {C_TEXT[i]}' for i in range(1, 6)]),
                      yaxis=dict(title='Likelihood', tickvals=list(range(1, 6)), showgrid=False,
                                 ticktext=[str(i) if compact else f'{i} {L_TEXT[i]}' for i in range(1, 6)]))
    return fig
