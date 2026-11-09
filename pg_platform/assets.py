"""One asset register for the whole platform: joins the module results of the same physical asset.

Each module names assets its own way (TR-1 in GridGuard, T1-87 for the T1 differential relay, MOTOR-1 in PDGuard).
`site_asset` maps a module asset ID to one site asset, so Asset 360 can show every module's view of that asset.
The demo joins are illustrative (the module demos are separate simulated substations)."""
import re

import pandas as pd

NAMES = {
    'T': ('Power Transformer {}', 'Transformer'), 'RB': ('Shunt Reactor {} (Busbar)', 'Reactor'), 'RL': ('Shunt Reactor {} (Line)', 'Reactor'),
    'Line': ('{} 400 kV Overhead Line', 'Line'), 'M': ('HV Motor {}', 'Motor'), 'Q': ('GIS Bay {}', 'GIS'), 'BUS': ('GIS Busbar {}', 'GIS'),
    'CABLE': ('220 kV Cable {}', 'Cable'), 'Busbar': ('{} Protection Zone', 'Busbar'),
}
TYPE_ICON = {'Transformer': '🛢️', 'Reactor': '🌀', 'Line': '🗼', 'Motor': '⚙️', 'GIS': '🔌', 'Cable': '〰️', 'Busbar': '🧩', 'Other': '📦'}


def site_asset(module, aid):
    """The site asset ID that a module's asset ID belongs to."""
    a = str(aid).strip()
    if module == 'GridGuard':
        m = re.fullmatch(r'TR-?(\d+)', a)
        return f'T{m.group(1)}' if m else a
    if module == 'PDGuard':
        m = re.fullmatch(r'MOTOR-?(\d+)', a)
        return f'M-{int(m.group(1)):02d}' if m else a
    if module == 'RelayGuard':
        m = re.fullmatch(r'L(\d+)-M\d+', a)
        if m:
            return f'Line {m.group(1)}'
        m = re.fullmatch(r'(T\d+)-\w+', a)
        if m:
            return m.group(1)
        m = re.fullmatch(r'R(\d+)-\w+', a)
        if m:
            return f'RB{m.group(1)}'
        m = re.fullmatch(r'BB-87([A-Z])', a)
        if m:
            return f'Busbar {m.group(1)}'
    return a


def asset_info(sid):
    """(name, type) of a site asset."""
    for pre in sorted(NAMES, key=len, reverse=True):
        if sid.startswith(pre) and (pre in ('Line', 'Busbar') or re.fullmatch(rf'{pre}-?\d+(-CB\d+)?|{pre}\d+', sid)):
            n, t = NAMES[pre]
            return n.format(sid), t
    return sid, 'Other'


def combined_risk(risks):
    """Old rule (kept for reference): the highest module risk + 5 for every other module above 35 %."""
    r = sorted([float(x) for x in risks if x is not None], reverse=True)
    if not r:
        return 0.0
    return round(min(100.0, r[0] + 5 * sum(1 for x in r[1:] if x > 35)), 1)


def band(r):
    return 'EXCELLENT' if r <= 35 else ('WATCH' if r <= 70 else 'ACTION')


def latest_incidents(F):
    """FaultGuard rows are incidents; the line's state is its latest incident."""
    fg = F[F['Module'] == 'FaultGuard']
    if fg.empty or 'Date' not in fg:
        return fg
    return fg.assign(_d=pd.to_datetime(fg['Date'], errors='coerce')).sort_values('_d').groupby('Asset ID', as_index=False).last().drop(columns='_d')


def module_view(F):
    """One row per module asset (FaultGuard: latest incident per line)."""
    if F.empty:
        return F
    return pd.concat([F[F['Module'] != 'FaultGuard'], latest_incidents(F)], ignore_index=True)


def register(F, crit=None, n1=None, n1_source=''):
    """Site asset register: one row per site asset with its combined risk (likelihood x consequence, see risk.py).
    crit: {site asset: criticality 1-5 set by the engineer}; n1: {asset id: N-1 result of its outage}."""
    from pg_platform import risk as RK
    V = module_view(F)
    cols = ['Site Asset', 'Name', 'Type', 'Modules', 'Module Count', 'Likelihood', 'Consequence', 'Score', 'Priority', 'Priority Name',
            'Combined Risk (%)', 'Status', 'Main Driver', 'Action', 'Confirmed', 'Criticality', 'Steps']
    if V.empty:
        return pd.DataFrame(columns=cols)
    V = V.assign(**{'Site Asset': [site_asset(m, a) for m, a in zip(V['Module'], V['Asset ID'])]})
    V = V[~V['Site Asset'].astype(str).str.startswith('N-1 ')]          # N-1 results are the consequence of other assets
    crit, n1 = crit or {}, n1 or {}
    rows = []
    for sid, g in V.groupby('Site Asset'):
        if sid in ('-', '', 'None'):
            continue
        name, typ = asset_info(sid)
        if typ == 'Other':
            typ = str(g.iloc[0].get('Asset Type') or 'Other')
            name = str(g.iloc[0].get('Asset') or sid)
        res = RK.assess_asset(g, typ, name, crit.get(sid), n1.get(sid) or next((n1[a] for a in g['Asset ID'] if a in n1), None), n1_source)
        rows.append({'Site Asset': sid, 'Name': name, 'Type': typ, 'Modules': ', '.join(sorted(g['Module'].unique())), 'Module Count': g['Module'].nunique(),
                     **{k: res[k] for k in cols if k in res}})
    return pd.DataFrame(rows, columns=cols).sort_values(['Score', 'Consequence', 'Likelihood'], ascending=False, ignore_index=True)


def rows_of(F, sid):
    V = module_view(F)
    return V[[site_asset(m, a) == sid for m, a in zip(V['Module'], V['Asset ID'])]].sort_values('Risk (%)', ascending=False)
