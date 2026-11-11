"""NetworkGuard AI - 3D network view (self-contained HTML, works offline).

scene_data() turns the network (and optionally a load flow and a fault analysis) into a compact JSON scene;
html() puts it into the 3D template with three.js inlined, so the file opens in any browser without internet.
"""
import json
import os

import numpy as np
import pandas as pd

import networkguard_core as ng

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'networkguard_assets')


def _f(v, nd=3):
    try:
        v = float(v)
        return None if not np.isfinite(v) else round(v, nd)
    except (TypeError, ValueError):
        return None


def scene_data(data, lf=None, fs=None, title='Network'):
    """Compact scene: substations with their buses and equipment, lines, sources, breakers, the load flow and the fault."""
    d = ng.clean(data)
    lay = ng.sld_layout(d)
    pos = lay['pos']
    if not pos:
        return {'title': title, 'subs': [], 'buses': [], 'lines': [], 'trafos': [], 'shunts': [], 'sources': [], 'breakers': []}
    xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    S = 900.0 / span                                   # world units per layout unit
    cx, cy = float(xs.mean()), float(ys.mean())
    W = lambda x, y: (round((x - cx) * S, 2), round(-(y - cy) * S, 2))
    lfb = lf['bus'].set_index('Bus') if lf else None
    lfl = lf['line'].set_index('Line') if lf else None
    lft = lf['trafo'].set_index('Transformer') if lf else None
    buses = []
    for _, b in d['Buses'].iterrows():
        if b['Bus'] not in pos:
            continue
        x, z = W(*pos[b['Bus']])
        r = {'id': b['Bus'], 'sub': b['Substation'], 'kv': _f(b['kV'], 1), 'x': x, 'z': z}
        if lfb is not None and b['Bus'] in lfb.index:
            r['v'] = _f(lfb.at[b['Bus'], 'V (pu)']); r['st'] = str(lfb.at[b['Bus'], 'Status'])
        buses.append(r)
    subs = []
    for s, (x0, y0, n) in lay['subs'].items():
        x, z = W(x0, y0)
        subs.append({'id': s, 'x': x, 'z': z, 'n': int(n)})
    lines = []
    for br in lay['branches']:
        if br['kind'] != 'Line':
            continue
        r = {'id': br['id'], 'a': br['a'], 'b': br['b'], 'off': round(br['off'] * S, 2), 'cable': bool(br['cable'])}
        if lfl is not None and br['id'] in lfl.index:
            q = lfl.loc[br['id']]
            r.update(load=_f(q['Loading (%)'], 1), p=_f(q['P From (MW)'], 1), st=str(q['Status']), on=bool(q['In Service']))
        lines.append(r)
    L = d['Lines'].set_index('Line')
    for r in lines:
        if r['id'] in L.index:
            r['km'] = _f(L.at[r['id'], 'Length km'], 2)
            on_ = L.at[r['id'], 'Other Names'] if 'Other Names' in L.columns else ''
            r['alias'] = '' if pd.isna(on_) else str(on_)
    trafos = []
    for br in lay['branches']:
        if br['kind'] != 'Transformer':
            continue
        r = {'id': br['id'], 'a': br['a'], 'b': br['b'], 'off': round(br['off'] * S, 2)}
        if lft is not None and br['id'] in lft.index:
            r.update(load=_f(lft.at[br['id'], 'Loading (%)'], 1), p=_f(lft.at[br['id'], 'P HV (MW)'], 1), st=str(lft.at[br['id'], 'Status']))
        trafos.append(r)
    import networkguard_plots as npl
    out = {'title': title, 'scale': round(S, 3), 'bus_half': round(0.42 * S, 2), 'kvcol': {int(k): v for k, v in npl.KV_COL.items()}, 'subs': subs, 'buses': buses, 'lines': lines, 'trafos': trafos,
           'shunts': [{'id': s['id'], 'kind': s['kind'], 'bus': s['bus']} for s in lay['shunts'] if s['bus'] in pos],
           'sources': [{'id': s['id'], 'kind': s['kind'], 'bus': s['bus']} for s in lay['sources'] if s['bus'] in pos],
           'breakers': [{'id': b['id'], 'bus': b['bus'], 'el': b['element']} for b in lay['breakers'] if b['bus'] in pos]}
    if lf is not None:
        src = lf.get('sources')
        if src is not None and len(src):
            pm = {r['Source']: _f(r['P (MW)'], 1) for _, r in src.iterrows()}
            for s in out['sources']:
                s['p'] = pm.get(s['id'])
    if fs and 'sequence' in fs:
        out['fault'] = fault_scene(fs, d)
    return out


def fault_scene(fs, d):
    """The fault analysis as a timeline for the 3D playback."""
    bv = {}
    for _, r in fs['buses'].iterrows():
        if r['Bus'] == 'Fault point':
            continue
        bv[r['Bus']] = {'a': _f(r['Va (pu)']), 'b': _f(r['Vb (pu)']), 'c': _f(r['Vc (pu)']), 'low': _f(r['Lowest (pu)']), 'st': r['Status']}
    br = {}
    for _, r in fs['branches'].iterrows():
        nm = str(r['Branch'])
        key = fs['line'] if nm.startswith(fs['line'] + ' (') else nm
        side = nm[len(fs['line']) + 2:-6] if key == fs['line'] and nm.endswith(' side)') else None
        rec = {'i': _f(r['Highest (kA)'], 2), 'from': r['From'], 'to': r['To'], 'i0': _f(r['3I0 (kA)'], 2)}
        if side:
            br.setdefault(key, {'sides': {}})['sides'][side] = rec
        else:
            br[key] = rec
    seq = []
    for _, s in fs['sequence'].iterrows():
        seq.append({'t': _f(s['Time (s)'], 3), 'relay': s['Relay'], 'bus': s['Bus'], 'fn': s['Function'], 'op': s['Operates'], 'role': s['Role'],
                    'st': s['Status']})
    R = fs['relays']
    trip = []                                         # breaker openings: the relays of the faulted line (main protection)
    if len(R):
        for _, r in R.iterrows():
            if pd.notna(r.get('Trip time (s)')) and r['Protects'] == fs['line']:
                trip.append({'bus': r['Bus'], 'el': r['Protects'], 't': _f(r['Trip time (s)'], 3)})
    first = {}
    for x in trip:
        if x['bus'] not in first or x['t'] < first[x['bus']]['t']:
            first[x['bus']] = x
    grids = {r['Grid']: _f(r['Highest phase current (kA)'], 2) for _, r in fs['grids'].iterrows()} if len(fs['grids']) else {}
    main = [s['t'] for s in seq if s['role'] == 'Main']
    return {'line': fs['line'], 'from': fs['from'], 'to': fs['to'], 'pos': _f(fs['pos'], 4), 'd_km': _f(fs['d_km'], 1), 'len_km': _f(fs['length_km'], 2),
            'type': fs['type'], 'name': fs['fault'], 'ik': fs['Ik (kA)'], 'ends': {k: _f(v, 2) for k, v in fs['end_I'].items()},
            'buses': bv, 'branches': br, 'grids': grids, 'seq': seq, 'opens': list(first.values()), 'notes': list(fs['notes']),
            'cleared': max(main) if main else None, 'rf': _f(fs.get('rf'), 1)}


def _three():
    with open(os.path.join(ASSETS, 'three.min.js'), encoding='utf-8') as f:
        return f.read()


def html(data, lf=None, fs=None, title='Network', subtitle='', start='network'):
    """The 3D view as one self-contained HTML file (str). start: 'network', 'lf' (load flow) or 'fault' (fault playback)."""
    sc = scene_data(data, lf, fs, title)
    sc['subtitle'] = subtitle
    sc['start_mode'] = start
    with open(os.path.join(ASSETS, 'network3d.html'), encoding='utf-8') as f:
        t = f.read()
    js = json.dumps(sc, default=lambda o: o.item() if hasattr(o, 'item') else str(o)).replace('</', '<\\/')
    return t.replace('/*__THREE__*/', _three()).replace('__SCENE_JSON__', js).replace('__TITLE__', str(title).replace('<', ''))
