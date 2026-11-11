"""NetworkGuard AI engine: network data, load flow, voltage drop, short circuit (IEC 60909), protection coordination and
N-1 contingency for a transmission network.

The calculations use the standard methods that every network analysis program uses:
  * load flow: Newton-Raphson (pandapower),
  * short circuit: IEC 60909 (equivalent voltage source c*Un/sqrt(3) at the fault, positive and zero sequence impedances),
  * relay operating times: IEC 60255-151 / IEEE C37.112 inverse-time equations,
  * line parameters: the classic GMD / GMR equations with Carson's earth return for the zero sequence.
The results depend only on the equipment parameters entered, not on the program.

Network data is a dict of tables (one per equipment type). Every table row is one equipment; empty values are filled with
typical values and flagged, so the study can run before all data is available."""
import io
import math
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import pandapower as pp  # noqa: E402
import pandapower.shortcircuit as sc  # noqa: E402
import pandapower.topology as top  # noqa: E402

F_HZ = 50.0
CTI = 0.30   # coordination time interval (s)

# ------------------------------------------------------------------ equipment schemas
# field: (unit, description, typical value or None (required), kind)
SCHEMA = {
    'Substations': {
        'Substation': ('', 'Substation name', None, 'id'),
        'X': ('', 'Position on the single-line diagram (left to right)', 0.0, 'num'),
        'Y': ('', 'Position on the single-line diagram (bottom to top)', 0.0, 'num'),
        'Breaking kA': ('kA', 'Rated short-circuit breaking current of the circuit breakers', 40.0, 'num'),
        'Short-time kA (1 s)': ('kA', 'Rated short-time withstand current, 1 s (busbar and switchgear)', 40.0, 'num'),
        'Busbar A': ('A', 'Rated busbar current', 3150.0, 'num'),
    },
    'Buses': {
        'Bus': ('', 'Bus name', None, 'id'),
        'Substation': ('', 'Substation of the bus', None, 'text'),
        'kV': ('kV', 'Nominal voltage', 400.0, 'num'),
        'Vmin pu': ('pu', 'Lowest allowed voltage', 0.95, 'num'),
        'Vmax pu': ('pu', 'Highest allowed voltage', 1.05, 'num'),
        'Breaking kA': ('kA', 'Breaker rating at this voltage level if different from the substation value (empty = substation value)', '', 'num'),
    },
    'Lines': {
        'Line': ('', 'Line or cable circuit name, the same at both ends (one row per circuit)', None, 'id'),
        'Other Names': ('', 'Other names of the circuit (relay description, number such as L12), separated by commas', '', 'text'),
        'From Bus': ('', 'Sending-end bus', None, 'text'),
        'To Bus': ('', 'Receiving-end bus', None, 'text'),
        'Type': ('', 'OHL (overhead line) or Cable', 'OHL', 'text'),
        'Length km': ('km', 'Route length of the circuit', None, 'num'),
        'R1 ohm/km': ('ohm/km', 'Positive-sequence resistance', None, 'num'),
        'X1 ohm/km': ('ohm/km', 'Positive-sequence reactance', None, 'num'),
        'C1 nF/km': ('nF/km', 'Positive-sequence capacitance', None, 'num'),
        'R0 ohm/km': ('ohm/km', 'Zero-sequence resistance', None, 'num'),
        'X0 ohm/km': ('ohm/km', 'Zero-sequence reactance', None, 'num'),
        'C0 nF/km': ('nF/km', 'Zero-sequence capacitance', None, 'num'),
        'Rating A': ('A', 'Continuous current rating (lowest of conductor, CT, breaker, isolator)', 2000.0, 'num'),
        'Conductor': ('', 'Conductor name from the conductor library (used when R1/X1/C1 are empty)', '', 'text'),
        'Tower': ('', 'Tower name from the tower library (used when R1/X1/C1 are empty)', '', 'text'),
        'In Service': ('', '1 = in service, 0 = out of service', 1, 'num'),
    },
    'Transformers': {
        'Transformer': ('', 'Transformer name', None, 'id'),
        'HV Bus': ('', 'High-voltage bus', None, 'text'),
        'LV Bus': ('', 'Low-voltage bus', None, 'text'),
        'MVA': ('MVA', 'Rated power', 750.0, 'num'),
        'HV kV': ('kV', 'Rated HV voltage', 400.0, 'num'),
        'LV kV': ('kV', 'Rated LV voltage', 275.0, 'num'),
        'uk %': ('%', 'Short-circuit (impedance) voltage', 13.0, 'num'),
        'ur %': ('%', 'Resistive part of the impedance voltage (copper losses / rating)', 0.25, 'num'),
        'Vector Group': ('', 'Vector group, e.g. YNyn0, YNd11, Dyn11', 'YNyn0', 'text'),
        'Tap %': ('%', 'Tap position as % voltage change on the HV side', 0.0, 'num'),
        'In Service': ('', '1 = in service, 0 = out of service', 1, 'num'),
    },
    'Reactors': {
        'Reactor': ('', 'Shunt reactor name', None, 'id'),
        'Bus': ('', 'Connected bus (for a line reactor: the bus at that line end)', None, 'text'),
        'Line': ('', 'Line of a line reactor (empty for a bus reactor)', '', 'text'),
        'MVAr': ('MVAr', 'Rated reactive power at rated voltage', None, 'num'),
        'kV': ('kV', 'Rated voltage', 400.0, 'num'),
        'In Service': ('', '1 = in service, 0 = out of service', 1, 'num'),
    },
    'Grids': {
        'Grid': ('', 'External network (neighbouring grid) name', None, 'id'),
        'Bus': ('', 'Connection bus', None, 'text'),
        'Slack': ('', '1 = reference (balancing) grid of the load flow; one grid only', 0, 'num'),
        'P Import MW': ('MW', 'Power imported from this grid into the network (negative = export); not used for the slack', 0.0, 'num'),
        'V Setpoint pu': ('pu', 'Voltage kept by the grid at the connection bus', 1.0, 'num'),
        'Sk max MVA': ('MVA', 'Maximum short-circuit power of the grid at the connection bus', None, 'num'),
        'Sk min MVA': ('MVA', 'Minimum short-circuit power of the grid at the connection bus', None, 'num'),
        'R/X': ('', 'R/X ratio of the grid impedance', 0.1, 'num'),
        'X0/X1': ('', 'Zero to positive sequence reactance ratio', 1.0, 'num'),
        'R0/X0': ('', 'R/X ratio of the zero-sequence impedance', 0.1, 'num'),
    },
    'Converters': {
        'Converter': ('', 'HVDC converter or other power injection (no short-circuit contribution)', None, 'id'),
        'Bus': ('', 'Connection bus', None, 'text'),
        'P MW': ('MW', 'Active power injected into the network (negative = taken from the network)', 0.0, 'num'),
        'Q MVAr': ('MVAr', 'Reactive power injected into the network', 0.0, 'num'),
        'Rated MW': ('MW', 'Rated power', 600.0, 'num'),
        'In Service': ('', '1 = in service, 0 = out of service', 1, 'num'),
    },
    'Loads': {
        'Load': ('', 'Load name', None, 'id'),
        'Bus': ('', 'Connection bus', None, 'text'),
        'P MW': ('MW', 'Active power', 0.0, 'num'),
        'Q MVAr': ('MVAr', 'Reactive power', 0.0, 'num'),
        'In Service': ('', '1 = in service, 0 = out of service', 1, 'num'),
    },
    'Breakers': {
        'Breaker': ('', 'Circuit breaker name', None, 'id'),
        'Bus': ('', 'Bus the breaker connects to', None, 'text'),
        'Element': ('', 'Line, transformer, reactor or coupler the breaker switches', '', 'text'),
        'Breaking kA': ('kA', 'Rated short-circuit breaking current', 40.0, 'num'),
        'Rated A': ('A', 'Rated normal current', 2000.0, 'num'),
    },
    'Relays': {
        'Relay': ('', 'Relay name', None, 'id'),
        'Element': ('', 'Protected line or transformer', None, 'text'),
        'Bus': ('', 'Bus where the relay (CT) is located', None, 'text'),
        'Function': ('', '21 (distance) or 51 (phase overcurrent) or 51N (earth fault)', '21', 'text'),
        'CT Primary A': ('A', 'CT primary current', 2000.0, 'num'),
        'CT Secondary A': ('A', 'CT secondary current', 1.0, 'num'),
        'VT Primary kV': ('kV', 'VT primary voltage (phase to phase)', 400.0, 'num'),
        'VT Secondary V': ('V', 'VT secondary voltage (phase to phase)', 110.0, 'num'),
        'Curve': ('', 'IEC SI / IEC VI / IEC EI / IEC LTI / IEEE MI / IEEE VI / IEEE EI / DT', 'IEC SI', 'text'),
        'Pickup A': ('A', 'Overcurrent pickup, primary amperes', None, 'num'),
        'TMS': ('', 'Time multiplier (or definite time in s for DT)', 0.1, 'num'),
        'Inst Pickup A': ('A', 'Instantaneous (high-set) pickup, primary amperes; empty = off', '', 'num'),
        'Inst Time s': ('s', 'Instantaneous stage time', 0.05, 'num'),
        'Z1 %': ('%', 'Zone 1 reach as % of the protected line impedance', 80.0, 'num'),
        'Z2 %': ('%', 'Zone 2 reach as % of the protected line impedance', 120.0, 'num'),
        'Z3 %': ('%', 'Zone 3 reach as % of the protected line impedance', 220.0, 'num'),
        'T1 s': ('s', 'Zone 1 time', 0.0, 'num'),
        'T2 s': ('s', 'Zone 2 time', 0.4, 'num'),
        'T3 s': ('s', 'Zone 3 time', 1.0, 'num'),
    },
}
ORDER = list(SCHEMA)
TYPE_OF = {'Substations': 'Substation', 'Buses': 'Bus', 'Lines': 'Line', 'Transformers': 'Transformer', 'Reactors': 'Reactor', 'Breakers': 'Breaker',
           'Grids': 'Grid', 'Converters': 'Converter', 'Loads': 'Load', 'Relays': 'Relay'}
ICON = {'Substations': '🏢', 'Buses': '➖', 'Lines': '🗼', 'Transformers': '🔁', 'Reactors': '🌀', 'Grids': '🌐', 'Converters': '🔀', 'Breakers': '🔲',
        'Loads': '🏭', 'Relays': '🛡️'}

# conductor library: R20 (ohm/km, one sub-conductor), diameter (mm), GMR factor (GMR = k * radius), alpha (1/degC)
CONDUCTORS = pd.DataFrame([
    ['Flint AAAC 740.8 kcmil', 0.0897, 25.17, 0.768, 0.0036],
    ['Condor ACSR 795 kcmil', 0.0718, 27.76, 0.809, 0.0040],
    ['Drake ACSR 795 kcmil', 0.0717, 28.14, 0.809, 0.0040],
    ['Zebra ACSR 428 mm2', 0.0674, 28.62, 0.809, 0.0040],
    ['Moose ACSR 528 mm2', 0.0547, 31.77, 0.809, 0.0040],
], columns=['Conductor', 'R20 ohm/km', 'Diameter mm', 'GMR factor', 'Alpha'])

# tower library: sub-conductors per phase, bundle spacing (m), phase positions (x, y in m) of one circuit, earth resistivity
TOWERS = pd.DataFrame([
    ['400 kV double-circuit, quad bundle', 4, 0.45, -6.5, 30.0, -7.5, 41.0, -6.5, 52.0, 100.0],
    ['400 kV double-circuit, triple bundle', 3, 0.45, -6.5, 30.0, -7.5, 41.0, -6.5, 52.0, 100.0],
    ['400 kV single-circuit horizontal, quad', 4, 0.45, -11.0, 27.0, 0.0, 27.0, 11.0, 27.0, 100.0],
], columns=['Tower', 'Sub-conductors', 'Bundle Spacing m', 'Xa', 'Ya', 'Xb', 'Yb', 'Xc', 'Yc', 'Earth ohm.m'])
CABLE_TYPICAL = dict(r1=0.0125, x1=0.12, c1=230.0, r0=0.06, x0=0.08, c0=230.0)


def line_parameters(cond, tower, temp=75.0, f=F_HZ):
    """Per-km sequence parameters of an overhead line from its conductor and tower (transposed line, earth wires ignored).
    Returns dict r1, x1, c1 (nF/km), r0, x0, c0 and the intermediate values."""
    c = CONDUCTORS.set_index('Conductor').loc[cond]
    t = TOWERS.set_index('Tower').loc[tower]
    n, s = int(t['Sub-conductors']), float(t['Bundle Spacing m'])
    r = c['Diameter mm'] / 2000.0
    gmr = c['GMR factor'] * r
    A = s / (2 * math.sin(math.pi / n)) if n > 1 else 0.0
    gmr_b = (n * gmr * A ** (n - 1)) ** (1 / n) if n > 1 else gmr
    req_b = (n * r * A ** (n - 1)) ** (1 / n) if n > 1 else r
    P = [(t['Xa'], t['Ya']), (t['Xb'], t['Yb']), (t['Xc'], t['Yc'])]
    d = lambda i, j: math.dist(P[i], P[j])  # noqa: E731
    gmd = (d(0, 1) * d(1, 2) * d(0, 2)) ** (1 / 3)
    w = 2 * math.pi * f
    r1 = c['R20 ohm/km'] * (1 + c['Alpha'] * (temp - 20)) / n
    x1 = w * 2e-4 * math.log(gmd / gmr_b)
    c1 = 55.63 / math.log(gmd / req_b)
    De = 658.5 * math.sqrt(t['Earth ohm.m'] / f)
    re = math.pi ** 2 * f * 1e-4
    gmr3 = (gmr_b * gmd ** 2) ** (1 / 3)
    r0 = r1 + 3 * re
    x0 = 3 * w * 2e-4 * math.log(De / gmr3)
    # zero-sequence capacitance with images: H = mean distance to images
    H = [math.hypot(P[i][0] - P[j][0], P[i][1] + P[j][1]) for i in range(3) for j in range(3)]
    Hm = np.prod(H) ** (1 / 9)
    c0 = 55.63 / math.log(Hm ** 3 / (req_b * gmd ** 2))
    return dict(r1=r1, x1=x1, c1=c1, r0=r0, x0=x0, c0=c0, gmd=gmd, gmr_b=gmr_b, req_b=req_b, De=De)


# ------------------------------------------------------------------ data handling
def empty():
    return {k: pd.DataFrame(columns=list(v)) for k, v in SCHEMA.items()}


def id_col(kind):
    return next(f for f, v in SCHEMA[kind].items() if v[3] == 'id')


def clean(data):
    """All tables with all schema columns, numbers as numbers."""
    out = {}
    for k, sch in SCHEMA.items():
        df = data.get(k)
        df = pd.DataFrame(columns=list(sch)) if df is None else df.copy()
        for f in sch:
            if f not in df:
                df[f] = np.nan
        for f, (u, dsc, typ, kind) in sch.items():
            if kind in ('num',):
                df[f] = pd.to_numeric(df[f], errors='coerce')
            else:
                df[f] = df[f].where(df[f].notna(), '').astype(str).str.strip()
                df[f] = df[f].replace({'nan': '', 'None': ''})
        df = df[df[id_col(k)].astype(str).str.len() > 0]
        out[k] = df[list(sch)].reset_index(drop=True)
    return out


def element_table(data, kind, eid):
    """Parameter table of one equipment: Parameter, Value, Unit, Description, Source."""
    df = data[kind]
    row = df[df[id_col(kind)] == eid]
    r = row.iloc[0] if len(row) else pd.Series(dtype=object)
    rows = []
    for f, (u, dsc, typ, kind_) in SCHEMA[kind].items():
        v = r.get(f, np.nan)
        empty_ = v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v == '')
        src = 'entered'
        if empty_:
            src = 'typical value' if typ not in (None, '') else ('computed' if (kind == 'Lines' and f in LINE_COMPUTED) else 'missing')
            if kind == 'Lines' and f in LINE_COMPUTED:
                src = 'computed / typical'
        rows.append({'Parameter': f, 'Value': '' if empty_ else (v if kind_ != 'num' else float(v)), 'Unit': u, 'Description': dsc, 'Source': src})
    return pd.DataFrame(rows)


def set_element(data, kind, eid, table):
    """Write an edited parameter table back (adds the equipment if it is new)."""
    df = data[kind].copy()
    vals = {}
    for _, r in table.iterrows():
        f = r['Parameter']
        if f not in SCHEMA[kind]:
            continue
        v = r['Value']
        if SCHEMA[kind][f][3] == 'num':
            v = pd.to_numeric(v, errors='coerce')
        else:
            v = '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
        vals[f] = v
    vals[id_col(kind)] = vals.get(id_col(kind)) or eid
    idc = id_col(kind)
    if (df[idc] == eid).any():
        i = df.index[df[idc] == eid][0]
        for f, v in vals.items():
            df.at[i, f] = v
    else:
        df = pd.concat([df, pd.DataFrame([vals])], ignore_index=True)
    data = dict(data)
    data[kind] = df
    return clean(data)


def delete_element(data, kind, eid):
    data = dict(data)
    data[kind] = data[kind][data[kind][id_col(kind)] != eid].reset_index(drop=True)
    return data


def element_excel(data, kind, eid):
    """Excel form of one equipment: sheet 'Data' (Parameter / Value / Unit / Description) to fill in and upload."""
    t = element_table(data, kind, eid)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        t[['Parameter', 'Value', 'Unit', 'Description']].to_excel(xw, sheet_name='Data', index=False)
        pd.DataFrame({'Equipment type': [kind], 'Equipment': [eid],
                      'How to fill': ['Write the values in the Value column only, keep the Parameter names, then upload the file '
                                      'on the equipment page. Leave a value empty to use the typical value.']}).to_excel(xw, sheet_name='Info', index=False)
        ws = xw.sheets['Data']
        for col, w in zip('ABCD', (22, 18, 10, 90)):
            ws.column_dimensions[col].width = w
    return buf.getvalue()


def read_element_excel(file):
    """(kind, id, table) from an equipment Excel form."""
    x = pd.read_excel(file, sheet_name=None)
    t = x.get('Data', list(x.values())[0])
    t.columns = [str(c).strip() for c in t.columns]
    info = x.get('Info')
    kind = eid = None
    if info is not None and len(info):
        kind = str(info.iloc[0].get('Equipment type', '')) or None
        eid = str(info.iloc[0].get('Equipment', '')) or None
    return kind, eid, t[['Parameter', 'Value']]


def network_excel(data):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        for k in ORDER:
            data[k].to_excel(xw, sheet_name=k, index=False)
        CONDUCTORS.to_excel(xw, sheet_name='Conductor Library', index=False)
        TOWERS.to_excel(xw, sheet_name='Tower Library', index=False)
        rows = [{'Sheet': k, 'Column': f, 'Unit': u, 'Description': d, 'Typical': '' if t is None else t} for k, s in SCHEMA.items() for f, (u, d, t, _) in s.items()]
        pd.DataFrame(rows).to_excel(xw, sheet_name='Help', index=False)
    return buf.getvalue()


def read_network_excel(file):
    x = pd.read_excel(file, sheet_name=None)
    return clean({k: x[k] for k in ORDER if k in x})


# ------------------------------------------------------------------ completeness and filled values
RELAY_OC_FIELDS = ['Curve', 'TMS', 'Inst Time s']
RELAY_21_FIELDS = ['VT Primary kV', 'VT Secondary V', 'Z1 %', 'Z2 %', 'Z3 %', 'T1 s', 'T2 s', 'T3 s']
LINE_COMPUTED = ['R1 ohm/km', 'X1 ohm/km', 'C1 nF/km', 'R0 ohm/km', 'X0 ohm/km', 'C0 nF/km']


def filled(data):
    """Tables with the empty values filled (typical values or computed line parameters) and a list of the notes."""
    d = clean(data)
    notes = []
    for k, sch in SCHEMA.items():
        df = d[k]
        idc = id_col(k)
        for f, (u, dsc, typ, kind) in sch.items():
            if typ is None or f in LINE_COMPUTED:
                continue
            m = df[f].isna() if kind == 'num' else (df[f] == '')
            if k == 'Relays' and f in RELAY_OC_FIELDS:
                m = m & df['Function'].isin(['51', '51N'])
            if k == 'Relays' and f in RELAY_21_FIELDS:
                m = m & (df['Function'] == '21')
            if m.any() and typ != '':
                for e in df.loc[m, idc]:
                    notes.append((k, e, f, f'typical value {typ} {u}'.strip()))
                df.loc[m, f] = typ
        if k == 'Relays':
            for f in RELAY_OC_FIELDS + RELAY_21_FIELDS:
                typ = sch[f][2]
                if sch[f][3] == 'num':
                    df[f] = df[f].fillna(typ)
                else:
                    df[f] = df[f].replace('', typ)
        d[k] = df
    L = d['Lines']
    for i, r in L.iterrows():
        miss = [f for f in LINE_COMPUTED if pd.isna(r[f])]
        if not miss:
            continue
        p = None
        how = ''
        if r['Type'].lower().startswith('cab'):
            p = dict(CABLE_TYPICAL); how = 'typical 400 kV cable values'
        elif r['Conductor'] in set(CONDUCTORS['Conductor']) and r['Tower'] in set(TOWERS['Tower']):
            p = line_parameters(r['Conductor'], r['Tower']); how = f"computed from {r['Conductor']} on {r['Tower']}"
        else:
            p = line_parameters('Flint AAAC 740.8 kcmil', '400 kV double-circuit, quad bundle'); how = 'typical 400 kV quad-bundle line values'
        for f, key in zip(LINE_COMPUTED, ['r1', 'x1', 'c1', 'r0', 'x0', 'c0']):
            if pd.isna(r[f]):
                L.at[i, f] = round(p[key], 5)
        notes.append(('Lines', r['Line'], ', '.join(m.split()[0] for m in miss), how))
    d['Lines'] = L
    # zero-sequence of a non-earthed winding is handled by the vector group
    return d, notes


def missing_required(data):
    """Required values that are empty (no typical value exists)."""
    d = clean(data)
    out = []
    for k, sch in SCHEMA.items():
        idc = id_col(k)
        for f, (u, dsc, typ, kind) in sch.items():
            if typ is not None or kind == 'id' or f in LINE_COMPUTED:
                continue
            if k == 'Relays' and f == 'Pickup A':
                m = d[k]['Function'].isin(['51', '51N']) & d[k][f].isna()
            elif k == 'Grids' and f in ('Sk max MVA', 'Sk min MVA'):
                m = d[k][f].isna()
            else:
                m = d[k][f].isna() if kind == 'num' else (d[k][f] == '')
            for e in d[k].loc[m, idc]:
                out.append((k, e, f))
    return out


def validate_links(data):
    """Wrong references (bus names that do not exist and so on)."""
    d = clean(data)
    buses = set(d['Buses']['Bus'])
    subs = set(d['Substations']['Substation'])
    err = []
    for _, r in d['Buses'].iterrows():
        if r['Substation'] not in subs:
            err.append(f"Bus {r['Bus']}: substation '{r['Substation']}' not found")
    for k, cols in [('Lines', ['From Bus', 'To Bus']), ('Transformers', ['HV Bus', 'LV Bus']), ('Reactors', ['Bus']), ('Grids', ['Bus']),
                    ('Converters', ['Bus']), ('Loads', ['Bus']), ('Relays', ['Bus']), ('Breakers', ['Bus'])]:
        for _, r in d[k].iterrows():
            for c in cols:
                if r[c] not in buses:
                    err.append(f"{TYPE_OF[k]} {r[id_col(k)]}: {c} '{r[c]}' not found")
    elems = set(d['Lines']['Line']) | set(d['Transformers']['Transformer'])
    for _, r in d['Relays'].iterrows():
        if r['Element'] not in elems:
            err.append(f"Relay {r['Relay']}: element '{r['Element']}' not found")
    if (d['Grids']['Slack'] == 1).sum() != 1 and len(d['Grids']):
        err.append('Exactly one grid must be the slack (reference) grid')
    return err


# ------------------------------------------------------------------ pandapower model
def build(data, mode='pf', case='max', outages=(), load_scale=1.0, reactors_off=()):
    """pandapower network. mode 'pf' = load flow, 'sc' = short circuit (all grids as sources, no loads, no capacitances)."""
    d, _ = filled(data)
    net = pp.create_empty_network(f_hz=F_HZ, sn_mva=100)
    B = {}
    for _, r in d['Buses'].iterrows():
        B[r['Bus']] = pp.create_bus(net, vn_kv=float(r['kV']), name=r['Bus'], zone=r['Substation'],
                                    min_vm_pu=float(r['Vmin pu']), max_vm_pu=float(r['Vmax pu']))
    for _, r in d['Lines'].iterrows():
        if r['From Bus'] not in B or r['To Bus'] not in B:
            continue
        ins = bool(r['In Service']) and r['Line'] not in outages
        c1, c0 = (float(r['C1 nF/km']), float(r['C0 nF/km'])) if mode == 'pf' else (0.0, 0.0)
        pp.create_line_from_parameters(net, B[r['From Bus']], B[r['To Bus']], float(r['Length km']), float(r['R1 ohm/km']), float(r['X1 ohm/km']),
                                       c1, float(r['Rating A']) / 1000.0, name=r['Line'], in_service=ins,
                                       r0_ohm_per_km=float(r['R0 ohm/km']), x0_ohm_per_km=float(r['X0 ohm/km']), c0_nf_per_km=c0,
                                       endtemp_degree=80.0, type='cs' if r['Type'].lower().startswith('cab') else 'ol')
    for _, r in d['Transformers'].iterrows():
        if r['HV Bus'] not in B or r['LV Bus'] not in B:
            continue
        ins = bool(r['In Service']) and r['Transformer'] not in outages
        vg = r['Vector Group'] or 'YNyn0'
        vg_pp = vg if vg in ('Dyn', 'Yyn', 'YNd', 'YNyn', 'Yzn', 'Dd', 'Dy', 'Yd', 'Yy', 'YNy', 'YyN', 'YNdy', 'YNynd') else \
            ''.join(ch for ch in vg if not ch.isdigit()) or 'YNyn'
        if vg_pp not in ('Dyn', 'Yyn', 'YNd', 'YNyn', 'Yzn', 'Dd', 'Dy', 'Yd', 'Yy', 'YNy', 'YyN'):
            vg_pp = 'YNyn'
        shift = 0.0
        digits = ''.join(ch for ch in vg if ch.isdigit())
        if digits:
            shift = int(digits) * 30.0
        pp.create_transformer_from_parameters(
            net, B[r['HV Bus']], B[r['LV Bus']], sn_mva=float(r['MVA']), vn_hv_kv=float(r['HV kV']), vn_lv_kv=float(r['LV kV']),
            vkr_percent=float(r['ur %']), vk_percent=float(r['uk %']), pfe_kw=0.0, i0_percent=0.0, shift_degree=shift, name=r['Transformer'],
            in_service=ins, tap_side='hv', tap_neutral=0, tap_min=-20, tap_max=20, tap_step_percent=1.0,
            tap_pos=int(round(float(r['Tap %']))), vector_group=vg_pp, vk0_percent=float(r['uk %']), vkr0_percent=float(r['ur %']),
            mag0_percent=100.0, mag0_rx=0.0, si0_hv_partial=0.9)
    if mode == 'pf':
        for _, r in d['Reactors'].iterrows():
            if r['Bus'] in B:
                pp.create_shunt(net, B[r['Bus']], q_mvar=float(r['MVAr'] or 0), p_mw=0.0, vn_kv=float(r['kV']), name=r['Reactor'],
                                in_service=bool(r['In Service']) and r['Reactor'] not in reactors_off and r['Line'] not in outages)
        for _, r in d['Loads'].iterrows():
            if r['Bus'] in B:
                pp.create_load(net, B[r['Bus']], p_mw=float(r['P MW']) * load_scale, q_mvar=float(r['Q MVAr']) * load_scale, name=r['Load'],
                               in_service=bool(r['In Service']))
        for _, r in d['Converters'].iterrows():
            if r['Bus'] in B:
                pp.create_sgen(net, B[r['Bus']], p_mw=float(r['P MW']), q_mvar=float(r['Q MVAr']), name=r['Converter'],
                               in_service=bool(r['In Service']) and r['Converter'] not in outages)
        for _, r in d['Grids'].iterrows():
            if r['Bus'] not in B:
                continue
            if int(r['Slack'] or 0) == 1:
                pp.create_ext_grid(net, B[r['Bus']], vm_pu=float(r['V Setpoint pu']), name=r['Grid'])
            else:
                pp.create_gen(net, B[r['Bus']], p_mw=float(r['P Import MW']), vm_pu=float(r['V Setpoint pu']), name=r['Grid'],
                              in_service=r['Grid'] not in outages)
    else:
        for _, r in d['Grids'].iterrows():
            if r['Bus'] not in B or r['Grid'] in outages:
                continue
            skmax = float(r['Sk max MVA']) if not pd.isna(r['Sk max MVA']) else 10000.0
            if skmax <= 0:                                   # load-flow reference only, no short-circuit source
                continue
            skmin = float(r['Sk min MVA']) if not pd.isna(r['Sk min MVA']) else 0.6 * skmax
            pp.create_ext_grid(net, B[r['Bus']], vm_pu=1.0, name=r['Grid'], s_sc_max_mva=skmax, s_sc_min_mva=skmin,
                               rx_max=float(r['R/X']), rx_min=float(r['R/X']), x0x_max=float(r['X0/X1']), x0x_min=float(r['X0/X1']),
                               r0x0_max=float(r['R0/X0']), r0x0_min=float(r['R0/X0']))
    return net


def _bus_of(net):
    return dict(zip(net.bus.index, net.bus.name))


# ------------------------------------------------------------------ load flow and voltage drop
def load_flow(data, load_scale=1.0, outages=(), reactors_off=(), warn_loading=80.0):
    net = build(data, 'pf', outages=outages, load_scale=load_scale, reactors_off=reactors_off)
    pp.runpp(net, algorithm='nr', calculate_voltage_angles=True, init='dc', max_iteration=50, numba=False)
    bn = _bus_of(net)
    bus = pd.DataFrame({'Bus': net.bus.name, 'Substation': net.bus.zone, 'kV': net.bus.vn_kv, 'V (pu)': net.res_bus.vm_pu.round(4),
                        'V (kV)': (net.res_bus.vm_pu * net.bus.vn_kv).round(1), 'Angle (deg)': net.res_bus.va_degree.round(2),
                        'Vmin': net.bus.min_vm_pu, 'Vmax': net.bus.max_vm_pu})
    bus['Status'] = np.where(bus['V (pu)'].isna(), 'ACTION', np.where((bus['V (pu)'] < bus['Vmin']) | (bus['V (pu)'] > bus['Vmax']), 'ACTION',
                             np.where((bus['V (pu)'] < bus['Vmin'] + 0.01) | (bus['V (pu)'] > bus['Vmax'] - 0.01), 'WATCH', 'EXCELLENT')))
    rl = net.res_line
    line = pd.DataFrame({'Line': net.line.name, 'From': net.line.from_bus.map(bn), 'To': net.line.to_bus.map(bn), 'Length (km)': net.line.length_km,
                         'P From (MW)': rl.p_from_mw.round(1), 'Q From (MVAr)': rl.q_from_mvar.round(1), 'P To (MW)': rl.p_to_mw.round(1),
                         'Q To (MVAr)': rl.q_to_mvar.round(1), 'I (A)': (rl.i_ka * 1000).round(0), 'Rating (A)': (net.line.max_i_ka * 1000).round(0),
                         'Loading (%)': rl.loading_percent.round(1), 'Losses (MW)': rl.pl_mw.round(2),
                         'V From (pu)': rl.vm_from_pu.round(4), 'V To (pu)': rl.vm_to_pu.round(4), 'In Service': net.line.in_service})
    line['Voltage Drop (%)'] = ((line['V From (pu)'] - line['V To (pu)']) * 100).round(2)
    line['Status'] = np.where(~line['In Service'], 'OUT', np.where(line['Loading (%)'] > 100, 'ACTION', np.where(line['Loading (%)'] > warn_loading, 'WATCH', 'EXCELLENT')))
    rt = net.res_trafo
    trafo = pd.DataFrame({'Transformer': net.trafo.name, 'HV Bus': net.trafo.hv_bus.map(bn), 'LV Bus': net.trafo.lv_bus.map(bn),
                          'P HV (MW)': rt.p_hv_mw.round(1), 'Q HV (MVAr)': rt.q_hv_mvar.round(1), 'Loading (%)': rt.loading_percent.round(1),
                          'Losses (MW)': rt.pl_mw.round(2), 'In Service': net.trafo.in_service})
    trafo['Status'] = np.where(~trafo['In Service'], 'OUT', np.where(trafo['Loading (%)'] > 100, 'ACTION', np.where(trafo['Loading (%)'] > warn_loading, 'WATCH', 'EXCELLENT')))
    src = []
    for _, r in net.ext_grid.iterrows():
        src.append({'Source': r['name'], 'Type': 'Grid (slack)', 'Bus': bn[r.bus], 'P (MW)': round(net.res_ext_grid.at[_, 'p_mw'], 1),
                    'Q (MVAr)': round(net.res_ext_grid.at[_, 'q_mvar'], 1)})
    for i, r in net.gen.iterrows():
        src.append({'Source': r['name'], 'Type': 'Grid', 'Bus': bn[r.bus], 'P (MW)': round(net.res_gen.at[i, 'p_mw'], 1), 'Q (MVAr)': round(net.res_gen.at[i, 'q_mvar'], 1)})
    for i, r in net.sgen.iterrows():
        src.append({'Source': r['name'], 'Type': 'Converter', 'Bus': bn[r.bus], 'P (MW)': round(net.res_sgen.at[i, 'p_mw'], 1), 'Q (MVAr)': round(net.res_sgen.at[i, 'q_mvar'], 1)})
    for i, r in net.shunt.iterrows():
        src.append({'Source': r['name'], 'Type': 'Reactor', 'Bus': bn[r.bus], 'P (MW)': 0.0, 'Q (MVAr)': round(-net.res_shunt.at[i, 'q_mvar'], 1)})
    sources = pd.DataFrame(src)
    losses = float(rl.pl_mw.sum() + rt.pl_mw.sum())
    gen_mw = float(sources.loc[sources['Type'] != 'Reactor', 'P (MW)'].clip(lower=0).sum()) if len(sources) else 0.0
    charging = float(rl.ql_mvar.clip(upper=0).sum()) * -1
    summary = {'Losses (MW)': round(losses, 2), 'Losses (%)': round(100 * losses / gen_mw, 2) if gen_mw else 0.0,
               'Min V (pu)': round(bus['V (pu)'].min(), 4), 'Max V (pu)': round(bus['V (pu)'].max(), 4),
               'Max Line Loading (%)': round(line['Loading (%)'].max(), 1) if len(line) else 0.0,
               'Max Transformer Loading (%)': round(trafo['Loading (%)'].max(), 1) if len(trafo) else 0.0,
               'Line Charging (MVAr)': round(charging, 0),
               'Reactors (MVAr)': round(float(sources.loc[sources['Type'] == 'Reactor', 'Q (MVAr)'].sum()) * -1, 0) if len(sources) else 0.0,
               'Violations': int((bus['Status'] == 'ACTION').sum() + (line['Status'] == 'ACTION').sum() + (trafo['Status'] == 'ACTION').sum()),
               'Warnings': int((bus['Status'] == 'WATCH').sum() + (line['Status'] == 'WATCH').sum() + (trafo['Status'] == 'WATCH').sum())}
    return {'bus': bus, 'line': line, 'trafo': trafo, 'sources': sources, 'summary': summary, 'converged': bool(net.converged),
            'settings': {'load_scale': load_scale, 'outages': list(outages), 'reactors_off': list(reactors_off)}}


def voltage_drop(lf, limit_pct=3.0):
    """Voltage drop along every circuit and the bus voltages against their limits."""
    L = lf['line'][lf['line']['In Service']].copy()
    L['|Drop| (%)'] = L['Voltage Drop (%)'].abs()
    L['Status'] = np.where(L['|Drop| (%)'] > limit_pct, 'WATCH', 'EXCELLENT')
    L = L[['Line', 'From', 'To', 'Length (km)', 'V From (pu)', 'V To (pu)', 'Voltage Drop (%)', '|Drop| (%)', 'Loading (%)', 'Status']]
    B = lf['bus'].copy()
    B['Margin to Vmin (%)'] = ((B['V (pu)'] - B['Vmin']) * 100).round(2)
    B['Margin to Vmax (%)'] = ((B['Vmax'] - B['V (pu)']) * 100).round(2)
    rise = L[L['Voltage Drop (%)'] < 0]
    notes = []
    if len(rise):
        notes.append(f"{len(rise)} circuit(s) show a voltage rise toward the receiving end (Ferranti effect of the line charging, "
                     "typical for long lightly loaded EHV lines and cables).")
    hi = B[B['V (pu)'] > B['Vmax']]
    if len(hi):
        notes.append('High voltage at ' + ', '.join(hi['Bus']) + ': switch in shunt reactors or lower the generator / grid voltage set points.')
    lo = B[B['V (pu)'] < B['Vmin']]
    if len(lo):
        notes.append('Low voltage at ' + ', '.join(lo['Bus']) + ': switch out shunt reactors, raise the voltage set points or add reactive support.')
    return {'line': L.sort_values('|Drop| (%)', ascending=False), 'bus': B, 'notes': notes}


# ------------------------------------------------------------------ short circuit (IEC 60909)
def short_circuit(data, case='max', fault_bus=None):
    """IEC 60909 three-phase and single-phase fault at every bus and the breaker duty against the substation ratings."""
    d, _ = filled(data)
    net = build(data, 'sc')
    sc.calc_sc(net, case=case, fault='3ph', ip=True, ith=True, tk_s=1.0)
    r3 = net.res_bus_sc.copy()
    sc.calc_sc(net, case=case, fault='1ph')
    r1 = net.res_bus_sc.copy()
    subs = d['Substations'].set_index('Substation')
    rows = []
    for i, b in net.bus.iterrows():
        bb = d['Buses'].set_index('Bus')['Breaking kA'].get(b['name'], np.nan)
        own = not pd.isna(bb) and bb != ''
        brk = float(bb) if own else float(subs['Breaking kA'].get(b.zone, 40.0))
        stw = brk if own else float(subs['Short-time kA (1 s)'].get(b.zone, brk))
        ik3, ik1 = r3.at[i, 'ikss_ka'], r1.at[i, 'ikss_ka']
        ikmax = np.nanmax([ik3, ik1])
        rk, xk = r3.at[i, 'rk_ohm'], r3.at[i, 'xk_ohm']
        duty = 100 * ikmax / brk if brk else np.nan
        ip = r3.at[i, 'ip_ka']
        making = 2.5 * brk
        rows.append({'Bus': b['name'], 'Substation': b.zone, 'kV': b.vn_kv, 'Ik3 (kA)': round(ik3, 2), 'Ik1 (kA)': round(ik1, 2),
                     'Sk3 (MVA)': round(math.sqrt(3) * b.vn_kv * ik3, 0), 'ip (kA)': round(ip, 2), 'Ith 1 s (kA)': round(r3.at[i, 'ith_ka'], 2),
                     'X/R': round(xk / rk, 1) if rk else np.nan, 'Zk (ohm)': round(math.hypot(rk, xk), 2),
                     'Breaking (kA)': brk, 'Breaker Duty (%)': round(duty, 1), 'Making Duty (%)': round(100 * ip / making, 1),
                     'Short-time Duty (%)': round(100 * r3.at[i, 'ith_ka'] / stw, 1) if stw else np.nan})
    T = pd.DataFrame(rows)
    T['Status'] = np.where(T[['Breaker Duty (%)', 'Making Duty (%)', 'Short-time Duty (%)']].max(axis=1) > 100, 'ACTION',
                           np.where(T[['Breaker Duty (%)', 'Making Duty (%)', 'Short-time Duty (%)']].max(axis=1) > 90, 'WATCH', 'EXCELLENT'))
    out = {'bus': T, 'case': case}
    Bk = d['Breakers']
    if len(Bk):
        ik = T.set_index('Bus')
        rows = []
        for _, b in Bk.iterrows():
            if b['Bus'] not in ik.index:
                continue
            r = ik.loc[b['Bus']]
            I = max(r['Ik3 (kA)'], r['Ik1 (kA)'])
            duty = 100 * I / float(b['Breaking kA']) if b['Breaking kA'] else np.nan
            rows.append({'Breaker': b['Breaker'], 'Bus': b['Bus'], 'Element': b['Element'], 'Fault Current (kA)': round(I, 2), 'Breaking (kA)': b['Breaking kA'],
                         'Duty (%)': round(duty, 1), 'ip (kA)': r['ip (kA)'], 'Making Duty (%)': round(100 * r['ip (kA)'] / (2.5 * float(b['Breaking kA'])), 1),
                         'Status': 'ACTION' if max(duty, 100 * r['ip (kA)'] / (2.5 * float(b['Breaking kA']))) > 100 else
                         ('WATCH' if max(duty, 100 * r['ip (kA)'] / (2.5 * float(b['Breaking kA']))) > 90 else 'EXCELLENT')})
        out['breakers'] = pd.DataFrame(rows).sort_values('Duty (%)', ascending=False, ignore_index=True) if rows else pd.DataFrame()
    if fault_bus:
        out['contrib'] = fault_contributions(data, fault_bus, case)
    return out


def fault_contributions(data, bus, case='max', fault='3ph', net=None):
    """Fault at one bus: total current and the current in every branch (from-end and to-end)."""
    net = net or build(data, 'sc')
    bi = net.bus.index[net.bus.name == bus][0]
    sc.calc_sc(net, case=case, fault=fault, bus=bi, branch_results=True)
    bn = _bus_of(net)
    ik = float(net.res_bus_sc.at[bi, 'ikss_ka'])
    rows = []
    for i, l in net.line.iterrows():
        if not l.in_service:
            continue
        rows.append({'Branch': l['name'], 'Type': 'Line', 'From': bn[l.from_bus], 'To': bn[l.to_bus],
                     'I From (kA)': round(float(net.res_line_sc.at[i, 'ikss_from_ka']), 3), 'I To (kA)': round(float(net.res_line_sc.at[i, 'ikss_to_ka']), 3)})
    for i, t in net.trafo.iterrows():
        if not t.in_service:
            continue
        r = net.res_trafo_sc.loc[i] if hasattr(net, 'res_trafo_sc') and i in net.res_trafo_sc.index else None
        rows.append({'Branch': t['name'], 'Type': 'Transformer', 'From': bn[t.hv_bus], 'To': bn[t.lv_bus],
                     'I From (kA)': round(float(r['ikss_hv_ka']), 3) if r is not None else np.nan, 'I To (kA)': round(float(r['ikss_lv_ka']), 3) if r is not None else np.nan})
    grid = []
    return {'bus': bus, 'fault': fault, 'Ik (kA)': round(ik, 3), 'branches': pd.DataFrame(rows), 'grids': grid}


# ------------------------------------------------------------------ grid short-circuit power from disturbance records
# During a fault the network behind a relay is passive for the fault-caused (superimposed) quantities, so at the relay bus
#   V2 = -Zs2 * I2,  V0 = -Zs0 * I0,  dV1 = -Zs1 * dI1   (I measured from the bus into the protected line)
# Zs is the Thevenin impedance of everything behind the relay (all grids and all other circuits, the faulted circuit excluded).
# Sk = U^2 / |Zs| (IEC 60909-0 equivalent source). The model gives the same impedance with the faulted circuit out of service,
# so the grid short-circuit powers can be adjusted until the model matches the records.
_A = np.exp(2j * np.pi / 3)


def _seq3(xa, xb, xc):
    return (xa + xb + xc) / 3, (xa + _A * xb + _A * _A * xc) / 3, (xa + _A * _A * xb + _A * xc) / 3


def record_source(rec, scale='Auto', ct=1.0, vt=1.0):
    """Source impedance behind the relay from one COMTRADE record (faultguard_core record dict).
    Returns station, device, kV, Z1/Z0 (complex ohm, primary), Sk (MVA), X/R, X0/X1, R0/X0 and the way it was found."""
    import faultguard_core as fg
    A = fg.scale_primary(rec, scale, ct, vt)
    ch = fg.map_channels(rec)
    if any(ch.get(k) is None for k in ('IA', 'IB', 'IC', 'VA', 'VB', 'VC')):
        raise ValueError('the record has no three phase currents and voltages')
    ev = fg.detect(rec, A, ch)
    if ev.get('inception') is None:
        raise ValueError('no fault found in the record')
    P = fg.fault_phasors(A, ch, ev, ev['N'])
    Vp = _seq3(*[P['V' + p][0] for p in 'ABC']); Vf = _seq3(*[P['V' + p][1] for p in 'ABC'])
    Ip = _seq3(*[P['I' + p][0] for p in 'ABC']); If = _seq3(*[P['I' + p][1] for p in 'ABC'])
    kv = abs(Vp[1]) * math.sqrt(3) / 1000.0
    notes = []
    dI1 = If[1] - Ip[1]
    if abs(If[2]) > 0.05 * abs(dI1) and abs(If[2]) > 20:          # unbalanced fault: negative sequence (no load in it)
        z1, how = -Vf[2] / If[2], 'negative sequence (V2 / I2)'
    elif abs(dI1) > 20:
        z1, how = -(Vf[1] - Vp[1]) / dI1, 'positive sequence change (dV1 / dI1)'
    else:
        raise ValueError('the fault current change is too small to measure the source')
    flip = z1.imag < 0                                           # current polarity towards the bus
    if flip:
        z1 = -z1; notes.append('current polarity is towards the bus; sign corrected')
    z0 = None
    if abs(If[0]) > 0.05 * abs(dI1) and abs(If[0]) > 20:
        z0 = -Vf[0] / If[0] * (-1 if flip else 1)
        if z0.imag <= 0:
            z0 = None; notes.append('zero sequence not usable')
    sk = kv ** 2 / abs(z1) if kv > 1 else np.nan
    return {'station': rec.get('station') or '', 'device': rec.get('device') or '', 'kV': round(kv, 1), 'Z1': z1, 'Z0': z0,
            'Sk MVA': round(sk, 0), 'X/R': round(z1.imag / z1.real, 1) if z1.real > 0 else np.nan,
            'X0/X1': round(z0.imag / z1.imag, 2) if z0 is not None else np.nan,
            'R0/X0': round(z0.real / z0.imag, 2) if z0 is not None else np.nan, 'method': how, 'notes': notes}


def bus_for_station(data, station, kv=None):
    """Bus of the substation most like a recorder's station name (nearest voltage when the substation has several)."""
    B = data['Buses']
    if not len(B):
        return None
    s = [max(station_sim(station, r['Substation']), station_sim(station, r['Bus'])) for _, r in B.iterrows()]
    best = max(s)
    if best < 0.6:
        return None
    cand = B[[x >= best - 1e-9 for x in s]]
    if kv and len(cand) > 1:
        cand = cand.iloc[(pd.to_numeric(cand['kV'], errors='coerce') - kv).abs().argsort()]
    return cand.iloc[0]['Bus']


def model_source(data, bus, line=None, case='max'):
    """Thevenin impedance (complex ohm) of the model at a bus with the faulted circuit out: Z1 and Z0."""
    net = build(data, 'sc', outages=tuple([line]) if line else ())
    bi = net.bus.index[net.bus.name == bus][0]
    sc.calc_sc(net, case=case, fault='1ph', bus=bi)
    r = net.res_bus_sc.loc[bi]
    return complex(r['rk_ohm'], r['xk_ohm']), complex(r['rk0_ohm'], r['xk0_ohm'])


def _copy(data):
    return {k: v.copy() for k, v in data.items()}


def _scaled(data, grids, k1, k0=1.0):
    d = _copy(data)
    G = d['Grids']
    m = G['Grid'].isin(grids)
    for c in ('Sk max MVA', 'Sk min MVA'):
        G.loc[m, c] = pd.to_numeric(G.loc[m, c], errors='coerce') * k1
    if k0 != 1.0:
        x0 = pd.to_numeric(G.loc[m, 'X0/X1'], errors='coerce').fillna(1.0)
        G.loc[m, 'X0/X1'] = x0 * k0
    return d


def fit_grids(data, meas, grids=None, each=None):
    """Adjust the short-circuit power and the X0/X1 of the chosen grids so that the model's source impedances match the
    measured ones (meas: list of {'bus', 'line', 'Z1', 'Z0'}). One measuring point fixes one value, so all the grids get one
    common factor; records from several buses or faulted circuits give each grid its own factor (least squares on the
    logarithms, held towards the common factor where the records cannot tell the grids apart).
    Returns the new data, the comparison table per record and {'grids', 'mode', 'table' per grid}."""
    from scipy.optimize import least_squares
    d0, _ = filled(data)
    grids = list(grids) if grids else [g for g in d0['Grids']['Grid'] if not str(g).startswith(REF_PREFIX)]
    data = _copy(data)
    G = data['Grids']
    G.loc[G['Sk max MVA'].isna(), 'Sk max MVA'] = 10000.0                     # start from the values the model uses
    G.loc[G['Sk min MVA'].isna(), 'Sk min MVA'] = 0.6 * pd.to_numeric(G['Sk max MVA'])
    G.loc[G['X0/X1'].isna(), 'X0/X1'] = 1.0
    meas = [m for m in meas if m.get('bus')]
    setups = sorted({(m['bus'], m.get('line') or '') for m in meas})
    each = (len(setups) > 1 and len(grids) > 1) if each is None else each
    nets = {}
    for b, ln in setups:
        net = build(data, 'sc', outages=(ln,) if ln else ())
        nets[(b, ln)] = (net, net.bus.index[net.bus.name == b][0], net.ext_grid.copy())

    def zs(key, k1, k0):
        net, bi, base = nets[key]
        for i, r in base.iterrows():
            a, c = k1.get(r['name'], 1.0), k0.get(r['name'], 1.0)
            net.ext_grid.at[i, 's_sc_max_mva'] = r['s_sc_max_mva'] * a
            net.ext_grid.at[i, 's_sc_min_mva'] = r['s_sc_min_mva'] * a
            net.ext_grid.at[i, 'x0x_max'] = r['x0x_max'] * c
            net.ext_grid.at[i, 'x0x_min'] = r['x0x_min'] * c
        sc.calc_sc(net, case='max', fault='1ph', bus=bi)
        r = net.res_bus_sc.loc[bi]
        return complex(r['rk_ohm'], r['xk_ohm']), complex(r['rk0_ohm'], r['xk0_ohm'])

    def factors(x):
        return dict(zip(grids, np.exp(x))) if each else {g: float(np.exp(x[0])) for g in grids}

    n = len(grids) if each else 1
    lam = 0.15

    def res1(x):
        k = factors(x)
        r = [math.log(abs(zs((m['bus'], m.get('line') or ''), k, {})[0])) - math.log(abs(m['Z1'])) for m in meas]
        return np.array(r + ([lam * (v - x.mean()) for v in x] if each else []))

    x1 = least_squares(res1, np.zeros(n), bounds=(-4, 4), diff_step=1e-3).x
    k1 = factors(x1)
    m0 = [m for m in meas if m.get('Z0') is not None]
    k0 = {}
    if m0:
        def res0(x):
            k = factors(x)
            r = [math.log(abs(zs((m['bus'], m.get('line') or ''), k1, k)[1])) - math.log(abs(m['Z0'])) for m in m0]
            return np.array(r + ([lam * (v - x.mean()) for v in x] if each else []))
        k0 = factors(least_squares(res0, np.zeros(n), bounds=(-3, 3), diff_step=1e-3).x)
    new = _copy(data)
    for g in grids:
        mk = new['Grids']['Grid'] == g
        for c in ('Sk max MVA', 'Sk min MVA'):
            new['Grids'].loc[mk, c] = pd.to_numeric(new['Grids'].loc[mk, c]) * k1[g]
        new['Grids'].loc[mk, 'X0/X1'] = pd.to_numeric(new['Grids'].loc[mk, 'X0/X1']) * k0.get(g, 1.0)
    rows = []
    for m in meas:
        key = (m['bus'], m.get('line') or '')
        z1o, z0o = zs(key, {}, {})
        z1n, z0n = zs(key, k1, k0)
        kv = float(data['Buses'].set_index('Bus')['kV'].get(m['bus'], 0) or 0)
        rows.append({'Record': m.get('name', ''), 'Bus': m['bus'], 'Faulted circuit': m.get('line') or '',
                     'Measured Sk (MVA)': round(kv ** 2 / abs(m['Z1'])), 'Model Sk before (MVA)': round(kv ** 2 / abs(z1o)),
                     'Model Sk after (MVA)': round(kv ** 2 / abs(z1n)),
                     'Measured Z0 (ohm)': round(abs(m['Z0']), 1) if m.get('Z0') is not None else np.nan,
                     'Model Z0 before (ohm)': round(abs(z0o), 1), 'Model Z0 after (ohm)': round(abs(z0n), 1)})
    Go = data['Grids'].set_index('Grid'); Gn = new['Grids'].set_index('Grid')
    gt = pd.DataFrame([{'Grid': g, 'Bus': Go.at[g, 'Bus'], 'Sk max before (MVA)': round(float(Go.at[g, 'Sk max MVA'])),
                        'Sk max after (MVA)': round(float(Gn.at[g, 'Sk max MVA'])), 'Factor': round(k1[g], 3),
                        'X0/X1 before': round(float(Go.at[g, 'X0/X1']), 2), 'X0/X1 after': round(float(Gn.at[g, 'X0/X1']), 2)} for g in grids])
    return new, pd.DataFrame(rows), {'grids': grids, 'mode': 'each' if each else 'common', 'table': gt}


REF_PREFIX = 'Reference '


def set_slack(data, name):
    """Make a grid, or any bus, the reference (slack) of the load flow. A bus without a grid gets a balancing source that
    exists only in the load flow (no short-circuit contribution)."""
    d = _copy(data)
    G = d['Grids'][~d['Grids']['Grid'].astype(str).str.startswith(REF_PREFIX)].reset_index(drop=True)
    if name in set(G['Grid']):
        G['Slack'] = (G['Grid'] == name).astype(int)
    else:
        G['Slack'] = 0
        row = {c: np.nan for c in G.columns}
        row.update({'Grid': REF_PREFIX + str(name), 'Bus': name, 'Slack': 1, 'P Import MW': 0.0, 'V Setpoint pu': 1.0,
                    'Sk max MVA': 0.0, 'Sk min MVA': 0.0})
        G = pd.concat([G, pd.DataFrame([row])], ignore_index=True)
    d['Grids'] = G
    return d


# ------------------------------------------------------------------ protection
CURVES = {'IEC SI': (0.14, 0.02, 0.0), 'IEC VI': (13.5, 1.0, 0.0), 'IEC EI': (80.0, 2.0, 0.0), 'IEC LTI': (120.0, 1.0, 0.0),
          'IEEE MI': (0.0515, 0.02, 0.114), 'IEEE VI': (19.61, 2.0, 0.491), 'IEEE EI': (28.2, 2.0, 0.1217)}


def oc_time(I, pickup, tms, curve='IEC SI', inst=None, inst_t=0.05):
    """Operating time (s) of an inverse-time overcurrent relay (IEC 60255-151 / IEEE C37.112); inf below pickup."""
    I = np.asarray(I, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        M = I / pickup
        if curve == 'DT':
            t = np.where(M > 1, tms, np.inf)
        else:
            k, a, b = CURVES.get(curve, CURVES['IEC SI'])
            t = np.where(M > 1, tms * (k / (M ** a - 1) + b), np.inf)
    if inst is not None and not (isinstance(inst, float) and np.isnan(inst)):
        t = np.where(I >= inst, np.minimum(t, inst_t), t)
    return t


def _line_z(d, line):
    L = d['Lines'].set_index('Line')
    r = L.loc[line]
    return complex(r['R1 ohm/km'], r['X1 ohm/km']) * r['Length km']


def protection(data, cti=CTI):
    """Overcurrent grading (primary / backup pairs for faults at every bus) and distance zone checks."""
    d, _ = filled(data)
    R = d['Relays']
    net = build(data, 'sc')
    L = d['Lines']
    ends = {r['Line']: (r['From Bus'], r['To Bus']) for _, r in L.iterrows()}
    ends.update({r['Transformer']: (r['HV Bus'], r['LV Bus']) for _, r in d['Transformers'].iterrows()})
    # ---- overcurrent coordination
    oc = R[R['Function'].isin(['51', '51N'])]
    pairs = []
    cache = {}
    for fb in d['Buses']['Bus']:
        if fb not in set(net.bus.name):
            continue
        for fault in ('3ph', '1ph'):
            key = (fb, fault)
            try:
                cache[key] = fault_contributions(data, fb, 'max', fault, net=build(data, 'sc'))
            except Exception:
                continue
    for _, bk in oc.iterrows():
        e = bk['Element']
        if e not in ends:
            continue
        a, b = ends[e]
        far = b if bk['Bus'] == a else a
        fault = '1ph' if bk['Function'] == '51N' else '3ph'
        fc = cache.get((far, fault))
        if fc is None:
            continue
        br = fc['branches'].set_index('Branch')
        ib = float(br.at[e, 'I From (kA)'] if bk['Bus'] == br.at[e, 'From'] else br.at[e, 'I To (kA)']) * 1000 if e in br.index else 0.0
        for _, pr in oc[(oc['Bus'] == far) & (oc['Element'] != e) & (oc['Function'] == bk['Function'])].iterrows():
            pe = pr['Element']
            ip_ = fc['Ik (kA)'] * 1000 - (float(br.at[pe, 'I From (kA)'] if far == br.at[pe, 'From'] else br.at[pe, 'I To (kA)']) * 1000 if pe in br.index else 0.0)
            tp = float(oc_time(ip_, pr['Pickup A'], pr['TMS'], pr['Curve'], pr['Inst Pickup A'], pr['Inst Time s']))
            tb = float(oc_time(ib, bk['Pickup A'], bk['TMS'], bk['Curve'], bk['Inst Pickup A'], bk['Inst Time s']))
            margin = tb - tp
            st_ = 'EXCELLENT' if margin >= cti else ('WATCH' if margin >= cti - 0.1 else 'ACTION')
            if np.isinf(tb):
                st_ = 'WATCH'
            pairs.append({'Fault at': far, 'Fault': fault, 'Primary': pr['Relay'], 'Primary I (A)': round(ip_, 0), 'Primary t (s)': round(tp, 3),
                          'Backup': bk['Relay'], 'Backup I (A)': round(ib, 0), 'Backup t (s)': round(tb, 3) if np.isfinite(tb) else None,
                          'Margin (s)': round(margin, 3) if np.isfinite(margin) else None, 'Required (s)': cti, 'Status': st_,
                          'Note': 'backup does not pick up' if np.isinf(tb) else ('' if margin >= cti else 'grading margin too small')})
    P = pd.DataFrame(pairs)
    # ---- distance zones
    dist = []
    zl = {ln: _line_z(d, ln) for ln in L['Line']}
    for _, r in R[R['Function'] == '21'].iterrows():
        ln = r['Element']
        if ln not in zl:
            continue
        a, b = ends[ln]
        far = b if r['Bus'] == a else a
        Z = zl[ln]
        nxt = [(x, zl[x]) for x, (p, q) in ends.items() if x in zl and x != ln and far in (p, q) and set((p, q)) != set((a, b))]
        shortest = min((abs(z) for _, z in nxt), default=None)
        longest = max((abs(z) for _, z in nxt), default=None)
        z1, z2, z3 = (r['Z1 %'] / 100 * abs(Z), r['Z2 %'] / 100 * abs(Z), r['Z3 %'] / 100 * abs(Z))
        ratio = (r['CT Primary A'] / r['CT Secondary A']) / (r['VT Primary kV'] * 1000 / r['VT Secondary V'])
        notes, st_ = [], 'EXCELLENT'
        if r['Z1 %'] > 90:
            notes.append('zone 1 above 90 %: risk of over-reach into the next line'); st_ = 'ACTION'
        elif r['Z1 %'] < 70:
            notes.append('zone 1 below 70 %: short instantaneous coverage'); st_ = 'WATCH'
        if r['Z2 %'] < 115:
            notes.append('zone 2 below 115 %: the remote end may not be covered'); st_ = 'ACTION'
        if shortest is not None and z2 > abs(Z) + 0.5 * shortest:
            notes.append('zone 2 reaches beyond 50 % of the shortest next line: check the zone 2 time against that line'); st_ = max(st_, 'WATCH', key=['EXCELLENT', 'WATCH', 'ACTION'].index)
        if longest is not None and z3 < abs(Z) + longest:
            notes.append('zone 3 does not reach the end of the longest next line (remote backup partly by breaker failure / local backup)')
        rating = float(L.set_index('Line').at[ln, 'Rating A'])
        zload = (0.9 * float(d['Buses'].set_index('Bus').at[r['Bus'], 'kV']) * 1000 / math.sqrt(3)) / (1.2 * rating)
        ang = math.atan2(Z.imag, Z.real)
        z3_load = z3 * max(math.cos(ang - math.radians(30)), 0.0)   # mho reach in the load direction (30 deg)
        if z3_load > 0.8 * zload:
            notes.append(f'zone 3 reach at the load angle ({z3_load:.0f} ohm) is above 80 % of the minimum load impedance '
                         f'({zload:.0f} ohm at 0.9 pu, 120 % rating): load encroachment risk, use a load blinder')
            st_ = max(st_, 'WATCH', key=['EXCELLENT', 'WATCH', 'ACTION'].index)
        if not (0 <= r['T2 s'] - r['T1 s'] and r['T3 s'] - r['T2 s'] >= cti):
            notes.append('zone times not graded by at least the coordination interval'); st_ = max(st_, 'WATCH', key=['EXCELLENT', 'WATCH', 'ACTION'].index)
        dist.append({'Relay': r['Relay'], 'Line': ln, 'Bus': r['Bus'], 'Line Z (ohm)': round(abs(Z), 2), 'Line Angle (deg)': round(math.degrees(math.atan2(Z.imag, Z.real)), 1),
                     'Z1 (ohm)': round(z1, 2), 'Z2 (ohm)': round(z2, 2), 'Z3 (ohm)': round(z3, 2),
                     'Z1 sec (ohm)': round(z1 * ratio, 3), 'Z2 sec (ohm)': round(z2 * ratio, 3), 'Z3 sec (ohm)': round(z3 * ratio, 3),
                     'T1 (s)': r['T1 s'], 'T2 (s)': r['T2 s'], 'T3 (s)': r['T3 s'], 'Load Z (ohm)': round(zload, 1),
                     'Next Lines': ', '.join(x for x, _ in nxt), 'Status': st_, 'Notes': '; '.join(notes) or 'reaches and times as recommended'})
    D = pd.DataFrame(dist)
    return {'oc_pairs': P, 'distance': D, 'cti': cti}


def oc_curve_points(relay_row, imin=None, imax=None, n=120):
    p = float(relay_row['Pickup A'])
    I = np.logspace(math.log10(p * 1.05), math.log10(imax or p * 40), n)
    t = oc_time(I, p, float(relay_row['TMS']), relay_row['Curve'], relay_row['Inst Pickup A'], float(relay_row['Inst Time s'] or 0.05))
    return I, t


# ------------------------------------------------------------------ N-1 contingency
def contingency(data, load_scale=1.0, warn_loading=80.0):
    d = clean(data)
    elems = [('Line', x) for x in d['Lines'].loc[d['Lines']['In Service'].fillna(1) == 1, 'Line']] + \
            [('Transformer', x) for x in d['Transformers'].loc[d['Transformers']['In Service'].fillna(1) == 1, 'Transformer']]
    base = load_flow(data, load_scale)
    rows = []
    for typ, e in elems:
        try:
            lf = load_flow(data, load_scale, outages=(e,), warn_loading=warn_loading)
            ok = lf['converged']
        except Exception:
            ok = False
        if not ok:
            rows.append({'Outage': e, 'Type': typ, 'Converged': False, 'Max Loading (%)': None, 'Most Loaded': '-', 'Min V (pu)': None,
                         'Max V (pu)': None, 'Islanded Buses': '-', 'Violations': '-', 'Status': 'ACTION'})
            continue
        B, Ln, Tr = lf['bus'], lf['line'], lf['trafo']
        isl = B[B['V (pu)'].isna()]['Bus'].tolist()
        br = pd.concat([Ln[['Line', 'Loading (%)', 'Status']].rename(columns={'Line': 'Branch'}),
                        Tr[['Transformer', 'Loading (%)', 'Status']].rename(columns={'Transformer': 'Branch'})])
        br = br[br['Status'] != 'OUT']
        w = br.sort_values('Loading (%)', ascending=False).iloc[0] if len(br) else None
        vio = [f"{r['Branch']} {r['Loading (%)']:.0f}%" for _, r in br[br['Loading (%)'] > 100].iterrows()] + \
              [f"{r['Bus']} {r['V (pu)']:.3f} pu" for _, r in B[(B['V (pu)'] < B['Vmin']) | (B['V (pu)'] > B['Vmax'])].iterrows()]
        st_ = 'ACTION' if vio or isl else ('WATCH' if (w is not None and w['Loading (%)'] > warn_loading) else 'EXCELLENT')
        rows.append({'Outage': e, 'Type': typ, 'Converged': True, 'Max Loading (%)': round(w['Loading (%)'], 1) if w is not None else None,
                     'Most Loaded': w['Branch'] if w is not None else '-', 'Min V (pu)': round(B['V (pu)'].min(), 4), 'Max V (pu)': round(B['V (pu)'].max(), 4),
                     'Islanded Buses': ', '.join(isl) or '-', 'Violations': '; '.join(vio) or '-', 'Status': st_})
    T = pd.DataFrame(rows)
    return {'table': T.sort_values(['Status', 'Max Loading (%)'], key=lambda s: s.map({'ACTION': 0, 'WATCH': 1, 'EXCELLENT': 2}) if s.name == 'Status' else -s.fillna(0)),
            'base': base['summary']}


# ------------------------------------------------------------------ validation against published results
IEEE9 = {'V': [1.000, 1.000, 1.000, 0.987, 0.975, 1.003, 0.986, 0.996, 0.958], 'A': [0.000, 9.669, 4.771, -2.407, -4.017, 1.926, 0.622, 3.799, -4.350]}
IEEE14 = {'V': [1.060, 1.045, 1.010, 1.018, 1.020, 1.070, 1.062, 1.090, 1.056, 1.051, 1.057, 1.055, 1.050, 1.036],
          'A': [0.00, -4.98, -12.72, -10.33, -8.78, -14.22, -13.37, -13.36, -14.94, -15.10, -14.79, -15.07, -15.16, -16.04]}


def validation():
    """Load flow on the IEEE 9 and 14 bus test systems against the published results, and IEC 60909 short circuits against
    hand calculation with the standard formulas."""
    import pandapower.networks as pn
    out = []
    for name, case, ref in (('IEEE 9-bus', pn.case9, IEEE9), ('IEEE 14-bus', pn.case14, IEEE14)):
        net = case(); pp.runpp(net, numba=False)
        for i in range(len(ref['V'])):
            out.append({'Test': name, 'Quantity': f'Bus {i + 1} voltage (pu)', 'Published': ref['V'][i], 'NetworkGuard': round(net.res_bus.vm_pu.iloc[i], 3)})
            out.append({'Test': name, 'Quantity': f'Bus {i + 1} angle (deg)', 'Published': ref['A'][i], 'NetworkGuard': round(net.res_bus.va_degree.iloc[i], 2 if name == 'IEEE 14-bus' else 3)})
    # IEC 60909 hand calculation: grid 20 000 MVA (R/X 0.1, X0/X1 1, R0/X0 0.1) + 100 km 400 kV line
    net = pp.create_empty_network(f_hz=50)
    a = pp.create_bus(net, 400); b = pp.create_bus(net, 400)
    pp.create_ext_grid(net, a, s_sc_max_mva=20000, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1)
    pp.create_line_from_parameters(net, a, b, 100, 0.03, 0.32, 0, 2, r0_ohm_per_km=0.2, x0_ohm_per_km=0.9, c0_nf_per_km=0, endtemp_degree=80)
    c, Un = 1.1, 400.0
    Zq = c * Un ** 2 / 20000; Xq = Zq / math.sqrt(1 + 0.1 ** 2); Rq = 0.1 * Xq
    Z1 = complex(Rq + 3.0, Xq + 32.0); Z0 = complex(0.1 * Xq + 20.0, Xq + 90.0)
    sc.calc_sc(net, case='max', fault='3ph', ip=True)
    ik3 = c * Un / (math.sqrt(3) * abs(Z1))
    kappa = 1.02 + 0.98 * math.exp(-3 * Z1.real / Z1.imag)
    out += [{'Test': 'IEC 60909 hand calculation', 'Quantity': "Ik3'' at the grid bus (kA)", 'Published': round(c * Un / (math.sqrt(3) * Zq), 3),
             'NetworkGuard': round(net.res_bus_sc.ikss_ka.iloc[0], 3)},
            {'Test': 'IEC 60909 hand calculation', 'Quantity': "Ik3'' at the line end (kA)", 'Published': round(ik3, 3), 'NetworkGuard': round(net.res_bus_sc.ikss_ka.iloc[1], 3)},
            {'Test': 'IEC 60909 hand calculation', 'Quantity': 'ip at the line end (kA), meshed method B', 'Published': round(kappa * math.sqrt(2) * ik3 * 1.15, 3) if False else round(kappa * math.sqrt(2) * ik3, 3),
             'NetworkGuard': round(net.res_bus_sc.ip_ka.iloc[1], 3)}]
    sc.calc_sc(net, case='max', fault='1ph')
    out.append({'Test': 'IEC 60909 hand calculation', 'Quantity': "Ik1'' at the line end (kA)", 'Published': round(math.sqrt(3) * c * Un / abs(2 * Z1 + Z0), 3),
                'NetworkGuard': round(net.res_bus_sc.ikss_ka.iloc[1], 3)})
    # IEC TR 60909-4 example network (grids, transformers, three-winding transformer, power station units, generators, motors)
    try:
        import iec60909_4_example as ex
        import contextlib
        import io as _io
        import logging
        logging.getLogger('pandapower').setLevel(logging.ERROR)
        quiet = contextlib.redirect_stdout(_io.StringIO())

        def run(fault, case):
            net = ex.iec_60909_4()
            if case == 'min':
                net.line['endtemp_degree'] = 80.0
                net.ext_grid['s_sc_min_mva'] = net.ext_grid['s_sc_max_mva'] / 10
                net.ext_grid['rx_min'] = net.ext_grid['rx_max']
            with quiet:
                sc.calc_sc(net, fault=fault, case=case, ip=True, tk_s=0.1, kappa_method='C')
            names = [f"Bus {i + 1} ({kv:g} kV)" for i, kv in enumerate(net.bus.vn_kv)]
            return net.res_bus_sc, names
        T4 = 'IEC TR 60909-4 example'
        for (fault, case, col, ref, q, nd) in (('3ph', 'max', 'ikss_ka', ex.IK3_MAX, "Ik3'' max (kA)", 4), ('3ph', 'max', 'ip_ka', ex.IP3_MAX, 'ip 3-phase (kA)', 4),
                                               ('3ph', 'max', 'skss_mw', ex.SK3_MAX, "Sk3'' (MVA)", 2), ('3ph', 'min', 'ikss_ka', ex.IK3_MIN, "Ik3'' min (kA)", 4),
                                               ('2ph', 'max', 'ikss_ka', ex.IK2_MAX, "Ik2'' max (kA)", 4), ('2ph', 'max', 'ip_ka', ex.IP2_MAX, 'ip 2-phase (kA)', 4)):
            r, names = run(fault, case)
            for i, v in enumerate(ref):
                out.append({'Test': T4, 'Quantity': f'{names[i]}: {q}', 'Published': v, 'NetworkGuard': round(float(r[col].iloc[i]), nd)})
    except Exception as e:      # the example file is missing: say so in the table
        out.append({'Test': 'IEC TR 60909-4 example', 'Quantity': f'not run ({e})', 'Published': 0.0, 'NetworkGuard': 1.0})
    # relay curve: IEC SI, TMS 0.1, M = 10 -> 0.297 s
    out.append({'Test': 'IEC 60255-151 relay curve', 'Quantity': 'Standard inverse, TMS 0.1, 10 x pickup (s)', 'Published': 0.297,
                'NetworkGuard': round(float(oc_time(10.0, 1.0, 0.1, 'IEC SI')), 3)})
    out.append({'Test': 'IEC 60255-151 relay curve', 'Quantity': 'Very inverse, TMS 0.1, 10 x pickup (s)', 'Published': 0.150,
                'NetworkGuard': round(float(oc_time(10.0, 1.0, 0.1, 'IEC VI')), 3)})
    T = pd.DataFrame(out)
    T['Difference'] = (T['NetworkGuard'] - T['Published']).round(4)
    tol = np.where(T['Quantity'].str.contains('angle'), 0.02, np.where(T['Test'].str.startswith('IEEE'), 0.0015,
                   np.where(T['Quantity'].str.contains('MVA'), 0.05, 0.002)))
    T['Result'] = np.where(T['Difference'].abs() <= tol + 1e-9, 'PASS', 'CHECK')
    return T


# ------------------------------------------------------------------ demo network (simulated, anonymised)
def demo():
    """A 400 kV interconnection with nine substations (simulated values, anonymised names)."""
    S = pd.DataFrame([
        ['North Point', 1.0, 9.0, 40, 40, 3150], ['North Bay', 4.0, 9.0, 40, 40, 3150], ['Central Hub', 2.5, 7.0, 40, 40, 3150],
        ['Central Junction', 2.5, 4.6, 40, 40, 3150], ['Island Link', 5.2, 4.6, 40, 40, 3150], ['Island City', 6.8, 4.6, 40, 40, 3150],
        ['South Hub', 2.5, 2.2, 40, 40, 3150], ['South Gate', 1.2, 0.5, 40, 40, 3150], ['East Gate', 5.0, 0.5, 63, 63, 4000]],
        columns=list(SCHEMA['Substations']))
    S['X'] = S['X'] * 1.7
    Bs = [['NP-400', 'North Point', 400], ['NB-400', 'North Bay', 400], ['NB-275', 'North Bay', 275], ['CH-400', 'Central Hub', 400],
          ['CJ-400', 'Central Junction', 400], ['IL-400', 'Island Link', 400], ['IC-400', 'Island City', 400], ['IC-220', 'Island City', 220],
          ['SH-400', 'South Hub', 400], ['SG-400', 'South Gate', 400], ['EG-400', 'East Gate', 400]]
    B = pd.DataFrame([[b, s, kv, 0.95, 1.05, 50 if kv == 275 else (40 if kv == 220 else np.nan)] for b, s, kv in Bs], columns=list(SCHEMA['Buses']))
    OHL, Q = 'Flint AAAC 740.8 kcmil', '400 kV double-circuit, quad bundle'
    lines = []
    for nm, a, b, km, n, typ in [('NP-CH', 'NP-400', 'CH-400', 92, 2, 'OHL'), ('NB-CH', 'NB-400', 'CH-400', 98, 2, 'OHL'),
                                 ('CH-CJ', 'CH-400', 'CJ-400', 285, 2, 'OHL'), ('CJ-IL', 'CJ-400', 'IL-400', 45, 2, 'Cable'),
                                 ('IL-IC', 'IL-400', 'IC-400', 42, 2, 'OHL'), ('CJ-SH', 'CJ-400', 'SH-400', 115, 1, 'OHL'),
                                 ('SH-SG', 'SH-400', 'SG-400', 300, 2, 'OHL'), ('SH-EG', 'SH-400', 'EG-400', 105, 2, 'OHL')]:
        for c in range(1, n + 1):
            lines.append({'Line': f'{nm}/{c}', 'From Bus': a, 'To Bus': b, 'Type': typ, 'Length km': km, 'Rating A': 2000,
                          'Conductor': OHL if typ == 'OHL' else '', 'Tower': Q if typ == 'OHL' else '', 'In Service': 1})
    L = pd.DataFrame(lines)
    T = pd.DataFrame([[f'NB-T{i}', 'NB-400', 'NB-275', 750, 400, 275, 14.0, 0.25, 'YNyn0', 0, 1] for i in (1, 2, 3)] +
                     [[f'IC-T{i}', 'IC-400', 'IC-220', 500, 400, 220, 13.0, 0.25, 'YNyn0', 0, 1] for i in (1, 2, 3)],
                     columns=list(SCHEMA['Transformers']))
    Rx = pd.DataFrame([['NP-R1', 'NP-400', '', 125, 400, 1], ['CH-R1', 'CH-400', '', 125, 400, 1], ['CH-R2', 'CH-400', '', 125, 400, 1],
                       ['CJ-R1', 'CJ-400', '', 125, 400, 1], ['CJ-R2', 'CJ-400', '', 125, 400, 1],
                       ['IL-R1', 'IL-400', 'CJ-IL/1', 250, 400, 1], ['IL-R2', 'IL-400', 'CJ-IL/2', 250, 400, 1],
                       ['CJ-LR1', 'CJ-400', 'CJ-IL/1', 250, 400, 1], ['CJ-LR2', 'CJ-400', 'CJ-IL/2', 250, 400, 1],
                       ['SH-R1', 'SH-400', '', 125, 400, 1], ['SG-R1', 'SG-400', '', 125, 400, 1], ['EG-R1', 'EG-400', '', 125, 400, 1]],
                      columns=list(SCHEMA['Reactors']))
    G = pd.DataFrame([['Grid North', 'NP-400', 1, 0, 1.02, 17000, 12000, 0.08, 1.0, 0.12],
                      ['Grid North-East', 'NB-275', 0, 150, 1.0, 18000, 12000, 0.08, 1.1, 0.12],
                      ['Grid Island', 'IC-220', 0, -400, 1.0, 9000, 6000, 0.1, 1.2, 0.15],
                      ['Grid South', 'SG-400', 0, -450, 1.0, 16000, 11000, 0.08, 1.1, 0.12],
                      ['Grid East', 'EG-400', 0, -300, 1.0, 26000, 18000, 0.08, 1.0, 0.12]], columns=list(SCHEMA['Grids']))
    C = pd.DataFrame([[f'HVDC-{i}', 'CH-400', 300, 0, 600, 1] for i in (1, 2, 3)], columns=list(SCHEMA['Converters']))
    Ld = pd.DataFrame([['CH-AUX', 'CH-400', 5, 2, 1], ['SH-AUX', 'SH-400', 3, 1, 1]], columns=list(SCHEMA['Loads']))
    rel = []
    for _, l in L.iterrows():
        for end in ('From Bus', 'To Bus'):
            side = 'A' if end == 'From Bus' else 'B'
            rel.append({'Relay': f"{l['Line']}-21{side}", 'Element': l['Line'], 'Bus': l[end], 'Function': '21', 'CT Primary A': 2000,
                        'CT Secondary A': 1, 'VT Primary kV': 400, 'VT Secondary V': 110, 'Z1 %': 80, 'Z2 %': 120, 'Z3 %': 220,
                        'T1 s': 0.0, 'T2 s': 0.4, 'T3 s': 1.0})
    tms = {'NP-CH': 0.40, 'NB-CH': 0.40, 'CH-CJ': 0.30, 'CJ-SH': 0.20, 'SH-SG': 0.10, 'SH-EG': 0.10}
    for _, l in L[L['Line'].str.split('/').str[0].isin(list(tms))].iterrows():
        t_ = tms[l['Line'].split('/')[0]]
        rel.append({'Relay': f"{l['Line']}-51NA", 'Element': l['Line'], 'Bus': l['From Bus'], 'Function': '51N', 'CT Primary A': 2000,
                    'CT Secondary A': 1, 'Curve': 'IEC SI', 'Pickup A': 400, 'TMS': t_})
    for _, t in T.iterrows():
        rel.append({'Relay': f"{t['Transformer']}-51", 'Element': t['Transformer'], 'Bus': t['HV Bus'], 'Function': '51', 'CT Primary A': 1500,
                    'CT Secondary A': 1, 'Curve': 'IEC SI', 'Pickup A': 1300, 'TMS': 0.3, 'Inst Pickup A': 9000, 'Inst Time s': 0.05})
    Rl = pd.DataFrame(rel)
    # an over-reaching zone 1 and a short zone 2 time to show the checks
    Rl.loc[Rl['Relay'] == 'IL-IC/1-21A', 'Z1 %'] = 95
    Rl.loc[Rl['Relay'] == 'SH-EG/1-21A', 'T2 s'] = 0.3
    Rl.loc[Rl['Relay'] == 'SH-EG/1-21A', 'T3 s'] = 0.5
    sub_of = dict(zip(B['Bus'], B['Substation']))
    brk = dict(zip(S['Substation'], S['Breaking kA']))
    Bk = [{'Breaker': f"{l['Line']} CB-{side}", 'Bus': l[end], 'Element': l['Line'], 'Breaking kA': brk[sub_of[l[end]]], 'Rated A': 2000}
          for _, l in L.iterrows() for end, side in (('From Bus', 'A'), ('To Bus', 'B'))]
    Bk += [{'Breaker': f"{t['Transformer']} CB-{side}", 'Bus': t[end], 'Element': t['Transformer'], 'Breaking kA': 50 if side == 'LV' and t['LV kV'] == 275 else 40,
            'Rated A': 3150} for _, t in T.iterrows() for end, side in (('HV Bus', 'HV'), ('LV Bus', 'LV'))]
    return clean({'Substations': S, 'Buses': B, 'Lines': L, 'Transformers': T, 'Reactors': Rx, 'Grids': G, 'Converters': C, 'Loads': Ld,
                  'Breakers': pd.DataFrame(Bk), 'Relays': Rl})


# ------------------------------------------------------------------ building a network step by step
def _append(data, kind, rows):
    data = clean(data)
    data[kind] = pd.concat([data[kind], pd.DataFrame(rows)], ignore_index=True)
    data[kind] = data[kind].drop_duplicates(id_col(kind), keep='last')
    return clean(data)


def next_position(data):
    S = clean(data)['Substations']
    n = len(S)
    return float((n % 4) * 3.0), float(9.0 - (n // 4) * 3.0)


def add_substation(data, name, kvs, x=None, y=None, breaking=40.0):
    if x is None or y is None:
        x, y = next_position(data)
    data = _append(data, 'Substations', [{'Substation': name, 'X': x, 'Y': y, 'Breaking kA': breaking, 'Short-time kA (1 s)': breaking, 'Busbar A': 3150}])
    return _append(data, 'Buses', [{'Bus': f'{name} {kv:g}', 'Substation': name, 'kV': kv, 'Vmin pu': 0.95, 'Vmax pu': 1.05} for kv in kvs])


def add_line(data, a, b, circuits=1, length=None, typ='OHL', conductor='', tower='', rating=2000.0, name=None, breakers=True):
    d = clean(data)
    sub = dict(zip(d['Buses']['Bus'], d['Buses']['Substation']))
    base = name or f"{sub.get(a, a)}-{sub.get(b, b)}"
    rows, bk = [], []
    brk = dict(zip(d['Substations']['Substation'], d['Substations']['Breaking kA']))
    for c in range(1, int(circuits) + 1):
        ln = f'{base}/{c}'
        rows.append({'Line': ln, 'From Bus': a, 'To Bus': b, 'Type': typ, 'Length km': length, 'Rating A': rating, 'Conductor': conductor, 'Tower': tower, 'In Service': 1})
        if breakers:
            bk += [{'Breaker': f'{ln} CB-A', 'Bus': a, 'Element': ln, 'Breaking kA': brk.get(sub.get(a), 40.0), 'Rated A': rating},
                   {'Breaker': f'{ln} CB-B', 'Bus': b, 'Element': ln, 'Breaking kA': brk.get(sub.get(b), 40.0), 'Rated A': rating}]
    data = _append(data, 'Lines', rows)
    return _append(data, 'Breakers', bk) if bk else data


def add_transformer(data, hv, lv, count=1, mva=None, uk=None, name=None, breakers=True):
    d = clean(data)
    kv = dict(zip(d['Buses']['Bus'], d['Buses']['kV'])); sub = dict(zip(d['Buses']['Bus'], d['Buses']['Substation']))
    n0 = (d['Transformers']['HV Bus'] == hv).sum()
    rows, bk = [], []
    for k in range(1, int(count) + 1):
        t = f"{name or sub.get(hv, hv) + ' T'}{n0 + k}"
        rows.append({'Transformer': t, 'HV Bus': hv, 'LV Bus': lv, 'MVA': mva, 'HV kV': kv.get(hv), 'LV kV': kv.get(lv), 'uk %': uk, 'In Service': 1})
        if breakers:
            bk += [{'Breaker': f'{t} CB-HV', 'Bus': hv, 'Element': t}, {'Breaker': f'{t} CB-LV', 'Bus': lv, 'Element': t}]
    data = _append(data, 'Transformers', rows)
    return _append(data, 'Breakers', bk) if bk else data


def add_reactor(data, bus, mvar=None, line='', count=1, name=None):
    d = clean(data)
    kv = dict(zip(d['Buses']['Bus'], d['Buses']['kV'])); sub = dict(zip(d['Buses']['Bus'], d['Buses']['Substation']))
    n0 = (d['Reactors']['Bus'] == bus).sum()
    return _append(data, 'Reactors', [{'Reactor': f"{name or sub.get(bus, bus) + ' R'}{n0 + k}", 'Bus': bus, 'Line': line, 'MVAr': mvar, 'kV': kv.get(bus), 'In Service': 1}
                                      for k in range(1, int(count) + 1)])


def add_grid(data, name, bus, sk_max=None, sk_min=None, slack=False, p_import=0.0):
    data = clean(data)
    if slack:
        data['Grids']['Slack'] = 0
    return _append(data, 'Grids', [{'Grid': name, 'Bus': bus, 'Slack': 1 if slack or not len(data['Grids']) else 0, 'P Import MW': p_import, 'V Setpoint pu': 1.0,
                                    'Sk max MVA': sk_max, 'Sk min MVA': sk_min}])


def add_converter(data, name, bus, p=0.0, rated=600.0):
    return _append(data, 'Converters', [{'Converter': name, 'Bus': bus, 'P MW': p, 'Q MVAr': 0.0, 'Rated MW': rated, 'In Service': 1}])


def add_breaker(data, name, bus, element='', breaking=40.0, rated=2000.0):
    return _append(data, 'Breakers', [{'Breaker': name, 'Bus': bus, 'Element': element, 'Breaking kA': breaking, 'Rated A': rated}])


# ------------------------------------------------------------------ single-line diagram geometry
def sld_layout(data):
    """Positions for the diagram: substation boxes, bus bars, branches (with an offset per parallel circuit), shunts and sources."""
    d = clean(data)
    S = d['Substations'].set_index('Substation')
    pos = {}
    subs = {}
    for s, g in d['Buses'].groupby('Substation', sort=False):
        x0, y0 = (float(S.at[s, 'X']), float(S.at[s, 'Y'])) if s in S.index else (0.0, 0.0)
        g = g.sort_values('kV', ascending=False)
        for k, (_, b) in enumerate(g.iterrows()):
            pos[b['Bus']] = (x0, y0 - 0.75 * k)
        subs[s] = (x0, y0, len(g))
    br = []
    groups = {}
    for _, r in d['Lines'].iterrows():
        key = tuple(sorted((r['From Bus'], r['To Bus'])))
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        n = len(rs)
        for k, r in enumerate(rs):
            off = (k - (n - 1) / 2) * 0.12
            br.append({'id': r['Line'], 'kind': 'Line', 'a': r['From Bus'], 'b': r['To Bus'], 'off': off, 'cable': r['Type'].lower().startswith('cab')})
    for _, r in d['Transformers'].iterrows():
        sib = d['Transformers'][(d['Transformers']['HV Bus'] == r['HV Bus']) & (d['Transformers']['LV Bus'] == r['LV Bus'])]
        k = list(sib['Transformer']).index(r['Transformer'])
        br.append({'id': r['Transformer'], 'kind': 'Transformer', 'a': r['HV Bus'], 'b': r['LV Bus'], 'off': (k - (len(sib) - 1) / 2) * 0.22, 'cable': False})
    sh = [{'id': r['Reactor'], 'kind': 'Reactor', 'bus': r['Bus']} for _, r in d['Reactors'].iterrows()]
    src = [{'id': r['Grid'], 'kind': 'Grid', 'bus': r['Bus']} for _, r in d['Grids'].iterrows()] + \
          [{'id': r['Converter'], 'kind': 'Converter', 'bus': r['Bus']} for _, r in d['Converters'].iterrows()] + \
          [{'id': r['Load'], 'kind': 'Load', 'bus': r['Bus']} for _, r in d['Loads'].iterrows()]
    bk = [{'id': r['Breaker'], 'bus': r['Bus'], 'element': r['Element']} for _, r in d['Breakers'].iterrows()]
    return {'pos': pos, 'subs': subs, 'branches': br, 'shunts': sh, 'sources': src, 'breakers': bk}


# ------------------------------------------------------------------ platform register
def asset_rows(lf=None, scr=None, prot=None, cont=None):
    """Findings per equipment for the platform register: risk 0-100 %, status, driver, action."""
    rows = []

    def add(aid, name, typ, risk, status, driver, action):
        rows.append({'Asset ID': aid, 'Asset': name, 'Asset Type': typ, 'Risk (%)': round(float(risk), 1), 'Status': status,
                     'Main Driver': driver, 'Action': action})
    if lf:
        for _, r in lf['line'].iterrows():
            if r['Status'] == 'OUT':
                continue
            rk = float(np.clip((r['Loading (%)'] - 50) * 1.4, 0, 100))
            add(r['Line'], f"Circuit {r['Line']}", 'Line', rk, r['Status'], f"Load flow: loading {r['Loading (%)']:.0f} %",
                'Reduce the transfer or reinforce the circuit' if r['Status'] == 'ACTION' else ('Watch the loading trend' if r['Status'] == 'WATCH' else 'No action'))
        for _, r in lf['trafo'].iterrows():
            if r['Status'] == 'OUT':
                continue
            rk = float(np.clip((r['Loading (%)'] - 50) * 1.4, 0, 100))
            add(r['Transformer'], f"Transformer {r['Transformer']}", 'Transformer', rk, r['Status'], f"Load flow: loading {r['Loading (%)']:.0f} %",
                'Share the load or reduce the transfer' if r['Status'] != 'EXCELLENT' else 'No action')
        for _, r in lf['bus'].iterrows():
            if r['Status'] == 'EXCELLENT':
                continue
            add(r['Bus'], f"Bus {r['Bus']}", 'Bus', 75 if r['Status'] == 'ACTION' else 45, r['Status'], f"Voltage {r['V (pu)']:.3f} pu",
                'Adjust reactors / voltage set points')
    if scr:
        for _, r in scr['bus'].iterrows():
            duty = max(r['Breaker Duty (%)'], r['Making Duty (%)'], r['Short-time Duty (%)'])
            add(f"{r['Substation']} switchgear", f"{r['Substation']} {r['kV']:.0f} kV switchgear", 'Switchgear', float(np.clip((duty - 50) * 1.6, 0, 100)), r['Status'],
                f"Short circuit: duty {duty:.0f} % of rating ({max(r['Ik3 (kA)'], r['Ik1 (kA)']):.1f} kA)",
                'Replace or limit the fault level (split busbar, series reactor)' if r['Status'] == 'ACTION' else
                ('Plan for the fault-level growth' if r['Status'] == 'WATCH' else 'No action'))
    if prot:
        for _, r in prot['distance'].iterrows():
            if r['Status'] != 'EXCELLENT':
                add(r['Relay'], f"Relay {r['Relay']}", 'Protection relay', 80 if r['Status'] == 'ACTION' else 50, r['Status'], f"Distance settings: {r['Notes']}",
                    'Review the zone settings')
        for _, r in prot['oc_pairs'].iterrows():
            if r['Status'] != 'EXCELLENT':
                add(r['Backup'], f"Relay {r['Backup']}", 'Protection relay', 80 if r['Status'] == 'ACTION' else 50, r['Status'],
                    f"Coordination with {r['Primary']}: {r['Note'] or 'margin'}", 'Regrade TMS / pickup')
    if cont is not None:
        for _, r in cont['table'].iterrows():
            if r['Status'] == 'ACTION':
                why = r['Violations'] if r['Violations'] != '-' else (f"separates {r['Islanded Buses']}" if r['Islanded Buses'] != '-' else 'no solution')
                add(f"N-1 {r['Outage']}", f"Outage of {r['Outage']}", 'Contingency', 80, 'ACTION', f"N-1: {why}",
                    'Plan a remedial action (second circuit, special protection scheme or operating limit)')
    R = pd.DataFrame(rows)
    if len(R):
        R = R.sort_values('Risk (%)', ascending=False).drop_duplicates('Asset ID')
    return R


# ------------------------------------------------------------------ fault study: a real fault on a line
FAULT_OF_TYPE = {'AG': '1ph', 'BG': '1ph', 'CG': '1ph', 'AB': '2ph', 'BC': '2ph', 'CA': '2ph', 'AC': '2ph',
                 'ABG': '2ph', 'BCG': '2ph', 'CAG': '2ph', 'ACG': '2ph', 'ABC': '3ph', 'ABCG': '3ph'}
FAULT_LABEL = {'1ph': 'Single phase to ground', '2ph': 'Phase to phase', '3ph': 'Three phase'}


def _norm(s):
    return re.sub(r'[^A-Z0-9]+', ' ', str(s).upper()).split()


_STOP = {'AL', 'KV', 'SS', 'S', 'SUB', 'SUBSTATION', 'STATION', 'LINE', 'L', 'OHL', 'CABLE', 'CIRCUIT', 'GIS', 'AIS', 'BAY'}


def _stn(s):
    """A station name reduced to letters (prefix 'AL', 'kV', 'SS' and similar words removed)."""
    w = [x for x in re.sub(r'[^A-Z ]', ' ', str(s or '').upper()).split() if x not in _STOP]
    return ''.join(w)


def station_sim(a, b):
    """How alike two spellings of a station name are (0..1), tolerant of short codes and spelling variants
    (e.g. a five-letter code against the full name, a missing vowel, 'AL' written or not)."""
    import difflib
    A, B = _stn(a), _stn(b)
    if not A or not B:
        return 0.0
    if A in B or B in A:
        return 1.0 if min(len(A), len(B)) >= 3 else 0.5
    cons = lambda x: x[:1] + re.sub(r'[AEIOUY]', '', x[1:])
    r = max(difflib.SequenceMatcher(None, A, B).ratio(), difflib.SequenceMatcher(None, cons(A), cons(B)).ratio())
    short, long_ = (A, B) if len(A) <= len(B) else (B, A)
    if len(short) <= 4 and short[0] == long_[0]:                 # an abbreviation: its letters in order inside the long name
        it = iter(long_)
        if all(ch in it for ch in short):
            r = max(r, 0.85)
    return r


def _circuit(texts):
    for x in texts or []:
        m = re.search(r'(?<![A-Z0-9])(\d{1,3})_([A-Z]{2,})_([A-Z]{2,})', str(x or '').upper().replace('-', '_'))
        if m:
            return m.group(1), m.group(2), m.group(3)
    return None


def line_for_relay(data, own='', remote='', names=()):
    """The network line a relay protects, from its own station, the far-end text (relay description) and any names that
    carry the circuit (file or folder names such as <circuit>_<station>_<station>). Returns (line, end bus, score);
    the end bus is the relay's own end. Names come only from the uploaded files."""
    d = clean(data)
    sub = dict(zip(d['Buses']['Bus'], d['Buses']['Substation']))
    cc = _circuit(list(names) + [remote, own])
    rem_words = [w for w in re.split(r'[^A-Z]+', re.sub(r'\b\d+[A-Z]*\d*[A-Z]*\b|\bL\d+\b', ' ', str(remote or '').upper())) if len(w) >= 3]
    best = (None, None, 0.0)
    for _, r in d['Lines'].iterrows():
        ends = [(r['From Bus'], sub.get(r['From Bus'], r['From Bus'])), (r['To Bus'], sub.get(r['To Bus'], r['To Bus']))]
        for i in (0, 1):
            (bus_o, so), (_, sr) = ends[i], ends[1 - i]
            own_s = station_sim(own, so) if own else 0.0
            rem_s = max((station_sim(w, sr) for w in rem_words), default=0.0)
            pair = max(min(station_sim(cc[1], so), station_sim(cc[2], sr)), min(station_sim(cc[2], so), station_sim(cc[1], sr))) if cc else 0.0
            pair_ok = pair >= 0.6
            if cc and not pair_ok:
                continue                                  # the circuit name points to another line (e.g. one leaving the network)
            own_ok = own_s >= 0.6 and own_s > station_sim(own, sr)
            if not (pair_ok or (own_ok and rem_s >= 0.6)):
                continue                                  # both ends of the line must be recognised
            s = 2 * own_s + 2 * rem_s
            if pair_ok:
                num = re.match(r'\s*(\d{1,3})', str(r['Line']))
                s += 1.5 * 2 * pair + (3 if (num and num.group(1) == cc[0]) else 0)
            if s > best[2] + 1e-9:
                best = (r['Line'], bus_o, s)
    return best


def match_line(name, data):
    """The network line that a name from another source (relay description, FaultGuard title) refers to.
    Uses the line ID, its other names, and its end substations. Returns (line, score)."""
    import re as _re  # noqa: F401
    d = clean(data)
    toks = [t for t in _norm(name) if t not in ('KV', 'LINE', 'CIRCUIT', 'OHL', '400', '380', '275', '220', '132')]
    nums = {t.lstrip('L') for t in toks if re.fullmatch(r'L?\d{1,3}', t)}
    words = [t for t in toks if not re.fullmatch(r'L?\d{1,3}', t) and len(t) >= 3]
    sub = dict(zip(d['Buses']['Bus'], d['Buses']['Substation']))
    best, score = None, 0
    for _, r in d['Lines'].iterrows():
        ends = ' '.join(_norm(f"{sub.get(r['From Bus'], r['From Bus'])} {sub.get(r['To Bus'], r['To Bus'])} {r['Line']} {r.get('Other Names', '')}"))
        s = sum(2 for w in words if w[:4] in ends) + sum(1 for w in words if w[:3] in ends and w[:4] not in ends)
        own = set(re.findall(r'\d{1,3}', f"{r['Line']} {r.get('Other Names', '')}"))
        s += 3 * len(nums & own)
        if s > score:
            best, score = r['Line'], s
    return best, score


def fault_study(data, line, d_km, fault='1ph', rf=0.0, case='max', measured=None):
    """Fault on a line at d_km from its From Bus: total current, current from each end, bus voltages, branch currents and
    the expected relay operation. measured: {'I_total': kA, 'I_from': kA, 'I_to': kA, 'loc_km': km} (any of them)."""
    from pandapower.shortcircuit.toolbox import calc_sc_on_line
    import logging
    logging.getLogger('pandapower').setLevel(logging.ERROR)
    d, _ = filled(data)
    L = d['Lines'].set_index('Line')
    if line not in L.index:
        raise ValueError(f'Line {line} not found')
    r = L.loc[line]
    length = float(r['Length km'])
    pos = min(max(float(d_km) / length, 0.001), 0.999)
    net = build(data, 'sc')
    ix = int(net.line.index[net.line.name == line][0])
    a_bus, b_bus = int(net.line.at[ix, 'from_bus']), int(net.line.at[ix, 'to_bus'])
    aux, fb = calc_sc_on_line(net, ix, pos, fault=fault, case=case, branch_results=True, r_fault_ohm=float(rf or 0.0))
    bn = dict(zip(aux.bus.index, aux.bus.name))
    ik = float(aux.res_bus_sc.at[fb, 'ikss_ka'])
    seg = aux.line[(aux.line.from_bus == fb) | (aux.line.to_bus == fb)]
    I = {}
    for j, s in seg.iterrows():
        other = s.to_bus if s.from_bus == fb else s.from_bus
        cur = aux.res_line_sc.at[j, 'ikss_from_ka'] if s.from_bus == other else aux.res_line_sc.at[j, 'ikss_to_ka']
        I[int(other)] = float(cur)
    ia, ib = I.get(a_bus, 0.0), I.get(b_bus, 0.0)
    # bus voltages during the fault (from the branch results)
    V = {}
    for j, l_ in aux.line.iterrows():
        for side, b_ in (('from', l_.from_bus), ('to', l_.to_bus)):
            v = aux.res_line_sc.at[j, f'vm_{side}_pu'] if f'vm_{side}_pu' in aux.res_line_sc else np.nan
            if pd.notna(v) and b_ != fb:
                V[bn[b_]] = float(v)
    buses = pd.DataFrame([{'Bus': b, 'Voltage during fault (pu)': round(v, 3), 'Dip (%)': round(100 * (1 - v), 1)} for b, v in V.items()]).sort_values('Voltage during fault (pu)') \
        if V else pd.DataFrame(columns=['Bus', 'Voltage during fault (pu)', 'Dip (%)'])
    br = []
    for j, l_ in aux.line.iterrows():
        if not l_.in_service:
            continue
        nm = line + (' (fault side A)' if (l_.from_bus == fb or l_.to_bus == fb) and a_bus in (l_.from_bus, l_.to_bus) else
                     ' (fault side B)' if (l_.from_bus == fb or l_.to_bus == fb) else '') if (fb in (l_.from_bus, l_.to_bus)) else l_['name']
        br.append({'Branch': nm, 'From': bn[l_.from_bus], 'To': bn[l_.to_bus], 'I From (kA)': round(float(aux.res_line_sc.at[j, 'ikss_from_ka']), 3),
                   'I To (kA)': round(float(aux.res_line_sc.at[j, 'ikss_to_ka']), 3)})
    branches = pd.DataFrame(br)
    # ---- expected relay operation
    ZL = complex(float(r['R1 ohm/km']), float(r['X1 ohm/km'])) * length
    zl = {ln: complex(rr['R1 ohm/km'], rr['X1 ohm/km']) * rr['Length km'] for ln, rr in L.iterrows()}
    ends = {ln: (rr['From Bus'], rr['To Bus']) for ln, rr in L.iterrows()}
    A, B = r['From Bus'], r['To Bus']
    da, db = pos * length, (1 - pos) * length
    rel = []
    R = d['Relays']
    cur_by_line = {row['Branch']: row for row in br}
    for _, rr in R.iterrows():
        e = rr['Element']
        if e not in ends:
            continue
        p, q = ends[e]
        if e == line:
            frac = pos if rr['Bus'] == A else 1 - pos
            zseen = abs(ZL) * frac
            i_rel = ia if rr['Bus'] == A else ib
            how = 'on the faulted line'
        elif rr['Bus'] in (p, q) and ({p, q} & {A, B}) and rr['Bus'] not in (A, B):
            near = A if A in (p, q) else B
            i_adj = cur_by_line.get(e)
            i_adj = float(max(i_adj['I From (kA)'], i_adj['I To (kA)'])) if i_adj else 0.0
            i_end = ia if near == A else ib
            infeed = (i_end / i_adj) if i_adj > 1e-6 else np.inf
            zseen = abs(zl[e]) + abs(ZL) * (pos if near == A else 1 - pos) * infeed
            i_rel = i_adj
            how = f'back-up through {near} (infeed ×{infeed:.2f})' if np.isfinite(infeed) else 'no current'
        else:
            continue
        if rr['Function'] == '21':
            zr = abs(zl[e])
            zs = [(1, rr['Z1 %'] / 100 * zr, rr['T1 s']), (2, rr['Z2 %'] / 100 * zr, rr['T2 s']), (3, rr['Z3 %'] / 100 * zr, rr['T3 s'])]
            hit = next(((z, t) for z, reach, t in zs if zseen <= reach), None)
            rel.append({'Relay': rr['Relay'], 'Bus': rr['Bus'], 'Function': 'Distance', 'Sees': how, 'Current (kA)': round(i_rel, 2),
                        'Impedance seen (ohm)': round(zseen, 2) if np.isfinite(zseen) else None,
                        'Operates in': f'zone {hit[0]}' if hit else 'no zone', 'Expected time (s)': round(float(hit[1]), 3) if hit else None})
        elif rr['Function'] in ('51', '51N') and (rr['Function'] == '51N') == (fault == '1ph'):
            t = float(oc_time(i_rel * 1000, rr['Pickup A'], rr['TMS'], rr['Curve'], rr['Inst Pickup A'], rr['Inst Time s']))
            rel.append({'Relay': rr['Relay'], 'Bus': rr['Bus'], 'Function': 'Earth fault' if rr['Function'] == '51N' else 'Overcurrent', 'Sees': how,
                        'Current (kA)': round(i_rel, 2), 'Impedance seen (ohm)': None, 'Operates in': 'inverse time' if np.isfinite(t) else 'below pickup',
                        'Expected time (s)': round(t, 3) if np.isfinite(t) else None})
    relays = pd.DataFrame(rel)
    if len(relays):
        relays = relays.sort_values('Expected time (s)', na_position='last', ignore_index=True)
    out = {'line': line, 'from': A, 'to': B, 'length_km': length, 'd_km': pos * length, 'pos': pos, 'fault': fault, 'rf': rf, 'case': case,
           'Ik (kA)': round(ik, 3), 'I_from (kA)': round(ia, 3), 'I_to (kA)': round(ib, 3), 'buses': buses, 'branches': branches, 'relays': relays}
    # ---- comparison with the measured values
    m = measured or {}
    cmp_ = []
    for k, lab, val in (('I_total', 'Total fault current (kA)', ik), ('I_from', f'Current from {A} (kA)', ia), ('I_to', f'Current from {B} (kA)', ib),
                        ('loc_km', f'Distance from {A} (km)', pos * length)):
        if m.get(k) not in (None, '') and not (isinstance(m.get(k), float) and np.isnan(m.get(k))):
            mv = float(m[k])
            diff = 100 * (val - mv) / mv if mv else np.nan
            cmp_.append({'Quantity': lab, 'Measured': round(mv, 3), 'Calculated': round(val, 3), 'Difference (%)': round(diff, 1),
                         'Status': 'EXCELLENT' if abs(diff) <= 10 else ('WATCH' if abs(diff) <= 25 else 'ACTION')})
    out['compare'] = pd.DataFrame(cmp_)
    return out


def _end_currents(net, ix, pos, fault, rf, case='max'):
    from pandapower.shortcircuit.toolbox import calc_sc_on_line
    a_bus, b_bus = int(net.line.at[ix, 'from_bus']), int(net.line.at[ix, 'to_bus'])
    aux, fb = calc_sc_on_line(net, ix, pos, fault=fault, case=case, branch_results=True, r_fault_ohm=float(rf or 0.0))
    I = {}
    for j, s_ in aux.line[(aux.line.from_bus == fb) | (aux.line.to_bus == fb)].iterrows():
        other = s_.to_bus if s_.from_bus == fb else s_.from_bus
        I[int(other)] = float(aux.res_line_sc.at[j, 'ikss_from_ka'] if s_.from_bus == other else aux.res_line_sc.at[j, 'ikss_to_ka'])
    return I.get(a_bus, 0.0), I.get(b_bus, 0.0)


def locate_from_currents(data, line, fault='1ph', rf=0.0, I_from=None, I_to=None, n=20):
    """Fault position that best explains the measured end currents (scan along the line, then refine)."""
    import logging
    logging.getLogger('pandapower').setLevel(logging.ERROR)
    if not I_from and not I_to:
        return None
    d, _ = filled(data)
    length = float(d['Lines'].set_index('Line').at[line, 'Length km'])
    net = build(data, 'sc')
    ix = int(net.line.index[net.line.name == line][0])

    def err(pos):
        ia, ib = _end_currents(net, ix, pos, fault, rf)
        e = (((ia - I_from) / I_from) ** 2 if I_from else 0) + (((ib - I_to) / I_to) ** 2 if I_to else 0)
        return e / (bool(I_from) + bool(I_to))
    grid = [(k / n, err(k / n)) for k in range(1, n)]
    p0, e0 = min(grid, key=lambda x: x[1])
    for p in np.linspace(max(0.005, p0 - 1 / n), min(0.995, p0 + 1 / n), 11):
        e = err(p)
        if e < e0:
            p0, e0 = p, e
    return {'loc_km': round(p0 * length, 1), 'rms_error_pct': round(100 * math.sqrt(e0), 1)}


# ============================================================== fault analysis on the network (real fault applied)
PHASE_FAULTS = {'AG': ('1ph', 'A'), 'BG': ('1ph', 'B'), 'CG': ('1ph', 'C'),
                'AB': ('2ph', 'C'), 'BC': ('2ph', 'A'), 'CA': ('2ph', 'B'), 'AC': ('2ph', 'B'),
                'ABG': ('2phg', 'C'), 'BCG': ('2phg', 'A'), 'CAG': ('2phg', 'B'), 'ACG': ('2phg', 'B'),
                'ABC': ('3ph', 'A'), 'ABCG': ('3ph', 'A')}
FAULT_NAME = {'1ph': 'single phase to ground', '2ph': 'phase to phase', '2phg': 'two phases to ground', '3ph': 'three phase'}
RELAY_OP_S, BREAKER_S = 0.02, 0.05          # relay pick-up and breaker opening times added to every setting
_A = np.exp(2j * np.pi / 3)


def _seq_matrices(aux, fb):
    """Positive- and zero-sequence bus impedance matrices of the short-circuit network (pandapower internals), the fault bus
    index in them and the branch admittances (from / to)."""
    from pandapower.shortcircuit import calc_sc
    from pandapower.shortcircuit.ppc_conversion import _init_ppc, _create_k_updated_ppci, _get_is_ppci_bus
    from pandapower.shortcircuit.impedance import _calc_ybus, _calc_zbus
    from pandapower.pd2ppc_zero import _pd2ppc_zero
    from pandapower.pypower.idx_brch_sc import K_ST
    calc_sc(aux, bus=fb, fault='1ph', case='max', branch_results=False)      # sets the calculation options
    ppc, p1 = _init_ppc(aux)
    pb = _get_is_ppci_bus(aux, np.array([fb]))
    _, p1, _ = _create_k_updated_ppci(aux, p1, ppci_bus=pb)
    _calc_ybus(p1); _calc_zbus(aux, p1)
    _, p0 = _pd2ppc_zero(aux, ppc['branch'][:, K_ST])
    _calc_ybus(p0); _calc_zbus(aux, p0)
    return p1, p0, int(pb[0])


def _phases(v0, v1, v2, special):
    """Phase values from sequence values computed with phase A as the special phase, relabelled so that `special` is it."""
    a = v0 + v1 + v2
    b = v0 + _A ** 2 * v1 + _A * v2
    c = v0 + _A * v1 + _A ** 2 * v2
    order = 'ABC'
    k = order.index(special)
    lab = [order[(k + i) % 3] for i in range(3)]             # computed a, b, c -> actual labels
    return dict(zip(lab, (a, b, c)))


def _branch_rows(aux, p1):
    """For every in-service line and transformer of the network: its row in the internal branch table (matched by end buses)."""
    from pandapower.pypower.idx_brch import F_BUS, T_BUS
    lk = aux._pd2ppc_lookups['bus']
    f_, t_ = p1['branch'][:, F_BUS].real.astype(int), p1['branch'][:, T_BUS].real.astype(int)
    used, rows = set(), []
    els = [('Line', j, r_['name'], int(r_.from_bus), int(r_.to_bus)) for j, r_ in aux.line.iterrows() if r_.in_service]
    els += [('Transformer', j, r_['name'], int(r_.hv_bus), int(r_.lv_bus)) for j, r_ in aux.trafo.iterrows() if r_.in_service]
    for kind, j, nm, a, b in els:
        ia, ib = int(lk[a]), int(lk[b])
        cand = [k for k in np.where(((f_ == ia) & (t_ == ib)) | ((f_ == ib) & (t_ == ia)))[0] if k not in used]
        if not cand:
            continue
        k = int(cand[0]); used.add(k)
        rows.append((kind, nm, a, b, k, f_[k] == ia))
    return rows


def fault_analysis(data, line, d_km, fault_type='BG', rf=0.0, prefault_pu=1.0, breaker_s=BREAKER_S):
    """A real fault applied to the network model (as a short-circuit study in ETAP): fault `fault_type` (AG, BC, ABG, ABC ...)
    on `line` at d_km from its From Bus. Symmetrical components on the whole network: the fault current, the current in every
    phase of every line and transformer, the phase voltages at every bus, the current from every grid, the operation of every
    relay (distance 21 with the real loop impedances, line differential 87L, overcurrent 51 / earth fault 51N) and the
    sequence of tripping with the coordination margins. Pre-fault voltage `prefault_pu` (1.0 = nominal), loads neglected."""
    from pandapower.shortcircuit.toolbox import calc_sc_on_line
    import logging
    logging.getLogger('pandapower').setLevel(logging.ERROR)
    ft = str(fault_type).upper().replace('-', '').replace(' ', '')
    if ft in ('1PH', '2PH', '3PH'):
        ft = {'1PH': 'AG', '2PH': 'BC', '3PH': 'ABC'}[ft]
    if ft not in PHASE_FAULTS:
        raise ValueError(f'Unknown fault type {fault_type}')
    kind, special = PHASE_FAULTS[ft]
    d, _ = filled(data)
    L = d['Lines'].set_index('Line')
    if line not in L.index:
        raise ValueError(f'Line {line} not found')
    r = L.loc[line]
    length = float(r['Length km'])
    pos = min(max(float(d_km) / length, 0.001), 0.999)
    net = build(data, 'sc')
    ix = int(net.line.index[net.line.name == line][0])
    aux, fb = calc_sc_on_line(net, ix, pos, fault='1ph', case='max')
    p1, p0, f = _seq_matrices(aux, fb)
    Z1, Z0 = p1['internal']['Zbus'], p0['internal']['Zbus']
    base = p1['baseMVA']
    vn = aux.bus.vn_kv.to_dict()
    lk = aux._pd2ppc_lookups['bus']
    zb = vn[fb] ** 2 / base
    zf = float(rf or 0.0) / zb
    E = complex(prefault_pu)
    z1, z2, z0 = Z1[f, f], Z1[f, f], Z0[f, f]
    if kind == '3ph':
        I1, I2, I0 = E / (z1 + zf), 0j, 0j
    elif kind == '1ph':
        I1 = I2 = I0 = E / (z1 + z2 + z0 + 3 * zf)
    elif kind == '2ph':
        I1 = E / (z1 + z2 + zf); I2, I0 = -I1, 0j
    else:
        zp = z2 * (z0 + 3 * zf) / (z2 + z0 + 3 * zf)
        I1 = E / (z1 + zp); I2 = -I1 * (z0 + 3 * zf) / (z2 + z0 + 3 * zf); I0 = -I1 * z2 / (z2 + z0 + 3 * zf)
    V1 = E - Z1[:, f] * I1
    V2 = -Z1[:, f] * I2
    V0 = -Z0[:, f] * I0
    ibase = lambda b: base / (np.sqrt(3) * vn[b])       # kA per pu at a bus
    If = _phases(I0, I1, I2, special)
    Ik = max(abs(v) for v in If.values()) * ibase(fb)
    bn = aux.bus.name.to_dict(); bn[fb] = 'Fault point'
    # bus voltages
    brows = []
    for b in aux.bus.index:
        if b == fb or b not in lk.keys() if hasattr(lk, 'keys') else False:
            continue
        k = int(lk[b])
        if k < 0 or k >= len(V1):
            continue
        ph = _phases(V0[k], V1[k], V2[k], special)
        vmin = min(abs(v) for v in ph.values())
        brows.append({'Bus': bn[b], 'kV': vn[b], 'Va (pu)': round(abs(ph['A']), 3), 'Vb (pu)': round(abs(ph['B']), 3), 'Vc (pu)': round(abs(ph['C']), 3),
                      'Lowest (pu)': round(vmin, 3), 'Dip (%)': round(100 * (1 - vmin / abs(E)), 1),
                      'Status': 'ACTION' if vmin < 0.5 else ('WATCH' if vmin < 0.9 else 'NORMAL')})
    buses = pd.DataFrame(brows).sort_values('Lowest (pu)', ignore_index=True) if brows else pd.DataFrame()
    # branch currents (each end, into the branch)
    Yf1, Yt1, Yf0, Yt0 = p1['internal']['Yf'], p1['internal']['Yt'], p0['internal']['Yf'], p0['internal']['Yt']
    i1f, i1t, i2f, i2t = Yf1 @ V1, Yt1 @ V1, Yf1 @ V2, Yt1 @ V2
    i0f, i0t = Yf0 @ V0, Yt0 @ V0
    rows, ends = [], {}
    A_, B_ = r['From Bus'], r['To Bus']
    for kd, nm, a, b, k, same in _branch_rows(aux, p1):
        fa = (i0f[k], i1f[k], i2f[k]) if same else (i0t[k], i1t[k], i2t[k])
        fb_ = (i0t[k], i1t[k], i2t[k]) if same else (i0f[k], i1f[k], i2f[k])
        for bus_, s_ in ((a, fa), (b, fb_)):
            ph = _phases(*s_, special)
            vph = _phases(V0[int(lk[bus_])], V1[int(lk[bus_])], V2[int(lk[bus_])], special) if bus_ != fb else None
            ends[(nm, bn.get(bus_, bus_))] = {'I': {p: v * ibase(bus_) for p, v in ph.items()}, 'I0': 3 * s_[0] * ibase(bus_),
                                              'V': {p: v * vn[bus_] / np.sqrt(3) for p, v in vph.items()} if vph else None}
        Ia = ends[(nm, bn.get(a, a))]['I']
        Ib = ends[(nm, bn.get(b, b))]['I']
        name = nm
        if nm.startswith(line) and '_aux_line' in str(nm):
            name = f"{line} ({bn.get(a, a) if a != fb else bn.get(b, b)} side)"
        rows.append({'Branch': name, 'Type': kd, 'From': bn.get(a, 'fault point'), 'To': bn.get(b, 'fault point'),
                     'Ia (kA)': round(abs(Ia['A']), 2), 'Ib (kA)': round(abs(Ia['B']), 2), 'Ic (kA)': round(abs(Ia['C']), 2),
                     '3I0 (kA)': round(abs(ends[(nm, bn.get(a, a))]['I0']), 2), 'Highest (kA)': round(max(abs(v) for v in Ia.values()), 2)})
    branches = pd.DataFrame(rows).sort_values('Highest (kA)', ascending=False, ignore_index=True)
    # currents from the two ends of the faulted line (into the fault)
    seg = {n_: (int(a_), int(b_)) for n_, a_, b_ in zip(aux.line['name'], aux.line.from_bus, aux.line.to_bus) if fb in (a_, b_)}
    end_I = {}
    for n_, (a_, b_) in seg.items():
        other = a_ if b_ == fb else b_
        e = ends.get((n_, bn[other]))
        if e:
            end_I[bn[other]] = max(abs(v) for v in e['I'].values())
    # current from every grid (sum of the branch currents leaving its bus)
    grids = []
    for _, g_ in aux.ext_grid.iterrows():
        gb = bn[g_.bus]
        tot = {p: sum(v['I'][p] for (n_, bb), v in ends.items() if bb == gb) for p in 'ABC'}
        grids.append({'Grid': g_['name'], 'Bus': gb, 'Highest phase current (kA)': round(max(abs(v) for v in tot.values()), 2)})
    grids = pd.DataFrame(grids).sort_values('Highest phase current (kA)', ascending=False, ignore_index=True) if grids else pd.DataFrame()
    # relays
    R = d['Relays']
    zl = {ln: (complex(rr['R1 ohm/km'], rr['X1 ohm/km']) * rr['Length km'], complex(rr['R0 ohm/km'], rr['X0 ohm/km']) * rr['Length km'])
          for ln, rr in L.iterrows()}
    rel = []
    for _, rr in R.iterrows():
        e, bus = rr['Element'], rr['Bus']
        key = None
        if e == line:
            for n_, (a_, b_) in seg.items():
                if bn.get(a_) == bus or bn.get(b_) == bus:
                    key = (n_, bus)
        else:
            key = (e, bus) if (e, bus) in ends else None
        if key is None or key not in ends:
            continue
        m = ends[key]
        I, I0, V = m['I'], m['I0'], m['V']
        fn = str(rr['Function']).upper()
        where = 'faulted line' if e == line else 'next line (back-up)'
        out = {'Relay': rr['Relay'], 'Bus': bus, 'Protects': e, 'Function': fn, 'Position': where,
               'Current (kA)': round(max(abs(v) for v in I.values()), 2), 'Seen': '', 'Operates': 'no', 'Relay time (s)': None}
        if fn == '87L':
            if e == line:
                out.update(Seen='internal fault', Operates='differential', **{'Relay time (s)': 0.0})
            else:
                out.update(Seen='external fault (stable)')
        elif fn == '21' and V and e in zl:
            z1l, z0l = zl[e]
            k0 = (z0l - z1l) / (3 * z1l) if abs(z1l) else 0
            loops = []
            for p in 'ABC':
                den = I[p] + k0 * I0
                if abs(den) > 0.05:
                    loops.append((p + 'N', V[p] / den))
            for p, q in (('A', 'B'), ('B', 'C'), ('C', 'A')):
                den = I[p] - I[q]
                if abs(den) > 0.05:
                    loops.append((p + q, (V[p] - V[q]) / den))
            zones = [(1, rr.get('Z1 %'), rr.get('T1 s')), (2, rr.get('Z2 %'), rr.get('T2 s')), (3, rr.get('Z3 %'), rr.get('T3 s'))]
            best = None
            for nm_, z in loops:
                for zn, pct, t in zones:
                    if pct is None or pd.isna(pct) or t is None or pd.isna(t):
                        continue
                    zr = float(pct) / 100 * z1l
                    if abs(z - zr / 2) <= abs(zr) / 2:          # inside the mho circle (forward)
                        if best is None or (zn, float(t)) < (best[1], best[2]):
                            best = (nm_, zn, float(t), z)
                        break
            zmin = min(loops, key=lambda x: abs(x[1])) if loops else None
            if best:
                out.update(Seen=f"loop {best[0]}: {abs(best[3]):.1f} Ω ∠{np.degrees(np.angle(best[3])):.0f}°", Operates=f'zone {best[1]}',
                           **{'Relay time (s)': best[2]})
            elif zmin:
                out.update(Seen=f"loop {zmin[0]}: {abs(zmin[1]):.1f} Ω ∠{np.degrees(np.angle(zmin[1])):.0f}° (outside the zones / reverse)")
        elif fn in ('51', '51N', '50/51', '50N/51N'):
            ground = 'N' in fn
            ia = abs(I0) * 1000 if ground else max(abs(v) for v in I.values()) * 1000
            t = float(oc_time(ia, rr['Pickup A'], rr['TMS'], rr['Curve'], rr['Inst Pickup A'], rr['Inst Time s']))
            if np.isfinite(t):
                out.update(Seen=f"{'3I0' if ground else 'phase'} {ia / 1000:.2f} kA", Operates='earth fault' if ground else 'overcurrent',
                           **{'Relay time (s)': round(t, 3)})
            else:
                out.update(Seen=f"{'3I0' if ground else 'phase'} {ia / 1000:.2f} kA (below pick-up)")
        if out['Relay time (s)'] is not None:
            out['Trip time (s)'] = round(out['Relay time (s)'] + RELAY_OP_S + breaker_s, 3)
        rel.append(out)
    relays = pd.DataFrame(rel)
    # sequence of tripping
    seq, notes = [], []
    if len(relays):
        relays['Trip time (s)'] = relays.get('Trip time (s)')
        relays = relays.sort_values('Trip time (s)', na_position='last', ignore_index=True)
        op = relays[relays['Trip time (s)'].notna()]
        main = op[op['Protects'] == line].groupby('Bus')['Trip time (s)'].min()
        cleared = float(main.max()) if len(main) else None
        for _, o in op.iterrows():
            role = 'Main' if o['Protects'] == line and o['Trip time (s)'] == main.get(o['Bus']) else \
                   ('Second main / back-up on the faulted line' if o['Protects'] == line else 'Back-up (trips only if the main protection fails)')
            margin = None if role == 'Main' or cleared is None else round(o['Trip time (s)'] - cleared, 3)
            st_ = 'NORMAL' if role == 'Main' else ('NORMAL' if margin is not None and margin >= 0.2 else ('WATCH' if margin is not None and margin >= 0 else 'ACTION'))
            seq.append({'Step': len(seq) + 1, 'Time (s)': o['Trip time (s)'], 'Relay': o['Relay'], 'Bus': o['Bus'], 'Function': o['Function'],
                        'Operates': o['Operates'], 'Role': role, 'Margin after clearing (s)': margin, 'Status': st_})
        for b_ in (A_, B_):
            if b_ in main.index:
                notes.append(f"{b_} end trips at {main[b_] * 1000:.0f} ms.")
            else:
                notes.append(f"No relay at the {b_} end of {line} in the relay table: that end is not shown as tripping.")
        if cleared is not None:
            notes.append(f"The fault is cleared from all known ends in {cleared * 1000:.0f} ms (relay {RELAY_OP_S * 1000:.0f} ms and breaker {breaker_s * 1000:.0f} ms included).")
        bad = [s_ for s_ in seq if s_['Status'] == 'ACTION']
        if bad:
            notes.append('Coordination problem: ' + ', '.join(f"{s_['Relay']} at {s_['Bus']} would trip before or with the main protection" for s_ in bad) + '.')
        elif any(s_['Role'].startswith('Back-up') for s_ in seq):
            notes.append('The back-up protections are graded after the main protection.')
    sequence = pd.DataFrame(seq)
    return {'line': line, 'from': A_, 'to': B_, 'length_km': length, 'd_km': pos * length, 'pos': pos, 'type': ft, 'kind': kind,
            'fault': f"{FAULT_NAME[kind]} ({ft})", 'rf': rf, 'prefault_pu': prefault_pu, 'Ik (kA)': round(Ik, 2),
            'end_I': {k: round(v, 2) for k, v in end_I.items()}, 'buses': buses, 'branches': branches, 'grids': grids,
            'relays': relays, 'sequence': sequence, 'notes': notes}
