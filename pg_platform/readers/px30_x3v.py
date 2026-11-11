"""Reader for MiCOM Px30 settings files (.x3v: P63x, P13x, P43x ...).

The file holds fixed 7-byte records: group (1 byte), address (1 byte), value
(signed 32-bit little-endian), flag (1 byte); the parameter address is
'ggg.aaa' as shown in the relay menu (e.g. 019.016 = Reference power Sref).
The records carry no names or scaling, so a name/scale map is built once from a
'Settings File Report' (PDF text) printed by Easergy Studio for the same model
and version, and then reused for every file of that model. Read-only.

    m   = build_map(report_text(pdf_text), records(x3v_bytes_of_the_same_file))
    out = decode(records(other_x3v_bytes), m)   # {'019.016': ('Reference power Sref', 650.0, 'MVA')}
"""
import re
import struct

TRANSFORMER_KEYS = [('sref_mva', '019.016'), ('v_end_a_kv', '019.017'), ('v_end_b_kv', '019.018'),
                    ('v_end_c_kv', '019.019'), ('v_end_d_kv', '019.037'), ('ct_a_a', '019.020'),
                    ('ct_b_a', '019.021'), ('ct_c_a', '019.022'), ('ct_d_a', '019.026'),
                    ('vecgr_a_b', '019.010'), ('vecgr_a_c', '019.011'), ('vecgr_a_d', '019.014'),
                    ('idiff_pickup_iref', '072.142'), ('idiff_hs1_iref', '072.143'), ('idiff_hs2_iref', '072.144'),
                    ('slope_m1', '072.145'), ('slope_m2', '072.146'), ('ir_m2_iref', '072.147'),
                    ('inrush_2nd_pct', '072.159'), ('overflux_5th_pct', '072.160')]


def records(data):
    """{'ggg.aaa': raw int} for every record (first occurrence wins)."""
    i = data.find(b'\x00\x00\x7a\x02\x00\x00\x00')      # 000.000 device type 634 aligns the grid
    start = i % 7 if i >= 0 else 0
    out = {}
    for k in range(start, len(data) - 6, 7):
        out.setdefault(f'{data[k]:03d}.{data[k + 1]:03d}', struct.unpack('<i', data[k + 2:k + 6])[0])
    return out


def report_text(txt):
    """{'ggg.aaa': (name, shown value)} from the text of an Easergy 'Settings File Report'."""
    d = {}
    for m in re.finditer(r'(\d{3}\.\d{3}):\s*(.+?):\s{1,}(\S.*?)\s*$', txt, re.M):
        d[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return d


def _num(s):
    m = re.match(r'^(-?\d+(?:\.\d+)?)', s)
    return float(m.group(1)) if m else None


def build_map(report, recs):
    """Name, scale and unit (or option text) of each address, learnt from one report + its file."""
    out = {}
    for a, (name, shown) in report.items():
        if a not in recs:
            continue
        x, raw = _num(shown), recs[a]
        unit = shown[len(shown.split()[0]):].strip() if shown.split() else ''
        if x is not None:
            out[a] = dict(name=name, kind='num', scale=(x / raw) if raw else None, unit=unit)
        else:
            out[a] = dict(name=name, kind='enum', enum={raw: shown})
    return out


def decode(recs, mp):
    out = {}
    for a, m in mp.items():
        if a not in recs:
            continue
        raw = recs[a]
        if m['kind'] == 'num':
            out[a] = (m['name'], round(raw * m['scale'], 6) if m['scale'] else raw, m['unit'])
        else:
            out[a] = (m['name'], m['enum'].get(raw, f'code {raw}'), '')
    return out


def transformer_data(recs, mp):
    """Rating, voltages, CT ratios, vector groups and differential settings of a P63x."""
    d = decode(recs, mp)
    return {k: d.get(a, ('', None, ''))[1] for k, a in TRANSFORMER_KEYS}
