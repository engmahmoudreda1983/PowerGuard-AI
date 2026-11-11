"""Reader for MiCOM S1 Agile / S1 Studio settings files (.set, Courier format).

Covers P44x, P54x, P14x, P64x and other Courier relays. The file stores one record
per setting cell: column/row address, the menu text, a type code, then the value
with its range. Floats are a 16-bit sign-magnitude mantissa and a power-of-ten
exponent (offset 0x7E). The reader only reads; it never writes a settings file.

Typical use:
    cells = read_set('relay.set')          # {'3001': ('Line Length', 85800.0), ...}
    line  = line_data(cells)               # line length, Z1, Z0, CT/VT, kZN
"""
import cmath
import math
import re

_REC = re.compile(rb'(.)(.)(.)\x00(.)\x00\x00\x00\x18(.)', re.S)


def _kval(b4):
    m = b4[0] | (b4[1] << 8)
    sign = -1 if m & 0x8000 else 1
    return round(sign * (m & 0x7FFF) * 10.0 ** (b4[2] - 0x7E), 9)


def cells(data):
    """All setting cells in the file: list of dicts (addr, label, fmt, rest)."""
    out = []
    for mt in _REC.finditer(data):
        n = mt.group(5)[0]
        s = mt.end()
        raw = data[s:s + n]
        if not 1 < n < 80:
            continue
        i = raw.find(b'%')
        label = raw[:i] if i >= 0 else raw
        if len(label) < 2 or not all(32 <= c < 127 for c in label):
            continue
        out.append(dict(addr=f'{mt.group(2)[0]:02X}{mt.group(1)[0]:02X}',
                        label=label.decode().strip(),
                        fmt=raw[i:].decode('latin1') if i >= 0 else '',
                        rest=data[s + n:s + n + 400]))
    return out


def value(c):
    """Decoded value of one cell: float, int, text or the selected option; None if not decoded."""
    r, f = c['rest'], c['fmt']
    if f.startswith('%k') and r[:2] == b',\x04':
        return _kval(r[2:6])
    if f.startswith('%s') and r[:2] == b'P\x02':
        idx = r[2] | (r[3] << 8)
        j = r.find(b'\xff\x00\xff\x00\xff\x00')
        if j < 0:
            return idx
        while r[j:j + 2] == b'\xff\x00':
            j += 2
        opts = [o.strip().decode('latin1') for o in r[j:].split(b'\x00')]
        return opts[idx] if idx < len(opts) and opts[idx] else idx
    if f.startswith('%s') and r[:1] == b'\x18':
        return r[2:2 + r[1]].decode('latin1').strip()
    if f.startswith('%u') and r[:2] == b'$\x02':
        return r[2] | (r[3] << 8)
    return None


def read_set(path_or_bytes):
    """{address: (label, value)} for every decoded cell (first occurrence wins), plus '_model'."""
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, 'rb').read()
    head = data[:400].decode('latin1')
    m = re.search(r'MODEL:\s*(\S+)', head)
    out = {'_model': (('Model', m.group(1)) if m else ('Model', ''))}
    for c in cells(data):
        v = value(c)
        if v is not None:
            out.setdefault(c['addr'], (c['label'], v))
    return out


def _find(cs, pat, col=None, cols=None):
    for c in cs:
        a = int(c['addr'][:2], 16)
        if col is not None and c['addr'][:2] != col:
            continue
        if cols and not (cols[0] <= a <= cols[1]):
            continue
        if re.fullmatch(pat, c['label'], re.I):
            v = value(c)
            if v is not None:
                return v


def line_data(path_or_bytes):
    """Line parameters from a distance / line-differential relay settings file (group 1).

    Returns dict with relay info, CT/VT ratios and the line in primary ohms:
    length_km, z1_ohm, angle_deg, r1/x1/r0/x0 per km, kzn, kzn_angle. Zero-sequence
    impedance follows Z0 = Z1 (1 + 3 kZN) (IEEE C37.113 residual compensation).
    """
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, 'rb').read()
    cs = cells(data)
    g1 = (0x30, 0x4F)
    d = dict(description=_find(cs, r'Description'), plant_reference=_find(cs, r'Plant Reference'),
             setting_values=_find(cs, r'Setting Values'), active_group=_find(cs, r'Active Group'),
             vt_prim=_find(cs, r'Main VT Primary', col='0A'), vt_sec=_find(cs, r"Main VT Sec'?y", col='0A'),
             ct_prim=_find(cs, r'Phase ?CT1? Primary', col='0A'), ct_sec=_find(cs, r"Phase ?CT1? Sec'?y", col='0A'),
             length_m=_find(cs, r'Line Length', cols=g1), z=_find(cs, r'Line Impedance', cols=g1),
             angle_deg=_find(cs, r'Line Angle', cols=g1), kzn=_find(cs, r'kZN Res Comp', cols=g1),
             kzn_angle=_find(cs, r'kZN Res Angle', cols=g1))
    m = re.search(r'MODEL:\s*(\S+)', data[:400].decode('latin1'))
    d['model'] = m.group(1) if m else ''
    if not (d['length_m'] and d['z'] and d['angle_deg']):
        return d
    k = 1.0
    if d['setting_values'] == 'Secondary' and all(d[x] for x in ('vt_prim', 'vt_sec', 'ct_prim', 'ct_sec')):
        k = (d['vt_prim'] / d['vt_sec']) / (d['ct_prim'] / d['ct_sec'])
    km = d['length_m'] / 1000.0
    z1 = cmath.rect(d['z'] * k / km, math.radians(d['angle_deg']))
    kz = cmath.rect(d['kzn'] or 0.0, math.radians(d['kzn_angle'] or 0.0))
    z0 = z1 * (1 + 3 * kz)
    d.update(length_km=km, z1_ohm=d['z'] * k, r1=z1.real, x1=z1.imag, r0=z0.real, x0=z0.imag,
             defaults=(d['z'] == 10.0 and d['angle_deg'] == 70.0))
    return d
