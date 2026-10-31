"""NetworkGuard data import: the single-line diagram from a PDF drawing, and equipment data from relay settings files
(PDF text, scanned PDF or picture by OCR, Excel, CSV, TXT).

Reading a drawing or a settings export is a proposal: the app shows what was found, the engineer checks it and applies it."""
import io
import math
import re

import numpy as np
import pandas as pd

import networkguard_core as ng

def device_of(txt):
    m = re.search(r'model\s*(?:number|no\.?)\s*:?\s*([A-Z]\d{3}[A-Z0-9]*)', txt, re.I)
    return m.group(1)[:4].upper() if m else ''



IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def _ocr_image(im):
    """Text from a picture of a printout or a screen (Tesseract). Upscales small pictures for better reading."""
    import pytesseract
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(im).convert('L')
    if im.width < 1800:
        f = 1800 / im.width
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    im = ImageOps.autocontrast(im)
    return pytesseract.image_to_string(im, config='--psm 6')



def extract_text(data, name=''):
    """Plain text of an uploaded file: PDF (text layer, or OCR when scanned), picture (OCR), CSV / TXT, Excel.
    Returns (text, how) where how says which reader was used."""
    low = name.lower()
    if low.endswith('.pdf') or data[:4] == b'%PDF':
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text(extraction_mode='layout') or '') for p in r.pages]
        txt = '\n'.join(pages)
        if len(re.sub(r'\s', '', txt)) >= 40 * max(1, len(pages)) * 0.3:
            return txt, 'PDF text'
        try:                                    # scanned PDF: render the pages and read them
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(data)
            out = []
            for i in range(min(len(doc), 60)):
                out.append(_ocr_image(doc[i].render(scale=300 / 72).to_pil()))
            return '\n'.join(out), 'scanned PDF (OCR)'
        except Exception as e:
            return txt, f'PDF text (OCR not available: {e})'
    if low.endswith(IMAGE_EXT) or data[:8] == b'\x89PNG\r\n\x1a\n' or data[:3] == b'\xff\xd8\xff':
        from PIL import Image
        return _ocr_image(Image.open(io.BytesIO(data))), 'picture (OCR)'
    if low.endswith(('.xlsx', '.xls')):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        rows = []
        for df in sheets.values():
            for r in df.astype(str).replace('nan', '').values.tolist():
                rows.append('  '.join(x for x in r if x))
        return '\n'.join(rows), 'Excel'
    for enc in ('utf-8-sig', 'utf-16', 'cp1252', 'latin-1'):
        try:
            return data.decode(enc), 'text'
        except Exception:
            continue
    return '', 'unreadable'



_NUM = r'([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(k|m|M)?\s*(ohms?|Ω|ohm|km|mi|m|%|deg|°|kv|v|ka|a|ms|s)?(?![a-z])'
SET_KEYS = [
    ('line_length', r'line\s*length'),
    ('line_angle', r'line\s*angle'),
    ('line_z', r'line\s*imped\w*|line\s*z1?\b(?!\s*angle)'),
    ('kzn_angle', r'k\s*z\s*n\d?\s*(?:res\w*\s*)?ang\w*|k0\s*ang\w*|kz0?\s*ang\w*'),
    ('kzn', r'k\s*z\s*n\d?\s*(?:res\w*\s*)?(?:comp\w*|magn?\w*|factor)?(?!\s*(?:res\w*\s*)?ang)|residual\s*comp\w*\s*factor|\bk0\b(?!\s*ang)'),
    ('kzm', r'k\s*z\s*m\s*(?:mutual\s*)?(?:comp\w*|set\w*)?(?!\s*ang)'),
    ('z1_reach', r'z(?:one)?\s*1\s*(?:ph\w*|gnd|ground|g|p)?\.?\s*reach|z1\s*(?:ph|gnd)?\s*reach'),
    ('z2_reach', r'z(?:one)?\s*2\s*(?:ph\w*|gnd|ground|g|p)?\.?\s*reach|z2\s*(?:ph|gnd)?\s*reach'),
    ('z3_reach', r'z(?:one)?\s*3\s*(?:ph\w*|gnd|ground|g|p)?\.?\s*reach|z3\s*(?:ph|gnd)?\s*reach'),
    ('r1_ph', r'r\s*1\s*ph\w*\.?\s*res\w*'), ('r2_ph', r'r\s*2\s*ph\w*\.?\s*res\w*'), ('r3_ph', r'r\s*3\s*ph\w*\.?\s*res\w*'),
    ('r1_gnd', r'r\s*1\s*g(?:n)?d\w*\.?\s*res\w*'), ('r2_gnd', r'r\s*2\s*g(?:n)?d\w*\.?\s*res\w*'), ('r3_gnd', r'r\s*3\s*g(?:n)?d\w*\.?\s*res\w*'),
    ('tilt', r'z\s*1\s*tilt\w*(?:\s*top\s*line)?'),
    ('z1_time', r't\s*z(?:one)?\s*1\s*(?:ph\w*|gnd|ground)?\.?\s*(?:delay|time)'),
    ('z3_time', r't\s*z(?:one)?\s*3\s*(?:ph\w*|gnd|ground)?\.?\s*(?:delay|time)'),
    ('z4_time', r't\s*z(?:one)?\s*4\s*(?:ph\w*|gnd|ground)?\.?\s*(?:delay|time)'),
    ('z2_time', r't\s*z(?:one)?\s*2\s*(?:ph\w*|gnd|ground)?\.?\s*(?:delay|time)?|z(?:one)?\s*2\s*(?:ph\w*|gnd)?\s*(?:time\s*)?delay'),
    ('ct_primary', r'(?:phase\s*)?ct\s*prim\w*'),
    ('ct_secondary', r'(?:phase\s*)?ct\s*sec\w*'),
    ('vt_primary', r'(?:main\s*)?vt\s*prim\w*'),
    ('vt_secondary', r'(?:main\s*)?vt\s*sec\w*'),
]
SET_LABEL = {'line_length': 'Line length (km)', 'line_z': 'Line impedance (Ω)', 'line_angle': 'Line angle (°)', 'kzn': 'kZN', 'kzn_angle': 'kZN angle (°)',
             'kzm': 'kZm', 'z1_reach': 'Zone 1 reach', 'z2_reach': 'Zone 2 reach', 'z2_time': 'Zone 2 time (ms)', 'ct_primary': 'CT primary (A)',
             'ct_secondary': 'CT secondary (A)', 'vt_primary': 'VT primary (V)', 'vt_secondary': 'VT secondary (V)', 'values_in': 'Setting values',
             'auto_reclose': 'Auto-reclose', 'device': 'Relay', 'description': 'Description'}


def _num_unit(text):
    m = re.search(_NUM, text, re.I)
    if not m:
        return None, None
    v = float(m.group(1).replace(',', '.'))
    pre, u = (m.group(2) or ''), (m.group(3) or '').lower()
    if pre == 'k':
        v *= 1000
    elif pre == 'm' and u not in ('', 'm'):
        v /= 1000
    elif pre == 'M':
        v *= 1e6
    return v, u


def parse_settings(text):
    """Line and protection settings from a relay settings export (text, PDF text or OCR). Tolerant of layouts:
    'Line Length   120.0 km', '0C01 Line Length: 120.0km', 'kZN Res Comp 800.0E-3' ..."""
    out = {'device': device_of(text)}
    m = re.search(r'description\s*[:=]?\s*(.+)', text, re.I)
    if m:
        out['description'] = m.group(1).strip()[:60]
    for k, pat in (('char_ph', r'phase\s*char\w*\.?\s*[:=]?\s*(quad\w*|mho)'), ('char_gnd', r'(?:ground|gnd|earth)\s*char\w*\.?\s*[:=]?\s*(quad\w*|mho)')):
        m = re.search(pat, text, re.I)
        if m:
            out[k] = 'quad' if m.group(1).lower().startswith('quad') else 'mho'
    m = re.search(r'setting\s*values\s*[:=]?\s*(primary|secondary)', text, re.I)
    if m:
        out['values_in'] = m.group(1).title()
    m = re.search(r'auto[\s\-]*reclos\w*\s*(?:mode|status)?\s*[:=]?\s*(enabled|disabled|in\s*service|out\s*of\s*service|on|off)\b', text, re.I)
    if m:
        out['auto_reclose'] = 'In service' if m.group(1).lower() in ('enabled', 'on', 'in service', 'inservice') else 'Out of service'
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        for key, pat in SET_KEYS:
            if key in out:
                continue
            m = re.search(r'(?:^|[\s:.\d])(' + pat + r')\s*[:=]?\s*', s, re.I)
            if not m:
                continue
            v, u = _num_unit(s[m.end():])
            if v is None:
                continue
            if key == 'line_length':
                v = v * 1.609 if u == 'mi' else (v / 1000 if u == 'm' and v > 2000 else v)
            if key in ('z1_time', 'z2_time', 'z3_time', 'z4_time'):
                v = v * 1000 if (u == 's' or (not u and v < 5)) else v
            if key in ('z1_reach', 'z2_reach'):
                out[key + '_unit'] = '%' if u == '%' else 'ohm'
            if key in ('kzn',) and v > 5:            # a percentage written without a unit
                v = v / 100
            out[key] = v
            break
    return out


def settings_line(sts, ct=None, vt=None):
    """Line data (per-km Z1/Z0) and protection settings from the parsed settings of one relay. Returns (line, setting, notes)."""
    line, setg, notes = {}, {}, []
    L = sts.get('line_length')
    if L:
        line['length_km'] = float(L)
    ctr = (sts['ct_primary'] / sts['ct_secondary']) if sts.get('ct_primary') and sts.get('ct_secondary') else ct
    vtr = (sts['vt_primary'] / sts['vt_secondary']) if sts.get('vt_primary') and sts.get('vt_secondary') else vt
    if sts.get('vt_primary'):
        line['kv'] = sts['vt_primary'] / 1000 if sts['vt_primary'] > 1000 else sts['vt_primary']
    Zl = sts.get('line_z')
    if Zl and L:
        if sts.get('values_in') == 'Secondary':
            if ctr and vtr:
                Zl = Zl * vtr / ctr; notes.append(f"Line impedance converted from secondary with CT {ctr:g} and VT {vtr:g}.")
            else:
                notes.append('Line impedance is in secondary ohms but the CT/VT ratios were not found: enter them under settings.')
        ang = np.radians(sts.get('line_angle', 85.0))
        z1 = Zl / L * np.exp(1j * ang)
        k = sts.get('kzn', 1.0 / 3 * 2)
        z0 = z1 * (1 + 3 * k * np.exp(1j * np.radians(sts.get('kzn_angle', 0.0))))
        line.update({'r1': float(z1.real), 'x1': float(z1.imag), 'r0': float(z0.real), 'x0': float(z0.imag)})
        if 'kzn' not in sts:
            notes.append('kZN not found in the settings: 0.67 assumed.')
    for z in ('z1', 'z2'):
        r = sts.get(z + '_reach')
        if r:
            if sts.get(z + '_reach_unit') == '%':
                setg[z + '_reach'] = r / 100
            elif Zl:
                setg[z + '_reach'] = r / sts['line_z']          # same base (primary or secondary) as the line impedance
    for n in (1, 2, 3, 4):
        if sts.get(f'z{n}_time') is not None:
            setg[f'z{n}_time_ms'] = float(sts[f'z{n}_time'])
    k = (vtr / ctr) if (sts.get('values_in') == 'Secondary' and ctr and vtr) else 1.0
    if sts.get('z3_reach') and sts.get('line_z'):
        setg['z3_reach'] = sts['z3_reach'] / 100 if sts.get('z3_reach_unit') == '%' else sts['z3_reach'] / sts['line_z']
    for kind in ('ph', 'gnd'):
        if sts.get('char_' + kind):
            setg['char_' + kind] = sts['char_' + kind]
        rr = [sts.get(f'r{i}_{kind}') for i in (1, 2, 3)]
        if any(x is not None for x in rr):
            setg['r_' + kind] = [None if x is None else float(x) * k for x in rr]
    if sts.get('tilt') is not None:
        setg['tilt_deg'] = float(sts['tilt'])
    if sts.get('auto_reclose'):
        setg['ar_expected'] = sts['auto_reclose'] == 'In service'
    if ctr:
        setg['_ct'] = ctr
    if vtr:
        setg['_vt'] = vtr
    return line, setg, notes


def best_settings(all_sts):
    """The settings file that carries the line data (normally the distance relay)."""
    score = lambda s: sum(k in s for k in ('line_length', 'line_z', 'line_angle', 'kzn', 'z1_reach')) * 10 + len(s)
    return max(all_sts, key=lambda kv: score(kv[1])) if all_sts else None


# ================================================================ relay settings -> network data
def _strs(P):
    for c in ('Current', 'Proposed'):
        if c in P:
            P[c] = P[c].map(lambda v: '' if v is None or (isinstance(v, float) and np.isnan(v)) else (f'{v:g}' if isinstance(v, (int, float, np.floating)) else str(v)))
    return P


def guess_line(sts, data):
    """The line a settings file belongs to, from its description (e.g. '21A SILAA L12'): best matching line name."""
    ln, sc_ = ng.match_line(sts.get('description') or '', data)
    if ln and sc_ >= 2:
        return ln
    d = ng.clean(data)
    desc = (sts.get('description') or '').upper()
    toks = [t for t in re.split(r'[^A-Z0-9]+', desc) if len(t) >= 2]
    best, score = None, 0
    for _, r in d['Lines'].iterrows():
        name = ' '.join([r['Line'], r['From Bus'], r['To Bus']]).upper()
        s = sum(2 if t in name else (1 if t[:3] in name else 0) for t in toks)
        if s > score:
            best, score = r['Line'], s
    return best


def guess_bus(sts, ends):
    """The line end of the relay: the end whose name appears in the relay description (e.g. '21A SILAA L12' -> Silla)."""
    toks = [t for t in re.split(r'[^A-Z]+', (sts.get('description') or '').upper()) if len(t) >= 3]
    for e in ends:
        name = re.sub(r'[^A-Z]', '', e.upper())
        if any(t[:3] in name for t in toks):
            return e
    return ends[0]


def settings_proposal(sts, data, line, bus, relay=None):
    """Rows (Table, Equipment, Parameter, Current, Proposed, From) to update a line and its relay from one relay settings file."""
    d = ng.clean(data)
    ln, sg, notes = settings_line(sts)
    rows = []

    def cur(tab, eid, f):
        df = d[tab]
        r = df[df[ng.id_col(tab)] == eid]
        return None if r.empty else r.iloc[0][f]

    if line:
        for f, k in (('Length km', 'length_km'), ('R1 ohm/km', 'r1'), ('X1 ohm/km', 'x1'), ('R0 ohm/km', 'r0'), ('X0 ohm/km', 'x0')):
            if k in ln:
                rows.append({'Table': 'Lines', 'Equipment': line, 'Parameter': f, 'Current': cur('Lines', line, f), 'Proposed': round(float(ln[k]), 5),
                             'From': 'line length' if k == 'length_km' else 'line impedance, angle and kZN of the relay'})
    dev = sts.get('device', '') or ''
    func = '21'
    if dev.upper().startswith('P54') and not sts.get('z2_reach'):
        func = '87L'
    relay = relay or (f"{line}-{func}{'A' if bus == cur('Lines', line, 'From Bus') else 'B'}" if line else f'{dev or "Relay"}-1')
    base = [('Element', line), ('Bus', bus), ('Function', func)]
    if sts.get('ct_primary'):
        base += [('CT Primary A', sts['ct_primary']), ('CT Secondary A', sts.get('ct_secondary', 1.0))]
    if sts.get('vt_primary'):
        base += [('VT Primary kV', sts['vt_primary'] / 1000 if sts['vt_primary'] > 1000 else sts['vt_primary']), ('VT Secondary V', sts.get('vt_secondary', 110.0))]
    for z, f in (('z1_reach', 'Z1 %'), ('z2_reach', 'Z2 %'), ('z3_reach', 'Z3 %')):
        if sg.get(z) is not None:
            base.append((f, round(100 * sg[z], 1)))
    for n, f in ((1, 'T1 s'), (2, 'T2 s'), (3, 'T3 s')):
        if sg.get(f'z{n}_time_ms') is not None:
            base.append((f, sg[f'z{n}_time_ms'] / 1000))
    for f, v in base:
        if v is None or v == '':
            continue
        rows.append({'Table': 'Relays', 'Equipment': relay, 'Parameter': f, 'Current': cur('Relays', relay, f), 'Proposed': v,
                     'From': f"{dev} settings" if dev else 'settings'})
    P = _strs(pd.DataFrame(rows))
    if len(P):
        P['Apply'] = True
    return P, notes + ([f"{dev}: line differential relay; the distance back-up zone 1 is kept."] if func == '87L' else [])


def equipment_values(text):
    """Transformer / reactor data found in a settings or test report text: MVA, MVAr, kV, impedance %, vector group, CT."""
    out = {}
    t = text
    m = re.search(r'(?:rated|reference|nominal|base)\s*(?:power|apparent\s*power|s\s*ref\w*|sn)\D{0,25}?(\d+(?:[.,]\d+)?)\s*MVA(?!r)', t, re.I)
    if m:
        out['MVA'] = float(m.group(1).replace(',', '.'))
    else:
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*MVA(?![rR])', t)
        if m:
            out['MVA'] = float(m.group(1).replace(',', '.'))
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*MVA[rR]', t)
    if m:
        out['MVAr'] = float(m.group(1).replace(',', '.'))
    kvs = sorted({float(x) for x in re.findall(r'(\d{2,3}(?:\.\d+)?)\s*kV', t, re.I) if 6 <= float(x) <= 800}, reverse=True)
    if kvs:
        out['HV kV'] = kvs[0]
        if len(kvs) > 1:
            out['LV kV'] = kvs[1]
    m = re.search(r'(?:impedance\s*voltage|short[\s-]*circuit\s*(?:voltage|impedance)|\bu\s*k\b|\bv\s*k\b|\bz\s*%|percent\s*impedance)\D{0,20}?(\d+(?:[.,]\d+)?)\s*%', t, re.I)
    if m:
        out['uk %'] = float(m.group(1).replace(',', '.'))
    m = re.search(r'\b(YN|Y|D)(yn|y|d|zn|z)(\d{1,2})\b', t)
    if m:
        out['Vector Group'] = m.group(0)
    s = parse_settings(t)
    for k, f in (('ct_primary', 'CT Primary A'), ('ct_secondary', 'CT Secondary A')):
        if s.get(k):
            out[f] = s[k]
    if s.get('device'):
        out['Relay'] = s['device']
    return out


def equipment_proposal(vals, data, kind, eid):
    d = ng.clean(data)
    df = d[kind]
    r = df[df[ng.id_col(kind)] == eid]
    fields = {'Transformers': ['MVA', 'HV kV', 'LV kV', 'uk %', 'Vector Group'], 'Reactors': ['MVAr', 'kV']}[kind]
    rows = []
    for f in fields:
        src = 'HV kV' if (kind == 'Reactors' and f == 'kV') else f
        if src in vals:
            rows.append({'Table': kind, 'Equipment': eid, 'Parameter': f, 'Current': None if r.empty else r.iloc[0][f], 'Proposed': vals[src],
                         'From': 'equipment document', 'Apply': True})
    return _strs(pd.DataFrame(rows))


def apply_proposal(data, P):
    """Write the ticked proposal rows into the network tables (adds equipment that does not exist yet)."""
    for (tab, eid), g in P[P['Apply']].groupby(['Table', 'Equipment']):
        t = ng.element_table(data, tab, eid)
        for _, r in g.iterrows():
            t.loc[t['Parameter'] == r['Parameter'], 'Value'] = r['Proposed']
        data = ng.set_element(data, tab, eid, t)
    return data


# ================================================================ single-line diagram from a PDF drawing
def _words(data):
    """Words with positions (x0, y0, x1, y1; y upwards) of the largest page, the page size and a picture of the page."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(data)
    best = max(range(len(doc)), key=lambda i: doc[i].get_textpage().count_chars())
    pg = doc[best]
    W, H = pg.get_size()
    tp = pg.get_textpage()
    words = []
    for i in range(tp.count_rects()):
        b = tp.get_rect(i)
        t = tp.get_text_bounded(*b).strip()
        if t:
            words.append((t, b[0], b[1], b[2], b[3]))
    img = pg.render(scale=min(2.0, 2400 / max(W, H))).to_pil()
    how = 'PDF text'
    if len(words) < 20:                               # scanned drawing: read it by OCR
        try:
            import pytesseract
            big = pg.render(scale=300 / 72).to_pil().convert('L')
            od = pytesseract.image_to_data(big, output_type=pytesseract.Output.DICT, config='--psm 11')
            k = W / big.width
            words = [(od['text'][i].strip(), od['left'][i] * k, H - (od['top'][i] + od['height'][i]) * k,
                      (od['left'][i] + od['width'][i]) * k, H - od['top'][i] * k) for i in range(len(od['text'])) if od['text'][i].strip()]
            how = 'scanned drawing (OCR)'
        except Exception as e:
            how = f'scanned drawing, OCR not available ({e})'
    buf = io.BytesIO(); img.save(buf, format='PNG')
    return words, W, H, buf.getvalue(), how


def _c(w):
    return ((w[1] + w[3]) / 2, (w[2] + w[4]) / 2)


def _compact(name):
    s = re.sub(r'\bSUBSTATION\b.*', '', name.upper())
    s = re.sub(r'\b(AL|EL|NEW)\b[\s-]*', '', s)
    return re.sub(r'[^A-Z]', '', s)


def sld_from_pdf(data):
    """Network proposal from a single-line diagram drawing: substations (name and voltages), circuits from the circuit names
    (e.g. 41_SALWA_SILLA = circuit 1 Salwa-Silla), shunt reactors (RB/RL), power transformers (T4xx), neighbouring grids
    (circuits to stations outside the drawing, connections to other networks) and HVDC converters.
    Returns (tables, log, page png)."""
    words, W, H, png, how = _words(data)
    log = [f'Read as {how}: {len(words)} text items.']
    # ---- substations: '<NAME> SUBSTATION' with the voltage written next to it
    subs = []
    for w in words:
        m = re.search(r'^([A-Z][A-Z .\-]*?)\s*SUBSTATION$', w[0].strip())
        if m and m.group(1).strip() and 'CONNECTION' not in w[0].upper():
            cx, cy = _c(w)
            kv = []
            for v in words:
                vx, vy = _c(v)
                mm = re.fullmatch(r'(\d{2,3})(?:\s*-\s*(\d{2,3}))?\s*kV', v[0].strip(), re.I)
                if mm and abs(v[1] - w[1]) < 30 and 0 < vy - cy < 20:
                    kv = [float(mm.group(1))] + ([float(mm.group(2))] if mm.group(2) else [])
            name = re.sub(r'\s+', ' ', m.group(1).strip()).title()
            subs.append({'name': name, 'x': cx, 'y': cy, 'kv': kv or [400.0], 'codes': set(), 'compact': _compact(m.group(1))})
    # code labels such as 4SALWA under the name
    for w in words:
        mm = re.fullmatch(r'\d([A-Z]{3,8})', w[0].strip())
        if mm and not mm.group(1).startswith('HVDC'):
            cx, cy = _c(w)
            near = [s for s in subs if abs(s['x'] - cx) < 60 and -40 < cy - s['y'] < 40]
            if near:
                s = min(near, key=lambda s: math.hypot(s['x'] - cx, s['y'] - cy))
                s['codes'].add(mm.group(1))
    log.append(f"{len(subs)} substations: " + ', '.join(f"{s['name']} ({'/'.join(f'{k:g}' for k in s['kv'])} kV)" for s in subs))
    # ---- circuits from names
    circ = []
    for w in words:
        mm = re.fullmatch(r'(\d{1,2})_([A-Z0-9]+)_([A-Z0-9]+)', w[0].strip())
        if mm and 'SPARE' not in w[0]:
            circ.append((int(mm.group(1)), mm.group(2), mm.group(3), _c(w)))
    seen = {}
    for n, a, b, pos in circ:
        seen.setdefault((n, a, b), pos)
    circ = [(n, a, b, p) for (n, a, b), p in seen.items()]
    codes = {}
    for s in subs:
        for c in s['codes']:
            codes[c] = s
    count = {}
    for n, a, b, _ in circ:
        for c in (a, b):
            count[c] = count.get(c, 0) + 1
    for c in sorted(count, key=lambda c: -count[c]):
        if c in codes:
            continue
        cand = [s for s in subs if not s['codes'] and (s['compact'].startswith(c[:3]) or c.startswith(s['compact'][:3]))]
        if cand:
            cand[0]['codes'].add(c); codes[c] = cand[0]
    # ---- positions on the diagram (keep the drawing layout)
    xs = [s['x'] for s in subs] or [0]; ys = [s['y'] for s in subs] or [0]
    sx = 12.0 / max(1.0, max(xs) - min(xs)); sy = 12.0 / max(1.0, max(ys) - min(ys)); sc = min(sx, sy)
    S, B = [], []
    for s in subs:
        S.append({'Substation': s['name'], 'X': round((s['x'] - min(xs)) * sc, 2), 'Y': round((s['y'] - min(ys)) * sc, 2)})
        for kv in s['kv']:
            B.append({'Bus': f"{s['name']} {kv:g}", 'Substation': s['name'], 'kV': kv})
    hv = {s['name']: f"{s['name']} {max(s['kv']):g}" for s in subs}
    L, G, ext = [], [], {}
    for n, a, b, pos in sorted(circ):
        sa, sb = codes.get(a), codes.get(b)
        if sa and sb and sa is not sb:
            L.append({'Line': f'{n}_{a}_{b}', 'Other Names': f"{sa['name']}-{sb['name']}", 'From Bus': hv[sa['name']], 'To Bus': hv[sb['name']], 'Type': 'OHL'})
        else:
            inside = sa or sb
            other = b if inside is sa else a
            if inside is None:
                continue
            ext.setdefault((inside['name'], other), []).append(pos)
    # neighbouring grids: country written under the nearest 'CONNECTION TO' text
    countries = [w for w in words if re.fullmatch(r'(KUWAIT|IRAQ|UAE|QATAR|BAHRAIN|KSA|SAUDI ARABIA|OMAN|[A-Z ]+-QATAR)', w[0].strip().upper())]
    def country_near(x, y):
        if not countries:
            return ''
        w = min(countries, key=lambda w: math.hypot(_c(w)[0] - x, _c(w)[1] - y))
        t_ = w[0].strip().split('-')[-1]
        t_ = t_.upper() if len(t_) <= 3 else t_.title()
        return t_ if math.hypot(_c(w)[0] - x, _c(w)[1] - y) < 400 else ''
    grp = {}
    for (sname, other), ps in ext.items():
        x, y = ps[0]
        cn = country_near(x, y)
        grp.setdefault((sname, cn or other), []).append((other, len(ps)))
    for (sname, cn), items in grp.items():
        nm = cn if len(cn) > 3 else cn.upper()
        G.append({'Grid': f"Grid {nm} ({', '.join(o for o, _ in items)})" if cn not in [o for o, _ in items] else f'Grid {nm}',
                  'Bus': hv[sname], 'Circuits': sum(n for _, n in items)})
    for w in words:
        if re.search(r'CONNECTION TO THE (\d{3}) kV NETWORK', w[0], re.I):
            kv = float(re.search(r'(\d{3}) kV', w[0], re.I).group(1))
            x, y = _c(w)
            near = [s for s in subs if kv in s['kv'] and kv != max(s['kv'])]
            if near:
                s = min(near, key=lambda s: math.hypot(s['x'] - x, s['y'] - y))
                if math.hypot(s['x'] - x, s['y'] - y) < 500:
                    G.append({'Grid': f"Grid {country_near(x, y) or s['name']} {kv:g} kV", 'Bus': f"{s['name']} {kv:g}", 'Circuits': 0})
    for w in words:
        mm = re.search(r'(\d{3})\s*KV\s*NETWORK\s+([A-Z]+)', w[0].upper())
        if mm and 'CONNECTION' not in w[0].upper():
            kv = float(mm.group(1)); x, y = _c(w)
            near = [s for s in subs if kv in s['kv']]
            if near:
                s = min(near, key=lambda s: math.hypot(s['x'] - x, s['y'] - y))
                G.append({'Grid': f"Grid {mm.group(2).title()} {kv:g} kV", 'Bus': f"{s['name']} {kv:g}", 'Circuits': 0})
    Gd = pd.DataFrame(G) if G else pd.DataFrame(columns=['Grid', 'Bus', 'Circuits'])
    if len(Gd):   # one grid per lower-voltage bus (the 'connection to' text and the side label name the same network)
        lv = Gd['Circuits'] == 0
        Gd = pd.concat([Gd[~lv], Gd[lv].drop_duplicates('Bus')], ignore_index=True)

    # ---- reactors and transformers: group the equipment labels of the drawing into substation boxes
    eq = [w for w in words if re.fullmatch(r'\d{2}[A-Z]{2}\d{1,2}|R[BL]\d{2}|T\d{3}(?:\s*\(.*\))?|\d[AB]\d\s*-\s*\d+\s*kV', w[0].strip())]
    box_of = {}
    if len(eq) >= 3 and subs:
        from sklearn.cluster import DBSCAN
        P = np.array([_c(w) for w in eq])
        lab = DBSCAN(eps=50, min_samples=3).fit(P).labels_
        cl = {k: P[lab == k] for k in set(lab) if k >= 0}
        big = {k: v for k, v in cl.items() if len(v) >= 15} or cl
        for k, v in cl.items():                      # small fragments join the nearest large group
            if k not in big:
                c = v.mean(0)
                kk = min(big, key=lambda b: np.min(np.hypot(*(big[b] - c).T)))
                big[kk] = np.vstack([big[kk], v])
        bb = {k: (v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()) for k, v in big.items()}

        def dist(s, b):
            x0, y0, x1, y1 = b
            return math.hypot(max(x0 - s['x'], 0, s['x'] - x1), max(y0 - s['y'], 0, s['y'] - y1))
        pairs = sorted(((dist(s, b), i, k) for i, s in enumerate(subs) for k, b in bb.items()))
        used_s, used_k = set(), set()
        for dd, i, k in pairs:
            if i in used_s or k in used_k or dd > 250:
                continue
            used_s.add(i); used_k.add(k); box_of[k] = subs[i]
        log.append(f"{len(box_of)} substation boxes recognised on the drawing.")

        def owner(x, y):
            if box_of:
                k = min(box_of, key=lambda k: dist({'x': x, 'y': y}, bb[k]))
                return box_of[k]
            return min(subs, key=lambda s: math.hypot(s['x'] - x, s['y'] - y))
    else:
        def owner(x, y):
            return min(subs, key=lambda s: math.hypot(s['x'] - x, s['y'] - y))
    R, T = {}, {}
    for w in words:
        t = w[0].strip()
        x, y = _c(w)
        if re.fullmatch(r'R[BL]\d{2}', t):
            s = owner(x, y)
            R[(s['name'], t)] = 1
        m = re.fullmatch(r'(T\d{3})(?:\s*\(.*\))?', t)
        if m and not m.group(1).startswith('T00'):
            s = owner(x, y)
            if len(s['kv']) > 1:
                T[(s['name'], m.group(1))] = 1
    Rx = pd.DataFrame([{'Reactor': f'{s} {r}', 'Bus': hv[s], 'Line': '', 'kV': max(next(x['kv'] for x in subs if x['name'] == s))} for s, r in sorted(R)],
                      columns=['Reactor', 'Bus', 'Line', 'kV'])
    Tr = []
    for (s, t) in sorted(T):
        kv = sorted(next(x['kv'] for x in subs if x['name'] == s), reverse=True)
        Tr.append({'Transformer': f'{s} {t}', 'HV Bus': f'{s} {kv[0]:g}', 'LV Bus': f'{s} {kv[1]:g}', 'HV kV': kv[0], 'LV kV': kv[1]})
    Tr = pd.DataFrame(Tr, columns=['Transformer', 'HV Bus', 'LV Bus', 'HV kV', 'LV kV'])
    # ---- HVDC converters
    C = []
    conv = sorted({w[0].strip() for w in words if re.fullmatch(r'\d?HVDC\d', w[0].strip())})
    if conv:
        xs_ = [_c(w) for w in words if w[0].strip() in conv]
        cx, cy = np.mean([p[0] for p in xs_]), np.mean([p[1] for p in xs_])
        s = min(subs, key=lambda s: math.hypot(s['x'] - cx, s['y'] - cy))
        for c in conv:
            C.append({'Converter': c.lstrip('0123456789') if c[0].isdigit() else c, 'Bus': hv[s['name']]})
        log.append(f"{len(conv)} HVDC converter(s) placed at {s['name']} (nearest substation): check the connection.")
    log.append(f"{len(L)} circuits between the substations, {len(Gd)} neighbouring grid connection(s), {len(Rx)} shunt reactors, {len(Tr)} power transformers.")
    if ext:
        log.append('Circuits to stations outside the drawing are modelled as neighbouring grids: ' + ', '.join(f'{o} at {s}' for s, o in ext))
    log.append('Line lengths and impedances, reactor ratings, transformer data and grid fault levels are not in the drawing: '
               'add them on the equipment data page (forms, Excel or relay settings files).')
    tabs = {'Substations': pd.DataFrame(S), 'Buses': pd.DataFrame(B), 'Lines': pd.DataFrame(L), 'Transformers': Tr, 'Reactors': Rx,
            'Grids': Gd.assign(Slack=0), 'Converters': pd.DataFrame(C, columns=['Converter', 'Bus'])}
    if len(tabs['Grids']):
        tabs['Grids'].loc[tabs['Grids'].index[0], 'Slack'] = 1
    return tabs, log, png


def to_network(tabs):
    """Proposal tables (as edited by the engineer) -> network data."""
    keep = {k: v.drop(columns=[c for c in ('Circuit Code', 'Circuits') if c in v.columns]) for k, v in tabs.items()}
    return ng.clean(keep)
