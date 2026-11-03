"""NetworkGuard PDF report: network study with the single-line diagram and the results of every analysis that was run."""
import io
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

import networkguard_core as ng  # noqa: E402
import networkguard_plots as npl  # noqa: E402
import pg_paths  # noqa: E402

NAVY = colors.HexColor('#0b2e59'); BLUE = colors.HexColor('#2f80ed'); GRID = colors.HexColor('#d5deea')
ST_COL = {'EXCELLENT': colors.HexColor('#1a9e5c'), 'WATCH': colors.HexColor('#d99a00'), 'ACTION': colors.HexColor('#d63a3a'),
          'PASS': colors.HexColor('#1a9e5c'), 'CHECK': colors.HexColor('#d63a3a'), 'OUT': colors.HexColor('#9aa5b5')}
ST_LAB = {'EXCELLENT': 'NORMAL', 'WATCH': 'WATCH', 'ACTION': 'ACTION', 'PASS': 'PASS', 'CHECK': 'CHECK', 'OUT': 'OUT'}
_ss = getSampleStyleSheet()
H1 = ParagraphStyle('h1', parent=_ss['Title'], fontSize=19, textColor=NAVY, alignment=TA_LEFT, spaceAfter=2)
H2 = ParagraphStyle('h2', parent=_ss['Heading2'], fontSize=13, textColor=NAVY, spaceBefore=8, spaceAfter=5)
SUB = ParagraphStyle('sub', parent=_ss['Normal'], fontSize=9, textColor=colors.HexColor('#5b6b82'))
TXT = ParagraphStyle('txt', parent=_ss['Normal'], fontSize=8.6, leading=11)
CELL = ParagraphStyle('cell', parent=_ss['Normal'], fontSize=7, leading=8.6)
HEAD = ParagraphStyle('head', parent=CELL, textColor=colors.white, fontName='Helvetica-Bold')
FOOT = '© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved. Confidential and proprietary.'


def _logo(h=15 * mm):
    try:
        from reportlab.lib.utils import ImageReader
        ir = ImageReader(pg_paths.LOGO); w, hh = ir.getSize()
        return Image(pg_paths.LOGO, width=h * w / hh, height=h)
    except Exception:
        return Spacer(1, h)


def _page(c, doc):
    c.saveState(); w, h = doc.pagesize
    c.setFillColor(NAVY); c.rect(0, h - 7 * mm, w, 7 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont('Helvetica-Bold', 8)
    c.drawString(12 * mm, h - 4.8 * mm, 'PowerGuard AI  |  NetworkGuard AI: Network Analysis')
    c.drawRightString(w - 12 * mm, h - 4.8 * mm, doc.title)
    c.setFillColor(colors.HexColor('#7a879b')); c.setFont('Helvetica', 7)
    c.drawString(12 * mm, 7 * mm, FOOT); c.drawRightString(w - 12 * mm, 7 * mm, f'Page {doc.page}')
    c.restoreState()


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (float, np.floating)):
        return f'{v:,.3f}'.rstrip('0').rstrip('.') if abs(v) < 100 else f'{v:,.1f}'
    return str(v)


def _table(df, W, status_col='Status', widths=None, max_rows=None):
    df = df.head(max_rows) if max_rows else df
    data = [[Paragraph(str(c), HEAD) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(f"<font color='white'><b>{ST_LAB.get(v, v)}</b></font>" if c == status_col and v in ST_COL else _fmt(v), CELL) for c, v in r.items()])
    widths = widths or [W / len(df.columns)] * len(df.columns)
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [('BACKGROUND', (0, 0), (-1, 0), NAVY), ('GRID', (0, 0), (-1, -1), 0.4, GRID), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f8fc')]), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    if status_col in df.columns:
        j = list(df.columns).index(status_col)
        for i, v in enumerate(df[status_col], start=1):
            if v in ST_COL:
                st.append(('BACKGROUND', (j, i), (j, i), ST_COL[v]))
    t.setStyle(TableStyle(st))
    return t


def _kpis(items, W):
    lab = ParagraphStyle('kl', parent=TXT, fontSize=7, leading=9, textColor=colors.HexColor('#5b6b82'))
    val = ParagraphStyle('kv', parent=TXT, fontSize=14, leading=17)
    t = Table([[[Paragraph(k.upper(), lab), Paragraph(f'<b>{v}</b>', val)] for k, v in items]], colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 0.5, GRID), ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID), ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7f9fd')),
                           ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    return t


def _img(fig, W, H=None, dpi=170):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(buf).getSize(); buf.seek(0)
    sc = min(W / w, (H or 1e9) / h)
    return Image(buf, width=w * sc, height=h * sc)


def _tcc_mpl(relays, faults):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for _, r in relays.iterrows():
        I, t = ng.oc_curve_points(r)
        ax.loglog(I, np.where(np.isfinite(t), t, np.nan), lw=1.6, label=f"{r['Relay']} ({r['Curve']}, {r['Pickup A']:.0f} A, TMS {r['TMS']})")
    for lab, i in faults:
        ax.axvline(i, color='#94a3b8', ls=':', lw=1); ax.text(i, 60, lab, fontsize=6, rotation=90, va='top')
    ax.set_ylim(0.01, 100); ax.set_xlabel('Current (A primary)'); ax.set_ylabel('Time (s)'); ax.grid(True, which='both', alpha=0.25); ax.legend(fontsize=6)
    return fig


VERIF_TEXT = [
    ('Purpose', 'This document shows that the NetworkGuard AI calculation engine gives the results published for standard test systems and '
                'the results of the standard formulas. It checks the engine, not a particular network: the results of a network study depend on '
                'the equipment data entered for that network.'),
    ('Load flow', 'Newton-Raphson load flow of the IEEE 9-bus (Anderson and Fouad) and IEEE 14-bus test systems, compared with their published '
                  'solutions (bus voltage magnitudes and angles).'),
    ('Short circuit', "IEC 60909 equivalent voltage source method: a 400 kV grid of 20 000 MVA (R/X 0.1, X0/X1 1.0, R0/X0 0.1) feeding a 100 km line "
                      "(R1 0.03, X1 0.32, R0 0.2, X0 0.9 ohm/km). Ik3'' = c·Un/(√3·|Z1|), Ik1'' = √3·c·Un/|2Z1 + Z0|, ip = κ·√2·Ik3'' with "
                      "κ = 1.02 + 0.98·e^(-3R/X), c = 1.1, computed by hand and compared."),
    ('IEC TR 60909-4 example', 'The example network of IEC TR 60909-4 (data for calculations according to IEC 60909): 380/110/30/10 kV with two '
                               'network feeders, two-winding and three-winding transformers, power station units with generators, asynchronous '
                               'motors and overhead lines and cables. The three-phase (maximum and minimum) and two-phase initial short-circuit '
                               "currents Ik'', the peak currents ip (method C) and the short-circuit powers Sk'' at every bus are compared with "
                               'the results published in the report.'),
    ('Relay curves', 'IEC 60255-151 inverse-time equation t = TMS·k/((I/Is)^α - 1): standard inverse (k 0.14, α 0.02) and very inverse '
                     '(k 13.5, α 1) at ten times the pickup with TMS 0.1.'),
    ('Acceptance', 'Voltages within 0.0015 pu, angles within 0.02°, currents within 0.002 kA, powers within 0.05 MVA and times within 0.002 s of the reference.'),
]


def _verification_body(va, W, appendix=False):
    out = [Paragraph(('Appendix: ' if appendix else '') + 'Verification of the calculation engine', H2)]
    for t, x in VERIF_TEXT:
        out += [Paragraph(f'<b>{t}.</b> {x}', TXT), Spacer(1, 3)]
    n = int((va['Result'] == 'PASS').sum())
    out += [Spacer(1, 4), Paragraph(f'<b>Result: {n} of {len(va)} checks pass.</b>', TXT), Spacer(1, 4), _table(va, W, status_col='Result')]
    return out


def verification_pdf(va):
    """Stand-alone verification document of the calculation engine (given to a client on request)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm, topMargin=13 * mm, bottomMargin=13 * mm,
                            title='Verification of the Calculation Engine')
    W = doc.width
    hd = Table([[[Paragraph('NetworkGuard AI: Verification of the Calculation Engine', H1),
                  Paragraph(f"Issued {datetime.now().strftime('%d %b %Y')}", SUB)], _logo()]], colWidths=['78%', '22%'])
    hd.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (1, 0), (1, 0), 'RIGHT'), ('LINEBELOW', (0, 0), (-1, 0), 1.2, BLUE)]))
    doc.build([hd, Spacer(1, 6)] + _verification_body(va, W), onFirstPage=_page, onLaterPages=_page)
    return buf.getvalue()


def study_pdf(data, res, title='Network Study', network_name='Network', engineer=''):
    """res: dict with any of 'lf', 'vd', 'sc', 'prot', 'cont', 'val'."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm, topMargin=13 * mm, bottomMargin=13 * mm, title=title)
    W = doc.width
    s = []
    hd = Table([[[Paragraph(title, H1), Paragraph(f"{network_name} · Issued {datetime.now().strftime('%d %b %Y %H:%M')}" + (f' · {engineer}' if engineer else ''), SUB)],
                 _logo()]], colWidths=['78%', '22%'])
    hd.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (1, 0), (1, 0), 'RIGHT'), ('LINEBELOW', (0, 0), (-1, 0), 1.2, BLUE)]))
    s += [hd, Spacer(1, 6)]
    d = ng.clean(data)
    s.append(_kpis([('Substations', len(d['Substations'])), ('Buses', len(d['Buses'])), ('Circuits', len(d['Lines'])), ('Transformers', len(d['Transformers'])),
                    ('Reactors', len(d['Reactors'])), ('Grids', len(d['Grids'])), ('Relays', len(d['Relays']))], W))
    lf, scr = res.get('lf'), res.get('sc')
    s.append(Paragraph('Single-line diagram' + (' (load flow results)' if lf else ''), H2))
    s.append(_img(npl.sld_matplotlib(data, lf=lf, mode='lf' if lf else 'network'), W, 118 * mm))
    _, notes = ng.filled(data)
    miss = ng.missing_required(data)
    s.append(Paragraph('Data used', H2))
    s.append(Paragraph(f'{len(notes)} value(s) were filled with typical or computed values and {len(miss)} required value(s) are missing. '
                       'Results based on typical values are indicative and must be confirmed with the equipment data.', TXT))
    if notes:
        s.append(_table(pd.DataFrame(notes, columns=['Table', 'Equipment', 'Parameter', 'Value used']), W, max_rows=40, widths=[W * .12, W * .16, W * .25, W * .47]))
    if lf:
        s += [PageBreak(), Paragraph('Load flow (Newton-Raphson)', H2)]
        sm = lf['summary']
        s.append(_kpis([('Losses', f"{sm['Losses (MW)']} MW ({sm['Losses (%)']}%)"), ('Min V', f"{sm['Min V (pu)']} pu"), ('Max V', f"{sm['Max V (pu)']} pu"),
                        ('Max line loading', f"{sm['Max Line Loading (%)']}%"), ('Max transformer', f"{sm['Max Transformer Loading (%)']}%"),
                        ('Violations', sm['Violations'])], W))
        st_ = lf.get('settings', {})
        s.append(Paragraph(f"Case: load scaling {st_.get('load_scale', 1) * 100:.0f} %, outages: {', '.join(st_.get('outages', [])) or 'none'}, "
                           f"reactors switched out: {', '.join(st_.get('reactors_off', [])) or 'none'}.", SUB))
        s.append(Paragraph('Bus voltages', H2))
        s.append(_table(lf['bus'][['Bus', 'Substation', 'kV', 'V (pu)', 'V (kV)', 'Angle (deg)', 'Vmin', 'Vmax', 'Status']], W))
        s.append(Paragraph('Circuit flows', H2))
        s.append(_table(lf['line'][['Line', 'From', 'To', 'Length (km)', 'P From (MW)', 'Q From (MVAr)', 'I (A)', 'Rating (A)', 'Loading (%)', 'Losses (MW)', 'Status']], W))
        if len(lf['trafo']):
            s.append(Paragraph('Transformers', H2))
            s.append(_table(lf['trafo'][['Transformer', 'HV Bus', 'LV Bus', 'P HV (MW)', 'Q HV (MVAr)', 'Loading (%)', 'Losses (MW)', 'Status']], W))
        s.append(Paragraph('Sources, converters and reactors', H2))
        s.append(_table(lf['sources'], W, status_col=None))
    vd = res.get('vd')
    if vd:
        s += [PageBreak(), Paragraph('Voltage drop', H2)]
        for n in vd['notes']:
            s.append(Paragraph('• ' + n, TXT))
        s.append(_table(vd['line'], W))
    if scr:
        s += [PageBreak(), Paragraph(f"Short circuit (IEC 60909, {scr['case']} case)", H2)]
        s.append(_img(npl.sld_matplotlib(data, scr=scr, mode='sc', fault=(scr.get('contrib') or {}).get('bus')), W, 95 * mm))
        s.append(_table(scr['bus'][['Bus', 'Substation', 'kV', 'Ik3 (kA)', 'Ik1 (kA)', 'Sk3 (MVA)', 'ip (kA)', 'Ith 1 s (kA)', 'X/R', 'Breaking (kA)',
                                    'Breaker Duty (%)', 'Making Duty (%)', 'Short-time Duty (%)', 'Status']], W))
        s.append(Paragraph('Breaker duty = highest of Ik3 and Ik1 against the rated breaking current; making duty = ip against 2.5 × breaking current (50 Hz); '
                           'short-time duty = Ith (1 s) against the rated short-time current. NORMAL ≤ 90 %, WATCH 90-100 %, ACTION > 100 %.', SUB))
        c = scr.get('contrib')
        if c:
            s.append(Paragraph(f"Fault at {c['bus']}: {c['fault']} {c['Ik (kA)']} kA - branch currents", H2))
            s.append(_table(c['branches'], W, status_col=None))
    pr = res.get('prot')
    if pr:
        s += [PageBreak(), Paragraph('Protection coordination', H2)]
        if len(pr['distance']):
            s.append(Paragraph('Distance protection (zone reaches and times)', H2))
            s.append(_table(pr['distance'][['Relay', 'Line', 'Line Z (ohm)', 'Z1 (ohm)', 'Z2 (ohm)', 'Z3 (ohm)', 'Z1 sec (ohm)', 'Z2 sec (ohm)', 'Z3 sec (ohm)',
                                             'T2 (s)', 'T3 (s)', 'Status', 'Notes']], W,
                            widths=[W * x for x in (.08, .06, .05, .05, .05, .05, .05, .05, .05, .04, .04, .06, .37)]))
        if len(pr['oc_pairs']):
            s.append(Paragraph(f"Overcurrent and earth-fault grading (required margin {pr['cti']} s)", H2))
            s.append(_table(pr['oc_pairs'], W))
        if pr.get('tcc') is not None and len(pr['tcc'][0]):
            s.append(_img(_tcc_mpl(*pr['tcc']), W, 85 * mm))
    ct = res.get('cont')
    if ct:
        s += [PageBreak(), Paragraph('N-1 contingency', H2), _table(ct['table'], W)]
    fs = res.get('fs')
    if fs:
        s += [PageBreak(), Paragraph(f"Fault study: {fs['line']}", H2),
              _kpis([('Fault', f"{fs['fault']} at {fs['d_km']:.1f} km"), ('Total', f"{fs['Ik (kA)']:.2f} kA"), (f"From {fs['from']}", f"{fs['I_from (kA)']:.2f} kA"),
                     (f"From {fs['to']}", f"{fs['I_to (kA)']:.2f} kA")], W)]
        if fs.get('estimate'):
            s.append(Paragraph(f"Location that best matches the measured currents: {fs['estimate']['loc_km']:.1f} km from {fs['from']}.", TXT))
        if len(fs['compare']):
            s += [Paragraph('Calculated against measured', H2), _table(fs['compare'], W)]
        s.append(_img(npl.sld_matplotlib(data, fault_line=(fs['line'], fs.get('pos', 0.5)), zoom=True,
                                          fault_label=f"{fs['fault']} fault: {fs['d_km']:.1f} km from {fs['from']}"), W, 105 * mm))
        if len(fs['relays']):
            s += [Paragraph('Expected relay operation', H2), _table(fs['relays'], W, status_col=None)]
        if len(fs['buses']):
            s += [Paragraph('Voltage during the fault', H2), _table(fs['buses'], W, status_col=None)]
    va = res.get('val')
    if va is not None:
        s += [PageBreak()] + _verification_body(va, W, appendix=True)
    s += [Spacer(1, 8), Paragraph('Methods: load flow by Newton-Raphson; short circuit by IEC 60909 (equivalent voltage source, c = 1.1 max / 1.0 min, line '
                                  'capacitances and loads neglected); relay times by IEC 60255-151 / IEEE C37.112; line parameters from conductor and tower '
                                  'geometry (GMD / GMR, Carson earth return, earth wires neglected).', SUB)]
    doc.build(s, onFirstPage=_page, onLaterPages=_page)
    return buf.getvalue()
