"""GridGuard AI - PDF reports (transformer forecast, thermal check, combined report)."""
import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

NAVY = colors.HexColor('#0b2e59'); LINE = colors.HexColor('#d5dce6'); MUTED = colors.HexColor('#5a6575')
SCOL = {'EXCELLENT': colors.HexColor('#1a9e5c'), 'WATCH': colors.HexColor('#d99a00'), 'ACTION': colors.HexColor('#d63a3a')}
SLAB = {'EXCELLENT': 'NORMAL', 'WATCH': 'WATCH', 'ACTION': 'ACTION REQUIRED'}
W = A4[0] - 30 * mm

_ss = getSampleStyleSheet()
H1 = ParagraphStyle('h1', parent=_ss['Heading1'], fontName='Helvetica-Bold', fontSize=17, textColor=NAVY, spaceAfter=4)
H2 = ParagraphStyle('h2', parent=_ss['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=NAVY, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle('b', parent=_ss['BodyText'], fontName='Helvetica', fontSize=9.2, leading=12.5)
SMALL = ParagraphStyle('s', parent=BODY, fontSize=7.8, leading=10, textColor=MUTED)
CELL = ParagraphStyle('c', parent=BODY, fontSize=8, leading=10)
CELLB = ParagraphStyle('cb', parent=CELL, fontName='Helvetica-Bold', textColor=colors.white)

_MAP = {'Ω': 'ohm', '≥': '>=', '≤': '<=', '−': '-', '…': '...', '→': '->', '×': 'x', '÷': '/', '²': '2', '·': '-', '’': "'", '‘': "'",
        '“': '"', '”': '"', '–': '-', '—': '-', '±': '+/-'}


def clean(x):
    s = '' if x is None else str(x)
    for k, v in _MAP.items():
        s = s.replace(k, v)
    s = s.encode('latin-1', 'replace').decode('latin-1')
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def P(x, st=BODY):
    return Paragraph(clean(x), st)


def table(rows, widths, head=True, zebra=True, align_right=()):
    data = [[Paragraph(clean(c), CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, LINE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]
    if head:
        st.append(('BACKGROUND', (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1 if head else 0, len(rows)):
            if i % 2 == 0:
                st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f5f7fa')))
    t.setStyle(TableStyle(st))
    return t


def banner(text, sub, status):
    c = SCOL.get(status, NAVY)
    t = Table([[Paragraph(f"<font color='{c.hexval()}' size='14'><b>{clean(text)}</b></font>", BODY)], [P(sub)]], colWidths=[W])
    t.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1.4, c), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return t


def fig_image(fig, width=W):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    from PIL import Image as PI
    w, h = PI.open(buf).size; buf.seek(0)
    return Image(buf, width=width, height=width * h / w)


def _doc(title):
    buf = io.BytesIO()

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY); canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white); canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(15 * mm, A4[1] - 8 * mm, 'GridGuard AI'); canvas.setFont('Helvetica', 8.5)
        canvas.drawRightString(A4[0] - 15 * mm, A4[1] - 8 * mm, clean(title).replace('&amp;', '&'))
        canvas.setFillColor(MUTED); canvas.setFont('Helvetica', 7.5)
        canvas.drawString(15 * mm, 9 * mm, f"Generated {datetime.now():%Y-%m-%d %H:%M}  -  PowerGuard AI  -  Confidential")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=18 * mm, bottomMargin=16 * mm,
                            title=title, author='GridGuard AI')
    return buf, doc, deco


def _fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{d}f}" if abs(v) >= 1 else f"{v:.4g}"
    return str(v)

import pandas as pd
GREEN, AMBER, RED, NAVYH = '#1a9e5c', '#d99a00', '#d63a3a', '#0b2e59'


def forecast_charts(fc, hist_ts=None, hist_load=None, hs_limit=110):
    ts = pd.DatetimeIndex(fc['ts'])
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    ax = axs[0]
    if hist_ts is not None and len(hist_ts):
        ax.plot(pd.DatetimeIndex(hist_ts), hist_load, color='#7d8ca3', lw=1, label='Measured load')
    ax.fill_between(ts, fc['load_lo'], fc['load_hi'], color='#4da6ff', alpha=.22, lw=0, label='80% band')
    ax.plot(ts, fc['load'], color=NAVYH, lw=1.6, label='AI forecast')
    ax.axhline(fc['rating'], color=RED, ls='--', lw=1); ax.text(ts[0], fc['rating'], f" Rating {fc['rating']:g} MVA", color=RED, fontsize=8, va='bottom')
    ax.set_ylabel('Load (MVA)'); ax.legend(fontsize=8, ncol=3, loc='upper left', frameon=False); ax.grid(alpha=.25)
    ax = axs[1]
    ax.plot(ts, fc['hot_spot'], color=RED, lw=1.5, label='Hot-spot'); ax.plot(ts, fc['top_oil'], color=AMBER, lw=1.3, label='Top oil')
    ax.plot(ts, fc['ambient'], color='#2a78d6', lw=1, label='Ambient (forecast)')
    for y, c, t in [(hs_limit, AMBER, f'{hs_limit:.0f} C'), (130, RED, '130 C')]:
        ax.axhline(y, color=c, ls='--', lw=0.9); ax.text(ts[-1], y, f' {t}', color=c, fontsize=8, va='bottom', ha='right')
    ax.set_ylabel('Temperature (C)'); ax.legend(fontsize=8, ncol=3, loc='upper left', frameon=False); ax.grid(alpha=.25)
    fig.autofmt_xdate()
    return fig


def forecast_pdf(a, fc, reasons, hist_ts=None, hist_load=None, hs_limit=110, weather_src=''):
    buf, doc, deco = _doc('Transformer Load & Thermal Forecast')
    S = [P('Transformer Load & Thermal Forecast', H1), P(f"{a['Transformer ID']}  -  {a.get('Bay') or ''}  -  forecast from {pd.Timestamp(fc['origin']):%Y-%m-%d %H:%M}, "
                                                          f"{len(fc['ts']) // 24} days", SMALL), Spacer(1, 4)]
    S.append(banner(f"{SLAB[a['Status']]}  -  failure risk {a['Overall Risk (%)']:.0f}%", f"Governing mode: {a['Governing Mode']}.  {a['Action']}", a['Status']))
    S.append(P('Transformer and forecast', H2))
    S.append(table([['Item', 'Value', 'Item', 'Value'],
                    ['Rating', f"{a['Rating (MVA)']:g} MVA, {a['Cooling']}", 'Load type', a['Load Type']],
                    ['Peak load', f"{a['Peak Load (MVA)']:.1f} MVA ({a['Peak Load (%)']:.0f}%) {pd.Timestamp(a['Peak Time']):%a %d %b %H:00}", 'Hours above rating', str(a['Hours Above Rating'])],
                    ['Max hot-spot', f"{a['Max Hot-Spot (C)']:.0f} C {pd.Timestamp(a['Hot-Spot Time']):%a %d %b %H:00}", 'Max top oil', f"{a['Max Top-Oil (C)']:.0f} C"],
                    ['Max ambient', f"{a['Max Ambient (C)']:.0f} C", 'Worst day ageing', f"{a['Worst Day Ageing (x)']:.2f} x normal"],
                    ['Load transfer to keep hot-spot <= limit', f"{a['Load Transfer Needed (MVA)']:.1f} MVA", 'Weather forecast', weather_src or '-']],
                   [W * 0.22, W * 0.32, W * 0.2, W * 0.26]))
    S.append(KeepTogether([P('Forecast', H2), fig_image(forecast_charts(fc, hist_ts, hist_load, hs_limit))]))
    S.append(P('Why this result', H2))
    for r in reasons:
        S.append(P(f"- {r}"))
    S.append(P('FMEA', H2))
    rows = [['Failure mode', 'Risk %', 'S', 'O', 'D', 'RPN', 'Level']] + [[m['Failure Mode'], f"{m['Risk (%)']:.0f}", m['S'], m['O'], m['D'], m['RPN'], m['RPN Level']] for m in a['FMEA']]
    S.append(table(rows, [W * 0.34, W * 0.12, W * 0.08, W * 0.08, W * 0.08, W * 0.12, W * 0.18]))
    S.append(Spacer(1, 6))
    S.append(P('Load forecast by the GridGuard AI model (gradient boosting on weather, calendar, load type and recent load). Temperatures by the IEEE C57.91 '
               'clause 7 thermal model; ageing by the IEEE C57.91 ageing acceleration factor (1.0 at a 110 C hot-spot). RPN = S x O x D.', SMALL))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


def combined_pdf(R, summary, source, n_a, n_w):
    buf, doc, deco = _doc('Substation Loading Forecast Report')
    S = [P('Substation Loading Forecast Report', H1), P(f"{len(R)} transformers  -  source: {source}", SMALL), Spacer(1, 4)]
    st = 'ACTION' if n_a else 'WATCH' if n_w else 'EXCELLENT'
    S.append(banner(f"{n_a} action  -  {n_w} watch  -  {len(R) - n_a - n_w} normal", summary, st))
    S.append(P('All transformers', H2))
    rows = [['Transformer', 'Rating', 'Peak %', 'Max HS C', 'Governing mode', 'Risk %', 'Status', 'RPN', 'Transfer MVA']]
    for _, r in R.iterrows():
        rows.append([r['Transformer ID'], f"{r['Rating (MVA)']:g} MVA", f"{r['Peak Load (%)']:.0f}", f"{r['Max Hot-Spot (C)']:.0f}", r['Governing Mode'],
                     f"{r['Overall Risk (%)']:.0f}", r['Status'], str(r['RPN']), f"{r['Load Transfer Needed (MVA)']:.1f}"])
    t = table(rows, [W * x for x in [0.11, 0.1, 0.08, 0.09, 0.2, 0.08, 0.12, 0.07, 0.15]])
    for i, (_, r) in enumerate(R.iterrows(), 1):
        t.setStyle(TableStyle([('TEXTCOLOR', (6, i), (6, i), SCOL[r['Status']])]))
    S.append(t)
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(R) + 1.1))
    rr = R.sort_values('Overall Risk (%)')
    ax.barh(rr['Transformer ID'], rr['Overall Risk (%)'], color=[{'EXCELLENT': GREEN, 'WATCH': AMBER, 'ACTION': RED}[s] for s in rr['Status']])
    for x in (35, 70):
        ax.axvline(x, color='#888', ls='--', lw=1)
    ax.set_xlim(0, 105); ax.set_xlabel('Failure risk (%)'); ax.grid(axis='x', alpha=.25)
    S.append(KeepTogether([P('Risk ranking', H2), fig_image(fig, W * 0.9)]))
    above = R[R['Status'] != 'EXCELLENT']
    if len(above):
        S.append(P('Transformers above the normal band', H2))
        for _, r in above.iterrows():
            block = [Paragraph(f"<b>{clean(r['Transformer ID'])}</b>  -  {clean(r['Governing Mode'])}  -  {r['Overall Risk (%)']:.0f}% {r['Status']}", BODY)]
            block += [P(f"- {f}") for f in r['Findings']]
            block += [P(f"RPN = {r['S']} x {r['O']} x {r['D']} = {r['RPN']} ({r['RPN Level']}).  Action: {r['Action']}", SMALL), Spacer(1, 5)]
            S.append(KeepTogether(block))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()
