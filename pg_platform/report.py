"""PDF reports of the platform: Fleet Status Report (all modules) and Asset 360 Report (one site asset)."""
import io
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import pg_paths

NAVY = colors.HexColor('#0b2e59')
BLUE = colors.HexColor('#2f80ed')
GRID = colors.HexColor('#d5deea')
ST_COL = {'EXCELLENT': colors.HexColor('#16a34a'), 'WATCH': colors.HexColor('#e6a100'), 'ACTION': colors.HexColor('#dc2626')}
ST_LAB = {'EXCELLENT': 'NORMAL', 'WATCH': 'WATCH', 'ACTION': 'ACTION'}

_ss = getSampleStyleSheet()
H1 = ParagraphStyle('h1', parent=_ss['Title'], fontSize=19, textColor=NAVY, alignment=TA_LEFT, spaceAfter=2)
H2 = ParagraphStyle('h2', parent=_ss['Heading2'], fontSize=12.5, textColor=NAVY, spaceBefore=10, spaceAfter=5)
SUB = ParagraphStyle('sub', parent=_ss['Normal'], fontSize=9.5, textColor=colors.HexColor('#5b6b82'))
TXT = ParagraphStyle('txt', parent=_ss['Normal'], fontSize=8.6, leading=10.8)
CELL = ParagraphStyle('cell', parent=_ss['Normal'], fontSize=7.6, leading=9.2)
HEAD = ParagraphStyle('head', parent=CELL, textColor=colors.white, fontName='Helvetica-Bold')
FOOT = '© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved. Confidential and proprietary.'


def _logo(h=16 * mm):
    try:
        from reportlab.lib.utils import ImageReader
        ir = ImageReader(pg_paths.LOGO); w, hh = ir.getSize()
        return Image(pg_paths.LOGO, width=h * w / hh, height=h)
    except Exception:
        return Spacer(1, h)


def _page(canvas, doc):
    canvas.saveState()
    w, h = doc.pagesize
    canvas.setFillColor(NAVY); canvas.rect(0, h - 7 * mm, w, 7 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white); canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(12 * mm, h - 4.8 * mm, 'PowerGuard AI  |  Asset Intelligence Platform')
    canvas.drawRightString(w - 12 * mm, h - 4.8 * mm, doc.title)
    canvas.setFillColor(colors.HexColor('#7a879b')); canvas.setFont('Helvetica', 7)
    canvas.drawString(12 * mm, 7 * mm, FOOT)
    canvas.drawRightString(w - 12 * mm, 7 * mm, f'Page {doc.page}')
    canvas.restoreState()


def _table(df, widths, status_col=None, max_rows=None):
    df = df.head(max_rows) if max_rows else df
    data = [[Paragraph(str(c), HEAD) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(f"<font color='white'><b>{ST_LAB.get(v, v)}</b></font>" if c == status_col and v in ST_COL else ('-' if pd.isna(v) else str(v)), CELL)
                     for c, v in r.items()])
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [('BACKGROUND', (0, 0), (-1, 0), NAVY), ('GRID', (0, 0), (-1, -1), 0.4, GRID), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f8fc')]),
          ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if status_col in df.columns:
        j = list(df.columns).index(status_col)
        for i, v in enumerate(df[status_col], start=1):
            if v in ST_COL:
                st += [('BACKGROUND', (j, i), (j, i), ST_COL[v]), ('TEXTCOLOR', (j, i), (j, i), colors.white)]
    t.setStyle(TableStyle(st))
    return t


def _kpis(items, width):
    lab = ParagraphStyle('kl', parent=TXT, fontSize=7, leading=9, textColor=colors.HexColor('#5b6b82'))
    cells = [[[Paragraph(k.upper(), lab), Paragraph(f"<font color='{c}'><b>{v}</b></font>", ParagraphStyle('kv', parent=TXT, fontSize=15, leading=18))]
              for k, v, c in items]]
    t = Table(cells, colWidths=[width / len(items)] * len(items))
    t.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 0.5, GRID), ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID),
                           ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7f9fd')), ('TOPPADDING', (0, 0), (-1, -1), 7),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 7), ('LEFTPADDING', (0, 0), (-1, -1), 8)]))
    return t


def _header(title, sub):
    t = Table([[[Paragraph(title, H1), Paragraph(sub, SUB)], _logo()]], colWidths=['75%', '25%'])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (1, 0), (1, 0), 'RIGHT'), ('LINEBELOW', (0, 0), (-1, 0), 1.2, BLUE),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    return t


def fleet_pdf(F, R, source_label, site='Demo substation'):
    """Fleet Status Report. F: module results, R: site asset register."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm, topMargin=13 * mm, bottomMargin=13 * mm,
                            title='Fleet Status Report')
    W = doc.width
    now = datetime.now().strftime('%d %b %Y %H:%M')
    s = [_header('Fleet Status Report', f'{site}  ·  Source: {source_label}  ·  Issued {now}'), Spacer(1, 8)]
    n = len(R); act = int((R['Status'] == 'ACTION').sum()); wat = int((R['Status'] == 'WATCH').sum())
    health = round(100 - R['Combined Risk (%)'].mean(), 1) if n else 100
    s.append(_kpis([('Site assets', n, '#0b2e59'), ('Action', act, '#dc2626'), ('Watch', wat, '#e6a100'), ('Normal', n - act - wat, '#16a34a'),
                    ('Fleet health index', f'{health}%', '#2f80ed'), ('Module results', len(F), '#0b2e59')], W))
    s.append(Paragraph('Status by module', H2))
    rows = []
    for m, g in F.groupby('Module'):
        w = g.sort_values('Risk (%)', ascending=False).iloc[0]
        rows.append({'Module': m, 'Results': len(g), 'Action': int((g['Status'] == 'ACTION').sum()), 'Watch': int((g['Status'] == 'WATCH').sum()),
                     'Normal': int((g['Status'] == 'EXCELLENT').sum()), 'Highest risk': f"{w['Asset ID']} ({w['Risk (%)']:.0f}%)", 'Main driver': w['Main Driver']})
    s.append(_table(pd.DataFrame(rows), [W * x for x in (.14, .07, .07, .07, .07, .16, .42)]))
    A = R[R['Status'] != 'EXCELLENT'][['Site Asset', 'Name', 'Modules', 'Combined Risk (%)', 'Status', 'Main Driver', 'Action']]
    s.append(Paragraph(f'Assets needing attention ({len(A)})', H2))
    s.append(_table(A, [W * x for x in (.07, .15, .17, .07, .07, .22, .25)], status_col='Status') if len(A) else Paragraph('No asset needs attention.', TXT))
    s.append(Paragraph('Site asset register', H2))
    s.append(_table(R[['Site Asset', 'Name', 'Type', 'Modules', 'Combined Risk (%)', 'Status', 'Main Driver']],
                    [W * x for x in (.08, .19, .08, .25, .08, .08, .24)], status_col='Status'))
    I = F[F['Module'] == 'FaultGuard']
    if len(I) and 'Incident' in I:
        s.append(Paragraph('Trips and incidents (FaultGuard)', H2))
        cols = [c for c in ['Incident', 'Date', 'Asset ID', 'Fault Type', 'Risk (%)', 'Status', 'Summary'] if c in I]
        s.append(_table(I[cols], [W * x for x in (.07, .09, .07, .07, .06, .07, .57)][:len(cols)], status_col='Status'))
    s.append(Spacer(1, 8))
    s.append(Paragraph('Combined risk of a site asset = highest module risk + 5 % for every other module above 35 %. '
                       'NORMAL ≤ 35 %, WATCH 35-70 %, ACTION > 70 %.', SUB))
    doc.build(s, onFirstPage=_page, onLaterPages=_page)
    return buf.getvalue()


def asset_pdf(sid, name, typ, rows, cr, status, extra=None):
    """Asset 360 Report of one site asset. rows: its module results."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f'Asset 360 Report - {sid}')
    W = doc.width
    s = [_header(f'Asset 360 Report: {name}', f"{typ}  ·  Site asset {sid}  ·  Issued {datetime.now().strftime('%d %b %Y %H:%M')}"), Spacer(1, 8)]
    s.append(_kpis([('Combined risk', f'{cr:.0f}%', '#dc2626' if status == 'ACTION' else ('#e6a100' if status == 'WATCH' else '#16a34a')),
                    ('Status', ST_LAB.get(status, status), '#0b2e59'), ('Modules', rows['Module'].nunique(), '#2f80ed')], W))
    s.append(Paragraph('Module results', H2))
    s.append(_table(rows[['Module', 'Asset ID', 'Risk (%)', 'Status', 'Main Driver', 'Action']], [W * x for x in (.15, .1, .08, .1, .27, .3)], status_col='Status'))
    for title, text in (extra or []):
        s.append(KeepTogether([Paragraph(title, H2), Paragraph(text, TXT)]))
    s.append(Spacer(1, 8))
    s.append(Paragraph('Combined risk = highest module risk + 5 % for every other module above 35 %.', SUB))
    doc.build(s, onFirstPage=_page, onLaterPages=_page)
    return buf.getvalue()
