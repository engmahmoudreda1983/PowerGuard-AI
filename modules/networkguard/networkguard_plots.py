"""NetworkGuard drawings: single-line diagram (Plotly for the page, matplotlib for the PDF), relay curves and distance zones."""
import math

import numpy as np
import plotly.graph_objects as go

import networkguard_core as ng

KV_COL = {400: '#d63a3a', 380: '#e07b00', 275: '#2a78d6', 220: '#1a9e5c', 132: '#7c3aed'}
ST_COL = {'EXCELLENT': '#1a9e5c', 'WATCH': '#d99a00', 'ACTION': '#d63a3a', 'OUT': '#9aa5b5'}
BUS_HALF = 0.42


def _top(lay, bus):
    """True for the highest bus of its substation (sources are drawn above it; below for lower buses)."""
    x, y = lay['pos'][bus]
    return not any(abs(xx - x) < 1e-6 and yy > y + 1e-6 for xx, yy in lay['pos'].values())


def kv_color(kv):
    return KV_COL.get(int(round(kv)), '#555555')


def _ends(lay, a, b, off):
    (x1, y1), (x2, y2) = lay['pos'][a], lay['pos'][b]
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    px, py = -dy / n * off, dx / n * off
    # leave the bus at the side facing the other end
    sx = max(-BUS_HALF * 0.8, min(BUS_HALF * 0.8, dx * 0.25))
    return (x1 + sx + px, y1 + py), (x2 - sx + px, y2 + py)


def sld_figure(data, lf=None, scr=None, fault=None, select=None, height=720, mode='network'):
    """Plotly single-line diagram. mode: 'network' (voltage colours), 'lf' (loading and voltage), 'sc' (fault levels)."""
    d = ng.clean(data)
    lay = ng.sld_layout(d)
    kv = dict(zip(d['Buses']['Bus'], d['Buses']['kV']))
    fig = go.Figure()
    # substation boxes
    for s, (x, y, n) in lay['subs'].items():
        fig.add_shape(type='rect', x0=x - 0.62, x1=x + 0.62, y0=y - 0.75 * (n - 1) - 0.42, y1=y + 0.42, line=dict(color='#c4d0e0', width=1.2),
                      fillcolor='rgba(240,245,252,0.85)', layer='below')
        fig.add_annotation(x=x, y=y + 0.52, text=f'<b>{s}</b>', showarrow=False, font=dict(size=11, color='#0b2e59'))
    LF = {}
    if lf:
        LF = {r['Line']: r for _, r in lf['line'].iterrows()}
        LF.update({r['Transformer']: r for _, r in lf['trafo'].iterrows()})
    # branches
    mx, my, mtxt, mcd, mcol, msym = [], [], [], [], [], []
    for br in lay['branches']:
        if br['a'] not in lay['pos'] or br['b'] not in lay['pos']:
            continue
        r = LF.get(br['id'])
        if br['kind'] == 'Line':
            (x1, y1), (x2, y2) = _ends(lay, br['a'], br['b'], br['off'])
            col = kv_color(kv.get(br['a'], 400))
            width = 2.2
            info = f"<b>{br['id']}</b> ({'cable' if br['cable'] else 'overhead line'})<br>{br['a']} → {br['b']}"
            if r is not None:
                col = ST_COL.get(r['Status'], col)
                width = 2 + min(5, max(0, (r['Loading (%)'] or 0) / 25))
                info += f"<br>P {r['P From (MW)']:.0f} MW, Q {r['Q From (MVAr)']:.0f} MVAr<br>I {r['I (A)']:.0f} A, loading {r['Loading (%)']:.0f} %"
            dash = 'dash' if br['cable'] else 'solid'
            if r is not None and r['Status'] == 'OUT':
                dash = 'dot'
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode='lines', line=dict(color=col, width=width, dash=dash), hoverinfo='skip', showlegend=False))
            mx.append((x1 + x2) / 2); my.append((y1 + y2) / 2); msym.append('circle')
            mtxt.append(info); mcd.append(['Lines', br['id']]); mcol.append(col)
        else:
            (x1, y1), (x2, y2) = lay['pos'][br['a']], lay['pos'][br['b']]
            x1 = x2 = x1 + br['off']
            col = '#334155' if r is None else ST_COL.get(r['Status'], '#334155')
            fig.add_trace(go.Scatter(x=[x1, x1], y=[y1, y2], mode='lines', line=dict(color=col, width=2), hoverinfo='skip', showlegend=False))
            ym = (y1 + y2) / 2
            for dy, c in ((0.07, kv_color(kv.get(br['a'], 400))), (-0.07, kv_color(kv.get(br['b'], 220)))):
                fig.add_trace(go.Scatter(x=[x1], y=[ym + dy], mode='markers', marker=dict(size=15, color='rgba(0,0,0,0)', line=dict(color=c, width=2)),
                                         hoverinfo='skip', showlegend=False))
            info = f"<b>{br['id']}</b> (transformer)<br>{br['a']} / {br['b']}"
            if r is not None:
                info += f"<br>P {r['P HV (MW)']:.0f} MW, loading {r['Loading (%)']:.0f} %"
            mx.append(x1); my.append(ym); mtxt.append(info); mcd.append(['Transformers', br['id']]); mcol.append(col); msym.append('circle')
    # buses
    BR = {r['Bus']: r for _, r in lf['bus'].iterrows()} if lf else {}
    SC = {r['Bus']: r for _, r in scr['bus'].iterrows()} if scr else {}
    bx, by, btxt, bcd = [], [], [], []
    for b, (x, y) in lay['pos'].items():
        col = kv_color(kv.get(b, 400))
        info = f"<b>{b}</b> {kv.get(b, 0):.0f} kV"
        lab = b
        if b in BR:
            rr = BR[b]
            col = ST_COL.get(rr['Status'], col) if mode == 'lf' else col
            info += f"<br>V {rr['V (pu)']:.3f} pu ({rr['V (kV)']:.1f} kV), angle {rr['Angle (deg)']:.1f}°"
            if mode == 'lf':
                lab = f"{b}  {rr['V (pu)']:.3f} pu"
        if b in SC:
            rr = SC[b]
            info += f"<br>Ik3 {rr['Ik3 (kA)']:.1f} kA, Ik1 {rr['Ik1 (kA)']:.1f} kA, duty {rr['Breaker Duty (%)']:.0f} %"
            if mode == 'sc':
                col = ST_COL.get(rr['Status'], col)
                lab = f"{b}  {max(rr['Ik3 (kA)'], rr['Ik1 (kA)']):.1f} kA"
        w = 7 if select == ('Buses', b) else 5
        fig.add_trace(go.Scatter(x=[x - BUS_HALF, x + BUS_HALF], y=[y, y], mode='lines', line=dict(color=col, width=w), hoverinfo='skip', showlegend=False))
        fig.add_annotation(x=x - BUS_HALF, y=y - 0.13, text=lab, showarrow=False, xanchor='left', font=dict(size=9, color='#334155'))
        for k in np.linspace(-BUS_HALF, BUS_HALF, 7):
            bx.append(x + k); by.append(y); btxt.append(info); bcd.append(['Buses', b])
    fig.add_trace(go.Scatter(x=bx, y=by, mode='markers', marker=dict(size=9, color='rgba(0,0,0,0)'), text=btxt, hovertemplate='%{text}<extra></extra>',
                             customdata=bcd, showlegend=False, name='buses'))
    # shunts and sources
    per_bus = {}
    sx, sy, stx, scd, ssym, scol = [], [], [], [], [], []
    for item in lay['shunts'] + lay['sources']:
        if item['bus'] not in lay['pos']:
            continue
        x, y = lay['pos'][item['bus']]
        k = per_bus.get(item['bus'], 0); per_bus[item['bus']] = k + 1
        xx = x - BUS_HALF + 0.08 + 0.13 * k
        if item['kind'] == 'Reactor':
            yy = y - 0.24; sym, col = 'circle-x-open', '#475569'
        elif item['kind'] == 'Grid':
            yy = y + (0.25 if _top(lay, item['bus']) else -0.24); sym, col = 'diamond', '#0b2e59'
        elif item['kind'] == 'Converter':
            yy = y + (0.25 if _top(lay, item['bus']) else -0.24); sym, col = 'square', '#7c3aed'
        else:
            yy = y - 0.24; sym, col = 'triangle-down', '#64748b'
        fig.add_trace(go.Scatter(x=[xx, xx], y=[y, yy], mode='lines', line=dict(color='#94a3b8', width=1), hoverinfo='skip', showlegend=False))
        sx.append(xx); sy.append(yy); stx.append(f"<b>{item['id']}</b> ({item['kind'].lower()}) at {item['bus']}"); scd.append([
            {'Reactor': 'Reactors', 'Grid': 'Grids', 'Converter': 'Converters', 'Load': 'Loads'}[item['kind']], item['id']])
        ssym.append(sym); scol.append(col)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode='markers', marker=dict(size=11, symbol=ssym, color=scol, line=dict(color=scol, width=1.5)), text=stx,
                             hovertemplate='%{text}<extra></extra>', customdata=scd, showlegend=False, name='equipment'))
    fig.add_trace(go.Scatter(x=mx, y=my, mode='markers', marker=dict(size=11, color='white', line=dict(color=mcol, width=2), symbol=msym), text=mtxt,
                             hovertemplate='%{text}<extra></extra>', customdata=mcd, showlegend=False, name='branches'))
    if select and select[0] in ('Lines', 'Transformers'):
        for x, y, c in zip(mx, my, mcd):
            if c[1] == select[1]:
                fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=20, color='rgba(47,128,237,.25)', line=dict(color='#2f80ed', width=2)),
                                         hoverinfo='skip', showlegend=False))
    if fault and fault in lay['pos']:
        x, y = lay['pos'][fault]
        fig.add_trace(go.Scatter(x=[x], y=[y + 0.02], mode='markers+text', marker=dict(size=26, symbol='star', color='#facc15', line=dict(color='#d63a3a', width=2)),
                                 text=['⚡ fault'], textposition='top center', textfont=dict(color='#d63a3a', size=12), hoverinfo='skip', showlegend=False))
    # legend
    for lab, col, dash in (('400 kV', KV_COL[400], 'solid'), ('275 kV', KV_COL[275], 'solid'), ('220 kV', KV_COL[220], 'solid'), ('Cable', '#555', 'dash')):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=col, width=3, dash=dash), name=lab))
    if lf or scr:
        for lab, st in (('Normal', 'EXCELLENT'), ('Watch', 'WATCH'), ('Action', 'ACTION')):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=ST_COL[st]), name=lab))
    xs = [p[0] for p in lay['pos'].values()] or [0]
    ys = [p[1] for p in lay['pos'].values()] or [0]
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='white', paper_bgcolor='white', dragmode='pan',
                      xaxis=dict(visible=False, range=[min(xs) - 1, max(xs) + 1]), yaxis=dict(visible=False, range=[min(ys) - 1.2, max(ys) + 0.9], scaleanchor='x'),
                      legend=dict(orientation='h', y=0.01, x=0.01, bgcolor='rgba(255,255,255,.8)'), hoverlabel=dict(bgcolor='white'), clickmode='event+select')
    return fig


def sld_matplotlib(data, lf=None, scr=None, fault=None, mode='network', ax=None):
    """Static diagram for the PDF report."""
    import matplotlib.pyplot as plt
    d = ng.clean(data)
    lay = ng.sld_layout(d)
    kv = dict(zip(d['Buses']['Bus'], d['Buses']['kV']))
    own = ax is None
    if own:
        xs = [p[0] for p in lay['pos'].values()]; ys = [p[1] for p in lay['pos'].values()]
        fig, ax = plt.subplots(figsize=(max(6, (max(xs) - min(xs) + 2) * 0.8), max(4.5, (max(ys) - min(ys) + 2.2) * 0.62)))
    LF = {}
    if lf:
        LF = {r['Line']: r for _, r in lf['line'].iterrows()}
        LF.update({r['Transformer']: r for _, r in lf['trafo'].iterrows()})
    for s, (x, y, n) in lay['subs'].items():
        ax.add_patch(plt.Rectangle((x - 0.62, y - 0.75 * (n - 1) - 0.42), 1.24, 0.75 * (n - 1) + 0.84, fc='#f0f5fc', ec='#c4d0e0', lw=0.8, zorder=0))
        ax.text(x, y + 0.5, s, ha='center', fontsize=7, color='#0b2e59', fontweight='bold')
    for br in lay['branches']:
        r = LF.get(br['id'])
        if br['kind'] == 'Line':
            (x1, y1), (x2, y2) = _ends(lay, br['a'], br['b'], br['off'])
            col = ST_COL.get(r['Status'], '#888') if r is not None else kv_color(kv.get(br['a'], 400))
            ax.plot([x1, x2], [y1, y2], color=col, lw=1.4, ls='--' if br['cable'] else '-', zorder=1)
            if r is not None and mode == 'lf':
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{r['Loading (%)']:.0f}%", fontsize=5.5, ha='center', va='center',
                        bbox=dict(fc='white', ec='none', pad=0.3), zorder=3)
        else:
            (x1, y1), (x2, y2) = lay['pos'][br['a']], lay['pos'][br['b']]
            x1 += br['off']
            ax.plot([x1, x1], [y1, y2], color='#334155', lw=1, zorder=1)
            ym = (y1 + y2) / 2
            for dy, c in ((0.06, kv_color(kv.get(br['a'], 400))), (-0.06, kv_color(kv.get(br['b'], 220)))):
                ax.add_patch(plt.Circle((x1, ym + dy), 0.07, fc='white', ec=c, lw=1, zorder=2))
    BR = {r['Bus']: r for _, r in lf['bus'].iterrows()} if lf else {}
    SC = {r['Bus']: r for _, r in scr['bus'].iterrows()} if scr else {}
    for b, (x, y) in lay['pos'].items():
        col = kv_color(kv.get(b, 400))
        lab = b
        if mode == 'lf' and b in BR:
            col = ST_COL.get(BR[b]['Status'], col); lab = f"{b} {BR[b]['V (pu)']:.3f} pu"
        if mode == 'sc' and b in SC:
            col = ST_COL.get(SC[b]['Status'], col); lab = f"{b} {max(SC[b]['Ik3 (kA)'], SC[b]['Ik1 (kA)']):.1f} kA"
        ax.plot([x - BUS_HALF, x + BUS_HALF], [y, y], color=col, lw=3.2, zorder=2, solid_capstyle='butt')
        ax.text(x - BUS_HALF, y - 0.16, lab, fontsize=5.5, color='#334155')
    per = {}
    for it in lay['shunts'] + lay['sources']:
        if it['bus'] not in lay['pos']:
            continue
        x, y = lay['pos'][it['bus']]
        k = per.get(it['bus'], 0); per[it['bus']] = k + 1
        xx = x - BUS_HALF + 0.08 + 0.13 * k
        yy = y - 0.24 if (it['kind'] in ('Reactor', 'Load') or not _top(lay, it['bus'])) else y + 0.25
        ax.plot([xx, xx], [y, yy], color='#94a3b8', lw=0.6, zorder=1)
        mk = {'Reactor': 'o', 'Grid': 'D', 'Converter': 's', 'Load': 'v'}[it['kind']]
        ax.plot([xx], [yy], marker=mk, ms=3.5, mfc='white' if it['kind'] == 'Reactor' else '#0b2e59', mec='#334155', zorder=3)
    if fault and fault in lay['pos']:
        x, y = lay['pos'][fault]
        ax.plot([x], [y], marker='*', ms=16, mfc='#facc15', mec='#d63a3a', zorder=5)
    ax.set_aspect('equal'); ax.axis('off')
    if own:
        fig.tight_layout()
        return fig
    return ax


def tcc_figure(relays, faults=None, title='Overcurrent relay curves'):
    """Log-log time-current curves; faults: list of (label, current A)."""
    fig = go.Figure()
    imax = max([float(f[1]) for f in (faults or [])] + [float(r['Pickup A']) * 30 for _, r in relays.iterrows()] + [1000]) * 1.3
    cols = ['#0b2e59', '#d63a3a', '#1a9e5c', '#d99a00', '#7c3aed', '#2a78d6']
    for k, (_, r) in enumerate(relays.iterrows()):
        I, t = ng.oc_curve_points(r, imax=imax)
        t = np.where(np.isfinite(t), t, np.nan)
        fig.add_scatter(x=I, y=t, mode='lines', name=f"{r['Relay']} ({r['Curve']}, Is {r['Pickup A']:.0f} A, TMS {r['TMS']})", line=dict(width=2.4, color=cols[k % len(cols)]))
    for lab, i in faults or []:
        fig.add_vline(x=i, line=dict(color='#94a3b8', dash='dot'), annotation_text=lab, annotation_position='top', annotation_font_size=10)
    fig.update_layout(template='plotly_white', height=480, title=title, xaxis=dict(type='log', title='Current (A, primary)'),
                      yaxis=dict(type='log', title='Time (s)', range=[-2, 2]), legend=dict(orientation='h', y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def rx_figure(row, line_z_ohm, angle_deg, title=''):
    """Mho zones of a distance relay on the R-X plane (primary ohms) with the line impedance and the load area."""
    fig = go.Figure()
    a = math.radians(angle_deg)
    th = np.linspace(0, 2 * math.pi, 200)
    for z, col, nm in ((row['Z3 (ohm)'], '#d99a00', 'Zone 3'), (row['Z2 (ohm)'], '#2a78d6', 'Zone 2'), (row['Z1 (ohm)'], '#1a9e5c', 'Zone 1')):
        cx, cy, rr = z / 2 * math.cos(a), z / 2 * math.sin(a), z / 2
        fig.add_scatter(x=cx + rr * np.cos(th), y=cy + rr * np.sin(th), mode='lines', fill='toself', name=f"{nm} {z:.1f} Ω", line=dict(color=col, width=2),
                        fillcolor=col.replace('#', 'rgba(') if False else None, opacity=0.9)
    fig.add_scatter(x=[0, line_z_ohm * math.cos(a)], y=[0, line_z_ohm * math.sin(a)], mode='lines+markers', name=f'Line {line_z_ohm:.1f} Ω ∠{angle_deg:.0f}°',
                    line=dict(color='#0b2e59', width=3))
    zl = row['Load Z (ohm)']
    for s in (1, -1):
        fig.add_scatter(x=[0, zl * math.cos(math.radians(30)) * 1.0], y=[0, s * zl * math.sin(math.radians(30))], mode='lines', line=dict(color='#94a3b8', dash='dot'),
                        showlegend=s == 1, name=f'Load area (|Z| ≥ {zl:.0f} Ω, ±30°)')
    m = max(row['Z3 (ohm)'], zl) * 1.1
    fig.update_layout(template='plotly_white', height=460, title=title, xaxis=dict(title='R (Ω primary)', range=[-0.4 * m, m], zeroline=True),
                      yaxis=dict(title='X (Ω primary)', range=[-0.4 * m, m], scaleanchor='x'), legend=dict(orientation='h', y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    return fig
