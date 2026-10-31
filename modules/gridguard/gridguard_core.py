"""
GridGuard AI - load and thermal forecasting engine for substation power transformers.

One engine used by every page of the app, the substation analysis and the historical study.

  1. AI load forecast: a gradient-boosting model predicts the hourly load of each transformer up to 7 days ahead
     from the weather forecast, the calendar (weekend, Ramadan, Eid, holidays), the hour, the load type of the
     transformer and its recent load.
  2. Thermal forecast: the forecast load and ambient temperature drive the IEEE C57.91 (clause 7) top-oil and
     hot-spot model; insulation ageing follows the IEEE C57.91 ageing acceleration factor.
  3. Risk: overload, hot-spot, top-oil and ageing risks, fused into one failure risk with FMEA and an action.

Inputs (CSV):
  * Transformer register: Transformer ID, Rating (MVA), Cooling (ONAN/ONAF/OFAF), Load Type
                          (Residential/Commercial/Industrial/Mixed), optional Top-Oil Rise (K), Hot-Spot Gradient (K)
  * Load history        : Timestamp, Transformer ID, Load (MVA)          (hourly)
  * Weather             : Timestamp, Ambient (C)                        (hourly; history and forecast)
"""
import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- limits and constants
EXCELLENT_MAX, WATCH_MAX = 35.0, 70.0
DETECTION = 2
HORIZON_DAYS = 7
LOAD_TYPES = ['Residential', 'Commercial', 'Industrial', 'Mixed']
COOLING = {   # IEEE C57.91 typical values for a 65 C average winding rise transformer (hot-spot rise 80 K at rated load)
    'ONAN': {'to_rise': 55.0, 'hs_grad': 25.0, 'R': 3.2, 'n': 0.8, 'm': 0.8, 'tau_to': 210.0, 'tau_w': 7.0},
    'ONAF': {'to_rise': 50.0, 'hs_grad': 30.0, 'R': 4.5, 'n': 0.9, 'm': 0.8, 'tau_to': 150.0, 'tau_w': 7.0},
    'OFAF': {'to_rise': 45.0, 'hs_grad': 35.0, 'R': 6.0, 'n': 1.0, 'm': 0.8, 'tau_to': 90.0, 'tau_w': 7.0},
}
HS_NORMAL, HS_PLANNED, HS_EMERGENCY = 110.0, 130.0, 140.0     # IEEE C57.91 hot-spot limits (C)
TO_LIMIT = 105.0                                                # top-oil limit used for the risk (C)

MODES = {
    'Overload': {'S': 7, 'text': 'Load above the transformer rating'},
    'Hot-spot temperature': {'S': 9, 'text': 'Winding hot-spot above the IEEE C57.91 limits: insulation damage, gas bubbles'},
    'Top-oil temperature': {'S': 8, 'text': 'Top-oil temperature high: oil expansion, gasket and bushing stress'},
    'Insulation ageing': {'S': 6, 'text': 'Paper insulation ageing faster than normal'},
}
MODE_ORDER = list(MODES)
ACTIONS = {
    'Overload': 'Plan a load transfer to a neighbouring transformer or feeder for the peak hours; check the cooling is fully in service.',
    'Hot-spot temperature': 'Transfer load before the peak to keep the hot-spot below 110 °C; start all cooling stages early; avoid switching that adds load.',
    'Top-oil temperature': 'Check fans, pumps and radiators before the peak; clean the coolers; consider pre-cooling with forced cooling.',
    'Insulation ageing': 'Limit the hours of high load in the coming days; include the unit in the next loading review and the DGA follow-up.',
}

# ---------------------------------------------------------------- Gulf calendar (Saudi working week: Friday and Saturday off)
RAMADAN = [(date(2024, 3, 11), date(2024, 4, 9)), (date(2025, 3, 1), date(2025, 3, 29)), (date(2026, 2, 18), date(2026, 3, 19)),
           (date(2027, 2, 8), date(2027, 3, 9))]
HOLIDAYS = [(date(2024, 4, 7), date(2024, 4, 17)), (date(2024, 6, 14), date(2024, 6, 22)), (date(2025, 3, 28), date(2025, 4, 6)),
            (date(2025, 6, 4), date(2025, 6, 12)), (date(2026, 3, 18), date(2026, 3, 28)), (date(2026, 5, 24), date(2026, 6, 1)),
            (date(2027, 3, 8), date(2027, 3, 17)), (date(2027, 5, 14), date(2027, 5, 22))]
FIXED_HOLIDAYS = [(2, 22), (9, 23)]       # Founding Day, National Day


def _in(d, spans):
    return any(a <= d <= b for a, b in spans)


def calendar_frame(ts):
    """Calendar columns for an hourly DatetimeIndex."""
    ts = pd.DatetimeIndex(ts)
    d = ts.normalize()
    days = pd.Series(d.date, index=ts)
    uniq = {x: (_in(x, RAMADAN), _in(x, HOLIDAYS) or (x.month, x.day) in FIXED_HOLIDAYS) for x in set(days)}
    ram = np.array([uniq[x][0] for x in days], float)
    hol = np.array([uniq[x][1] for x in days], float)
    dow = ts.dayofweek.to_numpy()
    return pd.DataFrame({'hour': ts.hour.to_numpy(), 'dow': dow, 'weekend': np.isin(dow, [4, 5]).astype(float),
                         'ramadan': ram, 'holiday': hol, 'doy': ts.dayofyear.to_numpy()}, index=ts)


# ---------------------------------------------------------------- Gulf climate (normal daily max / min by month)
CLIMATE = {
    'Dammam': [(21, 11), (23, 13), (27, 16), (33, 21), (39, 26), (43, 29), (44, 31), (44, 31), (41, 28), (36, 23), (29, 18), (23, 13)],
    'Riyadh': [(20, 9), (23, 11), (28, 15), (33, 20), (40, 25), (43, 27), (44, 29), (44, 29), (41, 26), (35, 21), (27, 15), (22, 10)],
    'Jeddah': [(29, 19), (29, 19), (31, 21), (33, 23), (35, 25), (37, 27), (38, 28), (38, 29), (36, 27), (35, 25), (32, 22), (30, 20)],
    'Kuwait': [(19, 8), (22, 10), (28, 15), (34, 20), (41, 26), (46, 30), (48, 32), (47, 32), (43, 28), (36, 22), (27, 15), (21, 10)],
    'Doha': [(22, 14), (23, 15), (27, 18), (32, 22), (38, 26), (41, 29), (42, 31), (42, 31), (39, 28), (35, 24), (29, 20), (24, 16)],
    'Abu Dhabi': [(24, 14), (26, 16), (30, 19), (35, 23), (40, 27), (42, 30), (44, 32), (44, 32), (41, 29), (37, 25), (31, 21), (26, 17)],
    'Muscat': [(25, 17), (26, 18), (30, 21), (35, 25), (40, 30), (40, 31), (38, 30), (36, 28), (36, 27), (35, 25), (30, 21), (26, 18)],
    'Manama': [(20, 14), (21, 15), (25, 18), (30, 22), (35, 26), (38, 29), (39, 31), (39, 31), (37, 29), (33, 26), (28, 21), (22, 16)],
}


def normal_temperature(ts, city='Dammam', shift=0.0):
    """Normal hourly ambient temperature for a city (smooth seasonal curve, minimum about 05:00, maximum about 14:00)."""
    ts = pd.DatetimeIndex(ts)
    tab = np.array(CLIMATE[city], float)
    mid = np.array([15 + 30.44 * i for i in range(12)])
    doy = ts.dayofyear.to_numpy().astype(float)
    x = np.r_[mid - 365.25, mid, mid + 365.25]
    tmax = np.interp(doy, x, np.r_[tab[:, 0], tab[:, 0], tab[:, 0]])
    tmin = np.interp(doy, x, np.r_[tab[:, 1], tab[:, 1], tab[:, 1]])
    h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    shape = np.where((h >= 5) & (h <= 14.5), 0.5 - 0.5 * np.cos(np.pi * (h - 5) / 9.5),
                     0.5 + 0.5 * np.cos(np.pi * ((h - 14.5) % 24) / 14.5))
    return tmin + (tmax - tmin) * shape + shift


# ---------------------------------------------------------------- IEEE C57.91 thermal model
def cooling_params(cooling, to_rise=None, hs_grad=None):
    p = dict(COOLING.get(str(cooling).upper().strip(), COOLING['ONAF']))
    if to_rise is not None and not pd.isna(to_rise) and float(to_rise) > 0:
        p['to_rise'] = float(to_rise)
    if hs_grad is not None and not pd.isna(hs_grad) and float(hs_grad) > 0:
        p['hs_grad'] = float(hs_grad)
    return p


def faa(hs):
    """IEEE C57.91 ageing acceleration factor (1.0 at a 110 C hot-spot)."""
    return np.exp(15000.0 / 383.0 - 15000.0 / (np.asarray(hs, float) + 273.0))


def thermal(K, ambient, p, state=None, dt_min=60.0, sub=6):
    """Top-oil and hot-spot temperatures for hourly per-unit loads K (load / rating) and ambient (C).
    K and ambient may be 1-D (one series) or 2-D (series x hours). state = (top-oil rise, hot-spot gradient) at the start."""
    K = np.atleast_2d(np.asarray(K, float)); A = np.atleast_2d(np.asarray(ambient, float))
    if A.shape[0] == 1 and K.shape[0] > 1:
        A = np.repeat(A, K.shape[0], 0)
    ns, nh = K.shape
    R, n, m = p['R'], p['n'], p['m']
    if state is None:
        k0 = K[:, 0]
        to = p['to_rise'] * ((k0 ** 2 * R + 1) / (R + 1)) ** n
        hg = p['hs_grad'] * k0 ** (2 * m)
    else:
        to, hg = (np.broadcast_to(np.asarray(s, float), (ns,)).copy() for s in state)
    TO = np.empty((ns, nh)); HS = np.empty((ns, nh)); TOR = np.empty((ns, nh)); HG = np.empty((ns, nh))
    dt = dt_min / sub
    a_to = 1 - math.exp(-dt / p['tau_to']); a_w = 1 - math.exp(-dt / p['tau_w'])
    for j in range(nh):
        k = K[:, j]
        to_u = p['to_rise'] * ((k ** 2 * R + 1) / (R + 1)) ** n
        hg_u = p['hs_grad'] * k ** (2 * m)
        for _ in range(sub):
            to = to + (to_u - to) * a_to
            hg = hg + (hg_u - hg) * a_w
        TOR[:, j] = to; HG[:, j] = hg
        TO[:, j] = A[:, j] + to; HS[:, j] = TO[:, j] + hg
    return {'top_oil': TO, 'hot_spot': HS, 'state': (TOR[:, -1], HG[:, -1])}


def max_scale_for(K, ambient, p, state, hs_limit=HS_NORMAL):
    """Largest factor f (0.3 to 1.0) so that f x the forecast load keeps the hot-spot at or below hs_limit."""
    if thermal(K, ambient, p, state)['hot_spot'].max() <= hs_limit:
        return 1.0
    lo, hi = 0.3, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if thermal(np.asarray(K) * mid, ambient, p, state)['hot_spot'].max() <= hs_limit:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- AI load model: features
FEATURES = ['hour', 'dow', 'weekend', 'ramadan', 'holiday', 'doy_s', 'doy_c', 'temp', 'temp_ema', 'temp_dmax', 'temp_dmin',
            'cdh', 'lag168', 'lag168_temp_diff', 'type_res', 'type_com', 'type_ind', 'type_mix', 'h_s', 'h_c']


def type_flags(t):
    t = str(t).strip().lower()
    return [float(t.startswith('res')), float(t.startswith('com')), float(t.startswith('ind')),
            float(not (t.startswith('res') or t.startswith('com') or t.startswith('ind')))]


def ema(x, span_h=4.0):
    a = 1 - math.exp(-1 / span_h)
    out = np.empty(len(x)); acc = x[0]
    for i, v in enumerate(x):
        acc = acc + a * (v - acc); out[i] = acc
    return out


def feature_frame(ts, temp, load_type, lag168_ratio, lag168_temp):
    """Model features for target hours ts. temp: ambient at ts; lag168_ratio: load one week before / level;
    lag168_temp: ambient one week before."""
    ts = pd.DatetimeIndex(ts)
    c = calendar_frame(ts)
    temp = np.asarray(temp, float)
    s = pd.Series(temp, index=ts)
    day = ts.normalize()
    dmax = s.groupby(day).transform('max').to_numpy(); dmin = s.groupby(day).transform('min').to_numpy()
    te = ema(temp)
    f = pd.DataFrame({'hour': c['hour'], 'dow': c['dow'], 'weekend': c['weekend'], 'ramadan': c['ramadan'], 'holiday': c['holiday'],
                      'doy_s': np.sin(2 * np.pi * c['doy'] / 365.25), 'doy_c': np.cos(2 * np.pi * c['doy'] / 365.25),
                      'temp': temp, 'temp_ema': te, 'temp_dmax': dmax, 'temp_dmin': dmin, 'cdh': np.maximum(te - 24, 0),
                      'lag168': np.asarray(lag168_ratio, float), 'lag168_temp_diff': temp - np.asarray(lag168_temp, float)}, index=ts)
    fl = type_flags(load_type)
    for k, v in zip(['type_res', 'type_com', 'type_ind', 'type_mix'], fl):
        f[k] = v
    f['h_s'] = np.sin(2 * np.pi * f['hour'] / 24); f['h_c'] = np.cos(2 * np.pi * f['hour'] / 24)
    return f[FEATURES]


def level_at(load, idx):
    """Mean load of the 7 days (168 h) before position idx (the forecast origin)."""
    a = max(0, idx - 168)
    return float(np.nanmean(load[a:idx])) if idx > a else float('nan')


def training_table(history, register, temps):
    """Day-ahead training rows: for every hour of every day, the origin is the start of the previous day."""
    rows = []
    reg = {str(r['Transformer ID']): r for r in register.to_dict('records')}
    T = temps.set_index(pd.to_datetime(temps['Timestamp']))['Ambient (C)'].astype(float).sort_index()
    for tid, g in history.groupby('Transformer ID'):
        tid = str(tid)
        if tid not in reg:
            continue
        s = g.assign(_t=pd.to_datetime(g['Timestamp'])).set_index('_t')['Load (MVA)'].astype(float).sort_index()
        s = s[~s.index.duplicated()].asfreq('h').interpolate(limit=6)
        tt = T.reindex(s.index).interpolate(limit=6).to_numpy()
        L = s.to_numpy()
        ts = s.index
        # level = mean of the 168 h ending at the start of the previous day (origin)
        day_start = np.arange(len(L)) - ts.hour.to_numpy()
        origin = day_start - 24
        cs = np.r_[0, np.nancumsum(np.nan_to_num(L))]; cnt = np.r_[0, np.cumsum(~np.isnan(L))]
        lo = np.clip(origin - 168, 0, None); hi = np.clip(origin, 0, None)
        level = (cs[hi] - cs[lo]) / np.maximum(cnt[hi] - cnt[lo], 1)
        lag = np.r_[np.full(168, np.nan), L[:-168]]
        lagt = np.r_[np.full(168, np.nan), tt[:-168]]
        ok = (origin >= 168) & ~np.isnan(L) & ~np.isnan(lag) & ~np.isnan(tt) & ~np.isnan(lagt) & (level > 0)
        f = feature_frame(ts, tt, reg[tid].get('Load Type', 'Mixed'), lag / np.where(level > 0, level, np.nan), lagt)
        f = f[ok]
        f['y'] = (L / np.where(level > 0, level, np.nan))[ok]
        f['level'] = level[ok]; f['Transformer ID'] = tid; f['load'] = L[ok]
        rows.append(f)
    return pd.concat(rows) if rows else pd.DataFrame(columns=FEATURES + ['y'])


def train_model(table, max_iter=400, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=seed)
    m.fit(table[FEATURES].to_numpy(float), table['y'].to_numpy(float))
    return m


def load_model(path='gridguard_model.pkl'):
    import pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------- forecast one transformer
def forecast(bundle, reg, hist_ts, hist_load, temp_hist_ts, temp_hist, fc_temp_ts, fc_temp, origin=None, days=HORIZON_DAYS):
    """Hourly load and thermal forecast from origin (default: the hour after the last load reading) for `days` days.
    hist_*: load history (MVA). temp_hist: ambient history; fc_temp: ambient forecast covering the horizon."""
    model = bundle['model']
    s = pd.Series(np.asarray(hist_load, float), index=pd.DatetimeIndex(hist_ts)).sort_index()
    s = s[~s.index.duplicated()].asfreq('h').interpolate(limit=6)
    if origin is None:
        origin = s.index[-1] + pd.Timedelta(hours=1)
    origin = pd.Timestamp(origin).floor('h')
    past = s[s.index < origin]
    if len(past) < 168:
        raise ValueError('At least 7 days of hourly load history are needed before the forecast start.')
    level = float(past.iloc[-168:].mean())
    th = pd.Series(np.asarray(temp_hist, float), index=pd.DatetimeIndex(temp_hist_ts)).sort_index()
    tf = pd.Series(np.asarray(fc_temp, float), index=pd.DatetimeIndex(fc_temp_ts)).sort_index()
    allT = pd.concat([th[th.index < origin], tf[tf.index >= origin]])
    allT = allT[~allT.index.duplicated(keep='last')]
    ts = pd.date_range(origin, periods=days * 24, freq='h')
    temp = allT.reindex(ts).interpolate(limit_direction='both').to_numpy()
    lag_ts = ts - pd.Timedelta(hours=168)
    lag = s.reindex(lag_ts).to_numpy()
    lag_t = allT.reindex(lag_ts).interpolate(limit_direction='both').to_numpy()
    # beyond 7 days the week-before load is itself a forecast: fill recursively
    pred = np.empty(len(ts))
    X = feature_frame(ts, temp, reg.get('Load Type', 'Mixed'), np.zeros(len(ts)), lag_t)
    for d0 in range(0, len(ts), 168):
        sl = slice(d0, min(d0 + 168, len(ts)))
        lr = lag[sl].copy()
        if d0 >= 168:
            lr = pred[d0 - 168:d0 - 168 + len(lr)] * level
        X.iloc[sl, FEATURES.index('lag168')] = lr / level
        pred[sl] = model.predict(X.iloc[sl].to_numpy(float)) * level
    rating = float(reg.get('Rating (MVA)') or 1)
    p = cooling_params(reg.get('Cooling', 'ONAF'), reg.get('Top-Oil Rise (K)'), reg.get('Hot-Spot Gradient (K)'))
    # thermal state at the origin: run the last 3 days of history
    h3 = past.iloc[-72:]
    t3 = allT.reindex(h3.index).interpolate(limit_direction='both').to_numpy()
    st = thermal(h3.to_numpy() / rating, t3, p)['state']
    sig = bundle.get('err_sd_by_day', [0.04] * days)
    band = np.array([sig[min(int(i // 24), len(sig) - 1)] for i in range(len(ts))])
    th_ = thermal(pred / rating, temp, p, st)
    hi = thermal(pred * (1 + 1.28 * band) / rating, temp, p, st)
    return {'ts': ts, 'load': pred, 'load_hi': pred * (1 + 1.28 * band), 'load_lo': pred * (1 - 1.28 * band), 'ambient': temp,
            'K': pred / rating, 'top_oil': th_['top_oil'][0], 'hot_spot': th_['hot_spot'][0], 'hot_spot_hi': hi['hot_spot'][0],
            'faa': faa(th_['hot_spot'][0]), 'state': st, 'params': p, 'rating': rating, 'level': level, 'origin': origin}


# ---------------------------------------------------------------- risk
def lin(x, x0, x1, y0=0.0, y1=100.0):
    if x1 == x0:
        return y1 if x >= x1 else y0
    return float(min(max(y0, y1), max(min(y0, y1), y0 + (y1 - y0) * (x - x0) / (x1 - x0))))


def band3(x, a, b, c, d):
    """0-35 from a to b, 35-70 from b to c, 70-100 from c to d."""
    if x <= b:
        return lin(x, a, b, 0, 35)
    if x <= c:
        return lin(x, b, c, 35, 70)
    return lin(x, c, d, 70, 100)


def status_of(r):
    return 'EXCELLENT' if r <= EXCELLENT_MAX else ('WATCH' if r <= WATCH_MAX else 'ACTION')


def rpn_level(rpn):
    return 'Low' if rpn < 40 else ('Medium' if rpn < 100 else 'High')


def assess(reg, fc, hs_watch=HS_NORMAL):
    """Risk of one transformer from its forecast."""
    ts, K, HS, TO, FA = fc['ts'], fc['K'], fc['hot_spot'], fc['top_oil'], fc['faa']
    rating = fc['rating']
    i_hs, i_k, i_to = int(np.argmax(HS)), int(np.argmax(K)), int(np.argmax(TO))
    days = pd.Series(FA, index=ts).groupby(ts.normalize()).mean()
    feqa = float(days.max()); feqa_day = days.idxmax()
    hours_over = int((K > 1.0).sum()); hours_hs = int((HS > hs_watch).sum())
    R, why = {}, {}
    f0 = lambda x: f"{x:.0f}"
    R['Overload'] = min(100.0, band3(float(K.max()), 0.6, 0.9, 1.1, 1.3) + (min(10, hours_over / 4) if hours_over else 0))
    why['Overload'] = [f"Peak load {fc['load'][i_k]:.1f} MVA ({K.max() * 100:.0f}% of the {rating:g} MVA rating) on {ts[i_k]:%a %d %b %H:00}."
                       + (f" {hours_over} hour(s) above the rating in the next {len(ts) // 24} days." if hours_over else " No hour above the rating.")]
    R['Hot-spot temperature'] = band3(float(HS.max()), 90, HS_NORMAL, HS_PLANNED, 150)
    why['Hot-spot temperature'] = [f"Winding hot-spot up to {f0(HS.max())} °C on {ts[i_hs]:%a %d %b %H:00} (IEEE C57.91: {HS_NORMAL:.0f} °C normal, "
                                   f"{HS_PLANNED:.0f} °C planned overload, {HS_EMERGENCY:.0f} °C emergency)."
                                   + (f" {hours_hs} hour(s) above {hs_watch:.0f} °C." if hours_hs else '')
                                   + (f" With a heavier load (upper band) up to {f0(fc['hot_spot_hi'].max())} °C." if fc['hot_spot_hi'].max() > hs_watch else '')]
    R['Top-oil temperature'] = band3(float(TO.max()), 70, 90, TO_LIMIT, 115)
    why['Top-oil temperature'] = [f"Top oil up to {f0(TO.max())} °C on {ts[i_to]:%a %d %b %H:00} at {fc['ambient'][i_to]:.0f} °C ambient."]
    R['Insulation ageing'] = band3(feqa, 0.0, 1.0, 4.0, 16.0)
    why['Insulation ageing'] = [f"Worst day {feqa_day:%a %d %b}: paper ageing {feqa:.2f}x the normal rate (IEEE C57.91 ageing factor); "
                                f"{float(FA.sum()):.0f} hours of normal life used in {len(ts)} hours."]
    risks = {k: round(float(v), 1) for k, v in R.items()}
    gov = max(MODE_ORDER, key=lambda k: risks[k])
    others = sum(1 for k in MODE_ORDER if k != gov and risks[k] > EXCELLENT_MAX)
    overall = round(min(100.0, risks[gov] + 5 * others), 1)
    S = MODES[gov]['S']; O = max(1, math.ceil(overall / 10)); rpn = S * O * DETECTION
    fmea = []
    for k in sorted(MODE_ORDER, key=lambda k: -risks[k]):
        o = max(1, math.ceil(risks[k] / 10)); r = MODES[k]['S'] * o * DETECTION
        fmea.append({'Failure Mode': k, 'Risk (%)': risks[k], 'S': MODES[k]['S'], 'O': o, 'D': DETECTION, 'RPN': r, 'RPN Level': rpn_level(r)})
    st = status_of(overall)
    f = max_scale_for(K, fc['ambient'], fc['params'], fc['state'], hs_watch)
    transfer = float(fc['load'].max() * (1 - f)) if f < 1 else 0.0
    act = ACTIONS[gov] if st != 'EXCELLENT' else 'No action needed: normal loading expected. Keep the forecast running.'
    if transfer > 0:
        act += f" Transfer about {transfer:.1f} MVA at the peak keeps the hot-spot at or below {hs_watch:.0f} °C."
    return {'Transformer ID': str(reg.get('Transformer ID', '')), 'Bay': reg.get('Bay', ''), 'Load Type': reg.get('Load Type', ''),
            'Cooling': str(reg.get('Cooling', '')), 'Rating (MVA)': rating, 'Forecast From': fc['origin'],
            'Peak Load (MVA)': round(float(fc['load'].max()), 2), 'Peak Load (%)': round(float(K.max() * 100), 1), 'Peak Time': ts[i_k],
            'Max Hot-Spot (C)': round(float(HS.max()), 1), 'Hot-Spot Time': ts[i_hs], 'Max Top-Oil (C)': round(float(TO.max()), 1),
            'Max Ambient (C)': round(float(np.max(fc['ambient'])), 1), 'Hours Above Rating': hours_over, 'Hours Hot-Spot Above Limit': hours_hs,
            'Worst Day Ageing (x)': round(feqa, 2), 'Load Transfer Needed (MVA)': round(transfer, 1),
            'Mode Risks': risks, 'Reasons': why, 'Governing Mode': gov, 'Other Modes Above Normal': others, 'Overall Risk (%)': overall,
            'Status': st, 'S': S, 'O': O, 'D': DETECTION, 'RPN': rpn, 'RPN Level': rpn_level(rpn), 'FMEA': fmea, 'Action': act}


def findings(a):
    r = []
    for k, v in sorted(a['Mode Risks'].items(), key=lambda kv: -kv[1]):
        if k == a['Governing Mode'] or v > EXCELLENT_MAX:
            r.append(f"{k} ({v:.0f}%): " + ' '.join(a['Reasons'][k]))
    n = a['Other Modes Above Normal']
    if n:
        r.append(f"{n} other failure mode(s) above normal add {5 * n} points to the overall risk.")
    if a['Status'] == 'EXCELLENT':
        r.append('All failure modes within normal limits for the forecast period.')
    return r


# ---------------------------------------------------------------- demo substation (synthetic)
DEMO_REGISTER = [
    # id, bay, load type, rating, cooling, base pu, cooling pu, growth/yr
    ('TR-1', 'Feeder group A', 'Residential', 50.0, 'ONAF', 0.34, 0.50, 0.035),
    ('TR-2', 'Feeder group B', 'Residential', 50.0, 'ONAF', 0.39, 0.58, 0.050),
    ('TR-3', 'Commercial centre', 'Commercial', 40.0, 'ONAF', 0.34, 0.42, 0.030),
    ('TR-4', 'Industrial area', 'Industrial', 60.0, 'OFAF', 0.58, 0.16, 0.020),
    ('TR-5', 'Old town', 'Mixed', 30.0, 'ONAN', 0.37, 0.48, 0.060),
    ('TR-6', 'New district', 'Commercial', 40.0, 'ONAF', 0.26, 0.30, 0.030),
]
DEMO_START, DEMO_END = '2024-01-01', '2026-08-31 23:00'
DEMO_CITY = 'Dammam'


def demo_register():
    return pd.DataFrame([{'Transformer ID': t, 'Bay': b, 'Load Type': lt, 'Rating (MVA)': r, 'Cooling': c,
                          'Top-Oil Rise (K)': COOLING[c]['to_rise'], 'Hot-Spot Gradient (K)': COOLING[c]['hs_grad']}
                         for t, b, lt, r, c, *_ in DEMO_REGISTER])


def _profile(load_type, hour, weekend, ramadan, holiday):
    h = hour.astype(float)
    res = 0.70 + 0.12 * np.exp(-((h - 14) / 3.5) ** 2) + 0.30 * np.exp(-((h - 22) / 3.0) ** 2) - 0.08 * np.exp(-((h - 6) / 2.0) ** 2)
    res_r = 0.72 + 0.05 * np.exp(-((h - 15) / 3.0) ** 2) + 0.38 * np.exp(-(((h - 1) % 24 - 0) / 3.5) ** 2) + 0.30 * np.exp(-((h - 22) / 2.5) ** 2)
    com = 0.55 + 0.45 / (1 + np.exp(-(h - 9.5) * 1.6)) - 0.55 / (1 + np.exp(-(h - 23.2) * 2.0)) + 0.10 * np.exp(-((h - 20) / 2.0) ** 2)
    com_r = 0.42 + 0.25 / (1 + np.exp(-(h - 10) * 1.4)) + 0.45 * np.exp(-(((h - 23) % 24) / 3.5) ** 2) - 0.12 * np.exp(-((h - 18) / 1.2) ** 2)
    ind = 0.88 + 0.12 / (1 + np.exp(-(h - 6.5) * 2)) - 0.12 / (1 + np.exp(-(h - 18) * 2))
    lt = load_type.lower()
    if lt.startswith('res'):
        p = np.where(ramadan > 0, res_r, res) * np.where(weekend + holiday > 0, 1.06, 1.0)
    elif lt.startswith('com'):
        p = np.where(ramadan > 0, com_r, com) * np.where(holiday > 0, 0.85, 1.0)
        p = np.where((weekend > 0) & (h < 15) & (ramadan == 0), p * 0.80, p)
    elif lt.startswith('ind'):
        p = ind * np.where(weekend > 0, 0.70, 1.0) * np.where(holiday > 0, 0.55, 1.0) * np.where(ramadan > 0, 0.92, 1.0)
    else:
        p = 0.5 * (np.where(ramadan > 0, res_r, res) + np.where(ramadan > 0, com_r, com))
    return p


def demo_weather(seed=7, start=DEMO_START, end=DEMO_END, city=DEMO_CITY):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq='h')
    base = normal_temperature(ts, city)
    nd = len(pd.date_range(ts[0].normalize(), ts[-1].normalize(), freq='D'))
    an = np.zeros(nd); e = 0.0
    for i in range(nd):
        e = 0.75 * e + rng.normal(0, 1.2); an[i] = e
    # summer heat waves
    for y in sorted(set(ts.year)):
        for _ in range(2):
            d0 = (pd.Timestamp(f'{y}-06-10') + pd.Timedelta(days=int(rng.integers(0, 70))) - ts[0].normalize()).days
            if 0 <= d0 < nd:
                L = int(rng.integers(3, 7)); an[d0:d0 + L] += np.linspace(1.5, 2.5, len(an[d0:d0 + L]))
    di = (ts.normalize() - ts[0].normalize()).days.to_numpy()
    t = base + np.interp(np.arange(len(ts)) / 24, np.arange(nd), an) * 1.0 + rng.normal(0, 0.4, len(ts))
    return pd.DataFrame({'Timestamp': ts, 'Ambient (C)': np.round(t, 1)})


def demo_loads(weather, seed=11):
    rng = np.random.default_rng(seed)
    ts = pd.DatetimeIndex(pd.to_datetime(weather['Timestamp']))
    T = weather['Ambient (C)'].to_numpy(float)
    c = calendar_frame(ts)
    te = ema(T, 5.0)
    years = (ts - ts[0]).days.to_numpy() / 365.25
    out = []
    for tid, bay, lt, rating, cool, base, cpu, gr in DEMO_REGISTER:
        prof = _profile(lt, c['hour'].to_numpy(), c['weekend'].to_numpy(), c['ramadan'].to_numpy(), c['holiday'].to_numpy())
        coolload = cpu * np.clip((te - 22) / 22, 0, None) ** 1.25 * (1.10 if lt.startswith('res') else 1.0)
        if lt.startswith('com'):
            coolload *= np.where((c['hour'] >= 9) & (c['hour'] <= 23), 1.15, 0.55)
        elif lt.startswith('res'):
            coolload *= np.where((c['hour'] >= 12) | (c['hour'] <= 2), 1.08, 0.85)
        lvl = (1 + gr) ** years
        if tid == 'TR-6':                      # new district: a large commercial customer connected in March 2026
            lvl = lvl * np.where(ts >= pd.Timestamp('2026-03-01'), 1.55, 1.0)
        noise = np.exp(np.convolve(rng.normal(0, 0.025, len(ts)), np.ones(3) / 3, 'same'))
        load = rating * (base * prof + coolload) * lvl * noise
        out.append(pd.DataFrame({'Timestamp': ts, 'Transformer ID': tid, 'Load (MVA)': np.round(load, 2)}))
    return pd.concat(out, ignore_index=True)


def demo_data():
    w = demo_weather()
    return {'register': demo_register(), 'loads': demo_loads(w), 'weather': w}


def weather_forecast_from(weather, origin, days=HORIZON_DAYS, seed=0):
    """A realistic weather forecast from actual weather: error grows with the lead time (demo only)."""
    rng = np.random.default_rng(seed + int(pd.Timestamp(origin).value // 3_600_000_000_000) % 100000)
    w = weather.assign(_t=pd.to_datetime(weather['Timestamp'])).set_index('_t')['Ambient (C)']
    ts = pd.date_range(origin, periods=days * 24, freq='h')
    act = w.reindex(ts).interpolate(limit_direction='both').to_numpy()
    lead = np.arange(len(ts)) / 24.0
    err = np.repeat(rng.normal(0, 1, days), 24)[:len(ts)] * (0.6 + 0.35 * lead)
    return pd.DataFrame({'Timestamp': ts, 'Ambient (C)': np.round(act + err, 1)})


# ---------------------------------------------------------------- historical study (fast replay)
def _ema_rows(x, span_h=4.0):
    a = 1 - math.exp(-1 / span_h)
    out = np.empty_like(x); acc = x[:, 0].copy()
    for j in range(x.shape[1]):
        acc = acc + a * (x[:, j] - acc); out[:, j] = acc
    return out


def replay(bundle, register, loads, weather, start, end, seed=0, forecast_weather=None):
    """Every day at 00:00 from start to end, forecast each transformer 7 days ahead (as forecast() does) and keep, for every
    target day, the forecast and the actual daily maximum hot-spot and load. Returns one row per transformer, origin, target day."""
    model = bundle['model']; D = HORIZON_DAYS; H = D * 24
    W = weather.assign(_t=pd.to_datetime(weather['Timestamp'])).set_index('_t')['Ambient (C)'].astype(float).sort_index()
    W = W[~W.index.duplicated()].asfreq('h').interpolate(limit_direction='both')
    origins = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq='D')
    out = []
    for r in register.to_dict('records'):
        tid = str(r['Transformer ID'])
        L = loads[loads['Transformer ID'].astype(str) == tid]
        if L.empty:
            continue
        s = L.assign(_t=pd.to_datetime(L['Timestamp'])).set_index('_t')['Load (MVA)'].astype(float).sort_index()
        s = s[~s.index.duplicated()].asfreq('h').interpolate(limit=6)
        org = origins[(origins - pd.Timedelta(hours=168) >= s.index[0]) & (origins + pd.Timedelta(hours=H - 1) <= s.index[-1])]
        if len(org) == 0:
            continue
        idx0 = s.index.get_indexer(org)
        Lv = s.to_numpy(); Tw = W.reindex(s.index).interpolate(limit_direction='both').to_numpy()
        n = len(org)
        tgt = idx0[:, None] + np.arange(H)[None, :]
        # weather forecast for every origin (same error model as weather_forecast_from), or a supplied one
        if forecast_weather is None:
            temp = np.empty((n, H))
            for i, o in enumerate(org):
                rng = np.random.default_rng(seed + int(o.value // 3_600_000_000_000) % 100000)
                lead = np.arange(H) / 24.0
                err = np.repeat(rng.normal(0, 1, D), 24)[:H] * (0.6 + 0.35 * lead)
                temp[i] = np.round(Tw[tgt[i]] + err, 1)
        else:
            temp = forecast_weather(org, Tw, tgt)
        cs = np.r_[0, np.nancumsum(Lv)]
        level = (cs[idx0] - cs[idx0 - 168]) / 168
        lag = Lv[tgt - 168]; lagt = Tw[tgt - 168]
        te = _ema_rows(temp)
        tday = temp.reshape(n, D, 24)
        dmax = np.repeat(tday.max(2), 24, 1); dmin = np.repeat(tday.min(2), 24, 1)
        ts = s.index[tgt.ravel()]
        hour = ts.hour.to_numpy(); c = calendar_frame(ts)
        doy = c['doy'].to_numpy()
        fl = type_flags(r.get('Load Type', 'Mixed'))
        X = np.column_stack([hour, c['dow'], c['weekend'], c['ramadan'], c['holiday'], np.sin(2 * np.pi * doy / 365.25),
                             np.cos(2 * np.pi * doy / 365.25), temp.ravel(), te.ravel(), dmax.ravel(), dmin.ravel(),
                             np.maximum(te.ravel() - 24, 0), (lag / level[:, None]).ravel(), (temp - lagt).ravel(),
                             np.full(n * H, fl[0]), np.full(n * H, fl[1]), np.full(n * H, fl[2]), np.full(n * H, fl[3]),
                             np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]).astype(float)
        pred = model.predict(X).reshape(n, H) * level[:, None]
        rating = float(r.get('Rating (MVA)') or 1)
        p = cooling_params(r.get('Cooling', 'ONAF'), r.get('Top-Oil Rise (K)'), r.get('Hot-Spot Gradient (K)'))
        act = thermal(Lv / rating, Tw, p)
        # thermal state at each origin = state after the last 72 h of history (as in forecast())
        h3 = idx0[:, None] + np.arange(-72, 0)[None, :]
        st = thermal(Lv[h3] / rating, Tw[h3], p)['state']
        fth = thermal(pred / rating, temp, p, st)
        fhs = fth['hot_spot'].reshape(n, D, 24).max(2); fk = (pred / rating).reshape(n, D, 24).max(2)
        ahs = act['hot_spot'][0][tgt].reshape(n, D, 24).max(2); ak = (Lv[tgt] / rating).reshape(n, D, 24).max(2)
        fl_ = pred.reshape(n, D, 24).max(2); al = Lv[tgt].reshape(n, D, 24).max(2)
        for k in range(D):
            out.append(pd.DataFrame({'Transformer ID': tid, 'Origin': org, 'Day Ahead': k + 1, 'Date': org + pd.Timedelta(days=k),
                                     'Forecast Max Hot-Spot (C)': fhs[:, k], 'Actual Max Hot-Spot (C)': ahs[:, k],
                                     'Forecast Peak (MVA)': fl_[:, k], 'Actual Peak (MVA)': al[:, k],
                                     'Forecast Peak (%)': fk[:, k] * 100, 'Actual Peak (%)': ak[:, k] * 100}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def historical_study(R, threshold=120.0, alert=117.0, gap_days=3, false_margin=10.0):
    """Events = episodes of days whose actual maximum hot-spot reaches `threshold` (days less than gap_days apart = one episode).
    An alert is a forecast daily maximum hot-spot at or above `alert` (a few degrees below the threshold, to allow for the
    forecast error). An episode is caught when a forecast issued before its first day (at the latest at 00:00 of that day)
    raised an alert for that day; the lead time is the earliest such forecast. False alerts = day-ahead alerts on days whose
    actual maximum stayed below threshold - false_margin."""
    if R.empty:
        return pd.DataFrame(), {}
    act = R[R['Day Ahead'] == 1][['Transformer ID', 'Date', 'Actual Max Hot-Spot (C)', 'Actual Peak (%)']]
    ev = []
    for tid, g in act[act['Actual Max Hot-Spot (C)'] >= threshold].sort_values('Date').groupby('Transformer ID'):
        cur = None
        for _, row in g.iterrows():
            if cur is not None and (row['Date'] - cur['end']).days <= gap_days:
                cur['end'] = row['Date']; cur['days'] += 1; cur['max'] = max(cur['max'], row['Actual Max Hot-Spot (C)'])
            else:
                if cur is not None:
                    ev.append(cur)
                cur = {'Transformer ID': tid, 'start': row['Date'], 'end': row['Date'], 'days': 1, 'max': row['Actual Max Hot-Spot (C)']}
        if cur is not None:
            ev.append(cur)
    rows = []
    for e in ev:
        f = R[(R['Transformer ID'] == e['Transformer ID']) & (R['Date'] == e['start']) & (R['Origin'] <= e['start'])]
        hit = f[f['Forecast Max Hot-Spot (C)'] >= alert]
        lead = int(hit['Day Ahead'].max()) if len(hit) else None
        d1 = f[f['Day Ahead'] == 1]
        rows.append({'Transformer ID': e['Transformer ID'], 'First Day': e['start'].date(), 'Last Day': e['end'].date(), 'Days Above': e['days'],
                     'Actual Max Hot-Spot (C)': round(e['max'], 1),
                     'Day-Ahead Forecast (C)': round(float(d1['Forecast Max Hot-Spot (C)'].iloc[0]), 1) if len(d1) else None,
                     'Warned (days ahead)': lead, 'Caught': 'Yes' if lead else 'No'})
    E = pd.DataFrame(rows)
    d1 = R[R['Day Ahead'] == 1]
    fa = int(((d1['Forecast Max Hot-Spot (C)'] >= alert) & (d1['Actual Max Hot-Spot (C)'] < threshold - false_margin)).sum())
    alerts = int((d1['Forecast Max Hot-Spot (C)'] >= alert).sum())
    by = R.assign(err=(R['Forecast Peak (MVA)'] / R['Actual Peak (MVA)'] - 1).abs() * 100,
                  herr=(R['Forecast Max Hot-Spot (C)'] - R['Actual Max Hot-Spot (C)']).abs())
    summ = {'events': len(E), 'caught': int((E['Caught'] == 'Yes').sum()) if len(E) else 0,
            'median_lead': float(E['Warned (days ahead)'].dropna().median()) if len(E) and E['Warned (days ahead)'].notna().any() else None,
            'false_alerts': fa, 'alerts': alerts,
            'peak_mape_by_day': by.groupby('Day Ahead')['err'].mean().round(2).to_dict(),
            'hs_mae_by_day': by.groupby('Day Ahead')['herr'].mean().round(2).to_dict()}
    return E, summ


# ---------------------------------------------------------------- MW / MVAR from the DCS
_SIDES = ('HV', 'MV', 'LV')


def _find(cols, side, kind):
    """Column of `kind` (MW or MVAR) for `side` (HV/MV/LV, or None for a single unnamed side)."""
    for c in cols:
        n = str(c).upper().replace('_', ' ').replace('-', ' ')
        if kind == 'MVAR':
            ok = bool(re.search(r'\bM\s*VAR\b|\bMVAR\b|\bMVR\b', n))
        else:
            ok = bool(re.search(r'\bMW\b', n)) and 'VAR' not in n
        if not ok:
            continue
        has_side = re.search(r'\b(HV|MV|LV)\b', n.replace('MVAR', '').replace('MW', ''))
        if side is None and not has_side:
            return c
        if side and has_side and has_side.group(1) == side:
            return c
    return None


def loads_from_power(df):
    """Load history from the DCS: MW and MVAR per side (HV, MV, LV) -> MVA per side, power factor and the transformer load
    (the HV side, which carries the whole transformer load; or the single MW/MVAR pair). A file that already has
    'Load (MVA)' is returned unchanged. Columns such as 'HV MW', 'HV MVAR', 'MV_MW', 'MV MVAR' or 'MW', 'MVAR' are recognised."""
    if 'Load (MVA)' in df.columns:
        return df, []
    out = df.copy(); sides = []
    for side in _SIDES + (None,):
        cw, cq = _find(df.columns, side, 'MW'), _find(df.columns, side, 'MVAR')
        if cw is None and cq is None:
            continue
        p = pd.to_numeric(df[cw], errors='coerce') if cw is not None else 0.0
        q = pd.to_numeric(df[cq], errors='coerce') if cq is not None else 0.0
        s = np.sqrt(np.asarray(p, float) ** 2 + np.asarray(q, float) ** 2)
        tag = side or 'HV'
        if tag in sides:
            continue
        out[f'{tag} MVA'] = np.round(s, 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[f'{tag} PF'] = np.round(np.where(s > 0, np.abs(np.asarray(p, float)) / s, np.nan), 3)
        sides.append(tag)
    if not sides:
        return df, []
    if 'HV' in sides:
        out['Load (MVA)'] = out['HV MVA']
    else:                                            # no HV metering: the windings below carry the whole load
        out['Load (MVA)'] = out[[f'{s} MVA' for s in sides]].sum(axis=1)
    return out, sides


def side_summary(df, sides, ratings=None):
    """Peak MVA, peak loading and average power factor of every side in the history."""
    rows = []
    for s in sides:
        mva = pd.to_numeric(df[f'{s} MVA'], errors='coerce')
        r = (ratings or {}).get(s)
        rows.append({'Side': s, 'Peak MVA': round(float(mva.max()), 2), 'Average MVA': round(float(mva.mean()), 2),
                     'Peak loading (%)': round(float(mva.max()) / r * 100, 1) if r else None,
                     'Average power factor': round(float(pd.to_numeric(df[f'{s} PF'], errors='coerce').mean()), 3)})
    return pd.DataFrame(rows)


def demo_power(loads, seed=3):
    """Demo DCS file: HV MW/MVAR from the demo MVA, split into MV and LV windings."""
    rng = np.random.default_rng(seed)
    L = loads.copy()
    pf = np.clip(0.93 + 0.03 * np.sin(np.arange(len(L)) * 2 * np.pi / 24) + rng.normal(0, 0.005, len(L)), 0.85, 0.99)
    s = L['Load (MVA)'].to_numpy(float)
    L['HV MW'] = np.round(s * pf, 2); L['HV MVAR'] = np.round(s * np.sqrt(1 - pf ** 2), 2)
    L['MV MW'] = np.round(L['HV MW'] * 0.7, 2); L['MV MVAR'] = np.round(L['HV MVAR'] * 0.72, 2)
    L['LV MW'] = np.round(L['HV MW'] * 0.3 * 0.99, 2); L['LV MVAR'] = np.round(L['HV MVAR'] * 0.27, 2)
    return L.drop(columns=['Load (MVA)'])
