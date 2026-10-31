# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential.
# ------------------------------------------------------------------------------
# PowerGuard AI - Core Risk Engine
# One place for every rule the dashboard uses, so all tabs (single assessment,
# batch, overall risk, historical study) give the same answer for the same data.
# ==============================================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. Model feature names (must match the trained Random Forest models)
# ------------------------------------------------------------------------------
# Only the 9 dissolved gases: they are what online DGA monitors measure. Oil physical
# properties (DBDS, PF, IFT, breakdown voltage, water) need manual lab tests and are not used.
DGA_FEATURES = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene',
                'Ethane', 'Acethylene']
THERMAL_FEATURES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month']

# Typical healthy values (also used as defaults in the input forms)
DGA_DEFAULTS = {'Hydrogen': 5.0, 'Oxigen': 500.0, 'Nitrogen': 10000.0, 'Methane': 2.0,
                'CO': 100.0, 'CO2': 300.0, 'Ethylene': 1.0, 'Ethane': 5.0,
                'Acethylene': 0.0}

# ------------------------------------------------------------------------------
# 2. Settings (all thresholds in one place - adjust per client / voltage class)
# ------------------------------------------------------------------------------
# Degradation Index is shown on the model's native 0-100 scale.
DI_SAFE_MAX = 30.0       # <= 30  -> SAFE
DI_WARNING_MAX = 45.0    # <= 45  -> WARNING, above -> RISKY
DI_PENALTY_POINTS = 8.0  # index points added per critical-factor point

# DGA risk transfer function: index 20 -> 0 %, index 50 -> 100 %
DGA_RISK_ZERO, DGA_RISK_SPAN = 20.0, 30.0
# Thermal risk transfer function: 40 C -> 0 %, 85 C -> 100 %
# Thermal risk transfer function aligned with typical top-oil settings: alarm 90 C, trip 100 C.
# 65 C -> 0 %, 90 C (alarm) -> ~71 %, 100 C (trip) -> 100 %
THERMAL_RISK_ZERO, THERMAL_RISK_SPAN = 65.0, 35.0
TOP_OIL_ALARM_C, TOP_OIL_TRIP_C = 90.0, 100.0
THERMAL_CLAMP = (20.0, 120.0)

# Fusion
W_DGA, W_THERMAL = 0.6, 0.4
CRITICAL_DIMENSION = 70.0       # a single dimension at/above this drives the result
OVERRIDE_TEMP = 90.0            # expert override: top-oil at the alarm setting
OVERRIDE_DI = 45.0              # expert override: degradation index
OVERRIDE_FLOOR = 75.0           # overall risk floor when an override fires

# Overall status bands
RISK_LOW_MAX = 35.0
RISK_MEDIUM_MAX = 70.0

# Key-gas condition limits (ppm), IEEE C57.104 four-condition method.
# Upper bound of Conditions 1, 2 and 3; above the last value = Condition 4.
# Verify against the edition the client uses before a commercial deployment.
GAS_CONDITION_LIMITS = {
    'Hydrogen':   (100, 700, 1800),
    'Methane':    (120, 400, 1000),
    'Acethylene': (1, 9, 35),
    'Ethylene':   (50, 100, 200),
    'Ethane':     (65, 100, 150),
    'CO':         (350, 570, 1400),
    'CO2':        (2500, 4000, 10000),
}
GAS_LABELS = {'Hydrogen': 'H2', 'Methane': 'CH4', 'Acethylene': 'C2H2', 'Ethylene': 'C2H4',
              'Ethane': 'C2H6', 'CO': 'CO', 'CO2': 'CO2'}
# Minimum overall risk implied by the worst gas condition
CONDITION_OVERALL_FLOOR = {1: 0.0, 2: 40.0, 3: 75.0, 4: 90.0}
# Minimum DGA-dimension risk implied by the worst gas condition
CONDITION_DGA_FLOOR = {1: 0.0, 2: 60.0, 3: 85.0, 4: 100.0}
# When every key gas is within IEEE Condition 1 (normal), the DGA risk is capped here, so a
# transformer the standard calls normal is never flagged on the model score alone.
CONDITION1_DGA_CAP = 35.0

# Gas guardrails: (feature, test, limit, points, message)
GAS_GUARDRAILS = [
    ('CO', 'gt', 500.0, 1.5, "High Carbon Monoxide (>500 ppm): severe carbonization of paper insulation."),
    ('Acethylene', 'gt', 10.0, 2.0, "High Acetylene (>10 ppm): active electrical arcing inside the tank."),
]

# Thermal model was trained on a small distribution transformer; inputs are scaled to it.
THERMAL_BASE_KVA = 50.0
SUMMER_MONTHS = (5, 6, 7, 8, 9)
SUMMER_PEAK_HOURS = (11, 16)
SUMMER_PENALTY_C = 0.0     # removed: the thermal model already accounts for the month (no double counting)
LOAD_PENALTY_START = 80.0   # % load factor
LOAD_PENALTY_C_PER_PCT = 1.2
MEASURED_DEVIATION_ALARM_C = 5.0   # |measured - predicted| above this raises a flag
DEVIATION_ALARM_FLOOR = 40.0       # overall risk floor (WATCH) when measured is much hotter than predicted


# ------------------------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------------------------
def to_kva(capacity, unit):
    """Rated capacity to kVA. unit is 'MVA' or 'kVA'."""
    return float(capacity) * (1000.0 if unit == 'MVA' else 1.0)


def to_kw(value, unit):
    """Active/reactive power to kW/kVAR. unit is 'MVA' (MW/MVAR inputs) or 'kVA'."""
    return float(value) * (1000.0 if unit == 'MVA' else 1.0)


def model_agreement(model, input_df):
    """Agreement between the forest's trees, shown as a percentage (65-99)."""
    try:
        preds = np.array([tree.predict(input_df.values)[0] for tree in model.estimators_])
        conf = 98.5 - (np.std(preds) * 2.5)
        return float(max(65.0, min(99.0, conf)))
    except Exception:
        return 92.5


def gas_condition(feature, value):
    limits = GAS_CONDITION_LIMITS.get(feature)
    if limits is None or value is None or pd.isna(value):
        return 1
    for cond, lim in enumerate(limits, start=1):
        if value <= lim:
            return cond
    return 4


def gas_conditions(dga):
    """Return {feature: condition 1-4} and the worst condition."""
    conds = {f: gas_condition(f, dga.get(f)) for f in GAS_CONDITION_LIMITS}
    return conds, max(conds.values()) if conds else 1


def duval_triangle_1(ch4, c2h4, c2h2):
    """Duval Triangle 1 (IEC 60599). Applied only when fault gases exceed Condition 1."""
    above = (gas_condition('Methane', ch4) > 1 or gas_condition('Ethylene', c2h4) > 1
             or gas_condition('Acethylene', c2h2) > 1)
    total = ch4 + c2h4 + c2h2
    if total <= 0 or not above:
        return "Not applicable (CH4, C2H4 and C2H2 below IEEE Condition 1 limits)"
    p_ch4, p_c2h4, p_c2h2 = (ch4 / total) * 100, (c2h4 / total) * 100, (c2h2 / total) * 100
    if p_ch4 >= 98:
        return "PD (Partial Discharge)"
    if p_c2h2 < 4 and p_c2h4 < 20:
        return "T1 (Thermal Fault < 300°C)"
    if p_c2h2 < 4 and p_c2h4 < 50:
        return "T2 (Thermal Fault 300-700°C)"
    if p_c2h2 < 15 and p_c2h4 >= 50:
        return "T3 (Thermal Fault > 700°C)"
    if p_c2h2 > 13 and p_c2h4 < 23:
        return "D1 (Low Energy Discharge)"
    if (p_c2h2 > 13 and p_c2h4 < 40) or (p_c2h2 > 29 and p_c2h4 >= 40):
        return "D2 (High Energy Discharge)"
    return "DT (Mixed Thermal & Electrical Faults)"


def duval_percentages(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total <= 0:
        return 33.33, 33.33, 33.33
    return (ch4 / total) * 100, (c2h4 / total) * 100, (c2h2 / total) * 100


# ------------------------------------------------------------------------------
# 4. DGA & oil-quality dimension
# ------------------------------------------------------------------------------
def assess_dga(dga, model_health, model_life=None):
    """dga: dict with the 9 gas DGA_FEATURES. Returns a dict with every DGA result."""
    row = {f: float(dga.get(f, DGA_DEFAULTS[f])) for f in DGA_FEATURES}
    X = pd.DataFrame([row])[DGA_FEATURES]

    raw_index = float(model_health.predict(X)[0])
    life_years = float(model_life.predict(X)[0]) if model_life is not None else None

    points, findings = 0.0, []
    for feat, test, limit, pts, msg in GAS_GUARDRAILS:
        val = row[feat]
        if (test == 'lt' and val < limit) or (test == 'gt' and val > limit):
            points += pts
            findings.append(msg)

    index = raw_index
    if points > 0:
        index = min(100.0, raw_index + points * DI_PENALTY_POINTS)

    conds, worst = gas_conditions(row)
    for feat, c in conds.items():
        if c >= 2:
            label = {2: "Condition 2 (above normal)", 3: "Condition 3 (high decomposition)",
                     4: "Condition 4 (excessive decomposition)"}[c]
            findings.append(f"{GAS_LABELS[feat]} = {row[feat]:,.0f} ppm: IEEE {label}.")

    risk = min(100.0, max(0.0, (index - DGA_RISK_ZERO) / DGA_RISK_SPAN * 100.0))
    risk = max(risk, CONDITION_DGA_FLOOR[worst])
    if worst == 1:
        risk = min(risk, CONDITION1_DGA_CAP)

    if worst == 1:
        status = "SAFE"       # all key gases within IEEE normal limits
    elif index <= DI_WARNING_MAX and worst <= 2:
        status = "WARNING"
    else:
        status = "RISKY"

    return {
        'inputs': row, 'input_df': X, 'raw_index': raw_index, 'index': index,
        'penalty_points': points, 'life_years': life_years,
        'gas_conditions': conds, 'worst_condition': worst,
        'risk': risk, 'status': status, 'findings': findings,
        'duval': duval_triangle_1(row['Methane'], row['Ethylene'], row['Acethylene']),
        'agreement': model_agreement(model_health, X),
    }


# ------------------------------------------------------------------------------
# 5. Thermal dimension
# ------------------------------------------------------------------------------
def assess_thermal(loads_kw, capacity_kva, hour, month, model_thermal, measured_top_oil=None):
    """loads_kw: dict HUFL, HULL, MUFL, MULL, LUFL, LULL in kW/kVAR."""
    total_active = loads_kw['HUFL'] + loads_kw['MUFL'] + loads_kw['LUFL']
    load_factor = (total_active / capacity_kva) * 100.0 if capacity_kva > 0 else 0.0
    ratio = THERMAL_BASE_KVA / capacity_kva if capacity_kva > 0 else 1.0

    X = pd.DataFrame([[loads_kw['HUFL'] * ratio, loads_kw['HULL'] * ratio,
                       loads_kw['MUFL'] * ratio, loads_kw['MULL'] * ratio,
                       loads_kw['LUFL'] * ratio, loads_kw['LULL'] * ratio,
                       int(hour), int(month)]], columns=THERMAL_FEATURES)
    base_temp = float(model_thermal.predict(X)[0])

    load_pen = (load_factor - LOAD_PENALTY_START) * LOAD_PENALTY_C_PER_PCT if load_factor > LOAD_PENALTY_START else 0.0
    summer = int(month) in SUMMER_MONTHS and SUMMER_PEAK_HOURS[0] <= int(hour) <= SUMMER_PEAK_HOURS[1]
    amb_pen = SUMMER_PENALTY_C if summer else 0.0
    temp = max(THERMAL_CLAMP[0], min(THERMAL_CLAMP[1], base_temp + load_pen + amb_pen))

    findings = []
    # The model always predicts. When a measured DCS value exists, both are compared and the risk
    # is assessed on the HIGHER of the two (protect the asset: never trust a low prediction over a hot reading).
    measured, deviation, deviation_flag = None, None, None
    assessed, basis = temp, 'Predicted'
    if measured_top_oil is not None and not pd.isna(measured_top_oil) and float(measured_top_oil) > 0:
        measured = float(measured_top_oil)
        deviation = measured - temp
        if measured > temp:
            assessed, basis = measured, 'Measured'
        if deviation > MEASURED_DEVIATION_ALARM_C:
            deviation_flag = 'ABOVE'
            findings.append(f"🚨 TEMPERATURE MISMATCH ALARM: measured top-oil {measured:.1f} °C is {deviation:.1f} °C above "
                            f"the model's expected {temp:.1f} °C at this load. Risk is assessed on the measured value. "
                            f"Verify the reading, then check cooling (fans, pumps, radiators) and oil level.")
        elif deviation < -MEASURED_DEVIATION_ALARM_C:
            deviation_flag = 'BELOW'
            findings.append(f"Measured top-oil {measured:.1f} °C is {-deviation:.1f} °C below the model's expected value: "
                            f"risk is kept on the higher (predicted) value. The model over-predicts here, or the sensor "
                            f"needs checking.")
        else:
            deviation_flag = 'MATCH'

    if assessed >= TOP_OIL_TRIP_C:
        findings.append(f"Trip Level ({assessed:.1f} °C, {basis.lower()}): top-oil at or above the {TOP_OIL_TRIP_C:.0f} °C trip setting.")
    elif assessed >= TOP_OIL_ALARM_C:
        findings.append(f"Alarm Level ({assessed:.1f} °C, {basis.lower()}): top-oil at or above the {TOP_OIL_ALARM_C:.0f} °C alarm setting. Check cooling and loading.")
    if load_factor > LOAD_PENALTY_START:
        findings.append(f"High Load Factor ({load_factor:.1f}%): accelerated thermal aging expected.")
    if summer and assessed >= TOP_OIL_ALARM_C - 10:
        findings.append("Summer peak hours: ambient heat is adding stress to the cooling system.")

    risk = min(100.0, max(0.0, (assessed - THERMAL_RISK_ZERO) / THERMAL_RISK_SPAN * 100.0))
    status = "NORMAL" if assessed < TOP_OIL_ALARM_C else ("WARNING" if assessed < TOP_OIL_TRIP_C else "RISKY")

    return {
        'input_df': X, 'base_temp': base_temp, 'temp': temp, 'assessed_temp': assessed, 'temp_basis': basis,
        'load_factor': load_factor,
        'load_penalty': load_pen, 'ambient_penalty': amb_pen, 'risk': risk,
        'status': status, 'findings': findings, 'measured': measured, 'measured_deviation': deviation,
        'deviation_flag': deviation_flag,
        'agreement': model_agreement(model_thermal, X),
    }



# ------------------------------------------------------------------------------
# 5b. Insulation life (IEEE C57.91 thermal aging of paper insulation)
# ------------------------------------------------------------------------------
NORMAL_INSULATION_LIFE_H = 180000.0          # IEEE C57.91 normal life at reference hot spot (~20.5 years)
COOLING_EXPONENT_M = {'ONAN': 0.8, 'ONAF': 0.8, 'OFAF': 0.8, 'ODAF': 1.0}   # winding exponent m
HOTSPOT_BUBBLE_LIMIT_C = 140.0


def insulation_life(top_oil_c, load_factor_pct, age_years, cooling='ONAN', upgraded_paper=True,
                    rated_hotspot_gradient=25.0, past_aging_factor=0.5):
    """Hot spot = top-oil + rated gradient x K^(2m). Aging factor FAA from the Arrhenius
    relation of IEEE C57.91 (reference 110 C for thermally upgraded paper, 95 C otherwise).
    Returns remaining life at normal aging and if the asset kept running at this operating point."""
    k = max(0.0, float(load_factor_pct) / 100.0)
    m = COOLING_EXPONENT_M.get(cooling, 0.8)
    gradient = rated_hotspot_gradient * (k ** (2 * m))
    hot_spot = float(top_oil_c) + gradient
    ref = 383.0 if upgraded_paper else 368.0            # 110 C or 95 C in kelvin
    faa = float(np.exp(15000.0 / ref - 15000.0 / (hot_spot + 273.0)))

    consumed_h = max(0.0, float(age_years)) * 8760.0 * max(0.0, float(past_aging_factor))
    remaining_h = max(0.0, NORMAL_INSULATION_LIFE_H - consumed_h)
    remaining_normal = remaining_h / 8760.0
    remaining_now = remaining_h / (8760.0 * faa) if faa > 0 else remaining_normal
    lol_per_day_pct = faa * 24.0 / NORMAL_INSULATION_LIFE_H * 100.0

    findings = []
    if hot_spot >= HOTSPOT_BUBBLE_LIMIT_C:
        findings.append(f"Hot spot {hot_spot:.0f} °C ≥ {HOTSPOT_BUBBLE_LIMIT_C:.0f} °C: risk of gas bubble formation in the insulation.")
    if faa >= 2:
        findings.append(f"Insulation is aging {faa:.1f}x faster than normal at this operating point.")
    if remaining_normal <= 0:
        findings.append("Normal insulation life (180,000 h) is already consumed at the given age; condition assessment is essential.")

    return {'hot_spot': hot_spot, 'gradient': gradient, 'faa': faa,
            'remaining_years_normal': remaining_normal, 'remaining_years_now': remaining_now,
            'loss_of_life_per_day_pct': lol_per_day_pct, 'consumed_pct': min(100.0, consumed_h / NORMAL_INSULATION_LIFE_H * 100.0),
            'findings': findings,
            'inputs': {'age_years': age_years, 'cooling': cooling, 'upgraded_paper': upgraded_paper,
                       'rated_hotspot_gradient': rated_hotspot_gradient, 'past_aging_factor': past_aging_factor}}

# ------------------------------------------------------------------------------
# 6. Fusion (hybrid: weighted average + critical dimension + expert overrides)
# ------------------------------------------------------------------------------
def fuse_risk(dga=None, thermal=None):
    """dga / thermal are the dicts returned above, or None when that data is missing.
    Weights are re-normalized over the dimensions that are available."""
    parts = {}
    if dga is not None:
        parts['DGA'] = (dga['risk'], W_DGA)
    if thermal is not None:
        parts['Thermal'] = (thermal['risk'], W_THERMAL)
    if not parts:
        return None

    total_w = sum(w for _, w in parts.values())
    weighted = sum(r * w for r, w in parts.values()) / total_w

    driver_name, (driver_risk, _) = max(parts.items(), key=lambda kv: kv[1][0])
    overall, driver, reasons = weighted, 'Combined', []

    if driver_risk >= CRITICAL_DIMENSION and driver_risk > overall:
        overall, driver = driver_risk, driver_name
        reasons.append(f"{driver_name} risk {driver_risk:.0f}% is above the {CRITICAL_DIMENSION:.0f}% critical limit.")

    if thermal is not None and thermal['assessed_temp'] >= OVERRIDE_TEMP and overall < OVERRIDE_FLOOR:
        overall, driver = OVERRIDE_FLOOR, 'Thermal'
        reasons.append(f"Expert override: {thermal['temp_basis'].lower()} top-oil {thermal['assessed_temp']:.1f} °C "
                       f"≥ {OVERRIDE_TEMP:.0f} °C.")
    if thermal is not None and thermal.get('deviation_flag') == 'ABOVE' and overall < DEVIATION_ALARM_FLOOR:
        overall, driver = DEVIATION_ALARM_FLOOR, 'Thermal'
        reasons.append(f"Temperature mismatch alarm: measured top-oil is {thermal['measured_deviation']:.1f} °C hotter "
                       f"than the model expects. The asset cannot be rated low risk until this is explained "
                       f"(minimum {DEVIATION_ALARM_FLOOR:.0f}%).")
    if dga is not None and dga['index'] > OVERRIDE_DI and dga['worst_condition'] >= 2 and overall < OVERRIDE_FLOOR:
        overall, driver = OVERRIDE_FLOOR, 'DGA'
        reasons.append(f"Expert override: degradation index {dga['index']:.1f} > {OVERRIDE_DI:.0f}.")
    if dga is not None:
        floor = CONDITION_OVERALL_FLOOR[dga['worst_condition']]
        if overall < floor:
            overall, driver = floor, 'DGA'
            reasons.append(f"IEEE gas Condition {dga['worst_condition']} sets a minimum overall risk of {floor:.0f}%.")

    overall = min(100.0, overall)
    missing = [d for d in ('DGA', 'Thermal') if d not in parts]
    band = overall_band(overall, driver)
    return {'overall': overall, 'weighted': weighted, 'driver': driver,
            'reasons': reasons, 'missing': missing, **band}


def overall_band(overall, driver='Combined'):
    if overall <= RISK_LOW_MAX:
        return {'status': 'EXCELLENT', 'label': 'EXCELLENT (Low Risk)', 'color': '#1a9e5c',
                'action': 'Continue normal operation. Next DGA sample as per schedule.'}
    if overall <= RISK_MEDIUM_MAX:
        action = {'DGA': 'Re-sample oil and shorten the DGA interval. Plan inspection.',
                  'Thermal': 'Reduce load and check cooling (fans, pumps, radiators).'}.get(
            driver, 'Schedule preventive maintenance. Review loading and oil condition.')
        return {'status': 'WATCH', 'label': 'WATCH (Medium Risk)', 'color': '#d99a00',
                'action': action}
    action = {'DGA': 'URGENT: internal fault suspected. Confirm with a lab re-sample and plan isolation and inspection.',
              'Thermal': 'URGENT: reduce load immediately and run all cooling stages. Inspect the cooling system.'}.get(
        driver, 'URGENT: isolate the transformer and initiate emergency FMEA.')
    return {'status': 'ACTION', 'label': 'ACTION REQUIRED (High Risk)', 'color': '#d63a3a',
            'action': action}


# ------------------------------------------------------------------------------
# 6b. FMEA: RPN = Severity x Occurrence x Detection (each 1-10)
# ------------------------------------------------------------------------------
# Severity of each failure mode (consequence if it develops to failure).
FMEA_SEVERITY = {
    'D2': ('High-energy arcing (D2)', 10),
    'T3': ('Thermal fault > 700 °C (T3)', 9),
    'D1': ('Low-energy discharge / sparking (D1)', 8),
    'DT': ('Mixed thermal + electrical fault (DT)', 8),
    'T2': ('Thermal fault 300-700 °C (T2)', 7),
    'PD': ('Partial discharge (PD)', 6),
    'T1': ('Thermal fault < 300 °C (T1)', 5),
    'PAPER': ('Paper insulation degradation (CO / CO2)', 7),
    'AGEING': ('Normal insulation ageing (no active fault)', 4),
    'THERMAL': ('Overheating: overload or cooling failure', 7),
    'COOLING': ('Cooling deficiency (measured much hotter than expected)', 7),
}
# Detection: PowerGuard is always continuous online monitoring (DGA monitor + DCS).
DETECTION_ONLINE = 2
RPN_MEDIUM_MIN, RPN_HIGH_MIN = 40, 100      # max possible RPN with online detection = 10 x 10 x 2 = 200


def _clamp_score(x):
    return int(max(1, min(10, x)))


def _occurrence(risk_pct):
    """Occurrence 1-10 from the dimension risk (0-100 %)."""
    return _clamp_score(int(np.ceil(risk_pct / 10.0)))


def rpn_level(rpn):
    if rpn >= RPN_HIGH_MIN:
        return 'High'
    if rpn >= RPN_MEDIUM_MIN:
        return 'Medium'
    return 'Low'


def fmea_assessment(dga=None, thermal=None):
    """Builds the FMEA rows for the active failure modes and returns the governing one (highest RPN)."""
    rows = []

    def add(key, source, risk):
        name, sev = FMEA_SEVERITY[key]
        s, o, d = sev, _occurrence(risk), DETECTION_ONLINE
        rows.append({'Failure Mode': name, 'Source': source, 'S': s, 'O': o, 'D': d,
                     'RPN': s * o * d, 'RPN Level': rpn_level(s * o * d)})

    if dga is not None:
        code = dga['duval'].split(' ')[0]
        if code in FMEA_SEVERITY and code not in ('PAPER', 'AGEING', 'THERMAL', 'COOLING'):
            add(code, 'DGA (Duval)', dga['risk'])
        conds = dga['gas_conditions']
        if max(conds.get('CO', 1), conds.get('CO2', 1)) >= 2:
            add('PAPER', 'DGA (CO / CO2)', dga['risk'])
    if thermal is not None and thermal.get('deviation_flag') == 'ABOVE':
        add('COOLING', 'Thermal (measured vs model)', max(thermal['risk'], DEVIATION_ALARM_FLOOR))
    elif thermal is not None and thermal['status'] != 'NORMAL':    # top oil at or above the alarm setting
        add('THERMAL', 'Thermal', thermal['risk'])
    if not rows:                                                   # no active failure mode
        risks = [x['risk'] for x in (dga, thermal) if x is not None]
        if not risks:
            return None
        add('AGEING', 'DGA + Thermal' if len(risks) == 2 else ('DGA' if dga else 'Thermal'), max(risks))
    rows.sort(key=lambda r: r['RPN'], reverse=True)
    return {'governing': rows[0], 'modes': rows}


# ------------------------------------------------------------------------------
# 7. Historical study (replay an asset's history in time order)
# ------------------------------------------------------------------------------
def thermal_inputs_from_row(row, unit):
    """Build kW/kVAR load dict from a SCADA row (columns in MW/MVAR or kW/kVAR)."""
    return {c: to_kw(row.get(c, 0.0) or 0.0, unit) for c in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']}


def run_historical_study(dga_df, scada_df, events_df, capacity_kva_by_asset, unit,
                         model_health, model_thermal, alert_threshold=70.0,
                         lookback_days=365, scada_window_hours=24):
    """Replays every DGA sample in date order and fuses it with the thermal state
    seen in the SCADA window before the sample. Uses only data available at that date."""
    dga_df = dga_df.copy()
    dga_df['Date'] = pd.to_datetime(dga_df['Date'])
    dga_df = dga_df.sort_values(['Asset ID', 'Date'])

    if scada_df is not None and len(scada_df):
        scada_df = scada_df.copy()
        scada_df['Timestamp'] = pd.to_datetime(scada_df['Timestamp'])
        scada_df = scada_df.sort_values(['Asset ID', 'Timestamp'])

    rows = []
    for asset, g in dga_df.groupby('Asset ID'):
        cap_kva = capacity_kva_by_asset.get(asset)
        prev = None
        for _, r in g.iterrows():
            dga_vals = {f: r.get(f, DGA_DEFAULTS[f]) for f in DGA_FEATURES}
            dga_vals = {f: (DGA_DEFAULTS[f] if pd.isna(v) else float(v)) for f, v in dga_vals.items()}
            d = assess_dga(dga_vals, model_health)

            # Rate of change of key fault gases since the previous sample
            rates, rising = {}, []
            if prev is not None:
                days = max(1.0, (r['Date'] - prev['Date']).days)
                for f in ('Hydrogen', 'Methane', 'Acethylene', 'Ethylene', 'Ethane', 'CO'):
                    rate = (dga_vals[f] - prev['vals'][f]) / days * 30.0
                    rates[f] = rate
                    if (prev['vals'][f] > 0 and dga_vals[f] > 1.5 * prev['vals'][f]
                            and gas_condition(f, dga_vals[f]) >= 2):
                        rising.append(GAS_LABELS[f])

            t = None
            if scada_df is not None and len(scada_df) and cap_kva:
                win = scada_df[(scada_df['Asset ID'] == asset) &
                               (scada_df['Timestamp'] <= r['Date'] + pd.Timedelta(hours=23, minutes=59)) &
                               (scada_df['Timestamp'] > r['Date'] - pd.Timedelta(hours=scada_window_hours))]
                results = []
                for _, s in win.iterrows():
                    meas = s.get('Top Oil Temp') if 'Top Oil Temp' in s else None
                    results.append(assess_thermal(thermal_inputs_from_row(s, unit), cap_kva,
                                                  s['Timestamp'].hour, s['Timestamp'].month, model_thermal, meas))
                if results:
                    # Sustained (average) temperature drives insulation aging; the peak is reported too.
                    peak = max(x['assessed_temp'] for x in results)
                    avg = float(np.mean([x['assessed_temp'] for x in results]))
                    t = dict(max(results, key=lambda x: x['assessed_temp']))
                    t['peak_temp'] = peak
                    t['temp'] = t['assessed_temp'] = avg
                    t['risk'] = min(100.0, max(0.0, (avg - THERMAL_RISK_ZERO) / THERMAL_RISK_SPAN * 100.0))
                    t['status'] = ("NORMAL" if avg < TOP_OIL_ALARM_C else
                                   "WARNING" if avg < TOP_OIL_TRIP_C else "RISKY")

            f = fuse_risk(d, t)
            if rising and f['overall'] < RISK_LOW_MAX + 5:
                f['overall'], f['driver'] = RISK_LOW_MAX + 5, 'DGA'
                f.update(overall_band(f['overall'], 'DGA'))
                f['reasons'].append("Fast-rising fault gas: " + ", ".join(rising))

            rows.append({
                'Asset ID': asset, 'Date': r['Date'],
                'Degradation Index': round(d['index'], 1), 'Worst Gas Condition': d['worst_condition'],
                'DGA Risk (%)': round(d['risk'], 1),
                'Avg Pred Temp (°C)': round(t['temp'], 1) if t else None,
                'Peak Pred Temp (°C)': round(t['peak_temp'], 1) if t else None,
                'Thermal Risk (%)': round(t['risk'], 1) if t else None,
                'Overall Risk (%)': round(f['overall'], 1), 'Status': f['status'],
                'Driver': f['driver'], 'Duval': d['duval'],
                'Rising Gases': ", ".join(rising), 'Thermal Data': 'Yes' if t else 'No',
                'H2 rate (ppm/30d)': round(rates.get('Hydrogen', 0.0), 1) if rates else None,
                'C2H2 rate (ppm/30d)': round(rates.get('Acethylene', 0.0), 1) if rates else None,
            })
            prev = {'Date': r['Date'], 'vals': dga_vals}

    timeline = pd.DataFrame(rows)

    # Alert episodes: consecutive alert samples of one asset, allowing gaps up to EPISODE_GAP_DAYS
    # between them (a fault that stays in alert across several samples is one episode).
    EPISODE_GAP_DAYS = 120
    episodes = []          # (asset, start, end, driver of the first alert)
    if len(timeline):
        alerts = timeline[timeline['Overall Risk (%)'] >= alert_threshold].sort_values(['Asset ID', 'Date'])
        for asset, a in alerts.groupby('Asset ID'):
            cur = None
            for _, ar in a.iterrows():
                if cur is not None and (ar['Date'] - cur[2]).days <= EPISODE_GAP_DAYS:
                    cur[2] = ar['Date']
                else:
                    if cur is not None:
                        episodes.append(tuple(cur))
                    cur = [asset, ar['Date'], ar['Date'], ar['Driver']]
            if cur is not None:
                episodes.append(tuple(cur))

    ev = None
    if events_df is not None and len(events_df):
        ev = events_df.copy()
        ev['Date'] = pd.to_datetime(ev['Date'])

    # Lead time: start of the alert episode that was still active (within the look-back window)
    # when the event happened.
    summary = []
    if ev is not None and len(timeline):
        for _, e in ev.iterrows():
            prior = [ep for ep in episodes if ep[0] == e['Asset ID'] and ep[1] < e['Date']
                     and (e['Date'] - ep[2]).days <= lookback_days]
            first = min(ep[1] for ep in prior) if prior else None
            summary.append({
                'Asset ID': e['Asset ID'], 'Event Date': e['Date'].date(),
                'Event': e.get('Event', ''),
                'First Alert Before Event': first.date() if first is not None else None,
                'Lead Time (days)': int((e['Date'] - first).days) if first is not None else None,
                'Caught': 'Yes' if first is not None else 'No',
            })
    events_summary = pd.DataFrame(summary)

    # False alarm: an episode with no recorded event between its start and its end + look-back window,
    # split by what drove the alert (DGA = possible internal fault, Thermal = loading/cooling).
    false_alarms = {'DGA': 0, 'Thermal': 0, 'Combined': 0}
    for asset, st, en, drv in episodes:
        followed = False
        if ev is not None:
            d = ev[ev['Asset ID'] == asset]['Date']
            followed = ((d >= st) & (d <= en + pd.Timedelta(days=lookback_days))).any()
        if not followed:
            false_alarms[drv if drv in false_alarms else 'Combined'] += 1

    return timeline, events_summary, false_alarms


def demo_history(seed=7):
    """Synthetic demo data (clearly labeled in the UI) to show how the study works."""
    rng = np.random.default_rng(seed)
    dga_rows, scada_rows = [], []
    dates = pd.date_range('2024-01-15', '2026-06-15', freq='90D')
    assets = {'TR-01': 'healthy', 'TR-02': 'arcing', 'TR-03': 'thermal'}
    for asset, kind in assets.items():
        for i, dt in enumerate(dates):
            k = i / (len(dates) - 1)
            v = dict(DGA_DEFAULTS)
            v.update({'Hydrogen': 20 + rng.normal(0, 3), 'Methane': 15 + rng.normal(0, 2),
                      'Ethylene': 8 + rng.normal(0, 1), 'Ethane': 12 + rng.normal(0, 2),
                      'Acethylene': 0.0, 'CO': 180 + rng.normal(0, 15), 'CO2': 1500 + rng.normal(0, 80)})
            if kind == 'arcing' and k > 0.45:
                s = (k - 0.45) / 0.55
                v['Acethylene'] = 2 + 60 * s ** 1.5
                v['Hydrogen'] = 30 + 900 * s ** 1.5
                v['Ethylene'] = 10 + 70 * s
            if kind == 'thermal' and k > 0.35:
                s = (k - 0.35) / 0.65
                v['Ethylene'] = 10 + 260 * s ** 1.3
                v['Methane'] = 20 + 300 * s ** 1.3
                v['Ethane'] = 15 + 90 * s
                v['CO'] = 200 + 500 * s
            v = {f: max(0.0, float(x)) for f, x in v.items()}
            dga_rows.append({'Asset ID': asset, 'Date': dt.date(), **v})
            for h in (3, 14):
                ts = pd.Timestamp(dt) + pd.Timedelta(hours=h)
                base = 0.55 + (0.35 if kind == 'thermal' and k > 0.6 else 0.0)
                p = base * (1.15 if h == 14 else 0.8) * 120  # MW on a 150 MVA unit
                scada_rows.append({'Asset ID': asset, 'Timestamp': ts,
                                   'HUFL': p * 0.5, 'HULL': p * 0.12, 'MUFL': p * 0.32,
                                   'MULL': p * 0.06, 'LUFL': p * 0.18, 'LULL': p * 0.03})
    events = pd.DataFrame([
        {'Asset ID': 'TR-02', 'Date': '2026-05-20', 'Event': 'Buchholz trip - internal arcing found'},
        {'Asset ID': 'TR-03', 'Date': '2026-04-10', 'Event': 'Forced outage - hot spot on tap changer'},
    ])
    assets_df = pd.DataFrame([{'Asset ID': a, 'Rated Capacity': 150.0} for a in assets])
    return pd.DataFrame(dga_rows), pd.DataFrame(scada_rows), events, assets_df
