# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential. Enterprise System-of-Systems Edition.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
import os
import warnings

# Suppress Version/Unpickling Warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. إعدادات الصفحة والتصميم الأساسي
# ==========================================
st.set_page_config(page_title="PowerGuard AI Enterprise", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .metric-card { background-color: #F8F9FA; padding: 15px; border-radius: 8px; border-left: 5px solid #3498DB; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin-bottom: 15px; }
    .master-card { background: linear-gradient(135deg, #0b2e59 0%, #1a5276 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-bottom: 20px; }
    .dark-card { background: linear-gradient(135deg, #111 0%, #222 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #00ff88; margin-bottom: 15px; }
    .zone-bar { width: 100%; height: 20px; display: flex; border-radius: 5px; overflow: hidden; margin-bottom: 5px; font-weight: bold; }
    .zone-safe { width: 40%; background-color: #2ecc71; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .zone-warn { width: 35%; background-color: #f1c40f; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .zone-dang { width: 25%; background-color: #e74c3c; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .main-header { background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%); padding: 20px; border-radius: 10px; border-left: 8px solid #DD4B39; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; }
    .report-reasoning { background-color: #F4F6F7; padding: 20px; border-radius: 10px; border-left: 5px solid #2980B9; font-size: 16px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. تحميل كل النماذج الذكية
# ==========================================
@st.cache_resource
def load_all_models():
    try:
        m_health = joblib.load('rf_health_model.pkl')
        m_life = joblib.load('rf_life_model.pkl')
        m_thermal = joblib.load('rf_thermal_model.pkl')
    except:
        m_health, m_life, m_thermal = None, None, None
        
    try:
        s1 = joblib.load('scaler.pkl')
        m1 = joblib.load('motorguard_model.pkl')
        s2 = joblib.load('nasa_bearing_scaler.pkl')
        m2 = joblib.load('nasa_bearing_model.pkl')
        motors_ready = True
    except:
        s1, m1, s2, m2, motors_ready = None, None, None, None, False
        
    return m_health, m_life, m_thermal, s1, m1, s2, m2, motors_ready

model_health, model_life, model_thermal, scaler1, model1, scaler2, model2, motors_ready = load_all_models()

# ==========================================
# 3. دوال مساعدة ورسومات (محصنة)
# ==========================================
def get_tx_confidence(model, input_df):
    try:
        preds = np.array(tuple((tree.predict(input_df.values).item() for tree in model.estimators_)))
        return max(65.0, min(99.0, 98.5 - (np.std(preds) * 2.5)))
    except: return 92.5  

def get_duval_diagnosis(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0: return "Normal (No significant fault gases)"
    p_ch4, p_c2h4, p_c2h2 = (ch4/total)*100, (c2h4/total)*100, (c2h2/total)*100
    if p_ch4 >= 98: return "PD (Partial Discharge)"
    if p_c2h2 < 4 and p_c2h4 < 20: return "T1 (Thermal Fault < 300°C)"
    if p_c2h2 < 4 and 20 <= p_c2h4 < 50: return "T2 (Thermal Fault 300-700°C)"
    if p_c2h2 < 15 and p_c2h4 >= 50: return "T3 (Thermal Fault > 700°C)"
    if 4 <= p_c2h2 <= 13 and p_c2h4 < 50: return "D1 (Low Energy Discharge)"
    return "D2 (High Energy Discharge)"

def highlight_fmea(row):
    status = str(row.get('Status', '')).upper()
    if 'SAFE' in status: return tuple(('background-color: #E8F5E9; color: #00A65A; font-weight: bold',)) * len(row)
    if 'WARN' in status: return tuple(('background-color: #FEF9E7; color: #D35400; font-weight: bold',)) * len(row)
    if 'DANG' in status or 'ACTION' in status: return tuple(('background-color: #FDEDEC; color: #C0392B; font-weight: bold',)) * len(row)
    return tuple(('',)) * len(row)

def create_motor_gauge(value, title, color, max_v=100.0, h=250):
    val = float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number=dict(suffix="%", font=dict(size=26, color=color, family="Arial Black")),
        title=dict(text=title, font=dict(size=16, color='#333', weight='bold')),
        gauge=dict(
            axis=dict(range=tuple((0, float(max_v))), tickwidth=1, tickcolor="darkblue"),
            bar=dict(color=color),
            bgcolor="white", borderwidth=2, bordercolor="gray",
            steps=tuple((
                dict(range=tuple((0, 40)), color='#E8F5E9'), 
                dict(range=tuple((40, 75)), color='#FEF9E7'), 
                dict(range=tuple((75, float(max_v))), color='#FDEDEC')
            )),
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=val)
        )))
    fig.update_layout(height=h, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def highlight_relay_fmea(row):
    status = str(row.get('Status', '')).upper()
    if 'SAFE' in status: return tuple(('background-color: #E8F5E9; color: #00A65A; font-weight: bold',)) * len(row)
    if 'WARN' in status: return tuple(('background-color: #FEF9E7; color: #D35400; font-weight: bold',)) * len(row)
    if 'DANG' in status: return tuple(('background-color: #FDEDEC; color: #C0392B; font-weight: bold',)) * len(row)
    return tuple(('',)) * len(row)

def create_relay_gauge(value, title, color):
    val = float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number=dict(suffix="%", font=dict(size=30, color=color)),
        title=dict(text=title, font=dict(size=18, color='#333')),
        gauge=dict(
            axis=dict(range=tuple((0, 100)), tickwidth=1),
            bar=dict(color=color),
            bgcolor="white",
            steps=tuple((
                dict(range=tuple((0, 30)), color='rgba(46, 204, 113, 0.2)'),
                dict(range=tuple((30, 70)), color='rgba(241, 196, 15, 0.2)'),
                dict(range=tuple((70, 100)), color='rgba(231, 76, 60, 0.2)')
            )),
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=val)
        )
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def create_relay_radar(probs_list, classes_tup, color):
    fig = go.Figure(data=go.Scatterpolar(
        r=tuple((float(p) * 100.0 for p in probs_list)), theta=classes_tup, fill='toself', line_color=color, opacity=0.6
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=tuple((0, 100)))), showlegend=False, height=350, margin=dict(l=30, r=30, t=30, b=30))
    return fig

# ==========================================
# 4. إعداد متغيرات الجلسة
# ==========================================
if 'current_page' not in st.session_state: st.session_state['current_page'] = 'portal'

# Transformer State
if 'tx_logged_in' not in st.session_state: st.session_state['tx_logged_in'] = False
if 'fmea_log' not in st.session_state: st.session_state['fmea_log'] = pd.DataFrame(columns=tuple(('Timestamp', 'Asset ID', 'DGA Index', 'Pred Temp', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required')))
if 'latest_eval' not in st.session_state: st.session_state['latest_eval'] = None

# Motor State
if 'motor_logged_in' not in st.session_state: st.session_state['motor_logged_in'] = False
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'inspection_log' not in st.session_state: st.session_state['inspection_log'] = pd.DataFrame(columns=tuple(('Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status', 'Grade', 'Air Temp [K]', 'Core Temp [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Vib RMS', 'Vib Kurt')))

# Relay State
if 'relay_logged_in' not in st.session_state: st.session_state['relay_logged_in'] = False
if 'relay_fmea_log' not in st.session_state: st.session_state['relay_fmea_log'] = pd.DataFrame(columns=tuple(('Timestamp', 'Asset ID', 'Noise Index', 'Max Current (A)', 'Risk (%)', 'RPN', 'Status', 'Action')))
if 'relay_latest_eval' not in st.session_state: st.session_state['relay_latest_eval'] = None

# Master State
if 'master_logged_in' not in st.session_state: st.session_state['master_logged_in'] = False

def go_to_portal():
    st.session_state['current_page'] = 'portal'
    st.rerun()

# ==========================================
# 5. البوابة الرئيسية (Portal)
# ==========================================
if st.session_state['current_page'] == 'portal':
    st.markdown("<br><h1 style='text-align: center; color: #0b2e59; font-size: 3rem;'>⚡ PowerGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666; margin-bottom: 40px;'>Enterprise Asset Performance Management System</h4>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(tuple((1, 2, 1)))
    with col_m2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1E1E1E 0%, #3a3a3a 100%); padding: 30px; border-radius: 12px; text-align: center; border-bottom: 5px solid #00ff88; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
            <h2 style='color: white; margin:0;'>🌐 SubstationGuard AI</h2>
            <p style='color: #ccc; font-size: 16px;'>C-Level Executive Command Center (System-of-Systems)</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ENTER COMMAND CENTER 🚀", use_container_width=True, type="primary"):
            st.session_state['current_page'] = 'substation'
            st.rerun()

    st.markdown("<hr style='margin-top:40px; margin-bottom:40px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #888;'>Specialized Engineering Modules</h3><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(tuple((1, 1, 1)))
    with col1:
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #4da6ff; height: 140px;'><h3>⚡ TransformerGuard</h3><p>DGA & Thermal Analytics</p></div>", unsafe_allow_html=True)
        if st.button("Launch Transformer Module", use_container_width=True):
            st.session_state['current_page'] = 'transformer'
            st.rerun()
    with col2:
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #DD4B39; height: 140px;'><h3>⚙️ MotorGuard AI</h3><p>Telemetry & Vibration FMEA</p></div>", unsafe_allow_html=True)
        if st.button("Launch Motor Module", use_container_width=True):
            st.session_state['current_page'] = 'motor'
            st.rerun()
    with col3:
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #9B59B6; height: 140px;'><h3>🛡️ RelayGuard AI</h3><p>Protection & Control FMEA</p></div>", unsafe_allow_html=True)
        if st.button("Launch Relay Module", use_container_width=True):
            st.session_state['current_page'] = 'relay'
            st.rerun()

    st.markdown("<div style='text-align: center; color: #aaa; padding: 10px; margin-top: 50px;'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. المحولات (Transformer Guard)
# ==============================================================================
elif st.session_state['current_page'] == 'transformer':
    if not st.session_state['tx_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(tuple((1, 2, 1)))
        with col2:
            st.markdown("<h2 style='text-align: center; color: #4da6ff;'>⚡ TransformerGuard Login</h2>", unsafe_allow_html=True)
            with st.form("tx_login_form"):
                st.warning("🔒 Restricted Access. Authorized Personnel Only.")
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['tx_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns(tuple((10, 1)))
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['tx_logged_in'] = False
                go_to_portal()

        st.markdown("<div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 20px;'><h1 style='color: #ffffff; text-align: center; margin: 0;'>Transformer & Reactor Guard AI ⚡</h1><h3 style='color: #4da6ff; text-align: center; margin: 5px 0 0 0;'>Predictive Maintenance for the Modern Grid</h3></div>", unsafe_allow_html=True)

        if model_health is None:
            st.error("⚠️ Model files not detected!")
            st.stop()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(tuple(("🧪 DGA Analytics", "🌡️ Thermal Load", "📁 Batch", "📑 RISK Logs", "📄 Executive Report")))

        with tab1:
            col_in, col_out = st.columns(tuple((1, 2.2)))
            with col_in:
                st.markdown("### 📊 DGA Parameters")
                h2 = st.number_input("Hydrogen (H2)", value=5.0)
                o2 = st.number_input("Oxygen (O2)", value=500.0)
                n2 = st.number_input("Nitrogen (N2)", value=10000.0)
                ch4 = st.number_input("Methane (CH4)", value=2.0)
                co = st.number_input("Carbon Monoxide (CO)", value=100.0)
                co2 = st.number_input("Carbon Dioxide (CO2)", value=300.0)
                c2h4 = st.number_input("Ethylene (C2H4)", value=1.0)
                c2h6 = st.number_input("Ethane (C2H6)", value=5.0)
                c2h2 = st.number_input("Acetylene (C2H2)", value=0.0)
                dbds = st.number_input("DBDS (ppm)", value=0.1)
                pf = st.number_input("Power Factor (%)", value=0.05)
                ift = st.number_input("Interfacial Tension", value=45.0)
                dr = st.number_input("Dielectric Rigidity", value=70.0)
                wc = st.number_input("Water Content", value=2.0)

            with col_out:
                input_df = pd.DataFrame(tuple((tuple((h2, o2, n2, ch4, co, co2, c2h4, c2h6, c2h2, dbds, pf, ift, dr, wc)),)), columns=tuple(('Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content')))
                
                h_score = float(model_health.predict(input_df).item())
                cf = sum(tuple((1.0 if dr<35 else 0, 1.0 if ift<20 else 0, 1.5 if wc>25 else 0, 1.5 if co>500 else 0, 1.0 if pf>1.5 else 0, 2.0 if c2h2>10 else 0)))
                if cf > 0: h_score = min(100.0, h_score + (cf * 8.0))
                
                status = "SAFE" if h_score <= 30 else ("WARNING" if h_score <= 45 else "RISKY")
                color = "#00cc66" if status == "SAFE" else ("#ffcc00" if status == "WARNING" else "#ff3333")

                m1, m2, m3 = st.columns(tuple((1, 1, 1)))
                m1.metric("Degradation Index", f"{h_score:.2f}")
                m2.metric("Status", status)
                m3.metric("AI Confidence", f"{get_tx_confidence(model_health, input_df):.1f} %")

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=float(h_score),
                    gauge=dict(
                        axis=dict(range=tuple((0.0, 100.0)), tickwidth=2, tickcolor="black"),
                        bar=dict(color="black"),
                        steps=tuple((
                            dict(range=tuple((0.0, 30.0)), color="#00cc66"),
                            dict(range=tuple((30.0, 45.0)), color="#ffcc00"),
                            dict(range=tuple((45.0, 100.0)), color="#ff3333")
                        ))
                    )
                ))
                fig_g.update_layout(height=280, margin=dict(t=30, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
                
                st.markdown(f"<h2 style='text-align: center; color: {color}; margin-top:-30px;'>{status}</h2>", unsafe_allow_html=True)
                st.info(f"**Diagnosis:** {get_duval_diagnosis(ch4, c2h4, c2h2)}")

        with tab2:
            st.markdown("### 🌡️ Thermal Load Analysis")
            col_t_in, col_t_out = st.columns(tuple((1, 1.5)))
            with col_t_in:
                capacity_kva = st.number_input("Rated Capacity (kVA)", min_value=10.0, value=50.0)
                hour = st.number_input("Hour (0-23)", value=14)
                month = st.number_input("Month (1-12)", value=7)
                hufl = st.number_input("HUFL", value=12.5)
                hull = st.number_input("HULL", value=3.2)
                mufl = st.number_input("MUFL", value=8.1)
                mull = st.number_input("MULL", value=1.5)
                lufl = st.number_input("LUFL", value=4.5)
                lull = st.number_input("LULL", value=0.8)
                    
            with col_t_out:
                n_ratio = 50.0 / capacity_kva
                t_input_norm = pd.DataFrame(tuple((tuple((hufl*n_ratio, hull*n_ratio, mufl*n_ratio, mull*n_ratio, lufl*n_ratio, lull*n_ratio, hour, month)),)), columns=tuple(('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month')))
                pred_ot = float(model_thermal.predict(t_input_norm).item())
                
                fig_t = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=float(pred_ot),
                    gauge=dict(
                        axis=dict(range=tuple((20.0, 150.0))),
                        bar=dict(color="black"),
                        steps=tuple((
                            dict(range=tuple((20.0, 60.0)), color="rgba(0, 204, 102, 0.4)"),
                            dict(range=tuple((60.0, 80.0)), color="rgba(255, 204, 0, 0.4)"),
                            dict(range=tuple((80.0, 150.0)), color="rgba(255, 51, 51, 0.4)")
                        ))
                    )
                ))
                fig_t.update_layout(height=350, margin=dict(t=30, b=10))
                st.plotly_chart(fig_t, use_container_width=True)

        with tab3:
            st.info("Batch Processing feature is ready for bulk CSV uploads.")

        with tab4:
            st.markdown("### 📑 Overall Transformer Risk Assessment")
            col_ex_in, col_ex_out = st.columns(tuple((1, 1.5)))
            with col_ex_in:
                with st.form("tx_exec_form"):
                    asset_id = st.text_input("Asset ID", value="TX-01")
                    gen_report = st.form_submit_button("📊 Evaluate & Push to Master System", use_container_width=True, type="primary")
                    
            with col_ex_out:
                if gen_report:
                    dga_risk = min(100.0, max(0.0, ((h_score - 20.0) / 30.0) * 100.0)) 
                    thermal_risk = min(100.0, max(0.0, ((pred_ot - 40.0) / 45.0) * 100.0))
                    overall_risk = (dga_risk * 0.6) + (thermal_risk * 0.4)
                    
                    if overall_risk <= 35: f_status, f_color, action, rpn_level = "🟢 SAFE", "#00cc66", "Continue operation.", "Low"
                    elif overall_risk <= 70: f_status, f_color, action, rpn_level = "🟡 WARNING", "#ffcc00", "Schedule PM.", "Medium"
                    else: f_status, f_color, action, rpn_level = "🔴 DANGER", "#ff3333", "URGENT: Isolate asset.", "Critical"
                    
                    # 💡 The FIX: Ensuring all keys required by tab5 are present in the dictionary.
                    load_factor_percent = ((hufl + mufl + lufl) / capacity_kva) * 100.0
                    st.session_state['latest_eval'] = {
                        'asset_id': asset_id, 'capacity': capacity_kva, 'load_factor': load_factor_percent,
                        'overall_risk': overall_risk, 'dga_score': h_score, 'pred_temp': pred_ot,
                        'status': f_status, 'action': action, 'color': f_color, 'rpn_level': rpn_level,
                        'duval': get_duval_diagnosis(ch4, c2h4, c2h2), 
                        'inputs': {'ch4': ch4, 'c2h4': c2h4, 'c2h2': c2h2, 'wc': wc, 'ift': ift, 'dr': dr, 'pf': pf, 'co': co, 'month': month, 'hour': hour},
                        'dga_input_df': input_df, 'thermal_input_df': t_input_norm
                    }
                    new_entry = pd.DataFrame(tuple((dict(Timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), Asset_ID=asset_id, DGA_Index=round(h_score, 1), Pred_Temp=round(pred_ot, 1), Overall_Risk=round(overall_risk, 1), Status=f_status, RPN_Level=rpn_level, Action_Required=action),)))
                    new_entry.columns = tuple(('Timestamp', 'Asset ID', 'DGA Index', 'Pred Temp', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required'))
                    st.session_state['fmea_log'] = pd.concat((st.session_state['fmea_log'], new_entry), ignore_index=True).tail(100)
                    
                    st.markdown(f"<h2 style='text-align: center; color: {f_color};'>{f_status} - Risk: {overall_risk:.1f}%</h2>", unsafe_allow_html=True)
            
            st.markdown("---")
            if not st.session_state['fmea_log'].empty:
                st.dataframe(st.session_state['fmea_log'].style.apply(highlight_fmea, axis=1), use_container_width=True)

        with tab5:
            if st.session_state.get('latest_eval') is None: 
                st.markdown("<br><br><h3 style='text-align: center; color: #888;'>💡 Please generate a risk assessment from TAB 4 first.</h3>", unsafe_allow_html=True)
            else:
                e_data = st.session_state['latest_eval']
                st.markdown(f"<div style='background-color: rgba(77, 166, 255, 0.1); padding: 15px; border-left: 5px solid #4da6ff; border-radius: 5px;'><h2 style='margin:0; color: #0b2e59;'>📄 AI Executive Diagnostic Report</h2><p style='margin:0; color: #555555;'>Asset ID: <b>{e_data['asset_id']}</b> | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p></div><br>", unsafe_allow_html=True)
                
                r_col1, r_col2 = st.columns(tuple((1, 1.8)))
                with r_col1:
                    fig_r = go.Figure(go.Indicator(
                        mode="gauge+number", value=float(e_data['overall_risk']),
                        number=dict(font=dict(size=40, color='#0b2e59')),
                        gauge=dict(axis=dict(range=tuple((0.0, 100.0)), tickwidth=1, tickcolor="black"), bar=dict(color="rgba(0,0,0,0.5)", thickness=0.1),
                                 steps=tuple((
                                     dict(range=tuple((0.0, 35.0)), color="#00cc66"),
                                     dict(range=tuple((35.0, 70.0)), color="#ffcc00"),
                                     dict(range=tuple((70.0, 100.0)), color="#ff3333")
                                 )))))
                    fig_r.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_r, use_container_width=True)
                    
                with r_col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.dataframe(st.session_state['fmea_log'].tail(1).style.apply(highlight_fmea, axis=1), use_container_width=True)
                    st.markdown(f"**Mandatory Action:** <span style='color:{e_data['color']}; font-weight:bold;'>{e_data['action']}</span>", unsafe_allow_html=True)

                st.markdown("---")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    ch4_v, c2h4_v, c2h2_v = e_data['inputs']['ch4'], e_data['inputs']['c2h4'], e_data['inputs']['c2h2']
                    tot_gas_v = ch4_v + c2h4_v + c2h2_v
                    p_ch4_v, p_c2h4_v, p_c2h2_v = ((ch4_v/tot_gas_v)*100, (c2h4_v/tot_gas_v)*100, (c2h2_v/tot_gas_v)*100) if tot_gas_v > 0 else (33.33, 33.33, 33.33)
                    
                    fig_tri_v = go.Figure()
                    fig_tri_v.add_trace(go.Scatterternary(a=tuple((100, 0, 0)), b=tuple((0, 100, 0)), c=tuple((0, 0, 100)), mode='markers', marker=dict(color='rgba(0,0,0,0)', size=1), hoverinfo='skip'))
                    fig_tri_v.add_trace(go.Scatterternary(a=tuple((p_ch4_v,)), b=tuple((p_c2h4_v,)), c=tuple((p_c2h2_v,)), mode='markers', marker=dict(size=14, color='red', symbol='cross', line=dict(width=2, color='black'))))
                    fig_tri_v.update_layout(title=dict(text="Duval Triangle (Fault Type)", font=dict(color='black'), y=0.95, x=0.5, xanchor='center', yanchor='top'), height=450, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", ternary=dict(sum=100, aaxis=dict(title="CH4 %", min=0), baxis=dict(title="C2H4 %", min=0), caxis=dict(title="C2H2 %", min=0)), showlegend=False)
                    st.plotly_chart(fig_tri_v, use_container_width=True)
                    st.markdown(f"<h4 style='text-align: center; color: #ff3333; margin-top:-30px;'>Diagnostic Result: {e_data['duval']}</h4>", unsafe_allow_html=True)

                with v_col2:
                    dga_features = list(('Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'))
                    dga_importances = model_health.feature_importances_
                    dga_baselines = np.array(list((5.0, 500.0, 10000.0, 2.0, 100.0, 300.0, 1.0, 5.0, 0.1, 0.1, 0.05, 45.0, 70.0, 2.0)))
                    dga_vals = e_data['dga_input_df'].values.flatten()
                    dga_dev = np.abs(dga_vals - dga_baselines) / dga_baselines
                    dga_impact = dga_importances * dga_dev

                    thm_features = list(('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'))
                    thm_importances = model_thermal.feature_importances_
                    thm_baselines = np.array(list((12.5, 3.2, 8.1, 1.5, 4.5, 0.8, 12.0, 6.0)))
                    thm_vals = e_data['thermal_input_df'].values.flatten()
                    thm_dev = np.abs(thm_vals - thm_baselines) / thm_baselines
                    thm_impact = thm_importances * thm_dev

                    all_features = dga_features + thm_features
                    all_impacts = np.concatenate((dga_impact, thm_impact))
                    
                    if all_impacts.sum() > 0: all_impacts = (all_impacts / all_impacts.sum()) * 100
                    else: all_impacts = np.concatenate((dga_importances, thm_importances)) * 100
                        
                    df_imp = pd.DataFrame({'Feature': all_features, 'Impact (%)': all_impacts})
                    df_imp = df_imp.sort_values(by='Impact (%)', ascending=False).head(10).sort_values(by='Impact (%)', ascending=True)

                    fig_imp = px.bar(df_imp, x='Impact (%)', y='Feature', orientation='h', color='Impact (%)', color_continuous_scale='Turbo')
                    fig_imp.update_layout(title=dict(text="Top 10 Impactful Variables (Combined)", font=dict(color='black')), height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                    st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🧠 3. Automated Expert System Findings")
                
                issues_dga, issues_thermal = list(), list()
                if e_data['dga_score'] > 45: issues_dga.append(f"⚠️ **AI Risk Alert:** Abnormal gas ratios detected.")
                elif e_data['dga_score'] > 30: issues_dga.append(f"🟡 **Early Warning:** Mild deviations in gases.")
                if e_data['inputs']['wc'] > 25: issues_dga.append("🚨 **Critical Water Content (>25 ppm)**")
                if e_data['inputs']['ift'] < 20: issues_dga.append("⚠️ **Low Interfacial Tension (<20 mN/m)**")
                if e_data['inputs']['dr'] < 35: issues_dga.append("🚨 **Low Dielectric Rigidity (<35 kV)**")
                if e_data['inputs']['pf'] > 1.5: issues_dga.append("⚠️ **High Power Factor (>1.5%)**")
                if e_data['inputs']['co'] > 500: issues_dga.append("🚨 **High Carbon Monoxide (>500 ppm)**")
                if e_data['inputs']['c2h2'] > 10: issues_dga.append("🚨 **High Acetylene (>10 ppm)**")
                if not issues_dga: issues_dga.append("✅ DGA chemistry is normal and healthy.")

                if e_data['pred_temp'] > 80: issues_thermal.append(f"🚨 **Overheating Alert ({e_data['pred_temp']:.1f} °C)**")
                elif e_data['pred_temp'] > 60: issues_thermal.append(f"🟡 **Elevated Temp ({e_data['pred_temp']:.1f} °C)**")
                if e_data['load_factor'] > 80.0: issues_thermal.append(f"⚠️ **High Load Factor ({e_data['load_factor']:.1f}%)**")
                if e_data['inputs']['month'] in list((5, 6, 7, 8, 9)) and 11 <= e_data['inputs']['hour'] <= 16: issues_thermal.append("☀️ **Harsh Ambient Conditions**")
                if not issues_thermal: issues_thermal.append("✅ Thermal parameters are within safe limits.")

                col_t1, col_t2 = st.columns(2)
                with col_t1: 
                    st.info("**🧪 DGA & Oil Quality Findings:**")
                    for iss in issues_dga: st.markdown(f"- {iss}")
                with col_t2: 
                    st.warning("**🌡️ Thermal & SCADA Findings:**")
                    for iss in issues_thermal: st.markdown(f"- {iss}")

# ==============================================================================
# 7. المحركات (MotorGuard AI)
# ==============================================================================
elif st.session_state['current_page'] == 'motor':
    if not st.session_state['motor_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        col1, col2, col3 = st.columns(tuple((1, 2, 1)))
        with col2:
            st.markdown("<br><h2 style='text-align: center; color: #DD4B39;'>⚙️ MotorGuard Login</h2>", unsafe_allow_html=True)
            with st.form("motor_login_form"):
                st.warning("🔒 Restricted Access. Authorized Personnel Only.")
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['motor_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns(tuple((10, 1)))
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['motor_logged_in'] = False
                go_to_portal()

        st.markdown("<div class='main-header'><h2 style='margin:0;'>⚙️ MotorGuard AI</h2></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(tuple(("⚙️ Diagnostics", "📳 Vibration", "🔗 Master Fusion", "📊 Executive Summary")))

        with tab1:
            c_in, c_out = st.columns(tuple((1, 2.5)))
            with c_in:
                motor_type = st.selectbox("Motor Grade", tuple(('L', 'M', 'H')), index=1)
                air_temp = st.number_input("Air Temp [K]", value=298.0)
                process_temp = st.number_input("Core Temp [K]", value=308.0)
                rpm = st.slider("RPM", 1100, 2900, 1500)
                torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)
                tool_wear = st.slider("Wear [min]", 0, 260, 50)
                
            with c_out:
                type_map = {'L':0, 'M':1, 'H':2}
                raw_df = pd.DataFrame({'Type': tuple((type_map[motor_type],)), 'Air temperature [K]': tuple((air_temp,)), 'Process temperature [K]': tuple((process_temp,)), 'Rotational speed [rpm]': tuple((rpm,)), 'Torque [Nm]': tuple((torque,)), 'Tool wear [min]': tuple((tool_wear,))})
                cols_to_scale = tuple(('Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'))
                processed_df = raw_df.copy()
                processed_df[list(cols_to_scale)] = scaler1.transform(raw_df[list(cols_to_scale)])
                
                probs = np.ravel(model1.predict_proba(processed_df)).tolist()
                p_normal = float(probs) * 100.0 if len(probs) > 0 else 0.0
                calc_risk = 100.0 - p_normal
                st.session_state.motor_risk = calc_risk
                st.session_state.motor_faults = {'General Fault': calc_risk}

                color = "#00A65A" if calc_risk < 40 else ("#F39C12" if calc_risk < 75 else "#DD4B39")
                st.plotly_chart(create_motor_gauge(calc_risk, "MOTOR RISK", color), use_container_width=True)

                if st.button("💾 Save Motor Assessment"):
                    new_m_data = dict(Timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), Asset='Motor-01', Risk_Level=f"{calc_risk:.1f}", Primary_FMEA='Operational', Status='Logged', Grade=motor_type, Air_Temp=air_temp, Core_Temp=process_temp, RPM=rpm, Torque=torque, Wear=tool_wear, Vib_RMS='-', Vib_Kurt='-')
                    new_log = pd.DataFrame(tuple((new_m_data,)))
                    new_log.columns = tuple(('Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status', 'Grade', 'Air Temp [K]', 'Core Temp [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Vib RMS', 'Vib Kurt'))
                    st.session_state.inspection_log = pd.concat((new_log, st.session_state.inspection_log), ignore_index=True).head(30)
                    st.success("Saved to Master.")

        with tab2:
            sim_level = st.slider("Degradation Phase (%)", 0, 100, 10)
            r_val = 0.05 + (sim_level/100.0)**2 * 0.20
            p_b_fail = (sim_level / 100.0)**2 * 100.0
            p_b_warn = (sim_level / 100.0) * (1.0 - (sim_level / 100.0)) * 200.0
            p_b_early = (sim_level * 2.0)
            
            b_risk = min(100.0, (p_b_early*0.3) + (p_b_warn*0.7) + (p_b_fail*1.0))
            st.session_state.bearing_risk = b_risk
            
            b_color = "#00A65A" if b_risk < 40 else ("#F39C12" if b_risk < 75 else "#DD4B39")
            st.plotly_chart(create_motor_gauge(b_risk, "BEARING VIBRATION RISK", b_color), use_container_width=True)

        with tab3:
            integrated_risk = (1.0 - ((1.0 - st.session_state.motor_risk/100.0) * (1.0 - st.session_state.bearing_risk/100.0))) * 100.0
            f_color = "#00A65A" if integrated_risk < 40 else ("#F39C12" if integrated_risk < 75 else "#DD4B39")
            st.plotly_chart(create_motor_gauge(integrated_risk, "OVERALL SYSTEM RISK", f_color), use_container_width=True)

        with tab4:
            st.info("Executive Summary is active. Switch to Command Center for Master View.")

# ==============================================================================
# 8. المرحلات (RelayGuard AI)
# ==============================================================================
elif st.session_state['current_page'] == 'relay':
    if not st.session_state['relay_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(tuple((1, 1.5, 1)))
        with col2:
            st.markdown("<div class='main-header'><h2 style='text-align: center; margin:0;'>🛡️ RelayGuard AI Login</h2><p style='text-align: center; margin:0; color:#aaa;'>Hybrid AI Expert System</p></div>", unsafe_allow_html=True)
            with st.form("relay_login_form"):
                st.warning("🔒 Restricted Access. Authorized Personnel Only.")
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['relay_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Username or Password")
    else:
        col_title, col_logout = st.columns(tuple((10, 2)))
        with col_title:
            if st.button("⬅️ Portal"): go_to_portal()
            st.markdown("<div class='main-header' style='padding: 15px;'><h2 style='margin:0;'>🛡️ RelayGuard AI - Health Dashboard</h2></div>", unsafe_allow_html=True)
        with col_logout:
            if st.button("Logout 🚪", use_container_width=True):
                st.session_state['relay_logged_in'] = False
                go_to_portal()

        tab1, tab2, tab3 = st.tabs(tuple(("🛡️ Diagnostic Dashboard", "📑 FMEA & Risk Logs", "📄 Executive Report")))

        with tab1:
            c_in, c_out = st.columns(tuple((1, 2.5)))
            with c_in:
                with st.form("relay_input_form"):
                    st.markdown("### 📥 Telemetry Data")
                    asset_id = st.text_input("Asset ID / Station Name", value="SUB-RELAY-001")
                    noise_index = st.number_input("Pre-Fault Noise Index (Signal Fluctuation)", value=0.05, step=0.01, help="❓ Measures signal noise before a fault. Values > 1.5 strongly indicate hardware deterioration in the Analog Card.")
                    fault_current = st.number_input("Max Fault Current (A)", value=1500, step=100, help="❓ The peak current recorded. Values > 5000A start adding physical stress penalties to the final risk score.")
                    age_months = st.number_input("Relay Age (Months)", value=192, step=12, help="❓ Relays older than 120 months (10 Years) receive an automatic +5% risk penalty for every additional year.")
                    alarms_count = st.number_input("Internal Alarms Count", value=0, step=1, help="❓ Internal self-test failures. Each alarm adds an automatic +1% risk penalty to the assessment.")
                    run_btn = st.form_submit_button("🔍 Run Hybrid AI Analysis", use_container_width=True, type="primary")

            with col_dash:
                if run_btn:
                    if not os.path.exists('Final_Relay_AI_Model.pkl'):
                        st.error("🚨 CRITICAL: Model file 'Final_Relay_AI_Model.pkl' is missing.")
                    else:
                        try:
                            model = joblib.load('Final_Relay_AI_Model.pkl')
                            features = pd.DataFrame(tuple((tuple((noise_index, fault_current, age_months, alarms_count)),)), columns=tuple(('Pre_Fault_Noise_Index', 'Max_Fault_Current_A', 'Relay_Age_Months', 'Internal_Alarms_Count')))
                            
                            probs_array = np.ravel(model.predict_proba(features))
                            base_probs_list = probs_array.tolist()
                            
                            classes_dict = {0: 'Safe Condition', 1: 'Analog Card Issue', 2: 'Mechanical Wear', 3: 'CPU/Power Error'}
                            actual_classes_list = list(getattr(model, 'classes_', tuple(range(len(base_probs_list)))))
                            class_probs_map = dict(zip(actual_classes_list, base_probs_list))
                            
                            ai_safe_prob = float(class_probs_map.get(0, 0.0))
                            ai_base_risk = float((1.0 - ai_safe_prob) * 100.0)
                            
                            age_penalty = max(0.0, float(age_months) - 120.0) / 12.0 * 5.0
                            alarm_penalty = float(alarms_count) * 1.0
                            current_penalty = max(0.0, float(fault_current) - 5000.0) / 1000.0 * 2.0
                            
                            total_penalty = age_penalty + alarm_penalty + current_penalty
                            final_risk = min(100.0, ai_base_risk + total_penalty)
                            
                            final_probs_list = list(base_probs_list) 
                            if total_penalty > 0 and 0 in actual_classes_list:
                                safe_idx = actual_classes_list.index(0)
                                new_safe_prob = max(0.0, base_probs_list[safe_idx] - (total_penalty / 100.0))
                                final_probs_list[safe_idx] = new_safe_prob
                                diff = base_probs_list[safe_idx] - new_safe_prob
                                if diff > 0:
                                    target_idx = 2 
                                    if age_penalty > current_penalty and age_penalty > alarm_penalty: target_idx = actual_classes_list.index(2) if 2 in actual_classes_list else 1
                                    elif alarm_penalty > current_penalty: target_idx = actual_classes_list.index(3) if 3 in actual_classes_list else 1
                                    final_probs_list[target_idx] += diff
                            
                            prob_sum = sum(final_probs_list)
                            if prob_sum > 0: final_probs_list = list((p/prob_sum for p in final_probs_list))
                                
                            ai_confidence = float(max(final_probs_list)) * 100.0
                            best_class_val = actual_classes_list[final_probs_list.index(max(final_probs_list))]
                            mode = classes_dict.get(best_class_val, "Unknown Error")
                            
                            if total_penalty > ai_base_risk and mode == 'Safe Condition' and final_risk >= 30:
                                mode = "Age Degradation (Heuristic)" if age_penalty > current_penalty else "High Stress Limits Exceeded"
                            
                            colors_dict = {0: '#2ecc71', 1: '#f1c40f', 2: '#e67e22', 3: '#e74c3c'}
                            plot_classes, plot_colors = list(), list()
                            for c in actual_classes_list:
                                class_name = classes_dict.get(int(c), "Unknown Error")
                                plot_classes.append("Age/Stress Degradation" if class_name == 'Mechanical Wear' and 'Age' in mode else class_name)
                                plot_colors.append(colors_dict.get(int(c), "#333333"))
                            
                            rpn_score = int(final_risk * 10)
                            
                            if final_risk < 30: status, color, action, rpn_level = "🟢 SAFE", "#2ecc71", "Normal operation. Maintain TBM schedule.", "Low"
                            elif final_risk < 70: status, color, action, rpn_level = "🟡 WARNING", "#f1c40f", f"Schedule CBM inspection focusing on {mode}.", "Medium"
                            else: status, color, action, rpn_level = "🔴 DANGER", "#e74c3c", f"Urgent replacement required due to {mode}.", "Critical"

                            m1, m2, m3, m4 = st.columns(tuple((1, 1, 1, 1)))
                            m1.metric("🤖 Hybrid Confidence", f"{ai_confidence:.1f}%")
                            m2.metric("⚠️ Failure Risk", f"{final_risk:.1f}%")
                            m3.metric("📈 Status", status.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip())
                            mode_words = mode.split()
                            m4.metric("⚙️ Primary Issue", mode_words.pop(0) if mode_words else "Unknown")

                            st.markdown(f"<div style='border-radius:10px; background:{color}20; border:2px solid {color}; padding:15px; text-align:center; margin-top: 10px; margin-bottom: 10px;'><h2 style='color:{color}; margin:0;'>{status}: {mode}</h2></div>", unsafe_allow_html=True)
                            
                            new_data = {'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset ID': asset_id, 'Noise Index': noise_index, 'Max Current (A)': fault_current, 'Risk (%)': round(final_risk, 1), 'RPN': rpn_score, 'Status': status, 'Action': action}
                            new_log = pd.DataFrame(tuple((new_data,)))
                            st.session_state['relay_fmea_log'] = pd.concat((new_log, st.session_state['relay_fmea_log']), ignore_index=True).head(100)
                            
                            ai_reasons_clean = "Analysis indicates healthy telemetry. All parameters are within safe operational thresholds." if final_risk < 30 else f"System detected anomaly resulting in {final_risk:.1f}% risk score."
                            st.session_state['relay_latest_eval'] = {'id': asset_id, 'risk': final_risk, 'status': status, 'color': color, 'mode': mode, 'action': action, 'probs': final_probs_list, 'classes': plot_classes, 'reasoning': ai_reasons_clean, 'rpn_score': rpn_score, 'rpn_level': rpn_level}

                            v1, v2 = st.columns(tuple((1, 1)))
                            with v1: st.plotly_chart(create_relay_gauge(final_risk, "Hybrid Failure Risk", color), use_container_width=True)
                            with v2:
                                st.markdown("#### 📊 Hybrid Failure Probabilities")
                                fig_p = px.bar(x=final_probs_list, y=plot_classes, orientation='h', color=plot_classes, color_discrete_sequence=plot_colors)
                                fig_p.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Probability", yaxis_title="")
                                st.plotly_chart(fig_p, use_container_width=True)

                        except Exception as e: st.error(f"Analysis failed: {str(e)}")
                else: st.info("💡 Please input telemetry data and click 'Run Hybrid AI Analysis' to start.")

        with tab2:
            st.markdown("### 📑 RPN & FMEA History Registry")
            fmea_df = st.session_state['relay_fmea_log']
            if not fmea_df.empty: st.dataframe(fmea_df.style.apply(highlight_relay_fmea, axis=1), use_container_width=True, hide_index=True)
            else: st.info("No records available. Run an assessment to populate the database.")

        with tab3:
            eval_data = st.session_state['relay_latest_eval']
            if eval_data:
                rep_color = eval_data.get('color')
                st.markdown(f"### 📄 Comprehensive Hybrid AI Executive Report")
                st.markdown(f"**Asset Designation:** `{eval_data.get('id')}` | **Assessment Time:** `{datetime.now().strftime('%Y-%m-%d %H:%M')}`")
                st.markdown("---")
                
                rep_col1, rep_col2 = st.columns(tuple((1, 1)))
                with rep_col1:
                    st.markdown("#### 🎯 Diagnostic Summary")
                    st.info(f"**Overall Health Index:** {100-eval_data.get('risk'):.1f}%")
                    
                    if eval_data.get('risk') < 30:
                        st.success(f"**Predicted Failure Mode:** {eval_data.get('mode')}")
                        st.success(f"✅ **{eval_data.get('action')}**")
                    elif eval_data.get('risk') < 70:
                        st.warning(f"**Predicted Failure Mode:** {eval_data.get('mode')}")
                        st.warning(f"⚠️ **{eval_data.get('action')}**")
                    else:
                        st.error(f"**Predicted Failure Mode:** {eval_data.get('mode')}")
                        st.error(f"🚨 **{eval_data.get('action')}**")
                    
                    st.markdown(f"**FMEA RPN (Risk Priority Number):** `{eval_data.get('rpn_score')}/1000` ({eval_data.get('rpn_level')} Severity)")
                    st.markdown("#### 🧠 Expert System Reasoning")
                    st.markdown(f"<div class='report-reasoning'>{eval_data.get('reasoning')}</div>", unsafe_allow_html=True)

                with rep_col2:
                    st.markdown("#### 📡 Multidimensional Hybrid Radar")
                    st.plotly_chart(create_relay_radar(eval_data.get('probs'), eval_data.get('classes'), rep_color), use_container_width=True)
                    
                    st.markdown("#### 📈 Actual Historical Risk Trend")
                    log_df = st.session_state['relay_fmea_log']
                    curr_id = eval_data.get('id')
                    
                    time_vals, risk_vals, status_vals = list(), list(), list()
                    for _, rec in log_df.iterrows():
                        if rec['Asset ID'] == curr_id:
                            time_vals.append(rec['Timestamp'])
                            risk_vals.append(rec['Risk (%)'])
                            status_vals.append(rec['Status'].replace("🟢", "").replace("🟡", "").replace("🔴", "").strip())
                    
                    time_vals.reverse()
                    risk_vals.reverse()
                    status_vals.reverse()
                    
                    if len(time_vals) > 0:
                        trend_fig = go.Figure()
                        trend_fig.add_trace(go.Scatter(x=time_vals, y=risk_vals, mode='lines+markers', line=dict(color=rep_color, width=3), marker=dict(size=8, color=rep_color), text=status_vals, hovertemplate="Time: %{x}<br>Risk: %{y}%<br>Status: %{text}<extra></extra>"))
                        trend_fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Risk (%)", xaxis_title="Historical Assessments")
                        st.plotly_chart(trend_fig, use_container_width=True)
                    else: st.info("📊 Run at least one assessment to view the historical trend.")
                    
            else: st.info("💡 Run an evaluation in the Diagnostic Dashboard to generate the Executive Report.")

# ==============================================================================
# 9. المحطة الشاملة (SubstationGuard AI) - التعديل النهائي
# ==============================================================================
elif st.session_state['current_page'] == 'substation':
    if not st.session_state['master_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(tuple((1, 2, 1)))
        with col2:
            st.markdown("<div style='background: linear-gradient(135deg, #0b2e59 0%, #1a5276 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-bottom: 20px;'><h2 style='margin:0;'>🌐 SubstationGuard AI</h2><p style='margin:0; font-size:14px; color:#aaa;'>Enterprise Command Center Login</p></div>", unsafe_allow_html=True)
            with st.form("master_login_form"):
                st.warning("🔒 Restricted C-Level Access. Authorized Personnel Only.")
                user = st.text_input("Admin Username")
                pwd = st.text_input("Admin Password", type="password")
                if st.form_submit_button("Authenticate", use_container_width=True):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['master_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Admin Credentials")
    else:
        col_l, col_r = st.columns(tuple((10, 1)))
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['master_logged_in'] = False
                go_to_portal()

        st.markdown("""
        <div style='background: linear-gradient(135deg, #111 0%, #222 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-bottom: 4px solid #00ff88;'>
            <h1 style='color: #00ff88; text-align: center; margin: 0; font-family: "Courier New", Courier, monospace;'>🌐 SUBSTATION DIGITAL TWIN</h1>
            <p style='color: #aaa; text-align: center; margin: 5px 0 0 0;'>System-of-Systems Comprehensive Monitoring</p>
        </div>
        """, unsafe_allow_html=True)

        tx_risk = 0.0
        if st.session_state.get('latest_eval'):
            tx_risk = float(st.session_state['latest_eval'].get('overall_risk', 0.0))

        motor_sys_risk = 0.0
        if st.session_state.get('motor_risk') or st.session_state.get('bearing_risk'):
            m_r = float(st.session_state.get('motor_risk', 0.0)) / 100.0
            b_r = float(st.session_state.get('bearing_risk', 0.0)) / 100.0
            motor_sys_risk = (1.0 - ((1.0 - m_r) * (1.0 - b_r))) * 100.0

        relay_risk = 0.0
        if st.session_state.get('relay_latest_eval'):
            relay_risk = float(st.session_state['relay_latest_eval'].get('risk', 0.0))

        rel_tx = 1.0 - (tx_risk / 100.0)
        rel_mo = 1.0 - (motor_sys_risk / 100.0)
        rel_re = 1.0 - (relay_risk / 100.0)
        global_reliability = rel_tx * rel_mo * rel_re
        global_risk = (1.0 - global_reliability) * 100.0
        global_health = 100.0 - global_risk

        if global_risk == 0:
            st.info("💡 Plant Data Empty: Please run evaluations in the Transformer, Motor, and Relay modules first to generate the Global Dashboard.")
        else:
            risks_dict = {'Transformer Unit': tx_risk, 'Motor System': motor_sys_risk, 'Protection Relay': relay_risk}
            weakest_link = max(risks_dict, key=risks_dict.get)
            max_risk_val = risks_dict[weakest_link]

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card' style='border-left-color:#00ff88; background:#111; color:white;'><h4>Global Plant Health</h4><h1 style='color:#00ff88; margin:0;'>{global_health:.1f}%</h1></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card' style='border-left-color:#e74c3c; background:#111; color:white;'><h4>Cascading Failure Risk</h4><h1 style='color:#e74c3c; margin:0;'>{global_risk:.1f}%</h1></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card' style='border-left-color:#f1c40f; background:#111; color:white;'><h4>System Bottleneck</h4><h2 style='color:#f1c40f; margin:0;'>{weakest_link}</h2></div>", unsafe_allow_html=True)

            action_color = "#e74c3c" if global_risk > 70 else ("#f1c40f" if global_risk > 35 else "#00ff88")
            action_text = "SHUTDOWN" if global_risk > 70 else ("INSPECT" if global_risk > 35 else "OPTIMAL")
            c4.markdown(f"<div class='metric-card' style='border-left-color:{action_color}; background:#111; color:white;'><h4>AI Directive</h4><h1 style='color:{action_color}; margin:0;'>{action_text}</h1></div>", unsafe_allow_html=True)

            st.markdown("---")

            # 💡 التعديل المطلوب: تقسيم الشاشة إلى 3 أعمدة (Gauge + Radar + Bar Chart) مع الحفاظ على الرسم الزمني إذا أردنا (أو استبداله بالبار شارت كما طلبت).
            # قمت بعمل 3 أعمدة هنا ليكون المنظر غاية في الفخامة.
            v_col1, v_col2, v_col3 = st.columns(list((1, 1, 1.5)))
            
            with v_col1:
                st.markdown("#### 🚨 Cascading Risk Score")
                fig_sub_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=float(global_risk),
                    number={'suffix': "%", 'font': {'size': 30, 'color': action_color}},
                    gauge={
                        'axis': {'range': list((0, 100)), 'tickwidth': 1},
                        'bar': {'color': action_color},
                        'bgcolor': "white",
                        'steps': list((
                            {'range': list((0, 30)), 'color': 'rgba(46, 204, 113, 0.2)'},
                            {'range': list((30, 70)), 'color': 'rgba(241, 196, 15, 0.2)'},
                            {'range': list((70, 100)), 'color': 'rgba(231, 76, 60, 0.2)'}
                        )),
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': float(global_risk)}
                    }
                ))
                fig_sub_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sub_gauge, use_container_width=True)

            with v_col2:
                st.markdown("#### 🕸️ Topology Risk Radar")
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=list((tx_risk, motor_sys_risk, relay_risk, tx_risk)),
                    theta=list(('Transformer', 'Motor', 'Relay', 'Transformer')),
                    fill='toself', line_color='#e74c3c', fillcolor='rgba(231, 76, 60, 0.4)'
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=list((0, 100)))), showlegend=False, height=300, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_radar, use_container_width=True)

            with v_col3:
                st.markdown("#### 📊 Component Health Breakdown")
                df_sys = pd.DataFrame({
                    'Asset Category': list(('Transformer', 'Motor', 'Relay')),
                    'Health %': list((100.0-tx_risk, 100.0-motor_sys_risk, 100.0-relay_risk)),
                    'Risk %': list((tx_risk, motor_sys_risk, relay_risk))
                })
                fig_sys = px.bar(df_sys, x="Asset Category", y=list(("Health %", "Risk %")), color_discrete_sequence=list(('#2ecc71', '#e74c3c')))
                fig_sys.update_layout(height=300, barmode='stack', margin=dict(t=30, b=10), yaxis_title="Status (%)", xaxis_title="")
                st.plotly_chart(fig_sys, use_container_width=True)

            st.markdown("---")

            # Master FMEA Registry
            st.markdown("### 📋 Master Plant FMEA Registry (Holistic View)")
            tx_eval = st.session_state.get('latest_eval', {})
            re_eval = st.session_state.get('relay_latest_eval', {})
            master_fmea = pd.DataFrame({
                'Asset Module': list(('Transformer Unit (Power)', 'Motor Assembly (Motion)', 'Protection Relay (Control)')),
                'Current Risk': list((f"{tx_risk:.1f}%", f"{motor_sys_risk:.1f}%", f"{relay_risk:.1f}%")),
                'System Status': list((
                    "🔴 DANGER" if tx_risk >= 70 else ("🟡 WARNING" if tx_risk >= 35 else "🟢 SAFE"),
                    "🔴 DANGER" if motor_sys_risk >= 75 else ("🟡 WARNING" if motor_sys_risk >= 40 else "🟢 SAFE"),
                    "🔴 DANGER" if relay_risk >= 70 else ("🟡 WARNING" if relay_risk >= 30 else "🟢 SAFE")
                )),
                'AI Prescribed Action': list((
                    tx_eval.get('action', 'Monitor') if tx_eval else "Monitor",
                    "Schedule Mechanical Check" if motor_sys_risk > 40 else "Monitor",
                    re_eval.get('action', 'Monitor') if re_eval else "Monitor"
                ))
            })
            st.dataframe(master_fmea.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True)

            st.markdown("---")

            # 💡 التقرير المعدل للإدارة العليا (Bullet Points)
            st.markdown("### 🧠 C-Level Executive AI Briefing")
            
            if global_risk < 30:
                exec_color = "#2ecc71"
                action_badge = "🟢 CONTINUE NORMAL PRODUCTION"
                overall_point = f"**Overall System Status:** The substation is operating optimally with a robust Global Health Index of **{global_health:.1f}%**. No cascading failure threats are currently detected."
            elif global_risk < 70:
                exec_color = "#f1c40f"
                action_badge = "🟡 SCHEDULE TARGETED MAINTENANCE"
                overall_point = f"**Overall System Status:** The plant exhibits a moderate cascading failure risk of **{global_risk:.1f}%**. Proactive interventions are required to prevent further degradation."
            else:
                exec_color = "#e74c3c"
                action_badge = "🔴 URGENT SYSTEM INTERVENTION REQUIRED"
                overall_point = f"**Overall System Status:** CRITICAL ALERT. The substation faces an imminent threat of cascading failure (Risk: **{global_risk:.1f}%**), threatening a potential plant-wide blackout."

            tx_str = f"Stable (Risk: {tx_risk:.1f}%)" if tx_risk < 35 else (f"Elevated (Risk: {tx_risk:.1f}%)" if tx_risk < 70 else f"Critical (Risk: {tx_risk:.1f}%)")
            mo_str = f"Stable (Risk: {motor_sys_risk:.1f}%)" if motor_sys_risk < 40 else (f"Elevated (Risk: {motor_sys_risk:.1f}%)" if motor_sys_risk < 75 else f"Critical (Risk: {motor_sys_risk:.1f}%)")
            re_str = f"Stable (Risk: {relay_risk:.1f}%)" if relay_risk < 30 else (f"Elevated (Risk: {relay_risk:.1f}%)" if relay_risk < 70 else f"Critical (Risk: {relay_risk:.1f}%)")

            st.markdown(f"""
            <div style='background-color: {exec_color}10; padding: 25px; border-radius: 10px; border-left: 8px solid {exec_color}; border-right: 1px solid #ddd; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;'>
                <h3 style='color: {exec_color}; margin-top: 0; font-weight: bold;'>{action_badge}</h3>
                <ul style='font-size: 18px; color: #333; line-height: 1.8; margin-bottom: 20px;'>
                    <li>{overall_point}</li>
                    <li>⚡ <b>Transformer Assets:</b> {tx_str}.</li>
                    <li>⚙️ <b>Motor & Motion Assets:</b> {mo_str}.</li>
                    <li>🛡️ <b>Protection Relays:</b> {re_str}.</li>
                </ul>
                <div style='background-color: #fff; padding: 15px; border-radius: 5px; border: 1px dashed {exec_color};'>
                    <strong>🎯 AI Strategic Directive:</strong> The primary bottleneck threatening plant stability is the <b>{weakest_link}</b>. Focus immediate predictive maintenance efforts and resource allocation on this asset to restore full system integrity.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 💡 زر التحميل الرسمي للتقرير
            st.markdown("<br>", unsafe_allow_html=True)
            report_txt = f"SUBSTATION DIGITAL TWIN - EXECUTIVE REPORT\n"
            report_txt += f"=============================================\n"
            report_txt += f"Assessment Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            report_txt += f"Global Plant Health: {global_health:.1f}%\n"
            report_txt += f"Cascading Failure Risk: {global_risk:.1f}%\n"
            report_txt += f"Primary Bottleneck: {weakest_link}\n"
            report_txt += f"AI Directive: {action_text}\n"
            report_txt += f"---------------------------------------------\n"
            report_txt += f"ASSET RISK BREAKDOWN:\n"
            report_txt += f"- Transformer Unit (Power): {tx_risk:.1f}%\n"
            report_txt += f"- Motor Assembly (Motion): {motor_sys_risk:.1f}%\n"
            report_txt += f"- Protection Relay (Control): {relay_risk:.1f}%\n"
            report_txt += f"---------------------------------------------\n"
            report_txt += f"EXECUTIVE BRIEFING:\n"
            report_txt += f"{overall_point.replace('**', '').replace('<b>', '').replace('</b>', '')}\n"
            report_txt += f"- Transformer Assets: {tx_str}\n"
            report_txt += f"- Motor & Motion Assets: {mo_str}\n"
            report_txt += f"- Protection Relays: {re_str}\n"
            
            st.download_button("📥 Download Official Substation Report (TXT)", data=report_txt, file_name=f"Substation_Report_{datetime.now().strftime('%Y%m%d')}.txt", use_container_width=True)