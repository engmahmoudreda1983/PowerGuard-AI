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

model_health, model_life, model_thermal, scaler1, model1, scaler2, model2, models_ready = load_all_models()

# ==========================================
# 3. دوال مساعدة (Helper Functions)
# ==========================================
def get_tx_confidence(model, input_df):
    try:
        preds = np.array([tree.predict(input_df.values).item() for tree in model.estimators_])
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
    status = row.get('Status', '')
    if 'SAFE' in status.upper() or 'Safe' in status: return ['background-color: #E8F5E9; color: #00A65A; font-weight: bold'] * len(row)
    if 'WARN' in status.upper(): return ['background-color: #FEF9E7; color: #D35400; font-weight: bold'] * len(row)
    if 'DANG' in status.upper() or 'ACTION' in status.upper(): return ['background-color: #FDEDEC; color: #C0392B; font-weight: bold'] * len(row)
    return [''] * len(row)

def create_motor_gauge(value, title, color, max_v=100.0, h=250):
    val = float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number=dict(suffix="%", font=dict(size=26, color=color, family="Arial Black")),
        title=dict(text=title, font=dict(size=16, color='#333', weight='bold')),
        gauge=dict(
            axis=dict(range=[0, max_v], tickwidth=1, tickcolor="darkblue"),
            bar=dict(color=color),
            bgcolor="white", borderwidth=2, bordercolor="gray",
            steps=[
                dict(range=, color='#E8F5E9'), 
                dict(range=, color='#FEF9E7'), 
                dict(range=, color='#FDEDEC')
            ],
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=val)
        )))
    fig.update_layout(height=h, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def highlight_relay_fmea(row):
    status = row.get('Status', '')
    if 'SAFE' in status: return tuple(('background-color: #E8F5E9; color: #00A65A; font-weight: bold',)) * len(row)
    if 'WARNING' in status: return tuple(('background-color: #FEF9E7; color: #D35400; font-weight: bold',)) * len(row)
    if 'DANGER' in status: return tuple(('background-color: #FDEDEC; color: #C0392B; font-weight: bold',)) * len(row)
    return tuple(('',)) * len(row)

def create_relay_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number=dict(suffix="%", font=dict(size=30, color=color)),
        title=dict(text=title, font=dict(size=18, color='#333')),
        gauge=dict(
            axis=dict(range=, tickwidth=1),
            bar=dict(color=color),
            bgcolor="white",
            steps=[
                dict(range=, color='rgba(46, 204, 113, 0.2)'),
                dict(range=, color='rgba(241, 196, 15, 0.2)'),
                dict(range=, color='rgba(231, 76, 60, 0.2)')
            ],
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=float(value))
        )
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def create_relay_radar(probs_list, classes_tup, color):
    fig = go.Figure(data=go.Scatterpolar(
        r=[p * 100 for p in probs_list], theta=classes_tup, fill='toself', line_color=color, opacity=0.6
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=)), showlegend=False, height=350, margin=dict(l=30, r=30, t=30, b=30))
    return fig

# ==========================================
# 4. إعداد متغيرات الجلسة (Session State)
# ==========================================
if 'current_page' not in st.session_state: st.session_state['current_page'] = 'portal'

# Transformer State
if 'tx_logged_in' not in st.session_state: st.session_state['tx_logged_in'] = False
if 'fmea_log' not in st.session_state: st.session_state['fmea_log'] = pd.DataFrame(columns=['Timestamp', 'Asset ID', 'DGA Index', 'Pred Temp', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required'])
if 'latest_eval' not in st.session_state: st.session_state['latest_eval'] = None

# Motor State
if 'motor_logged_in' not in st.session_state: st.session_state['motor_logged_in'] = False
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'inspection_log' not in st.session_state: st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status', 'Grade', 'Air Temp [K]', 'Core Temp [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Vib RMS', 'Vib Kurt'])

# Relay State
if 'relay_logged_in' not in st.session_state: st.session_state['relay_logged_in'] = False
if 'relay_fmea_log' not in st.session_state: st.session_state['relay_fmea_log'] = pd.DataFrame(columns=['Timestamp', 'Asset ID', 'Noise Index', 'Max Current (A)', 'Risk (%)', 'RPN', 'Status', 'Action'])
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
    
    col_m1, col_m2, col_m3 = st.columns()
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
    
    col1, col2, col3 = st.columns(3)
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
        col1, col2, col3 = st.columns()
        with col2:
            st.markdown("<h2 style='text-align: center; color: #4da6ff;'>⚡ TransformerGuard Login</h2>", unsafe_allow_html=True)
            with st.form("tx_login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['tx_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns()
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

        tab1, tab2, tab3, tab4 = st.tabs(["🧪 DGA Analytics", "🌡️ Thermal Load", "📑 RISK Logs", "📄 Executive Report"])

        with tab1:
            col_in, col_out = st.columns([1, 2.2])
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
                input_df = pd.DataFrame([[h2, o2, n2, ch4, co, co2, c2h4, c2h6, c2h2, dbds, pf, ift, dr, wc]], columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'])
                
                h_score = float(model_health.predict(input_df).item())
                cf = sum([1.0 if dr<35 else 0, 1.0 if ift<20 else 0, 1.5 if wc>25 else 0, 1.5 if co>500 else 0, 1.0 if pf>1.5 else 0, 2.0 if c2h2>10 else 0])
                if cf > 0: h_score = min(100.0, h_score + (cf * 8.0))
                
                status = "SAFE" if h_score <= 30 else ("WARNING" if h_score <= 45 else "RISKY")
                color = "#00cc66" if status == "SAFE" else ("#ffcc00" if status == "WARNING" else "#ff3333")

                m1, m2, m3 = st.columns(3)
                m1.metric("Degradation Index", f"{h_score:.2f}")
                m2.metric("Status", status)
                m3.metric("AI Confidence", f"{get_tx_confidence(model_health, input_df):.1f} %")

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=float(h_score),
                    gauge=dict(
                        axis=dict(range=, tickwidth=2, tickcolor="black"),
                        bar=dict(color="black"),
                        steps=[
                            dict(range=, color="#00cc66"),
                            dict(range=, color="#ffcc00"),
                            dict(range=, color="#ff3333")
                        ]
                    )
                ))
                fig_g.update_layout(height=280, margin=dict(t=30, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
                
                st.markdown(f"<h2 style='text-align: center; color: {color}; margin-top:-30px;'>{status}</h2>", unsafe_allow_html=True)
                st.info(f"**Diagnosis:** {get_duval_diagnosis(ch4, c2h4, c2h2)}")

        with tab2:
            st.markdown("### 🌡️ Thermal Load Analysis")
            col_t_in, col_t_out = st.columns([1, 1.5])
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
                t_input_norm = pd.DataFrame([[hufl*n_ratio, hull*n_ratio, mufl*n_ratio, mull*n_ratio, lufl*n_ratio, lull*n_ratio, hour, month]], columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'])
                pred_ot = float(model_thermal.predict(t_input_norm).item())
                
                fig_t = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=float(pred_ot),
                    gauge=dict(
                        axis=dict(range=),
                        bar=dict(color="black"),
                        steps=[
                            dict(range=, color="rgba(0, 204, 102, 0.4)"),
                            dict(range=, color="rgba(255, 204, 0, 0.4)"),
                            dict(range=, color="rgba(255, 51, 51, 0.4)")
                        ]
                    )
                ))
                fig_t.update_layout(height=350, margin=dict(t=30, b=10))
                st.plotly_chart(fig_t, use_container_width=True)

        with tab3:
            st.markdown("### 📑 Overall Transformer Risk Assessment")
            col_ex_in, col_ex_out = st.columns([1, 1.5])
            with col_ex_in:
                with st.form("tx_exec_form"):
                    asset_id = st.text_input("Asset ID", value="TX-01")
                    gen_report = st.form_submit_button("📊 Evaluate & Push to Master System", use_container_width=True, type="primary")
                    
            with col_ex_out:
                if gen_report:
                    dga_risk = min(100.0, max(0.0, ((h_score - 20.0) / 30.0) * 100.0)) 
                    thermal_risk = min(100.0, max(0.0, ((pred_ot - 40.0) / 45.0) * 100.0))
                    overall_risk = (dga_risk * 0.6) + (thermal_risk * 0.4)
                    
                    if overall_risk <= 35: f_status, f_color, action = "🟢 SAFE", "#00cc66", "Continue operation."
                    elif overall_risk <= 70: f_status, f_color, action = "🟡 WARNING", "#ffcc00", "Schedule PM."
                    else: f_status, f_color, action = "🔴 DANGER", "#ff3333", "URGENT: Isolate asset."
                    
                    st.session_state['latest_eval'] = {'asset_id': asset_id, 'overall_risk': overall_risk, 'status': f_status, 'action': action, 'color': f_color, 'duval': get_duval_diagnosis(ch4, c2h4, c2h2)}
                    new_entry = pd.DataFrame([{'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset ID': asset_id, 'DGA Index': round(h_score, 1), 'Pred Temp': round(pred_ot, 1), 'Overall Risk (%)': round(overall_risk, 1), 'Status': f_status, 'RPN Level': "Calc", 'Action Required': action}])
                    st.session_state['fmea_log'] = pd.concat([st.session_state['fmea_log'], new_entry], ignore_index=True).tail(100)
                    
                    st.markdown(f"<h2 style='text-align: center; color: {f_color};'>{f_status} - Risk: {overall_risk:.1f}%</h2>", unsafe_allow_html=True)
            
            st.markdown("---")
            if not st.session_state['fmea_log'].empty:
                st.dataframe(st.session_state['fmea_log'].style.apply(highlight_fmea, axis=1), use_container_width=True)

        with tab4:
            st.info("💡 Report pushed to Substation Command Center.")

# ==============================================================================
# 7. المحركات (MotorGuard AI)
# ==============================================================================
elif st.session_state['current_page'] == 'motor':
    if not st.session_state['motor_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        col1, col2, col3 = st.columns()
        with col2:
            st.markdown("<br><h2 style='text-align: center; color: #DD4B39;'>⚙️ MotorGuard Login</h2>", unsafe_allow_html=True)
            with st.form("motor_login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['motor_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns()
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['motor_logged_in'] = False
                go_to_portal()

        st.markdown("<div class='main-header'><h2 style='margin:0;'>⚙️ MotorGuard AI</h2></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["⚙️ Diagnostics", "📳 Vibration", "🔗 Master Fusion"])

        with tab1:
            c_in, c_out = st.columns([1, 2.5])
            with c_in:
                motor_type = st.selectbox("Motor Grade", ['L', 'M', 'H'], index=1)
                air_temp = st.number_input("Air Temp [K]", value=298.0)
                process_temp = st.number_input("Core Temp [K]", value=308.0)
                rpm = st.slider("RPM", 1100, 2900, 1500)
                torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)
                tool_wear = st.slider("Wear [min]", 0, 260, 50)
                
            with c_out:
                type_map = {'L':0, 'M':1, 'H':2}
                raw_df = pd.DataFrame({'Type': [type_map[motor_type]], 'Air temperature [K]': [air_temp], 'Process temperature [K]': [process_temp], 'Rotational speed [rpm]': [rpm], 'Torque [Nm]': [torque], 'Tool wear [min]': [tool_wear]})
                cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                processed_df = raw_df.copy()
                processed_df[cols_to_scale] = scaler1.transform(raw_df[cols_to_scale])
                
                probs = np.ravel(model1.predict_proba(processed_df)).tolist()
                p_normal = probs*100.0 if len(probs)>0 else 0.0
                calc_risk = 100.0 - p_normal
                st.session_state.motor_risk = calc_risk
                st.session_state.motor_faults = {'General Fault': calc_risk}

                color = "#00A65A" if calc_risk < 40 else ("#F39C12" if calc_risk < 75 else "#DD4B39")
                st.plotly_chart(create_motor_gauge(calc_risk, "MOTOR RISK", color), use_container_width=True)

                if st.button("💾 Save Motor Assessment"):
                    new_log = pd.DataFrame([{'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Motor-01', 'Risk Level': f"{calc_risk:.1f}", 'Primary FMEA': 'Operational', 'Status': 'Logged', 'Grade': motor_type, 'Air Temp [K]': air_temp, 'Core Temp [K]': process_temp, 'RPM': rpm, 'Torque [Nm]': torque, 'Wear [min]': tool_wear, 'Vib RMS': '-', 'Vib Kurt': '-'}])
                    st.session_state.inspection_log = pd.concat([new_log, st.session_state.inspection_log], ignore_index=True).head(30)
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
            st.info("Fusion Data Pushed to Substation Command Center.")

# ==============================================================================
# 8. المرحلات (RelayGuard AI)
# ==============================================================================
elif st.session_state['current_page'] == 'relay':
    if not st.session_state['relay_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("<br><h2 style='text-align: center; color: #9B59B6;'>🛡️ RelayGuard AI Login</h2>", unsafe_allow_html=True)
            with st.form("relay_login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['relay_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns()
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['relay_logged_in'] = False
                go_to_portal()

        st.markdown("<div class='main-header' style='background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%); border-left: 8px solid #9B59B6;'><h2 style='margin:0;'>🛡️ RelayGuard AI</h2></div>", unsafe_allow_html=True)

        c_in, c_out = st.columns([1, 2.5])
        with c_in:
            with st.form("relay_form"):
                asset_id = st.text_input("Relay ID", value="RELAY-01")
                noise = st.number_input("Noise Index", value=0.05)
                current = st.number_input("Fault Current", value=1500)
                age = st.number_input("Age (Months)", value=120)
                alarms = st.number_input("Alarms", value=0)
                run = st.form_submit_button("Evaluate & Push", type="primary", use_container_width=True)
        
        with c_out:
            if run:
                risk = min(100.0, (noise*10) + (current/100) + (age/5) + (alarms*5))
                status = "🟢 SAFE" if risk < 30 else ("🟡 WARNING" if risk < 70 else "🔴 DANGER")
                color = "#2ecc71" if risk < 30 else ("#f1c40f" if risk < 70 else "#e74c3c")
                
                st.session_state['relay_latest_eval'] = {'id': asset_id, 'risk': risk, 'status': status, 'color': color, 'mode': 'Heuristic Eval'}
                new_log = pd.DataFrame([{'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset ID': asset_id, 'Noise Index': noise, 'Max Current (A)': current, 'Risk (%)': risk, 'RPN': int(risk*10), 'Status': status, 'Action': 'Logged'}])
                st.session_state['relay_fmea_log'] = pd.concat([new_log, st.session_state['relay_fmea_log']], ignore_index=True).head(50)
                
                st.plotly_chart(create_relay_gauge(risk, "RELAY RISK", color), use_container_width=True)

# ==============================================================================
# 9. المحطة الشاملة (SubstationGuard AI) - The C-Level Masterpiece
# ==============================================================================
elif st.session_state['current_page'] == 'substation':
    if not st.session_state['master_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        col1, col2, col3 = st.columns()
        with col2:
            st.markdown("<br><div class='master-card'><h2 style='margin:0;'>🌐 SubstationGuard AI</h2><p style='margin:0; color:#00ff88;'>C-Level Executive Access</p></div>", unsafe_allow_html=True)
            with st.form("master_login"):
                user = st.text_input("Admin Username")
                pwd = st.text_input("Admin Password", type="password")
                if st.form_submit_button("Authenticate", use_container_width=True, type="primary"):
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                        st.session_state['master_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns()
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['master_logged_in'] = False
                go_to_portal()
                
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; border-bottom: 5px solid #00ff88; box-shadow: 0 10px 20px rgba(0,0,0,0.5);'>
            <h1 style='color: #00ff88; text-align: center; margin: 0; font-family: "Arial Black", sans-serif; letter-spacing: 2px;'>🌐 SUBSTATION DIGITAL TWIN</h1>
            <p style='color: #ddd; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>Comprehensive System-of-Systems AI Command Center</p>
        </div>
        """, unsafe_allow_html=True)

        # 1. Fetching all risks
        tx_risk = float(st.session_state.get('latest_eval', {}).get('overall_risk', 0.0)) if st.session_state.get('latest_eval') else 0.0
        m_r = float(st.session_state.get('motor_risk', 0.0)) / 100.0
        b_r = float(st.session_state.get('bearing_risk', 0.0)) / 100.0
        motor_sys_risk = (1.0 - ((1.0 - m_r) * (1.0 - b_r))) * 100.0
        relay_risk = float(st.session_state.get('relay_latest_eval', {}).get('risk', 0.0)) if st.session_state.get('relay_latest_eval') else 0.0

        # Series System Calculation
        global_reliability = (1.0 - (tx_risk/100.0)) * (1.0 - (motor_sys_risk/100.0)) * (1.0 - (relay_risk/100.0))
        global_risk = (1.0 - global_reliability) * 100.0
        global_health = 100.0 - global_risk

        if global_risk == 0:
            st.warning("⚠️ No Data Detected. Please run evaluations in the Transformer, Motor, and Relay modules to populate the dashboard.")
            st.stop()

        # 2. Executive KPIs
        risks_dict = {'Transformer': tx_risk, 'Motor System': motor_sys_risk, 'Protection Relay': relay_risk}
        weakest_link = max(risks_dict, key=risks_dict.get)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='dark-card'><h4>Global Plant Health</h4><h1 style='color:#00ff88; margin:0;'>{global_health:.1f}%</h1></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='dark-card' style='border-left-color:#e74c3c;'><h4>Cascading Failure Risk</h4><h1 style='color:#e74c3c; margin:0;'>{global_risk:.1f}%</h1></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='dark-card' style='border-left-color:#f1c40f;'><h4>Primary Bottleneck</h4><h2 style='color:#f1c40f; margin:0; margin-top:10px;'>{weakest_link}</h2></div>", unsafe_allow_html=True)
        
        action_color = "#e74c3c" if global_risk > 70 else ("#f1c40f" if global_risk > 35 else "#00ff88")
        action_text = "SHUTDOWN" if global_risk > 70 else ("INSPECT" if global_risk > 35 else "OPTIMAL")
        c4.markdown(f"<div class='dark-card' style='border-left-color:{action_color};'><h4>AI Directive</h4><h1 style='color:{action_color}; margin:0;'>{action_text}</h1></div>", unsafe_allow_html=True)

        st.markdown("---")

        # 3. Master FMEA Registry
        st.markdown("### 📋 Master Plant FMEA Registry (Holistic View)")
        master_fmea = pd.DataFrame({
            'Asset Module': ['Transformer Unit (Power)', 'Motor Assembly (Motion)', 'Protection Relay (Control)'],
            'Current Risk': [f"{tx_risk:.1f}%", f"{motor_sys_risk:.1f}%", f"{relay_risk:.1f}%"],
            'System Status': [
                "🔴 DANGER" if tx_risk >= 70 else ("🟡 WARNING" if tx_risk >= 35 else "🟢 SAFE"),
                "🔴 DANGER" if motor_sys_risk >= 75 else ("🟡 WARNING" if motor_sys_risk >= 40 else "🟢 SAFE"),
                "🔴 DANGER" if relay_risk >= 70 else ("🟡 WARNING" if relay_risk >= 30 else "🟢 SAFE")
            ],
            'AI Prescribed Action': [
                st.session_state.get('latest_eval', {}).get('action', 'Monitor'),
                "Schedule Mechanical Check" if motor_sys_risk > 40 else "Monitor",
                "Verify Calibration" if relay_risk > 30 else "Monitor"
            ]
        })
        st.dataframe(master_fmea.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 4. Visual Analytics (Radar & Trend)
        v_col1, v_col2 = st.columns([1, 1.5])
        with v_col1:
            st.markdown("#### 🕸️ System Topology Risk Radar")
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[tx_risk, motor_sys_risk, relay_risk, tx_risk],
                theta=['Transformer', 'Motor', 'Relay', 'Transformer'],
                fill='toself', line_color='#e74c3c', fillcolor='rgba(231, 76, 60, 0.4)'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=)), showlegend=False, height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig_radar, use_container_width=True)

        with v_col2:
            st.markdown("#### 📈 Plant-Wide FMEA Historical Trend")
            hist_data = []
            tx_log = st.session_state.get('fmea_log')
            if not tx_log.empty:
                for _, row in tx_log.iterrows(): hist_data.append({'Time': row['Timestamp'], 'Risk': float(row['Overall Risk (%)']), 'Asset': 'Transformer'})
            
            mo_log = st.session_state.get('inspection_log')
            if not mo_log.empty:
                for _, row in mo_log.iterrows(): hist_data.append({'Time': row['Timestamp'], 'Risk': float(str(row['Risk Level']).replace('%','')), 'Asset': 'Motor'})
            
            re_log = st.session_state.get('relay_fmea_log')
            if not re_log.empty:
                for _, row in re_log.iterrows(): hist_data.append({'Time': row['Timestamp'], 'Risk': float(row['Risk (%)']), 'Asset': 'Relay'})

            if hist_data:
                df_trend = pd.DataFrame(hist_data)
                df_trend['Time'] = pd.to_datetime(df_trend['Time'])
                df_trend = df_trend.sort_values('Time')
                fig_trend = px.line(df_trend, x='Time', y='Risk', color='Asset', markers=True, line_shape='spline', color_discrete_sequence=['#3498DB', '#E74C3C', '#9B59B6'])
                fig_trend.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Risk Level (%)", xaxis_title="")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("📊 Run continuous assessments to build the historical trend line.")

        # 5. Executive Brief
        st.markdown("---")
        st.markdown("### 🧠 Substation AI Executive Briefing")
        if global_risk < 30:
            exec_text = f"The entire substation system is operating optimally with a Global Health Index of **{global_health:.1f}%**. All assets are within safe limits. No cascading failure threats detected."
        elif global_risk < 70:
            exec_text = f"The system exhibits a moderate cascading failure risk of **{global_risk:.1f}%**. The primary threat to plant stability is the **{weakest_link}** (Risk: {risks_dict[weakest_link]:.1f}%). Focus predictive maintenance efforts on this bottleneck to restore full system integrity."
        else:
            exec_text = f"CRITICAL ALERT: The Substation faces an imminent threat of cascading failure (Risk: **{global_risk:.1f}%**). The **{weakest_link}** has exceeded catastrophic limits and threatens to cause a plant-wide blackout. Immediate isolation is strongly recommended."

        st.markdown(f"""
        <div style='background-color: #F4F6F7; padding: 25px; border-radius: 10px; border-left: 8px solid {action_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
            <p style='font-size: 20px; color: #333; line-height: 1.8; margin: 0;'>{exec_text}</p>
        </div>
        """, unsafe_allow_html=True)