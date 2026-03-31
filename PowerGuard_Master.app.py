import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime

# ==========================================
# 1. إعدادات الصفحة والتصميم الأساسي
# ==========================================
st.set_page_config(page_title="PowerGuard AI Enterprise", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .metric-card {
        background-color: #F8F9FA; padding: 15px; border-radius: 8px; border-left: 5px solid #3498DB; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin-bottom: 15px;
    }
    .zone-bar { width: 100%; height: 20px; display: flex; border-radius: 5px; overflow: hidden; margin-bottom: 5px; font-weight: bold; }
    .zone-safe { width: 40%; background-color: #2ecc71; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .zone-warn { width: 35%; background-color: #f1c40f; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .zone-dang { width: 25%; background-color: #e74c3c; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. تحميل كل النماذج الذكية
# ==========================================
@st.cache_resource
def load_all_models():
    # Transformer Models
    try:
        m_health = joblib.load('rf_health_model.pkl')
        m_life = joblib.load('rf_life_model.pkl')
        m_thermal = joblib.load('rf_thermal_model.pkl')
    except:
        m_health, m_life, m_thermal = None, None, None
        
    # Motor Models
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
# 3. دوال مساعدة للمحولات والمحركات
# ==========================================
def get_model_confidence(model, input_df):
    try:
        preds = np.array([tree.predict(input_df.values) for tree in model.estimators_])
        std = np.std(preds)
        return max(65.0, min(99.0, 98.5 - (std * 2.5)))
    except:
        return 92.5  

def get_duval_diagnosis(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0: return "Normal"
    p_ch4, p_c2h4, p_c2h2 = (ch4/total)*100, (c2h4/total)*100, (c2h2/total)*100
    if p_ch4 >= 98: return "PD"
    if p_c2h2 < 4 and p_c2h4 < 20: return "T1"
    if p_c2h2 < 4 and 20 <= p_c2h4 < 50: return "T2"
    if p_c2h2 < 15 and p_c2h4 >= 50: return "T3"
    if 4 <= p_c2h2 <= 13 and p_c2h4 < 50: return "D1"
    if p_c2h2 > 13 or (p_c2h2 > 4 and p_c2h4 >= 50): return "D2"
    return "DT"

def highlight_fmea(row):
    status = row.get('Status', '')
    if status == 'Safe': return ['background-color: #E8F5E9; color: #00A65A; font-weight: bold'] * len(row)
    if status == 'Warning': return ['background-color: #FEF9E7; color: #D35400; font-weight: bold'] * len(row)
    if status == 'Danger': return ['background-color: #FDEDEC; color: #C0392B; font-weight: bold'] * len(row)
    return [''] * len(row)

def create_gauge(value, title, color, max_v=100.0, h=250):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number={'suffix': "%", 'font': {'size': 26, 'color': color, 'family': "Arial Black"}},
        title={'text': title, 'font': {'size': 16, 'color': '#333', 'weight': 'bold'}},
        gauge={'axis': {'range': list((0, max_v)), 'tickwidth': 1, 'tickcolor': "darkblue"},
               'bar': {'color': color},
               'steps': [{'range': list((0, 40)), 'color': '#E8F5E9'}, {'range': list((40, 75)), 'color': '#FEF9E7'}, {'range': list((75, 100)), 'color': '#FDEDEC'}],
               'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': float(value)}}))
    fig.update_layout(height=h, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 4. إعداد متغيرات الجلسة (Session State)
# ==========================================
if 'current_page' not in st.session_state: st.session_state['current_page'] = 'portal'
if 'tx_logged_in' not in st.session_state: st.session_state['tx_logged_in'] = False
if 'motor_logged_in' not in st.session_state: st.session_state['motor_logged_in'] = False

# Transformer States
if 'fmea_log' not in st.session_state: st.session_state['fmea_log'] = pd.DataFrame(columns=['Timestamp', 'Asset ID', 'DGA Index', 'Pred Temp', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required'])
if 'latest_eval' not in st.session_state: st.session_state['latest_eval'] = None

# Motor States
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'inspection_log' not in st.session_state: 
    st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status', 'Grade', 'Air Temp [K]', 'Core Temp [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Vib RMS', 'Vib Kurt'])

def go_to_portal():
    st.session_state['current_page'] = 'portal'
    st.rerun()

# ==========================================
# 5. البوابة الرئيسية (Portal)
# ==========================================
if st.session_state['current_page'] == 'portal':
    st.markdown("<br><br><h1 style='text-align: center; color: #0b2e59;'>⚡ PowerGuard AI Enterprise Portal</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Select Diagnostic Module</h3><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(list((1, 2, 2, 1)))
    with col2:
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #4da6ff;'><h3>⚡ Transformer & Reactor</h3><p>DGA & Thermal Analytics</p></div>", unsafe_allow_html=True)
        if st.button("Launch Transformer Guard", use_container_width=True):
            st.session_state['current_page'] = 'transformer'
            st.rerun()
            
    with col3:
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #DD4B39;'><h3>⚙️ MotorGuard AI</h3><p>Telemetry & Vibration FMEA</p></div>", unsafe_allow_html=True)
        if st.button("Launch Motor Guard", use_container_width=True):
            st.session_state['current_page'] = 'motor'
            st.rerun()

    st.markdown("<div style='text-align: center; color: #888; padding: 10px; margin-top: 50px;'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim.</strong></div>", unsafe_allow_html=True)

# ==========================================
# 6. المحولات (Transformer Guard)
# ==========================================
elif st.session_state['current_page'] == 'transformer':
    if not st.session_state['tx_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        st.markdown("<br><br><h2 style='text-align: center; color: #4da6ff;'>⚡ TransformerGuard Login</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(list((1, 2, 1)))
        with col2:
            with st.form("tx_login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == "admin" and pwd == "DBA2026":
                        st.session_state['tx_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns(list((10, 1)))
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['tx_logged_in'] = False
                go_to_portal()

        st.markdown("<div style='background-color: #0b2e59; padding: 15px; border-radius: 10px; margin-bottom: 20px;'><h2 style='color: #fff; text-align: center; margin: 0;'>Transformer Guard AI ⚡</h2></div>", unsafe_allow_html=True)

        if model_health is None:
            st.error("⚠️ Transformer Models not found! Check .pkl files.")
        else:
            tab1, tab2 = st.tabs(["🧪 DGA & Oil Quality", "🌡️ THERMAL LOAD ANALYSIS"])

            with tab1:
                col_in, col_out = st.columns(list((1, 2.2)))
                with col_in:
                    with st.form("dga_form"):
                        h2 = st.number_input("H2", value=5.0)
                        ch4 = st.number_input("CH4", value=2.0)
                        co = st.number_input("CO", value=100.0)
                        c2h4 = st.number_input("C2H4", value=1.0)
                        c2h2 = st.number_input("C2H2", value=0.0)
                        dr = st.number_input("Dielectric Rigidity (kV)", value=70.0)
                        sub_dga = st.form_submit_button("Analyze Oil", use_container_width=True)
                with col_out:
                    if sub_dga:
                        input_df = pd.DataFrame([[h2, 500, 10000, ch4, co, 300, c2h4, 5, c2h2, 0.1, 0.05, 45.0, dr, 2.0]], columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'])
                        h_score = model_health.predict(input_df)
                        if dr < 35: h_score += 8.0
                        
                        st.metric("Degradation Index", f"{h_score:.2f} / 60")
                        
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number", value=h_score,
                            gauge={'axis': {'range': list((0, 60)), 'tickwidth': 2, 'tickcolor': "black"},
                                   'bar': {'color': "black"},
                                   'steps': [{'range': list((0, 30)), 'color': "#00cc66"}, {'range': list((30, 45)), 'color': "#ffcc00"}, {'range': list((45, 60)), 'color': "#ff3333"}]}))
                        fig_g.update_layout(height=280, margin=dict(t=30, b=10))
                        st.plotly_chart(fig_g, use_container_width=True)

            with tab2:
                col_t_in, col_t_out = st.columns(list((1, 1.5)))
                with col_t_in:
                    with st.form("thermal_form"):
                        hufl = st.number_input("HUFL", value=12.5)
                        sub_thm = st.form_submit_button("Predict Temp", use_container_width=True)
                with col_t_out:
                    if sub_thm:
                        t_in = pd.DataFrame([[hufl, 3.2, 8.1, 1.5, 4.5, 0.8, 14, 7]], columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'])
                        pred_ot = model_thermal.predict(t_in)
                        st.metric("Predicted Top Oil Temp", f"{pred_ot:.1f} °C")
                        fig_t = go.Figure(go.Indicator(
                            mode="gauge+number", value=pred_ot,
                            gauge={'axis': {'range': list((20, 120))}, 'bar': {'color': "black"},
                                   'steps': [{'range': list((20, 60)), 'color': "#00cc66"}, {'range': list((60, 80)), 'color': "#ffcc00"}, {'range': list((80, 120)), 'color': "#ff3333"}]}))
                        fig_t.update_layout(height=280)
                        st.plotly_chart(fig_t, use_container_width=True)

# ==========================================
# 7. المحركات (MotorGuard AI)
# ==========================================
elif st.session_state['current_page'] == 'motor':
    if not st.session_state['motor_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        st.markdown("<br><br><h2 style='text-align: center; color: #DD4B39;'>⚙️ MotorGuard Login</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(list((1, 2, 1)))
        with col2:
            with st.form("motor_login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    if user == "admin" and pwd == "DBA2026":
                        st.session_state['motor_logged_in'] = True
                        st.rerun()
                    else: st.error("❌ Invalid Credentials")
    else:
        col_l, col_r = st.columns(list((10, 1)))
        with col_l:
            if st.button("⬅️ Portal"): go_to_portal()
        with col_r:
            if st.button("Logout 🚪"):
                st.session_state['motor_logged_in'] = False
                go_to_portal()

        st.markdown("<div style='background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%); padding: 15px; border-radius: 10px; border-left: 8px solid #DD4B39;'><h2 style='color: white; margin: 0;'>⚙️ MotorGuard AI</h2></div><br>", unsafe_allow_html=True)

        if not models_ready:
            st.error("⚠️ Motor Models not found! Check .pkl files.")
        else:
            tab1, tab2, tab3 = st.tabs(["⚙️ Motor Diagnostics", "📳 Bearing Vibration", "📊 Fusion & Report"])

            with tab1:
                c_in, c_out = st.columns(list((1, 2.5)))
                with c_in:
                    motor_type = st.selectbox("Motor Quality", ['L', 'M', 'H'], index=1)
                    air_temp = st.number_input("Ambient Air [K]", value=298.0)
                    process_temp = st.number_input("Core Temp [K]", value=308.0)
                    rpm = st.slider("Speed [RPM]", 1100, 2900, 1500)
                    torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)
                    tool_wear = st.slider("Tool Wear [min]", 0, 260, 50)
                    
                with c_out:
                    m_dict = {'L': 0, 'M': 1, 'H': 2}
                    raw_df = pd.DataFrame({'Type': [m_dict[motor_type]], 'Air temperature [K]': [air_temp], 'Process temperature [K]': [process_temp], 'Rotational speed [rpm]': [rpm], 'Torque [Nm]': [torque], 'Tool wear [min]': [tool_wear]})
                    cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                    
                    processed_df = raw_df.copy()
                    processed_df[cols_to_scale] = scaler1.transform(raw_df[cols_to_scale])
                    
                    probs = np.array(model1.predict_proba(processed_df)).flatten().tolist()
                    p_normal = probs * 100 if len(probs) > 0 else 0.0
                    overall_risk_pct = 100.0 - p_normal
                    
                    st.session_state.motor_risk = overall_risk_pct
                    
                    st.plotly_chart(create_gauge(overall_risk_pct, "MASTER OVERALL RISK", "#DD4B39" if overall_risk_pct > 75 else "#F39C12" if overall_risk_pct > 40 else "#00A65A", h=280), use_container_width=True)

            with tab2:
                sim_level = st.slider("Degradation Phase (%)", 0, 100, 10)
                b_risk = (sim_level / 100.0)**2 * 100.0 + (sim_level / 100.0) * (1.0 - (sim_level / 100.0)) * 200.0
                b_risk = min(100.0, max(0.0, b_risk))
                st.session_state.bearing_risk = b_risk
                
                st.plotly_chart(create_gauge(b_risk, "BEARING RISK", "#DD4B39" if b_risk > 75 else "#F39C12" if b_risk > 40 else "#00A65A", h=280), use_container_width=True)

            with tab3:
                integrated_risk = (1.0 - ((1.0 - st.session_state.motor_risk/100.0) * (1.0 - st.session_state.bearing_risk/100.0))) * 100.0
                st.metric("Asset Health Score", f"{100 - integrated_risk:.1f}%")
                st.metric("Total System Risk", f"{integrated_risk:.1f}%")