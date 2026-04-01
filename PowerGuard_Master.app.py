# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential. Enterprise Portal Edition.
# ==============================================================================

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
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 20px; border-radius: 10px; border-left: 8px solid #DD4B39; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
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
        preds = np.array(list((tree.predict(input_df.values).item() for tree in model.estimators_)))
        std = np.std(preds)
        conf = 98.5 - (std * 2.5)
        return max(65.0, min(99.0, conf))
    except:
        return 92.5  

def get_duval_diagnosis(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0: return "Normal (No significant fault gases)"
    p_ch4, p_c2h4, p_c2h2 = (ch4/total)*100, (c2h4/total)*100, (c2h2/total)*100
    if p_ch4 >= 98: return "PD (Partial Discharge)"
    if p_c2h2 < 4 and p_c2h4 < 20: return "T1 (Thermal Fault < 300°C)"
    if p_c2h2 < 4 and 20 <= p_c2h4 < 50: return "T2 (Thermal Fault 300-700°C)"
    if p_c2h2 < 15 and p_c2h4 >= 50: return "T3 (Thermal Fault > 700°C)"
    if 4 <= p_c2h2 <= 13 and p_c2h4 < 50: return "D1 (Low Energy Discharge)"
    if p_c2h2 > 13 or (p_c2h2 > 4 and p_c2h4 >= 50): return "D2 (High Energy Discharge)"
    return "DT (Mixed Faults)"

def highlight_fmea(row):
    status = row.get('Status', '')
    if status == 'Safe': return list(('background-color: #E8F5E9; color: #00A65A; font-weight: bold',)) * len(row)
    if status == 'Warning': return list(('background-color: #FEF9E7; color: #D35400; font-weight: bold',)) * len(row)
    if status == 'Danger': return list(('background-color: #FDEDEC; color: #C0392B; font-weight: bold',)) * len(row)
    return list(('',)) * len(row)

def create_motor_gauge(value, title, color, max_v=100.0, h=250):
    val = float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={'suffix': "%", 'font': {'size': 26, 'color': color, 'family': "Arial Black"}},
        title={'text': title, 'font': {'size': 16, 'color': '#333', 'weight': 'bold'}},
        gauge={
            'axis': {'range': list((0, max_v)), 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': list((
                {'range': list((0, 40)), 'color': '#E8F5E9'}, 
                {'range': list((40, 75)), 'color': '#FEF9E7'}, 
                {'range': list((75, 100)), 'color': '#FDEDEC'}
            )),
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': val}
        }))
    fig.update_layout(height=h, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 4. إعداد متغيرات الجلسة (Session State)
# ==========================================
if 'current_page' not in st.session_state: st.session_state['current_page'] = 'portal'
if 'tx_logged_in' not in st.session_state: st.session_state['tx_logged_in'] = False
if 'motor_logged_in' not in st.session_state: st.session_state['motor_logged_in'] = False

if 'fmea_log' not in st.session_state: 
    st.session_state['fmea_log'] = pd.DataFrame(columns=list(('Timestamp', 'Asset ID', 'DGA Index', 'Pred Temp', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required')))
if 'latest_eval' not in st.session_state: 
    st.session_state['latest_eval'] = None

if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'inspection_log' not in st.session_state: 
    st.session_state['inspection_log'] = pd.DataFrame(columns=list(('Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status', 'Grade', 'Air Temp [K]', 'Core Temp [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Vib RMS', 'Vib Kurt')))

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
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #4da6ff; height: 160px;'><h3>⚡ Transformer & Reactor</h3><p>DGA & Thermal Analytics</p></div>", unsafe_allow_html=True)
        if st.button("Launch Transformer Guard", use_container_width=True):
            st.session_state['current_page'] = 'transformer'
            st.rerun()
            
    with col3:
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #DD4B39; height: 160px;'><h3>⚙️ MotorGuard AI</h3><p>Telemetry & Vibration FMEA</p></div>", unsafe_allow_html=True)
        if st.button("Launch Motor Guard", use_container_width=True):
            st.session_state['current_page'] = 'motor'
            st.rerun()

    st.markdown("<div style='text-align: center; color: #888; padding: 10px; margin-top: 50px;'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. المحولات (Transformer Guard)
# ==============================================================================
elif st.session_state['current_page'] == 'transformer':
    if not st.session_state['tx_logged_in']:
        if st.button("⬅️ Back to Portal"): go_to_portal()
        st.markdown("<br><br><h1 style='text-align: center; color: #4da6ff;'>⚡ TransformerGuard Login</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(list((1, 2, 1)))
        with col2:
            with st.form("tx_login_form"):
                st.warning("🔒 Restricted Access. Authorized Personnel Only.")
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    # السحب من الـ Secrets بدلاً من الكود الثابت
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
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

        st.markdown("""
        <div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: #ffffff; text-align: center; margin: 0;'>Transformer & Reactor Guard AI ⚡</h1>
            <h3 style='color: #4da6ff; text-align: center; margin: 5px 0 0 0;'>Predictive Maintenance for the Modern Grid</h3>
        </div>
        """, unsafe_allow_html=True)

        if model_health is None or model_thermal is None:
            st.error("⚠️ Model files not detected! Please ensure .pkl files are uploaded.")
            st.stop()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(list(("🧪 DGA & Quality", "🌡️ THERMAL LOAD", "📁 Batch", "📑 RISK Logs", "📄 AI Report")))

        # --- TAB 1 (Real-Time Auto Update) ---
        with tab1:
            col_in, col_out = st.columns(list((1, 2.2)))
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
                ift = st.number_input("Interfacial Tension", min_value=0.0, max_value=75.0, value=45.0)
                dr = st.number_input("Dielectric Rigidity (kV)", value=70.0)
                wc = st.number_input("Water Content (ppm)", value=2.0)

            with col_out:
                input_df = pd.DataFrame(list(((h2, o2, n2, ch4, co, co2, c2h4, c2h6, c2h2, dbds, pf, ift, dr, wc),)), 
                                        columns=list(('Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content')))
                
                h_score = float(model_health.predict(input_df).item())
                l_yrs = float(model_life.predict(input_df).item())
                diagnosis = get_duval_diagnosis(ch4, c2h4, c2h2)

                cf = 0.0
                if dr < 35: cf += 1.0        
                if ift < 20: cf += 1.0       
                if wc > 25: cf += 1.5        
                if co > 500: cf += 1.5       
                if pf > 1.5: cf += 1.0       
                if c2h2 > 10: cf += 2.0      
                
                if cf > 0:
                    h_score = min(60.0, h_score + (cf * 8.0))
                    l_yrs = max(0.2, l_yrs / (cf * 2.5)) 

                if h_score <= 30: status, color = "SAFE", "#00cc66"
                elif h_score <= 45: status, color = "WARNING", "#ffcc00"
                else: status, color = "RISKY", "#ff3333"

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Degradation Index", f"{h_score:.2f}")
                m2.metric("Status", status)
                m3.metric("Life Expectancy", f"{l_yrs:.1f} Yrs")
                m4.metric("AI Confidence", f"{get_tx_confidence(model_health, input_df):.1f} %")

                col_g, col_d = st.columns(list((1, 1.2)))
                with col_g:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=h_score,
                        gauge={'axis': {'range': list((0, 60)), 'tickwidth': 2, 'tickcolor': "black"},
                               'bar': {'color': "black"},
                               'steps': list((
                                   {'range': list((0, 30)), 'color': "#00cc66"},
                                   {'range': list((30, 45)), 'color': "#ffcc00"},
                                   {'range': list((45, 60)), 'color': "#ff3333"}
                               ))}))
                    fig_g.update_layout(height=280, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_g, use_container_width=True)
                    st.markdown(f"<h2 style='text-align: center; color: {color}; margin-top:-30px;'>{status}</h2>", unsafe_allow_html=True)

                with col_d:
                    tot_gas = ch4 + c2h4 + c2h2
                    if tot_gas > 0: p_ch4, p_c2h4, p_c2h2 = (ch4/tot_gas)*100, (c2h4/tot_gas)*100, (c2h2/tot_gas)*100
                    else: p_ch4, p_c2h4, p_c2h2 = 33.33, 33.33, 33.33 
                    
                    fig_tri = go.Figure()
                    fig_tri.add_trace(go.Scatterternary(a=list((100, 0, 0)), b=list((0, 100, 0)), c=list((0, 0, 100)), mode='markers', marker=dict(color='rgba(0,0,0,0)', size=1), hoverinfo='skip'))
                    fig_tri.add_trace(go.Scatterternary(a=list((p_ch4,)), b=list((p_c2h4,)), c=list((p_c2h2,)), mode='markers', marker=dict(size=14, color='red', symbol='cross')))
                    fig_tri.update_layout(title="Duval Triangle", height=320, margin=dict(b=0), ternary=dict(sum=100, aaxis=dict(title="CH4 %", min=0), baxis=dict(title="C2H4 %", min=0), caxis=dict(title="C2H2 %", min=0)), showlegend=False)
                    st.plotly_chart(fig_tri, use_container_width=True)
                    st.info(f"**Diagnosis:** {diagnosis}")
                
                st.markdown("---")
                st.markdown("### 📊 Variables Impact")
                features_list = list(('Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'))
                importances = model_health.feature_importances_
                baselines_dga = np.array(list((5.0, 500.0, 10000.0, 2.0, 100.0, 300.0, 1.0, 5.0, 0.1, 0.1, 0.05, 45.0, 70.0, 2.0)))
                input_vals = input_df.values.flatten()
                deviations = np.abs(input_vals - baselines_dga) / baselines_dga
                dynamic_impacts = importances * deviations
                if dynamic_impacts.sum() > 0: dynamic_impacts = (dynamic_impacts / dynamic_impacts.sum()) * 100
                else: dynamic_impacts = importances * 100
                df_imp = pd.DataFrame({'Feature': features_list, 'Impact (%)': dynamic_impacts}).sort_values(by='Impact (%)')
                fig_imp = px.bar(df_imp, x='Impact (%)', y='Feature', orientation='h', color='Impact (%)', color_continuous_scale='Blues')
                fig_imp.update_layout(height=350, coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)

        # --- TAB 2 (Real-Time Auto Update) ---
        with tab2:
            st.markdown("### 🌡️ Thermal Load Predictive Analysis")
            col_t_in, col_t_out = st.columns(list((1, 1.5)))
            with col_t_in:
                capacity_kva = st.number_input("Rated Capacity (kVA)", min_value=10.0, value=50.0)
                c1, c2 = st.columns(2)
                hour = c1.number_input("Hour (0-23)", min_value=0, max_value=23, value=14)
                month = c2.number_input("Month (1-12)", min_value=1, max_value=12, value=7)
                hufl = st.number_input("HUFL", value=12.5)
                hull = st.number_input("HULL", value=3.2)
                mufl = st.number_input("MUFL", value=8.1)
                mull = st.number_input("MULL", value=1.5)
                lufl = st.number_input("LUFL", value=4.5)
                lull = st.number_input("LULL", value=0.8)
                    
            with col_t_out:
                total_active_load = hufl + mufl + lufl
                load_factor_percent = (total_active_load / capacity_kva) * 100.0
                norm_ratio = 50.0 / capacity_kva
                t_input_norm = pd.DataFrame(list(((hufl*norm_ratio, hull*norm_ratio, mufl*norm_ratio, mull*norm_ratio, lufl*norm_ratio, lull*norm_ratio, hour, month),)), 
                                        columns=list(('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month')))
                
                pred_ot = float(model_thermal.predict(t_input_norm).item())
                load_penalty = (load_factor_percent - 80.0) * 1.2 if load_factor_percent > 80.0 else 0.0
                ambient_penalty = 12.0 if month in list((5, 6, 7, 8, 9)) and 11 <= hour <= 16 else 0.0
                
                if load_penalty > 0 or ambient_penalty > 0: pred_ot = pred_ot + load_penalty + ambient_penalty
                pred_ot = max(20.0, min(120.0, pred_ot))
                
                if pred_ot < 60: t_status, t_color, msg = "NORMAL", "#00cc66", "Temperature is within safe limits."
                elif pred_ot < 80: t_status, t_color, msg = "WARNING", "#ffcc00", "High load detected. Ensure cooling."
                else: t_status, t_color, msg = "RISKY", "#ff3333", "CRITICAL OVERHEATING! Reduce load."
                
                tm1, tm2, tm3 = st.columns(3)
                tm1.metric("Predicted Top Oil Temp", f"{pred_ot:.1f} °C")
                tm2.metric("Thermal Status", t_status)
                tm3.metric("AI Confidence", f"{get_tx_confidence(model_thermal, t_input_norm):.1f} %")
                
                fig_t = go.Figure(go.Indicator(
                    mode = "gauge+number", value = pred_ot,
                    gauge = {'axis': {'range': list((20, 120))}, 'bar': {'color': "black"},
                                'steps': list((
                                    {'range': list((20, 60)), 'color': "rgba(0, 204, 102, 0.4)"},
                                    {'range': list((60, 80)), 'color': "rgba(255, 204, 0, 0.4)"},
                                    {'range': list((80, 120)), 'color': "rgba(255, 51, 51, 0.4)"}
                                ))}))
                fig_t.update_layout(height=350, margin=dict(t=30, b=10))
                st.plotly_chart(fig_t, use_container_width=True)
                st.info(f"💡 **Recommendation:** {msg}")

                st.markdown("---")
                t_features_list = list(('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'))
                t_importances = model_thermal.feature_importances_
                baselines_scada = np.array(list((12.5, 3.2, 8.1, 1.5, 4.5, 0.8, 12.0, 6.0)))
                t_input_vals = t_input_norm.values.flatten()
                deviations_t = np.abs(t_input_vals - baselines_scada) / baselines_scada
                dynamic_impacts_t = t_importances * deviations_t
                if dynamic_impacts_t.sum() > 0: dynamic_impacts_t = (dynamic_impacts_t / dynamic_impacts_t.sum()) * 100
                else: dynamic_impacts_t = t_importances * 100
                df_t_imp = pd.DataFrame({'Feature': t_features_list, 'Impact (%)': dynamic_impacts_t}).sort_values(by='Impact (%)')
                fig_t_imp = px.bar(df_t_imp, x='Impact (%)', y='Feature', orientation='h', color='Impact (%)', color_continuous_scale='Reds')
                fig_t_imp.update_layout(height=300, coloraxis_showscale=False)
                st.plotly_chart(fig_t_imp, use_container_width=True)

        # --- TAB 3: BATCH ---
        with tab3:
            st.markdown("### 📁 Batch Historical Analysis")
            analysis_type = st.radio("Select Analysis Type:", list(("🧪 DGA (Health Index)", "🌡️ SCADA (Oil Temp)")))
            up = st.file_uploader("Upload File", type=list(('csv', 'xlsx')))
            if up:
                df_b = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
                if "DGA" in analysis_type:
                    req_cols = list(('Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'))
                    if all(c in df_b.columns for c in req_cols):
                        df_b['Predicted_Health'] = model_health.predict(df_b[req_cols])
                        st.plotly_chart(px.line(df_b, y='Predicted_Health', markers=True), use_container_width=True)
                elif "SCADA" in analysis_type:
                    req_cols = list(('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'))
                    if all(c in df_b.columns for c in req_cols):
                        df_b['Predicted_Oil_Temp'] = model_thermal.predict(df_b[req_cols])
                        st.plotly_chart(px.line(df_b, y='Predicted_Oil_Temp', markers=True, color_discrete_sequence=list(('#ff3333',))), use_container_width=True)

        # --- TAB 4: RISK ---
        with tab4:
            st.markdown("### 📑 Overall Transformer Risk Assessment (AHI)")
            col_ex_in, col_ex_out = st.columns(list((1, 1.5)))
            with col_ex_in:
                with st.form("tx_exec_form"):
                    c_id1, c_id2 = st.columns(2)
                    tx_list = list((f"T{i:02d} (Substation Unit)" for i in range(1, 11)))
                    asset_selection = c_id1.selectbox("Select Asset:", tx_list)
                    e_capacity = c_id2.number_input("Rated Capacity", min_value=10.0, value=50.0)
                    
                    e_h2 = st.number_input("H2", value=5.0)
                    e_ch4 = st.number_input("CH4", value=2.0)
                    e_co = st.number_input("CO", value=100.0)
                    e_co2 = st.number_input("CO2", value=300.0)
                    e_c2h4 = st.number_input("C2H4", value=1.0)
                    e_c2h2 = st.number_input("C2H2", value=0.0)
                    e_dr = st.number_input("DR (kV)", value=70.0)
                    e_wc = st.number_input("Water (ppm)", value=2.0)
                    e_pf = st.number_input("Power Factor", value=0.05)
                    e_ift = st.number_input("IFT", value=45.0)
                    
                    c_o1, c_o2, c_o3 = st.columns(3)
                    e_hufl = c_o1.number_input("Total Load", value=12.0)
                    e_hour = c_o2.number_input("Hour", value=5)
                    e_month = c_o3.number_input("Month", value=1)
                    
                    gen_report = st.form_submit_button("📊 Evaluate & Push to CMMS", use_container_width=True)
                    
            with col_ex_out:
                if gen_report:
                    asset_id = asset_selection.split(' ') 
                    dga_input = pd.DataFrame(list(((e_h2, 500.0, 10000.0, e_ch4, e_co, e_co2, e_c2h4, 5.0, e_c2h2, 0.1, e_pf, e_ift, e_dr, e_wc),)), 
                                            columns=list(('Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content')))
                    health_score = float(model_health.predict(dga_input).item())

                    cf_ex = 0.0
                    if e_dr < 35: cf_ex += 1.0
                    if e_ift < 20: cf_ex += 1.0
                    if e_wc > 25: cf_ex += 1.5
                    if e_co > 500: cf_ex += 1.5
                    if e_pf > 1.5: cf_ex += 1.0
                    if e_c2h2 > 10: cf_ex += 2.0
                    
                    if cf_ex > 0: health_score = min(60.0, health_score + (cf_ex * 8.0))
                    dga_risk = min(100.0, max(0.0, ((health_score - 20.0) / 30.0) * 100.0)) 
                    
                    load_pct_ex = (e_hufl / e_capacity) * 100.0
                    n_ratio = 50.0 / e_capacity
                    scada_input_norm = pd.DataFrame(list(((e_hufl*n_ratio, (e_hufl*0.2)*n_ratio, (e_hufl*0.6)*n_ratio, (e_hufl*0.1)*n_ratio, (e_hufl*0.3)*n_ratio, (e_hufl*0.05)*n_ratio, e_hour, e_month),)), 
                                           columns=list(('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month')))
                    ot_temp = float(model_thermal.predict(scada_input_norm).item())

                    load_pen = (load_pct_ex - 80.0) * 1.2 if load_pct_ex > 80.0 else 0.0
                    amb_pen = 12.0 if e_month in list((5, 6, 7, 8, 9)) and 11 <= e_hour <= 16 else 0.0
                    if load_pen > 0 or amb_pen > 0: ot_temp = ot_temp + load_pen + amb_pen
                    ot_temp = max(20.0, min(120.0, ot_temp))

                    thermal_risk = min(100.0, max(0.0, ((ot_temp - 40.0) / 45.0) * 100.0))
                    overall_risk = (dga_risk * 0.6) + (thermal_risk * 0.4)
                    if ot_temp >= 80.0 or health_score > 45.0: overall_risk = max(overall_risk, 75.0) 
                    
                    if overall_risk <= 35: final_status, final_color, final_action, rpn_level = "🟢 EXCELLENT (Low Risk)", "#00cc66", "Continue operation.", "Low"
                    elif overall_risk <= 70: final_status, final_color, final_action, rpn_level = "🟡 WATCH (Medium Risk)", "#ffcc00", "Schedule PM.", "Medium"
                    else: final_status, final_color, final_action, rpn_level = "🔴 ACTION REQUIRED", "#ff3333", "URGENT: Isolate asset.", "Critical"
                    
                    st.session_state['latest_eval'] = {
                        'asset_id': asset_id, 'capacity': e_capacity, 'load_factor': load_pct_ex,
                        'overall_risk': overall_risk, 'dga_score': health_score, 'pred_temp': ot_temp,
                        'status': final_status, 'action': final_action, 'rpn_level': rpn_level, 'color': final_color,
                        'duval': get_duval_diagnosis(e_ch4, e_c2h4, e_c2h2),
                        'inputs': {'dr': e_dr, 'wc': e_wc, 'ift': e_ift, 'pf': e_pf, 'co': e_co, 'c2h2': e_c2h2, 'month': e_month, 'hour': e_hour, 'ch4': e_ch4, 'c2h4': e_c2h4},
                        'dga_input_df': dga_input, 'thermal_input_df': scada_input_norm
                    }
                    
                    new_entry = pd.DataFrame(list(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset ID': asset_id, 'DGA Index': round(health_score, 1), 'Pred Temp': round(ot_temp, 1), 'Overall Risk (%)': round(overall_risk, 1), 'Status': final_status.split(' '), 'RPN Level': rpn_level, 'Action Required': final_action},)))
                    st.session_state['fmea_log'] = pd.concat((st.session_state['fmea_log'], new_entry), ignore_index=True).tail(100)
                    
                    st.markdown(f"<h2 style='text-align: center; color: {final_color}; border: 2px solid {final_color}; padding: 10px; border-radius: 5px;'>{final_status}</h2>", unsafe_allow_html=True)
                    
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Degradation Index", f"{health_score:.1f}")
                    rc2.metric("Predicted Temp", f"{ot_temp:.1f} °C")
                    rc3.metric("OVERALL RISK", f"{overall_risk:.1f} %")
                    rc4.metric("AI Confidence", f"{(get_tx_confidence(model_health, dga_input) + get_tx_confidence(model_thermal, scada_input_norm))/2:.1f} %")
                    
                    fig_risk = go.Figure(go.Indicator(
                        mode = "gauge+number", value = overall_risk,
                        gauge = {'axis': {'range': list((0, 100))}, 'bar': {'color': "black"},
                                 'steps': list((
                                     {'range': list((0, 35)), 'color': "rgba(0, 204, 102, 0.4)"}, 
                                     {'range': list((35, 70)), 'color': "rgba(255, 204, 0, 0.4)"}, 
                                     {'range': list((70, 100)), 'color': "rgba(255, 51, 51, 0.4)"}
                                 ))}))
                    fig_risk.update_layout(height=300, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📋 Dynamic Risk Logs")
            if not st.session_state['fmea_log'].empty:
                def color_risk(val): return f"color: {'#00cc66' if val == 'EXCELLENT' else '#ffcc00' if val == 'WATCH' else '#ff3333'}; font-weight: bold"
                try: st.dataframe(st.session_state['fmea_log'].style.map(color_risk, subset=['Status']), use_container_width=True)
                except: st.dataframe(st.session_state['fmea_log'].style.applymap(color_risk, subset=['Status']), use_container_width=True)

        # --- TAB 5: AI REPORT (RESTORED FULL VERSION) ---
        with tab5:
            if st.session_state.get('latest_eval') is None: 
                st.markdown("<br><br><h3 style='text-align: center; color: #888;'>💡 Please generate a risk assessment from TAB 4 first.</h3>", unsafe_allow_html=True)
            else:
                e_data = st.session_state['latest_eval']
                st.markdown(f"<div style='background-color: rgba(77, 166, 255, 0.1); padding: 15px; border-left: 5px solid #4da6ff; border-radius: 5px;'><h2 style='margin:0; color: #0b2e59;'>📄 AI Executive Diagnostic Report</h2><p style='margin:0; color: #555555;'>Asset ID: <b>{e_data['asset_id']}</b> | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p></div><br>", unsafe_allow_html=True)
                
                # 1. Overall System Risk & FMEA SECTION
                st.markdown("### 🎯 1. Overall System Risk & FMEA Directive")
                r_col1, r_col2 = st.columns(list((1, 1.8)))
                
                with r_col1:
                    fig_r = go.Figure(go.Indicator(
                        mode = "gauge+number", value = e_data['overall_risk'],
                        number={'font': {'size': 40, 'color': '#0b2e59'}},
                        gauge = {'axis': {'range': list((0, 100)), 'tickwidth': 1, 'tickcolor': "black"},
                                 'bar': {'color': "rgba(0,0,0,0.5)", 'thickness': 0.1},
                                 'steps': list((
                                     {'range': list((0, 35)), 'color': "#00cc66"},
                                     {'range': list((35, 70)), 'color': "#ffcc00"},
                                     {'range': list((70, 100)), 'color': "#ff3333"}
                                 ))}))
                    fig_r.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_r, use_container_width=True)
                    
                with r_col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    def color_r(val): return f"color: {'#00cc66' if val == 'EXCELLENT' else '#ffcc00' if val == 'WATCH' else '#ff3333'}; font-weight: bold"
                    latest_log = st.session_state['fmea_log'].tail(1)
                    try: st.dataframe(latest_log.style.map(color_r, subset=['Status']), use_container_width=True)
                    except: st.dataframe(latest_log.style.applymap(color_r, subset=['Status']), use_container_width=True)
                    st.markdown(f"**Mandatory Action:** <span style='color:{e_data['color']}; font-weight:bold;'>{e_data['action']}</span>", unsafe_allow_html=True)

                st.markdown("---")

                # 2. VISUAL ANALYTICS SECTION
                st.markdown("### 📊 2. AI Visual Analytics (Combined Root Cause Identifiers)")
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    ch4_v, c2h4_v, c2h2_v = e_data['inputs']['ch4'], e_data['inputs']['c2h4'], e_data['inputs']['c2h2']
                    tot_gas_v = ch4_v + c2h4_v + c2h2_v
                    if tot_gas_v > 0: p_ch4_v, p_c2h4_v, p_c2h2_v = (ch4_v/tot_gas_v)*100, (c2h4_v/tot_gas_v)*100, (c2h2_v/tot_gas_v)*100
                    else: p_ch4_v, p_c2h4_v, p_c2h2_v = 33.33, 33.33, 33.33 
                    
                    fig_tri_v = go.Figure()
                    fig_tri_v.add_trace(go.Scatterternary(a=list((100, 0, 0)), b=list((0, 100, 0)), c=list((0, 0, 100)), mode='markers', marker=dict(color='rgba(0,0,0,0)', size=1), hoverinfo='skip'))
                    fig_tri_v.add_trace(go.Scatterternary(a=list((p_ch4_v,)), b=list((p_c2h4_v,)), c=list((p_c2h2_v,)), mode='markers', marker=dict(size=14, color='red', symbol='cross', line=dict(width=2, color='black')), name='Diagnosis'))
                    fig_tri_v.update_layout(title={'text': "Duval Triangle (Fault Type)", 'font': {'color': 'black'}, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, height=450, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", ternary=dict(sum=100, aaxis=dict(title="CH4 %", min=0), baxis=dict(title="C2H4 %", min=0), caxis=dict(title="C2H2 %", min=0)), showlegend=False)
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
                    fig_imp.update_layout(title={'text': "Top 10 Impactful Variables (Combined)", 'font': {'color': 'black'}}, height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, font={'color': 'black'})
                    st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("---")

                # 3. TEXTUAL AI FINDINGS SECTION
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
        st.markdown("<br><br><h1 style='text-align: center; color: #DD4B39;'>⚙️ MotorGuard Login</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(list((1, 2, 1)))
        with col2:
            with st.form("motor_login_form"):
                st.warning("🔒 Restricted Access. Authorized Personnel Only.")
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", use_container_width=True):
                    # السحب من الـ Secrets بدلاً من الكود الثابت
                    if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
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

        st.markdown("""
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div><h2 style="margin:0;">⚙️ MotorGuard AI</h2><p style="margin:0; font-size: 14px; color:#CCCCCC;">Advanced Asset Performance Management</p></div>
                <div style="text-align: right;"><h4 style="margin:0; color:#00ff88;">⚡ LIVE TELEMETRY</h4></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not models_ready:
            st.error("⚠️ Motor Models not found! Check .pkl files.")
        else:
            tab1, tab2, tab3, tab4 = st.tabs(list(("⚙️ Motor Diagnostics", "📳 Bearing Vibration", "🔗 Reliability Fusion", "📊 Executive Summary")))

            # --- TAB 1: MOTOR DIAGNOSTICS ---
            with tab1:
                c_in, c_out = st.columns(list((1, 2.5)))
                with c_in:
                    st.markdown("#### 🎛️ Live Sensor Telemetry")
                    motor_options = {'L (Low - Standard Build)': 0, 'M (Medium - Commercial)': 1, 'H (High - Industrial Grade)': 2}
                    motor_type = st.selectbox("Motor Quality Grade ❓", list(motor_options.keys()), index=1)
                    st.markdown("---")
                    st.markdown("#### 🌡️ Thermal Metrics")
                    air_temp = st.number_input("Ambient Air Temp [K] ❓", min_value=290.0, max_value=310.0, value=298.0, step=0.1)
                    process_temp = st.number_input("Motor Core Temp [K] ❓", min_value=300.0, max_value=320.0, value=308.0, step=0.1)
                    st.markdown("---")
                    st.markdown("#### ⚙️ Mechanical Metrics")
                    rpm = st.slider("Rotational Speed [RPM] ❓", 1100, 2900, 1500, 10)
                    torque = st.slider("Applied Torque [Nm] ❓", 10.0, 80.0, 40.0, 0.5)
                    tool_wear = st.slider("Tool Wear Level [min] ❓", 0, 260, 50, 1)
                    
                with c_out:
                    raw_df = pd.DataFrame({'Type': list((motor_options[motor_type],)), 'Air temperature [K]': list((air_temp,)), 'Process temperature [K]': list((process_temp,)), 'Rotational speed [rpm]': list((rpm,)), 'Torque [Nm]': list((torque,)), 'Tool wear [min]': list((tool_wear,))})
                    cols_to_scale = list(('Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'))
                    processed_df = raw_df.copy()
                    processed_df[cols_to_scale] = scaler1.transform(raw_df[cols_to_scale])
                    
                    probs_matrix = getattr(model1, 'predict_proba')(processed_df)
                    probs_list = np.array(probs_matrix).flatten().tolist()
                    
                    if hasattr(model1, 'classes_'):
                        classes = list((str(c).strip().upper() for c in getattr(model1, 'classes_')))
                        probs_dict = dict(zip(classes, probs_list))
                        p_normal = float(probs_dict.get('NORMAL', 0.0)) * 100.0
                        p_hdf_raw = float(probs_dict.get('HEAT DISSIPATION FAILURE (HDF)', probs_dict.get('HDF', 0.0))) * 100.0
                        p_osf_raw = float(probs_dict.get('OVERSTRAIN FAILURE (OSF)', probs_dict.get('OSF', 0.0))) * 100.0
                        p_pwf_raw = float(probs_dict.get('POWER FAILURE (PWF)', probs_dict.get('PWF', 0.0))) * 100.0
                        p_twf_raw = float(probs_dict.get('TOOL WEAR FAILURE (TWF)', probs_dict.get('TWF', 0.0))) * 100.0
                        raw_overall_risk = 100.0 - p_normal
                    else:
                        p_normal = float(probs_list.pop(0)) * 100.0 if len(probs_list) > 0 else 0.0
                        raw_overall_risk = 100.0 - p_normal
                        p_hdf_raw = p_osf_raw = p_pwf_raw = p_twf_raw = 0.0

                    expert_twf, expert_pwf, expert_osf, expert_hdf = 0.0, 0.0, 0.0, 0.0
                    if tool_wear >= 200: expert_twf = min(50.0 + ((tool_wear - 200.0) / 40.0) * 45.0, 98.0)
                    power = torque * rpm
                    if power > 75000: expert_pwf = min(40.0 + ((power - 75000.0) / 15000.0) * 55.0, 98.0)
                    elif power < 40000: expert_pwf = min(40.0 + ((40000.0 - power) / 10000.0) * 55.0, 98.0)
                    strain = torque * tool_wear
                    strain_limit = 11000 if motor_options[motor_type] == 0 else (12000 if motor_options[motor_type] == 1 else 13000)
                    warning_limit = strain_limit * 0.8
                    if strain > warning_limit: expert_osf = min(40.0 + ((strain - warning_limit) / (strain_limit - warning_limit)) * 55.0, 98.0)
                    temp_diff = process_temp - air_temp
                    if temp_diff <= 10.0 and rpm < 1500: expert_hdf = min(40.0 + ((10.0 - temp_diff) / 2.0) * 55.0, 98.0)

                    p_hdf = max(p_hdf_raw, expert_hdf)
                    p_osf = max(p_osf_raw, expert_osf)
                    p_pwf = max(p_pwf_raw, expert_pwf)
                    p_twf = max(p_twf_raw, expert_twf)

                    calc_overall_risk = (1.0 - ((1.0 - p_hdf/100.0) * (1.0 - p_osf/100.0) * (1.0 - p_pwf/100.0) * (1.0 - p_twf/100.0))) * 100.0
                    overall_risk_pct = min(100.0, max(raw_overall_risk, calc_overall_risk))
                    ai_confidence = p_normal if overall_risk_pct < 50 else max(list((p_hdf, p_osf, p_pwf, p_twf)))

                    fault_dict = {'Heat (HDF)': p_hdf, 'Overstrain (OSF)': p_osf, 'Power (PWF)': p_pwf, 'Wear (TWF)': p_twf}
                    highest_fault_name = max(fault_dict, key=fault_dict.get)

                    st.session_state.motor_risk = overall_risk_pct
                    st.session_state.motor_faults = {'HDF': p_hdf, 'OSF': p_osf, 'PWF': p_pwf, 'TWF': p_twf}

                    if overall_risk_pct < 40.0: status_color, status_bg, status_title, sys_status, diag_display = "#00A65A", "#E8F5E9", "✅ SYSTEM NORMAL", "Safe", "NORMAL"
                    elif overall_risk_pct < 75.0: status_color, status_bg, status_title, sys_status, diag_display = "#F39C12", "#FEF9E7", "⚠️ ELEVATED RISK", "Warning", highest_fault_name
                    else: status_color, status_bg, status_title, sys_status, diag_display = "#DD4B39", "#FDEDEC", f"🚨 DANGER: {highest_fault_name}", "Danger", highest_fault_name
                        
                    st.markdown(f"<div style='background-color:{status_bg}; border: 2px solid {status_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom:20px;'><h2 style='color:{status_color}; margin:0;'>{status_title}</h2><p style='color:#555; font-size:16px; margin:0;'>Live Assessment based on AI & Expert Overrides</p></div>", unsafe_allow_html=True)
                        
                    colA, colB = st.columns(list((1, 1.5)))
                    with colA:
                        st.markdown(f"<br><div class='metric-card'><h4 style='color:#666; margin-bottom:5px;'>AI Confidence Score</h4><h2 style='color:#3498DB; margin:0;'>{ai_confidence:.1f}%</h2></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card' style='border-left-color: {status_color};'><h4 style='color:#666; margin-bottom:5px;'>Primary FMEA Concern</h4><h2 style='color:{status_color}; margin:0;'>{diag_display}</h2></div>", unsafe_allow_html=True)
                    with colB:
                        st.markdown("<div class='zone-bar'><div class='zone-safe'>SAFE (0-40)</div><div class='zone-warn'>WARN (40-75)</div><div class='zone-dang'>DANGER (75-100)</div></div>", unsafe_allow_html=True)
                        st.plotly_chart(create_motor_gauge(overall_risk_pct, "MASTER OVERALL RISK", status_color, h=280), use_container_width=True)

                    st.markdown("---")
                    g1, g2, g3, g4 = st.columns(4)
                    with g1: st.plotly_chart(create_motor_gauge(p_hdf, "Heat (HDF)", "#E67E22", h=160), use_container_width=True)
                    with g2: st.plotly_chart(create_motor_gauge(p_osf, "Strain (OSF)", "#9B59B6", h=160), use_container_width=True)
                    with g3: st.plotly_chart(create_motor_gauge(p_pwf, "Power (PWF)", "#E74C3C", h=160), use_container_width=True)
                    with g4: st.plotly_chart(create_motor_gauge(p_twf, "Wear (TWF)", "#34495E", h=160), use_container_width=True)

                    new_log_raw = pd.DataFrame(list(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Motor Core', 'Risk Level': f"{overall_risk_pct:.1f}%", 'Primary FMEA': diag_display, 'Status': sys_status, 'Grade': str(motor_type).split(' '), 'Air Temp [K]': air_temp, 'Core Temp [K]': process_temp, 'RPM': rpm, 'Torque [Nm]': torque, 'Wear [min]': tool_wear, 'Vib RMS': '-', 'Vib Kurt': '-'},)))
                    st.session_state.inspection_log = pd.concat((new_log_raw, st.session_state.inspection_log), ignore_index=True).head(30)

            # --- TAB 2: BEARING VIBRATION ---
            with tab2:
                cv1, cv2 = st.columns(list((1.5, 1)))
                with cv1:
                    st.markdown("#### 🎚️ Simulator & Key Metrics")
                    sim_level = st.slider("Degradation Phase (%) ❓", 0, 100, 10)
                    r_val = 0.05 + (sim_level/100.0)**2 * 0.20
                    k_val = 3.0 + (sim_level/100.0)**1.5 * 5.0
                    p2p_val = r_val * 4.5
                    cr_val = 3.5 + (sim_level/100.0)
                    
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("RMS", f"{r_val:.3f}")
                    k2.metric("Kurtosis", f"{k_val:.2f}")
                    k3.metric("Peak-to-Peak", f"{p2p_val:.2f}")
                    k4.metric("Crest Factor", f"{cr_val:.2f}")

                with cv2:
                    p_b_fail = (sim_level / 100.0)**2 * 100.0
                    p_b_warn = (sim_level / 100.0) * (1.0 - (sim_level / 100.0)) * 200.0
                    p_b_early = (1.0 - (sim_level / 100.0))**2 * 100.0 if sim_level > 20 else (sim_level * 2.0)
                    p_b_norm = max(0.0, 100.0 - (p_b_fail + p_b_warn + p_b_early))
                    
                    tot = p_b_fail + p_b_warn + p_b_early + p_b_norm
                    if tot > 0: p_b_fail, p_b_warn, p_b_early, p_b_norm = (p_b_fail/tot)*100, (p_b_warn/tot)*100, (p_b_early/tot)*100, (p_b_norm/tot)*100
                    
                    raw_b_risk = (p_b_early * 0.3) + (p_b_warn * 0.7) + (p_b_fail * 1.0)
                    b_risk = 100.0 if raw_b_risk >= 85.0 else raw_b_risk
                    st.session_state.bearing_risk = b_risk
                    
                    b_color = "#00A65A" if b_risk < 40 else ("#F39C12" if b_risk < 75 else "#DD4B39")
                    b_status = "Safe" if b_risk < 40 else ("Warning" if b_risk < 75 else "Danger")
                    st.plotly_chart(create_motor_gauge(b_risk, "BEARING VIBRATION RISK", b_color, max_v=100.0, h=280), use_container_width=True)

                st.markdown("---")
                df_b = pd.DataFrame({'Phase': list(('Normal', 'Early Wear', 'Warning', 'Failure')), 'Probability': list((p_b_norm, p_b_early, p_b_warn, p_b_fail))})
                fig_b = px.bar(df_b, x='Probability', y='Phase', orientation='h', color='Phase', color_discrete_map={'Normal': '#2ecc71', 'Early Wear': '#f1c40f', 'Warning': '#e67e22', 'Failure': '#e74c3c'})
                fig_b.update_layout(height=250)
                st.plotly_chart(fig_b, use_container_width=True)

                new_log_b = pd.DataFrame(list(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Bearing', 'Risk Level': f"{b_risk:.1f}%", 'Primary FMEA': 'Vibration Anomalies', 'Status': b_status, 'Grade': '-', 'Air Temp [K]': '-', 'Core Temp [K]': '-', 'RPM': '-', 'Torque [Nm]': '-', 'Wear [min]': '-', 'Vib RMS': round(r_val, 3), 'Vib Kurt': round(k_val, 2)},)))
                st.session_state.inspection_log = pd.concat((new_log_b, st.session_state.inspection_log), ignore_index=True).head(30)

            # --- TAB 3: RELIABILITY FUSION ---
            with tab3:
                st.markdown("### 🔗 Master Reliability Fusion (Series System)")
                integrated_risk = (1.0 - ((1.0 - st.session_state.motor_risk/100.0) * (1.0 - st.session_state.bearing_risk/100.0))) * 100.0
                f_color = "#00A65A" if integrated_risk < 40 else ("#F39C12" if integrated_risk < 75 else "#DD4B39")
                
                fc1, fc2, fc3 = st.columns(3)
                fc1.markdown(f"<div class='metric-card' style='border-left-color:#3498DB;'><h4>Motor Risk Factor</h4><h2 style='color:#3498DB;'>{st.session_state.motor_risk:.1f}%</h2></div>", unsafe_allow_html=True)
                fc2.markdown(f"<div class='metric-card' style='border-left-color:#9B59B6;'><h4>Bearing Risk Factor</h4><h2 style='color:#9B59B6;'>{st.session_state.bearing_risk:.1f}%</h2></div>", unsafe_allow_html=True)
                fc3.markdown(f"<div class='metric-card' style='border-left-color:{f_color};'><h4>Total Integrated Risk</h4><h2 style='color:{f_color};'>{integrated_risk:.1f}%</h2></div>", unsafe_allow_html=True)

                col_w, col_g = st.columns(list((1.5, 1)))
                with col_w:
                    st.markdown("#### 📊 Risk Contributor Analysis")
                    fig_comp = go.Figure(data=list((
                        go.Bar(name='Motor Factor', x=list(('Risk Component',)), y=list((st.session_state.motor_risk,)), marker_color='#3498DB', text=list((f"{st.session_state.motor_risk:.1f}%",)), textposition='auto'),
                        go.Bar(name='System (Series)', x=list(('Risk Component',)), y=list((integrated_risk,)), marker_color=f_color, text=list((f"{integrated_risk:.1f}%",)), textposition='auto'),
                        go.Bar(name='Bearing Factor', x=list(('Risk Component',)), y=list((st.session_state.bearing_risk,)), marker_color='#9B59B6', text=list((f"{st.session_state.bearing_risk:.1f}%",)), textposition='auto')
                    )))
                    fig_comp.update_layout(barmode='group', height=350, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_comp, use_container_width=True)

                with col_g:
                    st.plotly_chart(create_motor_gauge(integrated_risk, "OVERALL SYSTEM RISK", f_color, max_v=100.0, h=300), use_container_width=True)
                    action_text = "EMERGENCY SHUTDOWN REQUIRED" if integrated_risk > 75 else ("SCHEDULE PREVENTIVE MAINTENANCE" if integrated_risk > 40 else "OPTIMAL PRODUCTION CONTINUES")
                    st.markdown(f"<div style='text-align:center; padding:15px; background-color:#F8F9FA; border-radius:5px; border: 2px dashed {f_color};'><h4 style='margin:0; color:#555;'>AI Recommended Action:</h4><h3 style='margin:0; color:{f_color};'>{action_text}</h3></div>", unsafe_allow_html=True)

            # --- TAB 4: EXECUTIVE SUMMARY ---
            with tab4:
                integrated_risk = (1.0 - ((1.0 - st.session_state.motor_risk/100.0) * (1.0 - st.session_state.bearing_risk/100.0))) * 100.0
                health_score = 100.0 - integrated_risk
                rul_days = int((health_score / 100.0) * 365 * 3) 
                
                st.markdown("### 📊 Executive Summary & Reporting")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Asset Health Score", f"{health_score:.1f}%")
                e2.metric("Unplanned Downtime Prob", f"{integrated_risk:.1f}%")
                e3.metric("Est. Remaining Useful Life", f"{rul_days} Days" if rul_days > 30 else "IMMINENT FAILURE")
                e4.metric("Maintenance Urgency", "CRITICAL" if integrated_risk > 75 else ("MODERATE" if integrated_risk > 40 else "LOW"))
                
                st.markdown("---")
                er1, er2 = st.columns(list((1.5, 2)))
                with er1:
                    st.markdown("#### 🎯 Holistic Fault Radar")
                    cats = list(('Heat (HDF)', 'Strain (OSF)', 'Power (PWF)', 'Wear (TWF)', 'Bearing (Vib)'))
                    vals = list((st.session_state.motor_faults.get('HDF',0), st.session_state.motor_faults.get('OSF',0), st.session_state.motor_faults.get('PWF',0), st.session_state.motor_faults.get('TWF',0), st.session_state.bearing_risk))
                    fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself', line_color='#DD4B39'))
                    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=list((0, 100)))), showlegend=False, height=350, margin=dict(t=30, b=30))
                    st.plotly_chart(fig_r, use_container_width=True)
                with er2:
                    st.markdown("#### 🛡️ System Viability Index")
                    df_viability = pd.DataFrame({'Component': list(('Motor Core', 'Bearing Assembly', 'Total System')), 'Safe %': list((100.0 - st.session_state.motor_risk, 100.0 - st.session_state.bearing_risk, health_score)), 'At Risk %': list((st.session_state.motor_risk, st.session_state.bearing_risk, integrated_risk))})
                    fig_viab = px.bar(df_viability, x="Component", y=list(("Safe %", "At Risk %")), color_discrete_sequence=list(('#2ecc71', '#e74c3c')))
                    fig_viab.update_layout(height=350, barmode='stack', margin=dict(t=40, b=10))
                    st.plotly_chart(fig_viab, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 📋 Comprehensive Diagnostic Log")
                if not st.session_state.inspection_log.empty:
                    st.dataframe(st.session_state.inspection_log.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True, height=250)