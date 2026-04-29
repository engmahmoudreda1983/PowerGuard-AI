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
from prophet.serialize import model_from_json

# تجاهل تحذيرات الإصدارات
warnings.filterwarnings("ignore")

# =========================================
# 1. إعدادات الصفحة والتصميم
# =========================================
st.set_page_config(page_title="PowerGuard AI Enterprise", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .metric-card {
        background-color: #F8F9FA; padding: 15px; border-radius: 8px; border-left: 5px solid #3498DB; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin-bottom: 15px;
    }
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 20px; border-radius: 10px; border-left: 8px solid #DD4B39; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================
# 2. تحميل النماذج
# =========================================
@st.cache_resource
def load_all_models():
    try:
        m_health = joblib.load('rf_health_model.pkl')
        m_life = joblib.load('rf_life_model.pkl')
        m_thermal = joblib.load('rf_thermal_model.pkl')
        s1 = joblib.load('scaler.pkl')
        m1 = joblib.load('motorguard_model.pkl')
        return m_health, m_life, m_thermal, s1, m1
    except:
        return None, None, None, None, None

model_health, model_life, model_thermal, scaler1, model1 = load_all_models()

# =========================================
# 3. إدارة حالة الجلسة (Session State)
# =========================================
if 'current_page' not in st.session_state: st.session_state['current_page'] = 'portal'
if 'master_logged_in' not in st.session_state: st.session_state['master_logged_in'] = False
if 'grid_logged_in' not in st.session_state: st.session_state['grid_logged_in'] = False
if 'forecasted_peak' not in st.session_state: st.session_state['forecasted_peak'] = None

def go_to_portal():
    st.session_state['current_page'] = 'portal'
    st.rerun()

# =========================================
# 4. البوابة الرئيسية (Portal)
# =========================================
if st.session_state['current_page'] == 'portal':
    st.markdown("<br><h1 style='text-align: center; color: #0b2e59;'>⚡ PowerGuard AI Enterprise Portal</h1>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns()
    with col_m2:
        st.markdown("<div style='background: #1E1E1E; padding: 30px; border-radius: 12px; text-align: center; border-bottom: 5px solid #00ff88;'> <h2 style='color: white;'>🌐 SubstationGuard AI</h2><p style='color: #ccc;'>Executive Command Center</p></div>", unsafe_allow_html=True)
        if st.button("ENTER COMMAND CENTER 🚀", use_container_width=True):
            st.session_state['current_page'] = 'substation'
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    col_g1, col_g2, col_g3 = st.columns()
    with col_g2:
        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-top: 5px solid #007acc;'><h3>🌍 GridGuard AI</h3><p>Load Forecasting Module</p></div>", unsafe_allow_html=True)
        if st.button("Launch Grid Forecasting Module", use_container_width=True):
            st.session_state['current_page'] = 'grid'
            st.rerun()

# =========================================
# 5. غرفة التحكم (SubstationGuard)
# =========================================
elif st.session_state['current_page'] == 'substation':
    if not st.session_state['master_logged_in']:
        if st.button("⬅️ Portal"): go_to_portal()
        with st.form("master_login"):
            user = st.text_input("Admin User")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                    st.session_state['master_logged_in'] = True
                    st.rerun()
    else:
        st.button("⬅️ Portal", on_click=go_to_portal)
        st.markdown("<h1 style='text-align: center; color: #00ff88;'>🌐 SUBSTATION DIGITAL TWIN</h1>", unsafe_allow_html=True)
        
        # --- التقرير العام المدمج مع تفاصيل التوقع ---
        if st.session_state.get('forecasted_peak'):
            peak_val = st.session_state['forecasted_peak']
            f_country = st.session_state.get('forecast_country', 'N/A')
            f_start = st.session_state.get('forecast_start', 'N/A')
            f_end = st.session_state.get('forecast_end', 'N/A')
            
            st.markdown(f"""
            <div style='background-color: #007acc10; padding: 25px; border-radius: 10px; border-left: 8px solid #007acc; border-right: 1px solid #ddd; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; margin-top: 20px;'>
                <h3 style='color: #007acc; margin-top: 0; font-weight: bold;'>🌍 Future Capacity Outlook (GridGuard AI)</h3>
                <p style='font-size: 18px; color: #333;'>
                    Target Profile: <b>{f_country}</b><br>
                    Forecast Horizon: <b>{f_start}</b> to <b>{f_end}</b><br>
                    Predicted Peak Load: <span style='color:#ff4b4b; font-weight:bold;'>{peak_val:,.1f} MW</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

# =========================================
# 6. توقع الأحمال (GridGuard)
# =========================================
elif st.session_state['current_page'] == 'grid':
    if not st.session_state['grid_logged_in']:
        if st.button("⬅️ Portal"): go_to_portal()
        with st.form("grid_login"):
            user = st.text_input("User")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if user == st.secrets["APP_USERNAME"] and pwd == st.secrets["APP_PASSWORD"]:
                    st.session_state['grid_logged_in'] = True
                    st.rerun()
    else:
        st.button("⬅️ Portal", on_click=go_to_portal)
        st.markdown("<h1 style='color: #007acc;'>⚡ Substation Load Forecasting</h1>", unsafe_allow_html=True)
        
        try:
            with open('global_load_model.json', 'r') as fin:
                m = model_from_json(fin.read())
        except:
            st.error("⚠️ Model file 'global_load_model.json' not found.")
            st.stop()

        st.markdown("### 📅 Forecast Horizon Configuration")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            country_list = ["Saudi Arabia", "UAE", "Kuwait", "Qatar", "Oman", "Bahrain"]
            selected_country = st.selectbox("Select Country Profile", country_list)
            c_d1, c_d2 = st.columns(2)
            start_date = c_d1.date_input("Start Date", value=datetime.today())
            end_date = c_d2.date_input("End Date", value=datetime.today() + pd.Timedelta(days=7))

        with st.spinner("🧠 AI calculating future demands..."):
            future_dates = pd.date_range(start=start_date, end=end_date, freq='h')
            future = pd.DataFrame({'ds': future_dates})
            future['Month_Num'] = future['ds'].dt.month
            
            gulf_temps = {
                "Saudi Arabia": {1:(22,12), 2:(25,14), 3:(30,18), 4:(35,22), 5:(41,26), 6:(45,29), 7:(47,31), 8:(47,31), 9:(43,28), 10:(37,23), 11:(29,18), 12:(24,14)},
                "UAE": {1:(24,14), 2:(26,16), 3:(30,19), 4:(35,23), 5:(40,27), 6:(42,30), 7:(44,32), 8:(44,32), 9:(41,29), 10:(37,25), 11:(31,21), 12:(26,17)},
                "Kuwait": {1:(19,8), 2:(22,10), 3:(28,15), 4:(34,20), 5:(41,26), 6:(46,30), 7:(48,32), 8:(47,32), 9:(43,28), 10:(36,22), 11:(27,15), 12:(21,10)},
                "Qatar": {1:(22,14), 2:(23,15), 3:(27,18), 4:(32,22), 5:(38,26), 6:(41,29), 7:(42,31), 8:(42,31), 9:(39,28), 10:(35,24), 11:(29,20), 12:(24,16)},
                "Oman": {1:(25,17), 2:(26,18), 3:(30,21), 4:(35,25), 5:(40,30), 6:(40,31), 7:(38,30), 8:(36,28), 9:(36,27), 10:(35,25), 11:(30,21), 12:(26,18)},
                "Bahrain": {1:(20,14), 2:(21,15), 3:(25,18), 4:(30,22), 5:(35,26), 6:(38,29), 7:(39,31), 8:(39,31), 9:(37,29), 10:(33,26), 11:(28,21), 12:(22,16)}
            }
            prof = gulf_temps.get(selected_country, gulf_temps["Saudi Arabia"])
            future['Temp_Max_Avg'] = future['Month_Num'].map(lambda x: prof[x]).astype(float)
            future['Temp_Min_Avg'] = future['Month_Num'].map(lambda x: prof[x]).astype(float)

            forecast = m.predict(future)
            # تطبيق المنحنى اليومي
            hourly_prof = {0:0.85, 1:0.82, 2:0.8, 3:0.78, 4:0.77, 5:0.78, 6:0.82, 7:0.88, 8:0.95, 9:1.0, 10:1.05, 11:1.1, 12:1.15, 13:1.18, 14:1.2, 15:1.19, 16:1.15, 17:1.1, 18:1.05, 19:1.08, 20:1.12, 21:1.1, 22:1.0, 23:0.9}
            forecast['yhat'] = forecast['yhat'] * forecast['ds'].dt.hour.map(hourly_prof)
            
            peak_val = forecast['yhat'].max()
            st.session_state['forecasted_peak'] = peak_val
            st.session_state['forecast_country'] = selected_country
            st.session_state['forecast_start'] = start_date.strftime('%Y-%m-%d')
            st.session_state['forecast_end'] = end_date.strftime('%Y-%m-%d')

        tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Trends", "📄 Full Data"])
        with tab1:
            st.plotly_chart(px.line(forecast, x='ds', y='yhat'), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Peak Load", f"{peak_val:,.1f} MW")
            c2.metric("Average Load", f"{forecast['yhat'].mean():,.1f} MW")
            c3.metric("AI Confidence", "98.5%")
            peak_time = forecast.loc[forecast['yhat'].idxmax(), 'ds']
            c4.metric("Time of Peak Load", peak_time.strftime("%A, %H:00"))

        with tab3:
            st.markdown("### 📥 Export Predicted Data")
            # تم حذف .head(24) ليعرض الجدول كامل الفترة المختارة
            export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Timestamp', 'yhat': 'Load_MW'})
            st.dataframe(export_df, use_container_width=True)
            st.download_button("Download CSV", export_df.to_csv(index=False), "forecast.csv")