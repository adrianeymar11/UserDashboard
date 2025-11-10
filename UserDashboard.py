# ============================================================
# USER-FOCUSED MENTAL HEALTH RISK DASHBOARD
# Author: Adrian Anthony A/L R. Vikneswaran (UTP)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Digital Wellbeing Predictor",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------------------
# STYLE SETTINGS
# ----------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #DFF7F9 0%, #F8FBFC 100%);
    font-family: 'Helvetica Neue', sans-serif;
}
h1, h2, h3 {
    color: #007C91;
    font-weight: 700;
}
.main-card {
    background-color: white;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    padding: 25px;
    margin-top: 20px;
}
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    width: 100%;
    height: 50px;
}
div.stButton > button:first-child {
    background-color: #007C91;
    color: white;
}
div.stButton > button:first-child:hover {
    background-color: #005C68;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
model_path = "RandomForest_user_pipeline.joblib"
if not os.path.exists(model_path):
    st.error("⚠️ Trained model file not found. Please place RandomForest_best_pipeline.joblib in the same folder.")
    st.stop()

model = joblib.load(model_path)

# ----------------------------------------
# LOGIN SYSTEM (with st.rerun)
# ----------------------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Digital Wellbeing Login</h1>", unsafe_allow_html=True)

# Simple credential store (can extend to database later)
users = {
    "adrian": "1234",
    "guest": "0000"
}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------------------
# LOGIN FORM
# ---------------------------
if not st.session_state.logged_in:
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    col_login, col_reset = st.columns(2)
    with col_login:
        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}! 🎉")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

    with col_reset:
        if st.button("Clear Fields"):
            st.session_state.clear()
            st.rerun()

# ---------------------------
# SIDEBAR LOGOUT
# ---------------------------
else:
    st.sidebar.success(f"👋 Logged in as: {st.session_state.username}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.sidebar.info("You have been logged out.")
        st.rerun()


# ----------------------------------------
# MAIN PREDICTION PANEL (Text + Dropdown Hybrid)
# ----------------------------------------
if st.session_state.logged_in:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📊 Digital Wellbeing Assessment (User Input Form)")
    st.markdown("Type numbers directly (e.g., 7.5) and choose your stress level from the dropdown menu.")

    # Text fields for numeric inputs
    age = st.text_input("🎂 Age", value="25")
    sleep_hours = st.text_input("💤 Average Sleep Hours", value="7")
    screen_time = st.text_input("🖥️ Screen Time (Hours/Day)", value="6")
    gaming_hours = st.text_input("🎮 Gaming Hours (Hours/Day)", value="2")
    social_media = st.text_input("📱 Social Media Usage (Hours/Day)", value="3")
    activity_hours = st.text_input("🏃 Physical Activity (Hours/Day)", value="1")
    support_system = st.text_input("🤝 Access to Support Systems (0–10)", value="5")
    online_support = st.text_input("🌐 Online Support Usage (0–10)", value="5")
    work_impact = st.text_input("🏢 Work/Study Environment Impact (0–10)", value="5")

    # Dropdown for stress level
    stress_level = st.selectbox("😰 Stress Level", ["Low", "Medium", "High"])

    # Safe conversion helper
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except ValueError:
            return default

    # Build DataFrame for prediction
    user_input = pd.DataFrame([{
        'Age': safe_float(age, 25),
        'Sleep_Hours': safe_float(sleep_hours, 7),
        'Screen_Time_Hours': safe_float(screen_time, 6),
        'Gaming_Hours': safe_float(gaming_hours, 2),
        'Social_Media_Usage_Hours': safe_float(social_media, 3),
        'Stress_Level': stress_level,
        'Physical_Activity_Hours': safe_float(activity_hours, 1),
        'Support_Systems_Access': safe_float(support_system, 5),
        'Online_Support_Usage': safe_float(online_support, 5),
        'Work_Environment_Impact': safe_float(work_impact, 5)
    }])

    # Predict
    if st.button("🔍 Analyse My Mental Health Risk"):
        with st.spinner('Analysing your wellbeing profile...'):
            prediction = model.predict(user_input)[0]
            proba = model.predict_proba(user_input)[0]
            confidence = max(proba) * 100

        color_map = {'Low': '#00A676', 'Medium': '#FACC15', 'High': '#E53935'}
        bg_color = color_map.get(prediction, '#0097A7')

        st.markdown(f"""
        <div style='background-color:{bg_color}; border-radius:10px; padding:20px; text-align:center;'>
            <h3 style='color:white;'>Predicted Risk Level: <strong>{prediction}</strong></h3>
            <p style='color:white;'>Model Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Personalized suggestions
        suggestions = []
        if safe_float(sleep_hours) < 6:
            suggestions.append("💤 Aim for at least 7–8 hours of sleep per night.")
        if safe_float(screen_time) > 8:
            suggestions.append("📱 Reduce screen time to below 6 hours per day.")
        if safe_float(gaming_hours) > 5:
            suggestions.append("🎮 Limit gaming sessions to under 3 hours daily.")
        if stress_level == "High":
            suggestions.append("🧘 Practice mindfulness or short relaxation breaks.")
        if safe_float(activity_hours) < 1:
            suggestions.append("🏃 Include 30–45 minutes of light physical activity daily.")
        if safe_float(support_system) < 5:
            suggestions.append("🤝 Connect more with family, friends, or counsellors.")
        if safe_float(online_support) < 4:
            suggestions.append("🌐 Explore online mental wellness resources.")
        if safe_float(work_impact) < 5:
            suggestions.append("🏢 Improve your work or study environment for comfort.")

        st.markdown("---")
        st.subheader("📋 Personalized Recommendations")
        if suggestions:
            for s in suggestions:
                st.markdown(f"- {s}")
        else:
            st.success("✅ Your habits are well-balanced! Keep maintaining your healthy lifestyle.")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size:14px; color:#475569;'>
    📊 Dashboard designed by <strong>Adrian Anthony (UTP)</strong> | PETRONAS Digital Project<br>
    Built with <strong>Streamlit</strong> | Random Forest Model | User Mental Health Insights
</div>
---
""", unsafe_allow_html=True)
