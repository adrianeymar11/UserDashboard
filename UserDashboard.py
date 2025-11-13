# ============================================================
# USER-FOCUSED MENTAL HEALTH RISK DASHBOARD
# Author: Adrian Anthony A/L R. Vikneswaran (UTP)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

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
# EXCEL LOGGING HELPER (BACKEND ONLY)
# ----------------------------------------
EXCEL_PATH = "user_submissions.xlsx"

def save_submission_to_excel(record: dict, excel_path: str = EXCEL_PATH):
    df_new = pd.DataFrame([record])

    if os.path.exists(excel_path):
        try:
            df_existing = pd.read_excel(excel_path)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_out = df_new
    else:
        df_out = df_new

    df_out.to_excel(excel_path, index=False, engine="openpyxl")

# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
model_path = "RandomForest_user_pipeline.joblib"
if not os.path.exists(model_path):
    st.error("⚠️ Trained model file not found. Please place RandomForest_user_pipeline.joblib in the same folder.")
    st.stop()

model = joblib.load(model_path)

# ----------------------------------------
# LOGIN SYSTEM
# ----------------------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Digital Wellbeing Login</h1>", unsafe_allow_html=True)

users = {
    "adrian": "1234",
    "guest": "0000"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Login", key="login_btn"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}! 🎉")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

if st.session_state.logged_in:
    st.sidebar.success(f"👋 Logged in as: {st.session_state.username}")
    if st.sidebar.button("🚪 Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.sidebar.info("You have been logged out.")
        st.rerun()

# ----------------------------------------
# MAIN PREDICTION PANEL
# ----------------------------------------
if st.session_state.logged_in:

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📊 Digital Wellbeing Assessment (User Input Form)")

    age = st.text_input("🎂 Age", "25")
    sleep_hours = st.text_input("💤 Average Sleep Hours", "7")
    screen_time = st.text_input("🖥️ Screen Time (Hours/Day)", "6")
    gaming_hours = st.text_input("🎮 Gaming Hours (Hours/Day)", "2")
    social_media = st.text_input("📱 Social Media Usage (Hours/Day)", "3")
    activity_hours = st.text_input("🏃 Physical Activity (Hours/Day)", "1")
    support_system = st.text_input("🤝 Support Systems (0–10)", "5")
    online_support = st.text_input("🌐 Online Support Usage (0–10)", "5")
    work_impact = st.text_input("🏢 Work/Study Environment Impact (0–10)", "5")

    stress_level = st.selectbox("😰 Stress Level", ["Low", "Medium", "High"])

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    user_input = pd.DataFrame([{
        'Age': safe_float(age),
        'Sleep_Hours': safe_float(sleep_hours),
        'Screen_Time_Hours': safe_float(screen_time),
        'Gaming_Hours': safe_float(gaming_hours),
        'Social_Media_Usage_Hours': safe_float(social_media),
        'Stress_Level': stress_level,
        'Physical_Activity_Hours': safe_float(activity_hours),
        'Support_Systems_Access': safe_float(support_system),
        'Online_Support_Usage': safe_float(online_support),
        'Work_Environment_Impact': safe_float(work_impact)
    }])

    if st.button("🔍 Analyse My Mental Health Risk", key="analyse_btn"):
        with st.spinner("Analysing your profile..."):
            prediction = model.predict(user_input)[0]
            proba = model.predict_proba(user_input)[0]
            confidence = max(proba) * 100

        # ------------------------------
        # Risk banner
        # ------------------------------
        color_map = {'Low': '#00A676', 'Medium': '#FACC15', 'High': '#E53935'}
        bg = color_map.get(prediction, "#0097A7")

        st.markdown(f"""
        <div style='background:{bg};border-radius:12px;padding:20px;text-align:center;'>
            <h3 style='color:white;'>Risk Level: <b>{prediction}</b></h3>
            <p style='color:white;'>Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ------------------------------
        # Personalized recommendations
        # ------------------------------
        suggestions = []

        # Rule-based suggestions
        if safe_float(sleep_hours) < 6:
            suggestions.append("💤 Aim for at least **7–8 hours of sleep** per night.")
        if safe_float(screen_time) > 8:
            suggestions.append("📱 Try to reduce **total daily screen time to below 6 hours**.")
        if safe_float(gaming_hours) > 5:
            suggestions.append("🎮 Limit **gaming sessions to under 3 hours per day**.")
        if stress_level == "High":
            suggestions.append("🧘 Practice **relaxation techniques** (breathing, mindfulness, short breaks).")
        if safe_float(activity_hours) < 1:
            suggestions.append("🏃 Include **30–45 minutes of physical activity** in your daily routine.")
        if safe_float(support_system) < 5:
            suggestions.append("🤝 Strengthen connections with **family, friends, or counsellors**.")
        if safe_float(online_support) < 4:
            suggestions.append("🌐 Explore **online mental wellness resources** or helplines.")
        if safe_float(work_impact) < 5:
            suggestions.append("🏢 Improve your **study/work environment** for comfort and focus.")

        st.markdown("---")
        st.subheader("📋 Personalized Recommendations")

        # If no specific rule fired, fall back based on model prediction
        if not suggestions:
            if prediction == "High":
                suggestions.append(
                    "⚠️ Your overall pattern indicates a **high risk level**. "
                    "Consider consulting a mental health professional and reviewing your "
                    "screen time, sleep, and stress management habits."
                )
            elif prediction == "Medium":
                suggestions.append(
                    "🟡 You are at a **moderate risk level**. Try to maintain a structured routine, "
                    "balance digital usage with offline activities, and ensure adequate sleep and exercise."
                )
            else:  # Low
                st.success(
                    "✅ Your habits look **well-balanced** based on your inputs. "
                    "Keep maintaining your healthy lifestyle."
                )

        # Show suggestions list (if any)
        if suggestions:
            for s in suggestions:
                st.markdown(f"- {s}")

        # ------------------------------
        # Backend Excel logging
        # ------------------------------
        proba_dict = dict(zip(model.classes_, proba))

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "username": st.session_state.username,
            **{k: v for k, v in user_input.iloc[0].items()},
            "prediction": prediction,
            "confidence_pct": round(confidence, 2)
        }
        for cls, p in proba_dict.items():
            record[f"proba_{cls}"] = round(float(p), 6)

        save_submission_to_excel(record)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size:14px; color:#475569;'>
    📊 Dashboard designed by <strong>Adrian Anthony (UTP)</strong><br>
    Built with <strong>Streamlit</strong> | Random Forest Model | User Mental Health Insights
</div>
---
""", unsafe_allow_html=True)
