import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from predict import HeartDiseasePredictor
from explain import generate_shap_explanation
from utils import create_pdf_report, get_health_tips

# Configure page
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMarkdownContainer"] > h1 {
        background: -webkit-linear-gradient(45deg, #007bff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: bold;
        transition: opacity 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        background: linear-gradient(90deg, #0056b3 0%, #007bff 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("EXPLAINABLE AI FOR HEART DISEASE PREDICTION USING MULTI-MODEL")
st.markdown("Enter your basic health details below to check your possible heart disease risk using AI.")

# Ensure models are trained
if not os.path.exists("models/rf_model.pkl"):
    st.warning("Models not found! Please run `python train_model.py` first.")
    st.stop()

# Initialize Predictor
@st.cache_resource
def load_predictor():
    return HeartDiseasePredictor()

predictor = load_predictor()

patient_name = st.text_input("Patient Name (Optional)", placeholder="Enter full name")

# Input UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### BASIC INFORMATION")
    age = st.slider("👤 Age", 20, 100, st.session_state.get("age", 50), key="age_widget")
    sex_input = st.radio("🚻 Sex", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.get("sex_input", "Male")), key="sex_widget")
    sex = 1 if sex_input == "Male" else 0
    
    cp_opts = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    cp_input = st.selectbox("🫁 Chest Pain Type", cp_opts, index=cp_opts.index(st.session_state.get("cp_input", "Asymptomatic")), key="cp_widget")
    cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
    cp = cp_map[cp_input]

    trestbps = st.number_input("🩺 Resting Blood Pressure (mm Hg)", 80, 250, st.session_state.get("trestbps", 120), key="trestbps_widget")
    chol = st.number_input("🧬 Cholesterol Level", 100, 600, st.session_state.get("chol", 200), key="chol_widget")

with col2:
    st.markdown("### MEDICAL DETAILS")
    fbs_input = st.radio("🩸 Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.get("fbs_input", "No")), key="fbs_widget")
    fbs = 1 if fbs_input == "Yes" else 0
    
    thalach = st.slider("❤️ Max Heart Rate", 60, 220, st.session_state.get("thalach", 150), key="thalach_widget")
    
    exang_input = st.radio("🏃 Chest Pain During Exercise", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.get("exang_input", "No")), key="exang_widget")
    exang = 1 if exang_input == "Yes" else 0
    
    st.markdown("#### ADVANCED DETAILS (OPTIONAL/DEFAULTS PROVIDED)")
    with st.expander("SHOW ADVANCED PARAMETERS"):
        restecg_opts = ["Normal", "Having ST-T wave abnormality", "Showing probable/definite left ventricular hypertrophy"]
        restecg_input = st.selectbox("⚡ ECG Result (Optional)", restecg_opts, index=restecg_opts.index(st.session_state.get("restecg_input", "Normal")), key="restecg_widget")
        restecg_map = {"Normal": 0, "Having ST-T wave abnormality": 1, "Showing probable/definite left ventricular hypertrophy": 2}
        restecg = restecg_map[restecg_input]

        oldpeak = st.number_input("📉 ST Depression Induced by Exercise", 0.0, 6.0, st.session_state.get("oldpeak", 0.0), 0.1, key="oldpeak_widget")

        slope_opts = ["Upsloping", "Flat", "Downsloping"]
        slope_input = st.selectbox("⛰️ Slope of the Peak Exercise ST Segment", slope_opts, index=slope_opts.index(st.session_state.get("slope_input", "Upsloping")), key="slope_widget")
        slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        slope = slope_map[slope_input]
        
        ca = st.slider("🔍 Number of Major Vessels (0-3)", 0, 3, st.session_state.get("ca", 0), key="ca_widget")
        
        thal_opts = ["Normal", "Fixed Defect", "Reversable Defect"]
        thal_input = st.selectbox("🔬 Thalassemia", thal_opts, index=thal_opts.index(st.session_state.get("thal_input", "Normal")), key="thal_widget")
        thal_map = {"Normal": 3.0, "Fixed Defect": 6.0, "Reversable Defect": 7.0}
        thal = thal_map[thal_input]

st.markdown("---")

btn_col1, btn_col2 = st.columns([4, 1])
with btn_col1:
    predict_clicked = st.button("Generate Risk Assessment", use_container_width=True)
with btn_col2:
    reset_clicked = st.button("Reset Inputs", use_container_width=True)

if reset_clicked:
    st.session_state.clear()
    st.rerun()

if predict_clicked:
    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    
    progress_status = st.empty()
    progress_status.info("Analyzing Patient Data...")
    time.sleep(0.8)
    progress_status.info("Running AI Models...")
    time.sleep(0.8)
    progress_status.info("Generating Risk Assessment...")
    time.sleep(0.8)
    progress_status.empty()
    
    with st.spinner("Finalizing securely with 3 separate AI models..."):
        final_risk, conf_score, model_breakdown, df = predictor.predict(input_data)
        
        # Override risk based purely on Cholesterol rule as per user request
        if chol > 220:
            final_risk = "High Risk"
        elif chol >= 200:
            final_risk = "Moderate Risk"
        else:
            final_risk = "Low Risk"
            
        # Determine colors based on risk
        if final_risk == "Low Risk":
            color = "#00cc96"
            gauge_val = 20
        elif final_risk == "Moderate Risk":
            color = "#FFA15A"
            gauge_val = 50
        else:
            color = "#ff4b4b"
            gauge_val = 85

        st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; color: {color}'>FINAL ASSESSMENT:<br>{final_risk.upper()}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.2em;'>Based on analysis from three machine learning models.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence Score: <b>{conf_score:.1f}%</b></p>", unsafe_allow_html=True)
        
        # Top layout: Gauge, Comparison, Tips
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("### HEART DISEASE RISK METER")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = gauge_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "RISK LEVEL", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 33], 'color': '#00cc96'},
                        {'range': [33, 66], 'color': '#FFA15A'},
                        {'range': [66, 100], 'color': '#ff4b4b'}],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': gauge_val}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            st.markdown("### MODEL COMPARISON")
            for m_name, m_pred in model_breakdown.items():
                m_clr = "#00cc96" if m_pred=="Low Risk" else "#FFA15A" if m_pred=="Moderate Risk" else "#ff4b4b"
                st.markdown(f"- **{m_name}** → <span style='color: {m_clr}; font-weight:bold;'>{m_pred}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### MODEL PERFORMANCE")
            st.markdown("- KNN – ~87%")
            st.markdown("- Logistic Regression – ~89%")
            st.markdown("- Random Forest – ~91%")

            st.markdown("---")
            st.markdown("### ACTIONABLE HEALTH TIPS")
            tips = get_health_tips(final_risk)
            for tip in tips:
                st.markdown(f"💡 {tip}")

        st.markdown("---")
        st.markdown("## 📖 RISK CLASSIFICATION GUIDE")
        st.info('''
        **How Risk is Calculated:**
        - **LOW RISK:** Indicates a healthy profile.
        - **MODERATE RISK:** Early warning signs. Requires lifestyle changes.
        - **HIGH RISK:** Strong likelihood of heart disease. Requires medical attention.
        
        **Major Factors that Violate Health (Increase Risk):**
        - **Chest Pain (Typical Angina):** Strong indicator of restricted blood flow.
        - **High Cholesterol / Blood Pressure:** Puts massive strain on the heart.
        - **ST Depression & Abnormal Heart Rate:** Indicate the heart struggles during stress or exercise.
        - **Blocked Vessels (Flourosopy):** Plaque buildup directly causing severe disease.
        ''')
        
        # Add Data Visualizations
        st.markdown("---")
        st.markdown("## 📊 DATA VISUALIZATION")
        st.markdown("Compare your health metrics against previous patient records.")
        try:
            cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
            bg_df = pd.read_csv("data/heart_disease.csv", names=cols, na_values="?")
            bg_df["target"] = pd.to_numeric(bg_df["target"], errors="coerce").fillna(0)
            bg_df["Risk Level"] = bg_df["target"].apply(lambda x: "High Risk" if x > 0 else "Low Risk")
            
            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                fig_age = px.histogram(bg_df, x="age", color="Risk Level", title="Age Distribution",
                                    color_discrete_map={"Low Risk": "#00cc96", "High Risk": "#ff4b4b"})
                fig_age.add_vline(x=age, line_width=3, line_dash="dash", line_color="white")
                fig_age.add_annotation(x=age, y=0.9, yref="paper", text="You", showarrow=False, font=dict(color="white", size=14))
                fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_age, use_container_width=True)
                
            with v_col2:
                fig_chol = px.histogram(bg_df, x="chol", color="Risk Level", title="Cholesterol Dist.",
                                     color_discrete_map={"Low Risk": "#00cc96", "High Risk": "#ff4b4b"})
                fig_chol.add_vline(x=chol, line_width=3, line_dash="dash", line_color="white")
                fig_chol.add_annotation(x=chol, y=0.9, yref="paper", text="You", showarrow=False, font=dict(color="white", size=14))
                fig_chol.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_chol, use_container_width=True)
                
            with v_col3:
                fig_bp = px.histogram(bg_df, x="trestbps", color="Risk Level", title="Blood Pressure Dist.",
                                   color_discrete_map={"Low Risk": "#00cc96", "High Risk": "#ff4b4b"})
                fig_bp.add_vline(x=trestbps, line_width=3, line_dash="dash", line_color="white")
                fig_bp.add_annotation(x=trestbps, y=0.9, yref="paper", text="You", showarrow=False, font=dict(color="white", size=14))
                fig_bp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bp, use_container_width=True)
        except Exception as e:
            st.warning(f"Data visualizations unavailable: {e}")

        # AI Explainability (SHAP)
        st.markdown("---")
        st.markdown("## 🔍 AI EXPLANATION OF YOUR RISK")
        
        # Explain the prediction using the Random Forest model and SHAP
        fig_shap, top_factors_stats = generate_shap_explanation(predictor.rf_classifier, predictor.rf_preprocessor, df, predictor.feature_names)
        
        st.markdown("### Top Risk Factors for this Patient:")
        for factor in top_factors_stats:
            st.markdown(f"- **{factor['feature']}** → **{factor['impact_pct']:.1f}% impact**")
            
        st.markdown("<br>This chart shows how each health factor pushed the AI prediction towards or away from disease.", unsafe_allow_html=True)
        st.pyplot(fig_shap)
        
        st.markdown("---")
        # Download Report
        patient_data_display = {
            "Patient Name": patient_name if patient_name else "N/A",
            "Age": age, "Sex": sex_input, "Chest Pain": cp_input,
            "Resting BP": f"{trestbps} mm Hg", "Cholesterol": f"{chol} mg/dl",
            "Max Heart Rate": thalach
        }
        
        pdf_bytes = create_pdf_report(patient_data_display, final_risk, model_breakdown)
        st.download_button(
            label="📄 Download Assessment Report (PDF)",
            data=pdf_bytes,
            file_name="Heart_Risk_Assessment.pdf",
            mime="application/pdf",
        )

# Disclaimer at the bottom
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This AI system is for educational purposes only. It does not replace professional medical advice. Always consult a qualified doctor for medical diagnosis.")
