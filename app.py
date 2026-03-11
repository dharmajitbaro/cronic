import streamlit as st
import numpy as np
import joblib

# 1. Load trained model and scaler
# Make sure these files are in the same folder as this script
try:
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or Scaler file not found. Please ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the directory.")

# 2. Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered", page_icon="❤️")

st.title("❤️ Heart Disease Risk Assessment")
st.write("Enter the patient's clinical data below for a digital diagnostic check.")
st.markdown("---")

# -------------------- INPUT SECTION -------------------- #
# Using columns to make the interface compact and easy to read
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 45)
    sex_label = st.selectbox("Sex", ["Female", "Male"])
    cp_label = st.selectbox(
        "Chest Pain Type", 
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    restecg_label = st.selectbox(
        "Resting ECG Results", 
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang_label = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
    slope_label = st.selectbox(
        "Slope of Peak Exercise ST", 
        ["Upsloping", "Flat", "Downsloping"]
    )
    ca = st.selectbox("Major Vessels Colored by Flourosopy", [0, 1, 2, 3])
    thal_label = st.selectbox(
        "Thalassemia Status", 
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )

# -------------------- DATA ENCODING -------------------- #
# Mapping user-friendly labels back to the numeric values your model expects
sex = 1 if sex_label == "Male" else 0
fbs = 1 if fbs_label == "Yes" else 0
exang = 1 if exang_label == "Yes" else 0

cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

# -------------------- PREDICTION LOGIC -------------------- #

# CRITICAL: This array maintains the exact order from your working code
input_data = np.array([[ 
    age, sex, cp_map[cp_label], trestbps, chol, fbs, 
    restecg_map[restecg_label], thalach, exang, oldpeak, 
    slope_map[slope_label], ca, thal_map[thal_label] 
]])

st.markdown("###")
if st.button("Calculate Risk Analysis"):
    # Apply the scaler (ensure it's the one from your training)
    input_scaled = scaler.transform(input_data)
    
    # Get prediction and probability
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Diagnostic Results")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected**")
        st.write(f"The model indicates a **{probability:.2%}** probability of heart disease.")
        st.progress(probability)
    else:
        st.success(f"✅ **Low Risk / No Heart Disease Detected**")
        st.write(f"The model indicates a **{probability:.2%}** probability of heart disease.")
        st.progress(probability)

    st.info("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")
