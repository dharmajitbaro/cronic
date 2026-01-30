import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.write("Prediction based on UCI Heart Disease Dataset")

# -------------------- INPUTS -------------------- #

age = st.number_input("Age", 20, 100, 45)

sex = st.selectbox("Sex", ["Female", "Male"])

cp = st.selectbox(
    "Chest Pain Type (cp)",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 250, 120)

chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No", "Yes"])

restecg = st.selectbox(
    "Resting ECG (restecg)",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)

thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])

oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment (slope)",
    ["Upsloping", "Flat", "Downsloping"]
)

ca = st.selectbox("Number of Major Vessels Colored (ca)", [0, 1, 2, 3])

thal = st.selectbox(
    "Thalassemia (thal)",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)

# -------------------- ENCODING -------------------- #

sex = 1 if sex == "Male" else 0

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_map[cp]

fbs = 1 if fbs == "Yes" else 0

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_map[restecg]

exang = 1 if exang == "Yes" else 0

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
slope = slope_map[slope]

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}
thal = thal_map[thal]

# -------------------- INPUT ARRAY -------------------- #
# ⚠️ ORDER MUST MATCH TRAINING DATA

input_data = np.array([[ 
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]])

# -------------------- PREDICTION -------------------- #

if st.button("Predict Heart Disease"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected\n\nRisk Probability: {probability:.2%}")
    else:
        st.success(f"✅ No Heart Disease Detected\n\nRisk Probability: {probability:.2%}")