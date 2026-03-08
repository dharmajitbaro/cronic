import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
# Note: Ensure these files are in your directory
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Health Assistant", layout="wide", page_icon="❤️")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("❤️ Heart Disease Risk Assessment")
st.write("Enter the patient's clinical data below to check for heart disease risk.")

# --- UI LAYOUT ---
tab1, tab2 = st.tabs(["📋 Patient Profile", "🔬 Clinical Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 45, help="Patient's age in years")
        sex = st.selectbox("Biological Sex", ["Female", "Male"])
        cp = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            help="Typical Angina: Heart-related pain. Asymptomatic: No pain."
        )
    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Is Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        restecg = st.selectbox(
            "Resting ECG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        )
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina?", ["No", "Yes"], help="Does exercise cause chest pain?")
    
    with col4:
        oldpeak = st.number_input("ST Depression (Relative to Rest)", 0.0, 10.0, 1.0, help="ST depression induced by exercise relative to rest")
        slope = st.selectbox(
            "Peak Exercise ST Segment Slope",
            ["Upsloping", "Flat", "Downsloping"]
        )
        ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0, help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox(
            "Thalassemia Status",
            ["Normal", "Fixed Defect", "Reversible Defect"]
        )

# --- ENCODING ---
# (Using your existing mapping logic)
sex_val = 1 if sex == "Male" else 0
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_val = 1 if fbs == "Yes" else 0
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_val = 1 if exang == "Yes" else 0
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

# Prepare data for prediction
input_data = np.array([[ 
    age, sex_val, cp_map[cp], trestbps, chol, fbs_val,
    restecg_map[restecg], thalach, exang_val, oldpeak,
    slope_map[slope], ca, thal_map[thal]
]])

# --- PREDICTION ---
st.markdown("---")
if st.button("🔍 Run Diagnostic Check"):
    with st.spinner('Analyzing medical data...'):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Results Display
        st.subheader("Results")
        if prediction == 1:
            st.error(f"### High Risk Detected")
            st.progress(probability)
            st.write(f"The model estimates a **{probability:.1%}%** probability of heart disease.")
            st.warning("**Note:** This is an AI-generated prediction. Please consult a cardiologist for a formal diagnosis.")
        else:
            st.success(f"### Low Risk Detected")
            st.progress(probability)
            st.write(f"The model estimates a **{probability:.1%}%** probability of heart disease.")
            st.info("Results suggest a healthy profile, but regular checkups are always recommended.")
