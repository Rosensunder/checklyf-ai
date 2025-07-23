# File: data/pcos_app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
with open("data/pcos_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="🩺 PCOS Predictor", page_icon="💊", layout="centered")

st.title("🌸 PCOS Prediction Tool")
st.markdown("Use this tool to predict if a person may have PCOS based on medical features.")

# Input fields
age = st.slider("🎂 Age (years)", 15, 45, 25)
bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=22.5)
weight = st.number_input("🏋️ Weight (kg)", min_value=30.0, max_value=120.0, value=60.0)
pulse = st.number_input("💓 Pulse rate (bpm)", min_value=50, max_value=150, value=80)
cycle_length = st.slider("🩸 Cycle Length (days)", 15, 40, 28)
hair_growth = st.selectbox("🧔 Facial Hair Growth?", ["No", "Yes"])
skin_darkening = st.selectbox("🌑 Skin Darkening?", ["No", "Yes"])
acne = st.selectbox("😣 Acne?", ["No", "Yes"])
fast_food = st.selectbox("🍔 Fast Food Intake?", ["No", "Yes"])
stress = st.slider("😰 Stress Level (1 to 10)", 1, 10, 5)

# Prepare input
input_data = pd.DataFrame([[
    age,
    bmi,
    weight,
    pulse,
    cycle_length,
    1 if hair_growth == "Yes" else 0,
    1 if skin_darkening == "Yes" else 0,
    1 if acne == "Yes" else 0,
    1 if fast_food == "Yes" else 0,
    stress
]], columns=[
    "Age", "BMI", "Weight", "Pulse Rate", "Cycle Length", 
    "Facial Hair", "Skin Darkening", "Acne", "Fast Food", "Stress Level"
])

# Predict
if st.button("🔍 Predict PCOS"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ You may be at risk of PCOS. (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of PCOS detected. (Confidence: {1 - probability:.2f})")

    # Visualization
    st.markdown("### 📊 Your Input Overview:")
    fig, ax = plt.subplots()
    input_data.iloc[0].plot(kind='bar', ax=ax, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
