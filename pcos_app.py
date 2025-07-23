import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# Load the model
with open("pcos_model.pkl", "rb") as f:
    model = pickle.load(f)

# Ensure CSV exists
if not os.path.exists("pcos_results.csv"):
    df_init = pd.DataFrame(columns=["Age", "BMI", "Hair Growth", "Cycle Length", "Acne", "Weight Gain", "Prediction"])
    df_init.to_csv("pcos_results.csv", index=False)

# Page config
st.set_page_config(page_title="💊 PCOS Prediction App", layout="centered")

# Title with style
st.markdown(
    "<h2 style='text-align: center; color: #6C63FF;'>🌸 PCOS Prediction Tool</h2>",
    unsafe_allow_html=True
)
st.markdown("Welcome! Fill in the health details below to check if you're at risk for PCOS.")
st.markdown("---")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age (years)", min_value=10, max_value=60, step=1)
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=10.0, max_value=50.0, step=0.1)
    cycle_length = st.number_input("📅 Cycle Length (days)", min_value=15, max_value=45, step=1)

with col2:
    hair_growth = st.selectbox("💇 Hair Growth (Unusual facial/body hair?)", ["No", "Yes"])
    acne = st.selectbox("😣 Acne Present?", ["No", "Yes"])
    weight_gain = st.selectbox("📈 Sudden Weight Gain?", ["No", "Yes"])

# Convert inputs
hair_growth_val = 1 if hair_growth == "Yes" else 0
acne_val = 1 if acne == "Yes" else 0
weight_gain_val = 1 if weight_gain == "Yes" else 0

# Predict button
if st.button("🔍 Predict PCOS"):
    input_data = np.array([[age, bmi, hair_growth_val, cycle_length, acne_val, weight_gain_val]])
    prediction = model.predict(input_data)[0]
    
    # Save to CSV
    new_entry = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Hair Growth": [hair_growth_val],
        "Cycle Length": [cycle_length],
        "Acne": [acne_val],
        "Weight Gain": [weight_gain_val],
        "Prediction": [prediction]
    })
    new_entry.to_csv("pcos_results.csv", mode="a", header=False, index=False)

    # Show result
    st.markdown("---")
    if prediction == 1:
        st.error("🔴 The person is likely to have **PCOS**. Please consult a doctor.")
        risk_value = 80
    else:
        st.success("🟢 The person is unlikely to have PCOS.")
        risk_value = 20

    # Plotly gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_value,
        title={'text': "PCOS Risk Score"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "crimson" if prediction == 1 else "green"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgreen"},
                   {'range': [50, 100], 'color': "lightpink"}]}
    ))
    st.plotly_chart(fig)

st.markdown("---")
st.caption("📌 *This prediction is not medical advice. Consult a professional for diagnosis.*")
