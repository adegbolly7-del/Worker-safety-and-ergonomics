import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
try:
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'posture_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# App title
st.title("Worker Ergonomic Risk Prediction")

# Description
st.write("""
This app predicts if a worker's posture is safe or unsafe based on input features.
The prediction is binary: 0 (Safe) or 1 (Unsafe), where Unsafe is determined if Posture Score <= 4.
""")

# Input fields for the features
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
hours_worked = st.number_input("Hours Worked Per Day", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
posture_score = st.number_input("Posture Score (1-10)", min_value=1, max_value=10, value=5, step=1)
movement_freq = st.number_input("Movement Frequency Per Hour", min_value=0, max_value=100, value=20, step=1)
breaks_taken = st.number_input("Breaks Taken Per Day", min_value=0, max_value=10, value=2, step=1)
pain_level = st.number_input("Reported Pain Level (0-10)", min_value=0, max_value=10, value=5, step=1)

# Prediction button
if st.button("Predict Risk Level"):
    # Prepare input data
    input_data = np.array([[age, hours_worked, posture_score, movement_freq, breaks_taken, pain_level]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display result
    if prediction == 0:
        st.success("**Prediction: Safe Posture**")
    else:
        st.error("**Prediction: Unsafe Posture**")

# Additional info
st.write("Note: This app assumes the model was trained on the provided dataset with features in the order: Age, Hours Worked Per Day, Posture Score, Movement Frequency Per Hour, Breaks Taken Per Day, Reported Pain Level.")
