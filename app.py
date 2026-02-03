import streamlit as st
import joblib
import numpy as np

# Load model and scaler safely
try:
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'posture_model.pkl' and 'scaler.pkl' are in the app folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# App title
st.title("Worker Ergonomic Risk Prediction")

# Description
st.write("""
This app predicts if a worker's posture is safe or unsafe based on input features.
The prediction is binary: 0 (Safe) or 1 (Unsafe).
""")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
hours_worked = st.number_input("Hours Worked Per Day", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
posture_score = st.number_input("Posture Score (1-10)", min_value=1, max_value=10, value=5, step=1)
movement_freq = st.number_input("Movement Frequency Per Hour", min_value=0, max_value=100, value=20, step=1)
breaks_taken = st.number_input("Breaks Taken Per Day", min_value=0, max_value=10, value=2, step=1)
pain_level = st.number_input("Reported Pain Level (0-10)", min_value=0, max_value=10, value=5, step=1)

# Prepare input
input_data = np.array([[age, hours_worked, posture_score, movement_freq, breaks_taken, pain_level]])

# Check input features against scaler
expected_features = getattr(scaler, "n_features_in_", None)
if expected_features is not None and input_data.shape[1] != expected_features:
    st.error(f"Feature mismatch! Scaler expects {expected_features} features, but got {input_data.shape[1]}.")
    st.stop()

# Prediction button
if st.button("Predict Risk Level"):
    try:
        # Scale input
        input_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        # Show result
        if prediction == 0:
            st.success("**Prediction: Safe Posture**")
        else:
            st.error("**Prediction: Unsafe Posture**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Note
st.write("Ensure that the features match the training data order: Age, Hours Worked, Posture Score, Movement Frequency, Breaks Taken, Pain Level.")
