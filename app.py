import streamlit as st
import joblib
import numpy as np

# Load model and scaler
try:
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("Worker Ergonomic Risk Prediction")
st.write("Predicts if a worker's posture is Safe (0) or Unsafe (1).")

# User inputs for 9 features
age = st.number_input("Age", 18, 100, 30, 1)
hours_worked = st.number_input("Hours Worked Per Day", 0.0, 24.0, 8.0, 0.1)
posture_score = st.number_input("Posture Score (1-10)", 1, 10, 5, 1)
movement_freq = st.number_input("Movement Frequency Per Hour", 0, 100, 20, 1)
breaks_taken = st.number_input("Breaks Taken Per Day", 0, 10, 2, 1)
pain_level = st.number_input("Reported Pain Level (0-10)", 0, 10, 5, 1)

# New 3 features – placeholders; replace names and ranges if known
feature_7 = st.number_input("Feature 7 (placeholder)", 0.0, 100.0, 0.0, 0.1)
feature_8 = st.number_input("Feature 8 (placeholder)", 0.0, 100.0, 0.0, 0.1)
feature_9 = st.number_input("Feature 9 (placeholder)", 0.0, 100.0, 0.0, 0.1)

# Prepare input array
input_data = np.array([[age, hours_worked, posture_score, movement_freq,
                        breaks_taken, pain_level, feature_7, feature_8, feature_9]])

# Scale and predict
try:
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("**Prediction: Safe Posture**")
    else:
        st.error("**Prediction: Unsafe Posture**")

except Exception as e:
    st.error(f"Error during prediction: {e}")

st.write("""
Note: The app assumes features are in this order:  
Age, Hours Worked, Posture Score, Movement Frequency, Breaks Taken, Pain Level, Feature 7, Feature 8, Feature 9.
""")
