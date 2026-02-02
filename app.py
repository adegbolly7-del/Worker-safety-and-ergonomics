import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Load dataset to get feature structure
# -----------------------------
@st.cache_data
def load_feature_columns():
    df = pd.read_csv("worker_safety_ergonomics_dataset.csv")

    # Columns to remove if they exist
    drop_cols = [
        'Worker_ID',
        'worker_id',
        'Posture_Score',
        'posture_score',
        'posture_label',
        'Posture_Label'
    ]

    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=existing_drop_cols)

    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded.columns

# -----------------------------
# App UI
# -----------------------------
st.title("🦺 Worker Safety & Ergonomics Predictor")
st.write("Predict posture risk level using ergonomic and safety parameters.")

st.divider()

# -----------------------------
# Input form
# -----------------------------
st.subheader("Enter Worker Data")

user_input = {}

for col in feature_columns:
    user_input[col] = st.number_input(
        label=col,
        value=0.0,
        step=0.1
    )

input_df = pd.DataFrame([user_input])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Posture Risk"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input).max()

        st.success(f"✅ Predicted Posture Label: **{prediction}**")
        st.info(f"📊 Confidence: **{prediction_proba:.2%}**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))

