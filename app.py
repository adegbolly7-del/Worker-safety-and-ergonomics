import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Worker Safety & Ergonomics", page_icon="🦺")

# --------------------------------------------------
# Load model & scaler
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# Load dataset & prepare features SAFELY
# --------------------------------------------------
@st.cache_data
def load_feature_columns():
    df = pd.read_csv("worker_safety_ergonomics_dataset.csv")

    # Automatically detect target-like columns
    drop_cols = []
    for col in df.columns:
        name = col.lower()
        if "posture" in name or "label" in name or "score" in name or "id" in name:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols, errors="ignore")

    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded.columns

feature_columns = load_feature_columns()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🦺 Worker Safety & Ergonomics Predictor")
st.write("Predict worker posture condition using ergonomic data")

st.divider()
st.subheader("Input Parameters")

user_input = {}
for col in feature_columns:
    user_input[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([user_input])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict"):
    try:
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]

        st.success(f"✅ Predicted Posture: **{pred}**")

        if hasattr(model, "predict_proba"):
            prob = np.max(model.predict_proba(scaled))
            st.info(f"📊 Confidence: **{prob:.2%}**")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
