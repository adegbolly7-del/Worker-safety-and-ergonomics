import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Worker Safety & Ergonomics Predictor",
    page_icon="🦺",
    layout="centered"
)

st.title("🦺 Worker Safety & Ergonomics Prediction")
st.write("This app follows the exact preprocessing steps used during model training.")

st.divider()

# -----------------------------
# Load model & scaler safely
# -----------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("posture_model.pkl") or not os.path.exists("scaler.pkl"):
        st.error("❌ Model or scaler file not found. Make sure 'posture_model.pkl' and 'scaler.pkl' exist.")
        return None, None
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()
if model is None or scaler is None:
    st.stop()  # stop app if artifacts are missing

# -----------------------------
# Load dataset & preprocessing structure
# -----------------------------
@st.cache_data
def load_training_structure():
    if not os.path.exists("worker_safety_ergonomics_dataset.csv"):
        st.error("❌ Dataset file not found. Make sure 'worker_safety_ergonomics_dataset.csv' exists.")
        return None, None, None, None

    df = pd.read_csv("worker_safety_ergonomics_dataset.csv")

    # Columns to drop (ID / target)
    potential_drop_cols = ['Worker_ID', 'Posture_Score', 'posture_label']
    drop_cols = [c for c in potential_drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Identify categorical & numeric
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # One-hot encode categorical columns
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X, categorical_cols, numeric_cols, X_encoded.columns

X_raw, cat_cols, num_cols, final_feature_columns = load_training_structure()
if X_raw is None:
    st.stop()

# -----------------------------
# User input UI
# -----------------------------
st.subheader("Enter Worker Information")
user_data = {}

# Numeric sliders
for col in num_cols:
    min_val = float(X_raw[col].min())
    max_val = float(X_raw[col].max())
    mean_val = float(X_raw[col].mean())

    user_data[col] = st.slider(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# Categorical dropdowns
for col in cat_cols:
    options = X_raw[col].dropna().unique().tolist()
    user_data[col] = st.selectbox(col, options)

# -----------------------------
# Convert input to DataFrame
# -----------------------------
input_df = pd.DataFrame([user_data])

# One-hot encode input and align with training features
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(
    columns=final_feature_columns,
    fill_value=0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Posture Condition"):
    try:
        scaled_input = scaler.transform(input_encoded)
        prediction = model.predict(scaled_input)[0]

        st.success(f"✅ Predicted Posture Label: **{prediction}**")

        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(scaled_input))
            st.info(f"📊 Prediction Confidence: **{confidence:.2%}**")

    except Exception as e:
        st.error("❌ Prediction failed. Check your input values or model/scaler compatibility.")
        st.code(str(e))
