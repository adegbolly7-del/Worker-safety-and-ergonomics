import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Worker Safety & Ergonomics Predictor",
    page_icon="🦺",
    layout="centered"
)

# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# Load dataset & rebuild preprocessing (EXACT)
# --------------------------------------------------
@st.cache_data
def load_training_structure():
    df = pd.read_csv("worker_safety_ergonomics_dataset.csv")

    # Remove target / ID columns safely (same idea as notebook)
    drop_cols = [
        c for c in df.columns
        if any(x in c.lower() for x in ["posture", "label", "score", "id"])
    ]

    X = df.drop(columns=drop_cols, errors="ignore")

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X, categorical_cols, numeric_cols, X_encoded.columns

X_raw, cat_cols, num_cols, final_feature_columns = load_training_structure()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🦺 Worker Safety & Ergonomics Prediction")
st.write("This app follows the exact preprocessing steps used during model training.")

st.divider()
st.subheader("Enter Worker Information")

user_data = {}

# -------- Numeric Inputs → Sliders
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

# -------- Categorical Inputs → Dropdowns
for col in cat_cols:
    options = X_raw[col].dropna().unique().tolist()
    user_data[col] = st.selectbox(col, options)

# --------------------------------------------------
# Convert input to DataFrame
# --------------------------------------------------
input_df = pd.DataFrame([user_data])

# Apply SAME encoding as training
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Align with training feature space
input_encoded = input_encoded.reindex(
    columns=final_feature_columns,
    fill_value=0
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Posture Condition"):
    try:
        scaled_input = scaler.transform(input_encoded)
        prediction = model.predict(scaled_input)[0]

        st.success(f"✅ Predicted Posture Label: **{prediction}**")

        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(scaled_input))
            st.info(f"📊 Prediction Confidence: **{confidence:.2%}**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))
