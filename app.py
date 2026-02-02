import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Worker Safety & Ergonomics",
    page_icon="🦺",
    layout="centered"
)

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("posture_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# Load dataset & recreate feature structure
# --------------------------------------------------
@st.cache_data
def get_feature_template():
    df = pd.read_csv("worker_safety_ergonomics_dataset.csv")

    # Remove target & ID columns safely
    target_like = ['posture', 'label', 'score', 'id']
    drop_cols = [c for c in df.columns if any(t in c.lower() for t in target_like)]

    X = df.drop(columns=drop_cols)

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded

feature_template = get_feature_template()
feature_columns = feature_template.columns

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🦺 Worker Safety & Ergonomics Prediction")
st.write("Predict worker posture condition based on ergonomic and safety inputs.")

st.divider()
st.subheader("Input Worker Parameters")

user_input = {}

for col in feature_columns:
    user_input[col] = st.number_input(
        col,
        value=0.0,
        step=0.1
    )

input_df = pd.DataFrame([user_input])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Posture Status"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(scaled_input))
            st.success(f"✅ Prediction: **{prediction}**")
            st.info(f"📊 Confidence: **{confidence:.2%}**")
        else:
            st.success(f"✅ Prediction: **{prediction}**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))
