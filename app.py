import streamlit as st
import pandas as pd
import joblib
import pickle

st.set_page_config(page_title="Insurance Charges Predictor")

st.title("💰 Insurance Charges Prediction")
st.write("Devloped by PRINCE RAJPUT")

# -----------------------------------
# Safe model loader
# -----------------------------------
model = None
error_msg = None

try:
    model = joblib.load("insurance_model.pkl")
except Exception as e1:
    try:
        with open("insurance_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e2:
        error_msg = str(e2)

if model is None:
    st.error("❌ Model load nahi ho pa raha")
    st.code(error_msg)
    st.stop()

# -----------------------------------
# User Inputs
# -----------------------------------
age = st.number_input("Age", 1, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# -----------------------------------
# Input DataFrame
# -----------------------------------
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# -----------------------------------
# Encoding
# -----------------------------------
input_encoded = pd.get_dummies(input_df, drop_first=True)

# -----------------------------------
# Column alignment (safe fallback)
# -----------------------------------
if hasattr(model, "feature_names_in_"):
    input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Charges"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💵 Predicted Insurance Charges: ₹ {prediction:,.2f}")
