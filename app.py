import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load model
# ---------------------------
with open("insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💰 Insurance Charges Prediction")
st.write("App devloped by PRINCE RAJPUT")

# ---------------------------
# User Inputs
# ---------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ---------------------------
# Create input dataframe
# ---------------------------
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# ---------------------------
# Encoding (same as training)
# ---------------------------
input_encoded = pd.get_dummies(input_df, drop_first=True)

# ---------------------------
# Column Alignment (MOST IMPORTANT)
# ---------------------------
model_features = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Charges"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💵 Predicted Insurance Charges: ₹ {prediction:,.2f}")
