import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Insurance Charges Prediction",
    layout="centered"
)

# ---------------- Load Model ----------------
model = joblib.load("insurance_model.pkl")

# ---------------- Title ----------------
st.title("💰 Insurance Charges Prediction App")
st.write("Predict medical insurance cost based on personal details")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Enter User Details")

age = st.sidebar.slider("Age", 18, 65, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# ---------------- Manual Encoding (IMPORTANT) ----------------
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],

    "sex_male": [1 if sex == "male" else 0],
    "smoker_yes": [1 if smoker == "yes" else 0],

    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0],
})

# ---------------- Prediction ----------------
if st.button("Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"💵 Estimated Insurance Charges: ₹ {prediction:,.2f}")
