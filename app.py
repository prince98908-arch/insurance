import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Insurance Charges Prediction",
    layout="centered"
)

# Load trained model (pipeline)
model = joblib.load("insurance_model.pkl")

# App title
st.title("💰 Insurance Charges Prediction App")
st.write("Predict medical insurance cost based on personal details")

# Sidebar inputs
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

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Prediction button
if st.button("Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]

    st.success(f"💵 Estimated Insurance Charges: ₹ {prediction:,.2f}")

# Footer
st.markdown("---")
st.markdown("📌 **Model**: Gradient Boosting Regressor")
st.markdown("📊 **Metric**: R² ≈ 0.84")
