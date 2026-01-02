import streamlit as st
import pandas as pd
import joblib

# 1. Trained Model Load karein
# Pakka kar lein ki 'insurance_model.pkl' aapke GitHub folder mein hai
model = joblib.load('insurance_model.pkl')

st.set_page_config(page_title="Insurance Predictor", layout="centered")

st.title("🏥 Health Insurance Predictor")
st.write("App devloped by PRINCE RAJPUT.")

# 2. User Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.selectbox("Children", options=[0, 1, 2, 3, 4, 5])

with col2:
    sex = st.selectbox("Sex", options=['male', 'female'])
    smoker = st.selectbox("Smoker", options=['yes', 'no'])
    region = st.selectbox("Region", options=['southeast', 'southwest', 'northeast', 'northwest'])

# 3. Prediction Button
if st.button("Predict Insurance Cost"):
    # Data ko waisa hi banayein jaisa model ko chahiye (Get Dummies format)
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0]
    })
    
    # Prediction result
    prediction = model.predict(input_data)
    
    st.success(f"### ₹ Estimated Charges: ₹{prediction[0]:,.2f}")
