import streamlit as st
import pandas as pd
import joblib

# 1. Model load karein
model = joblib.load('insurance_model.pkl')

st.title("Insurance Prediction")

# 2. Inputs
age = st.number_input("Age", 18, 100, 25)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.selectbox("Children", [0,1,2,3,4,5])
sex = st.selectbox("Sex", ['male', 'female'])
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

# 3. Predict Button
if st.button("Predict"):
    # Data convert karein
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
    
    # Prediction
    prediction = model.predict(input_data)
    st.success(f"Charges: {prediction[0]}")
