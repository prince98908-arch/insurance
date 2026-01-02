import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Model Load karein
model = joblib.load('insurance_model.pkl')

st.title("🏥 Insurance Price Predictor")

# 2. User Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 25)
    bmi = st.number_input("BMI", 10.0, 50.0, 28.5)
    children = st.selectbox("Children", [0,1,2,3,4,5])

with col2:
    sex = st.selectbox("Sex", ['male', 'female'])
    smoker = st.selectbox("Smoker", ['yes', 'no'])
    region = st.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

# 3. Predict Button
if st.button("Predict Charges"):
    # Data Format (Wahi order jo training mein tha)
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
    
    # Model Prediction (Jo Log format mein hai)
    prediction = model.predict(input_data)[0]
    
    # Log value ko asli value mein badalne ke liye Exponent (e^x) lena
    # Isse $8.98 wapas $7,900+ ya $16,000+ ban jayega
    final_charges = np.exp(prediction)
    
    st.success(f"### 💵 Estimated Charges: ${final_charges:,.2f}")
