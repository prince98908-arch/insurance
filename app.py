import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Model Load karein
model = joblib.load('insurance_model.pkl')

st.set_page_config(page_title="Insurance Predictor", layout="centered")
st.title("🏥 Health Insurance Predictor")

# 2. User Inputs (Columns sequence wahi hai jo aapne bataya)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 54)
    bmi = st.number_input("BMI", 10.0, 60.0, 35.4)
    children = st.selectbox("Children", [0, 1, 2, 3, 4, 5], index=1)

with col2:
    sex = st.selectbox("Sex", ['male', 'female'])
    smoker = st.selectbox("Smoker", ['yes', 'no'], index=1)
    region = st.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

# 3. Prediction Logic
if st.button("Predict Cost"):
    # Input DataFrame (Order: age, sex, bmi, children, smoker, regions...)
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

    # Model se prediction lena (Log value aayegi)
    log_pred = model.predict(input_data)[0]

    # Log ko asli value mein convert karna (Inverse of Log is Exp)
    # Isse $8.98 seedhe $16,884 ke paas pahunch jayega
    final_cost = np.exp(log_pred)

    st.success(f"### 💵 Estimated Charges: ${final_charges:,.2f}")
    
    # Debugging ke liye (Optional)
    # st.write(f"Raw Model Output: {log_pred}")
