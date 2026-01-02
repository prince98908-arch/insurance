import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor # Import जरूरी है

# 1. मॉडल लोड करने का सुरक्षित तरीका
@st.cache_resource # ताकि बार-बार लोड न हो
def load_model():
    try:
        return joblib.load('insurance_model.pkl')
    except:
        return None

model = load_model()

st.title("🏥 Health Insurance Cost Predictor")

if model is None:
    st.error("मॉडल फाइल 'insurance_model.pkl' नहीं मिली। कृपया चेक करें कि फाइल GitHub पर अपलोडेड है।")
else:
    # 2. इनपुट फॉर्म
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        children = st.selectbox("Children", [0,1,2,3,4,5])
    with col2:
        sex = st.selectbox("Sex", ['male', 'female'])
        smoker = st.selectbox("Smoker", ['yes', 'no'])
        region = st.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

    # 3. प्रेडिक्शन बटन
    if st.button("Predict Insurance Charges"):
        # डेटा को ठीक उसी फॉर्मेट में बनाएँ जैसा मॉडल को चाहिए
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

        try:
            # यहाँ सुनिश्चित करें कि model एक ऑब्जेक्ट है
            prediction = model.predict(input_data)
            st.success(f"### 💵 Estimated Charges: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
