import streamlit as st
import pandas as pd
import pickle

# 1️⃣ Load trained model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Charges Prediction")

# 2️⃣ Single input form
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict"):
    # 3️⃣ Create dataframe
    df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region]
    })

    # 4️⃣ One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

    # 5️⃣ Ensure all expected columns are present (manual fix)
    model_columns = [
        'age', 'bmi', 'children',
        'sex_female', 'sex_male',
        'smoker_no', 'smoker_yes',
        'region_northeast', 'region_northwest',
        'region_southeast', 'region_southwest'
    ]
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # 6️⃣ Prediction
    prediction = model.predict(df_encoded)[0]
    st.success(f"Predicted Insurance Charges: ${round(prediction, 2)}")
