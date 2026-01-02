import streamlit as st
import pandas as pd
import pickle

# 1️⃣ Load trained model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Charges Prediction")

# 2️⃣ Option for single input or batch via CSV
option = st.radio("Choose input method:", ("Single Input", "Upload CSV"))

if option == "Single Input":
    # Single input form
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict"):
        df = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        # One-hot encoding
        df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

        # Ensure all model columns present
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_columns]

        # Prediction
        prediction = model.predict(df_encoded)[0]
        st.success(f"Predicted Insurance Charges: {round(prediction, 2)}")

else:
    # Batch CSV upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data.head())

        # One-hot encoding
        df_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'])

        # Ensure all model columns present
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_columns]

        # Prediction
        predictions = model.predict(df_encoded)
        data['Predicted Charges'] = predictions
        st.write("Predictions:")
        st.dataframe(data)
        st.download_button(
            "Download Predictions CSV",
            data.to_csv(index=False).encode('utf-8'),
            file_name="predictions.csv",
            mime="text/csv"
        )
