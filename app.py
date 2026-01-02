import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# 1. डेटा लोड और मॉडल ट्रेनिंग (App के अंदर ही)
@st.cache_resource
def train_model():
    # सीधे आपके GitHub से डेटा लिंक
    url = "https://raw.githubusercontent.com/prince98908/arch-insurance/main/Health_insurance.xlsx%20-%20Health_Insurance.csv"
    df = pd.read_csv(url)
    
    # डेटा को तैयार करना (Preprocessing)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # असली मॉडल ट्रेनिंग (Object तैयार करना)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# मॉडल को एक्टिवेट करें
model = train_model()

st.set_page_config(page_title="Insurance Predictor", layout="centered")
st.title("🏥 Insurance Cost Predictor")

# 2. यूज़र इनपुट फॉर्म
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
if st.button("Predict"):
    # इनपुट डेटा को मॉडल के सीखे हुए कॉलम्स के हिसाब से सेट करना
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
    
    # यहाँ 'model' एक असली ट्रेंड ऑब्जेक्ट है, इसलिए एरर नहीं आएगा
    prediction = model.predict(input_data)
    st.success(f"### Estimated Cost: ${prediction[0]:,.2f}")
