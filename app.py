import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. मॉडल लोड करें
model = joblib.load('insurance_model.pkl')

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("🏥 Health Insurance Cost Predictor")
st.write("कृपया नीचे अपनी जानकारी भरें ताकि हम आपके बीमा खर्च का अनुमान लगा सकें।")

# 2. यूज़र इनपुट के लिए फॉर्म बनाना
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("उम्र (Age)", min_value=1, max_value=100, value=25)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.selectbox("बच्चों की संख्या (Children)", options=[0, 1, 2, 3, 4, 5])

    with col2:
        sex = st.selectbox("लिंग (Sex)", options=['male', 'female'])
        smoker = st.selectbox("धूम्रपान (Smoker?)", options=['yes', 'no'])
        region = st.selectbox("क्षेत्र (Region)", options=['southeast', 'southwest', 'northeast', 'northwest'])

# 3. डेटा प्री-प्रोसेसिंग (मॉडल की ट्रेनिंग के हिसाब से)
def preprocess_input(age, sex, bmi, children, smoker, region):
    # 'sex' और 'smoker' को 0/1 में बदलें (जैसे हमने ट्रेनिंग के समय किया था)
    sex_val = 1 if sex == 'male' else 0
    smoker_val = 1 if smoker == 'yes' else 0
    
    # 'region' के लिए dummy variables (अगर आपने One-Hot Encoding की थी)
    # ध्यान दें: ये columns वैसे ही होने चाहिए जैसे ट्रेनिंग के समय थे
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0
    
    # डेटा को DataFrame या Array में डालें
    # क्रम: age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest
    data = {
        'age': age,
        'sex': sex_val,
        'bmi': bmi,
        'children': children,
        'smoker': smoker_val,
        'region_northwest': region_northwest,
        'region_southeast': region_southeast,
        'region_southwest': region_southwest
    }
    return pd.DataFrame([data])

# 4. प्रेडिक्शन बटन
if st.button("Predict Insurance Charges"):
    input_df = preprocess_input(age, sex, bmi, children, smoker, region)
    
    try:
        prediction = model.predict(input_df)
        st.success(f"### 💵 अनुमानित खर्च: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}. कृपया सुनिश्चित करें कि इनपुट कॉलम्स मॉडल ट्रेनिंग के समान हैं।")

st.info("नोट: यह केवल एक मशीन लर्निंग मॉडल पर आधारित अनुमान है।")
