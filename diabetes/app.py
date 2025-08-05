import pandas as pd
import joblib
import streamlit as st
from xgboost import XGBClassifier

# Load the pre-trained model
model = joblib.load('model.pkl')
# Load the scaler
scaler = joblib.load('scaler.pkl')

st.title('diabetes Prediction')
# Input fields for user data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=300, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=30)
# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],  
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree_function],
    'Age': [age]
})
# Add a button to clear the input fields
if st.button('predict'):
# Scale the input data
    scaled_data = scaler.transform(input_data)
# Make prediction
    prediction = model.predict(scaled_data)
# Display the prediction result
    if prediction[0] == 1:
        st.warning('The model predicts that you have diabetes.')
    else:
        st.warning('The model predicts that you do not have diabetes.')
