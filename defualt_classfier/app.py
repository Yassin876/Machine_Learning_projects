import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Default Classifier", layout="centered")
st.title("Loan Default Classification App")

@st.cache_resource
def load_model():
    model_bundle = joblib.load('default_model_full.pkl')
    return model_bundle

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']

all_columns = [
    'year', 'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
    'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'loan_amount', 'rate_of_interest',
    'Interest_rate_spread', 'Upfront_charges', 'term', 'Neg_ammortization', 'interest_only',
    'lump_sum_payment', 'property_value', 'construction_type', 'occupancy_type', 'Secured_by',
    'total_units', 'income', 'credit_type', 'Credit_Score', 'co-applicant_credit_type',
    'age', 'submission_of_application', 'LTV', 'Region', 'Security_Type', 'dtir1'
]
int_cols = {'year','loan_amount','Credit_Score'}
float_cols = {'rate_of_interest','Interest_rate_spread','Upfront_charges','term','property_value','income','LTV','dtir1'}
field_help = {
    'year': 'Year of the loan record.',
    'loan_limit': 'Loan limit group/category.',
    'Gender': 'Gender of the main applicant.',
    'approv_in_adv': 'Pre-approval status.',
    'loan_type': 'Type/category of the loan.',
    'loan_purpose': 'Purpose of the loan (e.g. home, auto).',
    'Credit_Worthiness': 'Credit-worthiness rating.',
    'open_credit': 'Is open credit present?',
    'business_or_commercial': 'Is it a business/commercial loan?',
    'loan_amount': 'The amount of the loan.',
    'rate_of_interest': 'Loan interest rate (%).',
    'Interest_rate_spread': 'Interest rate spread (difference %).',
    'Upfront_charges': 'Upfront charges (monetary value).',
    'term': 'Loan term (in months).',
    'Neg_ammortization': 'Negative amortization present?',
    'interest_only': 'Is it an interest-only loan?',
    'lump_sum_payment': 'Is lump sum payment allowed?',
    'property_value': 'Value of the property.',
    'construction_type': 'Construction type (of property).',
    'occupancy_type': 'Who occupies property?',
    'Secured_by': 'What is the security/collateral?',
    'total_units': 'Total units in property.',
    'income': 'Applicant annual income.',
    'credit_type': 'Type of applicant credit report.',
    'Credit_Score': 'Applicant credit score.',
    'co-applicant_credit_type': 'Co-applicant credit report type.',
    'age': 'Age group of the applicant.',
    'submission_of_application': 'Submission method/way.',
    'LTV': 'Loan to Value ratio.',
    'Region': 'Geographic region.',
    'Security_Type': 'Type of security/collateral.',
    'dtir1': 'Debt-to-Income Ratio.',
}
data_inputs = {}
st.header("Enter client information for all features")
for col in all_columns:
    help_text = field_help.get(col, "")
    if col in label_encoders:
        options = label_encoders[col].classes_
        value = st.selectbox(col.replace("_", " ").title(), options, help=help_text)
        data_inputs[col] = value
    elif col in int_cols:
        value = st.number_input(col.replace("_", " ").title(), value=0, step=1, format="%d", help=help_text)
        data_inputs[col] = int(value)
    elif col in float_cols:
        value = st.number_input(col.replace("_", " ").title(), value=0.0, format="%.4f", help=help_text)
        data_inputs[col] = float(value)
    else:
        value = st.number_input(col.replace("_", " ").title(), value=0.0, help=help_text)
        data_inputs[col] = value

input_df = pd.DataFrame([data_inputs])
# Label encoding
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])
# Scaling numeric columns
X_scaled = scaler.transform(input_df)

if st.button('Predict'):
    prediction = model.predict(X_scaled)[0]
    if prediction == 0:
        prediction = "Not Default"
    else:
        prediction = "Default"
    st.success(f"Predicted Status: {prediction}")
    st.info("This result is based on your inputs and the trained model.")

st.caption("Developed using Streamlit and scikit-learn.")
