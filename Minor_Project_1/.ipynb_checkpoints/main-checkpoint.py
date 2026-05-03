import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="CreditWise Predictor", layout="wide")

st.title("💳 CreditWise - Loan Approval Predictor")

st.markdown("Fill in applicant details below:")

# ------------------- BASIC NUMERIC INPUTS -------------------
st.subheader("📊 Financial & Personal Details")

col1, col2, col3 = st.columns(3)

with col1:
    applicant_income = st.number_input("Applicant Income", value=0.0)
    coapplicant_income = st.number_input("Coapplicant Income", value=0.0)
    age = st.number_input("Age", value=18.0)

with col2:
    dependents = st.number_input("Dependents", value=0.0)
    credit_score = st.number_input("Credit Score", value=300.0)
    existing_loans = st.number_input("Existing Loans", value=0.0)

with col3:
    dti_ratio = st.number_input("DTI Ratio", value=0.0)
    savings = st.number_input("Savings", value=0.0)
    collateral_value = st.number_input("Collateral Value", value=0.0)

col4, col5 = st.columns(2)

with col4:
    loan_amount = st.number_input("Loan Amount", value=0.0)

with col5:
    loan_term = st.number_input("Loan Term", value=12.0)

education_level = st.selectbox("Education Level", [0, 1, 2])  # adjust if needed

# ------------------- CATEGORICAL INPUTS -------------------
st.subheader("📋 Categorical Details")

col6, col7, col8 = st.columns(3)

with col6:
    employment = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

with col7:
    marital = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

with col8:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

employer_category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

# ------------------- ONE-HOT ENCODING -------------------
def preprocess():
    features = []

    # Numeric features (ORDER matters!)
    features.extend([
        applicant_income,
        coapplicant_income,
        age,
        dependents,
        credit_score,
        existing_loans,
        dti_ratio,
        savings,
        collateral_value,
        loan_amount,
        loan_term,
        education_level
    ])

    # Employment Status
    features.extend([
        1 if employment == "Salaried" else 0,
        1 if employment == "Self-employed" else 0,
        1 if employment == "Unemployed" else 0
    ])

    # Marital Status (only Single exists in your data)
    features.append(1 if marital == "Single" else 0)

    # Loan Purpose
    features.extend([
        1 if loan_purpose == "Car" else 0,
        1 if loan_purpose == "Education" else 0,
        1 if loan_purpose == "Home" else 0,
        1 if loan_purpose == "Personal" else 0
    ])

    # Property Area
    features.extend([
        1 if property_area == "Semiurban" else 0,
        1 if property_area == "Urban" else 0,
        1 if property_area == "Rural" else 0
    ])

    # Gender
    features.append(1 if gender == "Male" else 0)

    # Employer Category
    features.extend([
        1 if employer_category == "Government" else 0,
        1 if employer_category == "MNC" else 0,
        1 if employer_category == "Private" else 0,
        1 if employer_category == "Unemployed" else 0
    ])
   
    return np.array([features])


# ------------------- PREDICTION -------------------
if st.button("🔍 Predict Loan Approval"):
    try:
        input_data = preprocess()
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Rejected (Confidence: {prob:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Ensure preprocessing matches training exactly.")