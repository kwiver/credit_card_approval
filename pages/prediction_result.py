import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

hide_default_sidebar = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """
st.markdown(hide_default_sidebar, unsafe_allow_html=True)

 # navigations
with st.sidebar:
    st.markdown("### 🏥 MediCost NG")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.page_link("app.py",                label="🏠 Home",               )
    st.page_link("pages/eda_dashboard.py",           label="📊 EDA Dashboard",      )
    st.page_link("pages/card_approval_predictor.py",    label="🔮 Credit Card Approval Predictor",       )
    st.page_link("pages/model_eval.py",       label="⚖️ Model Evaluation", )
    st.markdown("---")
    st.markdown("**Project Info**")
    st.markdown("Dataset: `nigeria_medical_insurance.csv`")
    st.markdown("Model: Regression ensemble")
    st.markdown("Version: 1.0.0")
    

st.title("📄 Prediction Result")
if "predicted_approval_status" not in st.session_state:
    st.warning("No prediction found. Please make a prediction first.")
    st.stop()

result = st.session_state.predicted_approval_status
prediction = result["prediction"]
approved_probability = result["approved_probability"]
declined_probability = result["declined_probability"]
risk_level = result["risk_level"]
risk_color = result["risk_color"]

applicant_age = st.session_state.applicant_age
applicant_gender = st.session_state.applicant_gender
education_type = st.session_state.education_type
family_status = st.session_state.family_staus
job_title = st.session_state.job_title
total_income = st.session_state.total_income
income_type = st.session_state.income_type
years_of_working = st.session_state.years_of_working
total_bad_debt = st.session_state.total_bad_debt
total_good_debt = st.session_state.total_good_debt

st.markdown("---")
st.subheader("Approval Prediction")
if prediction == "Approved":
    st.success(f"### ✅ Approved")
else:
    st.error(f"### ❌ Denied")
    
# risk_level


# check probability
st.subheader("Probability Score")
col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Approval Probability",
        value=f"{approved_probability:.2%}"
    )
with col2:
    st.metric(
        "Declined Probability",
        value=f"{declined_probability:.2%}"
    )

st.subheader("Risk Level Indicator")
st.markdown(f"Risk color: {risk_color} | Risk level: {risk_level}")
st.markdown("---")

st.subheader("Applicant Details") 
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"***Age:*** {applicant_age} years")
    st.markdown(f"***Gender***: {applicant_gender}")
    st.markdown(f"***Education***: {education_type}")
    st.markdown(f"***Family Status***: {family_status}")
    st.markdown(f"***Job Title***: {job_title}")
with col_b:
    st.markdown(f"***Total Income***: ₦{total_income:,.2f}")
    st.markdown(f"***Income Type***: {income_type}")
    st.markdown(f"***Years of Working***: {years_of_working} years")
    st.markdown(f"***Total Bad Debt***: {total_bad_debt}")
    st.markdown(f"***Total Good Debt***: {total_good_debt}")
    
    
# navigation
st.markdown("---")
button_col1, button_col2, button_col3, button_col4 = st.columns(4, gap="small")
with button_col1:
    if st.button("🔙 Back to Predictor", use_container_width=True):
        with st.spinner("Navigating back to predictor page..."):
            st.switch_page("pages/card_approval_predictor.py")
with button_col2:   
    if st.button("📊  Back to EDA Dashboard", use_container_width=True):
        with st.spinner("Navigating back to EDA dashboard..."):
            st.switch_page("pages/eda_dashboard.py")
with button_col3:
    if st.button("🏠 Back to Home page", use_container_width=True):
        with st.spinner("Navigating to Home page..."):
            st.switch_page("app.py")
with button_col4:
    if st.button("⚖️ Go to Model Evaluation", use_container_width=True):
        with st.spinner("Navigating to Model Evaluation page..."):
            st.switch_page("pages/model_eval.py")
    
# footer
st.markdown("---")
st.caption("Credit card approval prediction results")