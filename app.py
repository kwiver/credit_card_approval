import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def home():
    # page config
    st.set_page_config(
        page_title="Home Page: Credit Card Approval Automation System ",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    hide_default_sidebar = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(hide_default_sidebar, unsafe_allow_html=True)


    # header
    st.title("💳 Credit Card Approval Automation System")
    # st.caption("")
    st.markdown("---")
    
    
    # navigations
    with st.sidebar:
        st.markdown("### 💳 Card-Approval NG")
        st.markdown("---")
        st.markdown("**Navigation**")
        st.page_link("app.py",                label="🏠 Home",               )
        st.page_link("pages/eda_dashboard.py",           label="📊 EDA Dashboard",      )
        st.page_link("pages/card_approval_predictor.py",    label="🔮 Credit Card Approval Predictor",       )
        st.page_link("pages/model_eval.py",       label="⚖️ Model Evaluation", )
        st.markdown("---")
        st.markdown("**Project Info**")
        st.markdown("Dataset: `credit_card_approval.csv`")
        st.markdown("Model: Classification Ensemble")
        st.markdown("Version: 1.0.0")
        
    
    # about and business context
    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        st.markdown("""
            About the Project
            
            ### Why Automate Credit Approval?
            
            Manual credit card approval is slow, inconsistent, and prone to human bias.
            Financial institutions lose valuable customers through delayed decisions and reject qualified
            applicants due to poorly defined underwriting criteria.
            
            This project builds an end-to-end intelligent approval system, trained on 
            real applicant records from **Credit_Card_Approval.csv**, that predicts whether 
            an applicant should be approved or rejected based on their financial and demographic profile. 
            Every prediction is traceable, measurable, and auditable.
        """)

    with col_b:
        st.markdown("""
            Business Questions Answered
            
            ### What We Solve
            
            ⚡ ***Instant decisions*** Replace multi-day manual reviews with real-time model-driven approvals.
            
            ⚖️ ***Reduce approval bias*** Eliminate subjective human judgement from the underwriting process.
            
            🛡️ ***Minimise credit risk*** Flag high-risk applicants before they default, protecting the institution.
            
            📊 ***Data-driven criteria*** Replace gut-feel thresholds with statistically validated decision rules.
        """)
        
    
    st.markdown("---")
    st.markdown("""
        Methodology
        
        ### Project Objectives
    """)

    obj_cols = st.columns(4, gap="medium")
    with obj_cols[0]:
        st.markdown("""
            🧹
            
            **Data Cleaning**
            
            Fix messy multi-source records, handle nulls, standardise formats, correct inconsistent labels, and remove duplicates.
        """)
    with obj_cols[1]:
        st.markdown("""
            📊
            
            **Exploratory Analysis**
            
            Profile applicant demographics, detect outliers, analyse approval rates by feature segment, and map risk correlations.
        """)
    with obj_cols[2]:
        st.markdown("""
            🤖
            
            **Model Building**
            
            Train Logistic Regression, Random Forest, XGBoost, and SVM. Compare via accuracy, precision, recall, F1, and ROC-AUC.
        """)
    with obj_cols[3]:
        st.markdown("""
            📋
            
            **Business Insights**
            
            Surface the strongest predictors of approval and translate findings into policy recommendations for the credit team.
        """)
        
    st.markdown("---")
    
    st.markdown("""
        Data
        
        ### Dataset Overview
        
        The source file ***Credit_Card_Approval.csv***
        contains applicant-level records from multiple data entry sources, resulting in significant quality
        issues including missing values, inconsistent encoding, and duplicate entries, all resolved in the cleaning phase.
        It was also dicovered that the data set had significant class imbalance of about 99% to 1%            
    """)
    
    data = {
        "Column": [
            "applicant_gender",
            "total_income",
            "income_type",
            "education_type",
            "family_status",
            "job_title",
            "applicant_age",
            "years_of_working",
            "total_bad_debt",
            "total_good_debt",
            "status" 
        ],
        "Description": [
            "Applicant's gender (Male/Female)",
            "Total income of the applicant (₦)",
            "Income type of the applicant",
            "Applicant's level of education",
            "Applicant's family status",
            "Job title of the applicant",
            "Age of applicant (years)",
            "Total years applicant has worked for",
            "Total bad debt incurred by applicant",
            "Total good debt incurred by applicant",
            "Applicant creddit card approval status (1/0)"
        ],
        "Type": [
            "Categorical",
            "Numeric",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Target"
        ]
    }
    st.table(data)
        
    st.markdown("---")
    st.markdown("""
        Navigation
        
        ### What's Inside This Dashboard
        
        Three purpose-built pages take you from raw data insights all the way to live predictions.
    
    """)

    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.markdown("""
            📊 **EDA Dashboard**
            
            Interactive charts exploring applicant demographics, approval rates by income/debt,
            feature distributions, and correlation heatmaps across the cleaned dataset.
        """)
        if st.button("📊  EDA Dashboard", use_container_width=True):
            with st.spinner("EDA Dashboard Page Loading..."):
                st.switch_page("pages/eda_dashboard.py")
                
    with p2:
        st.markdown("""
            🔮 **Credit Card Approval Predictor**
            
            Enter an applicant's profile through a structured form and receive an instant Approved / Rejected decision
            with a confidence score from the best-performing model.
        """)
        if st.button("🔮  Credit Card Approval Predictor", use_container_width=True):
            with st.spinner("Credit Card Approval Predictor Page Loading..."):
                st.switch_page("pages/card_approval_predictor.py")
                
    with p3:
        st.markdown("""
            📋 **Model Evaluation**
            
            Check and compare model performance across all metrics; f1 score, precision, PU-AUC, recall, inspect the confusion matrix,
            probability distribution and feature importance.
                
        """)
        if st.button("⚖️  Model Evaluation", use_container_width=True):
            with st.spinner("Model Evaluation Metrics Page Loading..."):
                st.switch_page("pages/model_eval.py")
    
    # footer
    st.markdown("---")
    st.caption("Credit card approval automation system")
        
        
home()