# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from pages.model_eval import kpi_card

def card_approval_dashboard():
    # page config
    st.set_page_config(
        page_title="Credit card approval dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # navigations
    hide_default_sidebar = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
        """
    st.markdown(hide_default_sidebar, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 💳 Card-Approval NG")
        st.markdown("---")
        st.markdown("**Navigation**")
        st.page_link("app.py",                label="🏠 Home",               )
        st.page_link("pages/eda_dashboard.py",           label="📊 EDA Dashboard",      )
        st.page_link("pages/card_approval_predictor.py",    label="🔮 Card Approval Predictor",       )
        st.page_link("pages/model_eval.py",       label="⚖️ Model Evaluation", )
        st.markdown("---")
        st.markdown("**Project Info**")
        st.markdown("Dataset: `credit_card_approval.csv`")
        st.markdown("Model: Regression ensemble")
        st.markdown("Version: 1.0.0") 
        
    #load datast
    df = pd.read_csv("data/cleaned/cleaned_credit_card_approval_1.csv")
    
    # header
    st.title("💳 Credit Card Approval Dashboard")
    st.caption("A strategic overview of credit card approval data")
    st.markdown("---")
    
     # Sidebar Title
    st.sidebar.markdown("## 🧭 Dashboard Filters")
    st.sidebar.markdown("Fine-tune the dashboard using the filters below.")
    
     # gender filter
    applicant_gender = st.sidebar.multiselect(
        "Gender",
        options=sorted(df["applicant_gender"].unique().tolist()),
        default=sorted(df["applicant_gender"].unique().tolist())
    )
    
     # age filter
    min_age = int(df["applicant_age"].min())
    max_age = int(df["applicant_age"].max())

    age_range = st.sidebar.slider(
        "Select Applicant Age Range",
        min_value=min_age,
        max_value=max_age,
        step=1,
        value=(min_age, max_age)
    )
    
      # job_title filter
    job_title = st.sidebar.multiselect(
        "Job Title",
        options=sorted(df["job_title"].unique().tolist()),
        default=sorted(df["job_title"].unique().tolist())
    )
      # education filter
    education_type = st.sidebar.multiselect(
        "Education Level",
        options=sorted(df["education_type"].unique().tolist()),
        default=sorted(df["education_type"].unique().tolist())
    )
    
      # job_title filter
    family_status = st.sidebar.multiselect(
        "Family Status",
        options=sorted(df["family_status"].unique().tolist()),
        default=sorted(df["family_status"].unique().tolist())
    )
    
    
    # total_income filter
    min_income = float(df["total_income"].min())
    max_income = float(df["total_income"].max())

    income_range = st.sidebar.slider(
        "Select Total Income Range",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income)
    )
    
      # apply filter
    filtered_df = df[
        (df["applicant_age"].between(age_range[0], age_range[1])) &
        (df["total_income"] >= income_range[0]) &
        (df["total_income"] <= income_range[1]) &
        (df["applicant_gender"].isin(applicant_gender)) &
        (df["job_title"].isin(job_title)) &
        (df["education_type"].isin(education_type)) &
        (df["family_status"].isin(family_status))
    ]
    
    if not filtered_df.empty:
        approval_rate = filtered_df["status"].value_counts(normalize=True) * 100
        approved_percentage = approval_rate[1]
        avg_total_income = filtered_df["total_income"].mean()
        avg_age = filtered_df["applicant_age"].mean()
        top_job = (filtered_df["job_title"].value_counts().idxmax())
    
    else:
        approved_percentage = 0
        avg_age = 0
        avg_total_income = 0
        top_job = "None"
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card(
            "✅ Approval rate",
            f"{approved_percentage:.1f}%",
            color="#3498db"
        )

    with col2:
        kpi_card(
            "💰 Average Total Income",
            f"₦{avg_total_income:,.2f}",
            color="#27ae60"
        )

    with col3:
        kpi_card(
            "👴🏿Average Age",
            f"{avg_age:.0f} years",
            color="#e74c3c"
        )
    
    with col4:
        kpi_card(
            "💼 Top Job Title",
            f"{top_job}",
            color="#9b59b6"
        )
        
    st.markdown("---")
    
     # row 1 - family status distribution and gender distribution
    left, right = st.columns(2)
    with left:
        st.subheader("Family Status Distribution")
        family_counts = (
            filtered_df["family_status"]
            .value_counts()
            .reset_index(name="count")
        )
        family_counts.columns = ["family_status", "count"]

        family_counts = family_counts.sort_values("count")
        fig_family = px.bar(
            family_counts,
            x="count",
            y="family_status",
            orientation="h",
            title="Family Status Count",
            labels={
                "count": "Number of Applicants",
                "family_status": "Family Status"
            }
        )
        fig_family.update_layout(template="plotly_white")
        st.plotly_chart(fig_family, use_container_width=True)
        
        
    with right:
        st.subheader("Gender Distribution")
        fig_gender = px.pie(
            filtered_df,
            names="applicant_gender",
            title="Pie Chart ofGender Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        fig_gender.update_layout(showlegend=False) 
        
    st.markdown("---")
    
    
    left, right = st.columns(2)
    with left:
        st.subheader("Total Income Distribution")
        fig_income = px.histogram(
            filtered_df,
            x="total_income",
            nbins=40,
            marginal="box",
            color_discrete_sequence=["skyblue"],
            title="Histogram of Total Income with Box Plot"
        )
        st.plotly_chart(fig_income, use_container_width=True)
        fig_income.update_layout(showlegend=False)
        
    with right:
        st.subheader("Correlation Matrix")
        corr_matrix = filtered_df[["status", "total_income", "applicant_age", "years_of_working", "total_bad_debt", "total_good_debt"]].corr()
        fig_corr = px.imshow(
            corr_matrix, 
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        fig_corr.update_layout(showlegend=False)
    
    
    # row 3 good debt vs bad debt and approval by education level
    left, right = st.columns(2)
    with left:
        st.subheader("Good Debt vs Bad Debt by Approval Status")
        fig = px.scatter(
            filtered_df,
            x="total_good_debt",
            y="total_bad_debt",
            color="status",
            title="Good Debt vs Bad Debt by Approval Status"
        )
        st.plotly_chart(fig, use_container_width=True)
        fig.update_layout(showlegend=False)       
     
    with right:
        st.subheader("Approval by Education Level")
        edu_status = (
            df.groupby(["education_type", "status"])
            .size()
            .reset_index(name="count")
        )
        fig_edu = px.bar(
            edu_status,
            x="count",
            y="education_type",
            color="status",
            orientation="h",
            title="Approval by Education Level"
        )
        fig_edu.update_layout( yaxis=dict(categoryorder="total ascending"), showlegend=False)
        st.plotly_chart(fig_edu, use_container_width=True)
        
    st.markdown("---")
    
     # data overview
    st.subheader("Data Preview")
    st.dataframe(filtered_df.head())
    
    # prediction system
    st.markdown("---") 
    if st.button("Click Here to Use Card Approval Predictor", type="primary", use_container_width=True):
        with st.spinner("Page loading..."):
            st.switch_page("pages/card_approval_predictor.py")
    
    
    # footer
    st.markdown("---")
    st.caption("Credit card approval EDA")
    
    
     
        
        
        
    
card_approval_dashboard()