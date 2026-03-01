import streamlit as st
import numpy as np
import pandas as pd
import joblib

# page config
st.set_page_config(
    page_title = "Card Approval Automation System",
    page_icon = "💳",
    layout = "centered",
    initial_sidebar_state = "expanded"
)

# navigations
hide_default_sidebar = """
<style>
    [data-testid="stSidebarNav"] {display: none;}
</style>
    """
st.markdown(hide_default_sidebar, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Credit-Card")
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
    
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #2E86C1;
        }
        .sub-text {
            font-size:18px;
            color: gray;
        }
        .prediction-card {
            padding: 25px;
            border-radius: 15px;
            background-color: #F4F6F7;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/rf_model_pipeline.pkl")
        return model
    except FileNotFoundError as e:
        st.error(f"Model artifact not found: {e}")
        st.error("Please ensure lr_model_pipeline.pkl exists inside models directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

        
@st.cache_data()
def load_dataset():
    try:
        df = pd.read_csv("data/cleaned/cleaned_credit_card_approval.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"Dataset not found {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
        
def predict_approval_staus(applicant_data, model):
    
    try:
        input_df = pd.DataFrame([applicant_data])
        predicted_approval_status = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # risk level
        risk = probability[0]
        if risk >= 0.7:
            risk_level = "High",
            risk_color = "🔴"
        elif risk >= 0.4:
            risk_level = "Medium"
            risk_color = "🟡"
        else:
            risk_level = "Low"
            risk_color = "🟢"
        
        result = {
            "prediction": "Approved" if predicted_approval_status == 1 else "Denied",
            "approved_probability": probability[1],
            "declined_probability": probability[0],
            "risk_level": risk_level,
            "risk_color": risk_color
        }
        
        return result
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()
        
def main():
    # header
    st.title("💳 Credit Card Approval Automation System")
    st.write("Fill all the fields and get immediate result.")
    
    # load model and dataset
    model = load_model()

    df = load_dataset()
           
    st.markdown("---")
    st.subheader("Enter Applicant's Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Personal Information")
        applicant_age = st.number_input(
            "Age",
            min_value=22, 
            max_value=68, 
            value=25,
            # step=1,
            help="Applicant age"
        )
        
        applicant_gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Applicant's gender"
        )
        
        education_type = st.selectbox(
            "Level of Education",
            options=["Lower Secondary", "Secondary / Secondary Special",
                     "Incomplete Higher", "Higher Education", "Academic Degree"],
            help="Applicant's level of education"
        )
        
        family_status = st.selectbox(
            "Family Satus",
            options=["Single / Not Married", "Married", "Civil Marriage", "Separated", "Widow"],
            help="Applicant's family status"
        )
        
        job_title = st.selectbox(
            "Job Title",
            options=sorted(["Managers", "Core Staff", "Sales Staff", "Accountants",
                     "Laborers", "Cleaning Staff", "Cooking Staff", "Security Staff",
                     "Medicine Staff", "High Skill Tech Staff", "Drivers",
                     "Low-Skill Laborers", "Waiters/Barmen Staff", "Secretaries",
                     "Private Service Staff", "Hr Staff", "Realty Agents", "It Staff"]),
            help="Applicant's job type"
        )
        
        
    with col2:
        st.markdown("Income and Finanacial Information")
        income_type = st.selectbox(
            "Income Type",
            options=sorted(["Working", "State Servant", "Commercial Associate", "Student",
                            "Pensioner"]),
            help="Applicant's income type"
        )
        
        total_income = st.number_input(
            "Total Income",
            min_value=27_000, 
            max_value=1_575_000, 
            value=50_000,
            step=1_000,
            help="Applicant's total income"
        )
        
        years_of_working = st.number_input(
            "Years of Working",
            min_value=1,
            max_value=44,
            value=5,
            help="Number of years the applicant has been working"
        )
        
        total_bad_debt = st.number_input(
            "Total Bad Debt Incured",
            min_value=0,
            max_value=40,
            value=0,
            help="Applicant's total bad debt"
        )
        
        total_good_debt = st.number_input(
            "Total Good Debt Incured",
            min_value=1,
            max_value=61,
            value=1,
            help="Applicant's total good debt"
        )
    
    # get result
    if st.button("Predict Credit Card Approval", type="primary", use_container_width=True):
        
        if applicant_age is None or applicant_gender is None or education_type is None or family_status is None or job_title is None or income_type is None or total_income is None or years_of_working is None:
            st.warning("⚠️ Please fill all the fields")
        else:
            applicant_data = {
                "applicant_age": applicant_age,
                "applicant_gender": applicant_gender,
                "education_type": education_type,
                "family_status": family_status,
                "job_title": job_title,
                "income_type": income_type,
                "total_income": total_income,
                "years_of_working": years_of_working,
                "total_bad_debt": total_bad_debt,
                "total_good_debt": total_good_debt
            } 
            
            with st.spinner("Predicting Approval..."):
                result = predict_approval_staus(applicant_data, model)
                # risk = calculate_risk(age, bmi, smoker)
                if result is not None:
                    
                    # Save prediction
                    st.session_state.predicted_approval_status = result
                    
                    # save risk
                    # st.session_state.calculate_risk = risk

                    # Save patient data 
                    st.session_state.applicant_age = applicant_age
                    st.session_state.applicant_gender = applicant_gender
                    st.session_state.education_type = education_type
                    st.session_state.family_staus = family_status
                    st.session_state.job_title = job_title
                    st.session_state.income_type = income_type
                    st.session_state.total_income = total_income
                    st.session_state.years_of_working = years_of_working
                    st.session_state.total_bad_debt = total_bad_debt
                    st.session_state.total_good_debt = total_good_debt
                    
                # Switch page
                st.switch_page("pages/prediction_result.py")
              
    # footer
    st.markdown("---")
    st.caption("Credit card approval automation system")
        
if __name__  ==  "__main__":
    main()


    
 
  
