import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score,
    recall_score, precision_score
)

st.set_page_config(
    page_title = "Model evaluation Page",
    page_icon = "⚖️",
    layout = "wide",
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
    

st.title("⚖️ Model Evaluation: Linear Regression Metrics")
# load dataset
df = pd.read_csv("data/cleaned/cleaned_credit_card_approval.csv")

numerical_features = [
        "total_income",
        "applicant_age",
        "years_of_working",
        "total_bad_debt",
        "total_good_debt"
    ]

categorical_features = [
    "applicant_gender",
    "education_type",
    "family_status",
    "job_title",
    "income_type"
]

target_column = "status"


X = df[numerical_features + categorical_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# load model
model = joblib.load("models/rf_model_pipeline.pkl")
    
# make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] 
 
# calculate metrics
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_proba)

def kpi_card(title, value, icon="📊", color="#2E86C1"):
    st.markdown(
        f"""
        <div style="
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 4px 12px rgba(0,0,0,0.08);
            text-align: center;
            border-left: 6px solid {color};
            ">
            <div style="font-size:20px; color:gray;">{title}</div>
            <div style="font-size:22px; font-weight:bold; margin-top:5px;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_card(
        "F1 Score",
        f"{f1:.4f}",
        color="#3498db"
    )

with col2:
    kpi_card(
        "Precision",
        f"{precision:.4f}",
        color="#27ae60"
        )

with col3:
    kpi_card(
        "Recall",
        f"{recall:.4f}",
        color="#e74c3c"
        )
    
with col4:
    kpi_card(
        "PR AUC",
        f"{pr_auc:.4f}",
        color="#9b59b6"
        )

st.markdown("---")

col1, col2 = st.columns(2)

# confusion matrix
with col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(title="Matrix of actual vs predicted labels")
    st.plotly_chart(fig, use_container_width=True)


with col2:
    st.subheader("Prediction Probability Distribution")

    fig = px.histogram(
        y_proba,
        nbins=30,
        title="Distribution of Approval Probabilities"
    )

    st.plotly_chart(fig, use_container_width=True)
    
st.markdown("---")  


# feature importace
rf_model = model.named_steps["classifier"]
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
})
importance_df["feature"] = importance_df["feature"].str.replace("cat__", "", regex=False)
importance_df["feature"] = importance_df["feature"].str.replace("num__", "", regex=False)
importance_df = importance_df.sort_values("importance", ascending=False)

fig = px.bar(
    importance_df.sort_values("importance", ascending=True).head(10),
    x="importance",
    y="feature",
    orientation="h",
    title="Impact of Features on Approval Probability"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


st.subheader("Model Interpretation")
st.info("""
    The Random Forest model primarily relies on:
    
    - Employment type
    - Family status
    - Education background

    These features are the strongest predictors of credit card approval 
    within this dataset.
    Feature importance reflects contribution strength, 
    not whether the feature increases or decreases approval probability.
""")


# footer
st.markdown("---")
st.caption("Medical insurance cost model evaluation")


