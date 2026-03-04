# Data Cleaning Process — Credit Card Approval Dataset

## Overview

This document outlines the data cleaning steps applied to `Credit_Card_Approval.csv` in preparation for model training and EDA. Two versions of the cleaned dataset were produced:

- `data/cleaned/cleaned_credit_card_approval.csv` — for model training (missing values retained)
- `data/cleaned/cleaned_credit_card_approval_1.csv` - an imputed copy of the above — for EDA only
 
---

## Raw Dataset

- **Source:** `data/raw/Credit_Card_Approval.csv`
- **Shape:** 28,897 rows × 21 columns
- **Dtypes:** All columns loaded as `object`
- **Missing values:** ~5,700 per column (~20% each) on average
- **Issues:** Inconsistent column naming — mixed casing, leading/trailing whitespace, and special characters such as `??` and `#`

---

## Cleaning Steps

### 1. Column Standardization & Type Conversion

All columns were renamed to `snake_case` and cleaned of formatting artifacts (commas, `??`, `#`, `.0`, `NG`). Four cleaning functions were applied depending on column type:

- **Numeric columns** — special characters stripped, values cast to `Int64`
- **Gender column** — uppercased and reduced to first character, `M` → `Male`, `F` → `Female`, invalid entries → `NaN`
- **Binary (owned) columns** — `0` → `No`, `1` → `Yes`, invalid entries → `NaN`
- **Categorical columns** — special characters removed, title-cased, unrecognised values standardised to `NaN`

After this step, missing values approximately doubled per column (from ~5,700 to ~11,500), as previously masked invalid entries were correctly converted to `NaN`.

---

### 2. Feature Selection

11 features were selected for downstream use:

```
applicant_gender, total_income, income_type, education_type,
family_status, job_title, applicant_age, years_of_working,
total_bad_debt, total_good_debt, status
```

---

### 3. Removing Rows with Missing Target

Rows where `status` (the target variable) was null were dropped to ensure label integrity.

---

### 4. Deduplication

```python
df.duplicated().sum()  # → 2,464 duplicates found
df = df.drop_duplicates()
```

---

### 5. Income Filtering

Rows where `total_income > 800,000` were removed. The raw maximum was ~1,600,000; values above 800k were considered extreme and likely to distort the model.

```python
df = df[df["total_income"] <= 800_000]
```

> Remaining outliers were deliberately retained, as they may represent valid rare cases in applicant financial profiles and to preserve the negative class as the dataset show extreme class imbalace of 99.6% for the posotive class and 0.4% for the negative class

**Final shape after cleaning: 10,454 rows × 11 columns**

---

### 6. Export — Model-Ready Dataset

```python
df.to_csv("data/cleaned/cleaned_credit_card_approval.csv", index=False)
```

> Missing values are **intentionally kept** at this stage to avoid data leakage. They will be handled within the model pipeline using `SimpleImputer`.

---

## EDA Dataset — Manual Imputation

A separate copy was imputed manually with the same metrics as will be used in the pipeline for EDA only. Numeric columns were filled with their median or mean, and categorical columns were filled with `"Unknown"`.

**Numeric imputation:**
- `total_income` → median
- `total_bad_debt` → median
- `total_good_debt` → median
- `applicant_age` → mean (rounded)
- `years_of_working` → mean (rounded)

**Categorical imputation:**
- `applicant_gender`, `income_type`, `education_type`, `family_status`, `job_title` → `"Unknown"`

---

## Final Summary

- **Raw data:** 28,897 rows, all columns as `object`, heavily malformatted
- **After cleaning & feature selection:** types corrected, columns renamed, 11 features retained
- **After deduplication & income filter:** **10,454 rows × 11 columns**
- **For modelling:** missing values kept, to be handled by `SimpleImputer` in pipeline
- **For EDA:** missing values manually imputed with median/mean or `"Unknown"`






### Exploratory Data Analysis — Credit Card Approval Dataset

## Overview

This documents the key findings from the exploratory data analysis (EDA) conducted on the imputed version of the cleaned dataset. The analysis covers univariate, bivariate, and multivariate perspectives to understand the data structure and feature relationships with the target variable `status` (1 = Approved, 0 = Rejected).

- **Source:** `data/cleaned/cleaned_credit_card_approval_1.csv`
- **Shape:** 10,454 rows × 11 columns
- **Dtypes:** `int64` (6 columns), `object` (5 columns)
- **Missing values:** None (imputed prior to EDA)

---

## Descriptive Statistics

Key statistics for the numeric features:

- `total_income` — mean ~193,344, std ~94,622, range 27,000–787,500
- `applicant_age` — mean ~40.6, range 22–68
- `years_of_working` — mean ~7.4, range 1–44
- `total_bad_debt` — mean ~0.21, mostly zero with rare high values
- `total_good_debt` — mean ~20.0, range 1–61
- `status` — mean ~0.996, indicating a heavily imbalanced target

---

## Univariate Analysis

### Age Distribution

The boxplot shows applicant age is tightly distributed between approximately 30 and 50, with a median around 40. A small number of outliers exist at the upper end (above 60). The distribution is fairly symmetric with no strong skew.

### Income Distribution

The histogram reveals a right-skewed distribution. The bulk of applicants earn between 100,000 and 300,000, with frequency dropping sharply beyond 400,000. The earlier income cap of 800,000 is visible at the tail. This confirms the income distribution is not normal.

### Debt Comparison

The side-by-side boxplot of `total_bad_debt` and `total_good_debt` highlights a stark contrast. Bad debt is concentrated near zero with a few extreme outliers, while good debt is more widely spread with a median around 18 and outliers reaching up to 61. Most applicants carry little to no bad debt.

### Approval Rate

The count plot of `status` confirms a severe class imbalance:

- **Approved (1):** ~99.6%
- **Rejected (0):** ~0.4%

This imbalance is a critical consideration for model training and evaluation.

---

## Bivariate Analysis

### Income vs Approval Status

Approved applicants show a slightly wider income range and higher median compared to rejected ones, though both groups overlap significantly. The rejected group has a few high-income outliers, suggesting income alone is not a strong differentiator.

### Age vs Approval Status

Rejected applicants tend to be younger (median ~35) compared to approved ones (median ~40). Approved applicants also show a tighter age spread. This suggests age may carry some signal in the approval decision.

### Job Title vs Approval

The horizontal bar chart shows that most approval counts come from high-frequency job categories — `Unknown`, `Managers`, `Core Staff`, and `Laborers` dominate. Rejection counts are minimal across all job titles, consistent with the overall class imbalance.

### Good Debt vs Approval

The crosstab bar chart shows that the vast majority of approved applicants have 0 to 1 recorded bad debts. Rejections, while rare, appear spread across higher bad debt values, suggesting bad debt may be a meaningful rejection signal.

---

## Multivariate Analysis

### Correlation Heatmap

The correlation heatmap of numeric features reveals:

- `total_bad_debt` has the strongest (negative) correlation with `status` at **-0.32**, making it the most linearly predictive numeric feature
- All other numeric features show near-zero correlation with `status`
- `applicant_age` and `years_of_working` are moderately correlated with each other at **0.2**, which is expected
- No severe multicollinearity was detected among the features

### Feature Importance (Random Forest — Initial)

A preliminary `RandomForestClassifier` was fitted  to gauge relative importance. The top 10 features ranked by importance are:

1. `total_bad_debt` (~0.27)
2. `total_income` (~0.14)
3. `total_good_debt` (~0.12)
4. `applicant_age` (~0.10)
5. `years_of_working` (~0.06)
6. `income_type_Working`
7. `family_status_Married`
8. `applicant_gender_Male`
9. `income_type_Unknown`
10. `education_type_Higher Education`

`total_bad_debt` is the dominant predictive feature by a clear margin, consistent with the correlation heatmap. Financial features (`total_income`, `total_good_debt`) rank second and third, with demographic and categorical encoded features contributing smaller but non-trivial importance.

---

## Key Takeaways

- The dataset is **heavily imbalanced** (~99.6% approvals), which must be addressed during model training
- **`total_bad_debt`** is the strongest individual predictor of rejection
- **Age** and **income** carry moderate signal; applicants who are younger or lower-earning are more likely to be rejected
- Categorical features like `income_type` and `family_status` may contribute meaningful signal after encoding
- No strong multicollinearity exists among numeric features







# Model Evaluation & Comparison — Credit Card Approval

## Overview

Three classification models were trained and evaluated on the same dataset split to predict credit card approval status. All models shared the same train/test split of 8,363 training samples and 2,091 test samples, with the target variable `status` (1 = Approved, 0 = Rejected).

Given the severe class imbalance (~99.6% positive), **PR-AUC** (Precision-Recall AUC) was used as the primary evaluation metric rather than accuracy, as accuracy alone is misleading when one class dominates.

---

## Model 1 — Logistic Regression

**Top predictive features by absolute coefficient:**
- `total_good_debt` (1.28), `total_bad_debt` (-0.65), `applicant_age` (0.25), `applicant_gender` (0.23), `education_type` (0.19)

**Evaluation results:**

- PR-AUC — Train: 0.9992 / Test: 0.9997
- Recall — Train: 0.8492 / Test: 0.8417
- F1 — Train: 0.9180 / Test: 0.9135
- CV PR-AUC: 0.9988 ± 0.0006

**Classification report (test set):**
- Positive class — Precision: 1.00, Recall: 0.84, F1: 0.91
- Negative class — Precision: 0.01, Recall: 0.71, F1: 0.03
- Overall accuracy: 0.84

**Confusion matrix:** TP: 1754 | FP: 2 | FN: 330 | TN: 5

**Observation:** Logistic Regression achieves a strong PR-AUC but struggles significantly with recall on the positive class (0.84), missing 330 approvals. Its performance on the minority negative class is essentially non-functional with a near-zero F1 of 0.03. The model generalises well (train/test metrics are close) but is too conservative for a task where missing approvals carries real cost.

---

## Model 2 — Random Forest

**Top predictive features by importance:**
- `total_bad_debt` (0.240), `total_good_debt` (0.195), `total_income` (0.115), `applicant_age` (0.095), `family_status` (0.069)

**Evaluation results:**

- PR-AUC — Train: 1.0000 / Test: 0.9987
- Recall — Train: 0.9998 / Test: 1.0000
- F1 — Train: 0.9999 / Test: 0.9990
- CV PR-AUC: 0.9990 ± 0.0005

**Classification report (test set):**
- Positive class — Precision: 1.00, Recall: 1.00, F1: 1.00
- Negative class — Precision: 1.00, Recall: 0.43, F1: 0.60
- Overall accuracy: 1.00

**Confusion matrix:** TP: 2084 | FP: 4 | FN: 0 | TN: 3

**Observation:** Random Forest achieves near-perfect recall on the positive class with zero false negatives, meaning it correctly approves every legitimate applicant in the test set. It does overfit slightly on the negative minority class (train recall 0.9998 vs test 1.00 is negligible), but the near-perfect train PR-AUC of 1.0000 versus test of 0.9987 indicates minimal generalisation gap. Overall the strongest performer across all metrics.

---

## Model 3 — XGBoost

**Top predictive features by importance:**
- `total_bad_debt` (0.219), `total_good_debt` (0.167), `applicant_age` (0.113), `family_status` (0.097), `years_of_working` (0.084)

**Evaluation results:**

- PR-AUC — Train: 1.0000 / Test: 0.9982
- Recall — Train: 1.0000 / Test: 1.0000
- F1 — Train: 0.9998 / Test: 0.9993
- CV PR-AUC: 0.9983 ± 0.0008

**Classification report (test set):**
- Positive class — Precision: 1.00, Recall: 1.00, F1: 1.00
- Negative class — Precision: 1.00, Recall: 0.57, F1: 0.73
- Overall accuracy: 1.00

**Confusion matrix:** TP: 2084 | FP: 3 | FN: 0 | TN: 4

**Observation:** XGBoost matches Random Forest on positive class recall and F1, and marginally outperforms it on minority class recall (0.57 vs 0.43) and macro F1 (0.86 vs 0.80). However, its CV PR-AUC standard deviation is slightly higher (0.0008 vs 0.0005), suggesting marginally less stable generalisation across folds.

---

## Model Comparison Summary

- **PR-AUC (test):** RF 0.9987 > XGB 0.9982 > LR 0.9997\*
- **Recall — Positive (test):** RF 1.00 = XGB 1.00 > LR 0.84
- **F1 — Positive (test):** XGB 0.9993 > RF 0.9990 > LR 0.9135
- **Recall — Negative (test):** XGB 0.57 > RF 0.43 > LR 0.71\*
- **CV PR-AUC stability:** RF ±0.0005 > LR ±0.0006 > XGB ±0.0008
- **False Negatives:** RF 0 = XGB 0 < LR 330

> \*Note: Logistic Regression's higher PR-AUC on the test set (0.9997) reflects its probability calibration rather than decision-boundary performance. Its recall of 0.84 and 330 false negatives reveal it is not competitive with the tree-based models in practical terms.

---

## Best Model — Random Forest

**Random Forest is selected as the best model** for the following reasons:

**1. Zero false negatives.** In a credit approval context, failing to approve a qualified applicant (false negative) is a significant business cost. Random Forest is the only model that achieves FN = 0 on the test set, approving every legitimate applicant correctly.

**2. Strongest and most stable CV performance.** Its cross-validated PR-AUC of 0.9990 ± 0.0005 is both the highest and the most consistent across folds, indicating reliable generalisation to unseen data.

**3. Competitive on minority class.** While XGBoost has a slightly better negative class recall (0.57 vs 0.43), the difference is small and does not outweigh Random Forest's advantage in stability and recall consistency.

**4. No meaningful overfitting.** The train-to-test gap in PR-AUC (1.0000 → 0.9987) is negligible, and the model does not show signs of memorisation.

XGBoost is a close second and may be preferred if minority class recall is prioritised more heavily or if further hyperparameter tuning is applied. Logistic Regression, while interpretable and well-calibrated, is not suitable as the primary model given its failure to recover 330 approved applicants in the test set.





# Model Interpretation — Random Forest

## Feature Importance

The model ranks features by their average contribution to decision splits across all trees. Financial features dominate:

- `total_bad_debt` — 0.240 *(strongest predictor)*
- `total_good_debt` — 0.195
- `total_income` — 0.115
- `applicant_age` — 0.095
- `family_status` — 0.069
- `years_of_working` — 0.068
- `job_title` — 0.065
- `applicant_gender` — 0.065
- `education_type` — 0.049
- `income_type` — 0.039

The top three features alone account for roughly 55% of total importance, confirming that the model's decisions are primarily financially grounded.

---

## Key Approval Drivers

**Bad debt** is the most decisive factor; any recorded bad debt strongly signals rejection. **Good debt** works in the opposite direction; a history of successfully managed debt increases approval likelihood. **Income** is the third pillar, with higher earners consistently more likely to be approved. Beyond financials, **age** and **years of working** contribute meaningfully, reflecting employment stability and credit history length. Categorical features such as family status, job title, and education type each play a smaller but non-trivial role.

---

## High-Risk Applicant Profile

An applicant likely to be rejected by the model typically presents with:

- One or more recorded bad debts
- Little to no good debt history
- Low total income (below ~100,000)
- Younger age, generally under 35
- Short employment history
- Unknown or missing profile fields (job title, income type, education)

This profile closely resembles a first-time credit applicant, young, with limited financial history, which is worth acknowledging when interpreting rejections.

---

## Fairness and Bias in Lending

Several concerns warrant attention before deploying this model in a production environment.

**Gender and age** both appear in the top 10 features. In many jurisdictions, using these as inputs in a credit decision model is legally restricted. Their presence introduces regulatory and ethical exposure even if they are not the primary drivers.

**Imputed `Unknown` values** — applied to missing categorical fields during data preparation — are treated by the model as a meaningful category. If missing data is non-random and concentrated in specific demographic groups, `Unknown` becomes an unintended proxy for those groups.

**Features like `job_title` and `income_type`** can act as indirect proxies for socioeconomic background or ethnicity, meaning the model may perpetuate structural disadvantage under the appearance of neutral, data-driven decisions.

**The class imbalance** (~99.6% approvals) means the model was trained on very few rejection examples. This limits its ability to learn balanced rejection patterns and risks encoding the historical approval behaviour of the original lender.

Before deployment, it is advisable to remove or audit gender and age as inputs, evaluate model outcomes across demographic subgroups, and apply human review to borderline decisions.

