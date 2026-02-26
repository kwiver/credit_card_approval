# 🧹 Credit Card Approval Dataset – Data Cleaning Report

## 📌 Project Phase: Data Cleaning

This phase focused on transforming a highly inconsistent and intentionally messy dataset into a structured format suitable for building a Credit Card Approval Prediction Model.

---

# 1️⃣ Initial Data Assessment

### Dataset Overview

- **Total Rows:** 28,897  
- **Total Columns:** 21  
- **Initial Data Types:** All columns stored as `object`

### Initial Issues Identified

The dataset was intentionally messy and contained:

- Inconsistent column names (extra spaces, special characters like `??`)
- All numeric fields stored as strings
- Special characters in values (`#`, `??`, `NG`, `.0`, commas)
- Mixed numeric and text values in the same column
- Large volume of missing values
- Duplicate records
- Potential numeric outliers

---

# 2️⃣ Column Name Standardization

Column names were cleaned and standardized using dictionary-based renaming.

### Example Transformations

| Raw Column Name               Cleaned Column Name 

 ` Applicant_ID  `            - `applicant_id` 
 `TOTAL_INCOME??`             - `total_income` 
 `YEARS_OF_WORKING??`         - `years_of_working` 
 ` Status  `                  - `status` 

### Actions Taken

- Removed extra spaces
- Removed special characters
- Converted to lowercase
- Applied `snake_case` formatting

✔ Result: Clean, consistent, machine-readable column names.

---

# 3️⃣ Numeric Column Cleaning

Numeric columns contained:

- Commas (e.g., `270000,`)
- `.0` suffixes
- `NG`
- `??`
- `#`
- `unknown`
- Mixed string and numeric formats

### Cleaning Strategy

A custom cleaning function was applied to:

- Remove unwanted characters
- Strip whitespace
- Convert valid values to integers
- Replace invalid entries with `NaN`
- Cast columns to nullable integer type (`Int64`)

### Numeric Columns Cleaned

- `applicant_id`
- `total_children`
- `total_income`
- `total_family`
- `applicant_age`
- `years_of_working`
- `total_bad_debt`
- `total_good_debt`
- `status`

All numeric columns successfully converted from `object` to `Int64`.

---

# 4️⃣ Gender Column Standardization

The gender column contained inconsistent entries such as:

- `F`, `M`
- `unknown`
- `?`
- Blank spaces

### Cleaning Steps

- Converted values to string
- Trimmed whitespace
- Extracted first character
- Converted to uppercase
- Mapped values as follows:

 `M` - Male 
 `F` - Female 
 `U`, `N`, `?` - NaN 

Gender standardized to two clean categories — `Male` and `Female`.

---

# 5️⃣ Ownership Columns Cleaning

Binary ownership-related columns contained noisy values like:

- `0#`
- `1.0`
- `0 NG`
- `??`
- `unknown`

### Columns Cleaned

- `owned_car`
- `owned_realty`
- `owned_mobile_phone`
- `owned_work_phone`
- `owned_phone`
- `owned_email`

Values were standardized to:

- `Yes`
- `No`
- `NaN` (invalid/missing entries)

---

# 6️⃣ Categorical Column Cleaning

Categorical columns contained:

- `#`
- `??`
- `.0`
- `NG`
- `unknown`
- Empty strings
- Inconsistent capitalization

### Cleaning Steps

- Removed unwanted symbols using regex
- Trimmed whitespace
- Applied title-case formatting
- Replaced invalid values with `NaN`

### Columns Cleaned

- `income_type`
- `education_type`
- `family_status`
- `housing_type`
- `job_title`

Clean and standardized categorical features.

---

# 7️⃣ Missing Values Analysis (Post-Cleaning)

After converting invalid values to `NaN`, significant missing values were observed.

Missing values were intentionally preserved for proper handling during the modeling stage, making use of the SimpleImputer to handle it.

---

# 8️⃣ Feature Selection

The dataset was reduced to relevant modeling features:
- applicant_gender
- total_income
- income_type
- education_type
- family_status
- ob_title
- applicant_age
- years_of_working
- total_bad_debt
- total_good_debt
- status

Irrelevant or redundant columns were removed.

---

# 9️⃣ Target Variable Cleaning

The `status` column represents credit card approval outcome, which is the target

### Actions Taken

- Cleaned and converted to numeric
- Removed rows where `status` was missing

## 🔟 Duplicate Detection and Removal

During data validation, duplicate records were identified.

- **Total Duplicates Detected:** 2,464 rows  

These duplicate records were removed to ensure:

- Prevention of **data leakage**
- Reduction of **model bias**
- Avoidance of **inflated performance metrics**
- Improved overall dataset integrity

---

## 1️⃣1️⃣ Outlier Detection

Outliers were identified using the **Interquartile Range (IQR)** method.

### 📊 Outlier Counts

 Column               Outliers 

 total_income         488      
 applicant_age        0        
 years_of_working     597      
 total_bad_debt       1,275    
 total_good_debt      0        

---

## 🔎 Outlier Handling Strategy

Outliers were identified in the following columns:

 Column               Number of Outliers

 total_income         488              
 applicant_age        0                
 years_of_working     597              
 total_bad_debt       1,275            
 total_good_debt      0                

Outliers were primarily concentrated in financial variables such as income and debt-related features.

---

## Decision: Retain Outliers

Instead of removing or capping extreme values, the outliers were retained.

### 🔎 Rationale

In credit card approval modeling:

- Extreme income values may represent legitimate high-income applicants.
- High debt values may reflect genuine financial risk behavior.
- Removing or altering these values could introduce bias.
- Tree-based machine learning models are robust to extreme values.
- Preserving raw financial variance improves model realism.

### Final dataset:
contained 17452 rows and 11 coulms representing the most crucial features in the determination of credit card approval.



