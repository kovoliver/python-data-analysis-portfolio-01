# Python Data Analysis Portfolio #01 - Daily Department Performance Analysis

An end-to-end Python data analytics pipeline featuring automated data cleaning, outlier treatment, trend modeling, parametric and non-parametric statistical evaluation (ANOVA, Cramér's V), and linear/non-linear regression analysis.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Libraries:** `pandas`, `numpy`, `scipy`, `statsmodels`, `pingouin`
* **Custom Modules:** `helpers.py` (encapsulating statistical models and custom print formatters)

---

## 📂 Project Structure

```text
.
├── datasets/
│   ├── daily_department_performance.csv  # Raw dataset
│   ├── daily_department_cleaned.csv      # Cleaned dataset after outlier treatment
│   └── regional_customer_analysis.csv    # Categorical/demographic dataset
├── 01_data_preparation_daily_department.py  # Cleaning, coercion & outlier interpolation
├── 02_data_preparation_customer_analysis.py  # Categorical data prep
├── 03_ad_spend_and_revenue_trends.py         # Shape measures & non-linear trend models
├── 04_ad_spend_and_revenue_regression.py     # Linear, logarithmic, exponential & power regressions
├── 05_anova.py                              # One-way ANOVA hypothesis testing
├── 06_cramersv.py                           # Association analysis for categorical features
├── helpers.py                               # Reusable analytical functions & hypothesis test routines
└── README.md