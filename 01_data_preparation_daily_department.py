import pandas as pd
import numpy as np
from helpers import outlier_percentage, get_outlier_mask

df = pd.read_csv(
    "./datasets/daily_department_performance.csv"
)

print("===Basic Information===")
df.info()

print("===Fundamental Statistics===")
print(df.describe())

print("===Missing Value Percentage===")
missing_percentages = df.isnull().sum() / len(df) * 100
print(missing_percentages.map(lambda x: f"{x:.2f}%"))

print("===Convert non-numerical values to strictly NaN===")
df.loc[:, "ad_spend"] = pd.to_numeric(df["ad_spend"], errors="coerce")
df.loc[:, "revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

print("===Outlier Treatment===")
# Outliers will be converted to NaN to prevent 
# interpolated values from being outliers.

target_cols = ["ad_spend", "revenue"]

for target_col in target_cols:
    print(f"***Calculate Outlier Percentage ({target_col})***")
    outliers_pct = outlier_percentage(df[target_col])
    print(f"outliers ({target_col}):", outliers_pct, "%")

    print(f"***Outlier Treatment ({target_col})***")
    df.loc[get_outlier_mask(df[target_col]), target_col] = np.nan
    df[target_col] = df[target_col].interpolate()

print("***Save the treated DataFrame to a new file***")
df.to_csv("./datasets/daily_department_cleaned.csv", index=False)