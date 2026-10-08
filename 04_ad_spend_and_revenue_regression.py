import pandas as pd
from helpers import regression_model, print_regression_model

df = pd.read_csv(
    "./datasets/daily_department_cleaned.csv",
    parse_dates=["date"]
)

df = df.sort_values(by="date", ascending=True)
departments = df["department"].unique()
model_types = ["linear", "logarithmic", "exponential", "power"]

df_regression = (
    df[["department", "date", "ad_spend", "revenue"]]
    .rename(columns={"ad_spend": "x", "revenue": "y"})
)

df_grouped_date = df_regression.groupby("date")[["x", "y"]].sum()
y_all = df_grouped_date['y']

print("===Regression Models All Departments===")
for model_type in model_types:
    print(f"***{model_type}***")
    model, f_critical = regression_model(df_grouped_date, model_type)
    print_regression_model(model, f_critical, y_all, model_type)
    print()

print("===Regression Models By Departments===")
for department in departments:
    print(f"***{department} Department Regression Models***")
    df_department = df_regression[df_regression["department"] == department]
    y_dept = df_department['y']
    
    for model_type in model_types:
        print(f"**{model_type}**")
        model, f_critical = regression_model(df_department, model_type)
        print_regression_model(model, f_critical, y_dept, model_type)
        print()