import pandas as pd
from helpers import trend_model

df = pd.read_csv("./datasets/daily_department_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by="date", ascending=True)

print("===Measures of Shape===")
for col in ["ad_spend", "revenue"]:
    print(f"***{col}***")
    print(f"{col} skewness: {df[col].skew():.3f}")
    print(f"{col} kurtosis: {df[col].kurt():.3f}")

print()

def print_trend_models(y):
    b, a = trend_model(y, "linear")
    print(f"linear trend: f(x) = {a:.3f} + {b:.5f}x")

    b, a = trend_model(y, "logarithmic")
    print(f"logarithmic trend: f(x) = {a:.3f} + {b:.5f} * ln(x)")

    b, a = trend_model(y, "exponential")
    print(f"exponential trend: f(x) = {a:.3f} * {b:.5f}^x")

    b, a = trend_model(y, "power")
    print(f"power trend: f(x) = {a:.3f} * x^{b:.5f}\n")


print("===Trend Models===")
df_grouped_date = df.groupby("date")[["ad_spend", "revenue"]].sum()

for col in ["ad_spend", "revenue"]:
    print(f"***{col} (all departments)***")
    print_trend_models(df_grouped_date[col])