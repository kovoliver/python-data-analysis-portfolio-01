import pandas as pd
from helpers import cramersv

df = pd.read_csv("./datasets/regional_customer_analysis.csv")
df = df[['region', 'customer_size']]

results = cramersv(df["region"], df["customer_size"])
print("===Cramer's V Contingency Table===")
print(results.contingency_table)
print()
print("**Results**")
print("Coefficient: ", round(results.coeff, 4))
print("Is significant: ", results.significant)