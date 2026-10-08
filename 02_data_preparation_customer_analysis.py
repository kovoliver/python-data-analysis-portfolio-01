import pandas as pd
from helpers import cramersv

df = pd.read_csv("./datasets/regional_customer_analysis.csv")
df = df[['region', 'customer_size']]

print("===Initial Statistics===")
print(df.describe())

print("===Missing Value Percentage===")
missing_percentages = df.isnull().sum()/len(df) * 100
print(missing_percentages)