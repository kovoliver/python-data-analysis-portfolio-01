import pandas as pd
from helpers import one_way_anova

df = pd.read_csv("./datasets/daily_department_cleaned.csv");

print("===Association Between Department and Revenue===")
anova_results = one_way_anova(df, "revenue", "department", 0.05, True)
print("Eta squared: ", anova_results.eta_squared)
print("Is significant: ", anova_results.significant)