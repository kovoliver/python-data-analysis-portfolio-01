import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import pingouin as pg
from dataclasses import dataclass
from scipy.stats.contingency import association

@dataclass
class ANOVAResults:
    f_val: float
    f_critical: float
    df_between: int
    df_within: int
    ms_between: float
    ms_within: float
    eta_squared: float
    h0_passed: bool
    significant:bool

@dataclass
class CramersResults:
    contingency_table: pd.DataFrame
    coeff:float
    h0_passed: bool
    significant:bool

def get_iqr_bounds(col, multiplier=1.5):
    Q1 = col.quantile(.25)
    Q3 = col.quantile(.75)
    IQR = Q3 - Q1
    
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    return lower, upper

def count_outliers(col, multiplier = 1.5):
    lower, upper = get_iqr_bounds(col, multiplier)

    return (col.lt(lower)|col.gt(upper)).sum()

def outlier_percentage(col, multiplier = 1.5, digits = 3):
    return round(count_outliers(col, multiplier)/col.count() * 100, digits)

def get_outlier_mask(col, multiplier=1.5):
    lower, upper = get_iqr_bounds(col, multiplier)
    return (col < lower) | (col > upper)

def remove_outliers(col, multiplier=1.5):
    lower, upper = get_iqr_bounds(col, multiplier)
    col.clip(lower=lower, upper=upper)

def trend_model(y, trend_type):
    x = np.arange(1, len(y) + 1)

    if trend_type == "logarithmic" or trend_type == "power":
        x = np.log(x)
    if trend_type == "exponential" or trend_type == "power":
        y = np.log(y)

    model = np.polyfit(x, y, deg=1)

    b = model[0]
    a = model[1]

    if trend_type == "exponential":
        b = np.exp(b)
        a = np.exp(a)
    elif trend_type == "power":
        a = np.exp(a)

    return b, a

def regression_model(df, model_type):
    model = None

    if model_type == "linear":
        model = smf.ols('y ~ x', data=df).fit()
    elif model_type == "logarithmic":
        model = smf.ols('y ~ np.log(x)', data=df).fit()
    elif model_type == "exponential":
        model = smf.ols('np.log(y) ~ x', data=df).fit()
    elif model_type == "power":
        model = smf.ols('np.log(y) ~ np.log(x)', data=df).fit()

    f_critical = stats.f.ppf(0.95, 1, len(df) - 2)

    return model, f_critical

def calculate_rsd(y, y_hat, df):
    sse = np.sum((y - y_hat) ** 2)
    return np.sqrt(sse / (len(y) - df))

def print_regression_model(model, f_critical, y, model_type):
    fitted_values = model.fittedvalues

    if model_type not in ("linear", "logarithmic"):
        fitted_values = np.exp(fitted_values)

    rsd = calculate_rsd(y, fitted_values, df=2)

    h0_passed = bool(model.fvalue <= f_critical)
    significant = not h0_passed

    intercept = model.params.iloc[0]
    slope = model.params.iloc[1]

    if model_type in ("linear", "logarithmic"):
        print(f"func: {intercept:.4f} + {slope:.4f}x")
    elif model_type == "exponential":
        print(f"func: {np.exp(intercept):.4f} * {np.exp(slope):.4f}^x")
    elif model_type == "power":
        print(f"func: {np.exp(intercept):.4f} * x^{slope:.4f}")

    print(f"R^2: {model.rsquared:.4f}")
    print(f"RSD: {rsd:.4f}")
    print(f"F value: {model.fvalue:.4f}")
    print(f"F critical: {f_critical:.4f}")
    print(f"H0 passed: {h0_passed}")
    print(f"Significant: {significant}")

def one_way_anova(df, dv, cat, alpha=0.05, detailed=True) -> ANOVAResults:
    anova_table = pg.anova(data=df, dv=dv, between=cat, detailed=detailed)

    df_between = int(anova_table.loc[0, "DF"])
    df_within = int(anova_table.loc[1, "DF"])
    f_val = float(anova_table.loc[0, "F"])
    f_critical = float(stats.f.ppf(1 - alpha, df_between, df_within))
    h0_passed=bool(f_val <= f_critical)
    significant = not h0_passed

    return ANOVAResults(
        f_val=f_val,
        f_critical=f_critical,
        df_between=df_between,
        df_within=df_within,
        ms_between=float(anova_table.loc[0, "MS"]),
        ms_within=float(anova_table.loc[1, "MS"]),
        eta_squared=float(anova_table.loc[0, "np2"]),
        significant=significant,
        h0_passed=h0_passed
    )

def cramersv(
    cat1: pd.Series, cat2: pd.Series, alpha: float = 0.05
) -> CramersResults:
    contingency_table = pd.crosstab(
        cat1, cat2, margins=True, margins_name="Total"
    )

    observed = contingency_table.iloc[:-1, :-1].values
    chi2_val, _, dof, _ = stats.chi2_contingency(observed)

    if dof == 0:
        chi2_critical = float("nan")
        h0_passed = True
        significant = False
    else:
        chi2_critical = float(stats.chi2.ppf(1 - alpha, dof))
        h0_passed = bool(chi2_val <= chi2_critical)
        significant = not h0_passed

    coeff = float(association(observed, method="cramer"))

    return CramersResults(
        contingency_table=contingency_table,
        coeff=coeff,
        significant=significant,
        h0_passed=h0_passed,
    )