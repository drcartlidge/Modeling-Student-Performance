import pandas as pd
import itertools
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import chi2
import numpy as np

# === LOAD CSV AND GENERATE FEATURES FOR USER SETTINGS ===
csv_path = r'C:\Users\cartlimr\OneDrive - Prince William County Public Schools\Vaughan Elementary Case Study - 2025\Data from Christina\Second Pull\Vaughn Query_Student Dataset_20250530.csv'  # Replace with your file
df = pd.read_csv(csv_path)
df['school_classroom'] = df['School_2021'].astype(str) + '_' + df['Teacher_2021'].astype(str)  # group_var
# Identify reassigned students
df['reassigned'] = np.where(df['School_1819'] != df['School_2021'], 1, 0)
# Generate dummy variable for SPED
df['spedDummy'] = np.where(df['SPED'] == 'Y', 1, 0)
# Generate dummy variable for Gifted
df['giftedDummy'] = np.where(df['Gifted Recode'] == 'Y', 1, 0)

# === ONE-HOT ENCODING FOR CATEGORICAL VARIABLES ===
#categorical_features = ['Gender', '1819 EL', '2021 EL', 'LunchStatus', 'Race Recode', 'HomeLanguage']
categorical_features = ['Gender', 'EL_1819', 'EL_2021', 'LunchStatus', 'Race Recode']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)


# === DATASET CLEANSING ===
# Replace all spaces in column names with underscores
df.columns = df.columns.str.replace(' ', '_')
# Drop students not enrolled in 2021
df = df.dropna(subset=['School_2021'])
# Select only 3rd graders from 2018-19
df = df[(df['Grade_1819'] == 3)]
# Select only Vaughan ES students
df = df[df['School_2021'] == 'Vaughan ES']



# === USER SETTINGS ===
outcome_var = 'Math_2021'  # Dependent variable name
group_var = 'school_classroom'  # Random effects group
subset_col = 'reassigned'  # Column indicating subset group (e.g., 1 = reassigned, 0 = not)
exclude_cols = [outcome_var, group_var, subset_col, categorical_features, 'School_1819', 'School_2021',
       'Teacher_1819', 'Grade_1819', 'Teacher_2021', 'Grade_2021', 'SPED',
       'Gifted_Recode', 'HomeLanguage']  # Exclude from explanatory vars


# === CREATE SUBSETS ===
subset1 = df[df[subset_col] == 0]  # Not reassigned
subset2 = df[df[subset_col] == 1]  # Reassigned

# === CORRELATION MATRIX ===
corr = df.corr(numeric_only=True)
plt.figure(figsize=(50, 30))
heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", cbar_kws={'label': 'Correlation'}, annot_kws={"size": 20})
# Extract color bar
cbar = heatmap.collections[0].colorbar
# Increase the size of the color bar label
cbar.set_label('Correlation', size=30)
# Increase the size of the color bar tick labels
cbar.ax.tick_params(labelsize=20)
# Increase the size of the labels on the x-axis and y-axis:
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Correlation Matrix - Enrolled 5th Graders at Vaughan ES (SY2020-21)', fontsize=50)
plt.tight_layout()
# Save the plot as an image file
plt.savefig("plot.png")


# === MODEL COMBINATIONS ===
explanatory_vars = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

# subset only first 3 explanatory variables from explanatory_vars
#explanatory_vars = explanatory_vars[:3]


results = []

for i in range(1, len(explanatory_vars) + 1):
    for combo in itertools.combinations(explanatory_vars, i):
        fixed_formula = outcome_var + " ~ " + " + ".join(combo)
        try:
            model = smf.mixedlm(fixed_formula, data=df, groups=df[group_var])
            result = model.fit()
            results.append({
                'variables': combo,
                'aic': result.aic
            })
        except Exception as e:
            print(f"Skipping combo {combo} due to error: {e}")



# === RANKED OUTPUT ===
results = sorted(results, key=lambda x: x['aic'])

print("\n=== Top Models by AIC ===")
for i, r in enumerate(results[:10]):  # Show top 10
    print(f"\nModel {i + 1}: AIC: {r['aic']:.2f} | Variables: {r['variables']}")

    # === MULTICOLLINEARITY CHECK: VIF ===
    # VIF (Variance Inflation Factor) shows how much a variable is inflated due to multicollinearity.
    # Interpretation guide:
    # - VIF ≈ 1: No multicollinearity
    # - VIF 1–5: Low to moderate multicollinearity (generally acceptable)
    # - VIF 5–10: Moderate to high (review carefully)
    # - VIF > 10: High multicollinearity — consider dropping or combining variables

    X = df[list(r['variables'])].dropna()
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])]

    print("VIF:")
    print(vif_data.to_string(index=False))

# === LIKELIHOOD RATIO TEST ===
print("\n=== Likelihood Ratio Test: Subset Comparison ===")

# Use the best variable combination
best_vars = list(results[0]['variables'])

# 1. Combined Model with interaction terms
interaction_terms = ' + '.join([f'{var}*{subset_col}' for var in best_vars])
combined_formula = f"{outcome_var} ~ {interaction_terms}"

try:
    combined_model = smf.mixedlm(combined_formula, data=df, groups=df[group_var])
    combined_result = combined_model.fit()
    ll_combined = combined_result.llf
except Exception as e:
    print(f"Error fitting combined model: {e}")
    ll_combined = None

# 2. Separate models for subset1 and subset2
try:
    model1 = smf.mixedlm(f"{outcome_var} ~ {' + '.join(best_vars)}", data=subset1, groups=subset1[group_var])
    result1 = model1.fit()
    ll1 = result1.llf
except Exception as e:
    print(f"Error fitting model for subset 1: {e}")
    ll1 = None

try:
    model2 = smf.mixedlm(f"{outcome_var} ~ {' + '.join(best_vars)}", data=subset2, groups=subset2[group_var])
    result2 = model2.fit()
    ll2 = result2.llf
except Exception as e:
    print(f"Error fitting model for subset 2: {e}")
    ll2 = None

# 3. Likelihood Ratio Test
if ll_combined is not None and ll1 is not None and ll2 is not None:
    ll_separate = ll1 + ll2
    lr_stat = 2 * (ll_separate - ll_combined)
    df_diff = (result1.df_modelwc + result2.df_modelwc) - combined_result.df_modelwc
    p_value = chi2.sf(lr_stat, df_diff)

    print(f"\nLikelihood Ratio Test Statistic: {lr_stat:.2f}")
    print(f"Degrees of Freedom: {df_diff}")
    print(f"P-Value: {p_value:.4f}")

    if p_value < 0.05:
        print("→ Reject null hypothesis: model coefficients differ significantly between subsets.")
    else:
        print("→ Fail to reject null hypothesis: no significant difference between subsets.")
else:
    print("Likelihood ratio test could not be completed due to model fitting errors.")
