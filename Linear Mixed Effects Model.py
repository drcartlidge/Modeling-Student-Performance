import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import chi2

# === USER SETTINGS ===
csv_path = 'your_data.csv'  # Replace with your file
outcome_var = 'outcome'  # Dependent variable name
group_var = 'subject'  # Random effects group
subset_col = 'reassigned'  # Column indicating subset group (e.g., 1 = reassigned, 0 = not)
exclude_cols = [outcome_var, group_var, subset_col]  # Exclude from explanatory vars

# === LOAD CSV AND CREATE SUBSETS ===
df = pd.read_csv(csv_path)

subset1 = df[df[subset_col] == 0]  # Not reassigned
subset2 = df[df[subset_col] == 1]  # Reassigned

# === CORRELATION MATRIX ===
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# === MODEL COMBINATIONS ===
explanatory_vars = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

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
