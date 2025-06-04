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
from collections.abc import Iterable
from patsy import dmatrix
import pickle
import scipy.stats as stats

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

# === DATASET CLEANING ===
# Replace all spaces in column names with underscores
df.columns = df.columns.str.replace(' ', '_')
# Replace NaN values in 'EL_1819' with 'Not ELL'
df['EL_1819'] = df['EL_1819'].fillna('Not ELL')
# Replace NaN values in 'EL_2021' with 'Not ELL'
df['EL_2021'] = df['EL_2021'].fillna('Not ELL')

# Drop students not enrolled in 2021
df = df.dropna(subset=['School_2021'])

# === Study Subset ===
# Select grade criteria from 'Grade_1819'
df = df[df['Grade_1819'].isin([3])]

# Select only Vaughan ES students
df = df[df['School_2021'] == 'Vaughan ES']
# Reclassify various Level 6 categories in 'EL_1819' into only 'Level 6'
df['EL_1819'] = df['EL_1819'].replace(['Level 6- 1st Yr', 'Level 6- 2nd Yr', 'Level 6- 3rd Yr', 'Level 6- 4th Yr'], 'Level 6 - All Years')
# Reclassify various Level 6 categories in 'EL_2021' into only 'Level 6'
df['EL_2021'] = df['EL_2021'].replace(['Level 6- 1st Yr', 'Level 6- 2nd Yr', 'Level 6- 3rd Yr', 'Level 6- 4th Yr'], 'Level 6 - All Years')

# === ONE-HOT ENCODING FOR CATEGORICAL VARIABLES ===
#categorical_features = ['Gender', '1819 EL', '2021 EL', 'LunchStatus', 'Race Recode', 'HomeLanguage']
categorical_features = ['Gender', 'EL_1819', 'EL_2021', 'LunchStatus', 'Race_Recode']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# === USER SETTINGS ===
outcome_var = 'Math_2021'  # Dependent variable name
group_var = 'school_classroom'  # Random effects group
subset_col = 'reassigned'  # Column indicating subset group (e.g., 1 = reassigned, 0 = not)
exclude_cols = [outcome_var, group_var, subset_col, categorical_features, 'StudentID', 'LastName', 'FirstName', 'School_1819', 'School_2021',
       'Teacher_1819', 'Grade_1819', 'Teacher_2021', 'Grade_2021', 'SPED',
       'Gifted_Recode', 'HomeLanguage', 'Science_1819',
                'Membership_Days_1819', 'Total_Absence_Days_1819', 'Per_Attendance_1819',
                 'EL_1819_Level_2', 'EL_1819_Level_3', 'EL_1819_Level_4', 'EL_1819_Level_6___All_Years',
                'EL_1819_Not_ELL']  # Exclude from explanatory vars

# Create function for nested list within the list of exclude_cols
def flatten_exclude_cols(exclude_cols):
    flat_list = []
    for item in exclude_cols:
        if isinstance(item, Iterable) and not isinstance(item, str):
            flat_list.extend(item)
        else:
            flat_list.append(item)
    return flat_list

# Example usage:
exclude_cols = flatten_exclude_cols(exclude_cols)

# Manipulate dataframe for modeling - drop fields where all NANs or containing datafields not needed
df = df.drop(columns=['StudentID', 'LastName', 'FirstName', 'School_1819', 'School_2021',
       'Grade_1819', 'Teacher_1819', 'Teacher_2021', 'Grade_2021', 'HomeLanguage', 'SPED', 'Gifted_Recode',
                      'Science_1819', 'Per_Attendance_1819', 'Membership_Days_1819', 'Total_Absence_Days_1819'])

# === DATASET CLEANING ===
# Replace all spaces in column names with underscores
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('-', '_')

# Drop missing data
df = df.dropna()
# Reset index
df = df.reset_index(drop=True)
# Identify missing data
missing_data = df.isnull().sum()
print(missing_data)

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
plt.savefig("correlation_matrix_vaughan_5th_graders.png")

# === MODEL COMBINATIONS ===
explanatory_vars = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

results = []
print(f"Number of explanatory variables considered for model: {len(explanatory_vars)}")

# === LINEAR MIXED EFFECTS MODELING ===
# Calculate Marginal and Conditional R-squared
def calculate_r_squared(result):
    # Variance of fixed effects
    var_fixed = result.fittedvalues.var()

    # Variance of random effects
    random_effects = result.random_effects.values()
    var_random = pd.Series([re.var() for re in random_effects]).sum()

    # Residual variance
    var_residual = result.resid.var()

    # Marginal R-squared
    marginal_r2 = var_fixed / (var_fixed + var_random + var_residual)

    # Conditional R-squared
    conditional_r2 = (var_fixed + var_random) / (var_fixed + var_random + var_residual)

    return marginal_r2, conditional_r2

# Calculate total number of combinations
total_combos = sum(1 for i in range(1, len(explanatory_vars) + 1) for _ in itertools.combinations(explanatory_vars, i))
tested_combos = 0

for i in range(1, len(explanatory_vars) + 1):
    for combo in itertools.combinations(explanatory_vars, i):
        tested_combos += 1
        percent_complete = (tested_combos / total_combos) * 100
        print(f"[{percent_complete:.2f}%] Testing combo: {combo}")

        fixed_formula = outcome_var + " ~ " + " + ".join(combo)
        print(f"Fixed formula: {fixed_formula}")
        try:
            model = smf.mixedlm(fixed_formula, data=df, groups=df[group_var])
            result = model.fit(reml=False)
            marginal_r2, conditional_r2 = calculate_r_squared(result)
            print(f"Marginal R-squared (percent of variance explained by fixed effects): {marginal_r2:.4f}")
            print(
                f"Conditional R-squared (Marginal R-square variance explained plus variance explained by random effects): {conditional_r2:.4f}")
            print(result.aic)
            results.append({
                'variables': combo,
                'aic': result.aic,
                'marginal_r2': marginal_r2,
                'conditional_r2': conditional_r2
            })
        except Exception as e:
            print(f"Skipping combo {combo} due to error: {e}")


def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False

def autosave(filename='session.pkl'):
    exclude = {'__builtins__', '__name__', '__file__', '__doc__', '__package__'}
    session = {k: v for k, v in globals().items()
               if k not in exclude and not k.startswith('__') and is_picklable(v)}
    with open(filename, 'wb') as f:
        pickle.dump(session, f)





# === Optional function to test a single configuration of the linear random effects model ===

# Not written with parameters for easy manipulation. Must change all hard coded fields. Opportunity to enhance in future
def non_automated_lme():
    # Define your variables
    outcome_var = "Math_2021"

    explanatory_vars = ["Reading_1819", "giftedDummy", "Math_1819", "Reading_2021", "Science_2021", "spedDummy", "Gender_M",
                        "EL_2021_Level_3", "EL_2021_Level_4", "EL_2021_Level_6___All_Years",
                        "Race_Recode_Black_or_African_American", "Race_Recode_Hispanic", "Race_Recode_White",
                        "Membership_Days_2021", "Total_Absence_Days_2021"]  # 0.7269

    #explanatory_vars = ["Math_1819"]
    group_var = "school_classroom"  # Replace with your actual grouping variable name

    # Build the formula
    fixed_formula = outcome_var + " ~ " + " + ".join(explanatory_vars)
    print(f"Fixed formula: {fixed_formula}")

    # Fit the model
    try:
        model = smf.mixedlm(fixed_formula, data=df, groups=df[group_var])
        result = model.fit(reml=False)
        print("AIC:", result.aic)
    except Exception as e:
        print(f"Model fitting failed due to error: {e}")

    print (result.summary())

    print(df[group_var].value_counts())
    print(df[group_var].nunique())

    marginal_r2, conditional_r2 = calculate_r_squared(result)
    print(f"Marginal R-squared (percent of variance explained by fixed effects): {marginal_r2:.4f}")
    print(f"Conditional R-squared (Marginal R-square variance explained plus variance explained by random effects): {conditional_r2:.4f}")

    # ==== VIF - Check for multicollinearity on non-autoamted model generated ====

    # Build the formula for VIF calculation
    formula = " + ".join(explanatory_vars)

    # Create design matrix
    X = dmatrix(formula, data=df, return_type='dataframe')

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_data)


# === BOXPLOT OF OUTCOME VARIABLE ===
plt.figure(figsize=(10, 6))
sns.boxplot(x=df[group_var], y=df[outcome_var])
plt.title('Boxplots of Outcome Variable: Math2021 by Group Variable: Vaughan Elementary 5th Grade Classroom')
plt.show()
plt.savefig("boxplot_vaughan_5th_grade_classrooms_math2021.png")


# === Optional function to test a single configuration of the linear fixed effects model (no group) ===

# Not written with parameters for easy manipulation. Must change all hard coded fields. Opportunity to enhance in future

def non_automated_lfe_model():
    # Define your variables
    outcome_var = "Math_2021"
    #explanatory_vars = ["Reading_1819", "giftedDummy", "Math_1819", "spedDummy", "giftedDummy", "Gender_M",
     #                   "EL_2021_Level_3", "EL_2021_Level_4", "EL_2021_Level_6___All_Years"]
    explanatory_vars = ["Reading_1819", "giftedDummy", "Math_1819", "Reading_2021", "Science_2021", "spedDummy", "giftedDummy", "Gender_M",
                        "EL_2021_Level_3", "EL_2021_Level_4", "EL_2021_Level_6___All_Years"]
    #explanatory_vars = ["Math_1819"]

    # Build the formula
    fixed_formula = outcome_var + " ~ " + " + ".join(explanatory_vars)
    print(f"Fixed formula: {fixed_formula}")

    # Fit the model
    try:
        model = smf.ols(fixed_formula, data=df)
        result = model.fit()
        print("AIC:", result.aic)
    except Exception as e:
        print(f"Model fitting failed due to error: {e}")

    # R-squared from the model
    print("R-squared:", result.rsquared)
    # Print model parameters
    print(result.summary())

def multicollinearity_check(ranker):
    # === CHECK FOR MULTICOLLINEARITY OF TOP MODELS ===
    print(f"\n=== Top Models by {ranker} ===")
    for i, r in enumerate(results[:5]):  # Show top 5
        print(f"\nModel {i + 1}: AIC: {r['aic']:.2f} | Variables: {r['variables']} | Marginal R^2: {r['marginal_r2']:.2f} | Conditional R^2: {r['conditional_r2']:.2f}")

        # === MULTICOLLINEARITY CHECK: VIF ===
        # VIF (Variance Inflation Factor) shows how much a variable is inflated due to multicollinearity.
        # Interpretation guide:
        # - VIF ≈ 1: No multicollinearity
        # - VIF 1–5: Low to moderate multicollinearity (generally acceptable)
        # - VIF 5–10: Moderate to high (review carefully)
        # - VIF > 10: High multicollinearity — consider dropping or combining variables

        X = df[list(r['variables'])].dropna()
        X = sm.add_constant(X)
        # Dummy variables stored as Boolean must be converted to 1 or 0, prior to calculating VIF.
        X = X.astype(int)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])]
        print("VIF:")
        print(vif_data.to_string(index=False))



# === RANKED OUTPUT ===
# sort by AIC
results = sorted(results, key=lambda x: x['aic'])
# export results to csv
result_df = pd.DataFrame(results)
result_df.to_csv("model_results_aic_sorted.csv", index=False)
# analyze VIFs for top ranked models by AIC
multicollinearity_check('AIC')

# sort by marginal r-squared
results = sorted(results, key=lambda x: x['marginal_r2'], reverse=True)
# export results to csv
result_df = pd.DataFrame(results)
result_df.to_csv("model_results_r2_sorted.csv", index=False)
# analyze VIFs for top ranked models by marginal r-squared
multicollinearity_check('Marginal r-squared')

# === LIKELIHOOD RATIO TEST BASED ON CHI-SQUARED DISTRIBUTION ===
print("\n=== Likelihood Ratio Test: Subset Comparison ===")

# Use the best variable combination - set index from the order in the results list
best_vars = list(results[2]['variables'])

# 1. Combined Model with interaction terms
interaction_terms = ' + '.join([f'{var}*{subset_col}' for var in best_vars])
combined_formula = f"{outcome_var} ~ {interaction_terms}"

try:
    combined_model = smf.mixedlm(combined_formula, data=df, groups=df[group_var])
    combined_result = combined_model.fit(reml=False)
    ll_combined = combined_result.llf
except Exception as e:
    print(f"Error fitting combined model: {e}")
    ll_combined = None

# 2. Separate models for subset1 and subset2
try:
    model1 = smf.mixedlm(f"{outcome_var} ~ {' + '.join(best_vars)}", data=subset1, groups=subset1[group_var])
    result1 = model1.fit(reml=False)
    ll1 = result1.llf
except Exception as e:
    print(f"Error fitting model for subset 1: {e}")
    ll1 = None

try:
    model2 = smf.mixedlm(f"{outcome_var} ~ {' + '.join(best_vars)}", data=subset2, groups=subset2[group_var])
    result2 = model2.fit(reml=False)
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

# === ASSUMPTION CHECKS FOR SELECTED LINEAR MIXED EFFECTS MODEL ===
model_result_names = [result2]  # [combined_result, result1, result2] --- cycle through manually and update plt.titles below
for model in model_result_names:
    # pull out the residuals
    residuals = model.resid
    fitted_vals = model.fittedvalues

    # check for linearity
    sns.scatterplot(x=fitted_vals, y=residuals)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted values (Reassigned)')
    plt.show()
    plt.savefig("Residuals vs Fitted values.png")
    # A random scatter in the first plot would suggest linearity.

    # check for homoscedasticity using a scatter plot
    sns.scatterplot(x=fitted_vals, y=np.sqrt(np.abs(residuals)))
    plt.xlabel('Fitted values')
    plt.ylabel('sqrt(abs(Residuals))')
    plt.title('Scale-Location (Reassigned)')
    plt.show()
    plt.savefig("Scale-Location.png")
    # A random scatter around a horizontal line at zero in the second plot would suggest homoscedasticity.

    # check for normality of residuals
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals (Reassigned)')
    plt.show()
    plt.savefig("Distribution of Residuals.png")
    # In the histogram, if residuals are normally distributed, they will follow the outlined 'bell curve' closely.

    # perform a normal Q-Q plot of residuals
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.probplot(residuals, dist="norm", plot=ax)
    plt.title("Normal Q-Q plot (Reassigned)")
    plt.show()
    plt.savefig("Normal Q-Q plot.png")
    # In the QQ-plot, if residuals are normally distributed, they will follow the line closely.

# === LIKELIHOOD RATIO TEST BASED ON PARAMETRIC BOOTSTRAP ===
# Bootstrap parameters
n_bootstrap_samples = 1000
lr_stats = np.zeros(n_bootstrap_samples)

# Store the original outcome variable
original_outcome = df[outcome_var].copy()

# Generate bootstrap samples
for i in range(n_bootstrap_samples):
    # Generate bootstrap sample
    bootstrap_sample = np.random.choice(original_outcome, size=len(original_outcome), replace=True)

    # Update outcome variable with bootstrap sample
    df[outcome_var] = bootstrap_sample
    subset1[outcome_var] = bootstrap_sample[subset1.index]
    subset2[outcome_var] = bootstrap_sample[subset2.index]

    # Same model fitting as before
    try:
        combined_model = smf.mixedlm(combined_formula, data=df, groups=df[group_var])
        combined_result = combined_model.fit(reml=False)
        ll_combined = combined_result.llf
    except Exception as e:
        print(f"Error fitting combined model: {e}")
        ll_combined = None

    try:
        model1 = smf.mixedlm(f"{outcome_var} ~ {' + '.join(best_vars)}", data=subset1, groups=subset1[group_var])
        result1 = model1.fit(reml=False)
        ll1 = result1.llf
    except Exception as e:
        print(f"Error fitting model for subset 1: {e}")
        ll1 = None

    try:
        model2 = smf.mixedlm(f"{outcome_var} ~ {' + '.join(best_vars)}", data=subset2, groups=subset2[group_var])
        result2 = model2.fit(reml=False)
        ll2 = result2.llf
    except Exception as e:
        print(f"Error fitting model for subset 2: {e}")
        ll2 = None

    # Only calculate LR stat if all log-likelihoods are valid
    if ll_combined is not None and ll1 is not None and ll2 is not None:
        ll_separate = ll1 + ll2
        lr_stats[i] = 2 * (ll_separate - ll_combined)

# Restore the original outcome variable
df[outcome_var] = original_outcome
subset1[outcome_var] = original_outcome[subset1.index]
subset2[outcome_var] = original_outcome[subset2.index]

# Now calculate bootstrap p-value
p_value_bootstrap = np.mean(lr_stats >= lr_stat)

print(f"Bootstrap p-value: {p_value_bootstrap:.4f}")
if p_value_bootstrap < 0.05:
    print("→ Bootstrapping: Reject null hypothesis: model coefficients differ significantly between subsets.")
else:
    print("→ Bootstrapping: Fail to reject null hypothesis: no significant difference between subsets.")
# Save to hard drive the model, after running for large models.
autosave()
