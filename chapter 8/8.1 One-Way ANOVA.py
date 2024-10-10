"""
Program to demonstrate 1-way ANOVA
on the hydrological data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Avoiding Pandas warning for column assignment
pd.options.mode.chained_assignment = None

np.random.seed(11)

"""
Loading the dataset
"""
# Loading first dataset
data1 = pd.read_csv(
    filepath_or_buffer="../data/Godavari.csv",
    sep=",",
    header=0
).dropna()
df1 = data1[["Level", "Streamflow"]]  # Retrieving columns
df1.loc[:, "River"] = "Godavari"  # Creating new column
df1 = df1.iloc[:1000, :]

# Loading second dataset
data2 = pd.read_csv(
    filepath_or_buffer="../data/Cauvery.csv",
    sep=",",
    header=0
).dropna()
df2 = data2[["Level", "Streamflow"]]  # Retrieving columns
df2.loc[:, "River"] = "Cauvery"  # Creating new column
df2 = df2.iloc[:1000, :]


"""
One-way ANOVA
"""
############## First-part ##############
print("\n\nFirst part")
# Creating a combined dataframe
df = pd.concat(objs=[df1, df2], axis=0)

"""
Fit a linear model to be used
by the ANOVA routine below
"""
linearModel = ols(
    formula='Streamflow ~ C(River)', data=df).fit()

"""
Perform 1-way ANOVA on the
fitted model
"""
anova_result = sm.stats.anova_lm(linearModel, typ=1)

# Displaying the results
print("\n1. ANOVA result:")
print(anova_result)

# Access the p-values from the ANOVA table then
# compare the p-values to the desired
# significance level (e.g., 0.05)
# finally, print the significant results
alpha = 0.05
p_values = anova_result['PR(>F)'].dropna()
significant_results = p_values > alpha
if significant_results['C(River)']:
    print("\nRiver:")
    print("     Alternate Hypothesis => "
          "No significant difference => "
          "Means are equal")
else:
    print("\nRiver:")
    print("     Null Hypothesis => "
          "Difference is significant => "
          "Means are different")

print("-----------------------------------"
      "--------------------------------------")





############## Second-part ##############
print("\n\n\n\nSecond part")
# Adding noise to the data
amp = df1["Streamflow"] * 0.05
amp_noisy = amp * np.random.normal(
    loc=10, scale=2, size=df2.shape[0])
amp_noisy = amp_noisy + df2["Streamflow"]
df2["Streamflow"] = amp_noisy

# Creating a combined dataframe
dfnoisy = pd.concat(objs=[df1, df2], axis=0)

"""
Fit a linear model to be used
by the ANOVA routine below
"""
linearModel2 = ols(
    formula='Streamflow ~ C(River)', data=dfnoisy).fit()

"""
Perform 1-way ANOVA on the
fitted model
"""
anova_result2 = sm.stats.anova_lm(linearModel2, typ=1)
alpha2 = 0.05

# Displaying the results
print("\n2. ANOVA result:")
print(anova_result2)
p_values2 = anova_result2['PR(>F)'].dropna()
significant_results2 = p_values2 > alpha2
if significant_results2['C(River)']:
    print("\nRiver:")
    print("     Alternate Hypothesis => "
          "No significant difference => "
          "Means are equal")
else:
    print("\nRiver:")
    print("     Null Hypothesis => "
          "Difference is significant => "
          "Means are different")
