"""
Program to do a Mann-Whitney
Test using scipy in Python
"""

import numpy as np
import pandas as pd
from scipy import stats

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
# Loading second dataset
data2 = pd.read_csv(
    filepath_or_buffer="../data/Cauvery.csv",
    sep=",",
    header=0
).dropna()

df1 = data1[["Level", "Streamflow"]]  # Retrieving columns
df1.loc[:, "River"] = "Godavari"  # Creating new column
df1 = df1.iloc[:1000, :]

df2 = data2[["Level", "Streamflow"]]  # Retrieving columns
df2.loc[:, "River"] = "Cauvery"  # Creating new column
df2 = df2.iloc[:1000, :]


"""
Mann-Whitney Test
"""
alpha = 0.05  # significance
MW_statistic, p_val = stats.mannwhitneyu(
    x=df1["Streamflow"], y=df2["Streamflow"],
    method="auto"
)
print("\n1. Mann-Whitney Test")
print("   p-value={}".format(p_val))
if p_val < alpha:
    print("   The two distributions are different")
else:
    print("   The two distributions are same")

print("\n-----------------------------------"
      "--------------------------------------\n")

# Second case - Similar datasets
chr = 0.8  # character
amp_noisy = chr*df1["Streamflow"] + \
            (1.-chr)*df2["Streamflow"]*np.random.normal(
    loc=0.08, scale=0.01, size=df2.shape[0])
df2["Streamflow"] = amp_noisy
MW_statistic2, p_val2 = stats.mannwhitneyu(
    x=df1["Streamflow"], y=df2["Streamflow"],
    method="auto"
)
print("\n1. Mann-Whitney Test")
print("   p-value={}".format(p_val2))
if p_val2 < alpha:
    print("   The two distributions are different")
else:
    print("   The two distributions are same")
