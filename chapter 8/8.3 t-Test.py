"""
Program demonstrating t-Test
using scipy
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
df1 = data1[["Level", "Streamflow"]]  # Retrieving two columns
df1.loc[:, "River"] = "Godavari"  # Creating new column
df1 = df1.iloc[:1000, :]

# Loading second dataset
data2 = pd.read_csv(
    filepath_or_buffer="../data/Cauvery.csv",
    sep=",",
    header=0
).dropna()
df2 = data2[["Level", "Streamflow"]]  # Retrieving two columns
df2.loc[:, "River"] = "Cauvery"  # Creating new column
df2 = df2.iloc[:1000, :]


"""
t-Test
"""
# Performing the t-Test (First part)
alpha = 0.05
t_statistics = stats.ttest_ind(
    a=df1["Streamflow"], b=df2["Streamflow"],
    equal_var=False
)
print(t_statistics)
print("\n1. Different samples:")
print("     p={:0.010f}".format(t_statistics[1]))
if t_statistics[1] < alpha:
    print("     -> The sample means are different")
else:
    print("     -> The sample means are same")


print("\n-----------------------------------"
      "--------------------------------------\n")

# Modifying data to be similar to the first (Second part)
amp = df1["Streamflow"] * 0.05  # 5% of the amplitude
amp_noisy = amp * np.random.normal(
    loc=10, scale=2, size=df2.shape[0])
amp_noisy = amp_noisy + df2["Streamflow"]
df2["Streamflow"] = amp_noisy
t_statistics2 = stats.ttest_ind(
    a=df1["Streamflow"], b=df2["Streamflow"],
    equal_var=False
)
print(t_statistics2)
print("\n2. Modified samples:")
print("     p={:0.010f}".format(t_statistics2[1]))
if t_statistics2[1] < alpha:
    print("     -> The sample means are different")
else:
    print("     -> The sample means are same")
