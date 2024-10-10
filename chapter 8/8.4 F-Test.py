"""
Program to  perform F-Test
using scipy in Python
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
F-Test
"""
def F_Test(a, b):
    # Computing variances
    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)
    fstat = np.divide(var1, var2)

    # Getting DOFs of samples
    dof_numerator, dof_denominator = \
        a.shape[0]-1, b.shape[0]-1

    # Computing p-value using f distribution's cdf
    p_val = 1. - stats.f.cdf(
        x=fstat, dfn=dof_numerator, dfd=dof_denominator
    )

    # Returning results
    return fstat, p_val


# Performing the F-Test (First part)
alpha = 0.05
f_statistics = F_Test(
    a=df1["Streamflow"], b=df2["Streamflow"]
)
print(f_statistics)
print("\n1. Different samples:")
print("     p={:0.010f}".format(f_statistics[1]))
if f_statistics[1] < alpha:
    print("     -> The sample variances are different")
else:
    print("     -> The sample variances are same")


print("\n-----------------------------------"
      "--------------------------------------\n")

# Modifying data to be similar to the first (Second part)
amp = df1["Streamflow"] * 0.08  # 8% of the amplitude
amp_noisy = amp * np.random.normal(
    loc=10, scale=2, size=df2.shape[0])
amp_noisy = amp_noisy + df2["Streamflow"]
df2["Streamflow"] = amp_noisy
f_statistics2 = F_Test(
    a=df1["Streamflow"], b=df2["Streamflow"]
)
print(f_statistics2)
print("\n2. Modified samples:")
print("     p={:0.010f}".format(f_statistics2[1]))
if f_statistics2[1] < alpha:
    print("     -> The sample variances are different")
else:
    print("     -> The sample variances are same")
