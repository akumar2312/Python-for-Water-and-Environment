"""
Program for non-parametric interval
estimate on 1D stream flow data
"""
import pandas as pd
from scipy.stats import (norm, binom, tmean, tstd,
                         t, nct, bootstrap, mstats)
import numpy as np


"""
Load dataset
"""
data = pd.read_csv(
    filepath_or_buffer="../data/Godavari.csv",
    sep=",",
    header=0).dropna()
print("\nChecking data:")
try:
    data["time"] = pd.to_datetime(
        data['time'], infer_datetime_format=True)
    print("   Date format is okay!\n")
except ValueError:
    print("   Encountered error!\n")
    pass
x = data[["Streamflow"]].to_numpy().squeeze()
del data
print("Read data file")

"""
Non-parametric interval estimate 
"""
print("\n\nDoing non-parametric interval estimate")
qnt = binom.ppf(q=[0.025, 0.975],
                n=len(x), p=0.5,
                loc=0)
qnt = np.cast[int](qnt - 1)
print("Quantile values = ({}, {})".format(x[qnt[0]],
                                          x[qnt[1]]))

print("\nDone!")
