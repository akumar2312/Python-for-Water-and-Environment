"""
Program for Non-parametric prediction
interval estimate on 1D stream flow data
"""
import pandas as pd
from scipy.stats import (norm, binom, tmean, tstd,
                         t, nct, bootstrap, mstats)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
Non-parametric prediction interval
"""
print("\n\nDoing non-parametric PI estimate")
qi = mstats.mquantiles(
    a=x, prob=[0.05, 0.95],
    alphap=0, betap=0  # type 6 in R
    )
print(f"Lower Bound: {qi[0]}")
print(f"Upper Bound: {qi[1]}")

print("\nDone!")


"""
Plot the prediction interval
"""
def plot_hist_PI(x, pi_lower, pi_upper, bins=50, prcnt=95):
    # Plot histogram
    # plt.hist(x, bins=50, alpha=0.75, color='blue',
    #          label='Data distribution', density=True)

    sns.histplot(x, kde=True, bins=bins)

    plt.xlim([-400, 2500])

    # Plot intervals
    plt.axvline(pi_lower, color='red', linestyle='dashed',
                linewidth=2, label='{}% PI Lower Limit={:.2f}'.format(prcnt, pi_lower))
    plt.axvline(pi_upper, color='green', linestyle='dashed',
                linewidth=2, label='{}% PI Upper Limit={:.2f}'.format(prcnt, pi_upper))
    plt.legend()

    # plt.title("Histogram with {}% Confidence Interval".format(prcnt))
    plt.xlabel("Streamflow (cumecs)")
    plt.ylabel("Frequency")
    plt.grid(ls="--")
    plt.tight_layout()
    plt.savefig("uncer-nonparametric_pi_median.pdf", dpi=300)


plot_hist_PI(x,
             pi_lower=qi[0],
             pi_upper=qi[1]
             )
