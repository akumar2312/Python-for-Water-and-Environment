"""
Program for confidence interval estimate
on 1D stream flow lognormal data
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
Confidence interval estimate assuming
lognormal data
"""

print("\n\nDoing CI estimate")
minx = np.min(x)
logx = np.log(x - minx + 1)
xbar = tmean(logx)
xstd = tstd(logx)
n = len(x)

qnt = norm.ppf(q=[0.9])
C90 = np.exp(xbar + qnt * xstd)

nc = -5 * qnt
qnt = nct.ppf(q=[0.05, 0.95],  # non-central t distribution
              df=n - 1,
              nc=nc,
              loc=0, scale=1)

print("{:.4f} < C90 < {:.4f}".
      format(np.exp(xbar - 1 / np.sqrt(n) * qnt[1] * xstd),
             np.exp(xbar - 1 / np.sqrt(n) * qnt[0] * xstd)
             )
      )
print("\nDone!")


"""
Plot the prediction interval
"""
def plot_hist_PI(x, ci_lower, ci_upper, bins=50, prcnt=95):
    # Plot histogram
    # plt.hist(x, bins=50, alpha=0.75, color='blue',
    #          label='Data distribution', density=True)

    sns.histplot(x, kde=True, bins=bins)

    plt.xlim([-80, 1000])

    # Plot intervals
    plt.axvline(ci_lower, color='red', linestyle='dashed',
                linewidth=2, label='{}% CI Lower Limit={:.2f}'.format(prcnt, ci_lower))
    plt.axvline(ci_upper, color='green', linestyle='dashed',
                linewidth=2, label='{}% CI Upper Limit={:.2f}'.format(prcnt, ci_upper))
    plt.legend()

    # plt.title("Histogram with {}% Confidence Interval".format(prcnt))
    plt.xlabel("Streamflow (cumecs)")
    plt.ylabel("Frequency")
    plt.grid(ls="--")
    plt.tight_layout()
    plt.savefig("uncer-two_sided_ci_lognormal.pdf", dpi=300)


plot_hist_PI(x,
             ci_lower=np.exp(xbar - 1 / np.sqrt(n) * qnt[1] * xstd),
             ci_upper=np.exp(xbar - 1 / np.sqrt(n) * qnt[0] * xstd)
             )
