"""
Program for asymmetric prediction
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
Asymmetric prediction interval
"""
print("\n\nDoing asymmetric PI estimate")
minx = np.min(x)
logx = np.log(x - minx + 1)
xbar = tmean(logx)
xstd = tstd(logx)
n = len(x)
qnt = t.ppf(q=[0.05, 0.95],
            df=n - 1,
            loc=0, scale=1)
print("{:.4f} < x_mean < {:.4f}".format(
    np.exp(xbar + qnt[0] * np.sqrt(xstd ** 2 + xstd ** 2 / n)),
    np.exp(xbar + qnt[1] * np.sqrt(xstd ** 2 + xstd ** 2 / n))
)
)

print("Done!")


"""
Plot the prediction interval
"""
def plot_hist_PI(x, pi_lower, pi_upper, bins=50, prcnt=95):
    sns.histplot(x, kde=True, bins=bins)

    plt.xlim([-300, 2500])

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
    plt.savefig("uncer-asymmetric_pi.pdf", dpi=300)


plot_hist_PI(x,
             pi_lower=np.exp(xbar + qnt[0] * np.sqrt(xstd ** 2 + xstd ** 2 / n)),
             pi_upper=np.exp(xbar + qnt[1] * np.sqrt(xstd ** 2 + xstd ** 2 / n))
             )
