"""
Program for Bootstrap confidence
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
Bootstrap confidence interval estimate
"""
print("\n\nDoing Bootstrap CI estimate")
x = (x,)
ci = bootstrap(data=x,
               statistic=np.mean,
               confidence_level=0.95,
               n_resamples=2000,
               method="percentile")
lower_bound = ci.confidence_interval.low
upper_bound = ci.confidence_interval.high
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

print("\nDone!")


"""
Plot the confidence interval
"""
def plot_hist_CI(x, ci_lower, ci_upper, bins=50, prcnt=95):
    sns.histplot(x, kde=True, bins=bins)

    plt.xlim([-50, 1000])

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
    plt.savefig("uncer-bootstrap_ci_mean.pdf", dpi=300)


plot_hist_CI(x,
             ci_lower=lower_bound,
             ci_upper=upper_bound
             )
