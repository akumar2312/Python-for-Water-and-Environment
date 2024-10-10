"""
Program for non-parametric confidence
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
Non-parametric confidence interval estimate on 
percentile data
"""
print("\n\nDoing non-parametric CI estimate")
prob = 0.2
percentile = mstats.mquantiles(  # value of quantile
    a=x, prob=prob,
    alphap=0, betap=0
    )
print("Quantile value =", percentile)

qnt = binom.ppf(q=[0.025, 0.975],  # position of quantiles
                n=len(x), p=prob,
                loc=0)
qnt = np.cast[int](qnt)
print("95% CI positions =", qnt)

result = binom.pmf(k=np.arange(qnt[0]-1, qnt[1]),
                   n=len(x),
                   p=prob,
                   loc=0)
print("Probability sum upto 0.2 =", result.sum())


# Using bootstrap method
def percentile_statistic(data, axis=1):
    percentile = mstats.mquantiles(  # value of quantile
        a=data, prob=0.2,
        alphap=0, betap=0, axis=axis
    )
    percentile = np.squeeze(percentile)
    return percentile


x = (x,)
ci = bootstrap(data=x,
               statistic=percentile_statistic,
               confidence_level=0.95,
               n_resamples=10000,
               method="percentile")

# Get the lower and upper bounds of the confidence interval
lower_bound = ci.confidence_interval.low
upper_bound = ci.confidence_interval.high

print("\n\n{}th Percentile Bootstrap Confidence Interval:".format(prob*100))
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

print("\nDone!")


"""
Plot the prediction interval
"""
def plot_hist_PI(x, ci_lower, ci_upper, bins=50, prcnt=95):
    # Plot histogram
    # plt.hist(x, bins=50, alpha=0.75, color='blue',
    #          label='Data distribution', density=True)

    sns.histplot(x, kde=True, bins=bins)

    plt.xlim([-300, 2500])

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
    plt.savefig("uncer-nonparametric_2sided_quantile_ci.pdf", dpi=300)


plot_hist_PI(x,
             ci_lower=lower_bound,
             ci_upper=upper_bound
             )
