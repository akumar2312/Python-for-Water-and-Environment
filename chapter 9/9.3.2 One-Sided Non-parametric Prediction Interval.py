"""
Program for One-side Non-parametric prediction
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
    a=x, prob=0.9,
    alphap=0, betap=0  # type 6 in R
    )
print(f"Quantile value: {qi}")

print("\nDone!")


"""
Plot the prediction interval
"""
def plot_hist_PI(x, quantile_value, bins=30, prcnt=95):
    pp = sns.histplot(x, kde=True, bins=bins)

    plt.xlim([-400, 2500])

    plt.axvline(quantile_value, color='k', linestyle='dashed',
                linewidth=1, label='{:.2f}% Prediction Interval'.format(prcnt))

    # Shading the prediction interval
    plt.fill_between([quantile_value, np.max(x)],
                     y1=0, y2=8370, color='red', alpha=0.3,
                     label='Shaded Interval (>{:.2f}%)'.format(prcnt))

    # Labels and title
    plt.xlabel('Streamflow (cumecs)')
    plt.ylabel('Density')
    plt.grid(ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("uncer-one_sided_nonparametric_pi.pdf", dpi=300)


plot_hist_PI(x, quantile_value=np.squeeze(qi))
