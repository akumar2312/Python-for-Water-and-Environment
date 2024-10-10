"""
Quantile plot for single dataset
"""

# Import libraries
import pandas as pd
from scipy import stats
import pylab
import matplotlib.pyplot as plt


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

Level_data = data[["time", "Level"]]
del data
print("Read data file")


"""
Resample:
Downsample the time series
"""
Level_data = Level_data.resample(
    rule='1M', on="time").mean()


"""
Plotting the q-q plot
"""
stats.probplot(
    x=Level_data['Level'],
    dist="norm",
    plot=pylab
)
plt.xlabel("Sample quantiles")
plt.ylabel("Ranked Level data (m)")
plt.title("")
plt.grid(ls='--')
plt.axis('square')
plt.xlim(-4.5, 4.5)

plt.tight_layout()
plt.show()
plt.savefig("single_qq.pdf", dpi=300)
print("\n\nPlotted!")
