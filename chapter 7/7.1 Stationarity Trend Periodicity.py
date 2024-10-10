"""
Program to check for stationarity of a
time series signal and decompose it
into trend and seasonal components
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.seasonal import seasonal_decompose


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
df = data[["time", "Streamflow"]]
del data
print("Read data file")


"""
Downsample the time series
"""
resampled = df.resample('2M', on="time").mean()


"""
Transform the data
"""
def transform(x):
    x = x - minx + 10.0
    return x


def inverse_transform(x):
    x = x - 10.0 + minx
    return x


"""
Transform data
"""
minx = resampled.min()
resampled = transform(resampled)


"""
Decompose a signal (multiplicative/additive)
"""
decompose_result_mult = seasonal_decompose(
    resampled, model="multiplicative")
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,
                       figsize=(8, 6))
ax[0].plot(decompose_result_mult.observed, label='Time series')
ax[1].plot(decompose_result_mult.seasonal, label='Seasonal')
ax[2].plot(decompose_result_mult.trend, label='Trend')
ax[0].set_ylabel('Series')
ax[1].set_ylabel('Seasonal')
ax[2].set_ylabel('Trend')
ax[2].set_xlabel('Year')
ax[0].grid(ls='--')
ax[1].grid(ls='--')
ax[2].grid(ls='--')
plt.tight_layout()
plt.savefig('seasonal.pdf', dpi=300)


"""
Stationarity check
"""
def stationarity_adf_test(x, alpha=0.05):
    adftest_res = ADF(x, autolag="AIC")
    dfout = pd.Series(
        adftest_res[0:4],
        index=["ADF statistic", "ADF p-value",
               "ADF lags used", "ADF number of obs used"])
    for key, value in adftest_res[4].items():
        dfout["   Critical Value (%s)" % key] = value
    print(dfout)
    if dfout["ADF p-value"] > alpha:
        print("   Result: Non-stationary timeseries", "\n")
    else:
        print("   Result: Stationary timeseries", "\n")


print("\nChecking stationarity:")
stationarity_adf_test(resampled)
