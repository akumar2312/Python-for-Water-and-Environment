"""
Program to do time-series modelling
using the Autoregression model
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns

savePlots = 1

def plotGraph(df_, label, figsize, indicator, title,
              save=savePlots):
    if save:
        if indicator == 0:
            df_.plot(figsize=figsize)
            plt.xlabel(label[0])
            plt.ylabel(label[1])
            plt.legend()
        elif indicator == 1:
            fig, ax = plt.subplots(2, 1, figsize=figsize,
                                   sharex=True)
            df_["Streamflow"].hist(ax=ax[0])
            df_["Streamflow"].plot(kind='kde', ax=ax[1])
            ax[0].set_title("")
            ax[0].grid(ls="--")
            ax[1].grid(ls="--")
            ax[1].set_xlabel(label[0])
            ax[1].set_ylabel(label[1])
        elif indicator == 2:
            plt.figure(figsize=figsize)
            sns.boxplot(x=df_["Streamflow"].index.year,
                        y=df_["Streamflow"])
        elif indicator == 3:
            fig, ax = plt.subplots(2, 1, figsize=figsize,
                                   sharex=True)
            plot_acf(x=df_["Streamflow"], lags=40, ax=ax[0])
            ax[0].set_title("")
            ax[0].set_ylabel("ACF")
            plot_pacf(x=df_["Streamflow"],
                      lags=40, method="ywm", ax=ax[1])
            ax[1].set_xlabel("lag")
            ax[1].set_ylabel("PACF")
        elif indicator == 4:
            fig, ax = plt.subplots()
            df_.plot(y="Test series",
                     use_index=True, style="-x",
                     lw=3, ms=8, ax=ax)
            df_.plot(y="Predicted series",
                     use_index=True, style="-o",
                     lw=3, ms=8, alpha=0.6,
                     ax=ax)
            ax.grid('on', ls="--", which='minor', axis='both')
            plt.xlabel(label[0])
            plt.ylabel(label[1])
        plt.title("")
        plt.grid(ls="--")
        plt.tight_layout()
        plt.show()
        plt.savefig(title + "_.pdf", dpi=300)


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
Resample:
Downsample the time series
"""
resampled = df.resample('2M', on="time").mean()
plotGraph(df_=resampled,  # Time series plot
          label=["Year", "Streamflow"],
          figsize=(10, 4),
          indicator=0,
          title="ar-ts_resampled")

# Before transformation and standardization
plotGraph(df_=resampled,  # Histogram
          label=["Streamflow", ""],
          figsize=(7, 6),
          indicator=1,
          title="ar-histogram")
plotGraph(df_=resampled,  # Boxplot
          label=["Year", "Streamflow"],
          figsize=(14, 4),
          indicator=2,
          title="ar-boxplot")


"""
Pre-processing:
Data transformation and Visualization
"""


def transform_data(x):
    x = np.log(x - xmin + 100.)
    x = (x - meanx_) / stdx_
    return x


def inverse_transform(x):
    x = x * stdx_ + meanx_  # non-standardized
    x = np.exp(x) + xmin - 100.  # inverse log positive
    return x


xmin = np.array(resampled.min())[0]
meanx_ = np.array(resampled.mean())[0]
stdx_ = np.array(resampled.std())[0]

resampled = resampled.apply(transform_data)
print("\nDone data transformation")



# After transformation and standardization
plotGraph(df_=resampled,  # Time series plot
          label=["Year", "$log($" + "Streamflow" + "$)$"],
          figsize=(10, 4),
          indicator=0,
          title="ar-ts_resampled_after")
plotGraph(df_=resampled,  # Histogram
          label=["Streamflow", ""],
          figsize=(7, 6),
          indicator=1,
          title="ar-histogram_after")
plotGraph(df_=resampled,  # Boxplot
          label=["Year", "Streamflow"],
          figsize=(14, 4),
          indicator=2,
          title="ar-boxplot_after")


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


"""
Autocorrelation and Partial autocorrelation
"""
plotGraph(df_=resampled,  # Autocorrelation plot
          label=["", ""],
          figsize=(7, 6),
          indicator=3,
          title="ar-acorr")


"""
Data partition and Simple Autoregression fitting
"""
trainmask = (resampled.index >= "1981-01-01") & \
            (resampled.index <= "2002-12-31")
testmask = (resampled.index > "2002-12-31")
training_set = list(resampled["Streamflow"].loc[trainmask])
testing_set = list(resampled["Streamflow"].loc[testmask])
print("Created training and testing datasets\n")

"""
Single-step predictions
using autoregression
"""
forecasts = list()
for i in range(len(testing_set)):
    modelAR = AR(training_set, lags=3).fit()
    pred = np.squeeze(modelAR.forecast())
    forecasts.append(pred)
    training_set.append(testing_set[i])
    print("Obs={:0.04f}, Pred={:0.04f}    {}/{}".format(
        np.squeeze(inverse_transform(testing_set[i])),
        np.squeeze(inverse_transform(pred)),
        i + 1,
        len(testing_set))
    )
print("\nRoot mean square error:")
print("   Testset RMSE={:0.04}".
      format(MSE(testing_set, forecasts)))

"""
Result plots
"""
result = {"Year": resampled["Streamflow"].loc[testmask].index,
          "Test series":
              inverse_transform(np.array(testing_set)),
          "Predicted series":
              inverse_transform(np.array(forecasts))}
df_res = pd.DataFrame.from_dict(data=result)
df_res.set_index("Year", inplace=True)
plotGraph(df_=df_res,  # Prediction results
          label=["Year", "Streamflow"],
          figsize=(10, 4),
          indicator=4,
          title="ar-result")
print("Done!")
