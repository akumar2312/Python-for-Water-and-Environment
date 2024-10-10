"""
Demonstration of Maximum-likelihood estimation
for log-normal data
"""
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Loading the dataset
"""
data = pd.read_csv(
    filepath_or_buffer="../data/Godavari.csv",
    sep=",",
    header=0
).dropna()
streamflow_data = np.array(data["Streamflow"])
streamflow_data = streamflow_data[streamflow_data > 0]

# Log transformation of stream data
data = np.log10(streamflow_data)

# Plot preliminary histogram
plt.figure()
plt.hist(data,
         density=True, bins=100, log=True,
         label="log(data)")

"""
Fit the normal distribution parameters.
These formulae can be used to get the
parameters of the lognormal distribution:
mu_ = ln(mu) - (1/2) * ln(1 + (sig^2 / mu^2))
sig_ = (ln(1 + (sig^2 / mu^2)))**0.5
"""
fit = norm.fit(data=data)  # fit Normal distribution
mu, scale = fit  # in Normal distribution

N = int(data.shape[0]/2)  # generate N samples
samples = norm.rvs(loc=mu, scale=scale, size=N)

plt.hist(samples,  # Plot MLE estimated distribution
         density=True, bins=100, log=True,
         label="log(data$_{" + "MLE" + "}$)",
         alpha=.6)
plt.xlabel("Streamflow (m$^3$/s)")
plt.ylabel("log(Frequency)")
plt.legend()
plt.grid(ls="--")
plt.tight_layout()
plt.savefig("streamflow_log_hist.pdf", dpi=300)
