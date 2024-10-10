"""
Program to do simple linear
regression on the Godavari
streamflow data
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


savePlots = 1


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
df = data[["time", "Pressure",
           "Rel_humidity", "Level", "Streamflow"]]
del data
print("Read data file")


"""
Resample:
Downsample the time series
"""
df = df.resample('1M', on="time").mean()


"""
Data preprocessing
"""
X = df["Level"].values.reshape(-1, 1)
y = df["Streamflow"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=11)


"""
Linear Regression model fitting
"""
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
b0, b1 = linearRegression.intercept_, \
    linearRegression.coef_[0]
print("\n\nEstimated parameters: "
      "\n   (b0, b1) = ({:.2f}, {:.2f})".
      format(b0, b1))
print("R-squared value: \n    R^2 = {:.2f}".
      format(linearRegression.score(X_train, y_train)))


"""
Visualization of results
"""
# Plot the datapoints
sns.scatterplot(x=np.squeeze(X_train), y=y_train)
plt.grid(ls="--")
plt.xlabel("Level(m)")
plt.ylabel("Streamflow(cumecs)")

# Plot regression line
x1, x2 = np.min(X), np.max(X)
y1, y2 = (b1*x1 + b0), (b1*x2 + b0)
plt.plot([x1, x2], [y1, y2], 'r-', label="Linear Model")
plt.legend()
plt.grid(ls="--")

# Tidying the plot
plt.tight_layout()
plt.show()
plt.savefig("linearReg_.pdf", dpi=300)

print("Done!!")
