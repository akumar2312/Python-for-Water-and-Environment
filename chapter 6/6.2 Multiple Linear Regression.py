"""
Program to do multiple linear
regression on the Godavari
streamflow data
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
# X = df[["Pressure", "Level"]].values.reshape(-1, 2)
X = df[["Rel_humidity", "Level"]].values.reshape(-1, 2)
y = df["Streamflow"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=11)

"""
Linear Regression model fitting
"""
multipleRegression = LinearRegression()
multipleRegression.fit(X_train, y_train)
b0, b1, b2 = multipleRegression.intercept_, \
    multipleRegression.coef_[0], multipleRegression.coef_[1]
print("\n\nEstimated parameters: "
      "\n   (b0, b1, b2) = ({:.2f}, {:.2f}, {:.2f})".
      format(b0, b1, b2))
print("R-squared value: \n    R^2 = {:.2f}".
      format(multipleRegression.score(X_train, y_train)))

"""
Visualization of results
"""
# Extract min max
xmin, xmax = np.min(X_train[:, 0]), np.max(X_train[:, 0])
ymin, ymax = np.min(X_train[:, 1]), np.max(X_train[:, 1])
zmin, zmax = np.min(y_train), np.max(y_train)

# Surface equation
XX = np.linspace(xmin, xmax, 20)
YY = np.linspace(ymin, ymax, 20)
xx, yy = np.meshgrid(XX, YY)
zz = b0 + b1 * xx + b2 * yy

# Surface plot and Scatter 3d plot of points
fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, zz,
                rstride=1, cstride=1, alpha=0.5)
ax.scatter(X_train[:, 0], X_train[:, 1], y_train,
           marker='o', edgecolors='black', c='red', s=30)

# Tick numbers and locations
xtick_loc = np.linspace(xmin, xmax, 5)
ytick_loc = np.linspace(ymin, ymax, 5)
ztick_loc = np.linspace(zmin, zmax, 5)
ax.set_xticks(xtick_loc)
ax.set_yticks(ytick_loc)
ax.set_zticks(ztick_loc)

# Labels
ldist = 9
ax.set_xlabel('Relative Humidity (X)', labelpad=ldist)
ax.set_ylabel('Level (Y)', labelpad=ldist + 4)
ax.set_zlabel('Streamflow (Z)', labelpad=ldist)

# Tidying up
ax.view_init(elev=25, azim=-50)
ax.axis('auto')
plt.tight_layout()
# plt.show()

# Saving
plt.savefig("multipleReg_.pdf", dpi=300)
print("Done!!")
