"""
Program to demonstrate scatter
 matrix using seaborn
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Loading the dataset
"""
data1 = pd.read_csv( # Loading first dataset
    filepath_or_buffer="../data/Godavari.csv",
    sep=",",
    header=0
).dropna()
df1 = data1[[
    "Level", "Streamflow",
    "Pressure", "Rel_humidity"]]  # Retrieving columns
df1.loc[:, "River"] = "Godavari"  # Creating new column
df1 = df1.iloc[:2000, :]

data2 = pd.read_csv( # Loading second dataset
    filepath_or_buffer="../data/Cauvery.csv",
    sep=",",
    header=0
).dropna()
df2 = data2[[
    "Level", "Streamflow",
    "Pressure", "Rel_humidity"]]  # Retrieving columns
df2.loc[:, "River"] = "Cauvery"  # Creating new column
df2 = df2.iloc[:2000, :]

# Combining dataset
df = pd.concat([df1, df2])


"""
Visualization
"""
sns.color_palette("Set2")
sns.pairplot(df, hue="River", height=2.0, markers='x')
plt.show()
plt.savefig("multi_scatter.pdf", dpi=300)
