"""
Program to demonstrate parallel
plots using pandas
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


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

df = pd.concat([df1, df2])


"""
Visualization
"""
parallel_coordinates(
    frame=df, class_column='River',
    colormap=plt.get_cmap("Set2")
)
plt.grid(ls='--')
plt.tight_layout()
plt.show()
plt.savefig("multi_parallel.pdf", dpi=300)
