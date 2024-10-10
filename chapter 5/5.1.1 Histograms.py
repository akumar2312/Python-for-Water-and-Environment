"""
Histogram for single dataset
"""

# Import libraries
import pandas as pd
import seaborn as sns
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

Level_data = data[["Level"]]
del data
print("Read data file")


"""
Plotting the histogram
"""
sns.histplot(data=Level_data,
             bins=50)
plt.xlabel(Level_data.columns[0] + " (m)")
plt.grid(ls='--')
plt.tight_layout()
plt.show()
plt.savefig("single_hist.pdf", dpi=300)
print("\n\nPlotted!")
