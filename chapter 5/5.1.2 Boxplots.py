"""
Boxplot for single dataset
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

Level_data = data[["time", "Level"]]
del data
print("Read data file")

"""
Resample:
Downsample the time series
"""
Level_data = Level_data.resample('1M', on="time").mean()


"""
Plotting the boxplot
"""
sns.boxplot(data=Level_data,
            notch=True, showcaps=False,
            flierprops={"marker": "x"},
            boxprops={"facecolor": (.4, .6, .8, .5)},
            medianprops={"color": "coral"},
            )
plt.ylabel("(m)")
plt.grid(ls='--')
plt.tight_layout()
plt.show()
plt.savefig("single_box.pdf", dpi=300)
print("\n\nPlotted!")
