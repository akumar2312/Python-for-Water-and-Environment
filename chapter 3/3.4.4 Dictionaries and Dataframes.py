# Importing pandas module
import pandas as pd

# Define a dictionary containing river details
river_details = {'Name': ['Ganges',
                          'Brahmaputra',
                          'Yamuna',
                          'Godavari',
                          'Narmada'],
                 'Length (km)': [2525,
                                 3848,
                                 1376,
                                 1465,
                                 1312],
                 'Drainage Area (km2)': [1080000,
                                         651334,
                                         366223,
                                         312812,
                                         131200]
                }

# Convert the dictionary into DataFrame
river_df = pd.DataFrame(river_details)

# Print the data frame
print(river_df)
