#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    '''
    This function accepts a string, path to a file and reads the CSV data into pandas dataframe
    
    Parameters
    ----------
        filename: str
        
    Returns
    -------
        pandas.DataFrame objects
    '''    
    df = pd.read_csv(filename)
    
    # Drop unnecessary columns
    df = df.iloc[:, 2:]
    df.drop(columns = ['Country Code'], inplace = True)  # Drop Country Code Column
    df = df.set_index('Country Name')  # Set Country Names to Index
    df = df.apply(pd.to_numeric, errors = 'coerce')  # Convert to numeric
    df = df[df.index.notna()]  # Drop NaN rows

    # Process the column names
    cols = df.columns.tolist()
    cols = [year[:4] for year in cols]
    df.columns = cols
    
    return df, df.T


# READING IN THE DATA
# Create the DataFrames from the functions
dfArableYear, dfArableCountry = read_data('Arable land.csv')
dfForestYear, dfForestCountry = read_data('Forest area.csv')


# SUMMARY STATISTICS
# Summary statistics of Percentage Arable Land and Forest Area Per Year
summaryArable = dfArableYear.describe().T
summaryForest = dfForestYear.describe().T

# Visualize the mean value for each year
x = summaryArable.index
y1 = summaryArable['mean']
y2 = summaryForest['mean']
fig, ax1 = plt.subplots(figsize = (8, 6))
ax1.plot(x, y1, label = "Arable Land (% of land area)")
ax2 = ax1.twinx()
ax2.plot(x, y2, label = "Forest Area (% of land area)", color = 'red')
plt.xlabel('Year')
ax1.set_ylabel('Mean (%)')
plt.title("Arable land vs Forest Area - (% of land area)")
ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels(labels = x, rotation = 60)
ax1.legend()
ax2.legend()
plt.grid()
plt.show()

np.random.seed(140)  # For reproducibility
arableMean = dfArableCountry.mean().sample(10).sort_values()  # Mean % Arable land for 10 random countries by

# Visualization
arableMean.plot(kind = 'bar', figsize = (8, 6))
plt.title("Mean percentage of arable land for 10 random countries")
plt.xlabel("Country Name")
plt.ylabel("Mean percentage")
plt.grid()
plt.show()

# Median Percentage of Forest area for the First 10 countries
np.random.seed(140)
forestMedian = dfForestCountry.median().sample(10)
forestMedian.plot(kind = 'pie', figsize = (8, 6))
plt.title("Median Percentage of Forest Area for first 10 countries")
plt.show()

# Investigating Relationship Per Countries
np.random.seed(140)  # For reproducibility
forest_nine_years = dfForestYear.sample(10).iloc[:,10:]
countries = forest_nine_years.index.tolist()  # To allow to obtain same countries
years = forest_nine_years.columns.tolist()
arable_nine_years = dfArableYear.loc[countries, years]

# Visualisation 1
arable_nine_years.plot(kind = 'bar', figsize = (8, 6))
plt.title("Percentage of Arable Land per Country (2012 - 2020)")
plt.xlabel("Country Name")
plt.ylabel("Percentage of Arable Land")
plt.show()

# Visualisation 2
forest_nine_years.plot(kind = 'bar', figsize = (8, 6))
plt.title("Percentage of Forest Area per Country (2012 - 2020)")
plt.xlabel("Country Name")
plt.ylabel("Percentage of Forest Area")
plt.show()


# CORRELATION
# Computing Correlation Values Per Country
corrcoef_per_country = []
print("---------------------+--------------")
print("Country              |   Correlation")
print("---------------------+--------------")
for i in range(len(forest_nine_years)):
    country = forest_nine_years.index[i]
    forest = forest_nine_years.iloc[i,:]
    arable = arable_nine_years.iloc[i, :]
    r = np.corrcoef(forest, arable)[0,1]
    print(f"{country:<20s} |  {r:>7.4f}")
    corrcoef_per_country.append((country, r))
print("---------------------+--------------")

