import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import seaborn as sns

# Create datasets from csv files
X = pd.read_csv('Data/Data_X.csv')
Y = pd.read_csv('Data/Data_Y.csv')
New_X = pd.read_csv('Data/DataNew_X.csv')

merged_df = pd.merge(X, Y, on='ID')

'''
print(X.info())
print("")
print(X.describe())
print("")
print(X.isnull().sum())
print("")
print(X[X.duplicated(keep=False)])
print("")

'''
'''
# Plot distribution for each variable
sns.pairplot(X.describe())

'''
'''
# Plot distribution for each variable
for col in X.describe().columns:
    print(col)
    plt.figure()
    plt.hist(X[col])
    plt.title(col)

    # Add shape, center, and spread information
    plt.axvline(x=X[col].mean(), color='r', linestyle='--')
    plt.axvline(x=X[col].median(), color='g', linestyle='--')
    plt.text(x=X[col].mean() + X[col].std(), y=plt.ylim()[1] * 0.9,
             s='Mean: {:.2f}\nMedian: {:.2f}\nStd Dev: {:.2f}'.format(X[col].mean(), X[col].median(), X[col].std()))

    plt.show()

'''
'''
# Plot boxplots for each variable
for col in X.columns:
    # Filter out missing or invalid data
    col_data = X[col].dropna()
    if type(col_data[1]) == np.float64 or type(col_data[1]) == np.int64 :
        plt.figure()
        plt.boxplot([col_data])
        plt.title(col)
        plt.show()

'''
'''
# Plot scatterplots for each pair of variables
for i, col1 in enumerate(X.columns):
    for j, col2 in enumerate(X.columns):
        if i >= j:
            continue

        # Filter out missing or invalid data
        data_filt = X[[col1, col2]].dropna()

        plt.figure()
        plt.scatter(data_filt[col1], data_filt[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
'''

'''
# Filter numeric columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create correlation matrix
corr = X[numeric_cols].corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20,20))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f",
            linewidths=.5, ax=ax)
plt.show()

'''