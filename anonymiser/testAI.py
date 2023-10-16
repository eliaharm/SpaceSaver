# import panda
import pandas as pd

df = pd.read_csv("data/iris.csv")
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

df
# import csv file into pandas dataframe

# print the first 5 rows of the dataframe
print(df.head())

# print the shape of the dataframe
print(df.shape)

# print the data types of each column
print(df.dtypes)

# print a concise summary of the dataframe
print(df.info())

# print a summary of each column
print(df.describe())

# print the class distribution
print(df["species"].value_counts())

# box and whisker plots for each variable
df.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# histograms for each variable
df.hist()
plt.show()

# scatter plot matrix
scatter_matrix(df)
plt.show()

# create a 2d scatter plot for each pair of attributes
pd.plotting.scatter_matrix(
    df,
    c=df["species"],
    figsize=(10, 10),
    marker="o",
    hist_kwds={"bins": 20},
    s=60,
    alpha=0.8,
)
plt.show()

# group by species and summarize
print(df.groupby("species").mean())

# print the class distribution
print(df.groupby("species").size())

# read files recursively from OS
import glob

for name in glob.glob("data/iris.csv"):
    print(name)

# read files recursively from OS
import glob

for name in glob.glob("data/iris.csv"):
    print(name)

# read
