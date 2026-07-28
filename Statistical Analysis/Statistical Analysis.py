import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".\\dataset\\winequality-red.csv", sep=';')

# show first 5 row
print(df.head())

# data structure
print(df.info())

# 1. Mean
# each feature mean
print("features mean:\n", df.mean(numeric_only=True))

# 2. Median
# each feature median
print("\nfeatures median:\n", df.median(numeric_only=True))

# 3. SD
# standard deviation
print("\nstandard deviation:\n", df.std(numeric_only=True).round(6).values)

# statistical description
print("\nstatistical description:\n", df.describe())
