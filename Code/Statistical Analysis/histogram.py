import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".\\dataset\\winequality-red.csv", sep=';')

# histogram for each feature
for col in df.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=20, color="skyblue", edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    
    plt.savefig(f".\\histogram\\{col}_histogram.png", bbox_inches='tight', dpi=100)
    plt.close()