import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".\\dataset\\winequality-red.csv", sep=';')

# correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()