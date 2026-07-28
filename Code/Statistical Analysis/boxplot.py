import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".\\dataset\\winequality-red.csv", sep=';')

features = [col for col in df.columns if col != 'quality']

for i, feature in enumerate(features, 1):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='quality', y=feature, palette='Blues')
    plt.title(f'{feature} by Quality')
    plt.xlabel('Wine Quality Score')
    plt.ylabel(feature)
    plt.savefig(f'.\\boxplot\\{feature}_boxplot.png', bbox_inches='tight', dpi=100)
    plt.close()  
