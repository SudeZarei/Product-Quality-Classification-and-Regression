import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC', 'Gradient Boosting']
f1 = [0.7951, 0.7978, 0.8516, 0.7873, 0.8159]
colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

# Create Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(models, f1, color=colors, edgecolor='black')
plt.xlabel('Classification Models')
plt.ylabel('F1-Score (Weighted)')
plt.title('Comparison of Classification Models by F1-Score')
plt.ylim(0, 1)  # F1-Score ranges from 0 to 1
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig('classification_f1.png', dpi=300, bbox_inches='tight')
plt.close()