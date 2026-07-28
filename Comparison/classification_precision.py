import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC', 'Gradient Boosting']
precision = [0.7819, 0.8114, 0.8362, 0.7813, 0.8089]
colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

# Create Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(models, precision, color=colors, edgecolor='black')
plt.xlabel('Classification Models')
plt.ylabel('Precision (Weighted)')
plt.title('Comparison of Classification Models by Precision')
plt.ylim(0, 1)  # Precision ranges from 0 to 1
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig('classification_precision.png', dpi=300, bbox_inches='tight')
plt.close()