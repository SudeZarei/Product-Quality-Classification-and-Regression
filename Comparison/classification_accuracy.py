import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC', 'Gradient Boosting']
accuracy = [0.8313, 0.7875, 0.8719, 0.8313, 0.8313]
colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

# Create Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(models, accuracy, color=colors, edgecolor='black')
plt.xlabel('Classification Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Models by Accuracy')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig('classification_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()