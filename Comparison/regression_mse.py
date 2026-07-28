import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'Gradient Boosting']
mse = [0.3900, 0.6219, 0.3014, 0.3516, 0.3644]
colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

# Create Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(models, mse, color=colors, edgecolor='black')
plt.xlabel('Regression Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Regression Models by MSE')
plt.ylim(0, 0.7)  # Adjust y-axis for clarity
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig('regression_mse.png', dpi=300, bbox_inches='tight')
plt.close()