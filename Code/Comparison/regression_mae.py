import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'Gradient Boosting']
mae = [0.5035, 0.4656, 0.4220, 0.4540, 0.4853]
colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

# Create Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(models, mae, color=colors, edgecolor='black')
plt.xlabel('Regression Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Comparison of Regression Models by MAE')
plt.ylim(0, 0.6)  # Adjust y-axis for clarity
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig('regression_mae.png', dpi=300, bbox_inches='tight')
plt.close()