import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'Gradient Boosting']
r2 = [0.4032, 0.0484, 0.5389, 0.4620, 0.4424]
colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

# Create Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(models, r2, color=colors, edgecolor='black')
plt.xlabel('Regression Models')
plt.ylabel('R² Score')
plt.title('Comparison of Regression Models by R² Score')
plt.ylim(0, 1)  # R² ranges from 0 to 1
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig('regression_r2.png', dpi=300, bbox_inches='tight')
plt.close()