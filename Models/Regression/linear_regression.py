import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load preprocessed data
X_train = pd.read_csv('preprocessed_dataset\\X_train_reg.csv')
X_test = pd.read_csv('preprocessed_dataset\\X_test_reg.csv')
y_train = pd.read_csv('preprocessed_dataset\\y_train_reg.csv')
y_test = pd.read_csv('preprocessed_dataset\\y_test_reg.csv')

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Linear Regression Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")