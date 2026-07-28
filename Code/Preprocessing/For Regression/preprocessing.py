# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Load the red wine dataset
df = pd.read_csv(".\\dataset\\winequality-red.csv", sep=';')

# 1. Handling Missing Values
print("Missing values:\n", df.isnull().sum())
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 2. No Discretization
# All features and target ('quality') are kept continuous for regression
# No encoding is needed since there are no categorical features

# 3. Normalization/Standardization
# Separate features and target
X = data_imputed.drop('quality', axis=1)  # Features (all continuous)
y = data_imputed['quality']  # Target (numerical, 0-10)

# StandardScaler (zero mean, unit variance)
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)

# MinMaxScaler (scale to range [0,1])
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

# 4. Train/Test Split
# Split the df into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, test_size=0.2, random_state=42
)

# Print shapes to verify
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Save preprocessed df
X_train.to_csv('X_train_reg.csv', index=False)
X_test.to_csv('X_test_reg.csv', index=False)
y_train.to_csv('y_train_reg.csv', index=False)
y_test.to_csv('y_test_reg.csv', index=False)