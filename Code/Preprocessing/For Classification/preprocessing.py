# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the red wine dataset
df = pd.read_csv(".\\dataset\\winequality-red.csv", sep=';')

# 1. Handling Missing Values
print("Missing values:\n", df.isnull().sum())
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 2. Discretization
# Discretize 'quality' for classification
bins_quality = [0, 4, 6, 10]  # Low (0-4), Medium (5-6), High (7-10)
labels_quality = ['Bad', 'Medium', 'Good']
data_imputed['quality_binned'] = pd.cut(data_imputed['quality'], bins=bins_quality, labels=labels_quality, include_lowest=True)

# Discretize 'alcohol' (influential feature)
bins_alcohol = [0, 10, 12, 15]  # Low, Medium, High
labels_alcohol = ['Low', 'Medium', 'High']
data_imputed['alcohol_binned'] = pd.cut(data_imputed['alcohol'], bins=bins_alcohol, labels=labels_alcohol, include_lowest=True)

# Discretize 'residual sugar' (skewed distribution)
bins_sugar = [0, 2, 4, 20]  # Low, Medium, High
labels_sugar = ['Low', 'Medium', 'High']
data_imputed['residual_sugar_binned'] = pd.cut(data_imputed['residual sugar'], bins=bins_sugar, labels=labels_sugar, include_lowest=True)

# 3. Encoding Categorical Features
# Label Encoding for target (quality_binned)
label_encoder = LabelEncoder()
data_imputed['quality_binned_encoded'] = label_encoder.fit_transform(data_imputed['quality_binned'])

# One-Hot Encoding for input features (alcohol_binned, residual_sugar_binned)
data_onehot = pd.get_dummies(data_imputed, columns=['alcohol_binned', 'residual_sugar_binned'], prefix=['alcohol', 'sugar'])

# 4. Normalization/Standardization
# Separate features and target
X = data_onehot.drop(['quality', 'quality_binned', 'quality_binned_encoded'], axis=1)  # Features (continuous + one-hot encoded)
y = data_onehot['quality_binned_encoded']  # Target (label-encoded classes)

# StandardScaler (zero mean, unit variance)
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)

# MinMaxScaler (scale to range [0,1]) - optional, for testing
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, test_size=0.2, random_state=42
)

# Print shapes to verify
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Save preprocessed df
X_train.to_csv('X_train_clf.csv', index=False)
X_test.to_csv('X_test_clf.csv', index=False)
y_train.to_csv('y_train_clf.csv', index=False)
y_test.to_csv('y_test_clf.csv', index=False)