import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Load preprocessed data
X_train = pd.read_csv('preprocessed_dataset\\X_train_clf.csv')
X_test = pd.read_csv('preprocessed_dataset\\X_test_clf.csv')
y_train = pd.read_csv('preprocessed_dataset\\y_train_clf.csv')
y_test = pd.read_csv('preprocessed_dataset\\y_test_clf.csv')

# Train the model
model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# Print results
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:\n", cm)