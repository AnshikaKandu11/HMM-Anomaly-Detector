import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load detection results
data = pd.read_csv("data/detection_results.csv")

# Actual anomaly labels
actual = np.zeros(len(data), dtype=int)

actual[200:220] = 1
actual[500:520] = 1
actual[800:820] = 1

# Predicted anomaly labels
predicted = data["anomaly"].values

# Calculate evaluation metrics
accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted, zero_division=0)
recall = recall_score(actual, predicted, zero_division=0)
f1 = f1_score(actual, predicted, zero_division=0)

print("\n===== MODEL EVALUATION =====")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\nClassification Report:")
print(classification_report(actual, predicted, zero_division=0))