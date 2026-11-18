import numpy as np
import pandas as pd
import pickle

# Load trained HMM model
with open("hmm_model.pkl", "rb") as file:
    model, _ = pickle.load(file)

# Load normal training data
train = pd.read_csv("data/normal_data.csv")
train_X = train[["sensor"]].values

# Load test data
data = pd.read_csv("data/test_data.csv")
X = data[["sensor"]].values

# Sliding window settings
window_size = 20

# Calculate normal window likelihood
train_scores = []

for i in range(len(train_X) - window_size + 1):
    window = train_X[i:i + window_size]
    train_scores.append(model.score(window) / window_size)

# Use stricter anomaly threshold
threshold = np.percentile(train_scores, 1)

# Detect anomalies
data["anomaly"] = 0

for i in range(len(X) - window_size + 1):
    window = X[i:i + window_size]
    score = model.score(window) / window_size

    if score < threshold:
        data.loc[i:i + window_size - 1, "anomaly"] = 1

# Save results
data.to_csv("data/detection_results.csv", index=False)

print("Improved Anomaly Detection Completed!")
print("Window Size:", window_size)
print("Threshold:", threshold)
print("Anomalies Detected:", data["anomaly"].sum())
print("Results saved successfully!")