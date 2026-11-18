import pandas as pd
import pickle
import time

# Load trained model
with open("hmm_model.pkl", "rb") as file:
    model, _ = pickle.load(file)

# Load normal training data
train = pd.read_csv("data/normal_data.csv")
train_X = train[["sensor"]].values

# Calculate window threshold
window_size = 20
scores = []

for i in range(len(train_X) - window_size + 1):
    window = train_X[i:i + window_size]
    scores.append(model.score(window) / window_size)

threshold = pd.Series(scores).quantile(0.05)

# Load test stream
data = pd.read_csv("data/test_data.csv")
X = data["sensor"].values

print("Real-Time Anomaly Detection Started!")

# Process incoming sensor readings
for i in range(len(X) - window_size + 1):

    window = X[i:i + window_size].reshape(-1, 1)
    score = model.score(window) / window_size

    if score < threshold:
        print("Anomaly Detected at:", i)

    if i % 100 == 0:
        print("Processed:", i, "readings")

    time.sleep(0.01)

print("Real-Time Detection Completed!")