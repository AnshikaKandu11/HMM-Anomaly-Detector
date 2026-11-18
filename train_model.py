import numpy as np
import pandas as pd
import pickle
from hmmlearn.hmm import GaussianHMM

# Load normal sensor data
data = pd.read_csv("data/normal_data.csv")
X = data[["sensor"]].values

# Train Gaussian HMM
model = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=100,
    random_state=42
)

model.fit(X)

# Calculate training likelihood
scores = model.score_samples(X)[0]

# Set anomaly threshold
threshold = np.percentile(scores, 5)

# Save model and threshold
with open("hmm_model.pkl", "wb") as file:
    pickle.dump((model, threshold), file)

print("HMM Model trained successfully!")
print("Hidden States:", model.n_components)
print("Anomaly Threshold:", threshold)