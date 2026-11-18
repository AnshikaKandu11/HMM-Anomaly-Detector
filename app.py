import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

print("\n================================")
print(" HMM ANOMALY DETECTION SYSTEM")
print("================================")

# Load trained model
with open("hmm_model.pkl", "rb") as file:
    model, threshold = pickle.load(file)

# Load detection results
data = pd.read_csv("data/detection_results.csv")

# Display information
print("\nModel Information")
print("-----------------")
print("Algorithm: Gaussian HMM")
print("Hidden States:", model.n_components)
print("Threshold:", threshold)

print("\nDetection Results")
print("-----------------")
print("Total Sensor Readings:", len(data))
print("Detected Anomalies:", data["anomaly"].sum())
print("Normal Readings:", (data["anomaly"] == 0).sum())

# Display graph
graph = "data/anomaly_graph.png"

if os.path.exists(graph):
    print("\nDisplaying Anomaly Graph...")
    image = plt.imread(graph)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

print("\nProject Executed Successfully!")