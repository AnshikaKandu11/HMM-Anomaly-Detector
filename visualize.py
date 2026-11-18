import pandas as pd
import matplotlib.pyplot as plt

# Load detection results
data = pd.read_csv("data/detection_results.csv")

# Plot sensor readings
plt.figure(figsize=(14, 6))

plt.plot(
    data["sensor"],
    label="Sensor Readings",
    color="blue"
)

# Highlight anomalies
anomalies = data[data["anomaly"] == 1]

plt.scatter(
    anomalies.index,
    anomalies["sensor"],
    color="red",
    label="Detected Anomalies",
    s=15
)

plt.title("HMM Based Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Sensor Values")
plt.legend()
plt.grid(True)

plt.savefig("data/anomaly_graph.png")
plt.show()

print("Anomaly Graph Generated Successfully!")