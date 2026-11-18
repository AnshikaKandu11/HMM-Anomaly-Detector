import numpy as np
import pandas as pd

np.random.seed(42)

# Generate normal sensor data
normal_data = np.random.normal(50, 2, 2000)

# Generate test sensor data
test_data = np.random.normal(50, 2, 1000)

# Add unusual sensor readings
test_data[200:220] += 20
test_data[500:520] -= 25
test_data[800:820] += 30

# Save datasets
pd.DataFrame(normal_data, columns=["sensor"]).to_csv(
    "data/normal_data.csv", index=False
)

pd.DataFrame(test_data, columns=["sensor"]).to_csv(
    "data/test_data.csv", index=False
)

print("Sensor datasets generated successfully!")