import pandas as pd
from sklearn.ensemble import IsolationForest

# Load test data
df = pd.read_csv("data/test_processed.csv", parse_dates=["DateTime"])

# Fit Isolation Forest
iso = IsolationForest(contamination=0.01, random_state=42)
iso.fit(df[["AQI"]])

# Predict anomalies
df["Anomaly"] = iso.predict(df[["AQI"]])
df["Anomaly"] = df["Anomaly"].apply(lambda x: 1 if x==-1 else 0)

# Save anomalies
df.to_csv("data/test_with_anomalies.csv", index=False)

print("Anomaly detection done, anomalies saved")
