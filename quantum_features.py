import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load preprocessed train data
df = pd.read_csv("data/train_processed.csv", parse_dates=["DateTime"])
city_df = df[df["City"]=="Delhi"]

features = ["PM2.5","PM10","NO2","SO2","CO","O3","NH3","Temperature","Humidity","WindSpeed","Pressure","Rainfall"]
X_qml = city_df[features].values
y_qml = city_df["AQI"].values

# Standardize features
scaler = StandardScaler()
X_qml_scaled = scaler.fit_transform(X_qml)

# Save prepared features for quantum model
import numpy as np
np.save("data/X_qml.npy", X_qml_scaled)
np.save("data/y_qml.npy", y_qml)
print("Quantum ML features ready")
