import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load actual AQI and hybrid forecast
test = pd.read_csv("data/test_with_anomalies.csv")
hybrid = pd.read_csv("data/hybrid_forecast.csv")

# Align lengths
min_len = min(len(test), len(hybrid))
y_true = test["AQI"][:min_len]
y_pred = hybrid["Hybrid_AQI"][:min_len]

# Metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Hybrid Model Evaluation:\nRMSE: {rmse}\nMAE: {mae}\nR²: {r2}")
