import pandas as pd

# Load ARIMA forecast
arima = pd.read_csv("data/arima_forecast.csv")

# Rename column to match evaluation.py
arima.rename(columns={"AQI_ARIMA":"Hybrid_AQI"}, inplace=True)

# Save as hybrid file
arima.to_csv("data/hybrid_forecast.csv", index=False)
print("Dummy hybrid file created with correct column name.")
