import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Ensure 'data' folder exists
if not os.path.exists("data"):
    os.makedirs("data")

# File paths
train_file = "data/train_processed.csv"
arima_forecast_file = "data/arima_forecast.csv"

# Check if train data exists
if not os.path.exists(train_file):
    raise FileNotFoundError(f"{train_file} not found. Run preprocessing first.")

# Load train data
df = pd.read_csv(train_file, parse_dates=["DateTime"])

# Select city (Delhi example, you can loop for all cities)
city_name = "Delhi"
city_data = df[df["City"] == city_name]["AQI"]

# Check if ARIMA forecast exists
if not os.path.exists(arima_forecast_file):
    print("ARIMA forecast not found. Running ARIMA model...")
    # Fit ARIMA
    arima_model = ARIMA(city_data, order=(5,1,0))
    arima_fit = arima_model.fit()

    # Forecast next 24 hours
    forecast = arima_fit.forecast(24)

    # Save forecast
    forecast_df = pd.DataFrame({
        "DateTime": pd.date_range(city_data.index[-1]+1, periods=24, freq='H'),
        "AQI_ARIMA": forecast
    })
    forecast_df.to_csv(arima_forecast_file, index=False)
    print("ARIMA forecast saved to:", arima_forecast_file)
else:
    print("ARIMA forecast found. Loading from file...")
    forecast_df = pd.read_csv(arima_forecast_file, parse_dates=["DateTime"])

# Plot historical + forecast
plt.figure(figsize=(12,6))
plt.plot(city_data[-100:], label="Historical AQI")
plt.plot(forecast_df["DateTime"], forecast_df["AQI_ARIMA"], label="ARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title(f"{city_name} AQI Forecast (ARIMA)")
plt.legend()
plt.show()

# You can now continue with LSTM, Quantum, and hybrid integration
# For example: load LSTM predictions, merge with ARIMA, apply hybrid logic
