import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load train data
df = pd.read_csv("data/train_processed.csv", parse_dates=["DateTime"])

# Select city (Delhi)
delhi_data = df[df["City"]=="Delhi"]["AQI"]

# Fit ARIMA
arima_model = ARIMA(delhi_data, order=(5,1,0))
arima_fit = arima_model.fit()

# Forecast next 24 hours
forecast = arima_fit.forecast(24)

# Correct plotting
plt.figure(figsize=(12,6))
plt.plot(delhi_data[-100:], label="Historical AQI")  # last 100 points
plt.plot(pd.date_range(delhi_data.index[-1]+1, periods=24, freq='H'), forecast, label="ARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("Delhi AQI Forecast (ARIMA)")
plt.legend()
plt.show()
