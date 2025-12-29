import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load preprocessed train data
df = pd.read_csv("data/train_processed.csv", parse_dates=["DateTime"])

# Select one city (Delhi)
city_df = df[df["City"]=="Delhi"]
features = ["PM2.5","PM10","NO2","SO2","CO","O3","NH3","Temperature","Humidity","WindSpeed","Pressure","Rainfall"]
X = city_df[features].values
y = city_df["AQI"].values

# Create sequences (lookback=24 hours)
lookback = 24
X_seq, y_seq = [], []
for i in range(lookback, len(X)):
    X_seq.append(X[i-lookback:i])
    y_seq.append(y[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Train-test split
split = int(0.8*len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(lookback, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Predict LSTM AQI
y_pred = model.predict(X_test)
np.savetxt("data/lstm_forecast.csv", y_pred, delimiter=",")
