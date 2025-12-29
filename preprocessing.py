import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/all_excels_mixed.xlsx", parse_dates=["DateTime"])

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Normalize pollutant and meteorological features
features_to_scale = ["PM2.5","PM10","NO2","SO2","CO","O3","NH3",
                     "Temperature","Humidity","WindSpeed","Pressure","Rainfall"]
scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Create lag feature for AQI
df["AQI_Lag1"] = df["AQI"].shift(1)
df.dropna(inplace=True)

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Save processed data
train_df.to_csv("data/train_processed.csv", index=False)
test_df.to_csv("data/test_processed.csv", index=False)

print("Preprocessing Done")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
