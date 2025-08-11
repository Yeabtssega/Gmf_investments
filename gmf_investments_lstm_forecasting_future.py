import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# ----------------------------
# Settings
# ----------------------------
ASSET = "TSLA"
START_DATE = "2015-01-01"
END_DATE = datetime.date.today().strftime('%Y-%m-%d')
FUTURE_DAYS = 180  # change to 365 for 12 months

# ----------------------------
# Step 1: Load historical data
# ----------------------------
print(f"Fetching {ASSET} stock data from Yahoo Finance...")
df = yf.download(ASSET, start=START_DATE, end=END_DATE)
df = df[['Close']]  # Use 'Close' since auto_adjust=True
df = df.fillna(method='ffill')

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# ----------------------------
# Step 2: Load trained model
# ----------------------------
print("Loading trained LSTM model...")
model = load_model("lstm_tesla_model.h5")  # Make sure you saved it in Task 2

# ----------------------------
# Step 3: Prepare last 60 days as seed input
# ----------------------------
look_back = 60
last_sequence = scaled_data[-look_back:]
predictions_scaled = []

# Predict future prices step-by-step
print(f"Forecasting next {FUTURE_DAYS} days...")
for _ in range(FUTURE_DAYS):
    X_input = np.array([last_sequence])
    pred_scaled = model.predict(X_input, verbose=0)
    predictions_scaled.append(pred_scaled[0, 0])
    last_sequence = np.append(last_sequence[1:], pred_scaled, axis=0)

# ----------------------------
# Step 4: Inverse scale predictions
# ----------------------------
predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

# ----------------------------
# Step 5: Create future date index
# ----------------------------
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq='B')

forecast_df = pd.DataFrame(predictions, index=future_dates, columns=['Forecast'])

# ----------------------------
# Step 6: Plot results
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label="Historical Prices")
plt.plot(forecast_df.index, forecast_df['Forecast'], label="Forecast", color='orange')
plt.title(f"{ASSET} Price Forecast ({FUTURE_DAYS} Days)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()

# Save forecast to CSV
forecast_df.to_csv("tesla_future_forecast.csv")
print("Forecast saved to tesla_future_forecast.csv")
