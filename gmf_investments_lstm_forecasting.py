import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math

# Parameters
ASSET = "TSLA"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
LOOKBACK = 60  # days for sequence length

# Step 1: Download Tesla stock data
print(f"Fetching {ASSET} stock data from Yahoo Finance...")
df = yf.download(ASSET, start=START_DATE, end=END_DATE)
print(f"Data downloaded: {len(df)} rows")

# Fix: use 'Close' instead of 'Adj Close'
df = df[['Close']].dropna()
df = df.rename(columns={'Close': 'Price'})

# Step 2: Train/test split
train_size = int(len(df) * 0.9)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# Step 3: Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Step 4: Create sequences
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_test, y_test = create_sequences(test_scaled, LOOKBACK)

# Reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 5: Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next price

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

# Step 7: Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Step 8: Evaluate
actual_prices = test[LOOKBACK:].values

mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print("\nLSTM Forecast Evaluation:")
print(f"MAE = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAPE = {mape:.2f}%")

# Step 9: Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size+LOOKBACK:], actual_prices, color='blue', label='Actual Price')
plt.plot(df.index[train_size+LOOKBACK:], predicted_prices, color='red', label='Predicted Price')
plt.title(f'{ASSET} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price USD')
plt.legend()
plt.show()
