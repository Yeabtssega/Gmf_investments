import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
ASSET = 'TSLA'
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'  # adjust if you want
SEQ_LENGTH = 60  # days to look back for prediction
EPOCHS = 20
BATCH_SIZE = 32

print("Fetching Tesla stock data from Yahoo Finance...")
df = yf.download(ASSET, start=START_DATE, end=END_DATE)  # auto_adjust=True default
print(f"Data downloaded: {len(df)} rows")

# Use 'Close' price
tsla_series = df['Close'].fillna(method='ffill')

# Split chronologically: train/test split by date
split_date = '2023-01-01'
train = tsla_series.loc[:split_date]
test = tsla_series.loc[split_date:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='loss', patience=3)

print("Training LSTM model...")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=2)

# Predict on test set
y_pred_scaled = model.predict(X_test)

# Inverse transform to get original scale
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred_scaled)

# Define safe MAPE (avoid division by zero)
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Evaluate
mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mape = safe_mape(y_test_orig, y_pred_orig)

print("\nLSTM Forecast Evaluation:")
print(f"MAE = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAPE = {mape:.2f}%")
