import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# --- Load Data ---
df = pd.read_csv("data/gmf_cleaned_data.csv", parse_dates=["Date"])
tsla = df[["Date", "Adj Close_TSLA"]].sort_values("Date").set_index("Date")

# --- Train-test split ---
train = tsla[:'2023-12-31']
test = tsla['2024-01-01':]
print(f"Train size: {len(train)}, Test size: {len(test)}")

# --- ARIMA Modeling ---
print("\nFitting ARIMA model...")
arima_model = pm.auto_arima(train, seasonal=False, trace=True,
                            error_action='ignore', suppress_warnings=True,
                            stepwise=True)
print(arima_model.summary())

n_periods = len(test)
arima_forecast = arima_model.predict(n_periods=n_periods)
arima_pred = pd.Series(arima_forecast, index=test.index)

plt.figure(figsize=(12,6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test")
plt.plot(arima_pred.index, arima_pred, label="ARIMA Forecast")
plt.title("Tesla Adjusted Close Price - ARIMA Forecast")
plt.legend()
plt.savefig("plots/arima_forecast.png")
plt.show()

# --- LSTM Modeling ---

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
tsla_scaled = scaler.fit_transform(tsla)

train_size = len(train)
test_size = len(test)

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(tsla_scaled[:train_size])
X_test, y_test = create_sequences(tsla_scaled[train_size-60:])

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("\nTraining LSTM model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

lstm_pred_scaled = model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

lstm_pred_series = pd.Series(lstm_pred.flatten(), index=test.index[60:])

plt.figure(figsize=(12,6))
plt.plot(test.index, test, label="Actual")
plt.plot(lstm_pred_series.index, lstm_pred_series, label="LSTM Forecast")
plt.title("Tesla Adjusted Close Price - LSTM Forecast")
plt.legend()
plt.savefig("plots/lstm_forecast.png")
plt.show()

# --- Evaluation ---
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

arima_mae = mean_absolute_error(test, arima_pred)
arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
arima_mape = mape(test.values.flatten(), arima_pred.values)

test_lstm = test[60:]
lstm_mae = mean_absolute_error(test_lstm, lstm_pred_series)
lstm_rmse = np.sqrt(mean_squared_error(test_lstm, lstm_pred_series))
lstm_mape = mape(test_lstm.values.flatten(), lstm_pred_series.values)

print(f"\nModel Performance on Test Set:")
print(f"ARIMA MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}, MAPE: {arima_mape:.2f}%")
print(f"LSTM MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, MAPE: {lstm_mape:.2f}%")
