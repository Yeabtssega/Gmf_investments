import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Parameters
ASSET = 'TSLA'
START_DATE = '2015-01-01'
END_DATE = '2024-01-01'  # adjust as you want

print(f"Fetching {ASSET} stock data from Yahoo Finance...")
df = yf.download(ASSET, start=START_DATE, end=END_DATE, auto_adjust=True)  # auto_adjust=True means no 'Adj Close' column

print("Columns downloaded:", df.columns)

# Use Close price (already adjusted)
tsla_series = df['Close']

# Ensure datetime index with freq for statsmodels (daily frequency)
tsla_series.index = pd.to_datetime(tsla_series.index)
tsla_series = tsla_series.asfreq('B')  # business day frequency; fills missing days with NaN
tsla_series = tsla_series.fillna(method='ffill')  # forward-fill missing values

# Split into train and test chronologically
split_date = '2023-01-01'
train = tsla_series[:split_date]
test = tsla_series[split_date:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

print("Fitting ARIMA model...")

# Manually chosen ARIMA order (p,d,q). You can tune this or use auto_arima externally
p, d, q = 5, 1, 0
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Forecast test period length
forecast = model_fit.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_val = mape(test.values, forecast.values)

print("\nARIMA Forecast Evaluation:")
print(f"MAE = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAPE = {mape_val:.2f}%")

# Plot actual vs forecast
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test (Actual)')
plt.plot(test.index, forecast, label='ARIMA Forecast')
plt.title(f'{ASSET} Stock Price Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('plots/arima_forecast.png')
plt.show()
