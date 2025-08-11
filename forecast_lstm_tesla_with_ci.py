import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.linear_model import LinearRegression

# Paths
MODEL_PATH = r"C:\Users\HP\gmf_project\models\lstm_tesla_model.h5"
SCALER_PATH = r"C:\Users\HP\gmf_project\models\scaler.pkl"
OUTPUT_CSV = r"C:\Users\HP\gmf_project\forecast_outputs\lstm_forecast_6m_ci.csv"

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Download Tesla data (1 year)
tesla = yf.download("TSLA", period="1y", auto_adjust=True)
close_prices = tesla['Close'].values.reshape(-1, 1)
scaled_data = scaler.transform(close_prices)

window_size = 60
forecast_days = 126
num_simulations = 100  # Number of simulation runs

input_seq_base = scaled_data[-window_size:].copy()

all_preds = []

for sim in range(num_simulations):
    input_seq = input_seq_base.copy()
    preds = []
    for _ in range(forecast_days):
        x_input = input_seq.reshape(1, window_size, 1)
        pred = model.predict(x_input, verbose=0)[0,0]

        # Add Gaussian noise to simulate uncertainty
        noise_std = 0.005  # Adjust noise if needed
        pred_noisy = pred + np.random.normal(0, noise_std)

        preds.append(pred_noisy)
        input_seq = np.append(input_seq[1:], [[pred_noisy]], axis=0)
    all_preds.append(preds)

all_preds = np.array(all_preds)  # shape: (num_simulations, forecast_days)

# Calculate mean and 95% CI
pred_mean = np.mean(all_preds, axis=0)
pred_lower = np.percentile(all_preds, 2.5, axis=0)
pred_upper = np.percentile(all_preds, 97.5, axis=0)

# Inverse transform to price scale
pred_mean_price = scaler.inverse_transform(pred_mean.reshape(-1,1)).flatten()
pred_lower_price = scaler.inverse_transform(pred_lower.reshape(-1,1)).flatten()
pred_upper_price = scaler.inverse_transform(pred_upper.reshape(-1,1)).flatten()

# Forecast dates
last_date = tesla.index[-1]
forecast_dates = pd.bdate_range(last_date, periods=forecast_days+1)[1:]

# Save forecast
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Predicted_Mean': pred_mean_price,
    'CI_Lower_95': pred_lower_price,
    'CI_Upper_95': pred_upper_price
})
forecast_df.set_index('Date', inplace=True)
forecast_df.to_csv(OUTPUT_CSV)

# Plot
plt.figure(figsize=(14,7))
plt.plot(tesla['Close'], label='Historical Close')
plt.plot(forecast_df['Predicted_Mean'], label='Forecast Mean', color='blue')
plt.fill_between(forecast_df.index, forecast_df['CI_Lower_95'], forecast_df['CI_Upper_95'],
                 color='blue', alpha=0.3, label='95% Confidence Interval')
plt.title('TSLA Stock Price Forecast with Confidence Intervals (6 Months)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Trend analysis
X_trend = np.arange(forecast_days).reshape(-1,1)
y_trend = pred_mean_price.reshape(-1,1)
reg = LinearRegression().fit(X_trend, y_trend)
slope = reg.coef_[0][0]
trend = "upward" if slope > 0 else ("downward" if slope < 0 else "stable")

# Volatility analysis
rolling_window = 10
rolling_vol = pd.Series(pred_mean_price).rolling(window=rolling_window).std()
avg_volatility = rolling_vol.mean()

# Confidence interval width trend
ci_width = forecast_df['CI_Upper_95'] - forecast_df['CI_Lower_95']
ci_increasing = ci_width.iloc[-1] > ci_width.iloc[0]

# Historical volatility (scalar fix here)
historical_vol = np.std(tesla['Close'].pct_change().dropna()) * tesla['Close'].mean()

# Summary
print("\n===== Forecast Analysis Summary =====")
print(f"Overall trend: {trend} (slope = {slope:.4f} $/day)")
print(f"Average rolling volatility (window={rolling_window} days): ${avg_volatility:.2f}")
print(f"Confidence interval width {'increases' if ci_increasing else 'does not increase'} over time.")
print("Wider intervals towards the end indicate growing uncertainty in long-term forecasts.")
print("\nMarket Opportunities and Risks:")
if slope > 0:
    print("- Potential price increase suggests buying opportunities.")
else:
    print("- Potential price decline or stagnation suggests caution.")
if avg_volatility > historical_vol:
    print("- Higher than historical volatility indicates elevated risk.")
else:
    print("- Volatility in forecast within historical levels.")

print(f"\nForecast CSV saved at: {OUTPUT_CSV}")
