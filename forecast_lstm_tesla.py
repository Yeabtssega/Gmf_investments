import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Paths to saved files
MODEL_PATH = r"C:\Users\HP\gmf_project\models\lstm_tesla_model.h5"
SCALER_PATH = r"C:\Users\HP\gmf_project\models\scaler.pkl"
OUTPUT_CSV = r"C:\Users\HP\gmf_project\forecast_outputs\lstm_forecast_6m.csv"

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Download latest Tesla data (1 year)
tesla = yf.download("TSLA", period="1y")
close_prices = tesla['Close'].values.reshape(-1, 1)

# Scale data using saved scaler
scaled_data = scaler.transform(close_prices)

# Prepare input sequence for forecasting (last 60 days)
window_size = 60
input_seq = scaled_data[-window_size:]

forecast_days = 126
predictions = []

for _ in range(forecast_days):
    x_input = input_seq.reshape(1, window_size, 1)
    pred = model.predict(x_input, verbose=0)[0,0]
    predictions.append(pred)
    input_seq = np.append(input_seq[1:], [[pred]], axis=0)

# Inverse transform predictions
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# Create forecast dates (business days)
last_date = tesla.index[-1]
forecast_dates = pd.bdate_range(last_date, periods=forecast_days+1)[1:]

# Prepare forecast dataframe
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': predicted_prices.flatten()})
forecast_df.set_index('Date', inplace=True)

# Save forecast CSV
forecast_df.to_csv(OUTPUT_CSV)

# Plot historical + forecast
plt.figure(figsize=(14,7))
plt.plot(tesla['Close'], label='Historical Close')
plt.plot(forecast_df['Predicted_Close'], label='6-Month Forecast')
plt.title('TSLA Stock Price Forecast - 6 Months')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

print(f"Forecast saved to {OUTPUT_CSV}")
