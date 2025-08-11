import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Step 1: Download Tesla data (last 2 years)
tesla = yf.download('TSLA', period='2y')
close_prices = tesla['Close'].values.reshape(-1, 1)

# Step 2: Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Step 3: Prepare training data (using 60 days window)
window_size = 60
X = []
y = []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM input [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 4: Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Step 6: Save model and scaler
model.save(r"C:\Users\HP\gmf_project\models\lstm_tesla_model.h5")
joblib.dump(scaler, r"C:\Users\HP\gmf_project\models\scaler.pkl")

print("Training complete and model/scaler saved.")
