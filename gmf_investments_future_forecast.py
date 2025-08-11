# gmf_investments_future_forecast.py
"""
Forecast future Tesla prices (6-month and 12-month) using LSTM and ARIMA.
Produces confidence intervals for LSTM via Monte-Carlo residual simulation,
and uses statsmodels ARIMA for baseline with analytic CI.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

# ------- Settings -------
ASSET = "TSLA"
START_DATE = "2015-01-01"
END_DATE = "2025-07-31"       # up-to-date end date (you can change)
SEQ_LEN = 60                 # lookback days for LSTM
EPOCHS = 30                  # increase if you want longer training
BATCH = 32
MC_SIMS = 300                # Monte Carlo simulations for CI
HORIZONS = {"6m": 126, "12m": 252}  # trading days approx
P_DQ = (5, 1, 0)             # ARIMA p,d,q (you can tune)

OUT_DIR = "forecast_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Fix randomness to aid reproducibility (not perfect for TF on all machines)
np.random.seed(42)
tf.random.set_seed(42)

# ------- Fetch data -------
print("Downloading historical data...")
df = yf.download(ASSET, start=START_DATE, end=END_DATE, progress=False)  # auto_adjust default True
if df.empty:
    raise SystemExit("No data downloaded. Check network or ticker/date range.")

prices = df["Close"].ffill()   # use Close (auto-adjusted by yfinance)
prices.index = pd.to_datetime(prices.index)

# Force business-day frequency and forward fill missing trading days
prices = prices.asfreq("B").ffill()

# ------- Prepare training data for LSTM -------
# Use full history up to latest date for training (we want the model to leverage max info)
series = prices.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(series_scaled, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test internal split not strictly necessary here since we train on all for forecasting;
# but we can keep last chunk for quick residual estimate:
resid_hold = int(0.1 * len(y))  # 10% tail for residual estimation
X_train, y_train = X[:-resid_hold], y[:-resid_hold]
X_resid, y_resid = X[-resid_hold:], y[-resid_hold:]  # used to compute residual std

# ------- Build and train LSTM -------
print("Building and training LSTM (this may take a few minutes)...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=1)

# compute residuals on held tail to estimate noise level
pred_tail = model.predict(X_resid).flatten()
residuals = y_resid - pred_tail
resid_std = np.std(residuals)
print(f"Estimated residual std (scaled space): {resid_std:.6f}")

# ------- Function: iterative forecasting with LSTM for n steps -------
def forecast_lstm(model, last_window_scaled, n_steps):
    """
    Iteratively predict n_steps ahead using the LSTM and a sliding window.
    last_window_scaled: shape (seq_len,) scaled
    returns: numpy array of scaled predictions length n_steps
    """
    preds = []
    window = last_window_scaled.copy()
    for _ in range(n_steps):
        inp = window.reshape((1, SEQ_LEN, 1))
        p = model.predict(inp)[0, 0]
        preds.append(p)
        # slide window
        window = np.roll(window, -1)
        window[-1] = p
    return np.array(preds)

# Last observed window (scaled)
last_window = series_scaled[-SEQ_LEN:, 0]

# ------- Monte Carlo sims to build CI -------
def mc_forecasts(model, last_window, n_steps, sims=MC_SIMS):
    sim_paths = np.zeros((sims, n_steps))
    for s in range(sims):
        window = last_window.copy()
        path = []
        for t in range(n_steps):
            p = model.predict(window.reshape((1, SEQ_LEN, 1)))[0, 0]
            # add Gaussian residual noise scaled by resid_std
            p_noisy = p + np.random.normal(0, resid_std)
            path.append(p_noisy)
            window = np.roll(window, -1)
            window[-1] = p_noisy
        sim_paths[s, :] = path
    return sim_paths

# ------- Run MC for each horizon, inverse transform to price scale -------
results = {}
for name, days in HORIZONS.items():
    print(f"\nForecasting {name} horizon ({days} trading days)...")
    sims = mc_forecasts(model, last_window, days, sims=MC_SIMS)
    # convert scaled sims back to price space
    sims_reshaped = sims.reshape(-1, 1)
    sims_prices = scaler.inverse_transform(sims_reshaped).reshape(MC_SIMS, days)
    # compute median and percentile bands
    median = np.median(sims_prices, axis=0)
    p05 = np.percentile(sims_prices, 5, axis=0)
    p95 = np.percentile(sims_prices, 95, axis=0)
    p25 = np.percentile(sims_prices, 25, axis=0)
    p75 = np.percentile(sims_prices, 75, axis=0)
    # dates for forecast
    last_date = prices.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    results[name] = {
        "dates": future_dates,
        "median": median,
        "p05": p05,
        "p95": p95,
        "p25": p25,
        "p75": p75,
        "sims_prices": sims_prices
    }
    # Save CSV
    out_df = pd.DataFrame({
        "date": future_dates,
        "median": median,
        "p05": p05,
        "p95": p95,
        "p25": p25,
        "p75": p75
    })
    out_df.to_csv(os.path.join(OUT_DIR, f"lstm_forecast_{name}.csv"), index=False)
    print(f"Saved LSTM {name} forecast CSV.")

# ------- ARIMA baseline (optional) -------
print("\nFitting ARIMA baseline on full history (may be slow)...")
arima_order = P_DQ
arima = ARIMA(prices, order=arima_order).fit()
# we request forecast and conf_int for largest horizon
max_h = max(HORIZONS.values())
arima_forecast = arima.get_forecast(steps=max_h)
arima_mean = arima_forecast.predicted_mean
arima_ci = arima_forecast.conf_int(alpha=0.10)  # 90% CI -> similar to p05/p95

# Save ARIMA CSV trimmed for 6m and 12m
for name, days in HORIZONS.items():
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=days)
    arima_mean_slice = arima_mean.iloc[:days].values
    arima_ci_slice = arima_ci.iloc[:days].values
    df_arima = pd.DataFrame({
        "date": future_dates,
        "mean": arima_mean_slice,
        "ci_lower": arima_ci_slice[:, 0],
        "ci_upper": arima_ci_slice[:, 1]
    })
    df_arima.to_csv(os.path.join(OUT_DIR, f"arima_forecast_{name}.csv"), index=False)
    print(f"Saved ARIMA {name} forecast CSV.")

# ------- Plot results for each horizon -------
for name, info in results.items():
    dates = info["dates"]
    plt.figure(figsize=(12,6))
    # historical last 2 years for context
    hist_start = prices.index[-(252*2):] if len(prices) > 252*2 else prices.index
    plt.plot(prices.loc[hist_start], label="Historical")
    plt.plot(dates, info["median"], label="LSTM median forecast", color="tab:orange")
    plt.fill_between(dates, info["p05"], info["p95"], color="orange", alpha=0.2, label="LSTM 90% CI")
    plt.fill_between(dates, info["p25"], info["p75"], color="orange", alpha=0.35, label="LSTM 50% IQR")
    # overlay ARIMA mean + CI if available
    arima_csv = os.path.join(OUT_DIR, f"arima_forecast_{name}.csv")
    if os.path.exists(arima_csv):
        df_a = pd.read_csv(arima_csv, parse_dates=["date"]).set_index("date")
        plt.plot(df_a.index, df_a["mean"], label="ARIMA mean", color="tab:green")
        plt.fill_between(df_a.index, df_a["ci_lower"], df_a["ci_upper"], color="green", alpha=0.15, label="ARIMA 90% CI")
    plt.title(f"TSLA Forecast - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"forecast_{name}.png"))
    plt.close()
    print(f"Saved plot: {os.path.join(PLOTS_DIR, f'forecast_{name}.png')}")

# ------- Quick analysis summary (text) -------
def ci_width_percent(ci_lower, ci_upper, median):
    return ((ci_upper - ci_lower) / median) * 100

summary_lines = []
for name, info in results.items():
    med = info["median"]
    p05 = info["p05"]
    p95 = info["p95"]
    # compute mean CI width percent across horizon
    width_pct = np.mean((p95 - p05) / med) * 100
    trend_dir = np.sign(med[-1] - med[0])
    trend_text = "upward" if trend_dir > 0 else ("downward" if trend_dir < 0 else "flat")
    summary_lines.append(f"{name} forecast: median trend appears {trend_text}. Mean 90% CI width ≈ {width_pct:.1f}% of median.")
summary = "\n".join(summary_lines)
print("\nSUMMARY:")
print(summary)

# Save summary
with open(os.path.join(OUT_DIR, "forecast_summary.txt"), "w") as f:
    f.write(summary)

print("\nAll done. Outputs saved in folder:", OUT_DIR)
