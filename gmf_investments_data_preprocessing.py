# gmf_investments_data_preprocessing.py

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# -----------------------------
# Setup
# -----------------------------
ASSETS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-07-01"
END_DATE = "2025-07-31"

DATA_DIR = "data"
PLOTS_DIR = "plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Fetching data from Yahoo Finance...")
df = yf.download(ASSETS, start=START_DATE, end=END_DATE, auto_adjust=False)

# Flatten MultiIndex columns
df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
df.reset_index(inplace=True)

print("\nDEBUG: First 10 columns after flattening:\n", df.columns[:10].tolist())

# Fill missing values
df.ffill(inplace=True)
df.bfill(inplace=True)

# -----------------------------
# Daily Returns & Volatility
# -----------------------------
for ticker in ASSETS:
    df[f"{ticker}_Daily_Return"] = df[f"Adj Close_{ticker}"].pct_change()
    df[f"{ticker}_Rolling_Volatility"] = df[f"{ticker}_Daily_Return"].rolling(window=21).std()

# -----------------------------
# Augmented Dickey-Fuller Test
# -----------------------------
print("\nAugmented Dickey-Fuller Test Results:")
for ticker in ASSETS:
    series = df[f"Adj Close_{ticker}"].dropna()
    result = adfuller(series)
    print(f"{ticker}: ADF Statistic = {result[0]:.4f}, p-value = {result[1]:.4f}")

# -----------------------------
# Risk Metrics
# -----------------------------
print("\nRisk Metrics:")
for ticker in ASSETS:
    returns = df[f"{ticker}_Daily_Return"].dropna()
    var_95 = np.percentile(returns, 5) * 100
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    print(f"{ticker}: VaR(95%)={var_95:.4f}%, Sharpe={sharpe_ratio:.2f}")

# -----------------------------
# Save cleaned dataset
# -----------------------------
csv_path = os.path.join(DATA_DIR, "gmf_cleaned_data.csv")
parquet_path = os.path.join(DATA_DIR, "gmf_cleaned_data.parquet")
df.to_csv(csv_path, index=False)
df.to_parquet(parquet_path, index=False)
print(f"\nSaved cleaned dataset to:\n {csv_path}\n {parquet_path}")

# -----------------------------
# Plots
# -----------------------------
sns.set_style("whitegrid")

# Closing Prices
plt.figure(figsize=(12, 6))
for ticker in ASSETS:
    plt.plot(df["Date"], df[f"Adj Close_{ticker}"], label=ticker)
plt.title("Adjusted Closing Prices")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "closing_prices.png"))
plt.close()

# Daily Returns
plt.figure(figsize=(12, 6))
for ticker in ASSETS:
    plt.plot(df["Date"], df[f"{ticker}_Daily_Return"], label=ticker)
plt.title("Daily Returns")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "daily_returns.png"))
plt.close()

# Rolling Volatility
plt.figure(figsize=(12, 6))
for ticker in ASSETS:
    plt.plot(df["Date"], df[f"{ticker}_Rolling_Volatility"], label=f"{ticker} Volatility")
plt.title("21-Day Rolling Volatility")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rolling_volatility.png"))
plt.close()

print(f"Plots saved in '{PLOTS_DIR}' folder.")
