import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Backtest parameters
start_date = "2024-08-01"
end_date = "2025-07-31"
risk_free_rate = 0.02
trading_days = 252

# Fixed portfolio weights from Task 4 recommendation (BLENDED Max Sharpe, capped TSLA)
weights = {
    "TSLA": 0.50,
    "BND": 0.50,
    "SPY": 0.00
}

# Benchmark weights (60% SPY / 40% BND)
benchmark_weights = {
    "TSLA": 0.00,
    "BND": 0.40,
    "SPY": 0.60
}

tickers = list(weights.keys())

# Download adjusted close prices for backtesting window
print(f"Downloading historical data from {start_date} to {end_date} ...")
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

# Check data
if data.isnull().values.any():
    print("Warning: Missing data detected. Forward filling missing values.")
    data = data.fillna(method='ffill')

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Calculate strategy portfolio daily returns
weights_array = np.array([weights[t] for t in tickers])
strategy_returns = daily_returns.dot(weights_array)

# Calculate benchmark portfolio daily returns
benchmark_array = np.array([benchmark_weights[t] for t in tickers])
benchmark_returns = daily_returns.dot(benchmark_array)

# Calculate cumulative returns
strategy_cum = (1 + strategy_returns).cumprod()
benchmark_cum = (1 + benchmark_returns).cumprod()

# Calculate total return over backtest period
strategy_total_return = strategy_cum.iloc[-1] - 1
benchmark_total_return = benchmark_cum.iloc[-1] - 1

# Annualized Sharpe Ratio function
def sharpe_ratio(returns, rf=0.02, periods=252):
    excess_ret = returns - rf / periods
    return np.sqrt(periods) * excess_ret.mean() / excess_ret.std()

strategy_sharpe = sharpe_ratio(strategy_returns, risk_free_rate, trading_days)
benchmark_sharpe = sharpe_ratio(benchmark_returns, risk_free_rate, trading_days)

# Plot cumulative returns
plt.figure(figsize=(12,6))
plt.plot(strategy_cum, label="Model-driven Portfolio (Task 4)")
plt.plot(benchmark_cum, label="Benchmark (60% SPY / 40% BND)")
plt.title("Backtest Cumulative Returns (Aug 2024 - July 2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Print summary
print("\nBacktest Summary:")
print(f"Strategy total return: {strategy_total_return:.2%}")
print(f"Strategy annualized Sharpe Ratio: {strategy_sharpe:.4f}")
print(f"Benchmark total return: {benchmark_total_return:.2%}")
print(f"Benchmark annualized Sharpe Ratio: {benchmark_sharpe:.4f}")

if strategy_total_return > benchmark_total_return:
    print("\nResult: Your strategy outperformed the benchmark over the backtest period.")
else:
    print("\nResult: Your strategy underperformed the benchmark over the backtest period.")

print("\nInitial backtest suggests that the model-driven portfolio:")
print("- Captures forecast insights (via TSLA weighting)")
print("- Balances risk with BND allocation")
print("- Should be monitored and rebalanced regularly for best real-world performance")
