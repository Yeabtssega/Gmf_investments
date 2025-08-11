import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Download adjusted close prices
print(f"Downloading historical data from {start_date} to {end_date} ...")
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

if data.isnull().values.any():
    print("Warning: Missing data detected. Forward filling missing values.")
    data = data.fillna(method='ffill')

# Daily returns
daily_returns = data.pct_change().dropna()

# Generate month start dates for rebalancing
month_starts = daily_returns.resample('MS').first().index

# Initialize portfolio values
initial_value = 1_000_000  # $1 million start
portfolio_values = []
benchmark_values = []

# Start portfolio values
current_portfolio_value = initial_value
current_benchmark_value = initial_value

# Keep track of shares held for portfolio and benchmark
portfolio_shares = None
benchmark_shares = None

for i in range(len(month_starts)):
    # Define start and end dates for this month period
    period_start = month_starts[i]
    period_end = month_starts[i+1] if i+1 < len(month_starts) else daily_returns.index[-1]

    # Slice price data for period
    prices_period = data.loc[period_start:period_end]

    if portfolio_shares is None:
        # Initial purchase of shares for portfolio at period_start prices
        prices_start = prices_period.iloc[0]
        portfolio_shares = pd.Series({t: (current_portfolio_value * weights[t]) / prices_start[t] for t in tickers})

        prices_start_bench = prices_period.iloc[0]
        benchmark_shares = pd.Series({t: (current_benchmark_value * benchmark_weights[t]) / prices_start_bench[t] for t in tickers})
    else:
        # Rebalance at period_start prices to fixed weights
        prices_start = prices_period.iloc[0]
        current_portfolio_value = sum(portfolio_shares * prices_start)
        portfolio_shares = pd.Series({t: (current_portfolio_value * weights[t]) / prices_start[t] for t in tickers})

        current_benchmark_value = sum(benchmark_shares * prices_start)
        benchmark_shares = pd.Series({t: (current_benchmark_value * benchmark_weights[t]) / prices_start[t] for t in tickers})

    # Compute daily portfolio values during the period
    portfolio_vals_period = prices_period.dot(portfolio_shares)
    benchmark_vals_period = prices_period.dot(benchmark_shares)

    # Append all but last day to the results list
    if i < len(month_starts) - 1:
        portfolio_values.append(portfolio_vals_period.iloc[:-1])
        benchmark_values.append(benchmark_vals_period.iloc[:-1])
    else:
        # Last period: include all days
        portfolio_values.append(portfolio_vals_period)
        benchmark_values.append(benchmark_vals_period)

# Concatenate all monthly period values into one series
portfolio_values = pd.concat(portfolio_values)
benchmark_values = pd.concat(benchmark_values)

# Compute daily returns for strategy and benchmark
strategy_returns = portfolio_values.pct_change().dropna()
benchmark_returns = benchmark_values.pct_change().dropna()

# Calculate total returns
strategy_total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
benchmark_total_return = benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1

# Annualized Sharpe Ratio function
def sharpe_ratio(returns, rf=0.02, periods=252):
    excess_ret = returns - rf / periods
    return np.sqrt(periods) * excess_ret.mean() / excess_ret.std()

strategy_sharpe = sharpe_ratio(strategy_returns, risk_free_rate, trading_days)
benchmark_sharpe = sharpe_ratio(benchmark_returns, risk_free_rate, trading_days)

# Plot cumulative returns
plt.figure(figsize=(12,6))
plt.plot(portfolio_values / portfolio_values.iloc[0], label="Rebalanced Model-driven Portfolio")
plt.plot(benchmark_values / benchmark_values.iloc[0], label="Benchmark (60% SPY / 40% BND)")
plt.title("Backtest with Monthly Rebalancing (Aug 2024 - Jul 2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Summary
print("\nBacktest Summary with Monthly Rebalancing:")
print(f"Strategy total return: {strategy_total_return:.2%}")
print(f"Strategy annualized Sharpe Ratio: {strategy_sharpe:.4f}")
print(f"Benchmark total return: {benchmark_total_return:.2%}")
print(f"Benchmark annualized Sharpe Ratio: {benchmark_sharpe:.4f}")

if strategy_total_return > benchmark_total_return:
    print("\nResult: Your rebalanced strategy outperformed the benchmark over the backtest period.")
else:
    print("\nResult: Your rebalanced strategy underperformed the benchmark over the backtest period.")

print("\nRecommendation:")
print("- Monthly rebalancing helps adjust portfolio weights to your forecast-driven targets.")
print("- Continue refining your model and consider transaction costs next.")
