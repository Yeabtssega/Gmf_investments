import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === Load TSLA forecast data ===
forecast_csv = r"C:\Users\HP\gmf_project\forecast_outputs\lstm_forecast_6m_ci.csv"
tsla_forecast = pd.read_csv(forecast_csv)

# Detect predicted price column
possible_cols = ["Predicted_Mean", "mean", "Forecast", "Price"]
pred_col = None
for col in possible_cols:
    if col in tsla_forecast.columns:
        pred_col = col
        print(f"Using '{pred_col}' as TSLA predicted price column.")
        break
if pred_col is None:
    raise ValueError("No recognizable price column found in TSLA forecast CSV.")

# Convert predicted prices to returns
tsla_prices = tsla_forecast[pred_col]
tsla_returns = tsla_prices.pct_change().dropna()

# === Download BND and SPY historical data ===
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=180)  # 6 months of data

print(f"Downloading BND and SPY data from {start_date} to {end_date}...")
bnd = yf.download("BND", start=start_date, end=end_date)
spy = yf.download("SPY", start=start_date, end=end_date)

if bnd.empty or spy.empty:
    raise ValueError("BND or SPY data not found. Check ticker or date range.")

# Use 'Adj Close' if exists
bnd_prices = bnd['Adj Close'] if 'Adj Close' in bnd.columns else bnd['Close']
spy_prices = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']

# Compute returns and ensure 1D Series
bnd_ret = bnd_prices.pct_change().dropna().squeeze()
spy_ret = spy_prices.pct_change().dropna().squeeze()
tsla_returns = tsla_returns.squeeze()

# Align lengths
min_len = min(len(tsla_returns), len(bnd_ret), len(spy_ret))
returns_df = pd.DataFrame({
    "TSLA": tsla_returns.tail(min_len).reset_index(drop=True),
    "BND": bnd_ret.tail(min_len).reset_index(drop=True),
    "SPY": spy_ret.tail(min_len).reset_index(drop=True)
})

# === Annualized expected returns ===
expected_returns = returns_df.mean() * 252  # 252 trading days/year

# Covariance matrix
cov_matrix = returns_df.cov() * 252

# === Portfolio optimization functions ===
def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.dot(weights, returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
    p_return, p_vol = portfolio_performance(weights, returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_vol

def portfolio_volatility(weights, returns, cov_matrix):
    return portfolio_performance(weights, returns, cov_matrix)[1]

# Constraints
num_assets = len(expected_returns)
init_guess = num_assets * [1. / num_assets]
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# Max Sharpe Ratio Portfolio
max_sharpe = minimize(neg_sharpe_ratio, init_guess,
                      args=(expected_returns, cov_matrix),
                      method='SLSQP', bounds=bounds,
                      constraints=constraints)

# Minimum Volatility Portfolio
min_vol = minimize(portfolio_volatility, init_guess,
                   args=(expected_returns, cov_matrix),
                   method='SLSQP', bounds=bounds,
                   constraints=constraints)

# === Efficient Frontier ===
target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 50)
frontier_volatility = []
for tr in target_returns:
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_performance(w, expected_returns, cov_matrix)[0] - tr})
    res = minimize(portfolio_volatility, init_guess,
                   args=(expected_returns, cov_matrix),
                   method='SLSQP', bounds=bounds, constraints=cons)
    frontier_volatility.append(res.fun)

# === Plot Efficient Frontier ===
plt.figure(figsize=(10, 6))
plt.plot(frontier_volatility, target_returns, 'g--', label='Efficient Frontier')

# Plot key portfolios
max_sharpe_ret, max_sharpe_vol = portfolio_performance(max_sharpe.x, expected_returns, cov_matrix)
min_vol_ret, min_vol_vol = portfolio_performance(min_vol.x, expected_returns, cov_matrix)

plt.scatter(max_sharpe_vol, max_sharpe_ret, c='red', marker='*', s=200, label='Max Sharpe')
plt.scatter(min_vol_vol, min_vol_ret, c='blue', marker='*', s=200, label='Min Volatility')

plt.title('Efficient Frontier')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()

# === Print Portfolio Recommendations ===
print("\nMax Sharpe Ratio Portfolio Allocation")
print(pd.Series(max_sharpe.x, index=expected_returns.index))
print("Expected annual return:", round(max_sharpe_ret, 4))
print("Annual volatility:", round(max_sharpe_vol, 4))
print("Sharpe Ratio:", round((max_sharpe_ret - 0.02) / max_sharpe_vol, 4))

print("\nMinimum Volatility Portfolio Allocation")
print(pd.Series(min_vol.x, index=expected_returns.index))
print("Expected annual return:", round(min_vol_ret, 4))
print("Annual volatility:", round(min_vol_vol, 4))
print("Sharpe Ratio:", round((min_vol_ret - 0.02) / min_vol_vol, 4))
