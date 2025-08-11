# portfolio_optimization_final.py
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize

# ---------------- CONFIG ----------------
forecast_csv = r"C:\Users\HP\gmf_project\forecast_outputs\lstm_forecast_6m_ci.csv"
output_csv = r"C:\Users\HP\gmf_project\portfolio_optimization_results.csv"
risk_free_rate = 0.02
trading_days = 252
tsla_max_weight = 0.5   # cap TSLA at 50%
blend_alpha = 0.5       # 0.5 => 50% forecast + 50% historical

# ---------------- LOAD TSLA FORECAST ----------------
fc = pd.read_csv(forecast_csv)
numeric_cols = fc.select_dtypes(include=[np.number]).columns
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found in forecast CSV.")
price_col = numeric_cols[0]
print(f"Using '{price_col}' from forecast CSV as predicted price column.")

# compute forecasted daily returns from predicted prices
tsla_forecast_returns = fc[price_col].pct_change().dropna().reset_index(drop=True)
forecast_days = len(tsla_forecast_returns)

# ---------------- DOWNLOAD HISTORICAL PRICES (for cov & hist returns) ----------------
# We'll take ~1 year history for cov matrix and TSLA historical return
hist_end = datetime.today()
hist_start = hist_end - timedelta(days=365)
hist_start_str = hist_start.strftime("%Y-%m-%d")
hist_end_str = hist_end.strftime("%Y-%m-%d")

print(f"Downloading historical prices (TSLA,BND,SPY) from {hist_start_str} to {hist_end_str} ...")
tickers = ["TSLA", "BND", "SPY"]
data = yf.download(tickers, start=hist_start_str, end=hist_end_str, auto_adjust=True, progress=False)

if data.empty:
    raise ValueError("Failed to download historical data for TSLA/BND/SPY.")

# Some yfinance returns multiindex columns; pick 'Close' if exists or outer level
if ('Close' in data.columns):
    close = data['Close']
else:
    # if single-column format
    close = data

# Ensure all tickers present
for t in tickers:
    if t not in close.columns:
        raise ValueError(f"{t} not found in downloaded historical data.")

# historical daily returns (aligned)
hist_returns = close.pct_change().dropna()
# slice last 'forecast_days' for alignment if needed
min_hist_len = min(len(hist_returns), forecast_days)
hist_returns = hist_returns.tail(min_hist_len).reset_index(drop=True)

# historical TSLA mean annual return
tsla_hist_daily_mean = hist_returns['TSLA'].mean()
tsla_hist_annual = tsla_hist_daily_mean * trading_days

# covariance matrix (annualized) from historical returns (TSLA, BND, SPY)
cov_matrix = hist_returns.cov() * trading_days

# ---------------- EXPECTED RETURNS: PURE FORECAST & BLENDED ----------------
# Forecast annualized (from forecast returns)
tsla_forecast_annual = tsla_forecast_returns.mean() * trading_days

# Blended TSLA expected return
tsla_blended_annual = blend_alpha * tsla_forecast_annual + (1 - blend_alpha) * tsla_hist_annual

# BND & SPY expected returns: use recent historical daily mean (annualized)
bnd_annual = hist_returns['BND'].mean() * trading_days
spy_annual = hist_returns['SPY'].mean() * trading_days

# Build expected return vectors
expected_pure = np.array([tsla_forecast_annual, bnd_annual, spy_annual])
expected_blended = np.array([tsla_blended_annual, bnd_annual, spy_annual])

print("\nExpected annual returns (pure forecast / blended):")
print(f"TSLA forecast annual: {tsla_forecast_annual:.2%}")
print(f"TSLA historical annual: {tsla_hist_annual:.2%}")
print(f"TSLA blended annual: {tsla_blended_annual:.2%}")
print(f"BND annual: {bnd_annual:.2%}, SPY annual: {spy_annual:.2%}")

# ---------------- PORTFOLIO FUNCTIONS ----------------
def port_perf(weights, exp_returns, cov):
    r = np.dot(weights, exp_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return r, vol

def neg_sharpe(weights, exp_returns, cov, rf=risk_free_rate):
    r, vol = port_perf(weights, exp_returns, cov)
    return -(r - rf) / vol

def vol_only(weights, exp_returns, cov):
    return port_perf(weights, exp_returns, cov)[1]

# ---------------- CONSTRAINTS & BOUNDS (TSLA cap) ----------------
num_assets = 3
init = num_assets * [1.0 / num_assets]
bounds = ((0.0, tsla_max_weight), (0.0, 1.0), (0.0, 1.0))  # TSLA bound first
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

# ---------------- OPTIMIZE FOR a given expected return vector ----------------
def optimize_portfolios(exp_returns):
    # Max Sharpe
    res_sharpe = minimize(neg_sharpe, init, args=(exp_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=cons)
    # Min Vol
    res_minvol = minimize(vol_only, init, args=(exp_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=cons)
    return res_sharpe, res_minvol

res_pure_sharpe, res_pure_minvol = optimize_portfolios(expected_pure)
res_blend_sharpe, res_blend_minvol = optimize_portfolios(expected_blended)

# ---------------- COMPUTE METRICS ----------------
def summarize(res, exp_returns):
    w = res.x
    r, vol = port_perf(w, exp_returns, cov_matrix)
    sharpe = (r - risk_free_rate) / vol
    return {'weights': w, 'return': r, 'vol': vol, 'sharpe': sharpe}

sum_pure_sharpe = summarize(res_pure_sharpe, expected_pure)
sum_pure_minvol = summarize(res_pure_minvol, expected_pure)

sum_blend_sharpe = summarize(res_blend_sharpe, expected_blended)
sum_blend_minvol = summarize(res_blend_minvol, expected_blended)

# ---------------- PRINT RESULTS (comparison) ----------------
def print_summary(tag, s):
    print(f"\n--- {tag} ---")
    print(f"Weights -> TSLA: {s['weights'][0]:.2%}, BND: {s['weights'][1]:.2%}, SPY: {s['weights'][2]:.2%}")
    print(f"Expected Annual Return: {s['return']:.2%}")
    print(f"Annual Volatility: {s['vol']:.2%}")
    print(f"Sharpe Ratio: {s['sharpe']:.4f}")

print_summary("PURE FORECAST - Max Sharpe", sum_pure_sharpe)
print_summary("PURE FORECAST - Min Volatility", sum_pure_minvol)
print_summary("BLENDED (50/50) - Max Sharpe", sum_blend_sharpe)
print_summary("BLENDED (50/50) - Min Volatility", sum_blend_minvol)

# ---------------- SAVE TO CSV ----------------
rows = []
for name, s, exp_vec in [
    ("Pure_Max_Sharpe", sum_pure_sharpe, expected_pure),
    ("Pure_Min_Vol", sum_pure_minvol, expected_pure),
    ("Blend_Max_Sharpe", sum_blend_sharpe, expected_blended),
    ("Blend_Min_Vol", sum_blend_minvol, expected_blended),
]:
    rows.append({
        "Portfolio": name,
        "TSLA_weight": s['weights'][0],
        "BND_weight": s['weights'][1],
        "SPY_weight": s['weights'][2],
        "Expected_Return": s['return'],
        "Volatility": s['vol'],
        "Sharpe": s['sharpe']
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(output_csv, index=False)
print(f"\nSaved optimization results to: {output_csv}")

# ---------------- PLOT EFFICIENT FRONTIER (using blended expected returns) ----------------
# We'll compute the frontier by solving min volatility for target returns
target_rets = np.linspace(expected_blended.min(), expected_blended.max(), 50)
frontier_vols = []
for tr in target_rets:
    cons_tr = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w, tr=tr: port_perf(w, expected_blended, cov_matrix)[0] - tr}
    )
    res = minimize(vol_only, init, args=(expected_blended, cov_matrix),
                   method='SLSQP', bounds=bounds, constraints=cons_tr)
    if res.success:
        frontier_vols.append(res.fun)
    else:
        frontier_vols.append(np.nan)

plt.figure(figsize=(10,6))
plt.plot(frontier_vols, target_rets, 'k--', label='Efficient Frontier (blended)')
# plot the portfolios we computed (blended max sharpe & min vol)
plt.scatter(sum_blend_sharpe['vol'], sum_blend_sharpe['return'], c='red', marker='*', s=150, label='Blend Max Sharpe')
plt.scatter(sum_blend_minvol['vol'], sum_blend_minvol['return'], c='blue', marker='*', s=150, label='Blend Min Vol')
plt.xlabel('Annual Volatility')
plt.ylabel('Expected Annual Return')
plt.title('Efficient Frontier (blended expected returns)')
plt.legend()
plt.grid(True)
plt.show()

# ---------------- RECOMMENDATION ----------------
# Choose the blended Max Sharpe as the recommended (practical compromise)
recommended = sum_blend_sharpe
print("\n===== RECOMMENDATION =====")
print("I recommend the BLENDED (50/50 forecast/historical) Max Sharpe portfolio (TSLA cap 50%).")
print_summary("Recommended Portfolio (BLENDED Max Sharpe)", recommended)
