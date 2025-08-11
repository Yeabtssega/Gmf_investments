GMF Investment Project
Overview
This project aims to analyze, model, and forecast stock market data focusing primarily on Tesla (TSLA), with supplementary data from BND and SPY ETFs to create a balanced portfolio. The workflow is structured in five key tasks:

Task 1: Preprocess and Explore the Data
Objective
Load, clean, and understand the historical financial data to prepare for predictive modeling and portfolio optimization.

Data Sources
Historical price data extracted using Yahoo Finance (YFinance) for:

TSLA — high return potential with high volatility.

BND — a bond ETF providing stability and low risk.

SPY — a diversified S&P 500 ETF representing moderate risk exposure.

Steps
Data Cleaning and Validation

Checked for missing values and inconsistent data types.

Handled missing values through interpolation, filling, or removal where appropriate.

Normalized/scaled data where necessary for modeling compatibility.

Exploratory Data Analysis (EDA)

Visualized closing prices over time to observe long-term trends.

Calculated and plotted daily percentage changes to assess volatility.

Analyzed short-term trends via rolling means and standard deviations.

Identified outliers and anomalies in returns.

Examined days with unusually high or low returns to detect extreme market events.

Statistical Testing

Applied the Augmented Dickey-Fuller (ADF) test to check for stationarity of price and return series.

Discussed implications of stationarity results for time series modeling (e.g., need for differencing in ARIMA).

Risk Metrics and Insights

Computed foundational risk measures including:

Value at Risk (VaR)

Sharpe Ratio

Summarized overall direction of Tesla’s stock price and volatility patterns.

Task 2: Develop Time Series Forecasting Models
Built and compared forecasting models:

Classical statistical model: ARIMA/SARIMA.

Deep learning model: LSTM.

Trained models on chronological splits to preserve temporal order.

Optimized hyperparameters using grid search and model-specific tuning.

Evaluated models on test data using MAE, RMSE, and MAPE.

Discussed trade-offs between model complexity, accuracy, and interpretability.

Task 3: Forecast Future Market Trends
Generated 6–12 month future forecasts for Tesla stock price using the selected model.

Visualized forecasts with confidence intervals.

Analyzed long-term trends, volatility, and uncertainty.

Identified market opportunities and risks based on forecast results.

Task 4: Optimize Portfolio Based on Forecast
Used forecasted returns for TSLA combined with historical returns of BND and SPY.

Computed covariance matrix of daily returns for portfolio risk estimation.

Ran optimization simulations to generate the Efficient Frontier.

Identified and marked:

Maximum Sharpe Ratio Portfolio.

Minimum Volatility Portfolio.

Recommended optimal portfolio weights and justified choice based on risk-return trade-offs.

Task 5: Strategy Backtesting
Defined backtesting period (last year of data).

Created benchmark portfolio (60% SPY / 40% BND).

Simulated performance of the optimized portfolio over the backtesting period.

Compared cumulative returns and Sharpe Ratios to benchmark.

Summarized performance insights and viability of the forecast-driven strategy.

