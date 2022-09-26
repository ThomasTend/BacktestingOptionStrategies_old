"""
Author: Thomas Tendron

This file is a pipeline for the backtesting framework.
"""

# import relevant classes
from backtesting import *
from features import *

""" PIPELINE """

# Create backtest
bt = Backtest(asset = "stock", asset_id = "msft", target="Close", period="1y", days_to_pred = 2 , num_lag_features = 7, historical_start_idx=250)
# Plot historical stock price
bt.plot_historical()
# Plot prediction on same plot as historical price for visual comparison
bt.plot_backtest()
# Plot future prediction on same plot
bt.plot_future_prediction()
# Call plt.show()
bt.plot_all()
# Print MSE on test set
bt.score()
# Compute volatility:
vol = Volatility(bt.price)
vol.get_historical_volatility()
# Plot log change
vol.plot_log_change()
# Compute implied volatility
print("Implied Volatility is {}".format(vol.get_implied_volatility(S_0=67.96, OP_obs=1.29, K=70, T=4.8, r=0.01, option_type = "call")))
# Compute Value-at-Risk
v = VaR(bt.price)
v.get_single_stock_VaR()
# Compute Correlation and autocorrelation
bt.compute_corr()
bt.plot_corr()
bt.compute_autocorr()
bt.plot_autocorr()


# General TODO list
# Implement ARMA, Yule-Walker Equations, application of Method of Moments
# Removing non-stationary trending behavior from non-stationary series like RW using difference operators (Box, Jenkins) 
# Example of non-stationary processes: linear trend reversion model, SRW, pure integrated process, stochastic trend process
# Stationarity checks: Dickey-Fuller Test
# ARIMA models: determine order of differencing required to remove time trends, estimate unknown params of ARIMA, select ARIMA model

# option pricing with blackscholes
# volatility and implied volatility modelling
# strategy implementation and backtesting: pairs e.g. S+P or B+C
# model evolution of a position in the market

# Bonds
# Swaps
# Yield curve
# Regularized yield curve models
# regularized volatility surface
