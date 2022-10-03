"""
Author: Thomas Tendron

This file implements the backtesting tool.
"""

""" 4 - BACKTESTING TOOL"""

# data manipulation modules
import pandas as pd
# Graphing modules
import matplotlib.pyplot as plt

# import parent classes
from prediction import *
from portfolio import *

class Backtest(Prediction):
    """
    Backtest class: streamlines the backtesting process. With this class we can:
    1) Fetch a stock's data from Yahoo Finance
    2) Train a model available on sklearn or tensorflow on Historical data for a choice of independent and dependent variables
    3) Visualize predictions on the past vs historical data
    4) Simulate a trading strategy/portfolio on past data and obtain statistics on the returns that one would have gotten
    """
    def __init__(self, asset = "stock", asset_id = "MSFT", target="Close", period = "max", days_to_pred = 2, num_lag_features= 20, historical_start_idx = 250):
        """
        historical_start_idx is the number of days we look back from today. It is in units of days and has to be less than the days in period.
        """
        Prediction.__init__(self, asset, asset_id, target, period, days_to_pred, num_lag_features)
        self.make_train_test()
        self.train_and_predict()
        self.fig, self.ax = plt.subplots(1,1, figsize=(12,7))
        if historical_start_idx == -1:
            self.historical_start_idx = len(self.dates)
        else:
            self.historical_start_idx = historical_start_idx
        self.X_future = self.data_future[self.feature_cols]

    def get_future_dates(self, start_date):
        self.future_dates = []
        last_date = start_date
        for i in range(self.days_to_pred):
            self.future_dates.append(last_date + timedelta(days=i))

    def plot_historical(self):
        """
        Add historical plot to the axis.
        """
        self.ax.plot(self.dates[-self.historical_start_idx:], self.price.loc[self.dates[-self.historical_start_idx]:], label="Historical price")
    
    def plot_backtest(self):
        price_step_1 = [self.y_pred[-1][i] for i in range(self.days_to_pred)]
        self.get_future_dates(self.y_test.index[-1])
        self.ax.plot(self.future_dates, price_step_1, label="Backtest forecast")

    def plot_future_prediction(self):
        self.y_future = self.model.predict(self.X_future.to_numpy().reshape(1,-1))
        self.get_future_dates(self.data_future.index[-1])
        self.ax.plot(self.future_dates, self.y_future.flatten(), label="Future prediction")

    def plot_all(self):
        self.ax.set(xlabel="Time", ylabel="Price", title="Backtest Price vs Time")
        self.ax.legend()
        plt.show()

    def test_systematic_portfolio(self, start_date = pd.to_datetime("2022-06-06", format="%Y-%m-%d")):
        """
        We assume perfect liquidity (mainly can sell when we want to) and no transaction costs.
        """
        pf = Portfolio(asset = self.asset, asset_id = self.asset_id, target = self.target, period = self.period, days_to_pred = self.days_to_pred, num_lag_features = self.num_lag_features, start_date=start_date)
        # then compute PnL if we liquidate the positions at expiry
        self.initial_funds = sum(opt[3] for opt in pf.positions.values() if opt[1] in self.data.index)
        self.PnL = sum([opt[0]+self.data.loc[opt[1], self.target]-opt[3] if opt[2] >= self.data.loc[opt[1], self.target] else opt[0]+opt[2]-opt[3] for opt in pf.positions.values() if opt[1] in self.data.index])
        start = start_date.strftime("%Y-%m-%d")
        end = self.data.index[-1].strftime("%Y-%m-%d")
        print("By investing ${0} in {1} using the covered call strategy between {2} and {3}, the portfolio makes ${4}".format(self.initial_funds, self.asset_id, start, end, self.PnL))