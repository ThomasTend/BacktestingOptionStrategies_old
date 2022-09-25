"""
Author: Thomas Tendron

This file implements the Strategy Tool.
"""

""" 4 - BACKTESTING TOOL"""

# data manipulation modules
import pandas as pd
# Graphing modules
import matplotlib.pyplot as plt

from portfolio import *
from prediction import *

class Backtest(Prediction, Portfolio):
    """
    Define a class backtest() to streamline the backtesting process. With this class we can:
    1) Fetch a stock's data from Yahoo Finance
    2) Train a model available on sklearn or tensorflow on Historical data for a choice of independent and dependent variables
    3) Visualize predictions on the past vs historical data
    4) Simulate a trading strategy/portfolio on past data and obtain statistics on the returns that one would have gotten

    It inherits the Prediction and Strat classes.
    """
    def __init__(self, asset = "stock", asset_id = "msft", target="Close", period = "1y", days_to_pred = 2, num_lag_features= 20, historical_start_idx = 250):
        """
        historical_start_idx is the number of days we look back from today. It is in units of days and has to be less than the days in period.
        """
        Prediction.__init__(self, asset, asset_id, target, period, days_to_pred, num_lag_features)
        self.make_train_test()
        self.train_and_predict()
        self.fig, self.ax = plt.subplots(1,1)
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
        Plot historical data and predictions on the same axes.
        """
        self.ax.plot(self.dates[-self.historical_start_idx:], self.price.loc[self.dates[-self.historical_start_idx]:])
    
    def plot_backtest(self):
        price_step_1 = [self.y_pred[-1][i] for i in range(self.days_to_pred)]
        self.get_future_dates(self.y_test.index[-1])
        self.ax.plot(self.future_dates, price_step_1)

    def plot_future_prediction(self):
        self.y_future = self.model.predict(self.X_future.to_numpy().reshape(1,-1))
        self.get_future_dates(self.data_future.index[-1])
        self.ax.plot(self.future_dates, self.y_future.flatten())

    def plot_all(self):
        plt.show()
