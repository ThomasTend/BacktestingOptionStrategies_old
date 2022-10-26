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
    def __init__(self, asset = "stock", asset_id = "MSFT", target="Close", period = "max", days_to_pred = 3, num_lag_features= 10, hist_start_date = "2017-10-26", hist_end_date="2018-01-05", use_news_sentiment_feature=True, model_name='LR'):
        """
        historical_start_idx is the number of days we look back from today. It is in units of days and has to be less than the days in period.
        """
        Prediction.__init__(self, asset, asset_id, target, period, days_to_pred, num_lag_features, hist_start_date, hist_end_date)
        self.make_train_test(use_news_sentiment_feature)
        self.train_and_predict(model_name)
        self.X_future = pd.DataFrame(self.feature_scaler.transform(self.data_future[self.feature_cols]), index = self.data_future.index)

    def get_future_dates(self, start_date, past=True):
        self.future_dates = []
        if past==True:
            start_idx = list(self.data.index).index(start_date)
            self.future_dates = self.data.index[start_idx:start_idx+self.days_to_pred]
        else: 
            for i in range(self.days_to_pred):
                self.future_dates.append(start_date + timedelta(days=i))

    def create_ax_fig(self):
        self.fig, self.ax = plt.subplots(1,1, figsize=(12,7))

    def plot_historical(self):
        """
        Add historical plot to the axis.
        """
        self.ax.plot(self.data_tmp.index, self.data_tmp[self.target], label="Historical price")
    
    def plot_backtest(self):
        self.get_future_dates(self.y_test.index[-self.days_to_pred])
        self.y_pred = self.target_scaler.inverse_transform(self.y_pred)
        price = [self.y_pred[-self.days_to_pred][i] for i in range(self.days_to_pred)]
        self.ax.plot(self.future_dates, price, label="Backtest forecast")

    def plot_future_prediction(self):
        self.y_future = self.target_scaler.inverse_transform(self.model.predict(self.X_future.to_numpy().reshape(1,-1)))
        self.get_future_dates(self.data_future.index[-1], past=False)
        self.ax.plot(self.future_dates, self.y_future.flatten(), label="Future prediction")

    def plot_all(self):
        self.ax.set(xlabel="Time", ylabel="Price", title="Backtest Price vs Time")
        self.ax.legend()
        plt.show()

    def test_systematic_portfolio(self):
        """
        We assume perfect liquidity (mainly can sell when we want to) and no transaction costs.
        """
        pf = Portfolio(asset=self.asset, asset_id=self.asset_id, target=self.target, period=self.period, days_to_pred=self.days_to_pred, num_lag_features=self.num_lag_features, hist_start_date =self.hist_start_date, hist_end_date=self.hist_end_date)
        # then compute PnL if we liquidate the positions at expiry
        self.initial_funds = sum(opt[3] for opt in pf.positions.values() if opt[1] in self.data.index)
        self.PnL = sum([opt[0]+self.data.loc[opt[1], self.target]-opt[3] if opt[2] >= self.data.loc[opt[1], self.target] else opt[0]+opt[2]-opt[3] for opt in pf.positions.values() if opt[1] in self.data.index])
        print("By investing ${0} in {1} using the covered call strategy between {2} and {3}, the portfolio makes ${4} - a {5:.2f}% return.".format(self.initial_funds, self.asset_id, pf.days_to_end_1, pf.days_to_end_2, self.PnL, (self.PnL / self.initial_funds)*100 if self.initial_funds else 0))