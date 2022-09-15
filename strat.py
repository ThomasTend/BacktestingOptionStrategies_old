"""
Author: Thomas Tendron
"""
# Ignore some warnings
import warnings
warnings.simplefilter(action='ignore')


# data manipulation modules
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import numpy as np
from scipy import stats as st

# API to fetch data from Yahoo Finance
import yfinance as yf

# Machine Learning modules
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as TTS
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE

# Graphing modules
import matplotlib.pyplot as plt


# # no warning messages in matplotlib backend
# from PyQt5.QtCore import QLoggingCategory

# QLoggingCategory.setFilterRules('qt.qpa.dialogs.debug=false')

""" FEATURE SELECTION TOOLS"""

"""
Fourier class: compute Fourier features.
"""

class Fourier():
    def __init__(self):
        pass

"""
Pricing class: Price options using Black-Scholes and extensions
"""

class Option_Pricer():
    def __init__(self):
        pass

"""
Volatility class: compute volatility and implied volatility. It inherits Option_Pricer in order to compute the implied volatility from the market price and the pricing model.
"""

class Volatility(Option_Pricer):
    def __init__(self, price):
        self.price = price

    def get_returns(self):
        """
        histograms of percentage price changes (may have fat tails, multiple modes, not necessarily normal)
        """
        price_lag = self.price.shift(1)
        diff = self.price-price_lag
        diff.dropna(axis=0, inplace=True)
        self.returns = diff / price_lag
        self.returns.plot(kind="hist")
        plt.show()

"""VaR class: Computes the Value-at-Risk for a given portfolio."""

class VaR(Volatility):
    def __init__(self, time_series):
        """
        Typically, time_series is a stock price time series. 
        """
        Volatility.__init__(self, time_series)
        self.time_series = time_series
        self.get_returns()

    def simulate_from_cov_mat(self, positions):
        pass

    def get_single_stock_VaR(self, threshold=0.05, position_size=1):
        """
        If time_series is a stock price, prints single stock VaR calculation given a certain position size. 
        """
        z_score = st.norm.ppf(1-threshold)
        print("z_score for threshold {0:.2f} is {1:.2f}".format(threshold, z_score))
        print("Last available price is {}".format(self.time_series.loc[self.time_series.index[-1]]))
        print("Standard deviation of returns is {}".format(self.returns.std()))
        self.VaR = z_score*self.returns.std()*self.time_series.loc[self.time_series.index[-1]]*position_size
        print("-"*20)
        print("95% VaR is {}".format(self.VaR))

class Features(Fourier, VaR):
    """
    For now time_series is assumed to be a pandas data frame.    
    """
    def __init__(self, time_series, target):
        self.time_series = time_series
        self.target = target

    def compute_corr(self):
        print(self.time_series.corr()[self.target].sort_values(ascending=False))

"""PRICE PREDICTION TOOL"""

class Prediction(Features):
    def __init__(self, asset = "stock", asset_id = "msft", target="Close", period="1y", days_to_pred = 2, num_lag_features = 20):
        """
        Download stock data from Yahoo finance using arguments.
        """
        self.asset = asset
        self.asset_id = asset_id
        self.target = target
        self.period = period
        self.days_to_pred = days_to_pred
        self.num_lag_features = num_lag_features
        s = yf.Ticker(self.asset_id)
        self.data = s.history(period=self.period)
        self.data.index = pd.to_datetime(self.data.index)
        self.dates = self.data.index
        self.price = self.data[self.target]

    def rescale_data(self):
        """
        Centre and normalize data.
        """
        scaler = SS()
        for col in self.data.columns:
            self.data.loc[:, col] = scaler.fit_transform(self.data.loc[:, col].to_numpy().reshape(-1, 1))
        self.price = pd.DataFrame(scaler.fit_transform(self.price.to_numpy().reshape(-1, 1)), index=self.dates)
        
    def add_lags(self, cols_for_lag_features): 
        """
        Adds two day lags in the past for features and one day in the future for targets. ALways lags the target, can also lag other coloumns listed in argument cols.
        """
        # add name of target
        cols_for_lag_features.append(self.target)
        # features = past
        for col in cols_for_lag_features:
            for i in range(self.num_lag_features):
                self.data[col+"_lag_{}".format(i+1)] = self.data[col].shift(i+1)
        
        # target = future
        for i in range(self.days_to_pred-1): # -1 because self.target is included
                self.data[self.target + "_step_{}".format(i+1)] = self.data[self.target].shift(-(i+1))

        # Save today's features for true prediction task
        self.data_future = pd.DataFrame(self.data.loc[self.data.index[-1],:].to_numpy().reshape(1, -1), index=[self.data.index[-1]], columns=self.data.columns)
        other_to_rem = [col+"_lag_{}".format(self.num_lag_features) for col in cols_for_lag_features]
        self.data_future.drop(labels=[self.target+"_lag_{}".format(self.num_lag_features)]+other_to_rem, axis=1, inplace=True)
        d = {}
        for c in cols_for_lag_features:
            d[c]=c+"_lag_1"
        for i in range(self.num_lag_features):
            for col in self.data_future.columns:
                for c in cols_for_lag_features:
                    if col == c+"_lag_{}".format(i):
                        d[c+"_lag_{}".format(i)] = c+"_lag_{}".format(i+1)
        self.data_future.rename(mapper=d, axis=1, inplace=True)
        self.data_future.dropna(axis=1, inplace=True)

        # Drop rows containing a NaN value 
        self.data.dropna(axis=0, inplace=True)

    def preprocess_data(self, other_lag_features = ["Volume"]):
        # Remove extra columns
        cols_to_rem = list(set(self.data.columns) - set(other_lag_features + [self.target]))
        self.data.drop(labels=cols_to_rem, axis=1, inplace=True)
        # Scale data and add 2 day lags
        self.rescale_data()
        self.add_lags(other_lag_features)
        # Instantiate Feature object for analysis
        Features.__init__(self, self.data, self.target)
        # Save names of target columns
        self.target_cols = [self.target] + [self.target + "_step_{}".format(i+1) for i in range(self.days_to_pred-1)]
        # Save names of lags of the target used as features
        self.feature_cols = list(set(self.data.columns) - set(self.target_cols) - set(other_lag_features))

    def make_train_test(self):
        """
        This function prepares the train and test data.
        """
        # Preprocess
        self.preprocess_data()
        # train_test_split
        X = self.data[self.feature_cols]
        y = self.data[self.target_cols]
        X_train, X_test, y_train, y_test = TTS(X, y, shuffle=False)
        # Save the train and test sets to the model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def train_and_predict(self, model_name="LR"):
        self.model_name = model_name
        self.model = LR()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def score(self, metric="mse"):
        mse = MSE(self.y_test, self.y_pred)
        print("MSE is {}".format(mse))

"""STRATEGY SELECTION AND BUILDING"""

"""
Strat class: allows to define a long-short strategy. For example, 
Long Stock + Short Put (Covered call), 
Long Bond/stock + Long Call, 
Short stock + Long Put (Protective put), 
Long stock + short Put 

It inherits Option_Pricer.
"""

class Strat(Option_Pricer):
    def __init__(self):
        pass

"""BACKTESTING TOOL"""

"""
Define a class backtest() to streamline the backtesting process. With this class we can:
1) Fetch a stock's data from Yahoo Finance
2) Train a model available on sklearn or tensorflow on Historical data for a choice of independent and dependent variables
3) Visualize predictions on the past vs historical data
4) Simulate a trading strategy/portfolio on past data and obtain statistics on the returns that one would have gotten

It inherits the Prediction and Strat classes.
"""

class Backtest(Prediction, Strat):
    def __init__(self, asset = "stock", asset_id = "msft", target="Close", period = "1y", days_to_pred = 2, num_lag_features= 20, historical_start_idx = 225):
        """
        historical_start_idx is the number of days we look back from today. It is in units of days and has to be less than the days in period.
        """
        Prediction.__init__(self, asset, asset_id, target, period, days_to_pred, num_lag_features)
        self.make_train_test()
        self.train_and_predict()
        self.fig, self.ax = plt.subplots(1,1)
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
        

""" PIPELINE """

bt = Backtest(asset = "stock", asset_id = "baba", target="Close", period="1y", days_to_pred = 3 , num_lag_features = 30)
bt.plot_historical()
bt.plot_backtest()
bt.plot_future_prediction()
bt.plot_all()
bt.score()
# v = VaR(bt.price)
# v.get_single_stock_VaR()
# bt.compute_corr()

# TODO
# option pricing with blackscholes
# volatility and implied volatility modelling
# strategy implementation and backtesting: pairs e.g. S+P or B+C
# model evolution of a position in the market