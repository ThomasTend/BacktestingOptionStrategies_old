"""
Author: Thomas Tendron
"""

# data manipulation modules
from datetime import datetime as dt
import pandas as pd
import numpy as np
from scipy import stats as st

# API to fetch data from Yahoo Finance
import yfinance as yf

# Machine Learning modules
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as TTS
from sklearn.linear_model import LinearRegression as LR

# Graphing modules
import matplotlib.pyplot as plt

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
    def __init__(self):
        pass 

"""VaR class: Computes the Value-at-Risk for a given portfolio."""

class VaR():
    def __init__(self):
        pass 

class Features(Fourier, Option_Pricer, Volatility, VaR):
    def __init__(self):
        pass 

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
        self.price = self.data[self.target]

    def rescale_data(self):
        """
        Centre and normalize data.
        """
        scaler = SS()
        for col in self.data.columns:
            self.data.loc[:, col] = scaler.fit_transform(self.data.loc[:, col].to_numpy().reshape(-1, 1))

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

        # Drop rows containing a NaN value 
        self.data.dropna(axis=0, inplace=True)

    def preprocess_data(self, other_lag_features = ["Volume"]):
        # Remove extra columns
        cols_to_rem = list(set(self.data.columns) - set(other_lag_features + [self.target]))
        self.data.drop(labels=cols_to_rem, axis=1, inplace=True)
        # Scale data and add 2 day lags
        self.rescale_data()
        self.add_lags(other_lag_features)
        # Save names of target columns
        self.target_cols = [self.target] + [self.target+"_step_{}".format(i+1) for i in range(self.days_to_pred-1)]
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
        self.y_test = y_test#.reshape(1,-1)

    def train_and_predict(self, model="LR"):
        self.model = model
        lr = LR()
        lr.fit(self.X_train, self.y_train)
        self.y_pred = lr.predict(self.X_test)

"""STRATEGY SELECTION AND BUILDING"""

"""
Strat class: allows to define a long-short strategy. For example, 
Long Stock + Short Put (Covered call), 
Long Bond/stock + Long Call, 
Short stock + Long Put (Protective put), 
Long stock + short Put 
"""

class Strat():
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
    def __init__(self, asset = "stock", asset_id = "msft", target="Close", period = "1y", days_to_pred = 2, num_lag_features= 20):
        Prediction.__init__(self, asset, asset_id, target, period, days_to_pred, num_lag_features)
        self.make_train_test()
        self.train_and_predict()
        print(self.data.head())
        print("£"*40)
        print(self.price)

    def plot_backtest(self):
        """
        Plot historical data and predictions on the same axes.
        """
        fig, ax = plt.subplots(1,1)
        ax.plot(self.data.index, self.data[self.target]) 
        price_step_1 = [self.y_pred[i] for i in range(len(self.y_pred))]
        ax.plot(self.y_test.index, price_step_1)
        plt.show()

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

    def simulate_from_cov_mat(self, positions):
        pass

    def get_single_stock_VaR(self, threshold=0.05, position_size=1):
        """
        Prints single stock VaR calculation 
        """
        z_score = st.norm.ppf(1-threshold)
        print("z_score for threshold {0:.2f} is {1:.2f}".format(threshold, z_score))
        print("Last available price is {}".format(self.price.loc[self.price.index[-1]]))
        print("Standard deviation of returns is {}".format(self.returns.std()))
        self.VaR = z_score*self.returns.std()*self.price.loc[self.price.index[-1]]*position_size
        print("-"*20)
        print("95% VaR is {}".format(self.VaR))

""" PIPELINE """

bt = Backtest(asset = "stock", asset_id = "goog", target="Close", period="1y", days_to_pred = 1, num_lag_features = 10)
bt.plot_backtest()
bt.get_returns() # a lot seem to be heavy tailed. 
bt.get_single_stock_VaR()
# option pricing with blackscholes
# volatility and implied volatility modelling
# strategy implementation and backtesting: pairs e.g. S+P or B+C
# model evolution of a position in the market
