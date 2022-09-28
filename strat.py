"""
Author: Thomas Tendron

This file implements the Strategy Tool.

We use a stock options dataset from Kaggle with data between 13/11/2017 and 09/08/2019.
"""

# Ignore some warnings
import warnings
warnings.simplefilter(action='ignore')

# math and stats
import math
import statsmodels.api as sm
from numerical import *

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

# import parent classes
from prediction import *
from pricing import *

""" STRATEGY SELECTION AND BUILDING"""

class Strat(Prediction, Option_Pricer):
    """
    Strat class: allows to define a long-short or momentum strategies. For example, 
    Long underlying + Short call (Covered call), 
    Long underlying + Long Put (Protective put), 
    A protective put and a covered call on the same assets (Protective collar)
    Long call and put on same asset with same strike price and expiry (Long straddle) 
    """
    def __init__(self, asset = "stock", asset_id = "MSFT", target="Close", period="max", days_to_pred = 2, num_lag_features = 20):
        Prediction.__init__(self, asset = asset, asset_id = asset_id, target=target, period=period, days_to_pred = days_to_pred, num_lag_features = num_lag_features)
        # Load options data
        date_parser = lambda x : dt.strptime(x, '%d/%m/%Y') if pd.isnull(x) != True else x
        self.options = pd.read_csv("data/stock_options.csv", parse_dates = ['date', 'expiration_date'], date_parser = date_parser)
        # Create column with time to expiry
        self.options['time_to_expiry'] = self.options.expiration_date-self.options.date
        # Make time to expiry NaN if it's negative
        self.options.time_to_expiry.mask(self.options.time_to_expiry <= timedelta(), np.nan, inplace=True) 
        # Remove row if time to expiry is NaN or expiration date is Null (no other column contains NaN)         
        self.options.dropna(axis=0, inplace=True) # affects 2437 rows out of 62795

    def get_stock_options(self, start_date):
        """
        Get options tradable on start_date (pd datetime64 object) and for the symbol self.asset_id.
        """
        self.tradable = self.options[(self.options.sym == self.asset_id) & (self.options.date == start_date)]
        print(self.tradable.head())
        print(self.tradable.tail())

strat = Strat(asset = "stock", asset_id = "AMD", target="Close", period="max", days_to_pred = 2, num_lag_features = 20)
strat.get_stock_options(pd.to_datetime("2019-08-09", format="%Y-%m-%d"))