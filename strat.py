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

# Graphing modules
import matplotlib.pyplot as plt

# import parent classes
from prediction import *
from pricing import *

""" STRATEGY SELECTION AND BUILDING

We assume perfect liquidity (mainly can sell when we want to) and no transaction costs.
"""


class Strat(Prediction, Option_Pricer):
    """
    Strat class: allows to define long-short, momentum and mean-reversion strategies. For example, long underlying + short call (Covered Call), 
    long underlying + long put (Protective Put), protective put and covered call on the same assets (Protective Collar), long call and 
    put on same asset with same strike price and expiry (long straddle), momentum or mean-reversion based on autocorrelation.
    """
    def __init__(self, asset = "stock", asset_id = "MSFT", target="Close", period="max", days_to_pred = 2, num_lag_features = 20, start_date=pd.to_datetime("2019-08-09", format="%Y-%m-%d")):
        Prediction.__init__(self, asset = asset, asset_id = asset_id, target=target, period=period, days_to_pred = days_to_pred, num_lag_features = num_lag_features)
        self.start_date = start_date
        # Load options data
        date_parser = lambda x : dt.strptime(x, '%d/%m/%Y') if pd.isnull(x) != True else x
        self.options = pd.read_csv("data/stock_options.csv", parse_dates = ['date', 'expiration_date'], date_parser = date_parser)
        # Create column with time to expiry
        self.options['time_to_expiry'] = self.options.expiration_date-self.options.date
        # Make time to expiry NaN if it's negative
        self.options.time_to_expiry.mask(self.options.time_to_expiry <= timedelta(), np.nan, inplace=True) 
        # Remove row if time to expiry is NaN or expiration date is Null (no other column contains NaN)         
        self.options.dropna(axis=0, inplace=True) # affects 2437 rows out of 62795
        # define a dictionary of positions
        self.positions = {}

    def get_stock_options(self):
        """
        Get options tradable on start_date (pd datetime64 object) and for the symbol self.asset_id.
        """
        self.tradable = self.options[(self.options.sym == self.asset_id) & (self.options.date == self.start_date)]

    def covered_call(self):
        """
        If predictions suggest the stock price could decrease a bit in the next few days but
        is likely to go up more after that, and if the premium we would get from shorting a 
        call option is larger than the expected loss in the stock, then we can long the stock 
        and short a call with expiry when the stock starts going up in the predictions.
        """
        # proceed if there are options trading on self.start_date for stock self.asset_id 
        if len(self.tradable.shape.index) > 0:
            # predict self.days_to_pred days after self.start_date
            self.make_train_test(start_date=self.start_date)
            self.train_and_predict()
            self.X_future = self.data_future[self.feature_cols]
            self.y_future = self.model.predict(self.X_future.to_numpy().reshape(1,-1))
            # check conditions for a covered call
            # TODO

strat = Strat(asset = "stock", asset_id = "AMD", target="Close", period="max", days_to_pred = 2, num_lag_features = 20)
strat.get_stock_options()