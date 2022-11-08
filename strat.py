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

class Strat(Prediction, Volatility):
    """
    Strat class: allows to define long-short, momentum and mean-reversion strategies. For example, long underlying + short call (Covered Call), 
    long underlying + long put (Protective Put), protective put and covered call on the same assets (Protective Collar), long call and 
    put on same asset with same strike price and expiry (long straddle), momentum or mean-reversion based on autocorrelation.
    
    Volatility inerits Option_Pricer and Returns.
    """
    def __init__(self, asset="stock", asset_id="MSFT", target="Close", period="max", days_to_pred=3, num_lag_features=10, hist_start_date = "2017-10-26", hist_end_date="2018-01-05", model_name='Lasso', use_cv=True, params={}):
        Prediction.__init__(self, asset=asset, asset_id=asset_id, target=target, period=period, days_to_pred=days_to_pred, num_lag_features=num_lag_features, hist_start_date=hist_start_date, hist_end_date=hist_end_date, use_cv=use_cv, params=params)
        # Load options data
        date_parser = lambda x : dt.strptime(x, '%d/%m/%Y') if pd.isnull(x) != True else x
        self.options = pd.read_csv("data/stock_options.csv", parse_dates=['date', 'expiration_date'], date_parser=date_parser)
        # Create column with time to expiry
        self.options['time_to_expiry'] = self.options.expiration_date-self.options.date
        # Make time to expiry NaN if it's negative
        self.options.time_to_expiry.mask(self.options.time_to_expiry <= timedelta(), np.nan, inplace=True) 
        # Remove row if time to expiry is NaN or expiration date is Null (no other column contains NaN)         
        self.options.dropna(axis=0, inplace=True) # affects 2437 rows out of 62795
        # Add column for unique identification of contract (no contract names in the kaggle dataset)
        self.options['contract'] = pd.Series(np.arange(self.options.shape[0]), index=self.options.index)
        # define a dictionary of positions
        self.positions = {}
        self.cc_ctr = 0 # covered call counter
        self.model_name = model_name
        self.params = params

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
        
        Updates the self.positions dictionary with keys CC_asset_id_i, where i is an integer and tuple values of the form:
        (call_premium, expiry, strike, stock_price_at_init_time)
        """
        start_idx = list(self.data.index).index(self.start_date)
        days_to_end = self.data.index[start_idx:-self.days_to_pred]
        self.days_to_end_1 = days_to_end[0].strftime("%Y-%m-%d")
        self.days_to_end_2 = days_to_end[-1].strftime("%Y-%m-%d")
        # Recursive prediction 
        for start_date in days_to_end:
            self.start_date = start_date
            # Predict self.days_to_pred days after start_date
            self.make_train_test(is_strat=True)
            self.train_and_predict(model_name=self.model_name, is_strat=True, params=self.params) # GBR by default 
            # self.X_future = self.data_future[self.feature_cols]
            self.y_future = self.model.predict(self.X_future.reshape(1,-1)).flatten()
            K = self.data.loc[start_date, self.target]
            Volatility.__init__(self, self.price[:start_date])
            sigma = self.get_historical_volatility()
            expiry = start_date + timedelta(days=self.days_to_pred)
            predicted_stock_move = self.y_future[self.days_to_pred-1]-self.y_future[0]
            # print('predicted_stock_move is {}'.format(predicted_stock_move))
            if (predicted_stock_move < 0):
                S_0 = self.data.loc[start_date,self.target]
                Option_Pricer.__init__(self, T=self.days_to_pred, S_0=S_0, sigma=sigma, K=K, r=0.05, option_type="call")
                call_premium = self.price_european_call(sigma) # set ask price
                if call_premium > predicted_stock_move:
                    # long stock and short call
                    self.positions.update({'CC_{0}_{1}'.format(self.asset_id, self.cc_ctr):(call_premium, expiry, K, S_0)})
                    self.cc_ctr += 1
        # print('positions:')
        # print(self.positions)