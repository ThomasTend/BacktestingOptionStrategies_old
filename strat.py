"""
Author: Thomas Tendron

This file implements the Strategy Tool.
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
        Prediction.__init__(self, asset = "stock", asset_id = "MSFT", target="Close", period="max", days_to_pred = 2, num_lag_features = 20)
        
    def test(self):
        print('Options are ', self.s.options)
        self.opt = self.s.option_chain(date=self.s.options[0])
        print(type(self.opt))
        # print(self.opt.columns)
        print("Calls are")
        print(self.opt.calls.head())
        print(self.opt.calls.columns)
        print("Puts are")
        print(self.opt.puts.head())
        print(self.opt.puts.columns)

strat = Strat(asset = "stock", asset_id = "MSFT", target="Close", period="max", days_to_pred = 2, num_lag_features = 20)
strat.test()