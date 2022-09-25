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

from pricing import *

""" 3 - STRATEGY SELECTION AND BUILDING"""

class Strat(Option_Pricer):
    """
    Strat class: allows to define a long-short strategy. For example, 
    Long Stock + Short Put (Covered call), 
    Long Bond/stock + Long Call, 
    Short stock + Long Put (Protective put), 
    Long stock + short Put 

    It inherits Option_Pricer.
    """
    def __init__(self):
        pass

