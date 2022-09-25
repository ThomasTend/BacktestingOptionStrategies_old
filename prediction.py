"""
Author: Thomas Tendron

This file implements the prediction Tool.
"""

# Ignore some warnings
import warnings
warnings.simplefilter(action='ignore')

# stats
import statsmodels.api as sm

# data manipulation modules
import pandas as pd

# API to fetch data from Yahoo Finance
import yfinance as yf

# Machine Learning modules
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as TTS
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE

from features import *

""" 2 - PREDICTION TOOLS"""

class Prediction(Features):
    def __init__(self, asset = "stock", asset_id = "MSFT", target="Close", period="max", days_to_pred = 2, num_lag_features = 20):
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
        """
        Ready features and target for training.
        """
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
        
    def print_p_values(self):
        """
        Print p-values of LR model coefficients.
        """
        print(self.X_train)
        print(self.y_train)
        X_with_intercept = sm.add_constant(self.X_train)
        ols = sm.OLS(self.y_train[self.target], X_with_intercept).fit()
        print(ols.summary())

    def train_and_predict(self, model_name="LR"):
        """
        Train model of choice on train set and save prediction on test features.
        """
        self.model_name = model_name
        self.model = LR()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        #self.print_p_values()

    def score(self, metric="mse"):
        """
        Print evaluation metric for test target vs test prediction. 
        """
        mse = MSE(self.y_test, self.y_pred)
        print("MSE is {}".format(mse))