"""
Author: Thomas Tendron
"""
# Ignore some warnings
import warnings
warnings.simplefilter(action='ignore')

# stats
import statsmodels.api as sm

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

""" FEATURE SELECTION TOOLS"""

class Fourier():
    """
    Fourier class: compute Fourier features.
    """
    def __init__(self):
        pass

class Option_Pricer():
    """
    Pricing class: Price options using Black-Scholes and extensions
    """
    def __init__(self):
        pass

class Volatility(Option_Pricer):
    """
    Volatility class: compute volatility from historical price. 
    It inherits Option_Pricer in order to compute the implied volatility from the market price and the pricing model.
    Also includes functions to compute and plot returns and log change in price. 

    Definition of volatility: Annualized standard deviation of the change in price or value of a financial security. (often log change)

    TODO:
    - Historical/ sample volatility measures
    - geometric Brownian motion model
    - Poisson jump diffusion model
    - ARCH/GARCH models
    - Stochastic Volatility (SV) models

    """
    def __init__(self, price):
        self.price = price
        self.price_lag = self.price.shift(1)
        price_list = self.price.to_numpy()
        price_lag_list = self.price_lag.to_numpy()
        self.price_df = pd.DataFrame(np.concatenate((self.price.to_numpy(), self.price_lag.to_numpy()), axis=1), columns=["price", "price_lag"], index = self.price.index)
        self.price_df.dropna(axis=0, inplace=True)
        self.price_df["log_change"] = np.log(self.price_df.price / self.price_df.price_lag).to_numpy().flatten()

    def get_returns(self):
        """
        Computes percentage price change series: (P_t-P_{t-1}) / P_{t-1}.
        """
        diff = self.price-self.price_lag
        diff.dropna(axis=0, inplace=True)
        self.returns = diff / self.price_lag

    def plot_returns(self):
        """
        Plots a histogram of percentage price changes (may have fat tails, multiple modes, not necessarily normal)
        """
        self.returns.plot(kind="hist")
        plt.show()

    def get_log_change(self):
        """
        Computes the log change time series: log( (Price on day t) / (price on day t-1) ).
        """
        self.log_change_no_nan = np.log(self.price_df.price / self.price_df.price_lag).to_numpy()
        # drop NaN values
        mask = ~np.isnan(self.log_change_no_nan)
        self.log_change_no_nan = pd.Series(self.log_change_no_nan[mask], index=self.price_df.index[mask])
    
    def get_historical_volatility(self):
        """
        Get volatility (annualized std of log change) assuming stationarity.
        """
        self.get_log_change()
        print("Volatility is: {}".format(np.sqrt(252)*self.log_change_no_nan.std()))

    def plot_log_change(self):
        self.price_df.log_change.plot(kind="line")
        plt.show()

class VaR(Volatility):
    """
    VaR class: Computes the Value-at-Risk for a given portfolio.
    """
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
        self.target_feature_corr = self.time_series.corr()[self.target].sort_values(ascending=False)
        print(self.target_feature_corr)

    def plot_corr(self):
        """
        Compute correlations between target and features.
        """
        ax = self.target_feature_corr.plot(kind="line")
        ax.set_xticks(np.arange(len(self.target_feature_corr.index)))
        # ax.set_xticklabels(list(self.target_feature_corr.index))
        plt.show()

    def compute_autocorr(self, max_lag=200):
        """
        Computes the autocorrelation of the target series for all lags up to max_lag.
        High autocorrelation for many lags suggests momentum.
        Autocorrelation which decays quickly as we lag further into the past suggests high volatility and low momentum.
        """
        self.partial_autocorr = [self.time_series[self.target].autocorr(lag=i) for i in range(max_lag)]
        print(*("Partial autocorrelation with lag {} is {}".format(i, self.partial_autocorr[i]) for i in range(max_lag)), sep='\n')
        
    def plot_autocorr(self):
        """
        Cisualize partial autocorrelation of target.
        """
        horiz_axis = np.arange(len(self.partial_autocorr))
        df = pd.Series(self.partial_autocorr, index=horiz_axis)
        df.plot(kind="line")
        plt.show()

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

"""STRATEGY SELECTION AND BUILDING"""

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

"""
Portfolio class: set of positions and strategies.
"""

class Portfolio(Strat):
    def __init__(self):
        pass

"""BACKTESTING TOOL"""

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
        

""" PIPELINE """

# Create backtest
bt = Backtest(asset = "stock", asset_id = "sq", target="Close", period="max", days_to_pred = 2 , num_lag_features = 30, historical_start_idx=-1)
# Plot historical stock price
bt.plot_historical()
# Plot prediction on same plot as historical price for visual comparison
bt.plot_backtest()
# Plot future prediction on same plot
bt.plot_future_prediction()
# Call plt.show()
bt.plot_all()
# Print MSE on test set
bt.score()
# Compute volatility:
vol = Volatility(bt.price)
vol.get_historical_volatility()
# Plot log change
vol.plot_log_change()
# Compute Value-at-Risk
v = VaR(bt.price)
v.get_single_stock_VaR()
# Compute Correlation and autocorrelation
bt.compute_corr()
bt.plot_corr()
bt.compute_autocorr()
bt.plot_autocorr()

# TODO
# Implement ARMA, Yule-Walker Equations, application of Method of Moments
# Removing non-stationary trending behavior from non-stationary series like RW using difference operators (Box, Jenkins) 
# Example of non-stationary processes: linear trend reversion model, SRW, pure integrated process, stochastic trend process
# Stationarity checks: Dickey-Fuller Test
# ARIMA models: determine order of differencing required to remove time trends, estimate unknown params of ARIMA, select ARIMA model

# option pricing with blackscholes
# volatility and implied volatility modelling
# strategy implementation and backtesting: pairs e.g. S+P or B+C
# model evolution of a position in the market