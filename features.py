"""
Author: Thomas Tendron

This file implements the feature selection tool.
"""

""" 1 - FEATURE SELECTION TOOLS"""

# import os to get file sizes for news data
import os
from os import listdir
from os.path import isfile, join

# math
import math

# data manipulation modules
import pandas as pd
import numpy as np
from datetime import datetime as dt

# Graphing modules
import matplotlib.pyplot as plt

# import classes used in feature analysis, construction or selection
from pricing import *
from numerical import *

# Machine Learning models
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans as KM

class Returns():
    def __init__(self, price):
        self.price = (price-price.mean())/price.std()
        self.price_lag = self.price.shift(1)
        self.price_lag.fillna(method='backfill', inplace=True)
        price_list = self.price.to_numpy()
        price_lag_list = self.price_lag.to_numpy()
        self.price_df = pd.DataFrame(np.array(list(zip(self.price.to_numpy(), self.price_lag.to_numpy()))), columns=["price", "price_lag"], index = self.price.index)
        # self.price_df.dropna(axis=0, inplace=True)
        self.price_df["log_change"] = np.log(self.price_df.price / self.price_df.price_lag).to_numpy().flatten()

    def get_returns(self):
        """
        Computes percentage price change series: (P_t-P_{t-1}) / P_{t-1}.
        """
        diff = self.price-self.price_lag
        self.returns = diff / self.price_lag
        self.returns.dropna(axis=0, inplace=True)
        self.returns = pd.Series(self.returns.values.flatten(), index = self.returns.index)

    def plot_returns(self):
        """
        Plots a histogram of percentage price changes (may have fat tails, multiple modes, not necessarily normal)
        """
        self.returns.plot(kind="hist", figsize=(12,7), title="Percentage price change")
        plt.show()

    def get_log_change(self):
        """
        Computes the log change time series: log( (Price on day t) / (price on day t-1) ).
        """
        self.log_change = pd.Series(np.log(self.price_df.price / self.price_df.price_lag).to_numpy(), index=self.price.index, name='log_change')
        # drop NaN values
        mask = ~np.isnan(self.log_change)
        self.log_change_no_nan = pd.Series(self.log_change[mask], index=self.price_df.index[mask])

class Volatility(Option_Pricer, Returns):
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
        Returns.__init__(self, price)

    def get_historical_volatility(self):
        """
        Get volatility (annualized std of log change) assuming stationarity.
        """
        self.get_log_change()
        self.log_change.fillna(0, inplace=True)
        self.hist_vol_full = pd.Series([self.log_change[:i].std() for i in range(self.log_change.shape[0])], index=self.price.index, name="hist_vol") # daily hist vol
        hist_vol = np.sqrt(252)*self.log_change_no_nan[-min(252, self.log_change_no_nan.shape[0]):].std()
        # print("Volatility is: {}".format(hist_vol))
        return hist_vol

    def plot_historical_volatility(self):
        self.hist_vol_series = pd.Series([np.sqrt(252)*self.log_change_no_nan[:i].std() for i in range(len(self.log_change_no_nan))], index=self.log_change_no_nan.index, name="hist_vol")
        self.hist_vol_series.plot(kind="line", figsize=(12,7), xlabel="Time", ylabel="Historical volatility", title="Historical volatility vs Time")
        plt.show()

    def get_implied_volatility(self, S_0, OP_obs, K, T, r, option_type = "call"):
        """
        We use the bisection algorithm to approximate the implied volatility. By Brenner and Subrahmanyam 1988, start with the 
        good approximation (2 * math.pi / T)**0.5 * P / S_0, where P is the current option market price.
        Ignoring dividends and assuming stationarity for now.

        OP_obs is the current market price of the option.
        error is the allowed difference between the observed price and our calculated price. 
        """
        # Set parameters
        self.S_0 = S_0
        self.OP_obs = OP_obs
        self.K = K 
        self.T = T
        self.sigma = (2 * math.pi / (T/365))**0.5 * self.OP_obs / self.S_0 # initial guess
        self.r = r
        # Initialize Option_Pricer
        Option_Pricer.__init__(self, T = self.T, S_0 = self.S_0, sigma = self.sigma, K = self.K, r = self.r, option_type = option_type)
        # Set function to be the difference between observed price and BSM price
        f = lambda x: self.OP_obs - self.price_option(sigma=x, manual = True)
        # Return output of bisection algorithm
        num = Numerical()
        return num.bisection(f, 0.01, 2)

class VaR(Returns):
    """
    VaR class: Computes the Value-at-Risk for a given portfolio.
    TODO: variance-covariance method, MC simulation
    TODO: backtest accuracy of VaR, i.e. compare predicted losses vs realized losses.
    """
    def __init__(self, time_series):
        """
        Typically, time_series is a stock price time series. 
        """
        Returns.__init__(self, time_series)
        self.time_series = time_series
        self.get_returns()

    def simulate_from_cov_mat(self, positions):
        pass

    def get_single_stock_VaR(self, threshold=0.05, position_size=1):
        """
        If time_series is a stock price, prints single stock VaR calculation given a certain position size. 
        """
        z_score = st.norm.ppf(1-threshold)
        self.VaR = z_score*self.returns.std()*position_size

    def plot_log_change_and_VaR(self):
        self.get_single_stock_VaR()
        print("Latest 95% VaR is {:.2f}".format(self.VaR))
        nf_VaR_line = -st.norm.ppf(0.95)*self.returns.rolling(5).std() # TODO: revisit this
        nn_VaR_line = -st.norm.ppf(0.99)*self.returns.rolling(5).std()
        fig, ax = plt.subplots(1, 1, figsize=(12,7))
        ax.plot(self.price_df.index, self.price_df.log_change, label= "log-change in price") 
        ax.plot(self.price_df.index, nf_VaR_line, label="95% Historical VaR", color="orange")
        ax.plot(self.price_df.index, nn_VaR_line, label="99% Historical VaR", color="red")
        ax.set(xlabel="Time", ylabel="Log-change in price", title="Log-change in price vs Time")
        ax.legend()
        plt.tight_layout()
        plt.show()

class Correlations():
    """
    Correlation class: Computes correlation between independent variables and target.
    Computes the autocorrelation. Plots both. 
    """
    def __init__(self, time_series, target): 
        """
        For now, time_series is assumed to be a pandas data frame.
        """
        self.time_series = time_series
        self.target = target

    def compute_corr(self):
        self.target_feature_corr = self.time_series.corr()[self.target].sort_values(ascending=False)

    def plot_corr(self):
        """
        Compute correlations between target and features.
        """
        ax = self.target_feature_corr.plot(kind="line", figsize=(12,7), xlabel="Features", ylabel="Correlation with target", title="Target-feature correlations")
        ax.set_xticks(np.arange(len(self.target_feature_corr.index)))
        # if few features, print their names along the x-axis
        if len(self.target_feature_corr.index) < 20: 
            ax.set_xticklabels(list(self.target_feature_corr.index))
        plt.tight_layout()
        plt.show()

    def compute_autocorr(self, max_lag=20):
        """
        Computes the autocorrelation of the target series for all lags up to max_lag.
        High autocorrelation for many lags suggests momentum.
        Autocorrelation which decays quickly as we lag further into the past suggests high volatility and low momentum.
        """
        self.autocorr = np.array([self.time_series[self.target].autocorr(lag=i) for i in range(max_lag)])
        
    def plot_autocorr(self):
        """
        Visualize autocorrelation of target.
        """
        horiz_axis = np.arange(len(self.autocorr))
        df = pd.Series(self.autocorr, index=horiz_axis)
        df.plot(kind="line", figsize=(12,7), xlabel="Target lag", ylabel="Autocorrelation", title="Target autocorrelation")
        plt.tight_layout()
        plt.show()

class Sentiment_Classifier():
    """
    Uses k-means algorithm to classify financial news quotes into 
    one of two sentiment categories: positive, negative.
    Our data is not labelled so we use k-means. 
    """
    def __init__(self, news_data, text_col, asset_id): # TODO: input date and collect only score for that date
        """
        data is a dataframe containing financial news text.
        text_col is the column of interest, e.g. title or content.
        """
        self.news_data = news_data # list of dataframes
        self.text_col = text_col
        self.asset_id = asset_id
        self.kmeans = []
        self.ticker_news_data = []
        self.vectorizer = []
        for i in range(len(self.news_data)):
            # Sparse matrix frequency representation
            self.vectorizer.append(TfidfVectorizer())
            # Save fit for re-use in prediction
            self.ticker_news_data.append(self.news_data[i].loc[self.news_data[i].ticker == self.asset_id, self.text_col])
            if self.ticker_news_data[i].empty == False:
                self.vectorizer[i].fit(self.ticker_news_data[i])
                X = self.vectorizer[i].transform(self.ticker_news_data[i])
                # Run k-means on 2 clusters
                self.kmeans.append(KM(n_clusters=2).fit(X))
            else:
                self.kmeans.append(np.NaN)

    def classify_news(self, news_date):
        self.classification = []
        if news_date <= pd.to_datetime("2020-02-13", format="%Y-%m-%d"):
            for i in range(len(self.news_data)):
                X = self.news_data[i].loc[(self.news_data[i].ticker == self.asset_id) & (self.news_data[i].release_date == news_date), self.text_col]
                if X.empty == False:
                    # Run k-means on 2 clusters
                    self.classification.append(self.kmeans[i].predict(self.vectorizer[i].transform(X)))
                else:
                    self.classification.append(np.NaN)
        else: 
            self.classification = [np.NaN for i in range(len(self.news_data))]

    def get_symbol_news_score(self, news_date): 
        """
        Transforms sentiment class to +1 or -1 and sums all scores on a given day for symbol self.asset_id. 
        """
        # Predict news sentiment for asset_id on news_date
        self.classify_news(news_date)
        # select files where there is news on self.asset_id 
        self.classification = np.array(self.classification)
        self.classification = self.classification[~pd.isnull(self.classification)]
        if self.classification.size != 0:
            self.classification = [x for pred in self.classification for x in pred]
            return sum([1 if x==1 else -1 for x in self.classification])
        else:
            return 0 # no news or neutral news

class Fourier():
    """
    Fourier class: compute Fourier features.
    """
    def __init__(self):
        pass

def validate_date(date_string, format):
    """
    Returns True if date_string can be parsed to datetime in the given format, False otherwise. 
    """
    try:
        dt.strptime(date_string, format)
        return True
    except ValueError:
        return False

class Features(Correlations, Sentiment_Classifier, Volatility):
    def __init__(self, time_series, target, asset_id):
        Correlations.__init__(self, time_series, target)
        """ 
        Load news data. We use a financial news dataset about US equities from 2008/10/02 to 2017/02/13 that can be found on Kaggle. 
        """
        self.date_parser = lambda x : dt.strptime(x, '%d/%m/%Y') if (pd.isnull(x) != True) and validate_date(x, '%d/%m/%Y') else np.NaN
        # Run the following if the news.csv file was updated and is more than 100 MB; GitHub only allows 100 MB per file
        """
        self.news_data = pd.read_csv("data/US_equities_news/news.csv", parse_dates=['release_date'], date_parser=self.date_parser)
        # 90MB is approximately 25600 rows in these news files
        for i in range((os.path.getsize('data/US_equities_news/news.csv') // (90*1024**2))+1):
            next_news_csv = self.news_data.iloc[i*25600:(i+1)*25600+1, :]
            print('at step {}'.format(i+1))
            print(os.getcwd())
            next_news_csv.to_csv('data/US_equities_news/news_{}.csv'.format(i+1))
        # print([x for x in os.listdir() if x == 'testfile.csv'])
        """
        # We use all the news files for unsupervised fitting (k-means)
        self.news_data = []
        news_files_num = len([f for f in listdir('./data/US_equities_news/') if isfile(join('./data/US_equities_news/', f))])
        for i in range(1, news_files_num+1):
            self.news_data.append(pd.read_csv(os.path.join(os.path.dirname(__file__),'data/US_equities_news/news_{}.csv'.format(i)), parse_dates=['release_date'], date_parser=self.date_parser))
            self.news_data[i-1].dropna(axis=0, inplace=True)
            self.news_data[i-1].sort_values('release_date', inplace=True)
        Sentiment_Classifier.__init__(self, self.news_data, 'content', asset_id)