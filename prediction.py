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
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as MSE
from lightgbm import LGBMRegressor as LGBMR
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.tree import DecisionTreeRegressor as DTR

# import parent classes
from features import *

""" PREDICTION TOOLS"""

class Prediction(Features):
    def __init__(self, asset = "stock", asset_id = "MSFT", target="Close", period="max", days_to_pred = 2, num_lag_features = 20,  hist_start_date = "2020-06-17", hist_end_date="2022-06-06"):
        """
        Download stock data from Yahoo finance using arguments.
        """
        self.asset = asset
        self.asset_id = asset_id
        self.target = target
        self.period = period
        self.days_to_pred = days_to_pred
        self.num_lag_features = num_lag_features
        self.s = yf.Ticker(self.asset_id)
        self.data = self.s.history(period=self.period)
        self.data.index = pd.to_datetime(self.data.index)
        if hist_start_date == -1:
            self.hist_start_date = self.data.index[0]
        else:
            self.hist_start_date = pd.to_datetime(hist_start_date, format="%Y-%m-%d")
        if hist_end_date == -1:
            self.hist_end_date = self.data.index[-1]
        else:
            self.hist_end_date = pd.to_datetime(hist_end_date, format="%Y-%m-%d")
        start_idx = list(self.data.index).index(self.hist_start_date) + 2*self.num_lag_features
        try:
            if start_idx > len(self.data.index)-self.days_to_pred:
                raise ValueError('Not enough data ({} days available) to train with {} days of lag and predict {} days.'.format(len(self.data.index), self.num_lag_features, self.days_to_pred))
        except ValueError as e:
            print(e)
        self.start_date = self.data.index[start_idx] 
        self.data = self.data.loc[self.hist_start_date:self.hist_end_date,:]
        self.price = self.data[self.target]
        self.dates = self.data.index

    def rescale_data(self):
        """
        Centre and normalize data.
        """
        # Scale features
        self.feature_scaler = SS()
        self.feature_scaler.fit(self.X_train)
        self.X_train = self.feature_scaler.transform(self.X_train)
        self.X_test = self.feature_scaler.transform(self.X_test)
        # Scale target
        self.target_scaler = SS()
        self.target_scaler.fit(self.y_train)
        self.y_train = self.target_scaler.transform(self.y_train)
        self.y_test = pd.DataFrame(self.target_scaler.transform(self.y_test), index=self.y_test.index)
        
    def add_lags(self, cols_for_lag_features):
        """
        Adds two day lags in the past for features and one day in the future for targets. Always lags the target, can also lag other coloumns listed in argument cols.
        """
        # add name of target
        cols_for_lag_features.append(self.target)
        # features = past
        for col in cols_for_lag_features:
            for i in range(self.num_lag_features):
                self.data_tmp[col+"_lag_{}".format(i+1)] = self.data_tmp[col].shift(i+1)
        # target = future
        for i in range(self.days_to_pred-1): # -1 because self.target is included
                self.data_tmp[self.target + "_step_{}".format(i+1)] = self.data_tmp[self.target].shift(-(i+1))
        # Save today's features for true prediction task
        self.data_future = pd.DataFrame(self.data_tmp.loc[self.data_tmp.index[-1],:].to_numpy().reshape(1, -1), index=[self.data_tmp.index[-1]], columns=self.data_tmp.columns)
        other_to_rem = [col+"_lag_{}".format(self.num_lag_features) for col in cols_for_lag_features]
        self.data_future.drop(labels=[self.target+"_lag_{}".format(self.num_lag_features)] + other_to_rem, axis=1, inplace=True)
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
        # self.data_tmp.dropna(axis=0, inplace=True)
        # backfill
        self.data_tmp.fillna(method='backfill', inplace=True)

    def select_features(self, lag_feature_cols):
        """
        Once the features are built in the preprocess_data(self) function, the select_features(self) function
        selects only lag features with high autocorrelation and other features that have high correlation with
        the target.
        """
        if self.use_news_sentiment_feature == True:
            # Target-Feature correlation
            self.compute_corr()
            remove = list(self.target_feature_corr[(self.target_feature_corr <= 0.6) & (self.target_feature_corr >= -0.6)].index)
            # Autocorrelations
            self.compute_autocorr()
            lag_ctr = 0
            # Identify when autocorrelation is between - or + 0.7
            while self.autocorr[lag_ctr] > 0.6 or self.autocorr[lag_ctr] < -0.6:
                lag_ctr += 1
            # Select highest absolute autocorrelation lags
            for col in lag_feature_cols:
                for i in range(lag_ctr, self.num_lag_features+1):
                    remove.append(col+'_lag_{}'.format(i))
            self.feature_cols = list(set(self.feature_cols) - set(remove))
            # print(self.feature_cols)
        else: # need to instantiate Correlations class
            pass 


    def preprocess_data(self, other_lag_features = ["Volume"]):
        """
        Ready features and target for training. For now just lag of the price and volume. This is not really enough 
        because of the efficient market hypothesis. We need to use alternative data like classification of news
        using naive Baye's classifier. 
        """
        # Remove extra columns
        cols_to_rem = list(set(self.data_tmp.columns) - set(other_lag_features + [self.target]))
        self.data_tmp.drop(labels=cols_to_rem, axis=1, inplace=True)
        if self.use_news_sentiment_feature == True:
            # Initialize Feature class
            Features.__init__(self, self.data_tmp, self.target, self.asset_id)
            news_sentiments = [self.get_symbol_news_score(date) for date in self.data_tmp.index]
            self.data_tmp['sentiment'] = pd.Series(news_sentiments, index=self.data_tmp.index)
            # Initialize Volatility class 
            Volatility.__init__(self, self.data_tmp[self.target])
            self.get_historical_volatility()
            # Add log-returns feature (leave after Volatility class init)
            self.data_tmp['log_change'] = pd.Series(self.log_change, index=self.data_tmp.index)
            # Add rolling average
            short_rolling_price_mean = self.data_tmp[self.target].rolling(5).mean()
            short_rolling_price_mean.fillna(method='backfill', inplace=True)
            self.data_tmp['short_rolling_price_mean'] = pd.Series(short_rolling_price_mean, index=self.data_tmp.index)
            long_rolling_price_mean = self.data_tmp[self.target].rolling(20).mean()
            long_rolling_price_mean.fillna(method='backfill', inplace=True)
            self.data_tmp['long_rolling_price_mean'] = pd.Series(long_rolling_price_mean, index=self.data_tmp.index)
            # Plot short and long rolling averages
            # self.data_tmp.short_rolling_price_mean.plot()
            # self.data_tmp.long_rolling_price_mean.plot()
            # Add rolling std
            short_rolling_price_std = self.data_tmp[self.target].rolling(5).std()
            short_rolling_price_std.fillna(method='backfill', inplace=True)
            self.data_tmp['short_rolling_price_std'] = pd.Series(short_rolling_price_std, index=self.data_tmp.index)
            long_rolling_price_std = self.data_tmp[self.target].rolling(20).std()
            long_rolling_price_std.fillna(method='backfill', inplace=True)
            self.data_tmp['long_rolling_price_std'] = pd.Series(long_rolling_price_std, index=self.data_tmp.index)
            # Plot short and long rolling std
            # self.data_tmp.short_rolling_price_std.plot()
            # self.data_tmp.long_rolling_price_std.plot()
            # Add historical volatility feature
            self.data_tmp['hist_vol'] = pd.Series(self.hist_vol_full, index=self.data_tmp.index)
            # Add percentage price and volume change features
            self.data_tmp['prct_price_change'] = (self.data_tmp[self.target]-self.data_tmp[self.target].shift(1)) / self.data_tmp[self.target]
            self.data_tmp['prct_vol_change'] = (self.data_tmp['Volume']-self.data_tmp['Volume'].shift(1)) / self.data_tmp['Volume']
            # Add direction
            self.data_tmp['direction'] = self.data_tmp['prct_price_change'].apply(lambda x: 1 if x > 0 else -1)
            # backfill NaN values created
            self.data_tmp.fillna(method='backfill', inplace=True)
        # add lag features and targets
        if self.use_news_sentiment_feature == True:
            other_lag_features += ['log_change', 'hist_vol', 'short_rolling_price_mean', 'long_rolling_price_mean', 'short_rolling_price_std', 'long_rolling_price_std', 'prct_price_change', 'prct_vol_change', 'direction']
            self.add_lags(other_lag_features)
        else:
            self.add_lags(other_lag_features)
        # Save names of target columns
        self.target_cols = [self.target] + [self.target + "_step_{}".format(i+1) for i in range(self.days_to_pred-1)]
        # Save names of lags of the target used as features
        self.feature_cols = list(set(self.data_tmp.columns) - set(self.target_cols) - set(other_lag_features))
        # Select only the best features
        self.select_features(other_lag_features + [self.target])


    def make_train_test(self, use_news_sentiment_feature=True, is_strat=False):
        """
        This function prepares the train and test data.
        The variable self.start_date is used in the Strat class to choose suitable strategies on a given date. 
        """
        # truncate up to the date at which future predictions start
        self.data_tmp = self.data.copy()
        if is_strat == True:
            self.data_tmp = self.data_tmp.loc[:self.start_date, :]
        self.use_news_sentiment_feature = use_news_sentiment_feature
        # Preprocess
        self.preprocess_data()
        # train_test_split
        X = self.data_tmp[self.feature_cols]
        y = self.data_tmp[self.target_cols]
        X_train, X_test, y_train, y_test = TTS(X, y, shuffle=False)
        # if is_strat == False:
        #     print('Training period ends on {}.'.format(X_train.index[-1].strftime("%Y-%m-%d")))
        # Save and scale the train and test sets to the model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.rescale_data()
        
    def print_p_values(self):
        """
        Print p-values of LR model coefficients.
        """
        X_with_intercept = sm.add_constant(self.X_train)
        ols = sm.OLS(self.y_train[self.target], X_with_intercept).fit()
        print(ols.summary())

    def train_and_predict(self, model_name='LR'):
        """
        Train model of choice on train set and save prediction on test features.
        """
        if model_name == 'LR':
            self.model_name = model_name
            self.model = LR()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            #self.print_p_values()
        elif model_name == 'LGBMR':
            self.model_name = model_name
            self.model = MOR(LGBMR())
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
        elif model_name == 'Lasso': # LASSO
            self.model_name = model_name
            self.model = Lasso()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            #self.print_p_values()
        else: # DTR
            self.model_name = model_name
            self.model = MOR(DTR())
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

    def score(self, metric="mse"):
        """
        Print evaluation metric for test target vs test prediction. 
        """
        # forward fill last few values which are outside the testing range
        self.y_test.fillna(method='ffill', inplace=True)
        mse = MSE(self.y_test, self.y_pred)
        print("MSE is {}".format(mse))