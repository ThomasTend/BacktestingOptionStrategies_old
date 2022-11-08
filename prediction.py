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
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# import parent classes
from features import *

""" PREDICTION TOOLS"""

class Prediction(Features):
    def __init__(self, asset = "stock", asset_id = "MSFT", target="Close", period="max", days_to_pred = 2, num_lag_features = 10,  hist_start_date = "2020-06-17", hist_end_date="2022-06-06", use_cv=True, params={}):
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
        try: 
            start_idx = list(self.data.index).index(self.hist_start_date) + 10*self.num_lag_features 
        except ValueError as e:
            print('ValueError: Data is not available on date hist_start_date.')
        try:
            if self.data.index[start_idx]  > self.hist_end_date:
                try:
                    begin = list(self.data.index).index(self.hist_start_date)
                    end = list(self.data.index).index(self.hist_end_date)
                except ValueError as e:
                    print('ValueError: Data is not available on date hist_end_date.')
                finally:
                    raise ValueError('ValueError: Not enough data ({} days available) to train with {} days of lag and predict {} days.'.format(end-begin, self.num_lag_features, self.days_to_pred))
        except ValueError as e:
            print(e)
        self.start_date = self.data.index[start_idx] 
        self.data = self.data.loc[self.hist_start_date:self.hist_end_date,:]
        self.price = self.data[self.target]
        # Predict self.target change (returns) instead of self.target; Volatility class inherits Returns class
        Returns.__init__(self, self.data[self.target]) # returns stored in self.returns
        self.get_returns()
        self.data['returns'] = self.returns
        self.dates = self.data.index
        # number of PCA components to keep
        self.n_components = min(10, len(self.data.loc[self.hist_start_date:self.hist_end_date,:].index) // 10)
        self.use_cv = use_cv

    def rescale_data(self, is_strat=False):
        """
        Centre and normalize data.
        """
        if is_strat == False:
            # Scale features
            self.feature_scaler = SS()
            self.feature_scaler.fit(self.X_train)
            self.X_train = self.feature_scaler.transform(self.X_train)
            self.X_val = self.feature_scaler.transform(self.X_val)
            self.X_test = self.feature_scaler.transform(self.X_test)
            # Scale target
            self.target_scaler = SS()
            self.target_scaler.fit(self.y_train)
            self.y_train = self.target_scaler.transform(self.y_train)
            self.y_val = pd.DataFrame(self.target_scaler.transform(self.y_val), index=self.y_val.index)
            self.y_test = pd.DataFrame(self.target_scaler.transform(self.y_test), index=self.y_test.index)
        else:
            # Scale features
            self.feature_scaler = SS()
            self.feature_scaler.fit(self.X_train)
            self.X_train = self.feature_scaler.transform(self.X_train)
            # Scale target
            self.target_scaler = SS()
            self.target_scaler.fit(self.y_train)
            self.y_train = self.target_scaler.transform(self.y_train)
        
    def add_lags(self, cols_for_lag_features):
        """
        Adds two day lags in the past for features and one day in the future for targets. Always lags the target, can also lag other columns listed in argument cols.
        """
        self.data_tmp.loc[self.data_tmp.index[-1],'returns'] = np.NaN
        # add name of target
        cols_for_lag_features += [self.target, 'returns']
        # features = past
        for col in cols_for_lag_features:
            for i in range(self.num_lag_features):
                self.data_tmp[col+"_lag_{}".format(i+1)] = self.data_tmp[col].shift(i+1)

        # target = future
        for i in range(self.days_to_pred-1): # -1 because self.target is included
            self.data_tmp['returns' + "_step_{}".format(i+1)] = self.data_tmp['returns'].shift(-(i+1))
        
        # Save today's features for true prediction task
        self.data_future = pd.DataFrame(self.data_tmp.loc[self.data_tmp.index[-1],:].to_numpy().reshape(1, -1), index=[self.data_tmp.index[-1]], columns=self.data_tmp.columns)
        self.data_future.drop(labels= cols_for_lag_features, axis=1, inplace=True)
        self.data_future.dropna(axis=1, inplace=True)

        # Drop rows containing a NaN value 
        self.data_tmp.dropna(axis=0, inplace=True)
        # backfill
        self.data_tmp.fillna(method='backfill', inplace=True)
        self.data_tmp.drop(labels=list(set(cols_for_lag_features)-set(['returns'])), axis=1, inplace=True)

    def select_features(self, is_strat=False):#, lag_feature_cols):
        """
        Select features using PCA.
        """
        pca = PCA(n_components=self.n_components)
        pca.fit(self.X_train)
        self.X_train  = pca.transform(self.X_train) 
        self.X_future = pca.transform(self.X_future)
        if is_strat == False:
            self.X_test  = pca.transform(self.X_test) 
            self.X_val  = pca.transform(self.X_val) 
        component_names = [f'PC{i+1}' for i in range(self.n_components)]

    def preprocess_data(self, other_lag_features = ["Volume"], is_strat=False):
        """
        Ready features and target for training.
        """
        other_lag_features.append(self.target)
        # Remove extra columns
        cols_to_rem = list(set(self.data_tmp.columns) - set(other_lag_features + ['returns']))
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

            # Add rolling average, with an initial shift to only use data up to day t-1 to predict day t and beyond
            short_rolling_price_mean = self.data_tmp[self.target].shift(7)
            short_rolling_price_mean.fillna(method='backfill', inplace=True)
            short_rolling_price_mean = short_rolling_price_mean.rolling(7).mean()
            self.data_tmp['short_rolling_price_mean'] = pd.Series(short_rolling_price_mean, index=self.data_tmp.index)

            long_rolling_price_mean = self.data_tmp[self.target].shift(15)
            long_rolling_price_mean.fillna(method='backfill', inplace=True)
            long_rolling_price_mean = long_rolling_price_mean.rolling(15).mean()
            self.data_tmp['long_rolling_price_mean'] = pd.Series(long_rolling_price_mean, index=self.data_tmp.index)

            # Plot short and long rolling averages
            # self.data_tmp.short_rolling_price_mean.plot()
            # self.data_tmp.long_rolling_price_mean.plot()

            # Add rolling std, with an initial shift to only use data up to day t-1 to predict day t and beyond
            short_rolling_price_std = self.data_tmp[self.target].shift(7)
            short_rolling_price_std.fillna(method='backfill', inplace=True)
            short_rolling_price_std = short_rolling_price_std.rolling(7).std()
            self.data_tmp['short_rolling_price_std'] = pd.Series(short_rolling_price_std, index=self.data_tmp.index)
            long_rolling_price_std = self.data_tmp[self.target].shift(15)
            long_rolling_price_std.fillna(method='backfill', inplace=True)
            long_rolling_price_std = long_rolling_price_std.rolling(15).std()
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
            # ['sentiment', 'log_change', 'short_rolling_price_mean', 'long_rolling_price_mean', 'prct_price_change', 'direction']
            self.add_lags(other_lag_features)
        else:
            self.add_lags(other_lag_features)

        # Save names of target columns
        self.target_cols = ['returns'] + ['returns' + "_step_{}".format(i+1) for i in range(self.days_to_pred-1)]

        # Save names of lags of the target used as features
        self.feature_cols = list(set(self.data_tmp.columns) - set(self.target_cols) - set(other_lag_features))

    def make_train_test(self, use_news_sentiment_feature=True, is_strat=False):
        """
        This function prepares the train and test data.
        The variable self.start_date is used in the Strat class to choose suitable strategies on a given date. 
        """
        # truncate up to the date at which future predictions start
        self.data_tmp = self.data.copy()
        self.price_tmp = self.data_tmp[self.target]
        self.use_news_sentiment_feature = use_news_sentiment_feature
        if is_strat == True:
            start_idx = list(self.data_tmp.index).index(self.start_date)
            self.data_tmp = self.data_tmp.iloc[:start_idx+1, :]
            # Preprocess, self.start_date was dropped and saved in self.data_future
            self.preprocess_data(is_strat=True)
            self.X_train = self.data_tmp[self.feature_cols]
            self.y_train = self.data_tmp[self.target_cols]
        else:
            # Preprocess
            self.preprocess_data()
            # train_test_split
            X = self.data_tmp[self.feature_cols]
            y = self.data_tmp[self.target_cols]
            # Split full data into train and test sets
            X_train, X_test, y_train, y_test = TTS(X, y, shuffle=False, test_size=0.33)
            # Split the train set into train and validation sets
            X_train, X_val, y_train, y_val = TTS(X_train, y_train, shuffle=False, test_size=0.5) # train, validation and test sets are roughly of the same size
            print('Training period: {} to {}.'.format(X_train.index[0].strftime("%Y-%m-%d"), X_train.index[-1].strftime("%Y-%m-%d")))
            print('Validation period: {} to {}.'.format(X_val.index[0].strftime("%Y-%m-%d"), X_val.index[-1].strftime("%Y-%m-%d")))
            print('Testing period: {} to {}.'.format(X_test.index[0].strftime("%Y-%m-%d"), X_test.index[-1].strftime("%Y-%m-%d")))
            # Save and scale the train and test sets to the model
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
        
        # Standard rescaling
        self.rescale_data(is_strat)

        # Create features for future prediction
        self.X_future = pd.DataFrame(self.feature_scaler.transform(self.data_future[self.feature_cols]), index = self.data_future.index)

        # Select only the best features
        self.select_features(is_strat)
        
    def print_p_values(self):
        """
        Print p-values of LR model coefficients.
        """
        X_with_intercept = sm.add_constant(self.X_train)
        ols = sm.OLS(self.y_train[self.target], X_with_intercept).fit()
        print(ols.summary())

    def grid_search_cross_validation(self):
        """
        Use the validation set to tune the hyperparameters of our model. For now only GBR.  
        """
        params_grid = {
                    "estimator__n_estimators": [10, 20],
                    "estimator__max_depth": [5, 6, 7],
                    "estimator__max_features": ['sqrt'],
                    "estimator__min_samples_split": [0.3, 0.5, 0.7],
                    "estimator__min_samples_leaf": [0.3, 0.5, 0.7],
                    "estimator__learning_rate": [0.05, 0.1, 0.15],
                    "estimator__loss": ['squared_error'],
                    "estimator__n_iter_no_change": [2, 3, 4],
                    "estimator__validation_fraction": [0.1, 0.2],
                    "estimator__subsample": [0.3, 0.5, 0.7],
                    # "verbose":100,
                    "estimator__tol": [0.01, 0.02]
                }
        self.base_model = MOR(GBR())
        self.best_model = GridSearchCV(self.base_model, params_grid)
        self.best_model.fit(X=self.X_val, y=self.y_val)

    def train_and_predict(self, model_name='GBR', is_strat=False, params={}):
        """
        Train model of choice on train set and save prediction on test features.
        """
        if is_strat == False:
            if model_name == 'LR':
                self.model_name = model_name
                self.model = LR()
                self.model.fit(self.X_train, self.y_train)
                self.y_pred_val = self.model.predict(self.X_val)
                self.y_pred = self.model.predict(self.X_test)
            elif model_name == 'GBR':
                self.model_name = model_name
                if len(params) == 0: # params dictionary is empty
                    params = {
                        "n_estimators": 15,
                        "max_depth": 3,
                        "max_features": 'sqrt',
                        "min_samples_split": 0.5,
                        "min_samples_leaf": 0.5,
                        "learning_rate": 0.1,
                        "loss": "squared_error",
                        "n_iter_no_change": 3,
                        # "validation_fraction": 0.2,
                        "subsample":0.8,
                        # "verbose":100,
                        "tol":0.01
                    }
                else:
                    # remove 'estimator__' from parameter names (was needed in param grid search)
                    for k in list(params):
                        params[k[11:]] = params.pop(k)
                self.model = MOR(GBR(**params))
                self.model.fit(X=self.X_train, y=self.y_train)
                self.y_pred_val = self.model.predict(self.X_val)
                self.y_pred = self.model.predict(self.X_test)
            elif model_name == 'Lasso': # LASSO
                self.model_name = model_name
                self.model = Lasso()
                self.model.fit(self.X_train, self.y_train)
                self.y_pred_val = self.model.predict(self.X_val)
                self.y_pred = self.model.predict(self.X_test)
            else: # DTR
                self.model_name = model_name
                self.model = MOR(DTR())
                self.model.fit(self.X_train, self.y_train)
                self.y_pred_val = self.model.predict(self.X_val)
                self.y_pred = self.model.predict(self.X_test)
        elif self.use_cv == True: # is_start and use_cv are True
            if model_name == 'LR':
                self.model_name = model_name
                self.model = LR()
                self.model.fit(self.X_train, self.y_train)
            elif model_name == 'GBR':
                self.model_name = model_name
                self.model = MOR(GBR(**params))
                self.model.fit(X=self.X_train, y=self.y_train)
            elif model_name == 'Lasso': # LASSO
                self.model_name = model_name
                self.model = Lasso()
                self.model.fit(self.X_train, self.y_train)
            else: # DTR
                self.model_name = model_name
                self.model = MOR(DTR())
                self.model.fit(self.X_train, self.y_train)
        else: # is_start is True and use_cv is False
            pass

    def score(self, metric="mse"):
        """
        Print evaluation metric for test target vs test prediction. 
        """
        # Inverse transform and forward fill last few values which are outside the testing range
        self.y_test.fillna(method='ffill', inplace=True)
        self.y_test = pd.DataFrame(self.target_scaler.inverse_transform(self.y_test), index=self.y_test.index)
        self.y_val = pd.DataFrame(self.target_scaler.inverse_transform(self.y_val), index=self.y_val.index)
        self.y_train = self.target_scaler.inverse_transform(self.y_train)
        self.y_pred_train = self.target_scaler.inverse_transform(self.model.predict(self.X_train))
        # Train set
        train_mse = MSE(self.y_train, self.y_pred_train)
        print('Train MSE is {}'.format(train_mse))
        # Validation set
        val_mse = MSE(self.y_val, self.y_pred_val)
        print('Validation MSE is {}'.format(val_mse))
        # Test set
        test_mse = MSE(self.y_test, self.y_pred)
        print('Test MSE is {}'.format(test_mse))