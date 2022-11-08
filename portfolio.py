"""
Portfolio class: set of positions and strategies.
"""

from strat import *

class Portfolio(Strat):
    def __init__(self, asset = "stock", asset_id = "MSFT", target = "Close", period = "max", days_to_pred = 3, num_lag_features = 7, hist_start_date = "2017-10-26", hist_end_date="2018-01-05", model_name='Lasso', use_cv=True, params={}):
        Strat.__init__(self, asset = asset, asset_id = asset_id, target = target, period = period, days_to_pred = days_to_pred, num_lag_features = num_lag_features, hist_start_date=hist_start_date, hist_end_date=hist_end_date, model_name=model_name, use_cv=use_cv, params=params)
        self.get_stock_options()
        self.covered_call()