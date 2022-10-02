"""
Portfolio class: set of positions and strategies.
"""

from strat import *

class Portfolio(Strat):
    def __init__(self, asset = "stock", asset_id = "MSFT", target = "Close", period = "max", days_to_pred = 2, num_lag_features = 20, start_date = pd.to_datetime("2022-06-06", format="%Y-%m-%d")):
        Strat.__init__(self, asset = asset, asset_id = asset_id, target = target, period = period, days_to_pred = days_to_pred, num_lag_features = num_lag_features, start_date=start_date)
        self.get_stock_options()
        self.covered_call()