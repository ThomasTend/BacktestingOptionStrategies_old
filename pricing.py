"""
Author: Thomas Tendron

This file implements the Pricing Tool.
"""

# stats
from scipy import stats as st

# data manipulation modules
import pandas as pd
import numpy as np

class Option_Pricer():
    """
    Pricing class: Price options using Black-Scholes and extensions.
    For now only European options.
    """
    def __init__(self, T, S_0, sigma, K, r, option_type = "call"):
        """
        T = time to maturity (in years)
        S_0 = stock price at the time the option is priced
        sigma = volatility
        K = strike price
        r = continuously compounded risk-free rate. In practice, the zero-coupon risk-free interest rate for a maturity T.
        """
        self.T = T
        self.S_0 = S_0
        self.sigma = sigma
        self.K = K
        self.r = r
        self.d1 = (np.log(S_0 / K) + (r + sigma**2 / 2) * T) / (sigma * T**0.5)
        self.d2 = self.d1 - sigma * T**0.5
        self.option_type = option_type

    def set_strike_price(self):
        if self.option_type == "call":
            self.K = self.S_0 + 2.5
        else: # put
            self.K = self.S_0 - 2.5

    def compute_d1(self, sigma):
        return (np.log(self.S_0 / self.K) + (self.r + sigma**2 / 2) * self.T) / (sigma * self.T**0.5)

    def price_european_call(self, sigma, manual = False):
        if manual == True:
            d1 = self.compute_d1(sigma)
            d2 = d1 - sigma * self.T**0.5
            return self.S_0 * st.norm.cdf(d1) - self.K * np.exp(- self.r * self.T) * st.norm.cdf(d2)
        else:
            return self.S_0 * st.norm.cdf(self.d1) - self.K * np.exp(- self.r * self.T) * st.norm.cdf(self.d2)

    def price_european_put(self, sigma, manual = False):
        if manual == True:
            d1 = self.compute_d1(sigma)
            d2 = d1 - sigma * self.T**0.5
            return self.K * np.exp(- self.r * self.T) * st.norm.cdf(-d2) - self.S_0 * st.norm.cdf(-d1)
        else:
            return self.K * np.exp(- self.r * self.T) * st.norm.cdf(-self.d2) - self.S_0 * st.norm.cdf(-self.d1)

    def price_option(self, sigma, manual = False):
        if self.option_type == "call":
            return self.price_european_call(sigma, manual = manual)
        else: # put
            return self.price_european_put(sigma, manual = manual)

    def vega(self, sigma, manual = False):
        """
        We evaluate vega, the derivative of the Black-Scholes option price with respect to the volatility. Same for call or put.
        """
        if manual == True:
            d1 = self.compute_d1(sigma)
            return self.S_0 * self.T**0.5 * st.norm.pdf(d1)
        else:
            return self.S_0 * self.T**0.5 * st.norm.pdf(self.d1)