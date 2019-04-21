"""MC3-P3 Part 3: Rule-Based Trader."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import csv
from util import get_data, plot_data


def get_rolling_mean(values, window):
    return pd.rolling_mean(values, window=window)

def get_exponential_mean(values, span, min_periods):
    return pd.ewma(values, span=span, min_periods=min_periods)

def get_rolling_std(values, window):
    return pd.rolling_std(values, window=window)

def get_bollinger_bands(rm, rstd):
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band

def test_code():
    
    orders = []
    day = 20
    holdings = 0
    dates = pd.date_range('2010-01-01', '2011-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)
    price = df['AAPL']

    rm = get_rolling_mean(price, window=20)
    ema = get_exponential_mean(price, span=20, min_periods=0)
    rstd = get_rolling_std(price, window=20)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    
    prices_norm = price/price.ix[0]
    rm_norm = rm/price.ix[0]
    ema_norm = ema/price.ix[0]
    price_sma = price.ix[20:]/rm
    price_ema = price.ix[20:]/ema
    upper_band_norm = upper_band/price.ix[0]
    lower_band_norm = lower_band/price.ix[0]
    
    bbp = (price.ix[20:] - lower_band)/(upper_band - lower_band)

    while(day < price.shape[0]):
        if (bbp.ix[day] < 0):
            orders.append([price.index[day].date(), 'AAPL', 'BUY', 200])
            day = day + 21
            holdings = holdings + 200
        elif (bbp.ix[day] > 1) and (holdings >= 200):
            orders.append([price.index[day].date(), 'AAPL', 'SELL', 200])
            day = day + 21
            holdings = holdings - 200
        else:
            day = day + 1
    for order in orders:
        print "     ".join(str(x) for x in order)

if __name__ == "__main__":
    test_code()
