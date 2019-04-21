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
    dates = pd.date_range('2008-01-01', '2009-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)
    price = df['AAPL']

    rm = get_rolling_mean(df['AAPL'], window=20)
    ema = get_exponential_mean(df['AAPL'], span=20, min_periods=0)
    rstd = get_rolling_std(df['AAPL'], window=20)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    
    prices_norm = df['AAPL']/df['AAPL'].ix[0]
    rm_norm = rm/df['AAPL'].ix[0]
    ema_norm = ema/df['AAPL'].ix[0]
    price_sma = df['AAPL'].ix[20:]/rm
    price_ema = df['AAPL'].ix[20:]/ema
    upper_band_norm = upper_band/df['AAPL'].ix[0]
    lower_band_norm = lower_band/df['AAPL'].ix[0]
    
    bbp = (price.ix[20:] - lower_band)/(upper_band - lower_band)

    while(day < price.shape[0]):
        if (rm_norm.ix[day] < 0.95) and (bbp.ix[day] < 0):
            orders.append([price.index[day].date(), 'AAPL', 'BUY', 200])
            day = day + 21
        elif (rm_norm.ix[day] > 1.05) and (bbp.ix[day] > 1):
            orders.append([price.index[day].date(), 'AAPL', 'SELL', 200])
            day = day + 21
        else:
            day = day + 1
    for order in orders:
        print ("     ".join(str(x) for x in order))
#np.savetxt("order-rules.csv", orders)

if __name__ == "__main__":
    test_code()
