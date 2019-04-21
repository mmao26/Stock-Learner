"""MC3-P3 Part 4: Generating Training Data."""

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
    price = df['AAPL'].copy()

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

    Y = price.copy()
    for day in range(0, price.shape[0]-21):
        ret = (price[day+21]/price[day]) - 1.0
        if ret > 0.05:
            Y.ix[day] = 1
        elif ret < -0.05:
            Y.ix[day] = -1
        else:
            Y.ix[day] = 0
    Y = Y.ix[:price.shape[0]-21]

    data = pd.DataFrame(0.00, index = np.arange(464), columns = ['X1', 'X2', 'X3', 'Y'])
    data['X1'][0:464] = rm_norm.ix[20:484]
    data['X2'][0:464] = ema_norm.ix[20:484]
    data['X3'][0:464] = bbp.ix[20:484]
    data['Y'][0:464] = Y.ix[20:484]
    data_traning = np.array([data['X1'], data['X2'], data['X3'], data['Y']])
    np.savetxt("./orders-training.csv", data_traning.transpose(), delimiter=',')

if __name__ == "__main__":
    test_code()
