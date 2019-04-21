"""MC3-P3 Part 1: Technical Indicators."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
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

    dates = pd.date_range('2008-01-01', '2009-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)

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
    
    plt.figure(1)
    ax1 = prices_norm.plot(title="Simple Moving Average (Window = 20)", label='AAPL')
    rm_norm.plot(label='SMA', ax=ax1)
    price_sma.plot(label='Price/SMA', ax=ax1)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized Price")
    ax1.legend(loc=0)
    plt.grid(True)
    
    plt.figure(2)
    ax2 = prices_norm.plot(title="Exponential Moving Average (Window = 20)", label='AAPL')
    ema_norm.plot(label='EMA', ax=ax2)
    price_ema.plot(label='Price/EMA', ax=ax2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Normalized Price")
    ax2.legend(loc=0)
    plt.grid(True)
    
    plt.figure(3)
    ax3 = prices_norm.plot(title="Bollinger Bands (Window = 20)", label='AAPL')
    rm_norm.plot(label='SMA', ax=ax3)
    upper_band_norm.plot(label='Upper Band', ax=ax3)
    lower_band_norm.plot(label='Lower Band', ax=ax3)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Normalized Price")
    ax3.legend(loc=0)
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    test_code()
