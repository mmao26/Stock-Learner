"""MC3-P3 Part 6: ML-Based Trader Using out of Sample."""

import numpy as np
import math
import BagLearner as bl
import RTLearner as rt
import sys
import pandas as pd
import matplotlib.pyplot as plt
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

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = None
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    trainX = data[:, 0:-1]
    trainY = data[:, -1]

    # Obtain testX
    orders = []
    day = 20
    dates = pd.date_range('2010-01-01', '2011-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)
    price = df['AAPL'].copy()

    rm = get_rolling_mean(price, window=20)
    ema = get_exponential_mean(price, span=20, min_periods=0)
    rstd = get_rolling_std(price, window=20)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    rm_norm = rm/price.ix[0]
    ema_norm = ema/price.ix[0]
    
    bbp = (price.ix[20:] - lower_band)/(upper_band - lower_band)
    testX = np.zeros([484, 3])
    testX[:, 0] = rm_norm.ix[20:]
    testX[:, 1] = ema_norm.ix[20:]
    testX[:, 2] = bbp.ix[20:]


    # create a learner and train it
    #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    #learner = rt.RTLearner(leaf_size = 10, verbose=False)
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size': 5}, bags=1, verbose=False)
    learner.addEvidence(trainX, trainY) # train it

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions

    orders = []
    day = 20

    while(day < 504):
        if (predY[day-20] == 1.0):
            orders.append([price.index[day].date(), 'AAPL', 'BUY', 200])
            day = day + 21
        elif (predY[day-20] == -1.0):
            orders.append([price.index[day].date(), 'AAPL', 'SELL', 200])
            day = day + 21
        else:
            day = day + 1
    for order in orders:
        print "     ".join(str(x) for x in order)


