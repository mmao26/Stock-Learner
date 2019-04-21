"""MC3-P3 Part 4: ML-Based Trader."""

import numpy as np
import math
#import LinRegLearner as lrl
import BagLearner as bl
import RTLearner as rt
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import csv
from util import get_data, plot_data


if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = None
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    trainX = data[:, 0:-1]
    trainY = data[:, -1]
    
    # create a learner and train it
    #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    #learner = rt.RTLearner(leaf_size = 10, verbose=False)
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size': 5}, bags=2, verbose=False)
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions

    orders = []
    day = 20
    dates = pd.date_range('2008-01-01', '2009-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)
    price = df['AAPL']

    while(day < 484):
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

