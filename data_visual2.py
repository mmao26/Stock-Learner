"""MC3-P3 Part 4: ML-Based Trader."""

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


if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = None
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    day = 0
    trainX = data[:, 0:-1]
    trainY = data[:, -1]
    

    while(day < trainY.shape[0]):
        if (trainY[day] == 1.0):
            plt.scatter(trainX[day, 0], trainX[day, 2], color='g')
        elif (trainY[day] == -1.0):
            plt.scatter(trainX[day, 0], trainX[day, 2], color='r')
        else:
            plt.scatter(trainX[day, 0], trainX[day, 2], color='k')
        day = day + 1

    plt.grid(True)
    plt.show()

