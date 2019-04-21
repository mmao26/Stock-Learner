"""MC3-P3 Part 2: Best Strategy."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import csv
from util import get_data, plot_data

dates = pd.date_range('2008-01-01', '2009-12-31')
symbols = ['AAPL']
df = get_data(symbols, dates)
prices = df['AAPL']
num_share = 0
cash = 100000.0

value_bench = prices.copy()
value_bench.ix[0] = 100000.0

for i in range(1, prices.shape[0]):
    value_bench.ix[i] = 100000.0 - 200*prices.ix[0] + 200*prices.ix[i]

dr_bench = prices.copy()
dr_bench[1:] = (value_bench[1:]/ value_bench[:-1].values)-1
dr_bench = dr_bench.ix[1:]

Bench_Final = 100000 - 200*prices.ix[0] + 200*prices.ix[-1]
cr_bench = Bench_Final/100000.0 -1

value_best = prices.copy()
value_best.ix[0] = 100000.0

for i in range(1, prices.shape[0]):
    if prices.ix[i] > prices.ix[i-1] and cash >= prices.ix[i-1]*200:   # Buy
            cash = cash - prices.ix[i-1]*200
            num_share = num_share + 200
            value_best.ix[i] = cash + num_share*prices.ix[i]
    
    elif prices.ix[i] < prices.ix[i-1] and num_share >= 200: # Sell
            cash = cash + prices.ix[i-1]*200
            num_share = num_share - 200
            value_best.ix[i] = cash + num_share*prices.ix[i]
    else:
        num_share = num_share
        cash = cash
        value_best.ix[i] = value_best.ix[i-1]

dr_best = prices.copy()
dr_best[1:] = (value_best[1:]/ value_best[:-1].values)-1
dr_best = dr_best.ix[1:]

Final_Value = value_best.ix[-1] #cash + num_share * prices.ix[-1]
cr_best = Final_Value/100000.0 -1

normed_best = value_best/value_best.ix[0]
normed_bench = value_bench/value_bench.ix[0]
df_temp = pd.concat([normed_best, normed_bench], keys=['Best Possible Strategy', 'Benchmark'], axis=1)
ax = df_temp.plot(color=['b','k'])
ax.set_title('Best Possible Strategy and Benchmark')
ax.set_ylabel('Normalized Price')
ax.set_xlabel('Date')
plt.grid(True)
plt.show()

print "Final Value for Best Strategy:", Final_Value
print "Final Value for Benchmark:", Bench_Final

print "Cumulative Return for Best Strategy:", cr_best
print "Cumulative Return for Benchmark:", cr_bench

print "Volatility for Best Strategy:", np.std(dr_best)
print "Volatility for Benchmark:", np.std(dr_bench)

print "Mean of Daily Return for Best Strategy:", np.mean(dr_best)
print "Mean of Daily Return for Benchmark:", np.mean(dr_bench)

