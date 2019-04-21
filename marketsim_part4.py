"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
from util import get_data, plot_data

def compute_portvals(orders_file, start_val = 100000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    # Read-in CSV file using the function provided
    orders_df = pd.read_csv(orders_file, index_col = 'Date', parse_dates = True, na_values = ['nan'])
    # Sort the Data Column
    orders_df.sort_index()
    # Obtain start_date and end_date
    start_date = dt.datetime(2008,1,1) #orders_df.index[0]
    end_date = dt.datetime(2009,12,31) #orders_df.index[-1]

    # Compute portfolio values in leverage range
    def ComputePortvals_Leverage(orders, start_date, end_date, Levg_limit = 1.5):
        
        symbols = orders.get('Symbol').unique().tolist()  # Get the symbol list
        dates = pd.date_range(start_date, end_date)  # Obtain the date range
        Prices_all = get_data(symbols, dates)  # automatically adds SPY
        Prices = Prices_all[symbols]  # only portfolio symbols
        Prices['CASH'] = 1  # add a column with the value of cash to the Prices DataFrame
        
        # Establish a Trades dataframe
        Trades = pd.DataFrame(columns = Prices.columns, index = Prices.index)
        Trades = Trades.fillna(value = 0)     # Initialize a Trades dataframe
        for idx, row in orders.iterrows():
            if row['Order'] == "BUY":
                Trades.loc[idx][row['Symbol']] = Trades.loc[idx][row['Symbol']] + row['Shares']
                Trades.loc[idx]['CASH'] = Trades.loc[idx]['CASH'] - Prices.loc[idx][row['Symbol']] * row['Shares']
            else:
                Trades.loc[idx][row['Symbol']] = Trades.loc[idx][row['Symbol']] - row['Shares']
                Trades.loc[idx]['CASH'] = Trades.loc[idx]['CASH'] + Prices.loc[idx][row['Symbol']] * row['Shares']

        # Establish a Holdings dataframe
        Holdings = pd.DataFrame(columns = Prices.columns, index = Prices.index)
        Holdings = Holdings.fillna(value = 0)    # Initialize a Holdings dataframe
        Holdings['CASH'] = start_val
        Holdings = Holdings + Trades.cumsum()
        
        # Establish the Values dataframe
        Values = Prices * Holdings
        
        # Calculate portfolio Leverage
        temp1 = Values.ix[:, :-1][Values > 0].sum(axis = 1).fillna(value = 0) - Values.ix[:, :-1][Values < 0].sum(axis = 1).fillna(value = 0)
        temp2 = Values.ix[:, :-1][Values > 0].sum(axis = 1).fillna(value = 0) + Values.ix[:, :-1][Values < 0].sum(axis = 1).fillna(value = 0)
        cash = Values.ix[:, -1]
        leverage = temp1 / (temp2 + cash)

        # Identify and delete orders that cause leverage to exceed 1.5
        if any(i for i in leverage.index if leverage[i] >= Levg_limit):
            orders = orders.drop([next(i for i in leverage.index if leverage[i] >= Levg_limit)])
            return ComputePortvals_Leverage(orders, start_date, end_date)
        # No orders that cause leverage to exceed 1.5, return portfolio values
        else:
            return Values.sum(axis=1)

    portvals = ComputePortvals_Leverage(orders_df, start_date, end_date)
    return portvals


def author(self):
    return 'mmao33'

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of1 = "./orders-rule.csv"
    of2 = "./orders-ML.csv"
    sv = 100000

    # Process orders
    portvals_rule = compute_portvals(orders_file = of1, start_val = sv)
    portvals_ml = compute_portvals(orders_file = of2, start_val = sv)
    """
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    """
    dates = pd.date_range('2008-01-01', '2009-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)
    prices = df['AAPL']

    value_bench = prices.copy()
    value_bench.ix[0] = sv

    for i in range(1, prices.shape[0]):
        value_bench.ix[i] = sv - 200*prices.ix[0] + 200*prices.ix[i]

    normed_bench = value_bench/value_bench.ix[0]
    normed_rule = portvals_rule/portvals_rule.ix[0]
    normed_ml = portvals_ml/portvals_ml.ix[0]

    print "Volatility for benchmark:", np.std(normed_bench)
    print "Volatility for rule:", np.std(normed_rule)
    print "Volatility for ml:", np.std(normed_ml)

    print "Mean for benchmark:", np.mean(normed_bench)
    print "Mean for rule:", np.mean(normed_rule)
    print "Mean for ml:", np.mean(normed_ml)

    print "Cumulative Ret for benchmark:", normed_bench.ix[-1]
    print "Cumulative Ret for rule:", normed_rule.ix[-1]
    print "Cumulative Ret for ml:", normed_ml.ix[-1]

    df_temp = pd.concat([normed_rule, normed_ml, normed_bench], keys=['Rule-Based Portfolio', 'ML-Based Portfolio', 'Benchmark'], axis=1)
    ax = df_temp.plot(color=['b', 'g', 'k'])
    ax.set_title('Rule-Based Portfolio, ML-Based Portfolio and Benchmark')
    ax.set_ylabel('Normalized Price')
    ax.set_xlabel('Date')
    ax.legend(loc=0)
    plt.grid(True)
    plt.axvline(dt.datetime(2008,1,31), color='r', linestyle='dashed', linewidth=1.5)
    plt.axvline(dt.datetime(2008,3,3), color='g', linestyle='dashed', linewidth=1.5)
    plt.axvline(dt.datetime(2008,4,3), color='g', linestyle='dashed', linewidth=1.5)
    plt.axvline(dt.datetime(2008,9,17), color='r', linestyle='dashed', linewidth=1.5)

    plt.axvline(dt.datetime(2008,12,26), color='g', linestyle='dashed', linewidth=1.5)
    plt.axvline(dt.datetime(2008,2,17), color='g', linestyle='dashed', linewidth=1.5)
    plt.axvline(dt.datetime(2008,5,1), color='g', linestyle='dashed', linewidth=1.5)
    plt.axvline(dt.datetime(2008,6,18), color='g', linestyle='dashed', linewidth=1.5)
    plt.axvline(dt.datetime(2008,7,20), color='g', linestyle='dashed', linewidth=1.5)
    plt.show()
    
if __name__ == "__main__":
    test_code()
