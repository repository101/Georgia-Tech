"""MC1-P2: Optimize a portfolio. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			     			  	   		   	  			  	
Atlanta, Georgia 30332 			  		 			     			  	   		   	  			  	
All Rights Reserved 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Template code for CS 4646/7646 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			     			  	   		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			     			  	   		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			     			  	   		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			     			  	   		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			     			  	   		   	  			  	
or edited. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
We do grant permission to share solutions privately with non-students such 			  		 			     			  	   		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			     			  	   		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			     			  	   		   	  			  	
GT honor code violation. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
-----do not edit anything above this line--- 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Student Name: Tucker Balch (replace with your name) 			  		 			     			  	   		   	  			  	
GT User ID: smarchienne3 (replace with your User ID)
GT ID: 903430342 (replace with your GT ID)
""" 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from util import get_data


plt.style.use("ggplot")


def portfolio_values(prices, allocs):
    normed = prices / prices.iloc[0]
    allocated = normed * allocs
    return allocated.sum(axis=1)


def cumulative_return(port_val):
    cum_ret = (port_val / port_val[0]) - 1
    return cum_ret[-1]


def avg_daily_returns(port_val):
    daily_ret = (port_val / port_val.shift(1)) - 1
    return daily_ret[1:].mean()


def std_daily_returns(port_val):
    daily_ret = (port_val / port_val.shift(1)) - 1
    return daily_ret[1:].std()


def sharpe_ratio(port_val, k=np.sqrt(252)):
    return k * avg_daily_returns(port_val) / std_daily_returns(port_val)


def minus_f(allocs, prices):
    port_val = portfolio_values(prices, allocs)
    return -1 * sharpe_ratio(port_val)


# This is the function that will be tested by the autograder 			  		 			     			  	   		   	  			  	
# The student must update this code to properly implement the functionality 			  		 			     			  	   		   	  			  	
def optimize_portfolio(
        sd=dt.datetime(2008,1,1),
        ed=dt.datetime(2009,1,1),
        syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
        gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range 			  		 			     			  	   		   	  			  	
    dates = pd.date_range(sd, ed) 			  		 			     			  	   		   	  			  	
    prices_all = get_data(syms, dates)  # automatically adds SPY 			  		 			     			  	   		   	  			  	
    prices = prices_all[syms]  # only portfolio symbols 			  		 			     			  	   		   	  			  	
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio 			  		 			     			  	   		   	  			  	
    # note that the values here ARE NOT meant to be correct for a test case
    x0 = np.asarray([1. / len(syms)] * len(syms))
    bounds = tuple((0.0, 1.0) for _ in xrange(len(syms)))
    constraints = ({'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs)})
    sol = minimize(minus_f, x0, prices, bounds=bounds, constraints=constraints)
    allocs = sol["x"]

    # Get daily portfolio value
    port_val = portfolio_values(prices, allocs)

    cr, adr, sddr, sr = [
        cumulative_return(port_val),
        avg_daily_returns(port_val),
        std_daily_returns(port_val),
        sharpe_ratio(port_val)
    ]

    # Normalization
    norm_prices_SPY = prices_SPY / prices_SPY[0]

    # Compare daily portfolio value with SPY using a normalized plot 			  		 			     			  	   		   	  			  	
    if gen_plot:
        df_temp = pd.concat([port_val, norm_prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plt.subplots()
        df_temp.plot(title="Daily portfolio value and SPY")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.savefig("figure.png")
        plt.show()

    return allocs, cr, adr, sddr, sr 			  		 			     			  	   		   	  			  	


def test_code(): 			  		 			     			  	   		   	  			  	
    # This function WILL NOT be called by the auto grader 			  		 			     			  	   		   	  			  	
    # Do not assume that any variables defined here are available to your function/code 			  		 			     			  	   		   	  			  	
    # It is only here to help you set up and test your code 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # Define input parameters 			  		 			     			  	   		   	  			  	
    # Note that ALL of these values will be set to different values by 			  		 			     			  	   		   	  			  	
    # the autograder! 			  		 			     			  	   		   	  			  	

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio 			  		 			     			  	   		   	  			  	
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd = start_date,
        ed = end_date,
        syms = symbols,
        gen_plot = True
    )

    # Print statistics 			  		 			     			  	   		   	  			  	
    print "Start Date:", start_date 			  		 			     			  	   		   	  			  	
    print "End Date:", end_date 			  		 			     			  	   		   	  			  	
    print "Symbols:", symbols 			  		 			     			  	   		   	  			  	
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr 			  		 			     			  	   		   	  			  	
    print "Volatility (stdev of daily returns):", sddr 			  		 			     			  	   		   	  			  	
    print "Average Daily Return:", adr 			  		 			     			  	   		   	  			  	
    print "Cumulative Return:", cr


if __name__ == "__main__": 			  		 			     			  	   		   	  			  	
    # This code WILL NOT be called by the auto grader 			  		 			     			  	   		   	  			  	
    # Do not assume that it will be called 			  		 			     			  	   		   	  			  	
    test_code() 