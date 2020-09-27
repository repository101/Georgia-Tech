""""""
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
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Josh Adams (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: jadams334 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 903475599 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""

import scipy.optimize as spo
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data


# This is the function that will be tested by the autograder  		  	   		     		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		     		  		  		    	 		 		   		 		  
def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of symbols that make up the portfolio (note that your code should support any
        symbol in the data directory)
    :type syms: list
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case

    #region Generate Random Initial Allocations
    initial_allocs = [1/len(syms) for i in range(len(syms))]    # add code here to find the allocations
    #endregion

    cr, adr, sddr, sr = [
        0.25,
        0.001,
        0.0005,
        2.1,
    ]  # add code here to compute stats

    constraints = ({'type': 'eq', 'fun': lambda inputs: np.sum(inputs) - 1.0})
    bounds = [(0.0, 1.0) for i in range(len(syms))]

    optimized = spo.minimize(get_negative_sharpe_ratio, initial_allocs, prices, method='SLSQP',
                             constraints=constraints, bounds=bounds)

    results = get_portfolio_value(prices_dataframe=prices, allocs=optimized.x)

    optimized_allocs = optimized.x
    optimized_cumulative_return = results["cumulative_returns"]
    optimized_average_daily_return = results["average_daily_returns"]
    optimized_standard_deviation_of_daily_return = results["std_daily_returns"]
    optimized_sharpe_ratio = results["sharpe_ratio"]
    optimized_portfolio_value = results["port_val"]

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        plt.close("All")
        plt.style.use('ggplot')
        port_val_normalized = optimized_portfolio_value / optimized_portfolio_value[0]
        prices_SPY_normalized = prices_SPY / prices_SPY[0]

        df_temp = pd.concat([port_val_normalized, prices_SPY_normalized], keys=["Portfolio", "SPY"], axis=1)
        title = "Daily Portfolio Value and SPY"
        xlabel = "Date"
        ylabel = "Price"
        ax = df_temp.plot(title=title, fontsize=12, grid=True, alpha=0.75)
        ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
        ax.set_title(title, fontsize=15, weight='bold')
        ax.set_xlabel(xlabel, fontsize=15, weight='bold')
        ax.set_ylabel(ylabel, fontsize=15, weight='bold')
        plt.tight_layout()
        plt.savefig("plot.png")
        pass

    return optimized_allocs, optimized_cumulative_return, \
           optimized_average_daily_return, optimized_standard_deviation_of_daily_return,\
           optimized_sharpe_ratio

def get_daily_returns(port_value_df):
    # Parameter should be the portfolio_value_dataframe
    daily_returns = (port_value_df / port_value_df.shift(1)) - 1
    return daily_returns

def get_average_daily_returns(df):
    return df.mean()

def get_std_daily_returns(df):
    return df.std()

def fill_empty_data(df):
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def get_sharpe_ratio(daily_rets):
    return (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)

def get_negative_sharpe_ratio(allocations, portfolio):
    daily_return_2 = get_portfolio_value(prices_dataframe=portfolio, allocs=allocations, end_early=True)
    return ((daily_return_2.mean() / daily_return_2.std()) * np.sqrt(252)) * -1

def get_cumulative_returns(portfolio_dataframe):
    # Cumulative return should be calculated using a portfolio
    return (portfolio_dataframe[-1] / portfolio_dataframe[0]) - 1

def get_portfolio_value(prices_dataframe, allocs=[0.1, 0.1, 0.1, 0.1], start_val=1e6, end_early=False):
    normed = prices_dataframe / prices_dataframe.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    daily_returns = get_daily_returns(port_val)
    if end_early:
        return daily_returns
    daily_returns = daily_returns[1::]
    cumulative_return = get_cumulative_returns(port_val)
    average_daily_returns = get_average_daily_returns(daily_returns)
    std_daily_return = get_std_daily_returns(daily_returns)
    sharpe_ratio = get_sharpe_ratio(daily_returns)
    result_dict = {"port_val": port_val,
                   "daily_returns": daily_returns,
                   "cumulative_returns": cumulative_return,
                   "average_daily_returns": average_daily_returns,
                   "std_daily_returns": std_daily_return,
                   "sharpe_ratio": sharpe_ratio}
    return result_dict


def test_code():
    """  		  	   		     		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		     		  		  		    	 		 		   		 		  
    """
    # from optimization_tests import run_side_tests

    # run_side_tests()

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

    # Print statistics
    print(f"\nStart Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sum of Allocations: {allocations.sum()}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}\n")
    print("Finished")
    return


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()

