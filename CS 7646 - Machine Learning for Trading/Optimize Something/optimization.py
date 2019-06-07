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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
from scipy.optimize import minimize

def getSharpERatio(port_val, k=np.sqrt(252)):
	return k * getAverageDailyReturns(port_val) / getSTDDailyReturns(port_val)


def getPortfolioValues(prices, allocs):
	# Normalize the data
	normed = prices / prices.iloc[0]
	allocated = normed * allocs
	return allocated.sum(axis=1)


def optFunc(allocs, prices):
	# This is the function we will pass to minimize
	port_val = getPortfolioValues(prices, allocs)
	return -1 * getSharpERatio(port_val)


def getAverageDailyReturns(port_val):
	dailyReturns = (port_val / port_val.shift(1)) - 1
	return dailyReturns[1:].mean()


def getCumulativeReturns(port_val):
	cumulativeReturns = (port_val / port_val[0]) - 1
	return cumulativeReturns[-1]


def getSTDDailyReturns(port_val):
	stdDailyReturn = (port_val / port_val.shift(1)) - 1
	return stdDailyReturn[1:].std()


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
	# Read in adjusted closing prices for given symbols, date range
	dates = pd.date_range(sd, ed)
	prices_all = get_data(syms, dates)  # automatically adds SPY
	prices = prices_all[syms]  # only portfolio symbols
	prices_SPY = prices_all['SPY']  # only SPY, for comparison later

	# find the allocations for the optimal portfolio
	# note that the values here ARE NOT meant to be correct for a test case
	xGuess = np.asarray([1.0 / len(syms)] * len(syms))

	bounds = tuple((0.0, 1.0) for _ in xrange(len(syms)))

	constraints = ({'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs)})

	optimalResult = minimize(optFunc, xGuess, prices, bounds=bounds, constraints=constraints)

	allocs = optimalResult['x']

	port_val = getPortfolioValues(prices, allocs)

	cr, adr, sddr, sr = [getCumulativeReturns(port_val),
	                     getAverageDailyReturns(port_val,),
	                     getSTDDailyReturns(port_val),
	                     getSharpERatio(port_val)]

	# Normalize SPY
	normalizedSPYPrices = prices_SPY / prices_SPY[0]

	# Compare daily portfolio value with SPY using a normalized plot
	if gen_plot:
		# add code to plot here
		df_temp = pd.concat([port_val, normalizedSPYPrices], keys=['Portfolio', 'SPY'], axis=1)
		plt.subplots()
		plt.style.use("ggplot")
		plot_data(df_temp, title="Daily portfolio value and SPY", xlabel='Date', ylabel='Price')

	return allocs, cr, adr, sddr, sr


def test_code():
	# This function WILL NOT be called by the auto grader
	# Do not assume that any variables defined here are available to your function/code
	# It is only here to help you set up and test your code

	# Define input parameters
	# Note that ALL of these values will be set to different values by
	# the autograder!

	start_date = dt.datetime(2008, 6, 1)
	end_date = dt.datetime(2009, 6, 1)
	symbols = ['IBM', 'X', 'GLD', 'JPM']

	# Assess the portfolio
	allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

	# Print statistics
	print "Start Date:", start_date
	print "End Date:", end_date
	print "Symbols:", symbols
	print "Allocations:", allocations
	print "Sharpe Ratio:", sr
	print "Volatility (stdev of daily returns):", sddr
	print "Average Daily Return:", adr
	print "Cumulative Return:", cr


# HINTS FOR PANDAS
#  -- Fill Forward First --
#       - When stocks are missing data you take the last known value and fill forward that value until
#           you reach data that is known
#
#  -- Fill Backward Second --
#       - Fill data backward second and what that means is if starting data is
# 		    unknown take the first known data point and fill it backwards to the beginning of the time frame
#
#  -- Fill Forward and Backward in Pandas --
#		- pandas.fillna(method='ffill', inplace=True) ~~~ Fill Forward
#		- pandas.fillna(method='bfill', inplace=True) ~~~ Fill Backward

#  -- Calculate Daily Return the Plot and then make Histogram --
#       - First copy the dataframe to a variable
#           daily_returns = df.copy()
#       - Second Calculate daily returns using "rolling"
#           daily_returns[1:] = (df[1:]/ df[:-1].values) - 1
#      - Third Set the daily returns for row 0 to 0
#           daily_returns.ix[0,:] = 0
#      - TO PLOT
#           plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")
#      - TO Histogram
#           daily_returns.hist()
#               - Specify the number of bins for the histogram, pass bins=#
#                   daily_returns.hist(bins=20)

#   -- Add vertical lines to graph, such as standard deviation or mean --
#       - Calculate mean first
#           mean = daily_returns['SPY'].mean()
#           plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
#       - Calculate Standard deviation first
#           std = daily_returns['SPY'].std()
#           plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
#           plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
#           plt.show()

#   -- Calculate Kurtosis --
#       - The larger the number returned the 'fatter' the tail
#           daily_returns.kurtosis()

#   -- Graph 2 histograms on one graph --
#       - Create a dataframe with both sets of data
#           df = get_data(symbols, dates)
#       - Get the daily returns
#           daily_returns = compute_daily_returns(df)
#       - Plot both histograms on one graph
#           daily_returns['SPY'].hist(bins=20, label="SPY")
#           daily_returns['XOM'].hist(bins=20, label="XOM")
#           plt.legend(loc="upper right)
#           plt.show()

#   -- BETA --
#       - Beta means how reactive the stock is
#           - The slope suggests that as the stock market goes up X%, that stock also goes up X%
#           - The rise/run, as the stock market moves "run"% the stock moves "rise"%
#           - Slope of 2/1, as the stock market goes up 1% the stock goes up 2%

#   -- ALPHA --
#       - Alpha is the point at which Beta crosses the vertical axis
#       - As in the point at which Beta(X) is 0
#       - If Alpha is positive that stock is returning a little more than the market overall

#   -- Scatterplot two different stocks --
#       daily_returns.plot(kind='scatter', x='SPY', y='XOM')
#       plt.show()

#   -- Fit a line to Scatterplot --
#       beat_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
#       - This returns the polynomial coefficient and y-intercept , the 1 denotes the degree of the function

#   -- Graph the fitted line --
#       plt.plot(daily_returns['SPY'], beta_XOM * daily_returns['SPY'] + alpha_XOM, '-', color='r')

#   -- Calculate the Correlation Coefficient --
#       print daily_returns.corr(method='pearson')
#       - This will display a matrix of correlations


#   -- PORTFOLIO --
#       - Get Prices Dataframe
#       - Then Normalize the Data  => prices/prices[0] => prices divided by first row
#       - Multiple the normed values by the allocations => normed * allocs
#       - pos_vals = alloced * start_val  => position values
#       - port_val = pos_vals.sum(axis=1) => portfolio value
#       - Now calculate Daily_returns (hint, the first value will always be 0 in daily returns)
#           -- When using daily_returns do not include the first value in calculations
#           daily_rets = daily_rets[1:]
#       - Calculate [Cumulative_Returns, Average_Daily_Returns, STD_Daily_Returns, SharpE_Ration]
#           -- Cumulative_Returns = (port_val[-1]/port_val[0]) - 1
#           -- Average_Daily_Returns = daily_rets.mean()
#           -- STD_Daily_Returns = daily_rets.std()
#           -- SharpR_Ratio = (Portfolio_Return - RiskFree_RateOfReturn) / STD_of_PortfolioReturn
#               --- Sharp Ratio is typically based on an annual measure
#                   ---- When using the Sharp ratio with something other than annual you need to multiply by K
#                   ---- K = sqrt(# of sampler per year)
#                   ---- daily_k = sqrt(252)
#                   ---- weekly_k = sqrt(52)
#                   ---- monthly_k = sqrt(12).
#               --- mean(daily_returns - daily_riskFreeReturns)/ std(daily_returns)
#               --- Typical daily_riskFreeReturns = 0
#               --- There are 252 days in a trading year
#               --- (Take the 252 root of ((StartingValue) (RiskFree as a decimal)) )- 1
#           -- Calculate Error
#               --- np.sum((data[:,1] - (line[0] * data[:,0] + line[1]))**2).

#   -- OPTIMIZE --
#       - Minimize Using SciPy
#           -- import scipy.optimize as spo
#               f in minimize is a function we define
#                   def f(x):
#                       Y = (X - 1.5)**2 + 0.5
#                       return Y
#               options={'disp': True} just makes the process verbose.
#               Xguess = 2.0
#               min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True}).

#   -- PARAMETERIZED MODEL --
#       - Formula for a line f(x) = mx + b
#       - The minimizer with vary the slope and y intercept to find the min
#           .


if __name__ == "__main__":
	# This code WILL NOT be called by the auto grader
	# Do not assume that it will be called
	test_code()
	print ""
