"""MC2-P1: Market simulator. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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
GT User ID: jadams334(replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def author():
	return 'jadams334'  # replace tb34 with your Georgia Tech username.


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
	# this is the function the autograder will call to test your code
	# NOTE: orders_file may be a string, or it may be a file object. Your
	# code should work correctly with either input
	# TODO: Your code here

	# Orders Data Frame
	ordersDataFrame, symbols, start_date, end_date = get_Orders_DataFrame(orders_file)
	# At this point our orderDataFrame will have index of dates

	# Price Data Frame
	pricesDataFrame, spyDataFrame = get_Price_DataFrame(list(symbols), start_date, end_date)
	pricesDataFrame_Shape = pricesDataFrame.shape
	# Create new ordersDataFrame consisting of only days which SPY was traded.
	ordersDataFrame = ordersDataFrame[ordersDataFrame.index.isin(spyDataFrame.index)]       # Remove trades that occur on non tradeable days
	pricesDataFrame = pricesDataFrame[symbols]  # remove SPY
	pricesDataFrame.dropna(axis='rows', how='all', inplace=True)
	pricesDataFrame["Cash"] = 1.0

	# Trades Data Frame
	tradesDataFrame = pricesDataFrame.copy()
	tradesDataFrame[:] = 0.0
	tradesDataFrame = populate_TradeDataFrame(tradesDataFrame, ordersDataFrame,
	                                           pricesDataFrame, symbols,
	                                           commission, impact, start_val)
	# Holdings Data Frame
	holdingsDataFrame = get_Holdings_DataFrame(tradesDataFrame, start_val)

	# Values Data Frame
	valuesDataFrame = holdingsDataFrame * pricesDataFrame
	portvals = valuesDataFrame.sum(axis=1)

	rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())
	return rv


def get_Holdings_DataFrame(tradesDataFrame, start_val):
	holdingsDataFrame = tradesDataFrame.copy()
	holdingsDataFrame = holdingsDataFrame.cumsum(axis=0)        # This will add the previous values in Cash to the current value
	return holdingsDataFrame


def get_Price_DataFrame(symbols, start_date, end_date):
	if not isinstance(symbols, list):
		symbols = list(symbols)
	pricesDataFrame = get_data(symbols, pd.date_range(start_date, end_date), addSPY=True)
	# pricesDataFrame = pricesDataFrame.resample("D").fillna(method='ffill')
	# pricesDataFrame = pricesDataFrame.resample("D").fillna(method='bfill')
	spyDataFrame = pricesDataFrame['SPY']
	return pricesDataFrame, spyDataFrame


def get_Orders_DataFrame(orders_file):
	ordersDataFrame = pd.read_csv(orders_file, delimiter=",", header=0, parse_dates=['Date']).sort_values(by='Date')
	symbols = ordersDataFrame["Symbol"].unique()
	symbols = list(symbols)
	if '$SPX' not in symbols:
		symbols += ['$SPX']
	start_date = ordersDataFrame.iloc[0, 0]
	end_date = ordersDataFrame.iloc[-1, 0]
	ordersDataFrame.set_index('Date', inplace=True)
	return ordersDataFrame, symbols, start_date, end_date


def populate_TradeDataFrame(tradeDF, orderDF, pricesDf, symbols, commission, impact, start_val):
	tradeDF['Cash'] = 0.0
	tradeDF['Cash'][0] = start_val
	tradeDF[symbols] = 0.0
	symbol_dict = {}
	for i in symbols:
		symbol_dict[i] = 0
	for index, row in orderDF.iterrows():
		val = 0.0   # Reset Val to 0 after each iteration as to not carry over values
		# Date | Symbol | Order | Shares
		pricesForThatDay = pricesDf.loc[index]
		if row.Order.lower() == "buy":
			val = row.Shares + symbol_dict[row.Symbol]      # Val is used to keep track of total for the stocks and update dictionary
			tradeDF.loc[index][row.Symbol] = row.Shares + tradeDF.loc[index][row.Symbol]
			symbol_dict[row.Symbol] = val
			impact_price = pricesForThatDay[row.Symbol] * (1 + impact)
			price_shares = ((impact_price * row.Shares) * -1)
			cash_val = (tradeDF.loc[index]['Cash'])
			cashColumn = price_shares + cash_val - commission
			tradeDF.loc[index]['Cash'] = cashColumn
		else:
			# Here we are going to SELL so we want to subtract those shares (-row.Shares)
			# We location the date using tradeDF.loc on row.Date--essentially selecting the row/index == to that date.
			#       then since we are selling set the value in that row[Symbol] = to -row.shares

			# ORDER - 2011-12-27 Buy 2200 Shares of Google
			# ORDER - 2011-12-28 Sell 2200 Shares of IBM
			#   2011-12-27    AAPL   IBM   GOOG
			#                  0      0    +2200
			#
			#   2011-12-28    AAPL   IBM   GOOG
			#                  0    -2200    0

			val = -row.Shares + symbol_dict[row.Symbol]
			# We do not want to keep track of the stocks in the tradesDataFrame, just the movement of Buys/Sells
			tradeDF.loc[index][row.Symbol] = -row.Shares + tradeDF.loc[index][row.Symbol]
			symbol_dict[row.Symbol] = val
			impact_price = pricesForThatDay[row.Symbol] * (1 - impact)
			price_shares = (impact_price * row.Shares)
			cash_val = (tradeDF.loc[index]['Cash'])
			cashColumn = price_shares + cash_val - commission
			tradeDF.loc[index]['Cash'] = cashColumn
	return tradeDF


def test_code():
	# this is a helper function you can use to test your code
	# note that during autograding his function will not be called.
	# Define input parameters

	of = "./orders/orders-11.csv"
	sv = 1000000

	# Process orders
	portvals = compute_portvals(orders_file=of, start_val=sv)
	if isinstance(portvals, pd.DataFrame):
		portvals = portvals[portvals.columns[0]]  # just get the first column
	else:
		"warning, code did not return a DataFrame"

	# Get portfolio stats
	# Here we just fake the data. you should use your code from previous assignments.
	start_date = dt.datetime(2008, 1, 1)
	end_date = dt.datetime(2008, 6, 1)
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
	cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

	# Compare portfolio against $SPX
	print "Date Range: {} to {}".format(start_date, end_date)
	print
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
	print
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
	print
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
	print
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
	print
	print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
	test_code()
	# portvals = compute_portvals(orders_file="orders-short.csv", start_val=100000)
