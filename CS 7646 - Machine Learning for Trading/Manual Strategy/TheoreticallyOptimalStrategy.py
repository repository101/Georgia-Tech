"""Manual Strategy.TheoreticallyOptimalStrategy

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
import sys
import marketsimcode as msc
import indicators as ind
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
	return 'jadams334'  # replace tb34 with your Georgia Tech username.


class TheoreticallyOptimalStrategy(object):
	def __init__(self):
		pass
	
	def author(self):
		return 'jadams334'  # replace tb34 with your Georgia Tech username.
	
	def testPolicy(self, symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2010, 6, 1), sv=100000):
		# get_port_val(symbol=symbol, startDate=sd, endDate=ed)
		OrdersDataFrame, PricesDataFrame, Symbol = msc.Manual_Strategy(symbol=symbol, sd=sd, ed=ed, sv=sv,
		                                                               getSPY=True, dropNA=True,
		                                                               tradeAmount=1000)
		OrdersDF_Prep = msc.Prep_OrderDF(OrdersDataFrame, symbol=Symbol)
		# results_1000 = compute_portvals(generatedOrderDataFrame=OrdersDF_Prep, start_val=sv, manual=True,
		#                                 get_Prices=False, pricesDF=PricesDataFrame, symbols=symbol)
		# Get Normalized Prices
		PricesDataFrame['Dates'] = PricesDataFrame.index
		ValidIndex = PricesDataFrame.copy(deep=True)
		PricesDataFrame.reset_index(drop=True, inplace=True)
		previousValue = 0
		holdings = 0
		# Replacing Orders
		PricesDataFrame['Orders'] = 0
		sma_buy_thresh = 0.95
		sma_sell_thresh = 1.02
		bollinger_buy_thresh = 0.4
		bollinger_sell_thresh = 0.6
		rsi_buy_thresh = 40
		rsi_sell_thresh = 60
		for day, row in PricesDataFrame.iterrows():
			if (row["Simple Moving Average Ratio"] < sma_buy_thresh) and (
					row["Bollinger Band Percentage"] < bollinger_buy_thresh) and (
					row["Relative Strength Index"] < rsi_buy_thresh):
				# Buy
				if (holdings <= 0) and (holdings >= -1):
					# This mean we cannot buy we can only do Nothing or Sell
					# Check indicators
					PricesDataFrame['Orders'].loc[day] = 1000
					holdings += 1
			elif (row["Simple Moving Average Ratio"] > sma_sell_thresh) and (
					row["Bollinger Band Percentage"] > bollinger_sell_thresh) and (
					row["Relative Strength Index"] > rsi_sell_thresh):
				# Sell
				if (holdings >= 0) and (holdings <= 1):
					# This means we can only do Nothing or Buy
					# Check Indicators
					PricesDataFrame['Orders'].loc[day] = -1000
					holdings += -1
			else:
				# Something was not caught and we should do nothing
				continue
			if (holdings > 1) or (holdings < -1):
				continue
			previousValue = holdings
		PricesDataFrame['Orders'].fillna(0.0, inplace=True)
		port_val, resultDF = msc.portValQuick(PricesDataFrame)
		daily_returns = msc.get_daily_returns(PricesDataFrame['Adj Close'])
		PricesDataFrame["Daily Returns"] = daily_returns
		PricesDataFrame["Positions"] = 0
		Positions = PricesDataFrame['Positions'].copy()
		Holdings = 0
		for index, row in PricesDataFrame.iterrows():
			if row['Daily Returns'] == 0:
				Positions.loc[index] = 0
				
			else:
				if (row['Daily Returns'] < 0) and (Holdings < 1000):
					Positions[index-1] += 1000
					Holdings += 1000
					if Holdings == 0:
						Positions[index-1] += 1000
						Holdings += 1000
				elif (row['Daily Returns'] > 0) and (Holdings > -1000):
					Positions[index-1] -= 1000
					Holdings -= 1000
					if Holdings == 0:
						Positions[index-1] -= 1000
						Holdings -= 1000
		PricesDataFrame["Positions"] = Positions
		port_val, resultDF = msc.portValQuick(PricesDataFrame, Theo=True)
		results = PricesDataFrame["Positions"]
		results = results.to_frame()
		results.set_index(PricesDataFrame["Dates"], inplace=True)
		resultDF["Normalized Port Value"] = resultDF["Port Value"] / resultDF["Port Value"][0]
		benchmarkDF = resultDF.copy()
		benchmarkDF['Positions'] = 0
		benchmarkDF["Positions"][0] = 1000
		benchmarkPortVal, benchmarkDF = msc.portValQuick(benchmarkDF, Theo=True, Bench=True)
		benchmarkDF["Normalized Port Value"] = benchmarkDF["Port Value"] / benchmarkDF["Port Value"][0]
		benchmarkDF.set_index("Dates", inplace=True)
		ValidIndex["Daily Returns"] = benchmarkDF["Daily Returns"]
		ValidIndex["Positions"] = benchmarkDF["Positions"]
		ValidIndex["Port Value"] = benchmarkDF["Port Value"]
		ValidIndex["Normalized Port Value"] = benchmarkDF["Normalized Port Value"]
		# newAx = plt.subplot()
		# ValidIndex["Normalized Port Value"].plot(ax=newAx, title='Benchmark Vs Theoretically Optimal', label='Benchmark', color="g")
		# resultDF["Normalized Port Value"].plot(ax=newAx, label="Theoretically Optimal", color="r")
		# newAx.set_xlabel('Days')
		# newAx.set_ylabel('Port Val')
		# newAx.legend(loc='best')
		# newAx.legend(fancybox=True, shadow=True)
		# plt.tight_layout()
		# fileName = '{} Benchmark Vs Theoretically Optimal.png'.format(symbol[0])
		# try:
		# 	path = os.path.dirname(__file__)
		# 	plt.savefig(os.path.join(path, fileName), dpi=300)
		# except Exception as e:
		# 	print "An error occurred when saving the graph "
		# 	exc_type, exc_obj, exc_tb = sys.exc_info()
		# 	print exc_obj
		# 	print exc_tb.tb_lineno
		# 	print e
		print ""
		test = benchmarkDF["Daily Returns"].std()
		Benchmark_Daily_Return_STD = msc.get_std_daily_returns(CompleteDF=benchmarkDF)
		Benchmark_Cumulative_Daily_Returns = msc.get_cumulative_returns(CompleteDF=benchmarkDF)
		Benchmark_Average_Daily_Returns = msc.get_avg_daily_return(CompleteDF=benchmarkDF)
		Bench_avg = Benchmark_Average_Daily_Returns.mean()
		Optimal_Daily_Returns_STD = msc.get_std_daily_returns(CompleteDF=resultDF)
		Optimal_Cumulative_Daily_Returns = msc.get_cumulative_returns(CompleteDF=resultDF)
		Optimal_Average_Daily_Returns = msc.get_avg_daily_return(CompleteDF=resultDF)
		Optimal_avg = Optimal_Average_Daily_Returns.mean()
		
		return results


if __name__ == "__main__":
	test = TheoreticallyOptimalStrategy()
	results = test.testPolicy(symbol=["JPM"], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
	print ""
