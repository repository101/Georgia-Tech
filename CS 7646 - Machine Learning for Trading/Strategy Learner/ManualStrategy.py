"""Manual Strategy.ManualStrategy

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
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author(self):
	return 'jadams334'


class ManualStrategy():
	
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
		PricesDataFrame.reset_index(drop=True, inplace=True)
		previousValue = 0
		holdings = 0
		# Replacing Orders
		PricesDataFrame['Orders'] = 0
		sma_buy_thresh = 0.95
		sma_sell_thresh = 1.05
		bollinger_buy_thresh = 0.6
		bollinger_sell_thresh = 0.4
		rsi_buy_thresh = 45
		rsi_sell_thresh = 55
		PricesDataFrame['Orders'].fillna(0.0, inplace=True)
		port_val, resultDF = msc.portValQuick(PricesDataFrame)
		daily_returns = msc.get_daily_returns(PricesDataFrame['Adj Close'])
		PricesDataFrame["Daily Returns"] = daily_returns
		PricesDataFrame["Positions"] = 0
		Positions = PricesDataFrame['Positions'].copy()
		Holdings = 0
		long = []
		short = []
		for index, row in PricesDataFrame.iterrows():
			if row['Daily Returns'] == 0:
				Positions.loc[index] = 0
			
			else:
				if (row['Bollinger Band Percentage'] < bollinger_buy_thresh) and (Holdings < 1000) \
						and (row["Relative Strength Index"] > rsi_buy_thresh) \
						and (row['Simple Moving Average Ratio'] < sma_sell_thresh):
					Positions[index - 1] += 1000
					Holdings += 1000
					long.append(index)
					if Holdings == 0:
						Positions[index - 1] += 1000
						Holdings += 1000
				elif (row['Bollinger Band Percentage'] > bollinger_sell_thresh) and (Holdings > -1000) \
						and (row["Relative Strength Index"] > rsi_sell_thresh) \
						and (row['Simple Moving Average Ratio'] > sma_buy_thresh):
					Positions[index - 1] -= 1000
					Holdings -= 1000
					short.append(index)
					if Holdings == 0:
						Positions[index - 1] -= 1000
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
		plt.clf()
		ax = benchmarkDF["Normalized Port Value"].plot(title='Benchmark Vs Manual Strategy Out Sample', label='Benchmark',
		                                               color="g")
		resultDF["Normalized Port Value"].plot(ax=ax, label="Manual Strategy", color="r")
		ax.set_xlabel('Days')
		ax.set_ylabel('Port Val')
		ax.legend(loc='best')
		ax.legend(fancybox=True, shadow=True)
		plt.tight_layout()
		# for lg in long:
		# 	plt.axvline(x=lg, color="k", linestyle="--")
		# for sh in short:
		# 	plt.axvline(x=sh, color="b", linestyle="--")
		fileName = '{} Benchmark Vs Manual Strategy Out Sample.png'.format(symbol[0])
		try:
			path = os.path.dirname(__file__)
			plt.savefig(os.path.join(path, fileName), dpi=300)
		except Exception as e:
			print "An error occurred when saving the graph "
			exc_type, exc_obj, exc_tb = sys.exc_info()
			print exc_obj
			print exc_tb.tb_lineno
			print e
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
		RSI_Mean = resultDF['Relative Strength Index'].mean()
		BBP_Mean = resultDF["Bollinger Band Percentage"].mean()
		Simple_Move_Mean = resultDF["Simple Moving Average Ratio"].mean()
		return results
	
	
if __name__ == "__main__":
	test = ManualStrategy()
	results = test.testPolicy(symbol=["JPM"], sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
	print ""
