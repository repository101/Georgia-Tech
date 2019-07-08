"""Manual Strategy.marketsimcode

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
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
	return 'jadams334'  # replace tb34 with your Georgia Tech username.


def temp(orders):
	return


def Optimal_Strategy(priceDF):

	priceDF[["Port Value", "Impact Price", "Commission", "Cumulative Port Value", "Orders"]] = 0.0
	current_index = 0
	current_Adj_Close = 0
	previous_index = 0
	previous_row = 0
	previous_Adj_Close = 0
	current_Holding = 0
	numTrades = 0
	for index, row in priceDF.iterrows():
		if previous_index == 0:
			current_index = index
			previous_index = index
			current_Adj_Close = row["Adj Close"]
			previous_Adj_Close = row["Adj Close"]
			previous_row = row
		else:
			if current_Adj_Close == 0:
				current_Adj_Close = row["Adj Close"]
			else:
				current_Adj_Close = row["Adj Close"]
				
			if current_Adj_Close != previous_Adj_Close:
				if current_Adj_Close > previous_Adj_Close:
					# Buy
					if numTrades == 0:
						# initiate first trade of 1000
						current_Holding += 1000
						numTrades += 1
					else:
						numTrades += 1
				if current_Adj_Close < previous_Adj_Close:
					# Sell
					if numTrades == 0:
						# Initiate first trade of 1000
						current_Holding -= 1000
						numTrades += 1
					else:
						# Check holdings
						if (current_Holding >= 0):
							result = row
						numTrades += 1
			else:
				continue
			previous_row = row

	return


def get_cumulative_returns(port_val=None, CompleteDF=None, symbolToUse=None):
	if symbolToUse is not None:
		cumulative_return = (port_val[-1] / port_val[0]) - 1
		return cumulative_return[symbolToUse]
	else:
		if CompleteDF is not None:
			return (CompleteDF["Port Value"].iloc[-1] / CompleteDF["Port Value"].iloc[0]) - 1
		else:
			return (port_val[-1] / port_val[0]) - 1
		

def get_std_daily_returns(dailyReturnDataFrame=None, CompleteDF=None, symbolToUse=None):
	if symbolToUse is not None:
		dailyReturnDataFrame.ix[0, :] = 0
		return dailyReturnDataFrame[symbolToUse].std()
	else:
		if CompleteDF is not None:
			return CompleteDF["Daily Returns"].std()


def get_daily_returns(priceDataFrame):
	daily_rets = (priceDataFrame / priceDataFrame.shift(1)) - 1
	daily_rets.fillna(0, inplace=True)
	return daily_rets


def get_avg_daily_return(dailyReturnDataFrame=None, CompleteDF=None, symbolToUse=None):
	if symbolToUse is not None:
		dailyReturnDataFrame.ix[0, :] = 0
		return dailyReturnDataFrame[symbolToUse].mean()
	else:
		if CompleteDF is not None:
			return get_daily_returns(CompleteDF["Port Value"])


def get_port_val(symbol='JPM', startDate=dt.datetime(2010, 1, 1), endDate=dt.datetime(2011, 12, 31)):
	if isinstance(symbol, list):
		pricesDF = get_Price_DataFrame(symbols=symbol, start_date=startDate, end_date=endDate, getSPY=False)
	else:
		pricesDF = get_Price_DataFrame(symbols=[symbol], start_date=startDate, end_date=endDate, getSPY=False)
	pricesDF.dropna(axis='rows', how='all', inplace=True)
	generate_order_DataFrame(pricesDF)
	return pricesDF


def generate_order_DataFrame(pricesDataFrame):
	orderDataFrame = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
	orderDataFrame['Date'] = pricesDataFrame.index
	orderDataFrame['Symbol'] = 'JPM'
	orderDataFrame['Order'] = 0
	orderDataFrame['Shares'] = 0
	options = [-1, 0, 1]
	amount = [1000, 2000]
	bestVal = np.asarray([0, 0, 0, 0, 0])
	tempDF = orderDataFrame.copy()
	prevVal = 0
	totalHolding = 0
	for index, row in tempDF.iterrows():
		bestVal = np.asarray([0, 0, 0, 0, 0])
		if prevVal == 'Buy2K':
			options = [1]
			amount = [1000, 2000]
		if prevVal == 'Buy1K':
			options = [-1, 1]
			amount = [1000]
		if prevVal == 'Nothing':
			options = [-1, 1]
			amount = [1000, 2000]
		if prevVal == 'Sell1K':
			options = [-1, 1]
			amount = [1000]
		if prevVal == 'Sell2K':
			options = [1]
			amount = [1000, 2000]
		nothingDF = orderDataFrame.copy()
		nothingDF['Shares'][index] = 0
		nothingDF['Order'][index] = 0
		nothingResults = compute_portvals(manual=True, generatedOrderDataFrame=nothingDF)
		bestVal[2] = nothingResults.iloc[-1]
		for opt in options:
			for amt in amount:
				if ((totalHolding + (opt * amt)) > 2000) or ((totalHolding + (opt * amt)) < -2000):
					continue
				tmpDF = orderDataFrame.copy()
				tmpDF['Shares'][index] = amt
				tmpDF['Order'][index] = opt
				results = compute_portvals(manual=True, generatedOrderDataFrame=tmpDF)
				if opt == -1:
					if amt == 1000:
						# Sell 1000
						bestVal[1] = results.iloc[-1]
					else:
						# Sell 2000
						bestVal[0] = results.iloc[-1]
				else:
					if amt == 1000:
						# Buy 1000
						bestVal[3] = results.iloc[-1]
					else:
						# Buy 2000
						bestVal[4] = results.iloc[-1]
		largestIndex = bestVal.argmax()
		if largestIndex == 0:
			prevVal = "Sell2K"
			orderDataFrame['Order'][index] = "Sell"
			orderDataFrame['Shares'][index] = 2000
			totalHolding -= 2000
		if largestIndex == 1:
			prevVal = "Sell1K"
			orderDataFrame['Order'][index] = 'Sell'
			orderDataFrame['Shares'][index] = 1000
			totalHolding -= 1000
		if largestIndex == 2:
			prevVal = "Nothing"
			orderDataFrame['Order'][index] = 0
			orderDataFrame['Shares'][index] = 0
			totalHolding += 0
		if largestIndex == 3:
			prevVal = "Buy1K"
			orderDataFrame['Order'][index] = "Buy"
			orderDataFrame['Shares'][index] = -1000
			totalHolding += 1000
		if largestIndex == 4:
			prevVal = "Buy2K"
			orderDataFrame['Order'][index] = "Buy"
			orderDataFrame['Shares'][index] = -2000
			totalHolding += 2000
	results = compute_portvals(manual=True, generatedOrderDataFrame=orderDataFrame)
	return orderDataFrame


def Relative_Strength_Index(PriceDataFrame, symbols, lookback=14):
	# Ratio of
	# On days the stock goes up, how much it goes up
	#                       VS
	# On days the stock goes down, how much it goes down
	
	# Symbol Overbought:
	#   RSI > 70
	
	# Symbol Oversold:
	#   RSI < 30
	
	price_diff = PriceDataFrame['Adj Close'].diff()
	gain = price_diff.mask(price_diff < 0, 0)
	loss = price_diff.mask(price_diff > 0, 0)
	relative_strength = abs(gain.ewm(com=13, min_periods=14).mean() / loss.ewm(com=13, min_periods=14).mean())
	relative_strength_index = 100.0 - (100.0 / (1 + relative_strength))
	PriceDataFrame["Relative Strength Index"] = relative_strength_index
	return relative_strength_index


def BollingerBand_Percentage(PricesDataFrame, UpperBand, LowerBand):
	# Symbol Overbought:
	#   BollingerBand_% > 1
	
	# Symbol Oversold:
	#   BollingerBand_% < 0
	return (PricesDataFrame['Adj Close'] - LowerBand) / (UpperBand - LowerBand)


def PriceSMA_Ratio(PricesDataFrame, SimpleMovingAverage):
	# Symbol Overbought:
	#   Price/SMA_Ratio > 1.05
	
	# Symbol Oversold:
	#   Price/SMA_Ratio < 0.95
	return PricesDataFrame['Adj Close'] / SimpleMovingAverage


def getRollingMean(Prices_DF, window):
	try:
		rollingMean = pd.rolling_mean(Prices_DF, window=window)
		return rollingMean
	except Exception as RollingMeanException:
		print "Error occurred when attempting to calculate the rolling mean"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print RollingMeanException
	return Prices_DF


def getBollingerBands(rollingMean, rollingStandardDeviation):
	try:
		upperBand = rollingMean + rollingStandardDeviation * 2
		lowerBand = rollingMean - rollingStandardDeviation * 2
		return upperBand, lowerBand
	except Exception as BollingerBandException:
		print "Error occurred when attempting to calculate the Bollinger bands"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print BollingerBandException


def getRollingStandardDeviation(dataFrame, window):
	try:
		rollingStandardDeviation = pd.rolling_std(dataFrame, window=window)
		return rollingStandardDeviation
	except Exception as RollingStandardDeviationException:
		print "Error occurred when attempting to calculate the rolling standard deviation"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print RollingStandardDeviationException
	return dataFrame


def Manual_Strategy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2010, 6, 1),
                    sv=100000, getSPY=False, dropNA=False, tradeAmount=1000):
	lookback = 14
	dates = pd.date_range(sd, ed)
	if not isinstance(symbol, list):
		symbol = [symbol]
	columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
	tempDataFrame = pd.DataFrame()
	tempSPYDataFrame = pd.DataFrame()
	for i in columns:
		dataFrame = get_data(symbol, dates, colname=i)
		tempDataFrame[i] = dataFrame[symbol[0]]
		if getSPY:
			tempSPYDataFrame[i] = dataFrame['SPY']
	PricesDataFrame = tempDataFrame.copy()
	if getSPY:
		SpyDataFrame = tempSPYDataFrame.copy()
	SimpleMovingAverage = getRollingMean(PricesDataFrame['Adj Close'], window=lookback)
	RollingStandardDeviation = getRollingStandardDeviation(PricesDataFrame['Adj Close'], window=lookback)
	UpperBand, LowerBand = getBollingerBands(SimpleMovingAverage, RollingStandardDeviation)
	BollingerBandPercentage = BollingerBand_Percentage(PricesDataFrame, UpperBand, LowerBand)
	SimpleMovingAverageRatio = PriceSMA_Ratio(PricesDataFrame, SimpleMovingAverage)
	RelativeStrengthIndex = Relative_Strength_Index(PricesDataFrame, symbols=symbol, lookback=14)
	PricesDataFrame["Bollinger Band Percentage"] = BollingerBandPercentage
	PricesDataFrame["Simple Moving Average Ratio"] = SimpleMovingAverageRatio
	orders = PricesDataFrame["Adj Close"].copy()
	orders.ix[:] = np.NaN
	orders[(SimpleMovingAverageRatio < 0.98) & (BollingerBandPercentage < 0.3) & (
				RelativeStrengthIndex < 40)] = tradeAmount
	orders[(SimpleMovingAverageRatio > 1.02) & (BollingerBandPercentage > 0.7) & (
				RelativeStrengthIndex > 60)] = -tradeAmount
	if dropNA:
		orders.dropna(inplace=True)
	PricesDataFrame["Orders"] = orders
	PricesDataFrame.fillna(0, inplace=True)
	PricesDataFrame["Normalized Adj Close"] = PricesDataFrame["Adj Close"] / PricesDataFrame["Adj Close"][0]
	return orders, PricesDataFrame, symbol


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005,
                     manual=False, generatedOrderDataFrame=None, pricesDF=None, get_Prices=False, symbols=['JPM']):
	# this is the function the autograder will call to test your code
	# NOTE: orders_file may be a string, or it may be a file object. Your
	# code should work correctly with either input
	# TODO: Your code here
	if pricesDF is not None:
		pricesDataFrame = pricesDF.copy()
		manual = 'PASS'
	# Orders Data Frame
	if (manual != True) & (manual != "PASS"):
		ordersDataFrame, symbols, start_date, end_date = get_Orders_DataFrame(orders_file)
		if isinstance(symbols, list):
			pricesDataFrame, spyDataFrame = get_Price_DataFrame(symbols, start_date, end_date)
			
		else:
			pricesDataFrame, spyDataFrame = get_Price_DataFrame(list(symbols), start_date, end_date)
		pricesDataFrame = pricesDataFrame[symbols]  # remove SPY
		pricesDataFrame.dropna(axis='rows', how='all', inplace=True)
		pricesDataFrame["Cash"] = 1.0
		
	# At this point our orderDataFrame will have index of dates
	if (manual == True) & (manual != "PASS"):
		ordersDataFrame, symbols, start_date, end_date = get_Orders_DataFrame(manual=True,
		                                                                      orderDF=generatedOrderDataFrame)
		if isinstance(symbols, list):
			pricesDataFrame, spyDataFrame = get_Price_DataFrame(symbols, start_date, end_date)
		else:
			pricesDataFrame, spyDataFrame = get_Price_DataFrame(list(symbols), start_date, end_date)
		pricesDataFrame = pricesDataFrame[symbols]  # remove SPY
		pricesDataFrame.dropna(axis='rows', how='all', inplace=True)
		pricesDataFrame["Cash"] = 1.0
	# Price Data Frame
	pricesDataFrame_Shape = pricesDataFrame.shape
	# Create new ordersDataFrame consisting of only days which SPY was traded.
	if (manual == False) & (manual != "PASS"):
		ordersDataFrame = ordersDataFrame[
			ordersDataFrame.index.isin(spyDataFrame.index)]  # Remove trades that occur on non tradeable days
	else:
		if manual != "PASS":
			ordersDataFrame = ordersDataFrame[ordersDataFrame['Date'].isin(spyDataFrame.index)]
	if generatedOrderDataFrame is not None:
		ordersDataFrame = generatedOrderDataFrame.copy()

	# Trades Data Frame
	tradesDataFrame = pricesDataFrame.copy()
	tradesDataFrame[:] = 0.0
	tradesDataFrame = populate_TradeDataFrame(tradesDataFrame, ordersDataFrame,
	                                           pricesDataFrame, symbols,
	                                           commission, impact, start_val, manual=manual)
	# Holdings Data Frame
	holdingsDataFrame = get_Holdings_DataFrame(tradesDataFrame, start_val)
	
	# Values Data Frame
	valuesDataFrame = holdingsDataFrame * pricesDataFrame
	portvals = valuesDataFrame.sum(axis=1)
	
	rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())
	return rv


def helper_Manual_Strategy(orders_file="./orders/orders.csv", temp_start_val=1000000, temp_commission=9.95,
                           temp_impact=0.005):
	temp_ordersDataFrame, temp_symbols, temp_start_date, temp_end_date = get_Orders_DataFrame(orders_file)
	# At this point our orderDataFrame will have index of dates
	temp_symbols = ['JPM']
	# Price Data Frame
	temp_pricesDataFrame, temp_spyDataFrame = get_Price_DataFrame(temp_symbols, temp_start_date, temp_end_date,
	                                                              getSPY=True)
	# Create new ordersDataFrame consisting of only days which SPY was traded.
	temp_ordersDataFrame = temp_ordersDataFrame[
		temp_ordersDataFrame.index.isin(temp_spyDataFrame.index)]  # Remove trades that occur on non tradeable days
	temp_pricesDataFrame = temp_pricesDataFrame[temp_symbols]  # remove SPY
	temp_pricesDataFrame.dropna(axis='rows', how='all', inplace=True)
	temp_pricesDataFrame["Cash"] = 1.0
	
	# Trades Data Frame
	temp_tradesDataFrame = temp_pricesDataFrame.copy()
	temp_tradesDataFrame[:] = 0.0
	temp_tradesDataFrame = populate_TradeDataFrame(temp_tradesDataFrame, temp_ordersDataFrame,
	                                               temp_pricesDataFrame, temp_symbols,
	                                               temp_commission, temp_impact, temp_start_val)
	# Holdings Data Frame
	temp_holdingsDataFrame = get_Holdings_DataFrame(temp_tradesDataFrame, temp_start_val)
	
	# Values Data Frame
	temp_valuesDataFrame = temp_holdingsDataFrame * temp_pricesDataFrame
	temp_portvals = temp_valuesDataFrame.sum(axis=1)
	
	test_Daily_Returns = get_avg_daily_return(temp_portvals, 'JPM')
	test_Cumulative_Returns = get_cumulative_returns(temp_portvals, 'JPM')
	return


def get_Holdings_DataFrame(tradesDataFrame, start_val):
	holdingsDataFrame = tradesDataFrame.copy()
	holdingsDataFrame = holdingsDataFrame.cumsum(
		axis=0)  # This will add the previous values in Cash to the current value
	return holdingsDataFrame


def get_Price_DataFrame(symbols, start_date, end_date, getSPY=True):
	if not isinstance(symbols, list):
		symbols = list(symbols)
	pricesDataFrame = get_data(symbols, pd.date_range(start_date, end_date), addSPY=getSPY)
	# pricesDataFrame = pricesDataFrame.resample("D").fillna(method='ffill')
	# pricesDataFrame = pricesDataFrame.resample("D").fillna(method='bfill')
	
	if getSPY:
		spyDataFrame = pricesDataFrame['SPY']
		return pricesDataFrame, spyDataFrame
	else:
		return pricesDataFrame[symbols]


def get_Orders_DataFrame(orders_file=None, manual=False, orderDF=None):
	if not manual:
		path = os.getcwd()
		orders_file = orders_file.replace("./", "")
		fullPath = os.path.join(path, orders_file)
		ordersDataFrame = pd.read_csv(fullPath, delimiter=",", header=0, parse_dates=['Date']).sort_values(by='Date')
		symbols = ordersDataFrame["Symbol"].unique()
		symbols = list(symbols)
		if '$SPX' not in symbols:
			symbols += ['$SPX']
		if 'JPM' not in symbols:
			symbols += ['JPM']
		start_date = ordersDataFrame.iloc[0, 0]
		end_date = ordersDataFrame.iloc[-1, 0]
		ordersDataFrame.set_index('Date', inplace=True)
		return ordersDataFrame, symbols, start_date, end_date
	if manual:
		symbols = orderDF['Symbol'].unique()
		symbols = list(symbols)
		start_date = orderDF.iloc[0, 0]
		end_date = orderDF.iloc[-1, 0]
		return orderDF, symbols, start_date, end_date


def populate_TradeDataFrame(tradeDF, orderDF, pricesDf, symbols, commission, impact, start_val, manual=False):
	tradeDF['Cash'] = 0.0
	tradeDF['Cash'][0] = start_val
	tradeDF[symbols] = 0.0
	if not isinstance(symbols, list):
		symbols = [symbols]
	symbol_dict = {}
	if 'Date' in orderDF.columns:
		orderDF.set_index("Date", inplace=True)
	for i in symbols:
		symbol_dict[i] = 0
	for index, row in orderDF.iterrows():
		val = 0.0  # Reset Val to 0 after each iteration as to not carry over values
		# Date | Symbol | Order | Shares
		pricesForThatDay = pricesDf.loc[index]
		if isinstance(row.Order, str):
			if (row.Order.lower() == "buy") or (row.Order == 1):
				val = row.Shares + symbol_dict[
					row.Symbol]  # Val is used to keep track of total for the stocks and update dictionary
				tradeDF.loc[index][row.Symbol] = row.Shares + tradeDF.loc[index][row.Symbol]
				symbol_dict[row.Symbol] = val
				if manual:
					impact_price = pricesForThatDay["Adj Close"] * (1 + impact)
				else:
					impact_price = pricesForThatDay[row.Symbol] * (1 + impact)
				price_shares = ((impact_price * row.Shares) * -1)
				cash_val = (tradeDF.loc[index]['Cash'])
				cashColumn = price_shares + cash_val - commission
				tradeDF.loc[index]['Cash'] = cashColumn
				
			elif (row.Order.lower() == 'sell') or (row.Order == -1):
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
				if manual:
					impact_price = pricesForThatDay["Adj Close"] * (1 + impact)
				else:
					impact_price = pricesForThatDay[row.Symbol] * (1 + impact)
				price_shares = (impact_price * row.Shares)
				cash_val = (tradeDF.loc[index]['Cash'])
				cashColumn = price_shares + cash_val - commission
				tradeDF.loc[index]['Cash'] = cashColumn
		
			else:
				continue
		else:
			temp = row.Order
			if row.Order == 1:
				val = row.Shares + symbol_dict[
					row.Symbol]  # Val is used to keep track of total for the stocks and update dictionary
				tradeDF.loc[index][row.Symbol] = row.Shares + tradeDF.loc[index][row.Symbol]
				symbol_dict[row.Symbol] = val
				if manual:
					impact_price = pricesForThatDay["Adj Close"] * (1 + impact)
				else:
					impact_price = pricesForThatDay[row.Symbol] * (1 + impact)
				price_shares = ((impact_price * row.Shares) * -1)
				cash_val = (tradeDF.loc[index]['Cash'])
				cashColumn = price_shares + cash_val - commission
				tradeDF.loc[index]['Cash'] = cashColumn
				
			elif row.Order == -1:
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
				if manual:
					impact_price = pricesForThatDay["Adj Close"] * (1 + impact)
				else:
					impact_price = pricesForThatDay[row.Symbol] * (1 + impact)
				price_shares = (impact_price * row.Shares)
				cash_val = (tradeDF.loc[index]['Cash'])
				cashColumn = price_shares + cash_val - commission
				tradeDF.loc[index]['Cash'] = cashColumn
				
			else:
				continue
			# Do nothing
	return tradeDF


def portValQuick(priceDF, commission=9.95, impact=0.005, startVal=100000, Theo=False, Bench=False):
	results = priceDF.copy()
	results['Port Value'] = 0
	results['Port Value'][0] = startVal
	results["Impact Price"] = (1 + impact) * results['Adj Close']
	results["Commission"] = commission
	tempBuy = results[results['Orders'] > 0]
	tempBuy["Value"] = (-(tempBuy["Impact Price"]) * tempBuy["Orders"]) + tempBuy['Port Value'] - tempBuy["Commission"]
	tempSell = results[results['Orders'] < 0]
	tempSell["Value"] = (-(tempSell["Impact Price"]) * tempSell["Orders"]) + tempSell['Port Value'] - tempSell["Commission"]
	results['Port Value'] = 0
	for index, row in tempBuy.iterrows():
		portVal = row['Value']
		temp = results.loc[index]
		temp['Port Value'] = row['Value']
		results.loc[index] = temp
	for index, row in tempSell.iterrows():
		portVal = row['Value']
		temp = results.loc[index]
		temp['Port Value'] = row['Value']
		results.loc[index] = temp
	results['Cumulative Port Value'] = results['Port Value'].cumsum()
	if Theo:
		if Bench:
			benchDF = priceDF.copy()
			benchDF["Port Value"] = 0.0
			benchDF["Port Value"] = (benchDF["Adj Close"] * benchDF["Daily Returns"])*1000
			temp = benchDF["Port Value"]
			temp[0] = startVal + (benchDF["Positions"][0] * benchDF["Adj Close"][0])
			benchDF["Port Value"] = temp
			benchDF["Port Value"] = benchDF["Port Value"].cumsum()
			return benchDF['Port Value'].cumsum(), benchDF
		else:
			theoResultDF = priceDF.copy()
			temp = (theoResultDF["Positions"] * theoResultDF["Adj Close"] * theoResultDF["Daily Returns"]) + (theoResultDF["Positions"] * theoResultDF["Adj Close"])
			temp[0] += startVal
			theoResultDF["Port Value"] = temp.cumsum()
			theoResultDF["Port Value"].plot()
			return theoResultDF['Port Value'].cumsum(), theoResultDF
	else:
		return results['Port Value'].cumsum(), results


def test_code():
	# this is a helper function you can use to test your code
	# note that during autograding his function will not be called.
	# Define input parameters
	
	of = "./orders/orders-01.csv"
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
	return


def Prep_OrderDF(orderDf, symbol):
	resultDF = orderDf.copy()
	resultDF = resultDF.to_frame(name="Shares")
	resultDF['Symbol'] = symbol[0]
	resultDF['Date'] = resultDF.index
	resultDF['Order'] = 0
	resultDF['Order'][resultDF['Shares'] > 0] = 'Buy'
	resultDF['Order'][resultDF['Shares'] < 0] = 'Sell'
	return resultDF


def testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2010, 6, 1), sv=100000):
	# get_port_val(symbol=symbol, startDate=sd, endDate=ed)
	OrdersDataFrame, PricesDataFrame, Symbol = Manual_Strategy(symbol=symbol, sd=sd, ed=ed, sv=sv,
	                                                                          getSPY=True, dropNA=True,
	                                                                          tradeAmount=1000)
	OrdersDF_Prep = Prep_OrderDF(OrdersDataFrame, symbol=Symbol)
	# results_1000 = compute_portvals(generatedOrderDataFrame=OrdersDF_Prep, start_val=sv, manual=True,
	#                                 get_Prices=False, pricesDF=PricesDataFrame, symbols=symbol)
	# Get Normalized Prices

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
		if (row["Simple Moving Average Ratio"] < sma_buy_thresh) and (row["Bollinger Band Percentage"] < bollinger_buy_thresh) and (row["Relative Strength Index"] < rsi_buy_thresh):
			# Buy
			if (holdings <= 0) and (holdings >= -1):
				# This mean we cannot buy we can only do Nothing or Sell
				# Check indicators
				PricesDataFrame['Orders'].loc[day] = 1000
				holdings += 1
		elif (row["Simple Moving Average Ratio"] > sma_sell_thresh) and (row["Bollinger Band Percentage"] > bollinger_sell_thresh) and (row["Relative Strength Index"] > rsi_sell_thresh):
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
	port_val, resultDF = portValQuick(PricesDataFrame)
	resultDF["Cumulative Port Value"].plot()
	return


if __name__ == "__main__":
	test_code()
	testPolicy(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))
# portvals = compute_portvals(orders_file="orders-short.csv", start_val=100000)
