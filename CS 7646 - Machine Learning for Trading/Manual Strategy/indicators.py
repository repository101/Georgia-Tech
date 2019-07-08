"""Manual Strategy.Indicators

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
import matplotlib.pyplot as plt
import os
import sys
import marketsimcode as msc
from util import get_data, plot_data
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.style.use("ggplot")


def author():
	return 'jadams334'  # replace tb34 with your Georgia Tech username.


def getRollingMean(dataFrame, window):
	try:
		rollingMean = pd.rolling_mean(dataFrame, window=window)
		return rollingMean
	except Exception as RollingMeanException:
		print "Error occurred when attempting to calculate the rolling mean"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print RollingMeanException
	return dataFrame


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


def fillMissingValues(dataFrame):
	try:
		dataFrame.fillna(method='ffill', inplace=True)
	except Exception as FillForwardException:
		print "Error occurred when attempting to fill forward"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print FillForwardException
	try:
		dataFrame.fillna(method='bfill', inplace=True)
	except Exception as FillBackwardException:
		print "Error occurred when attempting to fill backward"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print FillBackwardException
	return dataFrame


def BollingerBands(symbols=None, startDate=None, endDate=None, chart=True, BollingerBandDataFrame=None):
	plt.clf()
	try:
		BollingerBandAX = plt.subplot()
		if startDate is None:
			startDate = dt.datetime(2008, 1, 1)
		if endDate is None:
			endDate = dt.datetime(2008, 12, 31)
		if symbols is None:
			symbols = ["JPM"]
		if BollingerBandDataFrame is None:
			dates = pd.date_range(startDate, endDate)
			BollingerBandDataFrame = get_data(symbols, dates)
			BollingerBandDataFrame = fillMissingValues(BollingerBandDataFrame)
		
		window = 20
		# 1. Computer Rolling Mean
		rollingMean_JPM = getRollingMean(BollingerBandDataFrame['Adj Close'], window=window)
		# 2. Compute Rolling Standard Deviation
		rollingStandardDeviation_JPM = getRollingStandardDeviation(BollingerBandDataFrame['Adj Close'], window=window)
		# 3. Compute Upper and Lower Bands
		upperBand_JPM, lowerBand_JPM = getBollingerBands(rollingMean_JPM, rollingStandardDeviation_JPM)
		BollingerBandDataFrame['Rolling Mean Adj Close'] = rollingMean_JPM
		BollingerBandDataFrame['Lower Bollinger Band'] = lowerBand_JPM
		BollingerBandDataFrame['Upper Bollinger Band'] = upperBand_JPM
		fillMissingValues(BollingerBandDataFrame)
		if chart:
			BollingerBandDataFrame['Adj Close'].plot(ax=BollingerBandAX, title="{} Bollinger Bands".format(symbols[0]),
			                                         label='Adj Close', color='forestgreen', style='.')
			BollingerBandDataFrame['Rolling Mean Adj Close'].plot(label="Rolling Mean", ax=BollingerBandAX, color='tomato', style='--')
			BollingerBandDataFrame['Upper Bollinger Band'].plot(label="Upper Band", ax=BollingerBandAX, color='dodgerblue')
			BollingerBandDataFrame['Lower Bollinger Band'].plot(label='Lower Band', ax=BollingerBandAX, color='dodgerblue')
			BollingerBandAX.fill_between(BollingerBandDataFrame.index, BollingerBandDataFrame['Upper Bollinger Band'], BollingerBandDataFrame['Lower Bollinger Band'], color='b', alpha=0.2)
			SavePlot(ax=BollingerBandAX, fileName="{} Bollinger Bands.png".format(symbols[0]),
			         symbols=symbols)
		return
	except Exception as BollingerBandException:
		print "Error occurred during the execution of BollingerBand method"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print BollingerBandException
	return
	

def SavePlot(ax=None, fileName=None, symbols=None):
	if symbols is not None:
		ax.set_xlabel('Date')
		ax.set_ylabel('Price')
		ax.legend(loc='best')
		ax.legend(fancybox=True, shadow=True)
		plt.tight_layout()
	try:
		path = os.path.dirname(__file__)
		if fileName is None:
			fileName = "{} Rolling Mean.png".format(symbols[0])
		plt.savefig(os.path.join(path, fileName), dpi=300)
	except Exception as e:
		print "An error occurred when saving the graph "
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print e

	return


def Momentum(momentum_symbols=None, momentum_StartDate=None, momentum_EndDate=None, momentum_Dates=None,
             chart=True, momentum_DataFrame=None):
	plt.clf()
	if momentum_StartDate is None:
		momentum_StartDate = dt.datetime(2008, 1, 1)
	if momentum_EndDate is None:
		momentum_EndDate = dt.datetime(2008, 12, 31)
	if momentum_symbols is None:
		momentum_symbols = ["JPM"]
	if momentum_Dates is None:
		momentum_Dates = pd.date_range(momentum_StartDate, momentum_EndDate)
	if momentum_DataFrame is None:
		momentum_DataFrame = get_data(momentum_symbols, momentum_Dates)
		momentum_DataFrame = fillMissingValues(momentum_DataFrame)
	try:
		MomentumAX = plt.subplot()
		# Normalized the data
		momentum_DataFrame['Adj Close Normalized'] = momentum_DataFrame['Adj Close']/momentum_DataFrame['Adj Close'].iloc[0]
		# momentum_DataFrame = momentum_DataFrame[symbols]
		FiveDayShift = momentum_DataFrame['Adj Close Normalized'].shift(-5)
		TwentyDayShift = momentum_DataFrame['Adj Close Normalized'].shift(-10)
		fillMissingValues(momentum_DataFrame)
		momentum_DataFrame["5DayMomentum"] = (momentum_DataFrame['Adj Close Normalized'] / FiveDayShift) - 0.2
		momentum_DataFrame["20DayMomentum"] = (momentum_DataFrame['Adj Close Normalized'] / TwentyDayShift) - 0.25
		momentum_DataFrame['Rolling Mean'] = getRollingMean(momentum_DataFrame['Adj Close Normalized'], window=20)
		momentum_DataFrame['12 EMA'] = pd.ewma(momentum_DataFrame['Adj Close'], span=12)
		# momentum_DataFrame['12 EMA'].plot(label='12 Day EMA', ax=MomentumAX, color='olivedrab')
		momentum_DataFrame['26 EMA'] = pd.ewma(momentum_DataFrame['Adj Close'], span=26)
		# momentum_DataFrame['26 EMA'].plot(label='26 Day EMA', ax=MomentumAX, color='deepskyblue')
		momentum_DataFrame['MACD'] = momentum_DataFrame['12 EMA'] - momentum_DataFrame['26 EMA']
		momentum_DataFrame['MACD CrossOver'] = pd.ewma(momentum_DataFrame['Adj Close'], span=9)
		momentum_DataFrame['MACD Signal'] = momentum_DataFrame['MACD'] - momentum_DataFrame['MACD CrossOver']
		# momentum_DataFrame['MACD'].plot(label="MACD", ax=MomentumAX)
		# momentum_DataFrame['MACD CrossOver'].plot(label="MACD Signal", ax=MomentumAX)
		# momentum_DataFrame['CrossOver'].plot(label="MACD CrossOver", ax=MomentumAX)
		momentum_DataFrame['Adj Close Normalized'].plot(ax=MomentumAX, title="{} Momentum".format(momentum_symbols[0]),
		                                                  label='Adj Close Normalized', color='forestgreen',
		                                                  style='-')
		momentum_DataFrame['Rolling Mean'].plot(label='Rolling Mean', ax=MomentumAX, color='tomato', style='--')
		momentum_DataFrame['5DayMomentum'].plot(label="5 Day Momentum", ax=MomentumAX, color='gold')
		momentum_DataFrame['20DayMomentum'].plot(label="20 Day Momentum", ax=MomentumAX, color='royalblue')
		SavePlot(ax=MomentumAX, fileName="{} Momentum.png".format(momentum_symbols[0]), symbols=momentum_symbols)
	except Exception as MomentumException:
		print "Error occurred when calculating 5 or 20 day momentum"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print MomentumException
	return


def OnBalanceVolume(onBalanceDataFrame, onBalance_symbols):
	try:
		plt.clf()
		
		OnBalanceAX = plt.subplot()
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)
		# if todays closing price is higher than previous day closing, OBV = previous OBV + todays volume
		# if todays closing price is lower than previous day closing, OBV = previous OBV - todays volume
		onBalanceDataFrame["Price Diff"] = onBalanceDataFrame['Adj Close'] - onBalanceDataFrame['Adj Close'].shift(1)
		onBalanceDataFrame["Volume Shift"] = onBalanceDataFrame['Volume'].shift(1)
		onBalanceDataFrame.fillna(0, inplace=True)
		onBalanceDataFrame['OBV'] = onBalanceDataFrame.apply(lambda x: (x['Volume'] + x['Volume Shift'])
																if x['Price Diff'] >= 0 else
																(x['Volume'] - x['Volume Shift']), axis=1)
		onBalanceDataFrame['Adj Close Normalized'] = onBalanceDataFrame['Adj Close'] / onBalanceDataFrame['Adj Close'].iloc[0]
		onBalanceDataFrame['Adj Close'].plot(ax=ax1, title="EVM vs. Adjusted Close",
		                                  label='Adj Close', color='dodgerblue', style="-")
		onBalanceDataFrame['OBV Normalized'] = onBalanceDataFrame['OBV']/onBalanceDataFrame['OBV'].iloc[0]
		onBalanceDataFrame['OBV Normalized'] = getRollingMean(onBalanceDataFrame['OBV Normalized'], window=20)
		onBalanceDataFrame['OBV Normalized'].plot(ax=ax2, title="On-Balance Volume",
		                                          label='On-Balance Volume', color='grey')
		onBalanceDataFrame['12 EMA'] = pd.ewma(onBalanceDataFrame['Adj Close'], span=12)
		onBalanceDataFrame['12 EMA'].plot(label='12 Day EMA', ax=ax1, color='olivedrab')
		onBalanceDataFrame['26 EMA'] = pd.ewma(onBalanceDataFrame['Adj Close'], span=26)
		onBalanceDataFrame['26 EMA'].plot(label='26 Day EMA', ax=ax1, color='deepskyblue')
		onBalanceDataFrame['MACD'] = onBalanceDataFrame['12 EMA'] - onBalanceDataFrame['26 EMA']
		onBalanceDataFrame['MACD Signal'] = pd.ewma(onBalanceDataFrame['Adj Close Normalized'], span=9)
		onBalanceDataFrame['MACD CrossOver'] = abs(onBalanceDataFrame['MACD'] - onBalanceDataFrame['MACD Signal'])
		ax1.fill_between(onBalanceDataFrame.index, onBalanceDataFrame['12 EMA'], onBalanceDataFrame['26 EMA'],
		                 where=onBalanceDataFrame['12 EMA'] >= onBalanceDataFrame['26 EMA'], facecolor='g',
		                 alpha=0.75, interpolate=True)
		ax1.fill_between(onBalanceDataFrame.index, onBalanceDataFrame['12 EMA'], onBalanceDataFrame['26 EMA'],
		                 where=onBalanceDataFrame['12 EMA'] < onBalanceDataFrame['26 EMA'], facecolor='r',
		                 alpha=0.75, interpolate=True)
		ax2.legend(loc='best')
		ax2.legend(fancybox=True, shadow=True)
		SavePlot(ax=ax1, fileName="{} On-Balance Volume Combined.png".format(onBalance_symbols[0]),
		         symbols=onBalance_symbols)
		plt.clf()
		OnBalanceAX = plt.subplot()
		onBalanceDataFrame['Adj Close'].plot(ax=OnBalanceAX, title="EVM vs. Adjusted Close",
	                                     label='Adj Close', color='dodgerblue', style="-")
		onBalanceDataFrame['12 EMA'].plot(label='12 Day EMA', ax=OnBalanceAX, color='olivedrab')
		onBalanceDataFrame['26 EMA'].plot(label='26 Day EMA', ax=OnBalanceAX, color='deepskyblue')
		OnBalanceAX.set_xlabel('Date')
		OnBalanceAX.set_ylabel('Price')
		try:
			RSI = msc.Relative_Strength_Index(PriceDataFrame=onBalanceDataFrame, symbols='JPM')
			onBalanceDataFrame["RSI"] = RSI
			onBalanceDataFrame["RSI"].plot(label="Relative Strength Index", ax=OnBalanceAX, color="firebrick")
			# onBalanceDataFrame['MACD'].plot(label="MACD", ax=OnBalanceAX)
			# onBalanceDataFrame['MACD CrossOver'].plot(label="MACD Signal", ax=OnBalanceAX)
			# onBalanceDataFrame['MACD Signal'].plot(label="MACD CrossOver", ax=OnBalanceAX)
			
		except Exception as OnBalanceVolumeException:
			print "Error occurred when attempting to call OnBalanceVolume"
			exc_type, exc_obj, exc_tb = sys.exc_info()
			print exc_obj
			print exc_tb.tb_lineno
			print OnBalanceVolumeException
		OnBalanceAX.fill_between(onBalanceDataFrame.index, onBalanceDataFrame['12 EMA'], onBalanceDataFrame['26 EMA'],
		                 where=onBalanceDataFrame['12 EMA'] >= onBalanceDataFrame['26 EMA'], facecolor='g',
		                 alpha=0.75, interpolate=True)
		OnBalanceAX.fill_between(onBalanceDataFrame.index, onBalanceDataFrame['12 EMA'], onBalanceDataFrame['26 EMA'],
		                 where=onBalanceDataFrame['12 EMA'] < onBalanceDataFrame['26 EMA'], facecolor='r',
		                 alpha=0.75, interpolate=True)

		SavePlot(ax=OnBalanceAX, fileName="{} EVM vs Adjusted Close.png".format(onBalance_symbols[0]),
		         symbols=onBalance_symbols)
		plt.clf()
		EMA_Ax = plt.subplot()
		onBalanceDataFrame['MACD'].plot(title="JPM MACD", label="MACD", ax=EMA_Ax, color="dodgerblue")
		onBalanceDataFrame['MACD CrossOver'].plot(label="MACD Signal", ax=EMA_Ax, color="gold")
		# onBalanceDataFrame['Adj Close'].plot(label="Adjusted Close", ax=EMA_Ax, color="b")
		onBalanceDataFrame['MACD Signal'].plot(label="MACD CrossOver", ax=EMA_Ax, color="b")
		EMA_Ax.fill_between(onBalanceDataFrame.index, onBalanceDataFrame['MACD'], onBalanceDataFrame['MACD CrossOver'],
		                 where=onBalanceDataFrame['MACD'] >= onBalanceDataFrame['MACD CrossOver'], facecolor='darkgreen',
		                 alpha=0.75, interpolate=True)
		EMA_Ax.fill_between(onBalanceDataFrame.index, onBalanceDataFrame['MACD'], onBalanceDataFrame['MACD CrossOver'],
		                 where=onBalanceDataFrame['MACD'] < onBalanceDataFrame['MACD CrossOver'], facecolor='firebrick',
		                 alpha=0.75, interpolate=True)
		SavePlot(ax=EMA_Ax, fileName="{} MACD.png".format(onBalance_symbols[0]),
		         symbols=onBalance_symbols)
		
	except Exception as OnBalanceVolumeException:
		print "Error occurred when attempting to calculate the On-Balance Volume"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print OnBalanceVolumeException
	return


def Indicators(symbols=None, startDate=None, endDate=None):
	dates = pd.date_range(startDate, endDate)
	columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
	tempDataFrame = pd.DataFrame()
	for i in columns:
		dataFrame = get_data(symbols, dates, colname=i)
		tempDataFrame[i] = dataFrame[symbols[0]]
	dataFrame = tempDataFrame
	BollingerBandDataFrame = dataFrame.copy()
	MomentumDataFrame = dataFrame.copy()
	OnBalanceVolumeDataFrame = dataFrame.copy()
	RelativeStrengthIndexDataFrame = dataFrame.copy()
	try:
		fillMissingValues(dataFrame)
	except Exception as IndicatorException:
		print "Error when attempting to call fillMissingValues"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print IndicatorException
	try:
		BollingerBands(BollingerBandDataFrame=BollingerBandDataFrame, symbols=symbols)
	except Exception as BollingerBandsException:
		print "Error when attempting to call BollingerBands"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print BollingerBandsException
	try:
		Momentum(momentum_DataFrame=MomentumDataFrame, momentum_symbols=symbols)
	except Exception as MomentumException:
		print "Error when attempting to call Momentum"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print MomentumException
	try:
		OnBalanceVolume(OnBalanceVolumeDataFrame, onBalance_symbols=symbols)
	except Exception as OnBalanceVolumeException:
		print "Error occurred when attempting to call OnBalanceVolume"
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print exc_obj
		print exc_tb.tb_lineno
		print OnBalanceVolumeException
	
	return
	
	
if __name__ == "__main__":
	symbols = ["JPM"]
	start_date = dt.datetime(2008, 1, 1)
	end_date = dt.datetime(2009, 12, 31)
	Indicators(symbols=symbols, startDate=start_date, endDate=end_date)
	
	# Should generate the charts that illustrate your indicators in the report
	# I can use two from this list [SMA, RSI, BolingerBands]

