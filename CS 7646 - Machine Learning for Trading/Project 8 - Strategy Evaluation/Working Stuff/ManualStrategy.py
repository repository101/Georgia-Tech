"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""

import os
import sys
import time
import random

import numpy as np
import pandas as pd
import datetime as dt
import util as ut
import matplotlib.pyplot as plt
import matplotlib as mpl

import marketsimcode as mkt
import indicators as ind

np.random.seed(42)
plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

def author():
	"""
	:return: The GT username of the student
	:rtype: str
	"""
	return "jadams334"  # Change this to your user ID


def compute_benchmark(sd, ed, sv, symbol, market_impact, commission_cost, max_holdings):
	try:
		# From provided file grade_strategy_learner.py
		date_idx = ut.get_data([symbol, ], pd.date_range(sd, ed)).index
		columns = ["Date", "Symbol", "Order", "Shares"]
		orders = pd.DataFrame(columns=columns, dtype=object)
		orders["Date"] = date_idx
		orders[["Symbol", "Order", "Shares"]] = symbol, "", 0
		orders.loc[0, ["Order", "Shares"]] = "Buy", 1000
		orders.loc[orders.index[-1], ["Order", "Shares"]] = "Sell", -max_holdings
		baseline_portvals = compute_portvals(orders, sd, ed, sv, market_impact, commission_cost)
		return baseline_portvals
	except Exception as compute_benchmark_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'compute_benchmark'", compute_benchmark_exception)


def compute_portvals(orders_df, start_date, end_date, startval, market_impact=0.0, commission_cost=0.0, ):
	try:
		# From provided file grade_strategy_learner.py
		"""Simulate the market for the given date range and orders file."""
		symbols = []
		orders = []
		orders_df = orders_df.sort_index()
		for _, order in orders_df.iterrows():
			shares = order["Shares"]
			action = order["Order"]
			symbol = order["Symbol"]
			date = order["Date"]
			if action.lower() == "sell":
				shares *= -1
			order = (date, symbol, shares)
			orders.append(order)
			symbols.append(symbol)
		symbols = list(set(symbols))
		dates = pd.date_range(start_date, end_date)
		prices_all = ut.get_data(symbols, dates)
		prices = prices_all[symbols]
		prices = prices.fillna(method="ffill").fillna(method="bfill")
		prices["_CASH"] = 1.0
		trades = pd.DataFrame(index=prices.index, columns=symbols)
		trades = trades.fillna(0)
		cash = pd.Series(index=prices.index)
		cash = cash.fillna(0)
		cash.iloc[0] = startval
		for date, symbol, shares in orders:
			price = prices[symbol][date]
			val = shares * price
			# transaction cost model
			val += commission_cost + (pd.np.abs(shares) * price * market_impact)
			positions = prices.loc[date] * trades.sum()
			totalcash = cash.sum()
			if (date < prices.index.min()) or (date > prices.index.max()):
				continue
			trades[symbol][date] += shares
			cash[date] -= val
		trades["_CASH"] = cash
		holdings = trades.cumsum()
		df_portvals = (prices * holdings).sum(axis=1)
		return df_portvals
	except Exception as compute_portvals_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'compute_portvals'", compute_portvals_exception)


def determine_y(data_df):
	try:
		data_df["Next_Day_Adjusted_Close"] = data_df["Adj Close"].shift(-1)
		data_df["Diff"] = data_df["Next_Day_Adjusted_Close"] - data_df["Adj Close"]
		data_df["Position"] = 0
		data_df.loc[data_df["Diff"] > 0, "Order"] = "BUY"
		data_df.loc[data_df["Diff"] < 0, "Order"] = "SELL"

		another_temp = data_df["RSI"].pct_change(periods=2)
		new_df = data_df.copy()
		plt.close("all")
		fig, ax1 = plt.subplots(figsize=(12, 6))
		ax1.plot((new_df["Close"] / new_df["Close"][0]), label="Close", linewidth=2)
		ax1.plot(new_df["BBP"], label="BBP", linewidth=1, linestyle="--")
		# ax1.plot(new_df["BBP"], label="BBP", linewidth=2, linestyle="--")
		# ax1.plot(new_df["OBV"], label="OBV", linewidth=2, linestyle="--")
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)
		plt.tight_layout()
		plt.savefig("Test_img.png")

		overbought = False
		oversold = False

		"""
		If BBP < 0 and RSI > 30:
		     BUY
	    else if BBP > 1 and RSI > 70:
	        SELL
	    else if 
		
		"""
		if data_df["RSI"] > 70 or data_df["BBP"] > 1:
			overbought = True

	except Exception as determine_y_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'determine_y'", determine_y_exception)


def populate_indicators(all_data_df, window=5, vortex_window=14, only_SPY=False, **kwargs):
	try:
		if only_SPY:
			return mkt.get_relative_strength_indicator(only_SPY=True, **kwargs)
		rsi_df = mkt.get_relative_strength_indicator(**kwargs)
		for i in rsi_df.columns:
			if i == "Price" or i == "Normalized_Price":
				continue
			all_data_df[i] = rsi_df[i]
		return \
			ind.calculate_obv(
				ind.calculate_vortex(
					ind.calculate_macd(
						ind.calculate_bollinger(all_data_df, window=window)), window=vortex_window))
	except Exception as populate_indicators_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'populate_indicators'", populate_indicators_exception)


def min_max_scale(df):
	try:
		return (df - df.mean()) / df.std()
	except Exception as min_max_scale_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'min_max_scale'", min_max_scale_exception)


def populate_training_data(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
                           market_impact=0.005, commission_cost=9.95, max_holding=1000, vortex_window=14,
                           window=10, n_days=10, min_max=False):
	try:
		kwargs = {"symbol": symbol, "sd": sd, "ed": ed, "lookback": 14}
		all_data_df = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
		all_data_with_indicators = populate_indicators(all_data_df, window=window, vortex_window=vortex_window,
		                                               **kwargs)
		# all_SPY_Data = populate_indicators(None, only_SPY=True, **kwargs)
		all_data_with_indicators["BBP"] = all_data_with_indicators["Bollinger_Percent"]
		# Reduce DF to only the indicators used
		indicator_columns = ["Close",
		                     "Adj Close",
		                     "RSI",
		                     "BBP",
		                     "MACD_Line",
		                     "Signal_Line",
		                     f"VI_{vortex_window}_Positive",
		                     f"VI_{vortex_window}_Negative",
		                     "OBV"]
		# new_X = min_max_scale(all_data_with_indicators[indicator_columns])
		temp_y = determine_y(all_data_with_indicators[indicator_columns])
		if min_max:
			data = {"X": min_max_scale(all_data_with_indicators[indicator_columns]),
			        "y": determine_y(min_max_scale(all_data_with_indicators[indicator_columns]))}
		else:
			data = {"X": all_data_with_indicators[indicator_columns],
			        "y": determine_y(all_data_with_indicators[indicator_columns])}
		return data
	except Exception as populate_training_data_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'populate_training_data'", populate_training_data_exception)


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
               market_impact=0.005, commission_cost=9.95, max_holding=1000, vortex_window=14, window=10):
	try:
		# baseline_portvals = compute_benchmark(sd=sd, ed=ed, sv=sv, symbol=symbol, market_impact=market_impact,
		#                                       commission_cost=commission_cost, max_holdings=max_holding)
		results = populate_training_data(symbol=symbol, sd=sd, ed=ed, sv=sv, market_impact=market_impact,
		                                 commission_cost=commission_cost, max_holding=max_holding,
		                                 vortex_window=vortex_window, window=window)
		return
	except Exception as testPolicy_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'testPolicy'", testPolicy_exception)
