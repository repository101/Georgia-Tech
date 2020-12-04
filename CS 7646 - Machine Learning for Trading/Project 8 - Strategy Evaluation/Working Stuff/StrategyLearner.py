""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
import datetime as dt
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import indicators as ind
import marketsimcode as mkt
import util as ut

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500
import BagLearner as bl
import RTLearner as rl

np.random.seed(42)


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                 ed=dt.datetime(2009, 12, 31), sv=100000, obv=4, rsi=15, bbp=15, macd=15, vortex=7,
                 n_day_return=15):
        """
        Constructor method
        """
        self.sv = sv
        self.ed = ed
        self.sd = sd
        self.symbol = symbol
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None
        self.max_holdings = 1000
        self.bags = 500
        self.obv = obv
        self.rsi = rsi
        self.bbp = bbp
        self.macd = macd
        self.vortex = vortex
        self.window = 10
        self.adj_close = None
        self.y_buy = 0.001
        self.y_sell = -0.001
        self.n_day_return = n_day_return
        self.indicator_columns = ["RSI",
                                  "BBP",
                                  "MACD",
                                  "Vortex",
                                  "OBV"]

    @staticmethod
    def author():
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jadams334"  # Change this to your user ID

    def populate_indicators(self, all_data_df, only_SPY=False, **kwargs):
        try:
            if only_SPY:
                return mkt.get_relative_strength_indicator(only_SPY=True, **kwargs)
            rsi_df = mkt.get_relative_strength_indicator(**kwargs)
            all_data_df["RSI"] = rsi_df["RSI"]
            all_data_df["SMA"] = rsi_df["SMA"]

            tg = ind.calculate_obv(
                    ind.calculate_vortex(
                        ind.calculate_macd(
                            ind.calculate_bollinger(all_data_df, window=self.bbp),
                            window=self.macd),
                        window=self.vortex),
                    window=self.obv)
            tg.dropna(inplace=True)
            return tg
        except Exception as populate_indicators_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'populate_indicators'", populate_indicators_exception)

    def min_max_scale(self, df):
        try:
            return (df - df.mean()) / df.std()
        except Exception as min_max_scale_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'min_max_scale'", min_max_scale_exception)

    def determine_y(self, data_df, n_day_return=1):
        try:
            data_df["Adj_Close_PCT"] = data_df["Adj Close"].pct_change(periods=n_day_return)
            temp_df = data_df.loc[data_df["RSI"] != 0, :].copy()
            result_corr = {}
            all_dataframe = []
            temp_df = data_df.copy()
            temp_df = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())
            temp_df[f"{n_day_return}_Day_Return"] = temp_df["Adj Close"].shift(-n_day_return)
            temp_df.loc[temp_df[f"{n_day_return}_Day_Return"] > self.y_buy, "Order"] = 1
            temp_df.loc[temp_df[f"{n_day_return}_Day_Return"] < self.y_sell, "Order"] = -1
            all_dataframe.append(temp_df)
            pt, ct = np.unique(temp_df["Order"], return_counts=True)
            if 0 in pt:
                idx = np.argwhere(pt == 0)[0]
                zero_ct = ct[idx]
                if zero_ct > np.floor(temp_df.shape[0] * 0.99):
                    temp_df.loc[temp_df.index.values[0], "Order"] = 1
            return temp_df["Order"]
        except Exception as determine_y_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'determine_y'", determine_y_exception)

    def populate_training_data(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), ybuy=0.1,
                               ysell=0.1, test_policy=False):
        try:
            kwargs = {"symbol": symbol, "sd": sd, "ed": ed, "lookback": self.rsi}
            all_data_df = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
            self.adj_close = all_data_df["Adj Close"]
            all_data_with_indicators = self.populate_indicators(all_data_df, **kwargs)

            # Reduce DF to only the indicators used
            if not test_policy:
                temp_y = self.determine_y(all_data_with_indicators, n_day_return=self.n_day_return)
                data = {"X": all_data_with_indicators.loc[temp_y.index, :],
                        "y": temp_y}
                return data
            elif test_policy:
                temp_y = self.learner.query(all_data_with_indicators[self.indicator_columns])
                temp_y = pd.DataFrame(columns=[symbol], data=temp_y.reshape(-1, 1))
                temp_y.set_index(all_data_with_indicators.index, drop=True, inplace=True)
                data = {"X": all_data_with_indicators.loc[temp_y.index, :],
                        "y": temp_y}
                return data
        except Exception as populate_training_data_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'populate_training_data'", populate_training_data_exception)

    def check_holdings(self, hold):
        try:
            if hold > 1000 or hold < -1000:
                return False
            else:
                return True
        except Exception as check_holdings_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'check_holdings'", check_holdings_exception)

    def determine_buy_sell_amount(self, holdings, buy_or_sell):
        try:
            if buy_or_sell == "BUY":
                if holdings == 0:
                    return 1000
                elif holdings == -1000:
                    return 2000
                else:
                    return 0
            elif buy_or_sell == "SELL":
                if holdings == 0:
                    return 1000
                elif holdings == 1000:
                    return 2000
                else:
                    return 0
            else:
                return 0
        except Exception as determine_buy_sell_amount_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'determine_buy_sell_amount'", determine_buy_sell_amount_exception)

    def find_optimal(self, stock_DF, symbol):
        try:
            columns = ["Date", "Symbol", "Order", "Shares"]
            temp_DF = pd.DataFrame(index=stock_DF.index)
            temp_DF["Date"] = stock_DF.index.values
            temp_DF["Symbol"] = symbol
            temp_DF["Order"] = ""
            temp_DF["Position"] = 0
            temp_DF.loc[stock_DF["Pred_y"] == 1, ["Order", "Position"]] = "BUY", 1000
            temp_DF.loc[stock_DF["Pred_y"] == -1, ["Order", "Position"]] = "SELL", -1000
            temp_DF = temp_DF.loc[temp_DF["Order"] != "", :]
            temp_DF["Shares"] = temp_DF["Position"].diff()
            temp_DF.dropna(inplace=True)
            temp_DF.loc[temp_DF.index.values[0], "Shares"] = temp_DF.loc[temp_DF.index.values[0], "Position"]
            return temp_DF.loc[(temp_DF["Shares"] != 0), columns]
        except Exception as find_optimal_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'find_optimal'", find_optimal_exception)

    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
                     ybuy=0.1, ysell=0.1):
        try:
            """
                    Trains your strategy learner over a given time frame.

                    :param symbol: The stock symbol to train on
                    :type symbol: str
                    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
                    :type sd: datetime
                    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
                    :type ed: datetime
                    :param sv: The starting value of the portfolio
                    :type sv: int
                    """
            # add your code to do learning here
            results = self.populate_training_data(symbol=symbol, sd=sd, ed=ed, ysell=ysell, ybuy=ybuy)
            kwargs = {"leaf_size": 25}
            self.learner = bl.BagLearner(learner=rl.RTLearner, use_mode=True, bags=self.bags, **kwargs)
            self.learner.add_evidence(data_x=results["X"][self.indicator_columns], data_y=results["y"])
            return
        except Exception as add_evidence_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'add_evidence'", add_evidence_exception)

    def convert_old_trade_to_new_trade_frame(self, trade_df):
        try:
            trade_df.reset_index(inplace=True, drop=True)
            result = pd.DataFrame()
            symbol = trade_df.loc[0, 'Symbol']
            result[f"{symbol}"] = trade_df["Shares"].copy()
            result.set_index(trade_df["Date"], inplace=True, drop=True)
            return result
        except Exception as convert_old_trade_to_new_trade_frame_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'convert_old_trade_to_new_trade_frame'",
                  convert_old_trade_to_new_trade_frame_exception)

    def convert_new_trade_back_to_old_trade_frame(self, trade_df):
        try:
            results = pd.DataFrame(columns=["Date", "Symbol", "Order", "Shares"],
                                   data=np.zeros(shape=(trade_df.shape[0], 4)), index=trade_df.index)
            symbol = trade_df.columns[0]
            results["Date"] = trade_df.index.copy()
            results["Order"] = ""
            results["Symbol"] = symbol
            results["Shares"] = np.abs(trade_df[symbol].copy())
            results.loc[trade_df[symbol] > 0, "Order"] = "BUY"
            results.loc[trade_df[symbol] < 0, "Order"] = "SELL"
            results.reset_index(inplace=True, drop=True)
            return results
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in ''", _exception)

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        # this method should use the existing policy and test it against new data
        try:
            """
                    Tests your learner using data outside of the training data

                    :param symbol: The stock symbol that you trained on on
                    :type symbol: str
                    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
                    :type sd: datetime
                    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
                    :type ed: datetime
                    :param sv: The starting value of the portfolio
                    :type sv: int
                    :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
                        a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
                        Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
                        long so long as net holdings are constrained to -1000, 0, and 1000.
                    :rtype: pandas.DataFrame
                    """
            # your code should return the same sort of data
            results = self.populate_training_data(symbol=symbol, sd=sd, ed=ed, test_policy=True)

            res = self.learner.query(results["X"][self.indicator_columns])
            temp_order = self.find_optimal(pd.DataFrame(index=results["X"].index, columns=["Pred_y"], data=res[0]),
                                           symbol=symbol)
            temp_order.dropna(inplace=True)
            new_trade_df = self.convert_old_trade_to_new_trade_frame(temp_order)
            return new_trade_df
        except Exception as testPolicy_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'testPolicy'", testPolicy_exception)

    def compute_benchmark(self, sd, ed, sv, symbol):
        try:
            # From provided file grade_strategy_learner.py
            data = ut.get_data([symbol, ], pd.date_range(sd, ed))
            data.sort_index(inplace=True)
            data.sort_index(inplace=True, )
            date_idx = data.index
            columns = ["Date", "Symbol", "Order", "Shares"]
            orders = pd.DataFrame(columns=columns, dtype=object)
            orders["Date"] = date_idx
            orders["Symbol"] = symbol
            orders["Order"] = ""
            orders["Shares"] = 0
            orders.loc[0, "Shares"] = 1000
            orders.loc[0, "Order"] = "Buy"
            orders.loc[orders.index[-1], "Shares"] = -self.max_holdings
            orders.loc[orders.index[-1], "Order"] = "Sell"
            baseline_portvals = self.compute_portvals(orders, sd, ed, sv, self.impact, self.commission)
            return baseline_portvals
        except Exception as compute_benchmark_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'compute_benchmark'", compute_benchmark_exception)

    def compute_portvals(self, orders_df, start_date, end_date, startval, market_impact=0.0,
                         commission_cost=0.0, ):
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

if __name__ == "__main__":
    print("One does not simply think up a strategy")
