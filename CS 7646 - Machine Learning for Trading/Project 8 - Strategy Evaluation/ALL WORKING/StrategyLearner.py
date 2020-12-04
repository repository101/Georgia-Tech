import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import BagLearner as bl
import RTLearner as rl
import DTLearner as dl
import indicators as ind
import marketsimcode as mkt
import util as ut

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

np.random.seed(42)

"""		  	   		     		  		  		    	 		 		   		 		  
Student Name: Josh Adams (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: jadams334 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 903475599 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jadams334"  # replace tb34 with your Georgia Tech username


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
    def __init__(self, verbose=False, impact=0.005, commission=0.0, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                 ed=dt.datetime(2009, 12, 31), sv=100000, obv=6, rsi=2, bbp=3, macd=10, vortex=7,
                 n_day_return=1, n_leaf=5):
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
        self.bags = 100
        self.n_leaf = n_leaf
        if self.n_leaf < 5:
            self.n_leaf = 5
        self.obv = obv
        self.rsi = rsi
        self.bbp = bbp
        self.macd = macd
        self.vortex = vortex
        self.window = 10
        self.adj_close = None
        self.current_holdings = 0
        self.y_buy = 0.0
        self.y_sell = 0.0
        self.n_day_return = n_day_return
        self.indicator_columns = ["RSI",
                                  "BBP",
                                  "MACD"]

    @staticmethod
    def author():
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jadams334"  # Change this to your user ID

    def populate_indicators(self, all_data_df, only_SPY=False, **kwargs):
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
        return tg[["Adj Close", "RSI", "BBP", "MACD", "Vortex", "OBV"]]

    def min_max_scale(self, df):
        return (df - df.mean()) / df.std()

    def determine_y(self, data_df, n_day_return=1):
        data_df[f"{n_day_return}_Day_Return"] = ((data_df["Adj Close"].shift(periods=n_day_return)
                                                 / data_df["Adj Close"]) - 1.0).shift(-n_day_return)
        data_df["Order"] = 0.0
        data_df.loc[data_df[f"{n_day_return}_Day_Return"] > (self.y_buy * (1 + self.impact)), "Order"] = -1
        data_df.loc[data_df[f"{n_day_return}_Day_Return"] < (self.y_sell * (1 - self.impact)), "Order"] = 1
        pt, ct = np.unique(data_df["Order"], return_counts=True)
        if 0 in pt:
            idx = np.argwhere(pt == 0)[0]
            zero_ct = ct[idx]
            if zero_ct > np.floor(data_df.shape[0] * 0.99):
                data_df.loc[data_df.index.values[0], "Order"] = 1
        return data_df

    def populate_training_data(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                               test_policy=False):
        kwargs = {"symbol": symbol, "sd": sd, "ed": ed, "lookback": self.rsi}
        all_data_df = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
        all_data_df.dropna(inplace=True)
        if not all_data_df.any().any():
            return False
        self.adj_close = all_data_df["Adj Close"]
        all_data_with_indicators = self.populate_indicators(all_data_df, **kwargs)
        # Reduce DF to only the indicators used
        if not test_policy:
            temp_y = self.determine_y(all_data_with_indicators, n_day_return=self.n_day_return)
            data = {"X": temp_y,
                    "y": temp_y["Order"]}
            return data
        elif test_policy:
            temp_y = self.learner.query(all_data_with_indicators[self.indicator_columns])
            temp_y = pd.DataFrame(columns=[symbol], data=temp_y.reshape(-1, 1))
            temp_y.set_index(all_data_with_indicators.index, drop=True, inplace=True)
            all_data_with_indicators["Pred_y"] = temp_y
            data = {"X": all_data_with_indicators.loc[temp_y.index, :],
                    "y": temp_y}
            return data

    def check_holdings(self, hold):
        if hold > 1000 or hold < -1000:
            return False
        else:
            return True

    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        results = self.populate_training_data(symbol=symbol, sd=sd, ed=ed)
        if results is False:
            return False
        kwargs = {"leaf_size": self.n_leaf}
        self.learner = bl.BagLearner(learner=rl.RTLearner, use_mode=True, bags=self.bags, **kwargs)
        self.learner.add_evidence(data_x=results["X"][self.indicator_columns], data_y=results["y"])
        return True

    def convert_old_trade_to_new_trade_frame(self, trade_df):
        trade_df.reset_index(inplace=True, drop=True)
        result = pd.DataFrame()
        symbol = trade_df.loc[0, 'Symbol']
        result[f"{symbol}"] = trade_df["Shares"].copy()
        result.set_index(trade_df["Date"], inplace=True, drop=True)
        return result

    def convert_new_trade_back_to_old_trade_frame(self, trade_df):
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

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        results = self.populate_training_data(symbol=symbol, sd=sd, ed=ed, test_policy=True)
        results["X"]["Trades"] = results["X"]["Pred_y"].diff(-1)
        results["y"].dropna(inplace=True)
        results = self.convert_trade_df(results)
        results["X"]["Current Trades"] = results["y"]
        return results["y"]

    def convert_trade_df(self, df):
        pred_y = df["y"]
        pred_y_no_nan = pred_y.copy()
        pred_y_no_nan[pred_y_no_nan == 0.0] = np.nan
        pred_y_no_nan.dropna(inplace=True)
        temp0 = pred_y.diff()
        temp1 = pred_y_no_nan.diff()
        temp2 = temp1.shift()
        temp = df["y"].diff() * 1000
        df["y"] = temp
        df["y"].loc[df["X"]["Pred_y"].first_valid_index()] = df["X"].loc[df["X"]["Pred_y"].first_valid_index(),
                                                                         "Pred_y"] * 1000
        return df

    def compute_benchmark(self, sd, ed, sv, symbol):
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
        baseline_portvals = self.compute_portvals(orders, sd, ed, sv, self.impact, self.commission)
        return baseline_portvals

    def compute_portvals(self, orders_df, start_date, end_date, startval, market_impact=0.0,
                         commission_cost=0.0, ):
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
            if (date < prices.index.min()) or (date > prices.index.max()):
                continue
            trades[symbol][date] += shares
            cash[date] -= val
        trades["_CASH"] = cash
        holdings = trades.cumsum()
        df_portvals = (prices * holdings).sum(axis=1)
        return df_portvals


if __name__ == "__main__":
    print("One does not simply think up a strategy")
