"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""

import datetime as dt
import time
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as mkt
import util as ut

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()


def author():
	"""
    :return: The GT username of the student
    :rtype: str
    """
	return "jadams334"  # Change this to your user ID


def verify_returned_dataframe_correct(dataframe, symbol, sd, ed):
    dataframe = dataframe.round()
    # Verify doesnt contain nan
    if dataframe.isnull().any()[0]:
        return False
    # Verify just one column
    if dataframe.shape[1] != 1:
        return False
    # Verify the columns name is the symbol
    if dataframe.columns[0] != symbol:
        return False
    # Verify the values are in between -2000 and 2000
    greater_than = dataframe[dataframe > 2000]
    less_than = dataframe[dataframe < -2001]
    if greater_than.any()[0] or less_than.any()[0]:
        return False
    # Verify the cumulative sum is at most 1000 and at least -1000
    cumulative = dataframe.cumsum()
    greater_than = cumulative[cumulative > 1000]
    less_than = cumulative[cumulative < -1000]
    if greater_than.any()[0] or less_than.any()[0]:
        return False
    # Verify the dates are the index
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        return False
    return True


def compute_portvals(orders_df, start_date, end_date, startval, market_impact=0.0,
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
        val += commission_cost + (np.abs(shares) * price * market_impact)
        if (date < prices.index.min()) or (date > prices.index.max()):
            continue
        trades[symbol][date] += shares
        cash[date] -= val
    trades["_CASH"] = cash
    holdings = trades.cumsum()
    df_portvals = (prices * holdings).sum(axis=1)
    return df_portvals


def run(results, sym="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
        oos_sd=dt.datetime(2010, 1, 1), oos_ed=dt.datetime(2011, 12, 31), impact=0.005, commission=9.95):
    in_sample_price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
    in_sample_price_df.drop("SPY", inplace=True, axis=1)
    out_sample_price_df = mkt.get_prices(symbols=[sym], start_date=oos_sd, end_date=oos_ed)
    out_sample_price_df.drop("SPY", inplace=True, axis=1)
    temp_impact_range = np.round(10. ** np.arange(-4, 1, 1), 4)
    impact_range = np.sort(np.hstack((temp_impact_range, np.zeros(shape=(1, )), -temp_impact_range)))
    impact_results = pd.DataFrame(index=impact_range, columns=["Number of Trades", "Cumulative Return"])
    # region Strategy Learner
    # region In Sample
    try:
        for temp_impact in impact_range:
            learner = sl.StrategyLearner(verbose=False, impact=temp_impact, commission=commission, bags=50)
            keep_going = learner.add_evidence(symbol=sym, sd=sd, ed=ed, sv=sv)
            if not keep_going:
                print(f"Something wrong with obtaining data of symbol: {sym}")
                exit()
            temp_trades = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
            if not verify_returned_dataframe_correct(temp_trades, symbol=sym, sd=sd, ed=ed):
                raise ValueError
            temp_trades_converted = learner.convert_new_trade_back_to_old_trade_frame(temp_trades)

            temp_portvals = compute_portvals(
                orders_df=temp_trades_converted,
                start_date=sd, end_date=ed,
                startval=sv, market_impact=impact,
                commission_cost=commission)

            temp_portvals_normalized = temp_portvals / temp_portvals[0]
            cumulative_returns = np.round((temp_portvals_normalized[-1] / temp_portvals_normalized[0]) - 1.0, 5)
            temp_trades[temp_trades == 0] = np.nan
            temp_trades.dropna(inplace=True)
            number_of_trades = temp_trades.shape[0]
            impact_results.loc[temp_impact, "Cumulative Return"] = cumulative_returns
            impact_results.loc[temp_impact, "Number of Trades"] = number_of_trades
            adj_close = in_sample_price_df[sym]
            adj_close_normalized = adj_close / adj_close[0]


        # endregion

        # endregion

        plot = "Experiment Two"
        adjusted_close_in_sample = results["Benchmark Results"]["In Sample"]["Adj Close Normalized"]
        title_in_sample = f"Experiment Two\n" \
                          f"Strategy Learner Behavior vs Impact"

        # region In Sample
        sample_type = "In Sample"
        plt.close("all")
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        # ax1 = ax.twinx()
        impact_results.plot(kind="bar", figsize=(10, 8), secondary_y=["Number of Trades"], title=title_in_sample, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        ax.set_ylabel("Cumulative Return", fontsize=10, weight='heavy')
        ax.set_xlabel("Impact", fontsize=20, weight='heavy')
        plt.savefig(f"{plot}_{sample_type}_{sym}_Impact.png")
        plt.close("all")
        print(f"Finished on {plot} {sample_type} {sym}")
    except Exception as _exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in ''", _exception)
    # endregion
    return
