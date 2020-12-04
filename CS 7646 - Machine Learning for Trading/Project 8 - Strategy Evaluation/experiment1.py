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
import experiment1 as exp1
import experiment2 as exp2
import util as ut

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()


def author():
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
    results["Strategy Learner Results"] = {"In Sample": {"PortVal": None,
                                                         "PortVal Normalized": None,
                                                         "Trades New": None,
                                                         "Trades": None,
                                                         "Adj Close": None,
                                                         "Adj Close Normalized": None,
                                                         "Cumulative Return": None
                                                         },
                                           "Out Sample": {"PortVal": None,
                                                          "PortVal Normalized": None,
                                                          "Trades New": None,
                                                          "Trades": None,
                                                          "Adj Close": None,
                                                          "Adj Close Normalized": None,
                                                          "Cumulative Return": None
                                                          }
                                           }
    in_sample_price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
    in_sample_price_df.drop("SPY", inplace=True, axis=1)
    out_sample_price_df = mkt.get_prices(symbols=[sym], start_date=oos_sd, end_date=oos_ed)
    out_sample_price_df.drop("SPY", inplace=True, axis=1)

    # region Strategy Learner
    # region In Sample
    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    keep_going = learner.add_evidence(symbol=sym, sd=sd, ed=ed, sv=sv)
    if not keep_going:
        print(f"Something wrong with obtaining data of symbol: {sym}")
        exit()
    results["Strategy Learner Results"]["In Sample"]["Trades New"] = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
    if not verify_returned_dataframe_correct(results["Strategy Learner Results"]["In Sample"]["Trades New"], symbol=sym,
                                             sd=sd, ed=ed):
        raise ValueError
    results["Strategy Learner Results"]["In Sample"]["Trades"] = learner.convert_new_trade_back_to_old_trade_frame(
        results["Strategy Learner Results"]["In Sample"]["Trades New"])
    results["Strategy Learner Results"]["In Sample"]["PortVal"] = compute_portvals(
        orders_df=results["Strategy Learner Results"]["In Sample"]["Trades"],
        start_date=sd, end_date=ed,
        startval=sv, market_impact=impact,
        commission_cost=commission)
    results["Strategy Learner Results"]["In Sample"]["PortVal Normalized"] = \
        results["Strategy Learner Results"]["In Sample"]["PortVal"] / \
        results["Strategy Learner Results"]["In Sample"]["PortVal"][0]
    results["Strategy Learner Results"]["In Sample"]["Cumulative Return"] = np.round(
        (results["Strategy Learner Results"]["In Sample"]["PortVal Normalized"][-1] /
         results["Strategy Learner Results"]["In Sample"]["PortVal Normalized"][0]) - 1.0, 5)
    results["Strategy Learner Results"]["In Sample"]["Adj Close"] = in_sample_price_df[sym]
    results["Strategy Learner Results"]["In Sample"]["Adj Close Normalized"] = in_sample_price_df[sym] / \
                                                                               in_sample_price_df[sym][0]

    # endregion

    # region Out Sample
    results["Strategy Learner Results"]["Out Sample"]["Trades New"] = learner.testPolicy(symbol=sym, sd=oos_sd,
                                                                                         ed=oos_ed, sv=sv)
    if not verify_returned_dataframe_correct(results["Strategy Learner Results"]["Out Sample"]["Trades New"],
                                             symbol=sym, sd=oos_sd, ed=oos_ed):
        raise ValueError
    results["Strategy Learner Results"]["Out Sample"]["Trades"] = learner.convert_new_trade_back_to_old_trade_frame(
        results["Strategy Learner Results"]["Out Sample"]["Trades New"])
    results["Strategy Learner Results"]["Out Sample"]["PortVal"] = compute_portvals(
        orders_df=results["Strategy Learner Results"]["Out Sample"]["Trades"],
        start_date=oos_sd, end_date=oos_ed,
        startval=sv, market_impact=impact,
        commission_cost=commission)
    results["Strategy Learner Results"]["Out Sample"]["PortVal Normalized"] = \
        results["Strategy Learner Results"]["Out Sample"]["PortVal"] / \
        results["Strategy Learner Results"]["Out Sample"]["PortVal"][0]
    results["Strategy Learner Results"]["Out Sample"]["Cumulative Return"] = np.round(
        (results["Strategy Learner Results"]["Out Sample"]["PortVal Normalized"][-1] /
         results["Strategy Learner Results"]["Out Sample"]["PortVal Normalized"][0]) - 1.0, 5)

    results["Strategy Learner Results"]["Out Sample"]["Adj Close"] = out_sample_price_df[sym]
    results["Strategy Learner Results"]["Out Sample"]["Adj Close Normalized"] = out_sample_price_df[sym] / \
                                                                                out_sample_price_df[sym][0]
    # endregion
    # endregion

    plot = "Experiment One"
    buy_long_color = "tab:blue"
    sell_short_color = "black"
    benchmark_color = "tab:green"
    color = "tab:red"

    benchmark_portval_in_sample = results["Benchmark Results"]["In Sample"]["PortVal Normalized"]
    benchmark_cumulative_in_sample_return = results["Benchmark Results"]["In Sample"]["Cumulative Return"]
    benchmark_portval_out_sample = results["Benchmark Results"]["Out Sample"]["PortVal Normalized"]
    benchmark_cumulative_out_sample_return = results["Benchmark Results"]["Out Sample"]["Cumulative Return"]
    adjusted_close_in_sample = results["Benchmark Results"]["In Sample"]["Adj Close Normalized"]
    adjusted_close_out_sample = results["Benchmark Results"]["Out Sample"]["Adj Close Normalized"]
    portval_in_sample = results["Strategy Learner Results"]["In Sample"]["PortVal Normalized"]
    cumulative_return_in_sample = results["Strategy Learner Results"]["In Sample"]["Cumulative Return"]
    trades_df_in_sample = results["Strategy Learner Results"]["In Sample"]["Trades New"]
    title_in_sample = f"In Sample\nStrategy Learner vs Manual Strategy vs Benchmark"

    portval_out_sample = results["Strategy Learner Results"]["Out Sample"]["PortVal Normalized"]
    cumulative_return_out_sample = results["Strategy Learner Results"]["Out Sample"]["Cumulative Return"]
    trades_df_out_sample = results["Strategy Learner Results"]["Out Sample"]["Trades New"]
    title_out_sample = f"Out Sample\nStrategy Learner vs Manual Strategy vs Benchmark"

    # region In Sample
    sample_type = "In Sample"
    plt.close("all")
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[8, 3, 3], figure=fig, wspace=0.05, hspace=0.15)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    ax0.set_title(f"{title_in_sample}", fontsize=20,
                  weight='bold')

    ax0.plot(results["Strategy Learner Results"]["In Sample"]["PortVal Normalized"],
             label=f"Strategy Learner - "
                   f"Return:{results['Strategy Learner Results']['In Sample']['Cumulative Return']:.3f}"
             , linewidth=3)

    ax0.plot(results["Manual Results"]["In Sample"]["PortVal Normalized"],
             label=f"Manual Strategy - "
                   f"Return:{results['Manual Results']['In Sample']['Cumulative Return']:.3f}"
             , linewidth=3)

    ax0.plot(results["Benchmark Results"]["In Sample"]["PortVal Normalized"],
             label=f"Benchmark - "
                   f"Return:{results['Benchmark Results']['In Sample']['Cumulative Return']:.3f}",
             linewidth=3)
    ax0.plot(adjusted_close_in_sample, label=f"Adjusted Close", linewidth=1, color="tab:orange",
             linestyle="--")
    ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
    plt.setp(ax0.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax0.xaxis.set_major_formatter(plt.NullFormatter())
    ax0.set_ylabel("Return", fontsize=20, weight='heavy')
    ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
    ax0.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)

    ax1.plot(results["Strategy Learner Results"]["In Sample"]["Trades New"][sym].cumsum(),
             label=f"Strategy Holdings", linewidth=2, color="tab:olive")
    ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.set_ylabel("Learner Holdings", fontsize=10, weight='heavy')
    ax1.grid(which='both', linestyle='-', linewidth='0.5', color='white')
    ax1.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)

    ax2.plot(results["Manual Results"]["In Sample"]["Trades New"][sym].cumsum(),
             label=f"Manual Holdings", linewidth=2, color="tab:olive")
    ax2.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax2.tick_params(which="major", bottom=True, left=True, labelsize=15)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax2.set_ylabel("Manual Holdings", fontsize=10, weight='heavy')
    ax2.set_xlabel("Date", fontsize=20, weight='heavy')
    ax2.grid(which='both', linestyle='-', linewidth='0.5', color='white')
    ax2.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)

    plt.savefig(f"{plot}_{sample_type}_{sym}.png")
    plt.close("all")
    print(f"Finished on {plot} {sample_type} {sym}")
    # endregion

    # region Out Sample
    sample_type = "Out Sample"
    plt.close("all")
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[8, 3, 3], figure=fig, wspace=0.05, hspace=0.15)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    ax0.set_title(f"{title_out_sample}", fontsize=20,
                  weight='bold')

    ax0.plot(results["Strategy Learner Results"]["Out Sample"]["PortVal Normalized"],
             label=f"Strategy Learner - "
                   f"Return:{results['Strategy Learner Results']['Out Sample']['Cumulative Return']:.3f}"
             , linewidth=3)

    ax0.plot(results["Manual Results"]["Out Sample"]["PortVal Normalized"],
             label=f"Manual Strategy - "
                   f"Return:{results['Manual Results']['Out Sample']['Cumulative Return']:.3f}"
             , linewidth=3)

    ax0.plot(results["Benchmark Results"]["Out Sample"]["PortVal Normalized"],
             label=f"Benchmark - "
                   f"Return:{results['Benchmark Results']['Out Sample']['Cumulative Return']:.3f}",
             linewidth=3)
    ax0.plot(adjusted_close_out_sample, label=f"Adjusted Close", linewidth=1, color="tab:orange",
             linestyle="--")
    ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
    plt.setp(ax0.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax0.xaxis.set_major_formatter(plt.NullFormatter())
    ax0.set_ylabel("Return", fontsize=20, weight='heavy')
    ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
    ax0.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)

    ax1.plot(results["Strategy Learner Results"]["Out Sample"]["Trades New"][sym].cumsum(),
             label=f"Learner Holdings", linewidth=2, color="tab:olive")
    ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.set_ylabel("Learner Holdings", fontsize=10, weight='heavy')
    ax1.grid(which='both', linestyle='-', linewidth='0.5', color='white')
    ax1.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)

    ax2.plot(results["Manual Results"]["Out Sample"]["Trades New"][sym].cumsum(),
             label=f"Manual Holdings", linewidth=2, color="tab:olive")
    ax2.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax2.tick_params(which="major", bottom=True, left=True, labelsize=15)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax2.set_ylabel("Manual Holdings", fontsize=10, weight='heavy')
    ax2.set_xlabel("Date", fontsize=20, weight='heavy')
    ax2.grid(which='both', linestyle='-', linewidth='0.5', color='white')
    ax2.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)

    plt.savefig(f"{plot}_{sample_type}_{sym}.png")
    plt.close("all")
    print(f"Finished on {plot} {sample_type} {sym}")
    # endregion
    return
