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


def get_chart_one(sym, benchmark_port_val, benchmark_cumulative_return, manual_strategy_port_val,
                  manual_strategy_cumulative_return, manual_trades, adj_close, sample_type, plot_adj=False, ax0=None,
                  ax1=None):
    buy_long_color = "tab:blue"
    sell_short_color = "black"
    benchmark_color = "tab:green"
    strategy_color = "tab:orange"
    manual_color = "tab:red"
    save = False
    if ax0 is None:
        fig, ax0 = plt.subplots(figsize=(10, 8))
        save = True

    ax0.plot(manual_strategy_port_val,
             label=f"Manual Strategy - Return:{manual_strategy_cumulative_return:.3f}",
             color=manual_color, linewidth=3)

    ax0.plot(benchmark_port_val, label=f"Benchmark - Return:{benchmark_cumulative_return:.3f}",
             color=benchmark_color, linewidth=3)

    long = manual_trades.loc[manual_trades[f"{sym}"] == 2000.0]
    short = manual_trades.loc[manual_trades[f"{sym}"] == -2000.0]

    for b in long.index.values:
        ax0.axvline(b, color=buy_long_color, alpha=0.9, linestyle="--", linewidth=1)
    for s in short.index.values:
        ax0.axvline(s, color=sell_short_color, alpha=0.9, linestyle="--", linewidth=1)

    normalized_adj = adj_close / adj_close[0]
    ax0.plot(normalized_adj, label=f"Adjusted Close", linewidth=1, color="tab:orange", linestyle="--")
    ax1.plot(manual_trades.cumsum(), label=f"Holdings", linewidth=2, color="tab:olive")

    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
    ax1.set_xlabel("Date", fontsize=20, weight='heavy')
    ax1.set_ylabel("Total Holdings", fontsize=20, weight='heavy')
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
    ax1.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)

    ax0.set_title(f"{sym} - {sample_type} Sample\nManual Strategy vs Benchmark", fontsize=20,
                  weight='bold')

    plt.setp(ax0.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
    ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
    ax0.xaxis.set_major_formatter(plt.NullFormatter())
    ax0.set_ylabel("Return", fontsize=20, weight='heavy')
    ax0.grid(which='major', linestyle='-', linewidth='0.5', color='white')
    ax0.legend(loc="best", markerscale=1.1, frameon=True,
               edgecolor="black", fancybox=True, shadow=True, fontsize=12)
    if save:
        plt.savefig(f"Chart_One_{sample_type}_Sample_{sym}.png")
        plt.close("all")
        print(f"Finished on {sample_type}-Sample {sym}")
    return


def convert_new_trade_back_to_old_trade_frame(trade_df):
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


def compute_benchmark(sd, ed, sv, symbol, impact, commission, max_holdings):
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
    orders.loc[orders.index[-1], "Shares"] = -max_holdings
    orders.loc[orders.index[-1], "Order"] = "Sell"
    baseline_portvals = compute_portvals(orders_df=orders, start_date=sd, end_date=ed, startval=sv,
                                         market_impact=impact, commission_cost=commission)
    return baseline_portvals


def test_manual_strategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, **kwargs):
    trade_df = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    order_df = convert_new_trade_back_to_old_trade_frame(trade_df)
    port_vals = compute_portvals(orders_df=order_df, start_date=sd, end_date=ed, startval=sv,
                                 market_impact=0.005, commission_cost=9.95)
    return


def check_autograde_requirements(symbol, add_evidence_time, in_sample_time, oos_time, results_check,
                                 in_sample_return, oos_return, in_sample_benchmark_return, oos_benchmark_return):
    if symbol not in ["ML4T-220", "AAPL", "UNH", "SINE_FAST_NOISE"]:
        return
    points = 0
    if symbol == "AAPL":
        # Add Evidence within 25 sec
        if add_evidence_time < 25.0:
            pt = 1
            points += pt
            print(f"Add Evidence Time was less than 25s. {pt} Point added, \tcurrent Points: {points}")
        # Check if returns are same
        if results_check[0].equals(results_check[1]):
            pt = 2
            points += pt
            print(
                f"Results were the same from two calls of test policy. {pt} Point added, \tcurrent Points: {points}")
        # in sample within 5 seconds
        if in_sample_time < 5.0:
            pt = 2
            points += pt
            print(f"In Sample Test Policy Time was less than 5s. {pt} Points added, \tcurrent Points: {points}")
        # In same cumulative return greater than 100
        if in_sample_return > in_sample_benchmark_return:
            pt = 5
            points += pt
            print(f"In Sample Return was greater than Benchmark. {pt} Points added, \tcurrent Points: {points}")
        # OOS time less than 5 seconds
        if oos_time < 5.0:
            pt = 5
            points += pt
            print(f"OOS Sample Test Policy Time was less than 5s. {pt} Points added, \tcurrent Points: {points}")
    elif symbol == "SINE_FAST_NOISE":
        # Add Evidence within 25 sec
        if add_evidence_time < 25.0:
            pt = 1
            points += pt
            print(f"Add Evidence Time was less than 25s. {pt} Point added, \tcurrent Points: {points}")
        # Check if returns are same
        if results_check[0].equals(results_check[1]):
            pt = 2
            points += pt
            print(
                f"Results were the same from two calls of test policy. {pt} Point added, \tcurrent Points: {points}")
        # in sample within 5 seconds
        if in_sample_time < 5.0:
            pt = 2
            points += pt
            print(f"In Sample Test Policy Time was less than 5s. {pt} Points added, \tcurrent Points: {points}")
        # In same cumulative return greater than 100
        if in_sample_return > 2.0:
            pt = 5
            points += pt
            print(f"In Sample Return was greater than 100%. {pt} Points added, \tcurrent Points: {points}")
        # OOS time less than 5 seconds
        if oos_time < 5.0:
            pt = 5
            points += pt
            print(f"OOS Sample Test Policy Time was less than 5s. {pt} Points added, \tcurrent Points: {points}")
    elif symbol == "UNH":
        # Add Evidence within 25 sec
        if add_evidence_time < 25.0:
            pt = 1
            points += pt
            print(f"Add Evidence Time was less than 25s. {pt} Point added, \tcurrent Points: {points}")
        # Check if returns are same
        if results_check[0].equals(results_check[1]):
            pt = 2
            points += pt
            print(
                f"Results were the same from two calls of test policy. {pt} Point added, \tcurrent Points: {points}")
        # in sample within 5 seconds
        if in_sample_time < 5.0:
            pt = 2
            points += pt
            print(f"In Sample Test Policy Time was less than 5s. {pt} Points added, \tcurrent Points: {points}")
        # In sample cumulative return greater than benchmark
        if in_sample_return > in_sample_benchmark_return:
            pt = 5
            points += pt
            print(f"In Sample Return was greater than 100%. {pt} Points added, \tcurrent Points: {points}")
        # OOS time less than 5 seconds
        if oos_time < 5.0:
            pt = 5
            points += pt
            print(f"OOS Sample Test Policy Time was less than 5s. {pt} Points added, \tcurrent Points: {points}")
    elif symbol == "ML4T-220":
        # Add Evidence within 25 sec
        if add_evidence_time < 25.0:
            pt = 1
            points += pt
            print(f"Add Evidence Time was less than 25s. {pt} Point added, \tcurrent Points: {points}")
        # Check if returns are same
        if results_check[0].equals(results_check[1]):
            pt = 2
            points += pt
            print(
                f"Results were the same from two calls of test policy. {pt} Point added, \tcurrent Points: {points}")
        # in sample within 5 seconds
        if in_sample_time < 5.0:
            pt = 2
            points += pt
            print(f"In Sample Test Policy Time was less than 5s. {pt} Points added, \tcurrent Points: {points}")
        # In same cumulative return greater than 100
        if in_sample_return > 1.0:
            pt = 5
            points += pt
            print(f"In Sample Return was greater than 100%. {pt} Points added, \tcurrent Points: {points}")
        # OOS cumulative return greater than 100
        if oos_return > 1.0:
            pt = 5
            points += pt
            print(f"Out Of Sample Return was greater than 100%. {pt} Points added, \tcurrent Points: {points}")
    print(f"Total Points for {symbol}: {points} / 15")
    return


def run_part1(sym, in_sample_benchmark_cumulative_return, out_sample_benchmark_cumulative_return,
              in_sample_manual_cumulative_return, out_sample_manual_cumulative_return,
              manual_strategy_in_sample_port_val, manual_strategy_out_sample_port_val,
              in_sample_benchmark_portval, out_sample_benchmark_portval):

    space_1 = "\t" * 1

    print(f"\n\n{space_1 * 7}~~Manual Strategy Daily Returns Table~~")
    print()
    print(f"{space_1 * 6}.:In-Sample:.{space_1 * 7}.:Out-Sample:.")
    print()
    print(f"{space_1 * 4}Manual Strategy{space_1 * 2}Benchmark{space_1 * 3}Manual Strategy{space_1 * 2}Benchmark")

    print(f"Cumulative {space_1 * 3}{in_sample_manual_cumulative_return:.3f} {space_1 * 3} "
          f"{in_sample_benchmark_cumulative_return:.3f}{space_1}"
          f"{space_1 * 4}{out_sample_manual_cumulative_return:.3f} {space_1 * 3} "
          f"{out_sample_benchmark_cumulative_return:.3f}")
    print(f"STDEV {space_1 * 4}{mkt.get_daily_returns(manual_strategy_in_sample_port_val).std():.3f} {space_1 * 3}"
          f" {mkt.get_daily_returns(in_sample_benchmark_portval).std():.3f}{space_1}"
          f"{space_1 * 4}{mkt.get_daily_returns(manual_strategy_out_sample_port_val).std():.3f} {space_1 * 3} "
          f"{mkt.get_daily_returns(out_sample_benchmark_portval).std():.3f}")
    print(f"Mean {space_1 * 4}{mkt.get_daily_returns(manual_strategy_in_sample_port_val).mean():.3f} {space_1 * 3}"
          f" {mkt.get_daily_returns(in_sample_benchmark_portval).mean():.3f}{space_1}"
          f"{space_1 * 4}{mkt.get_daily_returns(manual_strategy_out_sample_port_val).mean():.3f} {space_1 * 3} "
          f"{mkt.get_daily_returns(out_sample_benchmark_portval).mean():.3f}")

    print(f"Finished on Part 1 - Manual Strategy - {sym}")
    return


def run_learners(sym="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                 oos_sd=dt.datetime(2010, 1, 1), oos_ed=dt.datetime(2011, 12, 31),
                 sv=100000, commission=9.95, impact=0.005):
    buy_long_color = "tab:blue"
    sell_short_color = "black"
    benchmark_color = "tab:green"
    strategy_color = "tab:orange"
    manual_color = "tab:red"
    results = {
        "Manual Results": {"In Sample": {"PortVal": None,
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
                           },
        "Benchmark Results": {"In Sample": {"PortVal": None,
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
    }

    if sym != "JPM":
        print(f"The symbol that was passed in was {sym}, the correct symbol is 'JPM'. To resolve this issue \n"
              f"check the parameter sym that is being passed into the function 'run_learners'. The call to \n"
              f"'run_learners is inside the main function starting on line 686 in the file 'testproject.py'")
        raise ValueError
    if sd != dt.datetime(2008, 1, 1):
        raise ValueError
    if ed != dt.datetime(2009, 12, 31):
        raise ValueError
    if oos_sd != dt.datetime(2010, 1, 1):
        raise ValueError
    if oos_ed != dt.datetime(2011, 12, 31):
        raise ValueError
    if sv != 100000:
        raise ValueError
    if commission != 9.95:
        raise ValueError
    if impact != 0.005:
        raise ValueError

    in_sample_price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
    in_sample_price_df.drop("SPY", inplace=True, axis=1)
    out_sample_price_df = mkt.get_prices(symbols=[sym], start_date=oos_sd, end_date=oos_ed)
    out_sample_price_df.drop("SPY", inplace=True, axis=1)

    # region Manual Strategy
    # region In Sample
    results["Manual Results"]["In Sample"]["Trades New"] = ms.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
    results["Manual Results"]["In Sample"]["Trades"] = convert_new_trade_back_to_old_trade_frame(
        results["Manual Results"]["In Sample"]["Trades New"])
    results["Manual Results"]["In Sample"]["PortVal"] = compute_portvals(
        orders_df=results["Manual Results"]["In Sample"]["Trades"],
        start_date=sd, end_date=ed, startval=sv,
        market_impact=impact, commission_cost=commission)
    results["Manual Results"]["In Sample"]["PortVal Normalized"] = results["Manual Results"]["In Sample"]["PortVal"] / \
                                                                   results["Manual Results"]["In Sample"]["PortVal"][
                                                                       0]
    results["Manual Results"]["In Sample"]["Cumulative Return"] = np.round(
        (results["Manual Results"]["In Sample"]["PortVal Normalized"][-1] /
         results["Manual Results"]["In Sample"]["PortVal Normalized"][0]) - 1.0, 4)

    results["Manual Results"]["In Sample"]["Adj Close"] = in_sample_price_df[sym]
    results["Manual Results"]["In Sample"]["Adj Close Normalized"] = in_sample_price_df[sym] / in_sample_price_df[sym][
        0]
    # endregion

    # region Out Sample
    results["Manual Results"]["Out Sample"]["Trades New"] = ms.testPolicy(symbol=sym, sd=oos_sd, ed=oos_ed, sv=sv)
    results["Manual Results"]["Out Sample"]["Trades"] = convert_new_trade_back_to_old_trade_frame(
        results["Manual Results"]["Out Sample"]["Trades New"])
    results["Manual Results"]["Out Sample"]["PortVal"] = compute_portvals(
        orders_df=results["Manual Results"]["Out Sample"]["Trades"],
        start_date=oos_sd, end_date=oos_ed,
        startval=sv, market_impact=impact,
        commission_cost=commission)
    results["Manual Results"]["Out Sample"]["PortVal Normalized"] = results["Manual Results"]["Out Sample"]["PortVal"] / \
                                                                    results["Manual Results"]["Out Sample"]["PortVal"][
                                                                        0]
    results["Manual Results"]["Out Sample"]["Cumulative Return"] = np.round(
        (results["Manual Results"]["Out Sample"]["PortVal Normalized"][-1] /
         results["Manual Results"]["Out Sample"]["PortVal Normalized"][0]) - 1.0, 4)

    results["Manual Results"]["Out Sample"]["Adj Close"] = out_sample_price_df[sym]
    results["Manual Results"]["Out Sample"]["Adj Close Normalized"] = out_sample_price_df[sym] / \
                                                                      out_sample_price_df[sym][0]
    # endregion
    # endregion

    # region Benchmark
    # region In Sample
    results["Benchmark Results"]["In Sample"]["PortVal"] = compute_benchmark(symbol=sym, sd=sd, ed=ed, sv=sv,
                                                                             impact=impact,
                                                                             commission=commission, max_holdings=1000)
    results["Benchmark Results"]["In Sample"]["PortVal Normalized"] = results["Benchmark Results"]["In Sample"][
                                                                          "PortVal"] / \
                                                                      results["Benchmark Results"]["In Sample"][
                                                                          "PortVal"][0]
    results["Benchmark Results"]["In Sample"]["Cumulative Return"] = np.round(
        (results["Benchmark Results"]["In Sample"]["PortVal Normalized"][-1] /
         results["Benchmark Results"]["In Sample"]["PortVal Normalized"][0]) - 1.0, 4)

    results["Benchmark Results"]["In Sample"]["Adj Close"] = in_sample_price_df[sym]
    results["Benchmark Results"]["In Sample"]["Adj Close Normalized"] = in_sample_price_df[sym] / \
                                                                        in_sample_price_df[sym][0]
    # endregion

    # region Out Sample
    results["Benchmark Results"]["Out Sample"]["PortVal"] = compute_benchmark(symbol=sym, sd=oos_sd, ed=oos_ed, sv=sv,
                                                                              impact=impact, commission=commission,
                                                                              max_holdings=1000)
    results["Benchmark Results"]["Out Sample"]["PortVal Normalized"] = results["Benchmark Results"]["Out Sample"][
                                                                           "PortVal"] / \
                                                                       results["Benchmark Results"]["Out Sample"][
                                                                           "PortVal"][0]
    results["Benchmark Results"]["Out Sample"]["Cumulative Return"] = np.round(
        (results["Benchmark Results"]["Out Sample"]["PortVal Normalized"][-1] /
         results["Benchmark Results"]["Out Sample"]["PortVal Normalized"][0]) - 1.0, 4)
    results["Benchmark Results"]["Out Sample"]["Adj Close"] = out_sample_price_df[sym]
    results["Benchmark Results"]["Out Sample"]["Adj Close Normalized"] = out_sample_price_df[sym] / \
                                                                         out_sample_price_df[sym][0]
    # endregion
    # endregion

    return results


def plot_results(results, plot="", sym="JPM"):
    try:
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

        if plot == "Strategy Learner" or plot == "Manual Strategy":
            if plot == "Manual Strategy":
                line_name = "Manual Strategy"
                portval_in_sample = results["Manual Results"]["In Sample"]["PortVal Normalized"]
                cumulative_return_in_sample = results["Manual Results"]["In Sample"]["Cumulative Return"]
                trades_df_in_sample = results["Manual Results"]["In Sample"]["Trades New"]
                title_in_sample = f"{sym} - In Sample\nManual Strategy vs Benchmark"

                portval_out_sample = results["Manual Results"]["Out Sample"]["PortVal Normalized"]
                cumulative_return_out_sample = results["Manual Results"]["Out Sample"]["Cumulative Return"]
                trades_df_out_sample = results["Manual Results"]["Out Sample"]["Trades New"]
                title_out_sample = f"{sym} - Out Sample\nManual Strategy vs Benchmark"
            elif plot == "Strategy Learner":
                line_name = "Strategy Learner"
                portval_in_sample = results["Strategy Learner Results"]["In Sample"]["PortVal Normalized"]
                cumulative_return_in_sample = results["Strategy Learner Results"]["In Sample"]["Cumulative Return"]
                trades_df_in_sample = results["Strategy Learner Results"]["In Sample"]["Trades New"]
                title_in_sample = f"{sym} - In Sample\nStrategy Learner vs Benchmark"

                portval_out_sample = results["Strategy Learner Results"]["Out Sample"]["PortVal Normalized"]
                cumulative_return_out_sample = results["Strategy Learner Results"]["Out Sample"]["Cumulative Return"]
                trades_df_out_sample = results["Strategy Learner Results"]["Out Sample"]["Trades New"]
                title_out_sample = f"{sym} - Out Sample\nStrategy Learner vs Benchmark"
            # region In Sample
            sample_type = "In Sample"
            plt.close("all")
            plt.style.use("ggplot")
            fig = plt.figure(figsize=(16, 10), constrained_layout=True)
            gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])

            ax0.plot(portval_in_sample,
                     label=f"{line_name} - Return:{cumulative_return_in_sample:.3f}",
                     color=color, linewidth=3)

            ax0.plot(benchmark_portval_in_sample,
                     label=f"Benchmark - Return:{benchmark_cumulative_in_sample_return:.3f}",
                     color=benchmark_color, linewidth=3)
            in_sample_holdings = trades_df_in_sample[sym].cumsum()
            in_sample_holdings[in_sample_holdings == 0] = np.nan
            new_hold_in = in_sample_holdings.dropna()
            new_hold_in_diff = new_hold_in.diff()
            long = new_hold_in_diff.loc[new_hold_in_diff == 2000.0]
            short = new_hold_in_diff.loc[new_hold_in_diff == -2000.0]

            for b in long.index.values:
                ax0.axvline(b, color=buy_long_color, alpha=0.9, linestyle="--", linewidth=1)
            for s in short.index.values:
                ax0.axvline(s, color=sell_short_color, alpha=0.9, linestyle="--", linewidth=1)

            ax0.plot(adjusted_close_in_sample, label=f"Adjusted Close", linewidth=1, color="tab:orange", linestyle="--")

            ax1.plot(trades_df_in_sample[sym].cumsum(), label=f"Holdings", linewidth=2, color="tab:olive")

            plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                     rotation_mode="anchor")

            ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
            ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
            ax1.set_xlabel("Date", fontsize=20, weight='heavy')
            ax1.set_ylabel("Total Holdings", fontsize=20, weight='heavy')
            ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            ax1.legend(loc="best", markerscale=1.1, frameon=True,
                       edgecolor="black", fancybox=True, shadow=True, fontsize=12)

            ax0.set_title(f"{title_in_sample}", fontsize=20,
                          weight='bold')

            plt.setp(ax0.get_xticklabels(), rotation=30, ha="right",
                     rotation_mode="anchor")

            ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
            ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
            ax0.xaxis.set_major_formatter(plt.NullFormatter())
            ax0.set_ylabel("Return", fontsize=20, weight='heavy')
            ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
            ax0.legend(loc="best", markerscale=1.1, frameon=True,
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
            gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])

            ax0.plot(portval_out_sample,
                     label=f"{line_name} - Return:{cumulative_return_out_sample:.3f}",
                     color=color, linewidth=3)

            ax0.plot(benchmark_portval_out_sample,
                     label=f"Benchmark - Return:{benchmark_cumulative_out_sample_return:.3f}",
                     color=benchmark_color, linewidth=3)
            out_sample_holdings = trades_df_out_sample[sym].cumsum()
            out_sample_holdings[out_sample_holdings == 0] = np.nan
            new_hold_out = out_sample_holdings.dropna()
            new_hold_out_diff = new_hold_out.diff()
            long = new_hold_out_diff.loc[new_hold_out_diff == 2000.0]
            short = new_hold_out_diff.loc[new_hold_out_diff == -2000.0]

            for b in long.index.values:
                ax0.axvline(b, color=buy_long_color, alpha=0.9, linestyle="--", linewidth=1)
            for s in short.index.values:
                ax0.axvline(s, color=sell_short_color, alpha=0.9, linestyle="--", linewidth=1)

            ax0.plot(adjusted_close_out_sample, label=f"Adjusted Close", linewidth=1, color="tab:orange", linestyle="--")

            ax1.plot(trades_df_out_sample[sym].cumsum(), label=f"Holdings", linewidth=2, color="tab:olive")

            plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                     rotation_mode="anchor")

            ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
            ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
            ax1.set_xlabel("Date", fontsize=20, weight='heavy')
            ax1.set_ylabel("Total Holdings", fontsize=20, weight='heavy')
            ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            ax1.legend(loc="best", markerscale=1.1, frameon=True,
                       edgecolor="black", fancybox=True, shadow=True, fontsize=12)

            ax0.set_title(f"{title_out_sample}", fontsize=20,
                          weight='bold')

            plt.setp(ax0.get_xticklabels(), rotation=30, ha="right",
                     rotation_mode="anchor")

            ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
            ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
            ax0.xaxis.set_major_formatter(plt.NullFormatter())
            ax0.set_ylabel("Return", fontsize=20, weight='heavy')
            ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
            ax0.legend(loc="best", markerscale=1.1, frameon=True,
                       edgecolor="black", fancybox=True, shadow=True, fontsize=12)

            plt.savefig(f"{plot}_{sample_type}_{sym}.png")
            plt.close("all")
            print(f"Finished on {plot} {sample_type} {sym}")
            # endregion
        elif plot == "Experiment One" or plot == "Experiment Two":
            if plot == "Experiment One":
                exp1.run(results, sym=sym)
            elif plot == "Experiment Two":
                exp2.run(results, sym=sym)
        return
    except Exception as _exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in ''", _exception)


if __name__ == '__main__':


    sym = "JPM"
    results = run_learners(sym=sym, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    run_part1(sym=sym,
              in_sample_benchmark_cumulative_return=results["Benchmark Results"]["In Sample"]["Cumulative Return"],
              out_sample_benchmark_cumulative_return=results["Benchmark Results"]["Out Sample"]["Cumulative Return"],
              in_sample_manual_cumulative_return=results["Manual Results"]["In Sample"]["Cumulative Return"],
              out_sample_manual_cumulative_return=results["Manual Results"]["Out Sample"]["Cumulative Return"],
              manual_strategy_in_sample_port_val=results["Manual Results"]["In Sample"]["PortVal"],
              manual_strategy_out_sample_port_val=results["Manual Results"]["Out Sample"]["PortVal"],
              in_sample_benchmark_portval=results["Benchmark Results"]["In Sample"]["PortVal"],
              out_sample_benchmark_portval=results["Benchmark Results"]["Out Sample"]["PortVal"])
    plot_results(results=results, plot="Manual Strategy", sym=sym)
    plot_results(results=results, plot="Experiment One", sym=sym)
    plot_results(results=results, plot="Strategy Learner", sym=sym)
    plot_results(results=results, plot="Experiment Two", sym=sym)
    print()
