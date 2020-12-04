"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""
import datetime as dt
import time

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
        val += commission_cost + (pd.np.abs(shares) * price * market_impact)
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


def test_strategy_learner(symbols=["JPM"], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    test_sd = dt.datetime(2010, 1, 1)
    test_ed = dt.datetime(2011, 12, 31)

    for sym in symbols:
        print(f"Starting on {sym}")
        learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
        add_evidence_start_time = time.perf_counter()
        keep_going = learner.add_evidence(symbol=sym, sd=sd, ed=ed, sv=100000)
        add_evidence_end_time = time.perf_counter()
        add_evidence_elapsed_time = add_evidence_end_time - add_evidence_start_time
        if not keep_going:
            print(f"Something wrong with obtaining data of symbol: {sym}")
            continue
        # IN SAMPLE
        in_sample_test_policy_start_time = time.perf_counter()
        in_sample_temp_df_trades = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
        in_sample_test_policy_end_time = time.perf_counter()
        in_sample_test_policy_elapsed_time = in_sample_test_policy_end_time - in_sample_test_policy_start_time
        in_sample_temp_df_trades2 = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
        if not verify_returned_dataframe_correct(in_sample_temp_df_trades, symbol=sym, sd=sd, ed=ed):
            raise ValueError
        in_sample_price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
        in_sample_benchmark_portval = learner.compute_benchmark(symbol=sym, sd=sd, ed=ed, sv=sv)
        in_sample_price_df.drop("SPY", inplace=True, axis=1)
        in_sample_df_trades = learner.convert_new_trade_back_to_old_trade_frame(in_sample_temp_df_trades)
        in_sample_port_val = learner.compute_portvals(orders_df=in_sample_df_trades, start_date=sd, end_date=ed,
                                                      startval=sv, market_impact=0.005, commission_cost=9.95)

        in_sample_price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
        in_sample_benchmark_portval = compute_benchmark(symbol=sym, sd=sd, ed=ed, sv=sv, impact=0.00,
                                                        commission=9.95, max_holdings=1000)
        in_sample_benchmark_port_val_normalized = in_sample_benchmark_portval / in_sample_benchmark_portval[0]
        in_sample_benchmark_cumulative_return = np.round((in_sample_benchmark_port_val_normalized[-1] /
                                                          in_sample_benchmark_port_val_normalized[0]) - 1.0, 4)

        in_sample_cumulative_return = np.round(
            (in_sample_port_val[-1] / in_sample_port_val[0]) - 1.0, 5)
        in_sample_benchmark_return = np.round(in_sample_benchmark_cumulative_return, 5)

        print(f"In Sample Cumulative Return: {in_sample_cumulative_return} \t"
              f" Final Port Val: {in_sample_port_val[-1]}\nBenchmark Return: {in_sample_benchmark_return}")
        buy_long_color = "tab:blue"
        sell_short_color = "black"
        benchmark_color = "tab:green"
        strategy_color = "tab:orange"
        manual_color = "tab:red"

        sample_type = "IN"
        plt.close("all")
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        norm_port = in_sample_port_val / in_sample_port_val[0]
        ax0.plot(norm_port,
                 label=f"Strategy Learner - Return:{in_sample_cumulative_return:.3f}",
                 color=manual_color, linewidth=3)

        ax0.plot(in_sample_benchmark_portval / in_sample_benchmark_portval[0],
                 label=f"Benchmark - Return:{in_sample_benchmark_return:.3f}",
                 color=benchmark_color, linewidth=3)

        long = in_sample_temp_df_trades.loc[in_sample_temp_df_trades[f"{sym}"] == 2000.0]
        short = in_sample_temp_df_trades.loc[in_sample_temp_df_trades[f"{sym}"] == -2000.0]

        for b in long.index.values:
            ax0.axvline(b, color=buy_long_color, alpha=0.9, linestyle="--", linewidth=1)
        for s in short.index.values:
            ax0.axvline(s, color=sell_short_color, alpha=0.9, linestyle="--", linewidth=1)

        normalized_adj = in_sample_price_df[sym] / in_sample_price_df[sym][0]
        ax0.plot(normalized_adj, label=f"Adjusted Close", linewidth=1, color="tab:orange", linestyle="--")
        ax1.plot(in_sample_temp_df_trades[sym].cumsum(), label=f"Holdings", linewidth=2, color="tab:olive")

        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")

        ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
        ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
        ax1.set_xlabel("Date", fontsize=20, weight='heavy')
        ax1.set_ylabel("Total Holdings", fontsize=20, weight='heavy')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

        ax0.set_title(f"{sym} - {sample_type} Sample\nStrategy Learner vs Benchmark", fontsize=20,
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

        plt.savefig(f"Chart_One_{sample_type}_Sample_Strategy_Learner_{sym}.png")
        plt.close("all")
        print(f"Finished on IN-Sample {sym}")

        # OUT OF SAMPLE
        sample_type = "OUT"
        oos_test_policy_start_time = time.perf_counter()
        oos_temp_df_trades = learner.testPolicy(symbol=sym, sd=test_sd, ed=test_ed, sv=sv)
        oos_test_policy_end_time = time.perf_counter()
        oos_test_policy_elapsed_time = oos_test_policy_end_time - oos_test_policy_start_time
        if not verify_returned_dataframe_correct(oos_temp_df_trades, symbol=sym, sd=test_sd, ed=test_ed):
            raise ValueError
        oos_price_df = mkt.get_prices(symbols=[sym], start_date=test_sd, end_date=test_ed)
        oos_benchmark_portval = learner.compute_benchmark(symbol=sym, sd=test_sd, ed=test_ed, sv=sv)
        oos_price_df.drop("SPY", inplace=True, axis=1)
        oos_df_trades = learner.convert_new_trade_back_to_old_trade_frame(oos_temp_df_trades)
        oos_port_val = learner.compute_portvals(orders_df=oos_df_trades, start_date=test_sd, end_date=test_ed,
                                                      startval=sv, market_impact=0.005, commission_cost=9.95)
        oos_cumulative_return = np.round((oos_port_val[-1] / oos_port_val[0]) - 1.0, 5)
        oos_benchmark_return = np.round((oos_benchmark_portval[-1] / oos_benchmark_portval[0]) - 1.0, 5)
        print(f"OOS Cumulative Return: {oos_cumulative_return} \t"
              f" Final Port Val: {oos_port_val[-1]}\nBenchmark Return: {oos_benchmark_return}")

        plt.close("all")
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        oos_norm_port = oos_port_val / oos_port_val[0]
        ax0.plot(oos_norm_port,
                 label=f"Strategy Learner - Return:{oos_cumulative_return:.3f}",
                 color=manual_color, linewidth=3)

        ax0.plot(oos_benchmark_portval / oos_benchmark_portval[0],
                 label=f"Benchmark - Return:{oos_benchmark_return:.3f}",
                 color=benchmark_color, linewidth=3)

        long = oos_temp_df_trades.loc[oos_temp_df_trades[f"{sym}"] == 2000.0]
        short = oos_temp_df_trades.loc[oos_temp_df_trades[f"{sym}"] == -2000.0]

        for b in long.index.values:
            ax0.axvline(b, color=buy_long_color, alpha=0.9, linestyle="--", linewidth=1)
        for s in short.index.values:
            ax0.axvline(s, color=sell_short_color, alpha=0.9, linestyle="--", linewidth=1)

        oos_normalized_adj = oos_price_df[sym] / oos_price_df[sym][0]
        ax0.plot(oos_normalized_adj, label=f"Adjusted Close", linewidth=1, color="tab:orange", linestyle="--")
        ax1.plot(oos_temp_df_trades[sym].cumsum(), label=f"Holdings", linewidth=2, color="tab:olive")

        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")

        ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
        ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
        ax1.set_xlabel("Date", fontsize=20, weight='heavy')
        ax1.set_ylabel("Total Holdings", fontsize=20, weight='heavy')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

        ax0.set_title(f"{sym} - {sample_type} Sample\nStrategy Learner vs Benchmark", fontsize=20,
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

        plt.savefig(f"Chart_One_{sample_type}_Sample_Strategy_Learner_{sym}.png")
        plt.close("all")
        print(f"Finished on {sample_type}-Sample {sym}")

        check_autograde_requirements(symbol=sym, add_evidence_time=add_evidence_elapsed_time,
                                     in_sample_time=in_sample_test_policy_elapsed_time,
                                     oos_time=oos_test_policy_elapsed_time,
                                     results_check=[in_sample_temp_df_trades, in_sample_temp_df_trades2],
                                     in_sample_return=in_sample_cumulative_return,
                                     in_sample_benchmark_return=in_sample_benchmark_return,
                                     oos_return=oos_cumulative_return,
                                     oos_benchmark_return=oos_benchmark_return)

    return 0


def run_part1(sym, in_sample_price_df, out_sample_price_df,
              in_sample_benchmark_port_val_normalized, out_sample_benchmark_port_val_normalized,
              in_sample_benchmark_cumulative_return, out_sample_benchmark_cumulative_return,
              in_sample_manual_strategy_normalized, out_sample_manual_strategy_normalized,
              in_sample_manual_cumulative_return, out_sample_manual_cumulative_return,
              manual_strategy_in_sample_trades, manual_strategy_out_sample_trades,
              manual_strategy_in_sample_port_val, manual_strategy_out_sample_port_val,
              in_sample_benchmark_portval, out_sample_benchmark_portval):
    # region Manual Strategy Section
    plt.close("all")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # Chart One - In Sample
    get_chart_one(sym=sym, benchmark_port_val=in_sample_benchmark_port_val_normalized,
                  benchmark_cumulative_return=in_sample_benchmark_cumulative_return,
                  manual_strategy_port_val=in_sample_manual_strategy_normalized,
                  manual_strategy_cumulative_return=in_sample_manual_cumulative_return,
                  manual_trades=manual_strategy_in_sample_trades, sample_type="IN",
                  adj_close=in_sample_price_df[sym], ax0=ax0, ax1=ax1)
    plt.savefig("JPM_In_Sample_Manual_Strategy_Vs_Benchmark.png")

    plt.close("all")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # Chart One - Out Sample
    get_chart_one(sym=sym, benchmark_port_val=out_sample_benchmark_port_val_normalized,
                  benchmark_cumulative_return=out_sample_benchmark_cumulative_return,
                  manual_strategy_port_val=out_sample_manual_strategy_normalized,
                  manual_strategy_cumulative_return=out_sample_manual_cumulative_return,
                  manual_trades=manual_strategy_out_sample_trades, sample_type="OUT",
                  adj_close=out_sample_price_df[sym], ax0=ax0, ax1=ax1)

    plt.savefig("JPM_Out_Sample_Manual_Strategy_Vs_Benchmark.png")
    #           Get the In and Out Sample data for Manual Strategy and Benchmark
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


def run_project(sym="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                out_sample_sd=dt.datetime(2010, 1, 1), out_sample_ed=dt.datetime(2011, 12, 31),
                sv=100000, commission=9.95, impact=0.005):
    buy_long_color = "tab:blue"
    sell_short_color = "black"
    benchmark_color = "tab:green"
    strategy_color = "tab:orange"
    manual_color = "tab:red"

    if sym != "JPM":
        raise ValueError
    if sd != dt.datetime(2008, 1, 1):
        raise ValueError
    if ed != dt.datetime(2009, 12, 31):
        raise ValueError
    if out_sample_sd != dt.datetime(2010, 1, 1):
        raise ValueError
    if out_sample_ed != dt.datetime(2011, 12, 31):
        raise ValueError
    if sv != 100000:
        raise ValueError
    if commission != 9.95:
        raise ValueError
    if impact != 0.005:
        raise ValueError

    manual_strategy_in_sample_trades = ms.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv, commission_cost=commission,
                                                     market_impact=impact)
    manual_strategy_in_sample_trades_converted = convert_new_trade_back_to_old_trade_frame(
        manual_strategy_in_sample_trades)
    manual_strategy_in_sample_port_val = compute_portvals(orders_df=manual_strategy_in_sample_trades_converted,
                                                          start_date=sd, end_date=ed, startval=sv,
                                                          market_impact=impact, commission_cost=commission)
    in_sample_price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
    in_sample_benchmark_portval = compute_benchmark(symbol=sym, sd=sd, ed=ed, sv=sv, impact=impact,
                                                    commission=commission, max_holdings=1000)
    in_sample_price_df.drop("SPY", inplace=True, axis=1)
    in_sample_benchmark_port_val_normalized = in_sample_benchmark_portval / in_sample_benchmark_portval[0]
    in_sample_benchmark_cumulative_return = np.round((in_sample_benchmark_port_val_normalized[-1] /
                                                      in_sample_benchmark_port_val_normalized[0]) - 1.0, 4)
    in_sample_manual_strategy_normalized = manual_strategy_in_sample_port_val / manual_strategy_in_sample_port_val[
        0]
    in_sample_manual_cumulative_return = np.round(
        (in_sample_manual_strategy_normalized[-1] / in_sample_manual_strategy_normalized[0]) - 1.0, 4)

    manual_strategy_out_sample_trades = ms.testPolicy(symbol=sym, sd=out_sample_sd, ed=out_sample_ed, sv=sv,
                                                      commission_cost=commission, market_impact=impact)
    manual_strategy_out_sample_trades_converted = convert_new_trade_back_to_old_trade_frame(
        manual_strategy_out_sample_trades)
    manual_strategy_out_sample_port_val = compute_portvals(orders_df=manual_strategy_out_sample_trades_converted,
                                                           start_date=out_sample_sd, end_date=out_sample_ed,
                                                           startval=sv, market_impact=impact,
                                                           commission_cost=commission)
    out_sample_price_df = mkt.get_prices(symbols=[sym], start_date=out_sample_sd, end_date=out_sample_ed)
    out_sample_benchmark_portval = compute_benchmark(symbol=sym, sd=out_sample_sd, ed=out_sample_ed, sv=sv,
                                                     impact=impact, commission=commission, max_holdings=1000)
    out_sample_price_df.drop("SPY", inplace=True, axis=1)

    out_sample_benchmark_port_val_normalized = out_sample_benchmark_portval / out_sample_benchmark_portval[0]
    out_sample_benchmark_cumulative_return = np.round((out_sample_benchmark_port_val_normalized[-1] /
                                                       out_sample_benchmark_port_val_normalized[0]) - 1.0, 4)

    out_sample_manual_strategy_normalized = manual_strategy_out_sample_port_val / \
                                            manual_strategy_out_sample_port_val[0]
    out_sample_manual_cumulative_return = np.round(
        (out_sample_manual_strategy_normalized[-1] / out_sample_manual_strategy_normalized[0]) - 1.0, 4)

    # Initialize Learner
    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    # Training
    add_evidence_start_time = time.perf_counter()
    keep_going = learner.add_evidence(symbol=sym, sd=sd, ed=ed, sv=100000)
    add_evidence_end_time = time.perf_counter()
    add_evidence_elapsed_time = add_evidence_end_time - add_evidence_start_time
    # print(f"Training Finished, Elapsed Time: {add_evidence_elapsed_time:.4f}s")
    if not keep_going:
        print(f"Something wrong with obtaining data of symbol: {sym}")
        exit()

    # In Sample Strategy Results
    in_sample_strategy_start_time = time.perf_counter()
    in_sample_strategy_df_trades = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
    in_sample_strategy_end_time = time.perf_counter()
    in_sample_strategy_elapsed_time = in_sample_strategy_end_time - in_sample_strategy_start_time
    in_sample_strategy_df_trades_converted = learner.convert_new_trade_back_to_old_trade_frame(
        in_sample_strategy_df_trades)
    print(f"Learner Evaluation In Sample, Elapsed Time: {in_sample_strategy_elapsed_time:.4f}s")
    in_sample_strategy_port_val = mkt.compute_portvals(orders_DF=in_sample_strategy_df_trades_converted,
                                                       prices_DF=in_sample_price_df,
                                                       start_val=sv,
                                                       commission=commission, impact=impact)

    # Out Sample Strategy Results
    out_sample_strategy_start_time = time.perf_counter()
    out_sample_strategy_df_trades = learner.testPolicy(symbol=sym, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    out_sample_strategy_end_time = time.perf_counter()
    out_sample_strategy_elapsed_time = out_sample_strategy_end_time - out_sample_strategy_start_time
    print(f"Learner Evaluation Out Of Sample, Elapsed Time: {out_sample_strategy_elapsed_time:.4f}s")

    if not verify_returned_dataframe_correct(in_sample_strategy_df_trades, symbol=sym, sd=sd, ed=ed):
        raise ValueError
    in_sample_strategy_df_trades_converted = learner.convert_new_trade_back_to_old_trade_frame(
        in_sample_strategy_df_trades)
    in_sample_strategy_port_val = mkt.compute_portvals(orders_DF=in_sample_strategy_df_trades_converted,
                                                       prices_DF=in_sample_price_df,
                                                       start_val=sv,
                                                       commission=commission, impact=impact)

    in_sample_strategy_port_val_normalized = in_sample_strategy_port_val["PortVals"] / \
                                             in_sample_strategy_port_val["PortVals"][0]
    in_sample_strategy_cumulative_return = np.round(
        (in_sample_strategy_port_val["PortVals"][-1] / in_sample_strategy_port_val["PortVals"][0]) - 1.0, 5)

    out_sample_strategy_df_trades_converted = learner.convert_new_trade_back_to_old_trade_frame(
        out_sample_strategy_df_trades)
    out_sample_strategy_port_val = mkt.compute_portvals(orders_DF=out_sample_strategy_df_trades_converted,
                                                        prices_DF=out_sample_price_df,
                                                        start_val=sv,
                                                        commission=commission, impact=impact)

    out_sample_strategy_port_val_normalized = out_sample_strategy_port_val["PortVals"] / \
                                              out_sample_strategy_port_val["PortVals"][0]
    out_sample_strategy_cumulative_return = np.round(
        (out_sample_strategy_port_val["PortVals"][-1] / out_sample_strategy_port_val["PortVals"][0]) - 1.0, 5)

    run_part1(sym=sym, in_sample_price_df=in_sample_price_df, out_sample_price_df=out_sample_price_df,
              in_sample_benchmark_port_val_normalized=in_sample_benchmark_port_val_normalized,
              out_sample_benchmark_port_val_normalized=out_sample_benchmark_port_val_normalized,
              in_sample_benchmark_cumulative_return=in_sample_benchmark_cumulative_return,
              out_sample_benchmark_cumulative_return=out_sample_benchmark_cumulative_return,
              in_sample_manual_strategy_normalized=in_sample_manual_strategy_normalized,
              out_sample_manual_strategy_normalized=out_sample_manual_strategy_normalized,
              in_sample_manual_cumulative_return=in_sample_manual_cumulative_return,
              out_sample_manual_cumulative_return=out_sample_manual_cumulative_return,
              manual_strategy_in_sample_trades=manual_strategy_in_sample_trades,
              manual_strategy_out_sample_trades=manual_strategy_out_sample_trades,
              manual_strategy_in_sample_port_val=manual_strategy_in_sample_port_val,
              manual_strategy_out_sample_port_val=manual_strategy_out_sample_port_val,
              in_sample_benchmark_portval=in_sample_benchmark_portval,
              out_sample_benchmark_portval=out_sample_benchmark_portval)
    # exp2.
    return


if __name__ == '__main__':
    # run_project(sym="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # Test Manual Strategy

    # test_manual_strategy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    #
    # Test Strategy Learner

    # test_strategy_learner(symbols=["ML4T-220"],
    #                       sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    test_strategy_learner(symbols=["JPM", "AAPL", "SINE_FAST", "SINE_FAST_NOISE",
                                   "SINE_SLOW", "SINE_SLOW_NOISE", "ML4T-220", "UNH"],
                          sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # test_strategy_learner(symbols=["AAPL"], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print()
