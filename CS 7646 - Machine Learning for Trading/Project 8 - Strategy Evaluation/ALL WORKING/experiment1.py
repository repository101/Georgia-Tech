"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""

import datetime as dt
import os
import sys
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
    try:
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
    except Exception as verify_returned_dataframe_correct_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'verify_returned_dataframe_correct'", verify_returned_dataframe_correct_exception)


def get_chart_one(sym, benchmark_port_val, benchmark_cumulative_return, manual_strategy_port_val,
                  manual_strategy_cumulative_return, manual_trades, adj_close, sample_type, plot_adj=False, ax0=None,
                  ax1=None):
    try:
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
                 color=manual_color, linewidth=2)

        ax0.plot(benchmark_port_val, label=f"Benchmark - Return:{benchmark_cumulative_return:.3f}",
                 color=benchmark_color, linewidth=2)

        buy = manual_trades.loc[manual_trades[f"{sym}"] > 0]
        sell = manual_trades.loc[manual_trades[f"{sym}"] < 0]

        for b in buy.index.values:
            ax0.axvline(b, color=buy_long_color, alpha=0.4, linestyle="--", linewidth=0.5)
        for s in sell.index.values:
            ax0.axvline(s, color=sell_short_color, alpha=0.4, linestyle="--", linewidth=0.5)
        # normalized_adj = adj_close / adj_close[0]
        ax1.plot(adj_close, label=f"Adjusted Close", linewidth=1, color="tab:olive")

        # # Let the horizontal ax0es labeling appear on top.
        # ax1.tick_params(top=False, bottom=True,
        #                 labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")

        ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
        ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
        ax1.set_xlabel("Date", fontsize=20, weight='heavy')
        ax1.set_ylabel("Stock Price", fontsize=20, weight='heavy')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

        ax0.set_title(f"{sym} - {sample_type} Sample\nManual Strategy vs Benchmark", fontsize=20,
                      weight='bold')

        # # Let the horizontal ax0es labeling appear on top.
        # ax0.tick_params(top=False, bottom=True,
        #                 labeltop=False, labelbottom=True, labelsize=15)

        # Rotate the tick labels and set their alignment.
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
    except Exception as get_chart_one_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_chart_one'", get_chart_one_exception)


def convert_new_trade_back_to_old_trade_frame(trade_df):
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


def compute_portvals(orders_df, start_date, end_date, startval, market_impact=0.0,
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


def compute_benchmark(sd, ed, sv, symbol, impact, commission, max_holdings):
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
        orders.loc[orders.index[-1], "Shares"] = -max_holdings
        orders.loc[orders.index[-1], "Order"] = "Sell"
        baseline_portvals = compute_portvals(orders_df=orders, start_date=sd, end_date=ed, startval=sv,
                                             market_impact=impact, commission_cost=commission)
        return baseline_portvals
    except Exception as compute_benchmark_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'compute_benchmark'", compute_benchmark_exception)


def get_experiment_one_chart(sym, benchmark_port_val, benchmark_cumulative_return,
                             strategy_learner_port_val, strategy_learner_cumulative_return, manual_strategy_port_val,
                             manual_strategy_cumulative_return, manual_trades, adj_close, sample_type, extra_name=""):
    try:
        buy_long_color = "tab:blue"
        sell_short_color = "black"
        benchmark_color = "tab:green"
        strategy_color = "tab:orange"
        manual_color = "tab:red"

        plt.close("all")
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.plot(strategy_learner_port_val,
                 label=f"Strategy Learner - Return:{strategy_learner_cumulative_return:.3f}",
                 color=strategy_color, linewidth=2)
        ax1.plot(manual_strategy_port_val,
                 label=f"Manual Strategy - Return:{manual_strategy_cumulative_return:.3f}",
                 color=manual_color, linewidth=2)
        ax1.plot(benchmark_port_val, label=f"Benchmark - Return:{benchmark_cumulative_return:.3f}",
                 color=benchmark_color, linewidth=2)

        normalized_adj = adj_close / adj_close[0]
        ax1.plot(normalized_adj, label=f"Adjusted Close", linewidth=2, color="tab:olive")
        ax1.set_title(f"{extra_name}\nChart One - {sym} - {sample_type} Sample\nManual Strategy vs Benchmark", fontsize=15,
                      weight='bold')

        # Let the horizontal ax1es labeling appear on top.
        ax1.tick_params(top=False, bottom=True,
                        labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")

        ax1.tick_params(which="minor", bottom=False, left=False)
        ax1.set_xlabel("Date", fontsize=15, weight='heavy')
        ax1.set_ylabel("Cumulative Return", fontsize=15, weight='heavy')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        plt.tight_layout()
        strategy_learner_cumulative_return = np.round(strategy_learner_cumulative_return, 4)
        plt.savefig(f"Experiment_One_Chart_{sym}.png")
        plt.close("all")
        print(f"Finished on {sample_type}-Sample {sym}")
        return
    except Exception as get_chart_one_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_chart_one'", get_chart_one_exception)


def run_experiment(sym="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
                   commission=9.95, impact=0.005, out_sample_sd=dt.datetime(2010, 1, 1),
                   out_sample_ed=dt.datetime(2011, 12, 31)):
    try:
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

        in_sample_price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
        in_sample_benchmark_portval = compute_benchmark(symbol=sym, sd=sd, ed=ed, sv=sv, impact=impact,
                                                        commission=commission, max_holdings=1000)
        in_sample_price_df.drop("SPY", inplace=True, axis=1)

        manual_strategy_in_sample_trades = ms.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv, commission_cost=commission,
                                                         market_impact=impact)
        manual_strategy_in_sample_trades_converted = convert_new_trade_back_to_old_trade_frame(
            manual_strategy_in_sample_trades)
        manual_strategy_in_sample_port_val = compute_portvals(orders_df=manual_strategy_in_sample_trades_converted,
                                                              start_date=sd, end_date=ed, startval=sv,
                                                              market_impact=impact, commission_cost=commission)
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

        manual_in_cumulative = in_sample_manual_cumulative_return
        manual_out_cumulative = out_sample_manual_cumulative_return
        bench_in_cumulative = in_sample_benchmark_cumulative_return
        bench_out_cumulative = out_sample_benchmark_cumulative_return

        manual_in_std = mkt.get_daily_returns(manual_strategy_in_sample_port_val).std()
        manual_out_std = mkt.get_daily_returns(manual_strategy_out_sample_port_val).std()
        bench_in_std = mkt.get_daily_returns(in_sample_benchmark_portval).std()
        bench_out_std = mkt.get_daily_returns(out_sample_benchmark_portval).std()

        manual_in_mean = mkt.get_daily_returns(manual_strategy_in_sample_port_val).mean()
        manual_out_mean = mkt.get_daily_returns(manual_strategy_out_sample_port_val).mean()
        bench_in_mean = mkt.get_daily_returns(in_sample_benchmark_portval).mean()
        bench_out_mean = mkt.get_daily_returns(out_sample_benchmark_portval).mean()
        print(f"\n\n{space_1*7}~~Manual Strategy Daily Returns Table~~")
        print()
        print(f"{space_1*6}.:In-Sample:.{space_1*7}.:Out-Sample:.")
        print()
        print(f"{space_1*4}Manual Strategy{space_1*2}Benchmark{space_1*3}Manual Strategy{space_1*2}Benchmark")

        print(f"Cumulative {space_1*3}{manual_in_cumulative:.2f} {space_1*3} {bench_in_cumulative:.2f}{space_1}"
              f"{space_1*4}{manual_out_cumulative:.2f} {space_1*3} {bench_out_cumulative:.2f}")
        print(f"STDEV {space_1*4}{manual_in_std:.2f} {space_1*3} {bench_in_std:.2f}{space_1}"
              f"{space_1*4}{manual_out_std:.2f} {space_1*3} {bench_out_std:.2f}")
        print(f"Mean {space_1*4}{manual_in_mean:.2f} {space_1*3} {bench_in_mean:.2f}{space_1}"
              f"{space_1*4}{manual_out_mean:.2f} {space_1*3} {bench_out_mean:.2f}")

        # endregion

        # region Strategy Learner Section
        # endregion

        # region Experiment One Section
        #       In Sample Experiment One
        get_experiment_one_chart(sym=sym, benchmark_port_val=in_sample_benchmark_port_val_normalized,
                                 benchmark_cumulative_return=in_sample_benchmark_cumulative_return,
                                 strategy_learner_port_val=in_sample_strategy_port_val_normalized,
                                 strategy_learner_cumulative_return=in_sample_strategy_cumulative_return,
                                 manual_strategy_port_val=in_sample_manual_strategy_normalized,
                                 manual_strategy_cumulative_return=in_sample_manual_cumulative_return,
                                 manual_trades=manual_strategy_in_sample_trades, sample_type="IN",
                                 adj_close=in_sample_price_df[sym], )

        #       Out Sample Experiment One
        get_experiment_one_chart(sym=sym, benchmark_port_val=out_sample_benchmark_port_val_normalized,
                                 benchmark_cumulative_return=out_sample_benchmark_cumulative_return,
                                 strategy_learner_port_val=out_sample_strategy_port_val_normalized,
                                 strategy_learner_cumulative_return=out_sample_strategy_cumulative_return,
                                 manual_strategy_port_val=out_sample_manual_strategy_normalized,
                                 manual_strategy_cumulative_return=out_sample_manual_cumulative_return,
                                 manual_trades=manual_strategy_out_sample_trades, sample_type="OUT",
                                 adj_close=out_sample_price_df[sym])

        print(f"In Sample Cumulative Return: {in_sample_strategy_cumulative_return} \t"
              f" Final Port Val: {in_sample_strategy_port_val['PortVals'][-1]}\n"
              f"Benchmark Return: {in_sample_benchmark_cumulative_return}")
        # endregion

        # region Experiment Two Section
        # endregion

        print(f"Finished on Experiment 1 - {sym}")
        return
    except Exception as run_experiment_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_experiment'", run_experiment_exception)
