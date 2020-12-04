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
import pandas as pd
import numpy as np
import pickle

import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as mkt

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


def test_manual_strategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        return
    except Exception as test_manual_strategy_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'test_manual_strategy'", test_manual_strategy_exception)


def test_strategy_learner(symbols=["JPM"], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        test_sd = dt.datetime(2010, 1, 1)
        test_ed = dt.datetime(2011, 12, 31)
        use_out_of_sample = False
        for sym in symbols:
            print(f"Starting on {sym}")
            learner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0)
            learner.add_evidence(symbol=sym, sd=sd, ed=ed, sv=100000)
            if use_out_of_sample:
                temp_df_trades = learner.testPolicy(symbol=sym, sd=test_sd, ed=test_ed, sv=sv)
                if not verify_returned_dataframe_correct(temp_df_trades, symbol=sym, sd=test_sd, ed=test_ed):
                    raise ValueError
                price_df = mkt.get_prices(symbols=[sym], start_date=test_sd, end_date=test_ed)
                benchmark_portval = learner.compute_benchmark(symbol=sym, sd=test_sd, ed=test_ed, sv=sv)
            else:
                temp_df_trades = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
                if not verify_returned_dataframe_correct(temp_df_trades, symbol=sym, sd=sd, ed=ed):
                    raise ValueError
                price_df = mkt.get_prices(symbols=[sym], start_date=sd, end_date=ed)
                benchmark_portval = learner.compute_benchmark(symbol=sym, sd=sd, ed=ed, sv=sv)
            price_df.drop("SPY", inplace=True, axis=1)
            df_trades = learner.convert_new_trade_back_to_old_trade_frame(temp_df_trades)
            port_val = mkt.compute_portvals(orders_DF=df_trades, prices_DF=price_df, start_val=sv,
                                            commission=0.0, impact=0.0)
            cumulative_return = np.round((port_val["PortVals"][-1] / port_val["PortVals"][0]) - 1.0, 5)
            strategy_return = cumulative_return
            benchmark_return = np.round((benchmark_portval[-1] / benchmark_portval[0]) - 1.0, 5)
            print(f"Cumulative Return: {cumulative_return} \t"
                  f" Final Port Val: {port_val['PortVals'][-1]}\nBenchmark Return: {benchmark_return}")
            if strategy_return > benchmark_return:
                print("YAY!!! WE BEAT THE BENCHMARK ")

            plt.close("all")
            fig, ax1 = plt.subplots(figsize=(12, 8))

            ax1.set_title(f"{sym} vs Benchmark Performance", fontsize=15, weight='bold')
            ax1.plot((port_val["PortVals"] / port_val["PortVals"][0]), label=f"{sym} - Return:{strategy_return:.3f}",
                     color="Navy")
            ax1.plot((benchmark_portval / benchmark_portval[0]), label=f"Benchmark - Return:{benchmark_return:.3f}",
                     color="tab:green")
            ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            ax1.legend(loc="best", markerscale=1.1, frameon=True,
                       edgecolor="black", fancybox=True, shadow=True)
            plt.tight_layout()
            plt.savefig(f"TEST_{sym}.png")
            print(f"Finished on {sym}")
        return 0
    except Exception as test_strategy_learner_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'test_strategy_learner'", test_strategy_learner_exception)


if __name__ == '__main__':
    # data = pd.read_csv("Correlation.csv", index_col=0)
    # mx = data.max()
    # my = data.max(axis=1)

    # Test Manual Strategy
    # test_manual_strategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2008, 10, 25), sv=100000)

    # Test Strategy Learner
    # test_strategy_learner(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # test_strategy_learner(symbols=["JPM", "AAPL", "SINE_FAST", "SINE_FAST_NOISE",
    #                                "SINE_SLOW", "SINE_SLOW_NOISE", "ML4T-220",
    #                                "UNH"], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    test_strategy_learner(symbols=["JPM", "AAPL", "ML4T-220", "UNH"],
                          sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    test_strategy_learner(symbols=["AAPL"], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print()
