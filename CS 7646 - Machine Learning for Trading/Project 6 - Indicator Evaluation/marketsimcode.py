"""MC2-P1: Market simulator - Improved.

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
from matplotlib import gridspec

import indicators as ind
from util import get_data

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()


def author():
    return 'jadams334'  # replace tb34 with your Georgia Tech username.


def get_present_value(value, num_years, interest_rate):
    # PV = FutureValue / (1 + InterestRate)^num_years
    return value / (1 + interest_rate) ** num_years


def get_intrinsic_value(value, discount_rate):
    if discount_rate > 1.0:
        discount_rate /= 100.0
    return value / discount_rate


def get_book_value(total_assets, liabilities):
    return total_assets - liabilities


def market_capitalization(shares_outstanding, price):
    return shares_outstanding * price


def get_holdings(trades_df, start_val):
    try:
        holdings_df = trades_df.copy()
        holdings_df.iloc[0, :] = trades_df.iloc[0, :]
        holdings_df["Cash"].iloc[0] += start_val
        holdings_df = holdings_df.cumsum(axis=0)
        return holdings_df
    except Exception as get_holdings_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_holdings'", get_holdings_exception)


def get_prices(symbols, start_date, end_date):
    temp_prices = get_data(symbols, pd.date_range(start_date, end_date))
    temp_prices["Cash"] = np.ones(shape=(temp_prices.shape[0]))
    temp_prices.fillna(method='ffill', inplace=True)
    temp_prices.fillna(method='bfill', inplace=True)
    return temp_prices


def populate_trades_df_from_orders_df(trades_df, orders_df, prices_df, commission, impact):
    try:
        penalty_df = trades_df.iloc[:, :-1].copy()
        penalty_df[:] = 0
        commission_df = penalty_df.copy()
        for idx, row in orders_df.iterrows():
            if row["Order"] == "BUY":
                trades_df.iloc[idx][row["Symbol"]] += row["Shares"]
            elif row["Order"] == "SELL":
                trades_df.iloc[idx][row["Symbol"]] += row["Shares"] * -1
            else:
                continue
            commission_df.iloc[idx][row["Symbol"]] += commission
            penalty_df.iloc[idx][row["Symbol"]] += ((row["Shares"] * prices_df.iloc[idx][row["Symbol"]]) * impact)

        temp_DF = (trades_df * prices_df) * -1
        total_penalty = (commission_df + penalty_df).sum(axis=1)
        temp_DF["Cash"] -= total_penalty
        trades_df["Cash"] = temp_DF.sum(axis=1)
        return trades_df
    except Exception as populate_trades_df_from_orders_df_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'populate_trades_df_from_orders_df'",
              populate_trades_df_from_orders_df_exception)


def get_values(holdings, prices):
    try:
        return holdings * prices
    except Exception as get_values_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_values'", get_values_exception)


def get_average_daily_returns(df):
    return df.mean()


def get_std_daily_returns(daily_returns_df):
    return daily_returns_df.std()


def get_daily_returns(port_value_df):
    # Parameter should be the portfolio_value_dataframe
    daily_returns = (port_value_df / port_value_df.shift(1)) - 1
    return daily_returns


def get_sharpe_ratio(daily_rets):
    return (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)


def get_cumulative_returns(portfolio_dataframe):
    # Cumulative return should be calculated using a portfolio
    return (portfolio_dataframe[-1] / portfolio_dataframe[0]) - 1


def compute_portvals(orders_DF, prices_DF, start_val=1000000, commission=9.95, impact=0.005):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months

    # STEPS
    #   - Read Orders files. Get start_date, end_date, and symbols
    try:
        orders_DF.sort_index(inplace=True)
        dates = orders_DF.index.values
        dates_start = dates[0]
        dates_end = dates[-1]
        symbols = np.unique(orders_DF["Symbol"])

        # Get Trades Dataframe
        trades_DF = prices_DF.copy()
        trades_DF.iloc[:, :] = 0
        trades_DF = populate_trades_df_from_orders_df(trades_df=trades_DF, orders_df=orders_DF, prices_df=prices_DF,
                                                      commission=commission, impact=impact)

        # Get Holdings Dataframe
        holdings_DF = get_holdings(trades_df=trades_DF, start_val=start_val)

        # Get Values Dataframe
        values_DF = get_values(holdings_DF, prices_DF)

        # Get Portfolio Values Dataframe
        portvals = values_DF.sum(axis=1)

        # rv = pd.DataFrame(index=portvals.index, data=portvals.values)
        results = {"PortVals": portvals, "Trades_DF": trades_DF, "Orders_DF": orders_DF, "Prices_DF": prices_DF,
                   "Holdings_DF": holdings_DF, "Values_DF": values_DF}
        return results
    except Exception as compute_portvals_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'compute_portvals'", compute_portvals_exception)


def get_bollinger_band(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), fill_between=True,
                       extra="Figure_bollinger", marker_size=6, window=14):
    try:
        plt.close("all")
        # Get Prices

        temp_prices_DF = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
        prices_DF = ind.calculate_bollinger(temp_prices_DF, window=window)

        prices_DF["Sell_Signal"] = np.nan
        prices_DF["Buy_Signal"] = np.nan
        a = prices_DF.loc[prices_DF["Normalized"] > prices_DF["Bollinger_Top"], "Sell_Signal"]
        prices_DF.loc[prices_DF["Normalized"] > prices_DF["Bollinger_Top"], "Sell_Signal"] = prices_DF["Normalized"]
        prices_DF.loc[prices_DF["Normalized"] < prices_DF["Bollinger_Bottom"], "Buy_Signal"] = prices_DF["Normalized"]
        prices_DF["Bollinger_Percent_Sell"] = np.nan
        prices_DF["Bollinger_Percent_Buy"] = np.nan
        prices_DF.loc[prices_DF["Bollinger_Percent"] > 1, "Bollinger_Percent_Sell"] = prices_DF["Normalized"]
        prices_DF.loc[prices_DF["Bollinger_Percent"] < 0, "Bollinger_Percent_Buy"] = prices_DF["Normalized"]

        plt.close("all")
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))

        # Customize the major grid
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        if fill_between:
            ax1.fill_between(prices_DF.index.values, prices_DF["Normalized"],
                             prices_DF["SMA"] + prices_DF["Rolling_STD"], alpha=0.1,
                             color="tab:red")

            ax1.fill_between(prices_DF.index.values, prices_DF["Normalized"],
                             prices_DF["SMA"] - prices_DF["Rolling_STD"], alpha=0.10, color='green')

        ax1.plot(prices_DF.index.values, prices_DF["Normalized"], '-', color="navy",
                 label=symbol, lw=1)
        ax1.plot(prices_DF.index.values, prices_DF["Bollinger_Top"], '--', color="tab:red",
                 label="Upper_Bollinger", lw=0.75)
        ax1.plot(prices_DF.index.values, prices_DF["Bollinger_Bottom"], '--', color="green",
                 label="Lower_Bollinger", lw=0.75)

        ax1.plot(prices_DF.index.values, prices_DF["Buy_Signal"], linestyle="",
                 marker="o", color="tab:red", markersize=3)
        ax1.plot(prices_DF.index.values, prices_DF["Sell_Signal"], linestyle="",
                 marker="o", color="green", markersize=3)
        ax1.plot(prices_DF.index.values, prices_DF["Bollinger_Percent_Buy"], linestyle="",
                 marker="$+$", color="tab:red", markersize=marker_size, label="Buy Signal")
        ax1.plot(prices_DF.index.values, prices_DF["Bollinger_Percent_Sell"], linestyle="",
                 marker="$—$", color="green", markersize=marker_size, label="Sell Signal")

        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        ax1.set_title(f"Bollinger Bands\n{symbol} Normalized", fontsize=15, weight='bold')
        ax1.set_xlabel("Trading Date", fontsize=15, weight='heavy')
        ax1.set_ylabel("Normalized Price", fontsize=15, weight='heavy')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{symbol}_Bollinger_Bands_{extra}.png")
        return
    except Exception as get_bollinger_band_charts_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_bollinger_band_charts'", get_bollinger_band_charts_exception)


def get_baseline(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
                 commission=0, impact=0):
    try:
        prices_DF = get_prices(symbols=[symbol], start_date=sd, end_date=ed)
        prices_DF = prices_DF[[symbol]]  # remove SPY
        prices_DF["Cash"] = np.ones(shape=(prices_DF.shape[0]))
        trades_DF = prices_DF.copy()
        trades_DF.iloc[:, :] = 0
        baseline_orders_df = pd.DataFrame(columns=["Date", "Symbol", "Order", "Shares"])
        baseline_orders_df.loc[0] = [prices_DF.index.values[0], symbol, "BUY", 1000]
        baseline_results = compute_portvals(orders_DF=baseline_orders_df, prices_DF=prices_DF,
                                            start_val=sv, commission=commission, impact=impact)
        return baseline_results
    except Exception as exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in ", exception)


def get_relative_strength_indicator(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                    fill_between=True, extra="Figure_rsi", marker_size=6, lookback=14,
                                    upper_lim=70, lower_lim=30):
    try:
        plt.close("all")
        # Get Prices
        temp_prices_DF = get_prices(symbols=[symbol], start_date=sd, end_date=ed)
        unmodified_prices = temp_prices_DF.copy()
        temp_prices_DF.drop(["SPY"], axis=1, inplace=True)
        temp_prices_DF = temp_prices_DF / temp_prices_DF.iloc[0]
        daily_rets = temp_prices_DF.copy()

        # Normalize Prices
        rsi = ind.calculate_rsi(temp_prices_DF, daily_rets, 14)
        cols = ["RSI", "SMA", "Price", "Normalized_Price", "Buy_Signal", "Sell_Signal"]
        rsi_DF = pd.DataFrame(columns=cols,
                              data=np.zeros(shape=(temp_prices_DF.shape[0], len(cols))), index=temp_prices_DF.index)
        rsi_DF["RSI"] = rsi[symbol]
        rsi_DF["SMA"] = ind.get_simple_moving_average(temp_prices_DF[symbol], window=lookback)
        rsi_DF["Price"] = unmodified_prices[symbol]
        rsi_DF["Normalized_Price"] = temp_prices_DF[symbol]
        buy_lim = upper_lim
        sell_lim = lower_lim
        rsi_DF.loc[rsi_DF["RSI"] >= buy_lim, "Buy_Signal"] = rsi_DF["RSI"] - buy_lim
        rsi_DF.loc[rsi_DF["RSI"] <= sell_lim, "Sell_Signal"] = sell_lim - rsi_DF["RSI"]
        plt.close("all")
        # Normalize all values
        rsi_DF["Normalized_RSI"] = (rsi_DF["RSI"] - rsi_DF["RSI"].min()) / \
                                   (rsi_DF["RSI"].max() - rsi_DF["RSI"].min())
        buy_idx = np.where(rsi_DF["Buy_Signal"] > 0)
        sell_idx = np.where(rsi_DF["Sell_Signal"] > 0)
        rsi_DF["Normalized_Buy_Signal"] = 0
        rsi_DF["Normalized_Sell_Signal"] = 0
        rsi_DF.loc[rsi_DF["Buy_Signal"] > 0,
                   "Normalized_Buy_Signal"] = (rsi_DF["Buy_Signal"].iloc[buy_idx] -
                                               rsi_DF["Buy_Signal"].iloc[buy_idx].min()) / \
                                              (rsi_DF["Buy_Signal"].iloc[buy_idx].max() -
                                               rsi_DF["Buy_Signal"].iloc[buy_idx].min())

        rsi_DF.loc[rsi_DF["Sell_Signal"] > 0,
                   "Normalized_Sell_Signal"] = (rsi_DF["Sell_Signal"].iloc[sell_idx] -
                                                rsi_DF["Sell_Signal"].iloc[sell_idx].min()) / \
                                               (rsi_DF["Sell_Signal"].iloc[sell_idx].max() -
                                                rsi_DF["Sell_Signal"].iloc[sell_idx].min())
        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.plot(rsi_DF["Normalized_Price"], label=symbol, color="navy")
        ax0.plot(rsi_DF["SMA"], label="SMA", color="darkorange")
        ax0.fill_between(rsi_DF.index.values, rsi_DF["Normalized_Price"], rsi_DF["SMA"],
                         where=rsi_DF["Normalized_Sell_Signal"] > 0, color="tab:red", alpha=0.5)
        ax0.fill_between(rsi_DF.index.values, rsi_DF["Normalized_Price"], rsi_DF["SMA"],
                         where=rsi_DF["Normalized_Buy_Signal"] > 0, color="tab:red", alpha=0.5)
        ax0.set_title(f"Relative Strength Index\n{symbol}", fontsize=15, weight='bold')
        ax0.set_ylabel("Price", fontsize=15, weight='heavy')
        ax0.xaxis.set_major_formatter(plt.NullFormatter())
        ax0.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.plot(rsi_DF["RSI"], color="navy")
        ax1.set_xlabel("Trading Date", fontsize=12, weight='heavy')
        ax1.set_ylim([int(np.round(rsi_DF["RSI"].min() * 0.95)), int(np.round(rsi_DF["RSI"].max() * 1.05))])
        ax1.axhline(y=upper_lim, color="tab:red", linestyle="--", linewidth=1, label="Overbought", alpha=0.5)
        ax1.axhline(y=lower_lim, color="tab:green", linestyle="--", linewidth=1, label="Oversold", alpha=0.5)
        ax1.set_ylabel("Relative Strength Index", fontsize=12, weight='heavy')
        ax1.legend(bbox_to_anchor=(1.1, 1.05), markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{symbol}_Relative_Strength_Index_{extra}.png")
        return
    except Exception as get_relative_strength_indicator_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_relative_strength_indicator'", get_relative_strength_indicator_exception)


def get_vortex_indicator(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), fill_between=True,
                         extra="Figure_vortex", marker_size=6, window=14):
    try:
        plt.style.use("ggplot")
        all_data = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
        all_data["SMA"] = ind.get_simple_moving_average(all_data["Close"], window=window)
        results = ind.calculate_vortex(all_data, window=window)
        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.plot(results["Close"], label=symbol, linewidth=1.15, color="navy")
        ax0.plot(results["SMA"], label="SMA", linestyle="--", linewidth=1.15, color="darkorange")
        ax0.fill_between(results.index.values, results["Close"], results["SMA"],
                         where=results["VI-14 Positive"] > 1, color="tab:green", alpha=0.3)
        ax0.fill_between(results.index.values, results["Close"], results["SMA"],
                         where=results["VI-14 Negative"] > 1, color="tab:red", alpha=0.3)
        ax0.set_title(f"Vortex Indicator\n{symbol}", fontsize=15, weight='bold')
        ax0.set_ylabel("Price", fontsize=15, weight='heavy')
        ax0.xaxis.set_major_formatter(plt.NullFormatter())
        ax0.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.plot(results["VI-14 Positive"], color="tab:green", alpha=0.8, label="+ VI-14", linewidth=1)
        ax1.plot(results["VI-14 Negative"], color="tab:red", alpha=0.8, label="— VI-14", linewidth=1)
        ax1.set_xlabel("Trading Date", fontsize=12, weight='heavy')
        ax1.set_ylim([0.4, 1.4])
        ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_ylabel("Vortex Indicator", fontsize=12, weight='heavy')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{symbol}_Vortex_Indicator_{extra}.png")
        return
    except Exception as get_vortex_indicator_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_vortex_indicator'", get_vortex_indicator_exception)


def get_macd_indicator(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), fill_between=True,
                       extra="Figure_macd", marker_size=6, window=14):
    try:
        # MACD Line is the 12-day EMA minus the 26-day EMA
        # Signal Line is the 9-day EMA of the MACD Line
        # MACD Histogram would be MACD Line - Signal Line
        plt.close("all")
        all_data = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
        all_data["SMA"] = ind.get_simple_moving_average(all_data["Close"], window=window)
        results = ind.calculate_macd(all_data)
        results["MACD Hist"].hist(bins=int(np.round(results.shape[0] / 2)))
        results["MACD Hist Positive"] = 0
        results["MACD Hist Negative"] = 0
        results.loc[results["MACD Hist"] >= 0, "MACD Hist Positive"] = results["MACD Hist"][results["MACD Hist"] >= 0]
        results.loc[results["MACD Hist"] < 0, "MACD Hist Negative"] = results["MACD Hist"][results["MACD Hist"] < 0]

        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.plot(results["Close"], label=symbol, linewidth=1.15, color="navy")
        ax0.plot(results["SMA"], label="SMA", linestyle="--", linewidth=1.15, color="darkorange")

        ax0.set_title(f"Moving Average Convergence/Divergence Indicator\n{symbol}", fontsize=15, weight='bold')
        ax0.set_ylabel("Price", fontsize=15, weight='heavy')
        ax0.xaxis.set_major_formatter(plt.NullFormatter())
        ax0.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.plot(results["MACD Line"], color="navy", label="MACD", linewidth=1)
        ax1.plot(results["Signal Line"], color="tab:red", label="Signal", linewidth=1)
        ax1.plot(results["MACD Hist"], color="darkorange", label="MACD Hist", linewidth=1)

        ax1.fill_between(results.index.values, results["MACD Hist"], 0,
                         where=results["MACD Hist"] > 0, color="tab:green", alpha=0.3)
        ax1.fill_between(results.index.values, results["MACD Hist"], 0,
                         where=results["MACD Hist"] < 0, color="tab:red", alpha=0.3)

        ax1.set_xlabel("Trading Date", fontsize=12, weight='heavy')
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_ylabel("On-Balance Volume", fontsize=12, weight='heavy')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{symbol}_MACD_Indicator_{extra}.png")
        return
    except Exception as get_macd_indicator_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_macd_indicator'", get_macd_indicator_exception)


def get_obv_indicator(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), fill_between=True,
                      extra="Figure_obv", marker_size=6, window=14):
    try:
        plt.close("all")
        all_data = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
        all_data["SMA"] = ind.get_simple_moving_average(all_data["Close"], window=window)
        results = ind.calculate_obv(all_data)
        results["Absolute OBV"] = np.abs(results["OBV"])
        results["Positive OBV"] = results["OBV"]
        results["Negative OBV"] = results["OBV"]

        results.loc[results["Positive OBV"] <= 0, "Positive OBV"] = 0
        results.loc[results["Negative OBV"] > 0, "Negative OBV"] = 0
        results["OBV Normalized"] = (results["OBV"] - results["OBV"].mean()) / results["OBV"].std()
        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.plot(results["Close"], label=symbol, linewidth=1.15, color="navy")
        ax0.plot(results["SMA"], label="SMA", linestyle="--", linewidth=1.15, color="darkorange")
        ax0.fill_between(results.index.values, results["Close"], results["SMA"],
                         where=results["OBV"] > 0, color="tab:green", alpha=0.3)
        ax0.fill_between(results.index.values, results["Close"], results["SMA"],
                         where=results["OBV"] < 0, color="tab:red", alpha=0.3)
        ax0.set_title(f"On-Balance Volume\n{symbol}", fontsize=15, weight='bold')
        ax0.set_ylabel("Price", fontsize=15, weight='heavy')
        ax0.xaxis.set_major_formatter(plt.NullFormatter())
        ax0.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.plot(results["OBV"], color="navy", label="On-Balance Volume", linewidth=1)
        ax1.set_xlabel("Trading Date", fontsize=12, weight='heavy')
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_ylabel("On-Balance Volume", fontsize=12, weight='heavy')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{symbol}_OnBalance_Volume_Indicator_{extra}.png")
        return
    except Exception as get_obv_indicator_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_obv_indicator'", get_obv_indicator_exception)


def get_theoretically_optimal_strategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                       fill_between=True,
                                       extra="Figure_obv", marker_size=6, window=14):
    try:
        plt.close("all")
        all_data = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
        results = ind.calculate_bollinger(ind.calculate_obv(all_data), window=window)
        results = ind.calculate_macd(ind.calculate_vortex(results, window=window))
        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 5])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.plot(results["Close"], label=symbol, linewidth=1.15, color="navy")
        ax0.plot(results["SMA"], label="SMA", linestyle="--", linewidth=1.15, color="darkorange")

        ax0.fill_between(results.index.values, results["Close"], results["SMA"],
                         where=results["OBV"] > 0, color="tab:green", alpha=0.3)
        ax0.fill_between(results.index.values, results["Close"], results["SMA"],
                         where=results["OBV"] < 0, color="tab:red", alpha=0.3)

        ax0.set_title(f"Theoretically Optimal Strategy\n{symbol}", fontsize=15, weight='bold')
        ax0.set_ylabel("Price", fontsize=15, weight='heavy')
        ax0.xaxis.set_major_formatter(plt.NullFormatter())
        ax0.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.plot(results["OBV"], color="navy", label="On-Balance Volume", linewidth=1)
        ax1.set_xlabel("Trading Date", fontsize=12, weight='heavy')
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_ylabel("On-Balance Volume", fontsize=12, weight='heavy')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{symbol}_Theoretically_Optimal_Strategy_{extra}.png")
    except Exception as get_theoretically_optimal_strategy_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_theoretically_optimal_strategy'", get_theoretically_optimal_strategy_exception)
