"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import indicators as ind
import marketsimcode as mkt
import util as ut

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


def compute_portvals(orders_df, start_date, end_date, startval, market_impact=0.0, commission_cost=0.0, ):
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
        val += commission_cost + (pd.np.abs(shares) * price * market_impact)
        if (date < prices.index.min()) or (date > prices.index.max()):
            continue
        trades[symbol][date] += shares
        cash[date] -= val
    trades["_CASH"] = cash
    holdings = trades.cumsum()
    df_portvals = (prices * holdings).sum(axis=1)
    return df_portvals


def populate_indicators(all_data_df, window=9, vortex_window=14, only_SPY=False, **kwargs):
    if only_SPY:
        rsi_df = mkt.get_relative_strength_indicator(only_SPY=True, **kwargs)
    else:
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


def min_max_scale(df):
    return (df - df.mean()) / df.std()


"""
    RSI_SELL = 75
    RSI_BUY = 57
    MACD_SELL = 0.77 
    MACD_BUY = 0.12
    BBP_SELL = 0.63
    BBP_BUY = 0.39
"""


def determine_y(data_df, symbol, n_day_return=1, macd_buy=0.12, macd_sell=0.77, rsi_buy=57, rsi_sell=75,
                bbp_buy=0.39, bbp_sell=0.63, impact=0.005):
    data_df["Adj_Close_PCT"] = data_df["Adj Close"].pct_change(periods=n_day_return)
    data_df["Adj_Close_CumSum"] = data_df["Adj_Close_PCT"].cumsum()
    temp_df = data_df.loc[data_df["RSI"] != 0, :].copy()
    temp_df[f"{n_day_return}_Day_Return"] = temp_df["Adj_Close_PCT"].shift(-n_day_return)
    temp_df["Order"] = 0.0
    temp_df["RSI_Signal"] = 0.0
    temp_df["BBP_Signal"] = 0.0
    temp_df["MACD_Signal"] = 0.0
    temp_df["OBV_Signal"] = 0.0
    temp_df["VORTEX_Signal"] = 0.0
    temp_df["Adj_Close_Copy"] = temp_df["Adj Close"]
    temp_df.loc[temp_df["MACD"] > (macd_sell * (1 + impact)), "MACD_Signal"] = -1.0
    temp_df.loc[temp_df["MACD"] < (macd_buy * (1 - impact)), "MACD_Signal"] = 1.0
    temp_df.loc[temp_df["RSI"] > (rsi_sell * (1 + impact)), "RSI_Signal"] = -1.0
    temp_df.loc[temp_df["RSI"] < (rsi_buy * (1 - impact)), "RSI_Signal"] = 1.0
    temp_df.loc[temp_df["BBP"] > (bbp_sell * (1 + impact)), "BBP_Signal"] = -1.0
    temp_df.loc[temp_df["BBP"] < (bbp_buy * (1 - impact)), "BBP_Signal"] = 1.0
    # Test_DF = pd.DataFrame(index=temp_df.index, columns=["Previous Previous Order",
    #                                                      "Previous Order", "Order",
    #                                                      "Next Order", "Next Next Order",
    #                                                      "Adj Close", "RSI", "BBP",
    #                                                      "MACD"])
    # Test_DF["Adj Close"] = temp_df["Adj Close"]
    # Test_DF["RSI"] = temp_df["RSI"]
    # Test_DF["BBP"] = temp_df["BBP"]
    # Test_DF["MACD"] = temp_df["MACD"]
    # a = temp_df["Order"]
    # a[((Test_DF["RSI"] > (rsi_sell * (1 + impact))) & (Test_DF["BBP"] > (bbp_sell * (1 + impact)))) |
    #   ((Test_DF["MACD"] > (macd_sell * (1 + impact))) & (Test_DF["BBP"] > (bbp_sell * (1 + impact))))] = -1.0
    #
    # a[((Test_DF["RSI"] < (rsi_buy * (1 - impact))) & (Test_DF["BBP"] < (bbp_buy * (1 - impact)))) |
    #   ((Test_DF["MACD"] < (macd_buy * (1 - impact))) & (Test_DF["BBP"] < (bbp_buy * (1 - impact))))] = 1.0
    # Test_DF["Order"] = a
    # Test_DF["Next Order"] = a.shift(-1)
    # Test_DF["Previous Order"] = a.shift(1)
    # Test_DF["Next Next Order"] = a.shift(-2)
    # Test_DF["Previous Previous Order"] = a.shift(2)
    # zero_idx = Test_DF.loc[Test_DF["Order"] == 0]
    # Test_DF.loc[(Test_DF["Order"] == 0) &
    #             (Test_DF["Previous Previous Order"] == 1) &
    #             (Test_DF["Next Next Order"] == 1), "Order"] = -1
    # Test_DF.loc[(Test_DF["Order"] == 0) &
    #             (Test_DF["Previous Previous Order"] == -1) &
    #             (Test_DF["Next Next Order"] == -1), "Order"] = 1

    # temp_df.loc[temp_df["Vortex"] > vortex_sell, "VORTEX_Signal"] = -1.0
    # temp_df.loc[temp_df["Vortex"] < vortex_buy, "VORTEX_Signal"] = 1.0

    tmp = temp_df["RSI_Signal"] + temp_df["BBP_Signal"] + temp_df["MACD_Signal"]
    tmp[tmp <= -2] = -1
    tmp[tmp >= 2] = 1

    tmp.fillna(method="ffill", inplace=True)
    pt, ct = np.unique(temp_df, return_counts=True)
    if 0 in pt:
        idx = np.argwhere(pt == 0)[0]
        zero_ct = ct[idx]
        if zero_ct > np.floor(temp_df.shape[0] * 0.99):
            temp_df.loc[temp_df.index.values[0], "Order"] = 1

    if not ~np.isnan(pt).any():
        tmp.loc[temp_df["Adj Close"].first_valid_index()] = 1
    temp_df["Order"] = tmp
    trade_df = pd.DataFrame(index=data_df.index, columns=[symbol])
    trade_df[symbol] = tmp
    return temp_df, trade_df


def populate_training_data(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), only_SPY=False):
    kwargs = {"symbol": symbol, "sd": sd, "ed": ed, "lookback": 14, "only_SPY": only_SPY}
    all_data_df = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
    all_data_df.dropna(inplace=True)
    if not all_data_df.any().any():
        return False
    all_data_with_indicators = populate_indicators(all_data_df, **kwargs)
    return all_data_with_indicators


def convert_trade_df(df):
    temp_df = df.diff() * 1000
    temp_df.loc[df.first_valid_index()] = df.loc[df.first_valid_index()] * 1000
    return temp_df


# noinspection PyUnusedLocal
def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
               market_impact=0.005, commission_cost=9.95):
    results = populate_training_data(symbol=symbol, sd=sd, ed=ed)
    data_df, trades = determine_y(results, symbol=symbol)
    return convert_trade_df(trades)
