"""MC2-P1: Market simulator.

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

import numpy as np
import pandas as pd

from util import get_data


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
    holdings_df = trades_df.copy()
    holdings_df.iloc[0, :] = trades_df.iloc[0, :]
    holdings_df["Cash"].iloc[0] += start_val
    holdings_df = holdings_df.cumsum(axis=0)
    return holdings_df


def get_prices(symbols, start_date, end_date):
    temp_prices = get_data(symbols, pd.date_range(start_date, end_date))
    temp_prices.fillna(method='ffill', inplace=True)
    temp_prices.fillna(method='bfill', inplace=True)
    return temp_prices


def populate_trades_df_from_orders_df(trades_df, orders_df, prices_df, commission, impact):
    penalty_df = trades_df.iloc[:, :-1].copy()
    penalty_df[:] = 0
    commission_df = penalty_df.copy()
    for idx, row in orders_df.iterrows():
        if row["Order"] == "BUY":
            trades_df.loc[idx][row["Symbol"]] += row["Shares"]
        elif row["Order"] == "SELL":
            trades_df.loc[idx][row["Symbol"]] += row["Shares"] * -1
        else:
            continue
        commission_df.loc[idx][row["Symbol"]] += commission
        penalty_df.loc[idx][row["Symbol"]] += ((row["Shares"] * prices_df.loc[idx][row["Symbol"]]) * impact)

    temp_DF = (trades_df * prices_df) * -1
    total_penalty = (commission_df + penalty_df).sum(axis=1)
    temp_DF["Cash"] -= total_penalty
    trades_df["Cash"] = temp_DF.sum(axis=1)
    return trades_df





def get_values(holdings, prices):
    return holdings * prices


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


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
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
    orders_DF = pd.read_csv(orders_file, index_col=["Date"], parse_dates=True, na_values=['nan'], header=0,
                            usecols=["Date", "Symbol", "Order", "Shares"])
    orders_DF.sort_index(inplace=True)
    dates = orders_DF.index.values
    dates_start = dates[0]
    dates_end = dates[-1]
    symbols = np.unique(orders_DF["Symbol"])

    # Get Prices Dataframe
    prices_DF = get_prices(symbols=symbols, start_date=dates_start, end_date=dates_end)

    SPY_DF = prices_DF["SPY"]
    prices_DF = prices_DF[symbols]  # remove SPY
    prices_DF["Cash"] = np.ones(shape=(prices_DF.shape[0]))

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
    return portvals


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    # of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    temp_order_df = pd.read_csv(of, index_col=["Date"], parse_dates=True, na_values=['nan'])
    temp_order_df.sort_index(inplace=True)
    temp_dates = temp_order_df.index.values
    temp_dates_start = temp_dates[0]
    temp_dates_end = temp_dates[-1]
    symbols = np.unique(temp_order_df["Symbol"])
    symbols = np.append(symbols, ["$SPX"])
    temp_prices_df = get_prices(symbols=symbols, start_date=temp_dates_start, end_date=temp_dates_end)

    SPY_DF = temp_prices_df["$SPX"]

    start_date = temp_dates_start
    end_date = temp_dates_end

    temp_daily_return = get_daily_returns(portvals)
    cum_ret = get_cumulative_returns(portvals)
    avg_daily_ret = get_average_daily_returns(temp_daily_return)
    std_daily_ret = get_std_daily_returns(temp_daily_return)
    sharpe_ratio = get_sharpe_ratio(temp_daily_return)

    temp_SPY_daily_return = get_daily_returns(SPY_DF)
    cum_ret_SPY = get_cumulative_returns(SPY_DF)
    avg_daily_ret_SPY = get_average_daily_returns(temp_SPY_daily_return)
    std_daily_ret_SPY = get_std_daily_returns(temp_SPY_daily_return)
    sharpe_ratio_SPY = get_sharpe_ratio(temp_SPY_daily_return)

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
