import numpy as np
import pandas as pd

from util import get_data


def author():
    return "jadams334"  # Change this to your user ID


def calculate_exponential_moving_average(all_data, window):
    if len(all_data.shape) > 1:
        return all_data["Close"].ewm(span=window).mean()
    else:
        return all_data.ewm(span=window).mean()


def calculate_vortex(raw_dataframe, window=14):
    dataframe = raw_dataframe.copy()

    dataframe["VM_Positive"] = np.abs(dataframe["High"] - dataframe["Low"].shift(1))
    dataframe["VM_Negative"] = np.abs(dataframe["Low"] - dataframe["High"].shift(1))
    dataframe[f"VM_{window}_Positive"] = dataframe["VM_Positive"].rolling(window=window).sum()
    dataframe[f"VM_{window}_Negative"] = dataframe["VM_Negative"].rolling(window=window).sum()
    dataframe["TR"] = np.nanmax(((dataframe["High"] - dataframe["Low"]),
                                 (np.abs(dataframe["High"] - dataframe["Close"].shift(1))),
                                 (np.abs(dataframe["Low"] - dataframe["Close"].shift(1)))), axis=0)
    dataframe[f"TR_{window}"] = dataframe["TR"].rolling(window=window).sum()
    dataframe[f"VI_{window}_Positive"] = dataframe[f"VM_{window}_Positive"] / dataframe[f"TR_{window}"]
    dataframe[f"VI_{window}_Negative"] = dataframe[f"VM_{window}_Negative"] / dataframe[f"TR_{window}"]
    # Vortex Signal
    dataframe["Vortex"] = dataframe[f"VI_{window}_Positive"] - dataframe[f"VI_{window}_Negative"]
    # dataframe.dropna(inplace=True)
    # dataframe.fillna(method="ffill", inplace=True)
    # dataframe.fillna(method='bfill', inplace=True)
    return dataframe


def calculate_macd(raw_dataframe, window=9):
    dataframe = raw_dataframe.copy()
    dataframe["EMA_12_Day"] = calculate_exponential_moving_average(dataframe["Close"], window=12)
    dataframe["EMA_26_Day"] = calculate_exponential_moving_average(dataframe, window=26)

    dataframe["MACD_Line"] = (dataframe["EMA_12_Day"] - dataframe["EMA_26_Day"])
    dataframe["Signal_Line"] = calculate_exponential_moving_average(dataframe["MACD_Line"], window=window)
    dataframe["MACD"] = dataframe["MACD_Line"] - dataframe["Signal_Line"]
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)
    dataframe.dropna(inplace=True)

    return dataframe


def calculate_obv(raw_dataframe, window=7):
    # If the current closing price is above the prior closing price
    #    then current OBV = previous OBV + current volume
    # If the current closing price is below the prior closing price
    #    then current OBV = previous OBV - current volume
    # If the current closing price equals the prior closing price
    #    then current OBV = previous OBV
    dataframe = raw_dataframe.copy()
    dataframe["OBV_Line"] = np.nan

    def my_func(temp_data):
        # Current_price = temp_data[1]
        # Previous_price = temp_data[0]
        if temp_data[1] > temp_data[0]:
            # Current closing price was greater than previous closing price
            return 1
        elif temp_data[1] < temp_data[0]:
            # Current closing price was less than previous closing price
            return -1
        elif temp_data[1] == temp_data[0]:
            return 0

    a = dataframe["Close"].rolling(window=2, min_periods=2).apply(lambda x: my_func(x), raw=False)
    dataframe["Shifted_Volume"] = dataframe["Volume"].shift(-1)
    dataframe.loc[a == 1, "OBV_Line"] = dataframe.loc[a == 1, "Shifted_Volume"] + dataframe.loc[a == 1, "Volume"]
    dataframe.loc[a == -1, "OBV_Line"] = dataframe.loc[a == -1, "Shifted_Volume"] - dataframe.loc[a == -1, "Volume"]
    dataframe.loc[a == 0, "OBV_Line"] = dataframe.loc[a == 0, "Shifted_Volume"]
    dataframe["OBV_PCT_CNG"] = dataframe["OBV_Line"].pct_change(periods=window)
    dataframe["OBV"] = dataframe["OBV_PCT_CNG"]
    # dataframe.loc[dataframe["OBV_PCT_CNG"] > 0, "OBV"] = 1
    # dataframe.loc[dataframe["OBV_PCT_CNG"] < 0, "OBV"] = -1
    # dataframe.fillna(method='ffill', inplace=True)
    # dataframe.fillna(method='bfill', inplace=True)
    # dataframe.dropna(inplace=True)
    return dataframe


def calculate_rsi(prices_DF, lookback):
    temp_prices_DF = prices_DF / prices_DF.iloc[0]
    daily_rets = temp_prices_DF.copy()
    daily_rets.values[1:, :] = temp_prices_DF.values[1:, :] - temp_prices_DF.values[:-1, :]
    daily_rets.values[0, :] = np.nan

    rsi = temp_prices_DF.copy()
    rsi[:] = 0
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

    up_gain = temp_prices_DF.copy()
    up_gain.iloc[:] = 0
    up_gain.values[lookback:, :] = up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]

    down_loss = temp_prices_DF.copy()
    down_loss.iloc[:, :] = 0
    down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]
    rs = (up_gain / lookback) / (down_loss / lookback)
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:lookback, :] = np.nan
    rsi[rsi == np.inf] = 100
    rsi.fillna(method="ffill", inplace=True)
    rsi.fillna(method='bfill', inplace=True)
    return rsi


def calculate_bollinger(raw_prices_DF, window):
    temp_prices_DF = raw_prices_DF.copy()
    temp_prices_DF["Normalized"] = temp_prices_DF["Adj Close"] / temp_prices_DF["Adj Close"][0]
    temp_prices_DF["STD"] = temp_prices_DF["Normalized"].std(axis=0)
    temp_prices_DF["SMA"] = get_simple_moving_average(temp_prices_DF["Normalized"], window=window)
    temp_prices_DF["Rolling_STD"] = get_simple_moving_std(temp_prices_DF["Normalized"], num_std=2, window=window)
    temp_prices_DF["Bollinger_Top"] = temp_prices_DF["SMA"] + temp_prices_DF["Rolling_STD"]
    temp_prices_DF["Bollinger_Bottom"] = temp_prices_DF["SMA"] - temp_prices_DF["Rolling_STD"]
    temp_prices_DF["BBP"] = (temp_prices_DF["Normalized"] - temp_prices_DF["Bollinger_Bottom"]) / \
                            (temp_prices_DF["Bollinger_Top"] - temp_prices_DF["Bollinger_Bottom"])

    # temp_prices_DF.dropna(inplace=True)
    return temp_prices_DF


def get_simple_moving_average(price_df, window=10):
    sma = price_df.rolling(window=window, min_periods=window).mean()
    sma.fillna(method='ffill', inplace=True)
    sma.fillna(method='bfill', inplace=True)
    return sma


def get_simple_moving_std(price_df, window=10, num_std=2):
    bb = price_df.rolling(window=window, min_periods=window).std()
    bb.fillna(method='ffill', inplace=True)
    bb.fillna(method='bfill', inplace=True)
    return bb * num_std


def get_all_stock_data(symbol, sd, ed):
    data_DF = None
    columns = ("Open", "High", "Low", "Close", "Volume", "Adj Close")
    for col in columns:
        temp_data = get_data(symbols=[symbol], dates=pd.date_range(sd, ed), colname=col)
        if data_DF is None:
            data_DF = pd.DataFrame(index=temp_data.index, columns=columns,
                                   data=np.zeros(shape=(temp_data.shape[0], len(columns))))
        data_DF[col] = temp_data[symbol]
    return data_DF
