""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing Indicators 		  	   		     		  		  		    	 		 		   		 		  

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

import os
import sys

import numpy as np
import pandas as pd

from util import get_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jadams334"  # Change this to your user ID


def calculate_exponential_moving_average(all_data, window):
    try:
        if len(all_data.shape) > 1:
            return all_data["Close"].ewm(span=window).mean()
        else:
            return all_data.ewm(span=window).mean()
    except Exception as calculate_exponential_moving_average_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'calculate_exponential_moving_average'", calculate_exponential_moving_average_exception)


def calculate_vortex(raw_dataframe, window):
    try:
        dataframe = raw_dataframe.copy()

        dataframe["VM Positive"] = np.abs(dataframe["High"] - dataframe["Low"].shift(1))
        dataframe["VM Negative"] = np.abs(dataframe["Low"] - dataframe["High"].shift(1))
        dataframe["VM-14 Positive"] = dataframe["VM Positive"].rolling(window=window).sum()
        dataframe["VM-14 Negative"] = dataframe["VM Negative"].rolling(window=window).sum()
        dataframe["TR"] = np.nanmax(((dataframe["High"] - dataframe["Low"]),
                                     (np.abs(dataframe["High"] - dataframe["Close"].shift(1))),
                                     (np.abs(dataframe["Low"] - dataframe["Close"].shift(1)))), axis=0)
        dataframe["TR-14"] = dataframe["TR"].rolling(window=window).sum()
        dataframe["VI-14 Positive"] = dataframe["VM-14 Positive"] / dataframe["TR-14"]
        dataframe["VI-14 Negative"] = dataframe["VM-14 Negative"] / dataframe["TR-14"]
        dataframe.fillna(method="ffill", inplace=True)
        dataframe.fillna(method='bfill', inplace=True)
        return dataframe
    except Exception as get_vortex_indicator_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_vortex_indicator'", get_vortex_indicator_exception)


def calculate_macd(raw_dataframe):
    try:
        dataframe = raw_dataframe.copy()
        dataframe["EMA_12_Day"] = calculate_exponential_moving_average(dataframe["Close"], window=12)
        dataframe["EMA_26_Day"] = calculate_exponential_moving_average(dataframe, window=26)

        dataframe["MACD Line"] = (dataframe["EMA_12_Day"] - dataframe["EMA_26_Day"])
        dataframe["Signal Line"] = calculate_exponential_moving_average(dataframe["MACD Line"], window=9)
        dataframe["MACD Hist"] = dataframe["MACD Line"] - dataframe["Signal Line"]
        dataframe.fillna(method='ffill', inplace=True)
        dataframe.fillna(method='bfill', inplace=True)
        return dataframe
    except Exception as get_macd_indicator_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_macd_indicator'", get_macd_indicator_exception)


def calculate_obv(raw_dataframe):
    try:
        # If the current closing price is above the prior closing price
        #    then current OBV = previous OBV + current volume
        # If the current closing price is below the prior closing price
        #    then current OBV = previous OBV - current volume
        # If the current closing price equals the prior closing price
        #    then current OBV = previous OBV
        dataframe = raw_dataframe.copy()
        dataframe["OBV"] = np.nan

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
        dataframe["Shifted Volume"] = dataframe["Volume"].shift(-1)
        dataframe.loc[a == 1, "OBV"] = dataframe.loc[a == 1, "Shifted Volume"] + dataframe.loc[a == 1, "Volume"]
        dataframe.loc[a == -1, "OBV"] = dataframe.loc[a == -1, "Shifted Volume"] - dataframe.loc[a == -1, "Volume"]
        dataframe.loc[a == 0, "OBV"] = dataframe.loc[a == 0, "Shifted Volume"]
        dataframe.fillna(method='ffill', inplace=True)
        dataframe.fillna(method='bfill', inplace=True)
        return dataframe
    except Exception as get_obv_indicator_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_obv_indicator'", get_obv_indicator_exception)


def calculate_rsi(prices_DF, daily_rets, lookback):
    try:
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
    except Exception as calculate_rsi_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'calculate_rsi'", calculate_rsi_exception)


def calculate_bollinger(raw_prices_DF, window):
    try:
        temp_prices_DF = raw_prices_DF.copy()

        temp_prices_DF["Normalized"] = temp_prices_DF["Adj Close"] / temp_prices_DF["Adj Close"][0]
        temp_prices_DF["STD"] = temp_prices_DF["Normalized"].std(axis=0)
        temp_prices_DF["SMA"] = get_simple_moving_average(temp_prices_DF["Normalized"], window=window)
        temp_prices_DF["Rolling_STD"] = get_simple_moving_std(temp_prices_DF["Normalized"], num_std=2, window=window)
        temp_prices_DF["Bollinger_Top"] = temp_prices_DF["SMA"] + temp_prices_DF["Rolling_STD"]
        temp_prices_DF["Bollinger_Bottom"] = temp_prices_DF["SMA"] - temp_prices_DF["Rolling_STD"]
        temp_prices_DF["Bollinger_Percent"] = (temp_prices_DF["Normalized"] - temp_prices_DF["Bollinger_Bottom"]) / \
                                              (temp_prices_DF["Bollinger_Top"] - temp_prices_DF["Bollinger_Bottom"])
        return temp_prices_DF
    except Exception as calculate_bollinger_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'calculate_bollinger'", calculate_bollinger_exception)


def get_simple_moving_average(price_df, window=10):
    try:
        sma = price_df.rolling(window=window, min_periods=window).mean()
        sma.fillna(method='ffill', inplace=True)
        sma.fillna(method='bfill', inplace=True)
        return sma
    except Exception as get_simple_moving_average_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_simple_moving_average'", get_simple_moving_average_exception)


def get_simple_moving_std(price_df, window=10, num_std=2):
    try:
        bb = price_df.rolling(window=window, min_periods=window).std()
        bb.fillna(method='ffill', inplace=True)
        bb.fillna(method='bfill', inplace=True)
        return bb * num_std
    except Exception as get_simple_moving_std_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_simple_moving_std'", get_simple_moving_std_exception)


def get_all_stock_data(symbol, sd, ed):
    try:
        data_DF = None
        columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        for col in columns:
            temp_data = get_data(symbols=[symbol], dates=pd.date_range(sd, ed), colname=col)
            if data_DF is None:
                data_DF = pd.DataFrame(index=temp_data.index, columns=columns,
                                       data=np.zeros(shape=(temp_data.shape[0], len(columns))))
            data_DF[col] = temp_data[symbol]
        return data_DF
    except Exception as get_all_stock_data_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_all_stock_data'", get_all_stock_data_exception)
