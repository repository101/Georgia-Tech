""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing TheoreticallyOptimalStrategy  (c) 2016 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import indicators as ind
import marketsimcode as sim


def check_holdings(hold):
    try:
        if hold > 1000 or hold < -1000:
            return False
        else:
            return True
    except Exception as check_holdings_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'check_holdings'", check_holdings_exception)


def determine_buy_sell_amount(holdings, buy_or_sell):
    try:
        if buy_or_sell == "BUY":
            if holdings == 0:
                return 1000
            elif holdings == -1000:
                return 2000
            else:
                return 0
        elif buy_or_sell == "SELL":
            if holdings == 0:
                return 1000
            elif holdings == 1000:
                return 2000
            else:
                return 0
        else:
            return 0
    except Exception as determine_buy_sell_amount_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'determine_buy_sell_amount'", determine_buy_sell_amount_exception)


def find_optimal(stock_DF, symbol):
    try:
        total_holdings = 0
        columns = ["Date", "Symbol", "Order", "Shares"]
        order_df = pd.DataFrame(columns=columns)
        temp_DF = pd.DataFrame(index=stock_DF.index)
        temp_DF["Date"] = stock_DF.index.values
        temp_DF["Adj_Close"] = stock_DF["Adj Close"]
        temp_DF["Next_Day_Adjusted_Close"] = stock_DF["Adj Close"].shift(-1)
        temp_DF["Diff"] = temp_DF["Next_Day_Adjusted_Close"] - temp_DF["Adj_Close"]
        temp_DF["Order"] = ""
        temp_DF["Position"] = 0
        temp_DF.loc[temp_DF["Diff"] > 0, "Order"] = "BUY"
        temp_DF.loc[temp_DF["Diff"] < 0, "Order"] = "SELL"
        for i in range(temp_DF.shape[0]):
            if temp_DF.iloc[i]["Order"] == "SELL":
                if check_holdings(total_holdings):
                    amount = determine_buy_sell_amount(total_holdings, "SELL")
                    result = {"Date": temp_DF.iloc[i]["Date"], "Symbol": symbol,
                              "Order": temp_DF.iloc[i]["Order"], "Shares": amount}
                    order_df = order_df.append(result, ignore_index=True)
                    total_holdings -= amount
                    temp_DF.loc[temp_DF.iloc[i]["Date"], "Position"] = total_holdings
                else:
                    continue
            elif temp_DF.iloc[i]["Order"] == "BUY":
                if check_holdings(total_holdings):
                    amount = determine_buy_sell_amount(total_holdings, "BUY")
                    result = {"Date": temp_DF.iloc[i]["Date"], "Symbol": symbol,
                              "Order": temp_DF.iloc[i]["Order"], "Shares": amount}
                    order_df = order_df.append(result, ignore_index=True)
                    total_holdings += amount
                    temp_DF.loc[temp_DF.iloc[i]["Date"], "Position"] = total_holdings
                else:
                    continue
        return order_df
    except Exception as find_optimal_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'find_optimal'", find_optimal_exception)


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
               window=14, extra="Figure_"):
    try:
        baseline_results = sim.get_baseline(symbol=symbol, sd=sd, ed=ed, sv=sv)
        plt.close("all")
        all_data = ind.get_all_stock_data(symbol=symbol, sd=sd, ed=ed)
        results = ind.calculate_bollinger(ind.calculate_obv(all_data), window=window)
        results = ind.calculate_macd(ind.calculate_vortex(results, window=window))
        optimal_orders = find_optimal(results, symbol=symbol)
        optimal_prices_DF = sim.get_prices(symbols=[symbol], start_date=sd, end_date=ed)
        optimal_prices_DF = optimal_prices_DF[[symbol]]  # remove SPY
        optimal_prices_DF["Cash"] = np.ones(shape=(optimal_prices_DF.shape[0]))
        optimal_portval = sim.compute_portvals(orders_DF=optimal_orders, prices_DF=optimal_prices_DF,
                                               start_val=sv, commission=0, impact=0)
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
        ax1.plot(baseline_results["PortVals"] / baseline_results["PortVals"][0],
                 label="Baseline", linewidth=1.15, color="tab:green")
        ax1.plot(optimal_portval["PortVals"] / optimal_portval["PortVals"][0],
                 label="Optimal", linewidth=1.15, color="tab:red")
        ax1.set_title(f"Theoretically Optimal Strategy\n{symbol}", fontsize=15, weight='bold')
        ax1.set_ylabel("Performance", fontsize=15, weight='heavy')
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.set_xlabel("Trading Date", fontsize=12, weight='heavy')
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{symbol}_Theoretically_Optimal_Strategy_better.png")
        a = 8
        base_cumulative_returns = sim.get_cumulative_returns(baseline_results["PortVals"])
        base_daily_returns = sim.get_daily_returns(baseline_results["PortVals"])

        optimal_cumulative_returns = sim.get_cumulative_returns(optimal_portval["PortVals"] )
        optimal_daily_returns = sim.get_daily_returns(optimal_portval["PortVals"] )
        print("Baseline Statistics")
        print(f"\tCumulative Return: {base_cumulative_returns}")
        print(f"\tStandard Deviation of Daily Returns: {base_daily_returns.std()}")
        print(f"\tMean of Daily Returns: {base_daily_returns.mean()}")

        print("Theoretically Optimal Strategy Statistics")
        print(f"\tCumulative Return: {optimal_cumulative_returns}")
        print(f"\tStandard Deviation of Daily Returns: {optimal_daily_returns.std()}")
        print(f"\tMean of Daily Returns: {optimal_daily_returns.mean()}")
        return
    except Exception as testPolicy_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'testPolicy'", testPolicy_exception)


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jadams334"  # Change this to your user ID


if __name__ == '__main__':
    print()
