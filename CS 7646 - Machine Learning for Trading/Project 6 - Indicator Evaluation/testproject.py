""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing testproject  (c) 2020 Josh Adams 		  	   		     		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  


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

import marketsimcode as sim
import TheoreticallyOptimalStrategy as tos


def get_bollinger_band_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                              ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        # Figure 1 - Bollinger Bands
        print("\tBollinger Band - Figure 1: Starting")
        sim.get_bollinger_band(symbol=symbol, sd=sd, ed=ed, extra="Figure_1", marker_size=10)
        print("\tBollinger Band - Figure 1: Finished")

        # Figure 2 - Bollinger Bands - Closer
        print("\tBollinger Band - Figure 2: Starting")
        sim.get_bollinger_band(symbol=symbol, sd=dt.datetime(2009, 8, 1), ed=dt.datetime(2009, 12, 1),
                               extra="Figure_2", marker_size=10)
        print("\tBollinger Band - Figure 2: Finished")
        return
    except Exception as get_bollinger_band_charts_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_bollinger_band_charts'", get_bollinger_band_charts_exception)


def get_rsi_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        # Figure 3 - Relative Strength Index
        print("\tRelative Strength Index - Figure 3: Starting")
        sim.get_relative_strength_indicator(symbol=symbol, sd=sd, ed=ed,
                                            fill_between=True, extra="Figure_3",
                                            marker_size=6, lookback=14)
        print("\tRelative Strength Index - Figure 3: Finished")
        return
    except Exception as get_RSI_charts_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_RSI_charts'", get_RSI_charts_exception)


def get_macd_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        # Figure 4 - Moving Average Convergence Divergence
        print("\tMoving Average Convergence Divergence - Figure 4: Starting")
        sim.get_macd_indicator(symbol=symbol, sd=sd, ed=ed,
                               fill_between=True, extra="Figure_4",
                               marker_size=6, window=14)
        print("\tMoving Average Convergence Divergence - Figure 4: Finished")
        return
    except Exception as get_macd_charts_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_macd_charts'", get_macd_charts_exception)


def get_obv_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        # Figure 5 - On-Balance Volume
        print("\tOn-Balance Volume - Figure 5: Starting")
        sim.get_obv_indicator(symbol=symbol, sd=sd, ed=ed,
                              fill_between=True, extra="Figure_5",
                              marker_size=6, window=14)
        print("\tOn-Balance Volume - Figure 5: Finished")
        return
    except Exception as get_obv_charts_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_obv_charts'", get_obv_charts_exception)


def get_vortex_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                      ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        # Figure 6 - Vortex
        print("\tVortex Indicator - Figure 6: Starting")
        sim.get_vortex_indicator(symbol=symbol, sd=sd, ed=ed,
                                 fill_between=True, extra="Figure_6",
                                 marker_size=6, window=14)
        print("\tVortex Indicator - Figure 6: Finished")
        return
    except Exception as get_vortex_charts_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_vortex_charts'", get_vortex_charts_exception)


def get_theoretically_optimal_strategy_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                                              ed=dt.datetime(2009, 12, 31), sv=100000):
    try:
        sim.get_theoretically_optimal_strategy(symbol=symbol, sd=sd, ed=ed,
                                               extra="Figure_Theoretically_Optimal_Strategy", window=14)
    except Exception as get_theoretically_optimal_strategy_charts_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_theoretically_optimal_strategy_charts'",
              get_theoretically_optimal_strategy_charts_exception)


if __name__ == '__main__':
    print("Beginning Bollinger Bands")
    get_bollinger_band_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                              ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Finished Bollinger Bands")

    print("Beginning Relative Strength Index")
    get_rsi_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Finished Relative Strength Index")

    print("Beginning Moving Average Convergence Divergence")
    get_macd_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Finished Moving Average Convergence Divergence")

    print("Beginning On-Balance Volume")
    get_obv_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Finished On-Balance Volume")

    print("Beginning Vortex Indicator")
    get_vortex_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                      ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Finished Vortex Indicator")
    print("Beginning Theoretically Optimal Strategy")
    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Finished Theoretically Optimal Strategy")

    exit()
