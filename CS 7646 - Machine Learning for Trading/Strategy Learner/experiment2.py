"""
Template for implementing experiment1 (c) 2016 Tucker Balch

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
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import random
import numpy as np
import StrategyLearner as sl
import ManualStrategy as manstrat
import marketsimcode as mktsim
from matplotlib import pyplot as plt
from util import get_data




def author():
	return 'jadams334'


def experiment1():
	symbol = ["JPM"]
	sv = 100000
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	impact = 0.005
	
	# Benchmark
	benchmark_data = get_data(symbols=symbol, dates=pd.date_range(sd, ed))
	benchmark_data["Shares"] = np.zeros(shape=(len(benchmark_data)))
	benchmark_data["Shares"][0] = 1000
	benchmark_data["Shares"][-1] = -1000
	benchmark_data["Order"] = np.zeros(shape=(len(benchmark_data)))
	benchmark_data["Symbol"] = np.zeros(shape=(len(benchmark_data)), dtype=object)
	benchmark_data["Symbol"] = symbol[0]
	benchmark_data.drop(["SPY"], axis=1, inplace=True)
	benchmark_portVal = mktsim.compute_portvals_from_tester(benchmark_data, start_date=sd, end_date=ed,
	                                                    startval=sv, market_impact=impact, commission_cost=0.00)
	
	# Manual Strategy
	# manual_strategy = manstrat.ManualStrategy()
	# manual_strategy_dataframe = manual_strategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv, )
	
	# QLearner
	learner = sl.StrategyLearner(verbose=False, impact=impact)
	learner.addEvidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
	learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
	learner.test_policy__trades_dataframe["Shares"] = learner.test_policy__trades_dataframe[symbol]
	learner.test_policy__trades_dataframe["Order"] = np.zeros(shape=(len(learner.test_policy__trades_dataframe)))
	learner.test_policy__trades_dataframe["Symbol"] = symbol
	port_val = mktsim.compute_portvals_from_tester(learner.test_policy__trades_dataframe, start_date=sd, end_date=ed,
	                                                    startval=sv, market_impact=impact, commission_cost=0.00)
	
	return


if __name__ == "__main__":
	experiment1()